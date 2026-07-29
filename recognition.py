from __future__ import annotations

import json
import hashlib
import importlib.metadata
import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable


class CalibrationError(RuntimeError):
    """Raised when a stable game viewport cannot be established."""


@dataclass(frozen=True)
class CalibrationResult:
    client_size: tuple[int, int]
    content_top: int
    candidates: tuple[int, ...]
    inliers: tuple[int, ...]
    edge_strengths: tuple[float, ...]
    confidence: float
    calibrated_at: float
    generation: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CapturedFrame:
    raw: Any
    normalized: Any
    calibration: CalibrationResult
    viewport: tuple[int, int, int, int]
    timestamp: float
    frame_change: float = 0.0


@dataclass(frozen=True)
class OCRToken:
    text: str
    confidence: float
    rect: tuple[int, int, int, int]


_OCR_CACHE_LOCK = threading.RLock()
_OCR_RESULT_CACHE: OrderedDict[tuple[Any, ...], tuple[OCRToken, ...]] = OrderedDict()
_OCR_CACHE_HITS = 0
_OCR_CACHE_MISSES = 0
_SHARED_OCR_ENGINES: dict[tuple[Any, ...], "LocalOCR"] = {}


@dataclass(frozen=True)
class DetectedControl:
    name: str
    rect: tuple[int, int, int, int]
    confidence: float
    source: str
    text: str = ""


@dataclass
class Observation:
    timestamp: float
    viewport: tuple[int, int, int, int]
    title_candidates: dict[str, float] = field(default_factory=dict)
    controls: list[DetectedControl] = field(default_factory=list)
    numeric_values: dict[str, int] = field(default_factory=dict)
    template_scores: dict[str, float] = field(default_factory=dict)
    frame_change: float = 0.0
    state: str = "unknown"
    state_confidence: float = 0.0
    ocr_tokens: list[OCRToken] = field(default_factory=list)
    signals: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class ViewportCalibrator:
    """Calibrates once per client size and keeps the result stable between frames."""

    def __init__(self, config: dict[str, Any], cv2: Any, np: Any):
        self.config = config
        self.cv2 = cv2
        self.np = np
        settings = config.get("viewport_calibration", {})
        self.frame_count = max(5, int(settings.get("frame_count", 5)))
        self.max_deviation = max(0, int(settings.get("max_deviation", 2)))
        self.min_edge_strength = float(settings.get("min_edge_strength", 20.0))
        self.min_inliers = max(3, int(settings.get("min_inliers", 4)))
        self._cached: CalibrationResult | None = None
        self._generation = 0

    @property
    def cached(self) -> CalibrationResult | None:
        return self._cached

    def invalidate(self) -> None:
        self._cached = None

    def detect_candidate(self, image: Any) -> tuple[int, float]:
        reference_top = int(self.config.get("reference_content_top", 0))
        if reference_top <= 0:
            return 0, float("inf")
        start, end = self.config.get("content_top_search_range", [30, 90])
        end = min(int(end), image.shape[0] - 1)
        start = max(1, min(int(start), end))
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        row_means = gray.mean(axis=1)
        differences = row_means[start : end + 1] - row_means[start - 1 : end]
        offset = int(self.np.argmin(differences))
        return start + offset, max(0.0, float(-differences[offset]))

    def calibrate_frames(
        self, frames: Iterable[Any], client_size: tuple[int, int]
    ) -> CalibrationResult:
        frame_list = list(frames)
        if len(frame_list) < self.frame_count:
            raise CalibrationError(
                f"视口校准至少需要 {self.frame_count} 帧，实际只有 {len(frame_list)} 帧。"
            )
        measured = [self.detect_candidate(frame) for frame in frame_list]
        candidates = [item[0] for item in measured]
        strengths = [item[1] for item in measured]
        median = int(round(float(self.np.median(candidates))))
        inlier_indexes = [
            index
            for index, (candidate, strength) in enumerate(measured)
            if abs(candidate - median) <= self.max_deviation
            and strength >= self.min_edge_strength
        ]
        if len(inlier_indexes) < self.min_inliers:
            raise CalibrationError(
                "游戏内容顶部校准不可信："
                f"candidates={candidates}, strengths={[round(v, 2) for v in strengths]}, "
                f"required_inliers={self.min_inliers}"
            )
        inliers = [candidates[index] for index in inlier_indexes]
        content_top = int(round(float(self.np.median(inliers))))
        spread = max(inliers) - min(inliers)
        confidence = min(
            1.0,
            (len(inliers) / len(candidates))
            * (1.0 - min(spread, self.max_deviation + 1) / (self.max_deviation + 2)),
        )
        self._generation += 1
        result = CalibrationResult(
            client_size=tuple(map(int, client_size)),
            content_top=content_top,
            candidates=tuple(candidates),
            inliers=tuple(inliers),
            edge_strengths=tuple(round(value, 3) for value in strengths),
            confidence=round(confidence, 4),
            calibrated_at=time.time(),
            generation=self._generation,
        )
        self._cached = result
        logging.info(
            "视口校准完成：size=%sx%s content_top=%s candidates=%s confidence=%.3f generation=%s",
            client_size[0],
            client_size[1],
            content_top,
            candidates,
            confidence,
            self._generation,
        )
        return result

    def ensure(
        self,
        client_size: tuple[int, int],
        capture: Callable[[], Any],
    ) -> tuple[Any, CalibrationResult, bool]:
        normalized_size = tuple(map(int, client_size))
        if self._cached is not None and self._cached.client_size == normalized_size:
            return capture(), self._cached, False
        frames = [capture() for _ in range(self.frame_count)]
        result = self.calibrate_frames(frames, normalized_size)
        return frames[-1], result, True


def normalize_frame(
    image: Any,
    calibration: CalibrationResult,
    reference_size: Iterable[int],
    reference_content_top: int,
    cv2: Any,
    np: Any,
) -> Any:
    width, height = map(int, reference_size)
    content_top = int(calibration.content_top)
    reference_top = int(reference_content_top)
    if reference_top <= 0 or content_top <= 0:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if content_top >= image.shape[0]:
        raise CalibrationError(
            f"内容顶部 {content_top} 超出截图高度 {image.shape[0]}。"
        )
    chrome = cv2.resize(
        image[:content_top], (width, reference_top), interpolation=cv2.INTER_AREA
    )
    content = cv2.resize(
        image[content_top:],
        (width, height - reference_top),
        interpolation=cv2.INTER_AREA,
    )
    return np.vstack((chrome, content))


def frame_difference(previous: Any | None, current: Any, cv2: Any) -> float:
    if previous is None:
        return 0.0
    size = (160, 100)
    previous_gray = cv2.cvtColor(
        cv2.resize(previous, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
    )
    current_gray = cv2.cvtColor(
        cv2.resize(current, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
    )
    return float(cv2.mean(cv2.absdiff(previous_gray, current_gray))[0])


class LocalOCR:
    """Lazy RapidOCR adapter; only configured small ROIs are submitted."""

    BACKEND_PROVIDERS = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "dml": "DmlExecutionProvider",
    }

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        engine: Any | None = None,
        *,
        engine_factory: Callable[..., Any] | None = None,
        available_providers: Iterable[str] | None = None,
        np_module: Any | None = None,
    ):
        settings = (config or {}).get("recognition_v2", config or {})
        configured_backend = str(settings.get("ocr_backend", "auto")).lower()
        self.requested_backend = os.environ.get(
            "HUNTER_OCR_BACKEND", configured_backend
        ).lower()
        if self.requested_backend not in {"auto", "cpu", "cuda", "dml"}:
            raise ValueError(
                "recognition_v2.ocr_backend 必须是 auto、cpu、cuda 或 dml，"
                f"当前为 {self.requested_backend!r}。"
            )
        self.gpu_fallback = bool(settings.get("ocr_gpu_fallback", True))
        self._engine_factory = engine_factory
        self._available_providers_override = (
            tuple(map(str, available_providers))
            if available_providers is not None
            else None
        )
        self._np = np_module
        self._engine = engine
        self._load_attempted = engine is not None
        self._warned = False
        self._fallback_warned = False
        self.active_backend = "injected" if engine is not None else "uninitialized"
        self.actual_providers: dict[str, tuple[str, ...]] = {}
        self.load_error = ""
        try:
            self.model_version = importlib.metadata.version("rapidocr-onnxruntime")
        except importlib.metadata.PackageNotFoundError:
            self.model_version = "unavailable"
        if engine is not None:
            self.actual_providers = self._engine_providers(engine)

    @property
    def available_providers(self) -> tuple[str, ...]:
        if self._available_providers_override is not None:
            return self._available_providers_override
        try:
            import onnxruntime as ort

            return tuple(map(str, ort.get_available_providers()))
        except ImportError:
            return ()

    @property
    def actual_provider(self) -> str:
        primaries = [providers[0] for providers in self.actual_providers.values() if providers]
        if not primaries:
            return "unavailable"
        return primaries[0] if len(set(primaries)) == 1 else ",".join(primaries)

    @property
    def cache_identity(self) -> tuple[str, str, str]:
        return (self.requested_backend, self.active_backend, self.model_version)

    def runtime_info(self) -> dict[str, Any]:
        self._load()
        return {
            "requested_backend": self.requested_backend,
            "active_backend": self.active_backend,
            "actual_provider": self.actual_provider,
            "session_providers": {
                name: list(providers)
                for name, providers in self.actual_providers.items()
            },
            "available_providers": list(self.available_providers),
            "model_version": self.model_version,
            "gpu_fallback": self.gpu_fallback,
            "load_error": self.load_error,
        }

    @property
    def available(self) -> bool:
        self._load()
        return self._engine is not None

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            if self._engine_factory is None:
                from rapidocr_onnxruntime import RapidOCR

                self._engine_factory = RapidOCR
            candidates = self._backend_candidates()
            failures = []
            for backend in candidates:
                try:
                    self._initialize_backend(backend)
                    return
                except Exception as exc:
                    failures.append(f"{backend}: {exc}")
                    logging.warning("OCR 后端 %s 初始化/自检失败：%s", backend, exc)
            self.load_error = "; ".join(failures)
            if self.requested_backend != "auto" and not self.gpu_fallback:
                raise RuntimeError(self.load_error)
            self._engine = None
            self.active_backend = "unavailable"
        except ImportError as exc:
            self._engine = None
            self.active_backend = "unavailable"
            self.load_error = str(exc)

    def _backend_candidates(self) -> list[str]:
        providers = set(self.available_providers)
        if self.requested_backend == "auto":
            candidates = [
                backend
                for backend in ("cuda", "dml")
                if self.BACKEND_PROVIDERS[backend] in providers
            ]
            candidates.append("cpu")
            return candidates
        expected = self.BACKEND_PROVIDERS[self.requested_backend]
        if self.requested_backend == "cpu" or expected in providers:
            candidates = [self.requested_backend]
            if self.requested_backend != "cpu" and self.gpu_fallback:
                candidates.append("cpu")
            return candidates
        message = (
            f"{expected} 不在可用 Provider {sorted(providers)} 中"
        )
        if not self.gpu_fallback:
            raise RuntimeError(message)
        logging.warning("OCR 后端 %s 不可用，回退 CPU：%s", self.requested_backend, message)
        return ["cpu"]

    @staticmethod
    def _backend_kwargs(backend: str) -> dict[str, bool]:
        use_cuda = backend == "cuda"
        use_dml = backend == "dml"
        return {
            "det_use_cuda": use_cuda,
            "cls_use_cuda": use_cuda,
            "rec_use_cuda": use_cuda,
            "det_use_dml": use_dml,
            "cls_use_dml": use_dml,
            "rec_use_dml": use_dml,
        }

    @staticmethod
    def _engine_providers(engine: Any) -> dict[str, tuple[str, ...]]:
        holders = {
            "det": getattr(getattr(engine, "text_det", None), "infer", None),
            "cls": getattr(getattr(engine, "text_cls", None), "infer", None),
            "rec": getattr(getattr(engine, "text_rec", None), "session", None),
        }
        providers: dict[str, tuple[str, ...]] = {}
        for name, holder in holders.items():
            session = getattr(holder, "session", holder)
            if session is not None and hasattr(session, "get_providers"):
                providers[name] = tuple(map(str, session.get_providers()))
        return providers

    def _initialize_backend(self, backend: str) -> None:
        if self._engine_factory is None:
            raise RuntimeError("RapidOCR engine factory 未初始化")
        engine = self._engine_factory(**self._backend_kwargs(backend))
        providers = self._engine_providers(engine)
        expected = self.BACKEND_PROVIDERS[backend]
        if len(providers) != 3 or any(
            not values or values[0] != expected for values in providers.values()
        ):
            raise RuntimeError(
                f"RapidOCR 会话未全部使用 {expected}：{providers}"
            )
        if self._np is None:
            import numpy as np

            self._np = np
        probe = self._np.full((64, 192, 3), 255, dtype=self._np.uint8)
        engine(probe)
        self._engine = engine
        self.active_backend = backend
        self.actual_providers = providers
        self.load_error = ""
        logging.info(
            "OCR 后端已启用：requested=%s active=%s provider=%s available=%s",
            self.requested_backend,
            self.active_backend,
            self.actual_provider,
            list(self.available_providers),
        )

    def _fallback_after_inference_failure(self, exc: Exception) -> bool:
        if self.active_backend not in {"cuda", "dml"} or not self.gpu_fallback:
            return False
        if not self._fallback_warned:
            logging.warning(
                "OCR %s 推理失败，自动回退 CPU：%s", self.active_backend, exc
            )
            self._fallback_warned = True
        self._initialize_backend("cpu")
        return True

    def read(
        self, image: Any, offset: tuple[int, int] = (0, 0)
    ) -> list[OCRToken]:
        self._load()
        if self._engine is None:
            if not self._warned:
                logging.warning("RapidOCR 不可用，局部 OCR 信号已禁用。")
                self._warned = True
            return []
        try:
            result, _elapsed = self._engine(image)
        except Exception as exc:
            if not self._fallback_after_inference_failure(exc):
                raise
            result, _elapsed = self._engine(image)
        tokens: list[OCRToken] = []
        for item in result or []:
            box, text, confidence = item
            xs = [int(round(point[0])) for point in box]
            ys = [int(round(point[1])) for point in box]
            tokens.append(
                OCRToken(
                    text=str(text),
                    confidence=float(confidence),
                    rect=(
                        min(xs) + offset[0],
                        min(ys) + offset[1],
                        max(xs) + offset[0],
                        max(ys) + offset[1],
                    ),
                )
            )
        return tokens


def get_shared_local_ocr(config: dict[str, Any], np_module: Any) -> LocalOCR:
    settings = config.get("recognition_v2", {})
    key = (
        os.environ.get(
            "HUNTER_OCR_BACKEND", str(settings.get("ocr_backend", "auto"))
        ).lower(),
        bool(settings.get("ocr_gpu_fallback", True)),
        json.dumps(
            {name: value for name, value in settings.items() if name.startswith("ocr_")},
            sort_keys=True,
            ensure_ascii=True,
        ),
    )
    with _OCR_CACHE_LOCK:
        engine = _SHARED_OCR_ENGINES.get(key)
        if engine is None:
            engine = LocalOCR(config, np_module=np_module)
            _SHARED_OCR_ENGINES[key] = engine
        return engine


def clear_ocr_runtime(*, engines: bool = False) -> None:
    global _OCR_CACHE_HITS, _OCR_CACHE_MISSES
    with _OCR_CACHE_LOCK:
        _OCR_RESULT_CACHE.clear()
        _OCR_CACHE_HITS = 0
        _OCR_CACHE_MISSES = 0
        if engines:
            _SHARED_OCR_ENGINES.clear()


def ocr_cache_info() -> dict[str, int]:
    with _OCR_CACHE_LOCK:
        return {
            "entries": len(_OCR_RESULT_CACHE),
            "hits": _OCR_CACHE_HITS,
            "misses": _OCR_CACHE_MISSES,
        }


def _normalized_text(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).lower()


def text_similarity(actual: str, expected: str) -> float:
    left = _normalized_text(actual)
    right = _normalized_text(expected)
    if not left or not right:
        return 0.0
    if left in right or right in left:
        return min(1.0, min(len(left), len(right)) / max(1, len(right)) + 0.25)
    return SequenceMatcher(None, left, right).ratio()


class RecognitionEngine:
    def __init__(self, config: dict[str, Any], cv2: Any, np: Any, ocr: LocalOCR | None = None):
        self.config = config
        self.settings = config.get("recognition_v2", {})
        self.cv2 = cv2
        self.np = np
        self.ocr = ocr or get_shared_local_ocr(config, np)
        self._template_control_cache: dict[tuple[str, tuple[int, ...]], Any] = {}
        self._current_image_hash = ""
        self._ocr_cache_config = json.dumps(
            {
                name: value
                for name, value in self.settings.items()
                if name.startswith("ocr_")
            },
            sort_keys=True,
            ensure_ascii=True,
        )

    @staticmethod
    def _crop(image: Any, rect: Iterable[int]) -> tuple[Any, tuple[int, int]]:
        x1, y1, x2, y2 = map(int, rect)
        x1 = max(0, min(x1, image.shape[1]))
        x2 = max(x1, min(x2, image.shape[1]))
        y1 = max(0, min(y1, image.shape[0]))
        y2 = max(y1, min(y2, image.shape[0]))
        return image[y1:y2, x1:x2], (x1, y1)

    def _read_region(self, image: Any, rect: Iterable[int]) -> list[OCRToken]:
        global _OCR_CACHE_HITS, _OCR_CACHE_MISSES
        rect_key = tuple(map(int, rect))
        cache_enabled = bool(self.settings.get("ocr_cache", True))
        if cache_enabled:
            _ = self.ocr.available
        cache_key = (
            self._current_image_hash,
            rect_key,
            float(self.settings.get("ocr_scale", 2.0)),
            self.ocr.cache_identity,
            self.settings.get("ocr_model_version", self.ocr.model_version),
            self._ocr_cache_config,
        )
        if cache_enabled and self._current_image_hash:
            with _OCR_CACHE_LOCK:
                cached = _OCR_RESULT_CACHE.get(cache_key)
                if cached is not None:
                    _OCR_RESULT_CACHE.move_to_end(cache_key)
                    _OCR_CACHE_HITS += 1
                    return list(cached)
                _OCR_CACHE_MISSES += 1
        crop, offset = self._crop(image, rect)
        if crop.size == 0:
            return []
        scale = float(self.settings.get("ocr_scale", 2.0))
        if scale != 1.0:
            crop = self.cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=self.cv2.INTER_CUBIC,
            )
            tokens = self.ocr.read(crop)
            result = [
                OCRToken(
                    token.text,
                    token.confidence,
                    (
                        offset[0] + round(token.rect[0] / scale),
                        offset[1] + round(token.rect[1] / scale),
                        offset[0] + round(token.rect[2] / scale),
                        offset[1] + round(token.rect[3] / scale),
                    ),
                )
                for token in tokens
            ]
        else:
            result = self.ocr.read(crop, offset)
        if cache_enabled and self._current_image_hash:
            max_entries = max(1, int(self.settings.get("ocr_cache_max_entries", 2048)))
            cache_key = (
                self._current_image_hash,
                rect_key,
                float(self.settings.get("ocr_scale", 2.0)),
                self.ocr.cache_identity,
                self.settings.get("ocr_model_version", self.ocr.model_version),
                self._ocr_cache_config,
            )
            with _OCR_CACHE_LOCK:
                _OCR_RESULT_CACHE[cache_key] = tuple(result)
                _OCR_RESULT_CACHE.move_to_end(cache_key)
                while len(_OCR_RESULT_CACHE) > max_entries:
                    _OCR_RESULT_CACHE.popitem(last=False)
        return result

    @staticmethod
    def _enabled(spec: dict[str, Any], template_scores: dict[str, float]) -> bool:
        names = spec.get("when_templates", [])
        if not names:
            return True
        gate = float(spec.get("template_gate", 0.45))
        return max((template_scores.get(name, 0.0) for name in names), default=0.0) >= gate

    def _color_mask(self, hsv: Any, colors: Iterable[str]) -> Any:
        masks = []
        for color in colors:
            if color == "gold":
                masks.append(self.cv2.inRange(hsv, (10, 65, 90), (42, 255, 255)))
            elif color == "blue":
                masks.append(self.cv2.inRange(hsv, (85, 70, 120), (125, 255, 255)))
            elif color == "cyan":
                masks.append(self.cv2.inRange(hsv, (72, 45, 75), (105, 255, 255)))
            elif color == "gray":
                masks.append(self.cv2.inRange(hsv, (0, 0, 65), (179, 75, 235)))
            elif color == "red":
                low = self.cv2.inRange(hsv, (0, 90, 90), (8, 255, 255))
                high = self.cv2.inRange(hsv, (172, 90, 90), (179, 255, 255))
                masks.append(self.cv2.bitwise_or(low, high))
        if not masks:
            return self.np.zeros(hsv.shape[:2], dtype=self.np.uint8)
        mask = masks[0]
        for extra in masks[1:]:
            mask = self.cv2.bitwise_or(mask, extra)
        kernel = self.cv2.getStructuringElement(self.cv2.MORPH_RECT, (7, 5))
        return self.cv2.morphologyEx(mask, self.cv2.MORPH_CLOSE, kernel)

    def _control_candidates(self, image: Any, spec: dict[str, Any]) -> list[tuple[int, int, int, int, float]]:
        crop, offset = self._crop(image, spec["region"])
        if crop.size == 0:
            return []
        hsv = self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2HSV)
        mask = self._color_mask(hsv, spec.get("colors", ("gold", "blue")))
        contours, _hierarchy = self.cv2.findContours(
            mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE
        )
        min_width = int(spec.get("min_width", 70))
        min_height = int(spec.get("min_height", 22))
        max_height = int(spec.get("max_height", 100))
        min_fraction = float(spec.get("min_color_fraction", 0.12))
        candidates = []
        for contour in contours:
            x, y, width, height = self.cv2.boundingRect(contour)
            if width < min_width or height < min_height or height > max_height:
                continue
            fraction = float(self.cv2.contourArea(contour)) / max(1, width * height)
            if fraction < min_fraction:
                continue
            candidates.append(
                (
                    offset[0] + x,
                    offset[1] + y,
                    offset[0] + x + width,
                    offset[1] + y + height,
                    min(1.0, 0.55 + fraction),
                )
            )
        return sorted(candidates, key=lambda item: (item[1], item[0]))

    def _match_aliases(
        self, tokens: Iterable[OCRToken], aliases: dict[str, list[str]]
    ) -> tuple[str, float, str] | None:
        best: tuple[str, float, str] | None = None
        for token in tokens:
            for name, values in aliases.items():
                similarity = max(text_similarity(token.text, value) for value in values)
                score = token.confidence * similarity
                if best is None or score > best[1]:
                    best = (name, score, token.text)
        threshold = float(self.settings.get("ocr_match_threshold", 0.55))
        return best if best is not None and best[1] >= threshold else None

    def _template_control(self, image: Any, spec: dict[str, Any]) -> DetectedControl | None:
        source = Path(spec["source"])
        if not source.is_absolute():
            source = Path(__file__).resolve().parent / source
        template_region = tuple(map(int, spec["template_region"]))
        key = (str(source.resolve()), template_region)
        template = self._template_control_cache.get(key)
        if template is None:
            encoded = self.np.fromfile(source, dtype=self.np.uint8)
            reference = self.cv2.imdecode(encoded, self.cv2.IMREAD_GRAYSCALE)
            if reference is None:
                logging.warning("无法读取局部控件模板：%s", source)
                return None
            x1, y1, x2, y2 = template_region
            template = reference[y1:y2, x1:x2]
            if template.size == 0:
                logging.warning("局部控件模板区域为空：%s %s", source, template_region)
                return None
            self._template_control_cache[key] = template

        search, offset = self._crop(image, spec["search_region"])
        if search.size == 0:
            return None
        gray = self.cv2.cvtColor(search, self.cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < template.shape[0] or gray.shape[1] < template.shape[1]:
            return None
        _minimum, maximum, _minimum_at, maximum_at = self.cv2.minMaxLoc(
            self.cv2.matchTemplate(gray, template, self.cv2.TM_CCOEFF_NORMED)
        )
        score = float(maximum) if math.isfinite(maximum) else 0.0
        if score < float(spec.get("threshold", 0.75)):
            return None
        x1 = offset[0] + int(maximum_at[0])
        y1 = offset[1] + int(maximum_at[1])
        return DetectedControl(
            name=str(spec["name"]),
            rect=(x1, y1, x1 + template.shape[1], y1 + template.shape[0]),
            confidence=round(score, 4),
            source="template",
        )

    def detect_titles(
        self, image: Any, template_scores: dict[str, float]
    ) -> tuple[dict[str, float], list[OCRToken]]:
        candidates: dict[str, float] = {}
        tokens: list[OCRToken] = []
        cache: dict[tuple[int, ...], list[OCRToken]] = {}
        for name, spec in self.settings.get("title_regions", {}).items():
            if not self._enabled(spec, template_scores):
                candidates[name] = 0.0
                continue
            key = tuple(map(int, spec["region"]))
            if key not in cache:
                cache[key] = self._read_region(image, spec["region"])
                tokens.extend(cache[key])
            region_tokens = cache[key]
            score = 0.0
            for token in region_tokens:
                score = max(
                    score,
                    token.confidence
                    * max(text_similarity(token.text, alias) for alias in spec["aliases"]),
                )
            candidates[name] = score
        return candidates, tokens

    def detect_controls(
        self, image: Any, template_scores: dict[str, float]
    ) -> tuple[list[DetectedControl], list[OCRToken]]:
        controls: list[DetectedControl] = []
        all_tokens: list[OCRToken] = []
        for spec in self.settings.get("template_controls", []):
            if not self._enabled(spec, template_scores):
                continue
            control = self._template_control(image, spec)
            if control is not None:
                controls.append(control)
        for spec in self.settings.get("control_regions", []):
            if not self._enabled(spec, template_scores):
                continue
            aliases = spec.get("aliases", {})
            for x1, y1, x2, y2, color_confidence in self._control_candidates(image, spec):
                padding = int(spec.get("ocr_padding", 3))
                tokens = self._read_region(
                    image,
                    (x1 - padding, y1 - padding, x2 + padding, y2 + padding),
                )
                all_tokens.extend(tokens)
                matched = self._match_aliases(tokens, aliases)
                if matched is None:
                    continue
                name, ocr_confidence, text = matched
                controls.append(
                    DetectedControl(
                        name=name,
                        rect=(x1, y1, x2, y2),
                        confidence=round((color_confidence + ocr_confidence) / 2, 4),
                        source="color+ocr",
                        text=text,
                    )
                )
            for fixed in spec.get("fixed", []):
                tokens = self._read_region(image, fixed["region"])
                all_tokens.extend(tokens)
                matched = self._match_aliases(tokens, {fixed["name"]: fixed["aliases"]})
                if matched is None:
                    continue
                name, confidence, text = matched
                token_rects = [token.rect for token in tokens if text_similarity(token.text, text) > 0.8]
                rect = token_rects[0] if token_rects else tuple(map(int, fixed["region"]))
                controls.append(
                    DetectedControl(name, rect, round(confidence, 4), "ocr", text)
                )
        deduplicated: dict[tuple[str, tuple[int, int, int, int]], DetectedControl] = {}
        for control in controls:
            key = (control.name, control.rect)
            if key not in deduplicated or deduplicated[key].confidence < control.confidence:
                deduplicated[key] = control
        center_distance = float(
            self.settings.get("control_dedup_center_distance", 12.0)
        )
        spatially_deduplicated: list[DetectedControl] = []
        for control in sorted(
            deduplicated.values(), key=lambda item: item.confidence, reverse=True
        ):
            x1, y1, x2, y2 = control.rect
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            overlaps_existing = False
            for existing in spatially_deduplicated:
                if existing.name != control.name:
                    continue
                ex1, ey1, ex2, ey2 = existing.rect
                existing_center = ((ex1 + ex2) / 2.0, (ey1 + ey2) / 2.0)
                if (
                    (center[0] - existing_center[0]) ** 2
                    + (center[1] - existing_center[1]) ** 2
                    <= center_distance**2
                ):
                    overlaps_existing = True
                    break
            if not overlaps_existing:
                spatially_deduplicated.append(control)
        return spatially_deduplicated, all_tokens

    def detect_numbers(
        self, image: Any, template_scores: dict[str, float]
    ) -> tuple[dict[str, int], list[OCRToken]]:
        values: dict[str, int] = {}
        tokens: list[OCRToken] = []
        for name, spec in self.settings.get("numeric_regions", {}).items():
            if not self._enabled(spec, template_scores):
                continue
            region_tokens = self._read_region(image, spec["region"])
            tokens.extend(region_tokens)
            joined = " ".join(token.text for token in region_tokens)
            match = re.search(spec.get("pattern", r"\d+"), joined)
            if match:
                values[name] = int(match.group(int(spec.get("group", 0))))
        return values, tokens

    def _classify(self, observation: Observation) -> tuple[str, float, dict[str, dict[str, float]]]:
        all_signals: dict[str, dict[str, float]] = {}
        candidates: dict[str, float] = {}
        control_scores: dict[str, float] = {}
        for control in observation.controls:
            control_scores[control.name] = max(
                control_scores.get(control.name, 0.0), control.confidence
            )
        combined_text = " ".join(token.text for token in observation.ocr_tokens)
        for state, rule in self.settings.get("state_rules", {}).items():
            signals: dict[str, float] = {}
            template_names = rule.get("template_any", [])
            if template_names:
                name = max(template_names, key=lambda item: observation.template_scores.get(item, 0.0))
                score = observation.template_scores.get(name, 0.0)
                threshold = float(
                    self.config.get("page_thresholds", {}).get(
                        name, self.config.get("page_match_threshold", 0.78)
                    )
                )
                if score >= threshold:
                    signals["template"] = score
            title_names = rule.get("title_any", [])
            if title_names:
                score = max((observation.title_candidates.get(name, 0.0) for name in title_names), default=0.0)
                if score >= float(rule.get("title_threshold", 0.55)):
                    signals["title"] = score
            control_names = rule.get("control_any", [])
            if control_names:
                score = max((control_scores.get(name, 0.0) for name in control_names), default=0.0)
                if score >= float(rule.get("control_threshold", 0.55)):
                    signals["control"] = score
            numeric = rule.get("numeric_equals", {})
            if numeric and all(observation.numeric_values.get(name) == int(value) for name, value in numeric.items()):
                signals["numeric"] = 1.0
            numeric_any = rule.get("numeric_any", [])
            if numeric_any and any(name in observation.numeric_values for name in numeric_any):
                signals["numeric"] = 1.0
            text_aliases = rule.get("ocr_any", [])
            if text_aliases:
                score = max((text_similarity(combined_text, alias) for alias in text_aliases), default=0.0)
                if score >= float(rule.get("ocr_threshold", 0.55)):
                    signals["ocr"] = score
            all_signals[state] = signals
            minimum = int(rule.get("min_signals", 2))
            if len(signals) < minimum:
                continue
            confidence = sum(signals.values()) / len(signals)
            candidates[state] = confidence
        for foreground, occluded_states in self.config.get(
            "state_occlusion_overrides", {}
        ).items():
            if foreground in candidates:
                for state in occluded_states:
                    candidates.pop(str(state), None)
        if not candidates:
            return "unknown", 0.0, all_signals
        best_state = max(candidates, key=candidates.get)
        best_confidence = candidates[best_state]
        return best_state, round(best_confidence, 4), all_signals

    def observe(
        self,
        image: Any,
        *,
        viewport: tuple[int, int, int, int] = (0, 0, 0, 0),
        template_scores: dict[str, float] | None = None,
        frame_change: float = 0.0,
        timestamp: float | None = None,
    ) -> Observation:
        scores = dict(template_scores or {})
        if bool(self.settings.get("ocr_cache", True)):
            self._current_image_hash = hashlib.sha256(image.tobytes()).hexdigest()
        else:
            self._current_image_hash = ""
        titles, title_tokens = self.detect_titles(image, scores)
        controls, control_tokens = self.detect_controls(image, scores)
        numbers, number_tokens = self.detect_numbers(image, scores)
        observation = Observation(
            timestamp=time.time() if timestamp is None else timestamp,
            viewport=viewport,
            title_candidates=titles,
            controls=controls,
            numeric_values=numbers,
            template_scores=scores,
            frame_change=float(frame_change),
            ocr_tokens=title_tokens + control_tokens + number_tokens,
        )
        state, confidence, signals = self._classify(observation)
        transient_threshold = float(self.settings.get("stable_frame_change", 3.0))
        if state == "unknown" and observation.frame_change > transient_threshold:
            state = "unknown_transient"
            confidence = min(1.0, observation.frame_change / max(1.0, transient_threshold * 3))
        if state in {"loading", "unknown", "unknown_transient"}:
            observation.controls = []
        state_allowlists = self.settings.get("state_control_allowlists", {})
        if state in state_allowlists:
            allowed_controls = set(map(str, state_allowlists[state]))
            observation.controls = [
                control
                for control in observation.controls
                if control.name in allowed_controls
            ]
        observation.state = state
        observation.state_confidence = confidence
        observation.signals = signals
        return observation

    def annotate(self, image: Any, observation: Observation) -> Any:
        annotated = image.copy()
        for control in observation.controls:
            x1, y1, x2, y2 = control.rect
            self.cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
            self.cv2.putText(
                annotated,
                f"{control.name} {control.confidence:.2f}",
                (x1, max(16, y1 - 6)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 220, 255),
                1,
                self.cv2.LINE_AA,
            )
        self.cv2.putText(
            annotated,
            f"state={observation.state} confidence={observation.state_confidence:.2f}",
            (12, max(24, int(self.config.get("reference_content_top", 0)) - 10)),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (60, 240, 60),
            2,
            self.cv2.LINE_AA,
        )
        return annotated


class MultiFrameConsensus:
    def __init__(
        self,
        required_frames: int = 2,
        max_frame_change: float = 3.0,
        max_frame_change_by_state: dict[str, float] | None = None,
    ):
        self.required_frames = max(2, int(required_frames))
        self.max_frame_change = float(max_frame_change)
        self.max_frame_change_by_state = {
            str(state): float(value)
            for state, value in (max_frame_change_by_state or {}).items()
        }
        self._recent: deque[Observation] = deque(maxlen=self.required_frames)

    def reset(self) -> None:
        self._recent.clear()

    def update(self, observation: Observation) -> Observation | None:
        max_change = self.max_frame_change_by_state.get(
            observation.state, self.max_frame_change
        )
        if observation.state == "unknown" or observation.frame_change > max_change:
            self.reset()
            return None
        self._recent.append(observation)
        if len(self._recent) < self.required_frames:
            return None
        if len({item.state for item in self._recent}) != 1:
            return None
        return min(self._recent, key=lambda item: item.state_confidence)
