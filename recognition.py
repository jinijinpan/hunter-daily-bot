from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import deque
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

    def __init__(self, engine: Any | None = None):
        self._engine = engine
        self._load_attempted = engine is not None
        self._warned = False

    @property
    def available(self) -> bool:
        self._load()
        return self._engine is not None

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        except ImportError:
            self._engine = None

    def read(
        self, image: Any, offset: tuple[int, int] = (0, 0)
    ) -> list[OCRToken]:
        self._load()
        if self._engine is None:
            if not self._warned:
                logging.warning("RapidOCR 不可用，局部 OCR 信号已禁用。")
                self._warned = True
            return []
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
        self.ocr = ocr or LocalOCR()

    @staticmethod
    def _crop(image: Any, rect: Iterable[int]) -> tuple[Any, tuple[int, int]]:
        x1, y1, x2, y2 = map(int, rect)
        x1 = max(0, min(x1, image.shape[1]))
        x2 = max(x1, min(x2, image.shape[1]))
        y1 = max(0, min(y1, image.shape[0]))
        y2 = max(y1, min(y2, image.shape[0]))
        return image[y1:y2, x1:x2], (x1, y1)

    def _read_region(self, image: Any, rect: Iterable[int]) -> list[OCRToken]:
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
            return [
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
        return self.ocr.read(crop, offset)

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
        return list(deduplicated.values()), all_tokens

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
        best_state = "unknown"
        best_confidence = 0.0
        all_signals: dict[str, dict[str, float]] = {}
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
            if confidence > best_confidence:
                best_state = state
                best_confidence = confidence
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
    def __init__(self, required_frames: int = 2, max_frame_change: float = 3.0):
        self.required_frames = max(2, int(required_frames))
        self.max_frame_change = float(max_frame_change)
        self._recent: deque[Observation] = deque(maxlen=self.required_frames)

    def update(self, observation: Observation) -> Observation | None:
        if observation.state == "unknown" or observation.frame_change > self.max_frame_change:
            self._recent.clear()
            return None
        self._recent.append(observation)
        if len(self._recent) < self.required_frames:
            return None
        if len({item.state for item in self._recent}) != 1:
            return None
        return min(self._recent, key=lambda item: item.state_confidence)
