from __future__ import annotations

import argparse
import ctypes
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
RUNS_DIR = ROOT / "runs"


class SafetyStop(RuntimeError):
    """Raised when the visible game state is not safe to automate."""


class PageTimeout(SafetyStop):
    """Raised when an expected page does not appear before its deadline."""


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class ReferenceGeometry:
    reference_width: int
    reference_height: int
    window: Rect
    reference_content_top: int = 0
    window_content_top: int = 0

    def _map_y_to_screen(self, y: int) -> int:
        if self.reference_content_top <= 0 or y < self.reference_content_top:
            chrome_height = self.window_content_top or self.window.height
            reference_height = self.reference_content_top or self.reference_height
            return self.window.top + round(y * chrome_height / reference_height)
        content_height = self.window.height - self.window_content_top
        reference_height = self.reference_height - self.reference_content_top
        return self.window.top + self.window_content_top + round(
            (y - self.reference_content_top) * content_height / reference_height
        )

    def _map_y_to_reference(self, y: int) -> int:
        relative_y = y - self.window.top
        if self.window_content_top <= 0 or relative_y < self.window_content_top:
            chrome_height = self.window_content_top or self.window.height
            reference_height = self.reference_content_top or self.reference_height
            return round(relative_y * reference_height / chrome_height)
        content_height = self.window.height - self.window_content_top
        reference_height = self.reference_height - self.reference_content_top
        return self.reference_content_top + round(
            (relative_y - self.window_content_top) * reference_height / content_height
        )

    def point(self, point: Iterable[int]) -> tuple[int, int]:
        x, y = point
        scaled_x = self.window.left + round(x * self.window.width / self.reference_width)
        scaled_y = self._map_y_to_screen(y)
        return scaled_x, scaled_y

    def reference_point(self, point: Iterable[int]) -> tuple[int, int]:
        x, y = point
        reference_x = round(
            (x - self.window.left) * self.reference_width / self.window.width
        )
        reference_y = self._map_y_to_reference(y)
        return reference_x, reference_y

    def contains(self, point: Iterable[int]) -> bool:
        x, y = point
        return (
            self.window.left <= x < self.window.right
            and self.window.top <= y < self.window.bottom
        )


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def resolve_config_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_runtime_dependencies():
    try:
        import cv2
        import mss
        import numpy as np
        import pyautogui
        import win32con
        import win32gui
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "缺少运行依赖。请先执行：python -m pip install -r requirements.txt\n"
            f"原始错误：{exc}"
        ) from exc
    return cv2, mss, np, pyautogui, win32con, win32gui, Image


class DesktopGame:
    def __init__(self, config: dict[str, Any], execute: bool, run_dir: Path):
        (
            self.cv2,
            self.mss_module,
            self.np,
            self.pyautogui,
            self.win32con,
            self.win32gui,
            _,
        ) = load_runtime_dependencies()
        self.config = config
        self.execute = execute
        self.run_dir = run_dir
        self.window_handle = self._find_window()
        self.win32gui.ShowWindow(self.window_handle, self.win32con.SW_RESTORE)
        time.sleep(0.2)
        self.window_rect = self._window_rect()
        ref_width, ref_height = config["reference_size"]
        reference_content_top = config.get("reference_content_top", 0)
        self.content_top = reference_content_top
        self.geometry = ReferenceGeometry(
            ref_width,
            ref_height,
            self.window_rect,
            reference_content_top,
            self.content_top,
        )
        self.templates = self._load_templates()
        self.multi_anchor_templates = self._load_multi_anchor_templates()
        self.task_templates = self._load_task_templates()
        self.task_progress_templates = self._load_task_progress_templates()
        self.activity_templates = self._load_activity_templates()

    def _find_window(self) -> int:
        title_fragment = self.config["window_title_contains"]
        matches: list[tuple[int, str]] = []

        def collect(handle: int, _extra: object) -> None:
            if not self.win32gui.IsWindowVisible(handle):
                return
            title = self.win32gui.GetWindowText(handle)
            if title_fragment in title:
                matches.append((handle, title))

        self.win32gui.EnumWindows(collect, None)
        if not matches:
            raise SafetyStop(
                f"没有找到标题含“{title_fragment}”的可见窗口。请先在微信中打开小游戏。"
            )
        if len(matches) > 1:
            titles = "、".join(title for _, title in matches)
            raise SafetyStop(f"找到多个候选窗口，请只保留一个：{titles}")
        logging.info("目标窗口：%s", matches[0][1])
        return matches[0][0]

    def _window_rect(self) -> Rect:
        client_left, client_top, client_right, client_bottom = self.win32gui.GetClientRect(
            self.window_handle
        )
        left, top = self.win32gui.ClientToScreen(
            self.window_handle, (client_left, client_top)
        )
        right, bottom = self.win32gui.ClientToScreen(
            self.window_handle, (client_right, client_bottom)
        )
        rect = Rect(left, top, right, bottom)
        if rect.width < 600 or rect.height < 400:
            raise SafetyStop(f"小游戏窗口尺寸异常：{rect.width}x{rect.height}")
        return rect

    def focus(self, *, force: bool = False) -> None:
        if not self.execute and not force:
            return
        self.win32gui.ShowWindow(self.window_handle, self.win32con.SW_RESTORE)
        if self.win32gui.GetForegroundWindow() == self.window_handle:
            return
        try:
            self.win32gui.SetForegroundWindow(self.window_handle)
        except Exception as exc:
            raise SafetyStop("无法激活小游戏窗口，请手动点一下窗口后重试。") from exc
        time.sleep(0.4)

    def capture(self):
        self.window_rect = self._window_rect()
        monitor = {
            "left": self.window_rect.left,
            "top": self.window_rect.top,
            "width": self.window_rect.width,
            "height": self.window_rect.height,
        }
        with self.mss_module.MSS() as screen:
            shot = self.np.array(screen.grab(monitor))
        image = self.cv2.cvtColor(shot, self.cv2.COLOR_BGRA2BGR)
        self.content_top = self._detect_content_top(image)
        ref_width, ref_height = self.config["reference_size"]
        self.geometry = ReferenceGeometry(
            ref_width,
            ref_height,
            self.window_rect,
            self.config.get("reference_content_top", 0),
            self.content_top,
        )
        return image

    def _detect_content_top(self, image) -> int:
        reference_top = self.config.get("reference_content_top", 0)
        if reference_top <= 0:
            return 0
        start, end = self.config.get("content_top_search_range", [30, 90])
        end = min(int(end), image.shape[0] - 1)
        start = max(1, min(int(start), end))
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        row_means = gray.mean(axis=1)
        drops = row_means[start : end + 1] - row_means[start - 1 : end]
        content_top = start + int(self.np.argmin(drops))
        logging.debug("检测到游戏内容顶部：%d", content_top)
        return content_top

    def normalized_capture(self):
        image = self.capture()
        width, height = self.config["reference_size"]
        reference_top = self.config.get("reference_content_top", 0)
        if reference_top <= 0 or self.content_top <= 0:
            return self.cv2.resize(
                image, (width, height), interpolation=self.cv2.INTER_AREA
            )
        chrome = self.cv2.resize(
            image[: self.content_top],
            (width, reference_top),
            interpolation=self.cv2.INTER_AREA,
        )
        content = self.cv2.resize(
            image[self.content_top :],
            (width, height - reference_top),
            interpolation=self.cv2.INTER_AREA,
        )
        return self.np.vstack((chrome, content))

    def _load_templates(self) -> dict[str, Any]:
        templates: dict[str, Any] = {}
        for page in self.config["anchors"]:
            path = ASSETS_DIR / f"anchor_{page}.png"
            if not path.exists():
                raise SafetyStop(
                    f"缺少页面模板 {path.name}。请先运行：python bot.py prepare"
                )
            data = self.np.fromfile(path, dtype=self.np.uint8)
            template = self.cv2.imdecode(data, self.cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise SafetyStop(f"无法读取页面模板：{path}")
            templates[page] = template
        return templates

    def _load_multi_anchor_templates(self) -> dict[str, list[Any]]:
        templates: dict[str, list[Any]] = {}
        for page, regions in self.config.get("page_multi_anchors", {}).items():
            page_templates = []
            for index, _region in enumerate(regions):
                path = ASSETS_DIR / f"anchor_{page}_{index}.png"
                if not path.exists():
                    raise SafetyStop(
                        f"缺少页面联合模板 {path.name}。请先运行：python bot.py prepare"
                    )
                data = self.np.fromfile(path, dtype=self.np.uint8)
                template = self.cv2.imdecode(data, self.cv2.IMREAD_GRAYSCALE)
                if template is None:
                    raise SafetyStop(f"无法读取页面联合模板：{path}")
                page_templates.append(template)
            templates[page] = page_templates
        return templates

    def _load_task_templates(self) -> dict[str, Any]:
        templates: dict[str, Any] = {}
        for name in self.config.get("task_templates", {}):
            path = ASSETS_DIR / f"task_{name}.png"
            if not path.exists():
                raise SafetyStop(
                    f"缺少任务图标模板 {path.name}。请先运行：python bot.py prepare"
                )
            data = self.np.fromfile(path, dtype=self.np.uint8)
            template = self.cv2.imdecode(data, self.cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise SafetyStop(f"无法读取任务图标模板：{path}")
            templates[name] = template
        return templates

    def _load_task_progress_templates(self) -> dict[str, Any]:
        templates: dict[str, Any] = {}
        for name in self.config.get("task_progress_templates", {}):
            path = ASSETS_DIR / f"progress_{name}.png"
            if not path.exists():
                raise SafetyStop(
                    f"缺少任务进度模板 {path.name}。请先运行：python bot.py prepare"
                )
            data = self.np.fromfile(path, dtype=self.np.uint8)
            template = self.cv2.imdecode(data, self.cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise SafetyStop(f"无法读取任务进度模板：{path}")
            templates[name] = template
        return templates

    def _load_activity_templates(self) -> dict[str, Any]:
        templates: dict[str, Any] = {}
        for name in self.config.get("activity_templates", {}):
            path = ASSETS_DIR / f"activity_{name}.png"
            if not path.exists():
                raise SafetyStop(
                    f"缺少活跃度宝箱模板 {path.name}。请先运行：python bot.py prepare"
                )
            data = self.np.fromfile(path, dtype=self.np.uint8)
            template = self.cv2.imdecode(data, self.cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise SafetyStop(f"无法读取活跃度宝箱模板：{path}")
            templates[name] = template
        return templates

    def detect_page(self, image=None) -> tuple[str, dict[str, float]]:
        if image is None:
            image = self.normalized_capture()
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        scores: dict[str, float] = {}
        margin_x, margin_y = self.config.get("page_search_margin", [0, 0])
        for page, coordinates in self.config["anchors"].items():
            x1, y1, x2, y2 = coordinates
            template = self.templates[page]
            search_x1 = max(0, x1 - margin_x)
            search_y1 = max(0, y1 - margin_y)
            search_x2 = min(gray.shape[1], x2 + margin_x)
            search_y2 = min(gray.shape[0], y2 + margin_y)
            current = gray[search_y1:search_y2, search_x1:search_x2]
            if (
                current.shape[0] < template.shape[0]
                or current.shape[1] < template.shape[1]
            ):
                scores[page] = 0.0
                continue
            result = self.cv2.matchTemplate(
                current, template, self.cv2.TM_CCOEFF_NORMED
            )
            score = self.cv2.minMaxLoc(result)[1]
            numeric_score = float(score)
            scores[page] = numeric_score if math.isfinite(numeric_score) else 0.0

        for page, regions in self.config.get("page_multi_anchors", {}).items():
            templates = getattr(self, "multi_anchor_templates", {}).get(page, [])
            if len(templates) != len(regions):
                continue
            anchor_scores = []
            for coordinates, template in zip(regions, templates):
                x1, y1, x2, y2 = coordinates
                search_x1 = max(0, x1 - margin_x)
                search_y1 = max(0, y1 - margin_y)
                search_x2 = min(gray.shape[1], x2 + margin_x)
                search_y2 = min(gray.shape[0], y2 + margin_y)
                current = gray[search_y1:search_y2, search_x1:search_x2]
                if (
                    current.shape[0] < template.shape[0]
                    or current.shape[1] < template.shape[1]
                ):
                    anchor_scores.append(0.0)
                    continue
                score = float(
                    self.cv2.minMaxLoc(
                        self.cv2.matchTemplate(
                            current, template, self.cv2.TM_CCOEFF_NORMED
                        )
                    )[1]
                )
                anchor_scores.append(score if math.isfinite(score) else 0.0)

            required = int(
                self.config.get("page_multi_anchor_required", {}).get(page, 1)
            )
            anchor_threshold = self.config.get(
                "page_multi_anchor_thresholds", {}
            ).get(page, self.config.get("page_multi_anchor_threshold", 0.7))
            passing = sorted(
                (score for score in anchor_scores if score >= anchor_threshold),
                reverse=True,
            )
            scores[page] = (
                sum(passing[:required]) / required
                if required > 0 and len(passing) >= required
                else 0.0
            )
            logging.debug("页面 %s 联合锚点分数：%s", page, anchor_scores)
        return self._select_page_from_scores(scores), scores

    def _page_threshold(self, page: str) -> float:
        return float(
            self.config.get("page_thresholds", {}).get(
                page, self.config["page_match_threshold"]
            )
        )

    def _passing_pages(
        self, scores: dict[str, float], pages: Iterable[str] | None = None
    ) -> list[str]:
        candidates = scores if pages is None else pages
        return [
            page
            for page in candidates
            if scores.get(page, 0.0) >= self._page_threshold(page)
        ]

    def _select_page_from_scores(self, scores: dict[str, float]) -> str:
        passing = self._passing_pages(scores)
        if not passing:
            return "unknown"

        background_pages = set(self.config.get("background_pages", []))
        foreground = [page for page in passing if page not in background_pages]
        candidates = foreground or passing
        return max(candidates, key=lambda page: scores[page])

    def _expected_page_from_scores(
        self, expected: Iterable[str], scores: dict[str, float]
    ) -> str | None:
        passing = self._passing_pages(scores, expected)
        background_pages = set(self.config.get("background_pages", []))
        if any(page in background_pages for page in passing):
            has_foreground = any(
                page not in background_pages for page in self._passing_pages(scores)
            )
            if has_foreground:
                passing = [page for page in passing if page not in background_pages]
        return max(passing, key=lambda page: scores[page]) if passing else None

    def wait_for_page(
        self,
        expected: str,
        *,
        timeout: float | None = None,
        tolerate_unknown: bool = False,
    ) -> None:
        timeout = timeout or self.config["timeouts"]["page_seconds"]
        poll = self.config["timeouts"]["poll_seconds"]
        deadline = time.monotonic() + timeout
        last_scores: dict[str, float] = {}
        while time.monotonic() < deadline:
            page, last_scores = self.detect_page()
            matched = self._expected_page_from_scores({expected}, last_scores)
            if matched is not None:
                logging.info(
                    "已进入页面：%s；上下文匹配分数 %.3f",
                    expected,
                    last_scores[expected],
                )
                return
            if page == "unknown" and not tolerate_unknown:
                self.save_diagnostic("unknown-page")
                raise SafetyStop(
                    "出现未识别页面或确认弹窗，已停止操作。"
                    f"匹配分数：{last_scores}"
                )
            time.sleep(poll)
        self.save_diagnostic(f"timeout-{expected}")
        raise PageTimeout(f"等待页面 {expected} 超时。最后匹配分数：{last_scores}")

    def wait_for_one_of(
        self,
        expected: Iterable[str],
        *,
        timeout: float | None = None,
        tolerate_unknown: bool = False,
    ) -> str:
        expected_pages = set(expected)
        timeout = timeout or self.config["timeouts"]["page_seconds"]
        poll = self.config["timeouts"]["poll_seconds"]
        deadline = time.monotonic() + timeout
        last_scores: dict[str, float] = {}
        while time.monotonic() < deadline:
            page, last_scores = self.detect_page()
            matched = self._expected_page_from_scores(expected_pages, last_scores)
            if matched is not None:
                logging.info(
                    "已进入页面：%s；上下文匹配分数 %.3f",
                    matched,
                    last_scores[matched],
                )
                return matched
            if page == "unknown" and not tolerate_unknown:
                self.save_diagnostic("unknown-page")
                raise SafetyStop(
                    "出现未识别页面或确认弹窗，已停止操作。"
                    f"匹配分数：{last_scores}"
                )
            time.sleep(poll)
        expected_text = "、".join(sorted(expected_pages))
        self.save_diagnostic(f"timeout-{expected_text}")
        raise PageTimeout(f"等待页面 {expected_text} 超时。最后匹配分数：{last_scores}")

    def click_reference(
        self,
        point: Iterable[int],
        label: str,
        *,
        settle_seconds: float | None = None,
    ) -> None:
        absolute = self.geometry.point(point)
        logging.info("%s：参考坐标 %s，屏幕坐标 %s", label, tuple(point), absolute)
        if not self.execute:
            return
        self.focus()
        self.pyautogui.click(*absolute)
        delay = (
            self.config["timeouts"]["after_click_seconds"]
            if settle_seconds is None
            else settle_seconds
        )
        time.sleep(delay)

    def drag_reference(
        self, start: Iterable[int], end: Iterable[int], duration: float, label: str
    ) -> None:
        absolute_start = self.geometry.point(start)
        absolute_end = self.geometry.point(end)
        logging.info("%s：%s -> %s", label, absolute_start, absolute_end)
        if not self.execute:
            return
        self.focus()
        self.pyautogui.moveTo(*absolute_start, duration=0.2)
        self.pyautogui.dragTo(*absolute_end, duration=duration, button="left")
        time.sleep(self.config["timeouts"]["after_click_seconds"])

    def active_button(self, region: Iterable[int], image=None) -> bool:
        if image is None:
            image = self.normalized_capture()
        x1, y1, x2, y2 = region
        crop = image[y1:y2, x1:x2]
        hsv = self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        active_pixels = (saturation > 80) & (value > 105)
        return float(active_pixels.mean()) > 0.18

    def gold_button(self, region: Iterable[int], image=None) -> bool:
        if image is None:
            image = self.normalized_capture()
        x1, y1, x2, y2 = region
        crop = image[y1:y2, x1:x2]
        hsv = self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        gold_pixels = (hue >= 12) & (hue <= 38) & (saturation > 80) & (value > 105)
        return float(gold_pixels.mean()) > 0.18

    def red_indicator(self, region: Iterable[int], image=None) -> bool:
        if image is None:
            image = self.normalized_capture()
        x1, y1, x2, y2 = region
        crop = image[y1:y2, x1:x2]
        hsv = self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        red_pixels = (
            ((hue <= 7) | (hue >= 173))
            & (saturation > 120)
            & (value > 120)
        )
        return float(red_pixels.mean()) > 0.01

    def green_indicator(self, region: Iterable[int], image=None) -> bool:
        if image is None:
            image = self.normalized_capture()
        x1, y1, x2, y2 = region
        crop = image[y1:y2, x1:x2]
        hsv = self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        green_pixels = (hue >= 35) & (hue <= 90) & (saturation > 80) & (value > 90)
        return float(green_pixels.mean()) > 0.05

    def find_task(self, name: str, image=None) -> tuple[int, int, float] | None:
        if image is None:
            image = self.normalized_capture()
        template = self.task_templates[name]
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = self.config["task_search_region"]
        region = gray[y1:y2, x1:x2]
        result = self.cv2.matchTemplate(region, template, self.cv2.TM_CCOEFF_NORMED)
        _, score, _, location = self.cv2.minMaxLoc(result)
        threshold = self.config.get("task_search_thresholds", {}).get(
            name, self.config["task_search_threshold"]
        )
        if score < threshold:
            return None
        center_x = x1 + location[0] + template.shape[1] // 2
        center_y = y1 + location[1] + template.shape[0] // 2
        return center_x, center_y, float(score)

    def task_progress_complete(self, name: str, center: tuple[int, int], image=None) -> bool:
        if image is None:
            image = self.normalized_capture()
        template_name = self.config.get("task_completion_progress", {}).get(name, "one")
        template = self.task_progress_templates[template_name]
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        icon_x, icon_y = center
        button_x = self.config["task_button_x"]["left" if icon_x < 550 else "right"]
        width = template.shape[1]
        height = template.shape[0]
        search_x1 = max(0, button_x + 15)
        search_y1 = max(0, icon_y - 50)
        search_x2 = min(gray.shape[1], button_x + 85)
        search_y2 = min(gray.shape[0], icon_y + 5)
        region = gray[search_y1:search_y2, search_x1:search_x2]
        if region.shape[0] < height or region.shape[1] < width:
            return False
        score = float(
            self.cv2.minMaxLoc(
                self.cv2.matchTemplate(region, template, self.cv2.TM_CCOEFF_NORMED)
            )[1]
        )
        threshold = self.config.get("task_progress_thresholds", {}).get(
            template_name, self.config["task_progress_threshold"]
        )
        logging.info("任务 %s 进度模板 %s 匹配 %.3f", name, template_name, score)
        return score >= threshold

    def active_character_index(self, image=None) -> int:
        if image is None:
            image = self.normalized_capture()
        for index, region in enumerate(self.config["character_online_regions"]):
            if self.green_indicator(region, image):
                return index
        raise SafetyStop("切换角色页未识别到在线角色，已停止。")

    def activity_chest_score(self, name: str, image=None) -> float:
        if image is None:
            image = self.normalized_capture()
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = self.config["activity_templates"][name]["region"]
        current = gray[y1:y2, x1:x2]
        template = self.activity_templates[name]
        if current.shape != template.shape:
            return 0.0
        score = float(
            self.cv2.matchTemplate(current, template, self.cv2.TM_CCOEFF_NORMED)[0][0]
        )
        return score if math.isfinite(score) else 0.0

    def save_diagnostic(self, name: str) -> Path:
        path = self.run_dir / f"{name}.png"
        image = self.capture()
        encoded, buffer = self.cv2.imencode(".png", image)
        if not encoded:
            raise SafetyStop("无法编码诊断截图。")
        buffer.tofile(path)
        logging.info("保存截图：%s", path)
        return path


class DailyBot:
    SUPPORTED_TASK_ADAPTERS = frozenset(
        {
            "tower",
            "hunter_field",
            "resource_supply",
            "abyss",
            "monster_invasion",
            "hunter_league",
            "infinite_mystery",
        }
    )
    EXCLUDED_TASK_ADAPTERS = frozenset({"ladder"})

    def __init__(
        self,
        game: DesktopGame,
        config: dict[str, Any],
        *,
        resume: bool = False,
    ):
        self.game = game
        self.config = config
        self.resume = resume

    def run(self) -> None:
        self._configured_task_adapters()
        self.game.focus()
        page, scores = self.game.detect_page()
        logging.info("当前页面：%s；匹配分数：%s", page, scores)
        resumable_pages = {
            "main",
            "trial",
            "tower_result",
            "tower_changed",
            "tower_manual",
            "tower_battle_confirm",
            "tower_post_battle",
            "hunter_field",
            "hunter_quick_available",
            "hunter_failure",
            "hunter_confirm",
            "resource_dialog",
            "resource_confirm",
            "abyss",
            "abyss_victory",
            "abyss_cards",
            "abyss_finished",
            "abyss_exhausted",
            "monster_invasion",
            "monster_match",
            "monster_result",
            "monster_reward",
            "hunter_league",
            "hunter_league_victory",
            "hunter_league_failure",
            "hunter_league_rewards",
            "hunter_league_challenge_rewards",
            "infinite_rank_drop",
            "infinite_mystery",
            "infinite_map",
            "infinite_stage",
            "infinite_score",
            "infinite_next",
            "infinite_finished",
            "character_switch",
        }
        if page not in resumable_pages:
            self.game.save_diagnostic(f"startup-{page}")
            raise SafetyStop("请先把小游戏停在主界面或已识别的断点页面，再运行脚本。")

        if not self.game.execute:
            if page != "main":
                logging.info("干运行：将从断点页面 %s 继续后续任务。", page)
                return
            self._dry_run()
            return

        if page == "tower_result":
            self._finish_tower_quick_result()
            self._return_home()
        elif page == "tower_changed":
            self._resume_tower_changed()
        elif page == "tower_post_battle":
            self._exit_tower_post_battle()
            self._return_home()
        elif page == "tower_manual":
            self._run_tower_manual()
            self._return_home()
        elif page == "tower_battle_confirm":
            self._finish_tower_manual_battle()
            self._return_home()
        elif page == "trial":
            self._return_home()
        elif page in {"hunter_field", "hunter_quick_available"}:
            self._run_hunter_field(page)
            self._return_home()
        elif page == "hunter_failure":
            self._finish_hunter_failure()
            self._return_home()
        elif page == "hunter_confirm":
            self._finish_hunter_confirm()
            self._return_home()
        elif page == "resource_confirm":
            self._finish_resource_supply(page)
            self._return_home()
        elif page == "resource_dialog":
            self._run_resource_quick()
            self._return_home()
        elif page == "abyss":
            logging.info("从深渊挑战入口继续消耗剩余体力。")
            self._run_abyss()
            self._return_home()
        elif page in {"abyss_victory", "abyss_cards", "abyss_finished"}:
            logging.info("从深渊挑战断点页面 %s 继续结算。", page)
            self._run_abyss(initial_page=page)
            self._return_home()
        elif page == "abyss_exhausted":
            logging.info("启动时检测到深渊体力已耗尽，返回安全区。")
            self._finish_abyss_exhausted()
            self._return_home()
        elif page == "monster_invasion":
            self._run_monster_invasion()
            self._return_home()
        elif page == "monster_match":
            self._resume_monster_match()
            self._return_home()
        elif page == "monster_result":
            self._finish_monster_result()
            self._return_home()
        elif page == "monster_reward":
            self._finish_monster_reward()
            self._return_home()
        elif page == "hunter_league":
            self._run_hunter_league()
            self._return_home()
        elif page in {"hunter_league_victory", "hunter_league_failure"}:
            self._finish_hunter_league_result()
            self._return_home()
        elif page in {"hunter_league_rewards", "hunter_league_challenge_rewards"}:
            self._close_hunter_league_rewards(page)
            self._return_home()
        elif page == "infinite_rank_drop":
            self._dismiss_infinite_rank_drop()
            self._run_infinite_mystery()
            self._return_home()
        elif page == "infinite_mystery":
            self._run_infinite_mystery()
            self._return_home()
        elif page == "infinite_map":
            self._run_infinite_from_map()
            self._return_home()
        elif page == "infinite_stage":
            self._run_infinite_from_stage()
            self._return_home()
        elif page in {"infinite_score", "infinite_next", "infinite_finished"}:
            self._finish_infinite_mystery(page)
            self._return_home()
        elif page == "character_switch":
            self._switch_to_next_character(already_open=True)
        else:
            if self.resume:
                logging.info("续跑模式：跳过体力和每日补给，直接继续每日任务。")
            else:
                self._reclaim_free_stamina()
                self._collect_daily_supply()
        self._run_character_cycles()
        self.game.save_diagnostic("tasks-finished")
        self._report_unimplemented_tasks()

    def _dry_run(self) -> None:
        logging.info("干运行：将执行免费体力找回、每日补给、已捕获任务和任务领奖。")
        logging.info(
            "干运行：角色循环 %d 个角色。", self.config.get("character_cycle_count", 1)
        )
        for name in self._configured_task_adapters():
            logging.info("干运行：将按图标定位并执行任务适配器 %s。", name)
        logging.info(
            "干运行：将跳过未录制任务：%s",
            "、".join(self.config["skipped_task_names"]),
        )

    def _reclaim_free_stamina(self) -> None:
        points = self.config["points"]
        self.game.click_reference(points["main_stamina"], "打开体力获取")
        self.game.wait_for_page("stamina_get", tolerate_unknown=True)
        self.game.click_reference(points["stamina_reclaim"], "打开免费体力找回")
        self.game.wait_for_page("stamina_recovery", tolerate_unknown=True)
        self.game.click_reference(points["stamina_one_key"], "一键找回免费体力")
        page = self.game.wait_for_one_of(
            {"reward", "stamina_recovery"}, tolerate_unknown=True
        )
        if page == "reward":
            self._dismiss_reward("stamina_recovery")
        self.game.click_reference(points["popup_close"], "关闭体力找回")
        self.game.wait_for_page("stamina_get", tolerate_unknown=True)
        self.game.click_reference(points["popup_close"], "关闭体力获取")
        self.game.wait_for_page("main", tolerate_unknown=True)

    def _collect_daily_supply(self) -> None:
        points = self.config["points"]
        self.game.click_reference(points["main_welfare"], "打开福利")
        self.game.wait_for_page("welfare", tolerate_unknown=True)
        self.game.click_reference(points["welfare_daily_supply"], "打开每日补给")
        self.game.wait_for_page("supply", tolerate_unknown=True)

        regions = self.config["supply_claim_regions"]
        stop_scanning = False
        for pass_index in range(self.config["supply_max_passes"]):
            image = self.game.normalized_capture()
            active = [region for region in regions if self.game.active_button(region, image)]
            if not active:
                logging.info("每日补给页面已无可领取按钮。")
                break
            logging.info("第 %d 轮发现 %d 个可领取/补领按钮。", pass_index + 1, len(active))
            for region in active:
                x1, y1, x2, y2 = region
                self.game.click_reference(
                    ((x1 + x2) // 2, (y1 + y2) // 2), "领取每日补给"
                )
                page = self.game.wait_for_one_of(
                    {"supply", "reward"}, tolerate_unknown=True
                )
                if page == "reward":
                    self._dismiss_reward("supply")
                else:
                    logging.info("补给按钮未弹出奖励，结束本次补给扫描。")
                    stop_scanning = True
                    break
            if stop_scanning:
                break
        else:
            logging.warning("达到补给领取最大轮数，请根据运行截图确认是否领完。")

        self.game.save_diagnostic("supply-finished")
        self.game.click_reference(points["home"], "返回主界面")
        self.game.wait_for_page("main", tolerate_unknown=True)

    def _open_daily_tasks(self) -> None:
        self.game.click_reference(self.config["points"]["main_secretary"], "打开小秘书")
        self.game.wait_for_page("tasks", tolerate_unknown=True)

    def _reset_task_scroll_to_top(self) -> None:
        scroll = self.config["task_scroll"]
        for _ in range(int(scroll.get("reset_passes", 2))):
            self.game.drag_reference(
                scroll["to"],
                scroll["from"],
                scroll["duration_seconds"],
                "重置小秘书任务列表到顶部",
            )
            self.game.wait_for_page("tasks", tolerate_unknown=True)

    def _find_task_button(self, task: str) -> tuple[tuple[int, int], bool] | None:
        scroll = self.config["task_scroll"]
        self._reset_task_scroll_to_top()
        search_passes = int(scroll.get("search_passes", 3))
        for attempt in range(search_passes):
            image = self.game.normalized_capture()
            found = self.game.find_task(task, image)
            if found is not None:
                icon_x, icon_y, score = found
                button_x = self.config["task_button_x"]["left" if icon_x < 550 else "right"]
                region = [button_x - 65, icon_y - 20, button_x + 65, icon_y + 20]
                completed = self.game.gold_button(region, image)
                if not completed:
                    completed = self.game.task_progress_complete(
                        task, (icon_x, icon_y), image
                    )
                logging.info(
                    "任务 %s 图标匹配 %.3f，按钮坐标 %s，已完成=%s",
                    task,
                    score,
                    (button_x, icon_y),
                    completed,
                )
                return (button_x, icon_y), completed
            if attempt + 1 < search_passes:
                self.game.drag_reference(
                    scroll["from"],
                    scroll["to"],
                    scroll["duration_seconds"],
                    f"滚动查找任务 {task}",
                )
                self.game.wait_for_page("tasks", tolerate_unknown=True)
        logging.info("当前任务列表中未找到任务：%s", task)
        return None

    def _run_character_cycles(self) -> None:
        cycle_count = int(self.config.get("character_cycle_count", 1))
        available_count = len(self.config["character_row_points"])
        if not 1 <= cycle_count <= available_count:
            raise SafetyStop(
                f"角色循环数量必须在 1 到 {available_count} 之间，当前为 {cycle_count}。"
            )
        for role_index in range(cycle_count):
            logging.info("开始处理第 %d/%d 个角色的小秘书任务。", role_index + 1, cycle_count)
            self._run_captured_tasks()
            self._open_daily_tasks()
            self._claim_task_rewards()
            self.game.save_diagnostic(f"tasks-finished-role-{role_index + 1}")
            self._return_home()
            if role_index + 1 < cycle_count:
                self._switch_to_next_character()

    def _switch_to_next_character(self, *, already_open: bool = False) -> None:
        points = self.config["points"]
        if not already_open:
            self.game.click_reference(points["character_switch"], "打开切换角色")
            self.game.wait_for_page("character_switch", tolerate_unknown=True)
        current_index = self.game.active_character_index()
        rows = self.config["character_row_points"]
        next_index = (current_index + 1) % len(rows)
        logging.info("当前角色行 %d，切换到角色行 %d。", current_index + 1, next_index + 1)
        self.game.click_reference(rows[next_index], "选择下一个角色")
        self.game.wait_for_page("character_switch", tolerate_unknown=True)
        self.game.click_reference(points["character_start"], "开始游戏")
        self.game.wait_for_page(
            "main",
            timeout=self.config["timeouts"].get("character_switch_seconds", 45),
            tolerate_unknown=True,
        )

    def _open_task(self, task: str, page: str) -> bool:
        task_button = self._find_task_button(task)
        if task_button is None:
            return False
        point, completed = task_button
        if completed:
            logging.info("任务 %s 已完成，等待统一领奖。", task)
            return False
        self.game.click_reference(point, f"前往任务 {task}")
        self.game.wait_for_page(page, tolerate_unknown=True)
        return True

    def _open_tower_task(self) -> str | None:
        task_button = self._find_task_button("tower")
        if task_button is None:
            return None
        point, completed = task_button
        if completed:
            logging.info("任务 tower 已完成，等待统一领奖。")
            return None
        self.game.click_reference(point, "前往任务 tower")
        page = self.game.wait_for_one_of(
            {"tower", "tower_changed", "tower_manual"}, tolerate_unknown=True
        )
        if page == "tower_changed":
            self.game.click_reference(
                self.config["points"]["tower_changed_confirm"],
                "确认无尽塔难度变化",
            )
            page = self.game.wait_for_one_of(
                {"tower", "tower_manual"}, tolerate_unknown=True
            )
        return page

    def _open_hunter_task(self) -> str | None:
        task_button = self._find_task_button("hunter_field")
        if task_button is None:
            return None
        point, completed = task_button
        if completed:
            logging.info("任务 hunter_field 已完成，等待统一领奖。")
            return None
        self.game.click_reference(point, "前往任务 hunter_field")
        return self.game.wait_for_one_of(
            {"hunter_field", "hunter_quick_available"}, tolerate_unknown=True
        )

    def _run_captured_tasks(self) -> None:
        for task in self._configured_task_adapters():
            self._open_daily_tasks()
            started = False
            if task == "tower":
                tower_mode = self._open_tower_task()
                started = tower_mode is not None
                if tower_mode == "tower":
                    self._run_tower()
                elif tower_mode == "tower_manual":
                    self._run_tower_manual()
            elif task == "hunter_field":
                hunter_mode = self._open_hunter_task()
                started = hunter_mode is not None
                if hunter_mode is not None:
                    self._run_hunter_field(hunter_mode)
            elif task == "resource_supply":
                started = self._open_task(task, "resource_hub")
                if started:
                    self._run_resource_supply()
            elif task == "abyss":
                started = self._open_task(task, "abyss")
                if started:
                    self._run_abyss()
            elif task == "monster_invasion":
                started = self._open_task(task, "monster_invasion")
                if started:
                    self._run_monster_invasion()
            elif task == "hunter_league":
                started = self._open_task(task, "hunter_league")
                if started:
                    self._run_hunter_league()
            elif task == "infinite_mystery":
                started = self._open_infinite_task()
                if started:
                    self._run_infinite_mystery()
            self._return_home()

    def _configured_task_adapters(self) -> list[str]:
        adapters = list(self.config.get("captured_task_adapters", []))
        excluded = sorted(set(adapters) & self.EXCLUDED_TASK_ADAPTERS)
        if excluded:
            raise SafetyStop(
                "以下任务明确排除，不能加入 captured_task_adapters："
                + "、".join(excluded)
            )
        unsupported = sorted(set(adapters) - self.SUPPORTED_TASK_ADAPTERS)
        if unsupported:
            raise SafetyStop(
                "存在未实现的任务适配器：" + "、".join(unsupported)
            )
        duplicates = sorted({name for name in adapters if adapters.count(name) > 1})
        if duplicates:
            raise SafetyStop(
                "任务适配器不能重复配置：" + "、".join(duplicates)
            )
        return adapters

    def _click_until_transition(
        self,
        point: Iterable[int],
        label: str,
        source_pages: Iterable[str],
        target_pages: Iterable[str],
        *,
        timeout: float | None = None,
        settle_seconds: float | None = None,
    ) -> str:
        sources = set(source_pages)
        targets = set(target_pages)
        if sources & targets:
            raise ValueError("source_pages and target_pages must not overlap")
        max_clicks = int(self.config.get("transition_retry_max_clicks", 5))
        for attempt in range(max_clicks):
            retry = "" if attempt == 0 else f"（重试 {attempt + 1}/{max_clicks}）"
            self.game.click_reference(
                point,
                label + retry,
                settle_seconds=settle_seconds,
            )
            try:
                return self.game.wait_for_one_of(
                    targets,
                    timeout=timeout,
                    tolerate_unknown=True,
                )
            except PageTimeout as target_timeout:
                try:
                    page = self.game.wait_for_one_of(
                        sources,
                        timeout=max(
                            0.5,
                            float(self.config["timeouts"]["poll_seconds"]) * 2,
                        ),
                        tolerate_unknown=True,
                    )
                except PageTimeout:
                    raise target_timeout
            logging.warning(
                "%s 后仍停在页面 %s，将再次点击（%d/%d）。",
                label,
                page,
                attempt + 1,
                max_clicks,
            )
        raise SafetyStop(f"{label}连续点击后页面仍未切换，已停止。")

    def _run_tower(self) -> None:
        self._click_until_transition(
            self.config["points"]["tower_quick"],
            "无尽塔快速挑战",
            {"tower"},
            {"tower_result"},
        )
        self._finish_tower_quick_result()

    def _finish_tower_quick_result(self) -> None:
        self._click_until_transition(
            self.config["points"]["overlay_continue"],
            "关闭无尽塔结算",
            {"tower_result"},
            {"tower", "tower_manual"},
        )

    def _run_tower_manual(self) -> None:
        points = self.config["points"]
        self._click_until_transition(
            points["tower_start"],
            "无尽塔开始挑战",
            {"tower_manual"},
            {"tower_battle_confirm"},
        )
        self._finish_tower_manual_battle()

    def _finish_tower_manual_battle(self) -> None:
        points = self.config["points"]
        self._click_until_transition(
            points["tower_battle_confirm"],
            "确认进入无尽塔战斗",
            {"tower_battle_confirm"},
            {"reward"},
            timeout=self.config["timeouts"]["battle_seconds"],
        )
        self._click_until_transition(
            points["tower_reward_close"],
            "关闭无尽塔通关奖励",
            {"reward"},
            {"tower_post_battle"},
        )
        self._exit_tower_post_battle()

    def _exit_tower_post_battle(self) -> None:
        targets = {"tower", "tower_manual", "trial"}
        max_clicks = int(self.config.get("tower_exit_max_clicks", 5))
        retry_wait = float(self.config.get("tower_exit_retry_wait_seconds", 4))
        for attempt in range(max_clicks):
            if attempt == 0:
                point = self.config["points"]["tower_exit"]
                label = "退出无尽塔结算"
            else:
                point = self.config["points"]["tower_exit_retry"]
                label = f"无尽塔退出未完成，第 {attempt + 1} 次单击继续"
            self.game.click_reference(point, label)
            try:
                self.game.wait_for_one_of(
                    targets,
                    timeout=retry_wait,
                    tolerate_unknown=True,
                )
                return
            except PageTimeout:
                if attempt + 1 >= max_clicks:
                    raise
                logging.warning(
                    "无尽塔退出后仍未返回大厅，将继续单击（%d/%d）。",
                    attempt + 1,
                    max_clicks,
                )

    def _resume_tower_changed(self) -> None:
        self.game.click_reference(
            self.config["points"]["tower_changed_confirm"],
            "确认无尽塔难度变化",
        )
        page = self.game.wait_for_one_of(
            {"tower", "tower_manual"}, tolerate_unknown=True
        )
        if page == "tower":
            self._run_tower()
        else:
            self._run_tower_manual()
        self._return_home()

    def _run_hunter_field(self, mode: str = "hunter_field") -> None:
        points = self.config["points"]
        if mode == "hunter_quick_available":
            page = self._click_until_transition(
                points["hunter_quick"],
                "猎魔战场快速通关",
                {"hunter_quick_available"},
                {"hunter_confirm", "reward"},
            )
            if page == "hunter_confirm":
                self._finish_hunter_confirm()
                return
        else:
            page = self._click_until_transition(
                points["hunter_start"],
                "猎魔战场开始挑战",
                {"hunter_field"},
                {"hunter_failure", "reward"},
            )
            if page == "hunter_failure":
                self._finish_hunter_failure()
                return
        self._close_hunter_reward()

    def _finish_hunter_failure(self) -> None:
        self._click_until_transition(
            self.config["points"]["hunter_speed"],
            "猎魔战场速通上一关",
            {"hunter_failure"},
            {"hunter_confirm"},
        )
        self._finish_hunter_confirm()

    def _finish_hunter_confirm(self) -> None:
        self._click_until_transition(
            self.config["points"]["hunter_confirm"],
            "确认猎魔战场快速通关",
            {"hunter_confirm"},
            {"reward"},
        )
        self._close_hunter_reward()

    def _close_hunter_reward(self) -> None:
        points = self.config["points"]
        self._click_until_transition(
            points["overlay_continue"],
            "关闭猎魔战场奖励",
            {"reward"},
            {"hunter_field", "hunter_quick_available"},
        )

    def _run_resource_supply(self) -> None:
        self.game.click_reference(
            self.config["points"]["resource_magic_card"], "选择魔晶补给"
        )
        self.game.wait_for_page("resource_dialog", tolerate_unknown=True)
        self._run_resource_quick()

    def _run_resource_quick(self) -> None:
        self.game.click_reference(self.config["points"]["resource_speed"], "资源补给速通")
        page = self.game.wait_for_one_of(
            {"resource_confirm", "reward"}, tolerate_unknown=True
        )
        self._finish_resource_supply(page)

    def _finish_resource_supply(self, page: str) -> None:
        if page == "resource_confirm":
            self.game.click_reference(
                self.config["points"]["resource_confirm"],
                "确认资源补给快速通关",
            )
            self.game.wait_for_page("reward", tolerate_unknown=True)
        self._dismiss_reward("resource_dialog")
        self.game.click_reference(
            self.config["points"]["resource_close"], "关闭资源补给难度弹窗"
        )
        self.game.wait_for_page("resource_hub", tolerate_unknown=True)

    def _run_abyss(self, *, initial_page: str | None = None) -> None:
        points = self.config["points"]
        outcomes = {
            "abyss_victory",
            "abyss_cards",
            "reward",
            "abyss_finished",
            "abyss_exhausted",
            "stamina_get",
        }
        if initial_page is None:
            self.game.click_reference(points["abyss_single"], "深渊挑战单人挑战")
            page = self.game.wait_for_one_of(
                outcomes,
                timeout=self.config["timeouts"]["battle_seconds"],
                tolerate_unknown=True,
            )
        elif initial_page not in outcomes:
            raise SafetyStop(f"不支持的深渊断点页面：{initial_page}")
        else:
            page = initial_page
        max_runs = self.config["abyss_max_runs"]
        completed_runs = 0
        while completed_runs < max_runs:
            if page == "stamina_get":
                self._finish_abyss_after_stamina_exhausted()
                return
            if page == "abyss_exhausted":
                self._finish_abyss_exhausted()
                return

            if page == "abyss_victory":
                completed_runs += 1
                logging.info("深渊挑战已完成第 %d 次。", completed_runs)
                page = self._advance_abyss_page(
                    "abyss_victory",
                    points["overlay_continue"],
                    "继续深渊结算",
                    {
                        "abyss_cards",
                        "reward",
                        "abyss_finished",
                        "abyss_exhausted",
                        "stamina_get",
                    },
                )

            if page == "abyss_cards":
                page = self._advance_abyss_page(
                    "abyss_cards",
                    points["abyss_cards_skip"],
                    "跳过深渊翻牌",
                    {
                        "reward",
                        "abyss_finished",
                        "abyss_exhausted",
                        "abyss_victory",
                        "stamina_get",
                    },
                )

            if page == "reward":
                page = self._advance_abyss_page(
                    "reward",
                    points["overlay_continue"],
                    "关闭深渊奖励",
                    {
                        "abyss_finished",
                        "abyss_exhausted",
                        "abyss_victory",
                        "stamina_get",
                    },
                )

            if page in {"abyss_victory", "abyss_exhausted", "stamina_get"}:
                # The three-second countdown can start the next battle before
                # abyss_finished is sampled. Treat its result as a new run.
                continue

            if page != "abyss_finished":
                raise SafetyStop(f"深渊结算后出现未处理页面：{page}")

            if completed_runs >= max_runs:
                self.game.click_reference(
                    points["abyss_return_safe"], "达到保护上限，返回安全区"
                )
                self.game.wait_for_page("abyss", tolerate_unknown=True)
                raise SafetyStop(
                    f"深渊连续挑战达到保护上限 {max_runs} 次，已安全退出。"
                )
            logging.info("继续消耗深渊挑战剩余体力。")
            self.game.click_reference(
                points["abyss_retry"],
                "深渊再次挑战",
                settle_seconds=self.config["abyss_retry_settle_seconds"],
            )
            page = self.game.wait_for_one_of(
                outcomes,
                timeout=self.config["timeouts"]["battle_seconds"],
                tolerate_unknown=True,
            )

        raise SafetyStop(
            f"深渊连续挑战达到保护上限 {max_runs} 次，但未检测到体力不足。"
        )

    def _advance_abyss_page(
        self,
        current_page: str,
        point: Iterable[int],
        label: str,
        next_pages: set[str],
    ) -> str:
        max_clicks = self.config["abyss_page_action_max_clicks"]
        timeout = self.config["abyss_page_action_retry_seconds"]
        for attempt in range(1, max_clicks + 1):
            self.game.click_reference(point, label)
            try:
                return self.game.wait_for_one_of(
                    next_pages,
                    timeout=timeout,
                    tolerate_unknown=True,
                )
            except PageTimeout:
                logging.warning(
                    "深渊页面 %s 点击后仍未切换，第 %d/%d 次重试。",
                    current_page,
                    attempt,
                    max_clicks,
                )
        raise SafetyStop(
            f"深渊页面 {current_page} 连续点击 {max_clicks} 次后仍未切换，已停止。"
        )

    def _finish_abyss_after_stamina_exhausted(self) -> None:
        points = self.config["points"]
        logging.info("深渊挑战体力已不足，结束连续挑战。")
        self.game.click_reference(points["popup_close"], "关闭体力不足弹窗")
        page = self.game.wait_for_one_of(
            {"abyss", "abyss_finished"}, tolerate_unknown=True
        )
        if page == "abyss_finished":
            self.game.click_reference(points["abyss_return_safe"], "返回安全区")
            self.game.wait_for_page("abyss", tolerate_unknown=True)

    def _finish_abyss_exhausted(self) -> None:
        logging.info("深渊体力为 0 或只剩返回按钮，结束连续挑战。")
        self.game.click_reference(
            self.config["points"]["abyss_exhausted_return_safe"],
            "深渊体力耗尽，返回安全区",
        )
        self.game.wait_for_page("abyss", tolerate_unknown=True)

    def _run_monster_invasion(self) -> None:
        points = self.config["points"]
        for attempt in range(self.config["monster_invasion_max_attempts"]):
            self._click_until_transition(
                points["monster_challenge"],
                "魔物入侵挑战",
                {"monster_invasion"},
                {"monster_match"},
            )
            image = self.game.normalized_capture()
            if self.game.red_indicator(
                self.config["monster_exhausted_region"], image
            ):
                logging.info("魔物入侵挑战次数已用完。")
                self._close_monster_match()
                return
            self._finish_monster_match()
            logging.info("魔物入侵已完成第 %d 次。", attempt + 1)

    def _resume_monster_match(self) -> None:
        image = self.game.normalized_capture()
        if self.game.red_indicator(self.config["monster_exhausted_region"], image):
            self._close_monster_match()
            return
        self._finish_monster_match()

    def _close_monster_match(self) -> None:
        self._click_until_transition(
            self.config["points"]["monster_close"],
            "关闭魔物入侵匹配",
            {"monster_match"},
            {"monster_invasion"},
        )

    def _finish_monster_match(self) -> None:
        self._click_until_transition(
            self.config["points"]["monster_match"],
            "魔物入侵快速匹配",
            {"monster_match"},
            {"monster_result"},
            timeout=self.config["timeouts"]["battle_seconds"],
        )
        self._finish_monster_result()

    def _finish_monster_result(self) -> None:
        self._click_until_transition(
            self.config["points"]["monster_result_continue"],
            "确认魔物入侵胜负",
            {"monster_result"},
            {"monster_reward"},
        )
        self._finish_monster_reward()

    def _finish_monster_reward(self) -> None:
        self._click_until_transition(
            self.config["points"]["monster_reward_continue"],
            "关闭魔物入侵奖励",
            {"monster_reward"},
            {"monster_invasion"},
        )

    def _run_hunter_league(self) -> None:
        points = self.config["points"]
        results = {"hunter_league_victory", "hunter_league_failure"}
        for match_index in range(self.config["hunter_league_matches"]):
            self._click_until_transition(
                points["hunter_league_match"],
                "猎人联赛匹配战斗",
                {"hunter_league"},
                results,
                timeout=self.config["timeouts"]["battle_seconds"],
            )
            self._finish_hunter_league_result()
            logging.info("猎人联赛已完成第 %d 场。", match_index + 1)
        self._claim_hunter_league_rewards()

    def _finish_hunter_league_result(self) -> None:
        self._click_until_transition(
            self.config["points"]["hunter_league_result"],
            "关闭猎人联赛战果",
            {"hunter_league_victory", "hunter_league_failure"},
            {"hunter_league"},
        )

    def _close_hunter_league_rewards(self, page: str) -> None:
        self._click_until_transition(
            self.config["points"]["hunter_league_rewards_close"],
            "关闭猎人联赛奖励",
            {page},
            {"hunter_league"},
        )

    def _claim_hunter_league_rewards(self) -> None:
        points = self.config["points"]
        self.game.click_reference(
            points["hunter_league_rewards"], "打开猎人联赛奖励"
        )
        self.game.wait_for_page("hunter_league_rewards", tolerate_unknown=True)
        image = self.game.normalized_capture()
        if self.game.gold_button(self.config["hunter_league_claim_region"], image):
            self.game.click_reference(
                points["hunter_league_daily_claim"], "领取联赛每日奖励"
            )
            self._dismiss_reward("hunter_league_rewards")

        self.game.click_reference(
            points["hunter_league_challenge_tab"], "打开联赛挑战奖励"
        )
        self.game.wait_for_page("hunter_league_challenge_rewards", tolerate_unknown=True)
        image = self.game.normalized_capture()
        if self.game.gold_button(self.config["hunter_league_claim_region"], image):
            self.game.click_reference(
                points["hunter_league_challenge_claim"], "领取联赛挑战奖励"
            )
            self._dismiss_reward("hunter_league_challenge_rewards")
        self._close_hunter_league_rewards("hunter_league_challenge_rewards")

    def _open_infinite_task(self) -> bool:
        task_button = self._find_task_button("infinite_mystery")
        if task_button is None:
            return False
        point, completed = task_button
        if completed:
            logging.info("任务 infinite_mystery 已完成，等待统一领奖。")
            return False
        self.game.click_reference(point, "前往任务 infinite_mystery")
        page = self.game.wait_for_one_of(
            {"infinite_rank_drop", "infinite_mystery"}, tolerate_unknown=True
        )
        if page == "infinite_rank_drop":
            self._dismiss_infinite_rank_drop()
        return True

    def _dismiss_infinite_rank_drop(self) -> None:
        page = "infinite_rank_drop"
        for _ in range(self.config["infinite_rank_dismiss_max_clicks"]):
            if page == "infinite_mystery":
                return
            self.game.click_reference(
                self.config["points"]["infinite_rank_continue"],
                "关闭无限秘境段位结算",
            )
            page = self.game.wait_for_one_of(
                {"infinite_rank_drop", "infinite_mystery"}, tolerate_unknown=True
            )
        raise SafetyStop("无限秘境段位结算未能关闭，已停止。")

    def _run_infinite_mystery(self) -> None:
        points = self.config["points"]
        self._click_until_transition(
            points["infinite_start"],
            "无限秘境开始挑战",
            {"infinite_mystery"},
            {"infinite_map"},
        )
        self._run_infinite_from_map()

    def _run_infinite_from_map(self) -> None:
        self._click_until_transition(
            self.config["points"]["infinite_first_stage"],
            "选择无限秘境1-1",
            {"infinite_map"},
            {"infinite_stage"},
        )
        self._run_infinite_from_stage()

    def _run_infinite_from_stage(self) -> None:
        self._click_until_transition(
            self.config["points"]["infinite_stage_start"],
            "开始无限秘境关卡",
            {"infinite_stage"},
            {"infinite_score"},
            timeout=self.config["timeouts"]["battle_seconds"],
        )
        self._finish_infinite_mystery("infinite_score")

    def _finish_infinite_mystery(self, page: str) -> None:
        points = self.config["points"]
        max_scores = int(self.config["infinite_mystery_stage_count"])
        scores_closed = 0
        while True:
            if page == "infinite_score":
                scores_closed += 1
                if scores_closed > max_scores:
                    raise SafetyStop("无限秘境结算次数超过配置上限，已停止。")
                page = self._click_until_transition(
                    points["infinite_score_continue"],
                    "关闭无限秘境评分",
                    {"infinite_score"},
                    {"infinite_next", "infinite_finished"},
                )
                continue
            if page == "infinite_next":
                page = self._click_until_transition(
                    points["infinite_next"],
                    "无限秘境下一关",
                    {"infinite_next"},
                    {"infinite_score"},
                    timeout=self.config["timeouts"]["battle_seconds"],
                )
                continue
            if page == "infinite_finished":
                self._click_until_transition(
                    points["infinite_return_safe"],
                    "无限秘境返回安全区",
                    {"infinite_finished"},
                    {"infinite_mystery"},
                )
                return
            raise SafetyStop(f"无限秘境出现未处理结算页面：{page}")

    def _return_home(self) -> None:
        max_clicks = max(1, int(self.config.get("home_return_max_clicks", 3)))
        for attempt in range(max_clicks):
            label = "返回主界面"
            if attempt:
                label += f"（重试 {attempt + 1}/{max_clicks}）"
            self.game.click_reference(self.config["points"]["home"], label)
            try:
                page = self.game.wait_for_one_of(
                    {"main", "trial"}, tolerate_unknown=True
                )
                break
            except PageTimeout:
                if attempt + 1 >= max_clicks:
                    raise
                logging.warning(
                    "返回主界面未生效，将再次点击（%d/%d）。",
                    attempt + 1,
                    max_clicks,
                )
        if page == "trial":
            self.game.click_reference(
                self.config["points"]["home"], "从试炼大厅返回主界面"
            )
            self.game.wait_for_page("main", tolerate_unknown=True)

    def _dismiss_reward(self, return_page: str) -> None:
        self.game.click_reference(self.config["points"]["overlay_continue"], "关闭奖励弹窗")
        self.game.wait_for_page(return_page, tolerate_unknown=True)

    def _claim_task_rewards(self) -> None:
        self._reset_task_scroll_to_top()
        regions = self.config["reward_button_regions"]
        max_passes = self.config["task_claim_max_passes"]
        scanned_lower_list = False
        for pass_index in range(max_passes):
            image = self.game.normalized_capture()
            available = [region for region in regions if self.game.gold_button(region, image)]
            if not available:
                if not scanned_lower_list:
                    scroll = self.config["task_scroll"]
                    self.game.drag_reference(
                        scroll["from"],
                        scroll["to"],
                        scroll["duration_seconds"],
                        "滚动查找可领取任务奖励",
                    )
                    self.game.wait_for_page("tasks", tolerate_unknown=True)
                    scanned_lower_list = True
                    continue
                logging.info("当前可见任务页没有可领取奖励。")
                self._claim_activity_rewards()
                return
            x1, y1, x2, y2 = available[0]
            self.game.click_reference(
                ((x1 + x2) // 2, (y1 + y2) // 2), "领取每日任务奖励"
            )
            page = self.game.wait_for_one_of(
                {"tasks", "reward"}, tolerate_unknown=True
            )
            if page == "reward":
                self._dismiss_reward("tasks")
        logging.warning("达到任务领奖最大轮数，请检查运行截图。")

    def _claim_activity_rewards(self) -> None:
        threshold = self.config["activity_match_threshold"]
        for _ in range(len(self.config.get("activity_templates", {}))):
            image = self.game.normalized_capture()
            available = []
            for name, settings in self.config.get("activity_templates", {}).items():
                score = self.game.activity_chest_score(name, image)
                if score >= threshold:
                    available.append((int(name), score, settings["point"]))
            if not available:
                logging.info("没有可领取的活跃度宝箱。")
                return
            value, score, point = max(available)
            logging.info("领取活跃度 %d 宝箱，模板分数 %.3f", value, score)
            self.game.click_reference(point, f"领取活跃度 {value} 宝箱")
            page = self.game.wait_for_one_of(
                {"tasks", "reward"}, tolerate_unknown=True
            )
            if page == "reward":
                self._dismiss_reward("tasks")

    def _report_unimplemented_tasks(self) -> None:
        skipped = self.config["skipped_task_names"]
        logging.info("按配置跳过：%s", "、".join(skipped))
        logging.warning(
            "未列入 captured_task_adapters 的任务没有完整捕获链路，脚本不会点击其“前往”按钮。"
        )


def expected_asset_names(config: dict[str, Any]) -> set[str]:
    names = {f"anchor_{page}.png" for page in config["anchors"]}
    names.update(
        f"anchor_{page}_{index}.png"
        for page, regions in config.get("page_multi_anchors", {}).items()
        for index, _region in enumerate(regions)
    )
    names.update(f"task_{name}.png" for name in config.get("task_templates", {}))
    names.update(
        f"progress_{name}.png"
        for name in config.get("task_progress_templates", {})
    )
    names.update(
        f"activity_{name}.png" for name in config.get("activity_templates", {})
    )
    return names


def prepare_assets(config: dict[str, Any]) -> None:
    *_, Image = load_runtime_dependencies()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for page, source_name in config["reference_screenshots"].items():
        source = resolve_config_path(source_name)
        if not source.exists():
            raise SystemExit(f"参考截图不存在：{source}")
        coordinates = config["anchors"][page]
        with Image.open(source) as image:
            expected_size = tuple(config["reference_size"])
            if image.size != expected_size:
                raise SystemExit(
                    f"截图 {source.name} 尺寸为 {image.size}，预期为 {expected_size}。"
                )
            image.crop(tuple(coordinates)).save(ASSETS_DIR / f"anchor_{page}.png")
    for page, regions in config.get("page_multi_anchors", {}).items():
        source = resolve_config_path(config["reference_screenshots"][page])
        with Image.open(source) as image:
            expected_size = tuple(config["reference_size"])
            if image.size != expected_size:
                raise SystemExit(
                    f"截图 {source.name} 尺寸为 {image.size}，预期为 {expected_size}。"
                )
            for index, coordinates in enumerate(regions):
                image.crop(tuple(coordinates)).save(
                    ASSETS_DIR / f"anchor_{page}_{index}.png"
                )
    for name, task_template in config.get("task_templates", {}).items():
        source = resolve_config_path(task_template["source"])
        if not source.exists():
            raise SystemExit(f"任务图标截图不存在：{source}")
        with Image.open(source) as image:
            expected_size = tuple(config["reference_size"])
            if image.size != expected_size:
                raise SystemExit(
                    f"截图 {source.name} 尺寸为 {image.size}，预期为 {expected_size}。"
                )
            image.crop(tuple(task_template["region"])).save(
                ASSETS_DIR / f"task_{name}.png"
            )
    for name, progress_template in config.get("task_progress_templates", {}).items():
        source = resolve_config_path(progress_template["source"])
        if not source.exists():
            raise SystemExit(f"任务进度截图不存在：{source}")
        with Image.open(source) as image:
            expected_size = tuple(config["reference_size"])
            if image.size != expected_size:
                raise SystemExit(
                    f"截图 {source.name} 尺寸为 {image.size}，预期为 {expected_size}。"
                )
            image.crop(tuple(progress_template["region"])).save(
                ASSETS_DIR / f"progress_{name}.png"
            )
    for name, activity_template in config.get("activity_templates", {}).items():
        source = resolve_config_path(activity_template["source"])
        if not source.exists():
            raise SystemExit(f"活跃度宝箱截图不存在：{source}")
        with Image.open(source) as image:
            image.crop(tuple(activity_template["region"])).save(
                ASSETS_DIR / f"activity_{name}.png"
            )
    expected = expected_asset_names(config)
    stale_assets = [
        path
        for path in ASSETS_DIR.glob("*.png")
        if path.name not in expected
    ]
    for path in stale_assets:
        path.unlink()
    if stale_assets:
        stale_names = "、".join(sorted(path.name for path in stale_assets))
        print(f"已清理孤立模板：{stale_names}")
    print(f"页面模板已生成：{ASSETS_DIR}")


def configure_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(run_dir / "run.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="时空猎人·觉醒微信小游戏每日任务助手")
    parser.add_argument(
        "command", choices=("prepare", "inspect", "run"), help="准备模板、检查页面或运行流程"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="允许真实鼠标操作；不提供此参数时 run 仅做干运行",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从当前断点续跑；在主界面启动时跳过体力和每日补给",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config.json", help="配置文件路径"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        prepare_assets(config)
        return 0

    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except AttributeError:
        pass

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / timestamp
    configure_logging(run_dir)
    try:
        game = DesktopGame(config, execute=args.execute, run_dir=run_dir)
        if args.command == "inspect":
            game.focus(force=True)
            page, scores = game.detect_page()
            path = game.save_diagnostic(f"inspect-{page}")
            logging.info("识别页面：%s；匹配分数：%s；截图：%s", page, scores, path)
            return 0 if page != "unknown" else 2
        DailyBot(game, config, resume=args.resume).run()
        logging.info("当前已知流程执行完成。运行记录：%s", run_dir)
        return 0
    except SafetyStop as exc:
        logging.error("安全停止：%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
