from __future__ import annotations

import argparse
import ctypes
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from bot import DesktopGame, ROOT, SafetyStop, configure_logging, load_config


RECORDINGS_DIR = ROOT / "recordings"
VK_LBUTTON = 0x01
VK_F8 = 0x77


@dataclass
class Press:
    started_monotonic: float
    started_at: str
    screen_point: tuple[int, int]
    reference_point: tuple[int, int]
    before_file: str
    page_before: str


@dataclass(frozen=True)
class PendingAfterCapture:
    sequence: int
    due_monotonic: float
    expires_monotonic: float


class InteractionRecorder:
    def __init__(
        self,
        game: DesktopGame,
        output_dir: Path,
        interval_seconds: float,
        change_threshold: float,
        after_action_seconds: float,
    ):
        self.game = game
        self.output_dir = output_dir
        self.interval_seconds = interval_seconds
        self.change_threshold = change_threshold
        self.after_action_seconds = after_action_seconds
        self.timeline_path = output_dir / "timeline.jsonl"
        self.timeline = self.timeline_path.open("a", encoding="utf-8")
        self.user32 = ctypes.windll.user32
        self.press: Press | None = None
        self.event_index = 0
        self.frame_index = 0
        self.last_frame_at = 0.0
        self.last_thumbnail = None
        self.pending_after_captures: list[PendingAfterCapture] = []
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")

    def close(self) -> None:
        self.timeline.close()

    def _write_event(self, payload: dict[str, Any]) -> None:
        self.timeline.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.timeline.flush()

    def _key_down(self, virtual_key: int) -> bool:
        return bool(self.user32.GetAsyncKeyState(virtual_key) & 0x8000)

    def _foreground(self) -> bool:
        return self.game.win32gui.GetForegroundWindow() == self.game.window_handle

    def _cursor(self) -> tuple[int, int]:
        point = self.game.pyautogui.position()
        return int(point.x), int(point.y)

    def _save_image(self, image, filename: str) -> str:
        path = self.output_dir / filename
        compression = int(self.game.config.get("recording_png_compression", 6))
        encoded, buffer = self.game.cv2.imencode(
            ".png",
            image,
            [self.game.cv2.IMWRITE_PNG_COMPRESSION, compression],
        )
        if not encoded:
            raise SafetyStop("无法保存捕获截图。")
        buffer.tofile(path)
        return filename

    def _capture_named(self, filename: str):
        frame = self.game.capture_frame()
        image = frame.normalized
        saved = self._save_image(image, filename)
        page, scores = self.game.detect_page(image)
        return saved, image, page, scores

    def _record_changed_frame(self, force: bool = False, label: str = "change") -> None:
        now = time.monotonic()
        if not force and now - self.last_frame_at < self.interval_seconds:
            return
        if not self._foreground():
            return

        frame = self.game.capture_frame()
        image = frame.normalized
        thumbnail = self.game.cv2.resize(image, (160, 100), interpolation=self.game.cv2.INTER_AREA)
        gray = self.game.cv2.cvtColor(thumbnail, self.game.cv2.COLOR_BGR2GRAY)
        difference = None
        if self.last_thumbnail is not None:
            difference = float(
                self.game.np.mean(self.game.cv2.absdiff(gray, self.last_thumbnail))
            )
        self.last_frame_at = now
        if not force and difference is not None and difference < self.change_threshold:
            return

        self.frame_index += 1
        filename = f"frame-{self.frame_index:04d}-{label}.png"
        self._save_image(image, filename)
        page, scores = self.game.detect_page(image)
        self._write_event(
            {
                "type": "frame",
                "sequence": self.frame_index,
                "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "file": filename,
                "reason": label,
                "difference": difference,
                "page": page,
                "page_scores": scores,
            }
        )
        self.last_thumbnail = gray
        logging.info("关键帧 %s：page=%s difference=%s", filename, page, difference)

    def _on_press(self, cursor: tuple[int, int]) -> None:
        self.game.capture()
        if not self.game.geometry.contains(cursor):
            return
        self.event_index += 1
        filename = f"action-{self.event_index:04d}-before.png"
        saved, _, page, _ = self._capture_named(filename)
        reference = self.game.geometry.reference_point(cursor)
        self.press = Press(
            started_monotonic=time.monotonic(),
            started_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
            screen_point=cursor,
            reference_point=reference,
            before_file=saved,
            page_before=page,
        )
        logging.info("记录按下：屏幕 %s，参考 %s，页面 %s", cursor, reference, page)

    def _on_release(self, cursor: tuple[int, int]) -> None:
        if self.press is None:
            return
        press = self.press
        self.press = None
        duration = time.monotonic() - press.started_monotonic
        self.game.capture()
        reference_end = self.game.geometry.reference_point(cursor)
        distance = (
            (reference_end[0] - press.reference_point[0]) ** 2
            + (reference_end[1] - press.reference_point[1]) ** 2
        ) ** 0.5
        action_type = "drag" if distance >= 12 else "click"

        filename = f"action-{self.event_index:04d}-release.png"
        saved, _, page_at_release, scores = self._capture_named(filename)
        payload = {
            "type": action_type,
            "sequence": self.event_index,
            "started_at": press.started_at,
            "ended_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "duration_ms": round(duration * 1000),
            "screen_start": list(press.screen_point),
            "screen_end": list(cursor),
            "reference_start": list(press.reference_point),
            "reference_end": list(reference_end),
            "distance_reference_px": round(distance, 1),
            "page_before": press.page_before,
            "page_at_release": page_at_release,
            "page_at_release_scores": scores,
            "before_file": press.before_file,
            "release_file": saved,
        }
        self._write_event(payload)
        logging.info(
            "记录%s：%s -> %s，页面 %s -> %s",
            "拖动" if action_type == "drag" else "点击",
            press.reference_point,
            reference_end,
            press.page_before,
            page_at_release,
        )
        now = time.monotonic()
        self.pending_after_captures.append(
            PendingAfterCapture(
                sequence=self.event_index,
                due_monotonic=now + self.after_action_seconds,
                expires_monotonic=now + max(self.after_action_seconds + 5.0, 5.0),
            )
        )

    def _process_pending_after_captures(self) -> None:
        now = time.monotonic()
        remaining: list[PendingAfterCapture] = []
        for pending in self.pending_after_captures:
            if now < pending.due_monotonic:
                remaining.append(pending)
                continue
            if not self._foreground():
                if now < pending.expires_monotonic:
                    remaining.append(pending)
                else:
                    logging.warning(
                        "操作 %04d 的延迟截图因目标窗口未在前台而跳过。",
                        pending.sequence,
                    )
                continue

            filename = f"action-{pending.sequence:04d}-after.png"
            saved, _, page, scores = self._capture_named(filename)
            self._write_event(
                {
                    "type": "action_after",
                    "sequence": pending.sequence,
                    "captured_at": datetime.now()
                    .astimezone()
                    .isoformat(timespec="milliseconds"),
                    "page": page,
                    "page_scores": scores,
                    "file": saved,
                }
            )
            logging.info("操作 %04d 延迟截图：页面 %s", pending.sequence, page)
            self._record_changed_frame(
                force=True, label=f"after-action-{pending.sequence:04d}"
            )
        self.pending_after_captures = remaining

    def run(self) -> None:
        self._write_event(
            {
                "type": "session_start",
                "started_at": self.started_at,
                "window_title": self.game.win32gui.GetWindowText(self.game.window_handle),
                "reference_size": self.game.config["reference_size"],
                "stop_key": "F8",
            }
        )
        print("捕获已开始。请在小游戏内正常操作，完成后按 F8 停止。")
        print(f"记录目录：{self.output_dir}")
        self._record_changed_frame(force=True, label="start")

        left_was_down = self._key_down(VK_LBUTTON)
        f8_was_down = self._key_down(VK_F8)
        try:
            while True:
                f8_down = self._key_down(VK_F8)
                if f8_down and not f8_was_down:
                    break
                f8_was_down = f8_down

                foreground = self._foreground()
                left_down = self._key_down(VK_LBUTTON)
                cursor = self._cursor()
                if foreground and left_down and not left_was_down:
                    self._on_press(cursor)
                elif not left_down and left_was_down:
                    self._on_release(cursor)
                left_was_down = left_down

                if foreground:
                    self._process_pending_after_captures()
                    self._record_changed_frame()
                time.sleep(0.03)
        except KeyboardInterrupt:
            logging.info("收到 Ctrl+C，停止捕获。")
        finally:
            if self.press is not None:
                self.press = None
            self._record_changed_frame(force=True, label="stop")
            self._write_event(
                {
                    "type": "session_end",
                    "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "action_count": self.event_index,
                    "frame_count": self.frame_index,
                }
            )
            self.close()
        print(f"捕获完成：{self.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="记录时空猎人小游戏中的点击、拖动和页面变化")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="页面变化检查间隔秒数，默认 1.0",
    )
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=8.0,
        help="保存变化帧的灰度差阈值，默认 8.0",
    )
    parser.add_argument(
        "--after-action",
        type=float,
        default=0.8,
        help="松开鼠标后等待截图的秒数，默认 0.8",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval < 0.2:
        raise SystemExit("--interval 不能小于 0.2 秒。")
    if args.after_action < 0:
        raise SystemExit("--after-action 不能为负数。")

    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except AttributeError:
        pass

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = RECORDINGS_DIR / timestamp
    configure_logging(output_dir)
    try:
        config = load_config(args.config)
        game = DesktopGame(config, execute=False, run_dir=output_dir)
        recorder = InteractionRecorder(
            game,
            output_dir,
            interval_seconds=args.interval,
            change_threshold=args.change_threshold,
            after_action_seconds=args.after_action,
        )
        recorder.run()
        return 0
    except SafetyStop as exc:
        logging.error("无法开始捕获：%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
