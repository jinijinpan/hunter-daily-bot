import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot import DesktopGame, Rect, ReferenceGeometry


class ReferenceGeometryTests(unittest.TestCase):
    def test_point_at_reference_size(self):
        geometry = ReferenceGeometry(1091, 700, Rect(100, 50, 1191, 750))
        self.assertEqual((935, 158), geometry.point((835, 108)))

    def test_point_scales_to_larger_window(self):
        geometry = ReferenceGeometry(1091, 700, Rect(10, 20, 2192, 1420))
        self.assertEqual((1680, 236), geometry.point((835, 108)))

    def test_reference_point_reverses_screen_mapping(self):
        geometry = ReferenceGeometry(1091, 700, Rect(10, 20, 2192, 1420))
        self.assertEqual((835, 108), geometry.reference_point((1680, 236)))

    def test_content_area_scales_independently_from_fixed_chrome(self):
        geometry = ReferenceGeometry(
            1091, 700, Rect(0, 0, 1392, 839), 53, 53
        )
        self.assertEqual((0, 53), geometry.point((0, 53)))
        self.assertEqual((1392, 839), geometry.point((1091, 700)))
        self.assertEqual((835, 108), geometry.reference_point(geometry.point((835, 108))))

    def test_contains_uses_right_and_bottom_as_exclusive_edges(self):
        geometry = ReferenceGeometry(1091, 700, Rect(100, 50, 1191, 750))
        self.assertTrue(geometry.contains((100, 50)))
        self.assertTrue(geometry.contains((1190, 749)))
        self.assertFalse(geometry.contains((1191, 749)))
        self.assertFalse(geometry.contains((1190, 750)))


class WindowFocusTests(unittest.TestCase):
    def test_real_capture_focuses_window_before_reading_geometry(self):
        events = []
        image = object()
        calibration = SimpleNamespace(content_top=53, generation=1)
        game = DesktopGame.__new__(DesktopGame)
        game.execute = True
        game.focus = lambda: events.append("focus")
        game._window_rect = lambda: events.append("rect") or Rect(0, 0, 1091, 700)
        game.viewport_calibrator = SimpleNamespace(
            ensure=lambda _size, _capture: (image, calibration, True)
        )
        game.config = {"reference_size": [1091, 700], "reference_content_top": 53}
        game.cv2 = None
        game.np = None
        game.last_frame = None
        game.last_observation = None

        with patch("bot.normalize_frame", return_value=image), patch(
            "bot.frame_difference", return_value=0.0
        ):
            game.capture_frame()

        self.assertEqual(["focus", "rect"], events)

    def test_title_bar_click_is_verified_when_win32_activation_is_rejected(self):
        class Gui:
            foreground = 0
            positions = []

            @staticmethod
            def ShowWindow(_handle, _mode):
                pass

            @classmethod
            def GetForegroundWindow(cls):
                return cls.foreground

            @staticmethod
            def SetForegroundWindow(_handle):
                raise RuntimeError("foreground lock")

            @classmethod
            def SetWindowPos(cls, handle, insert_after, x, y, width, height, flags):
                cls.positions.append(
                    (handle, insert_after, x, y, width, height, flags)
                )

        class Mouse:
            clicks = []

            @classmethod
            def click(cls, x, y):
                cls.clicks.append((x, y))
                Gui.foreground = 123

        game = DesktopGame.__new__(DesktopGame)
        game.execute = True
        game.window_handle = 123
        game.win32gui = Gui
        game.win32con = SimpleNamespace(
            SW_RESTORE=9,
            HWND_TOPMOST=-1,
            HWND_NOTOPMOST=-2,
            SWP_NOMOVE=2,
            SWP_NOSIZE=1,
            SWP_NOACTIVATE=16,
        )
        game.pyautogui = Mouse
        game.content_top = 53
        game._window_rect = lambda: Rect(100, 200, 1554, 1109)

        with patch("bot.time.sleep"):
            game.focus()

        self.assertEqual([(827, 225)], Mouse.clicks)
        self.assertEqual(123, Gui.foreground)
        self.assertEqual(
            [
                (123, -1, 0, 0, 0, 0, 19),
                (123, -2, 0, 0, 0, 0, 19),
            ],
            Gui.positions,
        )


if __name__ == "__main__":
    unittest.main()
