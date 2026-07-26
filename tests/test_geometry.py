import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot import Rect, ReferenceGeometry


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


if __name__ == "__main__":
    unittest.main()
