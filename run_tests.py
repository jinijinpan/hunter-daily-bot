from __future__ import annotations

import argparse
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FULL_ONLY_PREFIXES = (
    "test_recognition_replay.RealScreenshotReplayTests.",
)


def iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_tests(item)
        else:
            yield item


def build_suite(profile: str) -> unittest.TestSuite:
    discovered = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    if profile == "full":
        return discovered
    selected = [
        test
        for test in iter_tests(discovered)
        if not any(prefix in test.id() for prefix in FULL_ONLY_PREFIXES)
    ]
    return unittest.TestSuite(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行猎人日常脚本测试")
    parser.add_argument("--profile", choices=("fast", "full"), default="fast")
    parser.add_argument("--verbosity", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.profile == "fast":
        os.environ["HUNTER_OCR_BACKEND"] = "cpu"
    suite = build_suite(args.profile)
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
