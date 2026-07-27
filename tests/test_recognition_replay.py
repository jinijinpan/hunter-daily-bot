import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot import DesktopGame, load_config
from recognition import (
    CalibrationResult,
    CalibrationError,
    CapturedFrame,
    MultiFrameConsensus,
    Observation,
    RecognitionEngine,
    ViewportCalibrator,
)
from replay_recognition import build_matcher, replay_sample


FIXTURE_DIR = ROOT / "tests" / "fixtures" / "recognition"


class ViewportCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "config.json")

    @staticmethod
    def frame(width=240, height=150, top=53):
        image = np.full((height, width, 3), 235, dtype=np.uint8)
        image[top:] = 25
        return image

    def test_calibration_is_cached_until_client_size_changes(self):
        calibrator = ViewportCalibrator(self.config, cv2, np)
        calls = []

        def capture():
            calls.append(True)
            return self.frame()

        _, first, recalibrated = calibrator.ensure((240, 150), capture)
        self.assertTrue(recalibrated)
        self.assertEqual(5, len(calls))
        self.assertEqual(53, first.content_top)

        _, cached, recalibrated = calibrator.ensure((240, 150), capture)
        self.assertFalse(recalibrated)
        self.assertEqual(6, len(calls))
        self.assertIs(first, cached)

        _, resized, recalibrated = calibrator.ensure((241, 150), capture)
        self.assertTrue(recalibrated)
        self.assertEqual(11, len(calls))
        self.assertEqual(2, resized.generation)

    def test_calibration_discards_a_single_outlier(self):
        calibrator = ViewportCalibrator(self.config, cv2, np)
        frames = [self.frame(top=value) for value in (53, 53, 71, 54, 53)]
        result = calibrator.calibrate_frames(frames, (240, 150))
        self.assertEqual(53, result.content_top)
        self.assertEqual((53, 53, 54, 53), result.inliers)

    def test_calibration_rejects_frames_without_a_strong_boundary(self):
        calibrator = ViewportCalibrator(self.config, cv2, np)
        flat = np.full((150, 240, 3), 100, dtype=np.uint8)
        with self.assertRaises(CalibrationError):
            calibrator.calibrate_frames([flat] * 5, (240, 150))


class RealScreenshotReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "config.json")
        cls.manifest = json.loads(
            (FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        cls.matcher = build_matcher(cls.config)
        cls.engine = RecognitionEngine(cls.config, cv2, np)

    def test_manifest_is_small_and_contains_runs_and_recordings(self):
        samples = self.manifest["samples"]
        self.assertLessEqual(len(samples), 8)
        sources = [sample["source"] for sample in samples]
        self.assertTrue(any(source.startswith("runs/") for source in sources))
        self.assertTrue(any(source.startswith("recordings/") for source in sources))

    def test_lossless_webp_fixtures_are_smaller_than_sources(self):
        for sample in self.manifest["samples"]:
            with self.subTest(file=sample["file"]):
                fixture = FIXTURE_DIR / sample["file"]
                source = ROOT / Path(sample["source"])
                self.assertTrue(fixture.exists())
                self.assertLess(fixture.stat().st_size, source.stat().st_size)

    def test_online_and_offline_normalization_and_scores_are_identical(self):
        for sample in self.manifest["samples"]:
            with self.subTest(file=sample["file"]):
                result = replay_sample(
                    sample, FIXTURE_DIR, self.config, self.matcher, self.engine
                )
                self.assertEqual(sample["expected_content_top"], result["content_top"])
                self.assertEqual(sample["expected_legacy_page"], result["legacy_page"])
                self.assertTrue(result["recalibrated"])
                self.assertGreaterEqual(result["calibration_confidence"], 0.95)
                if result["legacy_page"] != "unknown":
                    threshold = self.config.get("page_thresholds", {}).get(
                        result["legacy_page"], self.config["page_match_threshold"]
                    )
                    self.assertGreaterEqual(result["legacy_score"], threshold)
                self.assertEqual(sample["expected_state"], result["state"])
                controls = set(result["controls"])
                self.assertTrue(set(sample["required_controls"]).issubset(controls))
                self.assertTrue(set(sample["forbidden_controls"]).isdisjoint(controls))
                for rect in result["control_rects"].values():
                    self.assertLess(rect[0], rect[2])
                    self.assertLess(rect[1], rect[3])


class ObservationDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "config.json")
        cls.engine = RecognitionEngine(cls.config, cv2, np)

    def observation(self, state="tasks", confidence=0.9, change=0.0):
        return Observation(
            timestamp=1.0,
            viewport=(0, 0, 1091, 700),
            frame_change=change,
            state=state,
            state_confidence=confidence,
        )

    def test_single_template_signal_cannot_confirm_a_page(self):
        observation = self.observation(state="unknown", confidence=0.0)
        observation.template_scores = {"tasks": 0.99}
        state, confidence, signals = self.engine._classify(observation)
        self.assertEqual("unknown", state)
        self.assertEqual(0.0, confidence)
        self.assertEqual({"template": 0.99}, signals["tasks"])

    def test_consensus_requires_two_stable_frames(self):
        consensus = MultiFrameConsensus(required_frames=2, max_frame_change=3.0)
        self.assertIsNone(consensus.update(self.observation()))
        self.assertIsNotNone(consensus.update(self.observation()))

    def test_moving_or_unknown_frame_resets_consensus(self):
        consensus = MultiFrameConsensus(required_frames=2, max_frame_change=3.0)
        self.assertIsNone(consensus.update(self.observation()))
        self.assertIsNone(consensus.update(self.observation(change=8.0)))
        self.assertIsNone(consensus.update(self.observation()))
        self.assertIsNone(consensus.update(self.observation(state="unknown")))

    def test_annotation_draws_control_and_state(self):
        image = np.zeros((120, 240, 3), dtype=np.uint8)
        observation = self.observation()
        annotated = self.engine.annotate(image, observation)
        self.assertEqual(image.shape, annotated.shape)
        self.assertFalse(np.array_equal(image, annotated))


class DiagnosticConsistencyTests(unittest.TestCase):
    def test_diagnostic_reuses_the_last_captured_frame(self):
        config = load_config(ROOT / "config.json")
        calibration = CalibrationResult(
            client_size=(1091, 700),
            content_top=53,
            candidates=(53, 53, 53, 53, 53),
            inliers=(53, 53, 53, 53, 53),
            edge_strengths=(100.0,) * 5,
            confidence=1.0,
            calibrated_at=1.0,
            generation=1,
        )
        pixels = np.zeros((700, 1091, 3), dtype=np.uint8)
        frame = CapturedFrame(
            raw=pixels.copy(),
            normalized=pixels.copy(),
            calibration=calibration,
            viewport=(0, 0, 1091, 700),
            timestamp=1.0,
        )
        observation = Observation(
            timestamp=1.0,
            viewport=frame.viewport,
            state="unknown",
        )
        with tempfile.TemporaryDirectory() as directory:
            game = DesktopGame.__new__(DesktopGame)
            game.config = config
            game.cv2 = cv2
            game.np = np
            game.run_dir = Path(directory)
            game.last_frame = frame
            game.last_observation = observation
            game.recognition = RecognitionEngine(config, cv2, np)
            game.capture_frame = lambda: self.fail("诊断不应重新截帧")

            path = game.save_diagnostic("same-frame")

            self.assertEqual("same-frame-raw.png", path.name)
            for suffix in (
                "raw.png",
                "normalized.png",
                "annotated.png",
                "observation.json",
                "diagnostic.json",
            ):
                self.assertTrue((Path(directory) / f"same-frame-{suffix}").exists())


if __name__ == "__main__":
    unittest.main()
