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
        self.assertLessEqual(len(samples), 31)
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
                for name, count in sample.get("expected_control_counts", {}).items():
                    self.assertEqual(count, result["control_counts"].get(name, 0))
                if "expected_numeric_values" in sample:
                    self.assertEqual(
                        sample["expected_numeric_values"],
                        result["numeric_values"],
                    )
                for rect in result["control_rects"].values():
                    self.assertLess(rect[0], rect[2])
                    self.assertLess(rect[1], rect[3])

    def test_real_tower_result_detects_dismiss_control(self):
        image = cv2.imdecode(
            np.fromfile(ROOT / "references" / "tower_quick_result.png", dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        legacy_page, scores = self.matcher.detect_page(image)
        observation = self.engine.observe(image, template_scores=scores)

        self.assertEqual("tower_result", legacy_page)
        self.assertEqual("tower_result", observation.state)
        dismiss = [
            control for control in observation.controls
            if control.name == "dismiss_result"
        ]
        self.assertEqual(1, len(dismiss))
        self.assertEqual("ocr", dismiss[0].source)

    def test_ladder_reward_panel_detects_only_its_close_control(self):
        sample = next(
            item for item in self.manifest["samples"]
            if item["file"] == "milestone5a-ladder-reward-panel.png"
        )
        result = replay_sample(
            sample, FIXTURE_DIR, self.config, self.matcher, self.engine
        )

        self.assertEqual("unknown", result["legacy_page"])
        self.assertEqual("ladder_reward_panel", result["state"])
        self.assertEqual(["close_reward_panel"], result["controls"])
        x1, y1, x2, y2 = result["control_rects"]["close_reward_panel"]
        self.assertLessEqual(abs(((x1 + x2) // 2) - 941), 3)
        self.assertLessEqual(abs(((y1 + y2) // 2) - 154), 3)

    def test_milestone4r_resource_button_uses_ocr_and_outline_rectangle(self):
        sample = next(
            item for item in self.manifest["samples"]
            if item["file"] == "milestone4r-resource-dialog.png"
        )
        result = replay_sample(
            sample, FIXTURE_DIR, self.config, self.matcher, self.engine
        )

        self.assertEqual("resource_dialog", result["state"])
        x1, y1, x2, y2 = result["control_rects"]["resource_quick"]
        self.assertLessEqual(abs(x1 - 693), 3)
        self.assertLessEqual(abs(y1 - 474), 3)
        self.assertGreaterEqual(x2 - x1, 145)
        self.assertGreaterEqual(y2 - y1, 38)

    def test_real_tower_battle_confirmation_detects_actual_button(self):
        sample = next(
            item for item in self.manifest["samples"]
            if item["file"] == "milestone4r-tower-battle-confirm.webp"
        )
        result = replay_sample(
            sample, FIXTURE_DIR, self.config, self.matcher, self.engine
        )

        self.assertEqual("tower_battle_confirm", result["state"])
        x1, y1, x2, y2 = result["control_rects"]["confirm_battle"]
        self.assertLessEqual(abs(x1 - 573), 3)
        self.assertLessEqual(abs(y1 - 483), 3)
        self.assertGreaterEqual(x2 - x1, 110)
        self.assertGreaterEqual(y2 - y1, 38)

    def test_real_tower_failure_detects_continue_instruction(self):
        sample = next(
            item for item in self.manifest["samples"]
            if item["file"] == "milestone4r-tower-failure.webp"
        )
        result = replay_sample(
            sample, FIXTURE_DIR, self.config, self.matcher, self.engine
        )

        self.assertEqual("tower_failure", result["state"])
        self.assertIn("dismiss_tower_failure", result["controls"])

    def test_real_resource_confirmation_overrides_similar_hunter_prompt(self):
        sample = next(
            item for item in self.manifest["samples"]
            if item["file"] == "milestone4r-resource-confirm.webp"
        )
        result = replay_sample(
            sample, FIXTURE_DIR, self.config, self.matcher, self.engine
        )

        self.assertEqual("resource_confirm", result["legacy_page"])
        self.assertEqual("resource_confirm", result["state"])
        self.assertIn("resource_confirm", result["controls"])
        self.assertNotIn("hunter_confirm", result["controls"])

    def test_real_rank_overlay_detects_actual_dismiss_instruction(self):
        sample = next(
            item for item in self.manifest["samples"]
            if item["file"] == "milestone4r-rank-overlay.webp"
        )
        result = replay_sample(
            sample, FIXTURE_DIR, self.config, self.matcher, self.engine
        )

        self.assertEqual("rank_overlay", result["state"])
        self.assertIn("dismiss_rank_overlay", result["controls"])

    def test_rank_overlay_without_instruction_detects_title_fallback(self):
        sample = next(
            item for item in self.manifest["samples"]
            if item["file"] == "milestone4r-rank-overlay-title.webp"
        )
        result = replay_sample(
            sample, FIXTURE_DIR, self.config, self.matcher, self.engine
        )

        self.assertEqual("rank_overlay", result["state"])
        self.assertIn("dismiss_rank_overlay_title", result["controls"])
        self.assertNotIn("dismiss_rank_overlay", result["controls"])

    def test_real_rank_drop_detects_title_control_behind_template_gate(self):
        sample = next(
            item for item in self.manifest["samples"]
            if item["file"] == "milestone5a-rank-drop.webp"
        )
        result = replay_sample(
            sample, FIXTURE_DIR, self.config, self.matcher, self.engine
        )

        self.assertEqual("infinite_rank_drop", result["legacy_page"])
        self.assertEqual("rank_overlay", result["state"])
        self.assertIn("dismiss_rank_overlay_title", result["controls"])
        self.assertNotIn("dismiss_rank_overlay", result["controls"])

    def test_rank_tasks_reads_authoritative_hunter_league_progress(self):
        sample = next(
            item for item in self.manifest["samples"]
            if item["file"] == "milestone4r-rank-tasks-3-of-10.webp"
        )
        result = replay_sample(
            sample, FIXTURE_DIR, self.config, self.matcher, self.engine
        )

        self.assertEqual("rank_tasks", result["state"])
        self.assertEqual({"hunter_league_matches": 3}, result["numeric_values"])
        self.assertEqual(2, result["control_counts"]["league_go"])

    def test_rank_overview_detects_task_tab_and_home(self):
        sample = next(
            item for item in self.manifest["samples"]
            if item["file"] == "milestone4r-rank-overview.webp"
        )
        result = replay_sample(
            sample, FIXTURE_DIR, self.config, self.matcher, self.engine
        )

        self.assertEqual("rank_overview", result["state"])
        self.assertIn("rank_tasks_tab", result["controls"])
        self.assertIn("template", result["control_sources"]["rank_tasks_tab"])
        self.assertIn("home", result["controls"])

    def test_resource_confirmation_and_tower_exit_are_detected_controls(self):
        cases = [
            (
                "resource_confirm.png",
                "resource_confirm",
                "resource_confirm",
                "color+ocr",
            ),
            (
                "tower_post_battle.png",
                "tower_post_battle",
                "tower_exit",
                "template",
            ),
        ]
        for file_name, expected_state, control_name, expected_source in cases:
            with self.subTest(file=file_name):
                image = cv2.imdecode(
                    np.fromfile(ROOT / "references" / file_name, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                _legacy_page, scores = self.matcher.detect_page(image)
                observation = self.engine.observe(image, template_scores=scores)
                controls = {
                    control.name: control for control in observation.controls
                }
                self.assertEqual(expected_state, observation.state)
                self.assertIn(control_name, controls)
                self.assertEqual(expected_source, controls[control_name].source)

    def test_explicit_dismiss_instruction_is_the_only_exposed_control(self):
        sample = next(
            item for item in self.manifest["samples"]
            if item["file"] == "dismissible-overlay.webp"
        )
        result = replay_sample(
            sample, FIXTURE_DIR, self.config, self.matcher, self.engine
        )

        self.assertEqual("dismissible_overlay", result["state"])
        self.assertEqual(["dismiss_overlay"], result["controls"])
        x1, y1, x2, y2 = result["control_rects"]["dismiss_overlay"]
        self.assertGreaterEqual(x1, 400)
        self.assertGreaterEqual(y1, 625)
        self.assertLess(x2, 700)
        self.assertLessEqual(y2, 700)


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

    def test_consensus_supports_state_specific_animation_threshold(self):
        consensus = MultiFrameConsensus(
            required_frames=2,
            max_frame_change=6.0,
            max_frame_change_by_state={"main": 9.0},
        )
        self.assertIsNone(consensus.update(self.observation(state="main", change=7.5)))
        self.assertIsNotNone(consensus.update(self.observation(state="main", change=8.0)))

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
