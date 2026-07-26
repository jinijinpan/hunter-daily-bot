import json
import unittest
from pathlib import Path


try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


ROOT = Path(__file__).resolve().parents[1]

from bot import DesktopGame, expected_asset_names


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class TemplateRecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "config.json").open("r", encoding="utf-8") as stream:
            cls.config = json.load(stream)

    @staticmethod
    def read_gray(path: Path):
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise AssertionError(f"Cannot read image: {path}")
        return image

    @staticmethod
    def read_color(path: Path):
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise AssertionError(f"Cannot read image: {path}")
        return image

    @staticmethod
    def resolve_source(source: str) -> Path:
        path = Path(source)
        return path if path.is_absolute() else ROOT / path

    def normalize_capture_fixture(self, path: Path):
        image = self.read_color(path)
        width, height = self.config["reference_size"]
        reference_top = self.config["reference_content_top"]
        start, end = self.config["content_top_search_range"]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        row_means = gray.mean(axis=1)
        drops = row_means[start : end + 1] - row_means[start - 1 : end]
        content_top = start + int(np.argmin(drops))
        chrome = cv2.resize(image[:content_top], (width, reference_top))
        content = cv2.resize(
            image[content_top:], (width, height - reference_top)
        )
        return np.vstack((chrome, content))

    def configured_game(self):
        game = DesktopGame.__new__(DesktopGame)
        game.cv2 = cv2
        game.config = self.config
        game.templates = {
            page: self.read_gray(ROOT / "assets" / f"anchor_{page}.png")
            for page in self.config["anchors"]
        }
        game.multi_anchor_templates = {
            page: [
                self.read_gray(ROOT / "assets" / f"anchor_{page}_{index}.png")
                for index, _region in enumerate(regions)
            ]
            for page, regions in self.config.get("page_multi_anchors", {}).items()
        }
        return game

    def recognition_samples(self):
        for expected, source in self.config["reference_screenshots"].items():
            yield expected, self.resolve_source(source), False
        for expected, sources in self.config.get(
            "page_recognition_samples", {}
        ).items():
            for source in sources:
                yield expected, self.resolve_source(source), False
        for expected, sources in self.config.get("page_runtime_samples", {}).items():
            for source in sources:
                yield expected, self.resolve_source(source), True

    def test_each_recognition_sample_has_expected_page(self):
        game = self.configured_game()
        for expected, path, normalize in self.recognition_samples():
            with self.subTest(page=expected, sample=path.name):
                image = (
                    self.normalize_capture_fixture(path)
                    if normalize
                    else self.read_color(path)
                )
                detected, scores = game.detect_page(image)
                self.assertEqual(expected, detected)
                if expected != "unknown":
                    self.assertGreaterEqual(
                        scores[expected], game._page_threshold(expected)
                    )

    def test_every_reference_image_is_registered_as_a_sample(self):
        registered = {
            path.resolve() for _expected, path, _normalize in self.recognition_samples()
        }
        available = {path.resolve() for path in (ROOT / "references").glob("*.png")}
        self.assertSetEqual(available, registered)

    def test_generated_page_anchors_match_their_reference_crops(self):
        for page, source in self.config["reference_screenshots"].items():
            image = self.read_gray(self.resolve_source(source))
            x1, y1, x2, y2 = self.config["anchors"][page]
            expected = image[y1:y2, x1:x2]
            actual = self.read_gray(ROOT / "assets" / f"anchor_{page}.png")
            with self.subTest(page=page):
                self.assertEqual(expected.shape, actual.shape)
                score = float(
                    cv2.matchTemplate(expected, actual, cv2.TM_CCOEFF_NORMED)[0][0]
                )
                self.assertGreaterEqual(score, 0.995)

        for page, regions in self.config.get("page_multi_anchors", {}).items():
            image = self.read_gray(
                self.resolve_source(self.config["reference_screenshots"][page])
            )
            for index, (x1, y1, x2, y2) in enumerate(regions):
                expected = image[y1:y2, x1:x2]
                actual = self.read_gray(
                    ROOT / "assets" / f"anchor_{page}_{index}.png"
                )
                with self.subTest(page=page, anchor=index):
                    self.assertEqual(expected.shape, actual.shape)
                    score = float(
                        cv2.matchTemplate(
                            expected, actual, cv2.TM_CCOEFF_NORMED
                        )[0][0]
                    )
                    self.assertGreaterEqual(score, 0.995)

    def test_generated_asset_set_matches_config(self):
        actual = {path.name for path in (ROOT / "assets").glob("*.png")}
        self.assertSetEqual(expected_asset_names(self.config), actual)

    def test_abyss_exhausted_is_distinct_from_retry_available(self):
        game = self.configured_game()
        for expected in ("abyss_finished", "abyss_exhausted"):
            with self.subTest(page=expected):
                image = self.read_color(
                    self.resolve_source(self.config["reference_screenshots"][expected])
                )
                detected, scores = game.detect_page(image)
                self.assertEqual(expected, detected)
                self.assertGreaterEqual(
                    scores[expected], self.config["page_thresholds"][expected]
                )

    def test_abyss_cards_recognizes_later_flip_round_at_runtime_size(self):
        image = self.normalize_capture_fixture(
            ROOT / "references" / "abyss_cards_round2_raw.png"
        )
        game = self.configured_game()

        detected, scores = game.detect_page(image)

        self.assertEqual("abyss_cards", detected)
        self.assertGreaterEqual(
            scores["abyss_cards"], self.config["page_thresholds"]["abyss_cards"]
        )

    def test_page_recognition_tolerates_small_responsive_shift(self):
        image = self.read_gray(
            self.resolve_source(self.config["reference_screenshots"]["main"])
        )
        height, width = image.shape
        shifted = cv2.warpAffine(
            image,
            np.float32([[1, 0, 8], [0, 1, 5]]),
            (width, height),
            borderMode=cv2.BORDER_REFLECT,
        )
        game = self.configured_game()

        page, scores = game.detect_page(cv2.cvtColor(shifted, cv2.COLOR_GRAY2BGR))

        self.assertEqual("main", page)
        self.assertGreaterEqual(scores["main"], self.config["page_match_threshold"])

    def test_captured_battle_settlement_frames_remain_recognizable(self):
        samples = {
            "recordings/20260725-164147/action-0004-before.png": "tower_post_battle",
            "recordings/20260725-164147/frame-0027-change.png": "reward",
            "recordings/20260725-190746/action-0006-before.png": "monster_reward",
            "recordings/20260725-191355/action-0003-before.png": "hunter_league_victory",
            "recordings/20260725-191355/action-0006-before.png": "hunter_league_failure",
            "recordings/20260725-190429/action-0009-before.png": "infinite_score",
            "recordings/20260725-190429/action-0010-before.png": "infinite_next",
        }
        game = self.configured_game()
        for relative_path, expected in samples.items():
            with self.subTest(page=expected):
                page, scores = game.detect_page(self.read_color(ROOT / relative_path))
                self.assertEqual(expected, page)
                self.assertGreaterEqual(
                    scores[expected], game._page_threshold(expected)
                )

    def test_hunter_league_result_overlay_is_not_ready_lobby(self):
        game = self.configured_game()
        for sequence in (2, 7, 10, 33):
            with self.subTest(sequence=sequence, state="lobby"):
                path = (
                    ROOT
                    / "recordings"
                    / "20260725-191355"
                    / f"action-{sequence:04d}-before.png"
                )
                page, _scores = game.detect_page(self.read_color(path))
                self.assertEqual("hunter_league", page)
        for sequence in (3, 6, 8, 16, 32):
            with self.subTest(sequence=sequence, state="result"):
                path = (
                    ROOT
                    / "recordings"
                    / "20260725-191355"
                    / f"action-{sequence:04d}-before.png"
                )
                _page, scores = game.detect_page(self.read_color(path))
                self.assertLess(
                    scores["hunter_league"],
                    game._page_threshold("hunter_league"),
                )

    def test_returned_main_screen_is_recognized_with_dynamic_notifications(self):
        image = self.read_gray(ROOT / "references" / "main_returned.png")
        game = self.configured_game()

        page, scores = game.detect_page(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

        self.assertEqual("main", page)
        self.assertGreaterEqual(scores["main"], self.config["page_thresholds"]["main"])

    def test_legacy_main_screen_is_recognized_by_multiple_stable_anchors(self):
        image = self.read_color(ROOT / "references" / "main_legacy.png")
        game = self.configured_game()

        page, scores = game.detect_page(image)

        self.assertEqual("main", page)
        self.assertGreaterEqual(scores["main"], self.config["page_thresholds"]["main"])

    def test_main_background_behind_character_switch_is_not_main(self):
        image = self.read_color(ROOT / "references" / "character_switch.png")
        game = self.configured_game()

        page, scores = game.detect_page(image)

        self.assertEqual("character_switch", page)
        self.assertLess(scores["main"], self.config["page_thresholds"]["main"])

    def test_hunter_calculation_is_not_misclassified_as_failure(self):
        image = self.read_gray(ROOT / "references" / "hunter_calculating.png")
        x1, y1, x2, y2 = self.config["anchors"]["hunter_failure"]
        template = self.read_gray(ROOT / "assets" / "anchor_hunter_failure.png")
        score = float(
            cv2.matchTemplate(
                image[y1:y2, x1:x2], template, cv2.TM_CCOEFF_NORMED
            )[0][0]
        )
        self.assertLess(score, self.config["page_thresholds"]["hunter_failure"])

    def test_tower_calculation_is_not_misclassified_as_result(self):
        image = self.read_gray(ROOT / "references" / "tower_calculating.png")
        x1, y1, x2, y2 = self.config["anchors"]["tower_result"]
        template = self.read_gray(ROOT / "assets" / "anchor_tower_result.png")
        score = float(
            cv2.matchTemplate(
                image[y1:y2, x1:x2], template, cv2.TM_CCOEFF_NORMED
            )[0][0]
        )
        self.assertLess(score, self.config["page_thresholds"]["tower_result"])

    def test_tower_confirm_is_not_misclassified_as_hunter_confirm(self):
        image = self.read_color(
            ROOT / "references" / "tower_battle_confirm_current.png"
        )
        game = self.configured_game()

        page, scores = game.detect_page(image)

        self.assertEqual("tower_battle_confirm", page)
        self.assertGreaterEqual(
            scores["tower_battle_confirm"],
            self.config["page_match_threshold"],
        )
        self.assertLess(
            scores["hunter_confirm"],
            self.config["page_thresholds"]["hunter_confirm"],
        )

    def test_hunter_confirm_handles_dynamic_floor_and_reward_cards(self):
        image = self.read_color(ROOT / "references" / "hunter_confirm_current.png")
        game = self.configured_game()

        page, scores = game.detect_page(image)

        self.assertEqual("hunter_confirm", page)
        self.assertGreaterEqual(
            scores["hunter_confirm"],
            self.config["page_thresholds"]["hunter_confirm"],
        )

    def test_resource_confirm_is_not_misclassified_as_hunter_confirm(self):
        image = self.read_color(ROOT / "references" / "resource_confirm.png")
        game = self.configured_game()

        page, scores = game.detect_page(image)

        self.assertEqual("resource_confirm", page)
        self.assertLess(
            scores["hunter_confirm"],
            self.config["page_thresholds"]["hunter_confirm"],
        )

    def test_task_templates_handle_reordered_lists(self):
        samples = {
            "tasks.png": {
                "tower",
                "hunter_field",
                "abyss",
                "monster_invasion",
                "hunter_league",
            },
            "tasks_completed.png": {
                "tower",
                "hunter_field",
                "resource_supply",
                "monster_invasion",
                "hunter_league",
                "infinite_mystery",
            },
            "tasks_bottom.png": {
                "resource_supply",
                "abyss",
                "infinite_mystery",
            },
            "tasks_new.png": {
                "monster_invasion",
                "hunter_league",
                "infinite_mystery",
            },
        }
        x1, y1, x2, y2 = self.config["task_search_region"]
        for filename, expected in samples.items():
            image = self.read_gray(ROOT / "references" / filename)[y1:y2, x1:x2]
            for name in self.config["task_templates"]:
                with self.subTest(file=filename, task=name):
                    template = self.read_gray(ROOT / "assets" / f"task_{name}.png")
                    score = float(
                        cv2.minMaxLoc(
                            cv2.matchTemplate(
                                image, template, cv2.TM_CCOEFF_NORMED
                            )
                        )[1]
                    )
                    threshold = self.config["task_search_thresholds"][name]
                    if name in expected:
                        self.assertGreaterEqual(score, threshold)
                    else:
                        self.assertLess(score, threshold)

    def test_resource_supply_does_not_match_transaction_house(self):
        image = self.read_gray(ROOT / "references" / "tasks_claimed.png")
        x1, y1, x2, y2 = self.config["task_search_region"]
        template = self.read_gray(ROOT / "assets" / "task_resource_supply.png")
        score = float(
            cv2.minMaxLoc(
                cv2.matchTemplate(
                    image[y1:y2, x1:x2], template, cv2.TM_CCOEFF_NORMED
                )
            )[1]
        )

        self.assertLess(
            score, self.config["task_search_thresholds"]["resource_supply"]
        )

    def test_task_progress_distinguishes_completed_and_incomplete_rows(self):
        game = DesktopGame.__new__(DesktopGame)
        game.cv2 = cv2
        game.config = self.config
        game.task_templates = {
            name: self.read_gray(ROOT / "assets" / f"task_{name}.png")
            for name in self.config["task_templates"]
        }
        game.task_progress_templates = {
            name: self.read_gray(ROOT / "assets" / f"progress_{name}.png")
            for name in self.config["task_progress_templates"]
        }

        completed = self.read_color(ROOT / "references" / "tasks_completed.png")
        completed_tasks = {
            "tower",
            "hunter_field",
            "resource_supply",
            "monster_invasion",
            "hunter_league",
            "infinite_mystery",
        }
        for name in completed_tasks:
            with self.subTest(state="completed", task=name):
                found = game.find_task(name, completed)
                self.assertIsNotNone(found)
                self.assertTrue(game.task_progress_complete(name, found[:2], completed))

        claimed = self.read_color(ROOT / "references" / "tasks_claimed.png")
        claimed_tower = game.find_task("tower", claimed)
        self.assertIsNotNone(claimed_tower)
        self.assertTrue(
            game.task_progress_complete("tower", claimed_tower[:2], claimed)
        )

        incomplete_samples = {
            "tower": "tasks.png",
            "hunter_field": "tasks.png",
            "abyss": "tasks.png",
            "monster_invasion": "tasks_new.png",
            "hunter_league": "tasks_new.png",
        }
        for name, filename in incomplete_samples.items():
            with self.subTest(state="incomplete", task=name):
                image = self.read_color(ROOT / "references" / filename)
                found = game.find_task(name, image)
                self.assertIsNotNone(found)
                self.assertFalse(game.task_progress_complete(name, found[:2], image))

    def test_character_switch_reference_marks_first_role_online(self):
        image = self.read_color(ROOT / "references" / "character_switch.png")
        game = DesktopGame.__new__(DesktopGame)
        game.cv2 = cv2
        game.config = self.config

        self.assertEqual(0, game.active_character_index(image))

    def test_activity_chest_templates_do_not_match_claimed_chests(self):
        active = self.read_gray(ROOT / "references" / "tasks_activity.png")
        claimed = self.read_gray(ROOT / "references" / "tasks_claimed.png")
        threshold = self.config["activity_match_threshold"]
        for name, settings in self.config["activity_templates"].items():
            with self.subTest(chest=name):
                x1, y1, x2, y2 = settings["region"]
                template = self.read_gray(ROOT / "assets" / f"activity_{name}.png")
                active_score = float(
                    cv2.matchTemplate(
                        active[y1:y2, x1:x2], template, cv2.TM_CCOEFF_NORMED
                    )[0][0]
                )
                claimed_score = float(
                    cv2.matchTemplate(
                        claimed[y1:y2, x1:x2], template, cv2.TM_CCOEFF_NORMED
                    )[0][0]
                )
                self.assertGreaterEqual(active_score, threshold)
                self.assertLess(claimed_score, threshold)


if __name__ == "__main__":
    unittest.main()
