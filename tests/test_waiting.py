import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot import DesktopGame


class ContextAwareWaitingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "config.json").open("r", encoding="utf-8") as stream:
            cls.base_config = json.load(stream)

    def game_with_scores(self, detected_page, scores):
        game = DesktopGame.__new__(DesktopGame)
        game.config = copy.deepcopy(self.base_config)
        game.config["timeouts"]["page_seconds"] = 0.1
        game.config["timeouts"]["poll_seconds"] = 0
        game.detect_page = lambda: (detected_page, scores)
        game.save_diagnostic = lambda _name: None
        return game

    def test_wait_for_page_accepts_expected_high_score_when_global_winner_differs(self):
        game = self.game_with_scores(
            "hunter_confirm",
            {
                "hunter_confirm": 0.969,
                "tower_battle_confirm": 0.919,
            },
        )

        game.wait_for_page("tower_battle_confirm", tolerate_unknown=True)

    def test_wait_for_one_of_returns_highest_matching_expected_page(self):
        game = self.game_with_scores(
            "hunter_confirm",
            {
                "hunter_confirm": 0.969,
                "tower": 0.81,
                "tower_manual": 0.90,
            },
        )

        page = game.wait_for_one_of(
            {"tower", "tower_manual"}, tolerate_unknown=True
        )

        self.assertEqual("tower_manual", page)

    def test_main_background_is_rejected_when_a_foreground_page_passes(self):
        game = self.game_with_scores(
            "character_switch",
            {
                "main": 0.94,
                "character_switch": 0.86,
            },
        )
        scores = {
            "main": 0.94,
            "character_switch": 0.86,
        }

        self.assertEqual("character_switch", game._select_page_from_scores(scores))
        self.assertIsNone(game._expected_page_from_scores({"main"}, scores))


if __name__ == "__main__":
    unittest.main()
