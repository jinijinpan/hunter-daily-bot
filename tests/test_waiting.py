import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot import DesktopGame, PageTimeout, SafetyStop
from recognition import DetectedControl, Observation


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += max(0.0, float(seconds))


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


class ObservationWaitingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "config.json").open("r", encoding="utf-8") as stream:
            cls.base_config = json.load(stream)

    @staticmethod
    def observation(state, *, change=0.0, controls=()):
        return Observation(
            timestamp=1.0,
            viewport=(0, 0, 1091, 700),
            frame_change=change,
            state=state,
            state_confidence=0.95,
            controls=list(controls),
        )

    def sequence_game(self, observations, *, poll=0.0):
        game = DesktopGame.__new__(DesktopGame)
        game.config = copy.deepcopy(self.base_config)
        game.config["timeouts"]["poll_seconds"] = poll
        queue = list(observations)
        last = queue[-1]

        def observe_frame():
            nonlocal last
            if queue:
                last = queue.pop(0)
            return last

        game.observe_frame = observe_frame
        game.diagnostics = []
        game.save_diagnostic = game.diagnostics.append
        return game

    def test_wait_for_state_requires_consecutive_stable_frames(self):
        game = self.sequence_game(
            [
                self.observation("tasks"),
                self.observation("unknown"),
                self.observation("tasks"),
                self.observation("tasks"),
            ]
        )

        result = game.wait_for_state({"tasks"}, timeout=0.1, hard_timeout=0.2)

        self.assertEqual("tasks", result.state)

    def test_main_consensus_allows_real_background_animation(self):
        game = self.sequence_game(
            [
                self.observation("main", change=15.7),
                self.observation("main", change=15.6),
            ]
        )

        result = game.wait_for_state({"main"}, timeout=0.1, hard_timeout=0.2)

        self.assertEqual("main", result.state)

    def test_final_sampling_keeps_first_stable_frame_when_ocr_is_slow(self):
        game = self.sequence_game([self.observation("hunter_field")], poll=0.05)
        clock = FakeClock()

        def slow_observe_frame():
            clock.value += 0.15
            return self.observation("hunter_field", change=0.2)

        game.observe_frame = slow_observe_frame
        with patch("bot.time.monotonic", clock.monotonic), patch(
            "bot.time.sleep", clock.sleep
        ):
            result = game.wait_for_state(
                {"hunter_field"}, timeout=0.1, hard_timeout=0.6
            )

        self.assertEqual("hunter_field", result.state)
        self.assertLess(clock.value, 0.6)

    def test_transient_and_loading_frames_extend_only_to_hard_timeout(self):
        game = self.sequence_game(
            [
                self.observation("unknown_transient", change=20.0),
                self.observation("loading", change=15.0),
                self.observation("tasks"),
                self.observation("tasks"),
            ],
            poll=0.1,
        )
        clock = FakeClock()
        with patch("bot.time.monotonic", clock.monotonic), patch(
            "bot.time.sleep", clock.sleep
        ):
            result = game.wait_for_state(
                {"tasks"}, timeout=0.15, hard_timeout=1.0
            )

        self.assertEqual("tasks", result.state)
        self.assertGreater(clock.value, 0.15)
        self.assertLess(clock.value, 1.0)

    def test_continuous_loading_stops_at_hard_timeout(self):
        game = self.sequence_game(
            [self.observation("loading", change=20.0)], poll=0.05
        )
        clock = FakeClock()
        with patch("bot.time.monotonic", clock.monotonic), patch(
            "bot.time.sleep", clock.sleep
        ):
            with self.assertRaises(PageTimeout):
                game.wait_for_state(
                    {"tasks"}, timeout=0.1, hard_timeout=0.4
                )

        self.assertAlmostEqual(0.4, clock.value, places=6)
        self.assertEqual(["timeout-state-tasks"], game.diagnostics)

    def test_click_detected_control_uses_center_and_does_not_retry_in_transition(self):
        control = DetectedControl("quick_challenge", (10, 20, 30, 40), 0.99, "ocr")
        source = self.observation("tower_ready", controls=[control])
        game = self.sequence_game(
            [self.observation("unknown_transient", change=25.0)]
        )
        clicks = []
        game.execute = True
        game.geometry = SimpleNamespace(point=lambda point: (point[0] + 100, point[1] + 200))
        game.pyautogui = SimpleNamespace(click=lambda *point: clicks.append(point))
        game.focus = lambda: None

        result = game.click_detected_control(
            "quick_challenge",
            "快速挑战",
            allowed_states={"tower_ready"},
            target_states={"tower_result"},
            observation=source,
        )

        self.assertEqual("unknown_transient", result.state)
        self.assertEqual([(120, 230)], clicks)

    def test_click_accepts_expected_high_confidence_target_before_stable_consensus(self):
        control = DetectedControl("secretary", (10, 20, 30, 40), 0.99, "ocr")
        source = self.observation("main", controls=[control])
        target = self.observation("tasks", change=40.0)
        target.state_confidence = 0.99
        game = self.sequence_game([target])
        clicks = []
        game.execute = True
        game.geometry = SimpleNamespace(point=lambda point: point)
        game.pyautogui = SimpleNamespace(click=lambda *point: clicks.append(point))
        game.focus = lambda: None

        result = game.click_detected_control(
            "secretary",
            "打开小秘书",
            allowed_states={"main"},
            target_states={"tasks"},
            observation=source,
        )

        self.assertEqual("tasks", result.state)
        self.assertEqual([(20, 30)], clicks)

    def test_click_uses_state_specific_target_confidence(self):
        control = DetectedControl("home", (10, 20, 30, 40), 0.99, "template")
        source = self.observation("hunter_field", controls=[control])
        target = self.observation("main", change=64.0)
        target.state_confidence = 0.74
        game = self.sequence_game([target])
        clicks = []
        game.execute = True
        game.geometry = SimpleNamespace(point=lambda point: point)
        game.pyautogui = SimpleNamespace(click=lambda *point: clicks.append(point))
        game.focus = lambda: None

        result = game.click_detected_control(
            "home",
            "恢复主页",
            allowed_states={"hunter_field"},
            target_states={"main"},
            observation=source,
        )

        self.assertEqual("main", result.state)
        self.assertEqual([(20, 30)], clicks)

    def test_click_accepts_delayed_target_before_retrying_source_control(self):
        control = DetectedControl("home", (10, 20, 30, 40), 0.99, "template")
        source = self.observation("tasks", controls=[control])
        target = self.observation("main", change=9.0)
        game = self.sequence_game([target])
        game.execute = True
        game.geometry = SimpleNamespace(point=lambda point: point)
        clicks = []
        game.pyautogui = SimpleNamespace(click=lambda *point: clicks.append(point))
        game.focus = lambda: None
        game._verify_detected_click = lambda *_args, **_kwargs: None
        waited_for = []

        def wait_for_state(expected, **_kwargs):
            waited_for.append(set(expected))
            return target

        game.wait_for_state = wait_for_state

        result = game.click_detected_control(
            "home",
            "返回主界面",
            allowed_states={"tasks"},
            target_states={"main", "trial"},
            observation=source,
        )

        self.assertEqual("main", result.state)
        self.assertEqual([{"tasks", "main", "trial"}], waited_for)
        self.assertEqual([(20, 30)], clicks)

    def test_click_verifies_original_claim_disappeared_when_other_claims_remain(self):
        original = DetectedControl("claim", (10, 20, 30, 40), 0.99, "color+ocr")
        other = DetectedControl("claim", (110, 20, 130, 40), 0.98, "color+ocr")
        source = self.observation("tasks", controls=[original, other])
        remaining = self.observation("tasks", controls=[other])
        game = self.sequence_game([remaining, remaining])
        clicks = []
        game.execute = True
        game.geometry = SimpleNamespace(point=lambda point: point)
        game.pyautogui = SimpleNamespace(click=lambda *point: clicks.append(point))
        game.focus = lambda: None

        result = game.click_detected_control(
            "claim",
            "领取任务奖励",
            allowed_states={"tasks"},
            target_states={"reward"},
            observation=source,
            preferred_point=(20, 30),
        )

        self.assertEqual("tasks", result.state)
        self.assertEqual([(20, 30)], clicks)

    def test_click_detected_control_rejects_wrong_source_state(self):
        control = DetectedControl("home", (10, 20, 30, 40), 0.99, "template")
        source = self.observation("unknown", controls=[control])
        game = self.sequence_game([source])
        game.execute = True
        game.geometry = SimpleNamespace(point=lambda point: point)
        clicks = []
        game.pyautogui = SimpleNamespace(click=lambda *point: clicks.append(point))
        game.focus = lambda: None

        with self.assertRaises(SafetyStop):
            game.click_detected_control(
                "home",
                "恢复主页",
                allowed_states={"tasks"},
                observation=source,
            )

        self.assertEqual([], clicks)

    def test_recovery_uses_only_configured_safe_route(self):
        game = DesktopGame.__new__(DesktopGame)
        game.config = copy.deepcopy(self.base_config)
        states = iter(
            [self.observation("tasks"), self.observation("main")]
        )
        game.wait_for_state = lambda *_args, **_kwargs: next(states)
        actions = []

        def click(name, _label, **kwargs):
            actions.append((name, kwargs["allowed_states"], kwargs["target_states"]))
            return self.observation("unknown_transient", change=20.0)

        game.click_detected_control = click

        result = game.recover_to_state({"main"}, hard_timeout=2.0)

        self.assertEqual("main", result.state)
        self.assertEqual(
            [("home", {"tasks"}, {"main", "trial"})],
            actions,
        )

    def test_recovery_returns_target_already_verified_by_action(self):
        game = DesktopGame.__new__(DesktopGame)
        game.config = copy.deepcopy(self.base_config)
        calls = []

        def wait_for_state(*_args, **_kwargs):
            calls.append("wait")
            if len(calls) > 1:
                raise AssertionError("verified target must not be sampled again")
            return self.observation("hunter_field")

        game.wait_for_state = wait_for_state
        game.click_detected_control = lambda *_args, **_kwargs: self.observation(
            "main", change=64.0
        )

        result = game.recover_to_state({"main"}, hard_timeout=2.0)

        self.assertEqual("main", result.state)
        self.assertEqual(["wait"], calls)

    def test_recovery_dismisses_explicit_overlay_then_routes_home(self):
        game = DesktopGame.__new__(DesktopGame)
        game.config = copy.deepcopy(self.base_config)
        states = iter(
            [
                self.observation("dismissible_overlay"),
                self.observation("tasks"),
                self.observation("main"),
            ]
        )
        game.wait_for_state = lambda *_args, **_kwargs: next(states)
        actions = []

        def click(name, _label, **kwargs):
            actions.append((name, kwargs["allowed_states"], kwargs["target_states"]))
            return self.observation("unknown_transient", change=20.0)

        game.click_detected_control = click

        result = game.recover_to_state({"main"}, hard_timeout=2.0)

        self.assertEqual("main", result.state)
        self.assertEqual(
            [
                (
                    "dismiss_overlay",
                    {"dismissible_overlay"},
                    {"main", "tasks", "tower_ready", "trial"},
                ),
                ("home", {"tasks"}, {"main", "trial"}),
            ],
            actions,
        )

    def test_recovery_closes_ladder_reward_panel_and_waits_through_transients(self):
        close = DetectedControl(
            "close_reward_panel", (922, 130, 960, 178), 0.99, "template"
        )
        panel = self.observation("ladder_reward_panel", controls=[close])
        game = self.sequence_game(
            [
                panel,
                panel,
                self.observation("unknown_transient", change=20.0),
                self.observation("loading", change=18.0),
                self.observation("unknown_transient", change=12.0),
                self.observation("main"),
                self.observation("main"),
            ]
        )
        clicks = []
        game.execute = True
        game.geometry = SimpleNamespace(point=lambda point: point)
        game.pyautogui = SimpleNamespace(click=lambda *point: clicks.append(point))
        game.focus = lambda: None

        result = game.recover_to_state({"main"}, hard_timeout=2.0)

        self.assertEqual("main", result.state)
        self.assertEqual([(941, 154)], clicks)

    def test_ladder_reward_panel_close_failure_stops_at_retry_limit(self):
        close = DetectedControl(
            "close_reward_panel", (922, 130, 960, 178), 0.99, "template"
        )
        panel = self.observation("ladder_reward_panel", controls=[close])
        game = self.sequence_game([panel], poll=0.05)
        game.config["recognition_v2"]["wait"]["action_timeout_seconds"] = 0.1
        game.config["recognition_v2"]["wait"]["action_hard_timeout_seconds"] = 0.15
        game.config["recognition_v2"]["wait"]["action_max_retries"] = 2
        clicks = []
        game.execute = True
        game.geometry = SimpleNamespace(point=lambda point: point)
        game.pyautogui = SimpleNamespace(click=lambda *point: clicks.append(point))
        game.focus = lambda: None
        clock = FakeClock()

        with patch("bot.time.monotonic", clock.monotonic), patch(
            "bot.time.sleep", clock.sleep
        ):
            with self.assertRaisesRegex(SafetyStop, "2 次可信点击"):
                game.click_detected_control(
                    "close_reward_panel",
                    "关闭天梯奖励面板",
                    allowed_states={"ladder_reward_panel"},
                    target_states={"main", "hunter_league", "trial"},
                    observation=panel,
                )

        self.assertEqual([(941, 154), (941, 154)], clicks)
        self.assertIn("action-failed-close_reward_panel", game.diagnostics)

    def test_unknown_state_recovery_times_out_without_clicking(self):
        game = self.sequence_game([self.observation("unknown")], poll=0.05)
        clicks = []
        game.click_detected_control = lambda *_args, **_kwargs: clicks.append(True)
        clock = FakeClock()
        with patch("bot.time.monotonic", clock.monotonic), patch(
            "bot.time.sleep", clock.sleep
        ):
            with self.assertRaises(PageTimeout):
                game.recover_to_state({"main"}, hard_timeout=0.3)

        self.assertEqual([], clicks)


if __name__ == "__main__":
    unittest.main()
