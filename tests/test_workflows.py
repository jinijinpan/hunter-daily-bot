import json
import copy
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot import DailyBot, DesktopGame, PageTimeout, SafetyStop
from recognition import DetectedControl, Observation, OCRToken


class FakeGame:
    def __init__(
        self,
        pages=None,
        gold_results=None,
        red_results=None,
        green_results=None,
        progress_results=None,
        character_indices=None,
        detected_page=None,
    ):
        self.pages = list(pages or [])
        self.gold_results = list(gold_results or [])
        self.red_results = list(red_results or [])
        self.green_results = list(green_results or [])
        self.progress_results = list(progress_results or [])
        self.character_indices = list(character_indices or [])
        self.detected_page = detected_page
        self.execute = True
        self.clicks = []
        self.waits = []
        self.drags = []
        self.diagnostics = []

    def focus(self):
        pass

    def detect_page(self):
        if self.detected_page is None:
            raise AssertionError("No queued detected page")
        return self.detected_page, {self.detected_page: 1.0}

    def click_reference(self, point, label, **_kwargs):
        self.clicks.append((tuple(point), label))

    def wait_for_page(self, page, **_kwargs):
        self.waits.append(page)

    def wait_for_one_of(self, pages, **_kwargs):
        self.waits.append(tuple(sorted(pages)))
        if not self.pages:
            raise AssertionError("No queued page result")
        page = self.pages.pop(0)
        if isinstance(page, Exception):
            raise page
        if page not in pages:
            raise AssertionError(f"Queued page {page} is not in {pages}")
        return page

    def normalized_capture(self):
        return object()

    def drag_reference(self, start, end, duration, label):
        self.drags.append((tuple(start), tuple(end), duration, label))

    def active_button(self, _region, _image):
        return True

    def gold_button(self, _region, _image=None):
        if self.gold_results:
            return self.gold_results.pop(0)
        return True

    def red_indicator(self, _region, _image=None):
        if self.red_results:
            return self.red_results.pop(0)
        return False

    def green_indicator(self, _region, _image=None):
        if self.green_results:
            return self.green_results.pop(0)
        return False

    def task_progress_complete(self, _name, _center, _image=None):
        if self.progress_results:
            return self.progress_results.pop(0)
        return False

    def active_character_index(self, _image=None):
        if not self.character_indices:
            raise AssertionError("No queued character index")
        return self.character_indices.pop(0)

    def save_diagnostic(self, name):
        self.diagnostics.append(name)


class CapturedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "config.json").open("r", encoding="utf-8") as stream:
            cls.config = json.load(stream)

    def test_tower_uses_quick_challenge_and_closes_result(self):
        game = FakeGame(["tower_result", "tower"])
        DailyBot(game, self.config)._run_tower()
        self.assertEqual(
            [self.config["points"]["tower_quick"], self.config["points"]["overlay_continue"]],
            [list(point) for point, _label in game.clicks],
        )
        self.assertEqual(
            [
                ("tower_result",),
                ("tower", "tower_manual"),
            ],
            game.waits,
        )

    def test_v2_main_tasks_tower_result_and_home_chain(self):
        class ObservationGame:
            def __init__(self, states):
                self.states = list(states)
                self.actions = []
                self.recoveries = []

            def wait_for_state(self, expected, **_kwargs):
                state = self.states.pop(0)
                if state not in expected:
                    raise AssertionError(f"V2 state {state} not in {expected}")
                controls = []
                if state == "tower_ready":
                    controls.append(
                        DetectedControl(
                            "quick_challenge",
                            (790, 570, 1025, 665),
                            0.99,
                            "color+ocr",
                        )
                    )
                return Observation(
                    timestamp=1.0,
                    viewport=(0, 0, 1091, 700),
                    state=state,
                    state_confidence=0.95,
                    controls=controls,
                )

            def click_detected_control(self, name, _label, **kwargs):
                self.actions.append(
                    (name, kwargs["allowed_states"], kwargs["target_states"])
                )
                return Observation(
                    timestamp=1.0,
                    viewport=(0, 0, 1091, 700),
                    state="unknown_transient",
                    state_confidence=1.0,
                    frame_change=20.0,
                )

            def recover_to_state(self, expected, **_kwargs):
                self.recoveries.append(set(expected))
                return Observation(
                    timestamp=1.0,
                    viewport=(0, 0, 1091, 700),
                    state="main",
                    state_confidence=0.95,
                )

        game = ObservationGame(
            [
                "main",
                "tasks",
                "tasks",
                "tower_ready",
                "tower_ready",
                "tower_result",
                "tower_ready",
            ]
        )
        bot = DailyBot(game, self.config)
        bot._find_task_button = lambda _task: ((512, 316), False)

        bot._open_daily_tasks()
        self.assertEqual("tower", bot._open_tower_task())
        bot._run_tower()
        bot._return_home_v2()

        self.assertEqual(
            ["secretary", "go", "quick_challenge", "dismiss_result"],
            [name for name, _sources, _targets in game.actions],
        )
        self.assertEqual([{"main"}], game.recoveries)

    def test_v2_tower_confirms_changed_difficulty_then_runs_manual_path(self):
        controls = {
            "tower_changed": DetectedControl(
                "confirm_difficulty", (489, 484, 604, 524), 0.99, "color+ocr"
            ),
            "tower_ready": DetectedControl(
                "start_challenge", (791, 596, 943, 639), 0.99, "color+ocr"
            ),
            "tower_battle_confirm": DetectedControl(
                "confirm_battle", (573, 485, 686, 525), 0.99, "color+ocr"
            ),
            "reward": DetectedControl(
                "dismiss_reward", (478, 610, 614, 628), 0.99, "ocr"
            ),
            "tower_post_battle": DetectedControl(
                "tower_exit", (25, 626, 173, 669), 0.99, "template"
            ),
        }

        class TowerGame:
            def __init__(self):
                self.states = iter(
                    [
                        "tasks",
                        "tower_changed",
                        "tower_ready",
                        "tower_ready",
                        "tower_battle_confirm",
                        "reward",
                        "tower_post_battle",
                        "tower_ready",
                    ]
                )
                self.actions = []

            def wait_for_state(self, expected, **_kwargs):
                state = next(self.states)
                if state not in expected:
                    raise AssertionError(f"V2 state {state} not in {expected}")
                state_controls = [controls[state]] if state in controls else []
                return Observation(
                    timestamp=1.0,
                    viewport=(0, 0, 1091, 700),
                    state=state,
                    state_confidence=0.99,
                    controls=state_controls,
                )

            def click_detected_control(self, name, _label, **kwargs):
                self.actions.append(
                    (name, set(kwargs["allowed_states"]), set(kwargs["target_states"]))
                )
                return Observation(1.0, (0, 0, 1091, 700), state="unknown_transient")

        game = TowerGame()
        bot = DailyBot(game, self.config)
        bot._find_task_button = lambda _task: ((512, 316), False)

        self.assertEqual("tower", bot._open_tower_task())
        bot._run_tower()

        self.assertEqual(
            [
                "go",
                "confirm_difficulty",
                "start_challenge",
                "confirm_battle",
                "dismiss_reward",
                "tower_exit",
            ],
            [name for name, _sources, _targets in game.actions],
        )
        self.assertEqual(
            {"tower_ready"},
            game.actions[1][2],
        )

    def test_v2_tower_prefers_quick_challenge_when_both_controls_exist(self):
        quick = DetectedControl(
            "quick_challenge", (790, 570, 1025, 665), 0.99, "color+ocr"
        )
        start = DetectedControl(
            "start_challenge", (790, 510, 1025, 555), 0.99, "color+ocr"
        )

        class TowerGame:
            def __init__(self):
                self.states = iter(["tower_ready", "tower_ready"])
                self.actions = []

            def wait_for_state(self, expected, **_kwargs):
                state = next(self.states)
                controls = [quick, start] if state == "tower_ready" else []
                return Observation(
                    1.0, (0, 0, 1091, 700), controls=controls,
                    state=state, state_confidence=0.99,
                )

            def click_detected_control(self, name, _label, **kwargs):
                self.actions.append(name)
                return Observation(1.0, (0, 0, 1091, 700), state="tower_result")

        game = TowerGame()
        bot = DailyBot(game, self.config)
        bot._finish_tower_quick_result = lambda _observation: None

        bot._run_tower()

        self.assertEqual(["quick_challenge"], game.actions)

    def test_v2_vertical_chain_challenges_then_returns_and_claims_reward(self):
        def observed(state, controls=()):
            return Observation(
                timestamp=1.0,
                viewport=(0, 0, 1091, 700),
                state=state,
                state_confidence=0.99,
                controls=list(controls),
            )

        claim = DetectedControl(
            "claim", (450, 220, 578, 257), 0.9996, "color+ocr"
        )

        class VerticalGame:
            def __init__(self):
                self.observations = [
                    observed("main"),
                    observed("tasks"),
                    observed("tasks"),
                    observed("tower_ready", [
                        DetectedControl(
                            "quick_challenge",
                            (790, 570, 1025, 665),
                            0.99,
                            "color+ocr",
                        )
                    ]),
                    observed("tower_ready", [
                        DetectedControl(
                            "quick_challenge",
                            (790, 570, 1025, 665),
                            0.99,
                            "color+ocr",
                        )
                    ]),
                    observed("tower_result"),
                    observed("tower_ready"),
                    observed("main"),
                    observed("tasks", [claim]),
                    observed("tasks", [claim]),
                    observed("tasks"),
                    observed("tasks"),
                ]
                self.actions = []
                self.recoveries = []
                self.drags = []

            def wait_for_state(self, expected, **_kwargs):
                observation = self.observations.pop(0)
                if observation.state not in expected:
                    raise AssertionError(
                        f"V2 state {observation.state} not in {expected}"
                    )
                return observation

            def click_detected_control(self, name, _label, **kwargs):
                self.actions.append(name)
                if name == "claim":
                    return observed("reward", [
                        DetectedControl(
                            "dismiss_reward", (400, 570, 700, 660), 0.99, "ocr"
                        )
                    ])
                if name == "dismiss_reward":
                    return observed("tasks")
                return observed("unknown_transient")

            def recover_to_state(self, expected, **_kwargs):
                self.recoveries.append(set(expected))
                return observed("main")

            def drag_reference(self, start, end, _duration, label):
                self.drags.append((start, end, label))

            def save_diagnostic(self, _name):
                pass

        game = VerticalGame()
        bot = DailyBot(game, self.config)
        bot._find_task_button = lambda _task: ((512, 316), False)
        bot._claim_activity_rewards = lambda: None

        bot._open_daily_tasks()
        self.assertEqual("tower", bot._open_tower_task())
        bot._run_tower()
        bot._return_home_v2()
        bot._open_daily_tasks()
        bot._reset_task_scroll_to_top = lambda: None
        bot._claim_task_rewards()
        bot._return_home_v2()

        self.assertEqual(
            [
                "secretary",
                "go",
                "quick_challenge",
                "dismiss_result",
                "secretary",
                "claim",
                "dismiss_reward",
            ],
            game.actions,
        )
        self.assertEqual([{"main"}, {"main"}], game.recoveries)
        self.assertEqual([], game.observations)

    def test_v2_task_claims_choose_one_highest_confidence_control_at_a_time(self):
        def tasks(*controls):
            return Observation(
                timestamp=1.0,
                viewport=(0, 0, 1091, 700),
                state="tasks",
                state_confidence=0.99,
                controls=list(controls),
            )

        low = DetectedControl("claim", (10, 20, 30, 40), 0.80, "color+ocr")
        high = DetectedControl("claim", (110, 20, 130, 40), 0.99, "color+ocr")
        middle = DetectedControl("claim", (210, 20, 230, 40), 0.90, "color+ocr")

        class ClaimGame:
            def __init__(self):
                self.waits = [tasks(low, high, middle), tasks()]
                self.results = [tasks(low, middle), tasks(low), tasks()]
                self.points = []

            def wait_for_state(self, expected, **_kwargs):
                result = self.waits.pop(0)
                self.assert_expected = expected
                return result

            def click_detected_control(self, name, _label, **kwargs):
                self.points.append((name, kwargs["preferred_point"]))
                return self.results.pop(0)

            def drag_reference(self, *_args):
                pass

            def save_diagnostic(self, _name):
                pass

            def recover_to_state(self, expected, **_kwargs):
                return tasks()

        game = ClaimGame()
        bot = DailyBot(game, self.config)
        bot._reset_task_scroll_to_top = lambda: None
        bot._claim_activity_rewards = lambda: None

        bot._claim_task_rewards()

        self.assertEqual(
            [("claim", (120, 30)), ("claim", (220, 30)), ("claim", (20, 30))],
            game.points,
        )

    def test_v2_task_claim_does_not_repeat_a_stale_claim(self):
        claim = DetectedControl("claim", (10, 20, 30, 40), 0.99, "color+ocr")
        observation = Observation(
            timestamp=1.0,
            viewport=(0, 0, 1091, 700),
            state="tasks",
            state_confidence=0.99,
            controls=[claim],
        )

        class StaleClaimGame:
            def __init__(self):
                self.clicks = 0
                self.diagnostics = []

            def wait_for_state(self, expected, **_kwargs):
                self.assert_expected = expected
                return observation

            def click_detected_control(self, *_args, **_kwargs):
                self.clicks += 1
                return observation

            def save_diagnostic(self, name):
                self.diagnostics.append(name)

            def recover_to_state(self, expected, **_kwargs):
                return observation

        game = StaleClaimGame()
        bot = DailyBot(game, self.config)
        bot._reset_task_scroll_to_top = lambda: None

        with self.assertRaises(SafetyStop):
            bot._claim_task_rewards()

        self.assertEqual(1, game.clicks)
        self.assertEqual(["task-claim-stale-control"], game.diagnostics)

    def test_task_claim_can_be_rolled_back_to_legacy_independently(self):
        config = copy.deepcopy(self.config)
        config["recognition_v2"]["workflow_modes"]["main_tasks_claim"] = "legacy"
        regions = config["reward_button_regions"]
        game = FakeGame(gold_results=[False] * (len(regions) * 2))
        detected_actions = []
        game.wait_for_state = lambda *_args, **_kwargs: None
        game.click_detected_control = (
            lambda *args, **_kwargs: detected_actions.append((args, kwargs))
        )
        game.recover_to_state = lambda *_args, **_kwargs: None
        bot = DailyBot(game, config)
        bot._reset_task_scroll_to_top = lambda: None
        bot._claim_activity_rewards = lambda: None

        bot._claim_task_rewards()

        self.assertEqual([], detected_actions)
        self.assertEqual(1, len(game.drags))

    def test_v2_tasks_hunter_failure_reward_and_home_chain(self):
        class ObservationGame:
            def __init__(self, states):
                self.states = list(states)
                self.actions = []
                self.recoveries = []

            def wait_for_state(self, expected, **_kwargs):
                state = self.states.pop(0)
                if state not in expected:
                    raise AssertionError(f"V2 state {state} not in {expected}")
                controls = []
                if state == "hunter_field":
                    controls.append(
                        DetectedControl(
                            "hunter_start", (925, 593, 1077, 635), 0.99, "color+ocr"
                        )
                    )
                return Observation(
                    timestamp=1.0,
                    viewport=(0, 0, 1091, 700),
                    state=state,
                    state_confidence=0.99,
                    controls=controls,
                )

            def click_detected_control(self, name, _label, **kwargs):
                self.actions.append(
                    (name, kwargs["allowed_states"], kwargs["target_states"])
                )
                return Observation(
                    timestamp=1.0,
                    viewport=(0, 0, 1091, 700),
                    state="unknown_transient",
                    state_confidence=1.0,
                    frame_change=20.0,
                )

            def recover_to_state(self, expected, **_kwargs):
                self.recoveries.append(set(expected))
                return Observation(
                    timestamp=1.0,
                    viewport=(0, 0, 1091, 700),
                    state="main",
                    state_confidence=0.99,
                )

        game = ObservationGame(
            [
                "tasks",
                "hunter_field",
                "hunter_field",
                "hunter_failure",
                "hunter_confirm",
                "reward",
                "hunter_field",
                "main",
                "tasks",
            ]
        )
        bot = DailyBot(game, self.config)
        task_results = iter(
            [((1008, 503), False), ((1008, 503), True)]
        )
        bot._find_task_button = lambda _task: next(task_results)

        self.assertEqual("hunter_field", bot._open_hunter_task())
        bot._run_hunter_field("hunter_field")
        bot._return_home_v2()

        self.assertEqual(
            [
                "go",
                "hunter_start",
                "hunter_speed",
                "hunter_confirm",
                "dismiss_reward",
                "secretary",
            ],
            [name for name, _sources, _targets in game.actions],
        )
        self.assertEqual([{"main"}, {"main"}], game.recoveries)

    def test_v2_hunter_disabled_returns_to_tasks_and_requires_completed_progress(self):
        class DisabledGame:
            def __init__(self):
                self.actions = []
                self.recoveries = []

            def wait_for_state(self, expected, **_kwargs):
                self.assert_expected = expected
                return Observation(
                    timestamp=1.0,
                    viewport=(0, 0, 1091, 700),
                    state="hunter_field",
                    state_confidence=0.99,
                    controls=[
                        DetectedControl(
                            "quick_disabled", (820, 570, 1010, 665), 0.99, "color+ocr"
                        )
                    ],
                )

            def click_detected_control(self, name, _label, **_kwargs):
                self.actions.append(name)

            def recover_to_state(self, expected, **_kwargs):
                self.recoveries.append(set(expected))
                return Observation(
                    timestamp=1.0,
                    viewport=(0, 0, 1091, 700),
                    state="main",
                    state_confidence=0.99,
                )

        game = DisabledGame()
        bot = DailyBot(game, self.config)
        bot._open_daily_tasks = lambda: game.actions.append("secretary")
        bot._find_task_button = lambda _task: ((1012, 318), False)

        with self.assertRaisesRegex(SafetyStop, "任务仍未完成"):
            bot._run_hunter_field("hunter_field")

        self.assertEqual(["secretary"], game.actions)
        self.assertEqual([{"main"}], game.recoveries)

    def test_v2_hunter_disabled_accepts_completed_progress_after_safe_return(self):
        class DisabledGame:
            def wait_for_state(self, _expected, **_kwargs):
                return Observation(
                    timestamp=1.0,
                    viewport=(0, 0, 1091, 700),
                    state="hunter_field",
                    state_confidence=0.99,
                    controls=[
                        DetectedControl(
                            "quick_disabled", (820, 570, 1010, 665), 0.99, "color+ocr"
                        )
                    ],
                )

            @staticmethod
            def click_detected_control(_name, _label, **_kwargs):
                raise AssertionError("disabled hunter path must not click a challenge control")

            @staticmethod
            def recover_to_state(_expected, **_kwargs):
                return Observation(
                    timestamp=1.0,
                    viewport=(0, 0, 1091, 700),
                    state="main",
                    state_confidence=0.99,
                )

        game = DisabledGame()
        bot = DailyBot(game, self.config)
        bot._open_daily_tasks = lambda: None
        bot._find_task_button = lambda _task: ((1012, 318), True)

        bot._run_hunter_field("hunter_field")

    def test_v2_hunter_requires_completed_progress_after_reward(self):
        class HunterGame:
            def __init__(self):
                self.states = iter(
                    [
                        "hunter_quick_ready",
                        "hunter_confirm",
                        "reward",
                        "hunter_field",
                        "main",
                        "tasks",
                    ]
                )
                self.actions = []
                self.recoveries = []

            def wait_for_state(self, expected, **_kwargs):
                state = next(self.states)
                if state not in expected:
                    raise AssertionError(f"V2 state {state} not in {expected}")
                return Observation(
                    1.0, (0, 0, 1091, 700), state=state, state_confidence=0.99
                )

            def click_detected_control(self, name, _label, **_kwargs):
                self.actions.append(name)
                return Observation(1.0, (0, 0, 1091, 700), state="unknown_transient")

            def recover_to_state(self, expected, **_kwargs):
                self.recoveries.append(set(expected))
                return Observation(1.0, (0, 0, 1091, 700), state="main")

        game = HunterGame()
        bot = DailyBot(game, self.config)
        bot._find_task_button = lambda _task: ((1010, 503), False)

        with self.assertRaisesRegex(SafetyStop, "任务仍未完成"):
            bot._run_hunter_field("hunter_quick_ready")

        self.assertEqual(
            ["quick_clear", "hunter_confirm", "dismiss_reward", "secretary"],
            game.actions,
        )
        self.assertEqual([{"main"}], game.recoveries)

    def test_v2_resource_supply_uses_detected_controls_for_full_path(self):
        class ResourceGame:
            def __init__(self):
                self.states = iter(
                    [
                        "tasks",
                        "resource_hub",
                        "resource_hub",
                        "resource_dialog",
                        "resource_dialog",
                        "resource_confirm",
                        "reward",
                        "resource_dialog",
                        "resource_hub",
                    ]
                )
                self.actions = []

            def wait_for_state(self, expected, **_kwargs):
                state = next(self.states)
                if state not in expected:
                    raise AssertionError(f"V2 state {state} not in {expected}")
                return Observation(
                    1.0, (0, 0, 1091, 700), state=state, state_confidence=0.99
                )

            def click_detected_control(self, name, _label, **kwargs):
                self.actions.append(
                    (name, set(kwargs["allowed_states"]), set(kwargs["target_states"]))
                )
                return Observation(1.0, (0, 0, 1091, 700), state="unknown_transient")

            def click_reference(self, *_args, **_kwargs):
                raise AssertionError("V2 资源补给不得使用固定坐标")

        game = ResourceGame()
        bot = DailyBot(game, self.config)
        bot._find_task_button = lambda _task: ((512, 399), False)

        self.assertTrue(bot._open_resource_task())
        bot._run_resource_supply()

        self.assertEqual(
            [
                "go",
                "resource_magic",
                "resource_quick",
                "resource_confirm",
                "dismiss_reward",
                "close_resource",
            ],
            [name for name, _sources, _targets in game.actions],
        )
        self.assertEqual(
            {"resource_confirm", "reward"},
            game.actions[2][2],
        )

    def test_v2_resource_supply_stops_when_quick_control_is_missing(self):
        class ResourceGame:
            def wait_for_state(self, expected, **_kwargs):
                return Observation(
                    1.0,
                    (0, 0, 1091, 700),
                    state="resource_dialog",
                    state_confidence=0.99,
                    controls=[],
                )

            def click_detected_control(self, name, _label, **kwargs):
                observation = kwargs["observation"]
                if not any(control.name == name for control in observation.controls):
                    raise SafetyStop("未检测到可信控件 resource_quick")
                raise AssertionError("不应执行点击")

        with self.assertRaisesRegex(SafetyStop, "resource_quick"):
            DailyBot(ResourceGame(), self.config)._run_resource_quick()

    def test_v2_startup_from_tasks_recovers_home_before_workflow(self):
        game = FakeGame(detected_page="tasks")
        game.wait_for_state = lambda *_args, **_kwargs: None
        game.click_detected_control = lambda *_args, **_kwargs: None
        recoveries = []
        game.recover_to_state = lambda expected, **_kwargs: recoveries.append(set(expected))
        bot = DailyBot(game, self.config)
        bot._run_character_cycles = lambda: None
        bot._report_unimplemented_tasks = lambda: None

        bot.run()

        self.assertEqual([{"main"}], recoveries)

    def test_v2_unknown_startup_uses_safe_recovery_before_workflow(self):
        game = FakeGame(detected_page="unknown")
        game.wait_for_state = lambda *_args, **_kwargs: None
        game.click_detected_control = lambda *_args, **_kwargs: None
        recoveries = []

        def recover(expected, **_kwargs):
            recoveries.append(set(expected))
            return Observation(
                timestamp=1.0,
                viewport=(0, 0, 1091, 700),
                state="main",
                state_confidence=0.95,
            )

        game.recover_to_state = recover
        bot = DailyBot(game, self.config, resume=True)
        bot._run_character_cycles = lambda: None
        bot._report_unimplemented_tasks = lambda: None

        bot.run()

        self.assertEqual([{"main"}], recoveries)

    def test_tower_confirms_changed_difficulty_before_continuing(self):
        game = FakeGame(["tower_changed", "tower"])
        bot = DailyBot(game, self.config)
        bot._find_task_button = lambda _task: ((512, 316), False)

        self.assertEqual("tower", bot._open_tower_task())

        expected = [[512, 316], self.config["points"]["tower_changed_confirm"]]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])
        self.assertEqual(
            [
                ("tower", "tower_changed", "tower_manual"),
                ("tower", "tower_manual"),
            ],
            game.waits,
        )

    def test_tower_uses_manual_battle_when_quick_challenge_is_unavailable(self):
        game = FakeGame(
            ["tower_battle_confirm", "reward", "tower_post_battle", "tower_manual"]
        )
        DailyBot(game, self.config)._run_tower_manual()
        expected = [
            self.config["points"]["tower_start"],
            self.config["points"]["tower_battle_confirm"],
            self.config["points"]["tower_reward_close"],
            self.config["points"]["tower_exit"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])
        self.assertEqual(
            [
                ("tower_battle_confirm",),
                ("reward",),
                ("tower_post_battle",),
                ("tower", "tower_manual", "trial"),
            ],
            game.waits,
        )

    def test_tower_can_resume_from_battle_confirmation(self):
        game = FakeGame(["reward", "tower_post_battle", "tower_manual"])

        DailyBot(game, self.config)._finish_tower_manual_battle()

        expected = [
            self.config["points"]["tower_battle_confirm"],
            self.config["points"]["tower_reward_close"],
            self.config["points"]["tower_exit"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])
        self.assertEqual(
            [
                ("reward",),
                ("tower_post_battle",),
                ("tower", "tower_manual", "trial"),
            ],
            game.waits,
        )

    def test_v2_tower_can_resume_from_detected_battle_confirmation(self):
        controls = {
            "tower_battle_confirm": DetectedControl(
                "confirm_battle", (573, 483, 689, 523), 0.99, "color+ocr"
            ),
            "reward": DetectedControl(
                "dismiss_reward", (478, 610, 614, 628), 0.99, "ocr"
            ),
            "tower_post_battle": DetectedControl(
                "tower_exit", (25, 626, 173, 669), 0.99, "template"
            ),
        }

        class TowerGame:
            def __init__(self):
                self.states = iter(
                    [
                        "tower_battle_confirm",
                        "reward",
                        "tower_post_battle",
                        "tower_ready",
                    ]
                )
                self.actions = []

            def wait_for_state(self, expected, **_kwargs):
                state = next(self.states)
                if state not in expected:
                    raise AssertionError(f"V2 state {state} not in {expected}")
                state_controls = [controls[state]] if state in controls else []
                return Observation(
                    1.0,
                    (0, 0, 1091, 700),
                    controls=state_controls,
                    state=state,
                    state_confidence=0.99,
                )

            def click_detected_control(self, name, _label, **kwargs):
                self.actions.append(
                    (name, set(kwargs["allowed_states"]), set(kwargs["target_states"]))
                )
                return Observation(1.0, (0, 0, 1091, 700), state="unknown_transient")

        game = TowerGame()
        DailyBot(game, self.config)._finish_tower_manual_battle()

        self.assertEqual(
            ["confirm_battle", "dismiss_reward", "tower_exit"],
            [name for name, _sources, _targets in game.actions],
        )

    def test_v2_tower_closes_multiple_detected_reward_layers(self):
        reward_controls = [
            DetectedControl(
                "dismiss_reward", (478, 608, 614, 628), 0.99, "ocr"
            )
        ]

        class TowerGame:
            def __init__(self):
                self.states = iter(["reward", "tower_post_battle"])
                self.actions = []

            def wait_for_state(self, expected, **_kwargs):
                state = next(self.states)
                if state not in expected:
                    raise AssertionError(f"V2 state {state} not in {expected}")
                controls = reward_controls if state == "reward" else []
                return Observation(
                    1.0,
                    (0, 0, 1091, 700),
                    controls=controls,
                    state=state,
                    state_confidence=0.99,
                )

            def click_detected_control(self, name, _label, **_kwargs):
                self.actions.append(name)
                return Observation(1.0, (0, 0, 1091, 700), state="unknown_transient")

        game = TowerGame()
        first_reward = Observation(
            1.0,
            (0, 0, 1091, 700),
            controls=reward_controls,
            state="reward",
            state_confidence=0.99,
        )

        result = DailyBot(game, self.config)._close_tower_rewards_v2(first_reward)

        self.assertEqual("tower_post_battle", result.state)
        self.assertEqual(["dismiss_reward", "dismiss_reward"], game.actions)

    def test_v2_tower_failure_uses_detected_continue_and_returns_to_lobby(self):
        controls = {
            "tower_failure": [
                DetectedControl(
                    "dismiss_tower_failure", (478, 545, 613, 566), 0.99, "ocr"
                )
            ]
        }

        class TowerGame:
            def __init__(self):
                self.states = iter(["tower_failure", "tower_ready"])
                self.actions = []

            def wait_for_state(self, expected, **_kwargs):
                state = next(self.states)
                if state not in expected:
                    raise AssertionError(f"V2 state {state} not in {expected}")
                return Observation(
                    1.0,
                    (0, 0, 1091, 700),
                    controls=controls.get(state, []),
                    state=state,
                    state_confidence=0.99,
                )

            def click_detected_control(self, name, _label, **kwargs):
                self.actions.append(
                    (name, set(kwargs["allowed_states"]), set(kwargs["target_states"]))
                )
                return Observation(1.0, (0, 0, 1091, 700), state="unknown_transient")

        game = TowerGame()
        confirmation = Observation(
            1.0,
            (0, 0, 1091, 700),
            controls=[
                DetectedControl(
                    "confirm_battle", (573, 483, 689, 523), 0.99, "color+ocr"
                )
            ],
            state="tower_battle_confirm",
            state_confidence=0.99,
        )

        DailyBot(game, self.config)._finish_tower_manual_battle_v2(confirmation)

        self.assertEqual(
            ["confirm_battle", "dismiss_tower_failure"],
            [name for name, _sources, _targets in game.actions],
        )

    def test_tower_exit_retries_clicking_until_lobby_returns(self):
        game = FakeGame(
            [
                PageTimeout("first timeout"),
                PageTimeout("second timeout"),
                "trial",
            ]
        )

        DailyBot(game, self.config)._exit_tower_post_battle()

        expected = [
            self.config["points"]["tower_exit"],
            self.config["points"]["tower_exit_retry"],
            self.config["points"]["tower_exit_retry"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])
        self.assertEqual(
            [("tower", "tower_manual", "trial")] * 3,
            game.waits,
        )

    def test_resume_from_tower_changed_completes_tower_and_returns_home(self):
        game = FakeGame(["tower", "tower_result", "tower", "main"])
        DailyBot(game, self.config)._resume_tower_changed()
        expected = [
            self.config["points"]["tower_changed_confirm"],
            self.config["points"]["tower_quick"],
            self.config["points"]["overlay_continue"],
            self.config["points"]["home"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])
        self.assertEqual(
            [
                ("tower", "tower_manual"),
                ("tower_result",),
                ("tower", "tower_manual"),
                ("main", "trial"),
            ],
            game.waits,
        )

    def test_return_home_passes_through_trial_hub(self):
        game = FakeGame(["trial"])
        DailyBot(game, self.config)._return_home()
        self.assertEqual(
            [self.config["points"]["home"], self.config["points"]["home"]],
            [list(point) for point, _label in game.clicks],
        )
        self.assertEqual([("main", "trial"), "main"], game.waits)

    def test_return_home_retries_when_first_click_is_ignored(self):
        game = FakeGame([PageTimeout("home click ignored"), "main"])

        DailyBot(game, self.config)._return_home()

        self.assertEqual(
            [self.config["points"]["home"]] * 2,
            [list(point) for point, _label in game.clicks],
        )
        self.assertEqual([("main", "trial")] * 2, game.waits)

    def test_supply_skips_claimed_slots_and_continues_after_no_reward(self):
        game = FakeGame(
            ["supply", "reward"],
            green_results=[True, True, False, False, True, True],
        )
        DailyBot(game, self.config)._collect_daily_supply()
        claim_clicks = [
            point for point, label in game.clicks if label == "领取每日补给"
        ]
        expected = [
            ((x1 + x2) // 2, (y1 + y2) // 2)
            for x1, y1, x2, y2 in self.config["supply_claim_regions"][2:]
        ]
        self.assertEqual(expected, claim_clicks)
        self.assertEqual(["supply-finished"], game.diagnostics)

    def test_hunter_field_uses_captured_quick_pass_path(self):
        game = FakeGame(
            ["hunter_failure", "hunter_confirm", "reward", "hunter_field"]
        )
        DailyBot(game, self.config)._run_hunter_field()
        expected = [
            self.config["points"]["hunter_start"],
            self.config["points"]["hunter_speed"],
            self.config["points"]["hunter_confirm"],
            self.config["points"]["overlay_continue"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])

    def test_hunter_field_prefers_direct_quick_clear_when_available(self):
        game = FakeGame(
            [
                PageTimeout("quick button did not respond"),
                "hunter_quick_available",
                "hunter_confirm",
                "reward",
                "hunter_quick_available",
            ]
        )
        DailyBot(game, self.config)._run_hunter_field("hunter_quick_available")
        expected = [
            self.config["points"]["hunter_quick"],
            self.config["points"]["hunter_quick"],
            self.config["points"]["hunter_confirm"],
            self.config["points"]["overlay_continue"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])
        self.assertEqual(
            [
                ("hunter_confirm", "reward"),
                ("hunter_quick_available",),
                ("hunter_confirm", "reward"),
                ("reward",),
                ("hunter_field", "hunter_quick_available"),
            ],
            game.waits,
        )

    def test_hunter_confirm_can_resume_and_close_reward(self):
        game = FakeGame(["reward", "hunter_quick_available"])

        DailyBot(game, self.config)._finish_hunter_confirm()

        expected = [
            self.config["points"]["hunter_confirm"],
            self.config["points"]["overlay_continue"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])
        self.assertEqual(
            [
                ("reward",),
                ("hunter_field", "hunter_quick_available"),
            ],
            game.waits,
        )

    def test_resource_supply_closes_modal_before_returning(self):
        game = FakeGame(["reward"])
        DailyBot(game, self.config)._run_resource_supply()
        expected = [
            self.config["points"]["resource_magic_card"],
            self.config["points"]["resource_speed"],
            self.config["points"]["overlay_continue"],
            self.config["points"]["resource_close"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])
        self.assertEqual(
            [
                "resource_dialog",
                ("resource_confirm", "reward"),
                "resource_dialog",
                "resource_hub",
            ],
            game.waits,
        )

    def test_resource_supply_confirms_quick_clear_when_prompted(self):
        game = FakeGame(["resource_confirm"])
        DailyBot(game, self.config)._run_resource_supply()
        expected = [
            self.config["points"]["resource_magic_card"],
            self.config["points"]["resource_speed"],
            self.config["points"]["resource_confirm"],
            self.config["points"]["overlay_continue"],
            self.config["points"]["resource_close"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])
        self.assertEqual(
            [
                "resource_dialog",
                ("resource_confirm", "reward"),
                "reward",
                "resource_dialog",
                "resource_hub",
            ],
            game.waits,
        )

    def test_abyss_repeats_until_stamina_is_exhausted_then_returns(self):
        game = FakeGame(
            [
                "abyss_victory",
                "abyss_cards",
                "reward",
                "abyss_finished",
                "abyss_victory",
                "abyss_finished",
                "stamina_get",
                "abyss_finished",
            ]
        )
        DailyBot(game, self.config)._run_abyss()
        expected = [
            self.config["points"]["abyss_single"],
            self.config["points"]["overlay_continue"],
            self.config["points"]["abyss_cards_skip"],
            self.config["points"]["overlay_continue"],
            self.config["points"]["abyss_retry"],
            self.config["points"]["overlay_continue"],
            self.config["points"]["abyss_retry"],
            self.config["points"]["popup_close"],
            self.config["points"]["abyss_return_safe"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])
        self.assertEqual("abyss", game.waits[-1])

    def test_abyss_retries_victory_continue_until_page_changes(self):
        game = FakeGame(
            [
                "abyss_victory",
                PageTimeout("victory did not change"),
                "abyss_cards",
                "reward",
                "abyss_finished",
                "stamina_get",
                "abyss",
            ]
        )

        DailyBot(game, self.config)._run_abyss()

        continue_point = tuple(self.config["points"]["overlay_continue"])
        continue_clicks = [point for point, label in game.clicks if label == "继续深渊结算"]
        self.assertEqual([continue_point, continue_point], continue_clicks)

    def test_abyss_retries_cards_skip_and_reward_close(self):
        game = FakeGame(
            [
                "abyss_victory",
                "abyss_cards",
                PageTimeout("cards did not change"),
                "reward",
                PageTimeout("reward did not change"),
                "abyss_finished",
                "stamina_get",
                "abyss",
            ]
        )

        DailyBot(game, self.config)._run_abyss()

        labels = [label for _point, label in game.clicks]
        self.assertEqual(2, labels.count("跳过深渊翻牌"))
        self.assertEqual(2, labels.count("关闭深渊奖励"))

    def test_abyss_accepts_next_victory_when_countdown_skips_finished_page(self):
        game = FakeGame(
            [
                "abyss_victory",
                "reward",
                "abyss_victory",
                "abyss_finished",
                "stamina_get",
                "abyss",
            ]
        )

        DailyBot(game, self.config)._run_abyss()

        labels = [label for _point, label in game.clicks]
        self.assertEqual(2, labels.count("继续深渊结算"))
        self.assertEqual(1, labels.count("深渊再次挑战"))
        self.assertEqual(1, labels.count("关闭体力不足弹窗"))

    def test_abyss_returns_when_stamina_is_zero_and_only_safe_button_remains(self):
        game = FakeGame(
            [
                "abyss_victory",
                "reward",
                "abyss_exhausted",
                "abyss",
            ]
        )

        DailyBot(game, self.config)._run_abyss()

        expected = [
            self.config["points"]["abyss_single"],
            self.config["points"]["overlay_continue"],
            self.config["points"]["overlay_continue"],
            self.config["points"]["abyss_exhausted_return_safe"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])
        labels = [label for _point, label in game.clicks]
        self.assertNotIn("深渊再次挑战", labels)
        self.assertNotIn("关闭体力不足弹窗", labels)
        self.assertEqual("abyss", game.waits[-1])

    def test_abyss_can_resume_from_cards_without_starting_a_new_run(self):
        game = FakeGame(
            ["reward", "abyss_finished", "stamina_get", "abyss"]
        )

        DailyBot(game, self.config)._run_abyss(initial_page="abyss_cards")

        labels = [label for _point, label in game.clicks]
        self.assertEqual(
            [
                "跳过深渊翻牌",
                "关闭深渊奖励",
                "深渊再次挑战",
                "关闭体力不足弹窗",
            ],
            labels,
        )
        self.assertNotIn("深渊挑战单人挑战", labels)

    def test_startup_from_abyss_settlement_resumes_current_page(self):
        for detected_page in ("abyss_victory", "abyss_cards", "abyss_finished"):
            with self.subTest(page=detected_page):
                game = FakeGame(detected_page=detected_page)
                bot = DailyBot(game, self.config)
                calls = []
                bot._run_abyss = lambda *, initial_page: calls.append(initial_page)
                bot._return_home = lambda: calls.append("home")
                bot._run_character_cycles = lambda: calls.append("characters")
                bot._report_unimplemented_tasks = lambda: calls.append("report")

                bot.run()

                self.assertEqual(
                    [detected_page, "home", "characters", "report"], calls
                )

    def test_startup_from_abyss_continues_challenge_then_returns_home(self):
        game = FakeGame(detected_page="abyss")
        bot = DailyBot(game, self.config)
        calls = []
        bot._run_abyss = lambda: calls.append("abyss")
        bot._return_home = lambda: calls.append("home")
        bot._run_character_cycles = lambda: calls.append("characters")
        bot._report_unimplemented_tasks = lambda: calls.append("report")

        bot.run()

        self.assertEqual(["abyss", "home", "characters", "report"], calls)
        self.assertEqual(["tasks-finished"], game.diagnostics)

    def test_startup_from_exhausted_abyss_returns_then_continues_tasks(self):
        game = FakeGame(detected_page="abyss_exhausted")
        bot = DailyBot(game, self.config)
        calls = []
        bot._finish_abyss_exhausted = lambda: calls.append("exhausted")
        bot._return_home = lambda: calls.append("home")
        bot._run_character_cycles = lambda: calls.append("characters")
        bot._report_unimplemented_tasks = lambda: calls.append("report")

        bot.run()

        self.assertEqual(["exhausted", "home", "characters", "report"], calls)

    def test_startup_resumes_other_battle_settlement_pages(self):
        cases = [
            ("tower_result", "_finish_tower_quick_result", ()),
            ("hunter_failure", "_finish_hunter_failure", ()),
            ("monster_reward", "_finish_monster_reward", ()),
            ("hunter_league_failure", "_finish_hunter_league_result", ()),
            (
                "hunter_league_rewards",
                "_close_hunter_league_rewards",
                ("hunter_league_rewards",),
            ),
            ("infinite_next", "_finish_infinite_mystery", ("infinite_next",)),
        ]
        for page, method_name, expected_args in cases:
            with self.subTest(page=page):
                game = FakeGame(detected_page=page)
                bot = DailyBot(game, self.config)
                calls = []
                setattr(
                    bot,
                    method_name,
                    lambda *args, name=method_name: calls.append((name, args)),
                )
                bot._return_home = lambda: calls.append(("home", ()))
                bot._run_character_cycles = lambda: calls.append(("characters", ()))
                bot._report_unimplemented_tasks = lambda: calls.append(("report", ()))

                bot.run()

                self.assertEqual((method_name, expected_args), calls[0])
                self.assertEqual(
                    [("home", ()), ("characters", ()), ("report", ())],
                    calls[1:],
                )

    def test_monster_invasion_uses_quick_match_for_each_available_attempt(self):
        config = copy.deepcopy(self.config)
        config["monster_invasion_max_attempts"] = 2
        game = FakeGame(
            [
                "monster_match",
                "monster_result",
                "monster_reward",
                "monster_invasion",
            ]
            * 2,
            red_results=[False, False],
        )

        DailyBot(game, config)._run_monster_invasion()

        one_attempt = [
            config["points"]["monster_challenge"],
            config["points"]["monster_match"],
            config["points"]["monster_result_continue"],
            config["points"]["monster_reward_continue"],
        ]
        self.assertEqual(
            one_attempt * 2, [list(point) for point, _label in game.clicks]
        )

    def test_monster_invasion_stops_when_quick_match_is_disabled(self):
        game = FakeGame(
            ["monster_match", "monster_invasion"], red_results=[True]
        )

        DailyBot(game, self.config)._run_monster_invasion()

        expected = [
            self.config["points"]["monster_challenge"],
            self.config["points"]["monster_close"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])

    def test_monster_reward_retries_until_countdown_closes(self):
        game = FakeGame(
            [
                PageTimeout("countdown is still active"),
                "monster_reward",
                PageTimeout("countdown is still active"),
                "monster_reward",
                "monster_invasion",
            ]
        )

        DailyBot(game, self.config)._finish_monster_reward()

        self.assertEqual(
            [self.config["points"]["monster_reward_continue"]] * 3,
            [list(point) for point, _label in game.clicks],
        )

    def test_hunter_league_runs_configured_matches_and_claims_rewards(self):
        config = copy.deepcopy(self.config)
        config["hunter_league_matches"] = 2
        game = FakeGame(
            [
                "hunter_league_victory",
                "hunter_league",
                "hunter_league_failure",
                "hunter_league",
                "hunter_league",
            ],
            gold_results=[True, True],
        )

        DailyBot(game, config)._run_hunter_league()

        expected = [
            config["points"]["hunter_league_match"],
            config["points"]["hunter_league_result"],
            config["points"]["hunter_league_match"],
            config["points"]["hunter_league_result"],
            config["points"]["hunter_league_rewards"],
            config["points"]["hunter_league_daily_claim"],
            config["points"]["overlay_continue"],
            config["points"]["hunter_league_challenge_tab"],
            config["points"]["hunter_league_challenge_claim"],
            config["points"]["overlay_continue"],
            config["points"]["hunter_league_rewards_close"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])

    @staticmethod
    def _rank_tasks_observation(progress):
        return Observation(
            timestamp=1.0,
            viewport=(0, 0, 1091, 700),
            state="rank_tasks",
            state_confidence=0.99,
            numeric_values={"hunter_league_matches": progress},
            controls=[
                DetectedControl("league_go", (962, 502, 1047, 537), 0.99, "color+ocr"),
                DetectedControl("league_go", (962, 581, 1047, 616), 0.99, "color+ocr"),
            ],
            ocr_tokens=[
                OCRToken("参与10次猎人联赛。", 0.99, (469, 575, 628, 593)),
                OCRToken(f"进度：{progress}/10", 0.99, (468, 603, 541, 619)),
            ],
        )

    def test_v2_hunter_league_runs_only_remaining_matches_and_rechecks_ten(self):
        class ProgressGame:
            def __init__(self):
                self.actions = []
                self.diagnostics = []

            def click_detected_control(self, name, _label, **kwargs):
                self.actions.append((name, kwargs.get("preferred_point")))
                return Observation(
                    timestamp=1.0,
                    viewport=(0, 0, 1091, 700),
                    state="hunter_league",
                    state_confidence=0.99,
                )

            def save_diagnostic(self, name):
                self.diagnostics.append(name)

        game = ProgressGame()
        bot = DailyBot(game, self.config)
        observations = [
            self._rank_tasks_observation(3),
            self._rank_tasks_observation(10),
        ]
        events = []
        bot._open_rank_tasks = lambda _initial=None, **_kwargs: observations.pop(0)
        bot._run_hunter_league = lambda count=None: events.append(("run", count))
        bot._return_home_v2 = lambda: events.append(("home", None))

        completed = bot._complete_hunter_league_matches()

        self.assertTrue(completed)
        self.assertEqual([("run", 7), ("home", None)], events)
        self.assertEqual([("league_go", (548, 584))], game.actions)
        self.assertEqual(["hunter-league-complete"], game.diagnostics)

    def test_v2_hunter_league_rejects_incomplete_post_run_progress(self):
        class ProgressGame:
            def __init__(self):
                self.diagnostics = []

            def click_detected_control(self, *_args, **_kwargs):
                return Observation(
                    timestamp=1.0,
                    viewport=(0, 0, 1091, 700),
                    state="hunter_league",
                    state_confidence=0.99,
                )

            def save_diagnostic(self, name):
                self.diagnostics.append(name)

        game = ProgressGame()
        bot = DailyBot(game, self.config)
        observations = [
            self._rank_tasks_observation(3),
            self._rank_tasks_observation(9),
        ]
        bot._open_rank_tasks = lambda _initial=None, **_kwargs: observations.pop(0)
        bot._run_hunter_league = lambda _count=None: None
        bot._return_home_v2 = lambda: None

        with self.assertRaisesRegex(SafetyStop, "仅为 9/10"):
            bot._complete_hunter_league_matches()

        self.assertEqual(["hunter-league-progress-incomplete"], game.diagnostics)

    def test_open_rank_tasks_uses_detected_main_and_task_tab_controls(self):
        main = Observation(
            timestamp=1.0,
            viewport=(0, 0, 1091, 700),
            state="main",
            state_confidence=0.99,
        )
        overview = Observation(
            timestamp=1.0,
            viewport=(0, 0, 1091, 700),
            state="rank_overview",
            state_confidence=0.99,
        )
        tasks = self._rank_tasks_observation(3)
        tasks_transition = self._rank_tasks_observation(3)
        tasks_transition.numeric_values = {}

        class RankGame:
            def __init__(self):
                self.actions = []

            def recover_to_state(self, expected, **_kwargs):
                self.actions.append(("recover", set(expected)))
                return main

            def click_detected_control(self, name, _label, **kwargs):
                self.actions.append((name, set(kwargs["allowed_states"])))
                return overview if name == "rank_entry" else tasks_transition

            def wait_for_state(self, expected, **_kwargs):
                self.actions.append(("wait", set(expected)))
                return tasks

            def drag_reference(self, *_args, **_kwargs):
                raise AssertionError("visible target row must be resampled without scrolling")

        game = RankGame()
        observation = DailyBot(game, self.config)._open_rank_tasks()

        self.assertEqual(3, observation.numeric_values["hunter_league_matches"])
        self.assertEqual(
            [
                ("recover", {"main"}),
                ("rank_entry", {"main"}),
                ("rank_tasks_tab", {"rank_overview"}),
                ("wait", {"rank_tasks"}),
            ],
            game.actions,
        )

    def test_open_rank_tasks_scrolls_until_ten_match_row_is_visible(self):
        top = Observation(
            timestamp=1.0,
            viewport=(0, 0, 1091, 700),
            state="rank_tasks",
            state_confidence=0.99,
            ocr_tokens=[
                OCRToken("参与5次猎人联赛。", 0.99, (469, 609, 618, 629))
            ],
        )
        target = self._rank_tasks_observation(3)

        class ScrolledRankGame:
            def __init__(self):
                self.observations = [top, target]
                self.drags = []

            def wait_for_state(self, _expected, **_kwargs):
                return self.observations.pop(0)

            def drag_reference(self, start, end, duration, label):
                self.drags.append((start, end, duration, label))

            def recover_to_state(self, *_args, **_kwargs):
                raise AssertionError("initial rank_tasks must not recover home")

            def click_detected_control(self, *_args, **_kwargs):
                raise AssertionError("scroll search must not click a guessed control")

        game = ScrolledRankGame()
        observation = DailyBot(game, self.config)._open_rank_tasks("rank_tasks")

        self.assertEqual(3, observation.numeric_values["hunter_league_matches"])
        scroll = self.config["rank_task_scroll"]
        self.assertEqual(
            [(scroll["from"], scroll["to"], scroll["duration_seconds"])],
            [(start, end, duration) for start, end, duration, _label in game.drags],
        )

    def test_completed_secretary_row_does_not_skip_authoritative_league_check(self):
        config = copy.deepcopy(self.config)
        config["captured_task_adapters"] = ["hunter_league"]
        game = FakeGame()
        bot = DailyBot(game, config)
        events = []
        bot._use_v2_hunter_league_flow = lambda: True
        bot._open_daily_tasks = lambda: events.append("open")
        bot._scan_completed_tasks = lambda _tasks: {"hunter_league"}
        bot._complete_hunter_league_matches = lambda: events.append("league") or False
        bot._return_home_v2 = lambda: events.append("home")

        bot._run_captured_tasks()

        self.assertEqual(["open", "home", "open", "league", "home"], events)

    def test_hunter_league_result_retries_while_animation_is_visible(self):
        game = FakeGame(
            [
                PageTimeout("result animation is still active"),
                "hunter_league_failure",
                PageTimeout("result animation is still active"),
                "hunter_league_victory",
                "hunter_league",
            ]
        )

        DailyBot(game, self.config)._finish_hunter_league_result()

        self.assertEqual(
            [self.config["points"]["hunter_league_result"]] * 3,
            [list(point) for point, _label in game.clicks],
        )

    def test_hunter_league_result_dismisses_rank_promotion_overlay(self):
        game = FakeGame(["infinite_rank_drop", "hunter_league"])

        DailyBot(game, self.config)._finish_hunter_league_result()

        self.assertEqual(
            [
                self.config["points"]["hunter_league_result"],
                self.config["points"]["infinite_rank_continue"],
            ],
            [list(point) for point, _label in game.clicks],
        )

    def test_infinite_mystery_dismisses_rank_drop_and_runs_three_stages(self):
        game = FakeGame(
            [
                "infinite_rank_drop",
                "infinite_rank_drop",
                "infinite_mystery",
                "infinite_map",
                "infinite_stage",
                "infinite_score",
                "infinite_next",
                "infinite_score",
                "infinite_next",
                "infinite_score",
                "infinite_finished",
                "infinite_mystery",
            ]
        )
        bot = DailyBot(game, self.config)
        bot._find_task_button = lambda _task: ((512, 404), False)

        self.assertTrue(bot._open_infinite_task())
        bot._run_infinite_mystery()

        expected = [
            [512, 404],
            self.config["points"]["infinite_rank_continue"],
            self.config["points"]["infinite_rank_continue"],
            self.config["points"]["infinite_start"],
            self.config["points"]["infinite_first_stage"],
            self.config["points"]["infinite_stage_start"],
            self.config["points"]["infinite_score_continue"],
            self.config["points"]["infinite_next"],
            self.config["points"]["infinite_score_continue"],
            self.config["points"]["infinite_next"],
            self.config["points"]["infinite_score_continue"],
            self.config["points"]["infinite_return_safe"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])

    def test_completed_progress_skips_blue_task_button(self):
        game = FakeGame(gold_results=[False], progress_results=[True])
        game.find_task = lambda _task, _image: (152, 230, 0.99)

        result = DailyBot(game, self.config)._find_task_button("tower")

        self.assertEqual(((self.config["task_button_x"]["left"], 230), True), result)
        self.assertEqual([], game.clicks)

    def test_hunter_three_progress_threshold_separates_real_scores(self):
        class ScoreCV:
            COLOR_BGR2GRAY = 1
            TM_CCOEFF_NORMED = 2

            def __init__(self, score):
                self.score = score

            @staticmethod
            def cvtColor(image, _mode):
                return image[:, :, 0]

            @staticmethod
            def matchTemplate(_region, _template, _mode):
                return np.zeros((1, 1), dtype=np.float32)

            def minMaxLoc(self, _result):
                return 0.0, self.score, (0, 0), (0, 0)

        image = np.zeros((700, 1091, 3), dtype=np.uint8)
        game = object.__new__(DesktopGame)
        game.config = self.config
        game.task_progress_templates = {
            "three": np.zeros((19, 32), dtype=np.uint8)
        }

        game.cv2 = ScoreCV(0.904)
        self.assertTrue(game.task_progress_complete("hunter_field", (652, 318), image))

        game.cv2 = ScoreCV(0.869)
        self.assertFalse(game.task_progress_complete("hunter_field", (652, 318), image))

    def test_task_search_resets_to_top_before_matching(self):
        game = FakeGame()
        game.find_task = lambda _task, _image: (152, 230, 0.99)

        DailyBot(game, self.config)._find_task_button("tower")

        scroll = self.config["task_scroll"]
        self.assertEqual(
            [(tuple(scroll["to"]), tuple(scroll["from"]))] * scroll["reset_passes"],
            [(start, end) for start, end, _duration, _label in game.drags],
        )

    def test_task_search_scans_multiple_screens_from_the_top(self):
        game = FakeGame()
        results = [None, None, (652, 316, 0.99)]
        game.find_task = lambda _task, _image: results.pop(0)

        result = DailyBot(game, self.config)._find_task_button("hunter_field")

        self.assertEqual(((self.config["task_button_x"]["right"], 316), True), result)
        scroll = self.config["task_scroll"]
        directions = [(start, end) for start, end, _duration, _label in game.drags]
        self.assertEqual(
            [(tuple(scroll["to"]), tuple(scroll["from"]))] * scroll["reset_passes"]
            + [(tuple(scroll["from"]), tuple(scroll["to"]))] * 2,
            directions,
        )

    def test_completed_tasks_are_prescanned_and_not_opened_individually(self):
        config = copy.deepcopy(self.config)
        config["captured_task_adapters"] = ["tower", "hunter_field"]
        game = FakeGame()
        bot = DailyBot(game, config)
        events = []
        bot._open_daily_tasks = lambda: events.append("open")
        bot._scan_completed_tasks = lambda _tasks: {"tower", "hunter_field"}
        bot._open_tower_task = lambda: None
        bot._open_hunter_task = lambda: None
        bot._return_home = lambda: events.append("home")

        bot._run_captured_tasks()

        self.assertEqual(["open", "home"], events)

    def test_task_prescan_collects_completed_rows_across_multiple_screens(self):
        config = copy.deepcopy(self.config)
        config["task_scroll"]["reset_passes"] = 0
        config["task_scroll"]["search_passes"] = 2
        game = FakeGame(gold_results=[True, False, True])
        bot = DailyBot(game, config)
        screen = {"index": 0}

        def find_task(task, _image):
            visible = [
                {"tower": (152, 230, 0.99), "hunter_field": (652, 316, 0.99)},
                {"abyss": (652, 404, 0.99)},
            ][screen["index"]]
            return visible.get(task)

        game.find_task = find_task
        original_drag = game.drag_reference

        def drag_and_advance(start, end, duration, label):
            original_drag(start, end, duration, label)
            screen["index"] += 1

        game.drag_reference = drag_and_advance

        completed = bot._scan_completed_tasks(["tower", "hunter_field", "abyss"])

        self.assertEqual({"tower", "abyss"}, completed)
        self.assertEqual(1, len(game.drags))

    def test_incomplete_progress_keeps_task_available(self):
        game = FakeGame(gold_results=[False], progress_results=[False])
        game.find_task = lambda _task, _image: (652, 316, 0.99)

        result = DailyBot(game, self.config)._find_task_button("hunter_field")

        self.assertEqual(((self.config["task_button_x"]["right"], 316), False), result)

    def test_switch_character_selects_next_row_and_starts_game(self):
        game = FakeGame(character_indices=[0])

        DailyBot(game, self.config)._switch_to_next_character()

        expected = [
            self.config["points"]["character_switch"],
            self.config["character_row_points"][1],
            self.config["points"]["character_start"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])
        self.assertEqual(["character_switch", "character_switch", "main"], game.waits)

    def test_switch_character_wraps_from_third_row_to_first(self):
        game = FakeGame(character_indices=[2])

        DailyBot(game, self.config)._switch_to_next_character(already_open=True)

        expected = [
            self.config["character_row_points"][0],
            self.config["points"]["character_start"],
        ]
        self.assertEqual(expected, [list(point) for point, _label in game.clicks])
        self.assertEqual(["character_switch", "main"], game.waits)

    def test_character_cycle_runs_three_roles_and_switches_twice(self):
        game = FakeGame()
        bot = DailyBot(game, self.config)
        events = []
        bot._run_captured_tasks = lambda: events.append("tasks")
        bot._open_daily_tasks = lambda: events.append("open")
        bot._claim_task_rewards = lambda: events.append("claim")
        bot._return_home = lambda: events.append("home")
        bot._switch_to_next_character = lambda: events.append("switch")

        bot._run_character_cycles()

        self.assertEqual(
            [
                "tasks", "open", "claim", "home", "switch",
                "tasks", "open", "claim", "home", "switch",
                "tasks", "open", "claim", "home",
            ],
            events,
        )
        self.assertEqual(
            ["tasks-finished-role-1", "tasks-finished-role-2", "tasks-finished-role-3"],
            game.diagnostics,
        )

    def test_ladder_is_explicitly_excluded(self):
        self.assertNotIn("ladder", self.config["captured_task_adapters"])
        self.assertIn("天梯赛", self.config["skipped_task_names"])

    def test_ladder_adapter_is_rejected_at_runtime(self):
        config = copy.deepcopy(self.config)
        config["captured_task_adapters"].append("ladder")
        game = FakeGame(detected_page="main")

        with self.assertRaisesRegex(SafetyStop, "明确排除"):
            DailyBot(game, config).run()
        self.assertEqual([], game.clicks)

    def test_duplicate_adapter_is_rejected_to_avoid_repeating_tasks(self):
        config = copy.deepcopy(self.config)
        config["captured_task_adapters"].append("tower")

        with self.assertRaisesRegex(SafetyStop, "不能重复配置"):
            DailyBot(FakeGame(), config)._run_captured_tasks()


if __name__ == "__main__":
    unittest.main()
