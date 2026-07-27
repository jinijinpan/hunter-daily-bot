import json
import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot import DailyBot, PageTimeout, SafetyStop


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
