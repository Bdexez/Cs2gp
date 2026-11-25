import copy
import datetime as dt
import unittest

from cs2gp.ratings import EloConfig, EloEngine, TeamRating
from cs2gp.schema import MapResult


class RatingChronologyTests(unittest.TestCase):
    def setUp(self):
        self.day = dt.date(2026, 1, 2)
        self.engine = EloEngine()
        self.engine.observe(MapResult("recent", self.day, "a", "b", "mirage", 13, 5,
                                      best_of=1, roster_a=("p1", "p2", "p3")))

    def test_backdated_results_are_rejected_before_any_state_changes(self):
        original = copy.deepcopy(vars(self.engine))
        for a, b in (("a", "b"), ("c", "d")):
            old = MapResult("old", self.day - dt.timedelta(days=1), a, b, "nuke", 5, 13,
                            best_of=1, roster_a=("p4", "p5", "p6"))
            with self.subTest(teams=(a, b)), self.assertRaisesRegex(ValueError, "chronologique"):
                self.engine.observe(old)
            self.assertEqual(vars(self.engine), original)

    def test_invalid_calendar_dates_do_not_register_teams_or_rosters(self):
        original = copy.deepcopy(vars(self.engine))
        for date in (None, "2026-01-03", dt.datetime(2026, 1, 3), False):
            invalid = MapResult("invalid", date, "c", "d", "nuke", 13, 5,
                                best_of=1, roster_a=("p1",))
            with self.subTest(date=date), self.assertRaisesRegex(ValueError, "date calendaire"):
                self.engine.observe(invalid)
            self.assertEqual(vars(self.engine), original)

    def test_same_day_and_later_results_remain_supported(self):
        self.engine.observe(MapResult("same", self.day, "c", "d", "nuke", 13, 5, best_of=1))
        tomorrow = self.day + dt.timedelta(days=1)
        self.engine.observe(MapResult("next", tomorrow, "a", "b", "nuke", 5, 13, best_of=1))
        self.assertEqual(self.engine.maps_played("a"), 2)
        self.assertEqual(self.engine.teams["a"].last_date, tomorrow)

    def test_copy_retains_chronology_and_simulations_leave_original_unchanged(self):
        original = copy.deepcopy(vars(self.engine))
        clone = self.engine.copy()
        old = self.day - dt.timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "chronologique"):
            clone.observe(MapResult("old", old, "c", "d", "nuke", 13, 5, best_of=1))
        with self.assertRaisesRegex(ValueError, "chronologique"):
            self.engine.simulate_outcome("c", "d", "nuke", old)
        result = self.engine.simulate_outcome("a", "b", "nuke", self.day)
        self.assertTrue(result)
        self.assertEqual(vars(self.engine), original)

    def test_feature_extractor_keeps_all_history_when_old_map_is_rejected(self):
        from cs2gp.features import FeatureExtractor

        fx = FeatureExtractor()
        fx.observe_map(MapResult("recent", self.day, "a", "b", "mirage", 13, 5, best_of=1))
        original = copy.deepcopy(vars(fx))
        with self.assertRaisesRegex(ValueError, "chronologique"):
            fx.observe_map(MapResult("old", self.day - dt.timedelta(days=1), "a", "b",
                                     "nuke", 5, 13, best_of=1))
        self.assertEqual(vars(fx.elo), vars(original.pop("elo")))
        self.assertEqual({k: v for k, v in vars(fx).items() if k != "elo"}, original)


class RatingOrientationTests(unittest.TestCase):
    def test_updates_do_not_depend_on_team_order_at_unequal_uncertainty(self):
        day = dt.date(2026, 1, 1)
        for winner_a in (False, True):
            for use_mov in (False, True):
                with self.subTest(winner_a=winner_a, use_mov=use_mov):
                    first = EloEngine(EloConfig(use_mov=use_mov))
                    first.teams = {
                        "a": TeamRating(rating=1700, rd=40, last_date=day),
                        "b": TeamRating(rating=1500, rd=300, last_date=day),
                    }
                    reverse = copy.deepcopy(first)
                    a, b = (13, 5) if winner_a else (5, 13)
                    first.observe(MapResult("m", day, "a", "b", "mirage", a, b, best_of=1))
                    reverse.observe(MapResult("m", day, "b", "a", "mirage", b, a, best_of=1))
                    for team in ("a", "b"):
                        self.assertAlmostEqual(first.teams[team].rating, reverse.teams[team].rating, places=10)
                        self.assertAlmostEqual(first.teams[team].rd, reverse.teams[team].rd, places=10)
                        self.assertAlmostEqual(first.teams[team].fast_rating,
                                               reverse.teams[team].fast_rating, places=10)
                        self.assertEqual(first.teams[team].map_ratings, reverse.teams[team].map_ratings)


class MapInactivityTests(unittest.TestCase):
    def setUp(self):
        self.day = dt.date(2026, 1, 1)
        self.engine = EloEngine(EloConfig(map_prior_weight=0, use_mov=False))
        self.engine.observe(MapResult("nuke", self.day, "a", "b", "nuke", 13, 5, best_of=1))

    def test_map_rating_decays_by_its_own_last_played_date(self):
        before = self.engine.map_rating("a", "nuke", self.day)
        future = self.day + dt.timedelta(days=400)
        self.engine.observe(MapResult("mirage", future, "a", "b", "mirage", 13, 5, best_of=1))
        self.assertEqual(self.engine.teams["a"].map_last_dates["nuke"], self.day)
        self.assertAlmostEqual(self.engine.map_rating("a", "nuke", future), 1500+(before-1500)/2)

    def test_reading_future_rating_does_not_mutate_state(self):
        original = copy.deepcopy(self.engine.teams)
        future = self.day + dt.timedelta(days=10000)
        self.assertAlmostEqual(self.engine.map_rating("a", "nuke", future), 1500, places=5)
        self.assertEqual(self.engine.teams, original)

    def test_next_update_uses_the_decayed_map_rating(self):
        future = self.day + dt.timedelta(days=400)
        expected = self.engine.map_rating("a", "nuke", future)
        self.engine.cfg.k_map = 0
        self.engine.observe(MapResult("return", future, "a", "b", "nuke", 13, 5, best_of=1))
        self.assertAlmostEqual(self.engine.teams["a"].map_ratings["nuke"], expected)

    def test_reading_unknown_team_does_not_register_it(self):
        from cs2gp.features import FeatureExtractor

        original = copy.deepcopy(self.engine.teams)
        self.assertEqual(self.engine.rating("ghost", self.day), 1500.0)
        self.assertEqual(self.engine.deviation("ghost", self.day), self.engine.cfg.rd_initial)
        self.assertEqual(self.engine.trajectory("ghost", self.day), 0.0)
        self.assertEqual(self.engine.map_rating("ghost", "mirage", self.day), 1500.0)
        self.assertEqual(self.engine.maps_played("ghost"), 0)
        self.assertEqual(self.engine.teams, original)

        fx = FeatureExtractor()
        fx.match_features("ghost_a", "ghost_b", self.day)
        fx.map_features("ghost_a", "ghost_b", self.day, "mirage")
        fx.match_context("ghost_a", "ghost_b", self.day)
        self.assertEqual(fx.elo.teams, {})

    def test_leaderboard_active_within_days_filters_inactive_teams(self):
        later = self.day + dt.timedelta(days=100)
        self.engine.observe(MapResult("m2", later, "c", "d", "mirage", 13, 8, best_of=1))
        active = self.engine.leaderboard(on=later, min_maps=1, active_within_days=30)
        self.assertEqual({t for t, _, _ in active}, {"c", "d"})
        with self.assertRaises(ValueError):
            self.engine.leaderboard(on=later, min_maps=1, active_within_days=-1)

    def test_rating_summary_computes_distribution_and_handles_empty(self):
        summary = self.engine.rating_summary(on=self.day, min_maps=1)
        self.assertEqual(summary["n_teams"], 2)
        self.assertEqual(summary["top_team"], "a")
        self.assertEqual(summary["bottom_team"], "b")
        self.assertGreater(summary["max"], summary["min"])
        self.assertGreater(summary["mean_rd"], 0.0)
        empty = self.engine.rating_summary(on=self.day, min_maps=50)
        self.assertEqual(empty["n_teams"], 0)
        self.assertIsNone(empty["mean"])

    def test_copy_produces_isolated_engine_for_what_if_simulations(self):
        clone = self.engine.copy()
        before_a = self.engine.rating("a", self.day)
        clone.observe(MapResult("hyp", self.day + dt.timedelta(days=1), "a", "b", "nuke", 5, 13, best_of=1))
        self.assertEqual(self.engine.rating("a", self.day), before_a)
        self.assertLess(clone.rating("a", self.day + dt.timedelta(days=1)), before_a)
        self.assertEqual(self.engine.maps_played("a"), 1)
        self.assertEqual(clone.maps_played("a"), 2)


if __name__ == "__main__":
    unittest.main()
