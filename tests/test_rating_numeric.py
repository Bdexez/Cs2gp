import copy
import datetime as dt
import math
import unittest

import numpy as np

from cs2gp.ratings import EloConfig, EloEngine, TeamRating, expected_score
from cs2gp.schema import MapResult


class RatingNumericTests(unittest.TestCase):
    def test_ordinary_probabilities_match_the_elo_formula(self):
        for a in (800, 1500, 2400):
            for b in (800, 1500, 2400):
                with self.subTest(a=a, b=b):
                    self.assertAlmostEqual(expected_score(a, b),
                                           1 / (1 + 10 ** ((b - a) / 400)), places=15)

    def test_extreme_finite_ratings_saturate_without_overflow_or_numpy_warnings(self):
        with np.errstate(all="raise"):
            for a, b in ((-130000.0, 1500.0), (-1e308, 1e308), (1e308, 1e308)):
                for cast in (float, np.float64):
                    with self.subTest(a=a, b=b, cast=cast):
                        p = expected_score(cast(a), cast(b))
                        reverse = expected_score(cast(b), cast(a))
                        self.assertTrue(math.isfinite(p))
                        self.assertGreaterEqual(p, 0)
                        self.assertLessEqual(p, 1)
                        self.assertEqual(p + reverse, 1)
        self.assertEqual(expected_score(-130000, 1500), 0)
        self.assertEqual(expected_score(1e308, 1e308), .5)

    def test_small_representable_underdog_probability_is_preserved(self):
        p = expected_score(-120000, 0)
        self.assertGreater(p, 0)
        self.assertAlmostEqual(p / 1e-300, 1, places=12)

    def test_large_common_offset_does_not_change_a_small_representable_gap(self):
        for gap in (16, 32, 64):
            with self.subTest(gap=gap):
                self.assertEqual(expected_score(1e17 + gap, 1e17), expected_score(gap, 0))
                self.assertEqual(expected_score(1e17, 1e17 + gap), expected_score(0, gap))

    def test_extreme_gap_updates_are_finite_and_independent_of_team_order(self):
        day = dt.date(2026, 1, 1)
        for uncertainty in (False, True):
            for mov in (False, True):
                for won in (False, True):
                    with self.subTest(uncertainty=uncertainty, mov=mov, won=won):
                        engine = EloEngine(EloConfig(use_uncertainty=uncertainty, use_mov=mov))
                        engine.teams = {
                            "a": TeamRating(rating=-130000, rd=40, last_date=day,
                                            map_ratings={"mirage": -130000}),
                            "b": TeamRating(rating=130000, rd=300, last_date=day,
                                            map_ratings={"mirage": 130000}),
                        }
                        reverse = engine.copy()
                        sa, sb = (13, 5) if won else (5, 13)
                        engine.observe(MapResult("m", day, "a", "b", "mirage", sa, sb, best_of=1))
                        reverse.observe(MapResult("m", day, "b", "a", "mirage", sb, sa, best_of=1))
                        self.assertEqual(engine.teams, reverse.teams)
                        for tr in engine.teams.values():
                            for value in (tr.rating, tr.fast_rating, tr.rd, *tr.map_ratings.values()):
                                self.assertTrue(math.isfinite(value))

    def test_large_valid_k_values_support_successive_updates_and_simulations(self):
        day = dt.date(2026, 1, 1)
        for uncertainty in (False, True):
            with self.subTest(uncertainty=uncertainty):
                engine = EloEngine(EloConfig(k_base=200000, k_map=200000,
                                             use_mov=False, use_uncertainty=uncertainty))
                for i in range(4):
                    engine.observe(MapResult(str(i), day, "a", "b", "mirage", 13, 5, best_of=1))
                original = copy.deepcopy(vars(engine))
                engine.simulate_outcome("a", "b", "mirage", day)
                self.assertEqual(vars(engine), original)
                self.assertEqual(engine.maps_played("a"), 4)
                for team in ("a", "b"):
                    self.assertTrue(math.isfinite(engine.rating(team)))
                    self.assertTrue(math.isfinite(engine.map_rating(team, "mirage")))


if __name__ == "__main__":
    unittest.main()
