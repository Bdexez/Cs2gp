import datetime as dt
import unittest

from cs2gp.markets import SeriesDistribution, VetoPlan
from cs2gp.predict import Prediction
from cs2gp.pronostic import Fixture, Tipster, select_analysis_picks, simulate_tipster


class TipsterOptionsTests(unittest.TestCase):
    def test_invalid_options_fail_before_any_prediction(self):
        for options in (dict(abstention=float("nan")), dict(abstention=.4), dict(abstention=1.1),
                        dict(momentum=float("inf")), dict(edge_min=-.01),
                        dict(decider_temperature=0)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Tipster(None, **options)
        self.assertEqual(Tipster(None, abstention=.5, edge_min=0, momentum=-1).abstention, .5)

    def test_fixture_rejects_non_numeric_or_boolean_odds(self):
        for bad in (True, "2.0", float("nan"), 1.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Fixture(dt.date(2026, 1, 1), "navi", "g2", odds_a=bad)

    def test_pure_selection_cannot_emit_a_nan_probability_or_bypass_abstention(self):
        args = (dt.date(2026, 1, 1), "a", "b", 3, "")
        with self.assertRaises(ValueError):
            select_analysis_picks(*args, float("nan"), None)
        with self.assertRaises(ValueError):
            select_analysis_picks(*args, .8, None, abstention=float("nan"))

    def test_backtest_rejects_the_same_options_before_training(self):
        for options in (dict(abstention=float("nan")), dict(momentum=float("inf")),
                        dict(decider_temperature=-1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                simulate_tipster([], None, **options)

    def test_derived_markets_flag_absurd_edges_as_suspect_with_zero_stake(self):
        tipster = Tipster(None)
        day = dt.date(2026, 1, 1)
        fx = Fixture(
            day, "navi", "g2", best_of=3,
            odds_over=1.10, odds_under=9.00,  # implies ~89% over, ~11% under
            odds_a_sweep=5.00,                # implies 20% sweep
        )
        pred = Prediction(
            team_a="navi", team_b="g2", date=day, best_of=3, lan=False,
            p_a=0.75, p_a_match_model=0.75, p_a_from_maps=0.75,
            confidence="correcte", drivers=[], map_edges=[], data_quality={},
        )
        # Model predicts 60% 2-0 sweep (under 2.5 = 65%, vs market ~11% -> edge +54%)
        dist = SeriesDistribution(
            scores={(2, 0): 0.60, (2, 1): 0.15, (1, 2): 0.20, (0, 2): 0.05},
            order=[("mirage", 0.7), ("inferno", 0.6), ("nuke", 0.5)],
            best_of=3,
        )
        picks = tipster._value_picks(fx, pred, dist, [])
        by_market = {p.market: p for p in picks}
        self.assertEqual(by_market["total maps -2.5"].verdict, "SUSPECT")
        self.assertEqual(by_market["total maps -2.5"].stake, 0.0)
        self.assertEqual(by_market["handicap -1.5"].verdict, "SUSPECT")
        self.assertEqual(by_market["handicap -1.5"].stake, 0.0)

    def test_rationale_uses_human_labels_for_all_features(self):
        from cs2gp.features import FeatureExtractor

        class _DummyAn:
            fx = FeatureExtractor()

        tipster = Tipster(_DummyAn())
        day = dt.date(2026, 1, 1)
        dist = SeriesDistribution(
            scores={(2, 0): 0.5, (2, 1): 0.2, (1, 2): 0.2, (0, 2): 0.1},
            order=[("mirage", 0.6), ("inferno", 0.5), ("nuke", 0.5)],
            best_of=3,
        )
        veto = VetoPlan(order=dist.order, picked_by_a=["mirage"], picked_by_b=["inferno"], decider="nuke")
        for feature in ("trajectory_diff", "perf_residual_diff", "elo_diff_x_certainty",
                        "experience_diff", "roster_tenure_diff"):
            pred = Prediction(
                team_a="navi", team_b="g2", date=day, best_of=3, lan=False,
                p_a=0.7, p_a_match_model=0.7, p_a_from_maps=0.7,
                confidence="correcte", drivers=[("elo_diff", 0.5), (feature, 0.25)],
                map_edges=[], data_quality={},
            )
            lines = tipster._rationale(pred, veto, dist)
            joined = " ".join(lines)
            self.assertNotIn(feature, joined)


if __name__ == "__main__":
    unittest.main()
