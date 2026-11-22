import math
import unittest

from cs2gp.odds import (
    closing_line_value, devig_power, equity_curve, expected_value, find_value,
    kelly_fraction, market_diagnostics, roi_significance, simulate_flat, simulate_kelly,
)


class PowerDevigTests(unittest.TestCase):
    def test_both_outputs_follow_the_same_power_even_for_extreme_prices(self):
        for odds in ((1.1, 1.2), (100, 200), (1.01, 1.02), (1.75, 2.1),
                     (math.nextafter(1., 2.), 2.)):
            with self.subTest(odds=odds):
                pa, pb = devig_power(*odds)
                exponent = math.log(pa)/math.log(1/odds[0])
                self.assertAlmostEqual((1/odds[1])**exponent, pb, places=11)
                self.assertAlmostEqual(pa+pb, 1., places=14)
                reversed_probs = devig_power(*reversed(odds))
                self.assertAlmostEqual(pa, reversed_probs[1], places=14)

    def test_fair_prices_are_unchanged_and_equal_prices_stay_equal(self):
        for pa in (.01, .25, .5, .9):
            result = devig_power(1/pa, 1/(1-pa))
            self.assertAlmostEqual(result[0], pa, places=12)
        for odds in (1.01, 1.5, 2, 10, 1000):
            self.assertEqual(devig_power(odds, odds), (.5, .5))


class OddsValidationTests(unittest.TestCase):
    def test_invalid_probability_cannot_produce_a_maximum_stake(self):
        for p in (float("nan"), float("inf"), -.1, 1.1):
            for fn in (lambda: kelly_fraction(p, 2), lambda: expected_value(p, 2),
                       lambda: find_value(p, 2, 2)):
                with self.subTest(p=p), self.assertRaises(ValueError):
                    fn()

    def test_invalid_prices_and_staking_parameters_are_rejected(self):
        for odds in (1, 0, float("nan"), float("inf")):
            for fn in (kelly_fraction, expected_value):
                with self.subTest(odds=odds, fn=fn), self.assertRaises(ValueError):
                    fn(.7, odds)
        for options in (dict(cap=-1), dict(scale=-1), dict(cap=float("nan")),
                        dict(scale=float("inf")), dict(cap=1.1)):
            with self.assertRaises(ValueError):
                kelly_fraction(.7, 2, **options)
        self.assertEqual(kelly_fraction(.7, 2, scale=0), 0)
        self.assertEqual(kelly_fraction(.7, 2, cap=0), 0)

    def test_invalid_backtest_cannot_report_a_plausible_roi(self):
        for bets in ([(.7, 2., 2)], [(float("nan"), 2., 1)], [(.7, float("inf"), 0)]):
            with self.assertRaises(ValueError):
                simulate_flat(bets)
        for stake in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                simulate_flat([(.7, 2., 1)], stake=stake)
        result = simulate_flat([(.7, 2., 1), (.7, 2., 0)], stake=5)
        self.assertEqual(result["profit"], 0)
        self.assertEqual(result["turnover"], 10)
        self.assertEqual(result["max_drawdown"], 5.0)
        self.assertEqual(result["max_losing_streak"], 1)

    def test_closing_line_value_and_drawdown_metrics(self):
        # Took 2.20 on A; closing fair line is 1.90 / 1.90 (50% each) -> positive CLV
        clv = closing_line_value(2.20, 1.90, 1.90, side="a")
        self.assertAlmostEqual(clv["closing_p"], 0.5, places=10)
        self.assertGreater(clv["clv_prob"], 0.0)
        self.assertAlmostEqual(clv["clv_ev"], 0.10, places=10)
        with self.assertRaises(ValueError):
            closing_line_value(2.0, 1.9, 1.9, side="c")

        seq = [(0.6, 2.0, 1), (0.6, 2.0, 0), (0.6, 2.0, 0), (0.6, 2.0, 0), (0.6, 2.0, 1)]
        sim = simulate_flat(seq, stake=2.0)
        self.assertEqual(sim["max_drawdown"], 6.0)
        self.assertEqual(sim["max_losing_streak"], 3)

    def test_simulate_kelly_compounds_bankroll_and_tracks_relative_drawdown(self):
        empty = simulate_kelly([], initial_bankroll=100.0)
        self.assertEqual(empty["final_bankroll"], 100.0)
        self.assertEqual(empty["max_drawdown_pct"], 0.0)
        # kelly_fraction(0.6, 2.0, cap=0.05, scale=0.25) == 0.05
        res = simulate_kelly([(0.6, 2.0, 1), (0.6, 2.0, 0)], initial_bankroll=100.0)
        # 100 -> 105 -> 99.75 (drawdown = 5.25 / 105 = 5%)
        self.assertAlmostEqual(res["final_bankroll"], 99.75, places=10)
        self.assertAlmostEqual(res["max_drawdown_pct"], 0.05, places=10)
        self.assertEqual(res["max_losing_streak"], 1)
        with self.assertRaises(ValueError):
            simulate_kelly([(0.6, 2.0, 1)], initial_bankroll=0.0)

    def test_market_diagnostics_and_roi_significance(self):
        diag = market_diagnostics(1.50, 2.80)
        self.assertGreater(diag["vig"], 0.0)
        self.assertFalse(diag["is_arbitrage"])
        self.assertAlmostEqual(diag["power_a"] + diag["power_b"], 1.0, places=12)
        arb = market_diagnostics(2.10, 2.10)
        self.assertTrue(arb["is_arbitrage"])

        small = roi_significance([(0.6, 2.0, 1), (0.6, 2.0, 0)])
        self.assertFalse(small["significant_5pct"])
        strong = roi_significance([(0.7, 2.0, 1)] * 35 + [(0.7, 2.0, 0)] * 5)
        self.assertTrue(strong["significant_5pct"])
        self.assertLess(strong["p_value"], 0.01)
        self.assertGreater(strong["sharpe_ratio"], 0.5)

    def test_equity_curve_tracks_flat_and_kelly_trajectories(self):
        seq = [(0.6, 2.0, 1), (0.6, 2.0, 0)]
        self.assertEqual(equity_curve(seq, mode="flat", stake=2.0), [0.0, 2.0, 0.0])
        kcurve = equity_curve(seq, mode="kelly", initial_bankroll=100.0)
        self.assertEqual(len(kcurve), 3)
        self.assertAlmostEqual(kcurve[1], 105.0, places=10)
        self.assertAlmostEqual(kcurve[2], 99.75, places=10)
        with self.assertRaises(ValueError):
            equity_curve(seq, mode="unknown")


if __name__ == "__main__":
    unittest.main()
