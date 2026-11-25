import math
import unittest

import numpy as np

from cs2gp.odds import (
    closing_line_value, devig, devig_power, devig_shin, equity_curve, expected_value, find_value,
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

    def test_shin_devig_is_symmetric_and_handles_fair_and_vigged_markets(self):
        for odds in ((1.25, 4.0), (1.75, 2.10), (1.05, 10.0), (2.0, 2.0)):
            with self.subTest(odds=odds):
                pa, pb = devig_shin(*odds)
                self.assertAlmostEqual(pa + pb, 1.0, places=12)
                rev = devig_shin(*reversed(odds))
                self.assertAlmostEqual(pa, rev[1], places=12)
        self.assertEqual(devig_shin(1.8, 1.8), (0.5, 0.5))
        # On a vigged favorite/longshot market, Shin assigns higher prob to favorite than proportional
        diag = market_diagnostics(1.25, 4.20)
        self.assertGreater(diag["shin_z"], 0.0)
        self.assertGreater(diag["shin_a"], diag["proportional_a"])
        self.assertAlmostEqual(diag["shin_a"] + diag["shin_b"], 1.0, places=12)
        self.assertEqual(devig(1.75, 2.10, "shin"), devig_shin(1.75, 2.10))
        self.assertEqual(devig(1.75, 2.10, "power"), devig_power(1.75, 2.10))
        self.assertIsNotNone(find_value(0.65, 1.85, 2.00, devig_method="shin"))
        with self.assertRaises(ValueError):
            devig(1.80, 2.00, method="unknown")


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

    def test_roi_grouping_prevents_duplicate_markets_from_inventing_evidence(self):
        bets = [(0.7, 2.0, 1)] * 8 + [(0.7, 2.0, 0)] * 2
        baseline = roi_significance(bets)
        for repeats in (1, 3, 20):
            duplicated = [bet for bet in bets for _ in range(repeats)]
            groups = [i for i in range(len(bets)) for _ in range(repeats)]
            result = roi_significance(duplicated, groups=groups)
            self.assertEqual(result["n_groups"], 10)
            self.assertEqual(result["n"], 10 * repeats)
            self.assertFalse(result["significant_5pct"])
            for key in ("mean_return", "se_return", "t_stat", "p_value"):
                self.assertAlmostEqual(result[key], baseline[key], places=12)
        # L'ancienne hypothese d'independance surestime la preuve disponible.
        self.assertTrue(roi_significance(bets * 3)["significant_5pct"])

    def test_roi_clustered_error_matches_known_unequal_group_solution(self):
        bets = [(0.7, 2.0, won) for won in (1, 1, 0, 1, 0)]
        result = roi_significance(bets, groups=["a", "a", "b", "c", "c"])
        self.assertEqual(result["n_groups"], 3)
        self.assertAlmostEqual(result["mean_return"], .2)
        # Scores des groupes : 1,6 ; -1,2 ; -0,4. CR1 : (3/2)*4,16/25.
        self.assertAlmostEqual(result["se_return"], math.sqrt(.2496), places=12)
        self.assertAlmostEqual(result["t_stat"], .2 / math.sqrt(.2496), places=12)
        self.assertIsNone(roi_significance(bets, groups=["same"] * 5)["p_value"])
        self.assertEqual(roi_significance([], groups=[])["n_groups"], 0)
        for groups in (["a"], [[], [], [], [], []]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                roi_significance(bets, groups=groups)

    def test_roi_single_bet_groups_match_independent_calculation(self):
        bets = [(0.7, 2.0, 1)] * 35 + [(0.7, 2.0, 0)] * 5
        independent = roi_significance(bets)
        grouped = roi_significance(bets, groups=list(range(len(bets))))
        for key in ("mean_return", "se_return", "t_stat", "p_value"):
            self.assertAlmostEqual(independent[key], grouped[key], places=12)
        self.assertTrue(grouped["significant_5pct"])

    def test_equity_curve_tracks_flat_and_kelly_trajectories(self):
        seq = [(0.6, 2.0, 1), (0.6, 2.0, 0)]
        self.assertEqual(equity_curve(seq, mode="flat", stake=2.0), [0.0, 2.0, 0.0])
        kcurve = equity_curve(seq, mode="kelly", initial_bankroll=100.0)
        self.assertEqual(len(kcurve), 3)
        self.assertAlmostEqual(kcurve[1], 105.0, places=10)
        self.assertAlmostEqual(kcurve[2], 99.75, places=10)
        with self.assertRaises(ValueError):
            equity_curve(seq, mode="unknown")

    def test_bootstrap_bankroll_estimates_drawdown_quantiles(self):
        from cs2gp.odds import bootstrap_bankroll

        seq = [(0.65, 2.0, 1)] * 13 + [(0.65, 2.0, 0)] * 7
        kelly = bootstrap_bankroll(seq, mode="kelly", horizon=30, samples=200, initial_bankroll=100.0, seed=0)
        self.assertGreater(kelly["prob_profit"], 0.7)
        self.assertLessEqual(kelly["p05_final"], kelly["median_final"])
        self.assertLessEqual(kelly["median_final"], kelly["p95_final"])
        self.assertLessEqual(kelly["median_max_drawdown_pct"], kelly["p95_max_drawdown_pct"])

        flat = bootstrap_bankroll(seq, mode="flat", horizon=30, samples=200, stake=1.0, seed=0)
        self.assertGreater(flat["median_final"], 0.0)
        self.assertGreaterEqual(flat["p95_max_drawdown"], flat["median_max_drawdown"])
        with self.assertRaises(ValueError):
            bootstrap_bankroll([], mode="kelly")

    def test_bootstrap_matches_individual_seeded_trajectories_across_batches(self):
        from cs2gp.odds import bootstrap_bankroll

        seq = [(0.65, 2.0, 1), (0.7, 1.8, 0), (0.8, 1.5, 1)]
        samples, horizon, seed = 137, 17, 9
        draws = np.random.default_rng(seed).integers(0, len(seq), size=(samples, horizon))
        for mode in ("flat", "kelly"):
            with self.subTest(mode=mode):
                paths = np.array([
                    equity_curve([seq[i] for i in draw], mode=mode, stake=2.0,
                                 initial_bankroll=100.0)
                    for draw in draws
                ])
                peaks = np.maximum.accumulate(paths, axis=1)
                drawdowns = peaks - paths
                if mode == "kelly":
                    drawdowns /= peaks
                max_dds = drawdowns.max(axis=1)
                result = bootstrap_bankroll(seq, mode=mode, stake=2.0, initial_bankroll=100.0,
                                            samples=samples, horizon=horizon, seed=seed)
                self.assertEqual(result["prob_profit"], np.mean(paths[:, -1] > paths[:, 0]))
                for key, q in (("p05_final", .05), ("median_final", .5), ("p95_final", .95)):
                    self.assertAlmostEqual(result[key], np.quantile(paths[:, -1], q), places=10)
                suffix = "_pct" if mode == "kelly" else ""
                self.assertAlmostEqual(result["median_max_drawdown" + suffix],
                                       np.median(max_dds), places=10)
                self.assertAlmostEqual(result["p95_max_drawdown" + suffix],
                                       np.quantile(max_dds, .95), places=10)

    def test_bootstrap_relative_drawdown_is_invariant_to_initial_capital(self):
        from cs2gp.odds import bootstrap_bankroll

        seq = [(0.6, 2.0, 0)]
        for capital in (1e-100, 1e-15, 1.0, 1e100):
            with self.subTest(capital=capital):
                result = bootstrap_bankroll(seq, samples=1, horizon=1, initial_bankroll=capital)
                self.assertAlmostEqual(result["median_max_drawdown_pct"], .05, places=12)
                self.assertAlmostEqual(result["p95_max_drawdown_pct"], .05, places=12)


if __name__ == "__main__":
    unittest.main()
