import datetime as dt
from pathlib import Path
import unittest

import numpy as np

from cs2gp.backtest import walk_forward
from cs2gp.features import build_datasets
from cs2gp.schema import group_matches, read_csv
from cs2gp.sources import simulate_history, simulate_with_truth


class SimulationTests(unittest.TestCase):
    def test_bundled_sample_contains_only_complete_valid_series(self):
        root = Path(__file__).resolve().parents[1]
        matches = group_matches(read_csv(str(root / "data/sample_matches.csv")))
        self.assertGreater(len(matches), 400)
        self.assertTrue(all(m.is_complete for m in matches))

    def test_both_regimes_are_reproducible_and_have_distinct_maps(self):
        for regime in ("facile", "realiste"):
            for seed in (3, 7, 11):
                with self.subTest(regime=regime, seed=seed):
                    options = dict(n_teams=8, years=.5, seed=seed, regime=regime,
                                   start=dt.date(2023, 1, 1))
                    maps = simulate_history(**options)
                    self.assertEqual(maps, simulate_history(**options))
                    matches = group_matches(maps)
                    self.assertTrue(matches)
                    for match in matches:
                        self.assertTrue(match.is_complete)
                        self.assertEqual(len({m.map_name for m in match.maps}), len(match.maps))
                    ds, pds, _ = build_datasets(matches)
                    result = walk_forward(ds, pds, initial_frac=.6, use_gbdt=False)
                    self.assertTrue(np.all(np.isfinite(result.predictions)))
                    self.assertTrue(np.all((result.predictions > 0) & (result.predictions < 1)))
                    self.assertTrue(np.isfinite(result.metrics["log_loss"]))

    def test_truth_sampling_does_not_change_generated_history(self):
        options = dict(n_teams=4, years=.1, seed=7, start=dt.date(2023, 1, 1))
        expected = simulate_history(**options)
        actual, truth = simulate_with_truth(**options, veto_samples=20)
        self.assertEqual(actual, expected)
        self.assertEqual(set(truth), {m.match_id for m in actual})
        self.assertTrue(all(0 <= p <= 1 for p in truth.values()))

    def test_invalid_inputs_are_rejected(self):
        for options in (dict(n_teams=1), dict(n_teams=41), dict(years=0),
                        dict(years=float("nan")), dict(regime="missing")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                simulate_history(**options)

    def test_with_rosters_preserves_matches_and_tracks_reshuffles(self):
        from cs2gp.audit import audit_history

        opts = dict(n_teams=8, years=1.0, seed=7, start=dt.date(2024, 1, 1), regime="realiste")
        plain = simulate_history(**opts)
        rostered = simulate_history(**opts, with_rosters=True)
        self.assertEqual([(m.match_id, m.rounds_a, m.rounds_b) for m in plain],
                         [(m.match_id, m.rounds_a, m.rounds_b) for m in rostered])
        self.assertTrue(all(len(m.roster_a) == 5 and len(m.roster_b) == 5 for m in rostered))
        distinct_rosters = {m.roster_a for m in rostered if m.team_a == rostered[0].team_a}
        self.assertGreaterEqual(len(distinct_rosters), 1)
        self.assertEqual(audit_history(rostered)["roster_coverage"], 1.0)

        truth_rostered, truth = simulate_with_truth(**opts, veto_samples=30, with_rosters=True)
        self.assertEqual(truth_rostered, rostered)
        self.assertEqual(set(truth), {m.match_id for m in rostered})

    def test_realistic_regime_oracle_benchmark(self):
        from cs2gp.backtest import benchmark_against_truth

        rows, truth = simulate_with_truth(
            n_teams=12, years=1.5, seed=7, start=dt.date(2024, 1, 1),
            regime="realiste", veto_samples=50, with_rosters=True,
        )
        ds, pds, _ = build_datasets(group_matches(rows))
        res = walk_forward(ds, pds, initial_frac=0.5, use_gbdt=False)
        bench = benchmark_against_truth(res, truth)
        self.assertEqual(bench["n_matched"], len(res.y))
        self.assertGreater(bench["oracle_log_loss"], 0.45)
        self.assertLess(abs(bench["excess_log_loss"]), 0.05)
        self.assertLess(bench["mean_abs_prob_error"], 0.15)


if __name__ == "__main__":
    unittest.main()
