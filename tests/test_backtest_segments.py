import datetime as dt
import unittest

import numpy as np

from cs2gp.backtest import BacktestResult, benchmark_against_truth
from cs2gp.metrics import summarize


class BacktestSegmentTests(unittest.TestCase):
    def _sample_result(self) -> BacktestResult:
        n = 40
        dates = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(n)]
        y = np.array([1, 0] * (n // 2), dtype=float)
        preds = np.array([0.75, 0.25] * (n // 2), dtype=float)
        elo = np.array([0.65, 0.35] * (n // 2), dtype=float)
        meta = [
            {
                "match_id": f"m{i}",
                "best_of": 3 if i < 24 else 1,
                "lan": i % 2 == 0,
                "tier": "s" if i < 20 else "b",
            }
            for i in range(n)
        ]
        return BacktestResult(
            predictions=preds, y=y, dates=dates, meta=meta,
            elo_only=elo, folds=2,
            metrics=summarize(y, preds), elo_metrics=summarize(y, elo),
        )

    def test_breakdown_groups_by_metadata_and_computes_gain_vs_elo(self):
        res = self._sample_result()
        by_bo = res.breakdown(lambda m: f"bo{m['best_of']}")
        self.assertEqual(set(by_bo), {"bo1", "bo3"})
        self.assertEqual(by_bo["bo3"]["n"], 24)
        self.assertEqual(by_bo["bo1"]["n"], 16)
        self.assertGreater(by_bo["bo3"]["log_loss_gain_vs_elo"], 0.0)
        self.assertEqual(set(res.breakdown("tier")), {"b", "s"})

    def test_breakdown_respects_min_n_and_formats_report(self):
        res = self._sample_result()
        self.assertEqual(set(res.breakdown("best_of", min_n=20)), {"3"})
        formatted = res.format_breakdown("Par format", lambda m: f"bo{m['best_of']}")
        self.assertIn("Par format", formatted)
        self.assertIn("gain_elo=", formatted)
        report = res.report()
        self.assertIn("Calibration logistique (Cox)", report)
        self.assertIn("Decomposition de Brier (Murphy)", report)

    def test_benchmark_against_truth_computes_excess_losses(self):
        res = self._sample_result()
        truth = {f"m{i}": 0.80 if i % 2 == 0 else 0.20 for i in range(40)}
        bench = benchmark_against_truth(res, truth)
        self.assertEqual(bench["n_matched"], 40)
        self.assertAlmostEqual(bench["mean_abs_prob_error"], 0.05, places=10)
        self.assertGreater(bench["elo_excess_log_loss"], bench["excess_log_loss"])
        with self.assertRaises(ValueError):
            benchmark_against_truth(res, {"unknown": 0.5})


if __name__ == "__main__":
    unittest.main()
