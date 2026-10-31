import contextlib
import csv
import datetime as dt
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cs2gp.backtest import BacktestResult, benchmark_against_truth, benchmark_regimes, read_odds_csv
from cs2gp.cli import main
from cs2gp.metrics import summarize
from cs2gp.schema import write_csv
from cs2gp.sources import simulate_history


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
                "team_a": "a",
                "team_b": "b",
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

    def test_drift_report_export_csv_and_read_odds_csv(self):
        res = self._sample_result()
        res.fold_slices = [(0, 20), (20, 40)]
        drift = res.drift_report()
        self.assertEqual(len(drift), 2)
        self.assertEqual(drift[0]["fold"], 1)
        self.assertEqual(drift[0]["n"], 20)
        self.assertGreater(drift[0]["gain_vs_elo"], 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            csv_out = Path(tmp) / "preds.csv"
            self.assertEqual(res.export_csv(csv_out), 40)
            with csv_out.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 40)
            self.assertEqual(rows[0]["match_id"], "m0")

            odds_path = Path(tmp) / "odds.csv"
            odds_path.write_text(
                "match_id,odds_a,odds_b,closing_odds_a,closing_odds_b\n"
                "m0,1.90,1.90,1.60,2.30\nm1,2.10,1.80,2.00,1.85\n",
                encoding="utf-8",
            )
            lookup = read_odds_csv(odds_path)
            self.assertEqual(lookup["m0"], (1.90, 1.90, 1.60, 2.30))
            from cs2gp.backtest import bet_backtest
            bb = bet_backtest(res, lookup, min_edge=0.04)
            self.assertGreater(bb["clv_bets"], 0)
            self.assertIsNotNone(bb["avg_clv_prob"])

            dup_path = Path(tmp) / "dup.csv"
            dup_path.write_text("match_id,odds_a,odds_b\nm0,1.90,1.90\nm0,2.00,1.85\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplique"):
                read_odds_csv(dup_path)

            bad_path = Path(tmp) / "bad.csv"
            bad_path.write_text("match_id,odds_a,odds_b\nm0,0.95,2.00\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_odds_csv(bad_path)

    def test_cli_backtest_supports_by_fold_export_csv_and_odds_csv(self):
        rows = simulate_history(n_teams=8, years=1.0, seed=19, start=dt.date(2025, 1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "history.csv"
            export_path = Path(tmp) / "oos.csv"
            odds_path = Path(tmp) / "odds.csv"
            write_csv(str(data), rows)
            match_ids = sorted({m.match_id for m in rows})
            odds_lines = ["match_id,odds_a,odds_b,closing_odds_a,closing_odds_b"] + [
                f"{mid},1.95,1.95,1.85,2.05" for mid in match_ids
            ]
            odds_path.write_text("\n".join(odds_lines) + "\n", encoding="utf-8")

            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "backtest", "--data", str(data), "--no-gbdt", "--bootstrap-samples", "0",
                    "--by-fold", "--export-csv", str(export_path),
                    "--odds-csv", str(odds_path), "--json",
                ]), 0)
            payload = json.loads(out.getvalue())
            self.assertIn("by_fold", payload)
            self.assertIn("bet_backtest", payload)
            self.assertIn("favorite_reliability", payload)
            self.assertEqual(len(payload["temperatures"]), payload["folds"])
            self.assertGreater(payload["bet_backtest"]["matches_with_odds"], 0)
            self.assertGreater(payload["bet_backtest"]["avg_overround"], 1.0)
            self.assertTrue(export_path.is_file())

            text_out = io.StringIO()
            with contextlib.redirect_stdout(text_out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "backtest", "--data", str(data), "--no-gbdt", "--bootstrap-samples", "0",
                    "--odds-csv", str(odds_path),
                ]), 0)
            self.assertIn("Confrontation aux cotes historiques", text_out.getvalue())
            self.assertIn("CLV", text_out.getvalue())

    def test_benchmark_regimes_aggregates_multiple_seeds(self):
        report = benchmark_regimes(
            regimes=("realiste",), seeds=(3, 7), n_teams=8, years=1.0,
            veto_samples=20, initial_frac=0.5,
        )
        self.assertIn("realiste", report)
        self.assertEqual(len(report["realiste"]["runs"]), 2)
        self.assertIn("excess_log_loss", report["realiste"]["mean"])
        self.assertGreaterEqual(report["realiste"]["std"]["model_log_loss"], 0.0)
        with self.assertRaises(ValueError):
            benchmark_regimes(regimes=(), seeds=(1,))

    def test_walk_forward_importance_ranks_features_out_of_sample(self):
        from cs2gp.backtest import walk_forward_importance
        from cs2gp.features import build_datasets
        from cs2gp.schema import group_matches

        rows = simulate_history(n_teams=8, years=1.0, seed=13, start=dt.date(2025, 1, 1))
        mds, _, _ = build_datasets(group_matches(rows))
        imp = walk_forward_importance(
            mds, initial_frac=0.6, refit_every=150, min_train=80, repeats=3, seed=0, use_gbdt=False
        )
        self.assertEqual(len(imp), len(mds.feature_names))
        self.assertGreater(imp[0]["mean_delta"], 0.0)
        self.assertGreaterEqual(imp[0]["folds"], 1)


if __name__ == "__main__":
    unittest.main()
