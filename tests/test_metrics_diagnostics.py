import datetime as dt
import unittest

import numpy as np

from cs2gp import metrics


class MetricsDiagnosticsTests(unittest.TestCase):
    def test_murphy_brier_decomposition_identity_and_skill(self):
        rng = np.random.default_rng(42)
        p = rng.uniform(0.1, 0.9, size=400)
        y = (rng.random(400) < p).astype(float)
        d = metrics.brier_decomposition(y, p, bins=10)
        self.assertGreaterEqual(d["reliability"], 0.0)
        self.assertGreaterEqual(d["resolution"], 0.0)
        self.assertGreater(d["uncertainty"], 0.0)
        self.assertGreater(d["brier_skill_score"], 0.0)
        # Binned Murphy identity holds closely up to within-bin variance of p
        self.assertAlmostEqual(
            d["brier"], d["reliability"] - d["resolution"] + d["uncertainty"], delta=0.01
        )
        # Constant prediction equals base rate -> zero resolution, zero skill
        p_const = np.full_like(y, y.mean())
        d_const = metrics.brier_decomposition(y, p_const)
        self.assertAlmostEqual(d_const["reliability"], 0.0, places=10)
        self.assertAlmostEqual(d_const["resolution"], 0.0, places=10)
        self.assertAlmostEqual(d_const["brier_skill_score"], 0.0, places=10)

    def test_calibration_slope_detects_overconfidence_and_underconfidence(self):
        rng = np.random.default_rng(7)
        z = rng.normal(0.0, 1.4, size=2000)
        true_p = 1.0 / (1.0 + np.exp(-z))
        y = (rng.random(2000) < true_p).astype(float)

        calibrated = metrics.calibration_slope_intercept(y, true_p)
        self.assertTrue(calibrated["converged"])
        self.assertAlmostEqual(calibrated["slope"], 1.0, delta=0.15)
        self.assertAlmostEqual(calibrated["intercept"], 0.0, delta=0.15)

        overconfident_p = 1.0 / (1.0 + np.exp(-2.0 * z))
        over = metrics.calibration_slope_intercept(y, overconfident_p)
        self.assertTrue(over["converged"])
        self.assertLess(over["slope"], 0.65)

        underconfident_p = 1.0 / (1.0 + np.exp(-0.5 * z))
        under = metrics.calibration_slope_intercept(y, underconfident_p)
        self.assertTrue(under["converged"])
        self.assertGreater(under["slope"], 1.5)

    def test_calibration_slope_handles_degenerate_inputs(self):
        res = metrics.calibration_slope_intercept([1, 1, 1], [0.8, 0.7, 0.9])
        self.assertFalse(res["converged"])
        self.assertIsNone(res["slope"])
        res_const = metrics.calibration_slope_intercept([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
        self.assertFalse(res_const["converged"])

    def test_paired_block_bootstrap_supports_brier_metric(self):
        dates = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(40)]
        y = np.array([1, 0] * 20, dtype=float)
        p = np.array([0.8, 0.2] * 20, dtype=float)
        ref = np.full(40, 0.5)
        res = metrics.paired_block_bootstrap(y, p, ref, dates, samples=200, block_days=5, metric="brier")
        self.assertEqual(res["metric"], "brier")
        expected_gain = metrics.brier(y, ref) - metrics.brier(y, p)
        self.assertAlmostEqual(res["gain"], expected_gain, places=10)
        self.assertGreater(res["low"], 0.0)
        with self.assertRaises(ValueError):
            metrics.paired_block_bootstrap(y, p, ref, dates, metric="unknown")

    def test_quantile_reliability_table_creates_balanced_bins(self):
        rng = np.random.default_rng(5)
        p = rng.uniform(0.55, 0.90, size=100)
        y = (rng.random(100) < p).astype(float)
        rows = metrics.quantile_reliability_table(y, p, bins=5)
        self.assertEqual(len(rows), 5)
        self.assertEqual(sum(r["n"] for r in rows), 100)
        self.assertTrue(all(r["n"] == 20 for r in rows))
        self.assertTrue(all(a["predicted"] <= b["predicted"] for a, b in zip(rows, rows[1:])))

    def test_spiegelhalter_z_distinguishes_calibrated_from_miscalibrated(self):
        rng = np.random.default_rng(19)
        p = rng.uniform(0.15, 0.85, size=1000)
        y = (rng.random(1000) < p).astype(float)
        good = metrics.spiegelhalter_z(y, p)
        self.assertIsNotNone(good["z_stat"])
        self.assertFalse(good["miscalibrated_5pct"])

        # Severely overconfident predictions fail Spiegelhalter test
        bad_p = np.clip(0.5 + 1.8 * (p - 0.5), 0.02, 0.98)
        bad = metrics.spiegelhalter_z(y, bad_p)
        self.assertTrue(bad["miscalibrated_5pct"])
        self.assertLess(bad["p_value"], 0.01)

        # Constant 0.5 predictions have zero denominator -> graceful None
        deg = metrics.spiegelhalter_z([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
        self.assertIsNone(deg["z_stat"])

    def test_max_calibration_error_bounds_ece_and_exposes_in_summarize(self):
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
        p = np.array([0.8, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5], dtype=float)
        ece = metrics.expected_calibration_error(y, p, bins=10)
        mce = metrics.max_calibration_error(y, p, bins=10)
        self.assertGreaterEqual(mce, ece)
        self.assertAlmostEqual(mce, 0.5, places=10)
        self.assertAlmostEqual(metrics.summarize(y, p)["mce"], mce, places=10)

    def test_paired_loss_test_detects_significant_gain_on_log_loss_and_brier(self):
        rng = np.random.default_rng(11)
        p = np.where(np.arange(200) % 2 == 0, 0.85, 0.15)
        y = (rng.random(200) < p).astype(float)
        ref = np.full(200, 0.5)
        for metric in ("log_loss", "brier"):
            res = metrics.paired_loss_test(y, p, ref, metric=metric)
            self.assertGreater(res["gain"], 0.0)
            self.assertTrue(res["significant_5pct"])
            self.assertLess(res["p_value"], 0.01)
        with self.assertRaises(ValueError):
            metrics.paired_loss_test(y, p, ref, metric="unknown")

    def test_favorite_reliability_table_folds_symmetric_probabilities(self):
        y = np.array([1, 0, 1, 0, 1, 0], dtype=float)
        p = np.array([0.85, 0.15, 0.65, 0.35, 0.55, 0.45], dtype=float)
        rows = metrics.favorite_reliability_table(y, p, bins=5)
        self.assertEqual(sum(r["n"] for r in rows), 6)
        self.assertTrue(all(0.5 <= r["predicted"] <= 1.0 for r in rows))
        self.assertTrue(all(r["observed"] == 1.0 for r in rows))

    def test_hosmer_lemeshow_grows_under_nonlinear_miscalibration(self):
        rng = np.random.default_rng(33)
        p = rng.uniform(0.1, 0.9, size=600)
        y = (rng.random(600) < p).astype(float)
        good = metrics.hosmer_lemeshow(y, p, bins=10)
        bad = metrics.hosmer_lemeshow(y, 1.0 - p, bins=10)
        self.assertEqual(good["groups"], 10)
        self.assertEqual(good["df"], 8)
        self.assertLess(good["statistic"], bad["statistic"] * 0.1)

    def test_roc_auc_handles_perfect_random_ties_and_single_class(self):
        self.assertAlmostEqual(metrics.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0, places=10)
        self.assertAlmostEqual(metrics.roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]), 0.0, places=10)
        self.assertAlmostEqual(metrics.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5, places=10)
        self.assertAlmostEqual(metrics.roc_auc([0, 1, 0, 1], [0.2, 0.5, 0.5, 0.8]), 0.875, places=10)
        self.assertIsNone(metrics.roc_auc([1, 1, 1], [0.7, 0.8, 0.9]))
        summary = metrics.summarize([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(summary["roc_auc"], 1.0, places=10)
        self.assertIn("auc=1.000", metrics.format_summary("test", summary))


if __name__ == "__main__":
    unittest.main()
