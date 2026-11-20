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

    def test_calibration_slope_converges_from_extreme_probabilities_to_known_solution(self):
        # Deux groupes : taux observes 25 % et 75 %, p annoncees 0 et 1.
        # Avec le clipping de Cox, la solution non penalisee est analytique.
        y = np.array([0] * 75 + [1] * 25 + [0] * 25 + [1] * 75, dtype=float)
        p = np.repeat([0.0, 1.0], 100)
        result = metrics.calibration_slope_intercept(y, p)
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["intercept"], 0.0, places=8)
        expected_slope = np.log(3.0) / np.log((1.0 - 1e-6) / 1e-6)
        self.assertAlmostEqual(result["slope"], expected_slope, places=8)

    def test_calibration_slope_reaches_stationary_point_on_contradictory_extreme_inputs(self):
        for seed in (0, 1, 11, 42):
            rng = np.random.default_rng(seed)
            p = rng.choice([0.0, .00001, .01, .05, .1, .9, .95, .99, .99999, 1.0], 30)
            y = rng.integers(0, 2, 30)
            result = metrics.calibration_slope_intercept(y, p)
            with self.subTest(seed=seed):
                self.assertTrue(result["converged"])
                pc = np.clip(p, 1e-6, 1.0 - 1e-6)
                X = np.column_stack([np.ones(len(y)), np.log(pc / (1.0 - pc))])
                beta = np.array([result["intercept"], result["slope"]])
                q = 1.0 / (1.0 + np.exp(-(X @ beta)))
                gradient = X.T @ (y - q) - 1e-6 * (beta - [0.0, 1.0])
                np.testing.assert_allclose(gradient, 0.0, atol=1e-6)
                self.assertLess(metrics.log_loss(y, q), metrics.log_loss(y, p))
                reversed_result = metrics.calibration_slope_intercept(1 - y, 1 - p)
                self.assertTrue(reversed_result["converged"])
                self.assertAlmostEqual(result["slope"], reversed_result["slope"], places=8)
                self.assertAlmostEqual(result["intercept"], -reversed_result["intercept"], places=8)

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

    def test_sharpness_measures_confidence_entropy_and_decisiveness(self):
        flat = metrics.sharpness([0.5, 0.5, 0.5, 0.5])
        sharp = metrics.sharpness([0.9, 0.1, 0.8, 0.2])
        self.assertAlmostEqual(flat["mean_confidence"], 0.5, places=10)
        self.assertEqual(flat["pct_decisive_70"], 0.0)
        self.assertAlmostEqual(sharp["mean_confidence"], 0.85, places=10)
        self.assertEqual(sharp["pct_decisive_70"], 1.0)
        self.assertLess(sharp["mean_entropy"], flat["mean_entropy"])
        with self.assertRaises(ValueError):
            metrics.sharpness([])

    def test_confusion_summary_computes_counts_f1_and_balanced_accuracy(self):
        y = [1, 1, 0, 0]
        p = [0.8, 0.4, 0.3, 0.9]
        c = metrics.confusion_summary(y, p, threshold=0.5)
        self.assertEqual((c["tp"], c["fp"], c["tn"], c["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(c["precision"], 0.5)
        self.assertAlmostEqual(c["recall"], 0.5)
        self.assertAlmostEqual(c["f1"], 0.5)
        self.assertAlmostEqual(c["balanced_accuracy"], 0.5)
        with self.assertRaises(ValueError):
            metrics.confusion_summary(y, p, threshold=1.5)

    def test_confusion_summary_distinguishes_zero_f1_from_undefined(self):
        cases = [
            ([0, 1], [1, 0], 0.0),  # Toutes les predictions sont fausses.
            ([1, 1], [0, 0], 0.0),  # Aucun positif predit.
            ([0, 0], [1, 1], 0.0),  # Aucun positif observe.
            ([0, 0], [0, 0], None),  # Aucun positif observe ou predit.
            ([1, 1], [1, 1], 1.0),
        ]
        for y, p, expected in cases:
            with self.subTest(y=y, p=p):
                result = metrics.confusion_summary(y, p)
                self.assertEqual(result["f1"], expected)
                self.assertEqual(sum(result[k] for k in ("tp", "fp", "tn", "fn")), len(y))
                if len(set(y)) == 1:
                    self.assertIsNone(result["balanced_accuracy"])

    def test_confusion_summary_threshold_boundaries_and_validation(self):
        for threshold, counts in [(0.0, (1, 1, 0, 0)), (0.5, (1, 0, 1, 0)),
                                  (1.0, (0, 0, 1, 1))]:
            result = metrics.confusion_summary([0, 1], [0.0, 0.5], threshold=threshold)
            self.assertEqual(tuple(result[k] for k in ("tp", "fp", "tn", "fn")), counts)
        for threshold in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                metrics.confusion_summary([0, 1], [0.1, 0.9], threshold=threshold)
        for y, p in [([], []), ([0, 2], [0.1, 0.9]), ([0, 1], [0.5]),
                     ([0, 1], [0.1, float("nan")])]:
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                metrics.confusion_summary(y, p)


if __name__ == "__main__":
    unittest.main()
