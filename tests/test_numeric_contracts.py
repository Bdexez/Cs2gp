import unittest

import numpy as np

from cs2gp import metrics
from cs2gp.model import LogisticModel, MatchPredictor, recency_weights, series_win_probability


class NumericContractsTests(unittest.TestCase):
    def test_metrics_reject_silent_broadcasting_and_invalid_probabilities(self):
        functions = (metrics.log_loss, metrics.brier, metrics.accuracy,
                     metrics.expected_calibration_error, metrics.reliability_table, metrics.summarize)
        bad_pairs = (([1, 0], [[.9], [.1]]), ([1, 0], [.9]), ([], []),
                     ([1, 0], [.9, np.nan]), ([1, 0], [1.1, 0]),
                     ([1, .5], [.9, .1]), ([[1], [0]], [.9, .1]))
        for fn in functions:
            for y, p in bad_pairs:
                with self.subTest(metric=fn.__name__, y=y, p=p), self.assertRaises(ValueError):
                    fn(y, p)
        self.assertAlmostEqual(metrics.log_loss([1, 0], [.9, .1]), -np.log(.9))
        self.assertEqual(metrics.brier([1, 0], [1, 0]), 0)
        self.assertTrue(np.isfinite(metrics.log_loss([0, 1], [1, 0])))

    def test_calibration_requires_positive_integer_bins(self):
        for fn in (metrics.expected_calibration_error, metrics.reliability_table):
            for bins in (0, -1, 1.5, True):
                with self.subTest(bins=bins), self.assertRaises(ValueError):
                    fn([1, 0], [.9, .1], bins=bins)

    def test_training_rejects_inconsistent_arrays_and_weights(self):
        X = np.array([[1., 2.], [-1., -2.]])
        for x, y in ((X, [1]), (X, [[1], [0]]), (X, [1, np.nan]),
                     (X * np.nan, [1, 0]), (np.empty((2, 0)), [1, 0])):
            for model in (LogisticModel(), MatchPredictor(use_gbdt=False)):
                with self.subTest(model=type(model).__name__), self.assertRaises(ValueError):
                    model.fit(x, y)
        for weights in ([1], [[1], [1]], [0, 0], [1, -1], [np.inf, 1]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                LogisticModel().fit(X, [1, 0], weights)
        model = LogisticModel().fit(X, [1, 0], [1, 0])
        for bad in ([[1]], [[np.nan, 1]], np.empty((0, 2))):
            with self.assertRaises(ValueError):
                model.predict_proba(bad)

    def test_model_parameters_fail_before_fitting(self):
        X = np.arange(120, dtype=float).reshape(60, 2)
        y = np.tile([1, 0], 30)
        for options in (dict(l2=-1), dict(gbdt_weight=2), dict(recency_halflife=0),
                        dict(calibration_frac=np.nan)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                MatchPredictor(**options).fit(X, y)
        for options in (dict(max_iter=0), dict(tol=0), dict(l2=np.inf)):
            with self.assertRaises(ValueError):
                LogisticModel(**options).fit(X, y)
        with self.assertRaises(ValueError):
            recency_weights([], halflife=0)

    def test_zero_weight_rows_do_not_affect_residualization_or_scaling(self):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(40, 3))
        y = (X[:, 0] > 0).astype(float)
        clean = LogisticModel().fit(X, y)
        X_aug = np.vstack([X, [[100.0, -500.0, 900.0]]])
        y_aug = np.append(y, 0.0)
        w_aug = np.append(np.ones(len(y)), 0.0)
        weighted = LogisticModel().fit(X_aug, y_aug, sample_weight=w_aug)
        np.testing.assert_allclose(weighted.resid, clean.resid, atol=1e-12)
        np.testing.assert_allclose(weighted.scales, clean.scales, atol=1e-12)
        np.testing.assert_allclose(weighted.weights, clean.weights, atol=1e-10)

    def test_logistic_preserves_weighted_mirror_regularization_strength(self):
        # Pour une feature constante, l'optimum penalise se ramene a une
        # equation scalaire monotone, resolue ici independamment par dichotomie.
        y = np.array([0] * 40 + [1] * 60, dtype=float)
        w = np.linspace(0.25, 1.75, len(y))
        l2 = 8.0
        rate = np.average(y, weights=w)
        lo, hi = 1e-8, 1.0 - 1e-8
        for _ in range(60):
            q = (lo + hi) / 2.0
            residual = q + l2 * np.log(q / (1.0 - q)) / (2.0 * w.sum()) - rate
            if residual > 0:
                hi = q
            else:
                lo = q
        expected = (lo + hi) / 2.0
        model = LogisticModel(l2=l2).fit(np.ones((len(y), 1)), y, sample_weight=w)
        np.testing.assert_allclose(model.predict_proba([[1.0]]), [expected], atol=1e-12)

    def test_logistic_mirror_augmentation_matches_rescaled_regularization(self):
        rng = np.random.default_rng(52)
        X = rng.normal(size=(80, 4))
        y = rng.integers(0, 2, len(X))
        w = rng.uniform(0.0, 2.0, len(X))
        model = LogisticModel(l2=3.0).fit(X, y, sample_weight=w)
        mirrored = LogisticModel(l2=6.0).fit(
            np.vstack([X, -X]), np.concatenate([y, 1 - y]), sample_weight=np.tile(w, 2)
        )
        np.testing.assert_allclose(model.predict_proba(X), mirrored.predict_proba(X), atol=1e-12)
        np.testing.assert_allclose(model.predict_proba(X) + model.predict_proba(-X), 1.0, atol=1e-12)

    def test_explain_and_series_win_probability_validate_inputs(self):
        untrained = MatchPredictor(use_gbdt=False)
        with self.assertRaises(RuntimeError):
            untrained.explain(np.array([1.0, 2.0]))
        rng = np.random.default_rng(3)
        X = rng.normal(size=(60, 2))
        model = MatchPredictor(use_gbdt=False).fit(X, (X[:, 0] > 0).astype(float))
        with self.assertRaises(ValueError):
            model.explain(np.array([np.nan, 1.0]))
        with self.assertRaises(ValueError):
            model.explain(np.array([1.0, 2.0]), top=0)
        for bad_p in (float("nan"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                series_win_probability(bad_p, 3)
        for bad_bo in (0, 2, 4, -1, True):
            with self.assertRaises(ValueError):
                series_win_probability(0.6, bad_bo)

    def test_backtest_validates_momentum_and_min_edge(self):
        from cs2gp.backtest import BacktestResult, bet_backtest, walk_forward

        with self.assertRaises(ValueError):
            walk_forward(None, momentum=float("nan"))
        empty = BacktestResult(
            predictions=np.array([]), y=np.array([]), dates=[], meta=[],
            elo_only=np.array([]), folds=0, metrics={}, elo_metrics={},
        )
        with self.assertRaises(ValueError):
            bet_backtest(empty, {}, min_edge=float("nan"))


if __name__ == "__main__":
    unittest.main()
