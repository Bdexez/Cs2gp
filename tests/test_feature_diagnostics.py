import datetime as dt
import json
from dataclasses import replace
import unittest

import numpy as np

from cs2gp.features import Dataset, feature_correlation_report


class FeatureDiagnosticsTests(unittest.TestCase):
    def dataset(self, X, names=()):
        return Dataset(X=X, y=np.zeros(len(X)), dates=[dt.date(2026, 1, 1)] * len(X),
                       meta=[{} for _ in range(len(X))], feature_names=names)

    def test_large_feature_amplitudes_preserve_correlations_without_mutating_data(self):
        x = np.arange(1, 21, dtype=float)
        X = np.column_stack([x, 2 * x + 1, np.zeros_like(x), np.sin(x)])
        names = ("signal", "copy", "constant", "independent")
        baseline = feature_correlation_report(self.dataset(X, names), threshold=.95)
        large = X * 1e200
        previous = large.copy()
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            report = feature_correlation_report(self.dataset(large, names), threshold=.95)
        np.testing.assert_array_equal(large, previous)
        self.assertEqual(report["constant_features"], ["constant"])
        self.assertEqual(len(report["high_pairs"]), 1)
        self.assertEqual((report["high_pairs"][0]["a"], report["high_pairs"][0]["b"]), ("signal", "copy"))
        self.assertAlmostEqual(report["high_pairs"][0]["corr"], baseline["high_pairs"][0]["corr"], places=12)
        json.dumps(report, allow_nan=False)

    def test_small_variations_on_large_offsets_keep_their_linear_relationship(self):
        x = np.arange(20, dtype=float) * 2
        X = np.column_stack([1e16 + x, 1e16 + 3 * x, np.full_like(x, 1e16)])
        report = feature_correlation_report(self.dataset(X, ("a", "b", "constant")))
        self.assertEqual(report["constant_features"], ["constant"])
        self.assertEqual(len(report["high_pairs"]), 1)
        self.assertAlmostEqual(report["high_pairs"][0]["corr"], 1.0, places=12)

    def test_opposite_extreme_values_use_finite_fallback_centering(self):
        X = np.array([[1.7e308, -1.7e308, 1.7e308], [-1.7e308, 1.7e308, 1.7e308]])
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            report = feature_correlation_report(self.dataset(X, ("a", "b", "constant")))
        self.assertEqual(report["constant_features"], ["constant"])
        self.assertEqual(report["high_pairs"], [{"a": "a", "b": "b", "corr": -1.0}])
        json.dumps(report, allow_nan=False)

    def test_invalid_matrices_names_and_thresholds_raise_value_error(self):
        valid = self.dataset(np.eye(2))
        cases = [replace(valid, X=np.array([1.0, 2.0])),
                 replace(valid, X=np.array([[np.nan, 0], [0, 1]])),
                 replace(valid, X=np.array([[np.inf, 0], [0, 1]])),
                 replace(valid, X=np.array([[1j, 0], [0, 1j]])),
                 replace(valid, feature_names=("same", "same")),
                 replace(valid, feature_names=(" ", "x")),
                 replace(valid, feature_names=("one",))]
        for ds in cases:
            with self.subTest(ds=ds), self.assertRaises(ValueError):
                feature_correlation_report(ds)
        for threshold in (-.1, 1.1, True, np.nan, np.inf, ".5"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                feature_correlation_report(valid, threshold=threshold)

    def test_insufficient_or_empty_data_reports_no_correlations(self):
        for shape in ((0, 2), (1, 2), (3, 0)):
            with self.subTest(shape=shape):
                report = feature_correlation_report(self.dataset(np.ones(shape)))
                self.assertEqual(report["n_samples"], shape[0])
                self.assertEqual(report["n_features"], shape[1])
                self.assertEqual(report["constant_features"], [])
                self.assertEqual(report["high_pairs"], [])


if __name__ == "__main__":
    unittest.main()
