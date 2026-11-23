import datetime as dt
from dataclasses import replace
import unittest

import numpy as np

from cs2gp.features import Dataset


class DatasetSlicingTests(unittest.TestCase):
    def setUp(self):
        self.ds = Dataset(
            X=np.arange(18).reshape(6, 3), y=np.array([0, 1, 0, 1, 0, 1]),
            dates=[dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(6)],
            meta=[{"match_id": f"m{i}"} for i in range(6)],
            feature_names=("elo", "form", "pool"),
            pool_X=np.arange(84).reshape(6, 7, 2), versioned_pool=True,
        )

    def assert_selected(self, selected, expected):
        np.testing.assert_array_equal(selected.X, self.ds.X[expected])
        np.testing.assert_array_equal(selected.y, self.ds.y[expected])
        np.testing.assert_array_equal(selected.pool_X, self.ds.pool_X[expected])
        self.assertEqual(selected.dates, [self.ds.dates[i] for i in expected])
        self.assertEqual(selected.meta, [{"match_id": f"m{i}"} for i in expected])
        self.assertEqual(selected.feature_names, self.ds.feature_names)
        self.assertTrue(selected.versioned_pool)
        self.assertEqual(selected.X.shape, (len(expected), 3))
        self.assertEqual(selected.y.shape, (len(expected),))

    def test_boolean_mask_selects_matching_dates_metadata_and_pool_features(self):
        selected = self.ds.slice(np.array([False, True, False, True, False, True]))
        self.assert_selected(selected, [1, 3, 5])
        self.assertEqual(len(self.ds), 6)
        self.assertEqual(len(self.ds.meta), 6)

    def test_integer_indices_slices_and_scalars_preserve_order_and_row_dimensions(self):
        cases = [(slice(1, None, 2), [1, 3, 5]), (slice(None, None, -2), [5, 3, 1]),
                 (np.array([-1, 1, 1]), [5, 1, 1]), (2, [2]),
                 (np.array([4, 0], dtype=np.uint64), [4, 0])]
        for selector, expected in cases:
            with self.subTest(selector=selector):
                self.assert_selected(self.ds.slice(selector), expected)

    def test_empty_selections_keep_consistent_dataset_shapes(self):
        for selector in ([], np.array([], dtype=int), slice(3, 3), np.zeros(6, dtype=bool)):
            with self.subTest(selector=selector):
                selected = self.ds.slice(selector)
                self.assert_selected(selected, [])
                self.assert_selected(selected.slice(slice(None)), [])

    def test_optional_pool_features_are_preserved_as_absent(self):
        selected = replace(self.ds, pool_X=None).slice([False, True, False, True, False, False])
        self.assertIsNone(selected.pool_X)
        self.assertEqual(selected.dates, [self.ds.dates[1], self.ds.dates[3]])
        self.assertEqual([row["match_id"] for row in selected.meta], ["m1", "m3"])

    def test_invalid_selectors_are_rejected_before_creating_misaligned_data(self):
        for selector in ([True, False], True, [1.5], [[1, 2]], None, "one"):
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                self.ds.slice(selector)
        for selector in ([6], [-7], np.array([np.iinfo(np.uint64).max], dtype=np.uint64)):
            with self.subTest(selector=selector), self.assertRaises(IndexError):
                self.ds.slice(selector)


if __name__ == "__main__":
    unittest.main()
