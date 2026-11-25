import datetime as dt
import unittest
from unittest.mock import patch

import numpy as np

from cs2gp.model import MatchPredictor


class PredictorStateTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.X = rng.normal(size=(260, 3))
        self.y = (self.X[:, 0] > 0).astype(float)
        self.dates = [dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(len(self.y))]
        self.options = dict(use_gbdt=True, l2_grid=(.5, 3.0))
        self.model = MatchPredictor(**self.options).fit(self.X, self.y, self.dates, ("a", "b", "c"))
        self.next_dates = [day + dt.timedelta(days=400) for day in self.dates[:220]]

    def assert_preserved(self, model, before, expected):
        self.assertIs(model.logistic, before["logistic"])
        self.assertIs(model.gbdt, before["gbdt"])
        for name in ("l2", "temperature", "feature_names", "trained_on", "last_date"):
            self.assertEqual(getattr(model, name), before[name])
        np.testing.assert_array_equal(model.predict_proba(self.X), expected)

    def test_failed_refit_preserves_models_metadata_and_predictions_at_every_stage(self):
        before = vars(self.model).copy()
        expected = self.model.predict_proba(self.X)
        for target in ("cs2gp.model.select_l2", "cs2gp.model.LogisticModel.fit",
                       "cs2gp.model.MatchPredictor._fit_gbdt", "cs2gp.model._fit_temperature"):
            with self.subTest(stage=target), patch(target, side_effect=ValueError("echec voulu")):
                with self.assertRaisesRegex(ValueError, "echec voulu"):
                    self.model.fit(self.X[:220], 1 - self.y[:220], self.next_dates, ("x", "y", "z"))
            self.assert_preserved(self.model, before, expected)

    def test_invalid_feature_names_cannot_change_an_existing_model(self):
        before = vars(self.model).copy()
        expected = self.model.predict_proba(self.X)
        with self.assertRaises(ValueError):
            self.model.fit(self.X, self.y, self.dates, ("one",))
        self.assert_preserved(self.model, before, expected)

    def test_failed_initial_fit_keeps_model_untrained(self):
        model = MatchPredictor(**self.options)
        original_logistic = model.logistic
        with patch("cs2gp.model._fit_temperature", side_effect=ValueError("calibration impossible")):
            with self.assertRaises(ValueError):
                model.fit(self.X, self.y, self.dates, ("a", "b", "c"))
        self.assertIs(model.logistic, original_logistic)
        self.assertIsNone(model.logistic.weights)
        self.assertIsNone(model.gbdt)
        self.assertEqual(model.trained_on, 0)
        self.assertEqual(model.last_date, "")
        self.assertEqual(model.feature_names, ())
        with self.assertRaises(RuntimeError):
            model.predict_proba(self.X)

    def test_successful_refit_publishes_the_same_state_as_a_fresh_fit(self):
        previous = self.model.logistic
        expected = MatchPredictor(**self.options).fit(
            self.X[:220], 1 - self.y[:220], self.next_dates, ("x", "y", "z"),
        )
        self.assertIs(self.model.fit(self.X[:220], 1 - self.y[:220], self.next_dates, ("x", "y", "z")),
                      self.model)
        self.assertIsNot(self.model.logistic, previous)
        self.assertEqual(self.model.feature_names, expected.feature_names)
        self.assertEqual(self.model.trained_on, 220)
        self.assertEqual(self.model.last_date, expected.last_date)
        self.assertEqual(self.model.l2, expected.l2)
        self.assertEqual(self.model.temperature, expected.temperature)
        np.testing.assert_array_equal(self.model.predict_proba(self.X), expected.predict_proba(self.X))


if __name__ == "__main__":
    unittest.main()
