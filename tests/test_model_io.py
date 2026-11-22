import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cs2gp.model import MatchPredictor


class ModelIntegrityTests(unittest.TestCase):
    def test_invalid_model_metadata_is_rejected(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(60, 2))
        model = MatchPredictor(use_gbdt=False).fit(X, (X[:, 0] > 0).astype(float))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            original = json.loads(path.read_text())
            for changed in (dict(version=99), dict(scales=[0, 1]), dict(weights=[float("nan"), 1]),
                            dict(temperature=0), dict(resid=[1, 2]), dict(feature_names=["one"]),
                            dict(version=True), dict(temperature=True), dict(temperature="1"),
                            dict(has_gbdt="false"), dict(has_gbdt=1), dict(trained_on=-1),
                            dict(trained_on=True), dict(trained_on=1.5), dict(last_date=123),
                            dict(last_date="invalid"), dict(feature_names="ab"),
                            dict(feature_names=["duplicate", "duplicate"]),
                            dict(feature_names=[None, "x"]), dict(weights={"x": 1}),
                            dict(use_gbdt="false"), dict(calibration_frac=1.0),
                            dict(recency_halflife=0), dict(l2_grid=True), dict(l2_grid=[]),
                            dict(l2_grid={"one": 1.0}), dict(l2_grid=[-1.0])):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    path.write_text(json.dumps({**original, **changed}))
                    MatchPredictor.load(path)

    def test_model_json_requires_an_object_and_all_mandatory_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            for payload in (None, [], "model", 42, {"version": 1}):
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    path.write_text(json.dumps(payload))
                    MatchPredictor.load(path)

    def test_model_rejects_covariance_that_would_produce_false_intervals(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(60, 2))
        model = MatchPredictor(use_gbdt=False).fit(X, (X[:, 0] > 0).astype(float))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            original = json.loads(path.read_text())
            for covariance in ([[1, 2], [2, 1]], [[-1, 0], [0, -1]],
                               [[1, 5], [0, 1]], [[1, 0], [0, -1e-8]],
                               [[float("inf"), 0], [0, 1]], {"x": 1}, [[1]]):
                with self.subTest(covariance=covariance), self.assertRaisesRegex(ValueError, "covariance"):
                    path.write_text(json.dumps({**original, "cov": covariance}))
                    MatchPredictor.load(path)
            # La tolerance aux arrondis et les anciens formats sans covariance restent supportes.
            for covariance in ([[1, 1e-12], [0, 1]], [[1, 0], [0, -1e-12]], None):
                path.write_text(json.dumps({**original, "cov": covariance}))
                restored = MatchPredictor.load(path)
                np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))

    def test_untrained_save_fails_and_loaded_non_gbdt_preserves_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            with self.assertRaises(RuntimeError):
                MatchPredictor(use_gbdt=False).save(path)
            rng = np.random.default_rng(9)
            X = rng.normal(size=(60, 2))
            model = MatchPredictor(use_gbdt=False).fit(X, (X[:, 0] > 0).astype(float))
            model.save(path)
            restored = MatchPredictor.load(path)
            self.assertFalse(restored.use_gbdt)
            np.testing.assert_allclose(restored.predict_proba(X[:5]), model.predict_proba(X[:5]))
            self.assertEqual(restored.predict_interval(X[0]), model.predict_interval(X[0]))

    def test_predict_interval_is_symmetric_ordered_and_narrows_with_more_data(self):
        rng = np.random.default_rng(13)
        X_small = rng.normal(size=(80, 3))
        y_small = (rng.random(80) < 1 / (1 + np.exp(-X_small[:, 0]))).astype(float)
        X_large = rng.normal(size=(800, 3))
        y_large = (rng.random(800) < 1 / (1 + np.exp(-X_large[:, 0]))).astype(float)
        small = MatchPredictor(use_gbdt=False, calibration_frac=0.0).fit(X_small, y_small)
        large = MatchPredictor(use_gbdt=False, calibration_frac=0.0).fit(X_large, y_large)
        x = np.array([1.2, -0.4, 0.3])
        p = small.predict_one(x)
        lo, hi = small.predict_interval(x)
        self.assertLess(lo, p)
        self.assertLess(p, hi)
        lo_rev, hi_rev = small.predict_interval(-x)
        self.assertAlmostEqual(lo + hi_rev, 1.0, places=10)
        self.assertAlmostEqual(hi + lo_rev, 1.0, places=10)
        lo_big, hi_big = large.predict_interval(x)
        self.assertLess(hi_big - lo_big, hi - lo)
        with self.assertRaises(ValueError):
            small.predict_interval(x, z_score=0)

    def test_saved_model_preserves_training_settings_and_identical_refit(self):
        rng = np.random.default_rng(17)
        X = rng.normal(size=(320, 3))
        y = (rng.random(len(X)) < 1 / (1 + np.exp(-X[:, 0]))).astype(float)
        dates = [dt.date(2023, 1, 1) + dt.timedelta(days=3 * i) for i in range(len(X))]
        options = dict(use_gbdt=True, gbdt_weight=.27, calibration_frac=.15,
                       recency_halflife=120, l2_grid=(.5, 5.0), l2=17.0)
        model = MatchPredictor(**options).fit(X, y, dates, ("elo", "form", "pool"))
        self.assertIsNone(model.gbdt)  # GBDT demande, mais historique sous son seuil de 400.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            restored = MatchPredictor.load(path)
            for name in options.keys() - {"l2"}:
                self.assertEqual(getattr(restored, name), getattr(model, name))
            self.assertEqual(restored.l2, model.l2)
            np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))
            expected = MatchPredictor(**options).fit(X, y, dates, model.feature_names)
            restored.fit(X, y, dates)
            np.testing.assert_array_equal(restored.predict_proba(X), expected.predict_proba(X))

    def test_legacy_model_without_training_settings_keeps_previous_defaults(self):
        rng = np.random.default_rng(19)
        X = rng.normal(size=(60, 2))
        model = MatchPredictor(use_gbdt=False).fit(X, (X[:, 0] > 0).astype(float))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            model.save(path)
            payload = json.loads(path.read_text())
            for name in ("use_gbdt", "calibration_frac", "recency_halflife", "l2_grid"):
                payload.pop(name)
            payload["gbdt_weight"] = 0.0
            path.write_text(json.dumps(payload))
            restored = MatchPredictor.load(path)
            self.assertFalse(restored.use_gbdt)
            self.assertEqual(restored.gbdt_weight, 0.0)
            self.assertEqual(restored.calibration_frac, .2)
            self.assertEqual(restored.recency_halflife, 540.0)
            self.assertIsNone(restored.l2_grid)
            np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))

    def test_nonfinite_serialization_preserves_existing_json(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(60, 2))
        model = MatchPredictor(use_gbdt=False).fit(X, (X[:, 0] > 0).astype(float))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            previous = path.read_bytes()
            model.temperature = float("nan")
            with self.assertRaises(ValueError):
                model.save(path)
            self.assertEqual(path.read_bytes(), previous)

    def test_feature_diagnostics_identifies_true_signal_via_wald_z_score(self):
        rng = np.random.default_rng(21)
        X = rng.normal(size=(400, 3))
        y = (rng.random(400) < 1 / (1 + np.exp(-1.5 * X[:, 0]))).astype(float)
        mdl = MatchPredictor(use_gbdt=False, calibration_frac=0.0).fit(
            X, y, feature_names=("signal", "noise_a", "noise_b")
        )
        diag = mdl.logistic.feature_diagnostics()
        self.assertEqual(diag[0]["feature"], "signal")
        self.assertTrue(diag[0]["significant"])
        self.assertGreater(abs(diag[0]["z_score"]), 4.0)
        self.assertIsNotNone(diag[0]["se"])
        cond = mdl.logistic.condition_number()
        self.assertIsNotNone(cond)
        self.assertGreaterEqual(cond, 1.0)

    def test_permutation_importance_ranks_informative_feature_first(self):
        from cs2gp.model import permutation_importance

        rng = np.random.default_rng(29)
        X = rng.normal(size=(300, 3))
        y = (rng.random(300) < 1 / (1 + np.exp(-1.8 * X[:, 0]))).astype(float)
        mdl = MatchPredictor(use_gbdt=False, calibration_frac=0.0).fit(
            X[:200], y[:200], feature_names=("signal", "noise_a", "noise_b")
        )
        imp = permutation_importance(mdl, X[200:], y[200:], repeats=8, seed=0)
        self.assertEqual(imp[0]["feature"], "signal")
        self.assertGreater(imp[0]["mean_delta"], 0.05)
        with self.assertRaises(ValueError):
            permutation_importance(mdl, X[200:], y[200:], metric="bad")


if __name__ == "__main__":
    unittest.main()
