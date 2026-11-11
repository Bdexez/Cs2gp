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
                            dict(temperature=0), dict(resid=[1, 2]), dict(feature_names=["one"])):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    path.write_text(json.dumps({**original, **changed}))
                    MatchPredictor.load(path)

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
