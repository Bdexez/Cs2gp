import datetime as dt
import unittest

import numpy as np

from cs2gp.model import MatchPredictor, select_l2


class L2SelectionTests(unittest.TestCase):
    def test_select_l2_prefers_stronger_regularization_on_noisy_high_dim_data(self):
        rng = np.random.default_rng(7)
        n = 160
        X = rng.normal(size=(n, 35))
        # Weak true signal in col 0 only; 34 pure noise features -> small l2 overfits badly
        p_true = 1.0 / (1.0 + np.exp(-0.3 * X[:, 0]))
        y = (rng.random(n) < p_true).astype(float)
        dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(n)]

        res = select_l2(X, y, dates, grid=(0.01, 2.0, 40.0, 200.0), val_frac=0.35)
        self.assertGreaterEqual(res["best_l2"], 40.0)
        self.assertEqual(len(res["scores"]), 4)
        self.assertEqual(res["n_train"] + res["n_val"], n)

        pred = MatchPredictor(use_gbdt=False, l2_grid=(0.01, 2.0, 40.0, 200.0)).fit(X, y, dates)
        self.assertIn(pred.l2, (0.01, 2.0, 40.0, 200.0))
        self.assertGreaterEqual(pred.l2, 2.0)

    def test_select_l2_rejects_invalid_grid_and_val_frac(self):
        X = np.ones((60, 2))
        y = np.tile([1.0, 0.0], 30)
        with self.assertRaises(ValueError):
            select_l2(X, y, grid=())
        with self.assertRaises(ValueError):
            select_l2(X, y, grid=(-1.0,))
        with self.assertRaises(ValueError):
            select_l2(X, y, val_frac=1.0)
        with self.assertRaises(ValueError):
            MatchPredictor(use_gbdt=False, l2_grid=())


if __name__ == "__main__":
    unittest.main()
