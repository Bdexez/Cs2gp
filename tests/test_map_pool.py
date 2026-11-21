import datetime as dt
import unittest

from cs2gp.features import FeatureExtractor, build_datasets
from cs2gp.schema import ACTIVE_DUTY, MapResult, active_duty_on, group_matches


class MapPoolVersioningTests(unittest.TestCase):
    def test_default_pool_stays_constant_for_backward_compatibility(self):
        self.assertEqual(active_duty_on(dt.date(2024, 1, 1)), ACTIVE_DUTY)
        self.assertEqual(active_duty_on(dt.date(2025, 8, 1)), ACTIVE_DUTY)
        self.assertEqual(active_duty_on(None, versioned=True), ACTIVE_DUTY)

    def test_versioned_pool_tracks_historical_rotations(self):
        pool_2024_q1 = active_duty_on(dt.date(2024, 2, 1), versioned=True)
        self.assertEqual(len(pool_2024_q1), 7)
        self.assertIn("overpass", pool_2024_q1)
        self.assertIn("vertigo", pool_2024_q1)
        self.assertNotIn("dust2", pool_2024_q1)

        pool_2024_q3 = active_duty_on(dt.date(2024, 6, 1), versioned=True)
        self.assertIn("dust2", pool_2024_q3)
        self.assertNotIn("overpass", pool_2024_q3)
        self.assertIn("vertigo", pool_2024_q3)

        pool_2025_q1 = active_duty_on(dt.date(2025, 2, 1), versioned=True)
        self.assertIn("train", pool_2025_q1)
        self.assertNotIn("vertigo", pool_2025_q1)

        pool_2025_q3 = active_duty_on(dt.date(2025, 8, 1), versioned=True)
        self.assertIn("overpass", pool_2025_q3)
        self.assertNotIn("anubis", pool_2025_q3)

    def test_feature_extractor_and_build_datasets_respect_versioned_pool(self):
        day = dt.date(2024, 2, 1)
        fx = FeatureExtractor(versioned_pool=True)
        edges = fx.map_pool_edges("navi", "g2", day)
        self.assertEqual(set(edges), set(active_duty_on(day, versioned=True)))
        matches = group_matches([
            MapResult("m1", day, "navi", "g2", "overpass", 13, 8, best_of=1),
            MapResult("m2", dt.date(2025, 2, 1), "navi", "g2", "train", 13, 9, best_of=1),
        ])
        mds, _, _ = build_datasets(matches, warmup_maps=0, versioned_pool=True)
        self.assertEqual(mds.pool_X.shape, (2, 7, len(FeatureExtractor.MAP_FEATURES)))
        self.assertTrue(mds.versioned_pool)

    def test_analyzer_predict_uses_versioned_pool_for_target_date(self):
        from cs2gp.predict import Analyzer
        from cs2gp.sources import simulate_history

        rows = simulate_history(n_teams=8, years=1.0, seed=5, start=dt.date(2023, 10, 1))
        an = Analyzer(warmup_maps=2, versioned_pool=True).load_maps(rows).train(use_gbdt=False)
        teams = an.known_teams()
        pred_2024 = an.predict(teams[0], teams[1], on=dt.date(2024, 11, 1))
        self.assertEqual({e.map_name for e in pred_2024.map_edges},
                         set(active_duty_on(dt.date(2024, 11, 1), versioned=True)))
        pred_2025 = an.predict(teams[0], teams[1], on=dt.date(2025, 8, 1))
        self.assertEqual({e.map_name for e in pred_2025.map_edges},
                         set(active_duty_on(dt.date(2025, 8, 1), versioned=True)))


if __name__ == "__main__":
    unittest.main()
