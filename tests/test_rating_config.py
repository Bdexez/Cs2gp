import unittest

from cs2gp.ratings import EloConfig


class RatingConfigTests(unittest.TestCase):
    def test_invalid_configurations_fail_before_rating_updates(self):
        for values in (dict(idle_halflife_days=0), dict(rd_growth_days=-1),
                       dict(rd_reference=0), dict(rd_min=301), dict(rd_initial=float("nan")),
                       dict(k_base=-1), dict(map_prior_weight=-.1), dict(roster_shock=1.1),
                       dict(lan_bonus=float("inf")), dict(lan_bonus=True),
                       dict(use_mov="false"), dict(fast_multiplier=True)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                EloConfig.from_dict(values)

    def test_disabled_updates_and_fast_horizon_are_valid(self):
        cfg = EloConfig(k_base=0, k_map=0, fast_multiplier=0, map_prior_weight=0,
                        use_uncertainty=False, lan_bonus=-1)
        self.assertEqual(EloConfig.from_dict(cfg.to_dict()), cfg)

    def test_non_object_json_config_is_rejected_cleanly(self):
        from pathlib import Path
        import tempfile

        with self.assertRaisesRegex(ValueError, "objet JSON"):
            EloConfig.from_dict(["not", "a", "dict"])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "objet JSON"):
                EloConfig.load(p)

    def test_save_creates_parent_directories_and_roundtrips_atomically(self):
        from pathlib import Path
        import tempfile

        cfg = EloConfig(k_base=14.0, k_map=22.0)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nested" / "dir" / "elo.json"
            cfg.save(p)
            self.assertEqual(EloConfig.load(p), cfg)


if __name__ == "__main__":
    unittest.main()
