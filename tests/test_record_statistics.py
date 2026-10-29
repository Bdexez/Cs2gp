import datetime as dt
from dataclasses import replace
import unittest

from cs2gp.pronostic import Pick, SettledPick, TipsterRecord


class RecordStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.pick = Pick(dt.date(2026, 1, 1), "a", "b", 3, "", "vainqueur", "a",
                         .8, "solide", "PRONOSTIC", match_id="same")

    def test_many_picks_on_one_match_are_not_many_independent_observations(self):
        record = TipsterRecord([SettledPick(self.pick, False)]*100, 1, 0)
        stat = record.by(lambda p: p.tier)["solide"]
        self.assertEqual(stat["n"], 100)
        self.assertEqual(stat["n_matchs"], 1)
        self.assertAlmostEqual(stat["erreur_type"], .4)
        self.assertAlmostEqual(stat["brier"], 0.64, places=10)
        self.assertGreater(stat["log_loss"], 1.0)
        self.assertFalse(stat["significatif"])
        self.assertIn("n trop faible", record.report())

    def test_duplicate_markets_do_not_shrink_uncertainty(self):
        picks = [SettledPick(replace(self.pick, match_id=str(i)), False) for i in range(30)]
        single = TipsterRecord(picks, 30, 0).by(lambda p: p.tier)["solide"]
        doubled = TipsterRecord(picks + picks, 30, 0).by(lambda p: p.tier)["solide"]
        self.assertAlmostEqual(single["erreur_type"], doubled["erreur_type"])
        self.assertTrue(single["significatif"])
        self.assertEqual(single["n_matchs"], doubled["n_matchs"])

    def test_scoring_computes_proper_scoring_rules_on_settled_picks(self):
        self.assertIsNone(TipsterRecord([], 0, 0).scoring())
        settled = [
            SettledPick(replace(self.pick, probability=0.8, match_id="m1"), True),
            SettledPick(replace(self.pick, probability=0.8, match_id="m2"), True),
            SettledPick(replace(self.pick, probability=0.8, match_id="m3"), True),
            SettledPick(replace(self.pick, probability=0.8, match_id="m4"), True),
            SettledPick(replace(self.pick, probability=0.8, match_id="m5"), False),
        ]
        rec = TipsterRecord(settled, 5, 0)
        sc = rec.scoring()
        self.assertEqual(sc["n"], 5)
        self.assertAlmostEqual(sc["mean_predicted"], 0.8)
        self.assertAlmostEqual(sc["mean_observed"], 0.8)
        self.assertAlmostEqual(sc["ece"], 0.0, places=10)
        self.assertAlmostEqual(sc["brier"], 0.16, places=10)
        self.assertIn("spiegelhalter", sc)
        self.assertIn("brier_decomposition", sc)
        self.assertIn("Qualite probabiliste", rec.report())
        self.assertEqual(rec.to_dict()["scoring"]["n"], 5)


if __name__ == "__main__":
    unittest.main()
