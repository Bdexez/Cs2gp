import contextlib
import datetime as dt
import io
import json
from pathlib import Path
import tempfile
import unittest

from cs2gp.audit import audit_history, format_audit
from cs2gp.cli import main
from cs2gp.schema import MapResult, write_csv


class AuditTests(unittest.TestCase):
    def test_audit_detects_incomplete_series_out_of_pool_and_mr15_scores(self):
        day = dt.date(2026, 2, 1)
        maps = [
            MapResult("m1", day, "navi", "g2", "mirage", 13, 9, best_of=3, tier="s", lan=True,
                      map_index=0, roster_a=("a", "b"), roster_b=("c", "d")),
            MapResult("m1", day, "navi", "g2", "nuke", 16, 14, best_of=3, tier="s", lan=True,
                      map_index=1, roster_a=("a", "b"), roster_b=("c", "d")),
            # Incomplete BO3 (only 1 map), out-of-pool map, MR15-looking score, no rosters
            MapResult("m2", day + dt.timedelta(days=5), "navi", "g2", "vertigo", 16, 8,
                      best_of=3, tier="a", lan=False, map_index=0),
        ]
        report = audit_history(maps)
        self.assertEqual(report["n_maps"], 3)
        self.assertEqual(report["n_matches"], 2)
        self.assertEqual(report["n_complete_matches"], 1)
        self.assertEqual(report["n_incomplete_matches"], 1)
        self.assertEqual(report["span_days"], 5)
        self.assertEqual(report["out_of_pool_maps"], {"vertigo": 1})
        self.assertAlmostEqual(report["roster_coverage"], 2 / 3)
        self.assertEqual(report["round_regimes"]["mr12_regulation"], 1)
        self.assertEqual(report["round_regimes"]["mr12_overtime"], 1)
        self.assertEqual(report["round_regimes"]["mr15_suspect"], 1)
        self.assertEqual(report["round_regimes"]["other"], 0)
        self.assertGreaterEqual(len(report["warnings"]), 3)

        text = format_audit(report)
        self.assertIn("Audit historique", text)
        self.assertIn("vertigo", text)
        self.assertIn("MR15", text)

    def test_cli_audit_supports_text_and_json(self):
        day = dt.date(2026, 3, 1)
        maps = [MapResult("m1", day, "alpha", "beta", "mirage", 13, 7, best_of=1)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            write_csv(str(path), maps)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["audit", "--data", str(path), "--json"]), 0)
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["n_maps"], 1)
            self.assertEqual(payload["n_matches"], 1)
            self.assertEqual(payload["round_regimes"]["mr12_regulation"], 1)

    def test_audit_detects_calendar_gaps_warmup_and_same_day_collisions(self):
        d1 = dt.date(2026, 1, 1)
        d2 = dt.date(2026, 3, 15)  # 73-day gap
        maps = [
            MapResult("m1", d1, "navi", "g2", "mirage", 13, 8, best_of=1),
            MapResult("m2", d1, "g2", "navi", "nuke", 13, 10, best_of=1),
            MapResult("m3", d2, "navi", "faze", "inferno", 13, 7, best_of=1),
        ]
        report = audit_history(maps, warmup_maps=3)
        self.assertEqual(report["max_calendar_gap_days"], 73)
        self.assertEqual(report["same_day_pair_collisions"], 1)
        self.assertEqual(report["teams_under_warmup"], 2)  # g2 (2 maps), faze (1 map); navi has 3
        self.assertTrue(any("trou calendaire" in w for w in report["warnings"]))
        self.assertTrue(any("meme jour" in w for w in report["warnings"]))
        self.assertEqual(report["matches_by_year"], {"2026": 3})
        self.assertEqual(report["map_balance_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
