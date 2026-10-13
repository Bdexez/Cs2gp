import contextlib
import datetime as dt
import io
import json
from pathlib import Path
import tempfile
import unittest

from cs2gp import journal
from cs2gp.cli import main
from cs2gp.pronostic import Pick
from cs2gp.schema import write_csv
from cs2gp.sources import simulate_history


class CliJsonTests(unittest.TestCase):
    def test_ratings_predict_backtest_and_journal_emit_valid_json(self):
        rows = simulate_history(n_teams=8, years=1.0, seed=42, start=dt.date(2025, 1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "history.csv"
            write_csv(str(data), rows)

            # ratings --json + --export-csv
            ratings_csv = Path(tmp) / "ratings.csv"
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "ratings", "--data", str(data), "--top", "5", "--min-maps", "1",
                    "--export-csv", str(ratings_csv), "--json",
                ]), 0)
            ratings_payload = json.loads(out.getvalue())
            self.assertEqual(len(ratings_payload["ratings"]), 5)
            self.assertTrue(ratings_csv.is_file())
            self.assertEqual(len(ratings_csv.read_text(encoding="utf-8").splitlines()), 6)
            teams = [r["team"] for r in ratings_payload["ratings"]]

            # predict --json with odds
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "predict", teams[0], teams[1], "--data", str(data), "--no-gbdt",
                    "--odds-a", "1.85", "--odds-b", "2.00", "--json",
                ]), 0)
            pred_payload = json.loads(out.getvalue())
            self.assertAlmostEqual(pred_payload["p_a"] + pred_payload["p_b"], 1.0, places=10)
            self.assertIn("market", pred_payload)
            self.assertGreater(pred_payload["market"]["overround"], 1.0)

            # backtest --json --by-segment
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "backtest", "--data", str(data), "--no-gbdt", "--bootstrap-samples", "0",
                    "--by-segment", "--json",
                ]), 0)
            bt_payload = json.loads(out.getvalue())
            self.assertIn("metrics", bt_payload)
            self.assertIn("calibration", bt_payload)
            self.assertIn("brier_decomposition", bt_payload)
            self.assertIn("by_segment", bt_payload)

            # journal --json
            jpath = Path(tmp) / "journal.csv"
            journal.record([Pick(dt.date(2026, 1, 1), teams[0], teams[1], 3, "",
                                 "vainqueur", teams[0], 0.72, "solide", "PRONOSTIC")], jpath)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["journal", "--journal", str(jpath), "--json"]), 0)
            j_payload = json.loads(out.getvalue())
            self.assertIn("scorecard", j_payload)
            self.assertIn("financial", j_payload)

            # slate --json
            fpath = Path(tmp) / "fixtures.csv"
            fpath.write_text(f"date,team_a,team_b,best_of\n2026-01-10,{teams[0]},{teams[1]},3\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "slate", "--data", str(data), "--no-gbdt", "--fixtures", str(fpath),
                    "--momentum", "0", "--no-calibration", "--json",
                ]), 0)
            slate_payload = json.loads(out.getvalue())
            self.assertEqual(len(slate_payload["analyses"]), 1)
            self.assertIn("fixture", slate_payload["analyses"][0])

            # tipster-backtest --json
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "tipster-backtest", "--data", str(data), "--no-gbdt", "--json",
                ]), 0)
            tb_payload = json.loads(out.getvalue())
            self.assertIn("by_tier", tb_payload)
            self.assertIn("by_market", tb_payload)

            # team --json
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["team", teams[0], "--data", str(data), "--json"]), 0)
            team_payload = json.loads(out.getvalue())
            self.assertEqual(team_payload["team"], teams[0])
            self.assertEqual(len(team_payload["map_pool"]), 7)

            # train --bundle + inspect --json
            bundle = Path(tmp) / "analyzer.cs2gp"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "train", "--data", str(data), "--no-gbdt", "--bundle", str(bundle),
                ]), 0)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["inspect", str(bundle), "--json"]), 0)
            inspect_payload = json.loads(out.getvalue())
            self.assertEqual(inspect_payload["version"], 2)
            self.assertTrue(inspect_payload["has_calibration_history"])
            self.assertFalse(inspect_payload["has_gbdt"])

            # h2h --json
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["h2h", teams[0], teams[1], "--data", str(data), "--json"]), 0)
            h2h_payload = json.loads(out.getvalue())
            self.assertEqual(h2h_payload["team_a"], teams[0])
            self.assertEqual(h2h_payload["team_b"], teams[1])
            self.assertEqual(
                h2h_payload["series_a_wins"] + h2h_payload["series_b_wins"],
                h2h_payload["series_total"],
            )


if __name__ == "__main__":
    unittest.main()
