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
            self.assertEqual(ratings_payload["summary"]["n_teams"], 8)
            self.assertTrue(ratings_csv.is_file())
            self.assertEqual(len(ratings_csv.read_text(encoding="utf-8").splitlines()), 6)
            teams = [r["team"] for r in ratings_payload["ratings"]]

            # predict --json with odds and --markets
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "predict", teams[0], teams[1], "--data", str(data), "--no-gbdt",
                    "--odds-a", "1.85", "--odds-b", "2.00", "--markets", "--json",
                ]), 0)
            pred_payload = json.loads(out.getvalue())
            self.assertAlmostEqual(pred_payload["p_a"] + pred_payload["p_b"], 1.0, places=10)
            self.assertIn("market", pred_payload)
            self.assertIn("markets", pred_payload)
            self.assertIn("veto_sensitivity", pred_payload)
            self.assertGreater(len(pred_payload["markets"]), 5)
            self.assertGreater(pred_payload["market"]["overround"], 1.0)

            # backtest --json --by-segment
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "backtest", "--data", str(data), "--no-gbdt", "--bootstrap-samples", "0",
                    "--by-segment", "--by-team", "--json",
                ]), 0)
            bt_payload = json.loads(out.getvalue())
            self.assertIn("metrics", bt_payload)
            self.assertIn("calibration", bt_payload)
            self.assertIn("brier_decomposition", bt_payload)
            self.assertIn("sharpness", bt_payload)
            self.assertIn("by_segment", bt_payload)
            self.assertIn("by_team", bt_payload)

            # journal --json
            jpath = Path(tmp) / "journal.csv"
            journal.record([Pick(dt.date(2026, 1, 1), teams[0], teams[1], 3, "",
                                 "vainqueur", teams[0], 0.72, "solide", "PRONOSTIC")], jpath)
            j_csv = Path(tmp) / "journal_export.csv"
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "journal", "--journal", str(jpath), "--by-month",
                    "--export-csv", str(j_csv), "--json",
                ]), 0)
            j_payload = json.loads(out.getvalue())
            self.assertIn("scorecard", j_payload)
            self.assertIn("financial", j_payload)
            self.assertIn("by_month", j_payload)
            self.assertTrue(j_csv.is_file())

            # slate --json + --export-csv
            fpath = Path(tmp) / "fixtures.csv"
            slate_csv = Path(tmp) / "slate.csv"
            fpath.write_text(f"date,team_a,team_b,best_of\n2026-01-10,{teams[0]},{teams[1]},3\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "slate", "--data", str(data), "--no-gbdt", "--fixtures", str(fpath),
                    "--momentum", "0", "--no-calibration", "--export-csv", str(slate_csv), "--json",
                ]), 0)
            slate_payload = json.loads(out.getvalue())
            self.assertEqual(len(slate_payload["analyses"]), 1)
            self.assertIn("fixture", slate_payload["analyses"][0])
            self.assertTrue(slate_csv.is_file())

            # merge --json & teams --json
            merged_csv = Path(tmp) / "merged.csv"
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["merge", str(data), str(data), "--out", str(merged_csv), "--json"]), 0)
            self.assertTrue(merged_csv.is_file())
            merge_payload = json.loads(out.getvalue())
            self.assertGreater(merge_payload["duplicates_skipped"], 0)

            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["teams", "--data", str(data), "--json"]), 0)
            teams_payload = json.loads(out.getvalue())
            self.assertEqual(teams_payload["n_teams"], 8)

            # tipster-backtest --json + --export-csv
            tb_csv = Path(tmp) / "tipster_oos.csv"
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "tipster-backtest", "--data", str(data), "--no-gbdt",
                    "--export-csv", str(tb_csv), "--json",
                ]), 0)
            tb_payload = json.loads(out.getvalue())
            self.assertIn("by_tier", tb_payload)
            self.assertIn("by_market", tb_payload)
            self.assertTrue(tb_csv.is_file())

            # team --json
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["team", teams[0], "--data", str(data), "--json"]), 0)
            team_payload = json.loads(out.getvalue())
            self.assertEqual(team_payload["team"], teams[0])
            self.assertEqual(len(team_payload["map_pool"]), 7)

            # train --bundle + inspect --json + train --json
            bundle = Path(tmp) / "analyzer.cs2gp"
            model_json_path = Path(tmp) / "match.json"
            train_out = io.StringIO()
            with contextlib.redirect_stdout(train_out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "train", "--data", str(data), "--no-gbdt", "--bundle", str(bundle),
                ]), 0)
            self.assertIn("z=", train_out.getvalue())
            train_json_out = io.StringIO()
            with contextlib.redirect_stdout(train_json_out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "train", "--data", str(data), "--no-gbdt", "--out", str(model_json_path), "--json",
                ]), 0)
            train_payload = json.loads(train_json_out.getvalue())
            self.assertEqual(train_payload["kind"], "match_model")
            self.assertIn("condition_number", train_payload)
            self.assertGreater(len(train_payload["features"]), 0)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["inspect", str(bundle), "--json"]), 0)
            inspect_payload = json.loads(out.getvalue())
            self.assertEqual(inspect_payload["version"], 2)
            self.assertTrue(inspect_payload["has_calibration_history"])
            self.assertFalse(inspect_payload["has_gbdt"])

            # diff --json (same bundle -> zero deltas)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["diff", str(bundle), str(bundle), "--json"]), 0)
            diff_payload = json.loads(out.getvalue())
            self.assertEqual(diff_payload["delta_maps"], 0)
            self.assertEqual(diff_payload["delta_matches"], 0)
            self.assertEqual(diff_payload["elo_changes"], {})

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

            # audit --json includes feature correlation report
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["audit", "--data", str(data), "--json"]), 0)
            audit_payload = json.loads(out.getvalue())
            self.assertIn("features", audit_payload)
            self.assertIn("constant_features", audit_payload["features"])

            # history --json + --export-csv
            hist_csv = Path(tmp) / "hist.csv"
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "history", teams[0], "--data", str(data), "--last", "10",
                    "--export-csv", str(hist_csv), "--json",
                ]), 0)
            hist_payload = json.loads(out.getvalue())
            self.assertEqual(hist_payload["team"], teams[0])
            self.assertEqual(len(hist_payload["history"]), 10)
            self.assertTrue(hist_csv.is_file())

            # compare --json
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["compare", teams[0], teams[1], "--data", str(data), "--json"]), 0)
            cmp_payload = json.loads(out.getvalue())
            self.assertEqual(cmp_payload["team_a"]["team"], teams[0])
            self.assertEqual(len(cmp_payload["map_edges"]), 7)

            # movers --json
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["movers", "--data", str(data), "--days", "30", "--min-maps", "1", "--top", "3", "--json"]), 0)
            mv_payload = json.loads(out.getvalue())
            self.assertEqual(len(mv_payload["risers"]), 3)
            self.assertEqual(len(mv_payload["fallers"]), 3)

            # maps --json + --export-csv
            maps_csv = Path(tmp) / "maps.csv"
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["maps", "--data", str(data), "--export-csv", str(maps_csv), "--json"]), 0)
            maps_payload = json.loads(out.getvalue())
            self.assertEqual(len(maps_payload["maps"]), 7)
            self.assertGreater(maps_payload["maps"][0]["played"], 0)
            self.assertTrue(maps_csv.is_file())

            # events --json + --export-csv
            events_csv = Path(tmp) / "events.csv"
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "events", "--data", str(data), "--top", "5", "--export-csv", str(events_csv), "--json",
                ]), 0)
            ev_payload = json.loads(out.getvalue())
            self.assertGreater(len(ev_payload["events"]), 0)
            self.assertLessEqual(len(ev_payload["events"]), 5)
            self.assertTrue(events_csv.is_file())

            # fixtures --json (atomic write)
            gen_fx = Path(tmp) / "gen_fixtures.csv"
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "fixtures", "--data", str(data), "--teams", "4", "--count", "3",
                    "--out", str(gen_fx), "--json",
                ]), 0)
            fx_payload = json.loads(out.getvalue())
            self.assertEqual(fx_payload["n_fixtures"], 3)
            self.assertTrue(gen_fx.is_file())

    def test_public_package_api_exports(self):
        import cs2gp

        for symbol in cs2gp.__all__:
            self.assertTrue(hasattr(cs2gp, symbol))
        with self.assertRaises(AttributeError):
            _ = cs2gp.nonexistent_symbol


if __name__ == "__main__":
    unittest.main()
