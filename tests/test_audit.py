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

    def test_audit_flags_near_duplicate_team_names(self):
        day = dt.date(2026, 4, 1)
        maps = [
            MapResult("m1", day, "vitality", "eternalfire", "mirage", 13, 8, best_of=1),
            MapResult("m2", day, "vitality", "eternal-fire", "nuke", 13, 9, best_of=1),
        ]
        report = audit_history(maps)
        self.assertEqual(len(report["similar_team_pairs"]), 1)
        self.assertEqual(report["similar_team_pairs"][0][:2], ["eternal-fire", "eternalfire"])
        self.assertTrue(any("alias" in w for w in report["warnings"]))

    def test_audit_fixtures_flags_unknown_teams_and_missing_match_id_collisions(self):
        from cs2gp.audit import audit_fixtures
        from cs2gp.predict import Analyzer
        from cs2gp.pronostic import Fixture

        day = dt.date(2026, 4, 1)
        an = Analyzer(warmup_maps=0).load_maps([
            MapResult("m1", day, "navi", "g2", "mirage", 13, 8, best_of=1),
        ])
        fixtures = [
            Fixture(day + dt.timedelta(days=1), "navi", "g2"),
            Fixture(day + dt.timedelta(days=1), "g2", "navi"),
            Fixture(day + dt.timedelta(days=1), "navi", "mystery", odds_a=1.8, odds_b=2.0),
        ]
        res = audit_fixtures(fixtures, an)
        self.assertFalse(res["ok"])
        self.assertEqual(res["unknown_teams"], ["mystery"])
        self.assertEqual(res["missing_match_id_collisions"], 1)
        self.assertEqual(res["with_odds"], 1)
        self.assertGreaterEqual(len(res["thin_or_stale"]), 2)

        with tempfile.TemporaryDirectory() as tmp:
            hpath = Path(tmp) / "history.csv"
            fpath = Path(tmp) / "fixtures.csv"
            write_csv(str(hpath), [MapResult("m1", day, "navi", "g2", "mirage", 13, 8, best_of=1)])
            fpath.write_text("date,team_a,team_b\n2026-04-02,navi,mystery\n", encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["audit", "--data", str(hpath), "--fixtures", str(fpath), "--json"]), 0)
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["fixtures"]["unknown_teams"], ["mystery"])

    def test_audit_fixtures_detects_incomplete_odds_and_arbitrage(self):
        from cs2gp.audit import audit_fixtures
        from cs2gp.predict import Analyzer
        from cs2gp.pronostic import Fixture

        day = dt.date(2026, 4, 1)
        an = Analyzer(warmup_maps=0).load_maps([
            MapResult("m1", day, "navi", "g2", "mirage", 13, 8, best_of=1),
        ])
        fixtures = [
            Fixture(day + dt.timedelta(days=1), "navi", "g2", match_id="f1", odds_a=1.80),
            Fixture(day + dt.timedelta(days=1), "navi", "g2", match_id="f2", odds_a=2.10, odds_b=2.10),
        ]
        res = audit_fixtures(fixtures, an)
        self.assertFalse(res["ok"])
        self.assertEqual(len(res["incomplete_odds_pairs"]), 1)
        self.assertEqual(len(res["arbitrage_fixtures"]), 1)

    def test_audit_detects_team_a_winner_orientation_bias(self):
        day = dt.date(2026, 1, 1)
        maps = [
            MapResult(f"m{i}", day + dt.timedelta(days=i), "navi", "g2", "mirage", 13, 8, best_of=1)
            for i in range(35)
        ]
        report = audit_history(maps)
        self.assertEqual(report["team_a_win_rate"], 1.0)
        self.assertTrue(any("orientation biaisee" in w for w in report["warnings"]))

    def test_audit_fixtures_warns_on_past_dates(self):
        from cs2gp.audit import audit_fixtures
        from cs2gp.predict import Analyzer
        from cs2gp.pronostic import Fixture

        day = dt.date(2026, 4, 10)
        an = Analyzer(warmup_maps=0).load_maps([
            MapResult("m1", day, "navi", "g2", "mirage", 13, 8, best_of=1),
        ])
        res = audit_fixtures([Fixture(day - dt.timedelta(days=5), "navi", "g2", match_id="old")], an)
        self.assertEqual(len(res["past_fixtures"]), 1)
        self.assertTrue(any("avant le dernier match connu" in w for w in res["warnings"]))

    def fixture_history(self, count=1):
        from cs2gp.predict import Analyzer

        day = dt.date(2026, 1, 1)
        maps = [MapResult(f"m{i}", day + dt.timedelta(days=i), "navi", "g2", "mirage", 13, 5, 1)
                for i in range(count)]
        return day, Analyzer(warmup_maps=0).load_maps(maps)

    def test_fixture_audit_rejects_duplicate_ids_and_allows_distinct_ids(self):
        from cs2gp.audit import audit_fixtures
        from cs2gp.pronostic import Fixture

        day, an = self.fixture_history()
        future = day + dt.timedelta(days=1)
        duplicate = [Fixture(future, "navi", "g2", match_id=" shared "),
                     Fixture(future + dt.timedelta(days=1), "g2", "navi", match_id="shared")]
        report = audit_fixtures(duplicate, an)
        self.assertFalse(report["ok"])
        self.assertEqual(report["duplicate_match_ids"], ["shared"])
        distinct = [Fixture(future, "navi", "g2", match_id="one"),
                    Fixture(future, "g2", "navi", match_id="two")]
        self.assertTrue(audit_fixtures(distinct, an)["ok"])

    def test_fixture_audit_detects_mixed_id_collisions_after_alias_resolution(self):
        from cs2gp.audit import audit_fixtures
        from cs2gp.pronostic import Fixture

        day, an = self.fixture_history()
        future = day + dt.timedelta(days=1)
        fixtures = [Fixture(future, "nav", "g2"), Fixture(future, "g2", "navi", match_id="known")]
        report = audit_fixtures(fixtures, an)
        self.assertFalse(report["ok"])
        self.assertEqual(report["unknown_teams"], [])
        self.assertEqual(report["missing_match_id_collisions"], 1)
        other_day = [fixtures[0], Fixture(future + dt.timedelta(days=1), "g2", "navi")]
        self.assertEqual(audit_fixtures(other_day, an)["missing_match_id_collisions"], 0)

    def test_fixture_audit_uses_only_prior_history_for_confidence_and_team_resolution(self):
        from cs2gp.audit import audit_fixtures
        from cs2gp.predict import Analyzer
        from cs2gp.pronostic import Fixture

        day, an = self.fixture_history(30)
        historical = Fixture(day + dt.timedelta(days=1), "navi", "g2", match_id="old")
        report = audit_fixtures([historical], an)
        self.assertEqual(report["thin_or_stale"], [historical.label])
        missing = Fixture(day, "navi", "g2")
        unavailable = audit_fixtures([missing], an)
        self.assertFalse(unavailable["ok"])
        self.assertEqual(unavailable["no_prior_history"], [missing.label])
        extra = MapResult("later", day + dt.timedelta(days=31), "newteam", "navi", "mirage", 13, 5, 1)
        an.load_maps([*an.maps, extra])
        unknown = audit_fixtures([Fixture(historical.date, "newteam", "navi")], an)
        self.assertEqual(unknown["unknown_teams"], ["newteam"])
        self.assertFalse(unknown["ok"])
        empty = audit_fixtures([historical], Analyzer().load_maps([]))
        self.assertFalse(empty["ok"])
        self.assertEqual(empty["no_prior_history"], [historical.label])

    def test_cli_fixture_audit_json_exposes_duplicate_identifiers(self):
        day, an = self.fixture_history()
        with tempfile.TemporaryDirectory() as tmp:
            data, fixtures = Path(tmp) / "history.csv", Path(tmp) / "fixtures.csv"
            write_csv(data, an.maps)
            fixtures.write_text("date,team_a,team_b,match_id\n2026-01-02,navi,g2,same\n"
                                "2026-01-03,g2,navi,same\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["audit", "--data", str(data), "--fixtures", str(fixtures), "--json"]), 0)
            report = json.loads(out.getvalue())["fixtures"]
            self.assertEqual(report["duplicate_match_ids"], ["same"])
            self.assertFalse(report["ok"])


if __name__ == "__main__":
    unittest.main()
