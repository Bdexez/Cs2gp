import contextlib
import copy
import datetime as dt
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cs2gp.cli import main
from cs2gp.features import FeatureExtractor
from cs2gp.model import MatchPredictor
from cs2gp.predict import Analyzer
from cs2gp.schema import MapResult, write_csv


class ScoutingHistoryTests(unittest.TestCase):
    def setUp(self):
        self.day = dt.date(2026, 1, 1)
        self.cutoff = self.day + dt.timedelta(days=1)
        self.rows = [
            MapResult("m0", self.day, "navi", "g2", "mirage", 13, 5, 1, roster_a=("p1", "p2")),
            MapResult("m1", self.cutoff, "navi", "g2", "mirage", 5, 13, 1, roster_a=("p3", "p4")),
            MapResult("m2", self.cutoff, "navi", "g2", "nuke", 5, 13, 1),
            MapResult("m3", self.cutoff + dt.timedelta(days=1), "navi", "g2", "inferno", 5, 13, 1),
        ]
        self.an = Analyzer(warmup_maps=0, versioned_pool=True).load_maps(self.rows)

    def reports(self, analyzer, on):
        return (analyzer.team_profile("navi", on=on), analyzer.head_to_head("navi", "g2", on=on),
                analyzer.compare_teams("navi", "g2", on=on))

    def test_historical_reports_equal_a_prefix_and_ignore_target_day_and_future_results(self):
        past = Analyzer(warmup_maps=0, versioned_pool=True).load_maps(self.rows[:1])
        changed = [replace(row, rounds_a=row.rounds_b, rounds_b=row.rounds_a,
                           roster_a=("other",)) if row.date >= self.cutoff else row for row in self.rows]
        modified = Analyzer(warmup_maps=0, versioned_pool=True).load_maps(changed)
        with patch.object(MatchPredictor, "fit", side_effect=AssertionError("aucun entrainement")):
            expected = self.reports(past, self.cutoff)
            self.assertEqual(self.reports(self.an, self.cutoff), expected)
            self.assertEqual(self.reports(modified, self.cutoff), expected)
        profile, h2h, comparison = expected
        self.assertEqual(profile["maps_played"], 1)
        self.assertEqual(profile["last_match_date"], self.day.isoformat())
        self.assertEqual(h2h["series_total"], 1)
        self.assertGreater(h2h["weighted_h2h_score"], 0.0)
        self.assertEqual({row["map"] for row in comparison["map_edges"]},
                         set(past.fx.pool_on(self.cutoff)))

    def test_default_reports_keep_the_complete_latest_state(self):
        profile, h2h, comparison = self.reports(self.an, None)
        self.assertEqual(profile["maps_played"], 4)
        self.assertEqual(h2h["series_total"], 4)
        self.assertEqual(comparison["team_a"], profile)
        self.assertEqual(comparison["h2h"], h2h)
        self.assertEqual(self.an.team_profile("navi", on=self.day + dt.timedelta(days=10))["maps_played"], 4)

    def test_historical_alias_resolution_uses_only_previously_known_teams(self):
        extra = MapResult("later", self.day + dt.timedelta(days=3), "navistar", "other", "mirage", 13, 5, 1)
        self.an.load_maps([*self.rows, extra])
        self.assertEqual(self.an.team_profile("nav", on=self.cutoff)["team"], "navi")
        with self.assertRaisesRegex(ValueError, "ambigu"):
            self.an.resolve_team("nav")
        with self.assertRaisesRegex(ValueError, "inconnue"):
            self.an.team_profile("other", on=self.cutoff)
        with self.assertRaisesRegex(ValueError, "inconnue"):
            Analyzer().resolve_team("unknown")

    def test_cached_reads_are_pure_and_successful_import_invalidates_them(self):
        state = self.an._scouting_state(self.cutoff)
        live = self.an.fx
        teams, recent = copy.deepcopy(live.elo.teams), copy.deepcopy(live.recent)
        regression = live.elo._xy, live.elo._xx
        with patch.object(FeatureExtractor, "observe_match", side_effect=AssertionError("recalcul")):
            self.reports(self.an, self.cutoff)
        self.assertIs(self.an.fx, live)
        self.assertEqual(live.elo.teams, teams)
        self.assertEqual(live.recent, recent)
        self.assertEqual((live.elo._xy, live.elo._xx), regression)
        with self.assertRaises(ValueError):
            self.an.load_maps([self.rows[0], self.rows[0]])
        self.assertIs(self.an._scouting_state(self.cutoff), state)
        self.an.load_maps(self.rows[:-1])
        self.assertEqual(self.an._scouting_snapshots, {})
        self.assertIsNot(self.an._scouting_state(self.cutoff), state)

    def test_cache_is_bounded_and_empty_or_invalid_cutoffs_are_rejected(self):
        rows = [replace(self.rows[0], match_id=f"m{i}", date=self.day + dt.timedelta(days=i)) for i in range(15)]
        self.an.load_maps(rows)
        for i in range(1, 15):
            self.an.team_profile("navi", on=self.day + dt.timedelta(days=i))
        self.assertEqual(len(self.an._scouting_snapshots), 8)
        self.assertNotIn(self.cutoff, self.an._scouting_snapshots)
        for on in (self.day, self.day - dt.timedelta(days=1)):
            with self.assertRaisesRegex(ValueError, "aucun resultat anterieur"):
                self.an.team_profile("navi", on=on)
        for on in (dt.datetime(2026, 1, 2), "2026-01-02"):
            with self.assertRaisesRegex(ValueError, "date calendaire"):
                self.an.team_profile("navi", on=on)

    def test_cli_scouting_json_uses_only_results_before_explicit_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            write_csv(path, self.rows)
            for args in (["team", "navi"], ["h2h", "navi", "g2"], ["compare", "navi", "g2"]):
                out = io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main([*args, "--data", str(path), "--date", self.cutoff.isoformat(), "--json"]), 0)
                payload = json.loads(out.getvalue())
                if args[0] == "team":
                    self.assertEqual(payload["maps_played"], 1)
                elif args[0] == "h2h":
                    self.assertEqual(payload["maps_total"], 1)
                    self.assertGreater(payload["weighted_h2h_score"], 0)
                else:
                    self.assertEqual(payload["team_a"]["maps_played"], 1)
                    self.assertEqual(payload["h2h"]["maps_total"], 1)


if __name__ == "__main__":
    unittest.main()
