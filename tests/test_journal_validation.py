import contextlib
import csv
import datetime as dt
import io
import math
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from cs2gp import journal
from cs2gp.cli import main
from cs2gp.pronostic import Pick
from cs2gp.schema import MapResult, group_matches


class JournalValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "journal.csv"
        self.day = dt.date(2026, 1, 1)
        self.pick = Pick(self.day, "a", "b", 1, "", "vainqueur", "a", .8,
                         "solide", "PRONOSTIC", odds=2.0, stake=.02)

    def test_invalid_pick_rejects_whole_batch_without_changing_archive(self):
        journal.record([self.pick], self.path)
        archived = self.path.read_bytes()
        cases = [dict(probability=p) for p in (-.1, 1.1, math.nan, True)]
        cases += [dict(odds=o) for o in (1.0, math.inf, True)]
        cases += [dict(stake=s) for s in (-.01, 1.1, math.nan)]
        cases += [dict(best_of=2), dict(best_of=True), dict(edge=math.inf),
                  dict(team_a="mouz", team_b="mousesports"), dict(selection="unknown"),
                  dict(team_a=""), dict(event=None), dict(verdict="unknown"),
                  dict(data_through="invalid"), dict(date=dt.datetime(2026, 1, 1))]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                journal.record([replace(self.pick, match_id="new"), replace(self.pick, **changes)], self.path)
            self.assertEqual(self.path.read_bytes(), archived)
        with self.assertRaises(ValueError):
            journal.record([self.pick], self.path, issued=False)
        self.assertEqual(self.path.read_bytes(), archived)

    def test_damaged_values_are_localized_and_never_overwritten(self):
        first = journal._row(replace(self.pick, event="LAN\nParis"), self.day)
        second = journal._row(replace(self.pick, match_id="second"), self.day)
        cases = [("probabilite", "nan"), ("probabilite", "1.2"), ("mise", "-0.01"),
                 ("cote", "1"), ("edge", "inf"), ("best_of", "2"), ("team_a", "b"),
                 ("selection", "unknown"), ("resultat", "annule"), ("date", "invalid"),
                 ("emis_a", "invalid"), ("emis_le", "invalid"), ("carte_complete", "yes"),
                 ("pick_id", ""), ("donnees_jusquau", "invalid")]
        for field, value in cases:
            journal._write(self.path, [first, {**second, field: value}])
            archived = self.path.read_bytes()
            for operation in (lambda: journal.load(self.path),
                              lambda: journal.record([self.pick], self.path),
                              lambda: journal.settle(self.path, []),
                              lambda: journal.monthly_summary(self.path)):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "ligne 4"):
                    operation()
                self.assertEqual(self.path.read_bytes(), archived)

    def test_invalid_journal_cli_returns_error_without_emitting_nan_json(self):
        row = journal._row(self.pick, self.day)
        journal._write(self.path, [{**row, "probabilite": "nan", "resultat": "gagne"}])
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                self.assertRaises(SystemExit) as raised:
            main(["journal", "--journal", str(self.path), "--json"])
        self.assertEqual(out.getvalue(), "")
        self.assertIn("ligne 2", str(raised.exception))
        self.assertIn(str(self.path), str(raised.exception))

    def test_archive_round_trip_preserves_float_precision_and_valid_small_odds(self):
        pick = replace(self.pick, probability=.1234567890123456, odds=math.nextafter(1.0, 2.0),
                       stake=1e-10, edge=-.01234567890123456)
        journal.record([pick], self.path)
        restored = journal._pick_from_row(journal.load(self.path)[0])
        for field in ("probability", "odds", "stake", "edge"):
            self.assertEqual(getattr(restored, field), getattr(pick, field))

    def test_kelly_roi_is_invariant_to_a_tiny_recorded_stake(self):
        pick = replace(self.pick, stake=1e-10, match_id="tiny")
        journal.record([pick], self.path)
        results = group_matches([MapResult("tiny", self.day, "a", "b", "mirage", 13, 5, 1)])
        journal.settle(self.path, results)
        money = journal.financial_summary(self.path)
        self.assertEqual(money["mise_kelly_totale"], pick.stake)
        self.assertEqual(money["roi_kelly"], 1.0)

    def test_legacy_columns_and_empty_optional_metadata_remain_readable(self):
        row = journal._row(replace(self.pick, best_of=3), self.day)
        row["best_of"], row["emis_le"] = "", ""
        with self.path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=journal.COLUMNS[:17], extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        self.assertEqual(journal._pick_from_row(journal.load(self.path)[0]).best_of, 3)
        self.assertEqual(len(journal.pending_picks(self.path)), 1)


if __name__ == "__main__":
    unittest.main()
