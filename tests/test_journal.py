from concurrent.futures import ThreadPoolExecutor
import csv
import datetime as dt
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cs2gp import journal
from cs2gp.pronostic import Analysis, Fixture, Pick, read_fixtures
from cs2gp.schema import MapResult, group_matches


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "journal.csv"
        self.day = dt.date(2026, 1, 1)
        self.pick = Pick(self.day, "a", "b", 1, "", "vainqueur", "a", .8,
                         "solide", "PRONOSTIC")
        self.results = group_matches([
            MapResult("one", self.day, "a", "b", "mirage", 13, 5, 1, event="first"),
            MapResult("two", self.day, "a", "b", "mirage", 5, 13, 1, event="second"),
        ])

    def test_duplicate_in_batch_is_written_once(self):
        self.assertEqual(journal.record([self.pick, self.pick], self.path), 1)
        self.assertEqual(journal.record([self.pick], self.path), 0)
        self.assertEqual(len(journal.load(self.path)), 1)

    def test_match_id_disambiguates_and_survives_reload(self):
        pick = replace(self.pick, match_id="one", data_through="2025-12-31")
        journal.record([pick], self.path)
        for results in (self.results, self.results[::-1]):
            with self.subTest(order=[m.match_id for m in results]):
                path = self.path.with_name(results[0].match_id + ".csv")
                journal.record([pick], path)
                self.assertEqual(journal.settle(path, results)["pointes"], 1)
                row = journal.load(path)[0]
                self.assertEqual(row["resultat"], "gagne")
                self.assertEqual(row["donnees_jusquau"], "2025-12-31")
                self.assertIsNotNone(dt.datetime.fromisoformat(row["emis_a"]).tzinfo)
                self.assertEqual(journal.scorecard(path).settled[0].pick.match_id, "one")

    def test_unknown_id_never_falls_back_to_similar_match(self):
        journal.record([replace(self.pick, match_id="missing")], self.path)
        self.assertEqual(journal.settle(self.path, self.results)["pointes"], 0)

    def test_ambiguous_legacy_match_stays_pending(self):
        journal.record([self.pick], self.path)
        stats = journal.settle(self.path, self.results)
        self.assertEqual((stats["pointes"], stats["ambigus"], stats["en_attente"]), (0, 1, 1))
        self.assertEqual(journal.load(self.path)[0]["resultat"], "")

    def test_event_disambiguates_legacy_rows(self):
        journal.record([replace(self.pick, event="first")], self.path)
        self.assertEqual(journal.settle(self.path, self.results)["pointes"], 1)
        self.assertEqual(journal.load(self.path)[0]["resultat"], "gagne")

    def test_partial_series_cannot_settle(self):
        journal.record([replace(self.pick, best_of=3)], self.path)
        partial = group_matches([MapResult("partial", self.day, "a", "b", "nuke", 13, 5)])
        self.assertEqual(journal.settle(self.path, partial)["pointes"], 0)

    def test_legacy_header_can_be_appended_without_duplicates(self):
        row = journal._row(self.pick, self.day)
        old_fields = journal.COLUMNS[:17]
        row["pick_id"] = "old-format-id"
        with self.path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=old_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        self.assertEqual(journal.record([self.pick], self.path), 0)
        self.assertEqual(journal.record([replace(self.pick, match_id="two")], self.path), 1)
        self.assertEqual(len(journal.load(self.path)), 2)

    def test_failed_atomic_replace_preserves_previous_journal(self):
        journal.record([self.pick], self.path)
        previous = self.path.read_bytes()
        with patch("cs2gp.journal.os.replace", side_effect=OSError("interruption")):
            with self.assertRaises(OSError):
                journal.record([replace(self.pick, match_id="two")], self.path)
        self.assertEqual(self.path.read_bytes(), previous)

    def test_concurrent_records_preserve_all_distinct_picks(self):
        def write(i):
            return journal.record([self.pick, replace(self.pick, match_id=str(i))], self.path)
        with ThreadPoolExecutor(max_workers=6) as pool:
            added = list(pool.map(write, range(6)))
        rows = journal.load(self.path)
        self.assertEqual(sum(added), 7)
        self.assertEqual(len({r["pick_id"] for r in rows}), 7)

    def test_fixtures_accept_stable_match_id(self):
        path = self.path.with_name("fixtures.csv")
        path.write_text("date,team_a,team_b,match_id\n2026-01-01,a,b,source-42\n")
        self.assertEqual(read_fixtures(path)[0].match_id, "source-42")

    def test_card_archives_abstentions_and_counts_distinct_fixtures(self):
        first = replace(self.pick, match_id="one")
        picks = [first, replace(first, market="total maps -1.5", selection="oui")]
        card = [Analysis(Fixture(self.day, "a", "b", 1, match_id="one"),
                         None, None, None, [], picks),
                Analysis(Fixture(self.day, "c", "d", 1, match_id="skip"),
                         None, None, None, [], [], skipped="historique insuffisant")]
        self.assertEqual(journal.record_card(card, self.path), 3)
        self.assertEqual(journal.record_card(card, self.path), 0)
        row = next(r for r in journal.load(self.path) if r["verdict"] == "ABSTENTION")
        self.assertEqual(row["motif"], "historique insuffisant")
        stats = journal.settle(self.path, self.results)
        self.assertEqual(stats["en_attente"], 0)
        result = journal.scorecard(self.path)
        self.assertEqual((result.n_fixtures, result.n_skipped), (2, 1))
        self.assertTrue(result.coverage_known)
        self.assertEqual(len(result.settled), 2)

    def test_abstention_only_card_produces_a_coverage_report(self):
        card = [Analysis(Fixture(self.day, "a", "b", 1), None, None, None, [], [], skipped="serre")]
        journal.record_card(card, self.path)
        stats = journal.settle(self.path, self.results)
        self.assertEqual((stats["pointes"], stats["en_attente"]), (0, 0))
        record = journal.scorecard(self.path)
        self.assertEqual((record.n_fixtures, record.n_skipped), (1, 1))
        self.assertIn("100%", record.report())

    def test_pending_complete_card_already_reports_coverage(self):
        card = [Analysis(Fixture(self.day, "a", "b", 1), None, None, None, [], [self.pick])]
        journal.record_card(card, self.path)
        record = journal.scorecard(self.path)
        self.assertEqual((record.n_fixtures, record.n_skipped, record.n_pending), (1, 0, 1))
        self.assertTrue(record.coverage_known)

    def test_legacy_coverage_is_reported_as_unknown(self):
        journal.record([replace(self.pick, match_id="one")], self.path)
        journal.settle(self.path, self.results)
        record = journal.scorecard(self.path)
        self.assertFalse(record.coverage_known)
        self.assertIn("Couverture des abstentions indisponible", record.report())

    def test_aliases_match_raw_in_memory_results_in_either_orientation(self):
        pick = replace(self.pick, team_a="mousesports", team_b="g2", selection="mousesports",
                       match_id="alias")
        raw = replace(pick, team_a="mouz", selection="mouz")
        self.assertEqual(journal.pick_id(pick), journal.pick_id(raw))
        for flipped in (False, True):
            with self.subTest(flipped=flipped):
                path = self.path.with_name(f"alias-{flipped}.csv")
                journal.record([pick], path)
                result = MapResult("alias", self.day, "g2" if flipped else "mouz",
                                   "mouz" if flipped else "g2", "mirage",
                                   5 if flipped else 13, 13 if flipped else 5, 1)
                stats = journal.settle(path, group_matches([result]))
                self.assertEqual(stats["pointes"], 1)
                self.assertEqual(journal.load(path)[0]["resultat"], "gagne")

    def test_noop_settle_does_not_rewrite_journal(self):
        journal.record([replace(self.pick, match_id="missing")], self.path)
        with patch("cs2gp.journal._write") as mocked_write:
            stats = journal.settle(self.path, self.results)
        self.assertEqual(stats["pointes"], 0)
        mocked_write.assert_not_called()

    def test_financial_summary_reports_drawdown_and_losing_streak(self):
        p1 = replace(self.pick, match_id="one", odds=2.0, stake=0.02)
        p2 = replace(self.pick, match_id="two", odds=2.0, stake=0.02)
        journal.record([p1, p2], self.path)
        journal.settle(self.path, self.results)
        money = journal.financial_summary(self.path)
        self.assertEqual(money["paris"], 2)
        self.assertEqual(money["max_drawdown_mise_fixe"], 1.0)
        self.assertEqual(money["pire_serie_defaites"], 1)
        self.assertEqual(money["equity_curve_flat"], [0.0, 1.0, 0.0])

    def test_scorecard_and_financial_summary_support_date_filters(self):
        d1 = dt.date(2026, 1, 1)
        d2 = dt.date(2026, 2, 1)
        p1 = replace(self.pick, date=d1, match_id="m1", odds=2.0, stake=0.02)
        p2 = replace(self.pick, date=d2, match_id="m2", odds=2.0, stake=0.02)
        results = group_matches([
            MapResult("m1", d1, "a", "b", "mirage", 13, 5, 1),
            MapResult("m2", d2, "a", "b", "mirage", 5, 13, 1),
        ])
        journal.record([p1, p2], self.path)
        journal.settle(self.path, results)
        jan = journal.scorecard(self.path, since=dt.date(2026, 1, 1), until=dt.date(2026, 1, 31))
        self.assertEqual(len(jan.settled), 1)
        self.assertTrue(jan.settled[0].won)
        feb_money = journal.financial_summary(self.path, since=dt.date(2026, 2, 1))
        self.assertEqual(feb_money["paris"], 1)
        self.assertEqual(feb_money["profit_mise_fixe"], -1.0)
        with self.assertRaises(ValueError):
            journal.scorecard(self.path, since=d2, until=d1)

    def test_financial_history_is_independent_of_import_order(self):
        picks = [replace(self.pick, date=self.day + dt.timedelta(days=i), match_id=f"m{i}",
                         odds=3.0, stake=0.02) for i in range(4)]
        results = group_matches([
            MapResult(pick.match_id, pick.date, "a", "b", "mirage",
                      13 if i == 2 else 5, 5 if i == 2 else 13, 1)
            for i, pick in enumerate(picks)
        ])
        expected_money = expected_risk = None
        for order in ([0, 1, 2, 3], [0, 1, 3, 2], [3, 2, 1, 0]):
            with self.subTest(order=order):
                path = self.path.with_name("".join(map(str, order)) + ".csv")
                journal.record([picks[i] for i in order], path)
                journal.settle(path, results)
                archived = path.read_bytes()
                money = journal.financial_summary(path)
                self.assertEqual(money["equity_curve_flat"], [0.0, -1.0, -2.0, 0.0, -1.0])
                self.assertEqual(money["max_drawdown_mise_fixe"], 2.0)
                self.assertEqual(money["pire_serie_defaites"], 2)
                self.assertEqual([s.pick.match_id for s in journal.scorecard(path).settled],
                                 [p.match_id for p in picks])
                risk = journal.bankroll_risk(path, samples=100, seed=0)
                if expected_money is None:
                    expected_money, expected_risk = money, risk
                self.assertEqual(money, expected_money)
                self.assertEqual(risk, expected_risk)
                self.assertEqual(path.read_bytes(), archived)
                # La lecture brute conserve l'ordre de l'archive.
                self.assertEqual([row["match_id"] for row in journal.load(path)],
                                 [picks[i].match_id for i in order])

    def test_history_orders_same_day_emissions_in_utc_with_legacy_fallback(self):
        rows = []
        for name, issued in (("later", "2026-01-01T10:30:00+01:00"),
                             ("earlier", "2026-01-01T09:00:00Z"),
                             ("tie-b", "2026-01-01T09:00:00+00:00"),
                             ("tie-a", "2026-01-01T09:00:00"), ("legacy", "")):
            row = journal._row(replace(self.pick, match_id=name), self.day)
            row["emis_a"], row["pick_id"] = issued, name
            if name == "legacy":
                del row["emis_a"]
            rows.append(row)
        expected = ["legacy", "earlier", "tie-a", "tie-b", "later"]
        for order in (rows, rows[::-1]):
            with self.subTest(order=[r["pick_id"] for r in order]):
                self.assertEqual([r["pick_id"] for r in journal._filter_by_date(order)], expected)

    def test_bankroll_risk_bootstraps_settled_odds_from_journal(self):
        self.assertIsNone(journal.bankroll_risk(self.path))
        p1 = replace(self.pick, match_id="one", odds=2.0, stake=0.02)
        p2 = replace(self.pick, match_id="two", odds=2.0, stake=0.02)
        journal.record([p1, p2], self.path)
        journal.settle(self.path, self.results)
        risk = journal.bankroll_risk(self.path, samples=100, seed=0)
        self.assertEqual(risk["horizon"], 2)
        self.assertIn("p95_max_drawdown_pct", risk)

    def test_roi_significance_counts_matches_instead_of_correlated_markets(self):
        picks, results = [], []
        for i in range(10):
            day = self.day + dt.timedelta(days=i)
            first = replace(self.pick, date=day, match_id=f"m{i}", best_of=3, odds=2.0)
            picks.extend([first, replace(first, market="handicap -1.5"),
                          replace(first, market="total maps -2.5", selection="oui")])
            # Huit sweeps 2-0 puis deux defaites 1-2 : memes resultats sur les trois marches.
            outcomes = [True, True] if i < 8 else [True, False, False]
            for index, won in enumerate(outcomes):
                results.append(MapResult(first.match_id, day, "a", "b",
                                         ("mirage", "nuke", "inferno")[index],
                                         13 if won else 5, 5 if won else 13, 3, map_index=index))
        journal.record(picks, self.path)
        self.assertEqual(journal.settle(self.path, group_matches(results))["pointes"], 30)
        money = journal.financial_summary(self.path)
        self.assertEqual(money["paris"], 30)
        self.assertEqual(money["matchs"], 10)
        self.assertFalse(money["roi_significant_5pct"])
        self.assertAlmostEqual(money["roi_erreur_type"], 4.0 / 15.0, places=12)

    def test_monthly_summary_groups_calibration_and_pnl_by_month(self):
        d1 = dt.date(2026, 1, 10)
        d2 = dt.date(2026, 2, 15)
        p1 = replace(self.pick, date=d1, match_id="m1", odds=2.0, stake=0.02)
        p2 = replace(self.pick, date=d2, match_id="m2", odds=None, stake=None)
        results = group_matches([
            MapResult("m1", d1, "a", "b", "mirage", 13, 5, 1),
            MapResult("m2", d2, "a", "b", "mirage", 5, 13, 1),
        ])
        journal.record([p1, p2], self.path)
        journal.settle(self.path, results)
        months = journal.monthly_summary(self.path)
        self.assertEqual([m["month"] for m in months], ["2026-01", "2026-02"])
        self.assertEqual(months[0]["wins"], 1)
        self.assertAlmostEqual(months[0]["flat_profit"], 1.0)
        self.assertAlmostEqual(months[0]["flat_roi"], 1.0)
        self.assertEqual(months[1]["wins"], 0)
        self.assertIsNone(months[1]["flat_profit"])
        self.assertEqual(months[1]["priced_bets"], 0)

    def test_export_filtered_csv_respects_dates_and_settled_only(self):
        d1 = dt.date(2026, 1, 10)
        d2 = dt.date(2026, 2, 15)
        p1 = replace(self.pick, date=d1, match_id="m1")
        p2 = replace(self.pick, date=d2, match_id="m2")
        journal.record([p1, p2], self.path)
        journal.settle(self.path, group_matches([MapResult("m1", d1, "a", "b", "mirage", 13, 5, 1)]))
        out_csv = self.path.with_name("export.csv")
        n_settled = journal.export_filtered_csv(self.path, out_csv, settled_only=True)
        self.assertEqual(n_settled, 1)
        self.assertEqual(len(journal.load(out_csv)), 1)
        n_feb = journal.export_filtered_csv(self.path, out_csv, since=dt.date(2026, 2, 1))
        self.assertEqual(n_feb, 1)
        self.assertEqual(journal.load(out_csv)[0]["match_id"], "m2")

    def test_pending_picks_lists_unsettled_actionable_rows(self):
        d1 = dt.date(2026, 1, 10)
        d2 = dt.date(2026, 1, 12)
        p1 = replace(self.pick, date=d1, match_id="m1")
        p2 = replace(self.pick, date=d2, match_id="m2", odds=1.95)
        journal.record([p1, p2], self.path)
        journal.settle(self.path, group_matches([MapResult("m1", d1, "a", "b", "mirage", 13, 5, 1)]))
        pending = journal.pending_picks(self.path)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["match_id"], "m2")
        self.assertAlmostEqual(pending[0]["odds"], 1.95)


if __name__ == "__main__":
    unittest.main()
