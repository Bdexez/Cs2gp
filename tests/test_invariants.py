import datetime as dt
import unittest

from cs2gp.markets import build_markets, project_veto
from cs2gp.model import series_win_probability
from cs2gp.odds import devig_power, kelly_fraction
from cs2gp.predict import Analyzer
from cs2gp.pronostic import shape_series
from cs2gp.ratings import expected_score
from cs2gp.sources import simulate_history


class GlobalInvariantsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = simulate_history(n_teams=8, years=1.0, seed=31, start=dt.date(2025, 1, 1))
        cls.an = Analyzer(warmup_maps=4).load_maps(rows).train(use_gbdt=False)
        cls.on = max(m.date for m in cls.an.matches) + dt.timedelta(days=1)

    def test_prediction_is_exactly_antisymmetric_across_formats_venues_and_momentum(self):
        a, b = self.an.known_teams()[:2]
        for momentum in (0.0, 0.15):
            self.an.momentum = momentum
            for bo in (1, 3, 5):
                for lan in (False, True):
                    with self.subTest(bo=bo, lan=lan, momentum=momentum):
                        p_ab = self.an.predict(a, b, best_of=bo, lan=lan, on=self.on)
                        p_ba = self.an.predict(b, a, best_of=bo, lan=lan, on=self.on)
                        self.assertAlmostEqual(p_ab.p_a + p_ba.p_a, 1.0, places=10)
                        self.assertAlmostEqual(
                            p_ab.p_a_match_model + p_ba.p_a_match_model, 1.0, places=10
                        )
                        self.assertAlmostEqual(
                            p_ab.p_a_from_maps + p_ba.p_a_from_maps, 1.0, places=10
                        )
                        lo_ab, hi_ab = p_ab.p_a_interval
                        lo_ba, hi_ba = p_ba.p_a_interval
                        self.assertAlmostEqual(lo_ab + hi_ba, 1.0, places=10)
                        self.assertAlmostEqual(hi_ab + lo_ba, 1.0, places=10)
        self.an.momentum = 0.0

    def test_derived_markets_are_internally_coherent_for_all_formats(self):
        a, b = self.an.known_teams()[:2]
        for bo in (1, 3, 5):
            with self.subTest(bo=bo):
                pred = self.an.predict(a, b, best_of=bo, on=self.on)
                pool = [(e.map_name, e.p_a) for e in pred.map_edges]
                veto = project_veto(pool, bo)
                dist = shape_series(veto, bo, pred.p_a, 0.0, 1.05)
                lines = build_markets(dist, a, b)

                by_market: dict[str, list] = {}
                for line in lines:
                    by_market.setdefault(line.market, []).append(line)
                    self.assertGreaterEqual(line.probability, 0.0)
                    self.assertLessEqual(line.probability, 1.0)

                self.assertAlmostEqual(
                    sum(x.probability for x in by_market["vainqueur"]), 1.0, places=12
                )
                self.assertAlmostEqual(
                    sum(x.probability for x in by_market["score exact"]), 1.0, places=12
                )
                if bo >= 3:
                    line_val = bo - 0.5
                    over = by_market[f"total maps +{line_val}"][0].probability
                    under = by_market[f"total maps -{line_val}"][0].probability
                    self.assertAlmostEqual(over + under, 1.0, places=12)

    def test_core_probability_and_staking_functions_are_monotonic(self):
        # Elo expected_score increases with rating_a
        elos = [1300.0, 1450.0, 1500.0, 1600.0, 1800.0]
        ps = [expected_score(r, 1500.0) for r in elos]
        self.assertTrue(all(x < y for x, y in zip(ps, ps[1:])))

        # series_win_probability increases with p_map and amplifies advantage in longer series
        for p in (0.55, 0.65, 0.75):
            self.assertLess(series_win_probability(p, 1), series_win_probability(p, 3))
            self.assertLess(series_win_probability(p, 3), series_win_probability(p, 5))

        # devig_power decreases with odds_a (higher odds -> lower implied probability)
        p1, _ = devig_power(1.60, 2.30)
        p2, _ = devig_power(2.00, 1.85)
        self.assertGreater(p1, p2)

        # kelly_fraction is non-decreasing with model probability
        ks = [kelly_fraction(p, 2.0) for p in (0.45, 0.50, 0.55, 0.60, 0.70)]
        self.assertTrue(all(x <= y for x, y in zip(ks, ks[1:])))


if __name__ == "__main__":
    unittest.main()
