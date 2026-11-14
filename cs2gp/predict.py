"""Orchestration : charger, entrainer, predire un match a venir."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from .features import FeatureExtractor, build_datasets
from .markets import project_veto, series_distribution
from .model import MatchPredictor, map_predictor
from .ratings import EloConfig
from .schema import Match, MapResult, group_matches, normalize_team, read_csv

# Plafond de confiance. Aucun modele base sur des resultats passes ne peut
# honnetement depasser ce seuil sur un match unique.
MAX_CONFIDENCE = 0.96
# En dessous de ce volume de maps, on ne fait pas confiance a la note.
RELIABLE_MAPS = 60
# Poids du modele match face a la lecture par maps. Valeur mesuree par backtest
# (`cs2gp backtest` affiche l'optimum sur vos propres donnees) : les deux
# estimateurs se valent et se completent, l'optimum est plat autour de 0.5.
MATCH_MODEL_WEIGHT = 0.5
# Ecart entre les deux estimateurs a partir duquel il vaut la peine de prevenir.
# Fixe au 95e centile observe : a 0.15, l'avertissement se declenchait sur un
# match sur cinq et n'informait plus personne.
DIVERGENCE_NOTABLE = 0.25


def probability_from_pool(
    pool_probs: list[tuple[str, float]], best_of: int, momentum: float = 0.0
) -> float:
    """Probabilite de serie a partir des 7 probabilites de map, via le veto.

    Remplace l'ancienne moyenne du milieu du pool : le veto n'est pas un tirage
    au hasard, il est joue par deux equipes qui connaissent leurs terrains.
    Avant le tirage au sort de l'ordre du veto, aucun camp n'est garanti de
    choisir la map 1 : avec momentum non nul, on moyenne donc les deux ordres
    pour conserver exactement p(A|B) + p(B|A) = 1.
    """
    plan_a = project_veto(pool_probs, best_of)
    p_a_first = series_distribution(plan_a.order, best_of, momentum).p_a
    if momentum == 0.0 or best_of < 3:
        return p_a_first
    plan_b = project_veto([(name, 1.0 - p) for name, p in pool_probs], best_of)
    p_b_first = 1.0 - series_distribution(plan_b.order, best_of, momentum).p_a
    return 0.5 * (p_a_first + p_b_first)


def confidence_label(maps_played: int, days_since_last: int, h2h_n: int) -> str:
    """Niveau de confiance dans les *donnees*, pas dans le resultat.

    Distinction essentielle : une prediction a 85% sur deux equipes bien connues
    et une prediction a 85% sur une equipe vue six fois ne valent pas la meme
    chose, alors qu'elles affichent le meme nombre.
    """
    if maps_played < 25 or days_since_last > 60:
        return "faible"
    if maps_played < RELIABLE_MAPS or h2h_n == 0:
        return "moyenne"
    return "correcte"


def reliability(maps_played: int, days_since_last: int, certainty: float | None = None) -> float:
    """Facteur dans [0.55, 1] : quelle part de la confiance brute on garde.

    Si l'ecart-type des notes est disponible (`certainty`), il remplace la regle
    ad hoc fondee sur le nombre de maps et l'anciennete : c'est exactement la
    quantite que ces deux indicateurs essayaient d'approcher, en mieux, puisque
    elle tient compte de la qualite des adversaires rencontres et pas seulement
    de leur nombre.
    """
    if certainty is not None:
        return float(min(1.0, max(0.55, 0.55 + 0.45 * (certainty - 0.6) / 0.35)))
    volume = min(1.0, maps_played / RELIABLE_MAPS)
    freshness = 0.5 ** (max(0, days_since_last - 30) / 90.0)
    return 0.55 + 0.45 * volume * freshness


def combine_probability(
    p_match: float,
    p_from_maps: float,
    maps_played_min: int,
    days_idle_max: int,
    weight: float = None,
    certainty: float | None = None,
) -> tuple[float, float]:
    """Melange, retrecit et plafonne. Point de passage unique.

    Le backtest et la prediction en direct appellent tous deux cette fonction :
    c'est la seule facon de garantir que le chiffre mesure est bien le chiffre
    servi. Toute logique de combinaison ecrite en double finit par diverger.
    """
    w = MATCH_MODEL_WEIGHT if weight is None else weight
    p_blend = w * p_match + (1.0 - w) * p_from_maps
    p = 0.5 + (p_blend - 0.5) * reliability(maps_played_min, days_idle_max, certainty)
    return float(min(max(p, 1.0 - MAX_CONFIDENCE), MAX_CONFIDENCE)), float(p_blend)


@dataclass
class MapEdge:
    map_name: str
    elo_edge: float          # en points Elo, vu de A
    p_a: float               # proba que A gagne cette map
    a_experience: int
    b_experience: int


@dataclass
class Prediction:
    team_a: str
    team_b: str
    date: dt.date
    best_of: int
    lan: bool
    p_a: float               # proba finale que A gagne le match
    p_a_match_model: float   # modele match direct
    p_a_from_maps: float     # reconstitue depuis les probas par map
    confidence: str
    drivers: list[tuple[str, float]]
    map_edges: list[MapEdge]
    data_quality: dict
    warnings: list[str] = field(default_factory=list)
    p_a_interval: tuple[float, float] | None = None

    @property
    def favorite(self) -> str:
        return self.team_a if self.p_a >= 0.5 else self.team_b

    @property
    def p_favorite(self) -> float:
        return max(self.p_a, 1 - self.p_a)

    def fair_odds(self) -> tuple[float, float]:
        return 1.0 / max(self.p_a, 1e-6), 1.0 / max(1 - self.p_a, 1e-6)

    def to_dict(self) -> dict:
        oa, ob = self.fair_odds()
        return {
            "team_a": self.team_a,
            "team_b": self.team_b,
            "date": self.date.isoformat(),
            "best_of": self.best_of,
            "lan": self.lan,
            "p_a": self.p_a,
            "p_b": 1.0 - self.p_a,
            "p_a_match_model": self.p_a_match_model,
            "p_a_from_maps": self.p_a_from_maps,
            "p_a_interval": list(self.p_a_interval) if self.p_a_interval is not None else None,
            "favorite": self.favorite,
            "p_favorite": self.p_favorite,
            "fair_odds_a": oa,
            "fair_odds_b": ob,
            "confidence": self.confidence,
            "drivers": [{"feature": name, "contribution": val} for name, val in self.drivers],
            "map_edges": [
                {
                    "map": e.map_name,
                    "elo_edge": e.elo_edge,
                    "p_a": e.p_a,
                    "a_experience": e.a_experience,
                    "b_experience": e.b_experience,
                }
                for e in self.map_edges
            ],
            "data_quality": dict(self.data_quality),
            "warnings": list(self.warnings),
        }

    def report(self) -> str:
        oa, ob = self.fair_odds()
        lines = [
            f"{self.team_a}  vs  {self.team_b}"
            f"   [bo{self.best_of}{', LAN' if self.lan else ', online'}, {self.date}]",
            "",
            f"  {self.team_a:<16} {self.p_a:>6.1%}   cote juste {oa:.2f}",
            f"  {self.team_b:<16} {1 - self.p_a:>6.1%}   cote juste {ob:.2f}",
            "",
            f"  Favori : {self.favorite} ({self.p_favorite:.1%}) - confiance {self.confidence}",
            f"  Modele match {self.p_a_match_model:.1%} | reconstitue par maps "
            f"{self.p_a_from_maps:.1%} (vu de {self.team_a})",
        ]
        if self.p_a_interval is not None:
            lo, hi = self.p_a_interval
            lines.append(f"  IC 95% modele match : [{lo:.1%}, {hi:.1%}] (vu de {self.team_a})")
        lines += [
            "",
            "  Ce qui pese (contribution au logit, vu de " + self.team_a + ") :",
        ]
        for name, val in self.drivers:
            bar = "#" * min(20, int(abs(val) * 12))
            lines.append(f"    {name:<22} {val:+.3f}  {bar}")
        lines += ["", "  Pool de maps (avantage en points Elo, vu de " + self.team_a + ") :"]
        for e in self.map_edges:
            flag = ""
            if min(e.a_experience, e.b_experience) < 4:
                flag = "  (peu de donnees)"
            lines.append(
                f"    {e.map_name:<10} {e.elo_edge:+7.1f}   p={e.p_a:.1%}"
                f"   [{e.a_experience}v{e.b_experience} maps]{flag}"
            )
        q = self.data_quality
        lines += [
            "",
            f"  Historique : {self.team_a} {q['maps_a']} maps (derniere {q['last_a']}), "
            f"{self.team_b} {q['maps_b']} maps (derniere {q['last_b']}), "
            f"confrontations directes {q['h2h']}",
        ]
        for w in self.warnings:
            lines.append(f"  /!\\ {w}")
        return "\n".join(lines)


class Analyzer:
    """Point d'entree unique : donnees -> etat -> modele -> prediction."""

    def __init__(
        self,
        elo_config: EloConfig | None = None,
        warmup_maps: int = 12,
        momentum: float = 0.0,
        *,
        versioned_pool: bool = False,
    ) -> None:
        self.elo_config = elo_config
        self.warmup_maps = warmup_maps
        self.versioned_pool = bool(versioned_pool)
        # Effet d'enchainement entre maps. A mesurer sur vos donnees
        # (`markets.estimate_momentum`) avant d'etre suppose non nul.
        self.momentum = momentum
        self.maps: list[MapResult] = []
        self.matches: list[Match] = []
        self.fx: FeatureExtractor | None = None
        self.match_ds = None
        self.map_ds = None
        self.match_model: MatchPredictor | None = None
        self.map_model: MatchPredictor | None = None
        self._training_options: dict = {}
        self._momentum_auto = False
        self._snapshots: dict = {}
        self._calibration_history = None

    # ------------------------------------------------------------------ setup

    def save(self, path: str | Path) -> None:
        """Sauvegarde un instantane autonome, remplacant atomiquement le fichier."""
        from .snapshot import save_analyzer

        save_analyzer(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "Analyzer":
        """Restaure l'etat et les deux modeles, sans nouvel entrainement."""
        from .snapshot import load_analyzer

        return load_analyzer(path)

    def load_csv(self, path: str | Path) -> "Analyzer":
        return self.load_maps(read_csv(str(path)))

    def load_maps(self, maps: list[MapResult]) -> "Analyzer":
        """Remplace l'historique valide ; un echec conserve l'analyseur precedent."""
        maps = list(maps)
        matches = group_matches(maps)
        match_ds, map_ds, fx = build_datasets(
            matches, self.elo_config, self.warmup_maps, versioned_pool=self.versioned_pool
        )
        self.maps, self.matches = maps, matches
        self.match_ds, self.map_ds, self.fx = match_ds, map_ds, fx
        self.match_model = self.map_model = None
        self._training_options = {}
        self._snapshots.clear()
        self._calibration_history = None
        if self._momentum_auto:
            self.momentum = 0.0
        return self

    def train(self, **kwargs) -> "Analyzer":
        """Publie les deux modeles et leur calibration seulement apres reussite."""
        if self.match_ds is None:
            raise RuntimeError("charger des donnees avant d'entrainer")
        match_model = MatchPredictor(**kwargs).fit(
            self.match_ds.X, self.match_ds.y, self.match_ds.dates, self.match_ds.feature_names
        )
        map_model = map_predictor(**kwargs).fit(
            self.map_ds.X, self.map_ds.y, self.map_ds.dates, self.map_ds.feature_names
        )
        history = None
        momentum = self.momentum
        if self._momentum_auto:
            from .calibration import prequential_maps
            from .markets import estimate_momentum

            history = prequential_maps(self.match_ds, self.map_ds, **kwargs)
            diag = estimate_momentum(self.matches, history=history)
            momentum = float(diag["momentum"]) if diag.get("significatif") else 0.0
        self.match_model, self.map_model = match_model, map_model
        self._training_options = dict(kwargs)
        self._snapshots.clear()
        self._calibration_history = history
        self.momentum = momentum
        return self

    def as_of(self, on: dt.date) -> "Analyzer":
        """Reconstruit features, modeles et momentum avec les seuls jours anterieurs."""
        if self.match_model is None:
            raise RuntimeError("modele non entraine")
        if self.matches and self.matches[-1].date < on:
            return self
        key = (on, self.momentum, self._momentum_auto)
        if key not in self._snapshots:
            maps = [m for m in self.maps if m.date < on]
            if not maps:
                raise ValueError(f"aucun resultat anterieur au {on}")
            past = Analyzer(self.elo_config, self.warmup_maps,
                            momentum=0.0 if self._momentum_auto else self.momentum,
                            versioned_pool=self.versioned_pool)
            past.load_maps(maps).train(**self._training_options)
            if self._momentum_auto:
                past.apply_estimated_momentum()
            if len(self._snapshots) >= 8:
                self._snapshots.pop(next(iter(self._snapshots)))
            self._snapshots[key] = past
        return self._snapshots[key]

    # ------------------------------------------------------------------ usage

    def known_teams(self) -> list[str]:
        return sorted(self.fx.elo.teams) if self.fx else []

    def team_profile(self, name: str, on: dt.date | None = None) -> dict:
        """Profil complet d'une equipe : Elo, incertitude, forme, charge et pool de maps."""
        if self.fx is None or not self.matches:
            raise RuntimeError("charger des donnees avant de consulter un profil")
        team = self.resolve_team(name)
        as_of = on or max(m.date for m in self.matches)
        tr = self.fx.elo._read_team(team)
        win_rate, round_diff = self.fx._form(team, as_of)
        rec = self.fx.recent.get(team)
        pool_rows = []
        for mp in self.fx.pool_on(as_of):
            m_wr, m_recent_n = self.fx._map_form(team, mp, as_of)
            pool_rows.append({
                "map": mp,
                "rating": float(self.fx.elo.map_rating(team, mp, as_of)),
                "maps_played": int(tr.map_counts.get(mp, 0)),
                "recent_win_rate": float(m_wr),
                "recent_maps": int(m_recent_n),
            })
        pool_rows.sort(key=lambda row: -row["rating"])
        return {
            "team": team,
            "as_of": as_of.isoformat(),
            "rating": float(self.fx.elo.rating(team, as_of)),
            "deviation": float(self.fx.elo.deviation(team, as_of)),
            "trajectory": float(self.fx.elo.trajectory(team, as_of)),
            "maps_played": int(tr.maps_played),
            "last_match_date": tr.last_date.isoformat() if tr.last_date else None,
            "days_idle": int(max(0, (as_of - tr.last_date).days)) if tr.last_date else None,
            "recent_win_rate": float(win_rate),
            "recent_round_diff": float(round_diff),
            "perf_residual": float(self.fx._perf_residual(team, as_of)),
            "sos": float(self.fx._sos(team, as_of)),
            "workload_21d": int(self.fx._workload(team, as_of)),
            "streak": int(rec.streak) if rec else 0,
            "roster": list(tr.roster),
            "maps_with_roster": int(self.fx._roster_tenure(team)),
            "map_pool": pool_rows,
        }

    def head_to_head(self, team_a: str, team_b: str, on: dt.date | None = None) -> dict:
        """Historique direct entre deux equipes (series, maps et bilan par terrain)."""
        if self.fx is None or not self.matches:
            raise RuntimeError("charger des donnees avant de consulter l'historique direct")
        a, b = self.resolve_team(team_a), self.resolve_team(team_b)
        if a == b:
            raise ValueError("une equipe ne joue pas contre elle-meme")
        as_of = on or max(m.date for m in self.matches)
        encounters = [
            m for m in self.matches
            if {m.team_a, m.team_b} == {a, b} and (on is None or m.date < on)
        ]
        series_a = series_b = 0
        maps_a = maps_b = 0
        by_map: dict[str, dict] = {}
        recent = []
        for m in encounters:
            flipped = m.team_a != a
            ma, mb = (m.maps_b, m.maps_a) if flipped else (m.maps_a, m.maps_b)
            if m.is_complete:
                if ma > mb:
                    series_a += 1
                else:
                    series_b += 1
            for mp in m.maps:
                ra, rb = (mp.rounds_b, mp.rounds_a) if flipped else (mp.rounds_a, mp.rounds_b)
                won_a = int(ra > rb)
                maps_a += won_a
                maps_b += 1 - won_a
                slot = by_map.setdefault(mp.map_name, {"played": 0, "a_wins": 0, "b_wins": 0, "round_diff_a": 0})
                slot["played"] += 1
                slot["a_wins"] += won_a
                slot["b_wins"] += 1 - won_a
                slot["round_diff_a"] += ra - rb
            recent.append({
                "match_id": m.match_id,
                "date": m.date.isoformat(),
                "best_of": m.best_of,
                "event": m.event,
                "score": f"{ma}-{mb}",
                "winner": (a if ma > mb else b) if m.is_complete else None,
            })
        return {
            "team_a": a,
            "team_b": b,
            "as_of": as_of.isoformat(),
            "series_total": series_a + series_b,
            "series_a_wins": series_a,
            "series_b_wins": series_b,
            "maps_total": maps_a + maps_b,
            "maps_a_wins": maps_a,
            "maps_b_wins": maps_b,
            "weighted_h2h_score": float(self.fx._h2h(a, b, as_of)),
            "by_map": dict(sorted(by_map.items())),
            "recent_matches": recent[-10:],
        }

    def team_history(self, name: str) -> list[dict]:
        """Evolution chronologique de l'Elo, de l'incertitude RD et de la trajectoire."""
        from .ratings import EloEngine

        if self.fx is None or not self.maps:
            raise RuntimeError("charger des donnees avant de consulter l'historique d'une equipe")
        team = self.resolve_team(name)
        engine = EloEngine(self.elo_config)
        timeline: list[dict] = []
        for mp in sorted(self.maps, key=lambda m: (m.date, m.match_id, m.map_index)):
            involved = mp.team_a == team or mp.team_b == team
            before = engine.rating(team, mp.date) if involved else 0.0
            engine.observe(mp)
            if not involved:
                continue
            is_a = mp.team_a == team
            after = engine.rating(team, mp.date)
            timeline.append({
                "date": mp.date.isoformat(),
                "match_id": mp.match_id,
                "opponent": mp.team_b if is_a else mp.team_a,
                "map": mp.map_name,
                "won": bool(mp.a_won if is_a else not mp.a_won),
                "round_diff": int(mp.round_diff if is_a else -mp.round_diff),
                "rating_before": float(before),
                "rating_after": float(after),
                "delta": float(after - before),
                "rd_after": float(engine.deviation(team, mp.date)),
                "trajectory_after": float(engine.trajectory(team, mp.date)),
                "event": mp.event,
            })
        return timeline

    def compare_teams(self, team_a: str, team_b: str, on: dt.date | None = None) -> dict:
        """Comparaison directe de deux equipes (profils, pool de maps et H2H) sans entrainement."""
        if self.fx is None or not self.matches:
            raise RuntimeError("charger des donnees avant de comparer deux equipes")
        a, b = self.resolve_team(team_a), self.resolve_team(team_b)
        if a == b:
            raise ValueError("une equipe ne joue pas contre elle-meme")
        as_of = on or max(m.date for m in self.matches)
        prof_a = self.team_profile(a, on=as_of)
        prof_b = self.team_profile(b, on=as_of)
        edges = self.fx.map_pool_edges(a, b, as_of)
        pool = self.fx.pool_on(as_of)
        stakes = self.fx.elo.simulate_outcome(a, b, pool[0], as_of)
        return {
            "as_of": as_of.isoformat(),
            "team_a": prof_a,
            "team_b": prof_b,
            "elo_diff": float(prof_a["rating"] - prof_b["rating"]),
            "elo_win_prob_a": float(self.fx.elo.win_probability(a, b, as_of)),
            "certainty": float(self.fx.elo.certainty(a, b, as_of)),
            "elo_stakes": {
                "a_win_delta_a": stakes["if_a_wins"]["delta_a"],
                "a_loss_delta_a": stakes["if_b_wins"]["delta_a"],
                "b_win_delta_b": stakes["if_b_wins"]["delta_b"],
                "b_loss_delta_b": stakes["if_a_wins"]["delta_b"],
            },
            "map_edges": [
                {"map": m, "elo_edge": float(val)}
                for m, val in sorted(edges.items(), key=lambda kv: -kv[1])
            ],
            "h2h": self.head_to_head(a, b, on=on),
        }

    def movers(self, window_days: int = 30, min_maps: int = 20, top: int = 5) -> dict:
        """Equipes en plus forte hausse et baisse Elo sur les `window_days` derniers jours."""
        from .ratings import EloEngine
        from .validation import positive_integer

        positive_integer("window_days", window_days)
        positive_integer("top", top)
        if min_maps < 0:
            raise ValueError("min_maps doit etre positif ou nul")
        if self.fx is None or not self.maps:
            raise RuntimeError("charger des donnees avant de calculer les variations Elo")
        as_of = max(m.date for m in self.maps)
        cutoff = as_of - dt.timedelta(days=window_days)
        past_engine = EloEngine(self.elo_config)
        for mp in sorted(self.maps, key=lambda m: (m.date, m.match_id, m.map_index)):
            if mp.date <= cutoff:
                past_engine.observe(mp)
        rows = []
        for team, tr in self.fx.elo.teams.items():
            if tr.maps_played < min_maps:
                continue
            now_elo = float(self.fx.elo.rating(team, as_of))
            prev_elo = float(past_engine.rating(team, cutoff))
            rows.append({
                "team": team,
                "rating_before": prev_elo,
                "rating_after": now_elo,
                "delta": float(now_elo - prev_elo),
                "maps_played": int(tr.maps_played),
            })
        rows.sort(key=lambda r: -r["delta"])
        return {
            "as_of": as_of.isoformat(),
            "since": cutoff.isoformat(),
            "window_days": window_days,
            "risers": rows[:top],
            "fallers": list(reversed(rows[-top:])) if rows else [],
        }

    def map_stats(self) -> list[dict]:
        """Statistiques globales par map (volume, OT, ecart de rounds, reussite favori Elo)."""
        from .ratings import EloEngine

        if self.fx is None or not self.maps:
            raise RuntimeError("charger des donnees avant de calculer les statistiques par map")
        engine = EloEngine(self.elo_config)
        buckets: dict[str, dict] = {}
        for mp in sorted(self.maps, key=lambda m: (m.date, m.match_id, m.map_index)):
            ra_elo = engine.map_rating(mp.team_a, mp.map_name, mp.date)
            rb_elo = engine.map_rating(mp.team_b, mp.map_name, mp.date)
            slot = buckets.setdefault(mp.map_name, {
                "played": 0, "ot": 0, "abs_diff": 0, "rounds": 0,
                "fav_total": 0, "fav_wins": 0,
            })
            slot["played"] += 1
            hi, lo = max(mp.rounds_a, mp.rounds_b), min(mp.rounds_a, mp.rounds_b)
            if hi >= 16 and lo >= 12:
                slot["ot"] += 1
            slot["abs_diff"] += abs(mp.round_diff)
            slot["rounds"] += mp.rounds_a + mp.rounds_b
            if abs(ra_elo - rb_elo) > 1e-6:
                slot["fav_total"] += 1
                fav_a = ra_elo > rb_elo
                if (fav_a and mp.a_won) or (not fav_a and not mp.a_won):
                    slot["fav_wins"] += 1
            engine.observe(mp)
        rows = []
        for name, b in buckets.items():
            n = b["played"]
            rows.append({
                "map": name,
                "played": n,
                "ot_rate": float(b["ot"] / n),
                "avg_round_diff": float(b["abs_diff"] / n),
                "avg_total_rounds": float(b["rounds"] / n),
                "elo_favorite_win_rate": (
                    float(b["fav_wins"] / b["fav_total"]) if b["fav_total"] else None
                ),
            })
        rows.sort(key=lambda r: (-r["played"], r["map"]))
        return rows

    def event_stats(self, min_matches: int = 1) -> list[dict]:
        """Bilan par tournoi (dates, tier, LAN, volume et taux d'upsets Elo)."""
        from .ratings import EloEngine

        if min_matches < 1:
            raise ValueError("min_matches doit etre au moins 1")
        if self.fx is None or not self.matches:
            raise RuntimeError("charger des donnees avant de calculer les statistiques par tournoi")
        engine = EloEngine(self.elo_config)
        buckets: dict[str, dict] = {}
        for match in sorted(self.matches, key=lambda m: (m.date, m.match_id)):
            key = match.event or "(sans evenement)"
            slot = buckets.setdefault(key, {
                "date_min": match.date, "date_max": match.date,
                "tier": match.tier, "lan": match.lan,
                "n_matches": 0, "n_complete": 0, "fav_total": 0, "upsets": 0,
            })
            slot["date_min"] = min(slot["date_min"], match.date)
            slot["date_max"] = max(slot["date_max"], match.date)
            slot["n_matches"] += 1
            if match.is_complete:
                slot["n_complete"] += 1
                ra = engine.rating(match.team_a, match.date)
                rb = engine.rating(match.team_b, match.date)
                if abs(ra - rb) > 1e-6:
                    slot["fav_total"] += 1
                    fav_a = ra > rb
                    if (fav_a and not match.a_won) or (not fav_a and match.a_won):
                        slot["upsets"] += 1
            for mp in sorted(match.maps, key=lambda x: x.map_index):
                engine.observe(mp)
        rows = [
            {
                "event": name,
                "date_start": b["date_min"].isoformat(),
                "date_end": b["date_max"].isoformat(),
                "tier": b["tier"],
                "lan": bool(b["lan"]),
                "n_matches": b["n_matches"],
                "n_complete": b["n_complete"],
                "elo_upset_rate": float(b["upsets"] / b["fav_total"]) if b["fav_total"] else None,
            }
            for name, b in buckets.items()
            if b["n_matches"] >= min_matches
        ]
        rows.sort(key=lambda r: (r["date_end"], r["n_matches"]), reverse=True)
        return rows

    def calibration_history(self):
        """Historique hors entrainement partage par les calibrations et mis en cache."""
        from .calibration import prequential_maps

        if self.map_model is None:
            raise RuntimeError("entrainer avant de calibrer")
        if self._calibration_history is None:
            self._calibration_history = prequential_maps(
                self.match_ds, self.map_ds, **self._training_options
            )
        return self._calibration_history

    def estimate_momentum(self) -> dict:
        """Mesure l'effet d'enchainement entre maps sur l'historique charge.

        Reponds a : sachant ce que le modele sait deja des deux equipes et de la
        map, gagner la map precedente porte-t-il un signal supplementaire ? Le
        resultat inclut un test de significativite - on ne branche l'effet en
        production que s'il depasse le bruit (`significatif`). Sans cela on
        inventerait du momentum la ou il n'y en a pas, defaut classique des
        modeles publics.
        """
        from .markets import estimate_momentum

        if self.map_model is None:
            raise RuntimeError("entrainer le modele avant d'estimer le momentum")
        return estimate_momentum(self.matches, history=self.calibration_history())

    def apply_estimated_momentum(self) -> dict:
        """Estime le momentum et l'active seulement s'il est significatif.

        Retourne le diagnostic complet ; `self.momentum` n'est modifie que si
        l'effet tient statistiquement. C'est la seule quantite du systeme qu'on
        laisse a zero par defaut, precisement parce qu'elle est facile a
        halluciner.
        """
        diag = self.estimate_momentum()
        self._momentum_auto = True
        self._snapshots.clear()
        self.momentum = float(diag["momentum"]) if diag.get("significatif") else 0.0
        return diag

    def resolve_team(self, name: str) -> str:
        canonical = normalize_team(name)
        if self.fx and canonical in self.fx.elo.teams:
            return canonical
        candidates = [t for t in self.known_teams()
                      if normalize_team(t) == canonical or canonical in t or t in canonical]
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            raise ValueError(f"'{name}' est ambigu : {candidates[:8]}")
        raise ValueError(f"equipe inconnue : '{name}' (voir la commande `teams`)")

    def predict(
        self,
        team_a: str,
        team_b: str,
        best_of: int = 3,
        lan: bool = False,
        on: dt.date | None = None,
    ) -> Prediction:
        if self.match_model is None or self.fx is None:
            raise RuntimeError("modele non entraine")
        on = on or dt.date.today()
        past = self.as_of(on)
        if past is not self:
            return past.predict(team_a, team_b, best_of, lan, on)
        a, b = self.resolve_team(team_a), self.resolve_team(team_b)
        if a == b:
            raise ValueError("une equipe ne joue pas contre elle-meme")

        x = self.fx.match_features(a, b, on, best_of, lan)
        p_match = self.match_model.predict_one(x)

        edges: list[MapEdge] = []
        for m in self.fx.pool_on(on):
            xm = self.fx.map_features(a, b, on, m)
            edges.append(
                MapEdge(
                    map_name=m,
                    elo_edge=self.fx.elo.map_rating(a, m, on) - self.fx.elo.map_rating(b, m, on),
                    p_a=self.map_model.predict_one(xm),
                    a_experience=self.fx.elo.teams[a].map_counts.get(m, 0),
                    b_experience=self.fx.elo.teams[b].map_counts.get(m, 0),
                )
            )
        edges.sort(key=lambda e: -e.elo_edge)

        k = max(1, best_of)
        p_from_maps = probability_from_pool(
            [(e.map_name, e.p_a) for e in edges], best_of, self.momentum
        )


        quality = self.fx.match_context(a, b, on)
        quality["data_through"] = self.match_model.last_date
        n_h2h = quality["h2h_n"]
        thin, stale = quality["maps_played_min"], quality["days_idle_max"]

        # Le modele match voit la forme, le h2h, le repos ; la lecture par maps
        # voit le terrain. Ni l'un ni l'autre ne domine : le melange bat les deux.
        # Puis retrecissement si l'historique est mince ou perime, et plafond dur :
        # un bo3 de CS2 n'est jamais joue d'avance.
        p_final, p_blend = combine_probability(
            p_match, p_from_maps, thin, stale, certainty=quality["certainty"]
        )

        warnings: list[str] = []
        if thin < 25:
            warnings.append(
                f"historique mince ({thin} maps pour la moins connue) : proba peu fiable"
            )
        if stale > 60:
            warnings.append(
                f"une des equipes n'a pas joue depuis {stale} jours : note amortie vers la moyenne"
            )
        if abs(p_match - p_from_maps) > DIVERGENCE_NOTABLE:
            warnings.append(
                "modele match et lecture par maps divergent : le veto sera decisif"
            )
        if abs(p_final - p_blend) > 0.02:
            reason = (
                f"plafond de confiance {MAX_CONFIDENCE:.0%} : sur un match unique, "
                "aucun historique ne justifie mieux"
                if max(p_blend, 1 - p_blend) > MAX_CONFIDENCE
                else "historique trop mince ou trop ancien pour assumer cette assurance"
            )
            warnings.append(f"probabilite ramenee de {p_blend:.1%} a {p_final:.1%} ({reason})")
        if k >= 3 and abs(p_final - 0.5) < 0.04:
            warnings.append("match a pile ou face : aucune valeur a chercher ici")

        confidence = confidence_label(thin, stale, n_h2h)
        p_interval = (
            self.match_model.predict_interval(x)
            if self.match_model.logistic.cov is not None else None
        )

        return Prediction(
            team_a=a, team_b=b, date=on, best_of=best_of, lan=lan,
            p_a=float(p_final), p_a_match_model=float(p_match),
            p_a_from_maps=float(p_from_maps), confidence=confidence,
            drivers=self.match_model.explain(x), map_edges=edges,
            data_quality=quality, warnings=warnings,
            p_a_interval=p_interval,
        )
