"""Backtest walk-forward.

Le seul test honnete : n'entrainer que sur le passe, predire le futur, et
reentrainer periodiquement comme on le ferait en production. Une validation
croisee aleatoire donnerait ici des scores flatteurs et faux, parce que les
notes Elo d'un match donne encodent deja des matchs posterieurs.
"""

from __future__ import annotations

from bisect import bisect_left
import datetime as dt
from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np

from .csvio import atomic_write_csv, read_rows
from .features import Dataset
from .metrics import (
    brier, brier_decomposition, calibration_slope_intercept, format_summary, log_loss,
    paired_block_bootstrap, quantile_reliability_table, reliability_table, spiegelhalter_z, summarize,
)
from .model import MatchPredictor, map_predictor
from .ratings import expected_score
from .schema import ACTIVE_DUTY, active_duty_on
from .temporal import forward_splits
from .validation import finite_number, positive_integer


@dataclass
class BacktestResult:
    predictions: np.ndarray          # melange servi par `predict`
    y: np.ndarray
    dates: list[dt.date]
    meta: list[dict]
    elo_only: np.ndarray
    folds: int
    metrics: dict
    elo_metrics: dict
    match_only: np.ndarray | None = None
    maps_only: np.ndarray | None = None
    #  Melange avant garde-fou : sert a distinguer ce que coute le melange de
    #  ce que coute le retrecissement de securite, deux choses differentes.
    blended_raw: np.ndarray | None = None
    blend: float = 0.5
    best_blend: float | None = None
    comparison: dict | None = None
    fold_slices: list[tuple[int, int]] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            f"Backtest walk-forward : {self.folds} reentrainements, "
            f"{len(self.y)} matchs evalues "
            f"({self.dates[0]} -> {self.dates[-1]})",
            format_summary(f"melange (w={self.blend:.2f})", self.metrics),
        ]
        if self.match_only is not None:
            lines.append(format_summary("modele match seul", summarize(self.y, self.match_only)))
        if self.maps_only is not None:
            lines.append(format_summary("lecture par maps", summarize(self.y, self.maps_only)))
        if self.blended_raw is not None:
            lines.append(
                format_summary("melange sans garde-fou", summarize(self.y, self.blended_raw))
            )
        lines.append(format_summary("Elo seul (reference)", self.elo_metrics))
        if self.comparison is not None:
            c = self.comparison
            lines.append(f"Gain log-loss face a Elo : {c['gain']:+.4f} (positif = mieux).")
            if c["low"] is None:
                lines.append("IC indisponible : moins de deux blocs calendaires observes.")
            else:
                lines.append(f"IC 95% apparie [{c['low']:+.4f}, {c['high']:+.4f}], "
                             f"{c['blocks']} blocs de {c['block_days']} jours, "
                             f"{c['samples']} tirages.")
                if c["low"] <= 0 <= c["high"]:
                    lines.append("L'intervalle inclut zero : l'avantage sur Elo reste incertain.")
                lines.append("Intervalle conditionnel a cette periode ; ne mesure pas un changement de meta.")
        if self.best_blend is not None:
            lines.append(
                f"\nPoids optimal mesure sur cette periode : w={self.best_blend:.2f} "
                f"(w=1 : modele match seul, w=0 : lecture par maps). "
                f"Defaut du code : {self.blend:.2f}."
            )
        lines += ["", "Fiabilite (proba annoncee vs realisee) :"]
        for row in reliability_table(self.y, self.predictions):
            gap = row["observed"] - row["predicted"]
            lines.append(
                f"  {row['bin']:<10} n={row['n']:<5} annonce {row['predicted']:.1%} "
                f"observe {row['observed']:.1%}  ecart {gap:+.1%}"
            )
        cox = calibration_slope_intercept(self.y, self.predictions)
        if cox["converged"]:
            lines.append(
                f"\nCalibration logistique (Cox) : pente={cox['slope']:.3f} (ideal 1.000), "
                f"intercept={cox['intercept']:+.3f} (ideal 0.000)."
            )
        murphy = brier_decomposition(self.y, self.predictions)
        spieg = spiegelhalter_z(self.y, self.predictions)
        spieg_txt = (
            f" | Spiegelhalter Z={spieg['z_stat']:+.2f} (p={spieg['p_value']:.3f})"
            if spieg["z_stat"] is not None else ""
        )
        lines.append(
            f"Decomposition de Brier (Murphy) : fiabilite={murphy['reliability']:.4f}, "
            f"resolution={murphy['resolution']:.4f}, BSS={murphy['brier_skill_score']:+.1%}.{spieg_txt}"
        )
        return "\n".join(lines)

    def breakdown(self, key, *, min_n: int = 10) -> dict[str, dict]:
        """Ventile les metriques par segment (ex: 'best_of', 'lan', 'tier' ou callable)."""
        positive_integer("min_n", min_n)
        extractor = key if callable(key) else (lambda m: str(m.get(key, "inconnu")))
        groups: dict[str, list[int]] = {}
        for idx, meta in enumerate(self.meta):
            groups.setdefault(str(extractor(meta)), []).append(idx)
        out: dict[str, dict] = {}
        for label, indices in sorted(groups.items()):
            if len(indices) < min_n:
                continue
            y_sub = self.y[indices]
            p_sub = self.predictions[indices]
            elo_sub = self.elo_only[indices]
            summary = summarize(y_sub, p_sub)
            elo_ll = log_loss(y_sub, elo_sub)
            out[label] = {
                **summary,
                "elo_log_loss": elo_ll,
                "log_loss_gain_vs_elo": float(elo_ll - summary["log_loss"]),
            }
        return out

    def format_breakdown(self, title: str, key, *, min_n: int = 10) -> str:
        segments = self.breakdown(key, min_n=min_n)
        if not segments:
            return ""
        lines = [f"\n{title} :"]
        for label, stats in segments.items():
            lines.append(
                f"{format_summary('  ' + label, stats)}  "
                f"gain_elo={stats['log_loss_gain_vs_elo']:+.4f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        out = {
            "folds": self.folds,
            "n_matches": int(len(self.y)),
            "date_start": self.dates[0].isoformat() if self.dates else None,
            "date_end": self.dates[-1].isoformat() if self.dates else None,
            "blend": self.blend,
            "best_blend": self.best_blend,
            "metrics": dict(self.metrics),
            "elo_metrics": dict(self.elo_metrics),
            "comparison": dict(self.comparison) if self.comparison is not None else None,
            "calibration": calibration_slope_intercept(self.y, self.predictions) if len(self.y) else None,
            "spiegelhalter": spiegelhalter_z(self.y, self.predictions) if len(self.y) else None,
            "brier_decomposition": brier_decomposition(self.y, self.predictions) if len(self.y) else None,
            "reliability": reliability_table(self.y, self.predictions) if len(self.y) else [],
            "quantile_reliability": quantile_reliability_table(self.y, self.predictions) if len(self.y) else [],
        }
        if self.match_only is not None:
            out["match_only_metrics"] = summarize(self.y, self.match_only)
        if self.maps_only is not None:
            out["maps_only_metrics"] = summarize(self.y, self.maps_only)
        if self.blended_raw is not None:
            out["blended_raw_metrics"] = summarize(self.y, self.blended_raw)
        return out

    def drift_report(self) -> list[dict]:
        """Suivi par fold du log-loss, du gain vs Elo et de la pente de calibration."""
        if not len(self.y):
            return []
        slices = self.fold_slices
        if not slices:
            k = max(1, self.folds)
            bounds = np.linspace(0, len(self.y), k + 1, dtype=int)
            slices = [(int(a), int(b)) for a, b in zip(bounds, bounds[1:]) if b > a]
        rows: list[dict] = []
        for idx, (a, b) in enumerate(slices, 1):
            y_sub = self.y[a:b]
            p_sub = self.predictions[a:b]
            elo_sub = self.elo_only[a:b]
            summary = summarize(y_sub, p_sub)
            elo_ll = log_loss(y_sub, elo_sub)
            cox = calibration_slope_intercept(y_sub, p_sub)
            rows.append({
                "fold": idx,
                "date_start": self.dates[a].isoformat(),
                "date_end": self.dates[b - 1].isoformat(),
                "n": int(b - a),
                "log_loss": summary["log_loss"],
                "elo_log_loss": elo_ll,
                "gain_vs_elo": float(elo_ll - summary["log_loss"]),
                "brier": summary["brier"],
                "ece": summary["ece"],
                "calibration_slope": cox["slope"] if cox["converged"] else None,
            })
        return rows

    def export_csv(self, path: str | Path) -> int:
        """Exporte atomiquement les predictions hors-echantillon en CSV."""
        cols = [
            "match_id", "date", "team_a", "team_b", "best_of", "lan", "tier", "event",
            "y", "p_model", "p_elo", "p_match", "p_maps",
        ]
        rows = (
            [
                self.meta[i].get("match_id", ""),
                self.dates[i].isoformat(),
                self.meta[i].get("team_a", ""),
                self.meta[i].get("team_b", ""),
                self.meta[i].get("best_of", ""),
                int(bool(self.meta[i].get("lan", False))),
                self.meta[i].get("tier", ""),
                self.meta[i].get("event", ""),
                int(self.y[i]),
                f"{float(self.predictions[i]):.6f}",
                f"{float(self.elo_only[i]):.6f}",
                "" if self.match_only is None else f"{float(self.match_only[i]):.6f}",
                "" if self.maps_only is None else f"{float(self.maps_only[i]):.6f}",
            ]
            for i in range(len(self.y))
        )
        return atomic_write_csv(path, cols, rows)


def _elo_reference(ds: Dataset) -> np.ndarray:
    """Baseline : Elo brut converti en probabilite, sans modele appris."""
    i = ds.feature_names.index("elo_diff")
    return np.array([expected_score(x[i] * 100.0, 0.0) for x in ds.X])


def _pool_to_series(
    map_model: MatchPredictor,
    pool_x: np.ndarray,
    best_of: int,
    momentum: float = 0.0,
    pool_names: tuple[str, ...] = ACTIVE_DUTY,
) -> float:
    """Probabilites des 7 maps -> probabilite de serie, via le veto projete."""
    from .predict import probability_from_pool  # import local : evite un cycle

    ps = map_model.predict_proba(np.asarray(pool_x, dtype=float))
    return probability_from_pool(list(zip(pool_names, ps.tolist())), best_of, momentum)


def walk_forward(
    ds: Dataset,
    map_ds: Dataset | None = None,
    initial_frac: float = 0.4,
    refit_every: int = 150,
    min_train: int = 200,
    blend: float = 0.5,
    momentum: float = 0.0,
    evaluation_start: dt.date | None = None,
    bootstrap_samples: int = 0,
    block_days: int = 7,
    **model_kwargs,
) -> BacktestResult:
    """Entraine sur [0, i), predit [i, i+refit_every), avance, recommence.

    Si `map_ds` est fourni, evalue aussi la lecture par maps et leur melange,
    c'est-a-dire exactement ce que sert `Analyzer.predict`.
    """
    finite_number("blend", blend, maximum=1.0)
    finite_number("momentum", momentum, minimum=-math.inf)
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples doit etre un entier positif ou nul")
    positive_integer("block_days", block_days)
    n = len(ds)
    splits = forward_splits(ds.dates, initial_frac, min_train, refit_every, evaluation_start)
    start = splits[0][0]

    from .predict import MAX_CONFIDENCE, combine_probability, reliability  # import local : evite un cycle

    use_maps = map_ds is not None and ds.pool_X is not None
    p_match = np.zeros(n - start)
    p_maps = np.zeros(n - start) if use_maps else None
    folds = 0
    for i, j in splits:
        model = MatchPredictor(**model_kwargs)
        model.fit(ds.X[:i], ds.y[:i], ds.dates[:i], ds.feature_names)
        p_match[i - start : j - start] = model.predict_proba(ds.X[i:j])

        if use_maps:
            # Le modele map ne voit que les maps jouees avant la date du 1er match
            # predit : meme frontiere temporelle que le modele match.
            cutoff = ds.dates[i]
            k = bisect_left(map_ds.dates, cutoff)
            # Logistique seule au niveau map. Le GBDT a ete essaye ici (son
            # dataset est pourtant le plus gros, 4758 lignes contre 2416) :
            # +0.0005 de log-loss pour +0.0011 d'erreur type appariee, soit
            # rien, contre le double de temps de calcul.
            map_model = map_predictor(**model_kwargs)
            map_model.fit(map_ds.X[:k], map_ds.y[:k], map_ds.dates[:k], map_ds.feature_names)
            for r in range(i, j):
                pool_names = active_duty_on(ds.dates[r], versioned=ds.versioned_pool)
                p_maps[r - start] = _pool_to_series(
                    map_model, ds.pool_X[r], ds.meta[r].get("best_of", 3), momentum, pool_names
                )
        folds += 1

    y = ds.y[start:]
    elo_only = _elo_reference(ds)[start:]
    if use_maps:
        # Meme fonction de combinaison que `Analyzer.predict` : retrecissement
        # selon la fiabilite des donnees, puis plafond de confiance. Le chiffre
        # mesure ici est donc exactement celui qui sera servi.
        blended_raw = blend * p_match + (1.0 - blend) * p_maps
        preds = np.array([
            combine_probability(
                p_match[r], p_maps[r],
                ds.meta[start + r].get("maps_played_min", 999),
                ds.meta[start + r].get("days_idle_max", 0),
                weight=blend,
                certainty=ds.meta[start + r].get("certainty"),
            )[0]
            for r in range(len(y))
        ])
        grid = np.arange(0.0, 1.001, 0.05)
        rel = np.array([
            reliability(
                ds.meta[start + r].get("maps_played_min", 999),
                ds.meta[start + r].get("days_idle_max", 0),
                ds.meta[start + r].get("certainty"),
            )
            for r in range(len(y))
        ])

        def _ll(w: float) -> float:
            pb = w * p_match + (1.0 - w) * p_maps
            pw = np.clip(0.5 + (pb - 0.5) * rel, 1.0 - MAX_CONFIDENCE, MAX_CONFIDENCE)
            return log_loss(y, pw)

        best = float(min(grid, key=_ll))
    else:
        preds, best, blended_raw = p_match, None, None

    return BacktestResult(
        predictions=preds,
        y=y,
        dates=ds.dates[start:],
        meta=ds.meta[start:],
        elo_only=elo_only,
        folds=folds,
        metrics=summarize(y, preds),
        elo_metrics=summarize(y, elo_only),
        match_only=p_match if use_maps else None,
        maps_only=p_maps,
        blended_raw=blended_raw,
        blend=blend,
        best_blend=best,
        comparison=(paired_block_bootstrap(y, preds, elo_only, ds.dates[start:],
                                          samples=bootstrap_samples, block_days=block_days)
                    if bootstrap_samples else None),
        fold_slices=[(i - start, j - start) for i, j in splits],
    )


def read_odds_csv(path: str | Path) -> dict[str, tuple]:
    """Charge un CSV de cotes historiques : match_id,odds_a,odds_b[,closing_odds_a,closing_odds_b]."""
    from .odds import implied

    out: dict[str, tuple] = {}
    for line, row in read_rows(path, {"match_id", "odds_a", "odds_b"}):
        mid = row["match_id"].strip()
        if not mid:
            raise ValueError(f"ligne {line} invalide dans {path}: match_id vide")
        if mid in out:
            raise ValueError(f"ligne {line} invalide dans {path}: match_id duplique ({mid})")
        try:
            oa = float(row["odds_a"])
            ob = float(row["odds_b"])
            implied(oa)
            implied(ob)
            ca_raw = (row.get("closing_odds_a") or "").strip()
            cb_raw = (row.get("closing_odds_b") or "").strip()
            if bool(ca_raw) != bool(cb_raw):
                raise ValueError("closing_odds_a et closing_odds_b doivent etre fournis ensemble")
            if ca_raw:
                ca, cb = float(ca_raw), float(cb_raw)
                implied(ca)
                implied(cb)
                out[mid] = (oa, ob, ca, cb)
            else:
                out[mid] = (oa, ob)
        except ValueError as exc:
            raise ValueError(f"ligne {line} invalide dans {path}: {exc}") from exc
    return out


def bet_backtest(
    result: BacktestResult,
    odds_lookup: dict[str, tuple],
    min_edge: float = 0.04,
) -> dict:
    """Rejoue le backtest contre des cotes historiques, si on en dispose.

    `odds_lookup` : match_id -> (cote_a, cote_b) ou (cote_a, cote_b, cloture_a, cloture_b).
    Sans cotes reelles, cette fonction ne renvoie rien d'exploitable : ne pas
    substituer des cotes moyennes ou reconstituees, le biais serait total.
    """
    from .odds import (
        closing_line_value, find_value, overround, roi_significance, simulate_flat, simulate_kelly,
    )

    finite_number("min_edge", min_edge, maximum=1.0)
    plays = []
    overrounds = []
    clv_probs = []
    clv_evs = []
    for p, y, meta in zip(result.predictions, result.y, result.meta):
        odds = odds_lookup.get(meta["match_id"])
        if not odds:
            continue
        overrounds.append(overround(odds[0], odds[1]))
        bet = find_value(float(p), odds[0], odds[1], meta["team_a"], meta["team_b"], min_edge)
        if bet is None:
            continue
        won = int((bet.side == meta["team_a"] and y == 1) or (bet.side == meta["team_b"] and y == 0))
        plays.append((bet.model_p, bet.odds, won))
        if len(odds) == 4:
            side = "a" if bet.side == meta["team_a"] else "b"
            clv = closing_line_value(bet.odds, odds[2], odds[3], side=side)
            clv_probs.append(clv["clv_prob"])
            clv_evs.append(clv["clv_ev"])
    out = simulate_flat(plays)
    out["kelly"] = simulate_kelly(plays)
    out["significance"] = roi_significance(plays)
    out["matches_with_odds"] = len(overrounds)
    out["avg_overround"] = float(np.mean(overrounds)) if overrounds else None
    out["clv_bets"] = len(clv_probs)
    out["avg_clv_prob"] = float(np.mean(clv_probs)) if clv_probs else None
    out["avg_clv_ev"] = float(np.mean(clv_evs)) if clv_evs else None
    return out


def benchmark_against_truth(result: BacktestResult, truth: dict[str, float]) -> dict:
    """Mesure l'ecart residuel face aux probabilites vraies d'une simulation.

    Permet de savoir combien de signal reste sur la table entre le modele evalue
    et le plafond atteignable (oracle connaissant les niveaux latents).
    """
    indices = [i for i, m in enumerate(result.meta) if m.get("match_id") in truth]
    if not indices:
        raise ValueError("aucun match evalue ne correspond aux identifiants de truth")
    y = result.y[indices]
    p_model = result.predictions[indices]
    p_elo = result.elo_only[indices]
    p_true = np.array([truth[result.meta[i]["match_id"]] for i in indices], dtype=float)
    model_ll = log_loss(y, p_model)
    oracle_ll = log_loss(y, p_true)
    elo_ll = log_loss(y, p_elo)
    model_brier = brier(y, p_model)
    oracle_brier = brier(y, p_true)
    return {
        "n_matched": len(indices),
        "model_log_loss": model_ll,
        "oracle_log_loss": oracle_ll,
        "excess_log_loss": float(model_ll - oracle_ll),
        "elo_excess_log_loss": float(elo_ll - oracle_ll),
        "model_brier": model_brier,
        "oracle_brier": oracle_brier,
        "excess_brier": float(model_brier - oracle_brier),
        "mean_abs_prob_error": float(np.mean(np.abs(p_model - p_true))),
    }


def benchmark_regimes(
    regimes: tuple[str, ...] = ("facile", "realiste"),
    seeds: tuple[int, ...] = (3, 7, 11),
    *,
    n_teams: int = 12,
    years: float = 1.5,
    start: dt.date = dt.date(2024, 1, 1),
    with_rosters: bool = False,
    veto_samples: int = 50,
    **wf_kwargs,
) -> dict[str, dict]:
    """Benchmark reproductible multi-graines comparant modele, Elo et oracle."""
    from .features import build_datasets
    from .schema import group_matches
    from .sources import simulate_with_truth

    if not regimes or not seeds:
        raise ValueError("regimes et seeds doivent etre non vides")
    summary: dict[str, dict] = {}
    for regime in regimes:
        runs = []
        for seed in seeds:
            rows, truth = simulate_with_truth(
                n_teams=n_teams, years=years, seed=seed, start=start,
                regime=regime, veto_samples=veto_samples, with_rosters=with_rosters,
            )
            mds, pds, _ = build_datasets(group_matches(rows))
            res = walk_forward(mds, pds, **{"use_gbdt": False, **wf_kwargs})
            bench = benchmark_against_truth(res, truth)
            runs.append({
                "seed": seed,
                "n_matches": bench["n_matched"],
                "model_log_loss": bench["model_log_loss"],
                "elo_log_loss": float(res.elo_metrics["log_loss"]),
                "oracle_log_loss": bench["oracle_log_loss"],
                "excess_log_loss": bench["excess_log_loss"],
                "gain_vs_elo": float(res.elo_metrics["log_loss"] - bench["model_log_loss"]),
                "mean_abs_prob_error": bench["mean_abs_prob_error"],
            })
        keys = ("model_log_loss", "elo_log_loss", "oracle_log_loss", "excess_log_loss",
                "gain_vs_elo", "mean_abs_prob_error")
        summary[regime] = {
            "seeds": list(seeds),
            "runs": runs,
            "mean": {k: float(np.mean([r[k] for r in runs])) for k in keys},
            "std": {k: float(np.std([r[k] for r in runs])) for k in keys},
        }
    return summary


# ---------------------------------------------------------------------------
# Reglage des constantes.
#
# Les valeurs par defaut sont ajustees sur des mondes simules realistes. Elles
# ne sont pas universelles : le bon K depend de la vitesse a laquelle le niveau
# des equipes bouge, qui n'est pas la meme selon la periode, le circuit, ou le
# niveau de jeu couvert par vos donnees. Cette fonction refait le reglage sur
# vos donnees, avec un decoupage temporel qui empeche de se mentir a soi-meme.
# ---------------------------------------------------------------------------

TUNING_GRID: dict[str, tuple] = {
    "k_base": (8.0, 11.0, 14.0, 18.0, 24.0),
    "k_map": (12, 20, 26, 34),
    "rd_growth_days": (250, 700, 2000),
    "fast_multiplier": (0.0, 2.0, 3.0, 5.0),
    "idle_halflife_days": (400, 900, 2000),
    "map_prior_weight": (0.3, 0.65, 1.2),
}


def tune(
    matches,
    grid: dict | None = None,
    holdout_frac: float = 0.25,
    warmup_maps: int = 12,
    verbose: bool = True,
    **wf_kwargs,
):
    """Recherche par coordonnees sur les constantes du systeme de notation.

    Protocole : le reglage ne voit que le debut de l'historique, la validation
    porte sur la fin, jamais touchee. Un reglage valide sur les memes matchs que
    ceux qui l'ont produit ne prouve rien - c'est la facon la plus courante de
    fabriquer une amelioration qui n'existe pas.

    Retourne (config retenue, rapport).
    """
    from .features import build_datasets
    from .ratings import EloConfig

    if not math.isfinite(holdout_frac) or not 0 < holdout_frac < 1:
        raise ValueError("holdout_frac doit etre strictement entre 0 et 1")
    grid = TUNING_GRID if grid is None else grid
    matches = sorted(matches, key=lambda m: (m.date, m.match_id))
    cut = int(len(matches) * (1 - holdout_frac))
    if not 0 < cut < len(matches):
        raise ValueError("historique trop court pour reserver une periode de validation")
    validation_date = matches[cut].date
    cut = bisect_left([m.date for m in matches], validation_date)
    fit_matches = matches[:cut]

    def evaluate(cfg: EloConfig, subset, start_date=None) -> BacktestResult:
        mds, pds, _ = build_datasets(subset, cfg, warmup_maps)
        kwargs = {**wf_kwargs, "use_gbdt": False}
        if start_date is not None:
            kwargs["evaluation_start"] = start_date
        return walk_forward(mds, map_ds=pds, **kwargs)

    def score(cfg: EloConfig, subset) -> float:
        return evaluate(cfg, subset).metrics["log_loss"]

    current = EloConfig()
    baseline_fit = score(current, fit_matches)
    history = [("depart", None, baseline_fit)]
    best = baseline_fit

    for name, values in grid.items():
        trials = []
        for v in values:
            cfg = EloConfig.from_dict({**current.to_dict(), name: v})
            try:
                ll = score(cfg, fit_matches)
            except (ValueError, ZeroDivisionError):
                continue
            trials.append((ll, v))
            history.append((name, v, ll))
        if not trials:
            continue
        ll, v = min(trials)
        if ll < best - 1e-6:
            best = ll
            current = EloConfig.from_dict({**current.to_dict(), name: v})
            if verbose:
                print(f"  {name:<22} -> {v}   (log-loss {ll:.4f})")
        elif verbose:
            print(f"  {name:<22} -> inchange")

    # Tout le passe construit les ratings ; seules les cibles reservees comptent.
    baseline_validation = evaluate(EloConfig(), matches, validation_date)
    tuned_validation = evaluate(current, matches, validation_date)
    report = {
        "logloss_reglage_avant": baseline_fit,
        "logloss_reglage_apres": best,
        "logloss_validation_avant": baseline_validation.metrics["log_loss"],
        "logloss_validation_apres": tuned_validation.metrics["log_loss"],
        "essais": history,
        "n_matchs_reglage": cut,
        "n_matchs_total": len(matches),
        "n_matchs_validation": len(tuned_validation.y),
        "validation_debut": tuned_validation.dates[0].isoformat(),
        "validation_fin": tuned_validation.dates[-1].isoformat(),
        "reglage_match_ids": [m.match_id for m in fit_matches],
        "validation_match_ids": [m["match_id"] for m in tuned_validation.meta],
    }
    report["gain_reel"] = (
        report["logloss_validation_avant"] - report["logloss_validation_apres"]
    )
    return current, report
