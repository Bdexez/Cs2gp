"""Confrontation du modele au marche : retrait de la marge, valeur, mise.

Un modele qui "a raison a 65%" ne vaut rien en soi : le marche aussi. Ce qui
compte est l'ecart entre la probabilite du modele et la probabilite du marche
une fois la marge du bookmaker retiree, et la fiabilite de cet ecart.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from .validation import finite_number


def _probability(p: float) -> None:
    finite_number("probabilite", p, maximum=1.0)


def implied(odds: float) -> float:
    odds = float(odds)
    if not math.isfinite(odds) or odds <= 1.0:
        raise ValueError("une cote decimale doit etre finie et > 1.0")
    return 1.0 / odds


def overround(odds_a: float, odds_b: float) -> float:
    """Marge du bookmaker. 1.05 = 5% de vig."""
    return implied(odds_a) + implied(odds_b)


def devig_proportional(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Retrait proportionnel : simple, surestime un peu les gros favoris."""
    ia, ib = implied(odds_a), implied(odds_b)
    tot = ia + ib
    return ia / tot, ib / tot


def devig_power(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Retrait par puissance : p_i ∝ implied_i**k, k tel que la somme fasse 1.

    Corrige mieux le biais favori/outsider que le retrait proportionnel.
    """
    ia, ib = implied(odds_a), implied(odds_b)
    # La racine est positive, sans borne universelle a 0.5 ou 2 : une
    # surcote comme une marge tres forte peut sortir de cet ancien intervalle.
    lo, hi = 0.0, 1.0
    while ia**hi + ib**hi > 1.0:
        hi *= 2.0
    for _ in range(80):
        k = (lo + hi) / 2.0
        s = ia**k + ib**k
        if s > 1.0:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2.0
    s = ia**k + ib**k
    return ia**k / s, ib**k / s


def _shin_z(ia: float, ib: float) -> float:
    s = ia + ib
    if s <= 1.0:
        return 0.0

    def _sum_p(z: float) -> float:
        denom = 2.0 * (1.0 - z)
        pa = (math.sqrt(z * z + 4.0 * (1.0 - z) * ia * ia / s) - z) / denom
        pb = (math.sqrt(z * z + 4.0 * (1.0 - z) * ib * ib / s) - z) / denom
        return pa + pb

    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _sum_p(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def devig_shin(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Retrait de marge selon le modele de Shin (part d'inities z)."""
    ia, ib = implied(odds_a), implied(odds_b)
    s = ia + ib
    if s <= 1.0:
        return ia / s, ib / s
    z = _shin_z(ia, ib)
    denom = 2.0 * (1.0 - z)
    pa = (math.sqrt(z * z + 4.0 * (1.0 - z) * ia * ia / s) - z) / denom
    pb = (math.sqrt(z * z + 4.0 * (1.0 - z) * ib * ib / s) - z) / denom
    tot = pa + pb
    return float(pa / tot), float(pb / tot)


def devig(odds_a: float, odds_b: float, method: str = "power") -> tuple[float, float]:
    """Retrait de marge unifie : 'power' (defaut), 'shin' ou 'proportional'."""
    if method == "power":
        return devig_power(odds_a, odds_b)
    if method == "shin":
        return devig_shin(odds_a, odds_b)
    if method == "proportional":
        return devig_proportional(odds_a, odds_b)
    raise ValueError(f"methode de devig inconnue : {method!r} (attendu: power, shin, proportional)")


def market_diagnostics(odds_a: float, odds_b: float) -> dict:
    """Compare marge, arbitrage eventuel et probabilites sans marge (prop, power, Shin)."""
    ia, ib = implied(odds_a), implied(odds_b)
    ov = ia + ib
    pa_prop, pb_prop = devig_proportional(odds_a, odds_b)
    pa_pow, pb_pow = devig_power(odds_a, odds_b)
    pa_shin, pb_shin = devig_shin(odds_a, odds_b)
    return {
        "overround": float(ov),
        "vig": float(ov - 1.0),
        "is_arbitrage": bool(ov < 1.0),
        "implied_a": float(ia),
        "implied_b": float(ib),
        "proportional_a": float(pa_prop),
        "proportional_b": float(pb_prop),
        "power_a": float(pa_pow),
        "power_b": float(pb_pow),
        "shin_a": float(pa_shin),
        "shin_b": float(pb_shin),
        "shin_z": float(_shin_z(ia, ib)),
    }


def roi_significance(results: list[tuple[float, float, int]], *,
                     groups: Sequence[Hashable] | None = None) -> dict:
    """ROI moyen par pari, erreur type eventuellement regroupee par match.

    Avec groups, covariance sandwich CR1 d'une moyenne (constante seule).
    La p-value reste une approximation normale bilaterale ; au moins 30
    groupes sont requis pour annoncer une significativite a 5 %.
    """
    for p, odds, won in results:
        _probability(p)
        implied(odds)
        if won not in (0, 1):
            raise ValueError("le resultat d'un pari doit etre 0 ou 1")
    n = len(results)
    group_indices = None
    n_groups = n
    if groups is not None:
        if len(groups) != n:
            raise ValueError("groups doit identifier le match de chaque pari")
        labels = {}
        try:
            group_indices = [labels.setdefault(group, len(labels)) for group in groups]
        except TypeError as exc:
            raise ValueError("les identifiants de groups doivent etre hashables") from exc
        n_groups = len(labels)
    returns = np.array([(odds - 1.0) if won else -1.0 for _p, odds, won in results], dtype=float)
    mean_ret = float(returns.mean()) if n else 0.0
    std_ret = float(returns.std(ddof=1)) if n > 1 else None
    out = {
        "n": n, "n_groups": n_groups, "mean_return": mean_ret, "std_return": std_ret,
        "se_return": None, "sharpe_ratio": None, "t_stat": None, "p_value": None,
        "significant_5pct": False,
    }
    if n_groups < 2:
        return out
    if group_indices is None:
        se = std_ret / math.sqrt(n)
    else:
        scores = np.bincount(group_indices, weights=returns - mean_ret, minlength=n_groups)
        se = float(np.linalg.norm(scores) * math.sqrt(n_groups / (n_groups - 1)) / n)
    out["se_return"] = se
    if std_ret < 1e-12 or se < 1e-12:
        return out
    t_stat = mean_ret / se
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
    out.update(sharpe_ratio=float(mean_ret / std_ret), t_stat=float(t_stat),
               p_value=float(p_value), significant_5pct=bool(n_groups >= 30 and p_value < 0.05))
    return out


def expected_value(p: float, odds: float) -> float:
    """Rendement attendu par unite misee."""
    _probability(p)
    implied(odds)
    return float(p * (odds - 1.0) - (1.0 - p))


def closing_line_value(
    taken_odds: float, closing_odds_a: float, closing_odds_b: float,
    side: str = "a", *, devig_method: str = "power",
) -> dict:
    """Valeur face a la ligne de cloture (CLV), sans marge.

    Mesure a faible variance de la qualite d'un prix pris : battre la ligne de
    cloture regulierement est un signal bien plus fiable que le ROI sur peu de paris.
    """
    if side not in ("a", "b"):
        raise ValueError("side doit valoir 'a' ou 'b'")
    taken_p = implied(taken_odds)
    pa, pb = devig(closing_odds_a, closing_odds_b, method=devig_method)
    closing_p = pa if side == "a" else pb
    return {
        "closing_p": float(closing_p),
        "taken_implied": float(taken_p),
        "clv_prob": float(closing_p - taken_p),
        "clv_ev": float(expected_value(closing_p, taken_odds)),
    }


def kelly_fraction(p: float, odds: float, cap: float = 0.05, scale: float = 0.25) -> float:
    """Kelly fractionnaire et plafonne.

    Kelly plein suppose que `p` est exact. Il ne l'est jamais ici : on prend
    un quart de Kelly, plafonne, ce qui divise la variance sans effondrer la
    croissance esperee.
    """
    _probability(p)
    implied(odds)
    finite_number("cap", cap, maximum=1.0)
    finite_number("scale", scale, maximum=1.0)
    b = float(odds) - 1.0
    f = (p * b - (1.0 - p)) / b
    return float(max(0.0, min(cap, f * scale)))


@dataclass
class ValueBet:
    side: str
    model_p: float
    market_p: float
    odds: float
    edge: float
    ev: float
    stake: float

    def __str__(self) -> str:
        return (
            f"{self.side}: modele {self.model_p:.1%} vs marche {self.market_p:.1%} "
            f"@ {self.odds:.2f} -> edge {self.edge:+.1%}, EV {self.ev:+.1%}/unite, "
            f"mise {self.stake:.2%} bankroll"
        )


def find_value(
    p_a: float,
    odds_a: float,
    odds_b: float,
    team_a: str = "A",
    team_b: str = "B",
    min_edge: float = 0.03,
    *,
    devig_method: str = "power",
) -> ValueBet | None:
    """Retourne le cote qui offre de la valeur, ou None si le marche est correct."""
    _probability(p_a)
    finite_number("min_edge", min_edge, maximum=1.0)
    ma, mb = devig(odds_a, odds_b, method=devig_method)
    cands = [
        ValueBet(team_a, p_a, ma, odds_a, p_a - ma, expected_value(p_a, odds_a),
                 kelly_fraction(p_a, odds_a)),
        ValueBet(team_b, 1 - p_a, mb, odds_b, (1 - p_a) - mb, expected_value(1 - p_a, odds_b),
                 kelly_fraction(1 - p_a, odds_b)),
    ]
    best = max(cands, key=lambda c: c.edge)
    return best if best.edge >= min_edge and best.ev > 0 else None


def simulate_flat(results: list[tuple[float, float, int]], stake: float = 1.0) -> dict:
    """ROI a mise fixe. `results` = [(proba modele, cote prise, gagne 0/1)]."""
    finite_number("stake", stake, strict=True)
    for p, odds, won in results:
        _probability(p)
        implied(odds)
        if won not in (0, 1):
            raise ValueError("le resultat d'un pari doit etre 0 ou 1")
    if not results:
        return {
            "bets": 0, "roi": 0.0, "profit": 0.0, "turnover": 0.0,
            "hit_rate": 0.0, "max_drawdown": 0.0, "max_losing_streak": 0,
        }
    equity = peak = max_dd = 0.0
    streak = max_streak = 0
    for _p, odds, won in results:
        if won:
            equity += (odds - 1.0) * stake
            streak = 0
        else:
            equity -= stake
            streak += 1
            if streak > max_streak:
                max_streak = streak
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    profit = equity
    turnover = stake * len(results)
    return {
        "bets": len(results),
        "profit": float(profit),
        "turnover": float(turnover),
        "roi": float(profit / turnover),
        "hit_rate": float(np.mean([w for *_x, w in results])),
        "avg_odds": float(np.mean([o for _p, o, _w in results])),
        "max_drawdown": float(max_dd),
        "max_losing_streak": int(max_streak),
    }


def simulate_kelly(
    results: list[tuple[float, float, int]],
    initial_bankroll: float = 1.0,
    cap: float = 0.05,
    scale: float = 0.25,
) -> dict:
    """Simulation multiplicative de bankroll en Kelly fractionnaire plafonne."""
    finite_number("initial_bankroll", initial_bankroll, strict=True)
    finite_number("cap", cap, maximum=1.0)
    finite_number("scale", scale, maximum=1.0)
    for p, odds, won in results:
        _probability(p)
        implied(odds)
        if won not in (0, 1):
            raise ValueError("le resultat d'un pari doit etre 0 ou 1")
    if not results:
        return {
            "bets": 0, "final_bankroll": float(initial_bankroll), "return_pct": 0.0,
            "turnover": 0.0, "roi": 0.0, "hit_rate": 0.0,
            "max_drawdown_pct": 0.0, "max_losing_streak": 0,
        }
    bankroll = float(initial_bankroll)
    peak = bankroll
    max_dd_pct = 0.0
    turnover = 0.0
    streak = max_streak = 0
    hits = 0
    for p, odds, won in results:
        frac = kelly_fraction(p, odds, cap=cap, scale=scale)
        wager = bankroll * frac
        turnover += wager
        if won:
            bankroll += wager * (odds - 1.0)
            hits += 1
            streak = 0
        else:
            bankroll -= wager
            streak += 1
            if streak > max_streak:
                max_streak = streak
        if bankroll > peak:
            peak = bankroll
        if peak > 0:
            dd_pct = (peak - bankroll) / peak
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
    profit = bankroll - float(initial_bankroll)
    return {
        "bets": len(results),
        "final_bankroll": float(bankroll),
        "return_pct": float(profit / float(initial_bankroll)),
        "turnover": float(turnover),
        "roi": float(profit / turnover) if turnover > 1e-12 else 0.0,
        "hit_rate": float(hits / len(results)),
        "max_drawdown_pct": float(max_dd_pct),
        "max_losing_streak": int(max_streak),
    }


def equity_curve(
    results: list[tuple[float, float, int]],
    *,
    mode: str = "flat",
    stake: float = 1.0,
    initial_bankroll: float = 1.0,
    cap: float = 0.05,
    scale: float = 0.25,
) -> list[float]:
    """Trajectoire cumulee de la bankroll pari par pari (longueur n + 1)."""
    if mode not in ("flat", "kelly"):
        raise ValueError("mode doit valoir 'flat' ou 'kelly'")
    for p, odds, won in results:
        _probability(p)
        implied(odds)
        if won not in (0, 1):
            raise ValueError("le resultat d'un pari doit etre 0 ou 1")
    if mode == "flat":
        finite_number("stake", stake, strict=True)
        val = 0.0
        curve = [val]
        for _p, odds, won in results:
            val += (odds - 1.0) * stake if won else -stake
            curve.append(float(val))
        return curve
    finite_number("initial_bankroll", initial_bankroll, strict=True)
    finite_number("cap", cap, maximum=1.0)
    finite_number("scale", scale, maximum=1.0)
    val = float(initial_bankroll)
    curve = [val]
    for p, odds, won in results:
        wager = val * kelly_fraction(p, odds, cap=cap, scale=scale)
        val += wager * (odds - 1.0) if won else -wager
        curve.append(float(val))
    return curve


def bootstrap_bankroll(
    results: list[tuple[float, float, int]],
    *,
    mode: str = "kelly",
    horizon: int | None = None,
    samples: int = 500,
    stake: float = 1.0,
    initial_bankroll: float = 1.0,
    cap: float = 0.05,
    scale: float = 0.25,
    seed: int = 0,
) -> dict:
    """Reechantillonnage Monte Carlo de la bankroll, par lots de trajectoires.

    Seuls les capitaux finaux et drawdowns sont conserves pour les quantiles.
    La taille des lots limite la memoire sans changer l'ordre des tirages.
    """
    from .validation import positive_integer

    if mode not in ("flat", "kelly"):
        raise ValueError("mode doit valoir 'flat' ou 'kelly'")
    positive_integer("samples", samples)
    if not results:
        raise ValueError("results doit contenir au moins un pari")
    for p, odds, won in results:
        _probability(p)
        implied(odds)
        if won not in (0, 1):
            raise ValueError("le resultat d'un pari doit etre 0 ou 1")
    h = len(results) if horizon is None else horizon
    positive_integer("horizon", h)

    rng = np.random.default_rng(seed)
    n = len(results)
    if mode == "flat":
        finite_number("stake", stake, strict=True)
        rets = np.array([(odds - 1.0) * stake if won else -stake for _p, odds, won in results], dtype=float)
        start_value = 0.0
        drawdown_suffix = ""
    else:
        finite_number("initial_bankroll", initial_bankroll, strict=True)
        finite_number("cap", cap, maximum=1.0)
        finite_number("scale", scale, maximum=1.0)
        mults = np.array([
            1.0 + kelly_fraction(p, odds, cap=cap, scale=scale) * ((odds - 1.0) if won else -1.0)
            for p, odds, won in results
        ], dtype=float)
        start_value = float(initial_bankroll)
        drawdown_suffix = "_pct"

    finals = np.empty(samples)
    max_dds = np.empty(samples)
    # Au plus 64 trajectoires et environ 1 million de cellules par lot.
    batch_size = max(1, min(64, 1_000_000 // (h + 1)))
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        idx = rng.integers(0, n, size=(stop - start, h))
        paths = np.empty((stop - start, h + 1))
        paths[:, 0] = start_value
        if mode == "flat":
            np.cumsum(rets[idx], axis=1, out=paths[:, 1:])
        else:
            np.cumprod(mults[idx], axis=1, out=paths[:, 1:])
            paths[:, 1:] *= start_value
        finals[start:stop] = paths[:, -1]
        peaks = np.maximum.accumulate(paths, axis=1)
        drawdowns = peaks - paths
        if mode == "kelly":
            # Le pic inclut le capital initial strictement positif.
            drawdowns /= peaks
        max_dds[start:stop] = np.max(drawdowns, axis=1)
    return {
        "mode": mode, "horizon": h, "samples": samples,
        "prob_profit": float(np.mean(finals > start_value)),
        "median_final": float(np.median(finals)),
        "p05_final": float(np.quantile(finals, 0.05)),
        "p95_final": float(np.quantile(finals, 0.95)),
        "median_max_drawdown" + drawdown_suffix: float(np.median(max_dds)),
        "p95_max_drawdown" + drawdown_suffix: float(np.quantile(max_dds, 0.95)),
    }
