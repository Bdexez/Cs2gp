"""Audit de qualite d'un historique CS2 avant entrainement ou backtest."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from typing import Sequence

from .schema import ACTIVE_DUTY, MapResult, Match, group_matches


def _classify_score(rounds_a: int, rounds_b: int) -> str:
    hi, lo = (rounds_a, rounds_b) if rounds_a >= rounds_b else (rounds_b, rounds_a)
    if hi == 13 and lo <= 11:
        return "mr12_regulation"
    if hi >= 16 and lo >= 12 and (hi - 12) % 3 == 1 and 2 <= (hi - lo) <= 4:
        return "mr12_overtime"
    if hi == 16 and lo < 12:
        return "mr15_suspect"
    return "other"


def audit_history(
    maps: Sequence[MapResult],
    matches: Sequence[Match] | None = None,
    *,
    warmup_maps: int = 12,
) -> dict:
    """Produit un bilan structure de la qualite des donnees chargees."""
    maps = list(maps)
    matches = list(group_matches(maps)) if matches is None else list(matches)

    team_map_counts: Counter[str] = Counter()
    for mp in maps:
        team_map_counts[mp.team_a] += 1
        team_map_counts[mp.team_b] += 1
    teams = sorted(team_map_counts)
    teams_under_warmup = sum(1 for c in team_map_counts.values() if c < warmup_maps)
    complete = [m for m in matches if m.is_complete]
    incomplete = len(matches) - len(complete)

    formats = Counter(f"bo{m.best_of}" for m in matches)
    tiers = Counter(m.tier for m in matches)
    lan_count = sum(1 for m in matches if m.lan)
    maps_by_name = Counter(mp.map_name for mp in maps)
    active = set(ACTIVE_DUTY)
    out_of_pool = {name: count for name, count in sorted(maps_by_name.items()) if name not in active}

    with_rosters = sum(1 for mp in maps if mp.roster_a and mp.roster_b)
    roster_cov = with_rosters / len(maps) if maps else 0.0

    regimes = Counter(_classify_score(mp.rounds_a, mp.rounds_b) for mp in maps)
    for key in ("mr12_regulation", "mr12_overtime", "mr15_suspect", "other"):
        regimes.setdefault(key, 0)

    dates = sorted({m.date for m in matches})
    date_min = dates[0].isoformat() if dates else None
    date_max = dates[-1].isoformat() if dates else None
    span_days = (dates[-1] - dates[0]).days if dates else 0
    max_calendar_gap = max(((b - a).days for a, b in zip(dates, dates[1:])), default=0)

    pair_days = Counter((m.date, tuple(sorted((m.team_a, m.team_b)))) for m in matches)
    same_day_collisions = sum(1 for c in pair_days.values() if c > 1)

    matches_by_year = {str(y): c for y, c in sorted(Counter(m.date.year for m in matches).items())}
    active_counts = [maps_by_name.get(m, 0) for m in ACTIVE_DUTY]
    map_balance_ratio = (
        float(min(active_counts) / max(active_counts))
        if active_counts and max(active_counts) > 0 else None
    )
    similar_teams: list[list] = []
    for i, a in enumerate(teams):
        if len(a) < 4:
            continue
        for b in teams[i + 1 :]:
            if len(b) < 4:
                continue
            ratio = SequenceMatcher(None, a, b).ratio()
            if ratio >= 0.85:
                similar_teams.append([a, b, round(ratio, 3)])

    warnings: list[str] = []
    if incomplete:
        warnings.append(
            f"{incomplete} serie(s) incomplete(s) : exclues des cibles d'entrainement "
            "mais leurs maps alimentent l'Elo."
        )
    if maps and roster_cov == 0.0:
        warnings.append("aucun roster renseigne : les changements d'effectif ne sont pas detectes.")
    elif maps and roster_cov < 0.5:
        warnings.append(f"couverture roster partielle ({roster_cov:.0%}).")
    if out_of_pool:
        names = ", ".join(f"{k} ({v})" for k, v in out_of_pool.items())
        warnings.append(f"maps hors pool Active Duty actuel : {names}.")
    if regimes["mr15_suspect"]:
        warnings.append(
            f"{regimes['mr15_suspect']} map(s) ressemblent a du MR15 (CS:GO 16-x) : "
            "verifier l'homogeneite du reglement."
        )
    if regimes["other"]:
        warnings.append(
            f"{regimes['other']} map(s) avec score non standard MR12 (forfaits ou reglement atypique)."
        )
    if max_calendar_gap > 30:
        warnings.append(
            f"trou calendaire maximal de {max_calendar_gap} jours : verifier l'absence de coupure de collecte."
        )
    if same_day_collisions:
        warnings.append(
            f"{same_day_collisions} paire(s) d'equipes se rencontrent plusieurs fois le meme jour : "
            "fournir match_id dans les fixtures pour un reglement sans ambiguite."
        )
    if len(maps) >= 50 and map_balance_ratio is not None and map_balance_ratio < 0.25:
        warnings.append(
            f"pool Active Duty tres desequilibre (ratio min/max = {map_balance_ratio:.2f})."
        )
    if similar_teams:
        sample = ", ".join(f"{a}/{b}" for a, b, _ in similar_teams[:4])
        warnings.append(
            f"{len(similar_teams)} paire(s) de noms d'equipes tres proches ({sample}) : "
            "verifier les alias manquants ou fautes de frappe."
        )

    return {
        "n_maps": len(maps),
        "n_matches": len(matches),
        "n_complete_matches": len(complete),
        "n_incomplete_matches": incomplete,
        "n_teams": len(teams),
        "teams_under_warmup": teams_under_warmup,
        "similar_team_pairs": similar_teams,
        "date_min": date_min,
        "date_max": date_max,
        "span_days": span_days,
        "max_calendar_gap_days": max_calendar_gap,
        "same_day_pair_collisions": same_day_collisions,
        "matches_by_year": matches_by_year,
        "formats": dict(sorted(formats.items())),
        "lan_matches": lan_count,
        "online_matches": len(matches) - lan_count,
        "tiers": dict(sorted(tiers.items())),
        "maps_by_name": dict(sorted(maps_by_name.items())),
        "map_balance_ratio": map_balance_ratio,
        "out_of_pool_maps": out_of_pool,
        "roster_coverage": float(roster_cov),
        "round_regimes": dict(sorted(regimes.items())),
        "warnings": warnings,
    }


def format_audit(report: dict) -> str:
    """Met en forme le bilan d'audit pour le terminal."""
    lines = [
        f"Audit historique : {report['n_maps']} maps, {report['n_matches']} matchs "
        f"({report['n_complete_matches']} complets, {report['n_incomplete_matches']} incomplets), "
        f"{report['n_teams']} equipes",
    ]
    if report["date_min"]:
        lines.append(
            f"  Periode      : {report['date_min']} -> {report['date_max']} "
            f"({report['span_days']} jours)"
        )
    fmts = ", ".join(f"{k}={v}" for k, v in report["formats"].items()) or "aucun"
    tiers = ", ".join(f"{k.upper()}={v}" for k, v in report["tiers"].items()) or "aucun"
    lines.append(f"  Formats      : {fmts}")
    lines.append(
        f"  Contexte     : LAN={report['lan_matches']}, online={report['online_matches']} | "
        f"tiers {tiers}"
    )
    lines.append(f"  Rosters      : {report['roster_coverage']:.1%} des maps renseignees")
    r = report["round_regimes"]
    lines.append(
        f"  Scores       : MR12={r['mr12_regulation']}, OT={r['mr12_overtime']}, "
        f"MR15 suspect={r['mr15_suspect']}, autres={r['other']}"
    )
    if report["warnings"]:
        lines.append("")
        lines.append("Alertes :")
        for w in report["warnings"]:
            lines.append(f"  /!\\ {w}")
    return "\n".join(lines)
