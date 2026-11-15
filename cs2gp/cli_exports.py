"""Formatage et exports CSV partages par l'interface en ligne de commande."""

from __future__ import annotations

from .csvio import atomic_write_csv


def export_ratings_csv(path, analyzer, board, on, top: int) -> int:
    pool = list(analyzer.fx.pool_on(on))
    return atomic_write_csv(
        path,
        ["rank", "team", "rating", "deviation", "trajectory", "maps_played", "last_date", *pool],
        (
            [
                rank, team, f"{rating:.2f}",
                f"{analyzer.fx.elo.deviation(team, on):.2f}",
                f"{analyzer.fx.elo.trajectory(team, on):.2f}",
                played,
                tr.last_date.isoformat() if (tr := analyzer.fx.elo._read_team(team)).last_date else "",
                *(f"{analyzer.fx.elo.map_rating(team, mp, on):.2f}" for mp in pool),
            ]
            for rank, (team, rating, played) in enumerate(board[:top], 1)
        ),
    )


def export_history_csv(path, timeline: list[dict]) -> int:
    return atomic_write_csv(
        path,
        [
            "date", "match_id", "opponent", "map", "won", "round_diff",
            "rating_before", "rating_after", "delta", "rd_after", "trajectory_after", "event",
        ],
        (
            [
                r["date"], r["match_id"], r["opponent"], r["map"], int(r["won"]),
                r["round_diff"], f"{r['rating_before']:.2f}", f"{r['rating_after']:.2f}",
                f"{r['delta']:+.2f}", f"{r['rd_after']:.2f}", f"{r['trajectory_after']:+.2f}",
                r["event"],
            ]
            for r in timeline
        ),
    )


def export_slate_csv(path, card) -> int:
    csv_rows = []
    for a in card:
        fx = a.fixture
        if a.picks:
            for pk in a.picks:
                csv_rows.append([
                    pk.date.isoformat(), pk.match_id, pk.team_a, pk.team_b,
                    pk.best_of, pk.event, pk.market, pk.selection,
                    f"{pk.probability:.6f}", f"{pk.fair_odds:.3f}", pk.tier,
                    pk.verdict,
                    "" if pk.odds is None else f"{pk.odds:.3f}",
                    "" if pk.edge is None else f"{pk.edge:.6f}",
                    "" if pk.ev is None else f"{pk.ev:.6f}",
                    "" if pk.stake is None else f"{pk.stake:.6f}",
                    "",
                ])
        else:
            csv_rows.append([
                fx.date.isoformat(), fx.match_id, fx.team_a, fx.team_b,
                fx.best_of, fx.event, "abstention", "", "", "", "",
                "ABSTENTION", "", "", "", "", a.skipped or "",
            ])
    return atomic_write_csv(
        path,
        [
            "date", "match_id", "team_a", "team_b", "best_of", "event",
            "market", "selection", "probability", "fair_odds", "tier",
            "verdict", "odds", "edge", "ev", "stake", "skipped",
        ],
        csv_rows,
    )


def format_movers(mv: dict) -> str:
    lines = [f"Variations Elo sur {mv['window_days']} jours ({mv['since']} -> {mv['as_of']}) :"]
    if mv["risers"]:
        lines.append("\n  En hausse :")
        for r in mv["risers"]:
            lines.append(
                f"    {r['team']:<18} {r['rating_before']:>6.0f} -> {r['rating_after']:>6.0f} "
                f"({r['delta']:+6.1f})   [{r['maps_played']} maps]"
            )
    if mv["fallers"]:
        lines.append("\n  En baisse :")
        for r in mv["fallers"]:
            lines.append(
                f"    {r['team']:<18} {r['rating_before']:>6.0f} -> {r['rating_after']:>6.0f} "
                f"({r['delta']:+6.1f})   [{r['maps_played']} maps]"
            )
    return "\n".join(lines)


def export_maps_csv(path, rows: list[dict]) -> int:
    return atomic_write_csv(
        path,
        [
            "map", "played", "ot_rate",
            "avg_round_diff", "avg_total_rounds", "elo_favorite_win_rate",
        ],
        (
            [
                r["map"], r["played"], f"{r['ot_rate']:.6f}",
                f"{r['avg_round_diff']:.4f}", f"{r['avg_total_rounds']:.4f}",
                "" if r["elo_favorite_win_rate"] is None else f"{r['elo_favorite_win_rate']:.6f}",
            ]
            for r in rows
        ),
    )


def export_events_csv(path, rows: list[dict]) -> int:
    return atomic_write_csv(
        path,
        [
            "event", "n_matches", "n_complete", "tier", "lan",
            "date_start", "date_end", "elo_upset_rate",
        ],
        (
            [
                r["event"], r["n_matches"], r["n_complete"], r["tier"], int(r["lan"]),
                r["date_start"], r["date_end"],
                "" if r["elo_upset_rate"] is None else f"{r['elo_upset_rate']:.6f}",
            ]
            for r in rows
        ),
    )


def format_team_profile(prof: dict) -> str:
    lines = [
        f"{prof['team']}   [au {prof['as_of']}]",
        f"  Elo {prof['rating']:.0f} (+/-{prof['deviation']:.0f} RD) | "
        f"trajectoire {prof['trajectory']:+.1f} | {prof['maps_played']} maps",
        f"  Forme recente : {prof['recent_win_rate']:.1%} "
        f"({prof['recent_round_diff']:+.1f} rounds, residu {prof['perf_residual']:+.1f}) | "
        f"SOS {prof['sos']:.0f} | charge 21j {prof['workload_21d']} maps | serie {prof['streak']:+d}",
    ]
    if prof["roster"]:
        lines.append(f"  Roster ({prof['maps_with_roster']} maps) : {', '.join(prof['roster'])}")
    lines.append("\n  Pool de maps :")
    for row in prof["map_pool"]:
        lines.append(
            f"    {row['map']:<10} Elo {row['rating']:>6.0f}   "
            f"{row['maps_played']:>3} maps   forme {row['recent_win_rate']:.0%}"
        )
    return "\n".join(lines)


def format_compare(cmp: dict) -> str:
    a, b = cmp["team_a"], cmp["team_b"]
    h2h = cmp["h2h"]
    lines = [
        f"Comparaison : {a['team']} vs {b['team']}   [au {cmp['as_of']}]",
        f"  Elo     : {a['rating']:.0f} (+/-{a['deviation']:.0f}) vs "
        f"{b['rating']:.0f} (+/-{b['deviation']:.0f})   "
        f"(ecart {cmp['elo_diff']:+.0f}, proba Elo {cmp['elo_win_prob_a']:.1%})",
        f"  Forme   : {a['recent_win_rate']:.0%} ({a['recent_round_diff']:+.1f}r) vs "
        f"{b['recent_win_rate']:.0%} ({b['recent_round_diff']:+.1f}r) | "
        f"charge 21j {a['workload_21d']} vs {b['workload_21d']} maps",
    ]
    stakes = cmp.get("elo_stakes")
    if stakes:
        lines.append(
            f"  Enjeu   : {a['team']} ({stakes['a_win_delta_a']:+.1f} / {stakes['a_loss_delta_a']:+.1f}) | "
            f"{b['team']} ({stakes['b_win_delta_b']:+.1f} / {stakes['b_loss_delta_b']:+.1f})"
        )
    lines.append(
        f"  H2H     : {h2h['series_a_wins']}-{h2h['series_b_wins']} series, "
        f"{h2h['maps_a_wins']}-{h2h['maps_b_wins']} maps (score {h2h['weighted_h2h_score']:+.3f})"
    )
    lines.append("\n  Ecart Elo par map (vu de " + a["team"] + ") :")
    for row in cmp["map_edges"]:
        lines.append(f"    {row['map']:<10} {row['elo_edge']:+7.1f}")
    return "\n".join(lines)
