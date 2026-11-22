"""Interface en ligne de commande."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

from . import __version__
from .audit import audit_fixtures, audit_history, format_audit
from .backtest import bet_backtest, read_odds_csv, tune, walk_forward, walk_forward_importance
from .csvio import atomic_write_csv
from .features import feature_correlation_report
from .markets import build_markets, project_veto
from .metrics import format_summary, summarize
from .model import MatchPredictor
from .odds import find_value, overround
from .predict import MATCH_MODEL_WEIGHT, Analyzer
from .ratings import EloConfig
from .pronostic import (
    Fixture, Tipster, fit_decider_temperature, format_card, read_fixtures,
    shape_series, simulate_tipster,
)
from . import journal as journal_mod
from .schema import group_matches, merge_histories, parse_date, read_csv, write_csv
from .snapshot import diff_analyzers, inspect_analyzer
from .sources import simulate_history
from .validation import positive_integer

DEFAULT_DATA = "data/sample_matches.csv"


def _analyzer(args, train: bool = True) -> Analyzer:
    if getattr(args, "model", None):
        if (args.data is not None or args.warmup is not None or args.no_gbdt
                or args.elo_config or getattr(args, "versioned_pool", False)):
            raise ValueError("--model fournit deja les donnees et la configuration ; "
                             "retirer --data, --warmup, --no-gbdt, --elo-config et --versioned-pool")
        an = Analyzer.load(args.model)
        print(f"Analyseur charge depuis {args.model} : {len(an.maps)} maps, "
              f"historique jusqu'au {an.match_model.last_date}", file=sys.stderr)
        return an
    path = Path(args.data or DEFAULT_DATA)
    if not path.exists():
        sys.exit(
            f"fichier introuvable : {path}\n"
            f"  -> genere un jeu de test : cs2gp sample --out {path}\n"
            f"  -> ou fournis un CSV : match_id,date,team_a,team_b,map,rounds_a,rounds_b"
        )
    elo = EloConfig.load(args.elo_config) if getattr(args, "elo_config", None) else None
    if elo is not None:
        print(f"Constantes de notation chargees depuis {args.elo_config}", file=sys.stderr)
    an = Analyzer(
        elo_config=elo,
        warmup_maps=12 if args.warmup is None else args.warmup,
        versioned_pool=getattr(args, "versioned_pool", False),
    ).load_csv(path)
    if not an.matches:
        raise ValueError(f"aucun resultat exploitable dans {path}")
    print(
        f"{len(an.maps)} maps, {len(an.matches)} matchs, {len(an.known_teams())} equipes "
        f"({an.matches[0].date} -> {an.matches[-1].date}) | "
        f"{len(an.match_ds)} matchs exploitables apres rodage",
        file=sys.stderr,
    )
    if train:
        an.train(use_gbdt=not args.no_gbdt)
    return an


def cmd_sample(args) -> int:
    rows = simulate_history(n_teams=args.teams, years=args.years, seed=args.seed,
                            start=args.start, regime=args.regime,
                            with_rosters=args.with_rosters)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(str(out), rows)
    matches = group_matches(rows)
    print(f"{len(rows)} maps / {len(matches)} matchs simules -> {out}")
    print("Donnees synthetiques : utiles pour tester la chaine, pas pour parier.")
    return 0


def cmd_merge(args) -> int:
    sources = [read_csv(src) for src in args.inputs]
    merged, stats = merge_histories(sources)
    write_csv(args.out, merged)
    print(
        f"{stats['unique_out']} maps fusionnees -> {args.out} "
        f"({stats['total_in']} lues, {stats['duplicates_skipped']} doublon(s) ignores)"
    )
    return 0


def cmd_teams(args) -> int:
    an = _analyzer(args, train=False)
    for t in an.known_teams():
        print(f"{t:<20} {an.fx.elo.maps_played(t):>5} maps")
    return 0


def cmd_team(args) -> int:
    an = _analyzer(args, train=False)
    on = parse_date(args.date) if args.date else None
    prof = an.team_profile(args.team, on=on)
    if args.json:
        print(json.dumps(prof, indent=2, ensure_ascii=False))
        return 0
    print()
    print(f"{prof['team']}   [au {prof['as_of']}]")
    print(
        f"  Elo {prof['rating']:.0f} (+/-{prof['deviation']:.0f} RD) | "
        f"trajectoire {prof['trajectory']:+.1f} | {prof['maps_played']} maps"
    )
    print(
        f"  Forme recente : {prof['recent_win_rate']:.1%} "
        f"({prof['recent_round_diff']:+.1f} rounds, residu {prof['perf_residual']:+.1f}) | "
        f"SOS {prof['sos']:.0f} | charge 21j {prof['workload_21d']} maps | serie {prof['streak']:+d}"
    )
    if prof["roster"]:
        print(f"  Roster ({prof['maps_with_roster']} maps) : {', '.join(prof['roster'])}")
    print("\n  Pool de maps :")
    for row in prof["map_pool"]:
        print(
            f"    {row['map']:<10} Elo {row['rating']:>6.0f}   "
            f"{row['maps_played']:>3} maps   forme {row['recent_win_rate']:.0%}"
        )
    return 0


def cmd_inspect(args) -> int:
    info = inspect_analyzer(args.bundle)
    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0
    print()
    print(f"Instantane {info['format']} v{info['version']} (cs2gp {info['package_version']})")
    print(
        f"  Historique : {info['n_maps']} maps, {info['n_matches']} matchs "
        f"({info['date_min']} -> {info['date_max']})"
    )
    print(
        f"  Modele match : n={info['match_model']['trained_on']}, "
        f"T={info['match_model']['temperature']:.3f}, GBDT={info['has_gbdt']}"
    )
    print(
        f"  Modele map   : n={info['map_model']['trained_on']}, "
        f"T={info['map_model']['temperature']:.3f}, calibration_cache={info['has_calibration_history']}"
    )
    print(
        f"  Options : warmup={info['warmup_maps']}, versioned_pool={info['versioned_pool']}, "
        f"momentum={info['momentum']:+.3f} (auto={info['momentum_auto']})"
    )
    return 0


def cmd_diff(args) -> int:
    delta = diff_analyzers(args.bundle_a, args.bundle_b)
    if args.json:
        print(json.dumps(delta, indent=2, ensure_ascii=False))
        return 0
    print()
    print(f"Comparaison d'instantanes : {args.bundle_a} -> {args.bundle_b}")
    print(f"  Historique  : {delta['delta_maps']:+d} maps, {delta['delta_matches']:+d} matchs")
    print(f"  Periode A   : {delta['date_range_a'][0]} -> {delta['date_range_a'][1]}")
    print(f"  Periode B   : {delta['date_range_b'][0]} -> {delta['date_range_b'][1]}")
    print(
        f"  Temperature : match {delta['match_temperature_delta']:+.3f} | "
        f"map {delta['map_temperature_delta']:+.3f} | momentum {delta['momentum_delta']:+.3f}"
    )
    if delta["elo_changes"]:
        print(f"  Elo modifie : {', '.join(delta['elo_changes'])}")
    if delta["training_option_changes"]:
        print(f"  Options     : {', '.join(delta['training_option_changes'])}")
    return 0


def cmd_h2h(args) -> int:
    an = _analyzer(args, train=False)
    on = parse_date(args.date) if args.date else None
    h2h = an.head_to_head(args.team_a, args.team_b, on=on)
    if args.json:
        print(json.dumps(h2h, indent=2, ensure_ascii=False))
        return 0
    print()
    print(f"Confrontations directes : {h2h['team_a']} vs {h2h['team_b']}   [au {h2h['as_of']}]")
    print(
        f"  Series : {h2h['series_a_wins']}-{h2h['series_b_wins']} ({h2h['series_total']} completes) | "
        f"Maps : {h2h['maps_a_wins']}-{h2h['maps_b_wins']} ({h2h['maps_total']} jouees) | "
        f"score pondere {h2h['weighted_h2h_score']:+.3f}"
    )
    if h2h["by_map"]:
        print("\n  Par map :")
        for mp, stats in h2h["by_map"].items():
            print(
                f"    {mp:<10} {stats['a_wins']}-{stats['b_wins']}   "
                f"({stats['played']} maps, diff rounds {stats['round_diff_a']:+d})"
            )
    if h2h["recent_matches"]:
        print("\n  Dernieres series :")
        for item in h2h["recent_matches"][-5:]:
            print(f"    {item['date']}  bo{item['best_of']}  {item['score']}  [{item['event']}]")
    return 0


def cmd_history(args) -> int:
    if args.last is not None:
        positive_integer("last", args.last)
    an = _analyzer(args, train=False)
    timeline = an.team_history(args.team)
    shown = timeline[-args.last :] if args.last is not None else timeline
    if args.export_csv:
        written = atomic_write_csv(
            args.export_csv,
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
                for r in shown
            ),
        )
        print(f"{written} maps exportees -> {args.export_csv}", file=sys.stderr)
    if args.json:
        print(json.dumps({"team": an.resolve_team(args.team), "n_maps": len(shown), "history": shown},
                         indent=2, ensure_ascii=False))
        return 0
    print()
    print(f"Historique Elo de {an.resolve_team(args.team)} ({len(shown)}/{len(timeline)} maps) :")
    for r in shown[-20:]:
        outcome = "W" if r["won"] else "L"
        print(
            f"  {r['date']}  {outcome}  vs {r['opponent']:<16} {r['map']:<8} "
            f"{r['round_diff']:+3d}r  Elo {r['rating_after']:>6.1f} ({r['delta']:+5.1f}) "
            f"RD {r['rd_after']:>5.1f}"
        )
    return 0


def cmd_audit(args) -> int:
    an = _analyzer(args, train=False)
    report = audit_history(an.maps, an.matches)
    feat_report = (
        feature_correlation_report(an.match_ds)
        if an.match_ds is not None and len(an.match_ds) >= 2 else None
    )
    fx_report = audit_fixtures(read_fixtures(args.fixtures), an) if args.fixtures else None
    if args.json:
        payload = dict(report)
        if feat_report is not None:
            payload["features"] = feat_report
        if fx_report is not None:
            payload["fixtures"] = fx_report
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    print()
    print(format_audit(report))
    if feat_report is not None and (feat_report["constant_features"] or feat_report["high_pairs"]):
        print("\nFeatures (dataset match) :")
        if feat_report["constant_features"]:
            print(f"  Constantes : {', '.join(feat_report['constant_features'])}")
        if feat_report["high_pairs"]:
            pairs = ", ".join(f"{p['a']}/{p['b']} ({p['corr']:+.2f})" for p in feat_report["high_pairs"][:5])
            print(f"  Fortes correlations : {pairs}")
    if fx_report is not None:
        print(
            f"\nAudit fixtures : {fx_report['n_fixtures']} affiches "
            f"({fx_report['with_odds']} avec cotes) | "
            f"statut {'OK' if fx_report['ok'] else 'A VERIFIER'}"
        )
        for w in fx_report["warnings"]:
            print(f"  /!\\ {w}")
    return 0


def cmd_ratings(args) -> int:
    positive_integer("top", args.top)
    if args.min_maps < 0:
        raise ValueError("min-maps doit etre positif ou nul")
    an = _analyzer(args, train=False)
    on = max(m.date for m in an.matches)
    board = an.fx.elo.leaderboard(
        on=on, min_maps=args.min_maps, active_within_days=args.active_within
    )
    if args.export_csv:
        pool = list(an.fx.pool_on(on))
        written = atomic_write_csv(
            args.export_csv,
            ["rank", "team", "rating", "deviation", "trajectory", "maps_played", "last_date", *pool],
            (
                [
                    rank, team, f"{rating:.2f}",
                    f"{an.fx.elo.deviation(team, on):.2f}",
                    f"{an.fx.elo.trajectory(team, on):.2f}",
                    played,
                    tr.last_date.isoformat() if (tr := an.fx.elo._read_team(team)).last_date else "",
                    *(f"{an.fx.elo.map_rating(team, mp, on):.2f}" for mp in pool),
                ]
                for rank, (team, rating, played) in enumerate(board[: args.top], 1)
            ),
        )
        print(f"{written} equipes exportees -> {args.export_csv}", file=sys.stderr)
    if args.json:
        rows = []
        for rank, (team, rating, played) in enumerate(board[: args.top], 1):
            tr = an.fx.elo.teams[team]
            best = sorted(
                ((m, an.fx.elo.map_rating(team, m, on)) for m in tr.map_counts if tr.map_counts[m] >= 3),
                key=lambda kv: -kv[1],
            )[:3]
            rows.append({
                "rank": rank,
                "team": team,
                "rating": rating,
                "maps_played": played,
                "best_maps": [{"map": m, "rating": r} for m, r in best],
            })
        print(json.dumps({"as_of": on.isoformat(), "min_maps": args.min_maps, "ratings": rows},
                         indent=2, ensure_ascii=False))
        return 0
    print(f"Classement Elo au {on} (min {args.min_maps} maps)\n")
    print(f"{'#':<4}{'equipe':<20}{'elo':>8}{'maps':>7}   meilleures maps")
    for i, (team, rating, played) in enumerate(board[: args.top], 1):
        tr = an.fx.elo.teams[team]
        best = sorted(
            ((m, an.fx.elo.map_rating(team, m, on)) for m in tr.map_counts if tr.map_counts[m] >= 3),
            key=lambda kv: -kv[1],
        )[:3]
        tail = ", ".join(f"{m} {r:.0f}" for m, r in best)
        print(f"{i:<4}{team:<20}{rating:>8.0f}{played:>7}   {tail}")
    return 0


def cmd_train(args) -> int:
    an = _analyzer(args)
    ds = an.match_ds
    p_in = an.match_model.predict_proba(ds.X)
    print("\nAjustement en echantillon (optimiste par construction, sert de controle) :")
    print(format_summary("modele", summarize(ds.y, p_in)))
    print("\nPoids appris (features standardisees, vus de l'equipe A ; * = |z| > 1.96) :")
    for row in an.match_model.logistic.feature_diagnostics():
        w = row["weight"]
        bar = "#" * min(24, int(abs(w) * 10))
        z_str = "   n/a" if row["z_score"] is None else f"{row['z_score']:+6.2f}"
        star = "*" if row["significant"] else " "
        print(f"  {row['feature']:<22} {w:+.4f}  z={z_str} {star}  {bar}")
    print(f"\nTemperature de calibration : {an.match_model.temperature:.3f}")
    print(f"GBDT actif : {an.match_model.gbdt is not None}")
    out = Path(args.bundle or args.out or "models/match.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.bundle:
        an.save(out)
        print(f"\nAnalyseur autonome enregistre -> {out}")
    else:
        an.match_model.save(out)
        print(f"\nModele match enregistre -> {out}")
    print("Pour une mesure honnete de la performance : cs2gp backtest")
    return 0


def cmd_backtest(args) -> int:
    an = _analyzer(args, train=False)
    res = walk_forward(
        an.match_ds,
        map_ds=None if args.no_maps else an.map_ds,
        initial_frac=args.initial_frac,
        refit_every=args.refit_every,
        blend=args.blend,
        bootstrap_samples=args.bootstrap_samples,
        block_days=args.block_days,
        use_gbdt=not args.no_gbdt,
    )
    if args.export_csv:
        written = res.export_csv(args.export_csv)
        print(f"{written} predictions hors-echantillon exportees -> {args.export_csv}", file=sys.stderr)
    odds_summary = (
        bet_backtest(res, read_odds_csv(args.odds_csv), min_edge=args.min_edge)
        if args.odds_csv else None
    )
    importance = (
        walk_forward_importance(
            an.match_ds,
            initial_frac=args.initial_frac,
            refit_every=args.refit_every,
            use_gbdt=not args.no_gbdt,
        )
        if getattr(args, "importance", False) else None
    )
    if args.json:
        payload = res.to_dict()
        if args.by_year:
            payload["by_year"] = {
                str(y): summarize(res.y[idx], res.predictions[idx])
                for y in sorted({d.year for d in res.dates})
                if len(idx := [i for i, d in enumerate(res.dates) if d.year == y]) >= 20
            }
        if args.by_segment:
            payload["by_segment"] = {
                "format": res.breakdown(lambda m: f"bo{m.get('best_of', '?')}"),
                "context": res.breakdown(lambda m: "LAN" if m.get("lan") else "online"),
                "tier": res.breakdown(lambda m: f"tier {str(m.get('tier', '?')).upper()}"),
            }
        if args.by_fold:
            payload["by_fold"] = res.drift_report()
        if odds_summary is not None:
            payload["bet_backtest"] = odds_summary
        if importance is not None:
            payload["importance"] = importance
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    print()
    print(res.report())
    print(
        "\nLecture : `skill` = reduction du log-loss face au pile ou face. "
        "En CS2 pro, 0.10-0.20 est deja bon ; au-dela, suspecter une fuite de donnees.\n"
        "Le backtest applique le meme melange et les memes garde-fous que `predict`."
    )
    if args.by_year:
        print("\nPar annee :")
        years = sorted({d.year for d in res.dates})
        for y in years:
            idx = [i for i, d in enumerate(res.dates) if d.year == y]
            if len(idx) < 20:
                continue
            s = summarize(res.y[idx], res.predictions[idx])
            print(format_summary(f"  {y}", s))
    if args.by_segment:
        for title, key in (
            ("Par format", lambda m: f"bo{m.get('best_of', '?')}"),
            ("Par contexte", lambda m: "LAN" if m.get("lan") else "online"),
            ("Par tier", lambda m: f"tier {str(m.get('tier', '?')).upper()}"),
        ):
            block = res.format_breakdown(title, key)
            if block:
                print(block)
    if args.by_fold:
        print("\nPar fold (derive temporelle) :")
        for row in res.drift_report():
            slope = "n/a" if row["calibration_slope"] is None else f"{row['calibration_slope']:.2f}"
            print(
                f"  fold {row['fold']:<3} {row['date_start']} -> {row['date_end']} "
                f"n={row['n']:<4} logloss={row['log_loss']:.4f} "
                f"gain_elo={row['gain_vs_elo']:+.4f} ece={row['ece']:.3f} pente={slope}"
            )
    if odds_summary is not None:
        print("\nConfrontation aux cotes historiques :")
        vig_txt = (
            f" (marge moy. {odds_summary['avg_overround'] - 1:.1%})"
            if odds_summary.get("avg_overround") is not None else ""
        )
        sig = odds_summary.get("significance") or {}
        p_txt = f" (p={sig['p_value']:.3f})" if sig.get("p_value") is not None else ""
        print(
            f"  {odds_summary['bets']} paris sur {odds_summary['matches_with_odds']} matchs cotes{vig_txt} | "
            f"reussite {odds_summary['hit_rate']:.1%} | "
            f"profit {odds_summary['profit']:+.2f} u | ROI {odds_summary['roi']:+.1%}{p_txt} | "
            f"drawdown max {odds_summary['max_drawdown']:.2f} u"
        )
        kelly = odds_summary.get("kelly")
        if kelly and kelly["bets"]:
            print(
                f"  Kelly : rendement {kelly['return_pct']:+.1%} | "
                f"ROI {kelly['roi']:+.1%} | drawdown max {kelly['max_drawdown_pct']:.1%}"
            )
        if odds_summary.get("clv_bets"):
            print(
                f"  CLV ({odds_summary['clv_bets']} paris) : "
                f"ecart proba {odds_summary['avg_clv_prob']:+.2%} | "
                f"EV cloture {odds_summary['avg_clv_ev']:+.2%}"
            )
    if importance is not None:
        print("\nImportance hors-echantillon par permutation (delta log-loss) :")
        for row in importance[:10]:
            print(f"  {row['feature']:<22} {row['mean_delta']:+.5f}  (std folds {row['fold_std']:.5f})")
    return 0


def cmd_predict(args) -> int:
    if (args.odds_a is None) != (args.odds_b is None):
        raise ValueError("--odds-a et --odds-b doivent etre fournis ensemble")
    an = _analyzer(args)
    on = parse_date(args.date) if args.date else None
    pred = an.predict(args.team_a, args.team_b, best_of=args.bo, lan=args.lan, on=on)
    market_lines = None
    if getattr(args, "markets", False):
        past = an.as_of(pred.date)
        temp = 1.0 if getattr(args, "no_calibration", False) else fit_decider_temperature(past, momentum=past.momentum)
        veto = project_veto([(e.map_name, e.p_a) for e in pred.map_edges], pred.best_of)
        dist = shape_series(veto, pred.best_of, pred.p_a, past.momentum, temp)
        market_lines = build_markets(dist, pred.team_a, pred.team_b)
    if args.json:
        payload = pred.to_dict()
        if market_lines is not None:
            payload["markets"] = [
                {
                    "market": line.market,
                    "selection": line.selection,
                    "probability": line.probability,
                    "fair_odds": line.fair_odds,
                }
                for line in market_lines
            ]
        if args.odds_a is not None and args.odds_b is not None:
            bet = find_value(
                pred.p_a, args.odds_a, args.odds_b, pred.team_a, pred.team_b, min_edge=args.min_edge
            )
            payload["market"] = {
                "odds_a": args.odds_a,
                "odds_b": args.odds_b,
                "overround": overround(args.odds_a, args.odds_b),
                "value_bet": None if bet is None else {
                    "side": bet.side, "model_p": bet.model_p, "market_p": bet.market_p,
                    "odds": bet.odds, "edge": bet.edge, "ev": bet.ev, "stake": bet.stake,
                },
            }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    print()
    print(pred.report())
    if market_lines is not None:
        print("\n  Marches derives (cotes justes) :")
        for line in market_lines:
            print(f"    {line.market:<22} {line.selection:<16} {line.probability:>6.1%}   cote {line.fair_odds:.2f}")

    if args.odds_a and args.odds_b:
        print()
        print(f"  Marche : {args.odds_a:.2f} / {args.odds_b:.2f} "
              f"(marge {overround(args.odds_a, args.odds_b) - 1:.1%})")
        bet = find_value(
            pred.p_a, args.odds_a, args.odds_b, pred.team_a, pred.team_b, min_edge=args.min_edge
        )
        if bet is None:
            print(f"  Aucun ecart >= {args.min_edge:.0%} avec le marche : pas de valeur.")
        else:
            print(f"  {bet}")
            if abs(bet.edge) > 0.25:
                print(
                    "  /!\\ ecart enorme avec le marche. Dans 9 cas sur 10 cela veut dire\n"
                    "      que le marche sait quelque chose que les resultats passes ignorent\n"
                    "      (remplacant, roster fraichement change, forfait), pas que le modele\n"
                    "      a trouve une faille. Verifier la compo avant toute conclusion."
                )
            if pred.confidence == "faible":
                print("  /!\\ confiance faible sur cette prediction : l'ecart est probablement du bruit.")
    return 0


def _tipster(an: Analyzer, args) -> Tipster:
    # Momentum inter-maps : estime sur les donnees, applique seulement s'il est
    # significatif (voir Analyzer.apply_estimated_momentum). Un flag permet de le
    # forcer a zero ou a une valeur mesuree ailleurs.
    mode = getattr(args, "momentum", "auto")
    if mode == "auto":
        diag = an.apply_estimated_momentum()
        m = diag.get("momentum")
        if an.momentum:
            print(
                f"Momentum inter-maps : {an.momentum:+.3f} en logit (significatif, "
                f"n={diag.get('n_total', 0)}) - applique.",
                file=sys.stderr,
            )
        elif m is not None:
            print(
                f"Momentum inter-maps mesure {m:+.3f} mais non significatif "
                f"(+/-{2 * diag.get('erreur_type', 0):.3f}) : laisse a zero.",
                file=sys.stderr,
            )
    else:
        momentum = float(mode)
        if not math.isfinite(momentum):
            raise ValueError("le momentum doit etre fini")
        an.momentum = momentum
        an._momentum_auto = False
        an._snapshots.clear()
        if an.momentum:
            print(f"Momentum inter-maps force a {an.momentum:+.3f} en logit.", file=sys.stderr)

    temp = (
        1.0 if getattr(args, "no_calibration", False)
        else fit_decider_temperature(an, momentum=an.momentum)
    )
    print(f"Calibration de la longueur de serie : T = {temp:.3f}", file=sys.stderr)
    kw = {}
    if getattr(args, "edge_min", None) is not None:
        kw["edge_min"] = args.edge_min
    if getattr(args, "abstention", None) is not None:
        kw["abstention"] = args.abstention
    return Tipster(an, momentum=an.momentum, decider_temperature=temp,
                  calibrate_decider=not getattr(args, "no_calibration", False), **kw)


def cmd_fixtures(args) -> int:
    """Genere une liste de matchs a venir a partir des equipes les mieux notees."""
    import csv
    import itertools

    if args.teams < 2:
        raise ValueError("teams doit etre au moins 2")
    positive_integer("count", args.count)
    an = _analyzer(args, train=False)
    on = args.date or max(dt.date.today(), max(m.date for m in an.matches))
    board = [t for t, _r, _n in an.fx.elo.leaderboard(on=on, min_maps=40)][: args.teams]
    pairs = list(itertools.combinations(board, 2))[: args.count]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["date", "team_a", "team_b", "best_of", "lan", "event",
                    "odds_a", "odds_b", "odds_over", "odds_under",
                    "odds_a_sweep", "odds_b_sweep"])
        for i, (a, b) in enumerate(pairs):
            w.writerow([
                (on + dt.timedelta(days=1 + i // 3)).isoformat(), a, b, 3, 1,
                "Exemple", "", "", "", "", "", "",
            ])
    print(f"{len(pairs)} matchs a venir -> {out}")
    print("Colonnes de cotes laissees vides : le pronostiqueur repondra "
          "'qui gagne'.\nLes remplir bascule en mode marche ('ou le marche se trompe').")
    return 0


def cmd_slate(args) -> int:
    an = _analyzer(args)
    fixtures = read_fixtures(args.fixtures)
    card = _tipster(an, args).card(fixtures)
    added = journal_mod.record_card(card, args.record) if args.record else None
    if args.export_csv:
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
        written = atomic_write_csv(
            args.export_csv,
            [
                "date", "match_id", "team_a", "team_b", "best_of", "event",
                "market", "selection", "probability", "fair_odds", "tier",
                "verdict", "odds", "edge", "ev", "stake", "skipped",
            ],
            csv_rows,
        )
        print(f"{written} ligne(s) de carte exportee(s) -> {args.export_csv}", file=sys.stderr)
    if args.json:
        payload = {
            "analyses": [a.to_dict() for a in card],
            "recorded": added,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    print()
    print(format_card(card, verbose=args.verbose))
    if args.record:
        print(f"\n{added} ligne(s) archivee(s) dans {args.record} (pronostics et abstentions).")
        print("Ils pourront etre pointes apres coup : cs2gp settle --journal "
              f"{args.record}")
    return 0


def cmd_settle(args) -> int:
    an = _analyzer(args, train=False)
    stats = journal_mod.settle(args.journal, an.matches, window_days=args.window_days)
    print(f"Pointage : {stats['pointes']} nouveau(x) resultat(s), "
          f"{stats['en_attente']} en attente, "
          f"{stats['introuvables']} introuvable(s), {stats['ambigus']} ambigu(s).")
    return cmd_journal(args)


def cmd_journal(args) -> int:
    since = getattr(args, "since", None)
    until = getattr(args, "until", None)
    record = journal_mod.scorecard(args.journal, since=since, until=until)
    money = journal_mod.financial_summary(args.journal, since=since, until=until)
    if getattr(args, "json", False):
        print(json.dumps({
            "scorecard": None if record is None else record.to_dict(),
            "financial": money,
        }, indent=2, ensure_ascii=False))
        return 0
    if record is None:
        print("Aucun pronostic pointe pour le moment.")
        return 0
    print()
    print(record.report())
    if money:
        print()
        print("Bilan financier (uniquement les lignes ou une cote a ete enregistree) :")
        print(f"  {money['paris']} paris, cote moyenne {money['cote_moyenne']:.2f}, "
              f"reussite {money['taux_reussite']:.1%}")
        p_val = f", p={money['roi_p_value']:.3f}" if money.get("roi_p_value") is not None else ""
        print(f"  mise fixe : {money['profit_mise_fixe']:+.2f} unites, "
              f"ROI {money['roi_mise_fixe']:+.1%}{p_val}, "
              f"drawdown max {money['max_drawdown_mise_fixe']:.2f} u, "
              f"pire serie {money['pire_serie_defaites']} defaites")
        print(f"  Kelly     : {money['profit_kelly']:+.3f} pour "
              f"{money['mise_kelly_totale']:.2f} misees, ROI {money['roi_kelly']:+.1%}")
        print("  Sur moins de ~200 paris, ce ROI est domine par le hasard : "
              "ne pas en tirer de conclusion.")
    return 0


def cmd_tipster_backtest(args) -> int:
    an = _analyzer(args, train=False)
    rec = simulate_tipster(
        an.match_ds, an.map_ds,
        initial_frac=args.initial_frac,
        refit_every=args.refit_every,
        abstention=args.abstention,
        momentum=args.momentum,
        use_gbdt=not args.no_gbdt,
    )
    if getattr(args, "export_csv", None):
        written = rec.export_csv(args.export_csv)
        print(f"{written} pronostics regles exportes -> {args.export_csv}", file=sys.stderr)
    if args.json:
        print(json.dumps(rec.to_dict(), indent=2, ensure_ascii=False))
        return 0
    print()
    print(rec.report())
    print(
        "\nCe bilan repond a 'quand cet outil parle, qu'est-ce que ca vaut',\n"
        "pas a 'mes probabilites sont-elles justes' (voir `cs2gp backtest`).\n"
        "Un ecart n'est un defaut que s'il depasse sa marge (deux erreurs types) :\n"
        "ces lignes-la sont marquees '<- ecart reel, a corriger' et signalent un\n"
        "palier ou un marche qui ment a son lecteur. En dessous, l'ecart est du\n"
        "bruit d'echantillonnage et ne demontre rien."
    )
    return 0


def cmd_tune(args) -> int:
    an = _analyzer(args, train=False)
    if not args.json:
        print(
            f"\nReglage temporel avec {args.holdout:.0%} de fin d'historique reservee, "
            "frontieres alignees sur les journees.\n"
            "Les constantes par defaut sont ajustees sur des mondes simules realistes ;\n"
            "le bon reglage depend de la vitesse a laquelle bouge le niveau des equipes,\n"
            "qui n'est pas la meme selon la periode et le circuit couverts par vos donnees.\n"
        )
    cfg, report = tune(
        an.matches, holdout_frac=args.holdout, warmup_maps=an.warmup_maps, verbose=not args.json
    )
    saved = False
    out = Path(args.out)
    if report["gain_reel"] > 0.0005:
        out.parent.mkdir(parents=True, exist_ok=True)
        cfg.save(out)
        saved = True
    if args.json:
        print(json.dumps({"config": cfg.to_dict(), "saved": saved, "out": str(out), "report": report},
                         indent=2, ensure_ascii=False))
        return 0
    print(f"\n{report['n_matchs_reglage']} matchs de reglage ; "
          f"{report['n_matchs_validation']} matchs de validation "
          f"({report['validation_debut']} -> {report['validation_fin']}).")
    print(
        f"\nSur la periode de reglage  : {report['logloss_reglage_avant']:.4f} "
        f"-> {report['logloss_reglage_apres']:.4f}"
    )
    print(
        f"Sur la periode de validation : {report['logloss_validation_avant']:.4f} "
        f"-> {report['logloss_validation_apres']:.4f}   "
        f"(gain reel {report['gain_reel']:+.4f})"
    )
    if not saved:
        print(
            "\nLe gain ne survit pas a la validation : gardez les constantes par defaut.\n"
            "C'est le resultat le plus frequent quand l'historique est court."
        )
        return 0
    print(f"\nConstantes enregistrees -> {out}")
    print(f"Pour les utiliser : cs2gp --elo-config {out} backtest")
    return 0


def _common_options(parser: argparse.ArgumentParser) -> None:
    # Une option absente ne doit pas ecraser celle lue avant la sous-commande.
    parser.add_argument("--data", default=argparse.SUPPRESS,
                        help=f"CSV de resultats (defaut {DEFAULT_DATA})")
    parser.add_argument("--warmup", type=int, default=argparse.SUPPRESS,
                        help="maps minimum par equipe (defaut 12)")
    parser.add_argument("--no-gbdt", action="store_true", default=argparse.SUPPRESS,
                        help="regression logistique seule")
    parser.add_argument("--elo-config", default=argparse.SUPPRESS,
                        help="JSON de constantes de notation (voir `tune`)")
    parser.add_argument("--versioned-pool", action="store_true", default=argparse.SUPPRESS,
                        help="utilise le pool Active Duty historique a la date de chaque match")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cs2gp",
        description="Analyse de matchs CS2 professionnels et estimation des probabilites de victoire.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exemples :\n"
            "  cs2gp sample                       # genere un jeu de donnees de test\n"
            "  cs2gp backtest --by-year           # mesure la performance reelle\n"
            "  cs2gp ratings --top 15             # classement Elo courant\n"
            "  cs2gp predict vitality spirit --bo 3 --lan\n"
            "  cs2gp predict navi g2 --odds-a 1.75 --odds-b 2.10\n"
            "\n"
            "pronostiqueur :\n"
            "  cs2gp fixtures --out data/fixtures.csv    # matchs a venir (exemple)\n"
            "  cs2gp slate --fixtures data/fixtures.csv --record data/journal.csv\n"
            "  cs2gp settle --journal data/journal.csv   # pointe les resultats\n"
            "  cs2gp tipster-backtest                    # le bilan historique\n"
        ),
    )
    p.add_argument("--version", action="version", version=f"cs2gp {__version__}")

    # Options communes, acceptees avant comme apres la sous-commande.
    common = argparse.ArgumentParser(add_help=False)
    _common_options(common)
    _common_options(p)
    p.set_defaults(data=None, warmup=None, no_gbdt=False, elo_config=None, versioned_pool=False)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="genere un historique synthetique", parents=[common])
    s.add_argument("--out", default=DEFAULT_DATA)
    s.add_argument("--teams", type=int, default=32)
    s.add_argument("--years", type=float, default=3.0)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--start", type=parse_date, help="date initiale fixe pour reproduire le jeu")
    s.add_argument("--regime", choices=("facile", "realiste"), default="realiste",
                   help="regime du generateur (defaut realiste)")
    s.add_argument("--with-rosters", action="store_true",
                   help="genere des effectifs de 5 joueurs renouveles lors des remaniements")
    s.set_defaults(func=cmd_sample)

    s = sub.add_parser("merge", help="fusionne plusieurs CSV d'historique sans doublons ni conflits")
    s.add_argument("inputs", nargs="+", help="fichiers CSV source")
    s.add_argument("--out", required=True, help="CSV fusionne de sortie")
    s.set_defaults(func=cmd_merge)

    s = sub.add_parser("teams", parents=[common], help="liste les equipes connues")
    s.set_defaults(func=cmd_teams)

    s = sub.add_parser("team", parents=[common], help="profil detaille d'une equipe")
    s.add_argument("team")
    s.add_argument("--date", help="date de reference (defaut : dernier match connu)")
    s.add_argument("--json", action="store_true", help="sortie JSON structuree")
    s.set_defaults(func=cmd_team)

    s = sub.add_parser("inspect", help="verifie et affiche les metadonnees d'un instantane .cs2gp")
    s.add_argument("bundle", help="chemin vers l'archive .cs2gp")
    s.add_argument("--json", action="store_true", help="sortie JSON structuree")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("diff", help="compare deux instantanes .cs2gp sans unpickler le GBDT")
    s.add_argument("bundle_a", help="premier instantane .cs2gp")
    s.add_argument("bundle_b", help="second instantane .cs2gp")
    s.add_argument("--json", action="store_true", help="sortie JSON structuree")
    s.set_defaults(func=cmd_diff)

    s = sub.add_parser("h2h", parents=[common], help="bilan des confrontations directes entre deux equipes")
    s.add_argument("team_a")
    s.add_argument("team_b")
    s.add_argument("--date", help="exclut les matchs a partir de cette date")
    s.add_argument("--json", action="store_true", help="sortie JSON structuree")
    s.set_defaults(func=cmd_h2h)

    s = sub.add_parser("history", parents=[common], help="evolution chronologique de l'Elo d'une equipe")
    s.add_argument("team")
    s.add_argument("--last", type=int, default=None, help="limite aux N dernieres maps")
    s.add_argument("--export-csv", help="exporte la serie temporelle en CSV")
    s.add_argument("--json", action="store_true", help="sortie JSON structuree")
    s.set_defaults(func=cmd_history)

    s = sub.add_parser("audit", parents=[common], help="controle la qualite des donnees chargees")
    s.add_argument("--fixtures", help="CSV de matchs a venir a verifier avant prediction")
    s.add_argument("--json", action="store_true", help="sortie JSON structuree")
    s.set_defaults(func=cmd_audit)

    s = sub.add_parser("ratings", parents=[common], help="classement Elo et forces par map")
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--min-maps", type=int, default=20)
    s.add_argument("--active-within", type=int, default=None,
                   help="exclut les equipes inactives depuis plus de N jours")
    s.add_argument("--export-csv", help="exporte le classement et les notes par map en CSV")
    s.add_argument("--json", action="store_true", help="sortie JSON structuree")
    s.set_defaults(func=cmd_ratings)

    s = sub.add_parser("train", parents=[common], help="entraine et sauvegarde le modele")
    target = s.add_mutually_exclusive_group()
    target.add_argument("--out", help="modele match JSON seul (defaut models/match.json)")
    target.add_argument("--bundle", help="instantane autonome de l'analyseur, reutilisable avec --model")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("backtest", parents=[common], help="validation walk-forward (la seule qui compte)")
    s.add_argument("--initial-frac", type=float, default=0.4)
    s.add_argument("--refit-every", type=int, default=150)
    s.add_argument("--by-year", action="store_true")
    s.add_argument("--by-segment", action="store_true",
                   help="ventile les metriques par format, LAN/online et tier")
    s.add_argument("--by-fold", action="store_true",
                   help="affiche l'evolution du log-loss et de la calibration par fold")
    s.add_argument("--importance", action="store_true",
                   help="calcule l'importance hors-echantillon par permutation")
    s.add_argument("--export-csv", help="exporte les predictions hors-echantillon en CSV")
    s.add_argument("--odds-csv", help="CSV de cotes historiques (match_id,odds_a,odds_b)")
    s.add_argument("--min-edge", type=float, default=0.04,
                   help="ecart minimal avec le marche pour --odds-csv (defaut 0.04)")
    s.add_argument("--blend", type=float, default=MATCH_MODEL_WEIGHT,
                   help="poids du modele match dans le melange (1 = lui seul)")
    s.add_argument("--no-maps", action="store_true",
                   help="n'evaluer que le modele match")
    s.add_argument("--bootstrap-samples", type=int, default=1000,
                   help="tirages pour l'IC apparie face a Elo (defaut 1000, 0 desactive)")
    s.add_argument("--block-days", type=int, default=7,
                   help="taille des blocs calendaires de l'IC (defaut 7 jours)")
    s.add_argument("--json", action="store_true", help="sortie JSON structuree")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("predict", parents=[common], help="probabilites pour un match a venir")
    s.add_argument("--model", help="instantane d'analyseur produit par train --bundle")
    s.add_argument("team_a")
    s.add_argument("team_b")
    s.add_argument("--bo", type=int, default=3, choices=[1, 3, 5])
    s.add_argument("--lan", action="store_true")
    s.add_argument("--date", help="date du match (defaut : aujourd'hui ; passe reconstruit si necessaire)")
    s.add_argument("--odds-a", type=float, help="cote decimale sur team_a")
    s.add_argument("--odds-b", type=float, help="cote decimale sur team_b")
    s.add_argument("--min-edge", type=float, default=0.04)
    s.add_argument("--markets", action="store_true", help="affiche tous les marches derives et cotes justes")
    s.add_argument("--no-calibration", action="store_true",
                   help="desactive la calibration de longueur de serie pour --markets")
    s.add_argument("--json", action="store_true", help="sortie JSON structuree")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("fixtures", parents=[common],
                       help="genere une liste de matchs a venir (exemple)")
    s.add_argument("--out", default="data/fixtures.csv")
    s.add_argument("--teams", type=int, default=8)
    s.add_argument("--count", type=int, default=10)
    s.add_argument("--date", type=parse_date, help="date initiale du calendrier (defaut aujourd'hui)")
    s.set_defaults(func=cmd_fixtures)

    s = sub.add_parser("slate", parents=[common],
                       help="la carte de pronostics des matchs a venir")
    s.add_argument("--model", help="instantane d'analyseur produit par train --bundle")
    s.add_argument("--fixtures", required=True, help="CSV des matchs a venir")
    s.add_argument("--record", help="archive les pronostics dans ce journal")
    s.add_argument("--verbose", action="store_true", help="affiche la loi du score")
    s.add_argument("--no-calibration", action="store_true",
                   help="desactive la calibration de la longueur de serie")
    s.add_argument("--momentum", default="auto",
                   help="effet d'enchainement entre maps : 'auto' (mesure, applique "
                        "si significatif), '0' (aucun), ou un decalage en logit")
    s.add_argument("--abstention", type=float, default=None,
                   help="probabilite minimale pour se prononcer sans cotes (defaut 0.60)")
    s.add_argument("--edge-min", type=float, default=None,
                   help="ecart minimal avec le marche pour un pari avec cotes (defaut 0.04)")
    s.add_argument("--export-csv", help="exporte la carte de pronostics et abstentions en CSV")
    s.add_argument("--json", action="store_true", help="sortie JSON structuree")
    s.set_defaults(func=cmd_slate)

    s = sub.add_parser("settle", parents=[common],
                       help="pointe les pronostics archives contre les resultats")
    s.add_argument("--journal", default="data/journal.csv")
    s.add_argument("--window-days", type=int, default=3,
                   help="tolerance de date en jours pour les lignes sans match_id (defaut 3)")
    s.set_defaults(func=cmd_settle)

    s = sub.add_parser("journal", parents=[common],
                       help="bilan des pronostics deja pointes")
    s.add_argument("--journal", default="data/journal.csv")
    s.add_argument("--since", type=parse_date, help="inclut uniquement les matchs a partir de cette date")
    s.add_argument("--until", type=parse_date, help="inclut uniquement les matchs jusqu'a cette date")
    s.add_argument("--json", action="store_true", help="sortie JSON structuree")
    s.set_defaults(func=cmd_journal)

    s = sub.add_parser("tune", parents=[common],
                       help="regle les constantes de notation sur vos donnees")
    s.add_argument("--out", default="models/elo.json")
    s.add_argument("--holdout", type=float, default=0.25,
                   help="part finale de l'historique reservee a la validation")
    s.add_argument("--json", action="store_true", help="sortie JSON structuree")
    s.set_defaults(func=cmd_tune)

    s = sub.add_parser("tipster-backtest", parents=[common],
                       help="bilan historique du pronostiqueur (paliers, marches)")
    s.add_argument("--initial-frac", type=float, default=0.4)
    s.add_argument("--refit-every", type=int, default=150)
    s.add_argument("--abstention", type=float, default=0.60,
                   help="probabilite minimale pour se prononcer")
    s.add_argument("--momentum", type=float, default=0.0,
                   help="decalage momentum en logit a appliquer (defaut 0 ; mesurer "
                        "d'abord avec `slate --momentum auto`)")
    s.add_argument("--export-csv", help="exporte les pronostics regles en CSV")
    s.add_argument("--json", action="store_true", help="sortie JSON structuree")
    s.set_defaults(func=cmd_tipster_backtest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, RuntimeError, OSError) as exc:
        sys.exit(f"erreur : {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
