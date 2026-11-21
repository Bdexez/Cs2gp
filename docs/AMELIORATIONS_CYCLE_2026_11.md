# Ameliorations d'ingenierie et d'analyse de donnees (novembre 2026)

Revue complete du coeur `cs2gp` conduite en trois cycles incrementaux (fiabilite
d'etat et contrats numeriques, diagnostics probabilistes et audit de donnees,
outillage domaine CS2 et sortie structuree).

## Cycle 1 : Fiabilite d'etat et garde-fous numeriques

- **Lectures pures dans `EloEngine`** (`ratings.py`, `features.py`) : les
  methodes de consultation (`rating`, `deviation`, `trajectory`, `map_rating`,
  `maps_played`, `match_context`) n'inserent plus une equipe inconnue dans
  `EloEngine.teams`. `EloConfig` rejette desormais aussi les booleens pour
  `lan_bonus` et les champs morts `xy`/`xx` de `TeamRating` sont retires.
- **Residualisation et echelle ponderees** (`model.py`) : `LogisticModel`
  utilise `sample_weight` lors de la projection hors `elo_diff` et du calcul
  des echelles, de sorte qu'une observation de poids nul n'influence aucun
  parametre. `MatchPredictor.save` devient atomique, `explain` et
  `series_win_probability` valident leurs entrees, et `load` preserve `use_gbdt`.
- **Garde-fous marches derives** (`pronostic.py`) : le seuil `EDGE_ABSURDE`
  (verdict `SUSPECT`, mise nulle) s'applique desormais aux marches `total maps`
  et `handicap sweep`, pas seulement a `vainqueur`. Toutes les features de
  `MATCH_FEATURES` disposent d'un libelle francais dans `_rationale`.
- **Backtest, journal et CLI** (`backtest.py`, `journal.py`, `cli.py`) :
  validation de `momentum`/`min_edge`, recherche vectorisee de `best_blend` et
  `bisect_left` pour les coupures temporelles ; `settle` ne reecrit plus le
  journal lorsqu'aucune ligne n'a change ; la CLI rejette des cotes partielles
  (`--odds-a` seul) et les bornes invalides sur `ratings`/`fixtures`, tout en
  exposant `--window-days` et `--refit-every`.

## Cycle 2 : Diagnostics probabilistes, audit et sortie JSON

- **Diagnostics de calibration** (`metrics.py`) :
  - `brier_decomposition` : decomposition de Murphy (fiabilite, resolution,
    incertitude, Brier Skill Score).
  - `calibration_slope_intercept` : calibration logistique de Cox (pente et
    intercept) pour distinguer surconfiance et sous-confiance.
  - `paired_block_bootstrap(..., metric="brier")` : IC apparie par blocs pour
    le score de Brier en plus du log-loss.
- **Ventilation par segments** (`backtest.py`, `cli.py`) :
  `BacktestResult.breakdown` et `cs2gp backtest --by-segment` ventilent les
  metriques et le gain face a Elo par format (`bo1/bo3/bo5`), contexte
  (`LAN/online`) et `tier`.
- **Audit de qualite des donnees** (`audit.py`, `cli.py`) : `cs2gp audit`
  controle la periode, les series incompletes, la couverture des rosters, les
  maps hors Active Duty et les regimes de scores (MR12, OT, detection de scores
  MR15 suspects).
- **Sorties structurees `--json`** : methodes `to_dict()` sur `Prediction`,
  `BacktestResult`, `Pick`, `Analysis` et `TipsterRecord`, exposees via
  `--json` sur `audit`, `ratings`, `predict`, `backtest`, `slate`, `journal`
  et `tipster-backtest`.

## Cycle 3 : Pool versionne et metriques de risque

- **Pool Active Duty versionne** (`schema.py`, `features.py`) :
  `ACTIVE_DUTY_HISTORY` et `active_duty_on(date, versioned=True)` suivent les
  rotations historiques du pool CS2 tout en conservant `ACTIVE_DUTY` par defaut
  pour la retrocompatibilite des instantanes et tests existants.
- **CLV et risque financier** (`odds.py`, `journal.py`) : ajout de
  `closing_line_value` (CLV face a la ligne de cloture sans marge), ainsi que
  `max_drawdown` et la pire serie de defaites dans `simulate_flat` et
  `journal.financial_summary`.
