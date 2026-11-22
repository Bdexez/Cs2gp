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

## Cycle 4 : Profondeur statistique et integration bout-en-bout

- **Pool versionne bout-en-bout** (`predict.py`, `features.py`, `backtest.py`,
  `pronostic.py`, `snapshot.py`, `cli.py`) : option `versioned_pool` /
  `--versioned-pool` propagee de la construction des datasets jusqu'a
  `Analyzer.predict`, aux backtests et aux instantanes.
- **Intervalles de confiance analytiques** (`model.py`) : covariance
  asymptotique de Fisher dans `LogisticModel.decision_se` et intervalle de
  confiance 95 % dans `MatchPredictor.predict_interval`, persiste en JSON.
- **Audit et benchmark oracle** (`audit.py`, `backtest.py`) : detection des
  trous calendaires, equipes sous le rodage et collisions sur un meme jour ;
  `benchmark_against_truth` mesure l'exces de log-loss et de Brier face aux
  probabilites latentes de `simulate_with_truth`.

## Cycle 5 : Incertitude en sortie, scoring du journal et rosters simules

- **IC 95 % dans `Prediction`** (`predict.py`) : exposition de `p_a_interval`
  dans le rapport texte et dans `Prediction.to_dict()`.
- **Scores propres du pronostiqueur** (`pronostic.py`) : `TipsterRecord.scoring()`
  calcule log-loss, Brier et ECE sur les pronostics regles (rapport et JSON).
- **Rosters dans le simulateur** (`sources.py`, `cli.py`) : `with_rosters=True`
  (et `--with-rosters` sur `cs2gp sample`) genere des effectifs de 5 joueurs
  renouveles lors des remaniements sans perturber le flux aleatoire, permettant
  des experiences A/B pures et la validation bout-en-bout sur regime `realiste`.

## Cycle 6 : Regularisation chronologique, derive par fold et cotes historiques

- **Selection chronologique de `l2`** (`model.py`) : `select_l2` et option
  `MatchPredictor(..., l2_grid=...)` choisissant la penalite ridge sur la fin
  du jeu d'entrainement, sans toucher au holdout de calibration ni a l'evaluation.
- **Derive par fold et export CSV** (`backtest.py`, `cli.py`) :
  `BacktestResult.drift_report()` (`--by-fold`) suit le log-loss, le gain face
  a Elo, l'ECE et la pente de Cox fold par fold ; `export_csv()` (`--export-csv`)
  exporte atomiquement les predictions hors-echantillon.
- **Confrontation a des cotes historiques en CLI** (`backtest.py`, `cli.py`) :
  `read_odds_csv` (validation stricte, rejet des doublons et cotes <= 1) et
  `cs2gp backtest --odds-csv` affichant ROI, profit et drawdown.

## Cycle 7 : Simulation Kelly, profil d'equipe, inspection et benchmark multi-graines

- **Simulation de bankroll Kelly** (`odds.py`, `backtest.py`) : `simulate_kelly`
  calcule la croissance multiplicative de bankroll, le drawdown relatif maximal
  et la pire serie de defaites ; integre dans `bet_backtest`.
- **Profil d'equipe et inspection d'instantane** (`predict.py`, `snapshot.py`,
  `cli.py`) : nouvelles commandes `cs2gp team <equipe>` (Elo, RD, trajectoire,
  forme, SOS, charge 21j, roster, pool de maps) et `cs2gp inspect <bundle>`
  (verification SHA-256 et metadonnees sans unpickler le GBDT), en texte et `--json`.
- **Benchmark multi-graines** (`backtest.py`) : `benchmark_regimes` agrege
  moyenne et ecart-type des metriques modele / Elo / oracle sur plusieurs
  graines et regimes (`facile`, `realiste`).

## Cycle 8 : Confrontations directes, z-scores de Wald et veto stochastique

- **Confrontations directes (`cs2gp h2h`)** (`predict.py`, `cli.py`) :
  `Analyzer.head_to_head(a, b)` et `cs2gp h2h <a_team> <b_team> [--json]`
  resument series, maps, bilan par terrain et score pondere par recence.
- **Diagnostics de features (`feature_diagnostics`)** (`model.py`) : poids
  standardises, erreurs types de Fisher et z-scores de Wald par variable pour
  identifier les variables statistiquement significatives.
- **Veto stochastique** (`markets.py`) : `stochastic_veto_distribution`
  moyenne la loi du score sur des vetos softmax (reduit exactement a
  `project_veto` quand `temperature=0`) afin de quantifier la sensibilite au veto.
