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

## Cycle 9 : Diagnostics CLI, export du classement et suite d'invariants

- **Diagnostics CLI enrichis** (`cli.py`) : `cs2gp train` affiche les z-scores
  de Wald (`*` si `|z| > 1.96`) ; `cs2gp journal` affiche le drawdown maximal
  et la pire serie de defaites ; `cs2gp ratings --export-csv` exporte
  atomiquement le classement Elo et les 7 notes par map.
- **Suite d'invariants globaux et symetrie sous momentum** (`predict.py`,
  `tests/test_invariants.py`) : correction d'un biais d'ordre des equipes
  dans `probability_from_pool` lorsque `momentum != 0` (moyennage 50/50 des
  deux ordres de premier pick pour preserver exactement `p(A|B) + p(B|A) = 1`),
  verification de la coherence des marches derives sur BO1/BO3/BO5 et de la
  monotonicite des fonctions coeur (`expected_score`, `series_win_probability`,
  `devig_power`, `kelly_fraction`).

## Cycle 10 : Symetrie du veto stochastique, equilibre du pool et filtres journal

- **Symetrie du veto stochastique** (`markets.py`) : `stochastic_veto_distribution`
  moyenne les deux ordres de premier pick pour garantir exactement
  `p(A|B) + p(B|A) = 1` meme avec `temperature > 0` et `momentum != 0`.
- **Repartition annuelle et equilibre du pool** (`audit.py`) : `audit_history`
  ajoute `matches_by_year` et `map_balance_ratio` (alerte en cas de fort
  desequilibre des maps Active Duty).
- **Filtres temporels sur le journal** (`journal.py`, `cli.py`) : options
  `since`/`until` (`--since`, `--until` sur `cs2gp journal`) pour evaluer une
  periode donnee sans modifier le CSV.

## Cycle 11 : Significativite du ROI, diagnostics de cotes et diff d'instantanes

- **Significativite du ROI et diagnostics de marche** (`odds.py`) :
  `roi_significance` (t-statistique et p-value bilaterale) et
  `market_diagnostics` (overround, vig, arbitrage, comparaison devig
  proportionnel vs power).
- **Comparateur d'instantanes (`cs2gp diff`)** (`snapshot.py`, `cli.py`) :
  `diff_analyzers(a, b)` et `cs2gp diff <a.cs2gp> <b.cs2gp> [--json]` comparent
  deux bundles (volume, dates, temperatures, momentum, Elo et options) sans
  unpickler le GBDT.

## Cycle 12 : Fusion d'historiques, fiabilite par quantiles et significativite ROI

- **Significativite ROI integree** (`backtest.py`, `journal.py`, `cli.py`) :
  `roi_significance` expose dans `bet_backtest`, `financial_summary` et
  `cs2gp journal`.
- **Fusion d'historiques (`cs2gp merge`) et `write_csv` atomique** (`schema.py`,
  `cli.py`) : `merge_histories()` deduplique par `(match_id, map_name)` (meme
  en orientation inversee), enrichit les rosters/evenements manquants et
  rejette tout conflit de score.
- **Fiabilite par quantiles** (`metrics.py`) : `quantile_reliability_table()`
  decoupe les predictions en tranches d'effectifs egaux.

## Cycle 13 : Test de Spiegelhalter, fiabilite par quantiles et export de carte

- **Test analytique de Spiegelhalter** (`metrics.py`, `backtest.py`) :
  `spiegelhalter_z(y, p)` teste formellement la calibration du score de Brier
  (z-score et p-value bilaterale) ; expose avec `quantile_reliability` dans
  `BacktestResult.to_dict()`.
- **Export CSV de la carte (`cs2gp slate --export-csv`)** (`cli.py`) : ecriture
  atomique de tous les pronostics et abstentions de la carte du jour.

## Cycle 14 : Factorisation des ecritures atomiques, overround et sensibilite au veto

- **Ecritures atomiques centralisees** (`csvio.py`, `schema.py`, `model.py`,
  `backtest.py`, `cli.py`) : `atomic_write_bytes` et `atomic_write_csv`
  eliminent la duplication du motif `NamedTemporaryFile + fsync + replace`.
- **Marge bookmaker moyenne** (`backtest.py`) : `bet_backtest` rapporte
  `avg_overround` sur les matchs cotes pour contextualiser le ROI.
- **Sensibilite au veto** (`markets.py`) : `veto_sensitivity()` mesure la
  distance en variation totale entre la loi du score deterministe et la loi
  du score sous veto stochastique.

## Cycle 15 : Spiegelhalter en rapport, detection d'alias et courbe d'equite

- **Spiegelhalter dans `BacktestResult.report()`** (`backtest.py`) : affichage
  direct du z-score et de la p-value de Spiegelhalter a cote de la
  decomposition de Murphy.
- **Detection d'alias manquants** (`audit.py`) : `audit_history` repere les
  paires de noms d'equipes tres proches (`SequenceMatcher >= 0.85`) pour
  prevenir la scission silencieuse d'une equipe dans un CSV reel.
- **Courbe d'equite** (`odds.py`) : `equity_curve(results, mode="flat"|"kelly")`
  retourne la trajectoire cumulee pari par pari.

## Cycle 16 : Diagnostics journal, CLV en backtest et alias personnalises

- **Scoring enrichi du pronostiqueur** (`pronostic.py`) : `TipsterRecord.scoring()`
  inclut `spiegelhalter` et `brier_decomposition` (rapport et JSON).
- **CLV dans `bet_backtest`** (`backtest.py`) : `read_odds_csv` accepte les
  colonnes optionnelles `closing_odds_a, closing_odds_b` et `bet_backtest`
  calcule `clv_bets`, `avg_clv_prob` et `avg_clv_ev`.
- **Alias personnalises** (`schema.py`) : `normalize_team(name, extra_aliases=...)`
  permet de resoudre des variantes de noms detectees par `cs2gp audit`.

## Cycle 17 : Ratio de Sharpe, sortie cotes enrichie et `tune --json`

- **Ratio de Sharpe par pari** (`odds.py`) : `roi_significance()` expose
  `sharpe_ratio` (rendement moyen / ecart-type par pari).
- **Sortie texte `backtest --odds-csv` enrichie** (`cli.py`) : affiche marge
  moyenne, p-value du ROI, rendement/drawdown Kelly et CLV moyen lorsque des
  cotes de cloture sont fournies.
- **`cs2gp tune --json`** (`cli.py`) : exporte configuration retenue, statut
  d'enregistrement et rapport de validation temporelle en JSON structure.

## Cycle 18 : MCE, controle pre-vol des fixtures et courbe d'equite du journal

- **Maximum Calibration Error (`mce`)** (`metrics.py`) :
  `max_calibration_error(y, p)` et exposition dans `summarize()` pour reperer
  la pire tranche de calibration.
- **Controle pre-vol des fixtures (`audit_fixtures`)** (`audit.py`) : detecte
  les equipes inconnues, historiques minces/anciens et collisions sans `match_id`.
- **Courbe d'equite dans le journal** (`journal.py`) : `equity_curve_flat`
  exposee dans `financial_summary()` (et donc via `cs2gp journal --json`).

## Cycle 19 : Audit fixtures CLI, colinearite des features et affichage MCE

- **`cs2gp audit --fixtures`** (`cli.py`) : expose le controle pre-vol
  `audit_fixtures()` en ligne de commande (texte et `--json`).
- **Rapport de colinearite (`feature_correlation_report`)** (`features.py`) :
  identifie les colonnes constantes et les paires de features fortement
  correlees (`|r| >= threshold`) en NumPy pur.
- **Affichage `mce` dans `format_summary()`** (`metrics.py`) : rend l'ecart
  maximal de calibration visible a cote de `ece`.

## Cycle 20 : Marches derives dans `predict`, colinearite en audit et API publique

- **Colinearite dans `cs2gp audit`** (`cli.py`) : inclut `feature_correlation_report`
  du dataset match (features constantes et fortes correlations).
- **Grille complete des marches (`cs2gp predict --markets`)** (`cli.py`) :
  affiche toutes les probabilites et cotes justes des marches derives (texte et `--json`).
- **API publique stable** (`cs2gp/__init__.py`) : exporte `Analyzer`, `EloConfig`,
  `Fixture`, `Tipster`, `audit_history`, `walk_forward` dans `__all__`.

## Cycle 21 : Calibration `predict --markets`, test apparie de perte et filtre d'activite

- **Calibration alignee sur `predict --markets`** (`cli.py`) : utilise
  `fit_decider_temperature` (desactivable via `--no-calibration`) sur
  `an.as_of(pred.date)` pour correspondre exactement a `cs2gp slate`.
- **Test apparie de perte (`paired_loss_test`)** (`metrics.py`) : test t /
  Diebold-Mariano apparie sur log-loss ou Brier face a une reference.
- **Filtre d'activite dans `leaderboard` / `cs2gp ratings`** (`ratings.py`,
  `cli.py`) : `active_within_days` / `--active-within N` exclut les equipes
  inactives depuis plus de N jours.

## Cycle 22 : Importance par permutation, historique Elo et bootstrap de bankroll

- **Importance par permutation** (`model.py`) : `permutation_importance()`
  mesure la degradation hors-echantillon (log-loss ou Brier) par colonne,
  compatible avec la regression logistique seule comme avec l'ensemble GBDT.
- **Historique Elo d'une equipe (`cs2gp history`)** (`predict.py`, `cli.py`) :
  `Analyzer.team_history(team)` et `cs2gp history <team> [--last N] [--export-csv] [--json]`
  reconstruisent la trajectoire chronologique map par map (Elo, delta, RD).
- **Bootstrap Monte Carlo de bankroll** (`odds.py`) : `bootstrap_bankroll()`
  estime probabilite de profit, quantiles finaux (p05/mediane/p95) et drawdown
  p95 sur un horizon de paris donne (mise fixe ou Kelly).

## Cycle 23 : Importance walk-forward, scoring par groupe et CI

- **Importance par permutation en walk-forward** (`backtest.py`) :
  `walk_forward_importance()` agrege sans fuite temporelle la degradation
  hors-echantillon par feature sur les folds temporels.
- **Scoring propre par groupe** (`pronostic.py`) : `TipsterRecord.by()` expose
  `log_loss` et `brier` pour chaque palier et chaque marche.

## Cycle 24 : Fiabilite cote favori, `backtest --importance` et export tipster

- **Fiabilite cote favori (`favorite_reliability_table`)** (`metrics.py`) :
  replie les probabilites dans `[0.5, 1.0]` pour evaluer la calibration du
  favori annonce par tranche.
- **`cs2gp backtest --importance`** (`cli.py`) : expose `walk_forward_importance`
  directement en CLI (texte et `--json`).
- **Export CSV du `tipster-backtest`** (`pronostic.py`, `cli.py`) :
  `TipsterRecord.export_csv()` et `--export-csv` sur `cs2gp tipster-backtest`.

## Cycle 25 : Temperatures par fold, fiabilite favori et audit des cotes

- **Temperatures par fold et fiabilite favori** (`backtest.py`) : `walk_forward`
  enregistre la temperature de calibration apprise a chaque fold (`temperatures`)
  et `BacktestResult.to_dict()` expose `favorite_reliability`.
- **Audit des cotes dans `audit_fixtures`** (`audit.py`) : signale les paires
  de cotes incompletes (retrait de marge impossible) et les surebets (`overround < 1`).

## Cycle 26 : Analyse d'erreurs (`worst_misses`) et garde JSON EloConfig

- **Analyse d'erreurs (`worst_misses` / `--errors N`)** (`backtest.py`, `cli.py`) :
  identifie les N matchs les plus couteux en log-loss dans un backtest (texte et `--json`).
- **Validation stricte de `EloConfig.from_dict` / `load`** (`ratings.py`) :
  rejette explicitement tout fichier JSON qui n'est pas un objet.

## Cycle 27 : Risque Monte Carlo du journal, scouting `compare_teams` et garde `--errors`

- **Risque Monte Carlo dans le journal (`cs2gp journal --risk`)** (`journal.py`,
  `cli.py`) : `bankroll_risk()` applique `bootstrap_bankroll()` aux paris cotes
  du journal (texte et `--json`).
- **Scouting comparatif (`Analyzer.compare_teams`)** (`predict.py`) : compare
  deux equipes (profils, avantage map par map, probabilite Elo, H2H) sans
  necessiter d'entrainement.
- **Validation `--errors < 0`** (`cli.py`) : echoue immediatement avant le backtest.

## Cycle 28 : Commande `compare`, Hosmer-Lemeshow et variations Elo (`movers`)

- **Scouting CLI (`cs2gp compare`)** (`cli.py`) : expose `Analyzer.compare_teams()`
  en texte et `--json` sans entrainement.
- **Test de Hosmer-Lemeshow** (`metrics.py`) : `hosmer_lemeshow(y, p, bins=10)`
  calcule la statistique chi2 par quantiles pour reperer les defauts de
  calibration non lineaires.
- **Variations Elo (`Analyzer.movers`)** (`predict.py`) : identifie les plus
  fortes hausses et baisses Elo sur une fenetre glissante de N jours.

## Cycle 29 : Commande `movers`, Hosmer-Lemeshow en JSON et modularisation CLI

- **Commande CLI `cs2gp movers`** (`cli.py`, `cli_exports.py`) : expose les
  plus fortes hausses et baisses Elo recentes (texte et `--json`).
- **Hosmer-Lemeshow dans le backtest JSON** (`backtest.py`) :
  `BacktestResult.to_dict()` inclut `hosmer_lemeshow`.
- **Modularisation CLI** (`cli_exports.py`, `cli.py`) : extraction des exports
  CSV (`ratings`, `history`, `slate`) et du formatage `movers` dans un module
  dedie pour alleger `cli.py`.

## Cycle 30 : Statistiques par map (`cs2gp maps`), hygiene AST et reference CLI

- **Statistiques globales par map** (`predict.py`, `cli.py`) :
  `Analyzer.map_stats()` et `cs2gp maps [--json]` calculent volume, taux de
  prolongation (OT), ecart moyen de rounds, total de rounds et taux de
  victoire du favori Elo par map.
- **Verification d'hygiene AST et reference CLI** (`docs/REFERENCE.md`) :
  audit des imports inutilises sur tout `cs2gp/` et mise a jour complete de la
  reference CLI.

## Cycle 31 : Courbe d'abstention et bilan par tournoi (`event_stats`)

- **Courbe d'abstention (`abstention_curve` / `--abstention-curve`)** (`backtest.py`,
  `cli.py`) : mesure le compromis couverture / precision / log-loss / Brier par
  seuil de confiance pour calibrer objectivement la selectivite.
- **Statistiques par tournoi (`Analyzer.event_stats`)** (`predict.py`) :
  resume volume, dates, tier, LAN/online et taux d'upsets Elo par evenement.

## Cycle 32 : Commande `events` et detection du biais d'orientation `team_a`

- **Detection du biais d'orientation** (`audit.py`) : `audit_history()` calcule
  `team_a_win_rate` et alerte si un export place systematiquement le vainqueur
  dans la colonne `team_a`.
- **Commande CLI `cs2gp events`** (`cli.py`) : expose `Analyzer.event_stats()`
  en texte et `--json`.

## Cycle 33 : Bilan mensuel du journal (`--by-month`) et distribution Elo (`rating_summary`)

- **Bilan mensuel du journal** (`journal.py`, `cli.py`) : `monthly_summary()` et
  `cs2gp journal --by-month [--json]` ventilent volume, probabilite annoncee,
  taux de reussite, Brier et P&L/ROI mise fixe par mois (`YYYY-MM`).
- **Resume statistique Elo (`EloEngine.rating_summary`)** (`ratings.py`, `cli.py`) :
  calcule moyenne, ecart-type, quantiles (min/p25/mediane/p75/max) et RD moyen
  de l'ecosysteme, expose dans `cs2gp ratings` (texte et `--json`).

## Cycle 34 : ROC-AUC en NumPy pur, exports CSV `maps`/`events` et dates passees en audit

- **ROC-AUC exact (`roc_auc`)** (`metrics.py`) : statistique U de Mann-Whitney
  avec gestion vectorisee des ex-aequo en NumPy pur, exposee dans `summarize()`
  et affichee dans `format_summary()`.
- **Exports CSV `maps` et `events`** (`cli_exports.py`, `cli.py`) : ajout de
  `--export-csv` atomique sur `cs2gp maps` et `cs2gp events`.
- **Detection des fixtures passees (`audit_fixtures`)** (`audit.py`) : signale
  les affiches datees avant le dernier match connu de l'historique (`past_fixtures`).

## Cycle 35 : Retrait de marge de Shin (`devig_shin`) et export CSV filtre du journal

- **Retrait de marge selon Shin (`devig_shin`)** (`odds.py`) : calcule la part
  d'inities `shin_z` par dichotomie exacte et expose `shin_a`, `shin_b`,
  `shin_z` dans `market_diagnostics()`.
- **Export CSV filtre du journal** (`journal.py`, `cli.py`) :
  `export_filtered_csv()` et `cs2gp journal --export-csv` exportent de maniere
  atomique une tranche datee (ou reglee) du journal.

## Cycle 36 : Performance par equipe en backtest (`--by-team`) et sensibilite veto dans `predict`

- **Diagnostic par equipe (`BacktestResult.by_team` / `--by-team`)** (`backtest.py`,
  `cli.py`) : identifie les equipes les plus difficiles et les plus previsibles
  hors-echantillon (log-loss, gain vs Elo, Brier, precision).
- **Sensibilite au veto dans `cs2gp predict --markets`** (`cli.py`) : expose
  `veto_sensitivity()` (variation totale et ecart max sur le score exact) en
  texte et `--json` pour les series BO3/BO5.

## Cycle 37 : Conditionnement Fisher, `cs2gp train --json` et ventilation format/annee du tipster

- **Conditionnement numerique (`LogisticModel.condition_number`)** (`model.py`) :
  calcule le conditionnement de la matrice de covariance de Fisher pour reperer
  une colinearite excessive apres standardisation.
- **Sortie JSON sur `cs2gp train`** (`cli.py`) : `cs2gp train --json` emet
  metriques, poids + z-scores de Wald, temperature, `l2` et conditionnement Fisher.
- **Ventilation `by_format` et `by_year`** (`pronostic.py`) : `TipsterRecord.to_dict()`
  expose le suivi du pronostiqueur par format (`bo1`/`bo3`/`bo5`) et par annee.

## Cycle 38 : Copie isolee `EloEngine.copy()` et `--json` sur `teams` / `merge`

- **Copie isolee (`EloEngine.copy`)** (`ratings.py`) : clone explicitement
  l'etat Elo (notes, RD, maps, regression en ligne) pour des simulations
  hypothetiques sans muter l'instance source.
- **Sortie JSON sur `cs2gp teams` et `cs2gp merge`** (`cli.py`) : complete
  l'uniformite `--json` de l'interface CLI pour l'automatisation.

## Cycle 39 : Nettete probabiliste (`sharpness`) et ecriture atomique de `cs2gp fixtures`

- **Metrique de nettete (`sharpness`)** (`metrics.py`, `backtest.py`) : mesure
  confiance moyenne, ecart-type, entropie moyenne et part de predictions
  decisives (>= 70 %), exposee dans `BacktestResult.to_dict()`.
- **Ecriture atomique et `--json` sur `cs2gp fixtures`** (`cli.py`) : utilise
  `atomic_write_csv()` (plus aucune ecriture CSV non atomique dans le paquet)
  et ajoute `--json`.

## Cycle 40 : Simulation Elo *what-if* (`simulate_outcome`) et invariants analytiques

- **Simulation *what-if* (`EloEngine.simulate_outcome`)** (`ratings.py`) :
  calcule sans muter l'etat les gains/pertes Elo (global et map) en cas de
  victoire de A ou de B.
- **Invariants mathematiques etendus** (`tests/test_invariants.py`) : couvre
  l'antisymetrie de `roc_auc`, l'invariance par retournement de `sharpness`, la
  monotonicite de `devig_shin` et la purete/symetrie de `simulate_outcome`.

## Cycle 41 : Enjeux Elo dans `compare` et diagnostic d'environnement `cs2gp doctor`

- **Enjeux Elo dans `Analyzer.compare_teams` / `cs2gp compare`** (`predict.py`,
  `cli.py`) : expose `elo_stakes` (gain/perte Elo pour chaque equipe) en texte
  et `--json`.
- **Commande `cs2gp doctor`** (`cli.py`) : verifie en un coup d'oeil versions
  Python/NumPy/scikit-learn, disponibilite GBDT et lisibilite du fichier de
  donnees (texte et `--json`).

## Cycle 42 : Garde booleenne dans `temporal.py`, `sample --json` et modularisation CLI

- **Rejet des booleens dans `forward_splits`** (`temporal.py`) : `refit_every`
  et `min_train` rejettent explicitement `bool` (`isinstance(True, Integral)`).
- **Sortie `--json` sur `cs2gp sample`** (`cli.py`) : complete la couverture
  `--json` sur l'ensemble des sous-commandes analytiques et generatrices.
- **Extraction des formateurs CLI** (`cli_exports.py`, `cli.py`) : deplacement
  de `format_team_profile()` et `format_compare()` dans `cli_exports.py`.

## Cycle 43 : Retrait de marge configurable (`devig` / `--devig`)

- **Dispatcher unifie `devig()`** (`odds.py`) : selectionne `power` (defaut),
  `shin` ou `proportional`, et l'expose via `devig_method` dans `find_value()`
  et `closing_line_value()`.
- **Option `--devig` sur `cs2gp predict`** (`cli.py`) : permet de confronter
  directement une prediction au marche en mode `power`, `shin` ou `proportional`.

## Cycle 44 : `--devig` en backtest de paris et `cs2gp settle --json`

- **Retrait de marge configurable dans `bet_backtest()`** (`backtest.py`,
  `cli.py`) : `devig_method` / `--devig {power,shin,proportional}` propage la
  methode choisie a la detection de valeur et au calcul de CLV.
- **Sortie `--json` sur `cs2gp settle`** (`cli.py`) : rend l'ensemble complet
  des sous-commandes CLI 100 % exploitable en JSON pour l'automatisation.

## Cycle 45 : `--devig` sur `Tipster`/`slate` et contrat `--json` global CLI

- **Retrait de marge configurable dans `Tipster` / `cs2gp slate`** (`pronostic.py`,
  `cli.py`) : `devig_method` / `--devig {power,shin,proportional}` unifie le
  retrait de marge sur l'ensemble des commandes marche (`predict`, `backtest`, `slate`).
- **Test de contrat `--json` global** (`tests/test_cli_json.py`) : garantit que
  les 24 sous-commandes CLI exposent toutes `--json`.

## Cycle 46 : Ecriture atomique de `EloConfig.save()` et couverture CI

- **Ecriture atomique de `EloConfig.save()`** (`ratings.py`) : utilise
  `atomic_write_bytes()` (creation automatique des repertoires parents +
  `fsync`/`os.replace`), rendant 100 % des ecritures disque du paquet atomiques.
- **Extension des smoke checks CI** (`.github/workflows/tests.yml`) : ajout de
  `cs2gp doctor --json`, `cs2gp maps --json` et `cs2gp events --json`.

## Cycle 47 : Deduplication `snapshot.py` et suivi des pronostics en attente (`--pending`)

- **Deduplication de la verification d'archives** (`snapshot.py`) : extraction de
  `_read_verified_archive()` partage par `load_analyzer()` et `inspect_analyzer()`.
- **Pronostics en attente (`pending_picks` / `--pending`)** (`journal.py`,
  `cli.py`) : liste les pronostics actionnables non encore pointes (texte et `--json`).

## Cycle 48 : API publique enrichie (`cs2gp/__init__.py`) et matrice de confusion (`confusion_summary`)

- **API publique paresseuse etendue** (`cs2gp/__init__.py`) : table `_EXPORTS`
  declarant `audit_fixtures`, `devig`, `inspect_analyzer`, `diff_analyzers`,
  `roc_auc` et `sharpness` dans `__all__`.
- **Matrice de confusion parametree (`confusion_summary`)** (`metrics.py`) :
  calcule `tp`, `fp`, `tn`, `fn`, precision, rappel, specificite, F1 et
  exactitude equilibree pour un seuil donne.
  Exposee dans l'API publique et sous `confusion` dans le JSON du backtest
  (seuil 0,5). Le F1 vaut zero en presence d'erreurs sans vrai positif ;
  seuls les ratios sans denominateur valent `null`. Tests des classes
  uniques, des seuils limites et des entrees invalides.

## Cycle 49 : Memoire des simulations Monte Carlo et drawdown relatif

- **Simulation par lots (`bootstrap_bankroll`)** (`odds.py`) : conserve
  seulement les capitaux finaux et drawdowns maximaux entre les lots ;
  au plus 64 trajectoires par lot, avec une taille adaptee a l'horizon.
  Les tirages suivent le meme ordre pour une graine donnee. Sur 500
  simulations Kelly de 4 000 paris (graine 42 ; 13 gains et 7 pertes,
  probabilite 0,65 et cote 2), le pic mesure avec `tracemalloc` passe de
  92,39 Mio a 10,59 Mio, avec les memes quantiles et probabilite de profit.
  La memoire depend encore de la longueur d'une trajectoire et du nombre
  de resultats finaux conserves pour les quantiles.
- **Drawdown independant de l'unite de capital** (`odds.py`) : suppression
  du plancher absolu de 1e-12 sur les pics de bankroll. Le capital initial
  strictement positif est inclus dans chaque pic ; une perte de 5 % donne
  maintenant 5 % meme pour un capital initial de 1e-15 (auparavant 0,005 %).
  Tests de comparaison aux trajectoires individuelles en mise fixe et Kelly,
  de reproductibilite sur plusieurs lots et d'invariance au capital initial.

## Cycle 50 : Convergence du diagnostic de calibration de Cox

- **Newton amorti (`calibration_slope_intercept`)** (`metrics.py`) : controle
  de decroissance du cout logistique penalise avec `logaddexp`, sans saturer
  artificiellement les logits de l'optimisation. Le pas est reduit lorsqu'un
  pas complet augmente le cout ; une tolerance d'arrondi evite un blocage a
  proximite de l'optimum. Un cas reproduit avec des probabilites extremes et
  des resultats contradictoires passait auparavant a une pente de 8,5 millions,
  sans convergence. Il converge maintenant vers une pente de -0,0421.
- **Verification numerique** : solution analytique de deux groupes aux taux
  observes 25 % et 75 %, gradient penalise proche de zero et symetrie lors
  de l'inversion des equipes. Controle exploratoire de 200 graines aux
  probabilites extremes : aucune non-convergence observee apres correction.

## Cycle 51 : Integrite numerique des modeles JSON

- **Validation des covariances rechargees** (`model.py`) : rejet des matrices
  non symetriques ou ayant une valeur propre negative au-dela de la tolerance
  relative de 1e-10. Une covariance negative etait auparavant acceptee et
  pouvait donner un intervalle de largeur nulle apres le clipping de variance.
  Les modeles anciens sans covariance restent rechargeables.
- **Contrat des metadonnees** (`model.py`) : un modele doit etre un objet JSON
  contenant les parametres obligatoires, avec une version entiere, une
  temperature positive finie, des noms de features uniques, un indicateur
  GBDT booleen et des metadonnees d'entrainement valides. Les fichiers
  incorrects produisent une `ValueError` au lieu d'erreurs d'attribut ou de
  type non gerees. Tests des corruptions et des aller-retours d'instantanes.
- **Validation avec GBDT reel** : environnement virtuel isole sous `/tmp`,
  Python 3.14.10, NumPy 2.5.5 et scikit-learn 1.9.1. Les 257 tests passent
  sans test ignore, dont les sauvegardes/rechargements de l'ensemble GBDT
  et des instantanes complets.

## Cycle 52 : Augmentation miroir logistique sans duplication des donnees

- **Gradient et Hessienne equivalents** (`model.py`) : l'identite
  `sigmoid(-z) = 1 - sigmoid(z)` permet de remplacer la copie miroir par un
  facteur 2 dans le gradient et la Hessienne. La penalite L2 conserve ainsi
  sa force historique. La covariance reste calculee sur les observations
  originales, sans compter artificiellement les lignes miroir.
- **Mesure de memoire** : fit de 40 000 observations et 24 features gaussiennes,
  graine 52, labels tires selon `sigmoid(X[:, 0])`, threads BLAS fixes a 1.
  Pic `tracemalloc` de 46,77 Mio avant contre 23,58 Mio apres (environ -50 %).
  Ecart maximal observe de 1,2e-16 sur les coefficients et les probabilites
  du lot compare ; projection et echelles identiques.
- **Tests de non-regression** : optimum analytique unidimensionnel pondere
  pour controler la force L2 et equivalence avec une augmentation explicite
  dont la penalite est ajustee ; antisymetrie des probabilites preservee.

## Cycle 53 : Chronologie des bilans du journal

- **Lectures ordonnees** (`journal.py`) : les bilans, filtres et exports suivent
  la date du match puis l'emission convertie en UTC et l'identifiant stable du
  pronostic. Les journaux anciens sans heure d'emission utilisent `emis_le`.
  La lecture brute et les ecritures conservent l'ordre original du CSV.
- **Reproduction et tests** : quatre pronostics sur quatre jours donnaient un
  drawdown de 2 ou 3 unites et une serie de 2 ou 3 defaites selon l'ordre
  d'import. Les trois permutations testees donnent maintenant le meme bilan,
  le meme bootstrap a graine fixe et les memes pronostics ordonnes, sans
  reecriture du fichier. Tests des fuseaux horaires et du format ancien.
- **Resolution temporelle** : l'heure de reglement n'est pas disponible.
  L'emission departage les pronostics d'une meme journee ; les indicateurs
  intrajournaliers suivent cette convention et ne reconstituent pas l'ordre
  reel de cloture des rencontres.
