# vasicek_lab : la carte

Laboratoire du modèle de cascade dirigée. Chaque script est autonome, produit une figure
dans `figures/` et imprime ses diagnostics.

## Règles de structure, à ne pas casser

- **Le moteur reste à la racine.** `scr_engine.py`, `euro_cascade_model.py`,
  `severite_model.py`, `frequence_model.py`, `partial_id.py` sont les seuls modules
  importés par les scripts. Les déplacer casserait tout le dossier.
- **`partial_id.py`** porte le dispositif d'identification partielle (décomposition
  `W = S + A`, ensemble admissible, évaluateur à nombres aléatoires communs). Les
  scripts 30 et 31 en dépendent tous les deux : c'est ce qui garantit qu'ils publient
  les **mêmes** bornes. Ne pas dupliquer ce noyau ailleurs.
- **`figures/` reste à la racine.** Le mémoire le vise en dur
  (`\graphicspath{{../recherche/vasicek_lab/figures/}{../recherche/cascade_qualitative/figures/}}`).
  Tous les scripts y écrivent, quel que soit leur sous-dossier.
- **Aucun script numéroté n'est importé par un autre.** On peut donc les déplacer
  librement, à condition que l'expression qui calcule le dossier depuis `__file__`
  remonte au niveau de `vasicek_lab`.

## Lancement

Les [approfondissements Institut](../../resultats/institut_approfondissements/README.md)
utilisent `institut_incertitude.py`, le script `119` dans `8_benchmarks/` et le
script `120` dans `1_fondations/`. Ils conservent le moteur et le calage historiques.

La feuille de route de janvier, conservee dans le depot de travail, commence
avec `121` dans `5_etats/`, le module `institut_remediation.py` et ses tests
scientifiques dans `tests/`. Les résultats restent expérimentaux et séparés du manuscrit.
Le script `122`, dans `5_etats/`, poursuit avec les frontières d'efficacité et
de coût annuel entre P1 et P4, en incluant les calibrations extrêmes de `120`.
Le script `123`, dans `1_fondations/`, relit les calibrations de `120` et compare le
quantile annuel au calage central, la distribution des quantiles conditionnels et le
quantile du mélange des lois annuelles. Le script `124`, au même endroit, compare en
origine glissante une sévérité stationnaire, à échelle variable ou à deux régimes, ainsi
qu'un taux de dépassement constant ou en tendance. Le script `125` rééchantillonne les années
complètes, ou des blocs de trois années, pour propager aussi la fréquence relative des pertes enregistrées.
Le script `126`, dans `8_benchmarks/`, relie les constructions additive et amplifiée du `119` par une
dépendance graduée entre les composantes de coût d'un incident, à marges égales.
Le script `127`, dans `6_contagion/`, cherche des extrema du capital hors des sommets du pavé des
directions (recherche de contre-exemples, sans certification).
Le script `128`, dans `2_donnees/`, diagnostique par ses résidus compensés le comptage journalier de brèches
(lit `data/raw/cisa_kev.json`, non versionné).

```powershell
.venv\Scripts\python.exe recherche\vasicek_lab\<dossier>\<script>.py
```

## Le contenu, par étape du raisonnement

### 1_fondations : pourquoi un Vasicek dirigé
| script | figure | ce qu'il établit |
|---|---|---|
| `01_non_transitivite` | G1 | TRANS est asymétrique, sa symétrisée n'est pas PSD, les corrélations violent la transitivité d'un modèle à facteur unique. C'est la justification de tout le reste. |
| `02_seuil_Ki_horserace` | J | banc d'essai du seuil K_i, Phi^-1(PD) contre EVT |
| `04_seuil_Kj` | L | d'où vient le seuil K_j, ce que l'EVT peut et ne peut pas |
| `06_facteur_systemique` | N_facteur | calibration du facteur systémique Y_j et de la matrice Sigma_Y |
| `build_network_figure` | H1 | la structure dirigée de W sur les 5 piliers |

### 2_donnees : ce que la donnée fixe, et ce qu'elle refuse
| script | figure | ce qu'il établit |
|---|---|---|
| `05_faisabilite_donnees` | M | **W n'est pas calibrable.** Placebo directionnel z = -0,33. Le cœur du chapitre identifiabilité. |
| `08b_calibration_frequence_entree` | N_frequence | la fréquence d'entrée est calibrée, pas posée |
| `08c_hawkes_frequence_entree` | O | rejet du Hawkes : l'excitation est intra-journalière |
| `08d_poisson_compose_batch` | P | Poisson composé à paquets, signature MOVEit |
| `08h_hawkes_variantes_bessy_roland` | O2 | le rejet tient face au noyau à retard et à la logique two-phase |
| `28_test_vcdb_sous_piliers` | X | sous-piliers : P2 et P3 atteignables en open data, P4 s'effondre |
| `35_event_study_moveit` | Z6 | event-study du choc P4 MOVEit : DiD dévoré par le placebo (net nul), cible non attribuable. La tentative d'identification la plus dure, négative et assumée. |
| `57_biais_taille_et_troncature` | S20 | les deux biais déclarés d'OpRisk, mesurés : élasticité sévérité/taille par EMV tronquée (b = 0,087), et chain-ladder sur le triangle survenance x saisie |
| `60_descente_echelle_entite` | S21 | **la descente secteur vers entité, cohérente.** lambda estimé comme fonction de la taille (b_lambda = 0,074) et lu à la taille cible au lieu d'un seau, sévérité transposée à la même taille, les deux composées. SCR d'entité **169 M**, bande **[131,5 ; 183,3]**, socle **78,1**. |

> **Attention au protocole du panel firme-année.** Le filtre annuel s'applique AVANT le
> calcul de la fenêtre d'observation. L'inverser allonge les spans et fait tomber lambda
> de 0,10 à 0,04. Les scripts 08b et 60 doivent toujours reproduire les mêmes deux seaux
> (0,0988 toutes firmes, 0,2102 pour les firmes à au moins 10 événements).
>
> **Attention à la précision de lambda.** Les scripts 58 et 60 publient tous deux le SCR
> d'entité. Arrondir lambda à quatre décimales suffit à les faire diverger de 2 %. La
> valeur se reporte à pleine précision (le script 60 l'imprime pour cela).

### 3_marges : sévérité et fréquence
`07_severite` (S1), `08_frequence` (S2). Les deux marges, en unités normalisées.

### 4_scr : du modèle au capital
`09_agregation_scr` (S3), `10_sensibilite_scr` (S4), `11_copules_allocation` (S5),
puis le passage en euros : `12_scr_euro_cascade` (S6), `13_delta_dora_cascade` (S7),
`14_robustesse_delta_cascade` (S8), `15_tornado_robustesse_cascade` (S9).

### 5_etats : la conformité multi-états
`16_scr_multi_etats_global` (S10), `16b_scr_multi_etats_par_pilier` (S11),
`17_markov_trajectoire_scr` (S12), `18_priorisation_remediation` (S13),
`19_robustesse_multietats` (S14), `20_allocation_shapley_euler` (S15),
`20b_robustesse_interaction`, `21_prc_jacobs_residu` (S16),
`22_sensibilites_hypotheses` (S17), `24_scr_dora_distribution` (T).

> `18` énumère les 120 ordres de remédiation et `20` fait l'allocation de Shapley.
> Ce sont les deux endroits où la direction de la cascade a un effet du premier ordre,
> parce qu'un ordre est un objet discret qui bascule ou ne bascule pas.

### 6_contagion : W, et jusqu'où on peut aller
| script | figure | ce qu'il établit |
|---|---|---|
| `03_calibration_W` | K1, K2 | le protocole de calibration tenté, et son échec documenté |
| `25_scr_continu_contagion` | U | le SCR borné sur l'**amplitude** g de la contagion |
| `29_cascade_branchement` | Y | marche auto-évitante contre branchement multitype : SCR inchangé à 4 %, résultat de robustesse |
| `30_identification_partielle_W` | Z | le SCR borné sur la **direction** : W = S + A, S identifiée, A libre. Bornes exhaustives sur les 1024 sommets, valeur de l'information, priorité minimax. |
| `31_comparaison_regimes` | Z2 | les deux régimes face à face **avant élicitation** : poser W (un point) contre le borner (socle + bande). Test de biais du panel. |

> Bornes de référence (t = 1, ignorance totale de la direction) : socle **5275 M**,
> SCR dans **[6858 ; 8697] M**. Déterministes, énumération exhaustive des sommets.
> Les scripts 30 et 31 doivent toujours s'accorder sur ces valeurs ; si ce n'est plus
> le cas, c'est que quelqu'un a dupliqué le noyau au lieu d'importer `partial_id`.

### 7_elicitation
`105_graines_elicitation` : les dix graines du questionnaire et leurs quatre contrôles, valeurs
masquées dans cette version. Le moteur de Cooke (`26`, `26b`) reste hors de la version publique
tant que l'élicitation est ouverte. Le questionnaire est dans `../cascade_qualitative/elicitation/`.

### 8_benchmarks
`23_benchmark_copule` (S18), `27_benchmark_formule_standard` (W). Ensemble : la Formule
Standard exprime 0 % de l'effet DORA, la copule 62 %, la cascade 100 %.

### 9_cas_usage
`08e` (Q, pilier 2), `08f` (R, pilier 3), `08g` (S, pilier 4). Orientés KPI.

### notes/
`cours_memoire_ensae` : le cours de préparation à la soutenance, chapitre par chapitre. Les notes de
travail restent dans le dépôt de travail.
