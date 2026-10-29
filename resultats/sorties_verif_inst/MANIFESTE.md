# Manifeste du lot 2000-2026

Dernier lancement le 2026-10-29T16:21:14+01:00, code du commit `3aee10c` plus les
modifications non committees du depot a cet instant. Variable : `OPRISK_ANNEE_MIN=2000`.
Interpreteur : `python -X utf8 -I`. Effectifs : 570 pertes de l'echantillon de severite, 86 exces,
582/27 pour le numerateur de la frequence (inchange). Les sorties absentes de ce dossier sont lues
dans `resultats/sorties_verif/` : ce sont celles des scripts qui ne dependent pas de l'echantillon.
Les lignes des scripts non rejoues au dernier lancement sont celles du lancement precedent.

Les expériences 119 et 120 sont propres à la reprise Institut : le calage historique
reste inchangé et le bootstrap réestime conjointement la calibration dans un moteur séparé.
Le 26/10/2026, le script 130 (commentaire, puis explication de la bascule et tableau des
probabilités annuelles d'incident et de charge positive) et le script 16 (libellés de la
figure S10) ont été rejoués : nombres antérieurs inchangés. Journaux dans
`docs/collaboration/local/journaux_reexecution_2026-10-26/`.

Exécution 2026-10-27T22:34:44+01:00 : scripts 131. Journaux détaillés : `docs/collaboration/local/journaux_reexecution/2026-10-27_223409_094220`.

Exécution 2026-10-27T22:34:46+01:00 : scripts 131, 132. Journaux détaillés : `docs/collaboration/local/journaux_reexecution/2026-10-27_223409_094220`.

Exécution 2026-10-27T22:34:48+01:00 : scripts 131, 132, 133. Journaux détaillés : `docs/collaboration/local/journaux_reexecution/2026-10-27_223409_094220`.

Exécution 2026-10-28T10:15:51+01:00 : scripts 134. Journaux détaillés : `docs/collaboration/local/journaux_reexecution/2026-10-28_101401_117524`.

Exécution 2026-10-29T10:46:06+01:00 : scripts 135. Journaux détaillés : `docs/collaboration/local/journaux_reexecution/2026-10-29_104548_740893`.

Exécution 2026-10-29T14:22:05+01:00 : scripts 136. Journaux détaillés : `docs/collaboration/local/journaux_reexecution/2026-10-29_142002_279900`.

Exécution 2026-10-29T14:24:10+01:00 : scripts 136. Journaux détaillés : `docs/collaboration/local/journaux_reexecution/2026-10-29_142215_621220`.

Exécution 2026-10-29T15:06:27+01:00 : scripts 136. Journaux détaillés : `docs/collaboration/local/journaux_reexecution/2026-10-29_150418_890950`.

Exécution 2026-10-29T15:07:55+01:00 : scripts 136, 137. Journaux détaillés : `docs/collaboration/local/journaux_reexecution/2026-10-29_150418_890950`.

Exécution 2026-10-29T15:10:24+01:00 : scripts 137. Journaux détaillés : `docs/collaboration/local/journaux_reexecution/2026-10-29_150855_358343`.

Exécution 2026-10-29T15:32:55+01:00 : scripts 137. Journaux détaillés : `docs/collaboration/local/journaux_reexecution/2026-10-29_153039_804918`.

Exécution 2026-10-29T15:51:17+01:00 : scripts 138. Journaux détaillés : `docs/collaboration/local/journaux_reexecution/2026-10-29_155029_481460`.

Exécution 2026-10-29T15:55:29+01:00 : scripts 138. Journaux détaillés : `docs/collaboration/local/journaux_reexecution/2026-10-29_155433_010266`.

Exécution 2026-10-29T16:06:07+01:00 : scripts 138. Journaux détaillés : `docs/collaboration/local/journaux_reexecution/2026-10-29_160531_160436`.

Exécution 2026-10-29T16:21:14+01:00 : scripts 138. Journaux détaillés : `docs/collaboration/local/journaux_reexecution/2026-10-29_162001_352230`.

| script | fichier | code | Traceback | duree | octets | md5 |
|---|---|---|---|---|---|---|
| 14 | 14_robustesse_delta_cascade.py | 0 | non | 92 s | 1870 | 6b00d0ba |
| 16 | 16_scr_multi_etats_global.py | 0 | non | 197 s | 4246 | 6a47044d |
| 46 | 46_var_predictive.py | 0 | non | 48 s | 4139 | 72ad933b |
| 47 | 47_validation_adequation.py | 0 | non | 210 s | 9495 | a202e723 |
| 48 | 48_bande_modele.py | 0 | non | 20 s | 2670 | 45b33a43 |
| 51 | 51_scr_robuste.py | 0 | non | 249 s | 5065 | 72420fe8 |
| 63 | 63_calibration_figee.py | 0 | non | 1 s | 34146 | 47a55461 |
| 71 | 71_test_retrait_etage_modele.py | 0 | non | 5 s | 7159 | 10987fe6 |
| 72 | 72_intervalle_reporte_95.py | 0 | non | 42 s | 6013 | b03d42b5 |
| 89 | 89_backtest_hors_echantillon.py | 0 | non | 15 s | 15063 | f7f7ec9c |
| 90 | 90_invariance_ecart_derive.py | 0 | non | 30 s | 9402 | f3edd16e |
| 91 | 91_backtest_charge_agregee.py | 0 | non | 19 s | 13961 | 740640d7 |
| 93 | 93_regle_de_seuil.py | 0 | non | 738 s | 10890 | a097d0ee |
| 102 | 102_derive_nominale_ou_reelle.py | 0 | non | 16 s | 2635 | f25fc1df |
| 104 | 104_profilage_variables.py | 0 | non | 101 s | 24527 | ed9b0fa6 |
| 106 | 106_influence_plus_grosses_pertes.py | 0 | non | 41 s | 7060 | eddc0ed0 |
| 118 | 118_unite_observee_simulee.py | 0 | non | 9 s | 5019 | 6aa1b246 |
| 119 | 119_unite_perte_capital_institut.py | 0 | non | 3 s | 3438 | 224d5a28 |
| 120 | 120_bootstrap_joint_institut.py | 0 | non | 243 s | 8111 | 92c8c225 |
| 123 | 123_melange_predictif_institut.py | 0 | non | 129 s | 2833 | dc8271fa |
| 124 | 124_validation_temporelle_institut.py | 0 | non | 21 s | 5982 | b83b442d |
| 125 | 125_reechantillonnage_annuel_institut.py | 0 | non | 194 s | 3624 | cb3be8a1 |
| 126 | 126_dependance_couts_incident_institut.py | 0 | non | 1 s | 3887 | 79037f2d |
| 127 | 127_extrema_hors_sommets_institut.py | 0 | non | 269 s | 2993 | 8915c4bd |
| 128 | 128_residus_compenses_comptage_institut.py | 0 | non | 237 s | 3790 | 37f67c45 |
| 129 | 129_seuils_hugo_institut.py | 0 | non | 242 s | 1993 | 4ee0782d |
| 130 | 130_exposition_taille_entite.py | 0 | non | 277 s | 9537 | 0b27cb20 |
| 131 | 131_benchmark_covariables_depassement.py | 0 | non | 35 s | 5665 | efa8bb0a |
| 132 | 132_pilote_codage_textes.py | 0 | non | 2 s | 2410 | 28480676 |
| 133 | 133_codage_textes_complet.py | 0 | non | 2 s | 3512 | 7cc1f2c9 |
| 134 | 134_bruit_extrema_institut.py | 0 | non | 110 s | 999 | 395e0c3b |
| 135 | 135_figure_hill_institut.py | 0 | non | 18 s | 1930 | d42aaf23 |
| 136 | 136_descente_grandes_institutions.py | 0 | non | 128 s | 11553 | 27bbcf4d |
| 137 | 137_descente_diagnostics.py | 0 | non | 135 s | 14134 | 690af3a2 |
| 138 | 138_descente_observation.py | 0 | non | 73 s | 1591 | 7849aeef |

Publication publique : les empreintes des sorties dont les chemins locaux ont ete retires
sont recalculees sur les fichiers nettoyes. Les codes de retour et dates restent ceux des
executions sources, controlees avant nettoyage ; aucune simulation n'est relancee ici.
Voir `TRANSFORMATIONS_PUBLICATION.json` pour les anciennes et nouvelles empreintes.
