# Approfondissements Institut — octobre 2026

## Descente a l'entite : complements du 29 octobre

- [136.txt](../sorties_verif_inst/136.txt) : frequence sur les grandes institutions, pertes sous le seuil, VaR/ES et comparaison des entites ; entite notionnelle a 24,1 M EUR, scenario conditionnel.
- [137.txt](../sorties_verif_inst/137.txt) : bootstrap par entreprise, plan factoriel et decomposition additive des effets, sans validation d'un capital individuel.
- [138.json](138.json) : sensibilite a l'enregistrement des incidents ; les probabilites utilisees sont des scenarios, pas des taux estimes.

## Experiences anterieures

Ces experiences prolongent le protocole de reprise Institut du 20 octobre.
Elles concernent la variante `main_institut_150`, dont les sources sont dans `chapitres_inst/`.

- [119.json](119.json) : quatre constructions de perte sur les seize configurations,
  attributions de Shapley, interaction, contrôles numériques et comparaison Monte-Carlo.
- [120.json](120.json) : bootstrap conjoint des pertes entières, paramètres réestimés,
  résultats appariés par réplication, journal des échecs et intervalles conditionnels.

- [123.json](123.json) : mélange prédictif annuel des 2 000 calibrations de 120, quantiles
  du mélange à quatre niveaux, comparaison au calage central et aux quantiles conditionnels.

- [124.json](124.json) : validation chronologique (origine glissante 2014-2025) de trois lois
  de sévérité des excès et du taux de dépassement, et capital pour 2026 selon la loi retenue.

- [125.json](125.json) : rééchantillonnage annuel conjoint (années ou blocs de trois années), taux de
  dépassement, queue et fréquence relative réestimés, capitaux avec et sans propagation de la fréquence.

- [126.json](126.json) : dépendance entre les coûts d'un même incident à marges fixées, par un mélange
  de copules (choc commun de probabilité rho) et par des blocs de composantes comonotones ; capitaux
  aux seize configurations et décomposition nombre de composantes, dépendance, amplification.

- [127.json](127.json) : recherche d'extrema du capital hors des sommets du pavé des directions (moyenne
  lissée, ascension exacte par coordonnées, évolution différentielle du capital, condition du premier ordre).

- [128.json](128.json) : résidus compensés du comptage journalier de brèches (tendance, excitation interne,
  excitation externe du catalogue KEV), seuils recalibrés par bootstrap avec réestimation.

Les pertes individuelles restent dans `data/raw/`, sous licence. Les JSON ne contiennent
que des paramètres ajustés et des résultats agrégés. Les sorties lisibles et leurs
empreintes sont dans `../sorties_verif_inst/`.

Le complément `129.json` répond au point Hugo du 22 octobre : balayage de seuils avec KS
bilatéral et Anderson–Darling réestimés par bootstrap paramétrique, et comparaison EMV,
PWM, Hill et moments avec IC à 95 % sur les 86 excès des 570 pertes de calibration.
Il ne remplace pas les constantes gelées. Les IC sont conditionnels au seuil ; les quantiles
de sévérité EMV/PWM gardent aussi le taux de dépassement fixé. Aucun tirage n'est éliminé
au seul motif que l'indice atteint ou dépasse un. Les tables et la figure des slides sont
reproduites par `outils/ci/actualiser_complements_hugo.py`.

Depuis la racine, sous PowerShell :

```powershell
.venv\Scripts\python.exe -X utf8 recherche/vasicek_lab/8_benchmarks/119_unite_perte_capital_institut.py > resultats/sorties_verif_inst/119.txt
.venv\Scripts\python.exe -X utf8 recherche/vasicek_lab/1_fondations/120_bootstrap_joint_institut.py --replications 2000 > resultats/sorties_verif_inst/120.txt
.venv\Scripts\python.exe -X utf8 outils/ci/actualiser_approfondissements_institut.py
```

Le module `recherche/vasicek_lab/institut_incertitude.py` réutilise les tables exactes
du moteur. L'inversion de Fourier est inclinée exponentiellement et conserve la
sous-probabilité sur la portée : aucune renormalisation ni masse résiduelle ajoutée.
Les contrôles font varier le pas, la portée et l'inclinaison ; ils comparent aussi le
cas d'une seule perte par incident et une limite à sévérité déterministe calculable
par la loi binomiale négative amincie.

Les quatre règles sont des scénarios. La perte moyenne conserve plusieurs Bernoulli
et peut être sous le seuil ; la perte unique neutralise financièrement les canaux
qui ne modifient que le nombre de piliers. L'amplification par ce nombre est posée.

Le bootstrap est iid, à seuil fixé. La fréquence historique, la dispersion, les
multiplicateurs entre états et la cascade d'expert sont conditionnants. Il ne corrige
ni la sélection des pertes, ni la non-stationnarité, ni l'année 2026 incomplète.
Les quantiles bootstrap de l'écart et du rapport utilisent chaque paire d'états,
sans soustraction de bornes marginales. Les ajustements admissibles avec xi >= 1
sont conservés ; leur moyenne théorique est infinie, leur quantile annuel reste défini.
