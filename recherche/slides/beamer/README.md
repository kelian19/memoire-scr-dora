# Supports de soutenance

Le [support ENSAE](soutenance_ensae.pdf) accompagne le rapport court
[`main_ensae.pdf`](../../../memoire/main_ensae.pdf). Il comporte 21 diapositives principales sur
23 pages PDF, puis treize réserves R1 à R13, pages 24 à 37.
Les autres compléments après dépôt sont signalés dans le support.
Le fichier ENSAE courant contient des corrections postérieures au dépôt ; le script de préparation
du dépôt de travail distingue la copie du jury et ces corrections.
Le choix des paramètres distingue les bases issues des données, les hypothèses entre états
et les plages de scénario ; le multiplicateur de fréquence est détaillé par vecteur d'attaque.
Le seuil et les paramètres sont détaillés dans les réserves R1 et R2. La descente à l'entité
figure dans l'exposé, en diapositive 18, avec son statut de complément après dépôt non validé.

Le [support Institut](soutenance_institut.pdf) est distinct et accompagne
[`main_institut_150.tex`](../../../memoire/main_institut_150.tex). Il comporte un parcours de
25 diapositives sur 28 pages PDF, puis des diapositives de réserve pour les questions et des annexes.
Le PDF complet compte 79 pages. Les parcours ENSAE et Institut sont distincts.

Depuis la racine du dépôt :

```bash
python outils/ci/compiler_soutenance.py
python outils/ci/compiler_soutenance.py --deck institut
```

La [direction artistique](DIRECTION_ARTISTIQUE.md) est définie dans `direction_artistique.tex`.
Les schémas sources sont dans [figures_soutenance](../figures_soutenance/).
Les notes de répétition et comptes rendus internes restent dans le dépôt de travail.

Les tableaux `table129_*.tex` proviennent de `129.json` et peuvent être actualisés par
`python outils/ci/actualiser_complements_hugo.py`. Les calculs scientifiques demandent la base
sous licence ; les sorties déjà publiées permettent de contrôler la concordance des nombres.
