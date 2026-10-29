# Supports de soutenance

Le [support ENSAE](soutenance_ensae.pdf) accompagne le rapport court
[`main_ensae.pdf`](../../../memoire/main_ensae.pdf). Il comporte 22 diapositives sur 25 pages
PDF, du fait des étapes d'affichage. Les compléments après dépôt sont signalés dans le support.

Le [support Institut](soutenance_institut.pdf) est distinct et accompagne
[`main_institut_150.tex`](../../../memoire/main_institut_150.tex).

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
