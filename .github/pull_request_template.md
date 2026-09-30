## Ce que change cette pull request

<!-- En deux ou trois phrases : quoi, et pourquoi. -->

## Contrôles

- [ ] `python ci/harnais_strict.py` passe : aucun nombre non confirmé, aucun hors contrôle non déclaré
- [ ] la version Institut est régénérée (`build_institut.py`), jamais éditée à la main
- [ ] le mémoire compile, débordements au niveau de référence, moins de 200 pages
- [ ] aucune donnée brute, aucune image de signature dans les fichiers ajoutés
- [ ] tout nombre nouveau est imprimé par un script dont la sortie est versionnée dans `sorties_verif/`
