# Contribuer

Ce dépôt est un export public du mémoire et de ses compléments. Pour signaler une erreur,
préciser le document, la page, le passage et la sortie numérique concernée. Les sorties sont
dans `resultats/sorties_verif/` et `resultats/sorties_verif_inst/`.

Pour une correction, ouvrir une pull request contre `main` et exécuter :

```bash
python projet.py structure
python outils/ci/harnais_strict.py
python -m unittest discover -s outils/ci -p "test_*.py"
```

Pour les changements LaTeX, compiler le document concerné avec Tectonic et contrôler les
références, les citations, les débordements et la pagination. Les données brutes et les images
de signatures ne sont pas ajoutées au dépôt.

Chaque résultat nouveau doit avoir une source et un protocole explicites. Les hypothèses,
sensibilités et calculs effectivement exécutés doivent rester distincts. Les recalibrations
postérieures au rapport ENSAE sont identifiées comme telles ; l'ancienne version Institut
conservée reste séparée du manuscrit Institut actuel.
