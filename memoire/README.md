# Manuscrits

- [Institut actuel](main_institut_150.pdf) : 158 pages, avec correctifs de novembre.
- [ENSAE courant](main_ensae.pdf) : 100 pages.
- [Institut conservé](main_institut.pdf) : ancienne version de 195 pages.
- [Variante Overleaf Nexialog](overleaf/README.md).

Les sources `main_institut_150.tex`, `main_ensae.tex` et `main_institut.tex` lisent des dossiers
de chapitres distincts. Les bibliographies correspondantes sont versionnées. Pour compiler :

```bash
tectonic memoire/main_institut_150.tex
```

Le [guide des versions](../docs/VERSIONS.md) précise les statuts. Les images de signatures ne sont
pas distribuées ; leur absence est gérée par la couverture.

La version Institut est un document de travail : sa cible de remise reste de 150 pages. La chaîne
`python projet.py depot` bloque tant que ce dépassement n'est pas résolu. Une compilation directe
avec Tectonic produit un PDF de contrôle sans attester sa conformité à cette cible.
