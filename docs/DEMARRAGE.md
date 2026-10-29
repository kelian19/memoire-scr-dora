# Démarrage de la version publique

Lire le [mémoire Institut actuel](../memoire/main_institut_150.pdf), le
[guide des versions](VERSIONS.md) et les [limites présentées dans l'accueil](../README.md).

```bash
python -m venv .venv
# Activer l'environnement, puis :
pip install -r requirements.txt
python projet.py structure
python outils/ci/harnais_strict.py
tectonic memoire/main_institut_150.tex
```

La compilation utilise les figures présentes. Les simulations nécessitant SAS OpRisk ou les
autres données brutes demandent un accès séparé. Les contrôles numériques peuvent être exécutés
sans ces bases, à partir des sorties versionnées.

L'[exporteur Overleaf](../outils/ci/exporter_overleaf_institut.py) prépare un projet autonome.
La [variante Nexialog](../memoire/overleaf/README.md) attend le fond du cabinet séparément.
