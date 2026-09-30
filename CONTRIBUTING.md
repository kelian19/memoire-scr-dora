# Contribuer

Ce dépôt est la version publique d'un mémoire déposé : son texte ne change plus que pour corriger
une erreur. Les contributions les plus utiles sont donc des signalements.

## Signaler une erreur

Ouvrir une issue avec le modèle « Erreur dans le mémoire » : la page, la phrase ou le nombre en
cause, et ce qui devrait y figurer. Un nombre se vérifie contre la sortie du script que sa section
cite, dans `sorties_verif/`.

## Proposer une correction

Une pull request contre `main`. La vérification continue rejoue trois contrôles, qu'il vaut mieux
lancer d'abord en local :

| Contrôle | Commande |
|---|---|
| chaque nombre publié est confirmé par la sortie de son script | `python ci/harnais_strict.py` |
| la version Institut est à jour de la version ENSAE | `python exploratory/memoire_cascade/build_institut.py` |
| compilation des deux versions et contrôles du PDF | `python exploratory/memoire_cascade/refaire_tout.py --sans-harnais` |

Avec `make` : `make verifier`, `make institut`, `make memoire`. Les garde-fous du poste
s'installent une fois, `pip install pre-commit` puis `pre-commit install` : ils bloquent tout
commit de donnée brute ou d'image de signature et les fichiers de plus de 20 Mo.

Une correction acceptée est reportée dans le dépôt de travail, et revient ici avec la version
suivante.

## Règles du projet

- **Aucune donnée brute.** La base SAS OpRisk est sous licence : le dépôt versionne les sorties
  des scripts (`sorties_verif/`), pas la donnée.
- **Aucun nombre écrit à la main.** Chaque section déclare les scripts dont elle tire ses nombres
  (`% SOURCES-SCRIPTS: …`), et le harnais les confronte à leurs sorties. Un superlatif ou un
  comptage se calcule dans le script, il ne s'écrit pas.
- **`main_institut.tex` est généré** par `build_institut.py` : ne jamais l'éditer.
- **La calibration est gelée depuis le 7 août 2026** : les corrections d'erreur sont admises, les
  recalibrations non.
