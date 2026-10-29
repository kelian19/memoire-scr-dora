# Supplément du rapport ENSAE

Les fichiers S1 à S4 contiennent les adaptations par pilier, pistes de recherche,
questionnaire d'élicitation et diagnostics complémentaires au rapport court.
Le fichier `relais.tex` fournit les numéros de renvoi utilisés par le rapport.

Le [PDF du supplément](../materiel_complementaire/supplement_rapport_ENSAE.pdf) est distinct
du [rapport ENSAE](../main_ensae.pdf) et du [mémoire Institut actuel](../main_institut_150.pdf).
Le [constructeur](../materiel_complementaire/build_supplement.py) compile un maître temporaire
avec le rapport, puis extrait le supplément pour conserver ses renvois.

Les nombres sont contrôlés par `python outils/ci/harnais_strict.py`, depuis la racine du dépôt.
Les valeurs vraies des questions d'étalonnage restent hors publication.

Le matériel complémentaire dispose d'un [dépôt Zenodo](https://doi.org/10.5281/zenodo.22874524).
Une mise à jour de ce dépôt GitHub ne constitue pas une nouvelle publication sur Zenodo.
