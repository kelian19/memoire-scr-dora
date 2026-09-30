# sorties_verif : les sorties des scripts

Un fichier `NN.txt` par script cité dans le mémoire : sa sortie texte, versionnée. Le harnais,
`exploratory/memoire_cascade/verif_chiffres.py`, cherche chaque nombre publié d'une section parmi
les sorties des scripts que cette section cite, à l'arrondi d'écriture près.

```bash
# tout le mémoire, échoue au moindre écart
python ci/harnais_strict.py

# un chapitre, avec le détail des nombres
python exploratory/memoire_cascade/verif_chiffres.py \
    sorties_verif exploratory/memoire_cascade/chapitres/12_resultats.tex
```

Le harnais vérifie d'où vient un nombre, pas ce qu'une phrase en fait : un nombre confirmé peut
être mal commenté. Et plus une section cite de scripts, plus un petit entier y trouve une
correspondance par hasard. `exploratory/memoire_cascade/audit_sources.py` signale les
confirmations qui ne tiennent qu'à l'arrondi ou à une occurrence unique, et imprime la ligne
d'appui, qui se relit à la main.

## Différences avec le dépôt de travail

Quelques sorties diffèrent de celles du dépôt de travail, sans qu'aucun nombre du mémoire n'en
dépende :

- `65.txt` porte `Identite A` à `Identite D` à la place des raisons sociales des quatre
  assureurs, que le mémoire anonymise ;
- `105.txt` masque les valeurs vraies des dix graines du questionnaire d'élicitation ;
- `08e.txt`, `28.txt`, `44.txt` et `45.txt` ne nomment plus les praticiens consultés ;
- les chemins absolus des postes de travail sont retirés de toutes.
