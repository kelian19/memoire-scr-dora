# Contrôles du dépôt

[Architecture](../../docs/ARCHITECTURE.md) · [Contribution](../../CONTRIBUTING.md)

| Contrôle | Commande depuis la racine | Ce qu'il établit |
|---|---|---|
| Structure et navigation | `python projet.py structure` | Fichiers indispensables présents, liens de fichiers des guides maintenus valides, casse exacte ; aucune donnée brute, signature ou fichier de plus de 20 Mio suivi. |
| Tests du contrôle de structure | `python -m unittest discover -s outils/ci -p "test_*.py"` | Les cas de liens cassés, casse incorrecte et fichiers interdits sont effectivement refusés. |
| Structure et harnais strict | `python projet.py verifier` | Comparaison aux sorties enregistrées ; pas de nouveau calcul. |
| Compilation et contrôles PDF | `python projet.py compiler` | Délègue à la chaîne existante, avec Tectonic et PyMuPDF. |
| Audit des correspondances | `python projet.py audit` | Recherche de correspondances à examiner ; pas une preuve de validité scientifique. |

Le contrôle de structure et ses tests utilisent uniquement Python et Git. Le [workflow GitHub Actions](../../.github/workflows/verification.yml) les exécute sur Windows et Linux ; il conserve le harnais et la compilation existants sur Linux.

`vitrine/` est un outillage historique de préparation d'une vitrine du dépôt. Les README de sa surcouche sont des copies ; les guides de travail maintenus sont ceux reliés depuis l'accueil.
