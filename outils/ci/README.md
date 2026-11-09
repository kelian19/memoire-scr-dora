# Contrôles du dépôt

[Architecture](../../docs/ARCHITECTURE.md) · [Contribution](../../CONTRIBUTING.md)

| Contrôle | Commande depuis la racine | Ce qu'il établit |
|---|---|---|
| Structure et navigation | `python projet.py structure` | Fichiers indispensables présents, liens de fichiers des guides maintenus valides, casse exacte ; aucune donnée brute, signature ou fichier de plus de 20 Mio suivi. |
| Tests du contrôle de structure | `python -m unittest discover -s outils/ci -p "test_*.py"` | Les cas de liens cassés, casse incorrecte et fichiers interdits sont effectivement refusés. |
| Tests scientifiques | `python -m pytest recherche/vasicek_lab/tests src/tests -q` | Propriétés des modèles et régressions de simulation sur données synthétiques, après installation de `requirements-test.txt`. |
| Structure et harnais strict | `python projet.py verifier` | Comparaison aux sorties enregistrées ; pas de nouveau calcul. |
| Compilation et contrôles PDF | `python projet.py compiler` | Délègue à la chaîne existante, avec Tectonic et PyMuPDF. |
| Audit des correspondances | `python projet.py audit` | Recherche de correspondances à examiner ; pas une preuve de validité scientifique. |

Le contrôle de structure et ses tests utilisent uniquement Python et Git. Le [workflow GitHub Actions](../../.github/workflows/verification.yml) les exécute sur Windows et Linux ; il exécute aussi les tests scientifiques, le harnais et la compilation des trois versions sur Linux.

Les tests du moteur courant vérifient aussi la fréquence nulle, les probabilités invalides,
l'absence d'auto-propagation et les deux capitaux de référence à graines fixes. Une graine commune
ne garantit pas une comparaison monotone des pertes lorsque le taux de matérialité change.

Les commandes complètes des approfondissements sont dans leur
[guide de reproduction](../../resultats/institut_approfondissements/README.md).
Dans `lancer_lot_institut.py`, `119` désigne la variante de `8_benchmarks/` et `120` lance
2 000 réplications. Le lot sans argument reste celui des seize scripts historiques.

`vitrine/` prépare l'export public contrôlé. Sa surcouche porte les guides propres à cette publication ;
les guides du dépôt de travail sont ceux reliés depuis l'accueil. La liste de fichiers publiés et les
balayages de confidentialité sont définis dans `vitrine/exporter.py` ; ce dossier n'est pas exporté.
