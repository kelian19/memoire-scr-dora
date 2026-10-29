# Modules Python réutilisables

[Architecture du dépôt](../docs/ARCHITECTURE.md)

Ces modules sont encore importés par des scripts du projet. Ils restent à leur emplacement actuel.

| Dossier | Responsabilité |
|---|---|
| `frequency/` | Modèles de fréquence. |
| `severity/` | Sévérité, GPD et traitements des sources de pertes. |
| `aggregation/` | LDA, dépendance par copule et allocation d'Euler. |
| `compliance/` | Variable latente de conformité. |
| `scenarios/` | Scénarios, pont latent et bootstrap. |
| `utils/` | Configuration de cette chaîne de calcul. |
| `visualization/` | Fonctions de représentation. |

Le moteur de cascade courant se trouve dans [recherche/vasicek_lab/](../recherche/vasicek_lab/README.md). Ne pas présumer que les paramètres de `src/utils/config.py` pilotent toutes les expériences du laboratoire. Rechercher les imports et le script producteur avant de modifier une calibration ou de mutualiser du code.
