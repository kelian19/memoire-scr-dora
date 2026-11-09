# Architecture de la version publique

- [memoire/](../memoire/) : manuscrits, bibliographies, chapitres et figures incluses.
- [recherche/vasicek_lab/](../recherche/vasicek_lab/) : expériences et modules de calcul.
- [recherche/cascade_qualitative/](../recherche/cascade_qualitative/) : cascade et questionnaire publiable.
- [src/](../src/) et [src/utils/config.py](../src/utils/config.py) : modules réutilisés et configuration de la chaîne LDA ; toutes les expériences ne lisent pas cette configuration.
- [resultats/sorties_verif/](../resultats/sorties_verif/) : sorties historiques.
- [resultats/sorties_verif_inst/](../resultats/sorties_verif_inst/) : surcharges Institut et manifeste.
- [resultats/institut_approfondissements/](../resultats/institut_approfondissements/) : compléments numériques.
- [outils/ci/](../outils/ci/) : contrôles et exporteur Overleaf.

Les notes privées, les données brutes et les outils internes de collaboration ne font pas partie
de cet export. Les documents de lecture sont répertoriés dans l'[accueil](../README.md).

La [carte des calculs](../recherche/vasicek_lab/CARTE_CALCULS.md) indique le moteur courant,
ses scripts producteurs et les expériences historiques. L'[exemple synthétique](../recherche/vasicek_lab/exemples/README.md)
permet de suivre le calcul sans accès aux bases restreintes.
