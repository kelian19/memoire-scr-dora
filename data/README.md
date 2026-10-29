# Données locales

[Architecture](../docs/ARCHITECTURE.md) · [Démarrage](../docs/DEMARRAGE.md)

| Dossier | Usage | Suivi Git |
|---|---|---|
| `raw/` | Sources brutes, dont bases de pertes sous licence. | Exclu ; ne jamais ajouter les données de force. |
| `processed/` | Données retraitées locales. | Les formats de données sont exclus par défaut ; examiner toute exception. |

Les noms et formats attendus sont définis par les scripts consommateurs. La présence d'un fichier local n'autorise pas sa publication. Les sorties destinées à la traçabilité du mémoire sont dans [resultats/sorties_verif/](../resultats/sorties_verif/README.md), après examen de leur contenu.

Le harnais et la compilation peuvent fonctionner sans les données brutes. Une reproduction scientifique complète des analyses qui les utilisent exige leur accès autorisé et l'environnement correspondant.
