# memoire_cascade : le mémoire

## Compiler

Depuis la racine du dépôt :

```bash
python ci/harnais_strict.py                                         # les nombres, contre les sorties des scripts
python exploratory/memoire_cascade/refaire_tout.py --sans-harnais   # génère, compile, contrôle
```

`refaire_tout.py` régénère la version Institut, compile les deux versions avec tectonic 0.16.9,
puis contrôle les débordements, les renvois cassés (« ?? » comptés dans le PDF) et le nombre de
pages. Il ne corrige rien : son code de sortie vaut 1 si un contrôle échoue.

## Où est quoi

| Chemin | Rôle |
|---|---|
| `main_ensae.tex` | fichier maître du mémoire déposé à l'ENSAE : classe, préambule, ordre de lecture |
| `main_institut.tex` | version Institut des actuaires, **générée** par `build_institut.py` depuis la précédente, sans la partie « Le stage ». Ne jamais l'éditer |
| `preambule_v2.tex` | paquets, format de l'école (Times 12, interligne 1,5), encadrés, macros |
| `page_de_garde.tex`, `logos/` | couverture au gabarit de l'école. Les images de signature ne sont pas distribuées : la couverture compile sans elles |
| `chapitres/` | un fichier par chapitre ou annexe. Le nom du fichier ne dit pas la partie : `12b_adaptations_pilier.tex` est une annexe. `02_introduction.tex` n'est plus appelé, son contenu ayant rejoint le chapitre 1 |
| `remerciements.tex`, `note_synthese_separee.tex` | remerciements ; note de synthèse séparée, française et anglaise, qui reprend celle du mémoire sans la recopier |
| `references.bib`, `plainnat-fr.bst` | bibliographie, et le style `plainnat` traduit en français |
| `figures_externes/` | les seules figures non produites par un script, reproduites avec autorisation. Provenance dans son `README.md` |
| `materiel_complementaire/` | sources et PDF du matériel complémentaire publié sur Zenodo, et `build_matcomp.py` qui le régénère |
| `refaire_tout.py` | la chaîne complète |
| `verif_chiffres.py` | le harnais : confronte chaque nombre d'une section aux sorties des scripts qu'elle cite |
| `audit_sources.py` | signale les confirmations du harnais qui ne tiennent qu'à l'arrondi ou à une occurrence unique, à relire à la main |
| `controles_memoire.py`, `audit_figures.py`, `audit_style.py`, `carte_figures.py` | contrôles du PDF produit, taille imprimée du texte des figures, marqueurs de registre, carte des figures |
| `controles_depot/` | statut des liens externes du PDF, comparaison du matériel complémentaire à la version publiée |

## Règles de rédaction

- **Aucun nombre écrit à la main.** Chaque section déclare ses scripts sources
  (`\texttt{NN}` dans le texte, ou `% SOURCES-SCRIPTS: NN` en commentaire), et le harnais cherche
  chaque nombre publié dans leurs sorties. Une section qui publie des nombres sans source se
  déclare hors script avec son motif (`% HARNAIS-HORS-SCRIPT:` ou `% HARNAIS-HORS-SECTION:`).
- **Le nom de fichier d'une figure ne porte aucun chiffre** : le harnais lirait ce chiffre comme
  un nombre publié.
- `\VaR` et `\TVaR` sont réservées à la charge annuelle agrégée, `\qsev` et `\qbarsev` à la
  sévérité d'un sinistre.
- On écrit « tableau », jamais « table », pour l'objet numéroté ; l'euro s'écrit `\euro{}` ; les
  acronymes en capitales pleines ; l'italique par `\emph`.
