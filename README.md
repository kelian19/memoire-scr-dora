# Quantification du SCR lié à la non-conformité au règlement DORA

**Mémoire d'actuariat — ENSAE Paris · Nexialog Consulting**

[![Vérification](https://github.com/kelian19/memoire-scr-dora/actions/workflows/verification.yml/badge.svg?branch=main)](https://github.com/kelian19/memoire-scr-dora/actions/workflows/verification.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22874524.svg)](https://doi.org/10.5281/zenodo.22874524)
[![Code : MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Contenu : CC BY 4.0](https://img.shields.io/badge/contenu-CC%20BY%204.0-lightgrey.svg)](LICENSE-CONTENU.md)

Auteur : **Kélian Kaddouri**. Encadrement : **Hugo Rapior** (Nexialog Consulting) et
**Caroline Hillairet** (ENSAE Paris).

Ce dépôt publie un export du travail de recherche : manuscrits, code, sorties numériques et
contrôles. Mise à jour du **29 octobre 2026**. Les résultats restent conditionnels aux hypothèses
de fréquence, de sévérité et de propagation ; ils ne valident pas un capital individuel.

## Lire

| Document | Lien |
|---|---|
| **Mémoire Institut actuel — 152 pages** | [main_institut_150.pdf](memoire/main_institut_150.pdf) |
| Rapport ENSAE courant — 100 pages | [main_ensae.pdf](memoire/main_ensae.pdf) |
| Ancienne version Institut conservée — 195 pages | [main_institut.pdf](memoire/main_institut.pdf) |
| Soutenance ENSAE actuelle | [soutenance_ensae.pdf](recherche/slides/beamer/soutenance_ensae.pdf) |
| Préparation à la soutenance | [cours_memoire_ensae.pdf](recherche/vasicek_lab/notes/cours_memoire_ensae.pdf) |
| Sources Overleaf avec réglages Nexialog | [Mode d'emploi](memoire/overleaf/README.md) |
| Versions et statut des calculs | [Guide des versions](docs/VERSIONS.md) |
| Matériel complémentaire | [Zenodo](https://doi.org/10.5281/zenodo.22874524) |

Le nom `main_institut_150` conserve la cible initiale ; le PDF actuel compte 152 pages.
Les fichiers de la première publication restent accessibles à l'étiquette `v1.1.0`.

## Question, modèle et portée

Le travail examine le besoin de capital associé à des états de conformité DORA supposés.
Un modèle fréquence–sévérité et une cascade dirigée relient ces états à une distribution annuelle
de pertes. Sur le périmètre financier étudié, le scénario à quatre canaux passe de 6 049 à
20 188 M€, soit ×3,34. Cet écart est simulé ; il ne mesure pas un effet causal du règlement.

La direction de propagation n'est pas identifiée sur les sources disponibles. Pour une partie
symétrique posée et les autres paramètres fixés, les 1 024 sommets examinés donnent une plage
de 6 858 à 8 697 M€. Ce calcul aux sommets ne constitue pas une preuve d'extrema globaux ni
un intervalle couvrant toutes les incertitudes.

Les compléments d'octobre traitent la descente à l'entité : calibration sur les grandes
institutions, bootstrap par entreprise, comparaison VaR/ES et sensibilité à l'enregistrement
des incidents. La variante donne 24,1 M€ pour l'entité notionnelle, contre 169 M€ dans l'ancienne
lecture. La taille, le métier, les fenêtres d'observation et la sélection du registre restent
des limites explicites. Les scénarios de déclaration ne sont pas des taux estimés.

## Vérifier et compiler

```bash
git clone https://github.com/kelian19/memoire-scr-dora.git
cd memoire-scr-dora
python -m venv .venv
# Activer cet environnement, puis :
pip install -r requirements.txt
python projet.py structure
python outils/ci/harnais_strict.py
tectonic memoire/main_institut_150.tex
```

Le harnais compare les valeurs contrôlables du texte aux sorties enregistrées. Il ne rejoue pas
les simulations et ne valide pas les hypothèses scientifiques. Les scripts qui lisent les bases
brutes nécessitent un accès séparé à ces données. Les sorties Institut dans
`resultats/sorties_verif_inst/` remplacent les sorties historiques pour les seuls chapitres Institut.

L'exporteur Overleaf prépare les sources, figures et bibliographie. La variante Nexialog demande
le fichier `RequiredFiles/fond_latex_2.pdf` du projet du cabinet ; ce fond n'est pas fourni ici.
Les signatures ne sont pas distribuées sous forme d'images ; la compilation sait fonctionner sans elles.

## Organisation

```text
memoire/       manuscrits, sources LaTeX et bibliographies
recherche/     modèles, expériences, figures et soutenance
src/           modules Python
src/utils/     paramètres communs du modèle
resultats/     sorties vérifiées et compléments numériques
outils/ci/     contrôles et exporteur Overleaf
docs/          guides de cette publication
data/          documentation ; données brutes exclues
```

Les données sous licence, les notes internes et les identités des quatre entités SFCR sont
exclues ou masquées. Les valeurs vraies du questionnaire d'élicitation restent hors publication.
Ces restrictions sont conservées lors de chaque export.

Voir [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md) et
[CITATION.cff](CITATION.cff). Le code est sous licence MIT ; texte et figures sous CC BY 4.0,
sous réserve des droits des tiers, notamment sur les logos, illustrations externes et signatures.
