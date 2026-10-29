# Visuels du gabarit Nexialog

Ce dossier porte les **visuels de charte** du support de soutenance, et rien d'autre. Aucune
figure de résultat n'y vit : celles-ci restent dans `exploratory/vasicek_lab/figures/` et sont
produites par les scripts du dépôt.

## Provenance

Les six fichiers sont extraits du gabarit de présentation Nexialog Consulting, transmis par
Kélian le 11 septembre 2026 sous la forme d'un PDF de 53 pages (`Rapport ENSAE.pdf`, hors dépôt).
Extraction par `pymupdf`, sans retouche : ni recadrage, ni recolorisation, ni redimensionnement.
Le canal alpha des logos est reconstitué depuis le masque doux du PDF, donc la transparence est
celle du fichier d'origine.

| Fichier | Taille | Ce que c'est |
|---|---|---|
| `nexialog.png` | 416 x 151, alpha | logo Nexialog Consulting, version couleur |
| `nexialog_blanc.png` | 315 x 114, alpha | le même en blanc, pour les fonds sombres |
| `institut_actuaires.png` | 330 x 140 | logo de l'Institut des Actuaires |
| `ensae.png` | 250 x 91 | logo ENSAE / IP Paris, recopié de `memoire_cascade/logos/` |
| `think_smart.png` | 821 x 164, alpha | signature « THINK SMART ACT DIFFERENT » |
| `photo_couverture.jpg` | 2249 x 1499 | photographie de couverture du gabarit |
| `photo_section.jpg` | 1811 x 1392 | photographie des intercalaires |

## Un gain à signaler

`CLAUDE.md` note depuis le 9 septembre que le logo Nexialog de la page de garde du mémoire n'a
pas été posé, faute d'un PNG à fond transparent : le seul fichier disponible était un JPEG sur
fond noir extrait du rapport LUCY. **`nexialog.png` lève cette réserve.** La ligne à décommenter
est déjà dans `memoire_cascade/page_de_garde.tex`. Le geste n'est pas fait ici : il touche le
mémoire, et il appartient à Kélian de décider si le logo d'entreprise figure sur la page de garde
de l'Institut.

## Charte relevée sur le gabarit, et non devinée

Les valeurs ci-dessous sont **mesurées** dans le PDF, police par police et aplat par aplat. Elles
sont recopiées en tête de `build_soutenance_ppt.py`, qui en est le seul consommateur.

| Rôle | Valeur |
|---|---|
| Titre de diapositive | Georgia Bold 26 pt, `#223E55` |
| Sous-titre | Georgia Bold 18 pt, `#B10031` |
| Corps | Segoe UI 14 pt, `#122738` |
| Corps secondaire et légendes | Segoe UI 14 pt, `#595959`, italique pour les légendes |
| Aplat sombre | `#223E55`, variante profonde `#192E3F`, variante claire `#435B6E` |
| Aplat clair | `#F2F2F2`, filets `#D9D9D9` |
| Accent | `#B10031` |
| Pied de page | 8 pt, `#A5A5A5` |

Ces couleurs ne sont **pas** celles de `style_nexialog.py`, qui reste la source unique des
couleurs des **figures**. Les deux familles se ressemblent sans être identiques, et il ne faut pas
les confondre : une figure du mémoire garde ses couleurs de figure, une diapositive porte celles
du gabarit de présentation.

## Ce qu'il ne faut pas en faire

Ces visuels appartiennent à Nexialog Consulting. Ils servent le support de soutenance et le
support de soutenance seul. Ne pas les verser dans le mémoire, qui est destiné à être mis en ligne
par l'Institut, sans l'accord de l'entreprise.
