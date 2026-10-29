# La cascade en vidéo : le même incident, deux entités

`output/cascade_deux_etats.mp4` — 36 s, 1920x1080, 30 images par seconde, H.264, 2,7 Mo.
Sans voix off : tout le sens est porté par la scène et trois cartons de texte.

## Ce que la vidéo montre, et pourquoi elle apporte quelque chose

Le mémoire publie que **62,31 %** des sinistres touchent plus d'un pilier à l'état non conforme
contre **31,15 %** à l'état conforme. Dans un tableau, c'est une affirmation. À l'écran, deux
entités reçoivent **le même incident**, tiré avec **les mêmes aléas**, et l'une l'éteint quand
l'autre le propage. Le spectateur voit le mécanisme au lieu de lire son résultat, et c'est la
seule chose qu'une figure fixe ne sait pas faire.

## Ce qu'elle anime, et ce qu'elle n'anime pas

Elle anime **la marche auto-évitante**, qui est le moteur publié, et non la cascade par
branchement de la définition 6.1. Le script 98 a établi que les deux diffèrent. Depuis le pilier
courant `j`, la cascade continue avec probabilité `e_j = g * s_j / max_s`, le successeur est tiré
proportionnellement à `TRANS[j]`, et **un pilier déjà touché éteint la cascade**. Le script
importe `scr_engine` au lieu de réécrire ce noyau.

**Un seul canal bouge à l'écran.** La fréquence est tenue fixe, sans quoi les deux panneaux ne
recevraient pas les mêmes incidents et la mention « mêmes aléas » serait fausse. Le carton final
le dit : l'écart de capital publié, 6 049 vers 20 188 M€, met en jeu les **quatre** canaux, la
propagation seule ne pesant que 1 633 M€. Ne pas retirer cette réserve.

## Ce que le script ne fait pas

- **il ne recalcule aucun capital** : les deux capitaux du carton final sont lus dans
  `sorties_verif/68.txt` ;
- **il ne retape aucun seuil** : la matrice, les taux d'amorce et les gains viennent du moteur ;
- **il ne définit aucune couleur** : les deux teintes d'accent viennent de `style_nexialog.py`,
  source unique des couleurs des figures du mémoire.

## Les deux contrôles, avec arrêt dur

1. la **loi exacte** du moteur, agrégée sur les amorces, doit redonner les valeurs publiées par
   le script 74 : 31,15 % et 62,31 % de sinistres multi-piliers, 1,380 et 1,931 piliers en
   moyenne. Tolérance dans `config.yaml`. Si le moteur a dérivé, la vidéo ne se rend pas ;
2. les deux capitaux du carton final doivent redonner 6 049 et 20 188.

Au dernier rendu, les écarts valaient au plus `1,4e-4`.

## Relancer

```powershell
cd exploratory\video_cascade
..\..\.venv\Scripts\python.exe render_cascade.py --apercu   # aperçu 480p, 10 s
..\..\.venv\Scripts\python.exe render_cascade.py            # version finale, ~3,5 min
```

L'aperçu garde le titre, quatre incidents et le carton final : il valide **le rendu**, pas le
montage complet. Le rendu final prend environ trois minutes et demie.

## Changer quelque chose

Tout le réglage est dans `config.yaml`, et **rien d'autre ne doit être édité** :

| Ce que tu veux | Où |
|---|---|
| format vertical pour un réseau social | `video.largeur: 1080`, `video.hauteur: 1920` |
| fond sombre plutôt que le fond du mémoire | `theme.fond: sombre` (les teintes d'accent ne bougent pas) |
| vidéo plus courte ou plus longue | `sequences.*`, en secondes |
| plus ou moins d'incidents | `incidents.nombre` |
| cascade plus lente à l'écran | `incidents.pas_par_seconde` |
| une autre suite d'incidents | `incidents.graine` |

## Dépendances, et les deux pièges rencontrés

`imageio-ffmpeg` est requis et a été installé dans le `.venv` : matplotlib cherche un
**exécutable** ffmpeg, pas le paquet, donc le script lui désigne le binaire embarqué par
`get_ffmpeg_exe()`. Sans cela les seuls writers disponibles sont `pillow` et `html`.

**H.264 refuse une dimension impaire.** matplotlib rend 853 pixels pour 854 demandés, parce que
854/100 n'est pas exact en binaire, et le premier rendu s'est arrêté sur
`width not divisible by 2`. Le writer porte donc un filtre `scale=trunc(iw/2)*2:trunc(ih/2)*2`,
qui force les deux dimensions au pair quelle que soit la résolution demandée. Ne pas le retirer.

**Manim n'est pas utilisable sur ce poste**, et ce n'est pas un choix de confort : ses
dépendances `moderngl` et `glcontext` échouent à construire leur wheel, faute d'outils de
compilation C++, et ses formules exigeraient `latex` et `dvisvgm`, absents puisque le projet
compile avec tectonic, un binaire unique qui ne les fournit pas. Les formules de la vidéo
passent donc par le `mathtext` de matplotlib, qui ne demande aucune installation TeX.

## Où la vidéo peut servir

Pas dans le PDF déposé : une vidéo embarquée dans un PDF ne se joue que dans Adobe Acrobat sur
poste, ni dans un navigateur ni dans la bibliothèque en ligne de l'Institut. Elle sert au
**support de soutenance**, et en ligne avec un lien depuis le mémoire par la macro `\matcomp`.
