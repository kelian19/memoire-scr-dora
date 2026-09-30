#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""108 : LE TREILLIS DES QUATRE CANAUX, EN PROJECTION TRIDIMENSIONNELLE.

POURQUOI CETTE FIGURE, ET POURQUOI ELLE A LE DROIT D'ETRE EN TROIS DIMENSIONS. Le memoire
publie seize configurations de canaux dans un tableau. Un tableau de seize lignes se lit, mais il
cache ce qui fait la structure : ces seize configurations sont les SOMMETS D'UN HYPERCUBE A
QUATRE DIMENSIONS, un par canal relache. L'objet est donc reellement de dimension quatre, et le
projeter en trois dimensions n'est pas un ornement, c'est la seule facon de le montrer entier.
Une surface tridimensionnelle posee sur un objet unidimensionnel serait de la decoration ; ici
c'est l'inverse, on REDUIT la dimension de l'objet pour le rendre visible.

CE QUE LA FIGURE MONTRE, ET CHACUN DES TROIS POINTS EST UN RESULTAT DEJA PUBLIE.
  - la HAUTEUR est le capital. Les deux sommets opposes sont les deux etats publies, 6 049 M a
    l'etat conforme et 20 188 M a l'etat non conforme, et tout sommet intermediaire est une
    remediation partielle ;
  - toute arete MONTE. C'est le theoreme du coin superieur du script 88, relacher un canal de
    plus ne fait jamais baisser le capital, et il se lit ici d'un coup d'oeil au lieu de se lire
    sur soixante-cinq paires emboitees ;
  - la COULEUR est l'ecart a l'additivite, c'est-a-dire le capital observe moins ce que
    predirait la somme des quatre effets isoles. C'est la decomposition publiee : le sommet
    complet porte +5 001 M, les 35 % d'interaction du script 43, et le sommet prop+accum porte
    -330 M, la seule paire NEGATIVE du dossier, les deux canaux de co-occurrence etant
    substituts et non complements.

CE QUE LE SCRIPT LIT, ET IL NE CALCULE AUCUN CAPITAL. Les seize valeurs viennent de la section 1
de sorties_verif/68.txt et les quatre effets isoles de sa section 2. Aucune simulation, aucune
graine, aucun appel au moteur : le gel n'est pas touche et la figure ne peut pas diverger du
tableau qu'elle illustre. C'est le patron du script 101.

TROIS CONTROLES, ET ARRET DUR SI L'UN CEDE.
  1. les deux coins redonnent 6 049 et 20 188 ;
  2. la monotonie tient sur les TRENTE-DEUX aretes du treillis, ce qui reverifie le script 88
     sur les nombres publies et par un chemin independant ;
  3. la projection est INJECTIVE, sans quoi deux configurations se superposeraient et la figure
     mentirait sans le dire.

Sortie : figure S33_treillis_canaux_3d.png.
"""

import itertools
import os
import re
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import style_nexialog as st                                       # noqa: E402

st.appliquer(taille=10)
mpl.rcParams["font.family"] = ["DejaVu Sans", "Segoe UI", "sans-serif"]

CANAUX = ["freq", "det", "prop", "accum"]
NOMS = {"freq": "fréquence", "det": "détection", "prop": "propagation",
        "accum": "accumulation"}

WID = 88


def titre(t):
    print()
    print("=" * WID)
    print(t)
    print("=" * WID)


# ---------------------------------------------------------------------------------------
# 1. Lecture des sorties versionnees
# ---------------------------------------------------------------------------------------

chemin68 = os.path.join(DEPOT, "sorties_verif", "68.txt")
capital, isole = {}, {}
with open(chemin68, encoding="utf-8") as f:
    for ligne in f:
        m = re.match(r"\s+(conforme|[a-z+]+)\s+(\d+)\s+(-?\d+)\s+(\d)\s*$", ligne)
        if m and (m.group(1) == "conforme" or
                  all(p in CANAUX for p in m.group(1).split("+"))):
            cfg = frozenset() if m.group(1) == "conforme" else frozenset(m.group(1).split("+"))
            capital[cfg] = int(m.group(2))
        m2 = re.match(r"\s+(frequence|detection|propagation|accumulation)\s+(\d+) \+-\d+", ligne)
        if m2:
            cle = {"frequence": "freq", "detection": "det",
                   "propagation": "prop", "accumulation": "accum"}[m2.group(1)]
            isole[cle] = int(m2.group(2))

titre("108 - LE TREILLIS DES QUATRE CANAUX EN PROJECTION TRIDIMENSIONNELLE")
print(f"  lu dans {os.path.relpath(chemin68, DEPOT)} : {len(capital)} configurations, "
      f"{len(isole)} effets isoles.")

# ---------------------------------------------------------------------------------------
# 2. Controles d'identite, et ARRET DUR
# ---------------------------------------------------------------------------------------

vide, plein = frozenset(), frozenset(CANAUX)
arrets = []

if len(capital) != 16:
    arrets.append(f"le treillis compte {len(capital)} configurations et non seize")
if len(isole) != 4:
    arrets.append(f"la section 2 donne {len(isole)} effets isoles et non quatre")

if not arrets:
    if capital[vide] != 6049:
        arrets.append(f"l'etat conforme vaut {capital[vide]} et non 6049")
    if capital[plein] != 20188:
        arrets.append(f"l'etat non conforme vaut {capital[plein]} et non 20188")

# monotonie sur les trente-deux aretes : relacher un canal de plus ne baisse jamais le capital
aretes, violations = [], []
if not arrets:
    for cfg in capital:
        for c in CANAUX:
            if c not in cfg:
                sup = cfg | {c}
                aretes.append((cfg, sup))
                if capital[sup] < capital[cfg]:
                    violations.append((cfg, sup))
    if len(aretes) != 32:
        arrets.append(f"le treillis porte {len(aretes)} aretes et non trente-deux")
    if violations:
        arrets.append(f"{len(violations)} arete(s) descendent : la monotonie du script 88 a cede")

# ---------------------------------------------------------------------------------------
# 3. La disposition, et pourquoi ce n'est pas une projection d'hypercube
# ---------------------------------------------------------------------------------------
# PREMIERE VERSION ABANDONNEE, ET LE DIRE EVITE QU'ON LA REFASSE. L'hypercube a quatre
# dimensions se projette classiquement dans le plan par quatre directions non commensurables,
# et c'est l'objet mathematiquement exact. RENDU A LA TAILLE D'IMPRESSION DU MEMOIRE, environ
# quatorze centimetres, c'est illisible : seize sommets, trente-deux aretes et six etiquettes
# se recouvrent, et l'on ne distingue plus quel point porte quel etat. Regarde, puis jete.
#
# LA DISPOSITION RETENUE RANGE PAR NOMBRE DE CANAUX RELACHES. L'axe x est ce nombre, de zero
# a quatre, l'axe z le capital, et l'axe y separe les configurations d'un meme rang, ordonnees
# par capital croissant. Le y ne porte donc pas de grandeur, il porte une IDENTITE : sans lui
# les six configurations de rang deux se superposeraient et les aretes deviendraient un
# enchevetrement. Les aretes ne relient plus que des rangs consecutifs, ce qui les rend suivables.
rangs = {}
for cfg in capital:
    rangs.setdefault(len(cfg), []).append(cfg)

pos = {}
for r, liste in rangs.items():
    liste.sort(key=lambda c: capital[c])
    n = len(liste)
    ys = [0.0] if n == 1 else list(np.linspace(-1.0, 1.0, n))
    for cfg, y in zip(liste, ys):
        pos[cfg] = (float(r), float(y))

if not arrets:
    dmin = min(np.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1])
               for a, b in itertools.combinations(sorted(capital, key=sorted), 2))
    if dmin < 1e-9:
        arrets.append("deux configurations occupent la meme position : la disposition les confond")

print()
print("  Controles :")
if capital:
    print(f"    coin conforme                    : {capital.get(vide, 'absent')}   (attendu 6049)")
    print(f"    coin non conforme                : {capital.get(plein, 'absent')}   (attendu 20188)")
print(f"    aretes du treillis               : {len(aretes)}   (attendu 32)")
print(f"    aretes descendantes              : {len(violations)}   (attendu 0)")
if not arrets:
    print(f"    ecart minimal de disposition     : {dmin:.3f}   (doit etre > 0)")

if arrets:
    print()
    print("  ARRET DUR. Le script refuse de tracer :")
    for a in arrets:
        print("    - " + a)
    sys.exit(1)
print("    => controles passes, le script trace.")

# ---------------------------------------------------------------------------------------
# 4. L'ecart a l'additivite, recalcule depuis les valeurs publiees
# ---------------------------------------------------------------------------------------

base = capital[vide]
ecart = {cfg: capital[cfg] - (base + sum(isole[c] for c in cfg)) for cfg in capital}

titre("1. ECART A L'ADDITIVITE, PAR CONFIGURATION")
print("  Ecart = capital observe - (conforme + somme des effets isoles des canaux relaches).")
print()
print("    configuration              capital   additif    ecart")
for cfg in sorted(capital, key=lambda s: (len(s), sorted(s))):
    nom = "conforme" if not cfg else "+".join(c for c in CANAUX if c in cfg)
    add = base + sum(isole[c] for c in cfg)
    print(f"    {nom:24s}  {capital[cfg]:7d}  {add:8d}  {ecart[cfg]:+7d}")

print()
print(f"  Sommet complet : {ecart[plein]:+d} M, l'interaction des quatre canaux.")
pa = frozenset({"prop", "accum"})
print(f"  Sommet prop+accum : {ecart[pa]:+d} M, la seule paire negative du dossier.")
print()
print("  A NE PAS PUBLIER TEL QUEL, ET C'EST LE SEUL PIEGE DE CE SCRIPT. Le sommet complet")
print("  ressort ici a +5002 quand le memoire publie +5001, et l'ecart d'un million est un")
print("  ARRONDI, pas un desaccord : les quatre effets isoles sont lus arrondis au million dans")
print("  68.txt, et leur somme herite de quatre arrondis. La valeur qui fait foi reste celle du")
print("  script 68, soit 5001, et c'est elle que le memoire cite. Ne pas << corriger >> le")
print("  memoire sur la foi de cette figure.")
print()
print("  Les croises de paires se retrouvent au meme arrondi pres, ce qui est le controle qui")
print("  compte : le script 68 publie freq x det 1739, freq x accum 1666, freq x prop 1182,")
print("  det x accum 573, det x prop 515 et prop x accum -330.")

# ---------------------------------------------------------------------------------------
# 5. La figure
# ---------------------------------------------------------------------------------------

# DIVERGENT de la charte, retourne pour que le pole ROUGE porte l'ecart POSITIF, qui est
# l'alerte (la cascade coute plus que la somme de ses canaux), et le bleu l'ecart negatif.
cmap = LinearSegmentedColormap.from_list("nexialog_div", list(reversed(st.DIVERGENT)))
amp = max(abs(v) for v in ecart.values())
norm = TwoSlopeNorm(vmin=-amp, vcenter=0.0, vmax=amp)

fig = plt.figure(figsize=(7.4, 4.0), dpi=200)
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor(st.FOND)
fig.patch.set_facecolor(st.FOND)

# Aretes en gris RELEVE : a 0,9 point et en gris de grille pur, la structure du treillis ne
# se voyait plus a l'impression, et une figure ou l'on ne distingue pas les aretes ne montre
# plus l'objet qu'elle est censee montrer.
for a, b in aretes:
    xa, ya = pos[a]
    xb, yb = pos[b]
    ax.plot([xa, xb], [ya, yb], [capital[a], capital[b]],
            color="#C4C4C4", lw=0.9, zorder=1)

# Les quatre aretes issues de l'etat conforme sont tracees en plein : c'est la lecture
# « effet isole » du memoire, et elle se lit ainsi sans legende supplementaire.
for c in CANAUX:
    x0, y0 = pos[vide]
    x1, y1 = pos[frozenset({c})]
    ax.plot([x0, x1], [y0, y1], [capital[vide], capital[frozenset({c})]],
            color=st.ENCRE_2, lw=1.7, zorder=2)

for cfg in sorted(capital, key=lambda s: capital[s]):
    x, y = pos[cfg]
    ax.scatter([x], [y], [capital[cfg]], s=88, c=[cmap(norm(ecart[cfg]))],
               edgecolors=st.ENCRE_2, linewidths=0.6, depthshade=False, zorder=3)

ax.set_xlabel("canaux relâchés", color=st.ENCRE_2, labelpad=4)
# LABELPAD FAIBLE, ET C'EST LE CONTRAIRE DE L'INTUITION. L'axe du capital se dessine a droite
# a cet azimut ; un labelpad genereux pousse son titre VERS la barre de couleur et l'y fait
# entrer. On le colle donc a ses graduations et l'on ecarte la barre.
ax.set_zlabel("capital (M€)", color=st.ENCRE_2, labelpad=1)
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xlim(-0.35, 4.35)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(4500, 21800)
ax.set_yticks([])
ax.grid(False)
ax.xaxis.pane.set_facecolor(st.FOND); ax.yaxis.pane.set_facecolor(st.FOND)
ax.zaxis.pane.set_facecolor(st.FOND)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_edgecolor(st.GRILLE)
ax.view_init(elev=19, azim=-72)
ax.xaxis.set_tick_params(labelsize=8, pad=-1)
ax.zaxis.set_tick_params(labelsize=8, pad=2)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.07, aspect=16, location="right")
cb.set_label("écart à l'additivité (M€)", color=st.ENCRE_2, fontsize=8.5)
cb.ax.tick_params(labelsize=8, colors=st.ENCRE_3)
cb.outline.set_edgecolor(st.GRILLE)

fig.subplots_adjust(left=0.00, right=0.84, top=1.00, bottom=0.02)

# ETIQUETTES ANCREES EN COORDONNEES ECRAN, AVEC LIGNE DE RAPPEL. Posees en coordonnees 3D
# elles derivent a la projection : un nom de canal tombait sur un sommet voisin. proj_transform
# donne la position projetee reelle, et l'annotation part de la avec son trait de rappel.
from mpl_toolkits.mplot3d import proj3d                           # noqa: E402

fig.canvas.draw()


def etiquette(cfg, texte, decalage, taille, gras, couleur):
    x, y = pos[cfg]
    x2, y2, _ = proj3d.proj_transform(x, y, capital[cfg], ax.get_proj())
    ax.annotate(texte, xy=(x2, y2), xytext=decalage, textcoords="offset points",
                fontsize=taille, color=couleur, ha="center", va="center",
                fontweight=("bold" if gras else "normal"),
                style=("normal" if gras else "italic"),
                arrowprops=dict(arrowstyle="-", color=st.ENCRE_3, lw=0.7,
                                shrinkA=2, shrinkB=7), zorder=7)


DECAL = {"freq": (26, 28), "det": (16, -32), "prop": (-42, 8), "accum": (52, -28)}
for c in CANAUX:
    etiquette(frozenset({c}), NOMS[c], DECAL[c], 8.5, False, st.ENCRE_2)

etiquette(vide, "état conforme\n6 049 M€", (-24, 32), 9, True, st.ENCRE)
etiquette(plein, "état non conforme\n20 188 M€", (-16, 30), 9, True, st.ENCRE)
sortie = os.path.join(HERE, "figures", "S33_treillis_canaux_3d.png")
fig.savefig(sortie, dpi=200)
print()
print("  figure ecrite :", os.path.relpath(sortie, DEPOT))

# ---------------------------------------------------------------------------------------
# 6. Grandeurs citees
# ---------------------------------------------------------------------------------------

titre("2. GRANDEURS CITEES PAR LE MEMOIRE (sans separateur de milliers)")
print(f"  configurations du treillis                  : {len(capital)}")
print(f"  aretes du treillis                          : {len(aretes)}")
print(f"  aretes descendantes                         : {len(violations)}")
print(f"  capital a l'etat conforme                   : {capital[vide]}")
print(f"  capital a l'etat non conforme               : {capital[plein]}")
print(f"  ecart a l'additivite au sommet complet      : {ecart[plein]}")
print(f"  ecart a l'additivite sur prop+accum         : {ecart[pa]}")
print(f"  amplitude maximale de l'ecart a l'additivite : {amp}")

print()
print("=" * WID)
print("FIN 108")
print("=" * WID)
