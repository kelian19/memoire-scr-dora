#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""109 : LA SURFACE CONJOINTE DU PLAFOND ET DE LA SATURATION.

POURQUOI CELLE-CI A LE DROIT D'ETRE EN TROIS DIMENSIONS, ET LA PLUPART NON. Une surface
tridimensionnelle cache les valeurs derriere la perspective : posee sur une grandeur qui ne
depend que d'un parametre, elle enleve de l'information au lieu d'en ajouter. Ici l'objet
mesure est une INTERACTION, c'est-a-dire litteralement une courbure au-dessus de deux axes, et
le resultat du script 86 est que cette courbure CHANGE DE SIGNE. Un tableau de vingt cases le
dit ; une surface le montre, et c'est la seule figure du dossier dont l'objet soit reellement
de dimension deux.

CE QUE LA FIGURE MONTRE, EN DEUX PANNEAUX.
  (a) le capital sur le rectangle des deux reserves, plafond de severite kappa en abscisse et
      exposant de saturation theta en profondeur. On y voit que la ligne kappa = 0,5 % est
      PLATE : a ce plafond, la sensibilite a theta est exactement nulle ;
  (b) le terme d'interaction, defini au script 86 comme
      I = V(kappa, theta) - V(kappa, 1) - V(inf, theta) + V(inf, 1). Il est POSITIF a
      theta < 1, NUL le long de theta = 1, NEGATIF a theta > 1. C'est le resultat publie : le
      plafond n'est pas un contrepoids qui s'oppose a la saturation, c'est un AMORTISSEUR qui
      reduit sa prise dans les deux sens.

CE QUE LE SCRIPT LIT, ET IL NE SIMULE RIEN. Les deux grilles viennent de la section 3 de
sorties_verif/86.txt, telles qu'elles y sont imprimees. Aucun appel au moteur, aucune graine :
le gel n'est pas touche et la figure ne peut pas diverger du tableau qu'elle illustre.

L'AXE DES PLAFONDS EST ORDINAL, PAS NUMERIQUE, et c'est declare plutot que masque : ses cinq
crans sont 0,2 %, 0,5 %, 1 %, 2 % et << aucun >>, ce dernier n'etant pas un nombre. Les espacer
regulierement est le seul choix honnete ; les espacer selon leur valeur exigerait de placer
l'infini.

TROIS CONTROLES, ET ARRET DUR SI L'UN CEDE.
  1. les deux grilles sont completes, cinq plafonds sur cinq exposants pour le capital et
     quatre sur cinq pour l'interaction, qui n'est pas definie au plafond infini ;
  2. le coin libre, plafond absent et theta = 1, redonne la VaR libre de 179,1 M ;
  3. le signe de l'interaction suit celui de (1 - theta) sur toutes les cases non nulles.

Sortie : figure S34_plafond_saturation_3d.png.
"""

import os
import re
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import style_nexialog as st                                       # noqa: E402

st.appliquer(taille=10)
mpl.rcParams["font.family"] = ["DejaVu Sans", "Segoe UI", "sans-serif"]

WID = 88
VAR_LIBRE = 179.1
KAPPA_NOMS = ["0,2 %", "0,5 %", "1 %", "2 %", "aucun"]


def titre(t):
    print()
    print("=" * WID)
    print(t)
    print("=" * WID)


# ---------------------------------------------------------------------------------------
# 1. Lecture de la section 3 de 86.txt
# ---------------------------------------------------------------------------------------

chemin = os.path.join(DEPOT, "sorties_verif", "86.txt")
with open(chemin, encoding="utf-8") as f:
    lignes = f.read().split("\n")

# Les deux grilles se suivent, chacune precedee de sa ligne d'en-tete << kappa \ theta >>.
grilles, thetas = [], None
for i, ligne in enumerate(lignes):
    if "kappa \\ theta" in ligne:
        thetas = [float(v) for v in re.findall(r"\d\.\d\d", ligne)]
        table = {}
        for suite in lignes[i + 1:]:
            m = re.match(r"\s+(0,2 %|0,5 %|1 %|2 %|aucun)\s+(.+?)\s*$", suite)
            if not m:
                break
            vals = re.findall(r"[+-]?\d+\.\d", m.group(2))
            if len(vals) != len(thetas):
                break
            table[m.group(1)] = [float(v) for v in vals]
        grilles.append(table)

titre("109 - LA SURFACE CONJOINTE DU PLAFOND ET DE LA SATURATION")
print(f"  lu dans {os.path.relpath(chemin, DEPOT)} : {len(grilles)} grille(s), "
      f"exposants {thetas}.")

arrets = []
if len(grilles) != 2:
    arrets.append(f"{len(grilles)} grille(s) lues au lieu de deux (capital, puis interaction)")
if thetas is None or len(thetas) != 5:
    arrets.append("les cinq exposants de saturation ne sont pas lisibles")

capital, inter = (grilles + [{}, {}])[:2]
if not arrets:
    if sorted(capital) != sorted(KAPPA_NOMS):
        arrets.append(f"la grille de capital porte {sorted(capital)} et non les cinq plafonds")
    if len(inter) != 4 or "aucun" in inter:
        arrets.append("la grille d'interaction doit porter quatre plafonds, sans le plafond absent")

if not arrets:
    libre = capital["aucun"][thetas.index(1.00)]
    if abs(libre - VAR_LIBRE) > 0.05:
        arrets.append(f"le coin libre vaut {libre} et non {VAR_LIBRE}")

    # le signe de l'interaction suit celui de (1 - theta)
    fautes = []
    for k, ligne in inter.items():
        for t, v in zip(thetas, ligne):
            if abs(v) > 1e-9 and np.sign(v) != np.sign(1.0 - t):
                fautes.append((k, t, v))
    if fautes:
        arrets.append(f"{len(fautes)} case(s) d'interaction ne suivent plus le signe de (1 - theta)")

print()
print("  Controles :")
print(f"    plafonds de la grille capital    : {len(capital)}   (attendu 5)")
print(f"    plafonds de la grille interaction: {len(inter)}   (attendu 4)")
if not arrets:
    print(f"    coin libre (aucun, theta = 1)    : {libre}   (attendu {VAR_LIBRE})")
    print(f"    cases a signe contraire a 1-theta: {len(fautes)}   (attendu 0)")

if arrets:
    print()
    print("  ARRET DUR. Le script refuse de tracer :")
    for a in arrets:
        print("    - " + a)
    sys.exit(1)
print("    => controles passes, le script trace.")

# ---------------------------------------------------------------------------------------
# 2. Les deux surfaces
# ---------------------------------------------------------------------------------------

Z_CAP = np.array([capital[k] for k in KAPPA_NOMS])                # (5 kappa, 5 theta)
Z_INT = np.array([inter[k] for k in KAPPA_NOMS[:4]])              # (4 kappa, 5 theta)

seq = LinearSegmentedColormap.from_list("nexialog_seq", list(reversed(st.SEQUENTIEL_6)))
div = LinearSegmentedColormap.from_list("nexialog_div", list(reversed(st.DIVERGENT)))

fig = plt.figure(figsize=(7.6, 3.9), dpi=200)

for k, (Z, noms, cmap, norm, libelle, lettre) in enumerate((
        (Z_CAP, KAPPA_NOMS, seq, Normalize(Z_CAP.min(), Z_CAP.max()),
         "capital (M€)", "(a)  le capital sur les deux réserves"),
        (Z_INT, KAPPA_NOMS[:4], div,
         TwoSlopeNorm(vmin=-np.abs(Z_INT).max(), vcenter=0.0, vmax=np.abs(Z_INT).max()),
         "interaction (M€)", "(b)  le terme d'interaction change de signe"))):

    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.set_facecolor(st.FOND)
    X, Y = np.meshgrid(np.arange(len(thetas)), np.arange(len(noms)))
    # PASSER edgecolors A plot_surface EN MEME TEMPS QUE facecolors LEVE UNE ERREUR :
    # matplotlib construit lui-meme les aretes depuis les faces et les deux arguments
    # se percutent. On pose donc la couleur d'arete APRES coup, sur la collection rendue.
    surf = ax.plot_surface(X, Y, Z, facecolors=cmap(norm(Z)), rstride=1, cstride=1,
                           linewidth=0.35, antialiased=True, shade=False)
    surf.set_edgecolor(st.FOND_PANNEAU)

    ax.set_xticks(range(len(thetas)))
    ax.set_xticklabels([f"{t:.2f}".replace(".", ",") for t in thetas], fontsize=7)
    ax.set_yticks(range(len(noms)))
    ax.set_yticklabels(noms, fontsize=7)
    ax.set_xlabel("saturation $\\theta$", color=st.ENCRE_2, fontsize=8.5, labelpad=-2)
    ax.set_ylabel("plafond $\\kappa$", color=st.ENCRE_2, fontsize=8.5, labelpad=-1)
    ax.set_zlabel(libelle, color=st.ENCRE_2, fontsize=8.5, labelpad=-4)
    ax.zaxis.set_tick_params(labelsize=7, pad=-1)
    ax.tick_params(axis="both", pad=-2)
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(st.FOND)
        pane.set_edgecolor(st.GRILLE)
    ax.view_init(elev=22, azim=-58)
    ax.set_title(lettre, fontsize=9, color=st.ENCRE, pad=10)

# MARGES : a top=0.98 les deux titres de panneau etaient COUPES par le bord, et a
# right=0.99 l'etiquette de l'axe du panneau droit sortait de la figure. Les deux se
# voient au rendu et nulle part ailleurs.
fig.subplots_adjust(left=0.01, right=0.94, top=0.90, bottom=0.04, wspace=0.10)
sortie = os.path.join(HERE, "figures", "S34_plafond_saturation_3d.png")
fig.savefig(sortie, dpi=200)
print()
print("  figure ecrite :", os.path.relpath(sortie, DEPOT))

# ---------------------------------------------------------------------------------------
# 3. Grandeurs citees
# ---------------------------------------------------------------------------------------

i_un = thetas.index(1.00)
titre("1. GRANDEURS CITEES PAR LE MEMOIRE (sans separateur de milliers)")
print(f"  VaR libre, plafond absent et theta = 1        : {VAR_LIBRE}")
print(f"  capital minimal du rectangle                  : {Z_CAP.min():.1f}")
print(f"  capital maximal du rectangle                  : {Z_CAP.max():.1f}")
print(f"  interaction maximale en valeur absolue        : {np.abs(Z_INT).max():.1f}")
plat = capital["0,5 %"]
print(f"  capital a plafond 0,5 pourcent, les cinq theta : "
      f"{' '.join(f'{v:.1f}' for v in plat)}")
print(f"  etendue de cette ligne plate                  : {max(plat) - min(plat):.1f}")
print(f"  interaction le long de theta = 1              : "
      f"{' '.join(f'{inter[k][i_un]:.1f}' for k in KAPPA_NOMS[:4])}")

print()
print("=" * WID)
print("FIN 109")
print("=" * WID)
