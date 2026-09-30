#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1 : Le reseau dirige W et un mini-exemple de cascade.

Panneau A : la structure DIRIGEE de W sur les 5 piliers (derivee de TRANS).
            Les fleches epaisses partent de P1 (super-emetteur) ; peu reviennent
            vers lui -> asymetrie visible.
Panneau B : le sous-reseau {P1,P2,P4} de l'exemple chiffre de la note, avec ses
            poids ILLUSTRATIFS (arrondis pour le calcul a la main, ce ne sont pas
            les entrees calibrees de W) et la propagation d'un choc +2 sur P1.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

# ---------------------------------------------------------------- style
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2, MUTED, GRID = "#1b1e30", "#223e55", "#595959", "#f2f2f2"
ACCENT = "#a6002e"
PCOL = {1: "#204993", 2: "#4c79c7", 3: "#6491e1", 4: "#a6002e", 5: "#7d7d7d"}

TRANS = {
    1: {2: 0.80, 3: 0.70, 4: 0.70, 5: 0.40},
    2: {1: 0.20, 3: 0.40, 4: 0.30, 5: 0.60},
    3: {1: 0.30, 2: 0.50, 4: 0.30, 5: 0.30},
    4: {1: 0.20, 2: 0.80, 3: 0.30, 5: 0.40},
    5: {1: 0.10, 2: 0.20, 3: 0.20, 4: 0.20},
}


def node(ax, xy, label, color, r=0.15, big=False):
    rr = r * (1.25 if big else 1.0)
    ax.add_patch(Circle(xy, rr, facecolor=color, edgecolor="#fcfcfb",
                        lw=2, zorder=4))
    ax.text(*xy, label, ha="center", va="center", color="#fff",
            fontsize=12 if big else 11, fontweight="bold", zorder=5)


def arrow(ax, p0, p1, w, color, rad=0.16, r=0.16):
    # raccourcit aux bords des cercles
    d = np.array(p1) - np.array(p0)
    L = np.hypot(*d)
    u = d / L
    a = np.array(p0) + u * r
    b = np.array(p1) - u * r
    ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}",
                 arrowstyle="-|>", mutation_scale=13, lw=w, color=color,
                 alpha=0.85, zorder=3))


# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime
# cette figure sur 7,28 pouces : tracee plus large, elle subissait une
# reduction qui faisait tomber ses etiquettes sous 6 points. Le rapport
# largeur sur hauteur est conserve, donc la figure occupe la meme place
# sur la page ; c'est son texte qui y prend plus de place. La hauteur perd
# la bande du titre general, retire plus bas.
fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.9, 3.45),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# ---------------------------------------------------------------- panneau A : W sur 5 piliers
ang = {1: 90, 2: 90 - 72, 3: 90 - 144, 4: 90 + 144, 5: 90 + 72}
pos = {p: (np.cos(np.radians(a)), np.sin(np.radians(a))) for p, a in ang.items()}
for i in TRANS:
    for j, v in TRANS[i].items():
        if v >= 0.4:                      # on ne trace que les liens marques
            arrow(axA, pos[i], pos[j], w=0.6 + 4.2 * (v - 0.4), color=PCOL[i])
for p, xy in pos.items():
    node(axA, xy, f"P{p}", PCOL[p], big=(p == 1))
axA.text(*pos[1], "", zorder=6)
axA.set_xlim(-1.5, 1.5); axA.set_ylim(-1.5, 1.6)
axA.set_aspect("equal"); axA.axis("off")
axA.set_title("(a)  Le réseau dirigé $W$ sur les cinq piliers",
              fontsize=11, color=INK, pad=6)
axA.text(0, -1.5, "flèche i→j = i entraîne j ;  épaisseur ∝ force.  "
         "P1 émet fort, reçoit peu  →  asymétrie.",
         ha="center", va="top", fontsize=9, color=INK2)

# ---------------------------------------------------------------- panneau B : sous-reseau + calcul
pB = {1: (0.0, 0.0), 2: (2.0, -0.7), 4: (2.0, 0.9)}
W = {(2, 1): 0.5, (2, 4): 0.4, (4, 1): 0.3}   # W_jk : j recoit de k
for (j, k), w in W.items():
    arrow(axB, pB[k], pB[j], w=0.8 + 4.0 * w, color=PCOL[k], rad=0.10, r=0.22)
    mid = (np.array(pB[k]) + np.array(pB[j])) / 2
    # POIDS POSE A COTE DE LA FLECHE, JAMAIS DESSUS. Sur l'arc vertical de P4
    # vers P2 le poids tombait sur le trait et sous la valeur de P4 : il part
    # maintenant a gauche de l'arc.
    vertical = abs(pB[k][0] - pB[j][0]) < 1e-9
    if vertical:
        axB.text(mid[0] - 0.20, mid[1], f"{w:.1f}".replace(".", ","),
                 ha="right", va="center", fontsize=9, color=INK,
                 bbox=dict(boxstyle="round,pad=0.1", fc="#fcfcfb", ec="none"))
    else:
        axB.text(mid[0], mid[1] + 0.12, f"{w:.1f}".replace(".", ","),
                 ha="center", va="bottom", fontsize=9, color=INK,
                 bbox=dict(boxstyle="round,pad=0.1", fc="#fcfcfb", ec="none"))
Xval = {1: 2.0, 2: 1.24, 4: 0.6}
for p, xy in pB.items():
    node(axB, xy, f"P{p}", PCOL[p], r=0.22, big=(p == 1))
    # la valeur de P4 passe AU-DESSUS du cercle : au-dessous, elle entrait dans
    # l'arc qui descend vers P2.
    haut = (p == 4)
    axB.text(xy[0], xy[1] + (0.36 if haut else -0.36),
             f"$X = {Xval[p]:.2f}$".replace(".", "{,}"),
             ha="center", va="bottom" if haut else "top",
             fontsize=9.5, color=ACCENT if Xval[p] >= 1 else MUTED,
             fontweight="bold")
# « choc +2 » renvoye a gauche de sa fleche, qu'il recouvrait.
axB.text(-0.14, 0.52, "choc $+2$", ha="right", va="center",
         fontsize=9.5, color=INK, style="italic")
axB.annotate("", xy=(pB[1][0], 0.30), xytext=(pB[1][0], 0.75),
             arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
axB.set_xlim(-0.9, 2.9); axB.set_ylim(-1.6, 1.7)
axB.set_aspect("equal"); axB.axis("off")
axB.set_title("(b)  Cascade d'un choc $+2$ sur P1 (exemple chiffré)",
              fontsize=11, color=INK, pad=6)
axB.text(1.0, -1.5, "seuil $K = 1$  →  P1 et P2 déclenchent, pas P4.\n"
         "Même choc placé sur P2  →  reste local (un seul incident).",
         ha="center", va="top", fontsize=9, color=INK2)

# PAS DE TITRE GENERAL : la legende LaTeX porte le titre.
fig.tight_layout()

outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "H1_reseau_W.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("figure ecrite :", path)
