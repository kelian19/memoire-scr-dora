#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F12 : L'arbre récursif des ordres.

Illustre la formalisation récursive : la probabilité d'un ordre de n piliers se
CONSTRUIT récursivement à partir du préfixe de (n-1) piliers, en multipliant par
la transition ajoutée.  On le montre sur l'exemple à 3 piliers (6 feuilles), avec
les VRAIES valeurs ROOT / TRANS du modèle.  Le calcul reste identique à
build_cascade_workbook.py : l'arbre n'est qu'une relecture, pas un nouveau modèle.

    p(a→b→c) = ROOT[a] · TRANS[a→b] · TRANS[b→c]
             = p(a→b)  · TRANS[b→c]          <-- récurrence : enfant = parent × lien
"""

from itertools import permutations
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# --- ROOT / TRANS : source unique (cascade_model.py) ------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cascade_model import ROOT, TRANS


def proba_raw(order):
    prop = ROOT[order[0]]
    for a, b in zip(order, order[1:]):
        prop *= TRANS[a][b]
    return prop


def proba_score(order):
    return max(1, min(10, round(10 * proba_raw(order))))


# ---------------------------------------------------------------- style (miroir des autres figures)
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID = "#e1e0d9"
ACCENT = "#eb6834"
# couleur fixe par pilier (P1..P5), cohérente avec les autres figures
PCOL = {1: "#184f95", 2: "#3987e5", 3: "#86b6ef", 4: ACCENT, 5: "#a9a79e"}
PSHORT = {1: "P1", 2: "P2", 3: "P3", 4: "P4", 5: "P5"}

SUBSET = (1, 2, 3)  # exemple à 3 piliers : 6 feuilles, lisible

# colonnes x
X1, X2, X3, XLEAF = 0.0, 1.0, 2.0, 2.62


def node(ax, x, y, pill, r=0.16):
    ax.scatter([x], [y], s=1500, color=PCOL[pill], edgecolor="#fcfcfb",
               linewidth=1.6, zorder=4)
    ax.text(x, y, PSHORT[pill], ha="center", va="center", color="#ffffff",
            fontsize=11, fontweight="bold", zorder=5)


def edge(ax, x0, y0, x1, y1, label):
    ax.annotate("", xy=(x1 - 0.16, y1), xytext=(x0 + 0.16, y0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3,
                                shrinkA=0, shrinkB=0), zorder=2)
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    ax.text(xm, ym + 0.06, f"×{label:.2f}", ha="center", va="bottom",
            fontsize=8.2, color=INK2,
            bbox=dict(boxstyle="round,pad=0.12", fc="#fcfcfb", ec="none"),
            zorder=3)


def main():
    fig, ax = plt.subplots(figsize=(11.0, 6.4))

    # positions des feuilles : les 6 permutations, ordonnées par racine
    leaves = []
    for root in SUBSET:
        for perm in permutations([p for p in SUBSET if p != root]):
            leaves.append((root,) + perm)
    ys = list(range(len(leaves)))[::-1]  # haut → bas

    # niveau 1 : chaque racine centrée sur ses enfants
    root_rows = {}
    for r in SUBSET:
        rows = [ys[i] for i, o in enumerate(leaves) if o[0] == r]
        root_rows[r] = sum(rows) / len(rows)

    # racines (niveau 1) + badge ROOT
    for r in SUBSET:
        yr = root_rows[r]
        node(ax, X1, yr, r)
        ax.text(X1, yr + 0.42, f"ROOT = {ROOT[r]:.2f}", ha="center", va="bottom",
                fontsize=8.4, color=INK2, style="italic")

    # arêtes + niveaux 2 et 3 + feuilles
    for i, order in enumerate(leaves):
        y = ys[i]
        a, b, c = order
        yr = root_rows[a]
        # niveau 1 -> 2
        edge(ax, X1, yr, X2, y, TRANS[a][b])
        node(ax, X2, y, b)
        # niveau 2 -> 3
        edge(ax, X2, y, X3, y, TRANS[b][c])
        node(ax, X3, y, c)
        # feuille : ordre complet + proba
        pr = proba_raw(order)
        ps = proba_score(order)
        chain = "→".join(PSHORT[p] for p in order)
        box = FancyBboxPatch((XLEAF, y - 0.28), 1.15, 0.56,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             fc="#ffffff", ec=GRID, lw=1.0, zorder=3)
        ax.add_patch(box)
        ax.text(XLEAF + 0.10, y + 0.02, chain, ha="left", va="center",
                fontsize=9.2, color=INK, fontweight="bold", zorder=4)
        ax.text(XLEAF + 0.10, y - 0.17, f"10·({pr:.3f})", ha="left", va="center",
                fontsize=7.4, color=MUTED, zorder=4)
        # badge proba
        ax.scatter([XLEAF + 1.02], [y], s=430, color=PCOL[1] if ps >= 3 else GRID,
                   edgecolor="#fcfcfb", lw=1.2, zorder=4)
        ax.text(XLEAF + 1.02, y, f"{ps}", ha="center", va="center",
                fontsize=9.5, fontweight="bold",
                color="#ffffff" if ps >= 3 else INK2, zorder=5)

    # en-têtes de colonnes
    for x, lab in [(X1, "n = 1\nracine"), (X2, "n = 2\n+ 1 pilier"),
                   (X3, "n = 3\n+ 1 pilier"), (XLEAF + 0.55, "ordre complet\n→ proba /10")]:
        ax.text(x, ys[0] + 0.95, lab, ha="center", va="bottom",
                fontsize=8.6, color=MUTED, linespacing=1.3)

    # titre + sous-titre
    fig.subplots_adjust(top=0.80, left=0.02, right=0.98, bottom=0.10)
    fig.text(0.02, 0.945, "L'arbre récursif des ordres : la proba se construit lien par lien",
             fontsize=13.5, fontweight="bold", color=INK, ha="left")
    fig.text(0.02, 0.895,
             "Exemple à 3 piliers (P1·P2·P3). Chaque feuille = un ordre ; sa proba est le produit ROOT × transitions le long du chemin.",
             fontsize=9.2, color=INK2, ha="left")

    # encadré récurrence
    ax.text(X1 - 0.30, -1.15,
            r"$p(a\!\to\!b\!\to\!c)\;=\;p(a\!\to\!b)\;\times\;TRANS[b\!\to\!c]$"
            "        (proba de l'enfant = proba du parent × lien ajouté)\n"
            r"Récurrence : $N_n = n\cdot N_{n-1}$  →  2, 6, 24, 120 ordres pour 5 piliers.",
            ha="left", va="top", fontsize=9.6, color=INK, linespacing=1.9,
            bbox=dict(boxstyle="round,pad=0.6", fc="#f3f1ea", ec=GRID, lw=1.0))

    ax.set_xlim(-0.5, XLEAF + 1.35)
    ax.set_ylim(-1.9, ys[0] + 1.5)
    ax.axis("off")

    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "F12_arbre_recursif.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("écrit :", path)


if __name__ == "__main__":
    main()
