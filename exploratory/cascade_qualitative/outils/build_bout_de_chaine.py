#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure « bout de chaîne » : propension de chaque pilier à ÉTEINDRE la cascade.

Vue complémentaire à la carte de criticité (F1..F6). La criticité dit QUI amorce
et QUELLE gravité ; ici on lit QUI termine (le pilier va « vers rien »).

Métrique (miroir de l'état d'arrêt du modèle de propagation) :
  s_j    = force sortante du pilier j        = somme_k TRANS[j][k]
  e_j    = propension à propager             = g * s_j / max_k s_k
  term_j = propension à être en bout de chaîne = 1 - e_j = P(∅| j)

Le CLASSEMENT ne dépend que de s_j, donc il est invariant au gain de contagion g
(non calibré). Les valeurs en % sont données au cas de référence g = 1, où le
pilier le plus connecté (P1) propage toujours. Sortie qualitative : c'est une
vraisemblance d'être terminal, pas une probabilité d'extinction estimée.

Importe le jugement d'expert depuis cascade_model.py (source unique).
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cascade_model import TRANS  # noqa: E402

# noms courts (cohérents avec build_figures.py)
PILIERS = {
    1: "Gouvernance & risque TIC",
    2: "Gestion des incidents",
    3: "Tests de résilience (TLPT)",
    4: "Risque lié aux tiers ICT",
    5: "Partage d'informations",
}
N = len(PILIERS)
G = 1.0  # gain de contagion, cas de référence (le classement n'en dépend pas)

NIV = ["Quasi jamais", "Faible", "Moyenne", "Forte", "Très forte"]


def niveau(t):
    """Niveau qualitatif 0..4 de la propension terminale t in [0,1] (bornes /10)."""
    x = t * 10
    return 0 if x <= 2 else 1 if x <= 4 else 2 if x <= 6 else 3 if x <= 8 else 4


# ------------------------------------------------------------- métrique bout de chaîne
S = {j: sum(TRANS[j].values()) for j in range(1, N + 1)}   # force sortante
SMAX = max(S.values())
TERM = {j: 1.0 - G * S[j] / SMAX for j in S}                # propension terminale

# ------------------------------------------------------------- style / palette (miroir build_figures)
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "font.size": 11,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "text.color": "#0b0b0b",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#898781",
    "ytick.color": "#898781",
    "axes.grid": False,
})
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID = "#e1e0d9"
BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blues_seq", BLUES)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)


def build():
    order = sorted(range(1, N + 1), key=lambda j: TERM[j])   # ascendant : P5 en haut
    y = range(N)
    vals = [TERM[j] * 100 for j in order]
    labels = [f"P{j} : {PILIERS[j]}" for j in order]

    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    colors = [SEQ(0.18 + 0.82 * TERM[j]) for j in order]
    ax.barh(y, vals, color=colors, edgecolor="#fcfcfb", height=0.62, zorder=3)
    for yi, j in zip(y, order):
        t = TERM[j]
        ax.text(t * 100 + 1.5, yi, f"{NIV[niveau(t)]}  ({t * 100:.0f} %)",
                va="center", fontsize=9, color=INK2)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Propension à éteindre la cascade, aller « vers rien »  (%)", color=INK2)
    ax.set_title("Quels piliers sont en bout de chaîne", fontsize=15,
                 fontweight="bold", color=INK, pad=26, loc="left")
    ax.text(0, N - 0.28, "Miroir de la carte de criticité : P1/P4 amorcent, P5 termine. "
            "Classement robuste au gain g (ici g = 1).", fontsize=9, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(OUT, "F13_bout_de_chaine.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def dump_table():
    """Affiche les valeurs (contrôle + report dans la fiche LaTeX)."""
    print("pilier | force sortante s_j | propension terminale (g=1) | niveau")
    for j in sorted(range(1, N + 1), key=lambda k: TERM[k], reverse=True):
        print(f"P{j} {PILIERS[j]:<28} | {S[j]:.2f} | {TERM[j]*100:5.0f} % | {NIV[niveau(TERM[j])]}")


if __name__ == "__main__":
    build()
    dump_table()
    print("Figure : figures/F13_bout_de_chaine.png")
