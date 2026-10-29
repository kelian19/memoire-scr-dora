#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REMPLACE PAR LE SCRIPT 117 (19 octobre 2026, soir) : les ensembles compares ici ne sont pas
emboites, d'ou des resserrements negatifs. Le memoire cite le 117 ; ce script reste pour la trace.

116b : la figure de la valeur de l'information corrigee, tracee depuis sorties_verif/116.txt.

Aucun calcul de capital : les bandes, le resserrement moyen et son etendue, et les dix valeurs
par paire (moyenne, minimum, maximum sur quatre graines) sont LUS dans 116.txt ; le socle est lu
dans 55.txt. Une lecture incomplete arrete le script.

L'axe du panneau (b) descend sous zero : sur quatre graines, documenter le sens d'une seule paire
ELARGIT parfois la plage aux coins, parce qu'un coefficient a zero n'est pas un sommet du pave
initial et que le capital n'est pas lineaire en A. Les coins ne garantissent donc pas les extrema.

Sortie : figures/valeur_information_contrainte.png, nom sans chiffre. Rien sur la sortie standard.
"""

import os
import re
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SORTIES = os.path.abspath(os.path.join(HERE, "../../resultats/sorties_verif"))
PM = {"P1-P3", "P1-P4", "P1-P5", "P2-P3", "P2-P4", "P3-P5"}


def lire():
    t116 = open(os.path.join(SORTIES, "116.txt"), encoding="utf-8").read()
    t55 = open(os.path.join(SORTIES, "55.txt"), encoding="utf-8-sig").read()
    b = {}
    for cle, motif in (("tot", r"ignorance totale\s+\[(\d+) ; (\d+)\]"),
                       ("six", r"six sens, ensemble contraint\s+\[(\d+) ; (\d+)\]"),
                       ("dix", r"dix sens, ensemble contraint\s+\[(\d+) ; (\d+)\]")):
        m = re.search(motif, t116)
        if not m:
            sys.exit(f"ARRET : bande « {cle} » introuvable dans 116.txt.")
        b[cle] = (float(m.group(1)), float(m.group(2)))
    m = re.search(r"bruit resserrement moyenne ([\d.]+) ecart-type [\d.]+ minimum ([\d.]+) maximum ([\d.]+)",
                  t116)
    paires = re.findall(r"paire (P\d-P\d) moyenne (-?[\d.]+) minimum (-?[\d.]+) maximum (-?[\d.]+)", t116)
    socle = re.search(r"Socle sans contagion \(reference\) : (\d+) M", t55)
    if not (m and socle) or len(paires) != 10:
        sys.exit("ARRET : lecture de 116.txt ou 55.txt incomplete.")
    red = tuple(float(x) for x in m.groups())
    rows = [(p, float(a), float(lo), float(hi)) for p, a, lo, hi in paires]
    return b, red, rows, float(socle.group(1))


b, (moy, rmin, rmax), rows, SOCLE = lire()

mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 8.5,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"
ESP = " "


def fr(x, nd=0):
    return f"{x:,.{nd}f}".replace(",", ESP).replace(".", ",").replace("-", "−")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.3, 2.6), gridspec_kw={"width_ratios": [1, 1.35]})

etapes = [("ignorance\ntotale", b["tot"], MUTED), ("six sens\ndocumentés", b["six"], BLUE),
          ("dix sens\nconnus", b["dix"], GREEN)]
for i, (lab, (lo, hi), c) in enumerate(etapes):
    ax1.plot([i, i], [lo, hi], color=c, lw=15, solid_capstyle="butt", alpha=0.85)
    ax1.text(i, hi + 70, f"largeur\n{fr(hi - lo)} M€", ha="center", va="bottom",
             fontsize=7.5, color=INK2)
ax1.text(1, b["six"][0] - 90, f"−{fr(moy, 1)} % en\nmoyenne ({fr(rmin, 1)}\nà {fr(rmax, 1)} %)",
         ha="center", va="top", fontsize=7.2, color=BLUE)
ax1.text(2, b["dix"][0] - 90, "reste\nl'amplitude", ha="center", va="top", fontsize=7.2, color=GREEN)
ax1.axhline(SOCLE, color=ACCENT, ls=":", lw=1.1)
ax1.text(-0.5, SOCLE + 60, "socle sans contagion", fontsize=7.5, color=ACCENT, va="bottom")
ax1.set_xticks(range(3)); ax1.set_xticklabels([e[0] for e in etapes])
ax1.set_xlim(-0.55, 2.55)
ax1.set_ylim(5000, 9600)
ax1.yaxis.set_major_locator(MultipleLocator(1000))
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fr(v)))
ax1.set_ylabel("bande de besoin de capital (M€)")
ax1.set_title("(a)  L'information rétrécit la bande", fontsize=9.5, color=INK, loc="left")

yp = np.arange(len(rows))[::-1]
moys = np.array([r[1] for r in rows])
cols = [BLUE if r[0] in PM else MUTED for r in rows]
ax2.barh(yp, moys, height=0.68, color=cols, alpha=0.9)
ax2.errorbar(moys, yp, xerr=[moys - np.array([r[2] for r in rows]), np.array([r[3] for r in rows]) - moys],
             fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
for y_, r in zip(yp, rows):
    ax2.text(r[3] + 0.6, y_, f"{fr(r[1], 1)} %", va="center", fontsize=7.2, color=INK2)
ax2.axvline(0, color="#9a9a9a", lw=0.8)
ax2.set_yticks(yp); ax2.set_yticklabels([r[0] for r in rows])
ax2.set_ylim(-0.6, len(rows) - 0.4)
ax2.set_xlim(min(r[2] for r in rows) - 1.5, 34)
ax2.set_xlabel("plage levée si cette seule paire est documentée (%)")
ax2.set_title("(b)  Quelle dépendance documenter en priorité", fontsize=9.5, color=INK, loc="left")
ax2.legend(handles=[Patch(facecolor=BLUE, alpha=0.9, label="sens documenté\npar un post-mortem"),
                    Patch(facecolor=MUTED, alpha=0.9, label="sens tiré du\nclasseur d'expert")],
           loc="lower right", fontsize=7.2, frameon=False, handlelength=1.2)
for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout(w_pad=1.8)
fig.savefig(os.path.join(HERE, "figures", "valeur_information_contrainte.png"), dpi=200,
            bbox_inches="tight")
