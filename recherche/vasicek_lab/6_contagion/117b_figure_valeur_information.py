#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
117b : la figure de la valeur de l'information sur la grille commune, tracee depuis 117.txt.

Remplace 116b, dont les ensembles compares n'etaient pas emboites. Aucun calcul de capital : le
resserrement moyen et son etendue, et les dix valeurs par paire (moyenne, minimum, maximum sur
quatre graines) sont LUS dans 117.txt. Les bandes de la graine de reference sont lues dans 116.txt
(reference, six sens, dix sens : a cette graine ce sont exactement les filtres de la grille du 117,
et le script le controle en redonnant ses resserrements a 0,1 point) ; le socle est lu dans 55.txt.

Sortie : figures/valeur_information_grille.png, nom sans chiffre. Rien sur la sortie standard.
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
    t117 = open(os.path.join(SORTIES, "117.txt"), encoding="utf-8").read()
    t55 = open(os.path.join(SORTIES, "55.txt"), encoding="utf-8-sig").read()
    b = {}
    for cle, motif in (("tot", r"ignorance totale\s+\[(\d+) ; (\d+)\]"),
                       ("six", r"(?m)^\s+%d\s+\d+\s+\[\d+ ; \d+\]\s+\d+\s+\[(\d+) ; (\d+)\]" % 20260721),
                       ("dix", r"dix sens, ensemble contraint\s+\[(\d+) ; (\d+)\]")):
        m = re.search(motif, t117 if cle == "six" else t116)
        if not m:
            sys.exit(f"ARRET : bande « {cle} » introuvable dans 116.txt.")
        b[cle] = (float(m.group(1)), float(m.group(2)))
    m = re.search(r"six sens moyenne ([\d.]+) ecart-type [\d.]+ minimum ([\d.]+) maximum ([\d.]+)", t117)
    paires = re.findall(r"paire (P\d-P\d) moyenne (-?[\d.]+) minimum (-?[\d.]+) maximum (-?[\d.]+)", t117)
    r0 = re.search(r"(?m)^\s+20260721\s+\d+\s+\[\d+ ; \d+\]\s+(\d+)\s+\[\d+ ; \d+\]\s+\d+\s+([\d.]+) %\s+([\d.]+) %", t117)
    if not r0:
        sys.exit("ARRET : ligne de la graine de reference introuvable dans 117.txt.")
    w0 = b["tot"][1] - b["tot"][0]
    for cle, attendu in (("six", float(r0.group(2))), ("dix", float(r0.group(3)))):
        if abs(100 * (1 - (b[cle][1] - b[cle][0]) / w0) - attendu) > 0.1 or float(r0.group(1)) != w0:
            sys.exit(f"ARRET : la bande « {cle} » ne redonne pas le resserrement du 117 ({attendu} %).")
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
ax2.set_yticks(yp); ax2.set_yticklabels([r[0] for r in rows])
ax2.set_ylim(-0.6, len(rows) - 0.4)
ax2.set_xlim(0, 34)
ax2.set_xlabel("plage levée si cette seule paire est documentée (%)")
ax2.set_title("(b)  Quelle dépendance documenter en priorité", fontsize=9.5, color=INK, loc="left")
ax2.legend(handles=[Patch(facecolor=BLUE, alpha=0.9, label="sens documenté\npar un post-mortem"),
                    Patch(facecolor=MUTED, alpha=0.9, label="sens tiré du\nclasseur d'expert")],
           loc="lower right", fontsize=7.2, frameon=False, handlelength=1.2)
for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout(w_pad=1.8)
fig.savefig(os.path.join(HERE, "figures", "valeur_information_grille.png"), dpi=200,
            bbox_inches="tight")
