#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08f : cas d'usage PILIER 3 (tests de resilience, TLPT), oriente KPI.

P3 n'a PAS de donnee d'incident directe : la resilience est un CONTROLE (on teste), pas
un evenement. Son EFFET s'observe : une entite bien testee CONTIENT plus vite. Le KPI
proxy est donc la DUREE DE BRECHE (occurrence -> fin), c'est-a-dire le temps d'exposition
avant containment. DORA impose les tests de resilience (art. 24-27, TLPT) precisement
pour ecraser cette duree.

A ne pas confondre avec P2 : P2 = delai avant de DECLARER (detection + notification) ;
P3 = duree pendant laquelle la breche reste ACTIVE (containment / retablissement). Deux
etapes distinctes de la chaine de reponse.

CAVEAT ASSUME. end_breach_date est la fenetre de breche RAPPORTEE, pas un MTTR mesure en
SOC. C'est un proxy de la duree d'exposition, pas une mesure d'ingenierie. Donnee US,
proxy d'entites DORA (idem 08e).

Donnees attendues dans data/raw/ (non versionnees). Sortie : diagnostics + figure.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "data", "raw"))
SRC = os.path.join(RAW, "Data_Breach_Chronology.xlsx")
if not os.path.exists(SRC):
    sys.exit(f"donnee absente : {SRC}\n(les sources brutes ne sont pas versionnees)")

W = 78
TYPE_LAB = {"HACK": "Piratage", "UNKN": "Non precise", "DISC": "Divulgation",
            "INSD": "Interne", "PORT": "Perte de support", "PHYS": "Physique",
            "CARD": "Carte", "STAT": "Statique"}


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# =====================================================================================
titre("Cas d'usage P3 : duree de breche (temps de containment)")
# =====================================================================================
d = pd.read_excel(SRC, sheet_name="Data_Breach_Chronology",
                  usecols=["breach_date", "end_breach_date", "breach_type",
                           "organization_type"])
d["bd"] = pd.to_datetime(d.breach_date.astype(str).str.strip(), errors="coerce")
d["ed"] = pd.to_datetime(d.end_breach_date.astype(str).str.strip(), errors="coerce")
bsf = d[(d.organization_type == "BSF") & d.bd.notna() & d.ed.notna()].copy()
bsf["dur"] = (bsf.ed - bsf.bd).dt.days
ok = bsf[(bsf.dur >= 0) & (bsf.dur <= 3650)].copy()
D = ok.dur.to_numpy()
print(f"incidents BSF avec debut et fin : {len(ok)}")
qs = {q: np.quantile(D, q) for q in (0.5, 0.9, 0.99)}
print(f"duree (jours) : mediane {qs[0.5]:.0f} | q90 {qs[0.9]:.0f} | q99 {qs[0.99]:.0f} "
      f"| moyenne {D.mean():.0f}")
print(f"contenu le JOUR MEME (duree 0) : {(D == 0).mean():.1%}")

# fenetres de containment
print("\nKPI CONTAINMENT : part contenue dans chaque fenetre")
wins = [(0, "le jour meme"), (7, "sous 1 semaine"), (30, "sous 1 mois")]
share = {}
for thr, lab in wins:
    share[thr] = (D <= thr).mean()
    print(f"   {lab:<16} : {share[thr]:6.1%}")
frac_long = (D > 90).mean()
print(f"   au-dela de 90 j (breche prolongee) : {frac_long:.1%}")

# duree par type
print("\nKPI par type d'incident (mediane de duree) :")
by_type = (ok.groupby("breach_type").dur
           .agg(["size", "median"]).query("size >= 30").sort_values("median"))
for t, row in by_type.iterrows():
    print(f"   {TYPE_LAB.get(t, t):<16} n={int(row['size']):>5}  "
          f"mediane={row['median']:6.0f} j")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("La duree de containment est tres etalee : la moitie des breches durent moins")
print(f"d'une semaine ({share[7]:.0%} sous 7 j), mais {frac_long:.0%} restent actives au-dela")
print("de 90 jours. C'est cette QUEUE que les tests de resilience (P3, TLPT) visent a")
print("couper : l'ecart entre la mediane et la queue EST la marge de progression testable.")
print("KPI P3 propose : part des breches contenues sous 1 semaine, et longueur de la queue")
print("(q90). Deux nombres qu'un exercice de crise fait bouger, et qu'un audit DORA suit.")

# =====================================================================================
# figure
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 11,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
})
INK, INK2, MUTED, GRID = "#1b1e30", "#223e55", "#595959", "#f2f2f2"
ACCENT = "#a6002e"
BL = ["#7baafd", "#4c79c7", "#204993"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 4.8),
                                    gridspec_kw={"width_ratios": [1.1, 1, 1]})

# (a) distribution de la duree (avec les durees 0 mises a 0.5 pour l'echelle log)
Dpos = np.where(D == 0, 0.5, D)
bins = np.concatenate([[0.3], np.logspace(0, np.log10(3650), 34)])
ax1.hist(Dpos, bins=bins, color=BL[1], edgecolor="#fcfcfb")
ax1.set_xscale("log")
ax1.axvline(qs[0.5], color=INK, lw=1.8)
ax1.text(qs[0.5] * 1.15, ax1.get_ylim()[1] * 0.85, f"mediane\n{qs[0.5]:.0f} j", color=INK,
         fontsize=8.5)
ax1.axvline(90, color=ACCENT, lw=1.4, ls="--")
ax1.text(96, ax1.get_ylim()[1] * 0.6, f"90 j\n{frac_long:.0%} au-dela", color=ACCENT,
         fontsize=8.2)
ax1.set_xlabel("duree de breche : occurrence $\\to$ fin (jours, log)", color=INK2)
ax1.set_ylabel("nombre d'incidents", color=INK2)
ax1.set_title("(a)  Mediane courte, mais une longue queue", fontsize=11, color=INK, pad=8)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# (b) duree mediane par type
labs = [TYPE_LAB.get(t, t) for t in by_type.index]
meds = by_type["median"].to_numpy()
yb = np.arange(len(labs))
imax = int(np.argmax(meds))
cols = [ACCENT if i == imax else BL[1] for i in range(len(meds))]
ax2.barh(yb, meds, color=cols, edgecolor="#fcfcfb", height=0.66)
for y, m, n in zip(yb, meds, by_type["size"]):
    ax2.text(m + max(meds) * 0.02, y, f"{m:.0f} j  (n={int(n)})", va="center",
             fontsize=8, color=INK2)
ax2.set_yticks(yb); ax2.set_yticklabels(labs, fontsize=8.7)
ax2.set_xlim(0, meds.max() * 1.35)
ax2.set_xlabel("duree mediane de breche (jours)", color=INK2)
ax2.set_title("(b)  Les acces persistants (interne, carte) trainent", fontsize=11,
              color=INK, pad=8)
for s in ("top", "right", "left"):
    ax2.spines[s].set_visible(False)
ax2.tick_params(axis="y", length=0)

# (c) fenetres de containment (marches)
cats = ["jour meme", "sous 1 sem.", "sous 1 mois", "au-dela"]
vals = [share[0], share[7] - share[0], share[30] - share[7], 1 - share[30]]
cols3 = [BL[2], BL[1], BL[0], ACCENT]
left = 0.0
for c, v, col in zip(cats, vals, cols3):
    ax3.barh([0], [v * 100], left=left * 100, color=col, edgecolor="#fcfcfb", height=0.5)
    if v > 0.03:
        ax3.text((left + v / 2) * 100, 0, f"{c}\n{v:.0%}", ha="center", va="center",
                 fontsize=8.2, color="#ffffff" if col in (BL[2], ACCENT) else INK)
    left += v
ax3.set_xlim(0, 100); ax3.set_ylim(-0.8, 0.8); ax3.set_yticks([])
ax3.set_xlabel("part des incidents (%)", color=INK2)
ax3.set_title("(c)  Vitesse de containment (proxy de resilience)", fontsize=11,
              color=INK, pad=8)
ax3.text(0.5, -0.62, "la queue au-dela d'un mois est la cible des tests de resilience\n"
         "(P3 / TLPT) : c'est la que l'exercice de crise se mesure",
         transform=ax3.transAxes, ha="center", fontsize=8.2, color=INK2, style="italic")
for s in ("top", "right", "left", "bottom"):
    ax3.spines[s].set_visible(False)

fig.suptitle("R : cas d'usage P3, le temps de containment comme KPI de resilience",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "R_cas_usage_p3.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
