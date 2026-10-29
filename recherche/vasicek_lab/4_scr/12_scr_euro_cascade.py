#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12 : SCR en EUROS par la cascade, calibre COMME LE MEMOIRE (SAS OpRisk / PRC).

Objectif : produire, dans le track exploratory, un SCR chiffre en euros directement
comparable a celui du memoire, mais en passant par la CASCADE dirigee des piliers au
lieu de l'architecture a 4 briques + copule de Gumbel.

PRINCIPE (aucune recalibration nouvelle). On reutilise TELS QUELS les modules du vrai
modele :
  - severite : src.aggregation.lda.simulate_remediation_severity (GPD spliced en euros,
    parametres OpRisk xi=0,595 / PRC xi=1,033 de src.utils.config) ;
  - frequence : NegBin (sur-dispersion phi=9,2) et multiplicateurs de scenario DORA de
    src.frequency.negbin.
On ne remplace QUE l'agregation. Chez le memoire, un incident = un tirage de severite.
Ici, un incident AMORCE se propage a un ensemble S de piliers (noyau auto-evitant,
gain g) et chaque pilier touche tire sa severite ; la severite de l'incident est la
somme. La cascade internalise donc mecaniquement la co-occurrence que le memoire porte
par la brique prestataire + copule de Gumbel.

LECTURE. A g=0 (aucune propagation, |S|=1) la cascade se reduit a UN tirage de severite
par incident : c'est exactement le LDA du memoire sans propagation, et le SCR obtenu
retombe dans la bande du memoire (controle de coherence ; rappel : l'IC90% du memoire
sur la VaR vaut deja un facteur ~2,5, le SCR est une distribution large, pas un point).
A g>0, l'ecart est la PRIME DE PROPAGATION : le surcout de capital du a la contagion
inter-piliers d'un meme incident. C'est la vue structurelle (bottom-up) en regard de la
vue copule (top-down) du memoire.

PORTEE. Meme calibration euro que le memoire, memes biais (OpRisk = grandes entites ;
PRC = severite derivee Jacobs, plafonnee a 40 M€). L'amorce se repartit sur les 5 piliers
DORA selon ROOT (dire d'expert de la cascade), la ou le memoire decompose par vecteur
Hackmageddon : seuls le NIVEAU de frequence et la loi de severite sont communs, pas la
structure de propagation. Rien n'est lu dans data/raw : tout vient de la config figee.

Sortie : diagnostics (SCR euro par source x gain, prime de propagation) + figure
S6_scr_euro_cascade.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                              # noqa: E402  (moteur partage)
from euro_cascade_model import PARAMS, MEMOIRE_TOTAL, var    # noqa: E402

RNG = np.random.default_rng(20260715)
W = 74
NY = {"OPRISK": 150_000, "PRC": 50_000}     # PRC : lambda ~30x => moins d'annees suffisent


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def scr_euro(source, g, scenario, n_years, rng):
    """SCR euro (VaR 99,5%) par la cascade, parametres de severite = config figee."""
    sp = PARAMS[source]
    lam = ec.lambda_scenario(source, scenario, mode="center")
    losses = ec.simulate_euro(lam, g, sp["xi"], sp["sigma"], sp["u"], sp["p_u"],
                              sp["cap"], n_years, rng)
    return var(losses)


# ============================================================ SCR euro : g=0 (controle) vs g=0.9
titre("SCR en euros par la cascade, scenario S2 non-conforme (comme le memoire)")
print("  g=0 : aucune propagation (|S|=1) = le LDA du memoire sans propagation.")
print("  g=0,90 : cascade de base -> prime de propagation.")
print(f"  {'source':<9}{'g':>6}{'SCR euro (M EUR)':>18}{'reference memoire':>22}")
res = {}
for source in ("OPRISK", "PRC"):
    for g in (0.0, ec.G_BASE):
        v = scr_euro(source, g, "S2_non_conforme", NY[source], RNG)
        res[(source, g)] = v
        tag = f"~{MEMOIRE_TOTAL[source]:.0f} M EUR (total)" if g == 0 else ""
        print(f"  {source:<9}{g:>6.2f}{v:>18.0f}{tag:>22}")

titre("Prime de propagation (surcout de capital du a la cascade)")
for source in ("OPRISK", "PRC"):
    v0, v9 = res[(source, 0.0)], res[(source, ec.G_BASE)]
    print(f"  {source:<9}: g=0 -> {v0:>8.0f}   g=0,90 -> {v9:>8.0f} M EUR   "
          f"prime = +{100*(v9/v0-1):.0f} %")
print("  g=0 tombe dans la bande du memoire (controle) ; la propagation inter-piliers")
print("  d'un incident rencherit le capital : contrepartie mecaniste de la brique")
print("  prestataire + copule de Gumbel du memoire (dependance modelisee, non ajoutee).")

# ============================================================ balayage du gain g (OpRisk)
titre("Effet du gain de propagation g sur le SCR euro (OpRisk, S2)")
print(f"  {'g':>6}{'SCR euro (M EUR)':>18}")
gcurve = {}
for g in (0.0, 0.3, 0.6, 0.9, 1.0):
    gcurve[g] = scr_euro("OPRISK", g, "S2_non_conforme", 80_000, RNG)
    print(f"  {g:>6.2f}{gcurve[g]:>18.0f}")

# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2 = "#1b1e30", "#223e55"
ACCENT, BLUE, GREY = "#a6002e", "#3661ac", "#7d7d7d"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.2))

# panneau A : SCR euro vs g (OpRisk), avec repere memoire (total S2)
gs = sorted(gcurve)
axA.plot(gs, [gcurve[g] for g in gs], "-o", color=BLUE, lw=2.0, ms=4.5,
         label="cascade (OpRisk)")
axA.axhline(MEMOIRE_TOTAL["OPRISK"], color=ACCENT, ls="--", lw=1.3)
axA.text(1.0, MEMOIRE_TOTAL["OPRISK"], " SCR memoire (4 briques + copule)",
         ha="right", va="bottom", fontsize=8.5, color=ACCENT)
axA.set_xlabel("gain de propagation g", fontsize=9.5, color=INK2)
axA.set_ylabel("SCR = VaR 99,5 % (M€)", fontsize=9.5, color=INK2)
axA.set_title("(A)  A g=0, la cascade rejoint le memoire ; g>0 = prime de propagation",
              fontsize=10, color=INK, pad=6)
axA.legend(fontsize=8.6, frameon=False, loc="lower right")
axA.grid(alpha=0.25, lw=0.5)

# panneau B : barres memoire / cascade g=0 / cascade g=0.9, par source
srcs = ["OPRISK", "PRC"]
x = np.arange(len(srcs))
wbar = 0.26
mem = [MEMOIRE_TOTAL[s] for s in srcs]
c0 = [res[(s, 0.0)] for s in srcs]
c9 = [res[(s, ec.G_BASE)] for s in srcs]
axB.bar(x - wbar, mem, wbar, color=GREY, label="memoire (4 briques)")
axB.bar(x, c0, wbar, color=BLUE, label="cascade g=0 (sans propagation)")
axB.bar(x + wbar, c9, wbar, color=ACCENT, label="cascade g=0,90")
for i in range(len(srcs)):
    for xx, val in [(x[i]-wbar, mem[i]), (x[i], c0[i]), (x[i]+wbar, c9[i])]:
        axB.text(xx, val, f"{val:.0f}", ha="center", va="bottom", fontsize=7.8, color=INK)
axB.set_xticks(x)
axB.set_xticklabels(["OpRisk", "PRC"])
axB.set_ylabel("SCR = VaR 99,5 % (M€)", fontsize=9.5, color=INK2)
axB.set_title("(B)  SCR euro : memoire vs cascade, par source de severite",
              fontsize=10, color=INK, pad=6)
axB.legend(fontsize=8.2, frameon=False)
axB.grid(alpha=0.25, lw=0.5, axis="y")

fig.suptitle("SCR en euros par la cascade, calibre comme le memoire (SAS OpRisk / PRC)",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S6_scr_euro_cascade.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
