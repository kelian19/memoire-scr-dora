#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13 : Delta_DORA par la cascade, avec bootstrap deux niveaux (comme le memoire).

Delta_DORA = SCR(non-conforme) - SCR(conforme) = surcout de capital strictement
imputable au defaut de conformite, a graine COMMUNE entre les deux etats (common random
numbers : l'ecart ne vient que du changement de conformite, pas du bruit Monte-Carlo).
Reproduit la methode du notebook 07 du memoire, mais l'agregation passe par le Vasicek
DIRIGE (cascade : noyau e_j = g*s_j/max_s, cf. note_vasicek_dirige) au lieu des 4 briques
+ copule de Gumbel.

DEUX CANAUX de conformite (le second EST le Vasicek dirige) :
  1. FREQUENCE  : S0 conforme vs S2 non-conforme (multiplicateurs de src.frequency.negbin),
     canal du memoire.
  2. PROPAGATION : une entite conforme CONTIENT mieux la contagion -> gain g plus faible
     (g_c < g_nc). Canal propre a la cascade : la non-conformite laisse un incident se
     propager plus loin entre piliers.

BOOTSTRAP DEUX NIVEAUX (notebook 07) :
  - niveau 1 : incertitude des multiplicateurs (tirage dans les fourchettes sourcees) ;
  - niveau 2 : incertitude de severite (xi, sigma tires dans leur IC90 de la config,
    approx normale) -> aucune lecture de data/raw.
  Graine MC commune entre etats conforme/non-conforme a chaque iteration.

PORTEE. Meme calibration euro et memes biais que le memoire. Le canal PROPAGATION (g_c)
est un choix de modelisation propre a la cascade, non calibre : presente en sensibilite.
Ne touche ni src/ ni memoire/. Comparaison au Delta_DORA du memoire (OpRisk 3879 M€
[1497 ; 22249] ; PRC 2015 M€ [1607 ; 2366]).

Sortie : diagnostics + figure S7_delta_dora_cascade.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                              # noqa: E402
from euro_cascade_model import PARAMS, MEMOIRE_DELTA, var    # noqa: E402

W = 74
G_NC = ec.G_BASE          # gain non-conforme (propagation de base 0,90)
BOOT = {"OPRISK": dict(B=120, ny=20_000), "PRC": dict(B=60, ny=3_000)}


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def delta_once(source, g_nc, g_c, xi, sigma, ny, seed):
    """Un Delta_DORA : SCR(S2, g_nc) - SCR(S0, g_c), graine MC commune (CRN)."""
    sp = PARAMS[source]
    lam_nc = ec.lambda_scenario(source, "S2_non_conforme", mode="center")
    lam_c = ec.lambda_scenario(source, "S0_conforme", mode="center")
    rng_nc = np.random.default_rng(seed)
    rng_c = np.random.default_rng(seed)          # meme graine -> CRN
    loss_nc = ec.simulate_euro(lam_nc, g_nc, xi, sigma, sp["u"], sp["p_u"], sp["cap"], ny, rng_nc)
    loss_c = ec.simulate_euro(lam_c, g_c, xi, sigma, sp["u"], sp["p_u"], sp["cap"], ny, rng_c)
    return var(loss_nc), var(loss_c)


# ============================================================ bootstrap Delta_DORA (canal frequence)
titre("Delta_DORA par la cascade, bootstrap 2 niveaux (canal FREQUENCE, g fixe = 0,90)")
print("  g_nc = g_c = 0,90 : conformite agit sur la frequence (S2 vs S0), comme le memoire.")
boot = {}
for source in ("OPRISK", "PRC"):
    cfg = BOOT[source]
    sp = PARAMS[source]
    orng = np.random.default_rng(20260715)
    deltas, scr_nc_s, scr_c_s = [], [], []
    for b in range(cfg["B"]):
        # niveau 1 : multiplicateurs (S0 = (1,1), pas de variation ; S2 tire dans les bornes)
        lam_nc = ec.lambda_scenario(source, "S2_non_conforme", mode="sample", rng=orng)
        lam_c = ec.lambda_scenario(source, "S0_conforme", mode="sample", rng=orng)
        # niveau 2 : severite (xi, sigma dans IC90)
        xi_b, sg_b = ec.sample_severity_params(source, orng)
        seed_b = 1000 + b
        rng_nc = np.random.default_rng(seed_b)
        rng_c = np.random.default_rng(seed_b)
        v_nc = var(ec.simulate_euro(lam_nc, G_NC, xi_b, sg_b, sp["u"], sp["p_u"], sp["cap"], cfg["ny"], rng_nc))
        v_c = var(ec.simulate_euro(lam_c, G_NC, xi_b, sg_b, sp["u"], sp["p_u"], sp["cap"], cfg["ny"], rng_c))
        deltas.append(v_nc - v_c)
        scr_nc_s.append(v_nc)
        scr_c_s.append(v_c)
    d = np.array(deltas)
    boot[source] = d
    med = np.median(d)
    lo, hi = np.percentile(d, [5, 95])
    ref = MEMOIRE_DELTA[source]
    print(f"\n  {source} ({cfg['B']} tirages x {cfg['ny']:,} annees) :")
    print(f"    Delta_DORA median = {med:8.0f} M EUR   IC90% [{lo:.0f} ; {hi:.0f}]")
    print(f"    memoire (07)      = {ref['median']:8.0f} M EUR   IC90% "
          f"[{ref['ic90'][0]:.0f} ; {ref['ic90'][1]:.0f}]")

# ============================================================ canal propagation (le Vasicek dirige)
titre("Canal PROPAGATION : la conformite contient la contagion (g_c < g_nc), OpRisk")
print("  g_nc = 0,90 fixe (non-conforme) ; on baisse g_c (conforme mieux contenu).")
print(f"  {'g_c':>6}{'SCR non-conf.':>16}{'SCR conforme':>15}{'Delta_DORA':>13}")
prop = {}
sp = PARAMS["OPRISK"]
for g_c in (0.90, 0.70, 0.50, 0.30, 0.0):
    v_nc, v_c = delta_once("OPRISK", G_NC, g_c, sp["xi"], sp["sigma"], 80_000, seed=7)
    prop[g_c] = v_nc - v_c
    print(f"  {g_c:>6.2f}{v_nc:>16.0f}{v_c:>15.0f}{v_nc - v_c:>13.0f}")
print("  g_c=0,90 : conformite agit sur la seule frequence (canal 1).")
print("  g_c<0,90 : la conformite CONTIENT aussi la propagation -> Delta_DORA plus grand.")

# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2 = "#1b1e30", "#223e55"
ACCENT, BLUE, GREEN, GREY = "#a6002e", "#3661ac", "#008d87", "#7d7d7d"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.4, 5.3))

# panneau A : distribution bootstrap du Delta_DORA (OpRisk) + repere memoire
# borne d'affichage : la queue lourde produit de rares tirages extremes qui
# ecraseraient le gros de la distribution (le nombre omis est annote, pas cache).
d_op = boot["OPRISK"]
cut = float(np.percentile(d_op, 95))
n_omis = int((d_op > cut).sum())
axA.hist(d_op[d_op <= cut], bins=40, density=True, color=BLUE, alpha=0.55,
         label="cascade (bootstrap)")
axA.axvline(np.median(d_op), color=BLUE, lw=1.6, label=f"median cascade {np.median(d_op):.0f}")
ref = MEMOIRE_DELTA["OPRISK"]
axA.axvline(ref["median"], color=ACCENT, lw=1.6, ls="--",
            label=f"median memoire {ref['median']:.0f}")
axA.set_xlim(0, cut)
axA.text(0.98, 0.55, f"{n_omis}/{len(d_op)} tirages > {cut:.0f}\nomis (lisibilite)",
         transform=axA.transAxes, ha="right", va="top", fontsize=7.8, color=INK2)
axA.set_xlabel("Delta_DORA (M€)", fontsize=9.5, color=INK2)
axA.set_ylabel("densite", fontsize=9.5, color=INK2)
axA.set_title("(A)  Delta_DORA bootstrap, canal frequence (OpRisk)",
              fontsize=10, color=INK, pad=6)
axA.legend(fontsize=8.2, frameon=False)
axA.grid(alpha=0.25, lw=0.5)

# panneau B : Delta_DORA vs g_c (canal propagation = Vasicek dirige)
gcs = sorted(prop, reverse=True)
axB.plot(gcs, [prop[g] for g in gcs], "-o", color=GREEN, lw=2.0, ms=4.5)
axB.axhline(prop[0.90], color=GREY, ls=":", lw=1.2)
axB.text(0.0, prop[0.90], " canal frequence seul (g_c=0,90)", ha="left", va="bottom",
         fontsize=8.2, color=INK2)
axB.set_xlabel("gain de propagation conforme  g_c  (g_nc=0,90 fixe)", fontsize=9.5, color=INK2)
axB.set_ylabel("Delta_DORA (M€)", fontsize=9.5, color=INK2)
axB.set_title("(B)  Contenir la propagation accroit le surcout de non-conformite",
              fontsize=10, color=INK, pad=6)
axB.invert_xaxis()
axB.grid(alpha=0.25, lw=0.5)

fig.suptitle("Delta_DORA par la cascade (Vasicek dirige) : bootstrap et canal de propagation",
             fontsize=12.3, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S7_delta_dora_cascade.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
