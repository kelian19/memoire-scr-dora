#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
39 : P2/P3 comme couche de DETECTION, ancree sur une courbe de survie (VCDB).

P2 (gestion des incidents) et P3 (tests de resilience) ne sont pas des noeuds de severite :
ils gouvernent le TEMPS DE DETECTION. Un incident non detecte a temps escalade et devient
materiel ; detecte tot, il est contenu. Le modele actuel resume cela par un multiplicateur
grossier du taux de materialite p_u (0,85 conforme / 1,00 / 1,20 non conforme, non calibre).
Ce script remplace ce multiplicateur par une lecture de SURVIE, ancree sur donnee.

ANCRAGE (VCDB, secteur financier, n=278 decouvertes datees, script 28). La distribution du
delai de decouverte est BIMODALE :
    <= 1 jour : 47 %     |     semaines : 8 %     |     mois et + : 45 %
\og Detecte tout de suite, ou pas avant longtemps \fg. La materialite vient des incidents
NON detectes vite : on prend donc pi_slow = P(decouverte lente) = 45 % comme taux de
reference, et le taux de materialite s'ecrit  p_u(etat) = p_u_base x pi_slow(etat)/pi_slow_ref.

CE QUI EST IDENTIFIE, CE QUI EST BORNE. Le point pi_slow = 0,45 est OBSERVE. En revanche
l'ELASTICITE de la detection a la maturite P2/P3 (de combien un pilier conforme detecte-t-il
plus vite ?) n'est pas mesuree : on la BORNE par eta dans [0, eta_max], et l'on lit le SCR
en bande. On verifie enfin que le multiplicateur grossier actuel est UN point de cette bande,
donc pas arbitraire.

Sortie : diagnostics + figure Z10_detection_survie.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                                 # noqa: E402
from euro_cascade_model import PARAMS, var                      # noqa: E402

WID = 80
SRC = "OPRISK"
sp = PARAMS[SRC]
NY = 40_000
SEED = 20260721
G_NEUTRE = 0.90                    # gain neutre : on isole le canal detection
PI_REF = 0.45                      # VCDB : part de decouverte lente (mois et +), etat median
# multiplicateur grossier actuel (script 16), pour comparaison
CRUDE = {"C": 0.85, "PC": 1.00, "NC": 1.20}


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def pi_slow(state, eta):
    """Part de decouverte lente par etat, ancree a PI_REF en PC, ecartee de +/- eta.
    Bornee dans (0,1) par une logistique pour rester une probabilite."""
    base = np.log(PI_REF / (1 - PI_REF))                 # logit du point observe
    shift = {"C": -eta, "PC": 0.0, "NC": +eta}[state]
    return 1.0 / (1.0 + np.exp(-(base + shift)))


def scr_state(state, eta, seed=SEED, ny=NY):
    """SCR (VaR 99,5 %) : seul le canal detection varie (p_u module par la survie)."""
    mult = pi_slow(state, eta) / PI_REF
    p_u = min(0.999, sp["p_u"] * mult)
    rng = np.random.default_rng(seed)
    lam = sp["lam_ref"]
    return var(ec.simulate_euro(lam, G_NEUTRE, sp["xi"], sp["sigma"], sp["u"], p_u,
                                sp["cap"], ny, rng)), mult


# =====================================================================================
titre("1. La courbe de survie observee, et le multiplicateur qu'elle implique")
# =====================================================================================
print("  VCDB (finance, n=278) : <=1j 47 %  |  semaines 8 %  |  mois et + 45 %")
print("  bimodal : detecte tout de suite, ou tard. pi_slow de reference = 0,45.")
print(f"\n  {'eta':>6}{'pi_slow C':>11}{'pi_slow NC':>12}{'mult C':>9}{'mult NC':>10}")
for eta in (0.2, 0.4, 0.6, 0.8, 1.0):
    pc, pn = pi_slow("C", eta), pi_slow("NC", eta)
    print(f"  {eta:>6.1f}{pc:>11.3f}{pn:>12.3f}{pc/PI_REF:>9.2f}{pn/PI_REF:>10.2f}")
# quel eta reproduit le multiplicateur grossier ?
etas = np.linspace(0, 1.5, 300)
err = [abs(pi_slow("C", e)/PI_REF - CRUDE["C"]) + abs(pi_slow("NC", e)/PI_REF - CRUDE["NC"])
       for e in etas]
eta_crude = etas[int(np.argmin(err))]
print(f"\n  Le multiplicateur grossier (0,85 / 1,20) correspond a eta ~ {eta_crude:.2f} :")
print(f"    survie -> mult C = {pi_slow('C', eta_crude)/PI_REF:.2f}, "
      f"mult NC = {pi_slow('NC', eta_crude)/PI_REF:.2f}. Il n'est donc pas arbitraire,")
print("    c'est un point de la courbe de survie ancree sur le VCDB.")

# =====================================================================================
titre("2. Delta_DORA du canal detection, borne sur l'elasticite eta")
# =====================================================================================
print(f"  (frequence et propagation neutres : on isole ce que P2/P3 achetent par la detection)")
print(f"\n  {'eta':>6}{'SCR conforme':>15}{'SCR non conf.':>15}{'Delta detection':>17}")
etas_grid = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
scr_c_list, scr_n_list = [], []
for eta in etas_grid:
    sc, mc = scr_state("C", eta)
    sn, mn = scr_state("NC", eta)
    scr_c_list.append(sc)
    scr_n_list.append(sn)
    print(f"  {eta:>6.1f}{sc:>14.0f} M{sn:>14.0f} M{sn-sc:>15.0f} M")
scr_pc, _ = scr_state("PC", 0.0)
d_lo = min(n - c for c, n in zip(scr_c_list, scr_n_list))
d_hi = max(n - c for c, n in zip(scr_c_list, scr_n_list))
print(f"\n  SCR etat median (PC) = {scr_pc:.0f} M (point d'ancrage, ne depend pas de eta).")
print(f"  Delta_DORA du SEUL canal detection : de {d_lo:.0f} a {d_hi:.0f} M selon eta.")
print("  C'est une bande, comme le reste : le point observe (pi_slow=0,45) est fixe,")
print("  l'elasticite de la detection a la maturite ne l'est pas, donc on la borne.")

# =====================================================================================
titre("3. Verdict")
# =====================================================================================
print("  1. Le canal detection de P2/P3 n'est pas un multiplicateur pose : c'est une")
print("     courbe de survie, dont UN point (45 % de decouverte lente) est observe au VCDB.")
print(f"  2. Le multiplicateur grossier actuel (0,85/1,20) est retrouve a eta ~ {eta_crude:.2f} :")
print("     il est donc dans la bande, pas arbitraire.")
print("  3. L'elasticite de la detection a la maturite reste non identifiee : le Delta")
print(f"     detection est borne ({d_lo:.0f}-{d_hi:.0f} M), pas un point. Meme discipline que W et xi.")
print("  4. La donnee necessaire pour l'identifier : des delais de detection PAR PILIER et")
print("     PAR NIVEAU DE MATURITE, que le registre DORA pourrait porter. Meme conclusion")
print("     que partout : la limite est une specification de reporting.")

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
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Trois panneaux traces
# sur seize pouces puis imprimes sur 7,27 donnent 1,6 pouce par panneau et
# des etiquettes sous 5 points. Le rapport largeur sur hauteur est conserve.
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10.66, 3.07))

# (0) la courbe observee (bimodale)
ax0.bar([0, 1, 2], [0.47, 0.08, 0.45], color=[GREEN, MUTED, ACCENT], alpha=0.85)
ax0.set_xticks([0, 1, 2]); ax0.set_xticklabels(["≤ 1 jour", "semaines", "mois +"], fontsize=9)
ax0.set_ylabel("part des découvertes (VCDB, n=278)", color=INK2)
ax0.set_title("(a)  Détection bimodale : tôt,\nou pas avant longtemps", fontsize=11,
              color=INK, pad=8)

# (1) multiplicateur survie vs eta, avec le crude
ee = np.linspace(0, 1.3, 60)
ax1.plot(ee, [pi_slow("C", e)/PI_REF for e in ee], color=GREEN, lw=2, label="conforme")
ax1.plot(ee, [pi_slow("NC", e)/PI_REF for e in ee], color=ACCENT, lw=2, label="non conforme")
ax1.axhline(CRUDE["C"], color=GREEN, ls=":", lw=1)
ax1.axhline(CRUDE["NC"], color=ACCENT, ls=":", lw=1)
ax1.axvline(eta_crude, color=INK, ls="--", lw=1)
ax1.text(eta_crude, 1.25, f" crude\n η≈{eta_crude:.2f}", fontsize=8, color=INK)
ax1.set_xlabel("élasticité détection→maturité  $\\eta$", color=INK2)
ax1.set_ylabel("multiplicateur de $p_u$", color=INK2)
ax1.set_title("(b)  Le multiplicateur grossier est\nun point de la courbe de survie",
              fontsize=11, color=INK, pad=8)
ax1.legend(frameon=False, fontsize=8)

# (2) Delta detection borne
ax2.fill_between(etas_grid, [c for c in scr_c_list], [n for n in scr_n_list],
                 color=BLUE, alpha=0.18)
ax2.plot(etas_grid, scr_c_list, color=GREEN, lw=2, marker="o", ms=4, label="conforme")
ax2.plot(etas_grid, scr_n_list, color=ACCENT, lw=2, marker="o", ms=4, label="non conforme")
ax2.set_xlabel("élasticité  $\\eta$ (bornée)", color=INK2)
ax2.set_ylabel("SCR (VaR 99,5 %, M€)", color=INK2)
ax2.set_title("(c)  Δ détection borné sur $\\eta$", fontsize=11, color=INK, pad=8)
ax2.legend(frameon=False, fontsize=8)

for ax in (ax0, ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z10_detection_survie.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
