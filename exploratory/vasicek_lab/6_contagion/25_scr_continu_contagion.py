#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25 : SCR en CONTINU sur deux axes : conformite (score dans [0,1]) et contagion g.

Deux motivations, une seule construction.

(a) STRUCTURE D'ETAT. Les 3 etats C/PC/NC du 16 sont des coupes a 2 seuils sur une
    latente DEJA continue. Plutot que de figer un nombre d'etats (3 ? 5 ?), on rend la
    conformite CONTINUE : un score q dans [0,1] (0 = non conforme, 1 = conforme) module
    les canaux en douceur. Les etiquettes (NC/PC/C, ou une maturite a 5 niveaux) ne sont
    plus qu'une grille de lecture, pas un choix de modele. Zero seuil arbitraire.

(b) FEUILLE DE ROUTE, levier 4. W (la contagion dirigee) n'est pas calibrable : on ne
    l'estime pas, on BORNE le SCR sur sa region admissible. La force de contagion est le
    gain g (noyau e_j = g*s_j/max_s, Leontief => rho(W) <= g < 1). On balaie g dans (0,1)
    et on lit le SCR comme une FONCTION de g, jamais comme un point.

DECOUPLAGE HONNETE (identification partielle, levier 3) :
  - la CONFORMITE q pilote la frequence (calibree, S0->S2) et la detection p_u
    (canaux ancres sur donnee ou sur le 04) ;
  - la CONTAGION g est l'axe SEPARE, non calibrable, celui qu'on borne.
Ainsi ce qui est ancre et ce qui ne l'est pas ne se melangent pas dans un seul curseur.

ANCRAGE. Les valeurs aux bornes reprennent le calage du 16 :
  q=0 (NC) : frequence S2, p_u x1,20 ;   q=1 (C) : frequence S0, p_u x0,85.
  etats de reference places a (q, g) : NC (0,17 ; 0,90), PC (0,52 ; 0,68), C (0,85 ; 0,45).
Severite euros = SAS OpRisk via euro_cascade_model. Ne touche ni src/ ni memoire/.

Sortie : diagnostics + figure U_scr_continu.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                              # noqa: E402
from euro_cascade_model import PARAMS, var                   # noqa: E402

SOURCE = "OPRISK"
W = 74
NY_CURVE = 40_000
NY_GRID = 18_000
NQ, NG = 15, 15          # resolution des courbes 1D
NQ_H, NG_H = 9, 9        # resolution de la surface
G_FIX = 0.90             # contagion de base (non mitigee) pour la courbe conformite
Q_FIX = 0.50             # conformite mediane pour la courbe contagion
SEED = 20260721

# etats de reference (miroir du 16) sur les deux axes continus
REF = {"C": (0.85, 0.45), "PC": (0.52, 0.68), "NC": (0.17, 0.90)}
LABEL = {"C": "Conforme", "PC": "Partiellement", "NC": "Non conforme"}


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# canaux continus ancres sur le calage du 16
sp = PARAMS[SOURCE]
LAM_S0 = ec.lambda_scenario(SOURCE, "S0_conforme", mode="center")     # q = 1
LAM_S2 = ec.lambda_scenario(SOURCE, "S2_non_conforme", mode="center")  # q = 0


def lam_of_q(q):
    return LAM_S2 + q * (LAM_S0 - LAM_S2)          # frequence decroit avec la conformite


def pu_of_q(q):
    return min(0.999, sp["p_u"] * (1.20 - 0.35 * q))   # detection : p_u decroit avec q


def scr(q, g, ny, seed):
    rng = np.random.default_rng(seed)
    return var(ec.simulate_euro(lam_of_q(q), g, sp["xi"], sp["sigma"],
                                sp["u"], pu_of_q(q), sp["cap"], ny, rng))


# =====================================================================================
titre("Courbe 1 : SCR comme fonction continue de la CONFORMITE q (contagion g fixe)")
# =====================================================================================
qs = np.linspace(0.0, 1.0, NQ)
scr_q = np.array([scr(q, G_FIX, NY_CURVE, SEED + i) for i, q in enumerate(qs)])
print(f"  g fixe = {G_FIX}. SCR de q=0 (NC) a q=1 (C) :")
print(f"    q=0,0 : {scr_q[0]:8.0f} M   q=0,5 : {scr(0.5, G_FIX, NY_CURVE, SEED):8.0f} M"
      f"   q=1,0 : {scr_q[-1]:8.0f} M")
print(f"    gain de conformite (NC->C) : {scr_q[0]-scr_q[-1]:.0f} M de SCR en moins "
      f"(-{100*(1-scr_q[-1]/scr_q[0]):.0f} %)")
print("  Lecture : la courbe est lisse et monotone. Decouper en 3 (ou 5) niveaux n'est")
print("  qu'une grille de lecture, pas un choix de modele : le continu les subsume.")

# =====================================================================================
titre("Courbe 2 : SCR comme fonction de la CONTAGION g (conformite q fixe) -- levier 4")
# =====================================================================================
gs = np.linspace(0.05, 0.95, NG)
scr_g = np.array([scr(Q_FIX, g, NY_CURVE, SEED + 100 + i) for i, g in enumerate(gs)])
print(f"  q fixe = {Q_FIX} (conformite mediane). SCR de g=0,05 a g=0,95 :")
print(f"    g=0,05 : {scr_g[0]:8.0f} M   g=0,90 : {scr(Q_FIX, 0.90, NY_CURVE, SEED):8.0f} M")
print(f"    amplitude imputable a la contagion : x{scr_g[-1]/scr_g[0]:.2f} sur la plage admissible")
print("  On ne CALIBRE pas g : on lit l'intervalle de SCR qu'il engendre. C'est le")
print("  resultat honnete (levier 4) : 'la contagion ajoute entre X et Y au capital'.")

# =====================================================================================
titre("Surface : SCR(q, g), avec les 3 etats de reference comme points")
# =====================================================================================
qg = np.linspace(0, 1, NQ_H)
gg = np.linspace(0.1, 0.95, NG_H)
Z = np.zeros((NG_H, NQ_H))
for ig, g in enumerate(gg):
    for iq, q in enumerate(qg):
        Z[ig, iq] = scr(q, g, NY_GRID, SEED + 1000 + ig * NQ_H + iq)
print(f"  grille {NQ_H}x{NG_H} calculee. SCR min = {Z.min():.0f} M (q=1,g bas), "
      f"max = {Z.max():.0f} M (q=0,g haut).")
for e, (q, g) in REF.items():
    print(f"  etat {LABEL[e]:<14} a (q={q}, g={g}) : SCR = {scr(q, g, NY_CURVE, SEED+50):.0f} M")
print("  Les 3 etats du modele discret sont 3 POINTS sur cette surface : le continu ne")
print("  contredit pas le 16, il le generalise et rend le choix du nombre d'etats inutile.")

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
BL = ["#cde2fb", "#6491e1", "#4c79c7", "#2b559f", "#204993", "#09327a"]
SEQ = LinearSegmentedColormap.from_list("blues_seq", BL)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.8, 4.8),
                                    gridspec_kw={"width_ratios": [1, 1, 1.2]})

# (a) SCR vs conformite q, avec bandes NC/PC/C
ax1.plot(qs, scr_q, color=BL[4], lw=2.4, zorder=3)
bands = [(0.0, 0.35, "NC", BL[0]), (0.35, 0.70, "PC", BL[1]), (0.70, 1.0, "C", BL[2])]
for lo, hi, lab, col in bands:
    ax1.axvspan(lo, hi, color=col, alpha=0.18)
    ax1.text((lo + hi) / 2, scr_q.max() * 0.96, lab, ha="center", fontsize=8.5, color=INK2)
ax1.set_xlabel("score de conformité $q$  (0 = NC, 1 = C)", color=INK2)
ax1.set_ylabel("SCR (VaR 99,5 %, M€)", color=INK2)
ax1.set_title(f"(a)  SCR continu en conformité (g = {G_FIX})", fontsize=11, color=INK, pad=8)
ax1.text(0.5, scr_q.min() + (scr_q.max()-scr_q.min())*0.12,
         "3 ou 5 niveaux = une grille\nde lecture, pas un modèle", ha="center",
         fontsize=8.2, color=MUTED, style="italic")
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# (b) SCR vs contagion g
ax2.plot(gs, scr_g, color=ACCENT, lw=2.4, zorder=3)
ax2.fill_between(gs, scr_g[0], scr_g, color=ACCENT, alpha=0.10)
for e, (q, g) in REF.items():
    ax2.axvline(g, color=MUTED, lw=0.9, ls=":")
    ax2.text(g, scr_g.min(), f" {e}", fontsize=7.6, color=MUTED, rotation=90,
             va="bottom", ha="left")
ax2.set_xlabel("force de contagion $g$  (amplitude de $W$, $\\rho(W)\\leq g$)", color=INK2)
ax2.set_ylabel("SCR (VaR 99,5 %, M€)", color=INK2)
ax2.set_title(f"(b)  SCR borné sur $W$, non calibré (q = {Q_FIX})", fontsize=11,
              color=INK, pad=8)
ax2.text(0.06, scr_g.max() * 0.96, f"x{scr_g[-1]/scr_g[0]:.1f} sur la plage",
         fontsize=8.4, color=ACCENT, va="top")
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# (c) surface SCR(q, g)
im = ax3.imshow(Z, origin="lower", aspect="auto", cmap=SEQ,
                extent=[qg[0], qg[-1], gg[0], gg[-1]])
cs = ax3.contour(qg, gg, Z, colors="#ffffff", linewidths=0.6, alpha=0.5)
ax3.clabel(cs, fmt="%.0f", fontsize=6.5)
for e, (q, g) in REF.items():
    ax3.scatter([q], [g], s=55, color=ACCENT, edgecolor="#ffffff", zorder=5)
    ax3.text(q, g + 0.03, e, ha="center", fontsize=8.5, color=ACCENT, fontweight="bold")
ax3.set_xlabel("score de conformité $q$", color=INK2)
ax3.set_ylabel("force de contagion $g$", color=INK2)
ax3.set_title("(c)  Le SCR comme surface ; 3 états = 3 points", fontsize=11,
              color=INK, pad=8)
cb = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.03)
cb.set_label("SCR (M€)", fontsize=9, color=INK2)
cb.outline.set_edgecolor(GRID)

fig.suptitle("U : conformité continue et contagion bornée, plutôt que des états figés",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "U_scr_continu.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
