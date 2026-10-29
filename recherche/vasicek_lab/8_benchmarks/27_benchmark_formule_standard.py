#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27 : benchmark Formule Standard, et synthese des trois cadres d'agregation.

Complete le 23 (copule). Le 23 a montre ce qu'une copule symetrique ne peut pas faire ;
ici on ajoute la FORMULE STANDARD (Solvabilite II, risque operationnel, art. 204 du
reglement delegue), puis on synthetise les trois cadres sur UNE question :

    combien chaque cadre laisse-t-il la CONFORMITE DORA bouger le capital ?

  - Formule Standard : SCR_op = min(0,3 x BSCR ; Op) + 0,25 x Exp_ul, avec
    Op = max(0,03 x primes ; 0,03 x provisions). C'est un FACTEUR sur le volume. Il ne
    depend NI du profil de risque cyber NI de la conformite : SCR_op(conforme) =
    SCR_op(non conforme). Reponse a DORA = ZERO, par construction.
  - LDA-copule : exprime les canaux frequence et detection (marges par etat), mais PAS la
    propagation (aucun parametre de marge ne porte g). Reponse PARTIELLE (script 23).
  - Cascade dirigee : exprime les trois canaux. Reponse COMPLETE.

C'est le resultat de synthese de la phase 4 : la valeur de la cascade se mesure en
'fraction de l'effet DORA que le cadre sait exprimer'. La Formule Standard en exprime 0.

Perimetre OpRisk, memes canaux et calage que 16b/23/24. La copule est recalculee ICI sur
la meme base que la cascade (CRN) pour une comparaison coherente. Ne touche ni src/ ni
memoire/. Sortie : diagnostics + figure W_benchmark_sf.png.
"""

import os
import sys

import numpy as np
from scipy.stats import norm, rankdata
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                              # noqa: E402
from euro_cascade_model import PARAMS, var                   # noqa: E402
from src.aggregation.lda import simulate_remediation_severity  # noqa: E402
import scr_engine as eng                                     # noqa: E402

W = 74
PIL = eng.PIL
SCENARIO = {"C": "S0_conforme", "NC": "S2_non_conforme"}
G_PROP = {"C": 0.45, "NC": 0.90}
PU_MULT = {"C": 0.85, "NC": 1.20}
MULT = {st: ec.lambda_scenario("OPRISK", sc, mode="center") / PARAMS["OPRISK"]["lam_ref"]
        for st, sc in SCENARIO.items()}
_S = {j: eng.LAMBDA[j] for j in PIL}
SHARE = {j: _S[j] / sum(_S.values()) for j in PIL}
NY, SEED = 60_000, 909
PHI = ec.PHI
sp = PARAMS["OPRISK"]

# facteurs de la Formule Standard (reglement delegue UE 2015/35, art. 204, risque op.)
SF_FACTOR = 0.03          # facteur non-vie sur primes acquises / provisions
SF_CAP_BSCR = 0.30        # SCR_op plafonne a 30 % du BSCR


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# ============================================================ cascade C / NC
def cascade_annual(state_map, by_pillar=False):
    lam_vec = {j: sp["lam_ref"] * SHARE[j] * MULT[state_map[j]] for j in PIL}
    g_vec = {j: G_PROP[state_map[j]] for j in PIL}
    p_u_vec = {j: min(0.999, sp["p_u"] * PU_MULT[state_map[j]]) for j in PIL}
    rng = np.random.default_rng(SEED)
    return ec.simulate_euro_pp(lam_vec, g_vec, sp["xi"], sp["sigma"], sp["u"],
                               p_u_vec, sp["cap"], NY, rng, by_pillar=by_pillar)


titre("Cascade dirigee : SCR conforme vs non conforme")
M0 = cascade_annual({j: "C" for j in PIL}, by_pillar=True)
casc_C = M0.sum(axis=1)
casc_NC = cascade_annual({j: "NC" for j in PIL})
scr_casc = {"C": var(casc_C), "NC": var(casc_NC)}
d_casc = scr_casc["NC"] - scr_casc["C"]
print(f"  SCR cascade  C = {scr_casc['C']:.0f} M   NC = {scr_casc['NC']:.0f} M   "
      f"Delta = {d_casc:.0f} M  (+{100*d_casc/scr_casc['C']:.0f} %)")


# ============================================================ LDA-copule (meme base, CRN)
def marge_lda(j, etat):
    lam = sp["lam_ref"] * SHARE[j] * MULT[etat]
    p_u = min(0.999, sp["p_u"] * PU_MULT[etat])
    rng = np.random.default_rng(10_000 + 100 * j + (0 if etat == "C" else 1))
    r = lam / (PHI - 1.0)
    p = r / (r + lam)
    counts = rng.negative_binomial(r, p, size=NY)
    annual = np.zeros(NY)
    T = int(counts.sum())
    if T:
        yrs = np.repeat(np.arange(NY), counts)
        sev = simulate_remediation_severity(T, sp["xi"], sp["sigma"], sp["u"], p_u,
                                            sp["cap"], rng)
        np.add.at(annual, yrs, sev)
    return annual


def couple(margins, U):
    tot = np.zeros(U.shape[0])
    for c, j in enumerate(PIL):
        tot += np.quantile(margins[j], U[:, c], method="linear")
    return tot


titre("LDA-copule gaussienne (marges par etat, dependance calee sur la cascade)")
ranks = np.column_stack([rankdata(M0[:, c]) for c in range(len(PIL))])
R = 2.0 * np.sin(np.pi * np.corrcoef(ranks, rowvar=False) / 6.0)
np.fill_diagonal(R, 1.0)
U = norm.cdf(np.random.default_rng(777).standard_normal((NY, len(PIL))) @ np.linalg.cholesky(R).T)
marg = {(j, e): marge_lda(j, e) for j in PIL for e in ("C", "NC")}
cop_C = couple({j: marg[(j, "C")] for j in PIL}, U)
cop_NC = couple({j: marg[(j, "NC")] for j in PIL}, U)
scr_cop = {"C": var(cop_C), "NC": var(cop_NC)}
d_cop = scr_cop["NC"] - scr_cop["C"]
print(f"  SCR copule   C = {scr_cop['C']:.0f} M   NC = {scr_cop['NC']:.0f} M   "
      f"Delta = {d_cop:.0f} M  (+{100*d_cop/scr_cop['C']:.0f} %)")
print("  la propagation (g : 0,45 -> 0,90) n'a aucun parametre de marge : Delta sous-estime.")


# ============================================================ Formule Standard
def scr_op_sf(primes, provisions, bscr):
    op = max(SF_FACTOR * primes, SF_FACTOR * max(0.0, provisions))
    return min(SF_CAP_BSCR * bscr, op)


titre("Formule Standard (risque operationnel, art. 204)")
print("  SCR_op = min(0,3 x BSCR ; 0,03 x max(primes, provisions)).")
print("  Aucun terme ne depend du profil cyber ni de la conformite DORA :")
print("  SCR_op(conforme) = SCR_op(non conforme). Reponse a DORA = 0, par construction.")
d_sf = 0.0

# ============================================================ synthese
titre("SYNTHESE : fraction de l'effet DORA que chaque cadre sait exprimer")
print(f"  {'cadre':<22}{'Delta capital C->NC':>22}{'part de l effet cascade':>26}")
print(f"  {'Formule Standard':<22}{d_sf:>20.0f} M{'0 %':>26}")
print(f"  {'LDA-copule':<22}{d_cop:>20.0f} M{100*d_cop/d_casc:>24.0f} %")
print(f"  {'Cascade dirigee':<22}{d_casc:>20.0f} M{'100 % (reference)':>26}")
print("\n  Verdict phase 4 : la Formule Standard est structurellement AVEUGLE a la")
print("  conformite (facteur sur le volume) ; la copule en voit une partie (frequence,")
print("  detection) mais rate la propagation ; seule la cascade exprime l'effet complet.")
print("  La valeur du modele se mesure donc en effet DORA rendu visible, pas en un ecart")
print("  de SCR a interpreter dans l'absolu.")

# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 10,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
})
INK, INK2, MUTED, GRID = "#1b1e30", "#223e55", "#595959", "#f2f2f2"
ACCENT = "#a6002e"
BL = ["#7baafd", "#4c79c7", "#204993"]

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Trois panneaux traces
# sur seize pouces puis imprimes sur 7,27 donnent 1,6 pouce par panneau et
# des etiquettes sous 5 points. Le rapport largeur sur hauteur est conserve.
# Ramenee une seconde fois a 8,8 pouces : a 10,66 le facteur d'impression
# valait 0,65, donc une etiquette de 8,6 points s'imprimait a 5,6.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.8, 3.45),
                                    gridspec_kw={"width_ratios": [1.05, 1, 1.1]})

# (a) reponse du capital a la non-conformite DORA, par cadre
frames = ["Formule\nStandard", "LDA-copule", "Cascade\ndirigée"]
deltas = [d_sf, d_cop, d_casc]
cols = [MUTED, BL[1], ACCENT]
ax1.bar(range(3), deltas, color=cols, edgecolor="#fcfcfb", width=0.68)
for i, d in enumerate(deltas):
    frac = "0 % de l'effet" if i == 0 else f"{100*d/d_casc:.0f} % de l'effet"
    ax1.text(i, d + max(deltas) * 0.03, f"{d:.0f} M€\n{frac}", ha="center", va="bottom",
             fontsize=9, color=INK2)
ax1.set_xticks(range(3)); ax1.set_xticklabels(frames, fontsize=9.5)
ax1.set_ylabel("effet DORA sur le capital (M€)", color=INK2, fontsize=10)
ax1.set_ylim(0, max(deltas) * 1.32)
ax1.set_title("(a)  Effet DORA exprimé par chaque cadre", fontsize=10.5, color=INK, pad=8)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# (b) la Formule Standard est aveugle : C = NC
prem_demo = 200_000.0        # primes notionnelles (M€) d'une grande entite
bscr_demo = 0.5 * prem_demo
sf_c = scr_op_sf(prem_demo, prem_demo, bscr_demo)
sf_nc = scr_op_sf(prem_demo, prem_demo, bscr_demo)     # identique : aucun canal DORA
groups = ["Formule\nStandard", "Cascade\ndirigée"]
x = np.arange(2)
h = 0.36
ax2.bar(x - h / 2, [sf_c, scr_casc["C"]], h, color=BL[0], edgecolor="#fcfcfb", label="conforme")
ax2.bar(x + h / 2, [sf_nc, scr_casc["NC"]], h, color=ACCENT, edgecolor="#fcfcfb",
        label="non conforme")
ax2.set_xticks(x); ax2.set_xticklabels(groups, fontsize=9)
ax2.set_ylabel("SCR (M€)", color=INK2)
ax2.legend(frameon=False, fontsize=8.5)
ax2.set_title("(b)  La Formule Standard ne bouge pas ;\nla cascade double", fontsize=11,
              color=INK, pad=8)
ax2.annotate("identique", (0, sf_c), textcoords="offset points", xytext=(0, 6),
             ha="center", fontsize=8, color=MUTED, style="italic")
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# (c) deux axes orthogonaux : SF ~ volume, cascade ~ risque
prem = np.linspace(50_000, 600_000, 200)
sf_line = np.minimum(SF_CAP_BSCR * (0.5 * prem), SF_FACTOR * prem)
ax3.plot(prem / 1000, sf_line, color=MUTED, lw=2.2, label="Formule Standard ($\\propto$ primes)")
ax3.axhline(scr_casc["C"], color=BL[2], lw=2.0, ls="-", label="cascade cyber, conforme")
ax3.axhline(scr_casc["NC"], color=ACCENT, lw=2.0, ls="-", label="cascade cyber, non conforme")
ax3.fill_between(prem / 1000, scr_casc["C"], scr_casc["NC"], color=ACCENT, alpha=0.08)
ax3.set_xlabel("primes acquises  (Md€)", color=INK2)
ax3.set_ylabel("capital (M€)", color=INK2)
ax3.legend(frameon=False, fontsize=8.0, loc="upper left")
ax3.set_title("(c)  Axes orthogonaux : volume vs profil de risque", fontsize=11,
              color=INK, pad=8)
ax3.text(0.98, 0.05, "la FS suit la taille,\nla cascade suit le risque",
         transform=ax3.transAxes, ha="right", fontsize=8.2, color=INK2, style="italic")
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.93])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "W_benchmark_sf.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nfigure ecrite :", path)
