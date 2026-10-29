#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
48 : la bande de MODELE, troisieme etage d'incertitude du SCR.

Le memoire porte deja deux bandes : incertitude de PARAMETRE (bootstrap sur xi, script 46) et
incertitude d'IDENTIFICATION (bornes sur la direction W, chap. 10). Il en manque une troisieme,
celle que soulevent la remarque de Caroline et la question 'chaque modele peut-il etre remis en
cause par une autre methode ?' : l'incertitude de MODELE, le choix de la FAMILLE elle-meme.

On la chiffre en recalculant le SCR sous des familles concurrentes, sur les deux axes ou le
memoire est le plus expose :

  AXE 1 - STRUCTURE DE DEPENDANCE (a marges par pilier FIXEES) :
    independance, copule gaussienne, copule de Gumbel (queue lourde), comonotone (borne de
    Frechet), formule standard Solvabilite II (variance-covariance). C'est l'alternative directe
    a la cascade dirigee : un praticien agregerait par copule ou par formule standard.

  AXE 2 - FAMILLE DE SEVERITE (agregation en independance) :
    GPD-OpRisk (xi~0,60), GPD-PRC (xi~1,03, plafonnee), lognormale (queue fine). C'est le choix
    de loi de queue, celui qui pese le plus sur un quantile 99,5 %.

Message attendu : le NIVEAU se deplace dans une bande selon la famille, mais la structure
QUALITATIVE (l'ordre d'amorce ROOT, fixe) ne bouge pas. La cascade dirigee, notre modele, est un
mecanisme GENERATIF (elle modifie les marges) : on la place en repere, pas comme une copule sur
les memes marges.

Sortie : diagnostics + figure J4_bande_modele.png.
"""

import os
import sys

import numpy as np
from scipy import stats
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import euro_cascade_model as ec                                 # noqa: E402
from euro_cascade_model import PARAMS, PHI, var                 # noqa: E402
from src.aggregation.lda import simulate_remediation_severity   # noqa: E402
from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR  # noqa: E402
import scr_engine as eng                                        # noqa: E402

WID = 82
NY = 400_000
SEED = 20260727
TAU = 0.444                             # tau de Kendall commun (Gumbel theta=1,8, config)
THETA = 1.8
RHO_G = float(np.sin(np.pi * TAU / 2))  # correlation gaussienne de meme tau
RHO_SF = 0.50                           # correlation lineaire de la formule standard (illustratif)
PIL = eng.PIL


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# ---- copules exchangeables (reprises du script 42) ------------------------------------------
def gaussian_uniforms(n, d, rho, rng):
    Z = rng.standard_normal(n)
    eps = rng.standard_normal((n, d))
    return stats.norm.cdf(np.sqrt(rho) * Z[:, None] + np.sqrt(1 - rho) * eps)


def _rstable_pos(alpha, n, rng):
    U = np.pi * rng.random(n); W = rng.exponential(1.0, n)
    a = (np.sin((1 - alpha) * U) * np.sin(alpha * U) ** (alpha / (1 - alpha))
         / np.sin(U) ** (1 / (1 - alpha)))
    return (a / W) ** ((1 - alpha) / alpha)


def gumbel_uniforms(n, d, theta, rng):
    alpha = 1.0 / theta
    V = _rstable_pos(alpha, n, rng)
    E = rng.exponential(1.0, (n, d))
    return np.exp(-(E / V[:, None]) ** alpha)


# ---- marges par pilier : modele collectif independant (frequence NB x severite) -------------
w = np.array([eng.LAMBDA[j] for j in PIL], float)
w = w / w.sum()
LAM = PARAMS["OPRISK"]["lam_ref"]
lam_j = LAM * w


def gpd_sampler(p):
    return lambda n, rng: simulate_remediation_severity(n, p["xi"], p["sigma"], p["u"],
                                                        p["p_u"], p["cap"], rng)


# lognormale ajustee aux pertes cyber x finance (famille a queue fine)
_loss = (filter_finance(filter_cyber(load_clean(
    os.path.join(REPO, "data", "raw", "SAS_OpRisk_Global_Data_June_2026.xlsx"))))["loss"].to_numpy()
    * USD_EUR)
_s, _, _sc = stats.lognorm.fit(_loss, floc=0)


def logn_sampler(n, rng):
    return _sc * np.exp(_s * rng.standard_normal(n))


def pillar_losses(sev_sampler, ny, rng):
    """Matrice (ny, 5) des pertes annuelles par pilier, marges INDEPENDANTES."""
    L = np.zeros((ny, 5))
    for c in range(5):
        lj = lam_j[c]
        r = lj / (PHI - 1.0)
        counts = rng.negative_binomial(r, r / (r + lj), size=ny)
        T = int(counts.sum())
        yof = np.repeat(np.arange(ny), counts)
        sev = sev_sampler(T, rng)
        L[:, c] = np.bincount(yof, weights=sev, minlength=ny)
    return L


rng = np.random.default_rng(SEED)

# =====================================================================================
titre("AXE 1 : structure de dependance (marges OpRisk par pilier fixees)")
# =====================================================================================
L = pillar_losses(gpd_sampler(PARAMS["OPRISK"]), NY, rng)
Lsort = [np.sort(L[:, c]) for c in range(5)]
VaR_j = np.array([var(L[:, c]) for c in range(5)])


def agg_copula(U):
    tot = np.zeros(len(U))
    for c in range(5):
        idx = np.clip((U[:, c] * (NY - 1)).astype(int), 0, NY - 1)
        tot += Lsort[c][idx]
    return var(tot)


dep = {}
dep["independance"] = var(L.sum(axis=1))
dep["copule gaussienne"] = agg_copula(gaussian_uniforms(NY, 5, RHO_G, rng))
dep["copule de Gumbel"] = agg_copula(gumbel_uniforms(NY, 5, THETA, rng))
dep["comonotone (Frechet)"] = float(VaR_j.sum())
CORR = np.full((5, 5), RHO_SF); np.fill_diagonal(CORR, 1.0)
dep["formule standard SII"] = float(np.sqrt(VaR_j @ CORR @ VaR_j))
print(f"  SCR (VaR 99,5 %) selon la structure de dependance, marges identiques :")
for k, v in dep.items():
    print(f"    {k:<26}{v:>9.0f} M€")
dlo, dhi = min(dep.values()), max(dep.values())
print(f"  Bande de dependance : [{dlo:.0f} ; {dhi:.0f}] M€ (facteur {dhi/dlo:.1f}).")
# LA MEME BANDE EN MILLIARDS. La legende de la figure la cite en Md€, ce script ne l'imprimait
# qu'en M€ : le harnais ne pouvait apparier ni l'une ni l'autre. On imprime les deux formes,
# comme le script 67 le fait pour les parts de vecteur.
print(f"  soit, en milliards  : [{dlo/1000:.1f} ; {dhi/1000:.1f}] Md€")

# repere : notre modele generatif (cascade dirigee)
sp = PARAMS["OPRISK"]
scr_cascade = var(ec.simulate_euro(LAM, 0.90, sp["xi"], sp["sigma"], sp["u"], sp["p_u"],
                                   sp["cap"], NY, np.random.default_rng(SEED), phi=PHI))
print(f"  Repere : notre cascade dirigee (mecanisme generatif) = {scr_cascade:.0f} M€.")

# =====================================================================================
titre("AXE 2 : famille de severite (agregation en independance)")
# =====================================================================================
# axe FAMILLE de queue, a plafond fixe (non plafonne) : la comparaison est propre
sp_heavy = dict(PARAMS["OPRISK"]); sp_heavy["xi"] = 0.90        # estimation q90 (chap. 6), plus lourde
sev = {}
sev["lognormale (queue fine)"] = var(pillar_losses(logn_sampler, NY,
                                                   np.random.default_rng(SEED + 3)).sum(axis=1))
sev["GPD OpRisk (xi=0,60)"] = var(pillar_losses(gpd_sampler(PARAMS["OPRISK"]), NY,
                                                np.random.default_rng(SEED + 1)).sum(axis=1))
sev["GPD lourde (xi=0,90)"] = var(pillar_losses(gpd_sampler(sp_heavy), NY,
                                                np.random.default_rng(SEED + 2)).sum(axis=1))
print(f"  SCR (VaR 99,5 %) selon la famille de queue, agregation en independance :")
for k, v in sev.items():
    print(f"    {k:<26}{v:>9.0f} M€")
slo, shi = min(sev.values()), max(sev.values())
print(f"  Bande de severite : [{slo:.0f} ; {shi:.0f}] M€ (facteur {shi/slo:.1f}). La queue lourde")
print("  fait exploser le quantile : c'est le choix de FAMILLE qui pese le plus, de loin.")

# le plafond de reassurance comme MITIGATION separee (a ne pas confondre avec la famille)
sp_heavy_cap = dict(sp_heavy); sp_heavy_cap["cap"] = 40.0
scr_heavy_cap = var(pillar_losses(gpd_sampler(sp_heavy_cap), NY,
                                  np.random.default_rng(SEED + 2)).sum(axis=1))
print(f"  Levier distinct : plafonner la severite a 40 M€ (reassurance) ramene la famille lourde")
print(f"  de {sev['GPD lourde (xi=0,90)']:.0f} a {scr_heavy_cap:.0f} M€ — c'est une reponse au risque de")
print("  queue, pas le risque lui-meme (d'ou le plafond sur la source PRC, xi>1, du memoire).")

# ordre d'amorce (structure qualitative) : invariant par construction (ROOT fixe)
rank = np.argsort(-w)
print(f"\n  Ordre des piliers par propension d'amorce (ROOT, w) : "
      f"{' > '.join('P%d' % PIL[i] for i in rank)} — fixe, invariant a la famille.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
allv = list(dep.values()) + list(sev.values())
lo, hi = min(allv), max(allv)
print(f"  1. Le SCR se deplace dans une BANDE DE MODELE [{lo:.0f} ; {hi:.0f}] M€ (facteur {hi/lo:.1f})")
print("     selon la famille choisie : c'est le troisieme etage d'incertitude, apres le")
print("     parametre (bootstrap) et l'identification (bornes W).")
print(f"  2. Axe dependance : facteur {dhi/dlo:.1f} (independance -> comonotone). Axe severite :")
print(f"     facteur {shi/slo:.1f} (lognormale -> GPD lourde xi=0,90). La queue lourde domine.")
print("  3. La structure QUALITATIVE (ordre d'amorce ROOT) ne bouge pas : le niveau est")
print("     illustratif, l'ordre resiste. C'est, chiffree, la these du memoire.")
print("  4. Reponse a la question : oui, chaque brique a des alternatives ; on ne pretend pas")
print("     avoir LA methode, on borne l'effet du choix et on le lit en bande, comme le reste.")

# =====================================================================================
# figure J4
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 10.5,
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
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.69, 2.94),
                                    gridspec_kw={"width_ratios": [1.15, 1, 1]})

# (a) axe dependance
names = list(dep.keys()); vals = list(dep.values())
yp = np.arange(len(names))[::-1]
ax1.barh(yp, vals, color=BLUE, alpha=0.85)
ax1.axvline(scr_cascade, color=ACCENT, lw=1.8, ls="--")
ax1.set_ylim(-0.75, len(names) - 0.25)
ax1.text(scr_cascade, len(names) - 0.45, f" cascade {scr_cascade:.0f}", color=ACCENT,
         fontsize=8, va="center")
for y_, v in zip(yp, vals):
    ax1.text(v, y_, f" {v:.0f}", va="center", fontsize=8.5, color=INK2, zorder=5,
             bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=0.8))
ax1.set_yticks(yp); ax1.set_yticklabels(names, fontsize=8.5)
ax1.set_xlabel("SCR (VaR 99,5 %, M€)", color=INK2)
ax1.set_title("(a)  Axe dépendance\n(marges fixées)", fontsize=11, color=INK, pad=6)

# (b) axe severite
names2 = list(sev.keys()); vals2 = list(sev.values())
yp2 = np.arange(len(names2))[::-1]
cols2 = ["#7baafd", "#4c79c7", "#204993"]        # gradient : queue de plus en plus lourde
ax2.barh(yp2, vals2, color=cols2, alpha=0.9)
for y_, v in zip(yp2, vals2):
    ax2.text(v, y_, f" {v:.0f}", va="center", fontsize=8.5, color=INK2)
ax2.set_yticks(yp2); ax2.set_yticklabels(names2, fontsize=8.5)
ax2.set_xlabel("SCR (VaR 99,5 %, M€)", color=INK2)
ax2.set_title("(b)  Axe famille de sévérité\n(la queue lourde domine)", fontsize=11, color=INK, pad=6)

# (c) les trois etages d'incertitude
ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.set_title("(c)  Les trois étages d'incertitude", fontsize=11, color=INK, pad=6)
floors = [
    ("Paramètre", "bootstrap sur ξ", "VaR ×2,6", GREEN),
    ("Identification", "bornes sur la direction W", "cascade [6,9 ; 8,7] Md (chap. 10)", BLUE),
    ("Modèle", "famille de loi / dépendance", f"[{lo:.0f} ; {hi:.0f}] M€, ×{hi/lo:.1f} (ici)", ACCENT),
]
yy = 0.94
for name, what, num, col in floors:
    ax3.add_patch(plt.Rectangle((0.03, yy - 0.025), 0.05, 0.05, color=col, alpha=0.8))
    ax3.text(0.12, yy, name, fontsize=10.5, color=col, fontweight="bold", va="center")
    ax3.text(0.12, yy - 0.075, what, fontsize=8.5, color=INK2, va="center")
    ax3.text(0.12, yy - 0.140, num, fontsize=8, color=MUTED, va="center", style="italic")
    yy -= 0.27
ax3.text(0.03, 0.12, "Le SCR se cite en intervalle, jamais en point :\nles trois étages se cumulent.",
         fontsize=8.5, color=INK, style="italic", va="top")

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(w_pad=1.8)
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "J4_bande_modele.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
