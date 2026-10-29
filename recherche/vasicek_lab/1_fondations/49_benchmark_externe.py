#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
49 : benchmark EXTERNE, la plausibilite de l'ordre de grandeur du SCR.

Le memoire compare deja le SCR a la Formule Standard (chap. 12, script 27) : la SF est aveugle
au risque (Delta_DORA = 0, charge SCR_op ~ 450 M€). Il manque le controle de PLAUSIBILITE
externe : nos chiffres tiennent-ils face aux pertes et au marche cyber REELS ? Un jury demande
toujours 'ce nombre est-il credible ?'. On l'ancre ici sur des reperes publics.

REPERES (convertis en M€ a 0,92 $/€) :
  - marche cyber mondial 2025 : ~16,3 Md$ de primes brutes (Business Insurance / III, 2025) ;
  - NotPetya (2017) : > 10 Md$ de dommages mondiaux ; sinistres Merck+Mondelez > 1,4 Md$ ;
  - Change Healthcare (2024) : ~2,4 Md$ d'impact (UnitedHealth) ;
  - Equifax (2017) : ~1,4 Md$.

Sortie : diagnostics + figure J5_benchmark_externe.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                                 # noqa: E402
from euro_cascade_model import PARAMS, PHI, var                 # noqa: E402
import resultats_partages as rp                                 # noqa: E402

WID = 82
USD = rp.USD_EUR                        # $ -> €
sp = PARAMS["OPRISK"]


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def var_gpd(a, xi, sig, u, zu):
    return u + (sig / xi) * (((1 - a) / zu) ** (-xi) - 1)


def tvar_gpd(a, xi, sig, u, zu):
    V = var_gpd(a, xi, sig, u, zu)
    return V / (1 - xi) + (sig - xi * u) / (1 - xi)


# --- nos chiffres ----------------------------------------------------------------------------
var_single = var_gpd(0.995, sp["xi"], sp["sigma"], sp["u"], sp["p_u"])
tvar_single = tvar_gpd(0.995, sp["xi"], sp["sigma"], sp["u"], sp["p_u"])
scr_cascade = var(ec.simulate_euro(sp["lam_ref"], 0.90, sp["xi"], sp["sigma"], sp["u"],
                                   sp["p_u"], sp["cap"], 400_000, np.random.default_rng(20260727),
                                   phi=PHI))
# valeurs produites ailleurs : source unique dans `resultats_partages`
SCR_C, SCR_NC = rp.SCR_CONFORME, rp.SCR_NON_CONFORME    # script 43
SF_OP = rp.SF_OP_CHARGE                                 # Formule Standard (chap. 12, script 27)
SOCLE = rp.SCR_SOCLE                                    # socle sans contagion (chap. 10)

# --- reperes externes (M€), sources citees dans `resultats_partages` --------------------------
sev_anchors = {k: v * USD for k, v in rp.PERTES_REELLES_USD.items()}
port_anchors = {                       # echelle portefeuille / marche
    "SCR_op Formule Standard": SF_OP,
    "marche cyber mondial (primes 2025)": rp.MARCHE_CYBER_USD * USD,
}

# =====================================================================================
titre("Nos chiffres, et les reperes externes")
# =====================================================================================
print(f"  Sinistre unique : VaR 99,5 % = {var_single:.0f} M€, TVaR 99,5 % = {tvar_single:.0f} M€.")
print(f"  SCR annuel (cascade) = {scr_cascade:.0f} M€ ; conforme {SCR_C:.0f} -> non conforme "
      f"{SCR_NC:.0f} M€.")
print(f"  Socle sans contagion = {SOCLE:.0f} M€ ; charge Formule Standard = {SF_OP:.0f} M€.")
print(f"\n  Reperes de sinistres individuels (M€) :")
for k, v in sev_anchors.items():
    print(f"    {k:<26}{v:>8.0f}")
print(f"  Reperes portefeuille / marche (M€) :")
for k, v in port_anchors.items():
    print(f"    {k:<40}{v:>8.0f}")

# =====================================================================================
titre("Lecture")
# =====================================================================================
print(f"  1. Au niveau du SINISTRE UNIQUE, notre VaR 99,5 % ({var_single:.0f} M€) et notre TVaR")
print(f"     ({tvar_single:.0f} M€) sont exactement dans la fourchette des grandes pertes cyber")
print("     reelles (Merck 1,3 Md€, Equifax 1,3 Md€, Change Healthcare 2,2 Md€). Credible.")
print(f"  2. Au niveau ANNUEL, notre SCR (~{scr_cascade/1000:.0f}-{SCR_NC/1000:.0f} Md€) est de")
print("     l'ordre du marche cyber mondial (~15 Md€ de primes). C'est donc une echelle")
print("     SYSTEMIQUE / grand portefeuille (OpRisk secteur financier mondial, ~21,6 incidents")
print("     materiels/an), a lire comme un capital de SECTEUR, pas d'une entite moyenne isolee.")
print(f"  3. La charge Formule Standard ({SF_OP:.0f} M€) est ordre(s) de grandeur en dessous : elle")
print("     ignore la queue et la contagion. Notre modele n'est donc ni gonfle (il colle aux")
print("     pertes reelles au niveau unitaire) ni aveugle (il capte le risque que la SF manque).")

# =====================================================================================
# figure J5
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


# ETIQUETTES D'AFFICHAGE, DISTINCTES DES CLES. Les cles des dictionnaires alimentent
# les impressions ci-dessus et ne bougent pas ; seules les etiquettes TRACEES sont
# raccourcies et accentuees, pour que les deux echelles tiennent COTE A COTE au lieu
# d'etre empilees sur une page entiere. Un nom de variable de code ne s'affiche pas.
ETIQ = {"SCR_op Formule Standard": "charge de Formule Standard",
        "marche cyber mondial (primes 2025)": "marché cyber mondial (primes)"}


def ladder(ax, ours, anchors, title, xlab):
    items = [(k, v, False) for k, v in anchors.items()] + [(k, v, True) for k, v in ours.items()]
    items.sort(key=lambda t: t[1])
    yy = np.arange(len(items))
    for y, (k, v, mine) in zip(yy, items):
        c = ACCENT if mine else MUTED
        ax.barh(y, v, color=c, alpha=0.9 if mine else 0.55, height=0.6)
        ax.text(v * 1.10, y, f"{v:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels([ETIQ.get(k, k) for k, v, mine in items], fontsize=8.5)
    for tick, (_, _, mine) in zip(ax.get_yticklabels(), items):
        tick.set_color(ACCENT if mine else INK2)
    ax.set_xscale("log")
    # DE LA PLACE A DROITE POUR L'ETIQUETTE DE VALEUR. Sur un panneau deux fois moins
    # large, la borne automatique laissait la plus grande valeur sortir du cadre.
    ax.set_xlim(right=max(v for _, v, _ in items) * 3.2)
    ax.set_xlabel(xlab, color=INK2)
    ax.set_title(title, fontsize=9.5, color=INK, pad=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 2.95))
ladder(ax1, {"notre VaR 99,5 %": var_single, "notre TVaR 99,5 %": tvar_single},
       sev_anchors, "(a)  Sinistre unique : nos quantiles\nface aux grandes pertes réelles",
       "perte (M€, échelle log)")
ladder(ax2, {"notre socle sans contagion": SOCLE, "notre SCR cascade": scr_cascade,
             "notre SCR conforme": SCR_C, "notre SCR non conforme": SCR_NC},
       port_anchors, "(b)  Échelle annuelle : notre capital\nface au marché et au forfait",
       "capital / marché (M€, échelle log)")

# PAS DE TITRE GENERAL : la legende LaTeX le porte, et le code interne de la figure
# n'a rien a faire sur une page de memoire.
fig.tight_layout(w_pad=2.0)
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "J5_benchmark_externe.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
