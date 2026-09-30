#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20 : Allocation du capital DORA aux 5 piliers - Shapley (surcout) et Euler (niveau).

Deux questions d'allocation complementaires sur le SCR_DORA multi-etats en euros.

  SHAPLEY (surcout). Le script 16b decompose le Delta_DORA en basculant UN pilier a la
  fois (Delta_k marginal), mais ces Delta_k NE SOMMENT PAS au Delta total : la cascade est
  super-additive, il reste un terme d'interaction non attribue. La valeur de Shapley est
  l'attribution PRINCIPIELLE de ce surcout : phi_j = moyenne des contributions marginales du
  pilier j sur tous les ordres d'arrivee. Fonction caracteristique
      v(S) = SCR(piliers de S en NC, autres C) - SCR(tous C),   v(emptyset) = 0,
  et par construction SUM_j phi_j = v(tous) = Delta_DORA total : Shapley redistribue donc
  EXACTEMENT l'interaction laissee de cote au 16b (Delta_k = v({k}) est le cas marginal).

  EULER (niveau). Le SCR_DORA(NC) lui-meme (le niveau de capital de l'etat non conforme)
  s'alloue aux piliers par la regle d'Euler, coherente (les parts somment au total). Version
  TVaR (exacte) : C_j = E[L_j | L_total >= VaR_99,5%], ou L_j est la perte imputee au pilier
  TOUCHE j (matrice par pilier de simulate_euro_pp, by_pillar=True). On oppose la contribution
  au CAPITAL (queue) a la contribution a la perte MOYENNE : un pilier peut peser peu en
  moyenne et beaucoup dans la queue.

CRN entre configurations (Shapley) ; memes canaux et memes choix non calibres que 16/16b
(lecture C, les 3 canaux). Ne touche ni src/ ni memoire/.

Sortie : diagnostics (Shapley phi_j vs marginal 16b ; interaction redistribuee ; classement
vs ROOT ; Euler C_j capital vs moyenne) + figure S15_allocation_shapley_euler.png. Exit 0.
"""

import os
import sys
from math import factorial
from itertools import combinations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                              # noqa: E402
from euro_cascade_model import PARAMS, var                   # noqa: E402
import scr_engine as eng                                      # noqa: E402

W = 74
PIL = eng.PIL                                                 # [1,2,3,4,5]
PIL_LAB = {1: "P1 gouvernance", 2: "P2 incidents", 3: "P3 tests",
           4: "P4 tiers", 5: "P5 partage"}
ROOT_ORDER = [1, 4, 2, 3, 5]                                  # classement ROOT (modele qualitatif)
SCENARIO = {"C": "S0_conforme", "PC": "S1_partiel", "NC": "S2_non_conforme"}

# canaux par etat (identiques a 16/16b, lecture C = les 3 canaux)
G_PROP = {"C": 0.45, "PC": 0.68, "NC": 0.90}
PU_MULT = {"C": 0.85, "PC": 1.00, "NC": 1.20}
MULT_ETAT = {st: ec.lambda_scenario("OPRISK", sc, mode="center") / PARAMS["OPRISK"]["lam_ref"]
             for st, sc in SCENARIO.items()}

# part d'amorce de chaque pilier (~ ROOT, comme simulate_euro / 16b)
_SHARE = {j: eng.LAMBDA[j] for j in PIL}
_STOT = sum(_SHARE.values())
SHARE = {j: _SHARE[j] / _STOT for j in PIL}

NY_SHAP = 60_000        # par configuration (32 configs x 2 sources) : MEME resolution et MEME
                        # graine que 16b, pour que v(tous)=Delta total retombe sur le Delta du
                        # 16b (coherence des chiffres publies). Long (~30 min), lancer en fond.
NY_EULER = 240_000      # allocation de queue : conditionnement sur 0,5 % -> beaucoup d'annees
SEED = 909              # graine MC commune entre configurations (CRN)


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def _channels(source, state_map):
    """Dicts (lam_vec, g_vec, p_u_vec) par pilier pour une configuration d'etats."""
    sp = PARAMS[source]
    lam_vec = {j: sp["lam_ref"] * SHARE[j] * MULT_ETAT[state_map[j]] for j in PIL}
    g_vec = {j: G_PROP[state_map[j]] for j in PIL}
    p_u_vec = {j: min(0.999, sp["p_u"] * PU_MULT[state_map[j]]) for j in PIL}
    return lam_vec, g_vec, p_u_vec


def scr_config(source, state_map, ny=NY_SHAP, seed=SEED):
    """SCR (VaR 99,5 %) d'une configuration {pilier -> etat}, canaux par pilier, CRN."""
    sp = PARAMS[source]
    lam_vec, g_vec, p_u_vec = _channels(source, state_map)
    rng = np.random.default_rng(seed)
    return var(ec.simulate_euro_pp(lam_vec, g_vec, sp["xi"], sp["sigma"], sp["u"],
                                   p_u_vec, sp["cap"], ny, rng))


# ============================================================ Shapley (surcout Delta_DORA)
def shapley(source):
    """phi_j, v(S) tabulee, Delta total et base tous-C. v(S)=SCR(S en NC, reste C)-SCR(tous C)."""
    n = len(PIL)
    base_c = scr_config(source, {j: "C" for j in PIL})
    v = {}
    for r in range(n + 1):
        for S in combinations(PIL, r):
            cfg = {j: ("NC" if j in S else "C") for j in PIL}
            v[frozenset(S)] = scr_config(source, cfg) - base_c
    phi = {j: 0.0 for j in PIL}
    for j in PIL:
        others = [p for p in PIL if p != j]
        for r in range(len(others) + 1):
            for S in combinations(others, r):
                fs = frozenset(S)
                wgt = factorial(len(fs)) * factorial(n - len(fs) - 1) / factorial(n)
                phi[j] += wgt * (v[fs | {j}] - v[fs])
    return phi, v, v[frozenset(PIL)], base_c


# ============================================================ Euler (niveau SCR(NC))
def euler_nc(source, ny=NY_EULER, seed=SEED):
    """Allocation TVaR du SCR(tous NC) : C_j = E[L_j | L_total>=VaR] et contribution moyenne."""
    sp = PARAMS[source]
    lam_vec, g_vec, p_u_vec = _channels(source, {j: "NC" for j in PIL})
    rng = np.random.default_rng(seed)
    M = ec.simulate_euro_pp(lam_vec, g_vec, sp["xi"], sp["sigma"], sp["u"],
                            p_u_vec, sp["cap"], ny, rng, by_pillar=True)
    Ltot = M.sum(axis=1)
    VaR = float(np.quantile(Ltot, 0.995))
    tail = Ltot >= VaR
    C_tvar = {PIL[c]: float(M[tail, c].mean()) for c in range(len(PIL))}   # somme = TVaR
    C_mean = {PIL[c]: float(M[:, c].mean()) for c in range(len(PIL))}      # perte moyenne
    return C_tvar, C_mean, VaR, float(Ltot[tail].mean())


def rang(order):
    return " > ".join(f"P{k}" for k in order)


# ============================================================ execution
titre("Allocation Shapley du surcout Delta_DORA (NC vs C) aux 5 piliers")
print("  phi_j : part principielle du pilier j dans le surcout total, interaction comprise.")
print(f"  Reference classement ROOT (qualitatif) : {rang(ROOT_ORDER)}")
# LES PARTS D'AMORCE, EN CLAIR. Le memoire compare la part de Shapley de P1 (34 %) a sa part
# d'amorce (30 %) pour montrer que la propagation DEFORME l'amorce : c'est le test non
# trivial de la section. La part d'amorce n'etait imprimee nulle part, donc la moitie de la
# comparaison n'etait pas verifiable. Ce sont des ENTREES du modele, pas des resultats :
# le vecteur ROOT normalise, ni calibre ni estime.
print("  Parts d'amorce (ROOT normalise, entree du modele) : "
      + ", ".join(f"P{j} {100*SHARE[j]:.1f} %" for j in ROOT_ORDER))
shap = {}
for source in ("OPRISK", "PRC"):
    phi, v, d_tot, base_c = shapley(source)
    shap[source] = (phi, v, d_tot)
    marg = {k: v[frozenset({k})] for k in PIL}          # Delta_k marginal (= 16b)
    somme_marg = sum(marg.values())
    inter = d_tot - somme_marg
    order_phi = sorted(PIL, key=lambda k: phi[k], reverse=True)
    print(f"\n  {source}  (base tous C = {base_c:.0f} M ; Delta total NC vs C = {d_tot:.0f} M)")
    print(f"  {'pilier':<18}{'marginal 16b':>14}{'Shapley phi':>14}{'redistrib.':>13}{'part phi':>10}")
    for k in order_phi:
        redis = phi[k] - marg[k]
        print(f"  {PIL_LAB[k]:<18}{marg[k]:>13.0f}M{phi[k]:>13.0f}M{redis:>+12.0f}M{phi[k] / d_tot:>9.1%}")
    print(f"  somme marginaux 16b = {somme_marg:.0f} M  (NE somme PAS au total)")
    print(f"  somme Shapley       = {sum(phi.values()):.0f} M  (= Delta total, exact)")
    # LA SOMME DES PARTS, CALCULEE PLUTOT QU'ECRITE. Le tableau du chapitre resultats publie
    # une ligne « somme ... 100 % » que rien ne produisait : c'etait un cent de redaction, le
    # seul nombre du tableau dont personne ne pouvait dire d'ou il venait. Or c'est aussi le
    # seul controle qui atteste que l'allocation est une PARTITION du surcout, ce qui est
    # exactement la propriete que l'on invoque pour preferer Shapley au marginal. Elle merite
    # donc d'etre calculee.
    # ET LES PARTS PASSENT A UNE DECIMALE. A l'entier, la colonne OpRisk sommait a 101 % et il
    # fallait une note de bas de tableau pour dire que l'ecart venait de l'arrondi. Une note de
    # bas de tableau qui excuse un total faux est moins bonne qu'un total juste.
    print(f"  somme des parts     = {sum(phi.values()) / d_tot:.1%}  (partition exacte)")
    print(f"  interaction super-additive redistribuee = {inter:+.0f} M")
    print(f"  classement Shapley : {rang(order_phi)}"
          f"   {'== ROOT' if order_phi == ROOT_ORDER else '!= ROOT'}")

titre("Allocation d'Euler du NIVEAU SCR_DORA(NC) : contribution au capital vs a la moyenne")
print("  TVaR exacte : C_j = E[L_j | L_total >= VaR 99,5 %]. Somme des C_j = TVaR.")
eul = {}
for source in ("OPRISK", "PRC"):
    C_tvar, C_mean, VaR, tvar = euler_nc(source)
    eul[source] = (C_tvar, C_mean, VaR, tvar)
    s_tvar, s_mean = sum(C_tvar.values()), sum(C_mean.values())
    order_cap = sorted(PIL, key=lambda k: C_tvar[k], reverse=True)
    print(f"\n  {source}  (VaR 99,5 % tous NC = {VaR:.0f} M ; TVaR = {tvar:.0f} M)")
    print(f"  {'pilier':<18}{'part CAPITAL':>14}{'part MOYENNE':>14}{'ecart':>10}")
    for k in order_cap:
        pc, pm = C_tvar[k] / s_tvar, C_mean[k] / s_mean
        print(f"  {PIL_LAB[k]:<18}{pc:>13.0%}{pm:>13.0%}{pc - pm:>+9.0%}")
    print(f"  classement capital (Euler) : {rang(order_cap)}"
          f"   {'== ROOT' if order_cap == ROOT_ORDER else '!= ROOT'}")
    print("  Lecture : l'ecart capital - moyenne mesure la contribution PROPRE a la QUEUE"
          " (au-dela du poids moyen).")


# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2 = "#1b1e30", "#223e55"
ACCENT, BLUE, GREEN, GREY = "#a6002e", "#3661ac", "#008d87", "#7d7d7d"

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime
# cette figure sur 7,27 pouces : tracee plus large, elle subissait une
# reduction qui faisait tomber ses etiquettes sous 6 points. Le rapport
# largeur sur hauteur est conserve, donc la figure occupe la meme place
# sur la page ; c'est son texte qui y prend plus de place.
fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.9, 3.52))

# panneau A : Shapley (OpRisk) - marginal 16b vs Shapley phi (interaction redistribuee)
phi, v, d_tot = shap["OPRISK"]
marg = {k: v[frozenset({k})] for k in PIL}
order = sorted(PIL, key=lambda k: phi[k], reverse=True)
ypos = np.arange(len(order))
h = 0.38
axA.barh(ypos - h / 2, [marg[k] for k in order], h, color=GREY, alpha=0.85, label="marginal 16b (ne somme pas)")
axA.barh(ypos + h / 2, [phi[k] for k in order], h,
         color=[ACCENT if k == order[0] else BLUE for k in order], alpha=0.9,
         label="Shapley (somme = Delta total)")
for yp, k in zip(ypos, order):
    axA.annotate(f"{phi[k]:.0f} M", (phi[k], yp + h / 2), textcoords="offset points",
                 xytext=(4, 0), va="center", fontsize=8, color=INK2)
axA.set_yticks(ypos)
axA.set_yticklabels([PIL_LAB[k] for k in order], fontsize=9)
axA.invert_yaxis()
axA.set_xlabel("surcout attribue Delta_DORA (M€, OpRisk)", fontsize=9.3, color=INK2)
axA.set_title("(A)  Shapley : l'interaction super-additive redistribuee",
              fontsize=10, color=INK, pad=6)
axA.legend(fontsize=8.0, frameon=False, loc="lower right")
axA.grid(alpha=0.25, lw=0.5, axis="x")

# panneau B : Euler (OpRisk) - part capital (queue) vs part moyenne
C_tvar, C_mean, VaR, tvar = eul["OPRISK"]
s_tvar, s_mean = sum(C_tvar.values()), sum(C_mean.values())
order2 = sorted(PIL, key=lambda k: C_tvar[k], reverse=True)
x = np.arange(len(order2))
wbar = 0.38
axB.bar(x - wbar / 2, [C_tvar[k] / s_tvar for k in order2], wbar, color=ACCENT, alpha=0.9,
        label="part capital (queue, TVaR)")
axB.bar(x + wbar / 2, [C_mean[k] / s_mean for k in order2], wbar, color=BLUE, alpha=0.85,
        label="part perte moyenne")
axB.set_xticks(x)
axB.set_xticklabels([f"P{k}" for k in order2], fontsize=9)
axB.set_ylabel("part de la contribution", fontsize=9.5, color=INK2)
axB.set_title("(B)  Euler : contribution au CAPITAL vs a la MOYENNE (OpRisk)",
              fontsize=10, color=INK, pad=6)
axB.legend(fontsize=8.2, frameon=False, loc="upper right")
axB.grid(alpha=0.25, lw=0.5, axis="y")

fig.tight_layout(rect=[0, 0, 1, 0.95])

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S15_allocation_shapley_euler.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
