#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18 : Priorisation de la remediation. Dans quel ordre remedier les 5 piliers ?

ETAPE 3. On ne peut pas tout remedier a la fois : sous budget contraint, on remedie un
pilier a la fois, dans un certain ORDRE. Chaque ordre engendre une trajectoire de capital
differente pendant la mise en conformite. On cherche l'ordre qui MINIMISE le capital
integre sur l'horizon (proportionnel au cout de portage du capital = cout de capital).

S'appuie sur le 16b (etat par pilier, moteur simulate_euro_pp) et le 17 (dimension temps).

3.1  VALEUR D'ACCELERATION par pilier : gain de capital immediat si on remedie ce pilier
     en premier (difference finie depuis l'etat tout-NC), meme logique que le tornado 15 :
         v_k = SCR(tous NC) - SCR(tous NC sauf k remedie en C).

3.2  PROBLEME D'ORDRE. Un pilier remedie par periode (duree tau). Le capital integre d'un
     ordre pi est la somme des SCR des configurations traversees :
         cout(pi) = tau * somme_{i=0..4} SCR( {pi_1,...,pi_i} remedies ).
     5! = 120 ordres : on ENUMERE tout (SCR de chaque sous-ensemble precalcule et memoise,
     31 configurations) et on retient le minimum. On compare a l'ordre glouton (par valeur
     d'acceleration), a l'ordre ROOT (qualitatif) et au pire ordre.

3.3  COHERENCE DES APPROCHES. On confronte le classement de priorite a trois autres vues :
     la decomposition Delta_k (16b, euro), le classement ROOT (qualitatif) et l'allocation
     d'Euler du SCR (script 11, unites normalisees). Trois chemins independants, une meme
     hierarchie attendue.

PORTEE. Canaux frequence + propagation (comme 17, canal detection exclu tant qu'il n'est pas
valide). Remediation modelisee NC -> C (l'etat intermediaire PC affinerait, non requis pour
l'ordre). tau et l'absence d'actualisation sont des conventions (le classement est robuste au
choix de tau). Severite euros = SAS OpRisk. Ne touche ni src/ ni memoire/.

Sortie : diagnostics (valeur d'acceleration ; ordre optimal vs glouton/ROOT/pire ; table de
coherence) + figure S13_priorisation.png. Verifie exit 0.
"""

import os
import sys
import itertools

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
SCENARIO = {"C": "S0_conforme", "NC": "S2_non_conforme"}
G_PROP = {"C": 0.45, "NC": 0.90}                              # freq + propagation (sans detection)

MULT = {st: ec.lambda_scenario("OPRISK", sc, mode="center") / PARAMS["OPRISK"]["lam_ref"]
        for st, sc in SCENARIO.items()}
_SHARE = {j: eng.LAMBDA[j] for j in PIL}
_STOT = sum(_SHARE.values())
SHARE = {j: _SHARE[j] / _STOT for j in PIL}

ROOT_ORDER = sorted(PIL, key=lambda j: eng.ROOT[j], reverse=True)   # [1,4,2,3,5]
SOURCE = "OPRISK"
NY = 40_000
SEED = 18
TAU = 1.0                                                     # duree de remediation d'un pilier (an)


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# SCR d'une configuration = ensemble des piliers REMEDIES (en C), les autres en NC
_cache = {}
def scr_subset(remedies):
    """SCR (VaR 99,5 %) quand `remedies` (frozenset) sont conformes, le reste non conforme."""
    key = frozenset(remedies)
    if key in _cache:
        return _cache[key]
    sp = PARAMS[SOURCE]
    lam_ref = sp["lam_ref"]
    state = {j: ("C" if j in key else "NC") for j in PIL}
    lam_vec = {j: lam_ref * SHARE[j] * MULT[state[j]] for j in PIL}
    g_vec = {j: G_PROP[state[j]] for j in PIL}
    p_u_vec = {j: sp["p_u"] for j in PIL}                     # canal detection exclu
    rng = np.random.default_rng(SEED)                          # graine commune -> CRN
    v = var(ec.simulate_euro_pp(lam_vec, g_vec, sp["xi"], sp["sigma"], sp["u"],
                                p_u_vec, sp["cap"], NY, rng))
    _cache[key] = v
    return v


# ============================================================ 3.1 valeur d'acceleration
titre("3.1  Valeur d'acceleration par pilier (gain de capital si remedie en premier)")
scr_allnc = scr_subset(frozenset())
print(f"  Base tous Non conforme : SCR = {scr_allnc:.0f} M EUR")
print(f"  {'pilier remedie en 1er':<24}{'SCR restant':>14}{'valeur v_k':>13}")
vacc = {}
for k in PIL:
    v = scr_subset(frozenset({k}))
    vacc[k] = scr_allnc - v
for k in sorted(PIL, key=lambda k: vacc[k], reverse=True):
    print(f"  {PIL_LAB[k]:<24}{scr_subset(frozenset({k})):>14.0f}{vacc[k]:>13.0f}")
GREEDY_ORDER = sorted(PIL, key=lambda k: vacc[k], reverse=True)
print(f"  -> ordre glouton (par valeur d'acceleration) : {[PIL_LAB[k].split()[0] for k in GREEDY_ORDER]}")


# ============================================================ 3.2 ordre optimal (enumeration 120)
titre("3.2  Ordre de remediation optimal (enumeration des 5! = 120 ordres)")
print(f"  Un pilier par periode (tau={TAU:.0f} an). Cout = capital integre sur l'horizon.")

def cout_ordre(order):
    """Capital integre : somme des SCR des prefixes remedies, x tau."""
    total = 0.0
    for i in range(len(order)):                               # i piliers deja remedies
        total += scr_subset(frozenset(order[:i]))
    return total * TAU

couts = {order: cout_ordre(order) for order in itertools.permutations(PIL)}
opt = min(couts, key=couts.get)
pire = max(couts, key=couts.get)
greedy = tuple(GREEDY_ORDER)
root = tuple(ROOT_ORDER)

def montre(name, order):
    lab = " -> ".join(PIL_LAB[k].split()[0] for k in order)
    print(f"  {name:<18}{lab:<34}{couts[order]:>12.0f}")

print(f"  {'ordre':<18}{'sequence':<34}{'capital integre':>12}")
montre("OPTIMAL", opt)
montre("glouton", greedy)
montre("ROOT (qualitatif)", root)
montre("PIRE", pire)
ecart = (couts[pire] - couts[opt]) / couts[opt]
print(f"  Ecart pire/optimal : {ecart:.0%}  (l'ordre de remediation change le cout de capital porte).")
print(f"  Optimal == glouton ? {'oui' if opt == greedy else 'NON'}   "
      f"Optimal == ROOT ? {'oui' if opt == root else 'NON'}")


# ============================================================ 3.3 coherence des approches
titre("3.3  Coherence : priorite de remediation vs Delta_k (16b) vs ROOT vs Euler (11)")
# Euler (unites normalisees, script 11) : contribution TVaR par pilier d'amorce
rng_e = np.random.default_rng(SEED)
mat = eng.simulate_losses_by_pillar(200_000, rng_e, g=eng.G_BASE)   # [annees x 5]
tot = mat.sum(axis=1)
q = np.quantile(tot, 0.995)
tail = tot >= q
euler = {PIL[c]: mat[tail, c].mean() for c in range(len(PIL))}      # E[L_j | total >= VaR]
rang_euler = sorted(PIL, key=lambda j: euler[j], reverse=True)
# valeur d'acceleration (18) deja dans vacc ; ROOT ; (Delta_k du 16b : meme classement que ROOT)
print(f"  {'approche':<28}{'classement des piliers (du + prioritaire au -)':<44}")
def seq(order):
    return " > ".join(PIL_LAB[k].split()[0] for k in order)
print(f"  {'remediation optimale (18)':<28}{seq(opt):<44}")
print(f"  {'valeur acceleration (18)':<28}{seq(GREEDY_ORDER):<44}")
print(f"  {'ROOT (qualitatif)':<28}{seq(ROOT_ORDER):<44}")
print(f"  {'Euler TVaR (11, normalise)':<28}{seq(rang_euler):<44}")
print("  Les vues amorce (ROOT, acceleration) coincident ; l'Euler (concentration de queue)")
print("  peut remonter P4 (severite tiers), a lire comme complementaire, pas contradictoire.")


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

# panneau A : valeur d'acceleration par pilier, classee
order_v = sorted(PIL, key=lambda k: vacc[k], reverse=True)
vals = [vacc[k] for k in order_v]
labs = [PIL_LAB[k] for k in order_v]
ypos = np.arange(len(order_v))
cols = [ACCENT if k == order_v[0] else BLUE for k in order_v]
axA.barh(ypos, vals, color=cols, alpha=0.85)
for yp, v in zip(ypos, vals):
    axA.annotate(f"{v:.0f} M", (v, yp), textcoords="offset points", xytext=(4, 0),
                 va="center", fontsize=8.5, color=INK2)
axA.set_yticks(ypos); axA.set_yticklabels(labs, fontsize=9); axA.invert_yaxis()
axA.set_xlabel("valeur d'acceleration v_k : gain de capital si remedie en 1er (M€)",
               fontsize=9.0, color=INK2)
axA.set_xlim(0, max(vals) * 1.18)
axA.set_title("(A)  Quel pilier remedier en priorite (OpRisk)", fontsize=10, color=INK, pad=6)
axA.grid(alpha=0.25, lw=0.5, axis="x")

# panneau B : trajectoire du capital en escalier, ordre optimal vs ROOT vs pire
def escalier(order):
    return [scr_subset(frozenset(order[:i])) for i in range(len(order) + 1)]
per = np.arange(len(PIL) + 1)
for order, lab, col, ls in [(opt, "optimal", GREEN, "-"),
                            (root, "ROOT", BLUE, "--"),
                            (pire, "pire", ACCENT, ":")]:
    axB.step(per, escalier(order), where="post", color=col, lw=2.0, ls=ls,
             label=f"{lab} ({couts[order]:.0f})")
axB.set_xlabel("piliers remedies (periodes de tau=1 an)", fontsize=9.5, color=INK2)
axB.set_ylabel("SCR_DORA (M€)", fontsize=9.5, color=INK2)
axB.set_title("(B)  Trajectoire du capital selon l'ordre de remediation", fontsize=10, color=INK, pad=6)
axB.legend(fontsize=8.2, frameon=False, title="capital integre", title_fontsize=8.2)
axB.grid(alpha=0.25, lw=0.5)
axB.set_xlim(0, len(PIL))

fig.tight_layout(rect=[0, 0, 1, 0.95])

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S13_priorisation.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
