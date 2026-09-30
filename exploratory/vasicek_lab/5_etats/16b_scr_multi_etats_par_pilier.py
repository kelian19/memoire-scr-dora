#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16b : Vasicek multi-etats PAR PILIER. SCR_DORA et decomposition du Delta_DORA.

Suite du 16 (etat global) : ici chaque pilier a son PROPRE etat de conformite. C'est la
contribution de la decision B : la conformite interagit avec la cascade pilier par pilier
(un pilier non conforme propage plus et detecte moins que ses voisins conformes), au lieu
d'un etat uniforme a l'entite.

Les trois canaux du 16 deviennent PAR PILIER (dict pilier -> valeur), passes au moteur
simulate_euro_pp (agregation non reecrite, cf. 1.3) :
  - frequence : l'amorce du pilier j est modulee par l'etat de j (multiplicateur S0/S1/S2) ;
  - propagation : gain du pilier SOURCE, e_j = g(etat_j)*s_j/max_s (choix : pilier source) ;
  - detection : taux de materialite p_u du pilier TOUCHE, module par son etat.

DEUX SORTIES.
  1. Echelle de coherence : les configurations homogenes (tous C / tous PC / tous NC)
     doivent redonner les valeurs du 16 (lecture C). Controle du moteur par pilier.
  2. DECOMPOSITION du Delta_DORA : on bascule UN pilier de Conforme a Non conforme, les
     autres restant conformes, et on mesure le surcout Delta_k = SCR(k seul NC) - SCR(tous C).
     Cela chiffre la contribution propre de chaque pilier a la non-conformite, et prepare
     la priorisation de la remediation (script 18). Les Delta_k ne s'additionnent pas au
     Delta total (les cascades interagissent) : l'ecart est l'effet d'interaction, reporte.

Etat par pilier tire, ici, de facon DETERMINISTE par configuration (pas de bootstrap :
l'IC honnete du Delta agrege est deja etabli au 16/13-14). Graine MC COMMUNE entre
configurations (common random numbers : l'ecart ne vient que du changement d'etat).

PORTEE. Severite euros = SAS OpRisk. Memes choix non calibres qu'au 16 (triplet de gains,
triplet de detection, multiplicateurs de frequence). Le gain de propagation est indexe sur
le pilier SOURCE (choix de modelisation valide). Ne touche ni src/ ni memoire/.

Sortie : diagnostics (coherence homogene ; Delta_k par pilier, classement) + figure
S11_decomp_par_pilier.png. Verifie exit 0.
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
from euro_cascade_model import PARAMS, var                   # noqa: E402
import scr_engine as eng                                      # noqa: E402

W = 74
PIL = eng.PIL                                                 # [1,2,3,4,5]
PIL_LAB = {1: "P1 gouvernance", 2: "P2 incidents", 3: "P3 tests",
           4: "P4 tiers", 5: "P5 partage"}
SCENARIO = {"C": "S0_conforme", "PC": "S1_partiel", "NC": "S2_non_conforme"}

# canaux par etat (memes valeurs que le 16, lecture C = les 3 canaux)
G_PROP = {"C": 0.45, "PC": 0.68, "NC": 0.90}
PU_MULT = {"C": 0.85, "PC": 1.00, "NC": 1.20}

# multiplicateur de frequence par etat (global, source-independant)
MULT_ETAT = {st: ec.lambda_scenario("OPRISK", sc, mode="center") / PARAMS["OPRISK"]["lam_ref"]
             for st, sc in SCENARIO.items()}

# part d'amorce de chaque pilier (~ ROOT, comme simulate_euro)
_SHARE = {j: eng.LAMBDA[j] for j in PIL}
_STOT = sum(_SHARE.values())
SHARE = {j: _SHARE[j] / _STOT for j in PIL}

NY = 60_000
SEED = 909


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def scr_config(source, state_map, ny=NY, seed=SEED):
    """SCR (VaR 99,5 %) d'une configuration {pilier -> etat}, canaux par pilier, CRN."""
    sp = PARAMS[source]
    lam_ref = sp["lam_ref"]
    lam_vec = {j: lam_ref * SHARE[j] * MULT_ETAT[state_map[j]] for j in PIL}
    g_vec = {j: G_PROP[state_map[j]] for j in PIL}
    p_u_vec = {j: min(0.999, sp["p_u"] * PU_MULT[state_map[j]]) for j in PIL}
    rng = np.random.default_rng(seed)
    return var(ec.simulate_euro_pp(lam_vec, g_vec, sp["xi"], sp["sigma"], sp["u"],
                                   p_u_vec, sp["cap"], ny, rng))


def homogene(etat):
    return {j: etat for j in PIL}


# ============================================================ 1. coherence configurations homogenes
titre("Coherence : configurations homogenes (tous piliers dans le meme etat)")
print("  Doit redonner l'ordre du script 16 (lecture C, les 3 canaux).")
scr_hom = {}
for source in ("OPRISK", "PRC"):
    print(f"\n  {source} :")
    for etat in ("C", "PC", "NC"):
        v = scr_config(source, homogene(etat))
        scr_hom[(source, etat)] = v
        print(f"    tous {etat:<3} -> SCR = {v:8.0f} M EUR")


# ============================================================ 2. decomposition par pilier
titre("Decomposition du Delta_DORA : basculer UN pilier de Conforme a Non conforme")
print("  Delta_k = SCR(pilier k seul NC, autres C) - SCR(tous C).  Graine MC commune.")
decomp = {}
for source in ("OPRISK", "PRC"):
    base_c = scr_hom[(source, "C")]
    print(f"\n  {source}  (base tous C = {base_c:.0f} M EUR)")
    print(f"  {'pilier bascule en NC':<24}{'SCR':>12}{'Delta_k':>12}{'part':>9}")
    deltas = {}
    for k in PIL:
        cfg = homogene("C")
        cfg[k] = "NC"
        v = scr_config(source, cfg)
        deltas[k] = v - base_c
    decomp[source] = deltas
    d_tot = scr_hom[(source, "NC")] - base_c
    somme_k = sum(deltas.values())
    order = sorted(PIL, key=lambda k: deltas[k], reverse=True)
    for k in order:
        part = deltas[k] / somme_k if somme_k else 0.0
        print(f"  {PIL_LAB[k]:<24}{base_c + deltas[k]:>12.0f}{deltas[k]:>12.0f}{part:>8.0%}")
    print(f"  -> pilier le plus couteux en non-conformite : {PIL_LAB[order[0]]}")
    print(f"  Delta total (tous NC vs tous C) = {d_tot:.0f} M ; somme des Delta_k = {somme_k:.0f} M")
    print(f"  interaction (total - somme) = {d_tot - somme_k:+.0f} M "
          f"({'super-additif' if d_tot > somme_k else 'sous-additif'} : les cascades interagissent).")


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

# panneau A : echelle homogene (OpRisk), controle de coherence vs 16
etats = ["C", "PC", "NC"]
elab = {"C": "tous Conforme", "PC": "tous Partiel", "NC": "tous Non conforme"}
ys = [scr_hom[("OPRISK", e)] for e in etats]
axA.plot(range(len(etats)), ys, "-o", color=BLUE, lw=2.0, ms=6)
for xi_, y in zip(range(len(etats)), ys):
    axA.annotate(f"{y:.0f}", (xi_, y), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=8.5, color=BLUE)
axA.set_xticks(range(len(etats)))
axA.set_xticklabels([elab[e] for e in etats], fontsize=9)
axA.set_ylabel("SCR_DORA (M€)", fontsize=9.5, color=INK2)
axA.set_ylim(min(ys) * 0.9, max(ys) * 1.08)
axA.set_title("(A)  Configurations homogenes (OpRisk) : coherence avec le 16",
              fontsize=10, color=INK, pad=6)
axA.grid(alpha=0.25, lw=0.5)

# panneau B : decomposition par pilier (OpRisk), classee
d = decomp["OPRISK"]
order = sorted(PIL, key=lambda k: d[k], reverse=True)
vals = [d[k] for k in order]
labs = [PIL_LAB[k] for k in order]
ypos = np.arange(len(order))
cols = [ACCENT if k == order[0] else BLUE for k in order]
axB.barh(ypos, vals, color=cols, alpha=0.85)
for yp, v in zip(ypos, vals):
    axB.annotate(f"{v:.0f} M", (v, yp), textcoords="offset points", xytext=(4, 0),
                 va="center", fontsize=8.5, color=INK2)
axB.set_yticks(ypos)
axB.set_yticklabels(labs, fontsize=9)
axB.invert_yaxis()
axB.set_xlabel("Delta_k : surcout si ce pilier seul est Non conforme (M€)", fontsize=9.3, color=INK2)
axB.set_xlim(0, max(vals) * 1.18)
axB.set_title("(B)  Contribution de chaque pilier a la non-conformite (OpRisk)",
              fontsize=10, color=INK, pad=6)
axB.grid(alpha=0.25, lw=0.5, axis="x")

fig.tight_layout(rect=[0, 0, 1, 0.95])

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S11_decomp_par_pilier.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
