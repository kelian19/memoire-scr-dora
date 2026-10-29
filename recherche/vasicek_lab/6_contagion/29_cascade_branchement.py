#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
29 : la cascade comme BRANCHEMENT MULTITYPE, reconciliee avec le W du Vasicek.

Jusqu'ici la cascade simulee etait une MARCHE auto-evitante (un seul successeur par pas :
une chaine), alors que le W du Vasicek, via (I-W)^{-1} = I + W + W^2 + ..., est la moyenne
d'un BRANCHEMENT (multi-descendance, un arbre). Incoherence. On implemente le vrai
branchement multitype (scr_engine, mode='branching') et on le confronte a la marche et au W.

Construction (cascade independante) : un pilier tombe j infecte chaque autre pilier k
INDEPENDAMMENT avec proba p_jk = g*TRANS[j][k]/max_s. Alors :
  - nombre moyen de descendants directs de j = e_j = g*s_j/max_s  (IDENTIQUE a la marche) ;
  - matrice moyenne des descendants = W = g*TRANS/max_s ;
  - rho(W) <= g < 1 : sous-critique (progeniture finie), MEME condition que le Vasicek ;
  - (I-W)^{-1} = esperance de la progeniture (envelope lineaire, non auto-evitante).
La marche et le branchement partagent donc e_j et la sous-criticite ; ils different par la
STRUCTURE : chaine (>=1 successeur) vs arbre (plusieurs possibles). Le branchement est
l'objet stochastique dont le W du Vasicek est l'esperance : les deux representations sont
enfin coherentes.

Perimetre OpRisk (euro_cascade_model). Ne touche pas les defauts (mode='walk' partout
ailleurs). Sortie : diagnostics + figure Y_cascade_branchement.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import scr_engine as eng                                     # noqa: E402
import euro_cascade_model as ec                              # noqa: E402
from euro_cascade_model import PARAMS, var                   # noqa: E402
from cascade_model import ROOT, TRANS                        # noqa: E402

W_ = 74
PIL = eng.PIL
COL = {j: c for c, j in enumerate(PIL)}
G = 0.90                                                     # gain de base
MAXS = max(sum(TRANS[j].values()) for j in PIL)


def titre(s):
    print("\n" + "=" * W_ + f"\n{s}\n" + "=" * W_)


# =====================================================================================
titre("Sous-criticite : matrice moyenne W et rayon spectral")
# =====================================================================================
# A[k][j] = p_jk = esperance de descendants de type k depuis un parent de type j
A = np.zeros((len(PIL), len(PIL)))
for j in PIL:
    for k in PIL:
        if k != j:
            A[COL[k], COL[j]] = G * TRANS[j].get(k, 0.0) / MAXS
rho = max(abs(np.linalg.eigvals(A)))
e_j = {j: G * sum(TRANS[j].values()) / MAXS for j in PIL}
print(f"  gain g = {G} | max_s = {MAXS:.2f}")
print("  nombre moyen de descendants directs e_j = g*s_j/max_s :")
for j in PIL:
    print(f"     P{j} : {e_j[j]:.3f}")
print(f"  rayon spectral rho(W) = {rho:.3f}  ({'sous-critique < 1' if rho < 1 else 'CRITIQUE'})")
print("  => meme W et meme condition de stabilite que le Vasicek dirige.")

# envelope lineaire (esperance de progeniture, non auto-evitante) = (I-A)^{-1}
inv = np.linalg.inv(np.eye(len(PIL)) - A)

# =====================================================================================
titre("Marche vs branchement : portee attendue par pilier d'amorce")
# =====================================================================================
tw = eng.build_cascade_tables(G, mode="walk")
tb = eng.build_cascade_tables(G, mode="branching")


def reach_vec(table_j):
    ind, probs = table_j
    return probs @ ind                                       # esperance d'atteinte par pilier


Ew, Eb, Elin = {}, {}, {}
print(f"  {'amorce':<8}{'E|S| marche':>14}{'E|S| branchement':>18}{'envelope (I-W)^-1':>20}")
for j in PIL:
    Ew[j] = reach_vec(tw[j]).sum()
    Eb[j] = reach_vec(tb[j]).sum()
    Elin[j] = inv[:, COL[j]].sum()                           # somme des comptes attendus
    print(f"  P{j:<7}{Ew[j]:>14.2f}{Eb[j]:>18.2f}{Elin[j]:>20.2f}")
print("  Lecture : marche et branchement ont le MEME nombre moyen de descendants directs")
print("  (e_j), donc des portees voisines, toutes deux SOUS l'envelope lineaire (I-W)^-1")
print("  (que l'auto-evitement empeche d'atteindre). Ils different par la FORME, pas le")
print("  niveau moyen : c'est la distribution de taille qui change (panneau suivant).")

# =====================================================================================
titre("Distribution de la taille de cascade |S| (ponderee par l'amorce ROOT)")
# =====================================================================================
wroot = np.array([ROOT[j] for j in PIL]); wroot = wroot / wroot.sum()


def size_dist(tables):
    out = np.zeros(len(PIL) + 1)                             # index m = taille
    for c, j in enumerate(PIL):
        ind, probs = tables[j]
        sizes = ind.sum(axis=1).astype(int)
        for m, p in zip(sizes, probs):
            out[m] += wroot[c] * p
    return out[1:]                                           # tailles 1..5


sw, sb = size_dist(tw), size_dist(tb)
print(f"  {'taille |S|':<12}" + "".join(f"{m:>8}" for m in range(1, 6)))
print(f"  {'marche':<12}" + "".join(f"{v:>8.2f}" for v in sw))
print(f"  {'branchement':<12}" + "".join(f"{v:>8.2f}" for v in sb))
print(f"  taille moyenne : marche {sum((m+1)*v for m,v in enumerate(sw)):.2f}  "
      f"branchement {sum((m+1)*v for m,v in enumerate(sb)):.2f}")

# =====================================================================================
titre("Impact SCR (OpRisk) : marche vs branchement")
# =====================================================================================
sp = PARAMS["OPRISK"]
lam = ec.lambda_scenario("OPRISK", "S0_conforme", mode="center")
NY = 80_000
scr = {}
for mode in ("walk", "branching"):
    rng = np.random.default_rng(4242)                        # CRN
    loss = ec.simulate_euro(lam, G, sp["xi"], sp["sigma"], sp["u"], sp["p_u"], sp["cap"],
                            NY, rng, casc_mode=mode)
    scr[mode] = var(loss)
    print(f"  SCR (VaR 99,5%) {mode:<10} = {scr[mode]:.0f} M")
pct = 100 * (scr['branching'] / scr['walk'] - 1)
print(f"  ecart branchement vs marche : {pct:+.0f} %  (quasi inchange)")
print("  Le SCR ne bouge quasiment pas : au 99,5 %, la queue est portee par la perte")
print("  UNIQUE dominante (severite en xi~0,9, subexponentielle), pas par la LARGEUR de la")
print("  cascade. Passer de la chaine a l'arbre reforme la taille mais pas le capital.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("1. La cascade est desormais un vrai BRANCHEMENT multitype (arbre) disponible, pas")
print("   seulement une marche (chaine) : un pilier peut en declencher plusieurs d'un coup.")
print(f"2. Reconciliation avec le Vasicek : meme matrice moyenne W, meme e_j, meme")
print(f"   sous-criticite rho(W)={rho:.2f}<1 ; (I-W)^-1 est l'esperance de la progeniture.")
print("   Les deux representations du modele sont enfin le meme objet.")
print("3. Le branchement REDISTRIBUE la taille (plus de |S|=1 car les infections")
print("   independantes ratent souvent toutes, ET une queue de tailles plus grosse), a")
print(f"   portee moyenne quasi identique ; le SCR est quasi inchange ({pct:+.0f} %).")
print("4. C'est un resultat de ROBUSTESSE : la marche etait une simplification inoffensive")
print("   pour le capital ; le branchement donne l'objet theoriquement propre, coherent")
print("   avec W, quasi gratuitement. Defaut inchange ailleurs (mode='walk').")

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
BL = ["#7baafd", "#4c79c7", "#204993"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 4.9),
                                    gridspec_kw={"width_ratios": [1.1, 1.05, 0.85]})

# (a) portee attendue par amorce : marche / branchement / envelope
x = np.arange(len(PIL)); h = 0.27
ax1.bar(x - h, [Ew[j] for j in PIL], h, color=BL[0], edgecolor="#fcfcfb", label="marche (chaîne)")
ax1.bar(x, [Eb[j] for j in PIL], h, color=ACCENT, edgecolor="#fcfcfb", label="branchement (arbre)")
ax1.bar(x + h, [Elin[j] for j in PIL], h, color=BL[2], edgecolor="#fcfcfb",
        label="envelope $(I-W)^{-1}$")
ax1.set_xticks(x); ax1.set_xticklabels([f"P{j}" for j in PIL], fontsize=9)
ax1.set_ylabel("piliers atteints en moyenne  $E|S|$", color=INK2)
ax1.legend(frameon=False, fontsize=8.2)
ax1.set_title("(a)  Portée moyenne semblable, sous l'envelope $W$", fontsize=11,
              color=INK, pad=8)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# (b) distribution de la taille |S|
xs = np.arange(1, 6); h2 = 0.38
ax2.bar(xs - h2 / 2, sw, h2, color=BL[0], edgecolor="#fcfcfb", label="marche")
ax2.bar(xs + h2 / 2, sb, h2, color=ACCENT, edgecolor="#fcfcfb", label="branchement")
ax2.set_xticks(xs)
ax2.set_xlabel("taille de la cascade $|S|$ (nb de piliers)", color=INK2)
ax2.set_ylabel("probabilité (pondérée amorce)", color=INK2)
ax2.legend(frameon=False, fontsize=8.5)
ax2.set_title("(b)  Le branchement redistribue la taille", fontsize=11, color=INK, pad=8)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# (c) SCR marche vs branchement
ax3.bar([0, 1], [scr["walk"], scr["branching"]], color=[BL[0], ACCENT], edgecolor="#fcfcfb",
        width=0.6)
for i, k in enumerate(("walk", "branching")):
    ax3.text(i, scr[k] + max(scr.values()) * 0.015, f"{scr[k]:.0f}", ha="center",
             fontsize=9, color=INK2)
ax3.set_xticks([0, 1]); ax3.set_xticklabels(["marche", "branchement"], fontsize=9)
ax3.set_ylabel("SCR (VaR 99,5 %, M€)", color=INK2)
ax3.set_ylim(0, max(scr.values()) * 1.18)
ax3.set_title(f"(c)  SCR quasi inchangé ({pct:+.0f} %)", fontsize=11, color=INK, pad=8)
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)

fig.suptitle(f"Y : la cascade en branchement multitype, réconciliée avec $W$ "
             f"($\\rho(W)={rho:.2f}<1$)", fontsize=13, fontweight="bold", color=INK,
             x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Y_cascade_branchement.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nfigure ecrite :", path)
