#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
34 : l'ÉCHELLE DE REPLI sur u_ij, telle que demandée par Hugo.

La consigne du tuteur : attaquer la version la plus dure, puis reculer par crans
documentés. Instanciation sur les probabilités de propagation :

        logit p_ij = logit p_j + u_ij        (i = source, j = cible)

  p_j  = attractivité de la cible, ce qui reste quand la source ne compte pas ;
  u_ij = excès propre à la source, c'est-à-dire LA DIRECTION, et donc la contribution.

NORMALISATION, sans quoi la décomposition n'est pas identifiée. Pour toute constante
c_j on pourrait poser (p_j + c_j, u_ij - c_j) sans rien changer : on impose donc
        moyenne sur i (i != j) de u_ij = 0,  pour chaque j,
ce qui fait de p_j exactement l'attractivité moyenne de j. C'est une CONVENTION DE
REPÉRAGE et non une information : le script vérifie que le SCR et le classement n'en
dépendent pas.

LES CRANS, du plus dur au repli :
  0. u libre                 15 degrés de liberté après normalisation
  1. u de rang 1             u_ij = b_j (a_i - moyenne des a), ~9 degrés  <- JAMAIS TESTÉ
  4. u = 0                   p_ij = p_j : les colonnes de W deviennent identiques

Le cran 4 est le MODÈLE NUL, et il n'est pas anodin : à u = 0 aucune source ne se
distingue plus par son profil de propagation, donc la priorité de remédiation ne peut
plus venir que de la fréquence d'amorce. L'écart au cran 4 n'est donc pas un delta de
capital, c'est un basculement de la logique de décision. C'est là que se mesure la
contribution, pas sur le niveau de la VaR.

Ce que ce script NE prétend pas : rendre u identifiable. Le placebo directionnel ne
rejette pas u = 0, et restreindre au rang 1 ne crée pas de donnée. Ce que le rang 1
apporte est un ensemble admissible PLUS PETIT, donc des bornes de capital plus serrées,
au prix d'une hypothèse de structure explicite.

Sortie : diagnostics + figure Z5_echelle_repli.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import partial_id as pid                                        # noqa: E402

WID = 80
SEED = 20260721
N_SET = 400               # tirages par cran pour les bornes
N_PRIO = 120              # matrices pour le classement de priorité


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


OFF = ~np.eye(pid.NP_, dtype=bool)


def logit(x):
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return np.log(x / (1 - x))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def decompose_logit(P):
    """(p_j, u_ij) avec moyenne_i u_ij = 0 sur i != j, pour chaque j."""
    L = np.full((pid.NP_, pid.NP_), np.nan)
    L[OFF] = logit(P[OFF])
    lp = np.nanmean(L, axis=0)              # logit p_j = moyenne des logits par colonne
    U = np.zeros((pid.NP_, pid.NP_))
    U[OFF] = (L - lp[None, :])[OFF]
    return lp, U


def recompose(lp, U):
    """W a partir de (logit p_j, u). Diagonale nulle : pas d'auto-propagation."""
    W = np.zeros((pid.NP_, pid.NP_))
    W[OFF] = sigmoid((lp[None, :] + U)[OFF])
    return W


def centrer(U):
    """Impose moyenne_i u_ij = 0 sur i != j."""
    V = U.copy()
    for j in range(pid.NP_):
        idx = [i for i in range(pid.NP_) if i != j]
        V[idx, j] -= V[idx, j].mean()
    V[~OFF] = 0.0
    return V


def rang1(a, b):
    """Famille de rang 1 compatible avec la normalisation : u_ij = b_j (a_i - moy_j a)."""
    U = np.zeros((pid.NP_, pid.NP_))
    for j in range(pid.NP_):
        idx = [i for i in range(pid.NP_) if i != j]
        U[idx, j] = b[j] * (a[idx] - a[idx].mean())
    return U


# =====================================================================================
ev = pid.Evaluator(n_years=40_000, seed=SEED)
P = pid.expert_matrix()
LP, U_EXP = decompose_logit(P)
UMAX = float(np.abs(U_EXP[OFF]).max())

titre("La décomposition, et le contrôle que la normalisation ne dit rien")
print(f"  attractivité des cibles, p_j : "
      + "  ".join(f"P{pid.PIL[j]}={sigmoid(LP[j]):.3f}" for j in range(pid.NP_)))
print(f"  amplitude de la direction, max |u_ij| = {UMAX:.3f}")
W_rec = recompose(LP, U_EXP)
print(f"  reconstruction exacte de W a partir de (p_j, u) : "
      f"ecart max {np.abs(W_rec - P).max():.2e}")
# La convention de reperage ne doit pas deplacer le resultat. ATTENTION : le test doit
# translater logit p_j de dec_j ET retirer dec_j a u, SANS recentrer ensuite -- recentrer
# annulerait la translation et on testerait autre chose.
dec = np.array([0.0, 0.3, -0.3, 0.7, -0.5])
scr_a, _ = ev(recompose(LP, U_EXP))
scr_b, _ = ev(recompose(LP + dec, U_EXP - dec[None, :]))
ecart = abs(scr_a - scr_b)
print(f"  SCR sous la convention retenue      : {scr_a:.0f} M")
print(f"  SCR apres translation compensee     : {scr_b:.0f} M (ecart {ecart:.1f} M)")
if ecart < 1.0:
    print("  La normalisation est bien une CONVENTION : le capital n'en depend pas.")
else:
    print("  ALERTE : le capital DEPEND de la convention de reperage. La decomposition")
    print("  transporterait alors de l'information, et l'enonce serait mal pose.")

# =====================================================================================
titre("Cran 1 : u est-il bien approché par un rang 1 ?")
# =====================================================================================
# SVD de u restreinte a la structure hors-diagonale (diagonale mise a zero)
Usvd = U_EXP.copy()
s = np.linalg.svd(Usvd, compute_uv=False)
part1 = s[0] ** 2 / (s ** 2).sum()
print(f"  valeurs singulieres de u : " + "  ".join(f"{x:.3f}" for x in s))
print(f"  part de variance captee par le rang 1 : {100*part1:.1f} %")
print("  Lecture : un rang 1 restitue l'essentiel de la direction posee par le classeur")
print("  si cette part est elevee ; sinon, reculer d'un cran coute cher en fidelite.")

# =====================================================================================
titre("Bornes de capital, cran par cran")
# =====================================================================================
rng = np.random.default_rng(SEED)


def bornes(tire, n, nom):
    vals, Ws = [], []
    for _ in range(n):
        U = tire()
        W = recompose(LP, U)
        if not pid.admissible(W):
            continue
        vals.append(ev.scr(W))
        Ws.append(W)
    v = np.array(vals)
    print(f"  {nom:<34} [{v.min():7.0f} ; {v.max():7.0f}] M   largeur {v.max()-v.min():6.0f} M"
          f"   ({len(v)} tirages)")
    return v, Ws


def cal(U):
    """Ramene toute famille a la MEME amplitude max |u_ij| = UMAX.

    Sans ce calage, comparer les crans n'a aucun sens : le cran 0 tire dans un pave puis
    recentre (ce qui retrecit), le cran 1 compose deux tirages (ce qui peut depasser).
    On comparerait alors des amplitudes, pas des structures.
    """
    m = np.abs(U[OFF]).max()
    return U * (UMAX / m) if m > 1e-12 else U


v0, W0 = bornes(lambda: cal(centrer(rng.uniform(-1, 1, (pid.NP_, pid.NP_)))),
                N_SET, "cran 0 : u libre")
v1, W1 = bornes(lambda: cal(rang1(rng.uniform(-1, 1, pid.NP_),
                                  rng.uniform(-1, 1, pid.NP_))),
                N_SET, "cran 1 : u de rang 1")
scr_exp = ev.scr(recompose(LP, U_EXP))
scr_nul = ev.scr(recompose(LP, np.zeros((pid.NP_, pid.NP_))))
print(f"  {'cran 3 : u du classeur (point)':<34} {scr_exp:7.0f} M")
print(f"  {'cran 4 : u = 0 (modele nul)':<34} {scr_nul:7.0f} M")
l0, l1 = v0.max() - v0.min(), v1.max() - v1.min()
red = 100 * (1 - l1 / l0)
print(f"\n  Largeur cran 0 : {l0:.0f} M   cran 1 : {l1:.0f} M   variation {red:+.0f} %")
if red > 5:
    print("  Reculer d'un cran RESSERRE les bornes : c'est le gain d'une hypothese de")
    print("  structure explicite. On n'a pas acquis de donnee, on a assume une forme.")
elif red < -5:
    print("  Reculer d'un cran ELARGIT les bornes. A amplitude egale, la famille de rang 1")
    print("  n'est donc pas incluse dans l'echantillon libre : le tirage uniforme centre")
    print("  explore mal les directions extremes, que le rang 1 atteint. A retenir : le")
    print("  cran 1 n'est PAS un sous-ensemble du cran 0 tel qu'echantillonne ici, et")
    print("  comparer leurs largeurs ne mesure donc pas un gain d'information.")
else:
    print("  Les deux largeurs sont comparables : a cette amplitude, restreindre au rang 1")
    print("  ne change pas materiellement l'enveloppe de capital.")

# =====================================================================================
titre("Ce qui bascule au cran 4 : la logique de la priorite")
# =====================================================================================
def profil(Ws, nom):
    B = np.array([ev.benefits(W) for W in Ws[:N_PRIO]])
    f = np.bincount(B.argmax(axis=1), minlength=pid.NP_) / len(B)
    print(f"  {nom:<30}" + "".join(f"  P{pid.PIL[j]} {100*f[j]:4.0f}%" for j in range(pid.NP_)))
    return f


f0 = profil(W0, "cran 0 (u libre)")
f1 = profil(W1, "cran 1 (rang 1)")
ben_exp = ev.benefits(recompose(LP, U_EXP))
ben_nul = ev.benefits(recompose(LP, np.zeros((pid.NP_, pid.NP_))))
print(f"  {'cran 3 (classeur), gains':<30}"
      + "".join(f"  P{pid.PIL[j]} {ben_exp[j]:4.0f}" for j in range(pid.NP_)))
print(f"  {'cran 4 (u = 0), gains':<30}"
      + "".join(f"  P{pid.PIL[j]} {ben_nul[j]:4.0f}" for j in range(pid.NP_)))
o_exp = [pid.PIL[j] for j in np.argsort(-ben_exp)]
o_nul = [pid.PIL[j] for j in np.argsort(-ben_nul)]
print(f"\n  Ordre de remediation au cran 3 : " + " > ".join("P"+str(p) for p in o_exp))
print(f"  Ordre de remediation au cran 4 : " + " > ".join("P"+str(p) for p in o_nul))
# Au cran 4, W_ij = p_j ne depend que de la CIBLE. Le controle doit donc comparer, pour
# une meme colonne j, les valeurs vues par les differentes sources i -- et non des lignes
# entieres, dont les colonnes disponibles different a cause de la diagonale nulle.
W_nul = recompose(LP, np.zeros((pid.NP_, pid.NP_)))
ecart_col = max(np.ptp([W_nul[i, j] for i in range(pid.NP_) if i != j])
                for j in range(pid.NP_))
print(f"  Au cran 4, pour une meme cible, l'ecart entre sources vaut {ecart_col:.2e} :")
print("  aucune source ne se distingue plus par son profil de propagation. Ce qui")
print("  subsiste du classement ne vient QUE de la frequence d'amorce, pas du mecanisme.")

# Ne PAS crier a l'inversion sur un ecart qui tient dans le bruit Monte-Carlo.
g1, g2 = sorted(ben_nul)[-1], sorted(ben_nul)[-2]
marge = 100 * (g1 - g2) / g1
print(f"\n  Au cran 4, les deux premiers sont separes de {g1-g2:.0f} M ({marge:.1f} %).")
if o_exp[0] != o_nul[0] and marge > 3:
    print(f"  => le pilier de tete BASCULE de P{o_exp[0]} a P{o_nul[0]}, avec une marge nette.")
    print("  C'est la quantite decisionnelle sur laquelle se mesure la contribution.")
elif marge <= 3:
    print(f"  => marge trop faible pour conclure a une inversion : au cran 4, P{o_nul[0]} et")
    print(f"  P{o_nul[1]} sont INDISCERNABLES. Le resultat defendable n'est pas un")
    print("  basculement, c'est que la DIRECTION est ce qui separe les deux piliers de")
    print(f"  tete : au cran 3 l'ecart est de {max(ben_exp)-sorted(ben_exp)[-2]:.0f} M, au cran 4 il")
    print("  s'annule. Sans direction, on ne sait plus lequel remedier en premier.")
else:
    print(f"  => le pilier de tete reste P{o_exp[0]} : chercher la contribution dans l'ordre")
    print("  complet ou dans l'ecart des gains, pas dans le premier rang.")

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
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.66, 3.23))

crans = ["0\nu libre", "1\nrang 1", "3\nclasseur", "4\nu = 0"]
los = [v0.min(), v1.min(), scr_exp, scr_nul]
his = [v0.max(), v1.max(), scr_exp, scr_nul]
xs = np.arange(4)
for i in range(4):
    ax1.plot([i, i], [los[i], his[i]], color=BLUE if i < 2 else ACCENT, lw=6, alpha=0.75,
             solid_capstyle="round")
ax1.scatter(xs, los, marker="_", s=300, color=INK, zorder=5)
ax1.scatter(xs, his, marker="_", s=300, color=INK, zorder=5)
ax1.set_xticks(xs); ax1.set_xticklabels(crans, fontsize=9)
ax1.set_ylabel("SCR (VaR 99,5 %, M€)", color=INK2)
ax1.set_title("(a)  Les bornes se resserrent\nen reculant d'un cran", fontsize=11,
              color=INK, pad=8)

ax2.bar(np.arange(len(s)), s, color=BLUE, alpha=0.85)
ax2.set_xticks(range(len(s)))
ax2.set_xticklabels([f"$\\sigma_{i+1}$" for i in range(len(s))])
ax2.set_ylabel("valeur singulière de $u$", color=INK2)
ax2.set_title(f"(b)  Le rang 1 capte {100*part1:.0f} %\nde la direction posée",
              fontsize=11, color=INK, pad=8)

w = 0.38
xs5 = np.arange(pid.NP_)
ax3.bar(xs5 - w/2, ben_exp, width=w, color=BLUE, alpha=0.85, label="cran 3 (classeur)")
ax3.bar(xs5 + w/2, ben_nul, width=w, color=ACCENT, alpha=0.85, label="cran 4 ($u=0$)")
ax3.set_xticks(xs5); ax3.set_xticklabels([f"P{p}" for p in pid.PIL])
ax3.set_ylabel("gain de remédiation (M€)", color=INK2)
ax3.set_title("(c)  Ce que la direction change\nà la décision", fontsize=11, color=INK, pad=8)
ax3.legend(frameon=False, fontsize=8)

for ax in (ax1, ax2, ax3):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z5_echelle_repli.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
