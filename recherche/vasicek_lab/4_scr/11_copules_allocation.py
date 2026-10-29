#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11 : Structure de dependance (copules) et allocation du capital (Euler).

Deux ajouts actuariels sur le SCR de 09, sans double comptage. On FIXE les marges par
pilier issues du modele mecaniste (facteur commun + cascade) et on ne fait varier que
la COPULE qui les relie. Ainsi la copule ne s'empile pas sur le facteur : elle re-exprime
la dependance, ce qui permet deux choses.

  1. Copules. On agrege la perte sous quatre hypotheses de dependance, marges fixees :
       - independance  : borne de diversification maximale ;
       - copule gaussienne (correlation R estimee sur le mecaniste) : sert de CONTROLE,
         le mecaniste (facteur commun ~ copule gaussienne implicite) doit tomber a cote ;
       - mecaniste (dependance reelle du modele) ;
       - copule de Student (meme R, nu degres de liberte) : ajoute la DEPENDANCE DE QUEUE,
         les extremes co-occurrent. Le supplement t vs gaussienne = prime de dependance de
         queue. nu est un parametre NON identifiable sur nos donnees (axe de stress, comme xi).

  2. Allocation d'Euler. On repartit le SCR total entre les cinq piliers d'amorce par la
     regle d'Euler (contributions qui SOMMENT au total) : version VaR (Solvabilite II,
     E[L_j | total ~ VaR], estimee sur une bande, plus bruitee) et version TVaR
     (E[L_j | total >= VaR], exacte par construction). On compare la contribution au
     CAPITAL (queue) a la contribution a la perte MOYENNE : elles different, c'est l'interet.

PORTEE. Unites normalisees. La copule (et nu) est une hypothese de structure, non calibree
sur l'entite : le script quantifie la SENSIBILITE du SCR a cette hypothese, il ne pretend
pas la mesurer.

Sortie : diagnostics (SCR par copule, allocations) + figure S5_copules_allocation.png.
"""

import os
import sys

# AMORCE DE CHEMIN. Ce script importait les modules partages du lab en supposant que le
# repertoire courant etait la racine du lab : il echouait donc depuis la racine du depot,
# ou le harnais et les autres scripts sont lances. Meme amorce que les scripts 29 et 32.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
from scipy.stats import norm, t as student_t, rankdata
import matplotlib as mpl
import matplotlib.pyplot as plt

import scr_engine as eng

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(20260715)
W = 74
ALPHA = 0.995
NU = 4          # degres de liberte de la copule de Student (dependance de queue)
NY = 200_000

PIL = eng.PIL
LAB = [f"P{j}" for j in PIL]


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# ============================================================ marges mecanistes
titre("Modele mecaniste : pertes annuelles ventilees par pilier d'amorce")
MAT = eng.simulate_losses_by_pillar(NY, RNG)      # [NY x 5]
total_mech = MAT.sum(axis=1)
print(f"  {NY:,} annees ; cas de base g={eng.G_BASE}, xi=0,70, a=0,60.")
print(f"  {'pilier':<8}{'E[perte]':>11}{'part moyenne':>14}{'P(=0)':>9}")
for c, j in enumerate(PIL):
    col = MAT[:, c]
    print(f"  P{j:<7}{col.mean():>11.2f}{100*col.mean()/total_mech.mean():>13.1f}%"
          f"{(col == 0).mean():>9.3f}")


# ============================================================ copules (marges fixees)
def normal_scores(mat):
    """Scores normaux (rang -> uniforme -> Phi^-1), colonne par colonne."""
    n = mat.shape[0]
    Z = np.empty_like(mat, dtype=float)
    for c in range(mat.shape[1]):
        Z[:, c] = norm.ppf(rankdata(mat[:, c], method="average") / (n + 1))
    return Z


def inv_marginals(mat, U):
    """Inverse des CDF empiriques : agrege sum_c F_c^{-1}(U[:,c])."""
    tot = np.zeros(U.shape[0])
    for c in range(mat.shape[1]):
        tot += np.quantile(mat[:, c], U[:, c])
    return tot


R = np.corrcoef(normal_scores(MAT), rowvar=False)
Lchol = np.linalg.cholesky(R)
k = len(PIL)


def agg_independence(mat, rng):
    tot = np.zeros(mat.shape[0])
    for c in range(mat.shape[1]):
        tot += mat[rng.permutation(mat.shape[0]), c]
    return tot


def agg_gaussian(mat, rng, n):
    Zc = rng.standard_normal((n, k)) @ Lchol.T
    return inv_marginals(mat, norm.cdf(Zc))


def agg_student(mat, rng, n, nu):
    Zc = rng.standard_normal((n, k)) @ Lchol.T
    chi = rng.chisquare(nu, size=n)
    T = Zc / np.sqrt(chi / nu)[:, None]
    return inv_marginals(mat, student_t.cdf(T, nu))

titre("Matrice de correlation des scores normaux (copule gaussienne implicite du mecaniste)")
print("        " + "".join(f"{l:>8}" for l in LAB))
for i, l in enumerate(LAB):
    print(f"  {l:<6}" + "".join(f"{R[i, j]:>8.3f}" for j in range(k)))

titre(f"SCR (VaR {ALPHA:.1%}) selon la structure de dependance, marges identiques")
scen = {
    "independance": agg_independence(MAT, RNG),
    "copule gaussienne": agg_gaussian(MAT, RNG, NY),
    "mecaniste (reel)": total_mech,
    f"copule Student (nu={NU})": agg_student(MAT, RNG, NY, NU),
}
print(f"  {'structure':<26}{'VaR 99,5%':>12}{'TVaR 99,5%':>12}{'vs gaussienne':>15}")
v_gauss = eng.var(scen["copule gaussienne"], ALPHA)
for name, tot in scen.items():
    v, tv = eng.var(tot, ALPHA), eng.tvar(tot, ALPHA)
    rel = "" if "gaussienne" in name else f"{100*(v/v_gauss-1):+.1f}%"
    print(f"  {name:<26}{v:>12.0f}{tv:>12.0f}{rel:>15}")
print("\n  Lecture : mecaniste ~ copule gaussienne (controle de coherence) ;")
print("  Student > gaussienne = PRIME DE DEPENDANCE DE QUEUE (les extremes co-occurrent).")


# ============================================================ allocation d'Euler
titre("Allocation d'Euler : repartir le SCR mecaniste entre piliers (somme = total)")
v_mech = eng.var(total_mech, ALPHA)
tv_mech = eng.tvar(total_mech, ALPHA)

# TVaR-Euler : E[L_j | total >= VaR], somme exacte = TVaR
tail = total_mech >= v_mech
alloc_es = MAT[tail].mean(axis=0)
# VaR-Euler : E[L_j | total ~ VaR] sur une BANDE EN PROBABILITE autour de alpha
# (une bande symetrique en valeur biaiserait vers le bas sur une queue lourde).
lo = np.quantile(total_mech, ALPHA - 0.002)
hi = np.quantile(total_mech, ALPHA + 0.002)
band = (total_mech >= lo) & (total_mech <= hi)
alloc_var = MAT[band].mean(axis=0)
# contribution a la perte MOYENNE (pour comparaison)
alloc_mean = MAT.mean(axis=0)

print(f"  {'pilier':<8}{'part perte moy.':>17}{'part SCR (VaR)':>16}{'part SCR (TVaR)':>17}")
for c, j in enumerate(PIL):
    print(f"  P{j:<7}{100*alloc_mean[c]/alloc_mean.sum():>16.1f}%"
          f"{100*alloc_var[c]/alloc_var.sum():>15.1f}%"
          f"{100*alloc_es[c]/alloc_es.sum():>16.1f}%")
print(f"  {'controle somme':<8}{alloc_mean.sum():>16.1f} "
      f"{alloc_var.sum():>15.1f} {alloc_es.sum():>16.1f}")
print(f"  (VaR mecaniste = {v_mech:.1f} ; TVaR = {tv_mech:.1f} ; les sommes doivent coller)")
print("  Interet : la part de CAPITAL (queue) differe de la part de perte MOYENNE.")

# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2 = "#1b1e30", "#223e55"
ACCENT, BLUE, GREEN, GREY = "#a6002e", "#3661ac", "#008d87", "#7d7d7d"
PCOL = {1: "#204993", 2: "#4c79c7", 3: "#6491e1", 4: "#a6002e", 5: "#7d7d7d"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.4, 5.3))

# panneau A : SCR selon la copule
names = list(scen.keys())
vars_ = [eng.var(scen[n], ALPHA) for n in names]
cols = [GREY, BLUE, INK, ACCENT]
axA.bar(range(len(names)), vars_, color=cols)
for i, v in enumerate(vars_):
    axA.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
axA.axhline(vars_[1], color=BLUE, ls=":", lw=1.0)
axA.set_xticks(range(len(names)))
axA.set_xticklabels(["independance", "copule\ngaussienne", "mecaniste", f"copule\nStudent nu={NU}"],
                    fontsize=8.6)
axA.set_ylabel("SCR = VaR 99,5 % (unites normalisees)", fontsize=9.5, color=INK2)
axA.set_title("(A)  La dependance de queue coute du capital",
              fontsize=10.5, color=INK, pad=6)
axA.grid(alpha=0.25, lw=0.5, axis="y")

# panneau B : allocation d'Euler vs part de perte moyenne
x = np.arange(k)
wbar = 0.38
axB.bar(x - wbar/2, 100*alloc_mean/alloc_mean.sum(), wbar, color=GREY,
        label="part de la perte moyenne")
axB.bar(x + wbar/2, 100*alloc_es/alloc_es.sum(), wbar,
        color=[PCOL[j] for j in PIL], label="part du SCR (Euler-TVaR)")
axB.set_xticks(x)
axB.set_xticklabels(LAB)
axB.set_ylabel("part (%)", fontsize=9.5, color=INK2)
axB.set_title("(B)  Allocation d'Euler : capital de queue vs perte moyenne",
              fontsize=10.5, color=INK, pad=6)
axB.legend(fontsize=8.4, frameon=False)
axB.grid(alpha=0.25, lw=0.5, axis="y")

fig.suptitle("Copules et allocation : structurer la dependance, repartir le capital",
             fontsize=12.8, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S5_copules_allocation.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
