#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08 : Frequence a facteur systemique commun. Une seule source de risque.

Le nombre d'incidents d'un pilier sur un an est un Poisson MELANGE par un facteur
systemique commun Y ~ N(0,1) :

    N_j | Y ~ Poisson( lambda_j * exp(a_j*Y - a_j^2/2) )

Le terme -a_j^2/2 recentre l'esperance : E[N_j] = lambda_j exactement. Le MEME Y
pour tous les piliers fait deux choses d'un coup :
  - il cree de la SURDISPERSION (Var/E > 1), le fait stylise du cyber (les attaques
    arrivent par vagues) ;
  - il cree de la CO-OCCURRENCE entre piliers (Cov(N_j,N_k) > 0), la dependance
    INTER-incidents.
C'est ce qui remplace une binomiale negative bricolee a cote d'un Vasicek : une seule
source systemique, celle du lab (le meme Y que la correlation de defaut).

Taux de base : lambda_j proportionnel a ROOT[j] (la propension d'amorce du pilier fixe
combien d'incidents il declenche), importe de cascade_model (source unique).

PORTEE. Ce script produit la brique frequence, pas le SCR. LAMBDA_TOT (nombre attendu
d'incidents par an) est un ancrage de frequence, choix de modelisation, pas une mesure.
La charge a_j est ici COMMUNE (a) en cas de base ; a_j propre a chaque pilier (lie a
son role dans W) est l'enrichissement de l'agregation (09). Aucune severite ici : la
frequence se traite seule, la jonction avec 07 se fait en 09.

Sortie : diagnostics (E, surdispersion, correlation induite, exact vs Monte-Carlo)
+ figure S2_frequence.png.
"""

import os
import sys

import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASCADE = os.path.abspath(os.path.join(HERE, "..", "cascade_qualitative"))
sys.path.insert(0, CASCADE)
sys.path.insert(0, HERE)
from cascade_model import ROOT  # noqa: E402  (propension d'amorce = source unique)
from frequence_model import (  # noqa: E402  (source unique de la loi de frequence)
    LAMBDA_TOT, A_LOAD, PIL, LAMBDA, draw_counts, disp_index, corr_pair,
)

RNG = np.random.default_rng(20260715)
W = 74


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# ============================================================ diagnostics
NSIM = 400_000
C = draw_counts(NSIM, RNG)

titre("Taux de base par pilier (lambda_j proportionnel a ROOT[j])")
print(f"  LAMBDA_TOT={LAMBDA_TOT}  A_LOAD={A_LOAD}   (total lambda = {sum(LAMBDA.values()):.2f})")
print(f"  {'pilier':<8}{'ROOT':>7}{'lambda_j':>11}{'E[N_j] exact':>14}"
      f"{'MC':>9}{'Var/E exact':>13}{'MC':>9}")
for c, j in enumerate(PIL):
    col = C[:, c]
    di_ex = disp_index(LAMBDA[j])
    di_mc = col.var() / col.mean()
    print(f"  P{j:<7}{ROOT[j]:>7.2f}{LAMBDA[j]:>11.3f}{LAMBDA[j]:>14.3f}"
          f"{col.mean():>9.3f}{di_ex:>13.3f}{di_mc:>9.3f}")

titre("Surdispersion du total annuel : Poisson simple donnerait Var/E = 1")
Ntot = C.sum(axis=1)
di_tot_ex = disp_index(sum(LAMBDA.values()))
print(f"  E[N_tot] exact = {sum(LAMBDA.values()):.3f}   MC = {Ntot.mean():.3f}")
print(f"  Var/E  exact  = {di_tot_ex:.3f}   MC = {Ntot.var()/Ntot.mean():.3f}"
      f"   ({di_tot_ex:.2f}x un Poisson simple)")
print(f"  P(N_tot=0) MC = {(Ntot==0).mean():.4f}   P(N_tot>=30) MC = {(Ntot>=30).mean():.4f}")

titre("Co-occurrence : la dependance INTER-incidents, induite par le seul Y")
print("  correlation entre comptes de piliers (exact vs Monte-Carlo).")
print(f"  {'paire':<12}{'corr exact':>12}{'corr MC':>10}")
for a_i in range(len(PIL)):
    for b_i in range(a_i + 1, len(PIL)):
        j, k = PIL[a_i], PIL[b_i]
        ce = corr_pair(LAMBDA[j], LAMBDA[k])
        cm = np.corrcoef(C[:, a_i], C[:, b_i])[0, 1]
        print(f"  P{j}-P{k:<9}{ce:>12.4f}{cm:>10.4f}")

titre("Sensibilite a la charge systemique a (surdispersion et correlation)")
print("  a=0 => Poisson independants (Var/E=1, corr=0). a monte => vagues + co-occurrence.")
print(f"  {'a':>6}{'Var/E total':>14}{'corr P1-P4':>13}{'P(N_tot=0)':>13}")
for a in (0.0, 0.3, 0.6, 0.9, 1.2):
    Ca = draw_counts(150_000, RNG, a=a)
    Nt = Ca.sum(axis=1)
    di = Nt.var() / Nt.mean() if a > 0 else 1.0
    cc = np.corrcoef(Ca[:, PIL.index(1)], Ca[:, PIL.index(4)])[0, 1] if a > 0 else 0.0
    print(f"  {a:>6.1f}{di:>14.3f}{cc:>13.4f}{(Nt==0).mean():>13.4f}")

# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2 = "#1b1e30", "#223e55"
ACCENT, BLUE = "#a6002e", "#3661ac"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.2))

# panneau A : distribution du total annuel, melange vs Poisson simple
bins = np.arange(0, 45) - 0.5
axA.hist(Ntot, bins=bins, density=True, color=BLUE, alpha=0.55,
         label=f"melange (Var/E={di_tot_ex:.1f})")
lam_tot = sum(LAMBDA.values())
kk = np.arange(0, 45)
axA.plot(kk, stats.poisson.pmf(kk, lam_tot), "o-", color=ACCENT, ms=3.5, lw=1.4,
         label="Poisson simple (Var/E=1)")
axA.set_xlabel("nombre total d'incidents dans l'annee", fontsize=9.5, color=INK2)
axA.set_ylabel("probabilite", fontsize=9.5, color=INK2)
axA.set_title("(A)  Le facteur commun epaissit la queue de la frequence",
              fontsize=10.5, color=INK, pad=6)
axA.legend(fontsize=8.6, frameon=False)
axA.grid(alpha=0.25, lw=0.5)

# panneau B : surdispersion et correlation en fonction de a
agrid = np.linspace(0.0, 1.2, 13)
di_curve = [disp_index(lam_tot, a=a) for a in agrid]
cc_curve = [corr_pair(LAMBDA[1], LAMBDA[4], a=a) for a in agrid]
axB.plot(agrid, di_curve, "-o", color=BLUE, lw=1.8, ms=3.5, label="Var/E du total")
axB.set_xlabel("charge systemique commune  a", fontsize=9.5, color=INK2)
axB.set_ylabel("surdispersion  Var/E", fontsize=9.5, color=BLUE)
axB.tick_params(axis="y", labelcolor=BLUE)
axB2 = axB.twinx()
axB2.plot(agrid, cc_curve, "-s", color=ACCENT, lw=1.8, ms=3.5, label="corr P1-P4")
axB2.set_ylabel("correlation P1-P4", fontsize=9.5, color=ACCENT)
axB2.tick_params(axis="y", labelcolor=ACCENT)
axB.axvline(A_LOAD, color=INK2, ls=":", lw=1.0)
axB.text(A_LOAD, axB.get_ylim()[1], " base a=0,6", ha="left", va="top",
         fontsize=8, color=INK2)
axB.set_title("(B)  Une seule source Y : surdispersion ET co-occurrence",
              fontsize=10.5, color=INK, pad=6)
axB.grid(alpha=0.25, lw=0.5)

fig.suptitle("Frequence : un facteur systemique commun, pas deux couches de dependance",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S2_frequence.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
