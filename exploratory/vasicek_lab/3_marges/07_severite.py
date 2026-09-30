#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07 : Severite. De l'ordinal (GBASE) aux unites monetaires (normalisees).

Chaque pilier touche par un incident genere une perte L_j. La forme est un corps
lognormal prolonge par une queue GPD au-dela d'un seuil (POT). Le score GBASE ne
fixe PAS un montant : il fixe l'ECHELLE de la loi du pilier (position mu_j et seuil
u_j croissants en GBASE). L'indice de queue xi est COMMUN a tous les piliers (regime
de queue du risque cyber), et c'est l'axe de sensibilite. L'echelle de queue beta_j
est DERIVEE par continuite de densite au raccord (pas un parametre libre).

Severite d'une cascade sur l'ensemble S :  X(S) = somme_{j dans S} L_j, tirages
independants sachant S (la dependance est deja portee par la formation de S via la
cascade, et par Y via la frequence : pas de troisieme canal).

PORTEE. Ce script produit la brique severite, pas le SCR. L'ancrage des unites (le
niveau de mu0) est une convention de NORMALISATION, pas un calibrage sur donnees de
l'entite ; seule la structure relative entre piliers est portee par GBASE. xi est
laisse libre : base 0,7 ici, la valeur 0,9 ancree sur SAS OpRisk (cf. 05) est un
stress, jamais un point de calage. Les euros absolus ne sont pas revendiques.

Sortie : diagnostics (mu_j, u_j, beta_j, medianes, moyennes) + figure
S1_severite.png (densites par pilier et exemples de cascade).
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
from cascade_model import GBASE, PILIERS  # noqa: E402  (source unique du jugement)
from severite_model import (  # noqa: E402  (source unique de la loi de severite)
    MU0, CPENTE, SIGMA, P_U, XI_BASE,
    params_pilier, draw_loss, draw_cascade_severity, mean_pilier,
)

RNG = np.random.default_rng(20260715)
W = 74


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# ============================================================ diagnostics
titre("Parametres de severite par pilier (unites normalisees, base mediane P min = 1)")
print(f"  MU0={MU0}  CPENTE={CPENTE}  SIGMA={SIGMA}  P_U={P_U}  XI_BASE={XI_BASE}")
print(f"  {'pilier':<8}{'GBASE':>6}{'mu_j':>9}{'mediane':>10}{'u_j':>10}"
      f"{'beta_j':>10}{'E[L_j]':>11}")
NSIM = 400_000
for j in sorted(GBASE, key=lambda k: GBASE[k]):
    mu, u, beta = params_pilier(j)
    med = np.exp(mu)
    m_exact = mean_pilier(j)
    m_mc = draw_loss(j, NSIM, RNG).mean()
    print(f"  P{j:<7}{GBASE[j]:>6}{mu:>9.3f}{med:>10.3f}{u:>10.3f}"
          f"{beta:>10.3f}{m_exact:>11.3f}   (MC {m_mc:.3f})")

titre("Coherence de la queue : la moyenne est-elle finie et bien capturee ?")
print("  xi < 1 => moyenne finie ; on verifie exact vs Monte-Carlo (ecart attendu faible).")
for j in sorted(GBASE, key=lambda k: GBASE[k]):
    m_exact = mean_pilier(j)
    m_mc = draw_loss(j, NSIM, RNG).mean()
    err = abs(m_mc - m_exact) / m_exact * 100
    flag = "OK" if err < 3 else "A VOIR"
    print(f"  P{j} : exact {m_exact:8.3f}   MC {m_mc:8.3f}   ecart {err:5.2f} %   {flag}")

titre("Severite de quelques cascades  X(S) = somme des pertes des piliers touches")
exemples = [
    [1], [4], [1, 2], [4, 2], [1, 2, 3], [1, 2, 4], [1, 2, 3, 4, 5],
]
print(f"  {'ensemble S':<22}{'E[X]':>10}{'mediane':>11}{'q95':>11}"
      f"{'q99.5':>12}")
for S in exemples:
    x = draw_cascade_severity(S, NSIM, RNG)
    lab = "{" + ",".join(f"P{j}" for j in S) + "}"
    print(f"  {lab:<22}{x.mean():>10.2f}{np.median(x):>11.2f}"
          f"{np.quantile(x, 0.95):>11.2f}{np.quantile(x, 0.995):>12.2f}")

titre("Sensibilite a xi (severite d'une cascade lourde {P1,P2,P4})")
print("  quantile 99,5 % de X, en fonction de l'indice de queue commun xi.")
S = [1, 2, 4]
for xi in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
    x = draw_cascade_severity(S, NSIM, RNG, xi=xi)
    print(f"  xi = {xi:.2f}   q99.5 = {np.quantile(x, 0.995):>10.2f}"
          f"   E[X] = {x.mean():>8.2f}")

# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
PCOL = {1: "#204993", 2: "#4c79c7", 3: "#6491e1", 4: "#a6002e", 5: "#7d7d7d"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.2))

# panneau A : densite (approchee par histogramme lisse) de L_j, echelle log
grid = np.logspace(-1.2, 2.6, 400)
for j in sorted(GBASE, key=lambda k: GBASE[k]):
    s = draw_loss(j, 300_000, RNG)
    dens, edges = np.histogram(np.log10(s), bins=120, density=True)
    centers = 10 ** (0.5 * (edges[:-1] + edges[1:]))
    axA.plot(centers, dens, color=PCOL[j], lw=1.8,
             label=f"P{j} (GBASE {GBASE[j]})")
axA.set_xscale("log")
axA.set_xlabel("perte L_j  (unites normalisees, echelle log)", fontsize=9.5, color=INK2)
axA.set_ylabel("densite (en log10 L)", fontsize=9.5, color=INK2)
axA.set_title("(A)  Loi de perte par pilier : GBASE fixe l'echelle, xi la queue",
              fontsize=10.5, color=INK, pad=6)
axA.legend(fontsize=8.2, frameon=False)
axA.grid(alpha=0.25, lw=0.5)

# panneau B : sensibilite du q99.5 de X(S) a xi, pour quelques cascades
xis = np.linspace(0.5, 0.95, 10)
for S, col, lab in [([1], PCOL[1], "{P1}"),
                    ([1, 2], PCOL[2], "{P1,P2}"),
                    ([1, 2, 4], PCOL[4], "{P1,P2,P4}")]:
    q = [np.quantile(draw_cascade_severity(S, 200_000, RNG, xi=xi), 0.995)
         for xi in xis]
    axB.plot(xis, q, "-o", color=col, lw=1.8, ms=3.5, label=lab)
axB.axvspan(0.9, 0.95, color="#a6002e", alpha=0.10)
axB.text(0.925, axB.get_ylim()[1], "zone instable", ha="center", va="top",
         fontsize=8, color="#a6002e", rotation=90)
axB.set_xlabel("indice de queue commun  xi", fontsize=9.5, color=INK2)
axB.set_ylabel("quantile 99,5 % de X(S)", fontsize=9.5, color=INK2)
axB.set_title("(B)  La queue commande : q99,5 % vs xi",
              fontsize=10.5, color=INK, pad=6)
axB.legend(fontsize=8.6, frameon=False)
axB.grid(alpha=0.25, lw=0.5)

fig.suptitle("Severite : de l'ordinal GBASE aux unites monetaires",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S1_severite.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
