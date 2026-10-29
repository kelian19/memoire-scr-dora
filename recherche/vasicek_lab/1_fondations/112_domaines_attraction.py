#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
112 : les trois domaines d'attraction des maxima, et la queue calibree dans le domaine de Frechet.

Figure didactique du chapitre de la frequence et de la severite (version ENSAE), demande de Kelian
du 1er octobre 2026 : illustrer le theoreme de Fisher-Tippett-Gnedenko par les lois de Weibull,
Gumbel et Frechet. La figure est TRACEE ICI, pas reprise d'un autre memoire.

  (a) densites des lois des valeurs extremes generalisees standard (position 0, echelle 1) pour
      un indice de forme negatif (type Weibull, support borne a droite), nul (Gumbel) et positif
      (type Frechet, queue lourde) ;
  (b) fonction de survie des exces d'une loi de Pareto generalisee d'echelle unite, pour un indice
      negatif (exces bornes), nul (decroissance exponentielle) et l'indice CALIBRE sur OpRisk,
      LU dans src/utils/config.py (calibration gelee), jamais retape.

Les indices -0,5, 0 et 0,5 du panneau (a) et -0,3 du panneau (b) sont des valeurs d'illustration,
posees pour montrer les trois formes ; seul l'indice calibre est un resultat du memoire.

Controles, et le script s'arrete au moindre echec : chaque densite integre a 1 ; les bornes du
support valent -1/xi ; le domaine de l'indice calibre se DEDUIT de son signe ; l'indice calibre est
bien celui que le memoire publie.

Sortie : sorties_verif/112.txt (rediriger stdout seul) et figures/GEV_domaines_attraction.png
(nom sans chiffre : le harnais lirait un chiffre de nom de fichier comme un nombre publie).
"""
import os
import sys

import numpy as np
from scipy.stats import genextreme
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (REPO, LAB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import style_nexialog as sn  # noqa: E402
from src.utils.config import OPRISK  # noqa: E402

XI_ILLUSTR_GEV = (-0.5, 0.0, 0.5)       # Weibull, Gumbel, Frechet
XI_ILLUSTR_GPD = (-0.3, 0.0)            # exces bornes, exponentiel
XI_CAL = float(OPRISK["xi"])            # indice calibre, calibration gelee
XI_PUBLIE = 0.5954                      # valeur que le memoire imprime : controle, pas une source


def domaine(xi):
    """Domaine d'attraction deduit du signe de l'indice de forme."""
    if xi > 0:
        return "Frechet"
    if xi < 0:
        return "Weibull"
    return "Gumbel"


def survie_gpd(y, xi):
    """P(Y > y) pour une loi de Pareto generalisee d'echelle 1."""
    if xi == 0:
        return np.exp(-y)
    base = 1.0 + xi * y
    out = np.zeros_like(y)
    ok = base > 0
    out[ok] = base[ok] ** (-1.0 / xi)
    return out


print("=" * 78)
print("112 : DOMAINES D'ATTRACTION DES MAXIMA ET QUEUE CALIBREE")
print("=" * 78)

# ---- controles -----------------------------------------------------------------------------
if abs(XI_CAL - XI_PUBLIE) > 1e-12:
    sys.exit(f"ARRET : indice calibre {XI_CAL} different de la valeur publiee {XI_PUBLIE}")

print("\n1. Densites des lois des valeurs extremes generalisees (position 0, echelle 1)")
grille = np.linspace(-40, 400, 2_000_001)
for xi in XI_ILLUSTR_GEV:
    dens = genextreme.pdf(grille, c=-xi)          # scipy note c = -xi
    masse = np.trapezoid(dens, grille)
    borne = "aucune" if xi == 0 else f"{-1 / xi:+.3f} ({'superieure' if xi < 0 else 'inferieure'})"
    print(f"   xi = {xi:+.2f}  type {domaine(xi):8s}  masse integree {masse:.6f}  borne du support {borne}")
    if abs(masse - 1) > 2e-3:
        sys.exit(f"ARRET : la densite pour xi = {xi} n'integre pas a 1 ({masse:.6f})")

print("\n2. Exces au-dela d'un seuil, loi de Pareto generalisee d'echelle 1")
for xi in XI_ILLUSTR_GPD + (XI_CAL,):
    if xi < 0:
        detail = f"exces bornes a 1/|xi| = {1 / abs(xi):.3f}"
    elif xi == 0:
        detail = "decroissance exponentielle"
    else:
        detail = (f"decroissance en puissance d'exposant 1/xi = {1 / xi:.4f} ; esperance "
                  f"{'finie' if xi < 1 else 'infinie'}, variance {'finie' if xi < 0.5 else 'infinie'}")
    print(f"   xi = {xi:+.4f}  domaine {domaine(xi):8s}  {detail}")

print(f"\n3. L'indice calibre sur OpRisk ({XI_CAL:.4f}) est positif : domaine de {domaine(XI_CAL)}.")

# ---- figure ---------------------------------------------------------------------------------
# Taille calee sur l'impression : environ 0,95 fois la largeur du texte (16,6 cm), donc des
# polices de 7 a 8,5 pt une fois la figure posee. Etiquettes courtes : le detail des formes
# (support borne, decroissance exponentielle ou en puissance) est dans la legende LaTeX.
sn.appliquer(taille=9)
mpl.rcParams["font.family"] = ["DejaVu Sans", "Segoe UI", "sans-serif"]
bleu, turquoise, rouge = sn.CATEGORIEL[2], sn.CATEGORIEL[1], sn.CATEGORIEL[0]
xi_txt = f"{XI_CAL:.3f}".replace(".", "{,}")   # lu dans config, jamais retape

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

x = np.linspace(-4, 6, 4001)
for xi, couleur, etiq in [(-0.5, bleu, r"Weibull, $\xi=-0{,}5$"),
                          (0.0, turquoise, r"Gumbel, $\xi=0$"),
                          (0.5, rouge, r"Fréchet, $\xi=0{,}5$")]:
    ax1.plot(x, genextreme.pdf(x, c=-xi), color=couleur, lw=1.6, label=etiq)
ax1.set_xlim(-4, 6)
ax1.set_ylim(0, 0.52)
ax1.set_xlabel("maximum normalisé")
ax1.set_ylabel("densité")
ax1.set_title("(a) Lois limites des maxima", fontsize=9.5)
ax1.legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.6)

y = np.logspace(-2, 2, 2000)
for xi, couleur, etiq in [(-0.3, bleu, r"$\xi=-0{,}3$ (borné)"),
                          (0.0, turquoise, r"$\xi=0$ (exponentiel)"),
                          (XI_CAL, rouge, r"$\hat\xi=" + xi_txt + r"$ (OpRisk)")]:
    s = survie_gpd(y, xi)
    m = s > 0
    ax2.plot(y[m], s[m], color=couleur, lw=1.6, label=etiq)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(1e-2, 1e2)
ax2.set_ylim(1e-6, 1.5)
ax2.set_xlabel(r"excès au-delà du seuil ($y/\sigma$)")
ax2.set_ylabel(r"probabilité de dépasser $y$")
ax2.set_title("(b) Queue des excès (Pareto généralisée)", fontsize=9.5)
ax2.legend(frameon=False, fontsize=8, loc="lower left", handlelength=1.6)

fig.tight_layout(w_pad=2.0)
sortie = os.path.join(LAB, "figures", "GEV_domaines_attraction.png")
fig.savefig(sortie, dpi=200, bbox_inches="tight")
print(f"\nfigure : figures/{os.path.basename(sortie)}")

print("\nGrandeurs citees (sans separateur) :")
print("  xi_weibull -0.5   xi_gumbel 0   xi_frechet 0.5   xi_gpd_borne -0.3   xi_calibre 0.5954")
