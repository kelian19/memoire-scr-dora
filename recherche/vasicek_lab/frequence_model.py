# -*- coding: utf-8 -*-
"""Source UNIQUE de la loi de frequence (Poisson melange par un facteur commun Y).

Extrait de 08_frequence.py pour etre partage par 08 (diagnostics/figure) et le moteur
d'agregation (scr_engine). Meme principe que cascade_model.py.

N_j | Y ~ Poisson(lambda_j exp(a Y - a^2/2)), Y ~ N(0,1) commun a tous les piliers :
une seule source systemique (surdispersion ET co-occurrence). lambda_j proportionnel a
ROOT[j] (propension d'amorce). Voir ossature_scr.tex.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CASCADE = os.path.abspath(os.path.join(_HERE, "..", "cascade_qualitative"))
if _CASCADE not in sys.path:
    sys.path.insert(0, _CASCADE)
from cascade_model import ROOT  # noqa: E402  (propension d'amorce = source unique)

LAMBDA_TOT = 12.0   # incidents attendus par an (ancrage de frequence, choix de modele)
A_LOAD = 0.60       # charge systemique commune en cas de base (axe de sensibilite)

PIL = sorted(ROOT)                       # [1,2,3,4,5]
_SROOT = sum(ROOT.values())
LAMBDA = {j: LAMBDA_TOT * ROOT[j] / _SROOT for j in PIL}   # taux de base par pilier


def draw_counts(size, rng, a=A_LOAD):
    """Tire size annees. Renvoie un tableau (size x 5) des N_j, meme Y par annee."""
    Y = rng.standard_normal(size)
    m = np.exp(a * Y - a * a / 2.0)
    out = np.empty((size, len(PIL)), dtype=int)
    for c, j in enumerate(PIL):
        out[:, c] = rng.poisson(LAMBDA[j] * m)
    return out


def disp_index(lam, a=A_LOAD):
    """Var/E d'un Poisson melange : 1 + lam*(exp(a^2)-1)."""
    return 1.0 + lam * (np.exp(a * a) - 1.0)


def corr_pair(lj, lk, a=A_LOAD):
    """Correlation induite par Y entre deux piliers (charge commune)."""
    ea = np.exp(a * a) - 1.0
    cov = lj * lk * ea
    vj = lj + lj * lj * ea
    vk = lk + lk * lk * ea
    return cov / np.sqrt(vj * vk)
