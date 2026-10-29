# -*- coding: utf-8 -*-
"""Source UNIQUE de la loi de severite par pilier (corps lognormal + queue GPD).

Extrait de 07_severite.py pour etre partage par 07 (diagnostics/figure), le moteur
d'agregation (scr_engine) et la sensibilite. Meme principe que cascade_model.py :
un seul endroit ou vit le calage, pas de copie-colle desynchronisable.

GBASE fixe l'ECHELLE de la loi du pilier (position mu_j et seuil u_j), pas un montant.
xi est COMMUN a tous les piliers (regime de queue). beta_j est DERIVE par continuite
de densite au raccord. Voir ossature_scr.tex.
"""

import os
import sys

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_CASCADE = os.path.abspath(os.path.join(_HERE, "..", "cascade_qualitative"))
if _CASCADE not in sys.path:
    sys.path.insert(0, _CASCADE)
from cascade_model import GBASE  # noqa: E402  (jugement d'expert = source unique)

# ---- parametres (tous partages sauf l'echelle par pilier, portee par GBASE) ----
MU0 = 0.0            # ancrage : mediane du pilier le moins grave = exp(MU0) = 1 unite
RATIO_GRADE = 2.0    # choix de modelisation : chaque echelon de gravite GBASE DOUBLE la mediane
CPENTE = np.log(RATIO_GRADE)   # pente de la carte GBASE -> mu_j (= ln 2)
SIGMA = 0.80         # dispersion du corps lognormal (commune)
P_U = 0.90           # niveau du seuil de raccord corps -> queue
XI_BASE = 0.70       # indice de queue commun (axe de sensibilite ; 0,9 = stress SAS, cf. 05)

GMIN = min(GBASE.values())


def params_pilier(j, xi=XI_BASE):
    """(mu_j, u_j, beta_j) pour le pilier j. beta_j derive par continuite de densite."""
    mu = MU0 + CPENTE * (GBASE[j] - GMIN)
    z = stats.norm.ppf(P_U)
    u = np.exp(mu + SIGMA * z)
    beta = (1.0 - P_U) * u * SIGMA / stats.norm.pdf(z)
    return mu, u, beta


def draw_loss(j, size, rng, xi=XI_BASE):
    """Tire size pertes L_j par inverse de la CDF du spliced lognormal-GPD."""
    mu, u, beta = params_pilier(j, xi)
    out = np.empty(size)
    U = rng.random(size)
    body = U < P_U
    out[body] = np.exp(mu + SIGMA * stats.norm.ppf(U[body]))
    V = (U[~body] - P_U) / (1.0 - P_U)
    out[~body] = u + (beta / xi) * (np.power(1.0 - V, -xi) - 1.0)
    return out


def draw_cascade_severity(S, size, rng, xi=XI_BASE):
    """X(S) = somme_{j dans S} L_j, tirages independants sachant S."""
    total = np.zeros(size)
    for j in S:
        total += draw_loss(j, size, rng, xi)
    return total


def mean_pilier(j, xi=XI_BASE):
    """Moyenne exacte de L_j = contribution corps + contribution queue (finie si xi<1)."""
    mu, u, beta = params_pilier(j, xi)
    z = stats.norm.ppf(P_U)
    body = np.exp(mu + SIGMA ** 2 / 2.0) * stats.norm.cdf(z - SIGMA)
    tail = (1.0 - P_U) * (u + beta / (1.0 - xi))
    return body + tail
