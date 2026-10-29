#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Derive d'echelle de la loi de severite, extraite du script 89.

POURQUOI CE MODULE EXISTE. Le script 89 mesure la derive et chiffre ce qu'elle coute sur le
QUANTILE de severite. Le script 90 teste si elle est neutre sur l'ECART entre etats de
conformite, ce qui est une affirmation du memoire. Les deux doivent partir de la MEME estimation :
si 90 re-estimait la derive de son cote, un desaccord de la troisieme decimale se lirait comme un
effet de la cascade. Le moteur d'estimation vit donc ici, une fois, et 89 comme 90 le lisent.
C'est le meme motif que canaux_conformite.py entre les scripts 43 et 68.

CONTRAT. Aucun calcul a l'import : ce module expose des constantes et trois fonctions.

CE QU'IL ESTIME, ET CE QU'IL N'ESTIME PAS. La derive est portee par l'ECHELLE de la queue,
sigma_t = exp(ls + b (t - t_ref)), l'indice de queue restant CONSTANT. Ce choix n'est pas
innocent et il est le bon ici : avec 91 exces, un indice de queue rendu lui-meme non stationnaire
serait estime sur trop peu d'information, et surtout la question posee est celle de l'echelle,
la FORME ayant passe le test hors echantillon du script 89.

Le seuil reste celui de la calibration publiee, et le taux de depassement est COMPTE sur
l'echantillon. C'est different du protocole de la section 1 du script 89, qui re-estime le seuil
sur chaque echantillon d'apprentissage : la section 1 predit, celle-ci decrit. Ne pas confondre
les deux, leurs xi ne sont pas comparables (seuils differents, et xi decroit avec le seuil).
"""

import os
import sys

import numpy as np
from scipy import optimize, stats
from scipy.stats import genpareto

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.severity.oprisk_analysis import (  # noqa: E402
    USD_EUR, filter_cyber, filter_finance, load_clean,
)
from src.utils.config import OPRISK  # noqa: E402

FICHIER = "SAS_OpRisk_Global_Data_June_2026.xlsx"
AN_REF = 2025          # annee a laquelle l'echelle derivee est evaluee
U_PUB = float(OPRISK["seuil_u_eur"])


def charger_pertes():
    """Perimetre complet de la chaine publiee : cyber x finance, en M EUR, avec l'annee."""
    d = filter_finance(filter_cyber(load_clean(
        os.path.join(REPO, "data", "raw", FICHIER))))
    d = d.dropna(subset=["year"]).copy()
    d["year"] = d["year"].astype(int)
    d["loss_eur"] = d["loss"] * USD_EUR
    return d


def ajuster(d, u=U_PUB, an_ref=AN_REF):
    """Ajustements stationnaire et a echelle derivee sur les exces au-dessus de u.

    Rend un dict. Les deux branches partagent leurs exces, leur seuil et leur taux de
    depassement : leur seule difference est la derive, ce qui est la condition pour que
    l'ecart entre elles mesure la derive et rien d'autre.
    """
    x = d["loss_eur"].values
    an = d["year"].values.astype(float)
    m = x > u
    exc = x[m] - u
    an_e = an[m]
    p_u = float(m.mean())

    xi_s, _, sig_s = genpareto.fit(exc, floc=0.0)

    def nll(par, b_fixe=None):
        xi, ls = par[0], par[1]
        b = par[2] if b_fixe is None else b_fixe
        sig_t = np.exp(ls + b * (an_e - an_ref))
        if xi <= -0.5:
            return 1e12
        z = 1.0 + xi * exc / sig_t
        if np.any(z <= 0.0):
            return 1e12
        return float(np.sum(np.log(sig_t) + (1.0 + 1.0 / xi) * np.log(z)))

    res = optimize.minimize(nll, [xi_s, np.log(sig_s), 0.0], method="BFGS")
    xi_ns, ls_ns, b = res.x
    se_b = float(np.sqrt(np.diag(res.hess_inv))[2])
    res0 = optimize.minimize(lambda p: nll(np.append(p, 0.0), b_fixe=0.0),
                             [xi_s, np.log(sig_s)], method="Nelder-Mead")
    lr = 2.0 * (res0.fun - res.fun)
    return {
        "n": len(x), "n_exc": len(exc), "u": u, "p_u": p_u, "an_ref": an_ref,
        "xi_stat": float(xi_s), "sigma_stat": float(sig_s),
        "xi_derive": float(xi_ns), "sigma_derive": float(np.exp(ls_ns)),
        "b": float(b), "se_b": se_b,
        "lr": float(lr), "p_lr": float(stats.chi2.sf(max(lr, 0.0), 1)),
    }


def quantile_pot(q, xi, sigma, p_u, u=U_PUB):
    """Quantile de severite par la formule des depassements de seuil."""
    return u + (sigma / xi) * (((1.0 - q) / p_u) ** (-xi) - 1.0)
