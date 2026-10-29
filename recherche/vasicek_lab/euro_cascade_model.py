# -*- coding: utf-8 -*-
"""Simulateur euro-cascade partage : severite/frequence du memoire, agregation cascade.

Source unique du moteur euro-cascade, importe par 12 (SCR euro) et 13 (Delta_DORA
bootstrap). Reutilise TELS QUELS les modules du vrai modele :
  - severite : src.aggregation.lda.simulate_remediation_severity (GPD euro spliced) ;
  - parametres OpRisk/PRC : src.utils.config ; sur-dispersion phi de FREQUENCY.
La seule difference avec le memoire est l'AGREGATION : un incident amorce se propage a
un ensemble S de piliers (Vasicek dirige : noyau e_j = g*s_j/max_s, ensemble auto-evitant,
cf. scr_engine / note_vasicek_dirige) et chaque pilier touche tire sa severite.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.aggregation.lda import simulate_remediation_severity   # noqa: E402
from src.utils.config import OPRISK, PRC, FREQUENCY, SCR_DORA    # noqa: E402
import scr_engine as eng                                          # noqa: E402  (cascade dirigee)

PHI = FREQUENCY["dispersion_factor"]        # 9.2
PIL = eng.PIL
G_BASE = eng.G_BASE                          # gain de propagation de base (0.90)

# parametres euro par source (severite GPD + frequence de reference), depuis la config figee
PARAMS = {
    "OPRISK": dict(xi=OPRISK["xi"], sigma=OPRISK["sigma_eur"], u=OPRISK["seuil_u_eur"],
                   p_u=OPRISK["p_u"], cap=None,
                   xi_ic90=OPRISK["xi_ic90"], sigma_ic90=OPRISK["sigma_ic90"],
                   lam_ref=OPRISK["n_incidents"] / OPRISK["n_years"]),
    "PRC":    dict(xi=PRC["xi"], sigma=PRC["sigma_eur"], u=PRC["seuil_u_eur"],
                   p_u=PRC["p_u"], cap=SCR_DORA["cap_eur"],
                   xi_ic90=PRC["xi_ic90"], sigma_ic90=PRC["sigma_ic90"],
                   lam_ref=FREQUENCY["lambda_ref"]),
}

MEMOIRE_TOTAL = {"OPRISK": 9259.5, "PRC": 2772.7}      # SCR total 4 briques (chap. 5, S2)
MEMOIRE_DELTA = {                                       # Delta_DORA median + IC90 (config)
    "OPRISK": {"median": 3879.3, "ic90": [1496.9, 22249.3]},
    "PRC":    {"median": 2014.7, "ic90": [1606.9, 2366.1]},
}


def simulate_euro(lam, g, xi, sigma, u, p_u, cap, n_years, rng, phi=PHI, casc_mode="walk"):
    """Perte annuelle agregee (M€) par la cascade. lam = frequence annuelle d'AMORCES.

    casc_mode : 'walk' (marche auto-evitante, defaut) ou 'branching' (branchement multitype).
    """
    r = lam / (phi - 1.0)
    p = r / (r + lam)
    counts = rng.negative_binomial(r, p, size=n_years)
    T = int(counts.sum())
    annual = np.zeros(n_years)
    if T == 0:
        return annual
    year_of_event = np.repeat(np.arange(n_years), counts)
    tables = eng.build_cascade_tables(g, mode=casc_mode)
    w = np.array([eng.LAMBDA[j] for j in PIL], float)
    w /= w.sum()                                        # amorce ~ ROOT
    amorce = rng.choice(len(PIL), size=T, p=w)
    for c_am, j in enumerate(PIL):
        ev = np.where(amorce == c_am)[0]
        if ev.size == 0:
            continue
        yr = year_of_event[ev]
        ind, probs = tables[j]
        idx = rng.choice(len(probs), size=ev.size, p=probs)
        for s in range(len(probs)):
            sel = idx == s
            if not sel.any():
                continue
            yrs = yr[sel]
            for _ in range(int(ind[s].sum())):          # un tirage par pilier touche
                sev = simulate_remediation_severity(yrs.size, xi, sigma, u, p_u, cap, rng)
                annual += np.bincount(yrs, weights=sev, minlength=n_years)
    return annual


def simulate_euro_pp(lam_vec, g_vec, xi, sigma, u, p_u_vec, cap, n_years, rng, phi=PHI,
                     by_pillar=False, casc_mode="walk", xi_vec=None):
    """Perte annuelle par la cascade avec parametres PAR PILIER (script 16b).

    Meme moteur que simulate_euro (agregation non reecrite), mais les canaux sont
    indexes par pilier via des dict {pilier -> valeur} :
      - lam_vec  : frequence d'amorce par pilier (l'amorce est tiree ~ lam_vec) ;
      - g_vec    : gain de propagation du pilier SOURCE (e_j = g_vec[j]*s_j/max_s) ;
      - p_u_vec  : taux de materialite du pilier TOUCHE (severite per-pilier).
    Quand tous les piliers partagent le meme etat, se reduit exactement a simulate_euro.

    by_pillar=False (defaut) : renvoie la perte annuelle AGREGEE (vecteur n_years),
    identique au comportement anterieur. by_pillar=True : renvoie la matrice des pertes
    annuelles VENTILEES par pilier TOUCHE, forme (n_years, len(PIL)), colonne c = pilier
    PIL[c] ; sa somme sur les colonnes redonne exactement la perte agregee. Cette matrice
    sert a l'allocation d'Euler (contribution de chaque pilier a la queue, script 20).
    """
    nP = len(PIL)
    lam = float(sum(lam_vec[j] for j in PIL))
    annual_pp = np.zeros((n_years, nP))
    if lam <= 0:
        return annual_pp if by_pillar else annual_pp.sum(axis=1)
    r = lam / (phi - 1.0)
    p = r / (r + lam)
    counts = rng.negative_binomial(r, p, size=n_years)
    T = int(counts.sum())
    if T == 0:
        return annual_pp if by_pillar else annual_pp.sum(axis=1)
    year_of_event = np.repeat(np.arange(n_years), counts)
    tables = eng.build_cascade_tables(g_vec, mode=casc_mode)   # g_vec dict -> gains par pilier source
    w = np.array([lam_vec[j] for j in PIL], float)
    w /= w.sum()                                          # amorce ~ frequence par pilier
    amorce = rng.choice(len(PIL), size=T, p=w)
    for c_am, j in enumerate(PIL):
        ev = np.where(amorce == c_am)[0]
        if ev.size == 0:
            continue
        yr = year_of_event[ev]
        ind, probs = tables[j]
        idx = rng.choice(len(probs), size=ev.size, p=probs)
        for s in range(len(probs)):
            sel = idx == s
            if not sel.any():
                continue
            yrs = yr[sel]
            for c_p, pil in enumerate(PIL):              # un tirage par pilier TOUCHE, p_u propre
                if ind[s][c_p] == 1.0:
                    xi_p = xi if xi_vec is None else xi_vec[pil]   # queue propre au pilier
                    sev = simulate_remediation_severity(yrs.size, xi_p, sigma, u,
                                                        p_u_vec[pil], cap, rng)
                    annual_pp[:, c_p] += np.bincount(yrs, weights=sev, minlength=n_years)
    return annual_pp if by_pillar else annual_pp.sum(axis=1)


def var(x, alpha=0.995):
    return float(np.quantile(x, alpha))


def lambda_scenario(source, scenario, mode="center", rng=None):
    """Frequence annuelle sous un scenario DORA (multiplicateurs de src.frequency.negbin)."""
    from src.frequency.negbin import compute_lambda_scenario
    return compute_lambda_scenario(PARAMS[source]["lam_ref"], scenario,
                                   mode=mode, rng=rng)["lambda_global"]


def sample_severity_params(source, rng):
    """Tire (xi, sigma) dans leur IC90 (approx normale) : incertitude de severite bootstrap.

    Approximation : la normale non bornee peut tirer xi vers 1 et gonfler l'IC. Pour
    OpRisk, preferer oprisk_excesses()+bootstrap_sev_from_excesses() (reechantillonnage
    des vrais exces, methode du memoire), qui reste realiste. Voir script 14.
    """
    xi_lo, xi_hi = PARAMS[source]["xi_ic90"]
    sg_lo, sg_hi = PARAMS[source]["sigma_ic90"]
    xi = rng.normal((xi_lo + xi_hi) / 2, (xi_hi - xi_lo) / (2 * 1.645))
    sg = rng.normal((sg_lo + sg_hi) / 2, (sg_hi - sg_lo) / (2 * 1.645))
    return max(0.05, float(xi)), max(1.0, float(sg))


def oprisk_excesses():
    """Les exces reels OpRisk (M€) au-dessus du seuil, lus LOCALEMENT (data/raw, non pousse).

    Renvoie None si la donnee sous licence est absente. Perimetre cyber x finance,
    conversion USD->EUR, seuil u de la config (doit redonner ~91 exces, xi~0,595).
    """
    import os
    path = os.path.join(_REPO, "data", "raw", "SAS_OpRisk_Global_Data_June_2026.xlsx")
    if not os.path.exists(path):
        return None
    from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR
    cf = filter_finance(filter_cyber(load_clean(path)))
    losses = cf["loss"].values * USD_EUR
    u = PARAMS["OPRISK"]["u"]
    return losses[losses > u] - u


def bootstrap_sev_from_excesses(excesses, rng):
    """(xi, sigma) par reechantillonnage des exces reels + fit GPD (methode du memoire)."""
    from scipy.stats import genpareto
    s = rng.choice(excesses, size=len(excesses), replace=True)
    xi, _, sg = genpareto.fit(s, floc=0)
    return float(np.clip(xi, 0.05, 0.98)), max(1.0, float(sg))


def oprisk_losses():
    """Les pertes cyber x finance OpRisk (M€), lues LOCALEMENT. None si absente.

    Sert aux tests de robustesse au SEUIL POT (refit GPD a differents seuils).
    """
    import os
    path = os.path.join(_REPO, "data", "raw", "SAS_OpRisk_Global_Data_June_2026.xlsx")
    if not os.path.exists(path):
        return None
    from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR
    cf = filter_finance(filter_cyber(load_clean(path)))
    return cf["loss"].values * USD_EUR
