"""
frequency/negbin.py
--------------------
Modèle de fréquence NegBin (Negative Binomial) calibré sur la PRC 2025.
Scénarios de fréquence conditionnels au niveau de conformité DORA.

Justification NegBin : surdispersion observée (Var/Mean >> 1) dans les
données cyber — le processus de Poisson simple est rejeté.

Références :
  Farkas, Lopez & Thomas (2021)
  Kher, Lopez & Rapior (2023)
"""

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. CALIBRATION NEGBIN PAR MLE
# ---------------------------------------------------------------------------

def fit_negbin(counts: np.ndarray) -> dict:
    """
    Calibre la loi NegBin(r, p) sur une série de comptages mensuels.
    Paramétrage : E[X] = r(1-p)/p, Var[X] = r(1-p)/p²

    Parameters
    ----------
    counts : array d'entiers (nombre d'incidents par période)

    Returns
    -------
    dict : r, p, mu (moyenne), sigma2 (variance), dispersion_factor
    """
    mu_hat = np.mean(counts)
    var_hat = np.var(counts, ddof=1)

    if var_hat <= mu_hat:
        raise ValueError(
            f"Pas de surdispersion détectée (Var={var_hat:.2f} ≤ Mu={mu_hat:.2f}). "
            "Vérifier les données ou utiliser Poisson."
        )

    # Méthode des moments pour initialisation
    r_init = mu_hat ** 2 / (var_hat - mu_hat)
    p_init = mu_hat / var_hat

    # MLE
    def neg_loglik(params):
        r, p = params
        if r <= 0 or p <= 0 or p >= 1:
            return 1e10
        return -np.sum(nbinom.logpmf(counts.astype(int), r, p))

    result = minimize(neg_loglik, x0=[r_init, p_init],
                      method="Nelder-Mead",
                      options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 10000})

    r, p = result.x
    mu = r * (1 - p) / p
    sigma2 = r * (1 - p) / p ** 2
    dispersion = sigma2 / mu  # facteur de surdispersion (= 1 pour Poisson)

    params = {
        "r": r, "p": p,
        "mu": mu, "sigma2": sigma2,
        "dispersion_factor": dispersion,
        "n_periods": len(counts),
        "log_lik": -result.fun,
    }

    print(f"\n=== CALIBRATION NegBin ===")
    print(f"  r  = {r:.4f}")
    print(f"  p  = {p:.4f}")
    print(f"  μ  = {mu:.2f} incidents/période")
    print(f"  σ² = {sigma2:.2f}")
    print(f"  Facteur de surdispersion = {dispersion:.2f}  (Poisson = 1)")
    print(f"  Périodes calibrées       = {len(counts)}\n")

    return params


# ---------------------------------------------------------------------------
# 2. MAPPING VECTEURS D'ATTAQUE → MULTIPLICATEURS DORA
# ---------------------------------------------------------------------------

# Proportions par vecteur (source : Hackmageddon S1 2026, 1 041 incidents)
HACKMAGEDDON_PROPORTIONS = {
    "phishing_social_eng": 0.388,       # Art. 13 — Formation
    "exploit_vuln":        0.338,       # Art. 26 — TLPT
    "supply_chain_tiers":  0.158,       # Art. 28-44 — Tiers ICT
    "identifiants":        0.063,       # Art. 9 — Accès/MFA
    "autres":              0.053,       # Transverse
}

# Articles DORA par vecteur
DORA_MAPPING = {
    "phishing_social_eng": "Art. 13 — Sensibilisation & formation",
    "exploit_vuln":        "Art. 26 — Tests TLPT ; Art. 8-10",
    "supply_chain_tiers":  "Art. 28-44 — Gestion risque tiers ICT",
    "identifiants":        "Art. 9 — Contrôle d'accès & MFA",
    "autres":              "Transverse",
}

# Multiplicateurs de fréquence par vecteur et par niveau de conformité.
#
# FOURCHETTES DE SCÉNARIO POSÉES PAR L'AUTEUR. Les parts d'attaque sont
# citées ; aucune des références ci-dessous ne mesure un effet de conformité
# DORA sur la fréquence d'incidents d'une entité. Les bornes S1/S2 ne sont
# donc pas des intervalles de confiance ni une calibration de cet effet.
# IBM/Ponemon décrit des coûts ; ENISA compare des vecteurs d'attaque ;
# SecurityScorecard décrit la part des brèches d'origine tierce ; Meyer et al.
# mesurent la compromission par compte. Ces objets ne fixent pas nos bornes.
# Relecture des sources : docs/audits/2026-11-05_audit_jury_institut.md, § 8.

SOURCES_MULTIPLICATEURS = {
    "phishing_social_eng": "Ponemon/IBM Cost of a Data Breach 2025 — "
                            "formation : effet sur le coût, non sur la fréquence DORA",
    "exploit_vuln":        "ENISA Threat Landscape 2025 — conversion "
                            "intrusion 70% (exploit) vs 27% (phishing), "
                            "sans effet de conformité estimé",
    "supply_chain_tiers":  "IBM/Ponemon 2025 — coût des brèches ; "
                            "SecurityScorecard 2025 — part tierce 35.5% en 2024, "
                            "sans effet de conformité estimé",
    "identifiants":        "Microsoft Research, arXiv:2305.00945 — MFA "
                            "réduit compromission de 99.22% (population), "
                            "98.56% (identifiants déjà fuités)",
    "autres":              "Pas de source dédiée — multiplicateur transverse "
                            "prudent, borné par les autres vecteurs",
}

MULTIPLICATEURS_DORA = {
    "S0_conforme": {
        vecteur: (1.0, 1.0) for vecteur in HACKMAGEDDON_PROPORTIONS
    },
    "S1_partiel": {
        # (borne basse, borne haute) — non-conformité PARTIELLE
        "phishing_social_eng": (1.2, 1.6),   # fourchette posée par l'auteur
        "exploit_vuln":        (1.3, 1.8),   # fourchette posée par l'auteur
        "supply_chain_tiers":  (1.3, 1.9),   # fourchette posée par l'auteur
        "identifiants":        (1.5, 3.0),   # fourchette posée, effet MFA par compte
        "autres":              (1.1, 1.3),
    },
    "S2_non_conforme": {
        # non-conformité TOTALE — bornes posées par l'auteur
        "phishing_social_eng": (1.8, 3.0),
        "exploit_vuln":        (2.0, 2.6),   # jusqu'au ratio ENISA complet
        "supply_chain_tiers":  (1.9, 2.6),
        "identifiants":        (3.0, 8.0),   # toujours tempéré vs 99%+ Microsoft
        "autres":              (1.3, 1.6),
    },
}


def sample_multiplicateur(scenario: str, vecteur: str, rng=None) -> float:
    """
    Tire uniformément dans la fourchette [low, high] posée par l'auteur.
    Ce tirage explore une incertitude de scénario ; il ne constitue pas
    un bootstrap d'un effet de conformité estimé.
    """
    rng = rng or np.random
    low, high = MULTIPLICATEURS_DORA[scenario][vecteur]
    return rng.uniform(low, high)


def compute_lambda_scenario(lambda_ref: float, scenario: str,
                             mode: str = "center", rng=None) -> dict:
    """
    Calcule le λ effectif pour chaque vecteur sous un scénario DORA.
    Le λ global est la somme pondérée des λ par vecteur.

    Parameters
    ----------
    lambda_ref : fréquence de référence (scénario S0 conforme)
    scenario   : 'S0_conforme', 'S1_partiel', 'S2_non_conforme'
    mode       : 'center' (point central de la fourchette posée, déterministe)
                 ou 'sample' (tirage uniforme dans la fourchette de scénario)
    rng        : générateur numpy (utilisé seulement si mode='sample')

    Returns
    -------
    dict : lambda_par_vecteur, lambda_global, multiplicateur_global
    """
    if scenario not in MULTIPLICATEURS_DORA:
        raise ValueError(f"Scénario inconnu : {scenario}. "
                         f"Choisir parmi {list(MULTIPLICATEURS_DORA.keys())}")
    if mode not in ("center", "sample"):
        raise ValueError("mode doit être 'center' ou 'sample'")

    bounds = MULTIPLICATEURS_DORA[scenario]
    props = HACKMAGEDDON_PROPORTIONS
    rng = rng or np.random

    lambda_vecteur = {}
    lambda_global = 0.0

    for v, prop in props.items():
        low, high = bounds[v]
        m = rng.uniform(low, high) if mode == "sample" else (low + high) / 2
        lv = lambda_ref * prop * m
        lambda_vecteur[v] = lv
        lambda_global += lv

    mult_global = lambda_global / lambda_ref

    return {
        "scenario": scenario,
        "mode": mode,
        "lambda_ref": lambda_ref,
        "lambda_global": lambda_global,
        "multiplicateur_global": mult_global,
        "lambda_par_vecteur": lambda_vecteur,
    }


def scenario_comparison(lambda_ref: float) -> pd.DataFrame:
    """
    Compare les trois scénarios de conformité DORA.

    Returns
    -------
    DataFrame : un scénario par ligne
    """
    rows = []
    for sc in ["S0_conforme", "S1_partiel", "S2_non_conforme"]:
        res = compute_lambda_scenario(lambda_ref, sc)
        rows.append({
            "Scénario": sc,
            "λ_global": res["lambda_global"],
            "Multiplicateur": res["multiplicateur_global"],
            "Δλ vs S0": res["lambda_global"] - lambda_ref,
        })

    df = pd.DataFrame(rows)
    print("\n=== SCÉNARIOS DE CONFORMITÉ DORA ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSource multiplicateurs : Hackmageddon S1 2026 (1 041 incidents)")
    print("Fourchettes posées par l'auteur ; sources voisines, sans effet DORA estimé :")
    print("ENISA 2025, IBM/Ponemon 2025, SecurityScorecard 2025, "
          "Microsoft Research arXiv:2305.00945 — voir SOURCES_MULTIPLICATEURS\n")
    return df


# ---------------------------------------------------------------------------
# 3. SIMULATION NEGBIN SOUS SCÉNARIO DORA
# ---------------------------------------------------------------------------

def simulate_frequency(params: dict, scenario: str,
                        lambda_ref: float,
                        n_sim: int = 1_000_000,
                        seed: int = 42) -> np.ndarray:
    """
    Simule n_sim réalisations de la fréquence annuelle sous un scénario DORA.

    Parameters
    ----------
    params     : dict issu de fit_negbin()
    scenario   : scénario DORA
    lambda_ref : fréquence de référence annuelle
    n_sim      : nombre de simulations
    seed       : graine aléatoire

    Returns
    -------
    array de n_sim entiers (nombre d'incidents simulés)
    """
    if not np.isfinite(lambda_ref) or lambda_ref <= 0:
        raise ValueError("lambda_ref doit être une fréquence annuelle strictement positive")
    sc_res = compute_lambda_scenario(lambda_ref, scenario)

    # Conserver la forme r et recaler la moyenne sur la fréquence ANNUELLE.
    # params["mu"] est la moyenne par période de calibration (souvent mensuelle).
    r = params["r"]
    mu_new = sc_res["lambda_global"]
    p_new = r / (r + mu_new)

    counts = nbinom.rvs(r, p_new, size=n_sim, random_state=seed)
    return counts


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module negbin.py chargé.")
    print()
    print("Exemple :")
    print("  from src.frequency.negbin import fit_negbin, scenario_comparison")
    print("  counts = np.array([120, 145, 130, ...])  # incidents mensuels")
    print("  params = fit_negbin(counts)")
    print("  df = scenario_comparison(lambda_ref=params['mu'] * 12)")
