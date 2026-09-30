"""
aggregation/euler_allocation.py
-------------------------------
Allocation d'Euler du SCR agrégé entre les 3 briques physiques du modèle LDA :
remédiation, prestataire, sanction.

PRINCIPE
========
On part d'un vecteur de pertes annuelles simulées par brique :
    X = (X_rem, X_presta, X_sanc)
et de la perte totale :
    L = X_rem + X_presta + X_sanc

Pour une mesure de risque de type VaR_alpha, l'allocation d'Euler approchée
sur simulation consiste à conditionner sur les années situées dans une bande
autour du quantile :
    contribution_i ≈ E[X_i | L ≈ VaR_alpha(L)]

La somme des contributions est ensuite renormalisée pour sommer exactement
au SCR total estimé.

L'Aggravation (le delta contrefactuel) n'est pas allouée ici car elle n'est 
pas une composante additive de L, mais une différence entre deux VaR(L).
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.aggregation.lda import simulate_year_3_briques
from src.compliance.latent import pcd_conditional, ANCHORED_PARAMS, PROFILS_TYPES
from src.utils.config import PRC, OPRISK, FREQUENCY, SCR_DORA


# On retire 'aggravation' de la liste des briques additives
BRIQUES = ["remediation", "prestataire", "sanction"]


def var_empirical(losses: np.ndarray, alpha: float = 0.995) -> float:
    """VaR empirique au niveau alpha."""
    return float(np.quantile(losses, alpha))


def expected_shortfall(losses: np.ndarray, alpha: float = 0.995) -> float:
    """ES empirique au niveau alpha."""
    var = var_empirical(losses, alpha)
    tail = losses[losses >= var]
    if len(tail) == 0:
        return var
    return float(np.mean(tail))


def _tail_band_mask(losses: np.ndarray, alpha: float, bandwidth: float) -> np.ndarray:
    """
    Sélectionne une bande locale autour de la VaR.
    Ex. alpha=0.995, bandwidth=0.002  -> garde les pertes entre q(99.3%) et q(99.7%).
    """
    low_q = max(0.0, alpha - bandwidth)
    high_q = min(1.0, alpha + bandwidth)

    q_low = np.quantile(losses, low_q)
    q_high = np.quantile(losses, high_q)

    mask = (losses >= q_low) & (losses <= q_high)

    # garde-fou : si bande trop étroite, prendre au moins les points >= VaR
    if mask.sum() < 20:
        var = np.quantile(losses, alpha)
        mask = losses >= var

    return mask


def euler_allocation_var(
    simulation_result: dict,
    alpha: float = 0.995,
    bandwidth: float = 0.002,
) -> dict:
    """
    Allocation d'Euler approchée de la VaR par moyenne conditionnelle locale.
    """
    total = np.asarray(simulation_result["total"], dtype=float)
    var_total = var_empirical(total, alpha=alpha)

    mask = _tail_band_mask(total, alpha=alpha, bandwidth=bandwidth)
    local_n = int(mask.sum())

    raw_contrib = {}
    for b in BRIQUES:
        xb = np.asarray(simulation_result[b], dtype=float)
        raw_contrib[b] = float(np.mean(xb[mask]))

    raw_sum = sum(raw_contrib.values())

    if raw_sum <= 0:
        contrib = {b: 0.0 for b in BRIQUES}
    else:
        contrib = {b: raw_contrib[b] * var_total / raw_sum for b in BRIQUES}

    pct = {b: (contrib[b] / var_total if var_total > 0 else 0.0) for b in BRIQUES}

    return {
        "measure": "VaR",
        "alpha": alpha,
        "scr_total": var_total,
        "local_sample_size": local_n,
        "bandwidth": bandwidth,
        "contribution": contrib,
        "percent": pct,
        "raw_local_mean": raw_contrib,
    }


def euler_allocation_es(
    simulation_result: dict,
    alpha: float = 0.995,
) -> dict:
    """
    Allocation naturelle sur l'Expected Shortfall.
    """
    total = np.asarray(simulation_result["total"], dtype=float)
    var = var_empirical(total, alpha=alpha)
    es = expected_shortfall(total, alpha=alpha)

    mask = total >= var
    tail_n = int(mask.sum())

    contrib = {}
    for b in BRIQUES:
        xb = np.asarray(simulation_result[b], dtype=float)
        contrib[b] = float(np.mean(xb[mask]))

    pct = {b: (contrib[b] / es if es > 0 else 0.0) for b in BRIQUES}

    return {
        "measure": "ES",
        "alpha": alpha,
        "scr_total": es,
        "tail_sample_size": tail_n,
        "contribution": contrib,
        "percent": pct,
    }


def print_euler_report(allocation: dict) -> None:
    """Affichage lisible des contributions d'Euler."""
    print("=" * 72)
    print(f" ALLOCATION D'EULER — {allocation['measure']} {allocation['alpha']:.3%}")
    print("=" * 72)
    print(f" Capital agrégé : {allocation['scr_total']:,.1f} M€")

    if allocation["measure"] == "VaR":
        print(f" Bande locale : ±{allocation['bandwidth']:.3%}")
        print(f" Observations retenues : {allocation['local_sample_size']:,}")
    else:
        print(f" Observations de queue : {allocation['tail_sample_size']:,}")

    print("\n Décomposition par brique :")
    print(f" {'Brique':14s} {'Contribution':>15s} {'Part du capital':>18s}")
    print(f" {'-'*52}")
    for b in BRIQUES:
        c = allocation["contribution"][b]
        p = allocation["percent"][b]
        print(f" {b.capitalize():14s} {c:>14,.1f} M€ {p:>17.1%}")

    print(f" {'-'*52}")
    print(f" {'TOTAL':14s} {sum(allocation['contribution'].values()):>14,.1f} M€ {sum(allocation['percent'].values()):>17.1%}")
    print("=" * 72)


def build_severity_params(source: str) -> dict:
    """Construit les paramètres de sévérité à transmettre à lda.py."""
    source = source.upper()
    if source == "PRC":
        src = PRC
        severity_cap = SCR_DORA.get("cap_eur", 40.0)
    elif source == "OPRISK":
        src = OPRISK
        severity_cap = None
    else:
        raise ValueError("source doit être 'PRC' ou 'OPRISK'")

    return {
        "xi": src["xi"],
        "sigma": src["sigma_eur"],
        "u": src["seuil_u_eur"],
        "p_u": src["p_u"],
        "dispersion_factor": FREQUENCY["dispersion_factor"],
        "severity_cap": severity_cap,
    }


def run_euler_on_reference(
    lambda_annual: float,
    source: str = "OPRISK",
    alpha: float = 0.995,
    n_sim: int = 100_000,
    dependence: str = "gumbel",
    theta_env: float = 0.0,
    entity_key: str = "median",
    method: str = "var",
    bandwidth: float = 0.002,
    seed: int = 42,
) -> dict:
    """
    Lance la simulation LDA à 3 briques physiques puis calcule l'allocation d'Euler.
    """
    entity = PROFILS_TYPES[entity_key]
    severity_params = build_severity_params(source)
    pcd = pcd_conditional(entity, theta_env, ANCHORED_PARAMS)

    # Appel de la nouvelle fonction à 3 briques
    sim = simulate_year_3_briques(
        lambda_annual=lambda_annual,
        severity_params=severity_params,
        pcd_sanction=pcd,
        n_sim=n_sim,
        dependence=dependence,
        seed=seed,
    )

    if method.lower() == "var":
        alloc = euler_allocation_var(sim, alpha=alpha, bandwidth=bandwidth)
    elif method.lower() == "es":
        alloc = euler_allocation_es(sim, alpha=alpha)
    else:
        raise ValueError("method doit être 'var' ou 'es'")

    alloc["meta"] = {
        "source": source.upper(),
        "lambda_annual": lambda_annual,
        "dependence": dependence,
        "entity": entity.name,
        "theta_env": theta_env,
        "pcd_sanction": pcd,
        "n_sim": n_sim,
    }
    return alloc


if __name__ == "__main__":
    # Test d'allocation sur une fréquence de référence OPRISK
    lambda_ref_oprisk = OPRISK["n_incidents"] / 27

    alloc_var = run_euler_on_reference(
        lambda_annual=lambda_ref_oprisk,
        source="OPRISK",
        alpha=0.995,
        n_sim=100_000,
        dependence="gumbel",
        theta_env=0.0,
        entity_key="median",
        method="var",
        bandwidth=0.002,
        seed=42,
    )

    print("\n>>> Allocation d'Euler sur VaR\n")
    print(f"Contexte : {alloc_var['meta']}")
    print_euler_report(alloc_var)