"""
scenarios/bootstrap_delta_dora.py
-----------------------------------
Bootstrap à DEUX NIVEAUX sur le Delta_DORA (Architecture 3 Briques) :

  Niveau 1 — incertitude des MULTIPLICATEURS de scénario (ENISA/Ponemon/
             Microsoft Research), tirage uniforme dans les fourchettes sourcées.
  Niveau 2 — incertitude des PARAMÈTRES DE SÉVÉRITÉ (ξ, σ) de la brique 
             Remédiation, par rééchantillonnage bootstrap des excès GPD.

⚠️ TRANSPARENCE MÉTHODOLOGIQUE :
  - OpRisk : bootstrap RÉEL, rééchantillonnage des 582 incidents bruts.
  - PRC : bootstrap RÉEL également, rééchantillonnage des excès de la
    sévérité dérivée Jacobs (2258 excès, voir src/severity/prc_analysis.py
    et notebooks/13_prc_jacobs_calibration.py).

Résultat final : la distribution de Delta_DORA calculée sur le modèle complet 
avec Copule (et non plus un simple processus de Poisson/GPD à plat).
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.stats import genpareto
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.frequency.negbin import compute_lambda_scenario
from src.utils.config import PRC, OPRISK, FREQUENCY, SCR_DORA
from src.aggregation.lda import simulate_year_3_briques
from src.compliance.latent import pcd_conditional, ANCHORED_PARAMS, PROFILS_TYPES


# ---------------------------------------------------------------------------
# 1. BOOTSTRAP GPD SUR DONNÉES RÉELLES (OpRisk uniquement)
# ---------------------------------------------------------------------------

def load_oprisk_excesses(u: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """Charge les excès réels OpRisk au-dessus du seuil (cyber×finance UNIQUEMENT)."""
    u = OPRISK["seuil_u_eur"] if u is None else float(u)

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    path = os.path.join(
        project_root, "data", "raw", "SAS_OpRisk_Global_Data_June_2026.xlsx"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier OpRisk introuvable : {path}\n"
            "Place la base sous data/raw/ (gitignorée car sous licence)."
        )

    df = pd.read_excel(path, sheet_name="Datasets")
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    candidate_loss_cols = [
        "Loss Amount ($M)", "Current Value of Loss ($M)",
        "Loss Amount (M)", "Current Value of Loss (M)",
    ]
    chosen_col = next((c for c in candidate_loss_cols if c in df.columns), None)
    if chosen_col is None:
        raise ValueError("Aucune colonne de perte reconnue dans 'Datasets'.")

    df["loss_musd"] = pd.to_numeric(df[chosen_col], errors="coerce")
    df = df[(df["loss_musd"] > 0) & (df["loss_musd"] < 100_000)]  

    mask_cyber = df["Sub Risk Category"].isin(["Systems Security", "Systems"])
    mask_fin = df["Industry Sector Name"].apply(
        lambda v: "Financial" in str(v) if pd.notna(v) else False
    )
    df_cyber = df[mask_cyber & mask_fin].copy()

    usd_eur = OPRISK.get("usd_eur", 0.92)
    losses_eur = df_cyber["loss_musd"].to_numpy(dtype=float) * usd_eur
    excesses = losses_eur[losses_eur > u] - u

    if len(excesses) == 0:
        raise ValueError(f"Aucun excès au-dessus du seuil u={u} M€.")

    return excesses, u

def load_prc_excesses(u: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """Charge les excès réels PRC (sévérité dérivée Jacobs) au-dessus du seuil."""
    u = PRC["seuil_u_eur"] if u is None else float(u)

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    path = os.path.join(project_root, "data", "raw", "Data_Breach_Chronology.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier PRC introuvable : {path}\n"
            "Place la base sous data/raw/ (gitignorée)."
        )

    from src.severity.prc_analysis import load_prc, jacobs_severity_eur_m

    df = load_prc(path)
    severities = jacobs_severity_eur_m(df["total_affected"].values)
    excesses = severities[severities > u] - u

    if len(excesses) == 0:
        raise ValueError(f"Aucun excès au-dessus du seuil u={u} M€.")

    return excesses, u


def bootstrap_gpd_params(excesses: np.ndarray, rng) -> tuple:
    """Un tirage bootstrap (ξ, σ) par rééchantillonnage des excès réels."""
    n = len(excesses)
    sample = rng.choice(excesses, size=n, replace=True)
    xi_b, _, sigma_b = genpareto.fit(sample, floc=0)
    return xi_b, sigma_b


# ---------------------------------------------------------------------------
# 2. BOOTSTRAP DEUX NIVEAUX BRANCHÉ SUR LE MODÈLE COMPLET (lda.py)
# ---------------------------------------------------------------------------

def bootstrap_delta_dora(source: str, scenario_x: str = "S2_non_conforme",
                         scenario_ref: str = "S0_conforme",
                         alpha: float = 0.995,
                         n_boot: int = 500,
                         n_sim_per_boot: int = 50_000,
                         seed: int = 42) -> dict:

    rng = np.random.default_rng(seed)
    dispersion = FREQUENCY["dispersion_factor"]

    # --- Setup Source ---
    if source == "PRC":
        lambda_ref = FREQUENCY["lambda_ref"]
        excesses, u0 = load_prc_excesses()
        xi_fit, _, sigma_fit = genpareto.fit(excesses, floc=0)
        xi0, sigma0 = xi_fit, sigma_fit
        p_u0 = PRC["p_u"]
        cap = SCR_DORA.get("cap_eur", 40.0)
        bootstrap_severity = True
    elif source == "OPRISK":
        lambda_ref = OPRISK["n_incidents"] / 27
        excesses, u0 = load_oprisk_excesses()
        xi_fit, _, sigma_fit = genpareto.fit(excesses, floc=0)
        xi0, sigma0 = xi_fit, sigma_fit
        p_u0 = OPRISK["p_u"]
        cap = None
        bootstrap_severity = True
    else:
        raise ValueError("source doit être 'PRC' ou 'OPRISK'")

    deltas, scr_ref_list, scr_x_list = [], [], []
    lambda_ref_list, lambda_x_list = [], []
    mult_x_list = []

    # Mapping rapide des scénarios vers des profils (pour la PCD Sanction)
    # Dans ton design, le ref = conforme (leader) et x = dégradé (retard/median)
    profil_map = {
        "S0_conforme": "leader",
        "S1_partiel": "median",
        "S2_non_conforme": "retard"
    }
    
    pcd_ref = pcd_conditional(PROFILS_TYPES[profil_map.get(scenario_ref, "leader")], 0.0, ANCHORED_PARAMS)
    pcd_x = pcd_conditional(PROFILS_TYPES[profil_map.get(scenario_x, "median")], 0.0, ANCHORED_PARAMS)

    for i in range(n_boot):
        # 1. Tirage des fréquences (Incertitude Multiplicateurs)
        res_ref = compute_lambda_scenario(lambda_ref, scenario_ref, mode="sample", rng=rng)
        res_x = compute_lambda_scenario(lambda_ref, scenario_x, mode="sample", rng=rng)

        lam_ref, lam_x = res_ref["lambda_global"], res_x["lambda_global"]
        lambda_ref_list.append(lam_ref)
        lambda_x_list.append(lam_x)

        if "multiplicateur_global" in res_x:
            mult_x_list.append(res_x["multiplicateur_global"])
        elif lambda_ref > 0:
            mult_x_list.append(lam_x / lambda_ref)
        else:
            mult_x_list.append(np.nan)

        # 2. Tirage des paramètres GPD (Incertitude Sévérité)
        if bootstrap_severity:
            xi_b, sigma_b = bootstrap_gpd_params(excesses, rng)
        else:
            xi_b, sigma_b = xi0, sigma0
            
        severity_params = {
            "xi": xi_b, "sigma": sigma_b, "u": u0, "p_u": p_u0,
            "dispersion_factor": dispersion, "severity_cap": cap
        }

        # 3. Simulation du modèle complet (3 Briques + Copule)
        # Neutralisation du biais de sélection : graine commune pour le contrefactuel dans la même itération
        graine_iteration = seed + i 
        
        sim_ref = simulate_year_3_briques(
            lambda_annual=lam_ref, severity_params=severity_params,
            pcd_sanction=pcd_ref, n_sim=n_sim_per_boot, dependence="gumbel", 
            theta=1.2, seed=graine_iteration
        )
        
        sim_x = simulate_year_3_briques(
            lambda_annual=lam_x, severity_params=severity_params,
            pcd_sanction=pcd_x, n_sim=n_sim_per_boot, dependence="gumbel", 
            theta=1.8, seed=graine_iteration
        )

        # 4. Calcul des VaR et du Delta
        scr_ref = np.quantile(sim_ref["total"], alpha)
        scr_x = np.quantile(sim_x["total"], alpha)

        scr_ref_list.append(scr_ref)
        scr_x_list.append(scr_x)
        deltas.append(scr_x - scr_ref)

    deltas = np.array(deltas)
    lambda_ref_arr = np.array(lambda_ref_list)
    lambda_x_arr = np.array(lambda_x_list)
    mult_x_arr = np.array(mult_x_list, dtype=float)

    result = {
        "source": source,
        "scenario_x": scenario_x,
        "scenario_ref": scenario_ref,
        "bootstrap_severity": bootstrap_severity,
        "n_boot": n_boot,
        "delta_median": float(np.median(deltas)),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
        "ic90": np.percentile(deltas, [5, 95]).tolist(),
        "ic95": np.percentile(deltas, [2.5, 97.5]).tolist(),
        "scr_ref_median": float(np.median(scr_ref_list)),
        "scr_x_median": float(np.median(scr_x_list)),
        "lambda_ref_median": float(np.median(lambda_ref_arr)),
        "lambda_x_median": float(np.median(lambda_x_arr)),
        "mult_x_median": float(np.nanmedian(mult_x_arr)),
        "mult_x_ic90": np.nanpercentile(mult_x_arr, [5, 95]).tolist(),
        "distribution": deltas,
    }

    n_obs = len(excesses) if excesses is not None else 0
    flag = f"✓ bootstrap réel ({n_obs} excès)" if bootstrap_severity else "⚠ point fixe (pas de données brutes ici)"
    print(f"\n{'='*62}")
    print(f"  Δ_DORA (Modèle Complet) — {source} — {scenario_ref} → {scenario_x}")
    print(f"{'='*62}")
    print(f"  Sévérité (niveau 2)        : {flag}")
    print(f"  Multiplicateurs (niveau 1) : ✓ bootstrap réel (fourchettes sourcées)")
    print(f"  λ {scenario_ref:18s} médian = {result['lambda_ref_median']:.3f}")
    print(f"  λ {scenario_x:18s} médian = {result['lambda_x_median']:.3f}")
    print(f"  Mult. {scenario_x:14s} médian = {result['mult_x_median']:.3f}")
    print(f"  Mult. {scenario_x:14s} IC90%  = [{result['mult_x_ic90'][0]:.3f} ; {result['mult_x_ic90'][1]:.3f}]")
    print(f"  n_boot = {n_boot} × n_sim = {n_sim_per_boot:,}")
    print(f"\n  SCR {scenario_ref:18s} (médiane) = {result['scr_ref_median']:>10.1f} M€")
    print(f"  SCR {scenario_x:18s} (médiane) = {result['scr_x_median']:>10.1f} M€")
    print(f"\n  Δ_DORA médiane = {result['delta_median']:.1f} M€")
    print(f"  Δ_DORA IC90%   = [{result['ic90'][0]:.1f} ; {result['ic90'][1]:.1f}] M€")
    print(f"  Δ_DORA IC95%   = [{result['ic95'][0]:.1f} ; {result['ic95'][1]:.1f}] M€")
    print(f"{'='*62}\n")

    return result


def full_bootstrap_grid(n_boot: int = 300, n_sim_per_boot: int = 30_000) -> pd.DataFrame:
    rows = []
    for source in ["PRC", "OPRISK"]:
        for scenario_x in ["S1_partiel", "S2_non_conforme"]:
            res = bootstrap_delta_dora(source, scenario_x,
                                       n_boot=n_boot, n_sim_per_boot=n_sim_per_boot)
            rows.append({
                "Source": source,
                "Scénario": scenario_x,
                "Δ médiane (M€)": res["delta_median"],
                "IC90% bas": res["ic90"][0],
                "IC90% haut": res["ic90"][1],
                "Bootstrap sévérité": "Oui" if res["bootstrap_severity"] else "Non (point fixe)",
            })

    df = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("  GRILLE FINALE — Δ_DORA (M€), deux sources x deux scénarios")
    print("="*70)
    print(df.to_string(index=False))
    print("\n  → Le Δ_DORA n'est jamais un nombre : il varie selon le scénario")
    print("    de conformité ET selon la source de sévérité retenue.")
    return df


if __name__ == "__main__":
    # Test rapide
    bootstrap_delta_dora("OPRISK", "S1_partiel", n_boot=200, n_sim_per_boot=30_000)
    bootstrap_delta_dora("PRC", "S1_partiel", n_boot=200, n_sim_per_boot=30_000)