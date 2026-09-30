"""
compliance/latent.py
--------------------
Variable latente de conformité DORA — modèle structurel de Merton-Vasicek.

OBJECTIF
========
Modéliser la probabilité qu'une entité d'assurance se retrouve en défaut de
conformité DORA à un instant t, comme une variable latente (score continu
[0,1]), conditionnellement à l'état de l'environnement cyber.

Cadre (cf. section réglementaire du mémoire, §3) :

    C*_{i,t} = β^T X_{i,t} + γ·Θ_t + sqrt(1-γ²)·Z_{i,t}

avec :
    C*_{i,t}  capacité de conformité latente de l'entité i (loi N(0,1))
    X_{i,t}   covariables observables de qualité du cadre TIC
    β         sensibilités associées
    Θ_t       facteur de risque systématique du secteur ~ N(0,1)
    γ         corrélation de conformité (sensibilité aux chocs systémiques)
    Z_{i,t}   choc idiosyncratique ~ N(0,1)

Le défaut survient si C*_{i,t} < K_i, avec K_i = Φ⁻¹(p_i).

La probabilité de défaut conditionnelle (point-in-time) à Θ_t = θ :

    PCD_i(θ) = Φ((Φ⁻¹(p_i) - β^T X_{i,t} - γ·θ) / sqrt(1-γ²))

STATUT MÉTHODOLOGIQUE
=====================
Ce module est un SIMULATEUR PARAMÉTRIQUE, pas un modèle estimé.

- DORA est entré en application le 17/01/2025 ; il n'existe pas encore de
  panel d'entités avec leurs métriques de conformité observées sur la durée.
- Les coefficients β, la corrélation γ et la probabilité de référence p_i
  ne sont donc pas estimés empiriquement : ce sont des valeurs illustratives
  ou ancrées sur proxies publics.
- Le module démontre la mécanique du modèle, pas un résultat calibré.

Tout chiffre produit ici doit être présenté dans le mémoire comme une
illustration du mécanisme, jamais comme une probabilité estimée pour une
entité réelle.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. PARAMÈTRES
# ---------------------------------------------------------------------------

ILLUSTRATIVE_PARAMS = {
    "covariates": {
        "art30_contracts_pct": ("% contrats tiers conformes Art. 30", 0.45),
        "rto_inverse": ("Rapidité de rétablissement (1/RTO)", 0.35),
        "phishing_resist": ("Taux de résistance au phishing", 0.40),
        "security_budget_ratio": ("Budget sécurité / CA", 0.30),
    },
    "gamma": 0.35,
    "p_ref": 0.08,
}

ANCHORED_PARAMS = {
    "covariates": ILLUSTRATIVE_PARAMS["covariates"],
    "gamma": 0.68,
    "p_ref": 0.50,
}

P_REF_BY_PILLAR = {
    "incident_mgmt": 1 - 0.48,
    "ict_risk_mgmt": 1 - 0.25,
    "resilience_test": 1 - 0.08,
    "third_party": 1 - 0.08,
    "roi_data_quality": 1 - 0.065,
}

SOURCES_ANCHORED = {
    "p_ref_global": "Industry surveys fin 2025 (~50% pleinement conformes) — doragrc.com",
    "p_ref_pillar": (
        "Deloitte European Survey 2025 (36 entités, 28 pays) ; "
        "ESA dry-run 2024 (6,5% RoI conformes sur ~1000 entités)"
    ),
    "gamma": (
        "Synergy Research / Canalys 2025 — AWS+Azure+GCP ≈ 68% du marché "
        "cloud mondial ; 87% multi-cloud, 73% hybride (Q1 2026)"
    ),
    "beta": (
        "NON estimés — pas de panel d'entités avec métriques observées "
        "(DORA trop récent). Valeurs illustratives conservées."
    ),
}


# ---------------------------------------------------------------------------
# 2. OUTILS DE VALIDATION
# ---------------------------------------------------------------------------

def validate_params(params: dict) -> None:
    gamma = params["gamma"]
    p_ref = params["p_ref"]

    if not (0 <= gamma < 1):
        raise ValueError(f"gamma doit appartenir à [0,1[ ; reçu {gamma}")
    if not (0 < p_ref < 1):
        raise ValueError(f"p_ref doit appartenir à ]0,1[ ; reçu {p_ref}")


# ---------------------------------------------------------------------------
# 3. PROFIL D'ENTITÉ
# ---------------------------------------------------------------------------

@dataclass
class EntityProfile:
    """
    Profil de conformité d'une entité, exprimé en covariables normalisées.
    0 = médiane marché ; valeurs positives = meilleur que la médiane.
    """
    name: str
    covariates: Dict[str, float] = field(default_factory=dict)

    def beta_dot_x(self, params: dict) -> float:
        total = 0.0
        for key, (_label, beta) in params["covariates"].items():
            x = self.covariates.get(key, 0.0)
            total += beta * x
        return total


PROFILS_TYPES = {
    "leader": EntityProfile(
        "Entité mature (leader)",
        {
            "art30_contracts_pct": 1.5,
            "rto_inverse": 1.2,
            "phishing_resist": 1.3,
            "security_budget_ratio": 1.0,
        },
    ),
    "median": EntityProfile(
        "Entité médiane",
        {
            "art30_contracts_pct": 0.0,
            "rto_inverse": 0.0,
            "phishing_resist": 0.0,
            "security_budget_ratio": 0.0,
        },
    ),
    "retard": EntityProfile(
        "Entité en retard",
        {
            "art30_contracts_pct": -1.3,
            "rto_inverse": -1.0,
            "phishing_resist": -1.2,
            "security_budget_ratio": -0.8,
        },
    ),
}


# ---------------------------------------------------------------------------
# 4. PROBABILITÉ DE DÉFAUT DE CONFORMITÉ
# ---------------------------------------------------------------------------

def threshold_K(p_ref: float) -> float:
    return norm.ppf(p_ref)


def pcd_conditional(
    entity: EntityProfile,
    theta: float,
    params: Optional[dict] = None,
) -> float:
    params = params or ILLUSTRATIVE_PARAMS
    validate_params(params)

    gamma = params["gamma"]
    K = threshold_K(params["p_ref"])
    bx = entity.beta_dot_x(params)

    numerator = K - bx - gamma * theta
    denominator = np.sqrt(1.0 - gamma**2)
    return float(norm.cdf(numerator / denominator))


def pcd_unconditional(
    entity: EntityProfile,
    params: Optional[dict] = None,
    n_sim: int = 100_000,
    seed: int = 42,
) -> dict:
    params = params or ILLUSTRATIVE_PARAMS
    validate_params(params)

    rng = np.random.default_rng(seed)
    thetas = rng.standard_normal(n_sim)
    pcds = np.array([pcd_conditional(entity, t, params) for t in thetas])

    return {
        "entity": entity.name,
        "pcd_mean": float(pcds.mean()),
        "pcd_median": float(np.median(pcds)),
        "pcd_p95": float(np.percentile(pcds, 95)),
        "pcd_p99": float(np.percentile(pcds, 99)),
        "distribution": pcds,
    }


# ---------------------------------------------------------------------------
# 5. EFFET D'AMPLIFICATION SYSTÉMIQUE
# ---------------------------------------------------------------------------

def systemic_amplification(
    entity: EntityProfile,
    gammas: Optional[List[float]] = None,
    theta_crisis: float = -2.5,
    params: Optional[dict] = None,
) -> Dict[float, dict]:
    params = params or ILLUSTRATIVE_PARAMS
    gammas = gammas or [0.1, 0.3, 0.5, 0.7, 0.9]

    out = {}
    for g in gammas:
        p = dict(params)
        p["gamma"] = g
        validate_params(p)

        pcd_normal = pcd_conditional(entity, 0.0, p)
        pcd_crisis = pcd_conditional(entity, theta_crisis, p)
        out[g] = {
            "pcd_normal": pcd_normal,
            "pcd_crisis": pcd_crisis,
            "amplification": pcd_crisis / pcd_normal if pcd_normal > 0 else np.inf,
        }
    return out


# ---------------------------------------------------------------------------
# 6. RAPPORTS
# ---------------------------------------------------------------------------

def full_report(params: Optional[dict] = None) -> None:
    params = params or ILLUSTRATIVE_PARAMS
    validate_params(params)

    print("=" * 64)
    print(" VARIABLE LATENTE DE CONFORMITÉ DORA — modèle Merton-Vasicek")
    print(" (SIMULATEUR PARAMÉTRIQUE — paramètres illustratifs, non estimés)")
    print("=" * 64)
    print(f" γ (corrélation systémique) = {params['gamma']}")
    print(f" p_i (défaut de référence)  = {params['p_ref']:.1%}")
    print(f" Seuil K = Φ⁻¹(p_i)         = {threshold_K(params['p_ref']):.4f}")

    print("\n --- PCD par profil d'entité ---")
    print(f" {'Profil':28s} {'PCD(θ=0)':>10s} {'PCD moy.':>10s} {'PCD crise':>11s}")
    for entity in PROFILS_TYPES.values():
        pcd0 = pcd_conditional(entity, 0.0, params)
        unc = pcd_unconditional(entity, params, n_sim=50_000)
        pcd_crisis = pcd_conditional(entity, -2.5, params)
        print(f" {entity.name:28s} {pcd0:>9.2%} {unc['pcd_mean']:>9.2%} {pcd_crisis:>10.2%}")

    print("\n --- Amplification systémique (entité médiane) ---")
    print(" Effet de γ sur le ratio PCD(crise)/PCD(normal) :")
    amp = systemic_amplification(PROFILS_TYPES["median"], params=params)
    print(f" {'γ':>5s} {'PCD normal':>12s} {'PCD crise':>12s} {'amplif.':>10s}")
    for g, r in amp.items():
        print(f" {g:>5.1f} {r['pcd_normal']:>11.2%} {r['pcd_crisis']:>11.2%} {r['amplification']:>9.1f}x")

    print("\n → Plus γ est élevé, plus une crise cyber systémique amplifie la probabilité de défaut.")
    print("   C'est le lien quantitatif avec la contagion externe (tiers ICT).")
    print("=" * 64)


def report_anchored() -> None:
    print("=" * 66)
    print(" VARIABLE LATENTE DORA — PARAMÈTRES ANCRÉS (Option B)")
    print(" Proxies sourcés — voir SOURCES_ANCHORED")
    print("=" * 66)
    print(f" γ (proxy concentration cloud) = {ANCHORED_PARAMS['gamma']}")
    print(" [AWS+Azure+GCP ≈ 68% du marché — Synergy/Canalys 2025]")
    print(f" p_ref global = {ANCHORED_PARAMS['p_ref']:.0%}")
    print(" [~50% pleinement conformes fin 2025 — industry surveys]")

    print("\n --- PCD de référence PAR PILIER DORA (entité médiane, θ=0) ---")
    print(f" {'Pilier':22s} {'Non-conf.':>10s} {'PCD(θ=0)':>10s} {'PCD crise':>11s}")
    median = PROFILS_TYPES["median"]
    for pillar, p_ref in P_REF_BY_PILLAR.items():
        params = dict(ANCHORED_PARAMS)
        params["p_ref"] = p_ref
        pcd0 = pcd_conditional(median, 0.0, params)
        pcd_crisis = pcd_conditional(median, -2.5, params)
        print(f" {pillar:22s} {p_ref:>9.1%} {pcd0:>9.1%} {pcd_crisis:>10.1%}")

    print("\n Lecture : les piliers Tests de résilience et Risque tiers dominent le risque de défaut DORA.")
    print(" En cas de crise systémique avec γ élevé, la bascule en défaut devient massive.")
    print("=" * 66)


def compare_A_vs_B() -> None:
    median = PROFILS_TYPES["median"]

    print("=" * 70)
    print(" COMPARAISON A (illustratif) vs B (ancré sur données)")
    print(" Entité médiane — effet du choix de paramètres sur la PCD")
    print("=" * 70)

    print(f"\n {'Paramètre':28s} {'A (illustratif)':>18s} {'B (ancré)':>18s}")
    print(f" {'-' * 64}")
    print(
        f" {'γ (corrélation systémique)':28s} "
        f"{ILLUSTRATIVE_PARAMS['gamma']:>18.2f} "
        f"{ANCHORED_PARAMS['gamma']:>18.2f}"
    )
    print(
        f" {'p_ref (non-conformité)':28s} "
        f"{ILLUSTRATIVE_PARAMS['p_ref']:>17.0%} "
        f"{ANCHORED_PARAMS['p_ref']:>17.0%}"
    )

    pcd0_A = pcd_conditional(median, 0.0, ILLUSTRATIVE_PARAMS)
    pcd0_B = pcd_conditional(median, 0.0, ANCHORED_PARAMS)
    pcdc_A = pcd_conditional(median, -2.5, ILLUSTRATIVE_PARAMS)
    pcdc_B = pcd_conditional(median, -2.5, ANCHORED_PARAMS)

    print(f"\n {'PCD (environnement normal)':28s} {pcd0_A:>17.1%} {pcd0_B:>17.1%}")
    print(f" {'PCD (crise systémique)':28s} {pcdc_A:>17.1%} {pcdc_B:>17.1%}")

    print("\n Lecture :")
    print(" • B donne une PCD plus élevée que A car les proxies de non-conformité fin 2025 sont nettement plus sévères.")
    print(" • Sous B, la dépendance cloud forte accroît fortement l'effet d'une crise systémique.")
    print(" • L'écart A↔B mesure la sensibilité de la conclusion aux hypothèses.")
    print("=" * 70)


if __name__ == "__main__":
    full_report()
    print()
    report_anchored()
    print()
    compare_A_vs_B()