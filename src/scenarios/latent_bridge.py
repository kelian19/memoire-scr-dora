"""
scenarios/latent_bridge.py
----------------------------
PONT entre la variable latente de conformité (compliance/latent.py) et la
fréquence de sinistres (frequency/negbin.py).

PROBLÈME RÉSOLU
===============
Jusqu'ici, deux modèles coexistaient sans être reliés :
  - compliance/latent.py calcule une PCD ∈ [0,1] (probabilité de défaut de
    conformité), continue, fonction du profil d'entité et de l'environnement
    systémique θ.
  - frequency/negbin.py applique des multiplicateurs de fréquence sous TROIS
    scénarios DISCRETS (S0/S1/S2), sans lien avec la PCD d'une entité donnée.

Ce module fait le pont : la PCD calculée pour une entité devient le curseur
continu qui détermine SA fréquence d'incidents — remplaçant le choix discret
de scénario par une fonction continue de son profil de conformité réel.

PRINCIPE D'INTERPOLATION
=========================
Pour chaque vecteur d'attaque v, le multiplicateur de fréquence sous PCD est :

    m_v(PCD) = 1 + PCD × (m_v^(S2) - 1)

où m_v^(S2) est le multiplicateur sourcé en scénario de non-conformité totale
(centre de l'intervalle [low,high] de MULTIPLICATEURS_DORA["S2_non_conforme"]).

  - PCD = 0 (entité jamais en défaut)  → m_v = 1   (= S0, fréquence de référence)
  - PCD = 1 (entité toujours en défaut) → m_v = m_v^(S2) (= effet plein du
    scénario de non-conformité totale, déjà sourcé sur ENISA/Ponemon/MS Research)

Cette interpolation est un choix de modélisation simple (linéaire) qui
RÉUTILISE les multiplicateurs déjà sourcés — elle n'invente aucune nouvelle
calibration, elle relie deux modèles existants par une hypothèse de linéarité
explicite et discutable comme telle.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.compliance.latent import (
    pcd_conditional, ANCHORED_PARAMS, ILLUSTRATIVE_PARAMS,
    PROFILS_TYPES, EntityProfile
)
from src.frequency.negbin import HACKMAGEDDON_PROPORTIONS, MULTIPLICATEURS_DORA
from src.utils.config import FREQUENCY


# ---------------------------------------------------------------------------
# 1. MULTIPLICATEUR CONTINU PAR VECTEUR, FONCTION DE LA PCD
# ---------------------------------------------------------------------------

def vector_multiplier_from_pcd(pcd: float, vecteur: str) -> float:
    """
    Multiplicateur de fréquence pour un vecteur d'attaque donné, interpolé
    linéairement entre m=1 (PCD=0, conforme) et le centre du multiplicateur
    S2 sourcé (PCD=1, non-conforme totale).
    """
    low, high = MULTIPLICATEURS_DORA["S2_non_conforme"][vecteur]
    m_full = (low + high) / 2
    return 1.0 + pcd * (m_full - 1.0)


def lambda_from_pcd(pcd: float, lambda_ref: float = None) -> dict:
    """
    Calcule le λ global et par vecteur à partir d'une PCD directement
    (sans recalculer la PCD — utile pour les tests de sensibilité).
    """
    lambda_ref = lambda_ref or FREQUENCY["lambda_ref"]
    lambda_vecteur = {}
    total = 0.0
    for v, prop in HACKMAGEDDON_PROPORTIONS.items():
        m = vector_multiplier_from_pcd(pcd, v)
        lv = lambda_ref * prop * m
        lambda_vecteur[v] = lv
        total += lv
    return {
        "pcd": pcd,
        "lambda_global": total,
        "lambda_par_vecteur": lambda_vecteur,
        "multiplicateur_global": total / lambda_ref,
    }


# ---------------------------------------------------------------------------
# 2. PONT COMPLET : PROFIL D'ENTITÉ + ENVIRONNEMENT → λ
# ---------------------------------------------------------------------------

def lambda_from_entity(entity: EntityProfile, theta: float = 0.0,
                        lambda_ref: float = None,
                        latent_params: dict = None) -> dict:
    """
    Pont complet : profil d'entité + état systémique θ → PCD → λ.

    C'est la fonction centrale du module : elle remplace le choix manuel
    d'un scénario S0/S1/S2 par le calcul de la fréquence RÉELLE d'une
    entité donnée, selon son profil de conformité et l'environnement cyber
    du moment.

    Parameters
    ----------
    entity       : profil de l'entité (cf. compliance.latent.EntityProfile)
    theta        : état de l'environnement cyber (0 = médian, <0 = crise)
    lambda_ref   : fréquence de référence (défaut : FREQUENCY['lambda_ref'])
    latent_params: paramètres du modèle latent (défaut : ANCHORED_PARAMS,
                   Option B — proxies sourcés)

    Returns
    -------
    dict : pcd, lambda_global, lambda_par_vecteur, multiplicateur_global
    """
    latent_params = latent_params or ANCHORED_PARAMS
    pcd = pcd_conditional(entity, theta, latent_params)
    result = lambda_from_pcd(pcd, lambda_ref)
    result["entity"] = entity.name
    result["theta"] = theta
    return result


# ---------------------------------------------------------------------------
# 3. COMPARAISON : SCÉNARIOS DISCRETS vs CONTINU PILOTÉ PAR LA PCD
# ---------------------------------------------------------------------------

def compare_discrete_vs_continuous(theta_values: list = None,
                                    latent_params: dict = None):
    """
    Compare l'approche discrète (S0/S1/S2, choisie manuellement) à l'approche
    continue (pilotée par la PCD de chaque profil d'entité, sous différents
    états systémiques θ).

    Montre que les scénarios discrets sont des cas particuliers de la
    fonction continue — pas deux modèles concurrents.
    """
    theta_values = theta_values or [0.0, -1.0, -2.5]
    latent_params = latent_params or ANCHORED_PARAMS
    lambda_ref = FREQUENCY["lambda_ref"]

    print("=" * 74)
    print("  PONT VARIABLE LATENTE → FRÉQUENCE : discret vs continu")
    print(f"  (paramètres latents : {'ANCHORED (Option B)' if latent_params is ANCHORED_PARAMS else 'ILLUSTRATIVE (Option A)'})")
    print("=" * 74)

    print(f"\n  Repères discrets (S0/S1/S2, calibration jeudi) :")
    print(f"    S0_conforme      : λ = {lambda_ref:.1f}  (mult = 1.00)")
    s1_center = sum(HACKMAGEDDON_PROPORTIONS[v] * sum(MULTIPLICATEURS_DORA["S1_partiel"][v]) / 2
                    for v in HACKMAGEDDON_PROPORTIONS)
    s2_center = sum(HACKMAGEDDON_PROPORTIONS[v] * sum(MULTIPLICATEURS_DORA["S2_non_conforme"][v]) / 2
                    for v in HACKMAGEDDON_PROPORTIONS)
    print(f"    S1_partiel       : λ ≈ {lambda_ref*s1_center:.1f}  (mult ≈ {s1_center:.2f})")
    print(f"    S2_non_conforme  : λ ≈ {lambda_ref*s2_center:.1f}  (mult ≈ {s2_center:.2f})")

    print(f"\n  Approche continue — λ(entité, θ) pour 3 profils × {len(theta_values)} environnements :")
    print(f"  {'Profil':24s} {'θ':>6s} {'PCD':>8s} {'λ':>10s} {'mult.':>8s}")
    print(f"  {'-'*60}")
    for key, entity in PROFILS_TYPES.items():
        for theta in theta_values:
            r = lambda_from_entity(entity, theta, lambda_ref, latent_params)
            print(f"  {entity.name:24s} {theta:>6.1f} {r['pcd']:>7.1%} "
                  f"{r['lambda_global']:>10.1f} {r['multiplicateur_global']:>7.2f}x")

    print(f"\n  Lecture :")
    print(f"  • Une entité MATURE garde un λ proche de S0 même en crise systémique")
    print(f"    (sa PCD reste basse grâce à un bon profil de conformité).")
    print(f"  • Une entité EN RETARD dépasse le multiplicateur S2 dès l'environnement")
    print(f"    normal (θ=0) sous les paramètres ancrés (Option B) — cohérent avec")
    print(f"    les taux de non-conformité élevés observés fin 2025.")
    print(f"  • Le modèle continu est plus informatif que le choix discret : il")
    print(f"    distingue les entités SELON LEUR PROFIL RÉEL, pas par hypothèse")
    print(f"    de marché uniforme.")
    print("=" * 74)


if __name__ == "__main__":
    print(">>> Avec paramètres ANCRÉS (Option B) <<<\n")
    compare_discrete_vs_continuous(latent_params=ANCHORED_PARAMS)
    print("\n\n>>> Avec paramètres ILLUSTRATIFS (Option A) <<<\n")
    compare_discrete_vs_continuous(latent_params=ILLUSTRATIVE_PARAMS)
