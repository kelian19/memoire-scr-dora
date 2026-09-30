"""
src/visualization/plots.py
--------------------------
Génération des graphiques "publication-ready" pour le mémoire SCR DORA.
Données calibrées sur les logs du modèle LDA à 4 briques (Facteur Commun)
et du Pont Latent (Option B - Ancrée).
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# =====================================================================
# CONFIGURATION GLOBALE DU STYLE (Charte Graphique)
# =====================================================================
BRAND_BLUE = "#2563eb"
BRAND_DARK = "#111317"
BRAND_LIGHT = "#f6f7f9"

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Montserrat', 'Arial', 'sans-serif']
plt.rcParams['text.color'] = BRAND_DARK
plt.rcParams['axes.labelcolor'] = BRAND_DARK
plt.rcParams['xtick.color'] = BRAND_DARK
plt.rcParams['ytick.color'] = BRAND_DARK
plt.rcParams['axes.facecolor'] = BRAND_LIGHT
plt.rcParams['figure.facecolor'] = BRAND_LIGHT
plt.rcParams['axes.edgecolor'] = '#e2e8f0'
plt.rcParams['grid.color'] = '#e2e8f0'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# =====================================================================
# 1. DISTRIBUTION DU SCR (LDA) — OpRisk, scénario S2 non-conforme
# =====================================================================
def plot_scr_distribution(source: str = "OPRISK", n_sim: int = 100_000,
                           save_name: str = "distribution_scr_oprisk_s2.png"):
    """Trace la distribution simulée des pertes agrégées (état non-conforme,
    3 briques physiques) et sa VaR 99.5%, à partir du modèle LDA réel."""
    from src.aggregation.lda import scr_4_briques_report

    res_nc = scr_4_briques_report(source=source, n_sim=n_sim, dependence="gumbel")
    losses = res_nc["total"]
    var_alpha = res_nc["scr_total"]

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(losses, bins=250, kde=True, color=BRAND_BLUE, edgecolor="none", alpha=0.6, ax=ax)

    ax.axvline(var_alpha, color='#dc2626', linestyle='dashed', linewidth=2.5,
               label=f"VaR 99.5% = {var_alpha:,.1f} M€")

    ax.set_title(f"Distribution des Pertes Agrégées — {source}, non-conforme (S2)",
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Capital Requis (M€)", fontsize=12, fontweight='semibold')
    ax.set_ylabel("Fréquence (simulations)", fontsize=12, fontweight='semibold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Zoom sur la zone pertinente (coupe l'extrême queue visuellement illisible)
    ax.set_xlim(0, var_alpha * 1.3)

    ax.legend(frameon=True, facecolor=BRAND_LIGHT, edgecolor='#e2e8f0', fontsize=11)
    sns.despine()

    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé : {save_path}")


# =====================================================================
# 2. SENSIBILITÉ DE λ AU CHOC SYSTÉMIQUE θ (Pont Latent, Option B ancrée)
# =====================================================================
def plot_sensibilite_theta(lambda_ref: float = None, save_name: str = "sensibilite_theta.png"):
    """Trace λ(θ) pour les trois profils-types, calculé directement via
    lambda_from_entity (aucune valeur interpolée à la main)."""
    from src.scenarios.latent_bridge import lambda_from_entity
    from src.compliance.latent import PROFILS_TYPES
    from src.utils.config import FREQUENCY

    lambda_ref = lambda_ref or FREQUENCY["lambda_ref"]
    theta_grid = np.linspace(0, -3, 60)

    curves = {}
    for key, color, label in [
        ("leader", "#10b981", "Entité Leader"),
        ("median", "#f59e0b", "Entité Médiane"),
        ("retard", "#ef4444", "Entité En Retard"),
    ]:
        entity = PROFILS_TYPES[key]
        lam = [lambda_from_entity(entity, theta=t, lambda_ref=lambda_ref)["lambda_global"]
               for t in theta_grid]
        curves[key] = (color, label, np.array(lam))

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, (color, label, lam) in curves.items():
        ax.plot(theta_grid, lam, color=color, linewidth=3, label=label)

    ax.set_title("Sensibilité de la fréquence λ au choc systémique (θ)",
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Environnement Macro-Cyber (θ) [0 = Normal, -3 = Crise Majeure]",
                  fontsize=12, fontweight='semibold')
    ax.set_ylabel("Fréquence de Sinistralité (λ)", fontsize=12, fontweight='semibold')

    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True, facecolor=BRAND_LIGHT, edgecolor='#e2e8f0', loc='center left', fontsize=11)
    sns.despine()

    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé : {save_path}")


# =====================================================================
# 3. ALLOCATION D'EULER — comparaison PRC vs OpRisk (3 briques additives)
# =====================================================================
def plot_capital_allocation(entity_key: str = "median", theta_env: str = "0.0",
                             method: str = "var",
                             csv_path: str = None,
                             save_name: str = "euler_decomposition.png"):
    """Trace la décomposition du SCR (VaR 99.5%) par brique, sous les deux
    sources de sévérité, à partir du CSV produit par notebooks/04 (résultats
    réels de l'allocation d'Euler — pas de valeur codée en dur)."""
    csv_path = csv_path or os.path.join(FIGURES_DIR, "..", "tables", "results_euler_option_a.csv")

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    briques = ["remediation", "prestataire", "sanction"]
    data = {}
    for source in ["PRC", "OPRISK"]:
        row = next(
            (r for r in rows if r["source"] == source and r["entity_key"] == entity_key
             and r["theta_env"] == theta_env and r["method"] == method),
            None,
        )
        if row is None:
            raise ValueError(f"Aucune ligne trouvée pour {source}/{entity_key}/{theta_env}/{method}")
        data[source] = [float(row[f"{b}_pct"]) for b in briques]

    y = np.arange(len(briques))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_prc = ax.barh(y + height / 2, data["PRC"], height=height, color=BRAND_BLUE, label="PRC")
    bars_op = ax.barh(y - height / 2, data["OPRISK"], height=height, color="#f59e0b", label="OpRisk")

    for bars in (bars_prc, bars_op):
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                    f"{width:.1f}%", va='center', ha='left', fontsize=10,
                    color=BRAND_DARK, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels([b.capitalize() for b in briques])
    ax.set_title(f"Décomposition du SCR par Brique — VaR 99.5% (profil {entity_key})",
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Part du capital (%)", fontsize=12, fontweight='semibold')
    ax.set_xlim(0, 105)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.legend(frameon=True, facecolor=BRAND_LIGHT, edgecolor='#e2e8f0', fontsize=11)
    sns.despine(left=True)

    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé : {save_path}")


if __name__ == "__main__":
    print("Génération des graphiques à partir des résultats réels du pipeline...")
    plot_scr_distribution()
    plot_sensibilite_theta()
    plot_capital_allocation()
    print("Terminé ! Vérifie le dossier outputs/figures/")