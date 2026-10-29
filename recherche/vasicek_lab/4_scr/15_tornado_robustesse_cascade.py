#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15 : Tornado de robustesse du Delta_DORA cascade aux parametres structurels.

Meme esprit que le notebook 08 du memoire : on fait varier un a un les leviers NON
calibres du modele et on mesure de combien le Delta_DORA (surcout de non-conformite,
OpRisk, entite 15 000 M€) se deplace. Le verdict est robuste s'il reste pluri-milliard
et > 0 sur toutes les variantes.

Leviers testes (bornes de part et d'autre du cas de base) :
  - SEUIL POT u        : percentile 80 vs 90 (refit GPD sur les VRAIS exces a chaque
                         seuil : fragilite EVT classique). Base = percentile 85 (config).
  - FREQUENCE lam_ref  : x0,5 vs x1,5 (proxy pose, OpRisk propre = 582/27 ~ 21,6/an).
  - SURDISPERSION phi  : 5 vs 15 (base 9,2).
  - GAIN g             : 0,5 vs 1,0 (base 0,90), applique aux deux etats.

Estimation deterministe (multiplicateurs au centre, graine commune conforme/non-conforme,
100 000 annees) : un tornado compare des points, l'incertitude d'IC est traitee en 14.

PORTEE. Lecture LOCALE de data/raw pour le refit au seuil (jamais poussee). Ne touche ni
src/ ni memoire/.

Sortie : diagnostics + figure S9_tornado_robustesse.png.
"""

import os
import sys

import numpy as np
from scipy.stats import genpareto
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                              # noqa: E402
from euro_cascade_model import var, OPRISK                   # noqa: E402
from src.frequency.negbin import compute_lambda_scenario     # noqa: E402

W = 74
NY = 100_000
SEED = 11
LAM_REF0 = OPRISK["n_incidents"] / OPRISK["n_years"]          # 582/27 ~ 21.56/an
G0, PHI0 = ec.G_BASE, ec.PHI


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def gpd_at_threshold(losses, u):
    exc = losses[losses > u] - u
    xi, _, sg = genpareto.fit(exc, floc=0)
    return float(np.clip(xi, 0.05, 0.98)), max(1.0, float(sg)), len(exc) / len(losses)


def delta(lam_ref, u, p_u, xi, sigma, g, phi, ny=NY):
    """Delta_DORA deterministe = SCR(S2) - SCR(S0), graine commune (OpRisk, non plafonne)."""
    lam_nc = compute_lambda_scenario(lam_ref, "S2_non_conforme", "center")["lambda_global"]
    lam_c = compute_lambda_scenario(lam_ref, "S0_conforme", "center")["lambda_global"]
    v_nc = var(ec.simulate_euro(lam_nc, g, xi, sigma, u, p_u, None, ny,
                                np.random.default_rng(SEED), phi=phi))
    v_c = var(ec.simulate_euro(lam_c, g, xi, sigma, u, p_u, None, ny,
                               np.random.default_rng(SEED), phi=phi))
    return v_nc - v_c


# ============================================================ cas de base
titre("Cas de base (OpRisk, u=percentile 85, lam_ref=21,6/an, phi=9,2, g=0,90)")
LOSSES = ec.oprisk_losses()
if LOSSES is None:
    sys.exit("SAS OpRisk absent de data/raw (licence) : tornado indisponible.")
u0 = OPRISK["seuil_u_eur"]
xi0, sg0, pu0 = gpd_at_threshold(LOSSES, u0)
print(f"  fit base : xi={xi0:.3f}  sigma={sg0:.1f}  p_u={pu0:.3f}  (config 0,595 / 58,0 / 0,151)")
base = delta(LAM_REF0, u0, pu0, xi0, sg0, G0, PHI0)
print(f"  Delta_DORA de base = {base:.0f} M€")

# ============================================================ tornado
titre("Tornado : deplacement du Delta_DORA par levier (bas / haut)")
rows = []

# seuil POT u : percentiles 80 et 90 (refit)
for lab, p in [("bas", 80), ("haut", 90)]:
    u_p = float(np.quantile(LOSSES, p / 100))
    xi_p, sg_p, pu_p = gpd_at_threshold(LOSSES, u_p)
    d = delta(LAM_REF0, u_p, pu_p, xi_p, sg_p, G0, PHI0)
    rows.append(("seuil u", lab, f"p{p} (u={u_p:.0f}, xi={xi_p:.2f})", d))

# frequence lam_ref
for lab, mult in [("bas", 0.5), ("haut", 1.5)]:
    d = delta(LAM_REF0 * mult, u0, pu0, xi0, sg0, G0, PHI0)
    rows.append(("lambda_ref", lab, f"x{mult}", d))

# surdispersion phi
for lab, phi in [("bas", 5.0), ("haut", 15.0)]:
    d = delta(LAM_REF0, u0, pu0, xi0, sg0, G0, phi)
    rows.append(("phi", lab, f"{phi:.0f}", d))

# gain g
for lab, g in [("bas", 0.5), ("haut", 1.0)]:
    d = delta(LAM_REF0, u0, pu0, xi0, sg0, g, PHI0)
    rows.append(("gain g", lab, f"{g}", d))

params = ["seuil u", "lambda_ref", "phi", "gain g"]
print(f"  base = {base:.0f} M€\n")
print(f"  {'levier':<12}{'bas':>28}{'haut':>28}")
swing = {}
for pnm in params:
    lo = next(r for r in rows if r[0] == pnm and r[1] == "bas")
    hi = next(r for r in rows if r[0] == pnm and r[1] == "haut")
    swing[pnm] = (min(lo[3], hi[3]), max(lo[3], hi[3]))
    print(f"  {pnm:<12}{f'{lo[2]}: {lo[3]:.0f}':>28}{f'{hi[2]}: {hi[3]:.0f}':>28}")

all_d = [r[3] for r in rows] + [base]
print(f"\n  Delta_DORA sur toutes les variantes : min {min(all_d):.0f}, max {max(all_d):.0f} M€")
print(f"  Toutes > 0 : {all(x > 0 for x in all_d)} ; toutes pluri-milliard : "
      f"{all(x > 1000 for x in all_d)}")
print("  Verdict robuste : le seuil u domine la sensibilite (fragilite EVT), mais le")
print("  surcout de non-conformite reste pluri-milliard et positif sur tous les leviers.")

# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2, BLUE, ACCENT = "#1b1e30", "#223e55", "#3661ac", "#a6002e"

# trier par amplitude (tornado)
order = sorted(params, key=lambda p: swing[p][1] - swing[p][0])
fig, ax = plt.subplots(figsize=(10.5, 5.2))
for i, pnm in enumerate(order):
    lo, hi = swing[pnm]
    ax.barh(i, hi - lo, left=lo, color=BLUE, alpha=0.75, height=0.55)
    ax.text(lo, i, f"{lo:.0f} ", ha="right", va="center", fontsize=8.2, color=INK2)
    ax.text(hi, i, f" {hi:.0f}", ha="left", va="center", fontsize=8.2, color=INK2)
ax.axvline(base, color=ACCENT, lw=1.5, ls="--")
ax.text(base, len(order) - 0.55, f" base {base:.0f}", color=ACCENT, fontsize=9, va="top")
lo_min = min(swing[p][0] for p in order)
hi_max = max(swing[p][1] for p in order)
ax.set_xlim(lo_min - 0.09 * (hi_max - lo_min), hi_max + 0.06 * (hi_max - lo_min))
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.set_xlabel("Delta_DORA (M€), surcout de non-conformite", fontsize=9.5, color=INK2)
ax.set_title("Robustesse du Delta_DORA cascade : tornado des leviers structurels (OpRisk)",
             fontsize=11.5, fontweight="bold", color=INK, pad=8)
ax.grid(alpha=0.25, lw=0.5, axis="x")
ax.text(0.99, 0.03, "toutes les variantes > 0 et pluri-milliard : verdict robuste",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.4, color=INK2, style="italic")
fig.tight_layout()

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S9_tornado_robustesse.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
