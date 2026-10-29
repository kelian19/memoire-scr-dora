#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24 : SCR_DORA comme une DISTRIBUTION, et Delta_DORA(conforme vs non-conforme).

Les scripts 13 et 16 donnent le SCR par etat de conformite comme un POINT (VaR 99,5 %).
Ici on sort deux distributions, dans l'esprit de Hillairet-Lopez (etats de conformite) :

  1. SCR_DORA en distribution de MELANGE. L'entite n'est pas dans un etat fixe : sa
     conformite depend d'une latente pilotee par un facteur systemique Theta (modele du
     16). Chaque annee : Theta -> probabilites d'etat -> un etat (C / PC / NC) -> une
     perte sous cet etat. La perte agregee est un MELANGE sur les trois etats.
         SCR_DORA = VaR_99,5 % du melange.
     Ce n'est PAS la moyenne ponderee des VaR par etat (naif) : le melange est domine par
     la queue Non-conforme, et une CRISE (Theta bas) bascule la masse vers NC, donc
     epaissit la queue. C'est la l'apport : le capital integre l'incertitude de conformite
     ET sa correlation systemique.

  2. Delta_DORA(C vs NC) en distribution. Delta = SCR(NC) - SCR(C), en euros, par
     bootstrap 2 niveaux (frequence par etat + severite de queue reechantillonnee sur les
     exces reels OpRisk), graine MC commune entre etats (CRN). On sort la distribution du
     Delta, pas seulement sa mediane.

Constantes d'etat = MIROIR du script 16 (source de verite). Severite euros = SAS OpRisk
via euro_cascade_model. Choix non calibres (gains g par etat, detection p_u, probas
d'etat, gamma) : deja traites en sensibilite au 16. Ne touche ni src/ ni memoire/.

Sortie : diagnostics + figure T_scr_dora_distribution.png.
"""

import os
import sys

import numpy as np
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                              # noqa: E402
from euro_cascade_model import PARAMS, var                   # noqa: E402

SOURCE = "OPRISK"
W = 74

# ---- constantes d'etat : MIROIR du 16 (y modifier en premier) -----------------------
ETATS = ["C", "PC", "NC"]
LABEL = {"C": "Conforme", "PC": "Partiellement conforme", "NC": "Non conforme"}
SCENARIO = {"C": "S0_conforme", "PC": "S1_partiel", "NC": "S2_non_conforme"}
P_ETAT = {"NC": 0.35, "PC": 0.35, "C": 0.30}
GAMMA = 0.68
THETA_CRISE = -2.5
G_PROP = {"C": 0.45, "PC": 0.68, "NC": 0.90}
PU_MULT = {"C": 0.85, "PC": 1.00, "NC": 1.20}
K_BAS = norm.ppf(P_ETAT["NC"])
K_HAUT = norm.ppf(P_ETAT["NC"] + P_ETAT["PC"])

NY = 120_000            # annees simulees par etat (distribution lisse)
BOOT_B, BOOT_NY = 120, 12_000
RNG_SEED = 20260721


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def etat_probs(theta):
    """Probas (C, PC, NC) conditionnelles au facteur systemique theta (vectorise)."""
    denom = np.sqrt(1.0 - GAMMA ** 2)
    p_nc = norm.cdf((K_BAS - GAMMA * theta) / denom)
    p_ncpc = norm.cdf((K_HAUT - GAMMA * theta) / denom)
    return np.array([1.0 - p_ncpc, p_ncpc - p_nc, p_nc])     # ordre C, PC, NC


def loss_state(e, ny, seed):
    """Pertes annuelles (M€) sous l'etat e, canaux freq+prop+detec, graine donnee (CRN)."""
    sp = PARAMS[SOURCE]
    lam = ec.lambda_scenario(SOURCE, SCENARIO[e], mode="center")
    p_u = min(0.999, sp["p_u"] * PU_MULT[e])
    rng = np.random.default_rng(seed)
    return ec.simulate_euro(lam, G_PROP[e], sp["xi"], sp["sigma"], sp["u"], p_u,
                            sp["cap"], ny, rng)


# =====================================================================================
titre("Pertes par etat de conformite (severite OpRisk, canaux freq+prop+detec)")
# =====================================================================================
seed0 = 4242
loss = {e: loss_state(e, NY, seed0) for e in ETATS}           # CRN : meme graine
scr_state = {e: var(loss[e]) for e in ETATS}
stack = np.vstack([loss[e] for e in ETATS])                   # (3, NY), ordre ETATS
for e in ETATS:
    print(f"  {LABEL[e]:<26} perte moyenne {loss[e].mean():8.1f} M | "
          f"SCR (VaR 99,5%) = {scr_state[e]:9.1f} M")

# =====================================================================================
titre("SCR_DORA en distribution de melange (normal vs crise systemique)")
# =====================================================================================
rng = np.random.default_rng(RNG_SEED)


def mixture(theta_draw):
    """Melange : par annee, Theta -> probas d'etat -> etat -> perte de cet etat (CRN)."""
    probs = etat_probs(theta_draw)                            # forme (3,) ou (3, NY)
    if probs.ndim == 1:
        probs = np.repeat(probs[:, None], NY, axis=1)
    cum = np.cumsum(probs, axis=0)                            # (3, NY)
    u = rng.random(NY)
    idx = (u > cum[0]).astype(int) + (u > cum[1]).astype(int)  # 0=C,1=PC,2=NC
    return stack[idx, np.arange(NY)], idx


theta_norm = rng.standard_normal(NY)                          # environnement aleatoire
mix_norm, idx_norm = mixture(theta_norm)
mix_crise, idx_crise = mixture(np.full(NY, THETA_CRISE))      # crise systemique
scr_norm, scr_crise = var(mix_norm), var(mix_crise)

# reference naive : moyenne ponderee des VaR par etat (a NE PAS confondre avec le melange)
naive = sum(P_ETAT[e] * scr_state[e] for e in ETATS)

print(f"  part de temps Non-conforme  : normal {np.mean(idx_norm==2):.0%}  "
      f"crise {np.mean(idx_crise==2):.0%}  (bascule DORA)")
print(f"\n  SCR par etat            : C {scr_state['C']:.0f}  PC {scr_state['PC']:.0f}  "
      f"NC {scr_state['NC']:.0f} M")
print(f"  moyenne ponderee des VaR (naif) : {naive:.0f} M")
print(f"  SCR_DORA (melange, normal)      : {scr_norm:.0f} M  "
      f"({'+' if scr_norm>naive else ''}{100*(scr_norm/naive-1):.0f} % vs naif)")
print(f"  SCR_DORA (melange, crise)       : {scr_crise:.0f} M  "
      f"(x{scr_crise/scr_norm:.2f} vs normal)")
print("\n  Lecture : le melange n'est pas la moyenne des VaR. Sa queue 99,5 % est portee")
print("  par l'etat Non-conforme ; une crise systemique y bascule la masse et epaissit")
print("  encore la queue. Le SCR_DORA integre donc l'incertitude de conformite.")

# =====================================================================================
titre("Delta_DORA(NC vs C) en distribution : bootstrap 2 niveaux, CRN")
# =====================================================================================
EXC = ec.oprisk_excesses()
print("  severite de queue :", "reechantillonnage des exces reels OpRisk"
      if EXC is not None else "repli IC90 normal")
sp = PARAMS[SOURCE]
brng = np.random.default_rng(RNG_SEED + 7)
deltas = []
for b in range(BOOT_B):
    lam_c = ec.lambda_scenario(SOURCE, SCENARIO["C"], mode="sample", rng=brng)
    lam_nc = ec.lambda_scenario(SOURCE, SCENARIO["NC"], mode="sample", rng=brng)
    if EXC is not None:
        xi_b, sg_b = ec.bootstrap_sev_from_excesses(EXC, brng)
    else:
        xi_b, sg_b = ec.sample_severity_params(SOURCE, brng)
    seed_b = 9000 + b
    pu_c = min(0.999, sp["p_u"] * PU_MULT["C"])
    pu_nc = min(0.999, sp["p_u"] * PU_MULT["NC"])
    r_c = np.random.default_rng(seed_b)
    r_nc = np.random.default_rng(seed_b)                       # CRN
    v_c = var(ec.simulate_euro(lam_c, G_PROP["C"], xi_b, sg_b, sp["u"], pu_c,
                               sp["cap"], BOOT_NY, r_c))
    v_nc = var(ec.simulate_euro(lam_nc, G_PROP["NC"], xi_b, sg_b, sp["u"], pu_nc,
                                sp["cap"], BOOT_NY, r_nc))
    deltas.append(v_nc - v_c)
deltas = np.array(deltas)
d_med = np.median(deltas)
d_lo, d_hi = np.quantile(deltas, [0.05, 0.95])
print(f"  Delta_DORA(NC vs C) : mediane {d_med:.0f} M€  |  IC90 [{d_lo:.0f} ; {d_hi:.0f}] M€")
print(f"  surcout de capital relatif median : x{scr_state['NC']/scr_state['C']:.2f} "
      "(NC / C, calage central)")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("1. SCR_DORA est bien une DISTRIBUTION, pas un point : le melange sur les etats de")
print(f"   conformite en donne le 99,5 % = {scr_norm:.0f} M (normal), {scr_crise:.0f} M (crise).")
print("2. Il ne se lit PAS comme la moyenne des VaR par etat : la queue est portee par")
print("   le Non-conforme, et le facteur systemique correle les bascules (effet DORA).")
print("3. Delta_DORA(NC vs C) est une distribution large : le surcout de non-conformite")
print(f"   vaut {d_med:.0f} M€ en median mais l'IC90 va de {d_lo:.0f} a {d_hi:.0f} M€. A")
print("   presenter comme une fourchette, jamais comme un point.")

# =====================================================================================
# figure
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 11,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
})
INK, INK2, MUTED, GRID = "#1b1e30", "#223e55", "#595959", "#f2f2f2"
ACCENT = "#a6002e"
BL = ["#7baafd", "#4c79c7", "#204993"]
STATE_COL = {"C": BL[0], "PC": BL[1], "NC": BL[2]}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 4.8),
                                    gridspec_kw={"width_ratios": [1.1, 1, 1]})

# (a) queues des pertes par etat + melange (fonction de survie, log-log)
def surv(x):
    xp = np.sort(x[x > 0])[::-1]
    return xp, np.arange(1, len(xp) + 1) / len(x)


for e in ETATS:
    xp, sv = surv(loss[e])
    ax1.loglog(xp, sv, color=STATE_COL[e], lw=1.8, label=LABEL[e])
xp, sv = surv(mix_norm)
ax1.loglog(xp, sv, color=ACCENT, lw=2.2, ls="--", label="SCR_DORA (mélange)")
ax1.axhline(0.005, color=MUTED, lw=1, ls=":")
ax1.text(ax1.get_xlim()[0] * 1.2, 0.0056, "seuil 99,5 %", fontsize=7.6, color=MUTED)
ax1.set_xlabel("perte annuelle (M€, log)", color=INK2)
ax1.set_ylabel("P(perte > x)", color=INK2)
ax1.legend(frameon=False, fontsize=8, loc="lower left")
ax1.set_title("(a)  La queue du mélange suit le Non-conforme", fontsize=11, color=INK, pad=8)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# (b) SCR : par etat, naif, melange normal, melange crise
labs = ["C", "PC", "NC", "moy.\npondérée", "SCR_DORA\nnormal", "SCR_DORA\ncrise"]
vals = [scr_state["C"], scr_state["PC"], scr_state["NC"], naive, scr_norm, scr_crise]
cols = [BL[0], BL[1], BL[2], MUTED, ACCENT, "#a3330f"]
ax2.bar(range(6), vals, color=cols, edgecolor="#fcfcfb", width=0.7)
for i, v in enumerate(vals):
    ax2.text(i, v + max(vals) * 0.015, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
ax2.set_xticks(range(6)); ax2.set_xticklabels(labs, fontsize=7.8)
ax2.set_ylabel("SCR (VaR 99,5 %, M€)", color=INK2)
ax2.set_ylim(0, max(vals) * 1.16)
ax2.set_title("(b)  Le mélange n'est pas la moyenne des VaR", fontsize=11, color=INK, pad=8)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# (c) distribution bootstrap du Delta_DORA(NC vs C)
ax3.hist(deltas, bins=22, color=BL[1], edgecolor="#fcfcfb")
ax3.axvline(d_med, color=ACCENT, lw=2.2, label=f"médiane {d_med:.0f} M€")
ax3.axvline(d_lo, color=INK, lw=1.2, ls="--")
ax3.axvline(d_hi, color=INK, lw=1.2, ls="--", label=f"IC90 [{d_lo:.0f} ; {d_hi:.0f}]")
ax3.set_xlabel("Delta_DORA = SCR(NC) $-$ SCR(C)  (M€)", color=INK2)
ax3.set_ylabel("tirages bootstrap", color=INK2)
ax3.legend(frameon=False, fontsize=8.2)
ax3.set_title("(c)  Le surcoût de non-conformité, en fourchette", fontsize=11,
              color=INK, pad=8)
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)

fig.suptitle("T : SCR_DORA comme distribution, et Delta_DORA(conforme vs non-conforme)",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "T_scr_dora_distribution.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
