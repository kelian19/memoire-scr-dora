#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
52 : le declenchement de P4 comme TEMPS D'ARRET (extension dynamique du seuil z*).

Le script 42 pose le seuil de declenchement de P4 comme un niveau z* = K_4/sqrt(rho) du facteur
tiers commun. Si ce facteur est un PROCESSUS Z_t (le tiers partage evolue dans le temps), le
declenchement devient un TEMPS DE PREMIER PASSAGE :
        tau = inf{ t : Z_t >= z* } ,
un temps d'arret. Le 'moment' de la question de Hugo prend alors un sens temporel LITTERAL, et
le cadre se raccorde a la note martingale (script 40) : pour un mouvement brownien sans derive,
Z_t est une MARTINGALE et tau un temps d'arret ; la probabilite de declenchement AVANT un
horizon a une forme close (principe de reflexion) :
        P(tau <= T) = 2 Phi( - z* / sqrt(T) ) ,   (Z de variance unitaire par an).

C'est deux fois la probabilite statique P(Z_T >= z*) du script 42 : la lecture dynamique
capte tous les FRANCHISSEMENTS avant T, pas seulement la position finale.

HONNETETE. Pour un brownien SANS derive, le niveau est atteint p.s. mais E[tau] = +infini
(recurrence lente) : la version realiste est un facteur MOYEN-REVERSIF (Ornstein-Uhlenbeck),
qui donne un TAUX de declenchement fini et recurrent. On presente le cas brownien (martingale,
forme close, lien item 4) et on note l'OU comme le raffinement.

Sortie : diagnostics + figure Z16_p4_temps_arret.png.
"""

import os
import sys

import numpy as np
from scipy.stats import norm
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from euro_cascade_model import PARAMS                            # noqa: E402

WID = 82
P4_MARG = PARAMS["OPRISK"]["p_u"]
K4 = float(norm.ppf(1.0 - P4_MARG))
SEED = 20260727
rng = np.random.default_rng(SEED)


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def z_star(rho):
    return K4 / np.sqrt(rho)


def p_trigger_within(T, rho):
    """P(tau <= T) pour un brownien sans derive (principe de reflexion)."""
    return 2.0 * norm.cdf(-z_star(rho) / np.sqrt(T))


def fpt_density(t, rho):
    """Densite du temps de premier passage (loi de Levy) : z*/sqrt(2 pi t^3) exp(-z*^2/2t)."""
    zs = z_star(rho)
    return zs / np.sqrt(2 * np.pi * t ** 3) * np.exp(-zs ** 2 / (2 * t))


# =====================================================================================
titre("1. Le seuil z* devient un temps de premier passage")
# =====================================================================================
print(f"  K_4 = {K4:.3f} ; z*(rho) = K_4/sqrt(rho). Facteur tiers Z_t = brownien (martingale).")
print(f"  {'rho':>7}{'z*':>9}{'P(Z_1>=z*) statique':>22}{'P(tau<=1 an) dynamique':>26}")
for rho in (0.10, 0.25, 0.50, 0.75, 0.90):
    zs = z_star(rho)
    p_stat = 1 - norm.cdf(zs)
    p_dyn = p_trigger_within(1.0, rho)
    print(f"  {rho:>7.2f}{zs:>9.3f}{p_stat:>22.3f}{p_dyn:>26.3f}")
print("\n  La lecture dynamique (premier passage) vaut ~2x la statique : elle compte tous les")
print("  franchissements avant l'horizon, pas seulement la position a l'instant T. C'est le")
print("  'moment' de declenchement au sens propre, un temps d'arret.")

# =====================================================================================
titre("2. Horizon de declenchement et lien martingale")
# =====================================================================================
for rho in (0.25, 0.50, 0.75):
    # mediane du temps de premier passage : P(tau<=t_med)=0.5 -> z*/sqrt(t)=Phi^-1(0.75)
    t_med = (z_star(rho) / norm.ppf(0.75)) ** 2
    print(f"  rho={rho:.2f} : P(declenchement < 1 an) = {p_trigger_within(1.0,rho):.2f} ; "
          f"P(< 5 ans) = {p_trigger_within(5.0,rho):.2f} ; mediane du delai = {t_med:.1f} an(s).")
print("\n  Z_t sans derive est une MARTINGALE ; tau est un temps d'arret (note 40). Le niveau est")
print("  atteint presque surement mais E[tau]=+inf : le brownien pur donne la forme close et le")
print("  lien martingale ; un facteur moyen-reversif (OU) donnerait un TAUX recurrent fini, plus")
print("  realiste pour un facteur systemique. C'est le raccord dynamique des items 3 et 4.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. Le seuil z* de P4 (item 3) devient un TEMPS D'ARRET tau des que le facteur tiers est")
print("     un processus : le 'moment' de declenchement a un sens temporel litteral.")
print("  2. Forme close (principe de reflexion) : P(tau<=T)=2 Phi(-z*/sqrt(T)), ~2x la statique.")
print("  3. Lien martingale (item 4) : Z_t sans derive est une martingale, tau un temps d'arret ;")
print("     le fil des quatre pistes se referme (seuil -> temps d'arret -> martingale).")
print("  4. Raffinement assume : un facteur moyen-reversif (OU) pour un taux de declenchement")
print("     recurrent fini ; le brownien est le cas propre qui porte le lien theorique.")

# =====================================================================================
# figure Z16
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 10.5,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
})
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime cette
# figure sur 18,5 cm : tracee plus large, elle subit une reduction qui fait tomber
# ses etiquettes sous 7 points. A 8,6 pouces le facteur d'impression vaut 0,85,
# donc une police de 8,5 points s'imprime a 7,2.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.6, 3.8))

# (a) trajectoires browniennes et premier passage a z* (rho=0,50)
rho0 = 0.50; zs0 = z_star(rho0)
T, dt = 3.0, 0.004
steps = int(T / dt)
tgrid = np.linspace(dt, T, steps)
for k in range(7):
    incr = rng.normal(0, np.sqrt(dt), steps)
    Z = np.cumsum(incr)
    hit = np.argmax(Z >= zs0) if (Z >= zs0).any() else -1
    ax1.plot(tgrid, Z, lw=1.0, color=BLUE, alpha=0.6)
    if hit > 0:
        ax1.plot(tgrid[hit], zs0, "o", color=ACCENT, ms=5, zorder=5)
ax1.axhline(zs0, color=INK, lw=1.4)
ax1.text(0.05, zs0 + 0.12, f"$z^*$={zs0:.2f} (ρ={rho0:.2f})", fontsize=8.5, color=INK,
         va="bottom", bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none", alpha=0.85))
ax1.set_xlabel("temps (années)", color=INK2)
ax1.set_ylabel("facteur tiers  $Z_t$", color=INK2)
ax1.set_title("(a)  Facteur tiers (martingale)\net premier passage de $z^*$",
              fontsize=10.5, color=INK, pad=8)

# (b) densite du temps de premier passage (Levy) + P(tau<=1 an)
tt = np.linspace(0.02, 8, 400)
for rho, c in zip((0.25, 0.50, 0.75), ["#9dc3e6", BLUE, "#204993"]):
    ax2.plot(tt, fpt_density(tt, rho), color=c, lw=2, label=f"ρ={rho:.2f}")
ax2.axvline(1.0, color=ACCENT, ls="--", lw=1.3)
# au pied de la ligne : en haut, l'etiquette tombait sur les courbes et la legende
ax2.text(1.12, 0.012, "1 an", fontsize=8.5, color=ACCENT, va="bottom")
ax2.set_xlabel("délai de déclenchement  τ (années)", color=INK2)
ax2.set_ylabel("densité (loi de Lévy)", color=INK2)
ax2.legend(frameon=False, fontsize=8.5, title="chargement", title_fontsize=8.5, loc="upper right")
ax2.set_title("(b)  Délai de déclenchement\n(loi du premier passage)",
              fontsize=10.5, color=INK, pad=8)

# (c) P(declenchement dans l'annee) vs rho : statique vs dynamique
rhos = np.linspace(0.05, 0.95, 60)
p_stat = np.array([1 - norm.cdf(z_star(r)) for r in rhos])
p_dyn = np.array([p_trigger_within(1.0, r) for r in rhos])
ax3.plot(rhos, p_dyn, color=ACCENT, lw=2.2, label="dynamique : P(τ ≤ 1 an)")
ax3.plot(rhos, p_stat, color=BLUE, lw=2.2, ls="--", label="statique : P($Z_1 \\geq z^*$)")
ax3.set_xlabel("chargement sur le tiers  ρ", color=INK2)
ax3.set_ylabel("P(déclenchement en 1 an)", color=INK2, fontsize=9)
# marge haute : la legende tombait sur la courbe dynamique
ax3.set_ylim(0, max(p_dyn.max(), p_stat.max()) * 1.55)
ax3.legend(frameon=False, fontsize=8.5, loc="upper left")
ax3.set_title("(c)  Plus ρ croît, plus le\ndéclenchement est probable",
              fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2, ax3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# PAS DE TITRE GENERAL : la legende LaTeX du memoire porte le titre de la figure.
fig.tight_layout()
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z16_p4_temps_arret.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
