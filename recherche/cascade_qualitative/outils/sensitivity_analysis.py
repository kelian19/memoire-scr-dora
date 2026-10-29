#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ancrage documentaire + analyse de sensibilité du modèle qualitatif de cascade DORA.

Objet : répondre à la question « tout dépend des valeurs, non ? ».
On ne le nie pas : on le TESTE.

  F7 : Coefficients & ancrage (logique pure + DORA)  : d'où vient chaque nombre.
  F8 : Robustesse du classement                       : le Top-15 tient-il sous bruit ?
  F9 : « L'ordre renverse le verdict » sous bruit     : la conclusion phare survit-elle ?
  F10 : Plafond de criticité sous bruit                : « jamais Extrême » est-il structurel ?
  F11 : Tornado                                        : quelle hypothèse PORTE le résultat ?

Perturbation Monte-Carlo : chaque coefficient est tiré autour de sa valeur de base
(± une marge), sur N tirages ; on recalcule les 325 scénarios à chaque tirage et on
regarde si les CONCLUSIONS QUALITATIVES tiennent (pas les scores exacts).
"""

from itertools import combinations, permutations
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ---------------------------------------------------------------- baseline (miroir du .xlsx)
PILIERS = {
    1: "Gouvernance & risque TIC",
    2: "Gestion des incidents",
    3: "Tests de résilience (TLPT)",
    4: "Risque lié aux tiers ICT",
    5: "Partage d'informations",
}
N = 5
# --- jugement et barème de base : source unique (cascade_model.py) ----------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cascade_model import (ROOT as ROOT0, TRANS as TRANS0, GBASE as GBASE0,
                           KFAC as KFAC0, AMPB as AMPB0, AMPS as AMPS0)

ORDERS = [o for k in range(1, N + 1)
          for subset in combinations(range(1, N + 1), k)
          for o in permutations(subset)]      # 325 chaînes notées


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def score(order, ROOT, TRANS, GBASE, kfac, ampb, amps):
    """(p, g, c) arrondis + (p_raw, g_raw) continus pour un classement fin."""
    prop = ROOT[order[0]]
    for a, b in zip(order, order[1:]):
        prop *= TRANS[a][b]
    p_raw = 10 * prop
    bases = [GBASE[p] for p in order]
    etendue = min(10, max(bases) + kfac * (sum(bases) - max(bases)))
    if len(order) <= 1:
        g_raw = etendue
    else:
        links = [TRANS[a][b] for a, b in zip(order, order[1:])]
        g_raw = etendue * (ampb + amps * (sum(links) / len(links)))
    p = max(1, min(10, round(p_raw)))
    g = max(1, min(10, round(g_raw)))
    c = round((p * g) ** 0.5)
    return p, g, c, p_raw, g_raw


BASE = (ROOT0, TRANS0, GBASE0, KFAC0, AMPB0, AMPS0)

# ---------------------------------------------------------------- style / palette validée
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 11,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "text.color": "#0b0b0b", "axes.labelcolor": "#52514e",
    "xtick.color": "#898781", "ytick.color": "#898781", "axes.grid": False,
})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ACCENT = "#2a78d6", "#eb6834"
BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blues", BLUES)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)


def finish(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {os.path.relpath(path)}")


def chain(o):
    return "→".join(map(str, o))


# ================================================================ métriques
def order_flip_fraction(params):
    """Part des paires {i,j} où le sens causalement fort (au baseline) reste ≥ à son inverse."""
    hold = 0
    pairs = list(combinations(range(1, N + 1), 2))
    for i, j in pairs:
        coh = (i, j) if TRANS0[i][j] >= TRANS0[j][i] else (j, i)
        inc = coh[::-1]
        c_coh = score(coh, *params)[2]
        c_inc = score(inc, *params)[2]
        hold += (c_coh >= c_inc)
    return hold / len(pairs)


def ranking(params):
    """Chaînes triées du plus critique au moins critique (clé continue)."""
    scored = [(o, score(o, *params)) for o in ORDERS]
    scored.sort(key=lambda t: (t[1][2], t[1][4], t[1][3]), reverse=True)  # c, g_raw, p_raw
    return [o for o, _ in scored]


def max_crit(params):
    return max(score(o, *params)[2] for o in ORDERS)


def perturb(rng):
    R = {p: clip(ROOT0[p] + rng.uniform(-.15, .15), 0.05, 1.0) for p in ROOT0}
    T = {a: {b: clip(TRANS0[a][b] + rng.uniform(-.15, .15), 0.02, 0.98)
             for b in TRANS0[a]} for a in TRANS0}
    G = {p: clip(GBASE0[p] + rng.uniform(-1.2, 1.2), 1, 8) for p in GBASE0}
    return (R, T, G, rng.uniform(.25, .55), rng.uniform(.35, .65), rng.uniform(.6, 1.0))


# ================================================================ Monte-Carlo
NDRAW = 3000
rng = np.random.default_rng(20260708)

base_rank = ranking(BASE)
top15 = base_rank[:15]
rank_hist = {o: [] for o in top15}
flip_draws = np.empty(NDRAW)
maxc_draws = np.empty(NDRAW)

for d in range(NDRAW):
    P = perturb(rng)
    flip_draws[d] = order_flip_fraction(P)
    maxc_draws[d] = max_crit(P)
    pos = {o: i + 1 for i, o in enumerate(ranking(P))}
    for o in top15:
        rank_hist[o].append(pos[o])

base_flip = order_flip_fraction(BASE)
print(f"\n--- Synthèse sensibilité ({NDRAW} tirages) ---")
print(f"Ordre renverse le verdict : baseline {base_flip:.0%} ; "
      f"moyenne {flip_draws.mean():.0%} ; ≥80% dans {np.mean(flip_draws >= .8):.0%} des tirages")
print(f"Criticité max : Extrême (≥9) atteinte dans {np.mean(maxc_draws >= 9):.1%} des tirages "
      f"(baseline max = {max_crit(BASE)})")
stab = np.mean([np.mean(np.array(rank_hist[o]) <= 15) for o in top15])
print(f"Top-15 : un scénario du top baseline reste dans le top-15 dans {stab:.0%} des tirages\n")


# ================================================================ F7 : coefficients & ancrage
def fig7():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4),
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})
    fig.subplots_adjust(top=0.76, wspace=0.55, left=0.06, right=0.98)

    # (a) ROOT
    ax = axes[0]
    order = sorted(ROOT0, key=ROOT0.get)
    ax.barh([f"P{p}" for p in order], [ROOT0[p] for p in order],
            color=[SEQ(ROOT0[p]) for p in order], edgecolor="#fcfcfb")
    for p in order:
        ax.text(ROOT0[p] - .04, f"P{p}", f"{ROOT0[p]:.2f}", va="center", ha="right",
                color="#fff", fontsize=9)
    ax.set_xlim(0, 1.05); ax.set_title("ROOT : propension à AMORCER", fontsize=11,
                                       color=INK, loc="left", pad=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

    # (b) GBASE
    ax = axes[1]
    order = sorted(GBASE0, key=GBASE0.get)
    ax.barh([f"P{p}" for p in order], [GBASE0[p] for p in order],
            color=[SEQ(GBASE0[p] / 7) for p in order], edgecolor="#fcfcfb")
    for p in order:
        ax.text(GBASE0[p] - .2, f"P{p}", f"{GBASE0[p]}", va="center", ha="right",
                color="#fff", fontsize=9)
    ax.set_xlim(0, 7.5); ax.set_title("GBASE : gravité intrinsèque", fontsize=11,
                                      color=INK, loc="left", pad=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

    # (c) TRANS heatmap
    ax = axes[2]
    M = np.full((N, N), np.nan)
    for a in TRANS0:
        for b in TRANS0[a]:
            M[a - 1, b - 1] = TRANS0[a][b]
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1)
    for i in range(N):
        for j in range(N):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=9,
                        color="#fff" if M[i, j] >= .5 else INK)
            else:
                ax.text(j, i, "-", ha="center", va="center", color=MUTED)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels([f"P{j}" for j in range(1, N + 1)])
    ax.set_yticklabels([f"P{i}" for i in range(1, N + 1)])
    ax.set_title("TRANS : « i entraîne j » (asymétrique)", fontsize=11, color=INK,
                 loc="left", pad=8)
    ax.set_xlabel("vers j", color=INK2); ax.set_ylabel("de i", color=INK2)
    ax.tick_params(length=0)

    fig.text(0.06, 0.95, "F7 : D'où vient chaque coefficient (logique pure + DORA)",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.06, 0.905, "Aucune valeur n'est mesurée : jugements ORDONNÉS, ancrés sur DORA. "
             "La HIÉRARCHIE est défendable, pas les décimales : d'où l'analyse de sensibilité (F8-F11).",
             fontsize=9.5, color=INK2)
    fig.text(0.06, 0.865, "ROOT : P1 fondation · P4 surface externe · P5 rare déclencheur.   "
             "GBASE : P4 contagion systémique (cloud) · P5 = 1 (partage VOLONTAIRE, art. 45).   "
             "TRANS : asymétrie causale.", fontsize=8.5, color=MUTED)
    finish(fig, os.path.join(OUT, "F7_coefficients_ancrage.png"))


# ================================================================ F8 : robustesse du classement
def fig8():
    y = np.arange(len(top15))
    data = [rank_hist[o] for o in top15]           # top15 est déjà trié (meilleur en 0)
    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    bp = ax.boxplot(data, orientation="horizontal", positions=y, widths=0.6, patch_artist=True,
                    showfliers=False)
    for box in bp["boxes"]:
        box.set(facecolor="#cfe0f7", edgecolor=BLUE, linewidth=1.1)
    for med in bp["medians"]:
        med.set(color=ACCENT, linewidth=2)
    for w in bp["whiskers"] + bp["caps"]:
        w.set(color=BLUE, linewidth=1)
    ax.axvline(15, color=ACCENT, ls="--", lw=1.3)
    ax.text(15.6, len(top15) - 0.5, "frontière\ndu Top-15", color=ACCENT, fontsize=8.5, va="top")
    ax.set_yticks(y); ax.set_yticklabels([chain(o) for o in top15], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Rang du scénario sur 325 (1 = le plus critique) : sur 3000 perturbations",
                  color=INK2)
    ax.set_xlim(0, max(60, np.percentile(np.concatenate(data), 99)))
    ax.set_title("F8 : Le classement tient sous perturbation", fontsize=15,
                 fontweight="bold", color=INK, pad=26, loc="left")
    ax.text(0, -1.0, "Rangs des 15 chaînes du Top baseline quand on tire tous les "
            "coefficients au hasard. Ils restent près du sommet → le classement est robuste.",
            fontsize=9, color=INK2, transform=ax.transData)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    finish(fig, os.path.join(OUT, "F8_robustesse_classement.png"))


# ================================================================ F9 : l'ordre renverse le verdict
def fig9():
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.hist(flip_draws * 100, bins=np.arange(45, 102, 5), color="#9ec5f4",
            edgecolor="#fcfcfb", zorder=3)
    ax.axvline(base_flip * 100, color=ACCENT, lw=2, zorder=4)
    ax.text(base_flip * 100 - 1, ax.get_ylim()[1] * 0.92, f"baseline {base_flip:.0%}",
            color=ACCENT, ha="right", fontsize=9.5, fontweight="bold")
    ax.axvline(flip_draws.mean() * 100, color=INK2, ls="--", lw=1.3, zorder=4)
    ax.text(flip_draws.mean() * 100 + 1, ax.get_ylim()[1] * 0.75,
            f"moyenne {flip_draws.mean():.0%}", color=INK2, fontsize=9)
    ax.set_xlabel("Part des paires de piliers où l'ordre causal domine son inverse  (%)",
                  color=INK2)
    ax.set_ylabel("Nombre de tirages", color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.83)
    fig.text(0.09, 0.95, "F9 : « L'ordre renverse le verdict » survit au bruit",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.09, 0.90, "Sur 3000 jeux de coefficients tirés au hasard, la conclusion phare "
             "reste vraie pour la grande majorité des paires.", fontsize=9, color=INK2)
    finish(fig, os.path.join(OUT, "F9_ordre_robuste.png"))


# ================================================================ F10 : plafond de criticité (FRAGILE)
def fig10():
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    vals, cnts = np.unique(maxc_draws.astype(int), return_counts=True)
    cols = ["#9ec5f4" if v < 9 else ACCENT for v in vals]
    ax.bar(vals, cnts / NDRAW * 100, color=cols, edgecolor="#fcfcfb", width=0.8, zorder=3)
    breach = np.mean(maxc_draws >= 9)
    ax.axvline(8.5, color=ACCENT, ls="--", lw=1.3)
    ax.text(8.5, 27, "seuil « Extrême » (≥9)", color=ACCENT, fontsize=9, rotation=90,
            ha="center", va="center", backgroundcolor="#fcfcfb")
    for v, c in zip(vals, cnts):
        ax.text(v, c / NDRAW * 100 + 1.2, f"{c/NDRAW:.0%}", ha="center", fontsize=8.5, color=INK2)
    ax.text(0.02, 0.9, "Baseline : max = 8 (Majeure)", transform=ax.transAxes,
            fontsize=9, color=INK2)
    ax.set_xlabel("Criticité MAXIMALE atteinte parmi les 325 scénarios (par tirage)", color=INK2)
    ax.set_ylabel("Part des tirages  (%)", color=INK2)
    ax.set_xticks(range(int(vals.min()), 11))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.82)
    fig.text(0.09, 0.95, "F10 : Le plafond de criticité, lui, est FRAGILE",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.09, 0.895, f"Dès qu'on tire les coefficients au hasard, {breach:.0%} des mondes "
             "franchissent « Extrême » (baseline = 8, Majeure). Contrairement aux autres, "
             "cette conclusion n'est PAS robuste : à ne pas sur-vendre.",
             fontsize=9, color=INK2)
    finish(fig, os.path.join(OUT, "F10_plafond_fragile.png"))


# ================================================================ F11 : knockout : qui porte le résultat ?
def fig11():
    pairs = list(combinations(range(1, N + 1), 2))

    def order_effect(params):
        """Amplitude moyenne du renversement : |crit(i→j) − crit(j→i)| sur les 10 paires."""
        return np.mean([abs(score((i, j), *params)[2] - score((j, i), *params)[2])
                        for i, j in pairs])

    def sym_trans():
        return {a: {b: (TRANS0[a][b] + TRANS0[b][a]) / 2 for b in TRANS0[a]} for a in TRANS0}

    ROOT_flat = {p: np.mean(list(ROOT0.values())) for p in ROOT0}
    GBASE_flat = {p: np.mean(list(GBASE0.values())) for p in GBASE0}

    configs = [
        ("Modèle complet (baseline)", BASE, BLUE),
        ("sans asymétrie TRANS", (ROOT0, sym_trans(), GBASE0, KFAC0, AMPB0, AMPS0), "#9ec5f4"),
        ("sans écart ROOT", (ROOT_flat, TRANS0, GBASE0, KFAC0, AMPB0, AMPS0), "#9ec5f4"),
        ("sans écart GBASE", (ROOT0, TRANS0, GBASE_flat, KFAC0, AMPB0, AMPS0), "#9ec5f4"),
        ("sans aucun des trois", (ROOT_flat, sym_trans(), GBASE_flat, KFAC0, AMPB0, AMPS0), ACCENT),
    ]
    names = [c[0] for c in configs]
    vals = [order_effect(c[1]) for c in configs]
    cols = [c[2] for c in configs]
    y = np.arange(len(configs))[::-1]           # baseline en haut

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.barh(y, vals, color=cols, edgecolor="#fcfcfb", height=0.66, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.05, yi, f"{v:.2f}", va="center", fontsize=9, color=INK2)
    ax.axvline(vals[0], color=BLUE, ls=":", lw=1.2, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlabel("Amplitude moyenne du renversement  |Δ criticité entre un ordre et son inverse|",
                  color=INK2)
    ax.set_xlim(0, max(vals) * 1.18)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.82)
    fig.text(0.09, 0.95, "F11 : Qu'est-ce qui fait que l'ordre compte ?",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.09, 0.895, "On NEUTRALISE chaque hypothèse (coefficients rendus égaux) et on mesure "
             "ce qui reste de l'effet d'ordre. La barre qui chute le plus = l'hypothèse porteuse.",
             fontsize=9, color=INK2)
    finish(fig, os.path.join(OUT, "F11_knockout_hypotheses.png"))


if __name__ == "__main__":
    fig7(); fig8(); fig9(); fig10(); fig11()
    print(f"Figures dans : {OUT}")
