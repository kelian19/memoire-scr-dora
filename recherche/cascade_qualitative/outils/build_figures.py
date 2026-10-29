#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5 figures qualitatives à partir de cascade_piliers_DORA.xlsx.

Chaque figure isole UN « message caché » du modèle conceptuel :
  F1 : L'ordre renverse le verdict          (asymétrie i→j vs j→i)
  F2 : Jamais très probable ET très grave    (plafond de criticité)
  F3 : Mêmes piliers, autre histoire         (amplitude du ré-ordonnancement)
  F4 : Les scénarios les plus critiques       (classement : piliers amont seuls + cascades)
  F5 : Là où la cascade démarre décide        (pilier-racine → profil de criticité)

Le moteur de score (ROOT / TRANS / GBASE) et le barème sont importés de
cascade_model.py, source unique partagée avec le classeur et la sensibilité :
aucune re-déclaration, donc aucun risque de désynchronisation. Seuls les noms
courts des piliers (labels de figures) restent locaux.
"""

from itertools import combinations, permutations
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ---------------------------------------------------------------- moteur (miroir du .xlsx)
PILIERS = {
    1: "Gouvernance & risque TIC",
    2: "Gestion des incidents",
    3: "Tests de résilience (TLPT)",
    4: "Risque lié aux tiers ICT",
    5: "Partage d'informations",
}
N = len(PILIERS)

# --- jugement et barème : source unique (cascade_model.py) ------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cascade_model import (ROOT, TRANS, GBASE, CRIT_LAB,
                           lvl, proba_score, gravite_score, crit_score)


# ---------------------------------------------------------------- construction des records
def build_records():
    recs = []
    for k in range(0, N + 1):
        for subset in combinations(range(1, N + 1), k):
            perms = [()] if k == 0 else list(permutations(subset))
            for order in perms:
                ps, gs = proba_score(order), gravite_score(order)
                recs.append({
                    "subset": subset, "k": k, "order": order,
                    "p": ps, "g": gs, "c": crit_score(ps, gs),
                })
    return recs


RECS = build_records()
SCORED = [r for r in RECS if r["c"] is not None]

# ---------------------------------------------------------------- style / palette validée
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "font.size": 11,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "text.color": "#0b0b0b",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#898781",
    "ytick.color": "#898781",
    "axes.grid": False,
})
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID = "#e1e0d9"
ACCENT = "#eb6834"   # orange slot 8 : un seul point d'attention par figure
# rampe séquentielle BLEUE validée (palette.md, steps 100→700), pour la magnitude
BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blues_seq", BLUES)
# 5 pas ordinaux (criticité Faible→Extrême), lisibles sur fond clair (≥ step 250)
CRIT_STEPS = ["#b7d3f6", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)


def chain_lbl(order):
    return "→".join(str(x) for x in order)


def finish(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {os.path.relpath(path)}")


# ================================================================ F1 : asymétrie i→j
def fig1():
    M = np.full((N, N), np.nan)
    for r in SCORED:
        if r["k"] == 2:
            i, j = r["order"]
            M[i - 1, j - 1] = r["c"]

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(M, cmap=SEQ, vmin=1, vmax=8)
    for i in range(N):
        for j in range(N):
            if not np.isnan(M[i, j]):
                v = int(M[i, j])
                ax.text(j, i, CRIT_LAB[lvl(v)], ha="center", va="center",
                        fontsize=8.5, color="#ffffff" if v >= 5 else INK)
            else:
                ax.text(j, i, "-", ha="center", va="center", color=MUTED)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels([f"P{j}" for j in range(1, N + 1)])
    ax.set_yticklabels([f"P{i}" for i in range(1, N + 1)])
    ax.set_xlabel("… puis ce pilier  (aval)", color=INK2)
    ax.set_ylabel("La cascade part de ce pilier  (amont)", color=INK2)
    ax.set_xticks(np.arange(-.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-.5, N, 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", length=0)
    # souligner le couple emblématique 1↔2
    ax.add_patch(plt.Rectangle((0.5, -0.5), 1, 1, fill=False, ec=ACCENT, lw=2.4))  # 1→2 (P1→P2)
    ax.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False, ec=ACCENT, lw=2.4))  # 2→1 (P2→P1)
    fig.subplots_adjust(top=0.83)
    fig.text(0.125, 0.95, "L'ordre renverse le verdict", fontsize=15,
             fontweight="bold", color=INK)
    fig.text(0.125, 0.90, "Criticité d'une cascade à 2 piliers selon le SENS. "
             "Cadres orange : 1→2 « Majeure » vs 2→1 « Faible » : mêmes piliers.",
             fontsize=9, color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[1, 4, 8])
    cb.ax.set_yticklabels(["faible", "moyenne", "élevée"], fontsize=8)
    cb.outline.set_edgecolor(GRID)
    finish(fig, os.path.join(OUT, "F1_asymetrie_ordre.png"))


# ================================================================ F2 : plafond de criticité
def fig2():
    xs = np.array([r["p"] for r in SCORED], float)
    ys = np.array([r["g"] for r in SCORED], float)
    cs = np.array([r["c"] for r in SCORED], float)
    rng = np.random.default_rng(7)
    jx = xs + rng.uniform(-.18, .18, len(xs))
    jy = ys + rng.uniform(-.18, .18, len(ys))

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    # iso-criticité (moyenne géométrique) en fond
    gx = np.linspace(0.6, 10.4, 300)
    for c in [2, 4, 6, 8]:
        ax.plot(gx, (c ** 2) / gx, color=GRID, lw=1, zorder=1)
        ax.text(10.3, (c ** 2) / 10.3, f"crit {c}", fontsize=7.5, color=MUTED,
                va="center", ha="left")
    sc = ax.scatter(jx, jy, c=cs, cmap=SEQ, vmin=1, vmax=8, s=46,
                    edgecolor="#fcfcfb", linewidth=0.6, zorder=3)
    # zone interdite : très probable ET très grave
    ax.add_patch(plt.Rectangle((7.5, 7.5), 3.2, 3.2, facecolor=ACCENT, alpha=0.08,
                               edgecolor=ACCENT, lw=1.4, ls="--", zorder=2))
    ax.text(9.1, 9.1, "vide", ha="center", va="center", color=ACCENT,
            fontsize=11, fontweight="bold")
    ax.text(9.1, 8.5, "ni très probable\nET très grave", ha="center", va="top",
            color=ACCENT, fontsize=8.2)
    ax.set_xlim(0.4, 10.8); ax.set_ylim(0.4, 10.8)
    ax.set_xlabel("Probabilité conceptuelle  /10", color=INK2)
    ax.set_ylabel("Gravité conceptuelle  /10", color=INK2)
    ax.set_title("Jamais très probable ET très grave", fontsize=15,
                 fontweight="bold", color=INK, pad=24, loc="left")
    ax.text(0.4, 11.15, "325 scénarios ordonnés. La coche haut-droite reste vide "
            "→ la criticité plafonne à « Majeure ».", fontsize=9, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, ticks=[1, 4, 8])
    cb.set_label("criticité", fontsize=9, color=INK2)
    cb.outline.set_edgecolor(GRID)
    finish(fig, os.path.join(OUT, "F2_plafond_criticite.png"))


# ================================================================ F3 : amplitude du ré-ordonnancement
def fig3():
    sets = {}
    for r in SCORED:
        if r["k"] >= 2:
            sets.setdefault(r["subset"], []).append(r["c"])
    rows = []
    for subset, vals in sets.items():
        rows.append((subset, min(vals), max(vals)))
    rows.sort(key=lambda x: (x[2] - x[1], x[2]))   # tri par amplitude

    labels = ["·".join(f"P{p}" for p in s) for s, _, _ in rows]
    lo = np.array([r[1] for r in rows]); hi = np.array([r[2] for r in rows])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8.2, 8.6))
    for yi, l, h in zip(y, lo, hi):
        ax.plot([l, h], [yi, yi], color=GRID, lw=3, solid_capstyle="round", zorder=1)
    ax.scatter(lo, y, s=70, color="#b7d3f6", edgecolor="#2a78d6", lw=1.2,
               zorder=3, label="meilleur ordre (min)")
    ax.scatter(hi, y, s=70, color=ACCENT, edgecolor="#a83c12", lw=1.2,
               zorder=3, label="pire ordre (max)")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0.5, 8.7)
    ax.set_xlabel("Criticité : de l'ordre le plus favorable au plus défavorable", color=INK2)
    for xb, lab in zip([1.5, 3.5, 5.5, 7.5], CRIT_LAB[:4]):
        ax.axvline(xb + 0.5, color=GRID, lw=0.8, zorder=0)
    ax.set_title("Mêmes piliers, autre histoire", fontsize=15, fontweight="bold",
                 color=INK, pad=24, loc="left")
    ax.text(0.5, len(rows) + 0.4, "Pour chaque ENSEMBLE de piliers : écart de verdict "
            "entre son meilleur et son pire ordre d'apparition.", fontsize=9, color=INK2)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    finish(fig, os.path.join(OUT, "F3_amplitude_reordonnancement.png"))


# ================================================================ F4 : top des cascades critiques
def fig4():
    top = sorted(SCORED, key=lambda r: (r["c"], r["g"], r["p"]), reverse=True)[:15]
    top = top[::-1]
    labels = [chain_lbl(r["order"]) for r in top]
    crit = np.array([r["c"] for r in top], float)
    y = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    colors = [SEQ((c - 1) / 7) for c in crit]
    ax.barh(y, crit, color=colors, edgecolor="#fcfcfb", height=0.72, zorder=3)
    for yi, r in zip(y, top):
        ax.text(r["c"] + 0.12, yi, f"{CRIT_LAB[lvl(r['c'])]}  (p{r['p']}·g{r['g']})",
                va="center", fontsize=8.4, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10, fontfamily="DejaVu Sans")
    ax.set_xlim(0, 10.2)
    ax.set_xlabel("Score de criticité", color=INK2)
    ax.set_title("Les 15 scénarios les plus critiques", fontsize=15, fontweight="bold",
                 color=INK, pad=24, loc="left")
    ax.text(0, len(top) + 0.2, "En tête : des piliers amont seuls (P1 gouvernance / P4 tiers) "
            "et des cascades qui en partent vers l'aval ; jamais l'inverse.", fontsize=9, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    finish(fig, os.path.join(OUT, "F4_top_cascades.png"))


# ================================================================ F5 : pilier-racine décide
def fig5():
    # profil de criticité (répartition %) selon le pilier qui AMORCE la cascade
    roots = range(1, N + 1)
    prof = np.zeros((N, 5))
    for r in SCORED:
        prof[r["order"][0] - 1, lvl(r["c"])] += 1
    pct = prof / prof.sum(axis=1, keepdims=True) * 100
    # ordonner les racines par part de criticité élevée (Élevée+Majeure)
    order_idx = np.argsort(pct[:, 2:].sum(axis=1))
    pct = pct[order_idx]; root_lbls = [f"P{i+1} : {PILIERS[i+1]}" for i in order_idx]

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    left = np.zeros(N)
    for lv in range(5):
        ax.barh(range(N), pct[:, lv], left=left, color=CRIT_STEPS[lv],
                edgecolor="#fcfcfb", height=0.66, zorder=3,
                label=CRIT_LAB[lv])
        for yi in range(N):
            w = pct[yi, lv]
            if w >= 7:
                ax.text(left[yi] + w / 2, yi, f"{w:.0f}", ha="center", va="center",
                        fontsize=8, color="#ffffff" if lv >= 2 else INK)
        left += pct[:, lv]
    ax.set_yticks(range(N)); ax.set_yticklabels(root_lbls, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Répartition des scénarios de cette racine, par criticité  (%)", color=INK2)
    ax.set_title("Là où la cascade démarre décide de tout", fontsize=15,
                 fontweight="bold", color=INK, pad=30, loc="left")
    ax.text(0, N - 0.35, "Criticité selon le pilier qui AMORCE la chaîne. "
            "P1/P4 en tête → verdicts lourds ; P5 → quasi jamais.", fontsize=9, color=INK2)
    ax.legend(ncol=5, frameon=False, fontsize=8.4, loc="lower center",
              bbox_to_anchor=(0.5, -0.24), handlelength=1.1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    finish(fig, os.path.join(OUT, "F5_pilier_racine.png"))


# ================================================================ F6 : carte de tous les scores
def fig6():
    # tous les scénarios notés, du plus critique au moins critique
    rows = sorted(SCORED, key=lambda r: (r["c"], r["g"], r["p"]), reverse=True)
    n = len(rows)
    ncol = 5
    per = -(-n // ncol)                        # ceil : lignes par colonne

    chain_w, cell_w, gap = 2.6, 0.92, 0.7
    block_w = chain_w + 3 * cell_w + gap
    header_h = 1.6

    fig, ax = plt.subplots(figsize=(16.5, 12.8))
    ax.set_xlim(-0.3, ncol * block_w)
    ax.set_ylim(per + header_h, -1.4)          # y inversé (haut = plus critique)
    ax.axis("off")

    def cell(x, y, val, vmax):
        col = SEQ(min(val, vmax) / vmax)
        ax.add_patch(plt.Rectangle((x, y - 0.42), cell_w, 0.84, facecolor=col,
                                   edgecolor="#fcfcfb", lw=1.1))
        ax.text(x + cell_w / 2, y, str(val), ha="center", va="center",
                fontsize=6.8, color="#ffffff" if val / vmax >= 0.5 else INK)

    for c in range(ncol):
        x0 = c * block_w
        # en-tête de colonne
        ax.text(x0, -1.0, "chaîne", fontsize=8, color=MUTED, fontstyle="italic")
        for j, (lab, tip) in enumerate([("p", "proba"), ("g", "gravité"), ("c", "crit")]):
            ax.text(x0 + chain_w + j * cell_w + cell_w / 2, -1.0, lab,
                    ha="center", fontsize=8, color=MUTED, fontweight="bold")
        for r in range(per):
            i = c * per + r
            if i >= n:
                break
            rec = rows[i]
            y = r
            ax.text(x0, y, chain_lbl(rec["order"]), ha="left", va="center",
                    fontsize=7, color=INK)
            cell(x0 + chain_w + 0 * cell_w, y, rec["p"], 10)
            cell(x0 + chain_w + 1 * cell_w, y, rec["g"], 10)
            cell(x0 + chain_w + 2 * cell_w, y, rec["c"], 8)

    fig.subplots_adjust(top=0.90, left=0.02, right=0.99, bottom=0.02)
    fig.text(0.02, 0.965, "Carte des scores : 325 scénarios notés de piliers DORA",
             fontsize=16, fontweight="bold", color=INK)
    fig.text(0.02, 0.935, "Chaque chaîne d'apparition avec ses scores conceptuels "
             "proba (p) · gravité (g) · criticité (c). Trié du plus critique (haut) au moins critique. "
             "Plus foncé = plus élevé. (Le cas « aucun pilier actif » n'a pas de score.)",
             fontsize=9.5, color=INK2)
    finish(fig, os.path.join(OUT, "F6_carte_scores.png"))


if __name__ == "__main__":
    print(f"{len(SCORED)} scénarios notés. Génération des figures :")
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print(f"Figures dans : {OUT}")
