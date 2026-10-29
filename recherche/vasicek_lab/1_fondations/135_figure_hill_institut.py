#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
135 : graphique de Hill de l'annexe des valeurs extremes, version Institut a 150 pages.

La figure hill_plot_ic90_oprisk.png venait du script archive 01_oprisk_gpd_calibration.py,
qui filtrait le perimetre par ses propres categories et n'etait plus rejoue. Ce script la retrace
sur la population du script 47 (load_clean, filter_cyber, filter_finance, conversion USD_EUR),
celle qui imprime xi_Hill = 1,421 a k = 86 sous OPRISK_ANNEE_MIN=2000.

Bande : intervalle asymptotique a 90 %, xi_Hill(k) +/- 1,645 xi_Hill(k) / sqrt(k), convention du
script archive et de la legende. Reference : l'indice du maximum de vraisemblance, LU dans
src/utils/config.py (calibration gelee).

Controles, et le script s'arrete au moindre echec : la periode est bien 2000-2026 ; le nombre de
pertes et l'indice de reference sont ceux que le memoire publie ; Hill a k = 86 redonne 1,421.

Lancement : OPRISK_ANNEE_MIN=2000, sortie dans resultats/sorties_verif_inst/135.txt (stdout seul),
figure dans figures_inst/ (nom inchange, appele par l'annexe).
"""
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (REPO, LAB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import style_nexialog as sn  # noqa: E402
from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR  # noqa: E402
from src.utils.config import OPRISK  # noqa: E402

K_MAX = 200
Z90 = 1.645
N_PUBLIE = 570
XI_PUBLIE = 0.5954
HILL_86_PUBLIE = 1.421

print("=" * 78)
print("135 : GRAPHIQUE DE HILL, ECHANTILLON 2000-2026")
print("=" * 78)

if os.environ.get("OPRISK_ANNEE_MIN") != "2000":
    sys.exit("ARRET : lancer avec OPRISK_ANNEE_MIN=2000 (calibration de la version Institut)")

d = filter_finance(filter_cyber(load_clean(
    os.path.join(REPO, "data", "raw", "SAS_OpRisk_Global_Data_June_2026.xlsx"))))
loss = np.sort(d["loss"].to_numpy() * USD_EUR)[::-1]
xi_ref = float(OPRISK["xi"])
if loss.size != N_PUBLIE:
    sys.exit(f"ARRET : {loss.size} pertes, {N_PUBLIE} attendues")
if abs(xi_ref - XI_PUBLIE) > 1e-12:
    sys.exit(f"ARRET : indice de reference {xi_ref}, {XI_PUBLIE} attendu")

lx = np.log(loss[:K_MAX + 1])
k = np.arange(1, K_MAX + 1)
hill = np.cumsum(lx[:K_MAX]) / k - lx[1:K_MAX + 1]
se = hill / np.sqrt(k)
bas, haut = hill - Z90 * se, hill + Z90 * se
if abs(hill[85] - HILL_86_PUBLIE) > 5e-4:
    sys.exit(f"ARRET : Hill a k = 86 vaut {hill[85]:.4f}, {HILL_86_PUBLIE} attendu (script 47)")

dedans = (bas <= xi_ref) & (xi_ref <= haut)
k_dedans = k[dedans]
# Plus grand k a partir duquel la reference sort de la bande pour ne plus y revenir.
k_sortie = int(k[~dedans][k[~dedans] > k_dedans.max()].min()) if (~dedans).any() else None

print(f"\n  {loss.size} pertes cyber x finance, periode 2000-2026 ; reference EMV xi = {xi_ref:.4f}")
print(f"\n  {'k':>5}{'xi_Hill':>10}{'borne basse':>14}{'borne haute':>14}{'reference dans la bande':>26}")
for kk in (1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 86, 120, 150, 200):
    i = kk - 1
    print(f"  {kk:>5}{hill[i]:>10.3f}{bas[i]:>14.3f}{haut[i]:>14.3f}{'oui' if dedans[i] else 'non':>26}")
print(f"\n  la reference est dans la bande pour {dedans.sum()} valeurs de k sur {K_MAX},"
      f" de k = {k_dedans.min()} a k = {k_dedans.max()}")
print(f"  hors de la bande pour tout k >= {k_sortie}")
print(f"  minimum de xi_Hill sur k <= 10 : {hill[:10].min():.3f} ; maximum sur k <= {K_MAX} :"
      f" {hill.max():.3f} (k = {int(k[np.argmax(hill)])})")

sn.appliquer(taille=9)
mpl.rcParams["font.family"] = ["DejaVu Sans", "Segoe UI", "sans-serif"]
bleu, rouge = sn.CATEGORIEL[2], sn.CATEGORIEL[0]
ref_txt = f"{xi_ref:.3f}".replace(".", "{,}")

fig, ax = plt.subplots(figsize=(7.0, 3.0))
ax.fill_between(k, bas, haut, color=bleu, alpha=0.15, lw=0, label="intervalle asymptotique à 90 %")
ax.plot(k, hill, color=bleu, lw=1.6, label="estimateur de Hill")
ax.axhline(xi_ref, color=rouge, ls="--", lw=1.3, label=r"EMV, $\hat\xi=" + ref_txt + r"$")
ax.set_xlim(1, K_MAX)
ax.set_xlabel("nombre de plus grandes pertes retenues ($k$)")
ax.set_ylabel(r"estimation de $\xi$")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
sortie = os.path.join(LAB, "figures_inst", "hill_plot_ic90_oprisk.png")
fig.savefig(sortie, dpi=200, bbox_inches="tight")
print(f"\nfigure : figures_inst/{os.path.basename(sortie)}")

print("\nGrandeurs citees (sans separateur) :")
print(f"  n {loss.size}   k_dedans_min {k_dedans.min()}   k_dedans_max {k_dedans.max()}"
      f"   k_sortie {k_sortie}   hill_k86 {hill[85]:.3f}   hill_k200 {hill[-1]:.3f}")
