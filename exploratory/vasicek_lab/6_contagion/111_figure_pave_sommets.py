#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
111 : figure Z_pave_sommets, le pave admissible des directions et le capital qu'il donne.

LE NOM DE FICHIER NE PORTE AUCUN CHIFFRE, et c'est voulu : le harnais lit les nombres du nom
passe a \figover, et un premier nom « Z22_... » y versait un 22 confirme par hasard contre
lambda = 21,56 du script 30.

Figure du paragraphe « Ce que le capital devient » du chapitre de l'identification partielle.
Elle rend visible ce que les scripts 30 et 61 ne font qu'imprimer :
  (a) une coupe du pave admissible |A_jk| <= S_jk sur deux des dix paires de piliers, avec
      ses sommets, la direction nulle, le point d'expert et les points interieurs tires ;
  (b) le besoin de capital en chacun des 1 024 sommets et des 3 000 points interieurs, a t = 1,
      qui montre que les extremes vivent aux sommets et qu'aucun point interieur n'en sort.

Le calcul est celui du script 61 : meme moteur, meme graine, meme sequence de tirages. Le socle,
la direction nulle, le point d'expert et les bornes sont LUS dans sorties_verif/30.txt,
l'amplitude interieure dans 61.txt, et le script s'arrete si son propre calcul ne les retrouve
pas : sans ce controle, la figure pourrait montrer un autre modele que celui publie.

La sortie standard est deterministe et versionnee (111.txt) ; la duree du calcul et le recours
au cache partent sur la sortie d'erreur. FIG_CACHE=1 relit le dernier calcul pour retoucher la
mise en page, les controles etant passes dans les deux cas.
"""

import os
import re
import sys
import tempfile
import time

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RACINE = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import partial_id as pid                                        # noqa: E402
import style_nexialog as sn                                     # noqa: E402

SEED = 20260721            # meme graine que 30 / 55 / 56 / 61
N_INT = 3000
PAIRE_X, PAIRE_Y = 0, 2    # P1-P2 et P1-P4 dans l'ordre de pid.IU


def lire(nom):
    with open(os.path.join(RACINE, "sorties_verif", nom), encoding="utf-8") as f:
        return f.read()


def nombre(motif, texte):
    m = re.search(motif, texte)
    if not m:
        raise SystemExit(f"motif introuvable dans la sortie versionnee : {motif}")
    return [float(x) for x in m.groups()]


def fr(x):
    return f"{x:,.0f}".replace(",", " ")


s30, s61 = lire("30.txt"), lire("61.txt")
(SOCLE,) = nombre(r"SOCLE \(W = 0.*?:\s+(\d+) M", s30)
(NUL,) = nombre(r"W symetrique \(A = 0.*?:\s+(\d+) M", s30)
(EXPERT,) = nombre(r"W d'expert.*?:\s+(\d+) M", s30)
LO, HI = nombre(r"t = 1\.00 : SCR dans \[\s*(\d+) ;\s*(\d+)\]", s30)
INT_LO, INT_HI = nombre(r"amplitude interieure SCR : \[(\d+) ; (\d+)\]", s61)

CACHE = os.path.join(tempfile.gettempdir(), f"fig_pave_sommets_{SEED}.npz")
t0 = time.time()
P = pid.expert_matrix()
S, A_EXP = pid.decompose(P)
smax = S[pid.IU]
if os.environ.get("FIG_CACHE") and os.path.exists(CACHE):
    c = np.load(CACHE)
    v_scr, i_scr, i_coord = c["v_scr"], c["i_scr"], c["i_coord"]
    scr_nul, scr_exp = float(c["scr_nul"]), float(c["scr_exp"])
    print(f"resultats relus dans le cache {CACHE}", file=sys.stderr)
else:
    ev = pid.Evaluator(n_years=40_000, seed=SEED)
    rng = np.random.default_rng(SEED)
    for _ in range(2000):          # tirages consommes par la proposition P2 du script 61
        rng.random(pid.NFREE)
    scr_nul, _ = ev(S)
    scr_exp, _ = ev(P)
    v_scr = []
    for m in range(1 << pid.NFREE):
        sg = np.array([(m >> k & 1) * 2 - 1 for k in range(pid.NFREE)], dtype=float)
        W = pid.build_W(S, sg * smax)
        if pid.admissible(W):
            v_scr.append(ev(W)[0])
    v_scr = np.array(v_scr)
    i_scr, i_coord = [], []
    for _ in range(N_INT):
        a = (rng.random(pid.NFREE) * 2 - 1) * smax
        W = pid.build_W(S, a)
        if not pid.admissible(W):
            continue
        i_scr.append(ev(W)[0])
        i_coord.append(a / smax)
    i_scr, i_coord = np.array(i_scr), np.array(i_coord)
    np.savez(CACHE, v_scr=v_scr, i_scr=i_scr, i_coord=i_coord, scr_nul=scr_nul, scr_exp=scr_exp)
print(f"duree du calcul : {time.time() - t0:.0f} s", file=sys.stderr)

print("=" * 84)
print("Z_pave_sommets : le pave admissible des directions et le capital qu'il donne (t = 1)")
print("=" * 84)
print(f"  sommets evalues : {len(v_scr)}   points interieurs evalues : {len(i_scr)}")
print(f"  coupe tracee : paires P1-P2 et P1-P4, coordonnees A_jk / S_jk dans [-1 ; 1]")
print(f"  point d'expert dans la coupe : ({A_EXP[pid.IU][PAIRE_X] / smax[PAIRE_X]:+.3f} ; "
      f"{A_EXP[pid.IU][PAIRE_Y] / smax[PAIRE_Y]:+.3f})")
print("\n  controles contre les sorties versionnees 30 et 61 :")
controles = [
    ("borne basse aux sommets", v_scr.min(), LO),
    ("borne haute aux sommets", v_scr.max(), HI),
    ("minimum interieur", i_scr.min(), INT_LO),
    ("maximum interieur", i_scr.max(), INT_HI),
    ("direction nulle (A = 0)", scr_nul, NUL),
    ("point d'expert", scr_exp, EXPERT),
]
for nom, calcule, publie in controles:
    ecart = abs(round(calcule) - publie)
    print(f"    {nom:<26} calcule {calcule:8.1f}   publie {publie:6.0f}   "
          f"{'ok' if ecart <= 1 else 'ECART'}")
    if ecart > 1:
        raise SystemExit("le calcul ne retrouve pas la valeur publiee : figure non produite")
dehors = int(((i_scr < v_scr.min() - 1e-9) | (i_scr > v_scr.max() + 1e-9)).sum())
print(f"    points interieurs hors des bornes des sommets : {dehors}")
if dehors:
    raise SystemExit("un point interieur sort des bornes : l'hypothese des sommets tombe")
print("\n  Lecture : les sommets portent les extremes, de la borne basse a la borne haute ;")
print("  les points interieurs se concentrent au milieu et n'en sortent jamais.")

# =====================================================================================
sn.appliquer(10)
mpl.rcParams["font.family"] = ["DejaVu Sans", "Segoe UI", "sans-serif"]
ROUGE, TURQ, BLEU = sn.CATEGORIEL
FOND_TXT = dict(boxstyle="round,pad=0.12", fc=sn.FOND, ec="none", alpha=0.9)
u_exp = A_EXP[pid.IU] / smax

fig = plt.figure(figsize=(8.6, 3.25))
gsp = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.6], wspace=0.24)
ax1, ax2 = fig.add_subplot(gsp[0, 0]), fig.add_subplot(gsp[0, 1])

# (a) la coupe du pave
ax1.scatter(i_coord[:, PAIRE_X], i_coord[:, PAIRE_Y], s=3, color=BLEU, alpha=0.30, lw=0,
            zorder=1)
ax1.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, lw=1.0, ec=sn.SEQUENTIEL_6[3],
                        ls="--", zorder=2))
ax1.text(-0.47, -0.47, "t = 0,5", fontsize=7.5, color=sn.SEQUENTIEL_6[2], ha="left",
         va="bottom", zorder=4, bbox=FOND_TXT)
ax1.add_patch(Rectangle((-1, -1), 2, 2, fill=False, lw=1.5, ec=sn.ENCRE_2, zorder=2))
ax1.text(-0.96, -0.96, "t = 1", fontsize=7.5, color=sn.ENCRE_2, ha="left", va="bottom",
         zorder=4, bbox=FOND_TXT)
coins = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
ax1.scatter(coins[:, 0], coins[:, 1], s=48, color=ROUGE, zorder=5)
ax1.text(1.0, 1.1, "sommets : chaque paire\norientée à fond", fontsize=7.5, color=ROUGE,
         ha="right", va="bottom")
ax1.scatter([0], [0], s=40, marker="s", color=sn.ENCRE, zorder=5)
ax1.text(0.07, -0.05, "A = 0", fontsize=7.5, color=sn.ENCRE, ha="left", va="top", zorder=5,
         bbox=FOND_TXT)
ax1.scatter([u_exp[PAIRE_X]], [u_exp[PAIRE_Y]], s=120, marker="*", color=TURQ, zorder=6,
            ec=sn.ENCRE, lw=0.5)
ax1.text(u_exp[PAIRE_X] + 0.07, u_exp[PAIRE_Y] + 0.04, "expert", fontsize=7.5, color=TURQ,
         fontweight="bold", ha="left", va="bottom", zorder=6, bbox=FOND_TXT)
ax1.text(-0.96, 0.93, "points intérieurs", fontsize=7.5, color=BLEU, ha="left", va="top",
         zorder=4, bbox=FOND_TXT)
ax1.set_xlim(-1.25, 1.25)
ax1.set_ylim(-1.25, 1.42)
ax1.set_aspect("equal")
ax1.set_xticks([-1, 0, 1])
ax1.set_xticklabels(["−1\nP2 → P1", "0", "+1\nP1 → P2"], fontsize=8)
ax1.set_yticks([-1, 0, 1])
ax1.set_yticklabels(["P4 → P1  −1", "0", "P1 → P4  +1"], fontsize=8)
ax1.set_xlabel("paire P1–P2 : $A_{12}/S_{12}$", fontsize=8.5, labelpad=2)
ax1.set_ylabel("paire P1–P4 : $A_{14}/S_{14}$", fontsize=8.5, labelpad=2)
ax1.set_title("(a)  Le pavé des directions possibles,\ncoupe sur deux des dix paires",
              fontsize=9.5, pad=6)

# (b) le capital aux sommets et a l'interieur
bords = np.linspace(SOCLE - 60, HI + 160, 58)
poids_v = np.full(len(v_scr), 100 / len(v_scr))
poids_i = np.full(len(i_scr), 100 / len(i_scr))
ymax = max(np.histogram(v_scr, bins=bords, weights=poids_v)[0].max(),
           np.histogram(i_scr, bins=bords, weights=poids_i)[0].max())
ax2.hist(v_scr, bins=bords, weights=poids_v, color=ROUGE, alpha=0.55,
         label=f"les {fr(len(v_scr))} sommets")
ax2.hist(i_scr, bins=bords, weights=poids_i, color=BLEU, alpha=0.55,
         label=f"les {fr(len(i_scr))} points intérieurs")
ax2.axvline(SOCLE, color=sn.ENCRE_3, lw=1.4, ls=":")
ax2.text(SOCLE + 40, ymax * 1.50, f"socle, W = 0\n{fr(SOCLE)}", fontsize=7.5,
         color=sn.ENCRE_3, ha="left", va="bottom")
for x, nom in ((LO, "borne basse"), (HI, "borne haute")):
    ax2.axvline(x, color=ROUGE, lw=1.6)
    ax2.text(x, ymax * 1.50, f"{nom}\n{fr(x)}", color=ROUGE, fontsize=7.5, ha="center",
             va="bottom", fontweight="bold", bbox=FOND_TXT, zorder=5)
ax2.axvline(NUL, color=sn.ENCRE, lw=1.1, ls="--")
ax2.axvline(EXPERT, color=TURQ, lw=1.6)
ax2.text(NUL - 25, ymax * 1.24, f"A = 0\n{fr(NUL)}", fontsize=7.5, color=sn.ENCRE,
         ha="right", va="bottom", bbox=FOND_TXT)
ax2.text(EXPERT + 25, ymax * 1.24, f"expert\n{fr(EXPERT)}", fontsize=7.5, color=TURQ,
         ha="left", va="bottom", fontweight="bold", bbox=FOND_TXT)
ax2.annotate("", xy=(INT_LO, ymax * 1.07), xytext=(INT_HI, ymax * 1.07),
             arrowprops=dict(arrowstyle="|-|", color=BLEU, lw=1.2, mutation_scale=3))
ax2.text(INT_LO - 30, ymax * 1.07, f"intérieur : {fr(INT_LO)} à {fr(INT_HI)}", fontsize=7.5,
         color=BLEU, ha="right", va="center", bbox=FOND_TXT)
ax2.set_ylim(0, ymax * 1.72)
ax2.set_xlim(SOCLE - 120, HI + 230)
ax2.set_xlabel("besoin de capital à 99,5 % (M€)", fontsize=8.5, labelpad=2)
ax2.set_ylabel("part des points (%)", fontsize=8.5, labelpad=2)
ax2.tick_params(labelsize=8)
ax2.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(0.0, 0.40))
ax2.set_title("(b)  Le capital aux sommets et à l'intérieur, à t = 1 :\n"
              "aucun point intérieur ne sort des bornes", fontsize=9.5, pad=6)

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z_pave_sommets.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
