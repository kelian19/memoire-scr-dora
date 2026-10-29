#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 : D'ou vient reellement le seuil K_j, et ce que l'EVT peut (et ne peut pas) faire.

Le script 02 a montre que Phi^-1(PD) sous-provisionne et que l'EVT tombe juste sur le
CAPITAL. Il n'a jamais construit K_j. Ce script comble le trou, et corrige au passage
une erreur de raisonnement.

--------------------------------------------------------------------------------
DEUX SEUILS, DEUX ESPACES. Il faut cesser de les confondre.

  u_j : seuil de MATERIALITE (DORA art. 18). Vit dans l'espace des severites
        OBSERVABLES : clients affectes, heures d'indisponibilite, euros.
  K_j : seuil sur la variable LATENTE X_ij. Sans unite.

On ne peut pas "prendre u_j comme K_j" : ils ne vivent pas dans le meme espace.
Mais ils definissent le MEME evenement, l'incident majeur :

      { S_ij >= u_j }   ==   { X_ij >= K_j }

d'ou, en normalisant X a variance unitaire, une identite INEVITABLE :

      K_j = F^-1( 1 - p_j )      avec  p_j = P(S_ij >= u_j)

On inverse donc TOUJOURS une fonction de repartition. Ce qu'on reproche a
Phi^-1(PD), ce n'est pas l'inversion : ce sont ses DEUX ENTREES, une PD non
mesurable et une queue gaussienne.

--------------------------------------------------------------------------------
RESULTAT 1 (exact). Le biais de la frequence naive est exactement le taux moyen de
declaration a la barre :

      p_naif / p_vrai  =  E[ q(S) | S >= u ]

Il est toujours < 1 : on sous-estime p_j, donc on SUR-estime K_j, donc on
SOUS-PROVISIONNE. C'est le mecanisme derriere le "-14 %" du script 02.

RESULTAT 2 (le levier). Ce biais s'evanouit quand la barre u monte dans la queue,
car q(u) -> 1. La barre DORA n'est pas defendable parce qu'elle est "reglementaire" :
elle est defendable parce qu'elle est placee assez HAUT pour que la declaration y
soit quasi complete. C'est une propriete VERIFIABLE : il suffit d'estimer q(u).

RESULTAT 3 (negatif, et il faut l'assumer). On pourrait croire que l'EVT debiaise p_j
en extrapolant la queue depuis un seuil haut v (bien declare) vers la barre u :

      lambda_u = lambda_v * (1 + xi (v-u)/beta_u)^(1/xi),   beta_u = beta_v - xi (v-u)

CE N'EST PAS FIABLE, et la raison est structurelle. Le rapport se decompose en

      p_EVT / p_vrai  =  E[q(S) | S > v]  x  (ratio_estime / ratio_vrai)
                         ^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         plancher de          bruit d'estimation sur xi et
                         declaration a v      beta, amplifie par (v-u)

Quand v monte, le plancher tend vers 1 (bon) mais l'erreur de ratio s'ecarte de 1
(mauvais). Deux forces opposees : il existe un optimum interieur, QU'ON NE PEUT PAS
LOCALISER sans connaitre la verite. Et le gain eventuel provient du plancher, c'est-a-dire
du seul fait de "regarder plus haut" : exactement le levier du RESULTAT 2. L'extrapolation
EVT n'ajoute que du bruit a un levier qu'on avait deja.

Le role de l'EVT reste celui du script 02 : la SEVERITE au-dessus de u_j (donc le
capital), avec un indice de queue xi invariant a la troncature a gauche. Elle ne
touche pas au seuil.

Sortie : diagnostics + figure L_seuil_Kj.png
"""

import os
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

rng = np.random.default_rng(20260709)

# ---------------------------------------------------------------- verite (DGP)
# xi ANCRE SUR SAS OpRisk (pertes en dollars, secteur financier, POT q95) -> 0,85-0,94.
# On retient 0,90, comme le script 02. Variance infinie ; moyenne finie de justesse.
XI, SCALE = 0.90, 1.0      # severites a queue lourde (GPD), identiques au script 02
AREP, S0 = 1.3, 0.6        # sous-declaration : q(s) croit avec la taille
N = 2_000_000


def q_report(s):
    """Probabilite qu'un incident de severite s soit declare."""
    return 1.0 / (1.0 + np.exp(-AREP * (np.log(s + 1e-12) - np.log(S0))))


S = stats.genpareto.rvs(c=XI, scale=SCALE, size=N, random_state=rng)
reported = rng.random(N) < q_report(S)

print("=" * 74)
print("RESULTAT 1 : le biais de p_j est EXACTEMENT le taux de declaration a la barre")
print("=" * 74)
print(f"{'u':>6}{'q(u)':>9}{'E[q|S>=u]':>12}{'p_naif/p_vrai':>16}"
      f"{'K vrai':>10}{'K naif':>10}{'biais K':>10}")

barres = [0.5, 1.0, 3.0, 8.0, 15.0, 40.0]
rows = []
for u in barres:
    m = S >= u
    p_true = m.mean()
    p_naive = (m & reported).mean()
    Eq = q_report(S[m]).mean()
    K_true = stats.norm.ppf(1 - p_true)
    K_naive = stats.norm.ppf(1 - p_naive)
    rows.append((u, q_report(np.array([u]))[0], Eq, p_naive / p_true,
                 K_true, K_naive, K_naive - K_true))
    print(f"{u:>6.1f}{rows[-1][1]:>9.2f}{Eq:>12.4f}{p_naive/p_true:>16.4f}"
          f"{K_true:>+10.3f}{K_naive:>+10.3f}{K_naive-K_true:>+10.3f}")

rows = np.array(rows)
print("\n  Le rapport p_naif/p_vrai colle E[q(S)|S>=u] a la 4e decimale.")
print("  Biais de K toujours POSITIF => seuil trop haut => sous-provisionnement.")
print(f"  Il tombe de {rows[0,6]:+.3f} (barre u={rows[0,0]:g}, q={rows[0,1]:.2f}) a "
      f"{rows[-1,6]:+.3f} (barre u={rows[-1,0]:g}, q={rows[-1,1]:.2f}).")
print("  C'est LA le levier : monter la barre, pas raffiner l'estimateur de queue.\n")

# ---------------------------------------------------------------- resultat 3 : l'EVT ne debiaise pas
print("=" * 74)
print("RESULTAT 3 : l'extrapolation EVT du TAUX n'est pas pilotable (resultat negatif)")
print("=" * 74)
u_ref = 3.0                # barre volontairement trop basse (q(u)=0,89) : c'est le cas ou
                           # l'on aimerait qu'une correction EVT vienne au secours
p_true_ref = (S >= u_ref).mean()
K_true_ref = stats.norm.ppf(1 - p_true_ref)
K_naive_ref = stats.norm.ppf(1 - ((S >= u_ref) & reported).mean())
print(f"  barre u = {u_ref}   K vrai = {K_true_ref:+.4f}   K naif = {K_naive_ref:+.4f} "
      f"(biais {K_naive_ref-K_true_ref:+.4f})\n")
print(f"{'seuil haut v':>14}{'E[q|S>v]':>11}{'erreur ratio':>14}"
       f"{'K_EVT':>10}{'biais K':>10}")
evt = []
for v in [8, 15, 25, 40, 60, 100]:
    exc = S[reported & (S > v)] - v
    if exc.size < 50:
        continue
    xi_h, _, beta_v = stats.genpareto.fit(exc, floc=0)
    lam_v = ((S > v) & reported).sum() / N          # taux par observation
    beta_u = beta_v - xi_h * (v - u_ref)
    if beta_u <= 0:
        continue
    lam_u = lam_v * (1 + xi_h * (v - u_ref) / beta_u) ** (1 / xi_h)
    if not (0 < lam_u < 1):
        continue
    plancher = q_report(S[S > v]).mean()            # E[q(S) | S > v]
    err_ratio = (lam_u / p_true_ref) / plancher     # part NON expliquee par le plancher
    K_e = stats.norm.ppf(1 - lam_u)
    evt.append((v, plancher, err_ratio, K_e, K_e - K_true_ref))
    print(f"{v:>14.0f}{plancher:>11.4f}{err_ratio:>14.4f}"
          f"{K_e:>+10.4f}{K_e-K_true_ref:>+10.4f}")
evt = np.array(evt)
biais_naif = K_naive_ref - K_true_ref
print(f"\n  comptage naif : plancher E[q|S>=u] = {q_report(S[S >= u_ref]).mean():.4f}, "
      f"biais K = {biais_naif:+.4f}")
print("\n  Le PLANCHER tend vers 1 quand v monte : c'est le seul gain reel, et il est")
print("  identique a celui du RESULTAT 2 (monter la barre). L'ERREUR DE RATIO, elle,")
print(f"  s'ecarte de 1 sans regle : elle vaut {evt[:,2].min():.2f} a {evt[:,2].max():.2f} selon v.")
print(f"  Biais resultant sur K : de {evt[:,4].min():+.3f} a {evt[:,4].max():+.3f}, contre "
      f"{biais_naif:+.3f} pour le comptage naif.")
print("  Il existe un v optimal, mais on ne peut PAS le localiser sans connaitre la")
print("  verite. L'extrapolation EVT n'ajoute que du bruit a un levier deja disponible.\n")

# ---------------------------------------------------------------- ce que l'EVT fait vraiment
print("=" * 74)
print("Ce que l'EVT fait vraiment : l'indice de queue, invariant a la troncature")
print("=" * 74)
for v in [2, 4, 8]:
    xi_full = stats.genpareto.fit(S[S > v] - v, floc=0)[0]
    xi_obs = stats.genpareto.fit(S[reported & (S > v)] - v, floc=0)[0]
    print(f"  seuil {v:>2} : xi sur donnees completes = {xi_full:.3f} | "
          f"sur donnees declarees = {xi_obs:.3f}   (vrai {XI})")
print("  => la queue (donc le CAPITAL) est robuste ; le SEUIL, lui, se joue ailleurs.\n")

# ---------------------------------------------------------------- figure L
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2, MUTED, GRID = "#1b1e30", "#223e55", "#595959", "#f2f2f2"
ACCENT = "#a6002e"
BL = ["#7baafd", "#4c79c7", "#204993"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0),
                               gridspec_kw={"width_ratios": [1.05, 1]})

# --- panneau 1 : le biais de K_j s'evanouit quand la barre monte
u_g, q_u, biais = rows[:, 0], rows[:, 1], rows[:, 6]
ax1.plot(u_g, biais, "-o", color=ACCENT, lw=2.2, ms=6, zorder=3, label="biais de $K_j$")
ax1.axhline(0, color=INK, lw=1.0)
ax1.fill_between(u_g, 0, biais, color=ACCENT, alpha=0.10)
ax1.set_xscale("log")
ax1.set_xticks(u_g); ax1.set_xticklabels([f"{x:g}" for x in u_g])
ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax1.tick_params(axis="x", which="minor", bottom=False)
ax1.set_xlabel("barre de materialite $u_j$  (severite)", color=INK2)
ax1.set_ylabel("biais de $K_j$  (estime $-$ vrai)", color=ACCENT)
ax1.tick_params(axis="y", colors=ACCENT)
ax1.set_ylim(-0.03, 0.47)
ax1.grid(True, color=GRID, lw=0.7)
axb = ax1.twinx()
axb.plot(u_g, q_u, "-s", color=BL[2], lw=2, ms=5, zorder=3, label="taux de declaration $q(u)$")
axb.set_ylabel("taux de declaration a la barre $q(u_j)$", color=BL[2])
axb.tick_params(axis="y", colors=BL[2])
axb.set_ylim(0, 1.05)
ax1.annotate("barre trop basse :\nsous-provisionnement", xy=(0.5, 0.414),
             xytext=(0.75, 0.35), fontsize=8.5, color=ACCENT,
             arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))
ax1.annotate("declaration quasi complete :\n$K_j$ non biaise", xy=(12, 0.005),
             xytext=(2.6, 0.10), fontsize=8.5, color=BL[2],
             arrowprops=dict(arrowstyle="->", color=BL[2], lw=1))
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = axb.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper right")
ax1.set_title("(a)  Le seuil n'est pas biaise par l'EVT, mais par la barre",
              fontsize=11, color=INK, pad=8)

# --- panneau 2 : l'extrapolation EVT n'est pas pilotable
xpos = np.arange(len(evt) + 1)
vals = np.concatenate([[biais_naif], evt[:, 4]])
labs = ["comptage\nnaif"] + [f"EVT\nv={int(v)}" for v in evt[:, 0]]
cols = [MUTED] + [BL[1] if abs(b) < abs(biais_naif) else ACCENT for b in evt[:, 4]]
ax2.bar(xpos, vals, color=cols, edgecolor="#fcfcfb", width=0.68)
ax2.axhline(0, color=INK, lw=1.1)
ax2.axhline(biais_naif, color=MUTED, lw=1, ls="--")
for x, v in zip(xpos, vals):
    off = 0.012 * np.sign(v if v != 0 else 1)
    ax2.text(x, v + off, f"{v:+.3f}", ha="center",
             va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK2)
ax2.set_xticks(xpos); ax2.set_xticklabels(labs, fontsize=8)
ax2.set_ylabel("biais de $K_j$", color=INK2)
pad = 0.10 * max(abs(vals).max(), 1e-3)
ax2.set_ylim(min(vals.min(), 0) - 4 * pad, max(vals.max(), 0) + 4 * pad)
ax2.yaxis.grid(True, color=GRID, lw=0.8)
ax2.set_title("(b)  L'extrapolation EVT n'est pas pilotable",
              fontsize=11, color=INK, pad=8)
ax2.text(0.5, 0.03, "meilleur qu'a $v=15$, pire qu'a $v=100$ :\nl'optimum existe mais n'est pas localisable",
         transform=ax2.transAxes, ha="center", fontsize=8.5, color=MUTED, style="italic")

fig.suptitle("L : $K_j$ s'obtient toujours en inversant une repartition ; ce qui compte, c'est ou l'on place la barre",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])

outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "L_seuil_Kj.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("figure ecrite :", path)
