#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
56 : reformulation BAYESIENNE de la direction non identifiee.

Etat actuel du memoire : la direction A de W = S + A n'est pas identifiee, donc on BORNE
(chapitre 10) et on lit le capital en intervalle. C'est honnete mais fruste : toutes les
configurations de l'ensemble admissible sont traitees a egalite, y compris celles que les
rapports post-incident contredisent. Le cadre bayesien fait mieux, avec le MEME materiau.

LE MODELE. Pour chaque paire de piliers {lo, hi}, on pose
        theta ~ P(une chaine de defaillances documentee dans cette paire aille de lo vers hi)
et on relie theta au coefficient antisymetrique par
        a = S * (2 theta - 1),
qui envoie theta dans [0,1] exactement sur l'intervalle admissible [-S, +S] du chapitre 10 :
theta = 1/2 donne a = 0 (aucune direction, W symetrique), theta = 1 donne la direction pleine.

  PRIOR      : theta ~ Beta(alpha0, beta0). Par defaut Beta(1,1), UNIFORME : aucune information
               d'expert n'est injectee, ce qui rend l'exercice independant de toute elicitation.
  DONNEES    : le corpus de post-mortems du script 53 fournit, par paire, le nombre de chaines
               documentees dans chaque sens (n_lo, n_hi).
  POSTERIEUR : Beta(alpha0 + n_lo, beta0 + n_hi). Conjugaison exacte, aucun MCMC necessaire.

CE QUE CELA APPORTE, ET POURQUOI C'EST PLUS QUE COSMETIQUE.
  1. Les paires DOCUMENTEES se concentrent, les paires SANS DONNEE restent uniformes sur [0,1]
     donc a couvre tout [-S, S] : la structure d'identification du memoire est reproduite
     AUTOMATIQUEMENT, sans etre posee a la main.
  2. Le SCR n'est plus un intervalle de bornes mais une LOI A POSTERIORI : on peut en donner la
     mediane et un intervalle de credibilite, ce qui est le langage d'un modele interne.
  3. La reponse a la remarque de C. Hillairet devient propre : le quantile d'un objet incertain
     est le quantile de la loi PREDICTIVE a posteriori.

SENSIBILITE AU PRIOR (obligatoire pour un travail bayesien defendable). On refait tout sous
trois priors : uniforme Beta(1,1), Jeffreys Beta(1/2,1/2), et un prior SCEPTIQUE Beta(4,4) qui
tire vers theta = 1/2, c'est-a-dire vers l'absence de direction. Si la conclusion resiste au
prior sceptique, elle ne vient pas du prior.

Sortie : diagnostics + figure Z19_bayes_direction.png.
"""

import os
import sys
from itertools import combinations

import numpy as np
from scipy.stats import beta as beta_dist
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import partial_id as pid                                        # noqa: E402
import resultats_partages as rp                                 # noqa: E402

WID = 84
SEED = 20260721            # MEME graine que les scripts 30 et 55 : bandes comparables
N_POST = 400               # tirages a posteriori par prior
PIL = pid.PIL
PAIRS = list(combinations(range(len(PIL)), 2))
PAIR_LAB = [f"P{PIL[a]}-P{PIL[b]}" for a, b in PAIRS]

# --- donnees : comptes de chaines documentees par paire (script 53) ---------------------------
# (n_lo, n_hi) = nombre de chaines allant du pilier de PLUS PETIT numero vers l'autre, et inverse.
# CALCULES depuis les transitions brutes du corpus (`resultats_partages`), donc impossibles a
# desynchroniser du script 53. Les paires absentes du corpus ressortent a (0, 0) : aucune
# information, et le posterieur y reste le prior.
COUNTS = rp.postmortem_counts()
PRIORS = [("uniforme Beta(1,1)", 1.0, 1.0),
          ("Jeffreys Beta(1/2,1/2)", 0.5, 0.5),
          ("sceptique Beta(4,4)", 4.0, 4.0)]


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


ev = pid.Evaluator(n_years=40_000, seed=SEED)
P = pid.expert_matrix()
S, A_EXP = pid.decompose(P)
smax = S[pid.IU]
SCR_SOCLE, MEAN_SOCLE = ev(np.zeros((pid.NP_, pid.NP_)))
n_lo = np.array([COUNTS[l][0] for l in PAIR_LAB], dtype=float)
n_hi = np.array([COUNTS[l][1] for l in PAIR_LAB], dtype=float)

# =====================================================================================
titre("1. Le posterieur par paire : ce que la donnee concentre, ce qu'elle laisse libre")
# =====================================================================================
a0, b0 = 1.0, 1.0
print(f"  Prior de reference : Beta({a0:.0f},{b0:.0f}) uniforme (aucune elicitation).")
print(f"  {'paire':<8}{'n_lo':>6}{'n_hi':>6}{'E[theta]':>11}{'IC90 de theta':>20}"
      f"{'sens implique':>16}")
for m, lab in enumerate(PAIR_LAB):
    A_, B_ = a0 + n_lo[m], b0 + n_hi[m]
    e = A_ / (A_ + B_)
    q5, q95 = beta_dist.ppf([0.05, 0.95], A_, B_)
    lo_p, hi_p = lab.split("-")
    if q5 > 0.5:
        sens = f"{lo_p} -> {hi_p}"
    elif q95 < 0.5:
        sens = f"{hi_p} -> {lo_p}"
    else:
        sens = "indetermine"
    print(f"  {lab:<8}{n_lo[m]:>6.0f}{n_hi[m]:>6.0f}{e:>11.2f}"
          f"   [{q5:.2f} ; {q95:.2f}]{sens:>16}")
n_tranche = sum(1 for m in range(pid.NFREE)
                if (beta_dist.ppf(0.05, a0 + n_lo[m], b0 + n_hi[m]) > 0.5
                    or beta_dist.ppf(0.95, a0 + n_lo[m], b0 + n_hi[m]) < 0.5))
n_doc = int(((n_lo + n_hi) > 0).sum())
print("\n  Les 4 paires sans donnee gardent E[theta] = 0,50 et un IC90 quasi complet : le")
print("  posterieur reproduit de lui-meme l'ignorance totale, sans qu'on la pose.")
print(f"\n  A NOTER, ET C'EST PLUS SEVERE QUE LE SCRIPT 53 : sur les {n_doc} paires documentees,")
print(f"  seules {n_tranche} ont un IC90 qui EXCLUT 0,50, donc un sens veritablement tranche.")
print("  Avec 1 a 3 chaines observees, l'esperance penche nettement mais la credibilite a 90 %")
print("  ne suffit pas a exclure l'absence de direction. Le test de signe du script 53 (p tres")
print("  faible) repondait a une AUTRE question : la coherence GLOBALE des 17 transitions entre")
print("  elles. Le posterieur, lui, juge CHAQUE paire separement, et se montre plus prudent.")
print("  Les deux lectures sont justes ; la bayesienne est la plus conservatrice, donc la plus")
print("  defendable devant un jury.")

# =====================================================================================
titre("2. Loi a posteriori du SCR, sous les trois priors")
# =====================================================================================
print("  On tire theta a posteriori, on en deduit A = S(2 theta - 1), donc W, puis le SCR.")
print("  Rappel du module : a queue lourde la VaR est peu sensible a la structure, la MOYENNE")
print("  la revele mieux ; on donne donc les deux.\n")
print(f"  {'prior':<24}{'SCR median':>12}{'IC90 credible':>22}{'largeur':>9}"
      f"{'moyenne med.':>14}")
res = {}
for name, pa, pb in PRIORS:
    rng = np.random.default_rng(SEED)
    A_, B_ = pa + n_lo, pb + n_hi
    scrs, means, kept = [], [], 0
    for _ in range(N_POST):
        th = rng.beta(A_, B_)
        a_vec = smax * (2.0 * th - 1.0)
        W = pid.build_W(S, a_vec)
        if not pid.admissible(W):
            continue
        kept += 1
        v, mn = ev(W)
        scrs.append(v); means.append(mn)
    scrs, means = np.array(scrs), np.array(means)
    q5, q50, q95 = np.percentile(scrs, [5, 50, 95])
    res[name] = dict(scrs=scrs, means=means, q5=q5, q50=q50, q95=q95, kept=kept)
    print(f"  {name:<24}{q50:>10.0f} M   [{q5:.0f} ; {q95:.0f}]{q95-q5:>8.0f} M"
          f"{np.median(means):>12.0f} M")

# comparaison aux bornes frequentistes du chapitre 10
LO_BOUNDS, HI_BOUNDS = rp.BORNES_CONTAGION     # chapitre 10 / script 30 (source unique)
W_BOUNDS = HI_BOUNDS - LO_BOUNDS
ref = res["uniforme Beta(1,1)"]
print(f"\n  Bornes du chapitre 10 (ignorance totale, sommets) : [{LO_BOUNDS:.0f} ; {HI_BOUNDS:.0f}] M,")
print(f"  largeur {W_BOUNDS:.0f} M. Intervalle credible a 90 % sous prior uniforme :")
print(f"  [{ref['q5']:.0f} ; {ref['q95']:.0f}] M, largeur {ref['q95']-ref['q5']:.0f} M, soit"
      f" {100*(1-(ref['q95']-ref['q5'])/W_BOUNDS):.0f} % de moins.")
print(f"  Ce n'est PAS une contradiction : les bornes couvrent le PIRE cas admissible, le")
print(f"  credible couvre 90 % de la masse a posteriori. Les deux repondent a deux questions")
print(f"  differentes, et le memoire peut donner les deux : bornes opposables, credible informatif.")

# =====================================================================================
titre("3. Sensibilite au prior : la conclusion vient-elle du prior ?")
# =====================================================================================
u = res["uniforme Beta(1,1)"]; j = res["Jeffreys Beta(1/2,1/2)"]; s = res["sceptique Beta(4,4)"]
print(f"  mediane : uniforme {u['q50']:.0f} | Jeffreys {j['q50']:.0f} | sceptique {s['q50']:.0f} M")
spread = max(u['q50'], j['q50'], s['q50']) - min(u['q50'], j['q50'], s['q50'])
print(f"  ecart maximal entre priors : {spread:.0f} M, soit {100*spread/u['q50']:.1f} % de la mediane")
print(f"  et {100*spread/W_BOUNDS:.0f} % de la largeur des bornes du chapitre 10.")
print(f"\n  Le prior SCEPTIQUE Beta(4,4) tire vers theta = 1/2, donc vers l'ABSENCE de direction :")
print(f"  c'est le prior le plus defavorable a la these du memoire. Sa mediane reste a")
print(f"  {s['q50']:.0f} M contre {u['q50']:.0f} M sous prior uniforme. La conclusion ne vient donc pas")
print(f"  du prior : elle vient des comptes documentes, qui dominent un prior meme sceptique")
print(f"  des que quelques chaines sont observees.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. La direction non identifiee se traite proprement en bayesien : theta ~ Beta,")
print("     conjugaison exacte sur les comptes de chaines documentees, aucun MCMC, aucun prior")
print("     d'expert. W passe de « borne a la main » a « estime sous prior explicite ».")
print("  2. Le posterieur REPRODUIT SEUL la structure d'identification : paires documentees")
print("     concentrees, paires sans donnee uniformes sur tout l'intervalle admissible.")
print(f"  3. Le SCR devient une loi : mediane {u['q50']:.0f} M, credible 90 % "
      f"[{u['q5']:.0f} ; {u['q95']:.0f}] M,")
print(f"     nettement plus informatif que les bornes de pire cas [{LO_BOUNDS:.0f} ; {HI_BOUNDS:.0f}].")
print(f"  4. La conclusion resiste a un prior SCEPTIQUE (ecart de mediane {spread:.0f} M, "
      f"{100*spread/u['q50']:.1f} %) :")
print("     elle est portee par la donnee documentaire, pas par le choix du prior.")
print(f"  5. NUANCE IMPORTANTE : par paire, seules {n_tranche} des {n_doc} paires documentees ont un")
print("     sens tranche a 90 % de credibilite. Le corpus etablit une COHERENCE d'ensemble")
print("     (script 53) bien plus qu'une direction paire par paire. A dire dans ces termes.")
print("  6. Limite heritee : la vraisemblance repose sur le codage des post-mortems, donc sur")
print("     ses reserves propres (codeur unique, biais de narration des enquetes). Le bayesien")
print("     formalise l'incertitude ; il ne cree pas d'information.")

# =====================================================================================
# figure Z19
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

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Trois panneaux traces
# sur seize pouces puis imprimes sur 7,27 donnent 1,6 pouce par panneau et
# des etiquettes sous 5 points. Le rapport largeur sur hauteur est conserve.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.69, 3.9))

# (a) posterieurs par paire
xg = np.linspace(0, 1, 300)
for m, lab in enumerate(PAIR_LAB):
    A_, B_ = a0 + n_lo[m], b0 + n_hi[m]
    dens = beta_dist.pdf(xg, A_, B_)
    documented = (n_lo[m] + n_hi[m]) > 0
    ax1.plot(xg, dens, lw=2.0 if documented else 1.2,
             color=BLUE if documented else MUTED,
             alpha=0.9 if documented else 0.5)
ax1.axvline(0.5, color=ACCENT, ls="--", lw=1.3)
from matplotlib.lines import Line2D
# la cle des couleurs quitte le titre, trop large pour sa colonne, et passe en legende
ax1.set_ylim(top=ax1.get_ylim()[1] * 1.45)
ax1.legend(handles=[Line2D([0], [0], color=BLUE, lw=2.0, label="paire documentée"),
                    Line2D([0], [0], color=MUTED, lw=1.2, alpha=0.6, label="paire sans donnée"),
                    Line2D([0], [0], color=ACCENT, lw=1.3, ls="--",
                           label="θ = 1/2 : pas de direction")],
           loc="upper center", fontsize=8, frameon=True, framealpha=0.92,
           edgecolor="none", facecolor="#fcfcfb", borderpad=0.3)
ax1.set_xlabel("directionnalité $\\theta$ de la paire", color=INK2)
ax1.set_ylabel("densité a posteriori", color=INK2)
ax1.set_title("(a)  Loi a posteriori de θ\npar paire de piliers",
              fontsize=11, color=INK, pad=8)

# (b) loi a posteriori du SCR vs bornes du chapitre 10
cnt, _, _ = ax2.hist(u["scrs"], bins=34, color=BLUE, alpha=0.55, edgecolor="#fcfcfb")
ax2.axvspan(LO_BOUNDS, HI_BOUNDS, color=MUTED, alpha=0.16)
ax2.text(0.03, 0.97, "bornes du\nchapitre 10", transform=ax2.transAxes, ha="left", va="top",
         fontsize=8, color=INK2)
ax2.axvline(u["q50"], color=ACCENT, lw=2, label=f"médiane {u['q50']:.0f} M€")
for q in (u["q5"], u["q95"]):
    ax2.axvline(q, color=ACCENT, ls="--", lw=1.2)
ax2.set_xlabel("SCR (M€)", color=INK2)
ax2.set_ylabel("fréquence a posteriori", color=INK2)
# plus de legende : elle recouvrait le texte des bornes. Marge haute pour les etiquettes.
ax2.set_ylim(0, cnt.max() * 1.6)
ax2.text(0.97, 0.97, f"médiane {u['q50']:.0f} M€\nprior uniforme", transform=ax2.transAxes,
         ha="right", va="top", fontsize=8, color=ACCENT,
         bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none", alpha=0.9))
ax2.set_title("(b)  Le SCR devient une loi,\nplus étroite que les bornes", fontsize=11,
              color=INK, pad=8)

# (c) sensibilite au prior
names = [p[0] for p in PRIORS]
cols = [BLUE, GREEN, ACCENT]
for i, (nm, c) in enumerate(zip(names, cols)):
    r = res[nm]
    ax3.plot([i, i], [r["q5"], r["q95"]], color=c, lw=11, solid_capstyle="butt", alpha=0.8)
    ax3.plot([i], [r["q50"]], "o", color=INK, ms=6, zorder=5)
    ax3.text(i, r["q95"] + 25, f"{r['q50']:.0f}", ha="center", fontsize=9, color=INK2)
ax3.set_xticks(range(3))
ax3.set_xticklabels(["uniforme\nBeta(1,1)", "Jeffreys\nBeta(½,½)", "sceptique\nBeta(4,4)"],
                    fontsize=8.5)
ax3.set_ylabel("SCR (M€) : médiane, crédible 90 %", color=INK2, fontsize=9)
ax3.set_title(f"(c)  Robuste au prior :\n{100*spread/u['q50']:.1f} % d'écart de médiane",
              fontsize=11, color=INK, pad=8)

for ax in (ax1, ax2, ax3):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.88])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z19_bayes_direction.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
