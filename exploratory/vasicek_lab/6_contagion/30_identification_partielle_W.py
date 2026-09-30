#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30 : IDENTIFICATION PARTIELLE de la contagion dirigee -> BORNES de capital.

Le modele en couches :

  SOCLE IDENTIFIE (calibre, defendable) : frequence NegBin, severite GPD euro,
    accumulation. C'est le SCR \"nu\", obtenu a W = 0. Il ne depend d'aucun parametre
    non identifiable.

  COUCHE CONTAGION (partiellement identifiee) : la matrice dirigee W. On ne l'estime
    pas, on caracterise son ensemble admissible et on lit le capital en BORNES.

Le dispositif (decomposition W = S + A, ensemble admissible, evaluateur a nombres
communs) vit dans le module partage `partial_id` : voir sa docstring pour la construction
et pour la mise en garde de lecture sur la VaR.

Trois livrables :
  1. les bornes de capital, et leur retrecissement quand l'ignorance directionnelle t
     diminue, ce qui chiffre la VALEUR DE L'INFORMATION d'un registre DORA ;
  2. une priorite de remediation ROBUSTE (minimax regret sur l'ensemble) ;
  3. le test de la hierarchie de remediation sur tout l'ensemble.

Separation avec le 25 : le 25 borne l'AMPLITUDE de la contagion (le gain g). Ici g est
FIXE et on borne la DIRECTION. Les deux axes de non-identifiabilite ne sont jamais
melanges. Pour la comparaison des deux regimes (poser W contre le borner), voir le 31.

Sortie : diagnostics + figure Z_identification_partielle.png.
"""

import os
import sys
from itertools import combinations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import partial_id as pid                                        # noqa: E402

WID = 78
T_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
N_PRIOR = 250
N_REMED = 220
TAU = 0.5                 # dispersion du prior d'expert, en fraction de la co-occurrence
SEED = 20260721


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


ev = pid.Evaluator(n_years=40_000, seed=SEED)
P = pid.expert_matrix()
S, A_EXP = pid.decompose(P)
print(f"tirage commun : {ev.n_years} annees, {ev.T} incidents, lambda={ev.lam:.2f}/an, "
      f"g={pid.G_DEFAULT}")

# =====================================================================================
titre("Socle identifie, point d'expert, et ce que la contagion ajoute")
# =====================================================================================
SCR_SOCLE, MEAN_SOCLE = ev(np.zeros((pid.NP_, pid.NP_)))
SCR_EXPERT, MEAN_EXPERT = ev(P)
SCR_SYM, MEAN_SYM = ev(S)
print(f"  SOCLE (W = 0, aucune contagion)          : {SCR_SOCLE:9.0f} M")
print(f"  W symetrique (A = 0, direction nulle)    : {SCR_SYM:9.0f} M"
      f"   (+{100*(SCR_SYM/SCR_SOCLE-1):.1f} % / socle)")
print(f"  W d'expert (classeur qualitatif actuel)  : {SCR_EXPERT:9.0f} M"
      f"   (+{100*(SCR_EXPERT/SCR_SOCLE-1):.1f} % / socle)")
print(f"  rho(W) expert = {pid.rho(P):.3f}   (Leontief : doit rester < 1)")
print("  Le socle ne depend d'AUCUN parametre non identifiable : c'est le plancher")
print("  defendable. Tout ce qui est au-dessus est de la contagion, donc a borner.")

# =====================================================================================
titre("Bornes de capital sur l'ensemble admissible, en fonction de l'ignorance t")
# =====================================================================================
# LA DIMENSION DU PAVE, IMPRIMEE PLUTOT QUE SOUS-ENTENDUE. Le memoire ecrit « les 2^10 = 1 024
# sommets », et seul le 1 024 etait trace : l'exposant, qui est le nombre d'entrees hors
# diagonale libres, ne l'etait pas. Un exposant est un vrai nombre quand il compte quelque
# chose, et celui-ci compte les degres de liberte de la direction. Il vaut donc mieux
# l'imprimer que de laisser le lecteur le reconstituer.
_n_pil = len(pid.PILIERS) if hasattr(pid, "PILIERS") else 5
_n_libres = _n_pil * (_n_pil - 1) // 2
print(f"  degres de liberte de la direction : {_n_libres} paires hors diagonale, "
      f"donc 2^{_n_libres} = {2**_n_libres} sommets")
print("  (une paire non ordonnee par couple de piliers distincts ; chaque sommet fixe le sens")
print("  de chacune, ce qui rend l'enumeration exhaustive et le resultat deterministe.)\n")

bounds, means_b = {}, {}
for t in T_GRID:
    if t == 0.0:
        bounds[t], means_b[t] = (SCR_SYM, SCR_SYM, 1), (MEAN_SYM, MEAN_SYM)
        continue
    vals, mns, keep = [], [], 0
    for a_vec in pid.all_vertices(S, t):        # enumeration EXHAUSTIVE des 1024 sommets
        W = pid.build_W(S, a_vec)
        if not pid.admissible(W):
            continue
        keep += 1
        v, m = ev(W)
        vals.append(v)
        mns.append(m)
    vals, mns = np.array(vals), np.array(mns)
    bounds[t] = (float(vals.min()), float(vals.max()), keep)
    means_b[t] = (float(mns.min()), float(mns.max()))
    print(f"  t = {t:.2f} : SCR dans [{vals.min():8.0f} ; {vals.max():8.0f}] M"
          f"   largeur {vals.max()-vals.min():7.0f} M"
          f"   ({100*(vals.max()-vals.min())/SCR_SOCLE:5.1f} % du socle)"
          f"   [{keep}/1024 sommets admissibles]")

lo1, hi1, _ = bounds[1.0]
mlo1, mhi1 = means_b[1.0]
print(f"\n  Etat actuel des donnees (t = 1, ignorance totale de la direction) :")
print(f"    capital BORNE dans [{lo1:.0f} ; {hi1:.0f}] M, facteur {hi1/lo1:.2f} entre les bouts.")
print(f"    La contagion ajoute entre +{100*(lo1/SCR_SOCLE-1):.0f} % et "
      f"+{100*(hi1/SCR_SOCLE-1):.0f} % au socle ({SCR_SOCLE:.0f} M).")
print(f"    Le point d'expert ({SCR_EXPERT:.0f} M) est UN point de cet ensemble, pas sa mesure.")
print(f"\n  MISE EN GARDE DE LECTURE : a xi eleve, la queue annuelle obeit au principe de la")
print(f"  perte unique dominante, donc la VaR est PEU sensible a la structure de cascade")
print(f"  (le 29 l'avait deja constate entre marche et branchement). Lecture en MOYENNE,")
print(f"  estimateur bien moins bruite :")
print(f"    moyenne annuelle : socle {MEAN_SOCLE:.0f} M, bornes [{mlo1:.0f} ; {mhi1:.0f}] M"
      f"  (facteur {mhi1/mlo1:.2f})")
print("  NE PAS faire porter la demonstration par le seul niveau de VaR : la contribution")
print("  se lit sur la DECISION (priorite de remediation) et sur la moyenne.")

w_full = hi1 - lo1
w_half = bounds[0.5][1] - bounds[0.5][0]
print(f"\n  VALEUR DE L'INFORMATION : un registre contraignant l'asymetrie a la moitie du")
print(f"  niveau de co-occurrence (t = 0,5) reduirait la largeur des bornes de {w_full:.0f} M")
print(f"  a {w_half:.0f} M, soit -{100*(1-w_half/w_full):.0f} %. C'est la specification de")
print("  reporting DORA, chiffree en euros de capital.")

# =====================================================================================
titre("Lecture centrale a l'interieur des bornes : le prior d'expert (couche bayesienne)")
# =====================================================================================
rng_p = np.random.default_rng(SEED + 2)
prior_vals = np.array([ev.scr(pid.build_W(S, a))
                       for a in pid.sample_prior(S, A_EXP, TAU, N_PRIOR, rng_p)
                       if pid.admissible(pid.build_W(S, a))])
q05, q50, q95 = np.percentile(prior_vals, [5, 50, 95])
print(f"  prior centre sur le classeur qualitatif, dispersion tau = {TAU} x co-occurrence")
print(f"  lecture centrale : mediane {q50:.0f} M, intervalle a 90 % [{q05:.0f} ; {q95:.0f}] M")
print(f"  a comparer aux BORNES sans prior [{lo1:.0f} ; {hi1:.0f}] M")
print("  ATTENTION, a ecrire tel quel dans le memoire : ce prior est le classeur qualitatif")
print("  actuel, PAS une elicitation de Cooke. Les seances ne sont pas faites. Les BORNES,")
print("  elles, ne dependent d'aucun prior : c'est ce qui les rend opposables.")

# =====================================================================================
titre("Priorite de remediation ROBUSTE : minimax regret sur l'ensemble admissible")
# =====================================================================================
rng_r = np.random.default_rng(SEED + 3)
Wsets = [W for W in (pid.build_W(S, a)
                     for a in pid.sample_vertices(S, 1.0, N_REMED, rng_r))
         if pid.admissible(W)]
print(f"  {len(Wsets)} matrices admissibles retenues (sommets tires, t = 1)")
benefits = np.array([ev.benefits(W) for W in Wsets])
regret = benefits.max(axis=1)[:, None] - benefits
maxreg = regret.max(axis=0)
p95reg = np.percentile(regret, 95, axis=0)
freq_top = np.bincount(benefits.argmax(axis=1), minlength=pid.NP_) / len(Wsets)

# Le regret MAXIMAL est un extreme sur l'echantillon : statistique instable, on la teste.
rng_b = np.random.default_rng(SEED + 4)
NB = 600
win = np.zeros(pid.NP_)
for _ in range(NB):
    sel = rng_b.integers(0, len(Wsets), len(Wsets))
    win[int(np.argmin(regret[sel].max(axis=0)))] += 1
win /= NB

print(f"\n  {'pilier':<8}{'gain moyen':>13}{'regret max':>13}{'regret p95':>13}"
      f"{'% fois 1er':>12}{'% minimax':>11}")
for j in range(pid.NP_):
    print(f"  P{pid.PIL[j]:<7}{benefits[:, j].mean():>12.0f} M{maxreg[j]:>12.0f} M"
          f"{p95reg[j]:>12.0f} M{100*freq_top[j]:>11.1f} %{100*win[j]:>10.1f} %")
print("  (gain min nul pour tous : a t = 1, certains sommets annulent deja la ligne du")
print("   pilier, le remedier n'y change alors rien. Ce n'est pas une anomalie.)")

order = np.argsort(maxreg)
j1, j2 = int(order[0]), int(order[1])
gap = 100.0 * (maxreg[j2] - maxreg[j1]) / maxreg[j1]
print(f"\n  Action minimax regret : P{pid.PIL[j1]} (regret max {maxreg[j1]:.0f} M), devant "
      f"P{pid.PIL[j2]} ({maxreg[j2]:.0f} M), soit {gap:.1f} % d'ecart.")
print("  Stabilite du choix (bootstrap) : "
      + ", ".join(f"P{pid.PIL[j]} {100*win[j]:.0f} %" for j in range(pid.NP_) if win[j] > 0.005))
exp_top = int(np.argmax(ev.benefits(P)))
print(f"  Choix sous le seul W d'expert : P{pid.PIL[exp_top]}.")
if gap < 5.0 or win[j1] < 0.80:
    print(f"  VERDICT HONNETE : P{pid.PIL[j1]} et P{pid.PIL[j2]} sont a egalite pratique.")
    print("  L'ignorance directionnelle ne permet PAS de les departager, et il ne faut pas")
    print("  pretendre le contraire. Ce qui est solide : ils devancent nettement les autres.")
else:
    print(f"  Le choix P{pid.PIL[j1]} est stable sur l'ensemble admissible.")

# =====================================================================================
titre("Hierarchie de remediation : tient-elle sur tout l'ensemble admissible ?")
# =====================================================================================
print("  ATTENTION AU VOCABULAIRE : ce n'est PAS la non-transitivite du script 01. Le 01")
print("  montre que les correlations violent la contrainte de transitivite d'un modele a")
print("  facteur unique. Ici on teste la relation de preference MAJORITAIRE entre piliers,")
print("  par gain de remediation. Un resultat ici ne confirme ni n'infirme celui du 01.")
domin = np.zeros((pid.NP_, pid.NP_))
for j in range(pid.NP_):
    for k in range(pid.NP_):
        if j != k:
            domin[j, k] = float((benefits[:, j] > benefits[:, k]).mean())
maj = domin > 0.5
cycles = [(a, b, c) for a, b, c in combinations(range(pid.NP_), 3)
          for (x, y, z) in [(a, b, c), (a, c, b)] if maj[x, y] and maj[y, z] and maj[z, x]]
print("\n  dominance majoritaire (part des W ou le gain de la ligne depasse celui de la colonne)")
print("          " + "".join(f"   P{pid.PIL[k]}  " for k in range(pid.NP_)))
for j in range(pid.NP_):
    print(f"     P{pid.PIL[j]} " + "".join("   .   " if j == k else f" {domin[j, k]:5.2f} "
                                           for k in range(pid.NP_)))
if cycles:
    a, b, c = cycles[0]
    print(f"\n  CYCLE : P{pid.PIL[a]} > P{pid.PIL[b]} > P{pid.PIL[c]} > P{pid.PIL[a]}.")
    print("  La hierarchie n'est pas un ordre : elle depend du chemin, sur l'ensemble entier.")
else:
    off = domin[~np.eye(pid.NP_, dtype=bool)]
    near = int(((off > 0.45) & (off < 0.55)).sum())
    print("\n  Aucun cycle : la preference majoritaire est un ordre total sur l'ensemble.")
    print(f"  MAIS ne pas en conclure que la hierarchie est invariante : {near} paire(s) sont")
    print("  a moins de 5 points de 0,50, donc a pile ou face. Enonce defendable : la")
    print("  SEPARATION entre le groupe de tete et les autres est robuste, l'ORDRE INTERNE")
    print("  au groupe de tete ne l'est pas. Ecrire \"classement invariant\" serait une survente.")

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
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"

# quatre panneaux : une grille 2x2 tient en portrait la ou une rangee de quatre
# imposerait une page tournee (rapport h/l de 0,89 contre 0,24).
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.4, 4.8),
                                             gridspec_kw={"width_ratios": [1.1, 1]})
ts = np.array(T_GRID)
los = np.array([bounds[t][0] for t in T_GRID])
his = np.array([bounds[t][1] for t in T_GRID])
ax1.fill_between(ts, los, his, color=BLUE, alpha=0.18, label="bornes sur l'ensemble")
ax1.plot(ts, los, color=BLUE, lw=1.8)
ax1.plot(ts, his, color=BLUE, lw=1.8)
ax1.axhline(SCR_SOCLE, color=GREEN, lw=2.0, ls="--", label="socle identifié ($W=0$)")
ax1.scatter([1.0], [SCR_EXPERT], s=60, color=ACCENT, zorder=6, edgecolor="#fff",
            label="point d'expert")
ax1.set_xlabel("$t$ : ignorance de la direction  (1 = donnée actuelle)", color=INK2)
ax1.set_ylabel("SCR (VaR 99,5 %, M€)", color=INK2)
ax1.set_title("(a)  Ce que la donnée manquante coûte", fontsize=11, color=INK, pad=8)
# EN BAS A GAUCHE, ET NON EN HAUT. Rendre la legende opaque a corrige sa lisibilite mais
# l'a rendue MASQUANTE : en haut a gauche elle recouvrait le debut de la bande et le point
# d'expert a t = 0. La zone entre le socle et la borne basse est vide : la legende y va.
ax1.legend(frameon=True, facecolor="#fcfcfb", edgecolor="none", framealpha=0.88,
           fontsize=8, loc="lower left")

ax2.hist(prior_vals, bins=28, color=BLUE, alpha=0.55, edgecolor="white", linewidth=0.5)
ax2.axvspan(lo1, hi1, color=MUTED, alpha=0.13)
for v, c in [(lo1, INK2), (hi1, INK2), (SCR_EXPERT, ACCENT)]:
    ax2.axvline(v, color=c, lw=1.6, ls="--" if c == INK2 else "-")
ax2.set_xlabel("SCR (M€)", color=INK2)
ax2.set_ylabel("tirages du prior", color=INK2)
ax2.set_title("(b)  Lecture centrale dans les bornes", fontsize=11, color=INK, pad=8)
ax2.text(0.06, 0.96, f"bornes  [{lo1:.0f} ; {hi1:.0f}]\nprior 90 % [{q05:.0f} ; {q95:.0f}]",
         transform=ax2.transAxes, fontsize=8.2, color=INK2, va="top")

xs = np.arange(pid.NP_)
ax3.bar(xs - 0.2, benefits.mean(axis=0), width=0.38, color=BLUE, alpha=0.8, label="gain moyen")
ax3.bar(xs + 0.2, maxreg, width=0.38, color=ACCENT, alpha=0.8, label="regret maximal")
ax3.scatter([j1], [maxreg[j1]], s=90, marker="v", color=INK, zorder=6)
ax3.set_xticks(xs)
ax3.set_xticklabels([f"P{p}" for p in pid.PIL])
ax3.set_ylabel("M€ de SCR", color=INK2)
ax3.set_title(f"(c)  Priorité robuste : P{pid.PIL[j1]} (minimax)", fontsize=11, color=INK, pad=8)
ax3.set_ylim(0, 1.32 * max(maxreg.max(), benefits.mean(axis=0).max()))
ax3.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

im = ax4.imshow(np.where(np.eye(pid.NP_) == 1, np.nan, domin), cmap="RdBu_r", vmin=0, vmax=1)
for j in range(pid.NP_):
    for k in range(pid.NP_):
        if j != k:
            ax4.text(k, j, f"{domin[j, k]:.2f}".replace(".", ","), ha="center", va="center", fontsize=6.8,
                     color=INK if 0.25 < domin[j, k] < 0.75 else "#fff")
ax4.set_xticks(xs); ax4.set_xticklabels([f"P{p}" for p in pid.PIL])
ax4.set_yticks(xs); ax4.set_yticklabels([f"P{p}" for p in pid.PIL])
ax4.set_title("(d)  " + ("Hiérarchie cyclique" if cycles else "Hiérarchie sans cycle"),
              fontsize=11, color=INK, pad=8)
fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.03).outline.set_edgecolor(GRID)
for ax in (ax1, ax2, ax3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# RESERVE EN POUCES, PAS EN FRACTION. Un rect a 0,90 reserve 10 % de la HAUTEUR au
# titre : correct sur une figure large de 5 pouces de haut, deux fois trop sur une
# figure empilee de 10 pouces, ou cela creait un bandeau blanc sous le titre. On
# reserve donc une hauteur FIXE de 0,42 pouce, quelle que soit la taille de la figure.
_top = 1.0 - 0.26 / fig.get_figheight()
fig.tight_layout(h_pad=1.6)
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z_identification_partielle.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
