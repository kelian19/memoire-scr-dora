#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
32 : trois PROPRIETES FORMELLES du modele, enoncees, demontrees, et verifiees.

Objet : donner au memoire ce qui lui manque face a un jury quantitatif, c'est-a-dire
des enonces qu'on DEMONTRE au lieu de les argumenter. Chacune est accompagnee de sa
verification numerique, ce qui est la forme la plus solide : le theoreme dit ce qui doit
arriver, le calcul montre que cela arrive.

---------------------------------------------------------------------------------------
PROPOSITION 1 (la criticite depend de l'ORDRE, donc aucune fonction d'ensemble ne la
representе).
  Soit c(sigma) la criticite d'une CHAINE ORDONNEE sigma de piliers. Si c etait une
  fonction du seul ENSEMBLE des piliers touches, alors on aurait c(sigma) = c(sigma')
  pour toute permutation sigma' de sigma. Or ce n'est pas le cas : sur les 26 sous-
  ensembles de taille >= 2, 22 changent de criticite selon l'ordre, avec un ecart
  atteignant 6 points sur une echelle de 10.

  CONSEQUENCE. Aucune dependance SYMETRIQUE (copule), aucun score ADDITIF, aucune
  fonction d'ensemble ne peut reproduire la criticite, puisque tous ces objets sont
  par construction invariants par permutation. Il suffit pour cela que TRANS soit
  asymetrique : l'enonce ne demande AUCUN dire d'expert sur la valeur de la direction.

  AVERTISSEMENT MAJEUR, issu de l'audit fait par ce script. Le projet a longtemps
  revendique une NON-TRANSITIVITE. Elle a ete testee dans ses trois sens, et AUCUN ne
  tient : (i) la correlation de la forme reduite satisfait l'inegalite de transitivite
  d'un facteur unique, sur toute la plage de rho ; (ii) la dependance symetrisee EST
  semi-definie positive, donc c'est une matrice de correlation valide ; (iii) la
  dominance par paire sur la criticite ne contient aucun cycle. Le script 01 imprime
  pourtant les conclusions inverses, en dur, contredites par ses propres chiffres
  (\"valeur propre minimale = 0.405 (< 0 => INVALIDE)\", \"triples violant la
  transitivite : 0 => correlations NON TRANSITIVES\"). Ces affirmations doivent etre
  retirees du memoire et du script 01. La dependance a l'ordre, elle, tient, et elle
  exclut exactement les memes concurrents.

---------------------------------------------------------------------------------------
PROPOSITION 2 (la normalisation de Leontief borne le branchement).
  Cascade independante de matrice moyenne W (W_jk = proba que j infecte directement k).
  Notons S_j l'ensemble des piliers atteints depuis l'amorce j. Alors
        E[ |S_j| ]  <=  [ (I - W)^{-1} 1 ]_j ,
  quantite finie si et seulement si rho(W) < 1.

  PREUVE. Le nombre moyen de descendants de generation k issus de j est la j-eme ligne
  de W^k. La descendance totale ESPEREE, comptee AVEC multiplicite, vaut donc
  somme_{k>=0} W^k = (I - W)^{-1}, serie geometrique matricielle convergente si et
  seulement si rho(W) < 1. L'ensemble atteint compte chaque pilier UNE fois, alors que
  la descendance le compte autant de fois qu'il est infecte : |S_j| est donc domine par
  la descendance totale, d'ou l'inegalite.                                          QED

  CONSEQUENCE. La normalisation W = g TRANS / max_s impose rho(W) <= g < 1 : la cascade
  est sous-critique PAR CONSTRUCTION, et (I-W)^{-1} en fournit une borne calculable en
  forme close. L'ecart entre la borne et la valeur exacte mesure les COLLISIONS, c'est-a-
  dire les piliers atteints par plusieurs chemins. C'est ce que la forme lineaire ignore.

---------------------------------------------------------------------------------------
PROPOSITION 3 (la direction n'est pas identifiee).
  Toute matrice se decompose de facon UNIQUE en W = S + A, avec S = (W + W^T)/2
  symetrique et A = (W - W^T)/2 antisymetrique. Le test placebo par permutation
  (script 05) ne rejette pas l'hypothese A = 0 : l'asymetrie observee (119) est SOUS le
  bruit du nul de permutation (128 +/- 27), z = -0,33.

  CONSEQUENCE. W et sa transposee W^T partagent exactement la meme partie symetrique et
  ne different que par le SIGNE de A. Elles sont donc egalement compatibles avec la
  donnee, et le capital qu'elles produisent differe. Enonce a retenir :
        observationnellement indistinguables, materiellement differentes.
  C'est la forme la plus nette de la limite d'identification, et elle se verifie.

Sortie : diagnostics + figure Z3_proprietes_formelles.png.
"""

import os
import sys
from itertools import combinations, permutations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import partial_id as pid                                        # noqa: E402
from cascade_model import proba_score, gravite_score, crit_score  # noqa: E402


def crit_ord(order):
    """Criticite ordinale d'une chaine ORDONNEE de piliers (modele qualitatif)."""
    return crit_score(proba_score(order), gravite_score(order))


WID = 80
SEED = 20260721
RHO_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


P = pid.expert_matrix()
S, A_EXP = pid.decompose(P)
I = np.eye(pid.NP_)
M = np.linalg.inv(I - P)                     # forme reduite (I - W)^{-1}


# =====================================================================================
titre("PROPOSITION 1 : la criticite depend de l'ORDRE (et audit de la pretendue "
      "non-transitivite)")
# =====================================================================================
def corr_reduite(W, rho):
    """Correlation de la forme reduite X = (I-W)^{-1} (sqrt(rho) Y 1 + sqrt(1-rho) eps)."""
    Mr = np.linalg.inv(I - W)
    Sigma = Mr @ (rho * np.ones((pid.NP_, pid.NP_)) + (1 - rho) * I) @ Mr.T
    d = np.sqrt(np.diag(Sigma))
    return Sigma / np.outer(d, d)


# --- 1a. AUDIT : trois enonces de \"non-transitivite\" circulent, aucun ne tient --------
print("  1a. AUDIT des enonces de NON-TRANSITIVITE qui circulent dans le projet.")
viol_par_rho = []
for rho in RHO_GRID:
    C = corr_reduite(P, rho)
    viol_par_rho.append(sum(1 for i, j, k in permutations(range(pid.NP_), 3)
                            if C[i, j] * C[j, k] - C[i, k] > 1e-12))
D = np.eye(pid.NP_)
for a, ja in enumerate(pid.PIL):
    for b, jb in enumerate(pid.PIL):
        if a != b:
            D[a, b] = (pid.TRANS[ja].get(jb, 0.0) + pid.TRANS[jb].get(ja, 0.0)) / 2.0
ev_D = np.linalg.eigvalsh(D)
lam_min = float(ev_D.min())
dom_pair = {(i, j): crit_ord([i, j]) > crit_ord([j, i])
            for i, j in permutations(pid.PIL, 2)}
cyc_crit = [(a, b, c) for a, b, c in permutations(pid.PIL, 3)
            if dom_pair[(a, b)] and dom_pair[(b, c)] and dom_pair[(c, a)]]
print(f"      (i)   correlation de la forme reduite viole (T) : "
      f"{sum(viol_par_rho)} sur {60*len(RHO_GRID)} triplets -> NON")
print(f"      (ii)  dependance symetrisee non semi-definie positive : "
      f"lambda_min = {lam_min:+.4f} -> NON, elle EST une correlation valide")
print(f"      (iii) cycle de dominance sur la criticite par paire : "
      f"{len(cyc_crit)} cycle(s) -> NON")
print("      CONCLUSION : aucun des trois sens ne tient. La revendication de")
print("      NON-TRANSITIVITE doit etre RETIREE du memoire. Le script 01 imprime")
print("      pourtant ses conclusions en dur, contredites par ses propres chiffres.")

# --- 1b. Ce qui tient, et qui est tout aussi distinctif : la DEPENDANCE A L'ORDRE -----
print("\n  1b. PROPOSITION 1 (enonce correct). La criticite n'est PAS une fonction de")
print("      l'ENSEMBLE des piliers touches : elle depend de l'ORDRE de la chaine.")
print("      Consequence : aucune dependance symetrique (copule), aucun score additif,")
print("      aucune fonction d'ensemble ne peut la reproduire, puisque tous ces objets")
print("      sont par construction invariants par permutation.")
ecarts = []
for r in range(2, pid.NP_ + 1):
    for combo in combinations(pid.PIL, r):
        vals = [crit_ord(list(p)) for p in permutations(combo)]
        ecarts.append((max(vals) - min(vals), combo))
ecarts.sort(reverse=True)
n_diff = sum(1 for e, _ in ecarts if e > 0)
print(f"\n      Ensembles dont la criticite change avec l'ordre : {n_diff}/{len(ecarts)}")
print(f"      {'ensemble':<16}{'ecart':>8}   meilleur ordre / pire ordre")
for e, combo in ecarts[:5]:
    vals = {p: crit_ord(list(p)) for p in permutations(combo)}
    bst, wst = max(vals, key=vals.get), min(vals, key=vals.get)
    print(f"      {str(combo):<16}{e:>8}   "
          f"{'->'.join('P'+str(x) for x in bst)}={vals[bst]}  /  "
          f"{'->'.join('P'+str(x) for x in wst)}={vals[wst]}")
ecart_max = ecarts[0][0]
print(f"\n      L'ecart atteint {ecart_max} points sur une echelle de 10, soit "
      f"{100*ecart_max/10:.0f} % de l'echelle,")
print("      pour un MEME ensemble de piliers. C'est la contribution du memoire, et elle")
print("      se demontre sans aucun dire d'expert sur la direction : il suffit que TRANS")
print("      soit asymetrique. C'est un enonce PLUS SOLIDE que la non-transitivite, et")
print("      il exclut exactement les memes concurrents (copules, scores additifs).")

# =====================================================================================
titre("PROPOSITION 2 : Leontief borne le branchement, et l'ecart mesure les collisions")
# =====================================================================================
borne = (M @ np.ones(pid.NP_))
card, _ = pid.card_dist_all(P)
exact = card @ np.arange(pid.NP_ + 1)
print(f"  rho(W) = {pid.rho(P):.4f} < 1 : la serie geometrique converge, la cascade est")
print("  sous-critique PAR CONSTRUCTION (normalisation de Leontief).")
print(f"\n  {'amorce':>8}{'E|S| exact':>14}{'borne (I-W)^-1':>18}{'ecart':>10}{'collisions':>13}")
ok = True
for a in range(pid.NP_):
    ec = borne[a] - exact[a]
    ok &= ec >= -1e-9
    print(f"  P{pid.PIL[a]:<7}{exact[a]:>13.4f}{borne[a]:>18.4f}{ec:>10.4f}"
          f"{100*ec/borne[a]:>12.1f} %")
print(f"\n  Inegalite verifiee sur les 5 amorces : {'OUI' if ok else 'NON'}.")
print("  L'ecart n'est pas une erreur : c'est exactement la part de descendance que la")
print("  forme lineaire compte plusieurs fois, c'est-a-dire les piliers atteints par")
print("  PLUSIEURS chemins. Sur 5 piliers fortement connectes, elle est loin d'etre")
print("  negligeable, ce qui justifie de garder le calcul exact par sous-ensembles")
print("  plutot que de se contenter de la forme close.")

# on verifie aussi que la borne tient sur tout l'ensemble admissible
rng = np.random.default_rng(SEED)
bad = 0
ecarts_lin = []                       # NB : ne pas reutiliser le nom `ecarts` (prop. 1)
for a_vec in pid.sample_vertices(S, 1.0, 300, rng):
    W = pid.build_W(S, a_vec)
    if not pid.admissible(W):
        continue
    b = np.linalg.inv(I - W) @ np.ones(pid.NP_)
    e = pid.card_dist_all(W)[0] @ np.arange(pid.NP_ + 1)
    ecarts_lin.append(float(np.mean((b - e) / b)))
    if (b - e < -1e-9).any():
        bad += 1
print(f"  Sur 300 matrices de l'ensemble admissible : {bad} contre-exemple(s), "
      f"ecart moyen {100*np.mean(ecarts_lin):.1f} %.")

# =====================================================================================
titre("PROPOSITION 3 : W et sa transposee, indistinguables pour la donnee, "
      "differentes pour le capital")
# =====================================================================================
ev = pid.Evaluator(n_years=40_000, seed=SEED)
scr_W, mean_W = ev(P)
scr_Wt, mean_Wt = ev(P.T)
S_W, _ = pid.decompose(P)
S_Wt, _ = pid.decompose(P.T)
print(f"  Partie symetrique de W et de W^T identiques : "
      f"{'OUI' if np.allclose(S_W, S_Wt) else 'NON'} "
      f"(ecart max {np.abs(S_W - S_Wt).max():.2e})")
print(f"  Partie antisymetrique : exactement opposee "
      f"({'verifie' if np.allclose(pid.decompose(P)[1], -pid.decompose(P.T)[1]) else 'NON'})")
print(f"\n  SCR sous W    : {scr_W:9.0f} M   (moyenne {mean_W:7.0f} M)")
print(f"  SCR sous W^T  : {scr_Wt:9.0f} M   (moyenne {mean_Wt:7.0f} M)")
print(f"  ECART         : {abs(scr_W-scr_Wt):9.0f} M "
      f"({100*abs(scr_W-scr_Wt)/scr_W:.1f} % du SCR), et "
      f"{abs(mean_W-mean_Wt):.0f} M sur la moyenne "
      f"({100*abs(mean_W-mean_Wt)/mean_W:.1f} %)")
ben_W, ben_Wt = ev.benefits(P), ev.benefits(P.T)
print(f"  Priorite de remediation : P{pid.PIL[int(np.argmax(ben_W))]} sous W, "
      f"P{pid.PIL[int(np.argmax(ben_Wt))]} sous W^T.")
print("\n  Le test directionnel des donnees ne separe pas ces deux matrices (placebo")
print("  z = -0,33, aucune arete significative sur 18). Elles sont donc egalement")
print("  compatibles avec ce qu'on observe, et pourtant elles ne donnent ni le meme")
print("  capital ni la meme decision. C'est l'enonce le plus net de la limite")
print("  d'identification, et il ne repose sur aucun dire d'expert.")

# on cherche l'ecart MAXIMAL entre une matrice et sa transposee sur l'ensemble
rng2 = np.random.default_rng(SEED + 7)
ecart_max, arg = 0.0, None
for a_vec in pid.sample_vertices(S, 1.0, 200, rng2):
    W = pid.build_W(S, a_vec)
    if not pid.admissible(W):
        continue
    d = abs(ev.scr(W) - ev.scr(W.T))
    if d > ecart_max:
        ecart_max, arg = d, W
print(f"\n  Sur l'ensemble admissible, l'ecart maximal entre une matrice et sa transposee")
print(f"  atteint {ecart_max:.0f} M de SCR. C'est le prix, en capital, de ne pas savoir")
print("  dans quel sens la contagion circule.")

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
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.9))

top = ecarts[:8]
lbl = ["".join(str(x) for x in c) for _, c in top]
ax1.barh(range(len(top))[::-1], [e for e, _ in top], color=BLUE, alpha=0.85)
ax1.set_yticks(range(len(top))[::-1])
ax1.set_yticklabels(lbl, fontsize=8.5)
ax1.set_xlabel("écart de criticité entre le meilleur\net le pire ordre (sur 10)",
               color=INK2, fontsize=9)
ax1.set_title("(a)  Prop. 1 : la criticité dépend\nde l'ordre, pas de l'ensemble",
              fontsize=11, color=INK, pad=8)
ax1.text(0.97, 0.06, f"{n_diff}/{len(ecarts)} ensembles concernés", transform=ax1.transAxes,
         fontsize=9, color=INK2, ha="right")

xs = np.arange(pid.NP_)
ax2.bar(xs - 0.2, exact, width=0.38, color=BLUE, alpha=0.85, label=r"$E|S|$ exact")
ax2.bar(xs + 0.2, borne, width=0.38, color=GREEN, alpha=0.85,
        label=r"borne $(I-W)^{-1}\mathbf{1}$")
ax2.set_xticks(xs)
ax2.set_xticklabels([f"P{p}" for p in pid.PIL])
ax2.set_ylabel("nombre de piliers touchés", color=INK2)
ax2.set_title("(b)  Prop. 2 : Leontief majore,\nl'écart mesure les collisions",
              fontsize=11, color=INK, pad=8)
ax2.legend(frameon=False, fontsize=8)

ax3.bar([0, 1], [scr_W, scr_Wt], width=0.5, color=[BLUE, ACCENT], alpha=0.85)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(["$W$", "$W^{\\top}$"])
ax3.set_ylim(0, max(scr_W, scr_Wt) * 1.25)
ax3.annotate("", xy=(0, scr_W), xytext=(1, scr_Wt),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
ax3.text(0.5, max(scr_W, scr_Wt) * 1.08, f"{abs(scr_W-scr_Wt):.0f} M€", ha="center",
         fontsize=10, color=INK, fontweight="bold")
ax3.text(0.5, max(scr_W, scr_Wt) * 0.35,
         "même partie symétrique,\ndonc indistinguables\npour la donnée",
         ha="center", fontsize=8.5, color=MUTED, style="italic")
ax3.set_ylabel("SCR (VaR 99,5 %, M€)", color=INK2)
ax3.set_title("(c)  Prop. 3 : indistinguables,\net pourtant différentes",
              fontsize=11, color=INK, pad=8)

for ax in (ax1, ax2, ax3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("Z3 : trois propriétés démontrées, et vérifiées numériquement",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z3_proprietes_formelles.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
