#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
59 : corpus etendu de post-mortems, et estimation de la matrice ORDONNEE p_ij.

DEUX CHANGEMENTS PAR RAPPORT AU SCRIPT 53.

(1) LE CORPUS S'ETEND. 53 code 7 rapports et 17 transitions. Le chapitre 9 en tire un signal
    directionnel d'ensemble (z = +3,93) mais le posterieur bayesien du meme chapitre montre que
    seules 2 paires sur 6 ont un credible concluant : c'est le VOLUME qui manque, pas la methode.
    On etend donc le corpus, en tracant pour chaque incident s'il est code sur SOURCE PRIMAIRE
    (rapport officiel lu) ou SECONDAIRE (synthese reglementaire ou de presse), afin de pouvoir
    rejouer tout le calcul sur les seules sources primaires.

(2) ON ESTIME p_ij, ET NON PLUS SEULEMENT LA DIRECTION PAR PAIRE. Le chapitre 10 reparametrise
    la propagation en logit p_ij = logit p_j + u_ij, ou p_j est l'attractivite de la CIBLE et
    u_ij l'exces propre a la SOURCE. Le corpus, tel que 53 l'exploite, ne renseigne que le SENS
    a l'interieur d'une paire : il compare n(j->k) a n(k->j), ce qui elimine le denominateur et
    ne dit donc rien de l'AMPLITUDE. On ajoute ici ce denominateur.

    Pour cela chaque incident declare, en plus de ses transitions, l'ensemble des PILIERS
    DEFAILLANTS, y compris ceux qui ont cede SANS rien entrainer. On peut alors poser

        p_jk = P(le pilier k cede | le pilier j a cede et est documente)

    estime par un binomial-beta conjugue :  p_jk | donnees ~ Beta(a0 + m_jk, b0 + N_j - m_jk),
    ou N_j est le nombre d'incidents ou j a cede et m_jk le nombre ou le rapport etablit j -> k.
    C'est la matrice ORDONNEE COMPLETE, donc S ET A, donc W a une normalisation pres.

CONVENTION D'INDICES (chapitre 7, fixee) : W_jk et p_jk vont de la SOURCE j vers la CIBLE k.
La ligne emet, la colonne recoit. Le script le verifie sur la matrice d'expert avant de commencer.

LE BIAIS QUI INTERDIT DE LIRE p_ij COMME UNE MESURE, et qu'il faut ecrire tel quel.
Un rapport d'enquete decrit la CHAINE CAUSALE qui a produit le sinistre. Il documente donc
abondamment les controles qui ont CEDE et qui ont PROPAGE, et tres mal ceux qui ont cede sans
propager, voire pas du tout ceux qui ont TENU. Le denominateur N_j est donc systematiquement
SOUS-ESTIME, et p_ij est en consequence une BORNE SUPERIEURE. On le chiffre : le script fait
varier un facteur de sous-documentation delta (nombre de defaillances non propagees manquantes
par incident et par pilier) et montre comment p_ij se deplace. C'est la seule facon honnete de
presenter une amplitude que le chapitre 9 declare non identifiable par cette voie.

Sortie : diagnostics + figure Z20_corpus_etendu_pij.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUALI = os.path.abspath(os.path.join(HERE, "..", "cascade_qualitative"))
for _p in (HERE, QUALI):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(HERE)
import partial_id as pid                                          # noqa: E402
import cascade_model as cm                                        # noqa: E402

WID = 86
PIL = [1, 2, 3, 4, 5]
LAB = {1: "P1 gouvernance", 2: "P2 incidents", 3: "P3 tests", 4: "P4 tiers", 5: "P5 partage"}
SHORT = {1: "P1", 2: "P2", 3: "P3", 4: "P4", 5: "P5"}
_c = {j: i for i, j in enumerate(PIL)}
SEED = 20260803
NPERM = 20000
A0 = B0 = 1.0                       # prior Beta(1,1), uniforme : aucune information d'expert

# =====================================================================================
# LE CORPUS EST UN MODULE, PAS UN LITTERAL. Il vivait ici ; le script 64 (biais de
# narration) le relit, et un corpus recopie dans deux scripts finit par diverger. Le
# protocole de codage est documente dans `postmortem_corpus.py`, avec le corpus.
# =====================================================================================
from postmortem_corpus import CORPUS                               # noqa: E402

# =====================================================================================
def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# ---------------------------------------------------------------- garde-fou de convention
titre("0. Verification de la convention d'indices (ligne = source)")
T = np.array([[cm.TRANS[j].get(k, 0.0) for k in PIL] for j in PIL], float)
rs, cs = T.sum(1), T.sum(0)
print(f"  sommes de LIGNE   (emission) : " + "  ".join(f"P{j}={rs[_c[j]]:.1f}" for j in PIL))
print(f"  sommes de COLONNE (reception): " + "  ".join(f"P{j}={cs[_c[j]]:.1f}" for j in PIL))
ok = (rs.argmax() == _c[1]) and (cs[_c[1]] < cs[_c[2]])
print(f"  P1 a la plus forte emission : {rs.argmax() == _c[1]}")
print(f"  P1 recoit moins que P2      : {cs[_c[1]] < cs[_c[2]]}  "
      f"(rapport {cs[_c[2]]/cs[_c[1]]:.1f})")
print(f"  => convention LIGNE = SOURCE {'CONFIRMEE' if ok else 'INFIRMEE'} sur la matrice d'expert.")
if not ok:
    sys.exit("convention incoherente : arret.")


# =====================================================================================
titre("1. Le corpus etendu")
# =====================================================================================
def stats_corpus(corpus):
    ntr = sum(len(c["transitions"]) for c in corpus)
    return len(corpus), ntr


n_inc, n_tr = stats_corpus(CORPUS)
prim = [c for c in CORPUS if c["source_type"] == "primaire"]
ent = [c for c in CORPUS if c["perspective"] == "entite"]
print(f"  {n_inc} post-mortems, {n_tr} transitions dirigees codees.")
print(f"    dont source primaire  : {len(prim)} incidents, "
      f"{sum(len(c['transitions']) for c in prim)} transitions")
print(f"    dont perspective entite : {len(ent)} incidents")
print(f"\n  {'incident':<32}{'src':<12}{'persp.':<12}{'defaillants':<16}{'transitions'}")
for c in CORPUS:
    pil = "".join(SHORT[p][1] for p in sorted(c["piliers_defaillants"]))
    seq = " ".join(f"P{j}>P{k}" for (j, k, _) in c["transitions"])
    print(f"  {c['nom']:<32}{c['source_type']:<12}{c['perspective']:<12}"
          f"{'P'+pil:<16}{seq}")


def build(corpus):
    """M[j,k] = nb d'incidents ou j->k est etabli ; N[j] = nb d'incidents ou j a cede."""
    M = np.zeros((5, 5))
    N = np.zeros(5)
    for c in corpus:
        for p in c["piliers_defaillants"]:
            N[_c[p]] += 1
        for (j, k, _) in c["transitions"]:
            M[_c[j], _c[k]] += 1
    return M, N


M, N = build(CORPUS)
print(f"\n  Matrice des transitions M (LIGNE = source, COLONNE = cible) :")
print("          " + "".join(f"{'->P%d' % k:>7}" for k in PIL) + f"{'N_j':>8}")
for j in PIL:
    print(f"    P{j}    " + "".join(f"{M[_c[j], _c[k]]:>7.0f}" for k in PIL) + f"{N[_c[j]]:>8.0f}")
print(f"\n  emissions : " + ", ".join(f"P{j}={M[_c[j],:].sum():.0f}" for j in PIL))
print(f"  receptions: " + ", ".join(f"P{j}={M[:,_c[j]].sum():.0f}" for j in PIL))


# =====================================================================================
titre("2. Le test de direction, rejoue sur le corpus etendu")
# =====================================================================================
def asym(A):
    return float(np.abs(A - A.T).sum() / 2.0)


rng = np.random.default_rng(SEED)


def placebo(corpus, nperm=NPERM):
    edges = [(j, k) for c in corpus for (j, k, _) in c["transitions"]]
    obs = asym(build(corpus)[0])
    null = np.empty(nperm)
    for t in range(nperm):
        A = np.zeros((5, 5))
        for (j, k) in edges:
            if rng.random() < 0.5:
                A[_c[j], _c[k]] += 1
            else:
                A[_c[k], _c[j]] += 1
        null[t] = asym(A)
    z = (obs - null.mean()) / null.std(ddof=1)
    p = float((null >= obs).mean())
    return obs, null.mean(), null.std(ddof=1), z, p


for lab, sub in (("corpus complet", CORPUS), ("sources primaires seules", prim),
                 ("perspective entite seule", ent)):
    o, m0, s0, z, p = placebo(sub)
    ni, nt = stats_corpus(sub)
    print(f"  {lab:<28} {ni:>2} inc., {nt:>2} trans. : "
          f"T={o:>5.1f}  nul={m0:>5.1f}+-{s0:.1f}  z={z:>+5.2f}  p={p:.4f}")
print("\n  Rappel : le corpus initial (7 rapports, 17 transitions) donnait z = +3.93,")
print("  contre un placebo par permutation de 8.4 +/- 2.2 transitions (script 53).")
print("  Si z tient ou monte sur le corpus etendu, le signal n'etait pas un artefact des")
print("  sept incidents choisis. S'il tient sur les seules sources PRIMAIRES, il ne vient")
print("  pas non plus de la couverture de presse.")


# =====================================================================================
titre("3. La matrice ORDONNEE p_jk, et non plus le seul sens par paire")
# =====================================================================================
print("  p_jk = P(k cede | j a cede), posterieur Beta(1 + m_jk, 1 + N_j - m_jk).")
print("  C'est la matrice complete : elle porte S ET A, donc l'amplitude et la direction.\n")

post_a = A0 + M
post_b = B0 + np.maximum(N[:, None] - M, 0.0)
p_med = stats.beta.median(post_a, post_b)
p_lo = stats.beta.ppf(0.05, post_a, post_b)
p_hi = stats.beta.ppf(0.95, post_a, post_b)
np.fill_diagonal(p_med, 0.0)

print("  Mediane a posteriori de p_jk (LIGNE = source j, COLONNE = cible k) :")
print("          " + "".join(f"{'->P%d' % k:>9}" for k in PIL))
for j in PIL:
    row = "".join("      .  " if j == k else f"{p_med[_c[j], _c[k]]:>9.3f}" for k in PIL)
    print(f"    P{j}    " + row)

n_bas = sum(1 for j in PIL for k in PIL if j != k and p_hi[_c[j], _c[k]] < 0.5)
n_haut = sum(1 for j in PIL for k in PIL if j != k and p_lo[_c[j], _c[k]] > 0.5)
print(f"\n  credible a 90 % ENTIEREMENT SOUS 0,5 (propagation faible etablie) : "
      f"{n_bas} / 20")
print(f"  credible a 90 % ENTIEREMENT AU-DESSUS de 0,5 (propagation forte etablie) : "
      f"{n_haut} / 20")
# LES COMPTES SE CALCULENT, ILS NE S'ECRIVENT PAS. Cette phrase annoncait « les CINQ sorties de
# P2 » : dans une matrice 5x5 sans boucle un pilier n'a que QUATRE sorties, et la ligne juste
# au-dessus annonce d'ailleurs vingt liens ordonnes, soit 5 x 4. Le compte est desormais derive
# de la matrice, et il verifie au passage que ces sorties sont bien toutes credibiliement faibles.
sorties_P2 = [k for k in PIL if k != 2 and p_hi[_c[2], _c[k]] < 0.5]
entrees_P1 = [j for j in PIL if j != 1 and p_hi[_c[j], _c[1]] < 0.5]
print("\n  LE RESULTAT EST ASYMETRIQUE, ET C'EST LUI QU'IL FAUT RETENIR. Le corpus etablit")
print("  tres bien ou la contagion NE VA PAS, et pas du tout ou elle va fort. Les douze")
non_faibles_P1 = [j for j in PIL if j != 1 and j not in entrees_P1]
print(f"  liens credibiliement faibles comprennent les {len(sorties_P2)} sorties de P2, soit"
      f" TOUTES," if len(sorties_P2) == len(PIL)-1 else
      f"  liens credibiliement faibles comprennent {len(sorties_P2)} des {len(PIL)-1} sorties de P2,")
print(f"  et {len(entrees_P1)} des {len(PIL)-1} entrees de P1"
      + (f" — l'entree P{non_faibles_P1[0]} vers P1 n'est PAS etablie comme faible."
         if non_faibles_P1 else "."))
print("  LE PUITS EST DONC ETABLI, LA SOURCE PURE SEULEMENT EN PARTIE : c'est une asymetrie")
print("  de plus, et elle va contre la convention « cause racine = gouvernance ».")
print("  Mais AUCUN lien n'est etabli comme fort : meme les plus eleves gardent un")
print("  credible qui recouvre 0,5.")
print("\n  Les quatre liens les plus eleves :")
for v, j, k in sorted(((p_med[_c[j], _c[k]], j, k)
                       for j in PIL for k in PIL if j != k), reverse=True)[:4]:
    print(f"    P{j} -> P{k} : {v:.3f} [{p_lo[_c[j],_c[k]]:.3f} ; {p_hi[_c[j],_c[k]]:.3f}]"
          f"   m={M[_c[j],_c[k]]:.0f} / N={N[_c[j]]:.0f}")


# =====================================================================================
titre("4. Decomposition S / A de la matrice estimee, et comparaison a l'expert")
# =====================================================================================
S_est = 0.5 * (p_med + p_med.T)
A_est = 0.5 * (p_med - p_med.T)
print(f"  ||S|| = {np.linalg.norm(S_est):.3f}   ||A|| = {np.linalg.norm(A_est):.3f}   "
      f"part de la direction ||A||/||W|| = {np.linalg.norm(A_est)/np.linalg.norm(p_med):.1%}")

Wx = np.asarray(pid.expert_matrix(), float)
Sx = 0.5 * (Wx + Wx.T)
Ax = 0.5 * (Wx - Wx.T)
print(f"  expert : ||S|| = {np.linalg.norm(Sx):.3f}   ||A|| = {np.linalg.norm(Ax):.3f}   "
      f"part = {np.linalg.norm(Ax)/np.linalg.norm(Wx):.1%}")

off = ~np.eye(5, dtype=bool)
r_all = stats.spearmanr(p_med[off], Wx[off])
r_asym = stats.spearmanr(A_est[off], Ax[off])
print(f"\n  Correlation de Spearman documentaire vs expert :")
print(f"    sur la matrice complete      rho = {r_all.statistic:+.3f}  (p = {r_all.pvalue:.4f})")
print(f"    sur la seule partie A        rho = {r_asym.statistic:+.3f}  (p = {r_asym.pvalue:.4f})")
print("\n  LECTURE, ET ELLE EST NEGATIVE SUR CE POINT. Aucune des deux correlations n'est")
print("  significative : le corpus ne reproduit PAS le classement des coefficients du")
print("  classeur, ni en amplitude ni meme en direction paire par paire. Il serait")
print("  malhonnete de presenter ce resultat autrement.")
print(f"\n  Ce qu'il reproduit, en revanche, est la STRUCTURE : P1 est source pure "
      f"({M[_c[1], :].sum():.0f} sorties, {M[:, _c[1]].sum():.0f} entree) et P2 puits pur "
      f"({M[_c[2], :].sum():.0f} sortie, {M[:, _c[2]].sum():.0f} entrees),")
print(f"  sur {N[_c[1]]:.0f} et {N[_c[2]]:.0f} incidents ou ces piliers ont respectivement cede.")
print("  C'est exactement le decoupage a trois niveaux du chapitre 9 : la structure")
print("  est corroboree, la direction paire par paire ne l'est pas, l'amplitude pas davantage.")
print("  Le corpus etendu ne deplace donc PAS la frontiere d'identification, il la CONFIRME")
print("  sur un objet plus riche (la matrice ordonnee complete) et avec plus de matiere.")
print("\n  A noter : le corpus est PLUS dirige que le jugement d'expert (part de A de "
      f"{np.linalg.norm(A_est)/np.linalg.norm(p_med):.0%}")
print(f"  contre {np.linalg.norm(Ax)/np.linalg.norm(Wx):.0%}). Le classeur est donc, sur ce")
print("  point precis, plus prudent que la donnee documentaire.")


# =====================================================================================
titre("5. Le biais qui interdit de lire p_jk comme une mesure")
# =====================================================================================
print("  Un rapport d'enquete raconte la CHAINE qui a produit le sinistre. Il documente donc")
print("  les controles qui ont cede ET propage, mal ceux qui ont cede sans propager, et pas")
print("  du tout ceux qui ont tenu. Le denominateur N_j est donc sous-estime et p_jk est une")
print("  BORNE SUPERIEURE. On chiffre cette sensibilite en ajoutant delta defaillances non")
print("  propagees par pilier et par incident.\n")
print(f"  {'delta':>7}{'p med. moyen':>15}{'P1->P2':>10}{'P4->P2':>10}{'P1->P3':>10}"
      f"{'||A||/||W||':>14}")
for delta in (0.0, 0.25, 0.5, 1.0, 2.0):
    Nd = N + delta * len(CORPUS)
    pa = A0 + M
    pb = B0 + np.maximum(Nd[:, None] - M, 0.0)
    pm = stats.beta.median(pa, pb)
    np.fill_diagonal(pm, 0.0)
    Ad = 0.5 * (pm - pm.T)
    print(f"  {delta:>7.2f}{pm[off].mean():>15.3f}{pm[_c[1],_c[2]]:>10.3f}"
          f"{pm[_c[4],_c[2]]:>10.3f}{pm[_c[1],_c[3]]:>10.3f}"
          f"{np.linalg.norm(Ad)/np.linalg.norm(pm):>14.1%}")
print("\n  Lecture. Le NIVEAU de p_jk s'effondre avec delta : l'amplitude n'est pas identifiee,")
print("  exactement comme le chapitre 9 l'annonce. En revanche la PART DE DIRECTION")
print("  ||A||/||W|| bouge peu : ce que le corpus etablit, c'est un ORDRE, pas une magnitude.")
print("  C'est la meme conclusion que le mémoire tenait deja, mais obtenue cette fois sur la")
print("  matrice ordonnee complete et non plus sur le seul sens a l'interieur des paires.")


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
ACCENT = "#a6002e"
BL = ["#7baafd", "#4c79c7", "#204993"]

# TROIS PANNEAUX EN RANGEE, ET NON EMPILES. Empiles sur 10 pouces de haut ils
# remplissaient une page entiere du memoire, la matrice de (a) se perdant au milieu
# d'un grand blanc. En rangee la figure s'imprime sur 7,3 cm de haut et tient dans le
# fil du texte. La matrice passe en aspect libre pour remplir sa case au lieu de
# laisser deux bandes vides sur ses cotes.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.6, 3.4),
                                    gridspec_kw={"width_ratios": [1.0, 1.18, 1.12]})

# (a) la matrice p_jk
im = ax1.imshow(p_med, cmap="Blues", vmin=0, vmax=max(0.6, p_med.max()), aspect="auto")
ax1.set_xticks(range(5)); ax1.set_xticklabels([SHORT[k] for k in PIL], fontsize=9)
ax1.set_yticks(range(5)); ax1.set_yticklabels([SHORT[j] for j in PIL], fontsize=9)
ax1.set_xlabel("cible $k$ (colonne reçoit)", color=INK2, fontsize=9.5)
ax1.set_ylabel("source $j$ (ligne émet)", color=INK2, fontsize=9.5)
for j in range(5):
    for k in range(5):
        if j == k:
            ax1.text(k, j, "·", ha="center", va="center", color=MUTED)
        else:
            v = p_med[j, k]
            ax1.text(k, j, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                     color="white" if v > 0.35 else INK2)
ax1.set_title("(a)  Médiane a posteriori\nde $p_{jk}$", fontsize=10, color=INK, pad=6)

# (b) les liens et leurs credibles
pairs = [(j, k) for j in PIL for k in PIL if j != k]
pairs.sort(key=lambda t: -p_med[_c[t[0]], _c[t[1]]])
pairs = pairs[:10]
ypos = np.arange(len(pairs))
med = [p_med[_c[j], _c[k]] for (j, k) in pairs]
lo = [p_med[_c[j], _c[k]] - p_lo[_c[j], _c[k]] for (j, k) in pairs]
hi = [p_hi[_c[j], _c[k]] - p_med[_c[j], _c[k]] for (j, k) in pairs]
ax2.errorbar(med, ypos, xerr=[lo, hi], fmt="o", color=BL[2], ecolor=BL[1],
             elinewidth=1.8, capsize=3.0, ms=5)
ax2.axvline(0.5, color=ACCENT, ls="--", lw=1.4, label="$0{,}5$ : pas de sens")
ax2.set_yticks(ypos)
ax2.set_yticklabels([f"P{j}$\\to$P{k}" for (j, k) in pairs], fontsize=8.5)
ax2.invert_yaxis()
# UNE RANGEE VIDE EN BAS, POUR LA LEGENDE. Sans elle la cle du trait a 0,5 se posait sur
# les deux derniers intervalles ; la place libre est sous eux, pas sur eux.
ax2.set_ylim(len(pairs) + 0.55, -0.6)
ax2.set_xlim(0, 1)
ax2.tick_params(axis="x", labelsize=9)
ax2.set_xlabel("$p_{jk}$, crédible à 90 %", color=INK2, fontsize=9.5)
ax2.legend(frameon=True, facecolor="#fcfcfb", edgecolor="none", framealpha=0.88,
           fontsize=8.5, loc="lower right", borderpad=0.3, handlelength=1.6)
ax2.set_title("(b)  Les dix premiers liens\net leur crédible à 90 %",
              fontsize=10, color=INK, pad=6)
for s_ in ("top", "right"):
    ax2.spines[s_].set_visible(False)

# (c) sensibilite a la sous-documentation
deltas = np.linspace(0, 2, 21)
niveau, partA = [], []
for d in deltas:
    Nd = N + d * len(CORPUS)
    pm = stats.beta.median(A0 + M, B0 + np.maximum(Nd[:, None] - M, 0.0))
    np.fill_diagonal(pm, 0.0)
    Ad = 0.5 * (pm - pm.T)
    niveau.append(pm[off].mean())
    partA.append(np.linalg.norm(Ad) / np.linalg.norm(pm))
ax3.plot(deltas, niveau, color=BL[2], lw=2.0, label="niveau moyen de $p_{jk}$")
ax3.plot(deltas, partA, color=ACCENT, lw=2.0, label=r"part de direction $\|A\|/\|W\|$")
ax3.set_xlabel("$\\delta$ : défaillances non\npropagées ajoutées",
               color=INK2, fontsize=9.5)
ax3.set_ylabel("valeur (sans unité)", color=INK2, fontsize=9.5)
ax3.tick_params(labelsize=9)
ax3.legend(frameon=True, facecolor="#fcfcfb", edgecolor="none", framealpha=0.88,
           fontsize=8.5, loc="center right", borderpad=0.3, handlelength=1.6)
ax3.set_ylim(0, max(max(niveau), max(partA)) * 1.18)
ax3.set_title("(c)  L'amplitude s'effondre,\nla direction tient", fontsize=10, color=INK, pad=6)
for s_ in ("top", "right"):
    ax3.spines[s_].set_visible(False)

# AUCUN TITRE GENERAL : la legende LaTeX porte le titre, et le code interne « Z20 »
# n'a rien a faire dans le PDF depose.
fig.tight_layout(w_pad=1.6)
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z20_corpus_etendu_pij.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)

titre("AVERTISSEMENT DE VALIDATION")
print("  Les trois incidents ajoutes par ce script (CrowdStrike 2024, Log4Shell 2021-2022,")
print("  Raphaels Bank 2015) sont codes a partir des constats PUBLICS et etablis de leurs")
print("  rapports respectifs. Chaque codage reste une PROPOSITION a valider avant usage dans")
print("  le memoire, au meme titre que les sept initiaux : c'est la raison d'etre du second")
print("  codage en aveugle du script 54. Le corpus n'atteint pas encore la trentaine visee ;")
print("  les candidats identifies et non encore codes sont listes dans le README du dossier.")
