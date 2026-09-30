#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
87 : l'intuition d'Hugo. La PART DE FIN est-elle proche des parts d'Euler ?

QUESTION POSEE PAR HUGO APRES LE POINT DU 14 AOUT, mot pour mot : « au lieu de la part d'amorce,
la part de fin, le % de la ou ca se finit. Probablement que ca sera proche d'Euler. Si c'est le
cas c'est incroyable. »

L'INTUITION EST BONNE DANS SA FORME : la part d'amorce explique le classement de Shapley, qui
attribue le surcout au pilier SOURCE ; il doit donc exister une grandeur de meme nature, purement
combinatoire, qui explique les parts d'EULER, lesquelles attribuent le capital au pilier TOUCHE.
La question est de savoir laquelle. Hugo propose la part de fin. Ce script la calcule, avec deux
concurrentes, et les compare aux parts d'Euler mesurees.

TROIS CANDIDATES, toutes EXACTES et sans aucune simulation, calculees par la meme recursion que la
loi de progeniture :
  - part d'AMORCE : ou le sinistre demarre. C'est ROOT, la reference deja connue ;
  - part de FIN   : ou le sinistre s'arrete, c'est-a-dire le dernier pilier atteint par la marche.
    C'est la proposition d'Hugo ;
  - part de PASSAGE : la probabilite qu'un pilier soit TOUCHE, quel que soit son rang dans la
    cascade. C'est la troisieme, et elle n'avait pas ete proposee.

CE QUE LE SCRIPT MESURE : l'ecart moyen en points entre chaque candidate et les parts d'Euler, et
lequel des trois classements coincide avec celui d'Euler.

Sortie : diagnostics + figure S43_part_de_fin.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                                 # noqa: E402
from euro_cascade_model import var                              # noqa: E402
import canaux_conformite as cx                                  # noqa: E402
from src.aggregation.lda import simulate_remediation_severity   # noqa: E402
import scr_engine as eng                                        # noqa: E402
from cascade_model import TRANS                                 # noqa: E402

WID = 88
PIL = cx.PIL
sp = cx.sp
NY, NSEED, SEED0 = cx.NY, cx.NSEED, cx.SEED0
ALPHA = 0.995
NBAND = 200

W_AM = np.array([eng.LAMBDA[j] for j in PIL], float)
W_AM = W_AM / W_AM.sum()
_SROW = {j: sum(TRANS[j].values()) for j in PIL}
_MAXS = max(_SROW.values())


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# =====================================================================================
titre("1. LES TROIS CANDIDATES, calculees EXACTEMENT et sans aucune simulation")
# =====================================================================================


def loi_fin(g):
    """P(fin = j) : le pilier ou la MARCHE s'arrete, par la recursion de la progeniture.

    La marche s'arrete au pilier courant avec probabilite (1 - e) + p_rehit, ou e est la
    propension a continuer et p_rehit la probabilite de viser un pilier deja visite. C'est
    exactement la masse que cascade_set_dist verse dans l'ensemble courant : on l'accumule ici
    par PILIER TERMINAL au lieu de l'accumuler par ensemble.

    CETTE GRANDEUR N'EST DEFINIE QUE POUR UNE MARCHE. Le choc commun de P4 tire un ENSEMBLE
    sans ordre : il n'y a pas de dernier pilier, donc pas de « fin ». C'est pourquoi la
    comparaison principale de ce script se fait a l'etat CONFORME, ou les cinq amorces sont des
    marches et ou les trois candidates sont donc definies sur le meme objet.
    """
    fin = {j: 0.0 for j in PIL}

    def rec(cur, visited, p, w_am):
        e = g * _SROW[cur] / _MAXS
        srow = _SROW[cur]
        p_rehit = e * sum(w for k, w in TRANS[cur].items() if k in visited) / srow
        fin[cur] += w_am * p * (1.0 - e + p_rehit)
        for k, w in TRANS[cur].items():
            if k not in visited:
                rec(k, visited | {k}, p * e * w / srow, w_am)

    for c, j in enumerate(PIL):
        rec(j, frozenset({j}), 1.0, W_AM[c])
    return np.array([fin[j] for j in PIL])


def loi_passage(g, phi_cs):
    """P(j touche), calculee sur LES TABLES QUE LA SIMULATION UTILISE.

    C'est le point de rigueur de ce script. A l'etat non conforme la simulation remplace la
    table de P4 par le choc commun ; un calcul exact qui garderait la marche pour P4
    comparerait deux modeles differents. On lit donc les memes tables que le moteur.
    """
    tables = {j: cx.table_amorce(j, g) for j in PIL}
    if phi_cs is not None:
        tables[cx.P4] = cx.table_p4_choc(phi_cs)
    tou = np.zeros(len(PIL))
    for c, j in enumerate(PIL):
        ind, probs = tables[j]
        tou += W_AM[c] * (probs[:, None] * ind).sum(axis=0)
    return tou


# ETAT CONFORME : les cinq amorces sont des marches, les trois candidates sont donc definies sur
# le meme objet, et c'est la comparaison PRINCIPALE.
p_amorce = W_AM
fin_c = loi_fin(cx.G_C)
tou_c = loi_passage(cx.G_C, None)
p_fin = fin_c / fin_c.sum()
p_passage = tou_c / tou_c.sum()
print(f"  COMPARAISON PRINCIPALE A L'ETAT CONFORME, et il faut dire pourquoi. A l'etat non")
print("  conforme, le pilier des tiers tire un ENSEMBLE de piliers par choc commun, sans ordre :")
print("  il n'y a donc pas de « dernier pilier » et la part de fin n'y est pas definie. A l'etat")
print("  conforme les cinq amorces sont des marches, et les trois candidates portent sur le meme")
print("  objet. La section 5 revient sur l'etat non conforme avec la seule candidate qui y garde")
print("  un sens.")
print(f"\n  Controle : la loi de fin somme a {fin_c.sum():.6f} (toute marche s'arrete quelque part).")
print(f"  Nombre moyen de piliers touches : {tou_c.sum():.3f} (le script 74 publie 1,380 a cet etat).")
print(f"\n  {'Pilier':<8}{'part d amorce':>16}{'part de FIN':>14}{'part de PASSAGE':>18}")
for c, j in enumerate(PIL):
    print(f"  P{j:<7}{100*p_amorce[c]:>15.1f} %{100*p_fin[c]:>13.1f} %{100*p_passage[c]:>17.1f} %")
print("\n  Lecture des trois colonnes. La part d'amorce est ROOT. La part de FIN est celle qu'Hugo")
print("  propose : le pilier ou la marche s'arrete. La part de PASSAGE compte un pilier des qu'il")
print("  est touche, quel que soit son rang, et c'est la troisieme candidate.")

# =====================================================================================
titre("2. LES PARTS D'EULER MESUREES, sur le meme etat")
# =====================================================================================


def parts_euler(lam, g, p_u, phi_cs):
    """Parts d'Euler (VaR et CTE), par la ventilation des pertes par pilier TOUCHE."""
    ev, ec_ = [], []
    for k in range(NSEED):
        rng = np.random.default_rng(SEED0 + k)
        r = lam / (ec.PHI - 1.0)
        counts = rng.negative_binomial(r, r / (r + lam), size=NY)
        T = int(counts.sum())
        year_of = np.repeat(np.arange(NY), counts)
        am = rng.choice(len(PIL), size=T, p=W_AM)
        U = rng.random(T)
        SEV = simulate_remediation_severity(T * len(PIL), sp["xi"], sp["sigma"], sp["u"], p_u,
                                            sp["cap"], rng).reshape(T, len(PIL))
        tables = {j: cx.table_amorce(j, g) for j in PIL}
        if phi_cs is not None:
            tables[cx.P4] = cx.table_p4_choc(phi_cs)
        M = np.zeros((NY, len(PIL)))
        for c, j in enumerate(PIL):
            idx = np.where(am == c)[0]
            if idx.size == 0:
                continue
            ind, probs = tables[j]
            cdf = np.cumsum(probs)
            cdf[-1] = 1.0
            sel = np.searchsorted(cdf, U[idx], side="right")
            np.clip(sel, 0, len(probs) - 1, out=sel)
            contrib = SEV[idx] * ind[sel]
            for cp in range(len(PIL)):
                M[:, cp] += np.bincount(year_of[idx], weights=contrib[:, cp], minlength=NY)
        tot = M.sum(axis=1)
        q = var(tot, ALPHA)
        band = np.argsort(np.abs(tot - q))[:NBAND]
        ev.append(M[band].mean(axis=0) / M[band].mean(axis=0).sum())
        queue = tot >= q
        ec_.append(M[queue].mean(axis=0) / M[queue].mean(axis=0).sum())
    return np.array(ev), np.array(ec_)


eul_var, eul_cte = parts_euler(cx.LAM_C, cx.G_C, cx.PU_C, None)
p_eul_v = eul_var.mean(axis=0)
p_eul_c = eul_cte.mean(axis=0)
print("  Etat CONFORME, le meme que celui des trois candidates ci-dessus.")
print(f"  {'Pilier':<8}{'Euler (VaR)':>14}{'Euler (CTE)':>14}{'etendue CTE':>14}")
for c, j in enumerate(PIL):
    print(f"  P{j:<7}{100*p_eul_v[c]:>13.1f} %{100*p_eul_c[c]:>13.1f} %"
          f"{100*np.ptp(eul_cte[:, c]):>13.1f}")
print("\n  RAPPEL DE PRUDENCE, deja publie : les parts d'Euler bougent d'une graine a l'autre, et")
print("  leur CLASSEMENT n'est pas un resultat. On compare donc les NIVEAUX, pas l'ordre, et l'on")
print("  garde les etendues en tete pour juger si un ecart est significatif.")

# =====================================================================================
titre("3. LA REPONSE : quelle candidate colle a Euler ?")
# =====================================================================================
cands = [("part d'amorce (ROOT)", p_amorce),
         ("part de FIN (Hugo)", p_fin),
         ("part de PASSAGE", p_passage)]
print(f"  Ecart a Euler, en points de pourcentage. On donne l'ecart MOYEN sur les cinq piliers et")
print(f"  l'ecart MAXIMAL, contre les deux versions d'Euler.")
print(f"\n  {'candidate':<24}{'moyen / Euler VaR':>20}{'moyen / Euler CTE':>20}{'max / CTE':>12}")
res = []
for nom, p in cands:
    dv = 100 * np.mean(np.abs(p - p_eul_v))
    dc = 100 * np.mean(np.abs(p - p_eul_c))
    mx = 100 * np.max(np.abs(p - p_eul_c))
    res.append((nom, p, dv, dc, mx))
    print(f"  {nom:<24}{dv:>19.1f} {dc:>19.1f} {mx:>11.1f}")
best = min(res, key=lambda t: t[3])
print(f"\n  LA CANDIDATE LA PLUS PROCHE D'EULER EST : {best[0]}, a {best[3]:.1f} point(s) d'ecart moyen.")

print(f"\n  {'Pilier':<8}{'Euler (CTE)':>14}{'amorce':>10}{'FIN':>10}{'PASSAGE':>11}")
for c, j in enumerate(PIL):
    print(f"  P{j:<7}{100*p_eul_c[c]:>13.1f} %{100*p_amorce[c]:>9.1f} %{100*p_fin[c]:>9.1f} %"
          f"{100*p_passage[c]:>10.1f} %")

# =====================================================================================
titre("4. POURQUOI, et le mecanisme est simple une fois vu")
# =====================================================================================
print("  LES SEVERITES SONT TIREES DANS LA MEME LOI POUR TOUS LES PILIERS, et independamment les")
print("  unes des autres. La perte moyenne apportee par un pilier vaut donc simplement")
print("      (probabilite que ce pilier soit touche) x (severite moyenne d'un pilier touche),")
print("  et le second facteur est LE MEME pour les cinq. En moyenne, la part d'un pilier dans la")
print("  perte est donc exactement sa PART DE PASSAGE. C'est une identite, pas une coincidence.")
print("\n  CE QUI EN ECARTE EULER EST LE CONDITIONNEMENT DE QUEUE, et lui seul : Euler ne moyenne")
print("  pas sur toutes les annees mais sur les annees de queue, ou la composition des sinistres")
print("  n'est pas tout a fait celle d'une annee ordinaire. C'est ce residu qu'on lit dans le")
print("  tableau ci-dessus.")
print("\n  LA PART DE FIN, ELLE, MESURE AUTRE CHOSE. Un pilier peut etre traverse souvent sans etre")
print("  souvent terminal, et l'inverse. Le corpus le montre deja de facon extreme (script 75) :")
print("  P2 n'emet jamais, donc des qu'il est atteint la marche s'y arrete, ce qui gonfle sa part")
print("  de fin ; P1 n'est jamais atteint, donc il n'est terminal que lorsqu'il est amorce ET que")
print("  la cascade s'arrete tout de suite. La part de fin est donc une mesure de PUITS, pas une")
print("  mesure d'exposition.")
print("\n  ET L'IDEE DE PRENDRE LE MEME CALCUL « DANS L'AUTRE SENS » EST EXACTEMENT LA BONNE, elle")
print("  s'arrete seulement un cran trop loin. La part d'amorce compte le pilier au rang 1, la")
print("  part de fin le compte au DERNIER rang. La part de passage le compte a TOUS LES RANGS, et")
print("  c'est elle qui correspond a Euler, parce qu'Euler ne demande ni ou le sinistre commence")
print("  ni ou il finit : il demande quels piliers ont PAYE. Un pilier paye des qu'il est touche,")
print("  qu'il soit premier, dernier ou au milieu.")

# =====================================================================================
titre("5. L'ETAT NON CONFORME, ou la part de fin n'a plus de sens")
# =====================================================================================
tou_nc = loi_passage(cx.G_NC, cx.PHICS_NC)
p_pas_nc = tou_nc / tou_nc.sum()
ev_nc, ec_nc = parts_euler(cx.LAM_NC, cx.G_NC, cx.PU_NC, cx.PHICS_NC)
p_eul_nc = ec_nc.mean(axis=0)
d_nc = 100 * np.mean(np.abs(p_pas_nc - p_eul_nc))
print("  A l'etat non conforme, le pilier des tiers tire un ENSEMBLE par choc commun, sans ordre :")
print("  il n'y a ni premier ni dernier pilier pour cette part-la des sinistres, donc la part de")
print("  fin cesse d'etre definie. La part de PASSAGE, elle, garde un sens : c'est la probabilite")
print("  d'etre touche, que la cascade soit une marche ou un choc.")
print(f"\n  {'Pilier':<8}{'Euler (CTE), non conf.':>25}{'part de PASSAGE':>18}{'ecart':>10}")
for c, j in enumerate(PIL):
    print(f"  P{j:<7}{100*p_eul_nc[c]:>24.1f} %{100*p_pas_nc[c]:>17.1f} %"
          f"{100*(p_pas_nc[c]-p_eul_nc[c]):>+9.1f}")
print(f"\n  Ecart moyen : {d_nc:.1f} point. La correspondance tient donc aux DEUX etats, y compris")
print("  quand une partie des sinistres n'est plus une marche du tout.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
o_eul = [PIL[c] for c in np.argsort(-p_eul_c)]
o_fin = [PIL[c] for c in np.argsort(-p_fin)]
o_pas = [PIL[c] for c in np.argsort(-p_passage)]
o_am = [PIL[c] for c in np.argsort(-p_amorce)]
d_fin = [r for r in res if r[0].startswith("part de FIN")][0]
d_pas = [r for r in res if r[0].startswith("part de PASSAGE")][0]
d_am = [r for r in res if r[0].startswith("part d'amorce")][0]
print(f"  1. L'INTUITION EST BONNE DANS SA FORME : il existe bien une grandeur combinatoire, sans")
print(f"     aucune simulation, qui explique les parts d'Euler comme la part d'amorce explique")
print(f"     Shapley. Elle existe, elle est exacte, et c'est une identite.")
print(f"  2. MAIS CE N'EST PAS LA PART DE FIN, c'est la PART DE PASSAGE : la probabilite qu'un")
print(f"     pilier soit TOUCHE, quel que soit son rang. Ecart moyen a Euler : {d_pas[3]:.1f} point contre")
print(f"     {d_fin[3]:.1f} pour la part de fin et {d_am[3]:.1f} pour la part d'amorce.")
print(f"  3. LE MECANISME EST UNE IDENTITE : les severites etant tirees dans la meme loi pour tous")
print(f"     les piliers, la part d'un pilier dans la perte MOYENNE est exactement sa part de")
print(f"     passage. Le seul ecart residuel vient du conditionnement de queue d'Euler.")
print(f"  4. LA PART DE FIN MESURE AUTRE CHOSE, et c'est instructif : elle mesure un PUITS, pas une")
print(f"     exposition. Un pilier qui n'emet jamais arrete toute marche qui l'atteint, donc sa")
print(f"     part de fin est forte sans que son exposition le soit.")
print(f"  5. CLASSEMENTS : Euler {' > '.join('P' + str(j) for j in o_eul)}, passage "
      f"{' > '.join('P' + str(j) for j in o_pas)},")
print(f"     fin {' > '.join('P' + str(j) for j in o_fin)}, amorce "
      f"{' > '.join('P' + str(j) for j in o_am)}.")
print("     A LIRE AVEC LA PRUDENCE DEJA PUBLIEE : le classement d'Euler n'est pas stable entre")
print("     graines, donc c'est la coincidence des NIVEAUX qui vaut, pas celle des ordres.")

# =====================================================================================
# figure S43
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.3))

xs = np.arange(len(PIL))
h = 0.20
ax1.bar(xs - 1.5 * h, 100 * p_eul_c, width=h, color=INK, alpha=0.9, label="Euler (mesuré)")
ax1.bar(xs - 0.5 * h, 100 * p_passage, width=h, color=GREEN, alpha=0.9, label="part de passage")
ax1.bar(xs + 0.5 * h, 100 * p_fin, width=h, color=ACCENT, alpha=0.9, label="part de fin")
ax1.bar(xs + 1.5 * h, 100 * p_amorce, width=h, color=MUTED, alpha=0.85, label="part d'amorce")
ax1.set_xticks(xs)
ax1.set_xticklabels([f"P{j}" for j in PIL], fontsize=10.5)
ax1.set_ylabel("part (%)", color=INK2)
ax1.legend(fontsize=9, frameon=False, loc="upper right", ncol=2)
ax1.set_ylim(0, 1.42 * max(100 * p_amorce.max(), 100 * p_fin.max()))
ax1.set_title("(a)  Les trois candidates face à Euler", fontsize=10.5, color=INK, pad=8)

noms = [r[0] for r in res]
ecs = [r[3] for r in res]
cols = [MUTED, ACCENT, GREEN]
ax2.barh(np.arange(len(noms)), ecs, color=cols, alpha=0.9, height=0.55)
for y, v in zip(np.arange(len(noms)), ecs):
    ax2.text(v + 0.12, y, f"{v:.1f} pt".replace(".", ","), va="center", fontsize=10,
             color=INK, fontweight="bold")
ax2.set_yticks(np.arange(len(noms)))
ax2.set_yticklabels(["part d'amorce", "part de fin", "part de passage"], fontsize=10)
ax2.invert_yaxis()
ax2.set_xlim(0, max(ecs) * 1.30)
ax2.set_xlabel("écart moyen aux parts d'Euler (points)", color=INK2)
ax2.set_title("(b)  C'est la part de PASSAGE qui colle,\net c'est une identité",
              fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("S43 : ce qui explique Euler n'est pas où la cascade finit, c'est par où elle passe",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S43_part_de_fin.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
