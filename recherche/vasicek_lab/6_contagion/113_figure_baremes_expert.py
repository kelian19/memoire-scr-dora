#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
113 : figure Z_baremes_expert, les trois baremes d'expert du modele qualitatif de cascade.

LE NOM DE FICHIER NE PORTE AUCUN CHIFFRE : le harnais lit les nombres du nom passe a
\figover (motif dans l'en-tete du script 111).

Les trois baremes sont lus dans exploratory/cascade_qualitative/cascade_model.py, source
unique du jugement d'expert, et jamais recopies :
  ROOT[p]     propension du pilier p a amorcer une chaine ;
  TRANS[a][b] force dirigee « a entraine b », asymetrique, non normalisee ;
  GBASE[p]    gravite intrinseque du pilier p s'il tombe seul (echelle ordinale).
La figure reprend le contenu de la figure F7 de la fiche pedagogique du modele qualitatif,
sans son bandeau de titre (qui renvoyait aux figures F8 a F11, absentes du rapport) et aux
couleurs de la charte (style_nexialog). La sortie standard est deterministe et versionnee
(113.txt) : elle imprime les valeurs, les parts d'amorce et les sommes de ligne et de colonne
de TRANS que le chapitre de la cascade cite.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RACINE = os.path.dirname(os.path.dirname(HERE))
QUALI = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cascade_qualitative'))
for p in (HERE, QUALI):
    if p not in sys.path:
        sys.path.insert(0, p)
import style_nexialog as sn                                     # noqa: E402
from cascade_model import ROOT, TRANS, GBASE                    # noqa: E402

PIL = sorted(ROOT)
WID = 88


def titre(t):
    print()
    print("=" * WID)
    print(t)
    print("=" * WID)


# ---------------------------------------------------------------- controles
titre("113 - LES TROIS BAREMES D'EXPERT DU MODELE QUALITATIF : ROOT, TRANS, GBASE")
print("  source : exploratory/cascade_qualitative/cascade_model.py (source unique)")
assert sorted(GBASE) == PIL and sorted(TRANS) == PIL, "piliers incoherents entre baremes"
for a in PIL:
    assert sorted(TRANS[a]) == [b for b in PIL if b != a], f"ligne TRANS[{a}] incomplete"
asym = sum(abs(TRANS[a][b] - TRANS[b][a]) for a in PIL for b in PIL if a < b)
assert asym > 0, "TRANS symetrique : l'ordre ne compterait plus"
print(f"  controles : cinq piliers dans les trois baremes, vingt entrees TRANS, "
      f"asymetrie totale {asym:.2f} (> 0)")

titre("1. ROOT : propension a amorcer une chaine, et part d'amorce dans le moteur")
s_root = sum(ROOT.values())
parts = {p: 100 * ROOT[p] / s_root for p in PIL}
for p in sorted(PIL, key=ROOT.get, reverse=True):
    print(f"  P{p} : ROOT = {ROOT[p]:.2f}   part d'amorce = {parts[p]:.1f} %   (arrondie {parts[p]:.0f} %)")
print(f"  somme des parts : {sum(parts.values()):.1f} %")
print("  classement ROOT : " + " > ".join(f"P{p}" for p in sorted(PIL, key=ROOT.get, reverse=True)))

titre("2. TRANS : force dirigee « a entraine b », ligne = source, colonne = cible")
print("        " + "".join(f"{'P' + str(b):>7}" for b in PIL) + "   emission (somme de ligne)")
emis = {a: sum(TRANS[a].values()) for a in PIL}
rec = {b: sum(TRANS[a][b] for a in PIL if a != b) for b in PIL}
for a in PIL:
    cells = "".join(f"{TRANS[a][b]:>7.2f}" if b != a else f"{'-':>7}" for b in PIL)
    print(f"  P{a}  {cells}   {emis[a]:.2f}")
print("  reception (somme de colonne) : " + "  ".join(f"P{b} {rec[b]:.2f}" for b in PIL))
vals = [TRANS[a][b] for a in PIL for b in PIL if a != b]
print(f"  entrees de {min(vals):.2f} a {max(vals):.2f} ; plus grande emission : "
      f"P{max(emis, key=emis.get)} avec {max(emis.values()):.2f}")
print(f"  exemple d'asymetrie : TRANS[1][2] = {TRANS[1][2]:.2f} contre TRANS[2][1] = {TRANS[2][1]:.2f}")

titre("3. GBASE : gravite intrinseque du pilier s'il tombe seul (ordinale)")
for p in sorted(PIL, key=GBASE.get, reverse=True):
    print(f"  P{p} : GBASE = {GBASE[p]}")
print("  classement GBASE : " + " > ".join(f"P{p}" for p in sorted(PIL, key=GBASE.get, reverse=True)))

titre("3bis. LES COTES DU MODELE QUALITATIF, calculees sur toutes les chaines ordonnees")
from itertools import combinations, permutations                    # noqa: E402
from cascade_model import proba_score, gravite_score, crit_score    # noqa: E402
CHAINES = [o for k in range(1, len(PIL) + 1) for s in combinations(PIL, k) for o in permutations(s)]
cotes = [(proba_score(o), gravite_score(o)) for o in CHAINES]
crits = [crit_score(p, g) for p, g in cotes]
bornes = (min(min(p, g) for p, g in cotes), max(max(p, g) for p, g in cotes))
print(f"  {len(CHAINES)} chaines ordonnees notees ; probabilite et gravite cotees dans "
      f"[{bornes[0]} ; {bornes[1]}], criticite dans [{min(crits)} ; {max(crits)}]")
print("  echelle de notation des trois cotes : 1 a 10 (bornes posees par proba_score et gravite_score)")
print(f"  exemple : 1->2 criticite {crit_score(proba_score((1, 2)), gravite_score((1, 2)))}, "
      f"2->1 criticite {crit_score(proba_score((2, 1)), gravite_score((2, 1)))}")

# ---------------------------------------------------------------- figure
sn.appliquer(taille=10)
mpl.rcParams["font.family"] = ["DejaVu Sans", "Segoe UI", "sans-serif"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#EEF3FC"] + list(reversed(sn.SEQUENTIEL_6)))
VIRGULE = mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))

fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7),
                         gridspec_kw={"width_ratios": [1, 1, 1.15]})
fig.subplots_adjust(wspace=0.42, left=0.05, right=0.99, top=0.88, bottom=0.14)


def barres(ax, bareme, fmt, xmax, titre_ax):
    ordre = sorted(PIL, key=bareme.get)                 # le plus fort en haut
    rang = {p: i for i, p in enumerate(sorted(PIL, key=bareme.get, reverse=True))}
    ax.barh([f"P{p}" for p in ordre], [bareme[p] for p in ordre],
            color=[sn.ORDINAL_5[rang[p]] for p in ordre], edgecolor=sn.FOND, height=0.72)
    for p in ordre:
        ax.text(bareme[p] - 0.03 * xmax, f"P{p}", fmt(bareme[p]), va="center", ha="right",
                color="white", fontsize=9)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(VIRGULE)
    ax.set_title(titre_ax, fontsize=10.5, loc="left", pad=6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)


barres(axes[0], ROOT, lambda v: f"{v:.2f}".replace(".", ","), 1.05,
       "(a) ROOT : propension à amorcer")
barres(axes[1], GBASE, lambda v: f"{v}", 7.5, "(b) GBASE : gravité intrinsèque")

ax = axes[2]
M = np.full((len(PIL), len(PIL)), np.nan)
for a in PIL:
    for b in PIL:
        if a != b:
            M[a - 1, b - 1] = TRANS[a][b]
ax.imshow(M, cmap=SEQ, vmin=0, vmax=1)
for i in range(len(PIL)):
    for j in range(len(PIL)):
        if np.isnan(M[i, j]):
            ax.text(j, i, "-", ha="center", va="center", color=sn.ENCRE_3)
        else:
            ax.text(j, i, f"{M[i, j]:.1f}".replace(".", ","), ha="center", va="center",
                    fontsize=9, color="white" if M[i, j] >= 0.45 else sn.ENCRE)
ax.set_xticks(range(len(PIL)))
ax.set_yticks(range(len(PIL)))
ax.set_xticklabels([f"P{p}" for p in PIL])
ax.set_yticklabels([f"P{p}" for p in PIL])
ax.set_xlabel("pilier cible", color=sn.ENCRE_2)
ax.set_ylabel("pilier source", color=sn.ENCRE_2)
ax.set_title("(c) TRANS : « la source entraîne la cible »", fontsize=10.5, loc="left", pad=6)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

sortie = os.path.join(HERE, "figures", "Z_baremes_expert.png")
fig.savefig(sortie, dpi=200)
plt.close(fig)
print()
print("  figure ecrite : exploratory/vasicek_lab/figures/Z_baremes_expert.png")

titre("4. GRANDEURS CITEES PAR LE MEMOIRE (sans separateur)")
print("  ROOT P1 P4 P2 P3 P5 : " + " ".join(f"{ROOT[p]:.2f}" for p in (1, 4, 2, 3, 5)))
print("  parts d'amorce P1 P4 P2 P3 P5 (%) : " + " ".join(f"{parts[p]:.0f}" for p in (1, 4, 2, 3, 5)))
print("  GBASE P4 P1 P2 P3 P5 : " + " ".join(f"{GBASE[p]}" for p in (4, 1, 2, 3, 5)))
print(f"  TRANS min max : {min(vals):.2f} {max(vals):.2f}")
print(f"  emission P1 : {emis[1]:.2f} ; reception P2 : {rec[2]:.2f}")
print()
print("=" * WID)
print("FIN 113")
print("=" * WID)
