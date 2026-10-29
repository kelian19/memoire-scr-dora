#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
97 : les trois figures de la lecture de marche, REDESSINEES depuis la transcription.

POURQUOI CE SCRIPT EXISTE. Les trois figures de marche de l'introduction etaient des images
extraites du rapport LUCY 2026 de Nexialog Consulting, seules figures du memoire non produites
par un script. Elles posaient trois problemes : aucune n'etait rejouable, elles portaient la
charte de graphique du rapport et non celle du memoire, et reproduire des figures entieres d'un
rapport d'entreprise dans un document mis en ligne par l'Institut demandait une autorisation.
Les series, elles, sont transcrites depuis le rapport dans config.py sous LUCY_2026, sous le
statut de CITATION EXTERNE NON RECALCULABLE, et le script 63 en verifie la fidelite par quatre
controles d'identite. Ce script les redessine donc a partir de la meme transcription.

CE QUE CELA NE CHANGE PAS. Le statut de la source. Les valeurs restent celles que le rapport
publie ; aucune n'est recalculee depuis une donnee de marche, que le depot ne detient pas, et
aucune n'entre dans une calibration du modele. Ce script ne produit que des traces.

CE QUE CELA CHANGE. Les figures deviennent rejouables et tracables, elles passent a la charte du
memoire, et la question d'autorisation tombe : ce ne sont plus des reproductions mais des traces
des valeurs citees, comme les tableaux que le memoire construit deja depuis la meme source.

TROIS CONTROLES, repris du script 63 pour que ce script ne puisse pas tracer autre chose que ce
que le memoire publie : le rapport des sinistres aux primes doit redonner la serie des ratios, la
somme des quatre classes de taille doit redonner la charge annuelle, et l'indice tarifaire des
entreprises de taille intermediaire doit culminer ou le rapport le dit.

Sorties : trois figures L1, L2 et L3, plus les grandeurs citees.
"""

import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import style_nexialog as sn                            # noqa: E402
from src.utils.config import LUCY_2026 as L            # noqa: E402

WID = 82
sn.appliquer()

ANS = np.array(L["annees_serie"])
PRIMES = np.array(L["primes_serie_eur"], dtype=float)
SINIS = np.array(L["sinistres_serie_eur"], dtype=float)
SP = np.array(L["sp_serie"], dtype=float)
ANS_SEG = np.array(L["annees_segment"])
SEG = {"grandes entreprises": np.array(L["sp_serie_grandes"], dtype=float),
       "entreprises de taille intermediaire": np.array(L["sp_serie_eti"], dtype=float),
       "entreprises moyennes": np.array(L["sp_serie_moyennes"], dtype=float)}
INDICE = np.array(L["indice_taux_prime_eti"], dtype=float)
TAILLE = {k: np.array(v, dtype=float) for k, v in L["taille_sinistre_eur"].items()}

print("=" * WID)
print("97 : figures de la lecture de marche, redessinees depuis la transcription LUCY 2026")
print("=" * WID)
print("  source      :", L["source"])
print("  analyse     :", L["analyse"])
print("  statut      : citation externe non recalculable ; ce script ne fait que tracer")

# --------------------------------------------------------------------------------------------
# 1. LES TROIS CONTROLES D'IDENTITE, avant tout trace
# --------------------------------------------------------------------------------------------
print("\n" + "=" * WID)
print("1. Controles d'identite sur la transcription")
print("=" * WID)

ecart_sp = np.abs(SINIS / PRIMES - SP)
print(f"  ecart maximal entre sinistres/primes et la serie des ratios : {ecart_sp.max():.4f}")
assert ecart_sp.max() < 0.02, "la serie des ratios ne se retrouve pas depuis primes et sinistres"

somme_classes = sum(TAILLE.values())
ecart_classes = np.abs(somme_classes - SINIS)
print(f"  ecart maximal entre la somme des quatre classes et la charge annuelle : "
      f"{ecart_classes.max():.0f} M EUR")
assert ecart_classes.max() <= 3.0, "les classes de taille ne somment pas a la charge annuelle"

i_max = int(np.argmax(INDICE))
print(f"  sommet de l'indice tarifaire des entreprises de taille intermediaire : "
      f"exercice {ANS_SEG[i_max]}, indice {INDICE[i_max]:.0f}")
assert ANS_SEG[i_max] == 2023, "le sommet tarifaire n'est pas a l'exercice que le rapport donne"
print(f"  indice de l'exercice sous revue : {INDICE[-1]:.0f}")
print(f"  ratio des entreprises de taille intermediaire : {SEG['entreprises de taille intermediaire'][-2]:.0%} "
      f"en {ANS_SEG[-2]} puis {SEG['entreprises de taille intermediaire'][-1]:.0%} en {ANS_SEG[-1]}")

# --------------------------------------------------------------------------------------------
# 2. FIGURE L1 : primes, sinistres et ratio
# --------------------------------------------------------------------------------------------
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)

fig, ax = plt.subplots(figsize=(9.2, 3.6))
larg = 0.38
x = np.arange(len(ANS))
b1 = ax.bar(x - larg / 2, PRIMES, larg, label="primes souscrites", color=sn.CATEGORIEL[2])
b2 = ax.bar(x + larg / 2, SINIS, larg, label="sinistres indemnises", color=sn.CATEGORIEL[0])
ax.set_ylabel("primes et sinistres (M€)")
ax.set_xticks(x)
ax.set_xticklabels(ANS)
ax.set_ylim(0, max(PRIMES.max(), SINIS.max()) * 1.25)
for b, v in list(zip(b1, PRIMES)) + list(zip(b2, SINIS)):
    ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=7, color=sn.ENCRE_2)

ax2 = ax.twinx()
ax2.plot(x, SP * 100, color=sn.ENCRE, marker="s", markersize=4, linewidth=1.4,
         label="ratio sinistres sur primes")
for xi, v in zip(x, SP * 100):
    ax2.annotate(f"{v:.0f} %", (xi, v), textcoords="offset points", xytext=(0, 7),
                 ha="center", fontsize=7, color=sn.ENCRE)
ax2.set_ylabel("ratio sinistres sur primes (%)")
ax2.set_ylim(0, SP.max() * 100 * 1.25)
ax2.spines["top"].set_visible(False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=8, ncol=3)
fig.tight_layout()
p1 = os.path.join(outdir, "L1_lucy_primes_sinistres_ratio.png")
fig.savefig(p1, dpi=200)
plt.close(fig)
print("\nfigure ecrite :", p1)

# --------------------------------------------------------------------------------------------
# 3. FIGURE L2 : la divergence entre prix et risque par segment
#    Les trois segments sont ORDONNES par taille : rampe a une seule teinte, pas trois
#    couleurs categorielles.
# --------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 3.6))
teintes = sn.rampe(3, sn.SEQUENTIEL_6)
larg = 0.26
x = np.arange(len(ANS_SEG))
for k, (nom, serie) in enumerate(SEG.items()):
    dec = (k - 1) * larg
    b = ax.bar(x + dec, serie * 100, larg, label=nom, color=teintes[k])
    for bi, v in zip(b, serie * 100):
        ax.annotate(f"{v:.0f}", (bi.get_x() + bi.get_width() / 2, bi.get_height()),
                    ha="center", va="bottom", fontsize=6.5, color=sn.ENCRE_2)
ax.set_ylabel("ratio sinistres sur primes (%)")
ax.set_xticks(x)
ax.set_xticklabels(ANS_SEG)
ax.set_ylim(0, max(s.max() for s in SEG.values()) * 100 * 1.2)

ax2 = ax.twinx()
ax2.plot(x, INDICE, color=sn.ENCRE, marker="o", markersize=4, linewidth=1.4,
         label="indice du taux de prime, entreprises de taille intermediaire")
for xi, v in zip(x, INDICE):
    ax2.annotate(f"{v:.0f}", (xi, v), textcoords="offset points", xytext=(0, 7),
                 ha="center", fontsize=7, color=sn.ENCRE)
ax2.set_ylabel("indice du taux de prime (base 100 en 2020)")
ax2.set_ylim(0, INDICE.max() * 1.25)
ax2.spines["top"].set_visible(False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=7.5)
fig.tight_layout()
p2 = os.path.join(outdir, "L2_lucy_divergence_prix_risque.png")
fig.savefig(p2, dpi=200)
plt.close(fig)
print("figure ecrite :", p2)

# --------------------------------------------------------------------------------------------
# 4. FIGURE L3 : le montant indemnise par classe de taille de sinistre
#    Les quatre classes sont ordonnees : rampe a une seule teinte.
# --------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 3.6))
teintes = sn.rampe(4, sn.SEQUENTIEL_6)
larg = 0.2
x = np.arange(len(ANS))
for k, (nom, serie) in enumerate(TAILLE.items()):
    dec = (k - 1.5) * larg
    b = ax.bar(x + dec, serie, larg, label=nom.replace(" a ", " à ").replace("EUR", "€"),
               color=teintes[k])
    for bi, v in zip(b, serie):
        if v > 0:
            ax.annotate(f"{v:.0f}", (bi.get_x() + bi.get_width() / 2, bi.get_height()),
                        ha="center", va="bottom", fontsize=6.5, color=sn.ENCRE_2)
ax.set_ylabel("montant indemnise (M€)")
ax.set_xticks(x)
ax.set_xticklabels(ANS)
ax.set_ylim(0, max(s.max() for s in TAILLE.values()) * 1.2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=7.5, ncol=2)
fig.tight_layout()
p3 = os.path.join(outdir, "L3_lucy_taille_sinistres.png")
fig.savefig(p3, dpi=200)
plt.close(fig)
print("figure ecrite :", p3)

# --------------------------------------------------------------------------------------------
# 5. LECTURE, ce que les trois traces montrent
# --------------------------------------------------------------------------------------------
print("\n" + "=" * WID)
print("2. Ce que les trois figures portent")
print("=" * WID)
print(f"  L1 : le ratio remonte de {SP[-3]:.0%} en {ANS[-3]} a {SP[-2]:.0%} puis {SP[-1]:.0%},")
print(f"       pour un maximum de serie de {SP.max():.0%} a l'exercice {ANS[int(np.argmax(SP))]}.")
print(f"  L2 : le ratio des entreprises de taille intermediaire passe de "
      f"{SEG['entreprises de taille intermediaire'][-2]:.0%} a "
      f"{SEG['entreprises de taille intermediaire'][-1]:.0%} quand l'indice tarifaire recule de "
      f"{INDICE[i_max]:.0f} a {INDICE[-1]:.0f}.")
xxl = TAILLE["XXL, 10 a 40 M EUR"]
print(f"  L3 : {int((xxl == 0).sum())} exercices sur {len(ANS)} sans aucun sinistre de la classe "
      f"la plus haute ; maximum {xxl.max():.0f} M EUR, exercice sous revue {xxl[-1]:.0f} M EUR.")

print("\n" + "=" * WID)
print("grandeurs citees")
print("=" * WID)
print(f"  ratio 2023 {SP[-3]*100:.0f}")
print(f"  ratio 2024 {SP[-2]*100:.0f}")
print(f"  ratio 2025 {SP[-1]*100:.0f}")
print(f"  ratio maximal de la serie {SP.max()*100:.0f}")
print(f"  ratio 2019 {SP[0]*100:.0f}")
print(f"  ratio ETI 2024 {SEG['entreprises de taille intermediaire'][-2]*100:.0f}")
print(f"  ratio ETI 2025 {SEG['entreprises de taille intermediaire'][-1]*100:.0f}")
print(f"  ratio ETI maximal {max(SEG['entreprises de taille intermediaire'])*100:.0f}")
print(f"  indice tarifaire ETI sommet {INDICE[i_max]:.0f}")
print(f"  indice tarifaire ETI 2025 {INDICE[-1]:.0f}")
print(f"  classe XXL maximum {xxl.max():.0f}")
print(f"  classe XXL 2025 {xxl[-1]:.0f}")
print(f"  exercices sans classe XXL {int((xxl == 0).sum())}")
print(f"  ecart maximal du controle des ratios {ecart_sp.max():.4f}")
print(f"  ecart maximal du controle des classes {ecart_classes.max():.0f}")
