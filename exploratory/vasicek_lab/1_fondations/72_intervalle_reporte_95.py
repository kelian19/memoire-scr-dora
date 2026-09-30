#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
72 : l'intervalle reporte a 95 %, ce qu'il change et ce qu'il ne corrige pas.

DECISION A L'ORIGINE (Caroline Hillairet, compte rendu du 13 aout 2026, section 7) :
« Intervalle de confiance retenu pour le SCR : 95 % ». Le memoire publie aujourd'hui ses
intervalles a 90 %. Ce script produit le passage.

POURQUOI AUCUN BOOTSTRAP N'EST RELANCE, ET C'EST LE POINT DE METHODE. Rejouer un bootstrap
donnerait un nouvel echantillon, donc un IC90 legerement different du publie : on se retrouverait
avec deux valeurs du meme nombre, exactement le defaut que le projet a corrige sur la VaR
predictive. On passe donc de 90 a 95 % par la MEME approximation normale que celle que le moteur
emploie deja pour propager l'incertitude de severite (euro_cascade_model.tire_severite tire
(xi, sigma) dans leur IC90 sous approximation normale). L'ecart-type implicite se lit sur la
demi-largeur publiee, et le niveau se change en remplacant 1,645 par 1,96. La calibration ne
bouge pas d'un chiffre : le point, sigma, u et p_u sont inchanges, donc tous les SCR publies
aussi. Seul le NIVEAU DE L'INTERVALLE change.

CE QUE LE SCRIPT ETABLIT, ET LE SECOND POINT EST LE PLUS IMPORTANT :
  1. les intervalles a 95 % sur xi, sur le quantile de severite et sur le capital agrege ;
  2. la COUVERTURE REELLE des deux niveaux nominaux, mesuree a n fini. Le memoire publie deja
     que l'IC a 90 % n'en couvre que 86 : passer le nominal a 95 % NE CORRIGE PAS cette
     sous-couverture, il deplace le nominal sans reparer l'estimateur. Annoncer 95 % sans le
     dire serait remplacer un defaut declare par un defaut cache.

Sortie : diagnostics + figure S28_intervalle_95.png.
"""

import os
import sys

import numpy as np
from scipy.stats import genpareto, norm
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                                 # noqa: E402
from euro_cascade_model import PARAMS, var                      # noqa: E402

WID = 88
sp = PARAMS["OPRISK"]
XI = sp["xi"]
SIG = sp["sigma"]
U = sp["u"]
P_U = sp["p_u"]
N_U = 91                          # nombre d'exces de la calibration figee

Z90, Z95 = 1.645, 1.96            # quantiles normaux des deux niveaux nominaux
ALPHA = 0.995

# Intervalles PUBLIES a 90 %, repris de la configuration figee, non recalcules.
XI_IC90 = tuple(sp["xi_ic90"])
VAR_UNIT = 662.78                 # quantile de severite publie
VAR_IC90 = (411.5, 1037.1)        # son IC90 publie

M_COV = 2000                      # simulations de couverture, comme le script 47
NY = 60_000
NSEED = 4
# LA MEME GRAINE QUE LE SCRIPT 71, ET C'EST DELIBERE. Les deux scripts evaluent le capital
# agrege le long du meme axe (xi a calibration figee) : avec deux graines differentes ils
# publieraient deux valeurs du meme nombre, a 24 M pres, ce qui est le defaut que le projet a
# corrige sur la VaR predictive. Une seule graine, donc des bornes identiques d'un script a
# l'autre.
SEED0 = 20260812
GAIN = 0.90


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def a_95(point, ic90):
    """Passe un intervalle de 90 a 95 %, par la meme approximation normale que le moteur.

    Les deux demi-largeurs sont mises a l'echelle SEPAREMENT : l'intervalle publie est
    asymetrique (la borne basse est plus eloignee du point que la borne haute), et le
    symetriser au passage serait une modification silencieuse de l'objet publie.
    """
    bas, haut = point - ic90[0], ic90[1] - point
    r = Z95 / Z90
    return point - r * bas, point + r * haut


def scr_xi(xi, nseed=NSEED, ny=NY):
    """SCR agrege (VaR 99,5 %) a xi donne, tout le reste a la calibration publiee."""
    out = []
    for k in range(nseed):
        rng = np.random.default_rng(SEED0 + k)
        out.append(var(ec.simulate_euro(sp["lam_ref"], GAIN, xi, SIG, U, P_U,
                                        sp["cap"], ny, rng)))
    return float(np.mean(out))


# =====================================================================================
titre("1. Le passage de 90 a 95 %, sans toucher a la calibration")
# =====================================================================================
xi95 = a_95(XI, XI_IC90)
var95 = a_95(VAR_UNIT, VAR_IC90)
print(f"  Facteur de passage sur les demi-largeurs : {Z95:.2f} / {Z90:.3f} = {Z95/Z90:.4f}")
# LES DEUX FACTEURS SONT IMPRIMES, un par niveau. Une premiere version n'imprimait celui de 95 %
# que pour le quantile de severite, si bien que le facteur 3,52 de xi devait etre recalcule a la
# main dans la redaction : exactement ce que le projet s'interdit.
print(f"\n  {'grandeur':<30}{'IC 90 % (publie)':>24}{'IC 95 %':>24}{'x a 90':>9}{'x a 95':>9}")
print(f"  {'indice de queue xi':<30}"
      f"{f'[{XI_IC90[0]:.4f} ; {XI_IC90[1]:.4f}]':>24}"
      f"{f'[{xi95[0]:.4f} ; {xi95[1]:.4f}]':>24}"
      f"{XI_IC90[1]/XI_IC90[0]:>9.2f}{xi95[1]/xi95[0]:>9.2f}")
print(f"  {'quantile de severite (M)':<30}"
      f"{f'[{VAR_IC90[0]:.1f} ; {VAR_IC90[1]:.1f}]':>24}"
      f"{f'[{var95[0]:.1f} ; {var95[1]:.1f}]':>24}"
      f"{VAR_IC90[1]/VAR_IC90[0]:>9.2f}{var95[1]/var95[0]:>9.2f}")
print(f"\n  L'ASYMETRIE DE L'INTERVALLE PUBLIE EST CONSERVEE : la borne basse de xi est a "
      f"{XI - XI_IC90[0]:.4f}")
print(f"  du point quand la haute est a {XI_IC90[1] - XI:.4f}. Les deux demi-largeurs sont mises")
print("  a l'echelle separement ; symetriser au passage aurait modifie l'objet publie en silence.")
print(f"\n  LE FACTEUR PUBLIE SUR LE QUANTILE UNITAIRE PASSE DE {VAR_IC90[1]/VAR_IC90[0]:.2f} A "
      f"{var95[1]/var95[0]:.2f}.")
print("  C'est le nombre le plus visible du changement, et il apparait a plusieurs endroits du")
print("  memoire : le remplacer partout est une decision de redaction distincte de ce calcul.")

# =====================================================================================
titre("2. La couverture REELLE des deux niveaux, et elle ne suit pas le nominal")
# =====================================================================================
# MEME PROTOCOLE QUE LE SCRIPT 47 : on simule N_U exces sous la calibration publiee, on
# reajuste, on forme l'intervalle avec l'ecart-type asymptotique, et l'on compte les fois ou la
# vraie valeur y tombe. La seule difference est qu'on le fait aux DEUX niveaux nominaux.
rng = np.random.default_rng(20260727)
dedans90 = dedans95 = 0
for _ in range(M_COV):
    ech = genpareto.rvs(XI, scale=SIG, size=N_U, random_state=rng)
    c, _, s = genpareto.fit(ech, floc=0)
    sd = (1.0 + c) / np.sqrt(N_U)
    if abs(XI - c) <= Z90 * sd:
        dedans90 += 1
    if abs(XI - c) <= Z95 * sd:
        dedans95 += 1
cov90, cov95 = 100 * dedans90 / M_COV, 100 * dedans95 / M_COV
# TOUTE GRANDEUR SIMULEE SE PUBLIE AVEC SON BRUIT, c'est la convention du projet. Sans elle on
# ne peut pas dire si deux mesures d'une meme couverture sont d'accord.
se90 = 100 * np.sqrt(cov90 / 100 * (1 - cov90 / 100) / M_COV)
se95 = 100 * np.sqrt(cov95 / 100 * (1 - cov95 / 100) / M_COV)
print(f"  {M_COV} echantillons de {N_U} exces simules sous la calibration publiee.")
print(f"\n  {'niveau nominal':<20}{'couverture reelle':>22}{'manque':>12}")
print(f"  {'90 %':<20}{f'{cov90:.1f} % +- {se90:.1f}':>22}{cov90 - 90:>11.1f}")
print(f"  {'95 %':<20}{f'{cov95:.1f} % +- {se95:.1f}':>22}{cov95 - 95:>11.1f}")
print(f"\n  ACCORD AVEC LE SCRIPT 47, QUI PUBLIE 86 % POUR LE NIVEAU DE 90. Les deux mesures")
print(f"  sortent du meme estimateur sur deux flux de simulation independants, et leur ecart de")
print(f"  {abs(cov90 - 86.0):.1f} point vaut environ un ecart-type ({se90:.1f}) : c'est le meme nombre, "
      "mesure deux fois.")
print("  Il ne faut donc pas lire une contradiction entre les deux scripts, et il ne faut pas")
print("  non plus citer trois chiffres significatifs sur une couverture estimee sur deux mille")
print("  tirages.")
print(f"\n  LE MANQUE EST CONSTANT EN POINTS, ET C'EST INSTRUCTIF : {cov90 - 90:.1f} au niveau de 90 "
      f"comme {cov95 - 95:.1f}")
print("  au niveau de 95. Ce n'est donc pas un defaut proportionnel qui s'atenuerait en montant")
print("  le nominal : c'est un deficit a peu pres fixe, signature d'un ecart-type asymptotique")
print("  trop petit plutot que d'une forme d'intervalle mal choisie.")
print("\n  PASSER LE NOMINAL A 95 % NE CORRIGE PAS LA SOUS-COUVERTURE, il la deplace. Les deux")
print("  niveaux couvrent moins que ce qu'ils annoncent, et pour la meme raison : l'ecart-type")
print("  asymptotique sous-estime la dispersion de l'estimateur a quatre-vingt-onze exces.")
print("  Annoncer 95 % sans le dire remplacerait donc un defaut DECLARE par un defaut CACHE,")
print("  ce qui serait un mauvais echange. Le niveau se change, la reserve se garde.")
manque95 = 95.0 - cov95
print(f"\n  Pour atteindre reellement 95 %, il faudrait elargir la demi-largeur d'environ")
_fac = Z95 / norm.ppf(0.5 + cov95 / 200.0) if cov95 < 100 else 1.0
print(f"  un facteur {_fac:.2f} de plus, sous approximation normale. C'est le meme correctif que")
print("  celui deja publie pour le niveau de 90 %, et il va dans le meme sens.")

# =====================================================================================
titre("3. Ce que le nouveau niveau donne sur le capital agrege")
# =====================================================================================
s_pub = scr_xi(XI)
s_lo90, s_hi90 = scr_xi(XI_IC90[0]), scr_xi(XI_IC90[1])
s_lo95, s_hi95 = scr_xi(xi95[0]), scr_xi(xi95[1])
print(f"  Capital a la calibration publiee : {s_pub:.0f} M")
print(f"\n  {'niveau':<12}{'borne basse':>14}{'borne haute':>14}{'largeur':>11}{'facteur':>10}")
print(f"  {'90 %':<12}{s_lo90:>14.0f}{s_hi90:>14.0f}{s_hi90-s_lo90:>11.0f}{s_hi90/s_lo90:>10.2f}")
print(f"  {'95 %':<12}{s_lo95:>14.0f}{s_hi95:>14.0f}{s_hi95-s_lo95:>11.0f}{s_hi95/s_lo95:>10.2f}")
print(f"\n  Le passage a 95 % elargit la bande de capital de {(s_hi95-s_lo95)-(s_hi90-s_lo90):.0f} M, "
      f"soit {100*((s_hi95-s_lo95)/(s_hi90-s_lo90)-1):.0f} %.")
print("  L'elargissement est TRES ASYMETRIQUE, et la raison est la convexite du capital en xi :")
print(f"  la borne basse ne descend que de {s_lo90-s_lo95:.0f} M quand la haute monte de "
      f"{s_hi95-s_hi90:.0f} M.")
print("  C'est une propriete a annoncer : sur une queue lourde, elargir un intervalle de")
print("  confiance sur l'indice de queue ne l'elargit pas symetriquement sur le capital.")

# =====================================================================================
titre("4. Ce que ce changement coute, et ce qu'il n'est pas")
# =====================================================================================
print("  N'EST PAS UNE RECALIBRATION. Le point, sigma, u et p_u sont inchanges, donc tous les")
print("  capitaux publies aussi. Seul le niveau de l'intervalle reporte change, et il change par")
print("  la meme approximation normale que le moteur emploie deja.")
print("\n  COUTE une reecriture. Le facteur du quantile unitaire, cite a plusieurs endroits du")
print(f"  memoire, passe de {VAR_IC90[1]/VAR_IC90[0]:.2f} a {var95[1]/var95[0]:.2f}, et l'intervalle "
      f"sur xi de")
print(f"  [{XI_IC90[0]:.2f} ; {XI_IC90[1]:.2f}] a [{xi95[0]:.2f} ; {xi95[1]:.2f}]. Remplacer ces "
      "valeurs partout est une decision")
print("  de redaction, distincte de ce calcul : elle deplace des nombres publies, et la piste")
print("  d'audit demande de dire lesquels et pourquoi.")
print("\n  ET UNE REMARQUE QUI VA CONTRE L'INTUITION DU CHANGEMENT. Elargir l'intervalle rend le")
print("  resultat MOINS precis en apparence, alors que rien n'a ete perdu : c'est la meme")
print("  incertitude, annoncee a un niveau plus exigeant. La position de Caroline sur ce point")
print("  est explicite dans le compte rendu, un intervalle large n'est pas anormal en soi et ne")
print("  devient problematique que s'il est disproportionne.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print(f"  1. xi : [{XI_IC90[0]:.4f} ; {XI_IC90[1]:.4f}] a 90 % devient "
      f"[{xi95[0]:.4f} ; {xi95[1]:.4f}] a 95 %.")
print(f"  2. Quantile de severite : facteur {VAR_IC90[1]/VAR_IC90[0]:.2f} -> "
      f"{var95[1]/var95[0]:.2f}. C'est le nombre le plus visible.")
print(f"  3. Capital agrege : facteur {s_hi90/s_lo90:.2f} -> {s_hi95/s_lo95:.2f}, et")
print("     l'elargissement est asymetrique par convexite en xi.")
print(f"  4. COUVERTURE REELLE : {cov90:.1f} % pour un nominal de 90, {cov95:.1f} % pour un")
print("     nominal de 95. Changer le nominal NE CORRIGE PAS la sous-couverture : la reserve")
print("     deja publiee reste due, au nouveau niveau comme a l'ancien.")
print("  5. Aucune calibration n'est touchee. Le remplacement des valeurs dans le corps du")
print("     memoire est une decision de redaction, a prendre explicitement.")

# =====================================================================================
# figure S28
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.2))

# (a) les trois grandeurs, aux deux niveaux
lignes = [("indice de queue ξ", XI, XI_IC90, xi95),
          ("quantile de sévérité (M€)", VAR_UNIT, VAR_IC90, var95),
          ("capital agrégé (M€)", s_pub, (s_lo90, s_hi90), (s_lo95, s_hi95))]
for i, (lib, pt, i90, i95) in enumerate(lignes):
    y = 2 - i
    # normalisation par le point, pour mettre les trois sur une echelle commune
    ax1.plot([i95[0] / pt, i95[1] / pt], [y + 0.13, y + 0.13], color=ACCENT, lw=7,
             solid_capstyle="butt", alpha=0.85)
    ax1.plot([i90[0] / pt, i90[1] / pt], [y - 0.13, y - 0.13], color=BLUE, lw=7,
             solid_capstyle="butt", alpha=0.85)
    ax1.plot([1.0], [y], "o", color=INK, ms=6)
    ax1.text(i95[1] / pt + 0.06, y + 0.13, f"×{i95[1]/i95[0]:.2f}".replace(".", ","),
             va="center", fontsize=8.5, color=ACCENT)
    ax1.text(i90[1] / pt + 0.06, y - 0.13, f"×{i90[1]/i90[0]:.2f}".replace(".", ","),
             va="center", fontsize=8.5, color=BLUE)
ax1.set_yticks([2, 1, 0])
ax1.set_yticklabels([l[0] for l in lignes], fontsize=9)
ax1.axvline(1.0, color=INK2, lw=0.8, ls=":")
ax1.set_xlabel("borne rapportée à la valeur publiée", color=INK2)
ax1.set_xlim(0, max(i95[1] / pt for _, pt, _, i95 in lignes) * 1.35)
ax1.plot([], [], color=BLUE, lw=7, label="IC 90 % (actuel)")
ax1.plot([], [], color=ACCENT, lw=7, label="IC 95 % (retenu)")
# LA LEGENDE VA EN HAUT A DROITE, region vide : posee en bas a droite elle venait au contact de
# l'etiquette de la bande de capital, qui est la plus longue des trois.
ax1.legend(fontsize=8.5, frameon=False, loc="upper right")
ax1.set_title("(a)  Le passage à 95 %, sur les trois grandeurs\n(élargissement asymétrique sur "
              "le capital)", fontsize=10.5, color=INK, pad=8)

# (b) nominal contre reel
noms = ["90 %", "95 %"]
nom = [90.0, 95.0]
reel = [cov90, cov95]
x = np.arange(2)
ax2.bar(x - 0.19, nom, width=0.36, color=MUTED, alpha=0.85, label="niveau annoncé")
ax2.bar(x + 0.19, reel, width=0.36, color=ACCENT, alpha=0.9, label="couverture mesurée")
for xi_, n_, r_ in zip(x, nom, reel):
    ax2.text(xi_ - 0.19, n_ + 0.6, f"{n_:.0f}", ha="center", fontsize=9, color=INK2)
    ax2.text(xi_ + 0.19, r_ + 0.6, f"{r_:.1f}".replace(".", ","), ha="center", fontsize=9,
             color=ACCENT)
    ax2.annotate("", xy=(xi_ + 0.19, r_), xytext=(xi_ - 0.19, n_),
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9, ls=":"))
ax2.set_xticks(x)
ax2.set_xticklabels(noms, fontsize=10)
ax2.set_ylim(80, 100)
ax2.set_ylabel("pourcentage", color=INK2)
ax2.legend(fontsize=8.5, frameon=False, loc="lower right")
ax2.set_title("(b)  Changer le niveau annoncé ne corrige pas\nla sous-couverture, il la déplace",
              fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("S28 : l'intervalle reporté passe à 95 %, sans toucher la calibration ; "
             "et la sous-couverture, elle, ne se corrige pas en changeant le niveau",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S28_intervalle_95.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
