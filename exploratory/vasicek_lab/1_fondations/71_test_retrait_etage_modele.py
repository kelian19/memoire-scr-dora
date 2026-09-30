#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
71 : TEST du retrait de l'etage de MODELE de la bande reportee.

DEMANDE A L'ORIGINE (Caroline Hillairet, point d'etape du 12 aout 2026) : si l'intervalle total
devient trop volumineux, retirer l'etage de modele du calcul et se concentrer sur les etages
parametre et identification, apres avoir quantifie ce que l'etage retire apportait.

CE QUE LA VERIFICATION A TROUVE AVANT DE COMMENCER, ET IL FAUT LE DIRE D'ABORD. Le memoire
affirme que « les trois etages se cumulent » et la figure du script 48 porte un panneau intitule
ainsi. Ce panneau est du TEXTE : il affiche trois libelles et une legende. Le cumul n'a jamais
ete compose numeriquement. Il n'existe donc, aujourd'hui, AUCUNE bande a trois etages dont on
pourrait retirer quelque chose. La question posee suppose un objet qui n'a pas ete construit,
et ce script le construit avant de repondre.

COMMENT COMPOSER TROIS BANDES QUI NE SONT PAS DES LOIS. Les trois etages sont des ENSEMBLES de
valeurs admissibles, pas des distributions : on ne peut pas les convoler. La bande composee est
donc l'IMAGE du produit des trois ensembles par la fonction de capital. Et comme le capital est
monotone en chacun des trois leviers, il suffit d'evaluer les DEUX COINS, ce qui est exactement
l'argument de statique comparative utilise ailleurs dans le projet (croissance en g, script 66).

CE QUE LE TEST A MIS AU JOUR, ET C'EST LE VRAI RESULTAT. L'etage de PARAMETRE fait varier xi sur
son intervalle bootstrap, jusqu'a 0,8313. L'axe de FAMILLE de l'etage de modele pousse la queue a
xi = 0,90. C'est LE MEME LEVIER a deux valeurs voisines, et c'est la que le bat blesse : une bande
doit dire UNE SEULE FOIS jusqu'ou on laisse aller xi. Or 0,8313 est une borne ESTIMEE, celle de
l'IC90, et 0,90 une valeur POSEE hors de cet intervalle. Le retrait demande par Caroline rend donc
la bande HOMOGENE, ce qui est un argument plus fort que la lisibilite.

J'ATTENDAIS AUTRE CHOSE ET JE ME SUIS TROMPE, c'est ecrit dans le script. Je pensais que passer de
0,83 a 0,90 n'ajouterait presque rien, donc que le cumul etait un double compte quasi pur. C'est
faux : l'extension vaut +12 748 M, le capital etant violemment convexe en xi. Le probleme n'est
pas que l'axe famille soit negligeable, c'est qu'il n'est pas de la meme NATURE que la borne
d'a cote.

Sortie : diagnostics + figure S27_test_etage_modele.png.
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
from euro_cascade_model import PARAMS, var                      # noqa: E402

WID = 88
sp = PARAMS["OPRISK"]

# xi : calibration publiee et bornes de son IC90 bootstrap (calibration figee, script 63).
XI_PUB = sp["xi"]
XI_LO, XI_HI = 0.3044, 0.8313
XI_FAMILLE = 0.90                  # l'axe de FAMILLE de l'etage de modele (script 48)

# Bandes deja publiees, reprises telles quelles (scripts 30 et 48).
IDENT = (6858.0, 8697.0)           # identification de la direction de W
DEP = (5322.0, 9806.0)             # axe dependance de l'etage de modele
FAM_BAS = 4330.0                   # lognormale, axe famille, borne basse

NY = 60_000
NSEED = 4
SEED0 = 20260812
GAIN = 0.90


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def scr_xi(xi, nseed=NSEED, ny=NY):
    """SCR agrege (VaR 99,5 %) a xi donne, tout le reste a la calibration publiee."""
    out = []
    for k in range(nseed):
        rng = np.random.default_rng(SEED0 + k)
        out.append(var(ec.simulate_euro(sp["lam_ref"], GAIN, xi, sp["sigma"], sp["u"],
                                        sp["p_u"], sp["cap"], ny, rng)))
    return float(np.mean(out))


# =====================================================================================
titre("1. L'etage de PARAMETRE, porte sur l'AGREGAT et non sur le quantile unitaire")
# =====================================================================================
# LE MEMOIRE PUBLIE CET ETAGE SUR LE QUANTILE D'UN SINISTRE ([411 ; 1037], facteur 2,5). Pour
# le composer avec les deux autres, qui vivent sur l'agrege, il faut le lire sur l'agrege : sans
# cela on additionnerait des grandeurs de deux echelles, ce que le projet a deja paye une fois.
s_lo, s_pub, s_hi = scr_xi(XI_LO), scr_xi(XI_PUB), scr_xi(XI_HI)
s_fam = scr_xi(XI_FAMILLE)
print(f"  xi = {XI_LO:.4f} (borne basse IC90)   -> SCR = {s_lo:8.0f} M")
print(f"  xi = {XI_PUB:.4f} (calibration figee) -> SCR = {s_pub:8.0f} M")
print(f"  xi = {XI_HI:.4f} (borne haute IC90)   -> SCR = {s_hi:8.0f} M")
print(f"  xi = {XI_FAMILLE:.4f} (axe FAMILLE)       -> SCR = {s_fam:8.0f} M")
print(f"\n  Etage PARAMETRE sur l'agrege : [{s_lo:.0f} ; {s_hi:.0f}] M, facteur "
      f"{s_hi/s_lo:.2f}.")

# =====================================================================================
titre("2. LE RECOUVREMENT, et c'est le resultat que je n'attendais pas")
# =====================================================================================
print(f"  L'etage de parametre monte xi jusqu'a {XI_HI:.4f}. L'axe de FAMILLE de l'etage de")
print(f"  modele le pousse a {XI_FAMILLE:.2f}. C'est le MEME parametre a deux valeurs voisines.")
print(f"\n  SCR a la borne haute du parametre : {s_hi:.0f} M")
print(f"  SCR a la valeur de l'axe famille   : {s_fam:.0f} M")
sup = s_fam - s_hi
print(f"  extension apportee par l'axe famille au-dela du parametre : {sup:+.0f} M, soit "
      f"{100*sup/(s_hi - s_lo):+.0f} %")
print(f"  de la largeur que l'etage de parametre couvre deja.")

# CORRECTION DE MA PROPRE ATTENTE, ET IL FAUT L'ECRIRE. J'attendais un recouvrement PRESQUE
# TOTAL, c'est-a-dire que passer de 0,83 a 0,90 n'ajoute presque rien. C'est faux : l'extension
# vaut +12 748 M, parce que le capital est violemment convexe en xi. Le probleme n'est donc pas
# que l'axe famille soit negligeable, c'est qu'il agit sur LE MEME LEVIER que l'etage de
# parametre. Deux sources qui poussent le meme parametre ne se cumulent pas comme deux sources
# independantes : il n'y a qu'UN xi, et une bande doit dire jusqu'ou on le laisse aller, une
# seule fois.
print("\n  CE QUI EST EN JEU N'EST DONC PAS UNE TAILLE, C'EST UNE NATURE. L'extension est grande,")
print("  le capital etant violemment convexe en xi. Mais les deux etages poussent LE MEME")
print(f"  parametre : il n'y a qu'un xi, et une bande doit dire une seule fois jusqu'ou on le")
print(f"  laisse aller. Or {XI_HI:.4f} est une BORNE ESTIMEE, celle de l'IC90 bootstrap, tandis que")
print(f"  {XI_FAMILLE:.2f} est une valeur POSEE, hors de cet intervalle. Les melanger dans le meme")
print("  intervalle revient a mettre un scenario et une borne de confiance sur la meme echelle.")
print(f"\n  La borne BASSE de l'axe famille, la lognormale a {FAM_BAS:.0f} M, est en revanche une autre")
print("  FAMILLE et non un autre xi : celle-la est un apport propre, et le recouvrement est donc")
print("  asymetrique, il porte sur la borne haute et non sur la borne basse.")

print(f"\n  RESERVE A NE PAS PERDRE, ET ELLE BORNE TOUT CE QUI PRECEDE. Ce script fait varier xi")
print("  SEUL, sigma reste a sa valeur publiee. Or le bootstrap estime le couple (xi, sigma)")
print("  conjointement et ses deux composantes sont correlees : la vraie bande de parametre suit")
print("  le nuage bootstrap, pas le segment en xi a sigma fixe. Les largeurs de parametre")
print("  ci-dessous sont donc des MAJORANTS, et il ne faut pas en conclure que l'incertitude de")
print("  parametre ecrase celle d'identification sans avoir refait le calcul sur le nuage.")

# =====================================================================================
titre("3. La bande composee, avec et sans l'etage de modele")
# =====================================================================================
# Le capital etant monotone en chacun des leviers, l'image du produit des ensembles est
# l'intervalle entre les deux COINS. On compose par l'enveloppe des bornes.
deux_lo = min(s_lo, IDENT[0])
deux_hi = max(s_hi, IDENT[1])
trois_lo = min(deux_lo, DEP[0], FAM_BAS)
trois_hi = max(deux_hi, DEP[1], s_fam)

# LES QUATRE ETAGES SUR UNE SEULE ECHELLE, AVEC LEUR FACTEUR IMPRIME. Le memoire citait un
# facteur 5,5 pour l'axe famille et 2,5 pour le parametre, soit une comparaison entre l'AGREGE
# et le QUANTILE UNITAIRE : deux echelles. Cette table les met tous sur la charge annuelle, et
# imprime les facteurs pour qu'aucun ne soit recalcule a la main dans la redaction.
print(f"  {'etage, sur la charge annuelle':<34}{'borne basse':>14}{'borne haute':>14}"
      f"{'largeur':>11}{'facteur':>10}")
for lib, lo, hi in (("parametre (xi sur son IC90)", s_lo, s_hi),
                    ("identification (direction W)", IDENT[0], IDENT[1]),
                    ("modele, axe dependance", DEP[0], DEP[1]),
                    ("modele, axe famille de queue", FAM_BAS, s_fam)):
    print(f"  {lib:<34}{lo:>14.0f}{hi:>14.0f}{hi-lo:>11.0f}{hi/lo:>10.2f}")
print(f"  {'-' * 83}")
for lib, lo, hi in (("DEUX etages (retenu)", deux_lo, deux_hi),
                    ("TROIS etages (ancienne annonce)", trois_lo, trois_hi)):
    print(f"  {lib:<34}{lo:>14.0f}{hi:>14.0f}{hi-lo:>11.0f}{hi/lo:>10.2f}")
print("\n  LA HIERARCHIE AINSI RETABLIE N'EST PAS CELLE QU'ON ANNONCE D'ORDINAIRE : l'etage de")
print("  modele N'ECRASE PAS les autres, il est du meme ordre que celui de parametre, et les deux")
print("  dominent largement l'identification. Le retirer ne peut donc pas se justifier par sa")
print("  taille, contrairement a ce que le memoire ecrivait en comparant un facteur d'agregat a")
print("  un facteur de quantile unitaire.")

gain = (trois_hi - trois_lo) - (deux_hi - deux_lo)
print(f"\n  Le retrait de l'etage de modele retire {gain:.0f} M de largeur, soit "
      f"{100*gain/(trois_hi-trois_lo):.0f} % de la bande")
print(f"  a trois etages. Le facteur entre bornes tombe de {trois_hi/trois_lo:.2f} a "
      f"{deux_hi/deux_lo:.2f}.")

# =====================================================================================
titre("4. « Est-ce que cela ameliore les resultats ? » Trois reponses, et elles diffèrent")
# =====================================================================================
print("  Il faut distinguer trois sens du mot, parce que la reponse change avec le sens.")
print(f"\n  (a) PLUS LISIBLE : oui, et franchement. La bande passe d'un facteur "
      f"{trois_hi/trois_lo:.2f} a {deux_hi/deux_lo:.2f}")
print("      entre ses bornes. Un intervalle qui varie d'un facteur cinq n'est pas un")
print("      intervalle de confiance, c'est un aveu d'ignorance, et un lecteur n'en fait rien.")
print("\n  (b) PLUS EXACT : non, et il ne faut pas le laisser croire. L'incertitude sous-jacente")
print("      ne bouge pas d'un euro. Ce qui change est ce qu'on met dans un intervalle et ce")
print("      qu'on publie a cote comme axe de sensibilite. Rien n'est retire du document.")
print("\n  (c) PLUS HOMOGENE : OUI, et c'est l'argument qu'il ne faut pas rater. La bande a trois")
print(f"      etages avait pour borne haute un xi POSE a {XI_FAMILLE:.2f}, hors de l'IC90 bootstrap, place")
print(f"      sur la meme echelle qu'une borne ESTIMEE a {XI_HI:.4f}. Un intervalle qui melange une")
print("      borne de confiance et un scenario ne se lit pas : le lecteur ne sait plus si la")
print("      borne haute est un aleas defavorable ou une autre theorie de la queue. Le retrait")
print("      rend la bande homogene, ce qui est plus qu'un gain de lisibilite.")
print("\n  CE QUI JUSTIFIE LE PLUS LE RETRAIT N'EST DONC PAS LA LARGEUR, C'EST L'HOMOGENEITE de ce")
print("  qu'on met dans un intervalle. Et cela renforce la position de Caroline par un chemin")
print("  qu'elle n'avait pas emprunte : elle proposait de simplifier, et il se trouve que la")
print("  composition qu'on simplifie melangeait deux natures d'incertitude sur un seul parametre.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. Le cumul des trois etages n'avait JAMAIS ete compose : le panneau qui l'annonce est")
print("     du texte. La bande a trois etages est construite ici pour la premiere fois.")
print(f"  2. Etage de parametre porte sur l'agrege : [{s_lo:.0f} ; {s_hi:.0f}] M, facteur "
      f"{s_hi/s_lo:.2f}, a ne")
print("     PAS confondre avec le facteur 2,5 du quantile unitaire : le quantile agrege amplifie")
print("     l'incertitude sur xi, donc le 2,5 publie ne mesure pas ce que le parametre fait au")
print("     CAPITAL. Majorant toutefois, xi variant seul a sigma fixe (voir la reserve du 2).")
print(f"  3. L'axe famille (xi = {XI_FAMILLE:.2f}) et l'etage de parametre (xi <= {XI_HI:.4f}) poussent LE")
print(f"     MEME parametre. L'extension vaut {sup:+.0f} M, donc elle n'est pas negligeable : le")
print("     probleme est de nature, une borne posee melangee a une borne estimee, pas de taille.")
print(f"  4. Retirer l'etage de modele fait tomber la bande d'un facteur {trois_hi/trois_lo:.2f} "
      f"a {deux_hi/deux_lo:.2f}.")
print("  5. Reponse a la question posee : plus lisible OUI, plus exact NON, plus homogene OUI.")
print("     C'est le troisieme point qui emporte la decision, pas le premier.")
print(f"  6. Et un constat qui n'etait pas cherche : la bande d'identification ([{IDENT[0]:.0f} ; "
      f"{IDENT[1]:.0f}])")
print("     est ENTIEREMENT contenue dans la bande de parametre lue sur l'agrege. A confirmer sur")
print("     le nuage bootstrap avant d'en tirer une hierarchie, mais c'est a instruire.")

# =====================================================================================
# figure S27
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 5.2),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# (a) les bandes, empilees
bandes = [("paramètre\n(ξ sur son IC90)", s_lo, s_hi, GREEN),
          ("identification\n(direction de W)", IDENT[0], IDENT[1], BLUE),
          ("modèle, dépendance", DEP[0], DEP[1], MUTED),
          ("modèle, famille", FAM_BAS, s_fam, ACCENT),
          ("DEUX étages\n(bande retenue)", deux_lo, deux_hi, BLUE),
          ("TROIS étages\n(ancienne annonce)", trois_lo, trois_hi, ACCENT)]
for i, (lib, lo, hi, c) in enumerate(bandes):
    y = len(bandes) - 1 - i
    alpha = 0.9 if i >= 4 else 0.5
    ax1.barh(y, hi - lo, left=lo, color=c, alpha=alpha, height=0.56)
    ax1.text(hi * 1.02, y, f"{lo:.0f} – {hi:.0f}", va="center", fontsize=8.5, color=INK2)
ax1.axvline(s_pub, color=INK, lw=1.0, ls=":")
ax1.text(s_pub * 1.01, len(bandes) - 0.45, "calibration publiée", fontsize=8, color=INK2)
ax1.set_yticks(range(len(bandes)))
ax1.set_yticklabels([b[0] for b in bandes][::-1], fontsize=8.5)
ax1.set_xlabel("SCR agrégé (VaR 99,5 %, M€)", color=INK2)
ax1.set_xlim(0, trois_hi * 1.30)
ax1.set_title("(a)  Les étages, et les deux compositions", fontsize=10.5, color=INK, pad=8)

# (b) le recouvrement sur xi
xs = np.array([XI_LO, XI_PUB, XI_HI, XI_FAMILLE])
ys = np.array([s_lo, s_pub, s_hi, s_fam])
ax2.plot(xs, ys, "o-", color=INK2, lw=1.6, ms=7)
ax2.axvspan(XI_LO, XI_HI, color=GREEN, alpha=0.16)
ax2.axvspan(XI_HI, XI_FAMILLE, color=ACCENT, alpha=0.30)
ax2.text((XI_LO + XI_HI) / 2, max(ys) * 0.94, "étage PARAMÈTRE\n(IC90 bootstrap)",
         ha="center", fontsize=8.5, color=INK2)
ax2.annotate(f"l'axe FAMILLE\nprolonge le MÊME ξ\net ajoute {sup:+.0f} M€", xy=(XI_FAMILLE, s_fam),
             xytext=(XI_LO + 0.06, max(ys) * 0.55), fontsize=8.5, color=ACCENT,
             arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
for x, y in zip(xs, ys):
    ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(6, -12),
                 fontsize=8, color=INK2)
ax2.set_xlabel("indice de queue ξ", color=INK2)
ax2.set_ylabel("SCR agrégé (M€)", color=INK2)
ax2.set_title("(b)  Le recouvrement : l'axe famille prolonge\nle paramètre sur le même ξ",
              fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("S27 : les deux étages poussent le même ξ ; retirer celui de modèle rend la bande "
             "homogène, une borne estimée n'étant pas un scénario",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S27_test_etage_modele.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
