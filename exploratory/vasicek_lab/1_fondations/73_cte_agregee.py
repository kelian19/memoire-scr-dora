#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
73 : la CTE au niveau AGREGE, et la demonstration de sa non-existence sous PRC.

DEMANDE A L'ORIGINE (Caroline Hillairet, compte rendu du 13 aout 2026, section 7) : « les
indicateurs CTE / CVaR sont valides comme adaptes », avec un intervalle reporte a 95 %. Or la CTE
n'est aujourd'hui calculee NULLE PART au niveau agrege : le memoire n'en publie qu'un rapport sur
le quantile d'UN sinistre, et son argument d'existence est invoque sans etre montre. Ce script
comble les deux manques.

CE QU'IL FAUT DISTINGUER AVANT TOUT CALCUL, et c'est la seule chose qui evite le contresens :
« intervalle a 95 % » et « mesure de risque a 95 % » sont deux objets. Le premier est un niveau de
CONFIANCE sur un parametre, traite au script 72. Le second serait un changement de MESURE DE
CAPITAL, la reference reglementaire de Solvabilite II etant une VaR a 99,5 %. Ce script calcule la
CTE pour qu'elle soit disponible, et il donne surtout le nombre qui permet de comparer les deux
conventions : le niveau beta auquel la CTE egale la VaR reglementaire.

LE PLAFOND DE SEVERITE EST LE PIVOT DE L'ARGUMENT D'EXISTENCE, et il n'est pas le meme sur les
deux perimetres : OpRisk n'a AUCUN plafond (cap = None) et un indice de queue de 0,5954, donc une
esperance finie ; PRC est plafonne a 40 M et porte un indice de 1,0328, donc une esperance qui
n'existe pas sans ce plafond. La consequence est plus tranchante que ce que le memoire ecrit :
sous PRC la CTE simulee est finie SEULEMENT parce que le plafond la rend finie, et sa valeur est
alors une fonction du plafond plutot qu'une mesure du risque. On le demontre au lieu de l'affirmer.

Sortie : diagnostics + figure S29_cte_agregee.png.
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
from euro_cascade_model import PARAMS                           # noqa: E402

WID = 88
op = PARAMS["OPRISK"]
prc = PARAMS["PRC"]

NIVEAUX = (0.95, 0.99, 0.995)
GAIN = 0.90
NY = 200_000
NSEED = 4
SEED0 = 20260813

# Pour la demonstration d'existence : tailles croissantes et deux indices de queue, SANS plafond.
NY_SUITE = (50_000, 200_000, 800_000)
XI_FINI, XI_INFINI = op["xi"], prc["xi"]      # 0,5954 (esperance finie) et 1,0328 (infinie)


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def var_cte(pertes, beta):
    """(VaR, CTE) empiriques au niveau beta. CTE = moyenne au-dela de la VaR, incluse."""
    q = float(np.quantile(pertes, beta))
    queue = pertes[pertes >= q]
    return q, float(queue.mean()) if queue.size else q


def pertes(lam, xi, sigma, u, p_u, cap, ny, seed):
    rng = np.random.default_rng(seed)
    return ec.simulate_euro(lam, GAIN, xi, sigma, u, p_u, cap, ny, rng)


# =====================================================================================
titre("1. VaR et CTE agregees sur la calibration publiee, a trois niveaux")
# =====================================================================================
ech = [pertes(op["lam_ref"], op["xi"], op["sigma"], op["u"], op["p_u"], op["cap"],
              NY, SEED0 + k) for k in range(NSEED)]
print(f"  Perimetre OpRisk, calibration figee, {NY} annees par graine, {NSEED} graines.")
print(f"  Plafond de severite : {op['cap']} (aucun) ; indice de queue xi = {op['xi']:.4f} < 1,")
print("  donc l'esperance de la severite existe et la CTE agregee est bien definie.")
print(f"\n  {'niveau beta':<14}{'VaR (M)':>12}{'CTE (M)':>12}{'CTE / VaR':>12}")
tab = {}
for b in NIVEAUX:
    vs = [var_cte(e, b) for e in ech]
    v = float(np.mean([x[0] for x in vs]))
    c = float(np.mean([x[1] for x in vs]))
    tab[b] = (v, c)
    print(f"  {b:<14.3f}{v:>12.0f}{c:>12.0f}{c/v:>12.2f}")

v995, c995 = tab[0.995]
print(f"\n  Le capital publie par le memoire est la VaR a 99,5 %, soit {v995:.0f} M. La CTE au meme")
print(f"  niveau vaut {c995:.0f} M, soit {c995/v995:.2f} fois plus : c'est la moyenne des annees qui")
print("  DEPASSENT le capital, donc une grandeur de queue et non un niveau de couverture.")

# =====================================================================================
titre("2. Le niveau equivalent : a quel beta la CTE egale-t-elle la VaR reglementaire ?")
# =====================================================================================
# C'EST LE NOMBRE QUI PERMET DE COMPARER LES DEUX CONVENTIONS. Sans lui, « CTE a 95 % » et
# « VaR a 99,5 % » ne se comparent pas : on ne sait pas laquelle est la plus prudente.
grille = np.linspace(0.80, 0.9949, 400)
cte_grille = np.array([np.mean([var_cte(e, b)[1] for e in ech]) for b in grille])
i = int(np.argmin(np.abs(cte_grille - v995)))
beta_eq = float(grille[i])
print(f"  CTE_beta(L) = VaR_99,5%(L) = {v995:.0f} M  pour  beta = {beta_eq:.4f}, soit "
      f"{100*beta_eq:.2f} %.")
_, cte95 = tab[0.95]
print(f"\n  CONSEQUENCE DIRECTE, ET ELLE TRANCHE LA QUESTION. Une CTE a 95 % vaut {cte95:.0f} M, donc")
if cte95 < v995:
    print(f"  MOINS que la VaR reglementaire ({v995:.0f} M) : rapportee comme capital, elle serait")
    print(f"  MOINS prudente, dans un rapport de {v995/cte95:.2f}. Il faudrait monter le niveau de la")
    print(f"  CTE a {100*beta_eq:.2f} % pour retrouver l'exigence de Solvabilite II.")
else:
    print(f"  PLUS que la VaR reglementaire ({v995:.0f} M) : rapportee comme capital, elle serait")
    print(f"  PLUS prudente, dans un rapport de {cte95/v995:.2f}.")
print("\n  C'est pourquoi la CTE est publiee ici comme DIAGNOSTIC et non comme mesure de capital :")
print("  changer de mesure changerait le niveau d'exigence sans que personne l'ait decide, et le")
print("  niveau de 99,5 % est celui que le regime prudentiel fixe.")

# =====================================================================================
titre("3. Le rapport CTE / VaR comme diagnostic de forme de queue")
# =====================================================================================
asympt = 1.0 / (1.0 - op["xi"])
print(f"  Asymptote theorique du rapport pour une GPD d'indice xi : 1/(1-xi) = {asympt:.2f}.")
print(f"  Rapport mesure sur l'AGREGAT a 99,5 %                    : {c995/v995:.2f}.")
print(f"  Rapport mesure sur l'AGREGAT a 95 %                      : {tab[0.95][1]/tab[0.95][0]:.2f}.")
print("\n  LECTURE, ET LA PRUDENCE EST DE MISE. L'asymptote vaut pour le quantile d'UN sinistre")
print("  quand le niveau tend vers un ; l'agregat s'en approche par le principe du grand saut")
print("  unique, mais a niveau fini le rapport agrege reste EN DESSOUS. C'est le rapport, et non")
print("  les deux montants, qui informe : il dit de combien la moyenne de queue depasse le")
print("  quantile, donc a quel point la queue est epaisse au-dela du capital.")

# =====================================================================================
titre("4. L'existence de la CTE, DEMONTREE et non invoquee")
# =====================================================================================
# LE MEMOIRE ECRIT QUE SOUS PRC L'ESPERANCE EST INFINIE, DONC QUE LA MESURE N'EXISTE PAS. C'est
# une affirmation theorique : a xi > 1 l'esperance de la GPD diverge. On la MONTRE ici, en
# regardant ce que fait la CTE empirique quand l'echantillon grandit. Une esperance qui existe
# donne une CTE qui se stabilise ; une esperance qui n'existe pas donne une CTE qui derive.
# PREMIERE VERSION DE CE TEST, ET ELLE ETAIT FAUSSE. Je regardais si la CTE MONTE quand
# l'echantillon grandit, en attendant une derive. Les nombres ont dit le contraire : a
# xi = 1,0328 la CTE passait de 1 476 244 a 866 732 M sur seize fois plus d'annees, donc elle
# BAISSAIT, et mon texte affirmait pourtant qu'elle derivait vers le haut. La cause est que
# l'estimateur, a esperance infinie, est domine par un seul tirage : sur deux graines il n'a
# aucune tendance lisible, et lui en preter une etait une erreur de lecture de ma part.
#
# LE BON DIAGNOSTIC N'EST PAS LA TENDANCE, C'EST LA DISPERSION. Pour une loi d'esperance FINIE,
# la CTE empirique est un estimateur consistant : sa dispersion entre graines decroit comme
# 1/racine(n). Pour une loi d'esperance INFINIE, elle ne decroit pas : l'estimateur n'a pas de
# valeur vers laquelle converger. C'est cela qu'il faut mesurer, et c'est plus fort qu'une
# tendance : ce n'est pas que la valeur soit grande, c'est qu'il n'y en a pas.
NSEED_EXIST = 6
print("  Protocole : meme moteur, SANS plafond, a deux indices de queue, et l'on regarde la")
print(f"  DISPERSION de la CTE a 99,5 % entre {NSEED_EXIST} graines quand le nombre d'annees croit.")
print("  Une esperance finie rend l'estimateur consistant, donc sa dispersion relative decroit")
print("  comme 1/racine(n). Une esperance infinie ne lui donne rien vers quoi converger, donc la")
print("  dispersion NE DECROIT PAS. La tendance du niveau, elle, n'est pas lisible et ne doit pas")
print("  etre invoquee : a esperance infinie la CTE empirique est dominee par un tirage unique.")
print(f"\n  {'annees':>9}{'CTE (xi fini)':>16}{'disp. rel.':>12}"
      f"{'CTE (xi infini)':>18}{'disp. rel.':>12}")
suite_fini, suite_inf, cv_fini, cv_inf = [], [], [], []
for ny in NY_SUITE:
    ef = [var_cte(pertes(op["lam_ref"], XI_FINI, op["sigma"], op["u"], op["p_u"], None,
                         ny, SEED0 + k), 0.995)[1] for k in range(NSEED_EXIST)]
    ei = [var_cte(pertes(op["lam_ref"], XI_INFINI, op["sigma"], op["u"], op["p_u"], None,
                         ny, SEED0 + k), 0.995)[1] for k in range(NSEED_EXIST)]
    mf, mi = float(np.mean(ef)), float(np.mean(ei))
    sf = float(np.std(ef, ddof=1) / mf)
    si = float(np.std(ei, ddof=1) / mi)
    suite_fini.append(mf); suite_inf.append(mi); cv_fini.append(sf); cv_inf.append(si)
    print(f"  {ny:>9}{mf:>16.0f}{100*sf:>11.1f} %{mi:>18.0f}{100*si:>11.1f} %")

r_cv_fini = cv_fini[-1] / cv_fini[0]
r_cv_inf = cv_inf[-1] / cv_inf[0]
attendu = np.sqrt(NY_SUITE[0] / NY_SUITE[-1])

# CE QUI TRANCHE EST LE NIVEAU DE LA DISPERSION, PAS SA VITESSE DE DECROISSANCE, et il faut le
# dire dans cet ordre parce que j'avais d'abord annonce l'inverse. Ni l'une ni l'autre des deux
# series n'atteint la reference en 1/racine(n) : c'est ATTENDU, et c'est meme un resultat du
# projet. A xi = 0,5954 la VARIANCE de la severite est deja infinie (xi > 0,5) meme si son
# esperance existe, donc aucun estimateur de ce modele ne converge au rythme classique. Comparer
# les vitesses n'est donc pas concluant, d'autant qu'une dispersion mesuree sur six graines est
# elle-meme connue a 30 % pres. Le contraste qui tient est celui des NIVEAUX, et il vaut un
# ordre de grandeur.
print(f"\n  CE QUI TRANCHE EST LE NIVEAU DE LA DISPERSION :")
print(f"    a xi = {XI_FINI:.4f} elle va de {100*cv_fini[0]:.1f} a {100*cv_fini[-1]:.1f} % : la CTE "
      "se cite a deux chiffres ;")
print(f"    a xi = {XI_INFINI:.4f} elle reste entre {100*min(cv_inf):.1f} et {100*max(cv_inf):.1f} % : "
      "la CTE ne se cite PAS,")
print("      meme a un chiffre significatif, quelle que soit la taille de l'echantillon.")
print(f"  Un ordre de grandeur separe les deux, et ce contraste-la resiste au bruit d'une")
print("  dispersion estimee sur six graines.")
print(f"\n  LA VITESSE, ELLE, N'EST PAS CONCLUANTE, et il faut le dire : les facteurs valent")
print(f"  {r_cv_fini:.2f} et {r_cv_inf:.2f} sur seize fois plus d'annees, quand un estimateur "
      f"classique donnerait {attendu:.2f}.")
print(f"  AUCUNE des deux series n'atteint cette reference, et c'est attendu : a xi = {XI_FINI:.4f}")
print("  la VARIANCE de la severite est deja infinie meme si son esperance existe, donc aucun")
print("  estimateur de ce modele ne converge au rythme classique. Ce n'est donc pas un defaut de")
print("  la mesure, c'est une propriete de la queue, et elle vaut aussi pour la VaR.")
print("\n  CONCLUSION, ET ELLE PORTE SUR LA BONNE QUANTITE. Au-dela de xi = 1 la CTE empirique")
print("  n'est pas reproductible : deux echantillons de huit cents mille annees en donnent des")
print("  valeurs qui different de deux cinquiemes. Le niveau atteint, de l'ordre du million de")
print("  millions d'euros, est d'ailleurs sans interpretation : il mesure la taille du plus grand")
print("  tirage de l'echantillon, pas un risque. Une mesure de capital ne peut pas etre cela.")

# =====================================================================================
titre("5. Et sous PRC, la CTE ne survit que par le plafond")
# =====================================================================================
print(f"  Le perimetre PRC porte un plafond de severite de {prc['cap']:.0f} M, et un indice de")
print(f"  {prc['xi']:.4f} > 1. Le plafond rend donc l'esperance finie ARTIFICIELLEMENT. On le")
print("  verifie en le faisant varier : si la CTE en depend, elle mesure le plafond et non le")
print("  risque.")
print(f"\n  {'plafond (M)':>14}{'VaR 99,5 % (M)':>18}{'CTE 99,5 % (M)':>18}{'CTE / VaR':>12}")
caps = (prc["cap"], 2 * prc["cap"], 4 * prc["cap"])
cte_caps, var_caps = [], []
for cp in caps:
    vs = [var_cte(pertes(prc["lam_ref"], prc["xi"], prc["sigma"], prc["u"], prc["p_u"],
                         cp, NY, SEED0 + k), 0.995) for k in range(2)]
    v = float(np.mean([x[0] for x in vs]))
    c = float(np.mean([x[1] for x in vs]))
    var_caps.append(v)
    cte_caps.append(c)
    print(f"  {cp:>14.0f}{v:>18.0f}{c:>18.0f}{c/v:>12.2f}")
print(f"\n  Quadrupler le plafond multiplie la CTE par {cte_caps[-1]/cte_caps[0]:.2f} et la VaR par "
      f"{var_caps[-1]/var_caps[0]:.2f}.")

# ET CE N'EST PAS CE QUE J'ATTENDAIS, NI CE QUE J'AVAIS ECRIT. J'annoncais que la CTE serait
# « une fonction du plafond », en sous-entendant qu'elle en dependrait plus que la VaR. Les deux
# bougent du MEME facteur, a 0,01 pres : le plafond agit comme une echelle sur toute la loi, il
# ne distingue pas la CTE. L'information n'est donc pas dans le niveau, elle est dans le RAPPORT.
print("  LES DEUX BOUGENT DU MEME FACTEUR, et c'est instructif autrement que prevu : le plafond")
print("  agit comme une ECHELLE sur toute la loi, il ne distingue pas la CTE de la VaR. Ce n'est")
print("  donc pas le niveau qui porte l'information.")
print(f"\n  CE QUI LA PORTE EST LE RAPPORT, et il vaut {cte_caps[0]/var_caps[0]:.2f} sous PRC contre "
      f"{c995/v995:.2f} sur")
print("  la calibration publiee. Autrement dit, sur le perimetre plafonne la moyenne de queue est")
print("  a peine au-dessus du quantile : le plafond a retire la queue que la CTE est censee")
print("  mesurer, et la mesure se retrouve VIDEE DE SON CONTENU. Elle n'apporte alors presque")
print("  rien de plus que la VaR.")
print("\n  D'OU L'ARGUMENT CONTRE SON USAGE COMME NIVEAU DE COUVERTURE, et il tient en une")
print("  alternative : sur un perimetre d'indice superieur a 1, ou bien on ne plafonne pas et la")
print("  CTE n'a pas de valeur (section 4), ou bien on plafonne et elle n'apporte plus")
print("  d'information au-dela de la VaR. Dans les deux cas elle ne peut pas porter le capital.")
print("  La ou l'indice est inferieur a 1, elle est parfaitement legitime comme diagnostic, et")
print("  c'est exactement l'usage que le memoire en fait.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print(f"  1. CTE agregee, calibration publiee : {tab[0.95][1]:.0f} M a 95 %, {tab[0.99][1]:.0f} M "
      f"a 99 %, {c995:.0f} M a 99,5 %.")
print(f"  2. NIVEAU EQUIVALENT : la CTE egale la VaR reglementaire a beta = {100*beta_eq:.2f} %.")
print(f"     Une CTE a 95 % vaut donc {cte95:.0f} M contre {v995:.0f} : "
      f"{'moins' if cte95 < v995 else 'plus'} prudente que")
print("     l'exigence de Solvabilite II. Changer de mesure changerait le niveau d'exigence.")
print(f"  3. Rapport CTE/VaR agrege a 99,5 % : {c995/v995:.2f}, sous l'asymptote {asympt:.2f} de la")
print("     severite. C'est le rapport qui informe, pas les montants.")
print(f"  4. EXISTENCE DEMONTREE PAR LE NIVEAU DE LA DISPERSION, non par sa vitesse ni par une")
print(f"     tendance du niveau : {100*cv_fini[0]:.1f} a {100*cv_fini[-1]:.1f} % a xi = {XI_FINI:.4f} "
      f"contre {100*min(cv_inf):.1f} a {100*max(cv_inf):.1f} % a")
print(f"     xi = {XI_INFINI:.4f}. Un ordre de grandeur. Les VITESSES ({r_cv_fini:.2f} et "
      f"{r_cv_inf:.2f} contre {attendu:.2f} pour un")
print("     estimateur classique) ne concluent pas, la variance de la severite etant deja infinie")
print("     a l'indice publie : aucun estimateur de ce modele ne converge au rythme classique.")
print(f"  5. Sous PRC le plafond agit comme une ECHELLE : il multiplie la CTE et la VaR du meme")
print(f"     facteur ({cte_caps[-1]/cte_caps[0]:.2f} contre {var_caps[-1]/var_caps[0]:.2f}). "
      f"L'information est dans le RAPPORT, qui vaut")
print(f"     {cte_caps[0]/var_caps[0]:.2f} contre {c995/v995:.2f} sur la calibration publiee : le "
      "plafond a vide la mesure de")
print("     son contenu. Ou la CTE n'a pas de valeur, ou elle n'apporte rien de plus que la VaR.")
print("  6. Conclusion pour le report : CTE publiee comme DIAGNOSTIC, capital reste la VaR 99,5 %.")

# =====================================================================================
# figure S29
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 5.2))

# (a) VaR et CTE le long de beta, avec le niveau equivalent
ax1.plot(100 * grille, cte_grille, "-", color=ACCENT, lw=2, label="CTE$_\\beta$ agrégée")
var_grille = np.array([np.mean([var_cte(e, b)[0] for e in ech]) for b in grille])
ax1.plot(100 * grille, var_grille, "-", color=BLUE, lw=2, label="VaR$_\\beta$ agrégée")
ax1.axhline(v995, color=INK2, lw=1.0, ls="--")
ax1.plot([100 * beta_eq], [v995], "o", color=INK, ms=8)
ax1.annotate(f"CTE = VaR$_{{99,5\\%}}$\nà β = {100*beta_eq:.2f} %".replace(".", ","),
             xy=(100 * beta_eq, v995), xytext=(83, v995 * 1.45), fontsize=9, color=INK,
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
ax1.text(80.5, v995 * 1.04, f"capital publié, VaR$_{{99,5\\%}}$ = {v995:.0f} M€",
         fontsize=8.5, color=INK2)
ax1.set_xlabel("niveau β (%)", color=INK2)
ax1.set_ylabel("M€", color=INK2)
ax1.legend(fontsize=9, frameon=False, loc="upper left")
ax1.set_title("(a)  À quel niveau la CTE égale la VaR réglementaire",
              fontsize=10.5, color=INK, pad=8)

# (b) la demonstration d'existence
# LE PANNEAU TRACE LA DISPERSION ET NON LE NIVEAU. Une premiere version tracait le niveau et
# suggerait une derive que les nombres ne montrent pas : a esperance infinie le niveau n'a pas de
# tendance lisible, c'est sa dispersion qui refuse de decroitre.
_xi_fr = f"{XI_FINI:.4f}".replace(".", ",")
_xinf_fr = f"{XI_INFINI:.4f}".replace(".", ",")
ax2.plot(NY_SUITE, [100 * c for c in cv_fini], "o-", color=GREEN, lw=2, ms=8,
         label=f"ξ = {_xi_fr} (espérance finie)")
ax2.plot(NY_SUITE, [100 * c for c in cv_inf], "s-", color=ACCENT, lw=2, ms=8,
         label=f"ξ = {_xinf_fr} (espérance infinie)")
ref = [100 * cv_fini[0] * np.sqrt(NY_SUITE[0] / n) for n in NY_SUITE]
ax2.plot(NY_SUITE, ref, "--", color=MUTED, lw=1.6,
         label="ce qu'un estimateur classique donnerait")
# LA REFERENCE EST UN REPERE, PAS UNE ATTENTE : aucune des deux series ne l'atteint, la variance
# de la severite etant deja infinie a l'indice publie. Ce qui separe les deux series est le
# NIVEAU de leur dispersion, et c'est ce que l'axe montre.
ax2.set_ylim(min(min(cv_fini), min(ref) / 100) * 100 * 0.55, max(cv_inf) * 100 * 2.4)
for x, y in zip(NY_SUITE, cv_fini):
    ax2.annotate(f"{100*y:.1f}".replace(".", ",") + " %", (x, 100 * y),
                 textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8,
                 color=GREEN)
for x, y in zip(NY_SUITE, cv_inf):
    ax2.annotate(f"{100*y:.1f}".replace(".", ",") + " %", (x, 100 * y),
                 textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8,
                 color=ACCENT)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("nombre d'années simulées (échelle log)", color=INK2)
ax2.set_ylabel("dispersion relative de la CTE entre graines (%)", color=INK2)
ax2.legend(fontsize=8.5, frameon=False, loc="lower left")
# LE TITRE DISAIT « celle qui refuse de decroitre », et la serie orange DECROIT : elle passe de
# 51,4 a 38,6 %. Ce qui separe les deux series n'est pas la decroissance, c'est le NIVEAU, et un
# titre doit dire ce que la figure montre.
ax2.set_title("(b)  Ce qui sépare les deux n'est pas la pente mais le NIVEAU :\nquatre pour cent "
              "contre quarante", fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("S29 : la CTE agrégée se calcule sur la calibration publiée ; "
             "au-delà de ξ = 1 elle n'est pas reproductible, donc elle ne se cite pas",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S29_cte_agregee.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
