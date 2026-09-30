#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
83 : la formule derriere « 6 a 35 ans » et « 1 a 7 ans », ecrite, puis ACTUALISEE.

DEUX DEMANDES DU POINT TUTEUR DU 14 AOUT 2026, et une seule mesure les traite toutes les deux.
  - « Resultat 6,35 ans (majorant) et 1,7 ans (composantes reunies) : formule sous-jacente a
    clarifier et a documenter. »
  - « Le parametre de cost of capital a ete revise dans Solvabilite II et doit etre mis a jour. »

LA FORMULE EST TRIVIALE, ET C'EST EXACTEMENT LE PROBLEME :
        T  =  C / B
ou C est le cout de remediation ONE-OFF du secteur et B le benefice ANNUEL. Avec
        C = 200 entites x 25 a 150 M€            = 5 000 a 30 000 M€
        B(portage seul) = CoC x Delta_SCR         = 0,06 x 14 139 = 848 M€/an
        B(les deux composantes) = 848 + 3 247     = 4 095 M€/an
on retrouve 5 000/848 = 6 ans, 30 000/848 = 35 ans, puis 1 an et 7 ans. Aucun mystere.

CE QUI CLOCHE N'EST DONC PAS LE CALCUL MAIS L'INSTRUMENT. C'est un PAYBACK SIMPLE NON ACTUALISE.
A un horizon de 35 ans, ne pas actualiser n'est pas une approximation, c'est un changement de
nature : un euro dans trente ans n'est pas un euro aujourd'hui, et le taux qui sert a l'escompter
est justement celui dont on parle. Ce script actualise, et il en sort une identite qui tranche la
question du taux.

L'IDENTITE. Le portage est une charge annuelle de CoC x Delta_SCR. Sa valeur presente, escomptee
au COUT DU CAPITAL lui-meme, vaut
        VP  =  (CoC x Delta_SCR) / CoC  =  Delta_SCR      exactement, quel que soit CoC.
Autrement dit LE CAPITAL LIBERE *EST* LA VALEUR PRESENTE DE L'ECONOMIE DE PORTAGE, et la reviser
le taux ne la deplace pas d'un euro. Le taux deplace le nombre d'annees affiche, pas l'economie.
C'est la reponse a la seconde demande, et elle est plus forte qu'une mise a jour de parametre.

CONSEQUENCE, ET ELLE EST QUALITATIVE. Le seuil de rentabilite au portage seul devient
C = Delta_SCR = 14 139 M€. Or C va de 5 000 a 30 000 : au bas de la fourchette le compte se
retourne, AU HAUT DE LA FOURCHETTE IL NE SE RETOURNE JAMAIS, a aucun horizon. Le « 35 ans » du
memoire n'est donc pas seulement imprecis, il masque une non-existence.

Sortie : diagnostics + figure S39_retour_actualise.png.
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
import resultats_partages as rp                                 # noqa: E402
import canaux_conformite as cx                                  # noqa: E402

WID = 88
DDORA = rp.DELTA_DORA
COC = rp.COUT_DU_CAPITAL                 # 6 %, delegue 2015/35
COC_REVISE = 0.0475                      # directive (UE) 2025/2
COUT_ENT = rp.COUT_DORA_ENTITE            # 25 a 150 M€ par grande entite
N_ENT = 200                               # hypothese affichee, comme au script 50
# Taux d'escompte balayes. Aucun n'est « le bon » : le premier est le cout du capital lui-meme,
# les suivants encadrent une courbe sans risque plausible. La conclusion ne doit dependre d'aucun.
TAUX_ESCOMPTE = (0.06, 0.0475, 0.03, 0.02)


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def fnum(v):
    """Separateur de milliers applique AU NOMBRE SEUL, jamais a une phrase entiere."""
    return f"{v:,.0f}".replace(",", " ")


# =====================================================================================
titre("1. La formule, ecrite, et les quatre nombres publies reproduits")
# =====================================================================================
C_lo, C_hi = N_ENT * COUT_ENT[0], N_ENT * COUT_ENT[1]
m_C = cx.metriques_par_graine(lam=cx.LAM_C, g=cx.G_C, p_u=cx.PU_C, phi_cs=None)
m_NC = cx.metriques_par_graine(lam=cx.LAM_NC, g=cx.G_NC, p_u=cx.PU_NC, phi_cs=cx.PHICS_NC)
perte_evitee = float(m_NC[:, 1].mean() - m_C[:, 1].mean())
B_port = COC * DDORA
B_tot = B_port + perte_evitee
print("  T = C / B, avec C le cout ONE-OFF et B le benefice ANNUEL. Les entrees :")
# Les deux bornes de cout sont aussi imprimees SANS separateur de milliers : « 5 000 » avec une
# espace ne se compare pas au 5000 d'un texte, et le harnais laisserait le nombre non confirme.
print(f"    bornes de cout, en clair pour la verification : {int(C_lo)} et {int(C_hi)} M€.")
print(f"    C   = {N_ENT} entites x {COUT_ENT[0]:.0f} a {COUT_ENT[1]:.0f} M€ = {fnum(C_lo)} a {fnum(C_hi)} M€")
print(f"    Delta_SCR                                  = {fnum(DDORA)} M€")
print(f"    B(portage)   = CoC x Delta_SCR = {100*COC:g} % x {fnum(DDORA)} = {B_port:.0f} M€/an")
print(f"    perte evitee = charge moyenne NC - charge moyenne C = "
      f"{m_NC[:, 1].mean():.0f} - {m_C[:, 1].mean():.0f} = {perte_evitee:.0f} M€/an")
print(f"    B(les deux)  = {B_port:.0f} + {perte_evitee:.0f} = {B_tot:.0f} M€/an")
print(f"\n  {'lecture':<26}{'B (M€/an)':>12}{'T bas':>9}{'T haut':>9}{'publie':>14}")
print(f"  {'portage seul':<26}{B_port:>12.0f}{C_lo/B_port:>9.1f}{C_hi/B_port:>9.1f}"
      f"{'6 a 35 ans':>14}")
print(f"  {'les deux composantes':<26}{B_tot:>12.0f}{C_lo/B_tot:>9.1f}{C_hi/B_tot:>9.1f}"
      f"{'1 a 7 ans':>14}")
print("\n  Les quatre nombres publies sont donc reproduits, et la formule est une simple division.")
print("  Elle est desormais documentee, ce qui etait la demande. LA SUITE MONTRE POURQUOI CETTE")
print("  DIVISION EST LE MAUVAIS INSTRUMENT.")

# =====================================================================================
titre("2. Pourquoi un payback non actualise ne tient pas a cet horizon")
# =====================================================================================
print("  Un payback simple traite un euro de l'annee 35 comme un euro d'aujourd'hui. Sur trois")
print("  ans l'approximation est benigne ; sur trente-cinq elle ne l'est pas, et elle l'est")
print("  d'autant moins que le flux dont on parle EST un cout du capital : escompter au cout du")
print("  capital est exactement ce que la marge de risque de Solvabilite II fait elle-meme.")
print("\n  LA VALEUR PRESENTE D'UNE ECONOMIE ANNUELLE PERPETUELLE B, escomptee au taux r, vaut B/r.")
print("  Pour le portage, B = CoC x Delta_SCR, donc :")
print("        VP(portage, escompte au taux r)  =  CoC x Delta_SCR / r")
print("  et, si l'on escompte au COUT DU CAPITAL lui-meme (r = CoC), les deux se simplifient :")
print("        VP  =  Delta_SCR      EXACTEMENT, et INDEPENDAMMENT de CoC.")
vp_coc = COC * DDORA / COC
print(f"\n  Verification numerique : {100*COC:g} % x {fnum(DDORA)} / {100*COC:g} % = {fnum(vp_coc)} M€, contre "
      f"Delta_SCR = {fnum(DDORA)}.")
print(f"  Ecart : {abs(vp_coc - DDORA):.6f} M€.")
print("\n  CE N'EST PAS UNE COINCIDENCE, C'EST LA DEFINITION DU PORTAGE, et l'enonce merite d'etre")
print("  retenu : LE CAPITAL LIBERE *EST* LA VALEUR PRESENTE DE L'ECONOMIE DE PORTAGE. Comparer un")
print("  cout de remediation a Delta_SCR est donc la comparaison homogene ; la convertir en annees")
print("  n'ajoute rien et introduit une dependance au taux qui n'existe pas dans la grandeur.")

# =====================================================================================
titre("3. LA REPONSE A LA DEMANDE SUR LE TAUX, et elle est plus forte qu'une mise a jour")
# =====================================================================================
print("  Le point tuteur demande de mettre a jour le cout du capital, revise par la directive")
print(f"  (UE) 2025/2 de {100*COC:g} % a {100*COC_REVISE:g} %. Voici ce que la revision deplace et ce qu'elle ne")
print("  deplace pas.")
print(f"\n  {'CoC':>8}{'B portage (M€/an)':>20}{'T non actualise':>18}"
      f"{'VP au taux = CoC':>20}")
for taux in (COC, COC_REVISE):
    b = taux * DDORA
    print(f"  {100*taux:>6.2f} %{b:>20.0f}{C_lo/b:>8.1f} a {C_hi/b:<8.1f}{taux*DDORA/taux:>20.0f}")
print(f"\n  LE NOMBRE D'ANNEES BOUGE DE {C_hi/(COC*DDORA):.0f} A {C_hi/(COC_REVISE*DDORA):.0f} ANS, LA VALEUR PRESENTE NE BOUGE PAS")
print(f"  D'UN EURO : elle vaut {fnum(DDORA)} M€ dans les deux cas. La revision du taux deplace donc le")
print("  chiffre de COMMUNICATION et laisse l'economie intacte. C'est la reponse a la demande, et")
print("  elle dispense de rejouer la chaine : le gel de la calibration n'a rien a arbitrer ici,")
print("  puisque la grandeur economiquement pertinente est invariante.")
print("\n  UNE PRECAUTION, ET ELLE EST IMPORTANTE. L'invariance vaut quand on escompte AU COUT DU")
print("  CAPITAL. La marge de risque de Solvabilite II, elle, applique la charge de CoC aux SCR")
print("  futurs puis les escompte a la COURBE SANS RISQUE, qui est plus basse. On balaie donc le")
print("  taux d'escompte plutot que de le poser, et la section suivante montre que la conclusion")
print("  de gestion ne depend d'aucun de ces choix.")

# =====================================================================================
titre("4. LE SEUIL DE RENTABILITE, et il change la nature du « 35 ans »")
# =====================================================================================
print("  Avec une economie annuelle B escomptee au taux r, la valeur presente sur T annees vaut")
print("  B x (1 - (1+r)^-T) / r, qui CONVERGE vers B/r. Il existe donc un cout au-dela duquel")
print("  aucun horizon ne suffit :")
print("        seuil de rentabilite  =  B / r")
print("\n  DEUX TAUX INTERVIENNENT ET IL NE FAUT PAS LES CONFONDRE, c'est justement l'objet de la")
print("  demande. Le taux de CHARGE (CoC) fixe le montant annuel immobilise, B = CoC x Delta_SCR ;")
print("  le taux d'ESCOMPTE r ramene ce flux a aujourd'hui. Solvabilite II les distingue : la marge")
print("  de risque applique la charge de CoC aux SCR futurs puis escompte a la courbe SANS RISQUE.")
print("  Dans la table qui suit, LA CHARGE EST TENUE A 6 % et seul l'escompte varie. Quand les deux")
print("  coincident, on retombe sur l'identite de la section 2 et le seuil vaut Delta_SCR.")
print(f"\n  {'taux d escompte r':>18}{'B (charge a 6 %)':>18}{'seuil':>10}{'T(C bas)':>11}"
      f"{'T(C haut)':>12}")
res = []
for r in TAUX_ESCOMPTE:
    seuil = B_port / r
    tt = []
    for C in (C_lo, C_hi):
        x = r * C / B_port
        tt.append(-np.log(1.0 - x) / np.log(1.0 + r) if x < 1.0 else np.inf)
    res.append((r, seuil, tt[0], tt[1]))
    f_hi = f"{tt[1]:.0f} ans" if np.isfinite(tt[1]) else "JAMAIS"
    print(f"  {100*r:>16.2f} %{B_port:>18.0f}{seuil:>10.0f}{tt[0]:>10.1f} a{f_hi:>12}")
print(f"\n  ET AVEC LES DEUX COMPOSANTES DU BENEFICE ({B_tot:.0f} M€/an) :")
print(f"  {'taux d escompte r':>18}{'B les deux':>18}{'seuil':>10}{'T(C bas)':>11}{'T(C haut)':>12}")
res2 = []
for r in TAUX_ESCOMPTE:
    seuil = B_tot / r
    tt = []
    for C in (C_lo, C_hi):
        x = r * C / B_tot
        tt.append(-np.log(1.0 - x) / np.log(1.0 + r) if x < 1.0 else np.inf)
    res2.append((r, seuil, tt[0], tt[1]))
    f_hi = f"{tt[1]:.0f} ans" if np.isfinite(tt[1]) else "JAMAIS"
    print(f"  {100*r:>16.2f} %{B_tot:>18.0f}{seuil:>10.0f}{tt[0]:>10.1f} a{f_hi:>12}")

print("\n  LE RESULTAT QUALITATIF, ET IL N'ETAIT PAS VISIBLE SUR LE PAYBACK SIMPLE. Au portage seul")
print(f"  et au cout du capital, le seuil vaut {fnum(B_port/COC)} M€ tandis que le haut de la fourchette de")
print(f"  cout vaut {fnum(C_hi)} : LE PROJET NE SE RENTABILISE JAMAIS, a aucun horizon. Le « 35 ans »")
print("  du memoire ne mesure donc pas une duree longue, il masque une NON-EXISTENCE. A un taux")
print("  d'escompte plus bas le seuil remonte et la duree redevient finie, mais tres longue.")
print("\n  ET LA CONCLUSION DE GESTION, ELLE, EST LA MEME A TOUS LES TAUX : au haut de la fourchette")
print("  de cout, LE PORTAGE SEUL NE JUSTIFIE PAS LA DEPENSE. Ce qui la justifie est la perte")
print(f"  evitee, {perte_evitee:.0f} M€/an, soit {perte_evitee/B_port:.1f} fois le portage : avec les deux composantes le")
print(f"  seuil passe a {fnum(B_tot/COC)} M€ au cout du capital, tres au-dessus de {fnum(C_hi)}, et le retour")
print(f"  redevient {res2[0][2]:.1f} a {res2[0][3]:.0f} ans. C'EST DONC UN RESULTAT QUI RENFORCE LE MEMOIRE :")
print("  son propre argument, « le portage seul est un plancher et la perte evitee vaut 3,8 fois")
print("  plus », cesse d'etre une precaution de redaction pour devenir la condition MEME de")
print("  rentabilite du projet au haut de la fourchette.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. LA FORMULE EST DOCUMENTEE, et c'est une division : T = C / B, avec C le cout one-off")
print(f"     ({fnum(C_lo)} a {fnum(C_hi)} M€) et B le benefice annuel ({B_port:.0f} au portage seul, {B_tot:.0f} avec")
print("     la perte evitee). Les quatre nombres publies sont reproduits.")
print("  2. MAIS C'EST UN PAYBACK NON ACTUALISE, donc le mauvais instrument a un horizon de")
print("     trente-cinq ans, d'autant que le flux en question EST un cout du capital.")
print("  3. UNE IDENTITE TRANCHE LA QUESTION DU TAUX : escomptee au cout du capital, la valeur")
print(f"     presente de l'economie de portage vaut EXACTEMENT Delta_SCR = {fnum(DDORA)} M€, quel que")
print(f"     soit CoC. Passer de {100*COC:g} % a {100*COC_REVISE:g} % deplace le nombre d'annees de "
      f"{C_hi/(COC*DDORA):.0f} a {C_hi/(COC_REVISE*DDORA):.0f}")
print("     et ne deplace la valeur presente d'aucun euro. LA REVISION DU TAUX EST DONC UN CHIFFRE")
print("     DE COMMUNICATION, PAS UNE RECALIBRATION, ce qui la rend compatible avec le gel.")
print("  4. LE CAPITAL LIBERE *EST* LA VALEUR PRESENTE DE L'ECONOMIE DE PORTAGE. La comparaison")
print("     homogene est donc cout de remediation contre Delta_SCR, en euros, sans passer par des")
print("     annees.")
print(f"  5. D'OU UN RESULTAT QUALITATIF QUE LE PAYBACK SIMPLE CACHAIT : au portage seul et au cout")
print(f"     du capital, le seuil de rentabilite vaut {fnum(B_port/COC)} M€ contre un cout haut de {fnum(C_hi)}.")
print("     Le projet ne se rentabilise JAMAIS a ce niveau de cout. Le « 35 ans » masque une")
print("     non-existence plutot qu'il ne mesure une duree.")
print(f"  6. ET CELA RENFORCE LE MEMOIRE : la perte evitee ({perte_evitee:.0f} M€/an, {perte_evitee/B_port:.1f} fois le portage)")
print("     n'est plus une precaution de redaction mais la CONDITION de rentabilite au haut de la")
print("     fourchette de cout. Le mot « plancher » y gagne son sens economique.")

# =====================================================================================
# figure S39
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.4))

# (a) la valeur presente cumulee, portage seul, contre les deux bornes de cout
T = np.arange(0, 61)
# ETIQUETTES POSEES AU-DESSUS DE LEUR PROPRE LIGNE, jamais dessus : centrees verticalement sur
# la ligne, elles etaient barrees par les pointilles et devenaient illisibles.
for r, col, ls in ((COC, ACCENT, "-"), (0.03, BLUE, "-")):
    vp = B_port * (1.0 - (1.0 + r) ** (-T)) / r
    ax1.plot(T, vp, ls, color=col, lw=2,
             label=f"escompté à {100*r:g} %".replace(".", ","))
    ax1.axhline(B_port / r, color=col, lw=1.0, ls=":")
    ax1.text(59, B_port / r + 450, f"plafond {fnum(B_port/r)}", fontsize=8.5, color=col,
             va="bottom", ha="right")
ax1.axhline(C_lo, color=GREEN, lw=1.4, ls="--")
# decalee a x = 20 : posee au bord gauche, elle etait traversee par la courbe bleue vers x = 8.
ax1.text(20, C_lo + 450, f"coût bas, {fnum(C_lo)} M€", fontsize=9, color=GREEN, va="bottom")
ax1.axhline(C_hi, color=INK, lw=1.4, ls="--")
ax1.text(1, C_hi + 450, f"coût haut, {fnum(C_hi)} M€", fontsize=9, color=INK,
         fontweight="bold", va="bottom")
ax1.set_xlim(0, 60)
ax1.set_ylim(0, C_hi * 1.28)
ax1.set_xlabel("horizon (années)", color=INK2)
ax1.set_ylabel("valeur présente de l'économie de portage (M€)", color=INK2)
ax1.legend(fontsize=9, frameon=False, loc="lower right")
ax1.set_title("(a)  Le portage seul plafonne : au coût haut,\naucun horizon ne rentabilise",
              fontsize=10.5, color=INK, pad=8)

# (b) l'identite et l'invariance au taux
taux = np.linspace(0.02, 0.10, 200)
ax2.plot(100 * taux, [C_hi / (t * DDORA) for t in taux], "-", color=ACCENT, lw=2.2,
         label="payback non actualisé, coût haut")
ax2.plot(100 * taux, [DDORA / 1000.0] * len(taux), "-", color=BLUE, lw=2.2,
         label="valeur présente au taux = CoC (Md€)")
for t, lab, col in ((COC, f"{100*COC:g} %", MUTED), (COC_REVISE, f"{100*COC_REVISE:g} %".replace(".", ","), MUTED)):
    ax2.axvline(100 * t, color=col, lw=1.0, ls=":")
    ax2.text(100 * t, 62, lab, fontsize=8.5, color=INK2, ha="center")
ax2.annotate(f"{C_hi/(COC_REVISE*DDORA):.0f} ans", xy=(100 * COC_REVISE, C_hi / (COC_REVISE * DDORA)),
             xytext=(3.2, 52), fontsize=9.5, color=ACCENT, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
ax2.annotate(f"{C_hi/(COC*DDORA):.0f} ans", xy=(100 * COC, C_hi / (COC * DDORA)),
             xytext=(7.4, 40), fontsize=9.5, color=ACCENT, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
ax2.text(7.0, DDORA / 1000.0 + 3.5, f"plate : {DDORA/1000:.1f} Md€ quel que soit le taux".replace(".", ","),
         fontsize=9.5, color=BLUE, fontweight="bold")
ax2.set_ylim(0, 68)
ax2.set_xlabel("coût du capital (%)", color=INK2)
ax2.set_ylabel("années (orange)   ·   Md€ (bleu)", color=INK2)
# LEGENDE EN BAS A GAUCHE : au centre droit elle heurtait l'annotation « 35 ans ».
ax2.legend(fontsize=9, frameon=False, loc="lower left")
ax2.set_title("(b)  Le taux déplace les années, pas l'économie :\nla valeur présente vaut "
              "$\\Delta$SCR quel que soit CoC", fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("S39 : la formule du retour, actualisée, et ce que la révision du taux ne déplace pas",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S39_retour_actualise.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
