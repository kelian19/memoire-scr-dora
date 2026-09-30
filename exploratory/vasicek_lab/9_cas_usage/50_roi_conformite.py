#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
50 : le ROI de la conformite DORA — le capital libere face au cout de remediation.

Prolonge le tableau de bord KPI (script 43) : chaque levier de conformite libere du capital
(le Delta SCR entre non conforme et conforme) ; le remedier a un cout. On confronte les deux.

CE QUI EST SOLIDE (issu de notre modele) : le capital en jeu par levier (Delta SCR, script 43),
et sa valeur ANNUELLE au cout du capital de Solvabilite II (marge de risque, CoC = 6 %).
CE QUI EST EXTERNE ET INCERTAIN : le cout de remediation DORA (~25 a 150 M€ par grande entite,
estimations conseil/McKinsey 2025) et le nombre d'entites concernees. On l'assume comme tel et
on affiche les hypotheses.

MESSAGE : (1) le capital libere est CONCENTRE sur quelques leviers (la frequence domine), donc
le budget de remediation doit se prioriser, pas s'etaler ; (2) meme au seul cout de portage du
capital, la conformite offre un rendement recurrent qui compense une part materielle de son
cout ; l'ajout des pertes evitees (la reduction de la charge moyenne) rend le compte clairement
positif. La conformite cesse d'etre un pur cout : c'est un levier de capital.

Sortie : diagnostics + figure J6_roi_conformite.png.
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

WID = 82

# valeurs produites par le script 43 et ancres externes : source unique dans `resultats_partages`
LEVIERS = rp.LEVIERS           # (libelle, Delta SCR en M€, statut) par canal isole
DDORA = rp.DELTA_DORA          # ecart de capital DORA total (avec interaction)
COC = rp.COUT_DU_CAPITAL       # marge de risque Solvabilite II (reglement delegue 2015/35)
COST_ENTITY = rp.COUT_DORA_ENTITE   # cout de remediation par grande entite (M€)
N_ENTITIES = 200               # HYPOTHESE affichee : nb de grandes entites concernees
AMORT = 10                     # HYPOTHESE : amortissement du cout initial (annees)


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# =====================================================================================
titre("1. Capital libere par levier, et sa valeur annuelle (cout du capital 6 %)")
# =====================================================================================
print(f"  {'levier':<22}{'ΔSCR (M€)':>11}{'valeur/an (M€)':>16}{'statut':>12}")
rows = sorted(LEVIERS, key=lambda t: -t[1])
for name, d, statut in rows:
    print(f"  {name:<22}{d:>11.0f}{COC*d:>16.0f}{statut:>12}")
print(f"\n  Conformite TOTALE : ΔSCR = {DDORA:.0f} M€ libere, soit une valeur recurrente de")
print(f"  {COC*DDORA:.0f} M€/an au seul cout de portage du capital (6 %). Le capital libere est")
print(f"  CONCENTRE : la frequence seule ({rows[0][1]:.0f} M€) porte {100*rows[0][1]/sum(r[1] for r in rows):.0f} %"
      f" de la somme des leviers isoles.")

# =====================================================================================
titre("2. Face au cout de remediation DORA (hypotheses affichees)")
# =====================================================================================
one_off = (N_ENTITIES * COST_ENTITY[0], N_ENTITIES * COST_ENTITY[1])
annual_saving = COC * DDORA
pay_lo = one_off[0] / annual_saving
pay_hi = one_off[1] / annual_saving
print(f"  Cout de remediation (secteur) : {N_ENTITIES} entites x {COST_ENTITY[0]:.0f}-{COST_ENTITY[1]:.0f} M€")
print(f"  = {one_off[0]/1000:.1f} a {one_off[1]/1000:.1f} Md€ en one-off.")
print(f"  Economie recurrente (capital) : {annual_saving:.0f} M€/an.")
print(f"  Retour sur le SEUL portage du capital : {pay_lo:.0f} a {pay_hi:.0f} ans, et ces durees")
print("  sont des MAJORANTS puisque le benefice retenu est minore (voir section 3).")
print(f"  MAIS le ΔSCR est majoritairement de la PERTE EVITEE (pas seulement du portage) : en")
print(f"  ajoutant la baisse de la charge moyenne, le retour est bien plus court. Le portage")
print(f"  seul ({annual_saving:.0f} M€/an) est le PLANCHER conservateur du benefice.")

# =====================================================================================
titre("3. Ce que « portage » et « plancher » veulent dire, et le sens de la borne")
# =====================================================================================
# CETTE SECTION EXISTE PARCE QUE LES DEUX MOTS N'ONT PAS ETE COMPRIS AU POINT DU 7 AOUT, et la
# relecture montre que le memoire les avait effectivement melanges : il ecrivait que le RETOUR
# (une duree) etait un « plancher », alors que ce qui est minore est le BENEFICE. Minorer le
# benefice MAJORE la duree : le sens de la borne etait inverse. Aucun nombre ne change, seule
# la phrase, mais une borne annoncee a l'envers est une erreur de lecture, pas de style.
print("  PORTAGE. Detenir du capital n'est pas gratuit. Sous Solvabilite II le cout de detention")
print("  se lit dans la marge de risque, un pourcentage par an du capital immobilise. Liberer")
print(f"  {DDORA:.0f} M€ de SCR economise donc ce pourcentage CHAQUE ANNEE, et non une fois.")
print("\n  LE BENEFICE A DEUX COMPOSANTES, et ce script n'en monetise qu'une :")
print("    (1) le portage economise                       -> chiffre ci-dessous ;")
print("    (2) la charge de sinistralite evitee           -> chiffre ci-dessous aussi, desormais.")

# LA PERTE EVITEE ETAIT AFFIRMEE « MAJORITAIRE » SANS ETRE CHIFFREE. Un comparatif non chiffre
# n'est verifiable par personne : c'est la classe de defaut qui a produit les queues Bale. On la
# calcule sur le meme moteur que le Delta SCR (module partage, graines et resolution du 43).
m_C = cx.metriques_par_graine(lam=cx.LAM_C, g=cx.G_C, p_u=cx.PU_C, phi_cs=None)
m_NC = cx.metriques_par_graine(lam=cx.LAM_NC, g=cx.G_NC, p_u=cx.PU_NC, phi_cs=cx.PHICS_NC)
perte_evitee = float(m_NC[:, 1].mean() - m_C[:, 1].mean())
print(f"\n  Charge annuelle moyenne : {m_C[:, 1].mean():.0f} M€/an conforme contre "
      f"{m_NC[:, 1].mean():.0f} non conforme,")
print(f"  soit une PERTE EVITEE de {perte_evitee:.0f} M€/an, a comparer aux {annual_saving:.0f} M€/an")
print(f"  de portage : la perte evitee vaut {perte_evitee/annual_saving:.1f} fois le portage.")
print("  C'est ce qui justifie le mot « plancher », et il porte sur le BENEFICE.")
tot_benef = annual_saving + perte_evitee
print(f"\n  LES DEUX COMPOSANTES REUNIES : {tot_benef:.0f} M€/an, soit un retour de "
      f"{one_off[0]/tot_benef:.0f} a {one_off[1]/tot_benef:.0f} ans")
print(f"  contre {pay_lo:.0f} a {pay_hi:.0f} au portage seul. LA BORNE VA DONC DANS CE SENS-LA, et")
print("  c'est tout ce que le mot voulait dire : le benefice retenu est un MINORANT, donc la")
print("  duree affichee est un MAJORANT, et le compte reel est plus favorable, jamais moins.")
print("  A ASSUMER TOUTEFOIS : additionner les deux composantes ajoute un flux de compte de")
print("  resultat (la sinistralite evitee) a un cout du capital (le portage). C'est une")
print("  convention economique de ROI, defendable et courante, mais ce n'est PAS une sortie du")
print("  modele : le modele produit les deux termes separement, et c'est ainsi qu'ils sont")
print("  publies. Le retour au portage seul reste donc le chiffre de reference.")

# =====================================================================================
titre("4. Le taux de portage a change : 6 % historique, 4,75 % en vigueur")
# =====================================================================================
# DEUX TAUX CIRCULAIENT DANS LE PROJET SANS QUE RIEN NE LES RELIE : 6 % ici, 4,75 % dans les
# scripts 58 et 60 depuis la directive (UE) 2025/2. Le memoire citait le premier, le deck du
# 14 aout le second, pour la meme notion. La chaine publiee reste a 6 % (la calibration est
# gelee), et l'ecart est desormais imprime plutot que laisse a la sagacite du lecteur.
COC_REVISE = 0.0475                         # directive (UE) 2025/2, comme les scripts 58 et 60
print(f"  {'taux de portage':<34}{'valeur/an':>12}{'retour (majorant)':>22}")
for lib, taux in (("6 % (delegue 2015/35, historique)", COC),
                  ("4,75 % (directive UE 2025/2)", COC_REVISE)):
    val = taux * DDORA
    print(f"  {lib:<34}{val:>9.0f} M€{one_off[0]/val:>13.0f} a {one_off[1]/val:.0f} ans")
val_rev = COC_REVISE * DDORA
print(f"\n  Le taux revise abaisse le portage monetise de {100*(1-COC_REVISE/COC):.0f} % et allonge")
print(f"  le majorant du retour de {pay_hi:.0f} a {one_off[1]/val_rev:.0f} ans. AUCUNE CONCLUSION NE BOUGE,")
print("  et la raison est structurelle : la duree calculee est un MAJORANT, la composante")
print(f"  dominante du benefice ({perte_evitee:.0f} M€/an de perte evitee) n'etant pas monetisee ici.")
print("  La chaine publiee reste au taux historique : la calibration est gelee, et rejouer la")
print("  chaine pour un ecart qui ne renverse rien ferait bouger des nombres publies sans")
print("  qu'aucun deplacement soit attribuable a la correction.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. La conformite n'est pas un pur cout : elle libere du capital (ΔSCR), dont la valeur")
print(f"     recurrente est {annual_saving:.0f} M€/an (plancher, cout de portage seul).")
print("  2. Ce capital est CONCENTRE : prioriser la remediation par capital libere (frequence")
print("     et detection, les leviers CALIBRABLES) plutot que l'etaler uniformement.")
print("  3. Les leviers BORNES (propagation, accumulation) liberent aussi du capital mais se")
print("     remedient par la donnee (registre DORA) autant que par l'action directe.")
print("  4. C'est l'aboutissement du fil : du score de criticite abstrait a un arbitrage")
print("     economique chiffre, opposable a une direction financiere.")

# =====================================================================================
# figure J6
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 10.5,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
})
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime
# cette figure sur 7,27 pouces : tracee plus large, elle subissait une
# reduction qui faisait tomber ses etiquettes sous 6 points. Le rapport
# largeur sur hauteur est conserve, donc la figure occupe la meme place
# sur la page ; c'est son texte qui y prend plus de place.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.9, 3.09), gridspec_kw={"width_ratios": [1.1, 1]})

# La virgule decimale se pose sur le NOMBRE seul, jamais par un .replace sur la phrase.
_ratio_fr = f"{perte_evitee / annual_saving:.1f}".replace(".", ",")

# (a) capital libere et valeur annuelle par levier, priorise
names = [r[0] for r in rows]
dscr = [r[1] for r in rows]
cols = [BLUE if r[2] == "calibrable" else MUTED for r in rows]
yp = np.arange(len(rows))[::-1]
ax1.barh(yp, dscr, color=cols, alpha=0.9)
for y_, r in zip(yp, rows):
    ax1.text(r[1] + 60, y_, f"{r[1]:.0f} M€  →  {COC*r[1]:.0f} M€/an  ({r[2]})",
             va="center", fontsize=8.5, color=INK2)
ax1.set_yticks(yp); ax1.set_yticklabels(names, fontsize=9)
ax1.set_xlim(0, max(dscr) * 1.7)
ax1.set_xlabel("capital libéré ΔSCR (M€) ; bleu : calibrable, gris : borné", color=INK2, fontsize=9)
ax1.set_title("(a)  Le capital libéré par levier\n(et sa valeur annuelle à 6 %)", fontsize=11,
              color=INK, pad=8)

# (b) economie recurrente vs cout de remediation one-off
ax2.bar([0], [one_off[1]], width=0.5, color=ACCENT, alpha=0.35,
        label=f"remédiation one-off\n({N_ENTITIES} entités × {COST_ENTITY[0]:.0f}-{COST_ENTITY[1]:.0f} M€)")
ax2.bar([0], [one_off[0]], width=0.5, color=ACCENT, alpha=0.8)
ax2.bar([1], [annual_saving], width=0.5, color=GREEN, alpha=0.9,
        label="économie récurrente\n(capital, 6 %/an)")
ax2.text(0, one_off[1] + 400, f"{one_off[0]/1000:.0f}–{one_off[1]/1000:.0f} Md€", ha="center",
         fontsize=9, color=INK2)
ax2.text(1, annual_saving + 400, f"{annual_saving:.0f} M€/an", ha="center", fontsize=9, color=GREEN)
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["coût\n(one-off)", "bénéfice\n(récurrent)"], fontsize=9)
ax2.set_ylabel("M€", color=INK2)
ax2.legend(frameon=False, fontsize=8, loc="upper right")
# LE MOT « PLANCHER » NE QUALIFIE PAS UNE DUREE. Ce titre annoncait « 6-35 ans (plancher) »,
# donc l'inverse du sens reel : c'est le BENEFICE qui est minore, ce qui MAJORE la duree.
# L'inversion n'a pas ete comprise au point du 7 aout, et a juste titre.
ax2.set_title(f"(b)  Retour sur le portage seul : {pay_lo:.0f}–{pay_hi:.0f} ans, et c'est un "
              f"MAJORANT\n(la perte évitée, {_ratio_fr}× le portage, "
              "n'est pas monétisée ici)", fontsize=11, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.93])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "J6_roi_conformite.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
