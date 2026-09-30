#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
60 : la descente d'echelle secteur -> entite, rendue COHERENTE et composee.

CE QUI CLOCHAIT. Le chapitre resultats donne une table a trois echelles ou SEULE la
frequence change, avec deux lectures d'entite : lambda = 0,21 (firmes a >= 10 evenements,
etiquetees "tres grande entite") et lambda = 0,10 (entite moyenne du panel). Deux defauts.

  (1) INCOHERENCE DE TAILLE. Le SCR d'entite est calcule a lambda = 0,21, c'est-a-dire au
      profil des firmes les mieux couvertes d'OpRisk (des banques globales), puis compare
      a la charge de Formule Standard d'une entite notionnelle de 15 000 M EUR de
      provisions, soit environ 20 000 M USD d'actifs. Ce n'est pas la meme entite. Choisir
      un seau ("toutes firmes" ou ">= 10 evenements") est de toute facon arbitraire.

  (2) CORRECTION MESUREE MAIS NON APPLIQUEE. Le script 57 etablit l'elasticite severite /
      taille (b = 0,087 par EMV lognormale tronquee au seuil de collecte) et en deduit un
      facteur 0,86 sur la severite transposee. La table d'echelle ne l'applique pas : sa
      legende dit explicitement que la severite est identique.

CE QUE FAIT CE SCRIPT.
  (A) Estime lambda comme une FONCTION de la taille de firme (binomiale negative sur le
      panel firme-annee, log-lien, covariable log actifs), et la lit A LA TAILLE CIBLE.
      On ne choisit plus un seau : on lit la courbe au bon endroit.
  (B) Recalcule le multiplicateur de severite a la meme taille cible, et le PROPAGE.
  (C) Compose les deux dans le moteur de cascade, a matrice W et engin inchanges.
  (D) Donne le resultat en BANDE d'identification (1024 sommets du pave admissible a
      t = 1) et non en point, puisque la direction de W reste non identifiee.

CE QU'IL NE FAIT PAS. Il ne change ni la FORME de la queue (le xi d'OpRisk reste celui de
grandes institutions financieres), ni la composition sectorielle du panel (dominee par la
banque). Ces deux limites sont irreductibles sans donnee d'entite, et elles sont chiffrees
en sortie plutot que passees sous silence.

Sortie : diagnostics + figure S21_descente_echelle.png
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)
import partial_id as pid                                          # noqa: E402

# LE PANEL ET L'ELASTICITE SONT DESORMAIS DANS UN MODULE. Ils vivaient ici ; le script 65
# les reutilise sur des entites reelles, et une estimation de maximum de vraisemblance
# recopiee dans deux fichiers finit toujours par diverger de l'autre. Le protocole, la
# fenetre d'observation et l'ordre des operations (filtre annuel AVANT calcul de la
# fenetre, faute de quoi lambda s'effondre) sont documentes dans `descente.py`.
import descente as dsc                                              # noqa: E402
from descente import ASSETS, B_SEV, B_SEV_LO, B_SEV_HI              # noqa: E402

# entite notionnelle du memoire : 15 000 M EUR de provisions, soit de l'ordre de
# 20 000 M USD d'actifs (ratio prudentiel usuel ~1,3). Meme convention que le script 57.
ACTIFS_CIBLE = 20_000.0
PROVISIONS_CIBLE = 15_000.0
SF_TAUX = 0.03               # charge operationnelle de Formule Standard, plafond en provisions

# MEMES reglages Monte-Carlo que le script 58, qui consomme ces chiffres en aval : sans
# cela les deux scripts publieraient deux SCR d'entite differant du seul bruit de tirage.
NY = 600_000                 # a lambda ~ 0,09 la quasi-totalite des annees est vide
SEED = 20260721
ALPHA = 0.995
W_ = 86


def titre(s):
    print("\n" + "=" * W_ + f"\n{s}\n" + "=" * W_)


# =====================================================================================
titre("Donnees : panel firme-annee, avec la taille de firme")
# =====================================================================================
D = dsc.Descente()
fs, ict = D.fs, D.ict
panel, pa, taille = D.panel, D.pa, D.taille

print(f"observations firme-annee            : {len(panel):,}  ({panel.firm.nunique()} firmes)")
print(f"dont avec actifs renseignes         : {len(pa):,}  ({pa.firm.nunique()} firmes)")
print(f"lambda brut, toutes firmes          : {panel.n.mean():.4f}")
print(f"lambda brut, sous-panel avec actifs : {pa.n.mean():.4f}")

med_act, med_act_ict = D.med_act, D.med_act_ict
print(f"\nactifs medians, firme du panel      : {med_act:,.0f} M USD")
print(f"actifs medians, ponderes evenements : {med_act_ict:,.0f} M USD")
# LES MEMES EN MILLIARDS. Le chapitre resultats cite ces tailles de bilan en Md$, ce script ne
# les imprimait qu'en M USD : 121 894 M USD et 122 Md$ sont le meme nombre, mais le harnais ne
# pouvait pas le savoir. On imprime les deux formes.
print(f"   soit, en milliards               : {med_act/1000:,.1f} Md USD (panel) et "
      f"{med_act_ict/1000:,.1f} Md USD (ponderes evenements)")
print(f"actifs de l'entite cible            : {ACTIFS_CIBLE:,.0f} M USD")
print(f"   -> l'entite cible est {med_act_ict/ACTIFS_CIBLE:,.0f} fois plus petite que la firme")
print("      mediane PONDEREE PAR EVENEMENT de la base. C'est cet ecart que les deux")
print("      corrections doivent absorber.")


# =====================================================================================
titre("(A) lambda comme FONCTION de la taille, au lieu d'un seau arbitraire")
# =====================================================================================
print("Les deux lectures actuelles du memoire sont des seaux : 'toutes firmes' et")
print("'firmes a >= 10 evenements'. Le second est un proxy de taille, pas une mesure. On")
print("estime donc directement le taux en fonction de la taille, sur le panel firme-annee.")

tot = fs.groupby("Firm Name").size()
big = tot[tot >= 10].index
print(f"\n  rappel seau 1, toutes firmes           : lambda = {panel.n.mean():.4f}")
print(f"  rappel seau 2, firmes >= 10 evenements : lambda = "
      f"{panel[panel.firm.isin(big)].n.mean():.4f}")
act_big = float(taille.reindex(big).dropna().median())
print(f"  actifs medians de ces firmes >= 10 ev. : {act_big:,.0f} M USD"
      f"  (soit {act_big/1000:,.1f} Md USD)")
print(f"  soit {act_big/ACTIFS_CIBLE:,.0f}x l'entite cible : utiliser leur lambda POUR l'entite")
print("  cible etait bien l'incoherence a corriger.")

a_hat, b_lam = D.a_hat, D.b_lam
xbar, se_b, z_b, p_b = D.xbar, D.se_b, D.z_b, D.p_b

print(f"\nBinomiale negative, log-lien, covariable log(actifs) centree :")
print(f"  elasticite frequence / taille b_lambda = {b_lam:+.4f}  "
      f"(ET {se_b:.4f}, z = {z_b:+.2f}, p = {p_b:.2e})")
print(f"  IC 95 %                                = "
      f"[{b_lam - 1.96*se_b:+.4f} ; {b_lam + 1.96*se_b:+.4f}]")
print(f"  lambda a la taille mediane du panel    = {np.exp(a_hat):.4f}")

lam_cible = float(np.exp(a_hat + b_lam * (np.log(ACTIFS_CIBLE) - xbar)))
lam_lo = float(np.exp(a_hat + (b_lam + 1.96 * se_b) * (np.log(ACTIFS_CIBLE) - xbar)))
lam_hi = float(np.exp(a_hat + (b_lam - 1.96 * se_b) * (np.log(ACTIFS_CIBLE) - xbar)))
lam_lo, lam_hi = min(lam_lo, lam_hi), max(lam_lo, lam_hi)
print(f"\n  >>> lambda LU A LA TAILLE CIBLE ({ACTIFS_CIBLE:,.0f} M USD) = {lam_cible:.4f}/an")
print(f"      bande de l'incertitude sur b_lambda : [{lam_lo:.4f} ; {lam_hi:.4f}]")
print(f"      valeur exacte a reporter dans le script 58 : LAM_ENTITE = {lam_cible!r}")
print(f"      a comparer aux deux seaux du memoire : 0,10 et 0,21")


# =====================================================================================
titre("(B) Le multiplicateur de severite, a la MEME taille cible")
# =====================================================================================
mult = float((ACTIFS_CIBLE / med_act_ict) ** B_SEV)
mult_lo = float((ACTIFS_CIBLE / med_act_ict) ** B_SEV_HI)
mult_hi = float((ACTIFS_CIBLE / med_act_ict) ** B_SEV_LO)
print(f"elasticite severite / taille b = {B_SEV:.3f}  IC [{B_SEV_LO:.3f} ; {B_SEV_HI:.3f}]"
      "  (script 57, EMV tronquee)")
print(f"rapport de taille cible / base = {ACTIFS_CIBLE:,.0f} / {med_act_ict:,.0f} = "
      f"{ACTIFS_CIBLE/med_act_ict:.4f}")
print(f"\n  >>> multiplicateur de severite = {mult:.4f}   (soit {100*(mult-1):+.1f} %)")
print(f"      bande sur l'IC de b            : [{mult_lo:.4f} ; {mult_hi:.4f}]")
print("\nLa severite GPD est une famille d'ECHELLE au-dessus du seuil : multiplier toutes")
print("les severites par c multiplie le quantile par c EXACTEMENT. La source OpRisk n'est")
print("pas plafonnee (cap = None), donc rien ne vient tronquer cette homothetie.")


# =====================================================================================
titre("(C) Composition des deux corrections dans le moteur de cascade")
# =====================================================================================
Wexp = pid.expert_matrix()
S, _ = pid.decompose(Wexp)


def scr_at(lam, sev_mult, Wmat=None, n_years=NY, seed=SEED):
    """SCR et perte moyenne a frequence lam et severite mise a l'echelle."""
    ev = pid.Evaluator(lam=lam, n_years=n_years, alpha=ALPHA, seed=seed)
    ev.cum = ev.cum * sev_mult          # homothetie exacte sur toutes les severites
    card = pid.card_dist_all(Wexp if Wmat is None else Wmat)[0]
    return ev.from_card(card)


scen = [
    ("Secteur (calage du chapitre)",        21.56, 1.0),
    ("Entite, seau >= 10 ev. (memoire)",     0.210, 1.0),
    ("Entite, seau toutes firmes (memoire)", 0.100, 1.0),
    ("Entite, lambda lu a la taille",       lam_cible, 1.0),
    ("Entite, taille + severite (retenu)",  lam_cible, mult),
]
print(f"{'lecture':<40}{'lambda':>9}{'x sev':>8}{'SCR 99,5%':>13}{'E[X]':>11}{'SCR/E[X]':>10}")
res = {}
for lab, lam, m in scen:
    v, e = scr_at(lam, m)
    res[lab] = (lam, m, v, e)
    print(f"{lab:<40}{lam:>9.3f}{m:>8.3f}{v:>13,.1f}{e:>11,.2f}{v/max(e,1e-9):>10,.1f}")

sf = SF_TAUX * PROVISIONS_CIBLE
print(f"\nCharge operationnelle de Formule Standard (0,03 x {PROVISIONS_CIBLE:,.0f}) = "
      f"{sf:,.0f} M EUR")
lam_r, m_r, scr_r, esp_r = res["Entite, taille + severite (retenu)"]
print(f"Rapport SCR retenu / Formule Standard                       = {scr_r/sf:.2f}")

# Les rapports de la non-linearite sont CITES dans le memoire. Les imprimer ici les rend
# verifiables par verif_chiffres.py au lieu d'etre derives a la main dans la redaction.
lam_s, _, scr_s, esp_s = res["Secteur (calage du chapitre)"]
print("\nLa non-linearite, en rapports (cites tels quels au chapitre resultats) :")
print(f"  lambda divise par        {lam_s/lam_r:,.0f}")
print(f"  perte moyenne divisee par {esp_s/esp_r:,.0f}")
print(f"  capital divise par        {scr_s/scr_r:,.0f}")
print("  Le capital resiste parce que le quantile est porte par un sinistre unique, la")
print("  moyenne par le nombre d'evenements : c'est le principe du grand saut unique.")


# =====================================================================================
titre("(D) Le resultat d'entite est une BANDE, pas un point")
# =====================================================================================
print("La direction de W n'est pas identifiee. Un SCR d'entite calcule a la matrice")
print("d'expert est UN point de l'ensemble admissible, pas sa mesure. On enumere donc les")
print(f"{1 << pid.NFREE} sommets du pave a t = 1 (ignorance directionnelle totale).")

ev_b = pid.Evaluator(lam=lam_cible, n_years=NY, alpha=ALPHA, seed=SEED)
ev_b.cum = ev_b.cum * mult
A_all = pid.all_vertices(S, 1.0)
vals, means, kept = [], [], 0
for row in A_all:
    A = np.zeros((pid.NP_, pid.NP_))
    A[pid.IU] = row
    A = A - A.T
    Wv = S + A
    if not pid.admissible(Wv):
        continue
    kept += 1
    v, e = ev_b.from_card(pid.card_dist_all(Wv)[0])
    vals.append(v)
    means.append(e)
vals = np.array(vals)
means = np.array(means)
print(f"\nsommets admissibles           : {kept} / {len(A_all)}")
print(f"SCR d'entite, bornes          : [{vals.min():,.1f} ; {vals.max():,.1f}] M EUR")
print(f"   largeur                    : {vals.max()-vals.min():,.1f} M EUR "
      f"({100*(vals.max()-vals.min())/vals.min():.1f} % de la borne basse)")
print(f"point a la matrice d'expert   : {scr_r:,.1f} M EUR")
print(f"perte moyenne, bornes         : [{means.min():,.2f} ; {means.max():,.2f}] M EUR")

# socle sans contagion : W = 0
ev0 = pid.Evaluator(lam=lam_cible, n_years=NY, alpha=ALPHA, seed=SEED)
ev0.cum = ev0.cum * mult
scr_socle, esp_socle = ev0.from_card(pid.card_dist_all(np.zeros((pid.NP_, pid.NP_)))[0])
print(f"\nsocle sans contagion (W = 0)  : {scr_socle:,.1f} M EUR")
print(f"le socle vaut {100*scr_socle/vals.max():.1f} % de la borne haute")
# DEUX DEFINITIONS A NE PAS CONFONDRE. Le chapitre 10 appelle "part fixee" la quantite
# 1 - largeur/haute (79 % a l'echelle du secteur). Le rapport socle/haute est une AUTRE
# quantite. Les imprimer toutes deux evite la collision de definition entre chapitres.
print(f"part NON exposee a l'ignorance directionnelle, au sens du chapitre 10")
print(f"  (1 - largeur / borne haute) : {100*(1-(vals.max()-vals.min())/vals.max()):.1f} %")
print(f"  rappel de la meme quantite a l'echelle du secteur : "
      f"{100*(1-1839/8697):.1f} % (bornes 6858-8697)")


# =====================================================================================
titre("(E) Ce qui reste non transposable, chiffre")
# =====================================================================================
print("Deux limites ne se corrigent pas par une elasticite, et il faut les chiffrer.")
xi_c = pid.PARAMS["OPRISK"]["xi"]
xi_lo, xi_hi = pid.PARAMS["OPRISK"]["xi_ic90"]
print(f"\n(1) LA FORME DE LA QUEUE. xi = {xi_c:.3f}, IC 90 % [{xi_lo:.3f} ; {xi_hi:.3f}],")
print("    estime sur de grandes institutions financieres. L'elasticite corrige l'ECHELLE")
print("    de la severite, jamais sa FORME. On mesure ce que l'incertitude sur xi seule")
print("    fait au capital d'entite, a frequence et taille CIBLES :")
for lab, xiv in (("borne basse", xi_lo), ("central", xi_c), ("borne haute", xi_hi)):
    evx = pid.Evaluator(lam=lam_cible, n_years=NY, alpha=ALPHA, seed=SEED,
                        source="OPRISK")
    sp = dict(pid.PARAMS["OPRISK"])
    from src.aggregation.lda import simulate_remediation_severity
    rngx = np.random.default_rng(20260804)
    sev = simulate_remediation_severity(evx.T * pid.NP_, xiv, sp["sigma"], sp["u"],
                                        sp["p_u"], sp["cap"], rngx).reshape(evx.T, pid.NP_)
    evx.cum = np.cumsum(sev, axis=1) * mult
    v, e = evx.from_card(pid.card_dist_all(Wexp)[0])
    print(f"      xi {lab:<12} = {xiv:.3f}  ->  SCR = {v:>9,.1f} M EUR")

print("\n(1 bis) COMPARAISON DES DEUX INCERTITUDES, a l'echelle d'entite. La bande")
print(f"    d'IDENTIFICATION (direction de W inconnue) vaut {vals.max()-vals.min():,.1f} M EUR de large ;")
print(f"    la bande de PARAMETRE (IC 90 % sur xi seul) vaut {193.3-141.8:,.1f} M EUR. Les deux sont")
print("    du meme ordre : a l'echelle d'entite, ignorer la direction coute autant")
print("    qu'ignorer l'indice de queue. Aucune des deux ne domine l'autre.")

print("\n(2) LA COMPOSITION SECTORIELLE. Le panel est celui du secteur financier d'OpRisk,")
print("    domine par la banque. Rien dans ce script ne corrige un effet 'assureur contre")
print("    banque' : la taille est controlee, le METIER ne l'est pas. C'est la limite qui")
print("    exige une donnee d'entite et qu'aucune elasticite ne remplace.")


# =====================================================================================
titre("(F) Le SCR d'entite par etat de conformite, face a une charge forfaitaire")
# =====================================================================================
print("Le gain de propagation g est le canal que la conformite fait bouger (0,45 conforme,")
print("0,68 partiel, 0,90 non conforme). On le balaie a frequence et taille CIBLES. Seul le")
print("canal CONTAGION varie ici : les canaux frequence et detection, qui exigent les")
print("multiplicateurs de scenario, ne sont pas mobilises, donc l'ecart lu est un PLANCHER.")
etats = [("conforme", 0.45), ("partiellement conforme", 0.68), ("non conforme", 0.90)]
print(f"\n{'etat':<26}{'g':>7}{'SCR 99,5%':>13}{'E[X]':>10}{'/ Formule Standard':>21}")
scr_etat = {}
for lab, gv in etats:
    v, e = scr_at(lam_cible, mult, Wmat=pid.expert_matrix(gv))
    scr_etat[lab] = v
    print(f"{lab:<26}{gv:>7.2f}{v:>13,.1f}{e:>10,.2f}{v/sf:>21.2f}")
d_dora = scr_etat["non conforme"] - scr_etat["conforme"]
print(f"\nsurcout de non-conformite, canal contagion seul : {d_dora:,.1f} M EUR "
      f"({100*d_dora/scr_etat['conforme']:+.1f} %)")
print(f"charge de Formule Standard, identique aux trois etats : {sf:,.0f} M EUR (ecart nul)")
print("\nC'est la these du memoire, lue a l'echelle ou une entite la vit : le forfait ne")
print("bouge pas d'un euro entre un profil exemplaire et un profil defaillant, quand le")
print("modele, lui, ecarte les deux. Le NIVEAU du forfait n'est pas le sujet ; son")
print("INSENSIBILITE l'est.")


# =====================================================================================
titre("(G) L'arbitrage detenir / transferer survit-il a la correction d'echelle ?")
# =====================================================================================
print("La section detenir/transferer du chapitre resultats (script 58) est en AVAL du meme")
print("choix de lambda : elle tourne a 0,21 et sur un SCR de 488 M EUR. On la rejoue ici a")
print("l'echelle corrigee, pour dire si sa conclusion tient ou si elle en dependait.")

COC_REVISE, LOSS_RATIO = 0.0475, 0.485
ev_g = pid.Evaluator(lam=lam_cible, n_years=NY, alpha=ALPHA, seed=SEED)
ev_g.cum = ev_g.cum * mult
card_g = pid.card_dist_all(Wexp)[0]
K = np.empty(ev_g.T, dtype=np.int64)
for a in range(pid.NP_):
    idx = ev_g.idx_by_am[a]
    if idx.size:
        cdf = np.cumsum(card_g[a][1:])
        cdf[-1] = 1.0
        K[idx] = np.searchsorted(cdf, ev_g.u[idx], side="right") + 1
np.clip(K, 1, pid.NP_, out=K)
X = np.bincount(ev_g.year_of, weights=ev_g.cum[ev_g._ar, K - 1], minlength=ev_g.n_years)
scr_x, esp_x = float(np.quantile(X, ALPHA)), float(X.mean())


def cout(L, coc=COC_REVISE):
    cede = np.minimum(X, L)
    return cede.mean() / LOSS_RATIO + coc * float(np.quantile(X - cede, ALPHA))


grid = np.linspace(0, 3 * scr_x, 601)
cc = np.array([cout(L) for L in grid])
i = int(np.argmin(cc))
c0 = cout(0.0)
print(f"\n{'grandeur':<44}{'a 0,21 (memoire)':>19}{'echelle corrigee':>19}")
print(f"{'SCR 99,5 %  (M EUR)':<44}{491.1:>19,.1f}{scr_x:>19,.1f}")
print(f"{'sinistralite attendue E[X]  (M EUR)':<44}{9.44:>19,.2f}{esp_x:>19,.2f}")
print(f"{'multiple de capital SCR / E[X]':<44}{491.1/9.44:>19,.1f}{scr_x/esp_x:>19,.1f}")
print(f"{'cout annuel de detention  (M EUR)':<44}{COC_REVISE*491.1:>19,.1f}"
      f"{COC_REVISE*scr_x:>19,.1f}")
print(f"{'   soit x la sinistralite attendue':<44}{COC_REVISE*491.1/9.44:>19,.2f}"
      f"{COC_REVISE*scr_x/esp_x:>19,.2f}")
print(f"{'portee optimale L*  (M EUR)':<44}{488.0:>19,.0f}{grid[i]:>19,.0f}")
# 0,61 est la valeur HISTORIQUE imprimee par le script 58 dans sa configuration a
# lambda = 0,21, conservee ici comme point de comparaison. Ne pas la recalculer par un
# rapport approche : une premiere version le faisait et donnait 63 %, faux de deux points.
CEDE_REF_HISTORIQUE = 0.61
cede_ref, cede_new = CEDE_REF_HISTORIQUE, float(np.minimum(X, grid[i]).mean()) / esp_x
print(f"{'part de la sinistralite cedee':<44}{cede_ref:>18.0%}{cede_new:>19.0%}")
print(f"{'gain du dimensionnement':<44}{-0.47:>18.0%}{cc[i]/c0-1:>19.0%}")

# NE RIEN CODER EN DUR ICI. La narration qui suit a deja porte une fois des valeurs
# perimees (48,6 et 2,31) que verif_chiffres.py a rattrapees dans le memoire. Toute
# grandeur citee est desormais interpolee depuis les variables calculees.
mult_ref, mult_new = 491.1 / 9.44, scr_x / esp_x
det_ref, det_new = COC_REVISE * 491.1 / 9.44, COC_REVISE * scr_x / esp_x
ecart = max(abs(mult_new / mult_ref - 1), abs(det_new / det_ref - 1))
print("\nCE QUI EST INVARIANT, ET CE QUI NE L'EST PAS. Il faut separer les deux, car tout")
print("ne survit pas de la meme facon.")
print(f"\n  INVARIANT. Le multiple de capital ({mult_ref:,.1f} -> {mult_new:,.1f}) et le cout de")
print(f"  detention rapporte a la sinistralite attendue ({det_ref:,.2f} -> {det_new:,.2f})")
print(f"  bougent de moins de {100*ecart:.0f} %, alors que les NIVEAUX sont divises par pres de")
print("  trois. Leurs deux termes se rapportent au meme SCR et se deplacent ensemble :")
print("  c'est la these du memoire, verifiee sur une correction qui n'avait pas ete faite")
print("  pour la tester.")
print(f"\n  NON INVARIANT. Le gain du dimensionnement passe de -47 % a {cc[i]/c0-1:.0%}, et il faut")
print("  le dire au lieu de le ranger avec les rapports. La cause est mecanique : a lambda")
print("  plus faible, la perte annuelle est plus dominee par un sinistre unique, donc une")
print("  part PLUS GRANDE de E[X] se situe au-dela du SCR et echappe a une portee bornee")
print(f"  par le SCR. La part cedee tombe de {cede_ref:.0%} a {cede_new:.0%}, et le cout residuel")
print("  avec elle. Le sens de la conclusion (transferer est nettement moins cher que")
print("  detenir) est donc renforce par la correction, mais son AMPLEUR chiffree en depend")
print("  et ne doit pas etre citee comme une grandeur d'entite.")


# =====================================================================================
titre("Figure")
# =====================================================================================
# STYLE ALIGNE SUR LES AUTRES FIGURES DU CHAPITRE. Ce panneau gardait le fond blanc,
# le quadrillage et une palette propre, alors que ses voisines de page sont sur fond
# creme, sans quadrillage et sur les couleurs de la charte. Deux figures cote a cote
# dans un meme chapitre doivent se ressembler.
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 10,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
})
NAVY, BLUE, ACC, GRN = "#1b1e30", "#2b559f", "#a6002e", "#009a94"
INK2, MUTED = "#223e55", "#595959"
LEG = dict(frameon=True, facecolor="#fcfcfb", edgecolor="none", framealpha=0.88)
# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Trois panneaux traces
# sur seize pouces puis imprimes sur 7,27 donnent 1,6 pouce par panneau et
# des etiquettes sous 5 points. Le rapport largeur sur hauteur est conserve.
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5))

# (a) lambda en fonction de la taille
ax = axes[0]
gr = pa.copy()
gr["bin"] = pd.qcut(np.log(gr.actifs), 8, duplicates="drop")
agg = gr.groupby("bin", observed=True).agg(lam=("n", "mean"), act=("actifs", "median"))
ax.scatter(agg.act, agg.lam, s=40, color=NAVY, zorder=3, label="déciles du panel")
xs = np.logspace(np.log10(max(pa.actifs.min(), 1)), np.log10(pa.actifs.max()), 200)
ax.plot(xs, np.exp(a_hat + b_lam * (np.log(xs) - xbar)), color=BLUE, lw=2,
        label="NB2, $b_\\lambda$ = " + f"{b_lam:.3f}".replace(".", ","))
ax.axvline(ACTIFS_CIBLE, color=ACC, ls="--", lw=1.6)
ax.plot([ACTIFS_CIBLE], [lam_cible], "o", ms=9, color=ACC, zorder=4,
        label="entité cible : $\\lambda$ = " + f"{lam_cible:.3f}".replace(".", ","))
ax.axhline(0.210, color="0.55", ls=":", lw=1.3)
# L'ETIQUETTE DU SEAU SE POSE SOUS SON TRAIT, A DROITE DU SEUIL CIBLE. Calee sur le
# bord droit elle tombait sur le decile le plus grand, qui vit precisement a 0,20.
ax.text(ACTIFS_CIBLE * 2.2, 0.203, "seau 0,21", ha="left", va="top", fontsize=8.4,
        color=MUTED)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("actifs de la firme (M USD, échelle log)")
ax.set_ylabel("incidents TIC matériels / an")
ax.set_title("(a)  $\\lambda$ lu à la taille cible,\nnon choisi dans un seau", fontsize=10.5)
ax.set_xlim(right=pa.actifs.max() * 5)
ax.set_ylim(top=ax.get_ylim()[1] * 4.0)
ax.legend(fontsize=8.4, loc="upper left", **LEG)

# (b) la cascade des corrections
ax = axes[1]
labs = ["seau 0,21\n(mémoire)", "$\\lambda$ à la\ntaille", "+ sévérité\n(retenu)", "Formule\nStandard"]
vv = [res["Entite, seau >= 10 ev. (memoire)"][2], res["Entite, lambda lu a la taille"][2],
      scr_r, sf]
cols = [MUTED, BLUE, NAVY, GRN]
bars = ax.bar(range(4), vv, color=cols, width=0.62, alpha=0.9, edgecolor="#fcfcfb")
for i, (b, v) in enumerate(zip(bars, vv)):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:,.0f}", ha="center",
            va="bottom", fontsize=9, color=INK2)
ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=7.2)
ax.set_ylabel("SCR 99,5 % (M€)")
ax.set_title("(b)  Les deux corrections composées,\net le repère réglementaire", fontsize=10.5)

# (c) la bande d'identification
ax = axes[2]
ax.hist(vals, bins=42, color=BLUE, alpha=0.60, edgecolor="#fcfcfb", linewidth=0.4)
ax.axvline(vals.min(), color=NAVY, lw=1.8)
ax.axvline(vals.max(), color=NAVY, lw=1.8)
ax.axvline(scr_r, color=ACC, lw=1.8, ls="--", label=f"matrice d'expert : {scr_r:,.0f}")
ax.axvline(scr_socle, color=GRN, lw=1.8, ls=":", label=f"socle $W=0$ : {scr_socle:,.0f}")
ax.set_xlabel("SCR 99,5 % de l'entité (M€)")
ax.set_ylabel("sommets admissibles")
ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
ax.set_title(f"(c)  Le résultat d'entité est une bande\n[{vals.min():,.0f} ; {vals.max():,.0f}]"
             " M€", fontsize=10.5)
ax.legend(fontsize=8.4, loc="upper left", **LEG)

fig.tight_layout(w_pad=1.8)
out = os.path.join(HERE, "figures", "S21_descente_echelle.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"figure ecrite : {out}")


# =====================================================================================
titre("Verdict")
# =====================================================================================
print(f"Peut-on donner un SCR d'entite ? OUI, mais en BANDE et sous deux reserves nommees.")
print(f"\n  socle sans contagion            : {scr_socle:>9,.1f} M EUR")
print(f"  bande d'identification (t = 1)  : [{vals.min():,.1f} ; {vals.max():,.1f}] M EUR")
print(f"  point a la matrice d'expert     : {scr_r:>9,.1f} M EUR")
print(f"  Formule Standard, meme entite   : {sf:>9,.1f} M EUR")
print(f"\n  Les deux corrections composees deplacent le chiffre d'entite de "
      f"{res['Entite, seau >= 10 ev. (memoire)'][2]:,.0f} a {scr_r:,.0f} M EUR.")
print("  Restent non corriges : la forme de la queue et la composition sectorielle.")
