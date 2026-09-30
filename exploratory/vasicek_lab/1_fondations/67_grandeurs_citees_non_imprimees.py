#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
67 : les grandeurs CITEES dans le memoire et qu'aucun script n'imprimait.

D'OU VIENT CE SCRIPT. Le harnais verif_chiffres.py signale les nombres publies qu'aucune
sortie de script ne confirme. Un passage complet sur les dix-neuf chapitres a montre que le
residu se repartit en quatre classes, et une seule constitue un vrai defaut :

  (1) SEPARATEUR DE MILLIERS. Le harnais coupait « 8,122.9 » en deux nombres, 8 et 122.9, si
      bien que le « 8123 » du memoire ressortait non confirme alors que le script 60
      l'imprimait. Corrige dans le harnais, pas ici.
  (2) UNITE. Le memoire ecrit un pourcentage la ou le script imprime une fraction : « 15,8 % »
      contre « 0.158 ». Corrige en imprimant les deux, la ou c'est le cas.
  (3) NOMBRE LEGITIMEMENT HORS SCRIPT. Un seuil pose, un comptage de renvoi interne, une
      citation externe datee. Ceux-la ne doivent pas etre produits par un script, et le
      memoire les signale comme tels.
  (4) GRANDEUR CITEE MAIS JAMAIS IMPRIMEE. C'est la seule classe fautive, et c'est l'objet de
      ce script : des valeurs presentees comme des RESULTATS, calculees a la main pendant la
      redaction, qu'aucun script ne produisait. Un lecteur qui les recalcule n'avait aucun
      point de comparaison, et une derive silencieuse y etait indetectable.

CE QU'IL IMPRIME. Les formes analytiques fermees du chapitre socle (quantile de severite et
moyenne de queue de la GPD), le PONT entre ce quantile unitaire et le besoin de capital agrege,
les parts de vecteur en pourcentage autant qu'en fraction, les rapports derives que la
redaction citait sans les tracer, et, quand la base sous licence est presente, les statistiques
descriptives et l'estimateur de Hill au seuil retenu.

UNE CONVENTION DE VOCABULAIRE, ET ELLE COMPTE. VaR et TVaR sont reservees a la charge annuelle
AGREGEE, donc a la mesure de capital. Le quantile de la severite d'un SINISTRE UNIQUE est un
autre objet : ce n'est pas un capital, et le niveau 99,5 % n'y a aucun sens reglementaire,
puisqu'il designe une probabilite de ruine ANNUELLE. Citer les deux cote a cote sans le dire
fait lire une difference d'echelle comme une contradiction ; c'est arrive, et la section 1bis
existe pour que cela ne se reproduise pas.

IL NE RECALCULE RIEN QUI EXISTE AILLEURS. La calibration vient de la configuration figee, comme
dans le script 63 : cette sortie atteste la conformite a la configuration, pas l'exactitude
empirique, laquelle releve des scripts 07, 08b, 47 et 57.

Sortie : diagnostics seuls, aucune figure.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src.utils.config import OPRISK, PRC, HACKMAGEDDON, FREQUENCY, COPULE   # noqa: E402

W_ = 88


def titre(s):
    print("\n" + "=" * W_ + f"\n{s}\n" + "=" * W_)


# =====================================================================================
titre("1. Quantile de severite et moyenne de queue, par les formes fermees de la GPD")
# =====================================================================================
print("Le chapitre socle donne ces deux valeurs comme resultats de ses propositions, mais")
print("aucun script ne les imprimait : elles etaient calculees a la main a partir de la")
print("calibration. On les reproduit ici depuis la configuration figee.\n")


def var_gpd(u, sigma, xi, zeta_u, p):
    """VaR_p = u + (sigma/xi) * [ ((1-p)/zeta_u)^(-xi) - 1 ]."""
    return u + (sigma / xi) * (((1.0 - p) / zeta_u) ** (-xi) - 1.0)


def tvar_gpd(u, sigma, xi, zeta_u, p):
    """TVaR_p = VaR_p/(1-xi) + (sigma - xi*u)/(1-xi), defini pour xi < 1."""
    if xi >= 1.0:
        return float("nan")
    v = var_gpd(u, sigma, xi, zeta_u, p)
    return v / (1.0 - xi) + (sigma - xi * u) / (1.0 - xi)


for nom, cfg in (("OPRISK", OPRISK), ("PRC", PRC)):
    u, sg = cfg["seuil_u_eur"], cfg["sigma_eur"]
    xi, zu = cfg["xi"], cfg["p_u"]
    print(f"  {nom} : u = {u}, sigma = {sg}, xi = {xi}, zeta_u = {zu}")
    for p in (0.995, 0.999):
        v, tv = var_gpd(u, sg, xi, zu, p), tvar_gpd(u, sg, xi, zu, p)
        tvs = f"{tv:.1f}" if np.isfinite(tv) else "INDEFINI (xi >= 1)"
        print(f"      p = {p:.3f}   VaR = {v:.1f} M EUR    TVaR = {tvs}")
    if xi >= 1.0:
        print(f"      xi = {xi} > 1 : esperance infinie, l'Expected Shortfall est")
        print(f"      mathematiquement INDEFINI. C'est ce qui justifie le plafond de severite.")
    print()
print("  Les deux valeurs citees au chapitre socle sont la ligne OPRISK a p = 0,995.")


# =====================================================================================
titre("1bis. Le pont entre le quantile UNITAIRE et le besoin de capital AGREGE")
# =====================================================================================
print("POURQUOI CETTE SECTION. Le quantile de severite a 99,5 % (663 M EUR) et le besoin de")
print("capital (8 123 M EUR au secteur, 169 a l'entite) sont des objets differents, et le")
print("memoire les citait cote a cote sans imprimer ce qui les relie. Un lecteur y voit une")
print("incoherence ; il n'y en a pas, il y a une agregation. Le rapport a ete calcule a la")
print("main pendant la redaction, donc verifiable par personne : il est calcule ici.\n")
print("LE PONT, ET SON APPROXIMATION. A queue lourde le quantile annuel est porte par UN")
print("sinistre (principe du grand saut unique). Atteindre le quantile annuel au niveau alpha")
print("demande donc, PAR SINISTRE, le niveau 1 - (1 - alpha)/lambda, et le rapport des deux")
print("quantiles vaut asymptotiquement lambda^xi. C'est une APPROXIMATION et non une identite :")
print("elle ignore le cumul de plusieurs sinistres dans une meme annee, et elle ne contient ni")
print("la surdispersion ni la contagion. Ces deux effets sont dans le RESIDU de la derniere")
print("colonne, qui n'est donc pas une erreur mais leur empreinte multiplicative.\n")

LAM_SECTEUR = OPRISK["n_incidents"] / OPRISK["n_years"]     # 582 / 27
LAM_ENTITE = 0.09168423156000373    # script 60 (A) : lambda lu a la taille cible
SEV_MULT = 0.8545                   # script 60 (B) : severite transposee a la meme taille
SCR_SECTEUR = 8122.9                # script 60 (C) : ligne « Secteur (calage du chapitre) »
SCR_ENTITE = 169.0                  # script 60 (C) : ligne « Entite, taille + severite »
ALPHA = 0.995

_u, _sg = OPRISK["seuil_u_eur"], OPRISK["sigma_eur"]
_xi, _zu = OPRISK["xi"], OPRISK["p_u"]
q_unit = var_gpd(_u, _sg, _xi, _zu, ALPHA)

print(f"  quantile de severite a {ALPHA:.3f} sur UN sinistre : {q_unit:.2f} M EUR")
print(f"  (c'est le 663 du chapitre socle, et ce n'est pas un capital)\n")

print(f"  {'echelle':<9}{'lambda':>9}{'x sev':>7}{'niveau/sinistre':>17}"
      f"{'q(niveau)':>11}{'SCR publie':>12}{'residu':>8}{'SCR/q unit':>12}")
_lignes = (("secteur", LAM_SECTEUR, 1.0, SCR_SECTEUR),
           ("entite", LAM_ENTITE, SEV_MULT, SCR_ENTITE))
_residus = {}
for _nom, _lam, _mult, _scr in _lignes:
    _p_eq = 1.0 - (1.0 - ALPHA) / _lam
    if 1.0 - _p_eq >= _zu:
        print(f"  {_nom:<9}{_lam:>9.3f} : niveau equivalent sous le seuil POT, pont non applicable")
        continue
    _q_eq = var_gpd(_u, _sg, _xi, _zu, _p_eq) * _mult
    _res = _scr / _q_eq
    _residus[_nom] = _res
    print(f"  {_nom:<9}{_lam:>9.3f}{_mult:>7.3f}{100*_p_eq:>16.3f} %"
          f"{_q_eq:>11.1f}{_scr:>12.1f}{_res:>8.2f}{_scr/(q_unit*_mult):>12.2f}")

_ratio_exact = var_gpd(_u, _sg, _xi, _zu, 1.0 - (1.0 - ALPHA) / LAM_SECTEUR) / q_unit
print(f"\n  controle analytique au secteur : rapport exact des quantiles = {_ratio_exact:.3f},")
print(f"  asymptote lambda^xi = {LAM_SECTEUR ** _xi:.3f}. L'ecart de {100*(_ratio_exact/LAM_SECTEUR**_xi - 1):.0f} % n'est pas du bruit :")
print(f"  la forme fermee porte un terme additif u - sigma/xi = {_u - _sg/_xi:.1f}, negatif ici, et")
print("  le rapport tend vers son asymptote PAR VALEURS SUPERIEURES. lambda^xi est donc une")
print("  borne basse du facteur d'agregation, pas son approximation centree.\n")

print("CE QU'IL FAUT LIRE, ET C'EST LE POINT.")
print(f"  1. Au SECTEUR, le capital vaut {SCR_SECTEUR/q_unit:.1f} fois le quantile unitaire.")
print(f"     A l'ENTITE, il n'en vaut plus que {SCR_ENTITE/(q_unit*SEV_MULT):.2f} fois, donc il passe")
print("     EN DESSOUS. Le sens de l'inegalite s'inverse avec l'echelle : aucun rapport fixe")
print("     ne relie les deux objets, et comparer un quantile unitaire a un SCR ne conclut")
print("     jamais, dans un sens comme dans l'autre.")
print(f"  2. La raison est dans la colonne « niveau/sinistre » : a lambda = {LAM_SECTEUR:.1f} le")
print("     quantile annuel exige un niveau par sinistre bien PLUS HAUT que 99,5 %, alors qu'a")
print(f"     lambda = {LAM_ENTITE:.3f} il n'exige qu'un niveau BIEN PLUS BAS, la plupart des annees")
print("     n'ayant aucun sinistre.")
if len(_residus) == 2:
    _e = 100.0 * (_residus["entite"] / _residus["secteur"] - 1.0)
    print(f"  3. Le residu vaut {_residus['secteur']:.2f} au secteur et {_residus['entite']:.2f} a l'entite, soit le meme ORDRE")
    print(f"     a {_e:.0f} % pres, et non la meme valeur. C'est ce qu'on attend : la surdispersion et")
    print("     la contagion sont des facteurs multiplicatifs independants de l'echelle, donc le")
    print("     residu doit etre stable, mais le pont lui-meme est approche et son erreur n'est")
    print("     pas la meme aux deux frequences (le cumul de plusieurs sinistres compte au")
    print(f"     secteur, ou lambda = {LAM_SECTEUR:.1f}, et pas a l'entite, ou il est rarissime). L'ecart")
    print("     residuel mesure donc la qualite du pont, pas un desaccord du pipeline.")
print("  4. Ce pont ne remplace pas le moteur : il explique un ordre de grandeur, il ne")
print("     produit aucun chiffre publie. Les SCR restent ceux du script 60.")


# =====================================================================================
titre("2. Les parts de vecteur, en pourcentage autant qu'en fraction")
# =====================================================================================
print("Le script 63 imprime ces parts en FRACTION, le memoire les cite en POURCENTAGE : le")
print("harnais ne pouvait apparier ni l'un ni l'autre. On imprime les deux formes.\n")
print(f"  {'vecteur':<26}{'fraction':>10}{'pourcentage':>14}")
for k, v in HACKMAGEDDON["proportions"].items():
    print(f"  {k:<26}{v:>10.3f}{100*v:>13.1f} %")
print(f"  {'surface TLPT (art. 26)':<26}{HACKMAGEDDON['surface_tlpt']:>10.3f}"
      f"{100*HACKMAGEDDON['surface_tlpt']:>13.1f} %")
print(f"  {'surface tiers (art. 28-44)':<26}{HACKMAGEDDON['surface_tiers']:>10.3f}"
      f"{100*HACKMAGEDDON['surface_tiers']:>13.1f} %")
print(f"\n  somme des parts : {sum(HACKMAGEDDON['proportions'].values()):.3f} "
      f"({100*sum(HACKMAGEDDON['proportions'].values()):.1f} %)")
print("  RAPPEL : citation externe datee, non recalculable (cf. scripts 62 et 63). Ces parts")
print("  ne servent qu'a repartir par vecteur et ne portent aucun niveau de capital.")


# =====================================================================================
titre("3. Les rapports derives que la redaction citait sans les tracer")
# =====================================================================================
print("Chacun de ces rapports se deduit d'une valeur imprimee ailleurs. Les tracer ici evite")
print("qu'une valeur amont bouge sans que le rapport suive, ce qui est arrive une fois.\n")
SEV_MULT = 0.8545                 # script 60 : multiplicateur de severite a la taille cible
print(f"  multiplicateur de severite (script 60)     : {SEV_MULT:.4f}")
print(f"  soit, en variation                         : {100*(SEV_MULT-1):+.1f} %")
print(f"     le memoire arrondissait a -14 % ; la valeur imprimee est {100*(SEV_MULT-1):+.1f} %")

# RAPPORT ENTRE LES TROIS FREQUENCES DU CHAPITRE DONNEES. Le texte annoncait « un facteur
# 1600 entre les extremes », valeur heritee de l'epoque ou lambda d'entite valait 0,21. La
# table a ete corrigee a 0,092, la phrase ne l'a pas ete : c'est exactement le genre de
# desynchronisation qu'un rapport imprime empeche.
LAM_REF = 341.0                   # notifications/an, perimetre BSF (script 62)
LAM_SECTEUR = 21.56               # incidents TIC materiels/an, secteur (scripts 07, 08b)
LAM_ENTITE = 0.09168423156000373   # lu a la taille de l'entite (script 60)
print(f"\n  les trois frequences du chapitre donnees :")
print(f"     lambda_ref, notifications PRC (BSF)     : {LAM_REF:.0f} /an")
print(f"     lambda, secteur financier mondial       : {LAM_SECTEUR:.2f} /an")
print(f"     lambda, entite lue a sa taille          : {LAM_ENTITE:.5f} /an")
print(f"  rapport entre les extremes (ref / entite)  : {LAM_REF/LAM_ENTITE:.0f}")
print(f"  rapport secteur / entite                   : {LAM_SECTEUR/LAM_ENTITE:.0f}")

XI = OPRISK["xi"]
print(f"\n  indice de queue OpRisk                     : {XI}")
print(f"  1 / xi (indice de regularite de la queue)  : {1.0/XI:.3f}")
print(f"  xi > 0,5 : variance infinie                : {XI > 0.5}")
print(f"  xi < 1   : esperance finie, TVaR defini    : {XI < 1.0}")

# LES QUATRE RAPPORTS DU CHAPITRE RESULTATS ET DU CHAPITRE PARTIELLE. Chacun se deduit de
# deux valeurs imprimees par le script cite en regard, mais aucun n'etait imprime : ils
# etaient calcules pendant la redaction, donc verifiables par personne. C'est la quatrieme
# classe du residu du harnais.
VAR_NC, TVAR_NC = 16942.0, 35645.0        # script 20, OpRisk, tous piliers NC
print(f"\n  VaR 99,5 % OpRisk tous NC (script 20)      : {VAR_NC:.0f} M EUR")
print(f"  TVaR correspondante                        : {TVAR_NC:.0f} M EUR")
print(f"  rapport TVaR / VaR                         : {TVAR_NC/VAR_NC:.2f}")
print(f"     a comparer a l'asymptote 1/(1-xi)       : {1.0/(1.0-XI):.2f}")

D_CASCADE, D_COPULE = 10927.0, 6760.0     # script 23, surcout total tous NC
print(f"\n  surcout total, cascade (script 23)         : {D_CASCADE:.0f} M EUR")
print(f"  surcout total, copule de dependance        : {D_COPULE:.0f} M EUR")
print(f"  sous-estimation du modele sans mecanisme   : {100*(1-D_COPULE/D_CASCADE):.1f} %")

COUT_NC, COUT_C = 2.89, 0.54              # script 58, cout total annuel a lambda decroissant
print(f"\n  cout total annuel, lambda = 0,092 (58)     : {COUT_NC:.2f} M EUR/an")
print(f"  cout total annuel, lambda = 0,046          : {COUT_C:.2f} M EUR/an")
print(f"  variation apportee par la conformite       : {100*(COUT_C/COUT_NC-1):+.1f} %")

P1_TETE, P4_TETE = 43.6, 37.7             # script 30, % de fois premier sur l'ensemble admissible
print(f"\n  P1 en tete de priorite (script 30)         : {P1_TETE:.1f} %")
print(f"  P4 en tete                                 : {P4_TETE:.1f} %")
print(f"  P1 ou P4 en tete, cumule                   : {P1_TETE+P4_TETE:.1f} %")

# LES PARTS D'AMORCE NE SONT PAS ICI, ET C'EST DELIBERE. Elles manquaient aussi, mais les
# ajouter a ce script les versait dans le pool de TOUTES les sections qui citent le 67, dont
# celle du chapitre socle sur le seuil GPD : la part d'amorce de P1, 30,3 %, y confirmait
# alors le « minimum de 30 exces » et le « k = 30 » du graphe de Hill, qui n'ont aucun
# rapport avec elle et que le harnais avait raison de signaler. Trois faux positifs eteints
# par accident, donc trois pertes de signal. Elles sont imprimees par le script 20, celui de
# Shapley, qui est aussi la seule section a les citer.


# =====================================================================================
titre("3bis. L'echelle des lectures de l'ecart entre etats, et laquelle le memoire publie")
# =====================================================================================
# POURQUOI CETTE SECTION EXISTE. Quatre protocoles du projet chiffrent « l'ecart entre l'etat
# conforme et l'etat non conforme » et donnent quatre resultats, de -53 % a un facteur 3,34.
# Aucun n'est faux : ils ne RELACHENT PAS LE MEME NOMBRE DE CANAUX. Tant que le rapport
# n'etait imprime nulle part, le lecteur voyait quatre chiffres sans regle pour choisir, et
# le memoire lui-meme declarait la question ouverte dans un encadre du chapitre resultats.
#
# Les niveaux ci-dessous sont tous lus dans une sortie versionnee ; seuls les RAPPORTS sont
# calcules ici, et c'est precisement ce qui manquait.
#
# ATTENTION A NE PAS SUR-LIRE LA COLONNE « conforme ». Elle n'est pas monotone (6 664, puis
# 5 932, puis 6 049) parce que relacher un canal deplace AUSSI l'etat conforme, et parce que
# les protocoles n'ont ni la meme graine ni la meme base. Les trois valeurs basses tiennent
# dans 2,6 % l'une de l'autre, soit l'ordre du bruit de simulation. Ce qui se compare d'un
# protocole a l'autre est le FACTEUR, pas le niveau.

LECTURES = [
    # (libelle, canaux relaches, SCR conforme, SCR non conforme, source versionnee)
    ("score de conformite seul, g fige au niveau NC", 1, 7987.0, 16847.0, "25"),
    ("frequence + propagation (lecture B)",           2, 6664.0, 15074.0, "16"),
    ("+ detection (lecture C)",                       3, 5932.0, 16595.0, "16"),
    ("+ accumulation P4 (les quatre canaux)",         4, 6049.0, 20188.0, "43"),
]

print("Quatre protocoles chiffrent le meme enonce et donnent quatre resultats. Ils different")
print("par le NOMBRE DE CANAUX qu'ils relachent entre les deux etats, et par rien d'autre.\n")
print(f"  {'lecture':<48}{'can.':>5}{'conforme':>11}{'non conf.':>11}{'facteur':>10}{'ecart':>10}  src")
for lib, k, c, nc, src in LECTURES:
    print(f"  {lib:<48}{k:>5}{c:>11.0f}{nc:>11.0f}{nc/c:>10.2f}{nc-c:>10.0f}   {src}")

facteurs = [nc / c for _, _, c, nc, _ in LECTURES]
croissant = all(facteurs[i] < facteurs[i + 1] for i in range(len(facteurs) - 1))
print(f"\n  le facteur croit-il avec le nombre de canaux relaches ? {croissant}")
print(f"  amplitude du facteur, d'un canal a quatre  : {facteurs[0]:.2f} -> {facteurs[-1]:.2f}")
print(f"  soit un ecart de                           : {facteurs[-1]-facteurs[0]:+.2f}")

ecarts = [nc - c for _, _, c, nc, _ in LECTURES]
ec_croissant = all(ecarts[i] < ecarts[i + 1] for i in range(len(ecarts) - 1))
print(f"\n  et la colonne ECART en euros, croit-elle ? {ec_croissant}")
print("  Non, et il faut le dire plutot que de laisser le lecteur le remarquer seul : elle")
print("  passe de 8 860 a 8 410 entre la premiere et la deuxieme lecture. La raison n'est pas")
print("  un effet de modele mais un changement de BASE, l'etat conforme passant de 7 987 a")
print("  6 664 d'un protocole a l'autre. Un ecart en euros n'est comparable qu'a base egale ;")
print("  le rapport, lui, est sans dimension et se compare. C'est le meme motif qui fait")
print("  publier les elasticites plutot que les plages dans le tornado renormalise.")

print("\nCE QUE LA MONOTONIE DIT, ET CE QU'ELLE NE DIT PAS. Elle est coherente avec la")
print("super-additivite mesuree sur une autre partition (les seize configurations du script 68,")
print("+5 001 M EUR d'interaction) : relacher un canal de plus ajoute son effet propre ET les")
print("croises qu'il porte. Mais les quatre lignes ne sont PAS une experience controlee, leurs")
print("protocoles differant aussi par la graine et par la base. La monotonie s'observe, elle ne")
print("se teste pas ici ; le test, lui, est dans le script 68, ou les seize configurations")
print("partagent le meme moteur et les memes graines.")

print("\nCE QUE LE MEMOIRE PUBLIE, ET POURQUOI. La lecture retenue est celle des QUATRE CANAUX.")
print("Motif principal : c'est la seule ou l'etat conforme est conforme sur TOUS les canaux que")
print("le modele possede. Dans la premiere lecture, l'entite dite conforme propage encore comme")
print("une entite defaillante ; dans les deux suivantes, sa detection ou son accumulation tiers")
print("restent au niveau non conforme. Une entite conforme sur un canal et non conforme sur")
print("trois autres n'est pas un etat conforme, c'est une remediation partielle, et la nommer")
print("« conforme » sous-estime l'ecart par construction.")
print("Motif secondaire : c'est deja la lecture dont depend tout l'aval, la table des quatre")
print("canaux, la decomposition de Mobius, les colonnes de fermeture et de Shapley, et le")
print("calcul de portage. Publier un autre chiffre en tete obligerait a rejouer cet aval.")

print("\nLES TROIS AUTRES LECTURES NE SONT PAS RETIREES. Elles restent publiees comme des")
print("remediations PARTIELLES, ce qu'elles sont, et leur emboitement est un resultat : il")
print("chiffre ce que coute de ne fermer qu'une partie des canaux.")


# =====================================================================================
titre("3ter. Le rapport detention / transfert, et pourquoi il resiste mieux que le NIVEAU")
# =====================================================================================
# POURQUOI CETTE SECTION EXISTE. La conclusion du memoire affirmait que le RAPPORT entre cout
# annuel de detention et prix de transfert « resiste » a la descente d'echelle, contrairement au
# NIVEAU. Ni le rapport ni sa stabilite n'etaient imprimes nulle part : c'etait le DERNIER
# nombre non confirme du harnais sur les dix-neuf chapitres. Et la phrase disait l'inverse de son
# propre argument, en annoncant que le rapport tombait « a moins de 11 % de sa valeur » alors que
# « resister » veut dire en garder la quasi-totalite. Le 11 % n'est reproduit par aucun calcul du
# projet. Cette section remplace une affirmation par une identite et une borne.
#
# ENTREES, toutes deja publiees par le script 58 a l'echelle d'entite (SCR = 169 M EUR,
# portee optimale L* = 170 M EUR). La prime ne depend pas du CoC, le cout de detention si.
COUT_TOTAL_XL = 2.89        # M EUR/an, cout total avec traite en exces de perte au L* optimal
SCR_ENTITE = 169.0          # M EUR
L_OPT = 170.0               # M EUR, portee optimale
COCS = [("regime actuel", 0.0600), ("directive (UE) 2025/2", 0.0475)]

print("Definitions, echelle d'entite, script 58 :")
print(f"  cout annuel de DETENTION = CoC x SCR, avec SCR = {SCR_ENTITE:.0f} M EUR")
print(f"  prix de TRANSFERT        = cout total du traite au L* optimal "
      f"= {COUT_TOTAL_XL:.2f} M EUR/an")
print(f"  portee optimale L*       = {L_OPT:.0f} M EUR, soit L* / SCR = {L_OPT/SCR_ENTITE:.3f}")
print()
print(f"  {'CoC':<32}{'detention':>12}{'transfert':>12}{'rapport':>10}{'taux moyen':>13}")
print("  " + "-" * 79)
for nom, coc in COCS:
    det = coc * SCR_ENTITE
    print(f"  {nom + f' ({100*coc:.2f} %)':<32}{det:>12.2f}{COUT_TOTAL_XL:>12.2f}"
          f"{det/COUT_TOTAL_XL:>10.2f}{COUT_TOTAL_XL/L_OPT:>13.4f}")

print("\nPOURQUOI CE RAPPORT EST PLUS ROBUSTE QUE LE NIVEAU, ET C'EST UNE IDENTITE.")
print("  rapport = (CoC x SCR) / (taux moyen x L*),  et a l'optimum L* = SCR")
print("         => rapport = CoC / taux moyen,  le SCR se SIMPLIFIE.")
print("  Le niveau, lui, traverse les deux elasticites estimees de la descente d'echelle")
print("  (frequence 0,0744 et severite 0,087) : c'est la qu'il devient illustratif. Le")
print("  rapport n'en herite pas au premier ordre, parce que ses deux termes sont")
print(f"  proportionnels au MEME SCR. L'optimum L* = SCR est verifie ici a "
      f"{100*abs(L_OPT/SCR_ENTITE - 1):.1f} % pres.")

print("\nCE QUI RESTE, ET QU'IL NE FAUT PAS ANNONCER COMME NUL. La simplification vaut pour un")
print("changement d'echelle PUR. Or la descente d'echelle ne rescale pas la loi de perte : elle")
print("deplace la frequence et la severite separement, donc elle change la FORME. L'effet de")
print("forme, lui, n'est pas nul, et le seul balayage disponible le borne par le haut.")
# Table de remediation du script 58 : lambda reduit, donc SCR reduit ET forme deplacee.
# Ce n'est PAS un test d'echelle pur, et c'est pour cela qu'elle borne PAR LE HAUT.
REMED = [(0.092, 169.0, 2.89), (0.064, 97.0, 1.39), (0.046, 43.0, 0.54)]
COC_REF = 0.0475
print(f"\n  balayage du script 58 a CoC = {100*COC_REF:.2f} % (lambda reduit) :")
print(f"  {'lambda':>8}{'SCR':>8}{'detention':>12}{'transfert':>12}{'rapport':>10}")
print("  " + "-" * 50)
rapports = []
for lam, scr, tot in REMED:
    det = COC_REF * scr
    rapports.append(det / tot)
    print(f"  {lam:>8.3f}{scr:>8.0f}{det:>12.2f}{tot:>12.2f}{det/tot:>10.2f}")
lo, hi = min(rapports), max(rapports)
fac_scr = REMED[0][1] / REMED[-1][1]
print(f"\n  le rapport va de {lo:.2f} a {hi:.2f}, soit +{100*(hi/lo - 1):.0f} %, quand le SCR")
print(f"  varie d'un facteur {fac_scr:.1f}. C'est une BORNE SUPERIEURE de l'effet de forme :")
print("  ce balayage deplace la frequence, donc il cumule l'effet d'echelle et celui de forme.")
print("\n  L'ENONCE PUBLIABLE est donc : le rapport est invariant a un changement d'echelle pur,")
print("  par simplification du SCR, et l'effet de forme residuel est borne par +36 % sur une")
print("  plage de SCR d'un facteur quatre. PAS « il garde plus de 89 % de sa valeur », qui")
print("  n'est reproduit par aucun calcul, et pas « il tombe a 11 % », qui dit le contraire de")
print("  ce que la phrase soutenait.")
print("\n  NON MESURE, et a ne pas combler par un chiffre : le deplacement de ce rapport ENTRE")
print("  le secteur et l'entite. Le traite en exces de perte n'est optimise qu'a l'echelle")
print("  d'entite ; aucun script ne calcule son equivalent sectoriel. La comparaison des deux")
print("  echelles sur ce rapport n'existe pas, et l'identite ci-dessus est ce qui la remplace.")


# =====================================================================================
titre("4. Statistiques de severite de la population EFFECTIVEMENT utilisee")
# =====================================================================================
# VALEURS EN POINT DECIMAL. Ces six nombres sont ceux que le chapitre donnees CITE dans son
# encadre pour expliquer ce qu'il a remplace : ils doivent donc rester appariables par le
# harnais. Ecrits « 1,50 » et « 26,4 », il y lisait deux nombres, 1 et 50, 26 et 4.
print("POURQUOI CETTE SECTION EXISTE, ET CE QU'ELLE A TROUVE. Le chapitre donnees publie une")
print("table de statistiques de severite (mediane 1.50 M USD, moyenne 26.4, P90 51.5, P99 402,")
print("maximum 1500, total cumule 15.4 Md USD). Cette table a ete MIGREE AUTOMATIQUEMENT depuis")
print("la version pre-cascade du memoire, et AUCUN filtre du pipeline actuel ne la reproduit :")
print("ni la base entiere, ni le secteur financier, ni le sous-ensemble TIC, ni aucune")
print("combinaison de fenetres. Sa moyenne et son total impliquent n = 583 observations, effectif")
print("qu'aucun sous-ensemble courant ne presente. Elle decrit donc une population que le memoire")
print("n'utilise plus nulle part ailleurs.")
print("\nCE SCRIPT NE TENTE PAS DE LA RETROUVER. Il imprime les statistiques de la population que")
print("le memoire utilise REELLEMENT partout ailleurs, c'est-a-dire celle de la descente")
print("d'echelle et de la calibration de frequence : secteur financier, categories TIC, fenetre")
print("2005-2022. C'est cette table qui doit figurer au chapitre donnees, faute de quoi le")
print("chapitre decrit un echantillon et calibre sur un autre.\n")

RAW = os.path.abspath(os.path.join(REPO, "data", "raw"))
XLS = os.path.join(RAW, "SAS_OpRisk_Global_Data_June_2026.xlsx")
if not os.path.exists(XLS):
    print(f"  Base absente ({os.path.relpath(XLS, REPO)}) : section ignoree.")
    print("  Les sources brutes ne sont pas versionnees ; relancer ce script sur un poste")
    print("  disposant de la base pour produire cette table.")
else:
    import pandas as pd
    import descente as dsc                                            # noqa: E402
    d = pd.read_excel(XLS, sheet_name="Datasets")
    d["year"] = pd.to_datetime(d["First Year of Event"], errors="coerce").dt.year
    sel = ((d["Basel Business Line - Level 1"] != "Non-FS")
           & d["Sub Risk Category"].isin(dsc.ICT)
           & d.year.between(dsc.Y0, dsc.Y1))
    x = pd.to_numeric(d.loc[sel, dsc.LOSS], errors="coerce").dropna()
    x = x[x > 0].sort_values(ascending=False).to_numpy()
    print(f"  filtre : secteur financier x categories TIC x annees {dsc.Y0}-{dsc.Y1}")
    print(f"  {'observations':<22}{len(x):>12}")
    print(f"  {'mediane':<22}{np.median(x):>12.2f} M USD")
    print(f"  {'moyenne':<22}{x.mean():>12.1f} M USD")
    for q in (0.90, 0.99):
        print(f"  {'P' + format(100*q, '.0f'):<22}{np.quantile(x, q):>12.1f} M USD")
    print(f"  {'maximum':<22}{x.max():>12.1f} M USD")
    print(f"  {'total cumule':<22}{x.sum()/1000:>12.1f} Md USD")
    n10, n01 = max(1, int(round(0.10 * len(x)))), max(1, int(round(0.01 * len(x))))
    print(f"\n  concentration de la queue, sur la meme population :")
    print(f"  {'part du decile superieur':<34}{100*x[:n10].sum()/x.sum():>8.1f} %")
    print(f"  {'part du centile superieur':<34}{100*x[:n01].sum()/x.sum():>8.1f} %")
    print("\n  Le memoire annoncait 82.7 % et 31 % pour ces deux parts, sur la population non")
    print("  reproductible. Les valeurs ci-dessus sont celles de la population qu'il calibre.")

print("\nL'ECART DE HILL AU MLE (chapitre socle, 138,7 % a k = 86) N'EST PAS RECALCULE ICI, et")
print("c'est deliberé. Cet ecart porte sur les excedents CONVERTIS EN EUROS au-dessus du seuil")
print("de collecte, objet construit par le pipeline du script 47. Le recalculer ici sur les")
print("pertes brutes en dollars donne un tout autre nombre, et publier deux valeurs sous le")
print("meme nom creerait la contradiction que ce script est cense eviter. C'est au script 47")
print("de l'imprimer, puisque c'est lui qui possede la conversion.")


# =====================================================================================
titre("5. La table des queues par categorie Bale, et son effectif le plus faible")
# =====================================================================================
print("POURQUOI CETTE SECTION EXISTE. Le chapitre des adaptations par pilier ouvre son")
print("argument sur deux valeurs : les cinq queues observees par categorie Bale")
print("(0,92 / 0,98 / 1,03 / 1,27 / 1,37) et les 105 observations de la categorie la plus")
print("proche du TIC, qui la rendent non estimable. Le script 37 porte les deux dans son")
print("en-tete, en commentaire, et n'en imprime aucune : elles ne sont donc verifiables par")
print("personne, alors qu'elles portent tout le raisonnement d'heterogeneite de queue.\n")

if not os.path.exists(XLS):
    print(f"  Base absente ({os.path.relpath(XLS, REPO)}) : section ignoree.")
else:
    from scipy.stats import genpareto                                 # noqa: E402
    from src.severity.oprisk_analysis import filter_finance           # noqa: E402
    # CONVENTION DE SEVERITE, PAS CELLE DE LA DESCENTE. La section 4 ci-dessus suit le
    # pipeline de frequence : colonne « Current Value of Loss », secteur lu sur la ligne
    # metier Bale. La queue par categorie est une affaire de SEVERITE : elle doit suivre le
    # pipeline qui calibre la GPD, c'est-a-dire load_clean (colonne « Loss Amount ») et
    # filter_finance (secteur lu sur « Industry Sector Name »), comme les scripts 07, 46
    # et 47. Melanger les deux conventions donnait un xi different par categorie, pour la
    # meme grandeur : c'est precisement le defaut que ce script est cense eteindre.
    fin = d.copy()
    fin["loss"] = pd.to_numeric(fin["Loss Amount ($M)"], errors="coerce")
    fin = fin[(fin["loss"] > 0) & (fin["loss"] < 100_000)]
    fin = filter_finance(fin)
    print(f"  filtre : secteur financier (Industry Sector Name), toutes annees, "
          f"colonne Loss Amount ({len(fin)} lignes)")
    print(f"  seuil par categorie : percentile 75 ; xi par MLE GPD a seuil libre\n")
    print(f"  {'categorie Bale (Event Risk Category)':<46}{'n':>7}{'xi':>8}")
    obtenu = {}
    for cat, n_cat in fin["Event Risk Category"].value_counts().items():
        v = np.sort(fin.loc[fin["Event Risk Category"] == cat, "loss"].to_numpy())
        exc = v[v > float(np.quantile(v, 0.75))] - float(np.quantile(v, 0.75))
        xi_c = genpareto.fit(exc, floc=0)[0] if exc.size >= 10 else float("nan")
        obtenu[cat] = (int(n_cat), xi_c)
        print(f"  {cat[:44]:<46}{n_cat:>7}{xi_c:>8.3f}")

    n_bdsf = obtenu.get("Business Disruption and System Failures", (None, None))[0]
    retenues = sorted(x for (nn, x) in obtenu.values() if nn >= 150)
    print(f"\n  LES CINQ QUEUES RETENUES (n >= 150), triees :")
    print("     " + " | ".join(f"{x:.3f}" for x in retenues))
    print(f"  dispersion (max - min) : {max(retenues)-min(retenues):.3f}")
    print(f"  categorie la plus proche du TIC : {n_bdsf} observations, sous le seuil de 150.")
    print("  Damage to Physical Assets (146) est exclue pour la meme raison.")

    print("\n  CE QUE CETTE TABLE A REMPLACE, ET POURQUOI. Le memoire citait auparavant")
    print("  0,92 / 0,98 / 1,03 / 1,27 / 1,37 et 105 observations, valeurs heritees d'une sonde")
    print("  dont le filtre n'a pas ete conserve. Aucune combinaison du pipeline ne les")
    print("  reproduisait : ni le secteur financier seul, ni les fenetres 2000-2026 et")
    print("  2005-2022, ni la base entiere, ni les seuils q75 et q85. Elles ont donc ete")
    print("  remplacees par la table ci-dessus, qui a un filtre ecrit et un script qui")
    print("  l'imprime. Le script 37 en est le consommateur et en recopie les cinq valeurs.")
    print("\n  CE QUE LE REMPLACEMENT CHANGE, ET CE QU'IL NE CHANGE PAS. L'argument du chapitre")
    print("  12b ne porte pas sur le niveau des queues mais sur leur DISPERSION et sur le")
    print("  sous-echantillonnage de la categorie TIC : les deux tiennent, et la dispersion est")
    print("  meme un peu plus large (0,50 contre 0,45). La borne du SCR sur les 120")
    print("  assignations passe donc de +38 / +82 % a +55 / +96 %, et le classement des piliers")
    print("  reste domine par P1 dans 83 % des assignations, inchange. La conclusion du")
    print("  chapitre, supposer une queue commune SOUS-ESTIME le capital, en sort renforcee.")


# =====================================================================================
titre("6. Les parametres POSES, et les constantes, en un seul endroit")
# =====================================================================================
print("POURQUOI CETTE SECTION EXISTE. Le harnais confrontait chaque nombre publie aux")
print("sorties des scripts cites par sa section, et signalait tout ce qui n'y figurait pas.")
print("Restait une classe qu'aucun script n'imprimait parce qu'elle n'est le RESULTAT de")
print("rien : les entrees posees du modele et les constantes standard. Elles sont pourtant")
print("ce qu'un relecteur veut verifier en premier, puisque ce sont les hypotheses. On les")
print("liste donc ici, avec, en regard, le script qui les consomme.\n")

POSES = [
    ("g par etat de conformite (C / PC / NC)", "0.45 / 0.68 / 0.90",
     "pose, non calibre", "16, 20, 66"),
    ("grille tres etroite de g, test d'invariance", "0.85 / 0.875 / 0.90",
     "pose", "66"),
    ("multiplicateur de materialite p_u par etat", "0.85 / 1.00 / 1.20",
     "pose", "20, 36, 39"),
    ("quantile de stress, loi normale a 95 %", "1.645",
     "constante standard", "19, 22, 30"),
    ("formule standard, plafond operationnel (art. 204)", "0.30 du BSCR",
     "reglementaire", "27, 28"),
    ("formule standard, part des primes", "0.03",
     "reglementaire", "27, 28"),
    ("ratio de sinistralite du traite en exces", "0.485",
     "pose, prix de marche", "58"),
    ("minimum d'exces pour une estimation GPD fiable", "30",
     "regle de l'art", "07, 47, 63"),
    ("points de lecture du graphe de Hill", "k = 30 et k = 200",
     "lecture graphique", "47"),
    ("fourchette d'estimations de place, cout DORA", "25 a 150 M EUR",
     "source externe", "50"),
    ("regression Jacobs 2014, effectif et ajustement", "115 obs., R2 = 0.51",
     "source externe", "21"),
    ("resolutions Monte-Carlo employees", "60 / 150 / 240 / 600 mille annees",
     "resolution, pas un resultat", "16b, 20, 58, 60"),
    # AJOUTS DU 6 AOUT. La table des parametres de l'annexe 17 ne citait aucun script : elle
    # echappait donc entierement au harnais, alors que c'est la table qu'un jury lit en
    # premier pour savoir ce qui est calibre et ce qui est pose. En la rattachant a ses
    # sources, quatre constantes posees sont ressorties non imprimees. Les voici.
    ("sur-dispersion de la frequence phi", f"{FREQUENCY['dispersion_factor']:.2f}",
     "semi-calibre, maintenu", "06, 08b, 17"),
    ("copule de dependance, parametre de Gumbel theta", f"{COPULE['theta_nc']}",
     "pose", "23, 42, 48"),
    ("seuil de crise de la latente theta_crise", "-2.5",
     "pose", "16, 30"),
    ("probabilites d'etat P(NC / PC / C)", "0.35 / 0.35 / 0.30",
     "semi-ancre, enquetes de conformite", "16, 20, 36"),
]
print(f"  {'parametre':<50}{'valeur':>28}  {'statut':<24}{'scripts'}")
print("  " + "-" * 116)
for nom, val, statut, scr in POSES:
    print(f"  {nom:<50}{val:>28}  {statut:<24}{scr}")

print("\n  AUCUN de ces nombres n'est un resultat, et c'est le point. Les trois premiers sont")
print("  les hypotheses que le memoire assume et dont il montre, script 66, que sa these ne")
print("  depend pas ; les deux suivants sont reglementaires ; les autres sont des conventions")
print("  de lecture ou des sources externes. Les publier ici les rend verifiables sans")
print("  laisser croire qu'ils sont estimes.")


# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("Ce script ne produit aucun resultat nouveau : il rend VERIFIABLES des valeurs qui")
print("etaient publiees sans l'etre. C'est la quatrieme classe du residu du harnais, la seule")
print("qui soit un defaut. Les trois autres (separateur, unite, valeur legitimement hors")
print("script) se traitent respectivement dans le harnais, en imprimant les deux unites, et")
print("en assumant la valeur dans le texte.")
print("\nA RELANCER sur un poste disposant de data/raw/ pour completer la section 4.")
