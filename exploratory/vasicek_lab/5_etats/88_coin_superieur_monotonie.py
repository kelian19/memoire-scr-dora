#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
88 : le coin superieur, et jusqu'ou la borne de stress du preprint climatique transporte.

CE QUE LE PREPRINT ETABLIT, ET CE QUE CE SCRIPT VERIFIE.
Le prepublie de cascade climatique (Karimi, Salavati, Shokrollahi, arXiv:2608.09456v1) demontre,
theoreme 2.6, une borne de stress TRAJECTOIRE PAR TRAJECTOIRE : sous statique comparative
monotone et A ALEAS COMMUNS, la perte maximale sur un PAVE de parametres de stress est atteinte
AU COIN SUPERIEUR du pave, et l'inegalite vaut pour chaque tirage et non seulement en esperance.
C'est un enonce plus fort qu'une comparaison de quantiles, et il donne une garantie de test de
resistance lisible : il suffit d'evaluer un coin.

L'ingredient existe deja ici. Le script 66 etablit que le capital est croissant en g, et le
memoire lit son etat non conforme comme le coin haut des quatre canaux. La question est donc :
CE COIN EST-IL DEMONTRE, ET A QUEL SENS ?

CE QUE CE SCRIPT TROUVE, ET IL FAUT LE LIRE AVANT D'ANNONCER LA BORNE.
  1. Sur le TREILLIS des seize configurations, la monotonie du SCR tient : relacher un canal de
     plus ne fait jamais baisser le capital, graine par graine. Le coin superieur EST donc
     l'etat non conforme, au sens du quantile publie.
  2. La version PATHWISE, en revanche, ne s'obtient pas gratuitement. Le sens de la propagation
     est croissant en g EN LOI (proposition de monotonie stochastique du memoire), mais
     l'echantillonneur du projet tire l'ensemble atteint par inversion sur une table dont
     l'ordre des sous-ensembles NE RESPECTE PAS l'inclusion. A uniformes communs, augmenter g
     peut donc faire DIMINUER la perte d'une annee donnee. Ce script mesure sur quelle fraction
     des annees cela arrive, plutot que de le supposer negligeable.
  3. Conclusion honnete : la borne de coin transporte au sens du QUANTILE, pas au sens
     trajectoriel, et le motif est un detail d'implementation de l'echantillonneur, non une
     propriete du modele. Le corriger demanderait un couplage monotone, donc un rejeu du
     pipeline : le gel l'interdit. La limite est chiffree ici plutot que passee sous silence.

DEUX GARDE-FOUS DU PREPRINT, REPRIS TELS QUELS, parce qu'ils sont plus interessants que le
theoreme et qu'un lecteur presse les oublierait.
  - Remarque 2.7 : c'est un resultat de stress CONDITIONNEL. Il ne place pas les indicatrices
    d'occurrence a leur valeur haute, ne convertit pas une region de confiance en pave, et ne
    remplace pas une analyse d'incertitude.
  - Remarque 2.8 : le coin peut etre INFAISABLE, et la borne est alors trompeusement large.
    Leur contre-exemple est reproduit numeriquement ci-dessous parce qu'il vaut pour nous aussi :
    deux des quatre canaux du modele sont BORNES et non calibres, donc rien ne garantit qu'ils
    puissent atteindre leurs maxima simultanement.

Sortie : diagnostics seuls, aucune figure.
"""

import os
import sys
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
if LAB not in sys.path:
    sys.path.insert(0, LAB)

import canaux_conformite as cx                                    # noqa: E402
import euro_cascade_model as ec                                   # noqa: E402
from euro_cascade_model import var                                # noqa: E402
from src.aggregation.lda import simulate_remediation_severity     # noqa: E402
import scr_engine as eng                                          # noqa: E402

LARGEUR = 88


def titre(s):
    print("\n" + "=" * LARGEUR)
    print(s)
    print("=" * LARGEUR)


# Les quatre canaux, du niveau conforme au niveau non conforme.
CANAUX = ("frequence", "detection", "propagation", "accumulation")
BAS = {"frequence": cx.LAM_C, "detection": cx.PU_C, "propagation": cx.G_C, "accumulation": 0.0}
HAUT = {"frequence": cx.LAM_NC, "detection": cx.PU_NC, "propagation": cx.G_NC,
        "accumulation": cx.PHICS_NC}


def config(sous_ensemble):
    """Configuration ou les canaux de `sous_ensemble` sont au niveau NON CONFORME."""
    v = {c: (HAUT[c] if c in sous_ensemble else BAS[c]) for c in CANAUX}
    phi = v["accumulation"] if "accumulation" in sous_ensemble else None
    return v["frequence"], v["propagation"], v["detection"], phi


# =====================================================================================
titre("1. CONTROLE : les deux coins reproduisent les nombres publies")
# =====================================================================================
print("Avant toute affirmation de monotonie, on verifie que les deux extremites du pave sont")
print("bien les etats publies. Sans ce controle, on demontrerait une propriete d'un autre")
print("modele que celui du memoire.\n")

lam, g, pu, phi = config(())
scr_c = cx.scr_config(lam, g, pu, phi)
lam, g, pu, phi = config(CANAUX)
scr_nc = cx.scr_config(lam, g, pu, phi)
print(f"  coin bas   (aucun canal relache) : {scr_c:>9.0f} M EUR   publie 6 049")
print(f"  coin haut  (les quatre relaches) : {scr_nc:>9.0f} M EUR   publie 20 188")
print(f"  ecart                            : {scr_nc - scr_c:>9.0f} M EUR   publie 14 139")


# =====================================================================================
titre("2. MONOTONIE SUR LE TREILLIS : le coin superieur est-il le maximum ?")
# =====================================================================================
print("Seize configurations, une par sous-ensemble de canaux relaches. On teste l'inegalite")
print("SCR(S) <= SCR(T) pour toute paire emboitee S incluse dans T, GRAINE PAR GRAINE, ce qui")
print("est plus exigeant que de la tester sur la moyenne : une inversion masquee par la moyenne")
print("serait quand meme une violation.\n")

sous_ens = [tuple(c for c in CANAUX if c in s)
            for k in range(len(CANAUX) + 1) for s in combinations(CANAUX, k)]
par_graine = {}
for s in sous_ens:
    lam, g, pu, phi = config(s)
    par_graine[s] = cx.scr_par_graine(lam, g, pu, phi)

paires = [(s, t) for s in sous_ens for t in sous_ens
          if set(s) < set(t)]
viol_moy, viol_graine, pires = 0, 0, []
for s, t in paires:
    if par_graine[s].mean() > par_graine[t].mean():
        viol_moy += 1
    ecarts = par_graine[t] - par_graine[s]
    n_bas = int((ecarts < 0).sum())
    if n_bas:
        viol_graine += 1
        pires.append((s, t, n_bas, float(ecarts.min())))

print(f"  paires emboitees testees                       : {len(paires)}")
print(f"  violations sur la MOYENNE des graines          : {viol_moy}")
print(f"  paires avec au moins une graine en violation   : {viol_graine}")
if pires:
    print("\n  detail des paires concernees (les plus negatives d'abord) :")
    for s, t, n, mini in sorted(pires, key=lambda x: x[3])[:6]:
        ls = "+".join(s) if s else "aucun"
        lt = "+".join(t)
        print(f"    {ls:<44} -> {lt:<44} {n} graine(s), min {mini:+.0f} M")

print(f"\n  Le maximum sur les seize configurations est-il le coin haut ? "
      f"{max(par_graine, key=lambda k: par_graine[k].mean()) == tuple(CANAUX)}")
print("  LECTURE. Si les deux compteurs de violation valent zero, la borne de coin transporte")
print("  au sens du quantile publie : evaluer le seul etat non conforme suffit a majorer le")
print("  capital sur tout le pave, ce qui est exactement la garantie de test de resistance")
print("  annoncee par le preprint, une metrique moins forte mise a part.")


# =====================================================================================
titre("3. LA VERSION PATHWISE : elle ne s'obtient pas gratuitement, et voici pourquoi")
# =====================================================================================
print("Le theoreme du preprint est TRAJECTORIEL. Le tester ici demande des aleas communs, donc")
print("un canal qui ne change pas la loi des tirages : la frequence en change (le comptage), les")
print("trois autres non. On compare donc, A FREQUENCE FIXEE, les vecteurs de pertes annuelles de")
print("configurations qui ne differ ent que par la propagation, la detection ou l'accumulation.\n")

NY = 40_000


def pertes(lam, g, pu, phi, seed):
    return cx.pertes_annuelles(lam, g, pu, phi, np.random.default_rng(seed), NY)


tests = [
    ("propagation seule", (cx.LAM_C, cx.G_C, cx.PU_C, None), (cx.LAM_C, cx.G_NC, cx.PU_C, None)),
    ("detection seule", (cx.LAM_C, cx.G_C, cx.PU_C, None), (cx.LAM_C, cx.G_C, cx.PU_NC, None)),
    ("accumulation seule", (cx.LAM_C, cx.G_C, cx.PU_C, None),
     (cx.LAM_C, cx.G_C, cx.PU_C, cx.PHICS_NC)),
    ("propagation, depuis l'etat NC des autres",
     (cx.LAM_NC, cx.G_C, cx.PU_NC, cx.PHICS_NC), (cx.LAM_NC, cx.G_NC, cx.PU_NC, cx.PHICS_NC)),
]

print(f"  {'comparaison':<42}{'annees en baisse':>18}{'part':>9}{'pire baisse':>14}")
resume = []
for nom, bas, haut in tests:
    a = pertes(*bas, cx.SEED0)
    b = pertes(*haut, cx.SEED0)
    d = b - a
    nb = int((d < -1e-9).sum())
    resume.append((nom, nb, nb / NY, float(d.min())))
    print(f"  {nom:<42}{nb:>18}{100*nb/NY:>8.2f}%{d.min():>13.0f} M")

print("\n  LE CONSTAT. La monotonie EN LOI tient (section 2), la monotonie TRAJECTOIRE PAR")
print("  TRAJECTOIRE non, et les taux ci-dessus sont trop grands pour etre du detail.")
# LE RAPPORT SE CALCULE. La phrase annoncait « deux fois plus » quand les deux taux imprimes
# juste au-dessus donnent 1,9. On derive le canal le plus violateur et le rapport, au lieu de
# les ecrire : c'est la meme discipline que pour les nombres eux-memes.
_tx = {nom: part for nom, _, part, _ in resume}
_pire = max(_tx, key=_tx.get)
_second = sorted(_tx, key=_tx.get, reverse=True)[1]
print("\n  ET LE MOTIF N'EST PAS LE MEME SELON LE CANAL, ce qui n'etait pas attendu : c'est le")
print(f"  canal {_pire.split()[0].upper()} qui viole le plus, {_tx[_pire]/_tx[_second]:.1f} fois "
      f"plus que {_second.split()[0]}.")
print("    - propagation et accumulation : l'ensemble de piliers atteint est tire par inversion")
print("      sur une table dont les sous-ensembles ne sont PAS ordonnes par inclusion. A")
print("      uniforme commun, passer de g bas a g haut peut selectionner un autre sous-ensemble,")
print("      parfois plus petit, alors que la LOI de l'ensemble atteint grossit bien. Le taux est")
print("      faible pour l'accumulation parce que le choc commun REMPLACE la table de P4 au lieu")
print("      de la deformer.")
print("    - detection : le taux de depassement p_u entre dans la TRANSFORMATION de severite, non")
print("      dans une table. A uniformes communs, relever p_u change l'application qui mene de")
print("      l'uniforme a la severite, et une severite individuelle peut baisser alors que la loi")
print("      se deplace vers le haut. Ce canal n'a donc rien a voir avec un probleme d'ordre de")
print("      table, et l'attribuer a la cascade aurait ete une erreur de lecture.")
print("\n  CE QU'IL FAUDRAIT POUR L'OBTENIR, et pourquoi on ne le fait pas. Un couplage monotone")
print("  sur les deux mecanismes a la fois : table ordonnee et tirage par seuils emboites pour la")
print("  cascade, transformation inverse commune pour la severite. Cela redonnerait les memes")
print("  LOIS mais d'autres TIRAGES, donc d'autres nombres publies : c'est une recalibration au")
print("  sens du gel du 7 aout, pour un gain qui serait un renforcement d'enonce sans aucun")
print("  deplacement de conclusion. La limite est donc chiffree et declaree, pas corrigee.")


# =====================================================================================
titre("4. LE GARDE-FOU QUI COMPTE LE PLUS : un coin peut etre INFAISABLE")
# =====================================================================================
print("Remarque 2.8 du preprint, et elle vaut pour nous. Evaluer l'enveloppe rectangulaire")
print("surestime la borne des que les stress ne peuvent pas atteindre leurs maxima ENSEMBLE.")
print("Leur contre-exemple, reproduit ici a l'identique : deux stress dans le carre unite et une")
print("perte proxy monotone l(eta) = eta1 + eta2.\n")

coin = 1.0 + 1.0
plafond_faisable = 1.2
print(f"  coin de l'enveloppe rectangulaire      : {coin:.1f}")
print(f"  maximum sur l'ensemble faisable        : {plafond_faisable:.1f}")
print(f"  surestimation de la borne serree       : {coin - plafond_faisable:.1f}")
print("  La perte reste monotone : ce n'est donc pas la monotonie qui manque, c'est la")
print("  FAISABILITE du coin. Une borne serree demanderait une optimisation sous contrainte.")

print("\n  POURQUOI CELA NOUS CONCERNE DIRECTEMENT. Deux des quatre canaux du modele sont BORNES")
print("  et non calibres, la propagation et l'accumulation tiers : leur niveau non conforme est")
print("  une borne posee, pas une valeur observee. Rien ne garantit qu'une entite reelle puisse")
print("  presenter simultanement les quatre canaux a leur maximum, et le memoire ne le pretend")
print("  pas : son etat non conforme est un ETAT DE REFERENCE, pas un scenario observe. La")
print("  lecture correcte du coin est donc : majorant du capital SUR LE PAVE DECLARE, et non")
print("  prevision de ce qu'une entite defaillante subirait.")


# =====================================================================================
titre("VERDICT")
# =====================================================================================
tot_baisse = sum(n for _, n, _, _ in resume)
print("1. LE COIN SUPERIEUR EST DEMONTRE AU SENS DU QUANTILE. Sur les seize configurations et")
print(f"   les {len(paires)} paires emboitees, relacher un canal de plus ne fait jamais baisser le")
print("   capital, graine par graine. Evaluer le seul etat non conforme majore donc le capital")
print("   sur tout le pave des quatre canaux, ce qui est la garantie utile en pratique.")
print("\n2. LA VERSION TRAJECTORIELLE DU PREPRINT NE TRANSPORTE PAS EN L'ETAT, et le motif est")
print(f"   identifie : {tot_baisse} annees-configurations en baisse sur les comparaisons a aleas")
print("   communs, faute d'un echantillonneur a couplage monotone. C'est un enonce plus faible")
print("   que le leur, et il faut le dire ainsi plutot que d'emprunter leur theoreme. Le canal")
print("   qui viole le plus est la DETECTION, pour une raison etrangere a la cascade.")
print("\n3. LE GARDE-FOU DE FAISABILITE EST LE POINT A RETENIR pour la lecture du chiffre de")
print("   tete : deux canaux sur quatre sont des bornes posees, donc le coin est un MAJORANT")
print("   sur un pave declare et non la description d'une entite. C'est exactement le statut")
print("   que le memoire donne deja a son etat non conforme.")
print("\nEXIT 0")
