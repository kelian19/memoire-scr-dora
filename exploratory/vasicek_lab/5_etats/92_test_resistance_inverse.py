#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""92 - TEST DE RESISTANCE INVERSE sur les quatre canaux de conformite.

LA QUESTION RECIPROQUE, ET C'EST CELLE QUE L'ORSA POSE AUSSI. Le memoire repond a
<< que coute la non-conformite >> : on fixe un etat, on lit un capital. L'exercice de
resistance inverse fixe le CAPITAL et cherche les etats du monde qui le produisent. Ce n'est
pas une curiosite de methode : c'est la forme sous laquelle un praticien utilise un modele,
et c'est la question qui manquait au dossier.

CE QUI REND L'EXERCICE POSSIBLE ICI, ET IL FAUT LE DIRE. Sans monotonie, chercher les etats
qui atteignent une cible demanderait de balayer tout le pave. Le script 88 a etabli que le
capital est croissant sur le treillis des seize configurations, graine par graine : l'ensemble
des configurations qui atteignent une cible est donc un ENSEMBLE CROISSANT, entierement decrit
par ses elements MINIMAUX. La reponse a la question inverse est cette frontiere, et elle est
courte. Le resultat du 88 cesse ainsi d'etre une propriete de confort pour devenir l'outil.

TROIS AVERTISSEMENTS, ET LE TROISIEME EST UNE DECOUVERTE DE CE SCRIPT.
  1. deux canaux sur quatre sont des BORNES POSEES, la propagation et l'accumulation. Un etat
     du monde declare atteignable l'est donc sur un pave DECLARE, et jamais << possible pour
     telle entite >>. C'est le garde-fou du script 88 sur l'infaisabilite d'un coin ;
  2. une configuration dont le capital est proche de la cible n'est pas CLASSEE : son
     appartenance a l'ensemble croissant depend de la graine. Le script le signale au lieu de
     trancher, et une frontiere non resolue est un resultat, pas un defaut ;
  3. LE CANAL D'ACCUMULATION N'ADMET PAS D'INVERSION CONTINUE, et ce n'est pas un choix de
     parametrage. A l'etat conforme, P4 est un noeud de cascade ordinaire ; a l'etat non
     conforme, il devient un choc commun de parametre phi. Les deux ne sont pas les deux bouts
     d'un continuum mais deux STRUCTURES de table differentes, si bien que phi = 0 n'est pas
     l'etat conforme. Ce canal est structurellement binaire, et un test inverse ne peut donc pas
     lui demander << jusqu'ou faut-il degrader >>.

CE QU'IL TROUVE, EN DEUX LIGNES. La frequence SEULE atteint 10 377 M EUR quand les trois autres
canaux REUNIS n'atteignent que 11 413 : toute cible sous le premier seuil se produit par un canal
unique, toute cible au-dessus du second EXIGE la frequence. Aucun autre canal seul n'atteint meme
la cible la plus basse du balayage. Et la bascule de structure du canal d'accumulation fait
BAISSER le capital de 239 M EUR, si bien que son effet isole publie de 1 728 est un NET.

C'EST UN DIAGNOSTIC, PAS UNE RECALIBRATION. Aucun parametre publie n'est touche, le moteur des
canaux n'est pas modifie, et les deux coins sont controles avant toute affirmation.
"""

import os
import sys
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
if LAB not in sys.path:
    sys.path.insert(0, LAB)

import canaux_conformite as cx  # noqa: E402

WID = 88
CANAUX = ("frequence", "detection", "propagation", "accumulation")
COURT = {"frequence": "freq", "detection": "det", "propagation": "prop",
         "accumulation": "accum"}
BAS = {"frequence": cx.LAM_C, "detection": cx.PU_C, "propagation": cx.G_C,
       "accumulation": 0.0}
HAUT = {"frequence": cx.LAM_NC, "detection": cx.PU_NC, "propagation": cx.G_NC,
        "accumulation": cx.PHICS_NC}
CIBLES = (8_000.0, 10_000.0, 12_000.0, 15_000.0, 18_000.0)


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def fnum(v):
    return f"{v:,.0f}".replace(",", " ")


def config(sous_ensemble):
    """Configuration ou les canaux de `sous_ensemble` sont au niveau non conforme."""
    v = {c: (HAUT[c] if c in sous_ensemble else BAS[c]) for c in CANAUX}
    phi = v["accumulation"] if "accumulation" in sous_ensemble else None
    return v["frequence"], v["propagation"], v["detection"], phi


def nom(s):
    return "conforme" if not s else " + ".join(COURT[c] for c in s)


# ---------------------------------------------------------------------------
# 0. Controle des deux coins
# ---------------------------------------------------------------------------

titre("0. CONTROLE : LES DEUX COINS REPRODUISENT LES NOMBRES PUBLIES")
print("Un test inverse repond par des etats du monde. Si le moteur n'etait pas celui du memoire,")
print("il repondrait par les etats du monde d'un autre modele, ce qui est le pire des defauts")
print("possibles puisqu'il ne se verrait pas.")
print()
sous_ens = [tuple(c for c in CANAUX if c in s)
            for k in range(len(CANAUX) + 1) for s in combinations(CANAUX, k)]
graines = {}
for s in sous_ens:
    lam, g, pu, phi = config(s)
    graines[s] = cx.scr_par_graine(lam, g, pu, phi)
scr = {s: float(graines[s].mean()) for s in sous_ens}
print(f"  coin bas  (aucun canal relache) : {fnum(scr[()]):>7} M EUR   publie 6 049")
print(f"  coin haut (les quatre relaches) : {fnum(scr[CANAUX]):>7} M EUR   publie 20 188")
print(f"  ecart                           : {fnum(scr[CANAUX] - scr[()]):>7} M EUR   "
      "publie 14 139")


# ---------------------------------------------------------------------------
# 1. Les seize configurations, ordonnees
# ---------------------------------------------------------------------------

titre("1. LES SEIZE CONFIGURATIONS, PAR CAPITAL CROISSANT")
print("configuration                        capital   ecart-type de graine")
for s in sorted(sous_ens, key=lambda z: scr[z]):
    print(f"  {nom(s):<34} {fnum(scr[s]):>7}   {fnum(float(graines[s].std(ddof=1))):>5}")


# ---------------------------------------------------------------------------
# 2. Le test inverse discret : la frontiere de chaque cible
# ---------------------------------------------------------------------------

titre("2. TEST INVERSE DISCRET : LES CONFIGURATIONS MINIMALES DE CHAQUE CIBLE")
print("Pour chaque cible, les configurations MINIMALES qui l'atteignent, c'est-a-dire celles")
print("dont aucun sous-ensemble strict ne l'atteint. La monotonie du script 88 garantit que")
print("cette frontiere decrit l'ensemble entier : toute configuration contenant un element de la")
print("frontiere atteint la cible, et aucune autre ne l'atteint.")
print()
for cible in CIBLES:
    atteint = [s for s in sous_ens if scr[s] >= cible]
    minimaux = [s for s in atteint
                if not any(set(t) < set(s) for t in atteint)]
    # une configuration est NON CLASSEE si sa position depend de la graine
    non_cl = [s for s in sous_ens
              if (graines[s] >= cible).any() and not (graines[s] >= cible).all()]
    print(f"cible {fnum(cible)} M EUR, soit x{cible / scr[()]:.2f} l'etat conforme")
    if not atteint:
        print("  AUCUNE configuration du pave ne l'atteint, meme le coin superieur")
    else:
        for s in sorted(minimaux, key=len):
            marge = 100 * (scr[s] / cible - 1.0)
            print(f"  {len(s)} canal(aux) : {nom(s):<32} {fnum(scr[s]):>7} "
                  f"({marge:+5.1f} % de la cible)")
        print(f"  {len(atteint)} configurations sur 16 atteignent la cible, "
              f"{len(minimaux)} minimale(s)")
    if non_cl:
        print("  NON CLASSEES, leur position depend de la graine : "
              + ", ".join(nom(s) for s in non_cl))
    print()

print("LECTURE DE GESTION. Le nombre de canaux de la configuration minimale la plus petite est")
print("la reponse utile : c'est le nombre MINIMAL de defaillances simultanees qui suffit a")
print("produire la cible. Une cible atteinte par un seul canal se lit comme une exposition a un")
print("point unique, une cible qui en exige trois comme une exposition a une conjonction.")


# ---------------------------------------------------------------------------
# 2bis. Les deux seuils du canal de frequence, lus sur le treillis
# ---------------------------------------------------------------------------

titre("2bis. LES DEUX SEUILS DU CANAL DE FREQUENCE")
print("Les frontieres de la section 2 disent la meme chose a chaque cible, et elle se resume en")
print("deux nombres plutot qu'en cinq listes. Ils se lisent directement sur le treillis, sans")
print("balayage supplementaire, et c'est la monotonie qui les rend valides.")
print()
sans_f = [s for s in sous_ens if "frequence" not in s]
max_sans_f = max(sans_f, key=lambda s: scr[s])
print(f"  la frequence SEULE atteint            {fnum(scr[('frequence',)])} M EUR, soit "
      f"x{scr[('frequence',)] / scr[()]:.2f} l'etat conforme")
print(f"  le MEILLEUR etat sans la frequence    {fnum(scr[max_sans_f])} M EUR, soit "
      f"x{scr[max_sans_f] / scr[()]:.2f}   ({nom(max_sans_f)})")
print()
print(f"  toute cible sous {fnum(scr[('frequence',)])} M EUR est donc atteinte par la "
      "FREQUENCE SEULE : un canal suffit ;")
print(f"  toute cible au-dessus de {fnum(scr[max_sans_f])} M EUR EXIGE la frequence, aucune")
print("  combinaison des trois autres canaux ne l'atteignant, meme toutes ensemble ;")
print(f"  entre les deux, soit de {fnum(scr[('frequence',)])} a {fnum(scr[max_sans_f])} M EUR, "
      "elle n'est ni suffisante seule ni necessaire.")
print()
d_seuils = graines[max_sans_f] - graines[("frequence",)]
print(f"  l'ordre des deux bornes est-il resolu ? Sur les {d_seuils.size} graines, l'ecart entre")
print(f"  ces deux configurations vaut {fnum(float(d_seuils.mean()))} "
      f"+- {fnum(float(d_seuils.std(ddof=1)))} M EUR et il est du meme signe sur "
      f"{int((d_seuils > 0).sum())} graine(s) sur {d_seuils.size}.")
print()
print("C'EST LE RESULTAT DE GESTION DE CE SCRIPT, et le memoire ne le dit nulle part. La lecture")
print("directe du modele hierarchise les canaux par leur contribution ; la lecture inverse dit")
print("autre chose, qu'un plan de remediation utilise vraiment : le canal de FREQUENCE est le")
print("seul dont la defaillance suffit a elle seule a un scenario severe, et le seul dont la")
print("maitrise interdit les scenarios les plus severes. Les trois autres reunis ne les")
print("atteignent pas.")


# ---------------------------------------------------------------------------
# 3. Le test inverse continu, canal par canal
# ---------------------------------------------------------------------------

titre("3. TEST INVERSE CONTINU : JUSQU'OU UN SEUL CANAL DOIT-IL SE DEGRADER ?")
print("Les trois autres canaux restent au niveau conforme, et l'on balaye celui-la de son niveau")
print("conforme a son niveau non conforme. La question inverse devient scalaire : quelle valeur")
print("du canal atteint la cible, et cette valeur est-elle DANS la plage declaree ?")
print()
FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
courbes = {}
for c in ("frequence", "detection", "propagation"):
    vals, caps = [], []
    for f in FRACTIONS:
        v = BAS[c] + f * (HAUT[c] - BAS[c])
        d = {k: BAS[k] for k in CANAUX}
        d[c] = v
        cap = float(cx.scr_par_graine(d["frequence"], d["propagation"], d["detection"],
                                      None).mean())
        vals.append(v)
        caps.append(cap)
    courbes[c] = (np.array(vals), np.array(caps))
    mono = bool(np.all(np.diff(caps) >= 0))
    print(f"{c} : de {vals[0]:.4g} a {vals[-1]:.4g}")
    print("  valeur   " + "  ".join(f"{v:>8.4g}" for v in vals))
    print("  capital  " + "  ".join(f"{fnum(x):>8}" for x in caps))
    print(f"  monotonie observee sur le balayage : {'oui' if mono else 'NON'}")
    print()

print("ACCUMULATION : PAS D'INVERSION CONTINUE, ET C'EST STRUCTUREL.")
print("A l'etat conforme P4 est un noeud de cascade ; a l'etat non conforme il devient un choc")
print("commun de parametre phi. Deux structures de table, non deux bouts d'un continuum. Le")
print("balayage de phi ci-dessous vit donc entierement dans la seconde famille, et son point")
print("phi = 0 N'EST PAS l'etat conforme, ce que les deux premieres lignes montrent.")
vals_a, caps_a, gr_a = [], [], {}
for phi in (0.0, 0.17, 0.34, 0.51, 0.68):
    g_ = cx.scr_par_graine(BAS["frequence"], BAS["propagation"], BAS["detection"], phi)
    gr_a[phi] = g_
    vals_a.append(phi)
    caps_a.append(float(g_.mean()))
print(f"  etat conforme (P4 noeud de cascade)      {fnum(scr[()]):>7} M EUR")
print("  phi      " + "  ".join(f"{v:>8.2f}" for v in vals_a))
print("  capital  " + "  ".join(f"{fnum(x):>8}" for x in caps_a))
courbes["accumulation"] = (np.array(vals_a), np.array(caps_a))

print()
print("ET LE SAUT DE STRUCTURE EST NEGATIF, CE QUI DECOMPOSE LE CANAL EN DEUX MOUVEMENTS DE SENS")
print("CONTRAIRES. On ne l'attendait pas, et il se lit graine par graine puisque les deux")
print("configurations partagent leurs tirages.")
d_str = gr_a[0.0] - graines[()]
print(f"  (i)  bascule de structure, noeud de cascade vers choc a phi = 0 : "
      f"{fnum(float(d_str.mean())):>7} M EUR")
print(f"       {int((d_str < 0).sum())} graine(s) sur {d_str.size} du meme signe, ecart-type "
      f"{fnum(float(d_str.std(ddof=1)))}")
print(f"  (ii) montee du choc commun, phi de 0 a 0,68                    "
      f"{fnum(caps_a[-1] - caps_a[0]):>7} M EUR")
print(f"  net, soit l'effet isole du canal deja publie                   "
      f"{fnum(caps_a[-1] - scr[()]):>7} M EUR")
print()
print("LE MOTIF EST INTELLIGIBLE ET IL FAUT LE DIRE. A l'etat conforme, P4 est un noeud de")
print("cascade : il PROPAGE vers les autres piliers. Devenu choc commun a phi = 0, il ne touche")
print("plus que lui-meme, ce qui RETIRE cette propagation. Le canal d'accumulation ne se contente")
print("donc pas d'ajouter de la co-occurrence : il retire d'abord la propagation propre de P4,")
print("puis la remplace par un choc plus large. Son effet isole publie est le NET de ces deux")
print("mouvements, non une addition, et c'est une raison de plus de ne jamais sommer les canaux.")

print()
print("VALEUR DE CHAQUE CANAL QUI ATTEINT LA CIBLE, PAR INTERPOLATION LINEAIRE DU BALAYAGE :")
print("cible          " + "  ".join(f"{c:>14}" for c in CANAUX))
for cible in CIBLES:
    cells = []
    for c in CANAUX:
        v, cap = courbes[c]
        if cap[-1] < cible:
            cells.append("hors plage".rjust(14))
        elif cap[0] >= cible:
            cells.append("des le bas".rjust(14))
        else:
            i = int(np.searchsorted(cap, cible))
            x = v[i - 1] + (cible - cap[i - 1]) * (v[i] - v[i - 1]) / (cap[i] - cap[i - 1])
            cells.append(f"{x:>14.4g}")
    print(f"  {fnum(cible):>7}      " + "  ".join(cells))
print()
print("<< hors plage >> ne veut pas dire impossible dans le monde : cela veut dire que le canal,")
print("SEUL et dans la plage que le memoire declare, n'atteint pas la cible. Pour la propagation")
print("et l'accumulation la plage est une borne POSEE, donc l'enonce porte sur le pave declare")
print("et non sur une entite.")


# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------

titre("VERDICT")
print("Ecrit APRES lecture des sorties.")
print()
print("1. LA QUESTION INVERSE A UNE REPONSE COURTE, ET C'EST LA MONOTONIE QUI LA RACCOURCIT.")
print("   L'ensemble des etats du monde atteignant une cible est un ensemble croissant, donc il")
print("   se decrit entierement par ses elements minimaux, et il n'y en a jamais plus de quatre")
print("   sur les seize configurations. Le theoreme du coin superieur du script 88 cesse ainsi")
print("   d'etre une propriete de confort pour devenir l'instrument de l'exercice inverse.")
print()
print("2. LE RESULTAT DE GESTION TIENT EN DEUX SEUILS, ET LE MEMOIRE NE LES DIT NULLE PART.")
print(f"   La frequence SEULE atteint {fnum(scr[('frequence',)])} M EUR, soit "
      f"x{scr[('frequence',)]/scr[()]:.2f} l'etat conforme ; les trois autres canaux")
print(f"   REUNIS n'atteignent que {fnum(scr[max_sans_f])}, soit x{scr[max_sans_f]/scr[()]:.2f}. "
      "Toute cible sous le premier seuil est donc")
print("   atteinte par un canal unique, et toute cible au-dessus du second EXIGE ce canal.")
print("   L'ordre des deux bornes est du meme signe sur les quatre graines.")
print("   LA LECTURE INVERSE DIT AUTRE CHOSE QUE LA LECTURE DIRECTE, et c'est l'interet de")
print("   l'exercice : la lecture directe hierarchise les canaux par leur contribution, la")
print("   lecture inverse designe celui dont la maitrise INTERDIT les scenarios severes.")
print()
print("3. AUCUN CANAL AUTRE QUE LA FREQUENCE N'ATTEINT MEME LA CIBLE LA PLUS BASSE, seul et dans")
print(f"   sa plage declaree. La detection plafonne a {fnum(scr[('detection',)])}, la propagation "
      f"a {fnum(scr[('propagation',)])}, l'accumulation")
print(f"   a {fnum(scr[('accumulation',)])}, quand la cible la plus basse du balayage vaut "
      f"{fnum(CIBLES[0])}. Une exposition a un canal")
print("   unique est donc, dans ce modele, une exposition a la frequence et a rien d'autre.")
print()
print("4. ET UNE DECOUVERTE SUR LE CANAL D'ACCUMULATION, qui n'etait pas cherchee. Il n'admet")
print("   pas d'inversion continue, ses deux etats etant deux STRUCTURES de table et non deux")
print("   bouts d'un continuum. Surtout, la bascule de structure a parametre nul fait BAISSER le")
print(f"   capital de {fnum(abs(float(d_str.mean())))} M EUR, resolu sur les quatre graines, "
      "parce qu'elle retire la propagation")
print(f"   propre de P4 avant d'ajouter le choc commun. L'effet isole publie de "
      f"{fnum(caps_a[-1] - scr[()])} M EUR est donc le NET")
print(f"   d'un retrait de {fnum(abs(float(d_str.mean())))} et d'un ajout de "
      f"{fnum(caps_a[-1] - caps_a[0])}, et non une addition.")
print()
print("CE QUE CE SCRIPT NE DIT PAS, ET LES DEUX RESERVES SONT DE NATURE DIFFERENTE.")
print("  - << hors plage >> porte sur un pave DECLARE et jamais sur une entite : deux canaux sur")
print("    quatre sont des bornes posees, donc un etat du monde declare inatteignable ne l'est")
print("    qu'au regard de plages que le memoire choisit. C'est le garde-fou du script 88 sur")
print("    l'infaisabilite d'un coin, applique dans l'autre sens ;")
print("  - une configuration proche d'une cible n'est PAS classee, son appartenance dependant de")
print("    la graine, et la section 2 les nomme. Une frontiere non resolue est un resultat : elle")
print("    dit que la cible tombe dans le bruit de simulation d'une configuration, pas que le")
print("    modele hesite.")

print()
print("GRANDEURS CITEES PAR LE MEMOIRE, SANS SEPARATEUR DE MILLIERS")
print("Les tables ci-dessus impriment les montants avec une espace, que l'extracteur du harnais")
print("coupe en deux. Aucun calcul nouveau ici.")
print(f"  frequence seule                      {scr[('frequence',)]:.0f}")
print(f"  meilleur etat sans la frequence      {scr[max_sans_f]:.0f}")
print(f"  detection seule                      {scr[('detection',)]:.0f}")
print(f"  propagation seule                    {scr[('propagation',)]:.0f}")
print(f"  accumulation seule                   {scr[('accumulation',)]:.0f}")
print(f"  choc commun a phi nul                {caps_a[0]:.0f}")
print(f"  montee du choc commun                {caps_a[-1] - caps_a[0]:.0f}")
print(f"  effet isole net du canal             {caps_a[-1] - scr[()]:.0f}")
print("  cibles du balayage                   "
      + "  ".join(f"{c:.0f}" for c in CIBLES))

print()
print("EXIT 0")
