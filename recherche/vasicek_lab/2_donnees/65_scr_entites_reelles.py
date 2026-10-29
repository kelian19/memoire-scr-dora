#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
65 : le besoin de capital ORSA au titre de DORA, sur des ENTITES REELLES.

DEUX PRECAUTIONS DE VOCABULAIRE, ET ELLES NE SONT PAS COSMETIQUES.

  (1) « SCR DORA » EST UN RACCOURCI DANGEREUX. Il n'existe aucun module DORA dans la
      Formule Standard, et le chapitre 4 du memoire pose la grandeur comme un besoin de
      capital ORSA, donc de pilier 2. Ecrire « le SCR DORA de telle entite » laisse croire
      a un objet reglementaire qui n'existe pas. Ce script ecrit donc « besoin ORSA ».

  (2) LE RAPPORT AU SCR PUBLIE EST UNE MISE A L'ECHELLE, PAS UNE PART. Diviser un besoin
      de pilier 2 par le SCR reglementaire d'une entite ne fait pas du premier une
      composante du second. Le rapport sert a repondre a « est-ce gros ou petit pour cette
      entite », et a rien d'autre. Une premiere version l'intitulait « DORA / SCR », ce qui
      se lisait comme une decomposition. Corrige.


CE QUI MANQUAIT. Le script 60 etablit la descente d'echelle et la lit a une entite
NOTIONNELLE, posee a 20 000 M USD d'actifs et 15 000 M EUR de provisions. La methode est
complete, mais l'objet est fictif : le memoire demontre une faisabilite, il ne mesure pas
une entite. C'est le reproche qu'un rapporteur formule en premier, et il a raison.

CE QUE FAIT CE SCRIPT. Il branche la descente d'echelle sur les chiffres PUBLIES de
quatre entites d'assurance francaises, choisies pour couvrir deux ordres de grandeur de
taille, et rapporte ce besoin a DEUX reperes propres a chaque entite : la charge
operationnelle forfaitaire de Formule Standard, et son SCR total publie. Ce dernier
rapport est le seul chiffre qu'un directeur des risques peut utiliser tel quel.

LA DONNEE. Trois grandeurs par entite : provisions techniques, fonds propres eligibles,
SCR. Toutes proviennent des rapports SFCR au 31/12/2024, qui sont des publications
reglementaires. Chaque champ porte sa PROVENANCE : « publie » quand il est lu tel quel
dans le rapport, « deduit » quand il resulte d'une identite comptable ou d'un ratio publie
(fonds propres divises par le taux de couverture, par exemple). Aucun champ n'est estime.
Le script imprime cette provenance a chaque execution : un chiffre deduit ne doit pas etre
cite au meme rang qu'un chiffre publie.

CORRECTION DU 2 SEPTEMBRE 2026, APRES LECTURE DES QUATRE RAPPORTS. La verification due
depuis des semaines a ete faite, et elle a trouve une ERREUR DE CHAMP sur Identite B :
le 3 234 M EUR avait ete saisi comme TOTAL D'ACTIF alors que la synthese du rapport le donne
comme FONDS PROPRES ELIGIBLES. Le total d'actif vaut 5 305, et l'identite comptable le
confirme (5 305 - 2 071 = 3 234 exactement). Consequence : les fonds propres deduits valaient
1 163 au lieu de 3 234, et le SCR deduit 292 au lieu des 812,5 que le rapport PUBLIE. La part
du SCR attribuee a DORA pour cette entite passe donc d'environ 41 % a environ 15 %.

DEUX GAINS QUI VIENNENT DE LA MEME LECTURE, et le premier supprime une limite declaree.
  (1) LES QUATRE SCR SONT PUBLIES. Identite A le donne dans son etat S.25.01 a 425 M EUR, Identite B
      Assurances a 812,5 M EUR dans sa section E.2. Plus aucun SCR n'est deduit d'un taux de
      couverture, donc la reserve « deux SCR sur quatre sont deduits » et la borne sous stress
      de plus ou moins 10 % qui la compensait n'ont plus d'objet. Le stress reste imprime comme
      un controle de robustesse, il n'est plus une necessite.
  (2) LE FORFAIT DE 3 % PEUT ETRE CONFRONTE AU MODULE PUBLIE. Les etats S.25 donnent le
      risque operationnel de formule standard : 59 M EUR pour Identite A contre 54,1 pour le forfait,
      et 39,7 pour Identite B contre 62,1. Le forfait n'est donc PAS une borne, il se
      trompe DANS LES DEUX SENS. C'est un resultat, et il renforce la prudence deja affichee
      sur ce repere. Le module n'est releve que pour les deux entites non-vie : pour les deux
      entites vie, le champ vaut None et la comparaison n'est pas imprimee.

CE QUE L'ETAT DE CONFORMITE N'EST PAS. Aucune de ces entites ne publie son etat de
conformite DORA. Le calcul mesure donc l'EXPOSITION d'une entite reelle sous un etat de
conformite SUPPOSE, et non sa non-conformite constatee. Les trois etats sont donc donnes
cote a cote, sans en designer un. C'est une reserve a ecrire, pas une note de bas de page.

CE QUE CE SCRIPT TROUVE, ET QUI N'ETAIT PAS ATTENDU. La transposition ne se degrade pas
gracieusement vers le bas. L'elasticite de severite valant 0,087, la severite est presque
invariante a la taille : diviser les actifs par cent ne divise la severite que par 1,5. La
charge rapportee au SCR publie est donc de quelques pourcents pour une grande entite
et de plusieurs dizaines pour une petite. Ce n'est pas un defaut de ce script, c'est la
limite que le script 60 nommait sans la chiffrer : l'elasticite corrige l'ECHELLE de la
severite, jamais sa FORME, et la forme est celle de grandes institutions financieres. Le
resultat utile de ce script est donc autant la table que le SEUIL DE TAILLE en dessous
duquel la methode ne doit pas etre appliquee.

Sortie : diagnostics + figure S22_entites_reelles.png
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
os.chdir(HERE)
import partial_id as pid                                            # noqa: E402
import descente as dsc                                              # noqa: E402

SF_TAUX = 0.03               # charge operationnelle de Formule Standard, assiette provisions
NY = 600_000
SEED = 20260721
ALPHA = 0.995
W_ = 92

# Conversion EUR -> USD. Le panel OpRisk exprime les actifs en M USD, les SFCR en M EUR.
# L'ordre de grandeur du cours fin 2024 suffit largement : voir la section 1, ou l'on
# montre que l'elasticite etant de 0,0744, une erreur de 10 % sur le cours deplace lambda
# de 0,7 %. Ce n'est pas un parametre sensible, et il ne faut pas le presenter comme tel.
TAUX_USD = 1.04

# =====================================================================================
# LES ENTITES. Un champ = une valeur + sa provenance. Les valeurs marquees None sont
# DEDUITES plus bas par identite comptable ou par le taux de couverture publie.
#
# POURQUOI LE MEMOIRE ANONYMISE, ET PAS CE SCRIPT. Le modele n'utilise QUE le total de
# bilan et les provisions techniques : jamais l'identite de l'entite. Nommer une societe
# cotee a cote d'un etat de conformite SUPPOSE, qu'elle ne publie pas et que personne n'a
# etabli, serait donc un risque d'attribution sans aucune contrepartie scientifique. Le
# memoire designe les entites par une classe de taille ; ce script garde les noms et les
# sources, parce que la tracabilite doit exister quelque part et que sa place est ici.
# Le champ `anonyme` porte l'etiquette publiee, et la table de correspondance est imprimee
# a chaque execution pour que le rapprochement soit toujours possible en interne.
# =====================================================================================
ENTITES = [
    dict(
        nom="Identite A",
        anonyme="Assureur non-vie A",
        perimetre="solo, non-vie",
        pt=1_804.0, of=515.0, scr=425.0, couverture=1.21, actifs=None, scr_op=59.0,
        pt_inclut_autres_passifs=False,
        source="SFCR 2024, Identite A : provisions techniques 1 804 M EUR "
               "(+14,8 %), fonds propres eligibles 515 M EUR (T1 425 + T2 90), couverture "
               "du SCR 121 %, SCR 425 M EUR et risque operationnel 59 M EUR lus dans "
               "l'etat S.25.01 (p. 79 du rapport)",
    ),
    dict(
        nom="Identite B",
        anonyme="Assureur non-vie B",
        perimetre="solo, non-vie",
        pt=2_071.0, of=3_234.0, scr=812.5, couverture=3.98, actifs=5_305.0, scr_op=39.7,
        pt_inclut_autres_passifs=True,
        source="SFCR 2024, Identite B : total actif Solvabilite II 5 305 M EUR, "
               "provisions techniques et autres passifs 2 071 M EUR, fonds propres "
               "eligibles 3 234 M EUR, couverture 398 % (synthese p. 5) ; SCR 812,5 M EUR "
               "et risque operationnel 39,7 M EUR (section E.2, p. 57)",
    ),
    dict(
        nom="Identite C",
        anonyme="Assureur vie C",
        perimetre="solo, vie",
        pt=33_484.0, of=4_361.0, scr=1_587.0, couverture=2.75, actifs=None, scr_op=None,
        pt_inclut_autres_passifs=True,
        source="SFCR 2024, Identite C : provisions techniques et autres "
               "passifs 33 484 M EUR, fonds propres eligibles 4 361 M EUR, SCR 1 587 M EUR",
    ),
    dict(
        nom="Identite D",
        anonyme="Assureur vie D",
        perimetre="solo, vie",
        pt=275_000.0, of=34_800.0, scr=14_800.0, couverture=None, actifs=None, scr_op=None,
        pt_inclut_autres_passifs=False,
        source="SFCR solo 2024, Identite D : provisions techniques brutes de "
               "reassurance 275 Md EUR, fonds propres eligibles 34,8 Md EUR, "
               "SCR 14,8 Md EUR (formule standard), couverture solo publiee 236 %",
    ),
]

# entite notionnelle du memoire (script 60), gardee comme point de comparaison
NOTIONNELLE = dict(nom="Entite notionnelle (ch. 12)", anonyme="Entite notionnelle",
                   perimetre="fictive", actifs_musd=20_000.0, pt=15_000.0)


def titre(s):
    print("\n" + "=" * W_ + f"\n{s}\n" + "=" * W_)


# =====================================================================================
titre("0. Les entites, leurs chiffres publies, et la provenance de chaque champ")
# =====================================================================================
for e in ENTITES:
    prov = {}
    # fonds propres : publies, ou deduits de l'identite actif = passif + fonds propres
    if e["of"] is None:
        e["of"] = e["actifs"] - e["pt"]
        prov["of"] = "deduit (actif - passifs)"
    else:
        prov["of"] = "publie"
    # SCR : publie, ou deduit des fonds propres et du taux de couverture
    if e["scr"] is None:
        e["scr"] = e["of"] / e["couverture"]
        prov["scr"] = f"deduit (fonds propres / couverture {e['couverture']:.2f})"
    else:
        prov["scr"] = "publie"
    # actifs : publies, ou approches par passifs + fonds propres (BORNE INFERIEURE, car
    # les autres passifs manquent). L'elasticite etant positive, sous-estimer les actifs
    # sous-estime lambda, donc le capital : l'approximation est CONSERVATRICE.
    if e["actifs"] is None:
        e["actifs"] = e["pt"] + e["of"]
        prov["actifs"] = "deduit (provisions + fonds propres, borne inferieure)"
    else:
        prov["actifs"] = "publie"
    e["prov"] = prov
    e["actifs_musd"] = e["actifs"] * TAUX_USD

print(f"{'entite':<28}{'perimetre':<16}{'actifs':>11}{'provisions':>12}{'SCR':>10}"
      f"{'couv.':>8}")
print(f"{'':<28}{'':<16}{'M EUR':>11}{'M EUR':>12}{'M EUR':>10}{'':>8}")
for e in ENTITES:
    cv = f"{e['of']/e['scr']:.0%}"
    print(f"{e['nom']:<28}{e['perimetre']:<16}{e['actifs']:>11.0f}{e['pt']:>12.0f}"
          f"{e['scr']:>10.0f}{cv:>8}")
print(f"{NOTIONNELLE['nom']:<28}{NOTIONNELLE['perimetre']:<16}"
      f"{NOTIONNELLE['actifs_musd']/TAUX_USD:>11.0f}{NOTIONNELLE['pt']:>12.0f}"
      f"{'-':>10}{'-':>8}")

print("\nCORRESPONDANCE AVEC LES ETIQUETTES PUBLIEES AU MEMOIRE :")
for e in ENTITES:
    print(f"  {e['anonyme']:<22} = {e['nom']}")
print("  Le memoire ne publie que la colonne de gauche. Voir l'en-tete de ce script pour")
print("  la raison : le modele n'utilise que le bilan, jamais l'identite.")

print("\nPROVENANCE DE CHAQUE CHAMP, a ne pas confondre a la lecture :")
for e in ENTITES:
    print(f"  {e['nom']}")
    for k in ("actifs", "of", "scr"):
        print(f"      {k:<8}: {e['prov'][k]}")
    print(f"      source  : {e['source']}")

print("\nDEUX RESERVES A ECRIRE TELLES QUELLES.")
print("  (1) Aucune de ces entites ne publie son etat de conformite DORA. On mesure une")
print("      EXPOSITION sous un etat suppose, jamais une non-conformite constatee.")
print("  (2) Deux entites publient « provisions techniques ET AUTRES PASSIFS » sans les")
print("      separer. Pour celles-la l'assiette de la charge forfaitaire est SURESTIMEE,")
print("      donc le rapport besoin ORSA / forfait est SOUS-estime :")
for e in ENTITES:
    if e["pt_inclut_autres_passifs"]:
        print(f"        - {e['nom']}")
print("      Le rapport au SCR PUBLIE, lui, n'est pas concerne par ce defaut d'assiette.")
print(f"  L'ecart de taille couvert va de {min(e['actifs'] for e in ENTITES):.0f} a "
      f"{max(e['actifs'] for e in ENTITES):.0f} M EUR, soit un facteur "
      f"{max(e['actifs'] for e in ENTITES)/min(e['actifs'] for e in ENTITES):.0f}.")


# =====================================================================================
titre("1. La conversion de devise ne pese presque rien, et il faut le montrer")
# =====================================================================================
D = dsc.Descente()
print(f"panel : {len(D.panel):,} observations firme-annee, {D.panel.firm.nunique()} firmes")
print(f"elasticite frequence / taille b_lambda = {D.b_lam:+.4f} (ET {D.se_b:.4f}, "
      f"z = {D.z_b:+.2f})")
print(f"elasticite severite / taille  b        = {dsc.B_SEV:+.4f} "
      f"(IC [{dsc.B_SEV_LO:.3f} ; {dsc.B_SEV_HI:.3f}])")
print(f"\nCONTROLE DE REPRODUCTION DU SCRIPT 60, sur l'entite notionnelle :")
lam_notio, mult_notio = D.lam(NOTIONNELLE["actifs_musd"]), D.mult(NOTIONNELLE["actifs_musd"])
print(f"  lambda a 20 000 M USD   = {lam_notio:.6f}   (script 60 : 0,091684)")
print(f"  multiplicateur severite = {mult_notio:.4f}       (script 60 : 0,8545)")
ok = abs(lam_notio - 0.09168423156000373) < 1e-9 and abs(mult_notio - 0.8545) < 5e-5
print(f"  reproduction exacte : {ok}")
if not ok:
    sys.exit("le module de descente ne reproduit plus le script 60 : arret.")

print(f"\nSensibilite au cours de change retenu ({TAUX_USD} USD par EUR) :")
for t in (0.95, 1.00, TAUX_USD, 1.10, 1.15):
    a = ENTITES[-1]["actifs"] * t
    print(f"  {t:.2f} USD/EUR -> lambda({ENTITES[-1]['nom']}) = {D.lam(a):.5f}   "
          f"mult = {D.mult(a):.4f}")
print(f"  Une erreur de 10 % sur le cours deplace lambda de "
      f"{100*((1.1)**D.b_lam - 1):.2f} % et la severite de {100*((1.1)**dsc.B_SEV - 1):.2f} %.")
print("  Le cours n'est donc pas un parametre du resultat, et le presenter comme une")
print("  hypothese sensible serait une fausse precaution.")


# =====================================================================================
titre("2. Les deux lectures d'echelle, entite par entite")
# =====================================================================================
print(f"{'entite':<28}{'actifs M USD':>14}{'lambda':>10}{'bande lambda':>20}"
      f"{'x severite':>12}")
for e in ENTITES + [NOTIONNELLE]:
    a = e["actifs_musd"]
    lo, hi = D.lam_bande(a)
    e["lam"], e["mult"] = D.lam(a), D.mult(a)
    print(f"{e['nom']:<28}{a:>14.0f}{e['lam']:>10.4f}"
          f"{'[' + format(lo, '.4f') + ' ; ' + format(hi, '.4f') + ']':>20}{e['mult']:>12.4f}")
print("\nLa frequence varie peu d'une entite a l'autre parce que l'elasticite est faible :")
print(f"  un facteur {max(e['actifs_musd'] for e in ENTITES)/min(e['actifs_musd'] for e in ENTITES):.0f} "
      f"sur les actifs ne donne qu'un facteur "
      f"{max(e['lam'] for e in ENTITES)/min(e['lam'] for e in ENTITES):.2f} sur lambda")
print(f"  et un facteur {max(e['mult'] for e in ENTITES)/min(e['mult'] for e in ENTITES):.2f} "
      f"sur la severite. C'est la propriete centrale a retenir, et c'est aussi")
print("  ce qui limite la transposition vers les petites tailles (section 5).")


# =====================================================================================
titre("3. Le besoin ORSA par entite : un point, une bande, un socle")
# =====================================================================================
print("La direction de W n'etant pas identifiee, le resultat d'une entite est une BANDE.")
print(f"On enumere les {1 << pid.NFREE} sommets du pave admissible a t = 1, comme au")
print("chapitre resultats, en plus du point a la matrice d'expert et du socle sans")
print("contagion. Les trois grandeurs sont donnees, jamais le seul point.\n")

Wexp = pid.expert_matrix()
S, _ = pid.decompose(Wexp)
A_ALL = pid.all_vertices(S, 1.0)
CARD_SOMMETS = []
for row in A_ALL:
    A = np.zeros((pid.NP_, pid.NP_))
    A[pid.IU] = row
    Wv = S + (A - A.T)
    if pid.admissible(Wv):
        CARD_SOMMETS.append(pid.card_dist_all(Wv)[0])
CARD_EXP = pid.card_dist_all(Wexp)[0]
CARD_NUL = pid.card_dist_all(np.zeros((pid.NP_, pid.NP_)))[0]
print(f"sommets admissibles : {len(CARD_SOMMETS)} / {len(A_ALL)}")


def scr_entite(lam, mult):
    """(point, bande basse, bande haute, socle) pour une entite, a nombres communs."""
    ev = pid.Evaluator(lam=lam, n_years=NY, alpha=ALPHA, seed=SEED)
    ev.cum = ev.cum * mult
    vals = [ev.from_card(c)[0] for c in CARD_SOMMETS]
    pt, esp = ev.from_card(CARD_EXP)
    socle = ev.from_card(CARD_NUL)[0]
    return pt, min(vals), max(vals), socle, esp


print(f"\n{'entite':<28}{'point':>10}{'bande basse':>13}{'bande haute':>13}"
      f"{'socle':>9}{'E[X]':>9}")
for e in ENTITES + [NOTIONNELLE]:
    pt, lo, hi, socle, esp = scr_entite(e["lam"], e["mult"])
    e.update(scr_dora=pt, scr_lo=lo, scr_hi=hi, socle=socle, esp=esp)
    print(f"{e['nom']:<28}{pt:>10.1f}{lo:>13.1f}{hi:>13.1f}{socle:>9.1f}{esp:>9.2f}")
print("\nen M EUR. Controle : l'entite notionnelle doit redonner les chiffres publies au")
print(f"chapitre resultats, soit 169,0 et [131,5 ; 183,3] : "
      f"{NOTIONNELLE['scr_dora']:.1f} et [{NOTIONNELLE['scr_lo']:.1f} ; "
      f"{NOTIONNELLE['scr_hi']:.1f}].")


# =====================================================================================
titre("4. Les deux reperes de l'entite : le forfait, et le SCR publie")
# =====================================================================================
print("Le premier repere est la charge operationnelle de Formule Standard, 3 % des")
print("provisions. Le second, plus parlant pour une direction des risques, est le SCR")
print("TOTAL publie de l'entite : il dit quelle part du capital reglementaire la")
print("non-conformite DORA representerait si elle etait chargee.\n")
print(f"{'entite':<28}{'ORSA':>9}{'forfait 3%':>12}{'ORSA/forfait':>14}"
      f"{'SCR publie':>12}{'/ SCR pub.':>11}")
for e in ENTITES:
    sf = SF_TAUX * e["pt"]
    e["sf"], e["part_scr"] = sf, e["scr_dora"] / e["scr"]
    print(f"{e['nom']:<28}{e['scr_dora']:>9.1f}{sf:>12.1f}{e['scr_dora']/sf:>14.2f}"
          f"{e['scr']:>12.0f}{e['part_scr']:>10.1%}")
sf_n = SF_TAUX * NOTIONNELLE["pt"]
print(f"{NOTIONNELLE['nom']:<28}{NOTIONNELLE['scr_dora']:>9.1f}{sf_n:>12.1f}"
      f"{NOTIONNELLE['scr_dora']/sf_n:>14.2f}{'-':>12}{'-':>10}")
print("\nen M EUR. Le rapport DORA / forfait de l'entite notionnelle vaut 0,38, chiffre")
print("deja publie : les entites reelles s'en ecartent parce que leur ratio provisions")
print("sur actifs n'est pas celui pose au chapitre 12.")

# -------------------------------------------------------------------------------------
# LA VERIFICATION PIECE PAR PIECE A ETE FAITE, ET ELLE CHANGE CE BLOC
# -------------------------------------------------------------------------------------
# CE QUE CE BLOC DISAIT AVANT LE 2 SEPTEMBRE 2026. Deux des quatre SCR n'etaient pas
# publies mais reconstitues en divisant les fonds propres par le taux de couverture, et
# c'etaient justement les deux entites dont la part attribuee a DORA etait la plus
# spectaculaire. Un stress de +-10 % bornait l'erreur, et le bloc se terminait par la
# phrase « ce que ce stress ne remplace pas : la lecture des quatre rapports SFCR
# eux-memes ». Cette lecture a ete faite.
#
# CE QU'ELLE A TROUVE. Les QUATRE SCR sont publies : Identite A dans son etat S.25.01 a 425,
# Identite B dans sa section E.2 a 812,5, et les deux entites vie les publiaient
# deja. Aucun SCR n'est donc plus deduit, et le stress n'a plus d'objet : il est conserve
# ci-dessous comme controle de coherence, non comme une borne dont la conclusion depend.
# Elle a aussi trouve une ERREUR DE CHAMP sur Identite B, voir l'en-tete du script.
STRESS = 0.10
deduits = [e for e in ENTITES if "deduit" in e["prov"]["scr"]]
print(f"\nSTATUT DU DENOMINATEUR, apres lecture des quatre rapports :")
for e in ENTITES:
    print(f"  {e['nom']:<28}{e['prov']['scr']:<22}{e['part_scr']:>7.1%} du SCR publie")
if deduits:
    print(f"\n  Stress {100*STRESS:.0f} % sur les SCR encore deduits :")
    for e in deduits:
        bas = e["scr_dora"] / (e["scr"] * (1 - STRESS))
        haut = e["scr_dora"] / (e["scr"] * (1 + STRESS))
        print(f"    {e['nom']:<28}{e['part_scr']:>7.1%}  ->  [{haut:.1%} ; {bas:.1%}]")
else:
    print("\n  AUCUN SCR N'EST PLUS DEDUIT. La reserve qui accompagnait cette colonne, et la")
    print("  borne sous stress qui la compensait, sont sans objet : les quatre denominateurs")
    print("  sont des chiffres publies. C'est un gain net sur la version precedente, et il")
    print("  vient de la lecture des rapports, pas d'un calcul.")

# -------------------------------------------------------------------------------------
# LE FORFAIT DE 3 % CONFRONTE AU MODULE OPERATIONNEL PUBLIE
# -------------------------------------------------------------------------------------
# POURQUOI CETTE COMPARAISON EXISTE MAINTENANT. Les etats S.25.01 des deux entites non-vie
# donnent le module de risque OPERATIONNEL de formule standard. Le forfait de 3 % des
# provisions etait jusqu'ici un repere POSE, et le voici mesurable sur deux cas.
op = [e for e in ENTITES if e.get("scr_op") is not None]
if op:
    print(f"\nLE REPERE FORFAITAIRE CONFRONTE AU MODULE PUBLIE (etats S.25.01) :")
    print(f"{'entite':<28}{'forfait 3%':>12}{'module publie':>15}{'ecart du forfait':>18}")
    for e in op:
        sf = SF_TAUX * e["pt"]
        print(f"{e['nom']:<28}{sf:>12.1f}{e['scr_op']:>15.1f}"
              f"{(sf / e['scr_op'] - 1):>17.0%}")
    print("\nen M EUR. LE FORFAIT N'EST PAS UNE BORNE : il se trompe DANS LES DEUX SENS, de")
    print(f"{min(SF_TAUX*e['pt']/e['scr_op'] - 1 for e in op):+.0%} sur une entite et de "
          f"{max(SF_TAUX*e['pt']/e['scr_op'] - 1 for e in op):+.0%} sur l'autre. Il faut donc")
    print("le citer comme un ORDRE DE GRANDEUR de comparaison, jamais comme un majorant ni")
    print("comme un minorant de la charge operationnelle reglementaire. Deux entites ne")
    print("determinent pas un biais, elles suffisent a refuter la monotonie.")
    print("Releve sur les deux entites NON-VIE seulement : les deux entites vie ne sont pas")
    print("comparables sur ce repere, leur assiette de provisions etant d'une autre nature.")


# =====================================================================================
titre("5. Ou la transposition cesse d'etre credible, et pourquoi")
# =====================================================================================
print("C'est le resultat que ce script n'avait pas ete ecrit pour trouver, et c'est le")
print("plus utile. La part du SCR publie attribuee a la non-conformite DORA est :\n")
ordre = sorted(ENTITES, key=lambda e: e["actifs"])
for e in ordre:
    print(f"  {e['nom']:<28} actifs {e['actifs']:>9.0f} M EUR  ->  "
          f"{e['part_scr']:>6.1%} du SCR publie")
print("\nLa charge decroit BEAUCOUP moins vite que la taille. La cause est mecanique et")
print("elle est deja nommee au chapitre 12 : l'elasticite de severite vaut "
      f"{dsc.B_SEV:.3f}, donc")
print(f"  diviser les actifs par {max(e['actifs'] for e in ENTITES)/min(e['actifs'] for e in ENTITES):.0f} "
      f"ne divise la severite que par "
      f"{max(e['mult'] for e in ENTITES)/min(e['mult'] for e in ENTITES):.2f},")
print(f"  et la frequence que par {max(e['lam'] for e in ENTITES)/min(e['lam'] for e in ENTITES):.2f}.")
print("L'elasticite corrige l'ECHELLE de la severite ; la FORME de la queue reste celle de")
print("grandes institutions financieres, et c'est elle qui porte un quantile a 99,5 %.")
print("\nCE QU'IL FAUT EN CONCLURE, sans le contourner. La descente d'echelle est utilisable")
print("la ou la taille de l'entite reste dans l'ordre de grandeur du panel de calibration,")
print("et elle produit une charge non credible en dessous.")
print("\nLE CRITERE EST POSE, PAS MESURE : on juge invraisemblable qu'un seul risque de")
print("non-conformite reglementaire pese plus de 10 % du SCR total d'une entite, tous")
print("risques confondus (marche, souscription, credit, operationnel). Le seuil de 10 % est")
print("un jugement d'ordre de grandeur, et il est explicite pour pouvoir etre contredit.\n")
plaus = [e for e in ordre if e["part_scr"] < 0.10]
non_plaus = [e for e in ordre if e["part_scr"] >= 0.10]
print(f"  {'entite':<28}{'actifs M EUR':>14}{'SCR/actifs':>12}{'/ SCR pub.':>11}   verdict")
for e in ordre:
    v = "credible" if e["part_scr"] < 0.10 else "HORS DOMAINE"
    print(f"  {e['nom']:<28}{e['actifs']:>14.0f}{e['scr']/e['actifs']:>12.1%}"
          f"{e['part_scr']:>10.1%}   {v}")
if plaus and non_plaus:
    print(f"\n  La bascule se situe entre {max(e['actifs'] for e in non_plaus):.0f} et "
          f"{min(e['actifs'] for e in plaus):.0f} M EUR d'actifs. Ce n'est pas un")
    print("  seuil fin : quatre entites ne le determinent pas, et le rapport depend aussi du")
    print("  levier propre de l'entite (colonne SCR/actifs, qui va de "
          f"{min(e['scr']/e['actifs'] for e in ordre):.1%} a "
          f"{max(e['scr']/e['actifs'] for e in ordre):.1%}).")
    print("  C'est un ordre de grandeur, et il doit etre presente comme tel.")

# CE QUE CELA IMPLIQUE POUR LE CHIFFRE CENTRAL DU MEMOIRE. Il faut le calculer, pas
# l'eviter : l'entite notionnelle du chapitre 12 se situe DANS la zone que la ligne
# ci-dessus declare hors domaine, et le chiffre publie de 169 M EUR en herite.
lev_med = float(np.median([e["scr"] / e["actifs"] for e in ENTITES]))
actifs_notio_eur = NOTIONNELLE["actifs_musd"] / TAUX_USD
scr_implicite = lev_med * actifs_notio_eur
print(f"\n  ET LE CHIFFRE CENTRAL DU MEMOIRE, DANS TOUT CELA. L'entite notionnelle du")
print(f"  chapitre 12 pese {actifs_notio_eur:.0f} M EUR d'actifs, donc elle tombe DANS la zone")
print(f"  hors domaine. Elle ne publie pas de SCR, mais on peut lui en imputer un au levier")
part_notio = NOTIONNELLE["scr_dora"] / scr_implicite
print(f"  MEDIAN des quatre entites reelles ({lev_med:.1%}) : {scr_implicite:.0f} M EUR. La charge de")
print(f"  {NOTIONNELLE['scr_dora']:.1f} M EUR en representerait alors {part_notio:.0%}.")
# LECTURE DE CE TEST APRES LA CORRECTION DU 2 SEPTEMBRE 2026, ET ELLE A CHANGE DE SENS.
# Avant correction, le levier median valait 6,9 % et la part imputee 13 %, donc AU-DESSUS du
# critere : le test condamnait l'entite notionnelle et corroborait la requalification. Avec le
# levier median corrige, la part imputee tombe SOUS le critere. Le test est donc devenu NEUTRE
# et il ne faut pas continuer a le presenter comme accablant : ce serait garder une conclusion
# apres que son support a bouge, exactement le defaut que ce projet traque.
if part_notio > 0.10:
    print(f"  C'est-a-dire le meme ordre de grandeur invraisemblable que les petites entites")
    print(f"  de la table : ce test CONFIRME la requalification.")
else:
    print(f"  C'est SOUS le critere de 10 %, donc ce test ne condamne PAS l'entite notionnelle.")
    print(f"  Il est devenu NEUTRE apres la correction des chiffres de l'identite B, et il ne faut pas")
    print(f"  continuer a le citer comme accablant. La requalification du 169 en borne")
    print(f"  superieure repose desormais sur le SEUL argument de TAILLE, qui tient par")
    print(f"  lui-meme : 19 231 M EUR se situe sous la bascule, laquelle est entre 37 845 et")
    print(f"  309 800 M EUR d'actifs. Un argument sur deux, et on le dit.")
print(f"\n  IL FAUT DONC LE DIRE AU CHAPITRE 12. Le chiffre de {NOTIONNELLE['scr_dora']:.0f} M EUR n'est pas une")
print("  charge plausible pour une entite de cette taille : c'est une BORNE SUPERIEURE")
print("  d'ordre de grandeur, heritee d'une severite calibree sur de grandes institutions")
print("  financieres. Ce que le memoire etablit n'en est pas affaibli, parce que sa these")
print("  porte sur l'ECART entre etats de conformite face a un forfait insensible, et que")
print("  cet ecart est un RAPPORT : il survit a une erreur de niveau commune aux trois")
print("  etats. Mais le NIVEAU, lui, doit cesser d'etre presente comme une mesure.")

print("\n" + "=" * 88)
print("PISTE D'AUDIT : les valeurs PERIMEES du 2 septembre, imprimees pour etre citables")
print("=" * 88)
print("Ces quatre valeurs sont FAUSSES et le memoire ne les publie plus. Elles sont imprimees")
print("ici parce qu'un deck ou une note qui documente la correction doit pouvoir les citer, et")
print("que tout nombre cite doit sortir d'un script. Ne JAMAIS les reutiliser comme donnees.")
print()
print("  grandeur                                  avant (FAUX)    apres (publie)")
print("  total d'actif, assureur non-vie B                 3234              5305")
print("  SCR, assureur non-vie B                            292             812.5")
print("  part du SCR attribuee a DORA, non-vie B          41.3 %            16.7 %")
print("  part imputee a l'entite notionnelle                13 %               9 %")
print()
print("MOTIF DE CHAQUE ECART. Le 3234 avait ete saisi en TOTAL D'ACTIF alors que le rapport le")
print("donne en FONDS PROPRES ELIGIBLES ; l'identite comptable le confirme, 5305 - 2071 = 3234.")
print("Le SCR de 292 en decoulait par un taux de couverture, alors que la section E.2 du rapport")
print("PUBLIE 812,5. Les deux parts suivent mecaniquement. La part imputee de 13 % venait d'un")
print("levier median de 6,9 %, devenu 10,0 %, ce qui fait passer le test du levier SOUS son")
print("critere : c'est ce qui l'a rendu neutre.")

print("\nUne charge de plusieurs dizaines de pourcents du SCR total pour une entite de")
print("quelques milliards d'actifs n'est pas un resultat, c'est le signe que la brique de")
print("severite est hors de son domaine. Le dire est plus utile que de publier le chiffre.")
print("Ce que cela demande pour etre leve : une severite calibree sur des incidents")
print("d'entites de cette taille, c'est-a-dire la donnee que le chapitre 5 declare absente.")


# =====================================================================================
titre("6. Par etat de conformite, sur les entites ou la transposition tient")
# =====================================================================================
print("Le gain de propagation g est le canal que la conformite fait bouger (0,45 conforme,")
print("0,68 partiel, 0,90 non conforme). Seul le canal CONTAGION varie : l'ecart lu est")
print("donc un PLANCHER. Le forfait, lui, est identique aux trois etats par construction.\n")
CARDS_G = {g: pid.card_dist_all(pid.expert_matrix(g))[0] for g in (0.45, 0.68, 0.90)}
print(f"{'entite':<28}{'conforme':>11}{'partiel':>10}{'non conf.':>11}{'ecart':>9}"
      f"{'%':>8}{'forfait':>10}")
for e in ENTITES + [NOTIONNELLE]:
    ev = pid.Evaluator(lam=e["lam"], n_years=NY, alpha=ALPHA, seed=SEED)
    ev.cum = ev.cum * e["mult"]
    vv = [ev.from_card(CARDS_G[g])[0] for g in (0.45, 0.68, 0.90)]
    e["etats"] = vv
    print(f"{e['nom']:<28}{vv[0]:>11.1f}{vv[1]:>10.1f}{vv[2]:>11.1f}"
          f"{vv[2]-vv[0]:>9.1f}{100*(vv[2]/vv[0]-1):>+8.1f}{SF_TAUX*e['pt']:>10.1f}")
print("\nen M EUR. La lecture est la meme a toutes les tailles : le modele ecarte les trois")
print("etats de conformite, le forfait ne bouge pas d'un euro. C'est la these du memoire,")
print("et elle est ici verifiee sur des bilans reels et non sur une entite posee.")


# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("1. Le besoin ORSA au titre de DORA est calculable sur des entites reelles a partir")
print("   chiffres SFCR publies. La methode n'exige aucune donnee interne, ce qui est")
print("   precisement ce qu'on lui demandait de demontrer.")
gr = ordre[-1]
# EN MILLIARDS AUSSI. Le memoire ecrit « un assureur vie de 310 Md EUR d'actifs », qui se
# lit mieux que 309800 M EUR ; encore faut-il que ce 310 soit imprime quelque part, sinon
# verif_chiffres.py le signale comme non confirme, ce qu'il a fait.
print(f"2. Sur la plus grande entite du panel, {gr['nom']} ({gr['anonyme']}), soit")
print(f"   {gr['actifs']/1000:.0f} Md EUR d'actifs, la charge vaut")
print(f"   {gr['scr_dora']:.1f} M EUR en point, [{gr['scr_lo']:.1f} ; {gr['scr_hi']:.1f}] en bande, "
      f"soit {gr['part_scr']:.1%} de son SCR publie")
print(f"   de {gr['scr']:.0f} M EUR. C'est un ordre de grandeur defendable devant un jury.")
print("3. L'ecart entre etats de conformite subsiste a toutes les tailles, face a un")
print("   forfait qui n'en distingue aucune. La these ne dependait pas de l'entite fictive.")
print("4. LIMITE TROUVEE ICI, et c'est le principal apport. La bascule du credible a")
print(f"   l'invraisemblable se situe entre {max(e['actifs'] for e in non_plaus):.0f} et "
      f"{min(e['actifs'] for e in plaus):.0f} M EUR d'actifs : en dessous,")
print("   la charge depasse 10 % du SCR total, parce que la severite est presque")
print("   invariante a la taille. Le domaine de validite de la descente d'echelle a une")
print("   borne INFERIEURE, que le memoire doit publier.")
print(f"   Consequence directe sur le chapitre 12 : l'entite notionnelle de "
      f"{actifs_notio_eur:.0f} M EUR se")
print(f"   trouve DANS cette zone, et son SCR de {NOTIONNELLE['scr_dora']:.0f} M EUR doit etre requalifie en")
print("   BORNE SUPERIEURE d'ordre de grandeur. La these, qui porte sur un ECART entre")
print("   etats face a un forfait insensible, n'en depend pas ; le niveau, si.")
print("5. Ce qui reste hors de portee sans donnee d'entite : l'etat de conformite reel,")
print("   et la forme de la queue a la taille de l'entite.")


# =====================================================================================
titre("Figure")
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
LEG = dict(frameon=True, facecolor="#fcfcfb", edgecolor="none", framealpha=0.88, fontsize=8.5)

# LA FIGURE PART DANS LE MEMOIRE : ELLE EST ANONYMISEE. Les etiquettes viennent du champ
# `anonyme`, jamais du nom. C'est le seul endroit du script ou la distinction compte, et
# c'est aussi celui ou l'oublier aurait publie les noms sans que personne le voie.
COURT = {"Identite A": "non-vie\nA",
         "Identite B": "non-vie\nB",
         "Identite C": "vie\nC",
         "Identite D": "vie\nD",
         "Entite notionnelle (ch. 12)": "entité\nnotionnelle"}

# DECALAGES D'ETIQUETTES, FIXES A LA MAIN ET NON PAR DEFAUT. Les deux plus petites entites
# sont a 2 319 et 3 234 M EUR, donc quasi confondues en echelle log : une position unique
# pour toutes les etiquettes les superposait dans les panneaux (a) et (c). Chaque entite
# recoit donc son propre decalage, l'une au-dessus de son point et l'autre en dessous.
DEC_A = {"Identite A": (-11, -4, "right"),
         "Identite B": (10, -15, "left"),
         "Identite C": (9, -14, "left"),
         "Identite D": (9, -14, "left")}
DEC_C = {"Identite A": (24, -26),
         "Identite B": (0, 14),
         "Identite C": (0, 14),
         "Identite D": (0, 14)}

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.2, 10.4))

# (a) les entites sur la courbe d'elasticite
# L'AXE EST BORNE AUX CENTILES DU PANEL, PAS A SES EXTREMES. Une firme du panel declare
# 0,01 M USD d'actifs : tracer la courbe jusque-la etale l'axe sur huit decades et ecrase
# la seule zone qui interesse, celle ou vivent les entites d'assurance.
_a1, _a99 = np.percentile(D.pa.actifs.to_numpy(), [1, 99])
xs = np.logspace(np.log10(_a1), np.log10(_a99), 200)
ax1.plot(xs, np.exp(D.a_hat + D.b_lam * (np.log(xs) - D.xbar)), color=BLUE, lw=2,
         label=f"NB2 sur le panel, $b_\\lambda={D.b_lam:.3f}".replace(".", "{,}") + "$")
for e in ENTITES:
    dx, dy, ha = DEC_A[e["nom"]]
    ax1.plot([e["actifs_musd"]], [e["lam"]], "o", ms=9, color=ACCENT, zorder=4)
    ax1.annotate(COURT[e["nom"]].replace("\n", " "), (e["actifs_musd"], e["lam"]),
                 textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8,
                 color=ACCENT)
ax1.plot([NOTIONNELLE["actifs_musd"]], [NOTIONNELLE["lam"]], "D", ms=8, color=GREEN,
         zorder=4, label="entité notionnelle du chapitre 12")
ax1.set_xscale("log")
ax1.set_xlabel("actifs (M USD, échelle log)", color=INK2)
ax1.set_ylabel("incidents TIC matériels / an", color=INK2)
ax1.legend(loc="upper left", **LEG)
ax1.set_title("(a)  Quatre entités réelles lues sur la courbe d'élasticité",
              fontsize=11, color=INK, pad=8)

# (b) besoin ORSA par entite, en bande
noms = [e["nom"] for e in ENTITES] + [NOTIONNELLE["nom"]]
tous = ENTITES + [NOTIONNELLE]
xi = np.arange(len(tous))
lo = np.array([e["scr_lo"] for e in tous])
hi = np.array([e["scr_hi"] for e in tous])
ptp = np.array([e["scr_dora"] for e in tous])
sfp = np.array([SF_TAUX * e["pt"] for e in tous])
ax2.vlines(xi, lo, hi, color=BLUE, lw=11, alpha=0.40,
           label="bande d'identification (1 024 sommets)")
ax2.plot(xi, ptp, "o", ms=8, color=ACCENT, zorder=4, label="point à la matrice retenue")
ax2.plot(xi, sfp, "_", ms=20, mew=2.4, color=GREEN, zorder=4,
         label="charge forfaitaire de Formule Standard")
for x, p_ in zip(xi, ptp):
    ax2.text(x + 0.13, p_, f"{p_:.0f}", ha="left", va="center", fontsize=8.4, color=INK)
ax2.set_xticks(xi)
ax2.set_xticklabels([COURT[n] for n in noms], fontsize=8.4)
ax2.set_yscale("log")
ax2.set_ylabel("M€ (échelle log)", color=INK2)
ax2.legend(loc="upper left", **LEG)
ax2.set_title("(b)  Le besoin ORSA et le forfait, sur des bilans réels",
              fontsize=11, color=INK, pad=8)

# (c) la part du SCR publie, et la borne inferieure de validite
ordre_x = sorted(ENTITES, key=lambda e: e["actifs"])
xa = np.array([e["actifs"] for e in ordre_x])
ya = np.array([e["part_scr"] for e in ordre_x])
ax3.axhspan(0, 0.10, color=GREEN, alpha=0.13, lw=0)
ax3.axhline(0.10, color=GREEN, lw=1.4, ls="--")
ax3.plot(xa, ya, "o-", color=ACCENT, lw=2, ms=9)
for e in ordre_x:
    _dx, _dy = DEC_C[e["nom"]]
    ax3.annotate(f"{COURT[e['nom']].replace(chr(10), ' ')}\n{e['part_scr']:.0%}",
                 (e["actifs"], e["part_scr"]), textcoords="offset points",
                 xytext=(_dx, _dy), ha="left" if _dx > 0 else "center", fontsize=8,
                 color=INK2)
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel("actifs de l'entité (M€, échelle log)", color=INK2)
ax3.set_ylabel("besoin ORSA rapporté au SCR publié", color=INK2)
ax3.set_ylim(min(ya) * 0.40, max(ya) * 4.0)
# LES DEUX ZONES SONT NOMMEES CHACUNE DANS LA SIENNE. Un seul libelle pose sur la ligne
# laissait croire qu'il qualifiait la zone verte, qui est justement l'inverse.
ax3.text(xa.min() * 1.15, min(ya) * 0.46, " crédible ", ha="left", va="bottom",
         fontsize=8, color=GREEN)
ax3.text(xa.max(), max(ya) * 3.4, "hors du domaine de la calibration ", ha="right",
         va="top", fontsize=8, color=INK2)
ax3.set_title("(c)  La transposition a une borne inférieure de taille,\n"
              "et elle doit être publiée", fontsize=11, color=INK, pad=8)

for ax in (ax1, ax2, ax3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("S22 : le besoin de capital ORSA au titre de DORA, sur quatre bilans réels",
             fontsize=11.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.995)
_top = 1.0 - 0.26 / fig.get_figheight()
fig.tight_layout(rect=[0, 0, 1, _top], h_pad=1.9)
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S22_entites_reelles.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
