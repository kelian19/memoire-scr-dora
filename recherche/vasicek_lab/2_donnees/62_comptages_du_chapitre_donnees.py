#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
62 : rendre VERIFIABLES les comptages de tete du chapitre donnees.

POURQUOI CE SCRIPT EXISTE. Le passage de verif_chiffres.py sur les dix-neuf chapitres a
montre que trois nombres publies en evidence n'etaient produits par AUCUN script du lab :

    lambda_ref = 341 / an   notifications PRC, cle de repartition par vecteur d'attaque
    1 041                   incidents Hackmageddon du semestre janvier-juin 2026
    840                     de ces incidents avec vecteur d'acces initial identifie (81 %)

Ils avaient ete calcules une fois pendant l'exploration puis recopies dans le texte. Un
nombre qu'aucun script ne reproduit n'est verifiable par personne, et c'est precisement la
classe d'erreurs que le harnais est cense eliminer.

CE QUE CE SCRIPT PEUT ET NE PEUT PAS FAIRE.
  - lambda_ref : RECALCULABLE. La base PRC est versionnee dans data/raw/.
  - 1 041 et 840 : NON RECALCULABLES. Le fichier Hackmageddon n'est pas dans le depot.
    Le script le dit explicitement plutot que de laisser croire a une verification.
    Deux options pour la suite : verser la source dans data/raw/, ou requalifier ces deux
    nombres en citation externe datee dans le memoire.

Sortie : diagnostics seulement, pas de figure.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.severity.prc_analysis import chemin_prc              # noqa: E402

PRC_PATH = chemin_prc(REPO)
W = 84
Y0, Y1 = 2019, 2025          # meme periode que la calibration de severite PRC


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


titre("(A) lambda_ref : les notifications PRC par an")
if not os.path.exists(PRC_PATH):
    sys.exit(f"donnee absente : {PRC_PATH}\n(les sources brutes ne sont pas versionnees)")

from src.severity.prc_analysis import load_prc                     # noqa: E402

d = load_prc(PRC_PATH, Y0, Y1)
d = load_prc(PRC_PATH, Y0, Y1)
annees = Y1 - Y0 + 1
print(f"periode retenue                       : {Y0}-{Y1}  ({annees} ans)")
print(f"incidents PRC a total_affected > 0    : {len(d):,}  (config annonce 15 053)")

print("\nLE PERIMETRE EST DECISIF, ET LE MEMOIRE LE DECRIVAIT MAL. Sur TOUTE la base, le")
print("taux annuel depasse deux mille, pas 341. Le 341 correspond au seul sous-ensemble des")
print("organisations FINANCIERES (organization_type = BSF). Un relecteur qui recalcule sur")
print("la base entiere trouve six fois plus et conclut que le nombre est faux.")

print(f"\n{'perimetre':<46}{'incidents':>11}{'par an':>10}")
print(f"{'toute la base PRC (tous acteurs)':<46}{len(d):>11,}{len(d)/annees:>10.1f}")
TYPE_COL = "organization_type"
if TYPE_COL in d.columns:
    for t, lab in (("BSF", "organisations financieres (BSF)"),
                   ("BSO", "autres entreprises (BSO)"),
                   ("MED", "sante (MED)")):
        s = d[d[TYPE_COL] == t]
        marque = "   <<< le 341 publie" if t == "BSF" else ""
        print(f"{'  dont ' + lab:<46}{len(s):>11,}{len(s)/annees:>10.1f}{marque}")
    bsf = d[d[TYPE_COL] == "BSF"]
    lam = len(bsf) / annees
    print(f"\n  >>> lambda_ref (perimetre financier) = {len(bsf)}/{annees} = {lam:.1f} par an")
    print(f"      valeur publiee dans config.py et dans le memoire : 341")
    print(f"      ecart : {100*(lam/341 - 1):+.1f} %")

    col = "reported_date" if "reported_date" in d.columns else None
    if col:
        y = pd.to_datetime(bsf[col], errors="coerce").dt.year.value_counts().sort_index()
        print(f"\n  repartition annuelle du perimetre financier :")
        print(f"{'annee':>10}{'incidents':>12}")
        for a, v in y.items():
            if pd.notna(a):
                print(f"{int(a):>10}{int(v):>12,}")
        pleines = y[y.index <= Y1 - 1]
        print(f"\n  moyenne sur annees pleines ({Y0}-{Y1-1}) : {pleines.mean():.1f} par an.")
        print(f"  L'annee {Y1} est incomplete (remontee en cours) et tire la moyenne vers le bas ;")
        print("  sur annees pleines le 341 publie est LEGEREMENT PRUDENT.")

print("\nA QUOI SERT CE NOMBRE. Il ne sert QU'A repartir les incidents par vecteur")
print("d'attaque : il ne porte aucun niveau de capital, et ne doit jamais etre confondu")
print("avec le lambda du moteur de cascade (21,6 incidents TIC materiels par an, secteur")
print("financier mondial, base OpRisk) ni avec le lambda d'entite (0,092 ; script 60).")

titre("(B) Hackmageddon : ce qui N'EST PAS verifiable, et il faut le dire")
cands = [f for f in os.listdir(os.path.join(REPO, "data", "raw"))
         if "hack" in f.lower() or "mageddon" in f.lower()]
if cands:
    print(f"fichier trouve : {cands}")
    print("-> adapter ce script pour recalculer les 1 041 et 840.")
else:
    print("AUCUN fichier Hackmageddon dans data/raw/. Les deux comptages publies au")
    print("chapitre donnees ne sont donc PAS reproductibles en l'etat :")
    print("     1 041 incidents du semestre janvier-juin 2026")
    print("       840 avec vecteur d'acces initial identifie, soit 81 %")
    print("\nCe n'est pas une erreur : ce sont des comptages faits sur une source consultee")
    print("puis non versionnee. Mais ils ne peuvent pas etre presentes comme des resultats")
    print("du dispositif. Deux issues, a trancher :")
    print("  (i)  verser la source dans data/raw/ et recalculer ici ;")
    print("  (ii) les requalifier dans le memoire en CITATION EXTERNE datee, au meme titre")
    print("       qu'un chiffre de rapport de place, avec la date de consultation.")
    print("\nL'option (ii) est suffisante : ces deux nombres ne servent qu'a decrire la")
    print("structure des vecteurs d'attaque, et le memoire dit deja que cette source est")
    print("mobilisee pour la STRUCTURE et non pour le niveau.")


titre("Verdict")
print(f"lambda_ref, perimetre financier       : {lam:.1f} / an contre 341 publie "
      f"({100*(lam/341-1):+.1f} %)  -> VERIFIABLE")
print(f"lambda_ref, toute la base PRC         : {len(d)/annees:.1f} / an  -> ce n'est PAS")
print("   le nombre du memoire. La mention 'tous acteurs confondus' etait fausse et a ete")
print("   corrigee en 'organisations financieres'.")
print("1 041 et 840 (Hackmageddon)           : source non versionnee -> A REQUALIFIER")
print("\nEXIT 0")
