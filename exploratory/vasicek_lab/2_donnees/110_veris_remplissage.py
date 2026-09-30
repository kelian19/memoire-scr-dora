#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
110 : le schema VERIS porte les deux champs manquants, et ne les remplit pas.

POURQUOI CE SCRIPT EXISTE, ET C'EST UN DEFAUT DU DISPOSITIF QU'IL CORRIGE.
Le chapitre d'identifiabilite publie quatre nombres sur la VERIS Community Database :
le champ control_failure renseigne dans 13 incidents sur 10 591, la chronologie datee
sur au moins deux etapes dans 126, sur au moins trois dans 10, et l'intersection des
deux conditions dans UN SEUL incident, aucun en secteur financier. Ces quatre nombres
n'etaient imprimes par AUCUN script versionne : ils avaient ete calcules pendant la
redaction et saisis a la main.

ET LE HARNAIS NE L'AVAIT PAS VU, PARCE QU'IL LES CONFIRMAIT PAR HASARD. La section
cite cinq scripts (05, 57, 28, 40, 64) et le harnais cherche chaque nombre publie dans
l'UNION de leurs sorties. Dans ce pool, le 13 se confirmait contre une ligne du triangle
chain-ladder du script 57, le 126 contre un montant en euros du script 64, le 0,12
contre un ecart de forme de Dirichlet du script 40. Aucune de ces correspondances n'a
de rapport avec VERIS. C'est exactement le piege que le projet documente depuis aout :
un pool de plusieurs milliers de valeurs produit toujours une correspondance fortuite,
et une correspondance numerique seule ne conclut jamais.

LE RESULTAT DE CE SCRIPT EST DONC RASSURANT, ET C'EST LE POINT : les quatre nombres
publies sont VRAIS. Ce qui manquait n'etait pas leur exactitude, c'etait leur source.

LE PIEGE DE DEFINITION, MESURE ICI. << Datee >> peut se lire de deux facons, et les deux
ne donnent pas le meme compte :
  - LACHE   : l'etape porte une valeur (timeline.X.value non nul) ;
  - STRICTE : l'etape porte une valeur ET une unite de temps REELLE, c'est-a-dire
              Seconds a Years, et non Unknown, NA ou Never.
Quatre incidents portent une valeur sans unite utilisable, deux en compromission et deux
en exfiltration. La definition lache donne donc 127 et 11 la ou la stricte donne 126 et
10. LE MEMOIRE PUBLIE LA STRICTE, et c'est la bonne : une duree sans unite ne date rien,
donc elle ne porte aucune anteriorite, qui est precisement ce que W exigerait.

ARRET DUR, repris du patron des scripts 95, 106 et 107 : le script refuse de tourner si
la base a derive, c'est-a-dire si elle ne porte pas exactement les 10 591 incidents et
les 980 du secteur financier que le script 28 publie deja. Sans ce controle, une mise a
jour de la VCDB deplacerait les taux publies en silence.

Donnee : data/raw/vcdb.csv (non versionne, sous licence ouverte ; source vz-risk/VCDB).
Aucune figure. Ne touche ni src/ ni memoire/.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "data", "raw"))
CSV = os.path.join(RAW, "vcdb.csv")
if not os.path.exists(CSV):
    sys.exit(f"donnee absente : {CSV}\n(VCDB non versionne ; telecharger vz-risk/VCDB)")

W = 74
STAGES = ["compromise", "exfiltration", "discovery", "containment"]
STAGES_FR = {"compromise": "compromission", "exfiltration": "exfiltration",
             "discovery": "decouverte", "containment": "confinement"}
UNITES_REELLES = ["Seconds", "Minutes", "Hours", "Days", "Weeks", "Months", "Years"]
UNITES_FLOUES = ["Unknown", "NA", "Never"]
FIN = "victim.industry2.52"          # NAICS 52 = finance

N_ATTENDU = 10591                     # publie par le script 28
N_FIN_ATTENDU = 980                   # publie par le script 28


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# ============================================================ chargement
cols = pd.read_csv(CSV, nrows=0).columns.tolist()
voulues = ["control_failure", FIN] + [f"timeline.{s}.value" for s in STAGES]
for s in STAGES:
    for u in UNITES_REELLES + UNITES_FLOUES:
        voulues.append(f"timeline.{s}.unit.{u}")
use = [c for c in voulues if c in cols]
df = pd.read_csv(CSV, usecols=use, low_memory=False)

n = len(df)


def bo(c):
    return df[c].fillna(0).astype(bool) if c in df else pd.Series(False, index=df.index)


fin = bo(FIN)
n_fin = int(fin.sum())

# ============================================================ arret dur
titre("0. Controle d'identite : la base est-elle celle du memoire ?")
print(f"  incidents               : {n} (attendu {N_ATTENDU})")
print(f"  dont secteur financier  : {n_fin} (attendu {N_FIN_ATTENDU})")
if n != N_ATTENDU or n_fin != N_FIN_ATTENDU:
    sys.exit("\nARRET : la base a derive. Les taux publies ne valent plus, et les\n"
             "republier depuis une autre extraction serait une recalibration silencieuse.")
print("  => base conforme, les taux publies restent valides.")

# ============================================================ 1. le champ de domaine
titre("1. control_failure : le champ qui nomme le domaine de controle defaillant")
cf = df["control_failure"].notna() & (df["control_failure"].astype(str).str.strip() != "")
n_cf, n_cf_fin = int(cf.sum()), int((cf & fin).sum())
print(f"  renseigne dans          : {n_cf} incidents sur {n} ({100 * n_cf / n:.2f} %)")
print(f"  dont secteur financier  : {n_cf_fin}")
print("  Ce champ est la taxonomie PAR DOMAINE DE CONTROLE que W exige, et le schema")
print("  la prevoit. Elle n'est pas remplie : le defaut porte sur le REMPLISSAGE, jamais")
print("  sur la conception.")

# ============================================================ 2. la chronologie
titre("2. La chronologie en quatre etapes : ce qui porte l'anteriorite")
lache, stricte = {}, {}
print(f"  {'etape':<16}{'valeur':>8}{'+ unite reelle':>16}{'valeur sans unite':>20}")
for s in STAGES:
    val = df[f"timeline.{s}.value"].notna()
    unite = pd.Series(False, index=df.index)
    for u in UNITES_REELLES:
        unite = unite | bo(f"timeline.{s}.unit.{u}")
    lache[s], stricte[s] = val, val & unite
    print(f"  {STAGES_FR[s]:<16}{int(val.sum()):>8}{int((val & unite).sum()):>16}"
          f"{int((val & ~unite).sum()):>20}")

n_lache = pd.DataFrame(lache).sum(axis=1)
n_stricte = pd.DataFrame(stricte).sum(axis=1)

print(f"\n  {'etapes datees':<22}{'definition lache':>18}{'definition STRICTE':>20}")
for k in (1, 2, 3, 4):
    print(f"  au moins {k}{'':<13}{int((n_lache >= k).sum()):>18}"
          f"{int((n_stricte >= k).sum()):>20}")
print("\n  LE MEMOIRE PUBLIE LA COLONNE STRICTE. Une duree sans unite ne date rien,")
print("  donc elle ne porte aucune anteriorite : c'est la seule lecture defendable.")

n_2 = int((n_stricte >= 2).sum())
n_3 = int((n_stricte >= 3).sum())
print(f"\n  datee sur au moins deux etapes : {n_2} ({100 * n_2 / n:.2f} %)")
print(f"  datee sur au moins trois etapes : {n_3}")

# ============================================================ 3. l'intersection
titre("3. L'intersection, qui est ce qu'identifier W exigerait SIMULTANEMENT")
inter = cf & (n_stricte >= 2)
n_i, n_i_fin = int(inter.sum()), int((inter & fin).sum())
print(f"  control_failure renseigne ET chronologie datee sur >= 2 etapes : {n_i}")
print(f"  dont secteur financier                                        : {n_i_fin}")
print("\n  C'est le resultat qui ferme la troisieme echappatoire. Identifier W demande")
print("  la taxonomie ET l'anteriorite sur le MEME incident. Toute la base publique en")
print("  porte un seul exemplaire, et aucun dans le secteur qui nous interesse.")

# ============================================================ verdict
titre("VERDICT")
print("  La specification existe, elle est publiee, elle est adoptee par la profession,")
print("  et elle n'est pas remplie. Le probleme n'est donc pas de savoir QUOI collecter,")
print("  question a laquelle VERIS repond deja, mais d'en rendre le remplissage")
print("  obligatoire. C'est ce qu'une notification d'incident majeur sous DORA peut")
print("  imposer, et qu'une base declarative volontaire ne peut pas.")

# =====================================================================================
# GRANDEURS CITEES, sans separateur de milliers ni signe.
# Le harnais coupe un nombre sur l'espace de milliers : << 10 591 >> y devient 10 et 591.
# Ce bloc n'ajoute aucun calcul, il reexpose les memes valeurs sous une forme lisible
# par le controle. Meme convention que les scripts 90 et 91.
# =====================================================================================
titre("Grandeurs citees")
print(f"  incidents de la base                      {n}")
print(f"  incidents en secteur financier            {n_fin}")
print(f"  control_failure renseigne                 {n_cf}")
print(f"  part de control_failure en pourcent       {100 * n_cf / n:.2f}")
print(f"  chronologie datee sur >= 2 etapes         {n_2}")
print(f"  part de la chronologie en pourcent        {100 * n_2 / n:.2f}")
print(f"  chronologie datee sur >= 3 etapes         {n_3}")
print(f"  intersection des deux conditions          {n_i}")
print(f"  intersection en secteur financier         {n_i_fin}")
