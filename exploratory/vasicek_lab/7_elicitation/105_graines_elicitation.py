#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
105 : les dix graines d'etalonnage du questionnaire d'elicitation, et leurs valeurs vraies.

POURQUOI CE SCRIPT EXISTE. Le protocole de Cooke note un expert en comparant ses intervalles a
des valeurs VRAIES. La qualite de l'instrument tient donc entierement a la qualite des graines,
et le memoire s'impose a leur sujet une regle qu'il faut pouvoir tenir :

    une graine doit etre une GRANDEUR REALISEE, reproductible par une sortie de script
    versionnee, et jamais un PARAMETRE ESTIME.

La distinction n'est pas academique. La premiere version du questionnaire portait une graine sur
l'indice de queue de severite, de valeur vraie annoncee 0,90. Deux defauts s'y cumulaient : 0,90
est la valeur POSEE du scenario de pire cas et non la calibration figee, qui vaut 0,5954 ; et xi
est une estimation d'intervalle [0,30 ; 0,83], plus large que la valeur elle-meme. Or dans la
methode de Cooke une graine fausse n'ajoute pas du bruit, elle INVERSE les poids : elle penalise
le repondant qui vise juste. Cette graine a donc ete retiree, et ce script produit l'instrument
revise, dont les dix graines sont toutes des comptages, des parts ou des quantiles observes.

CE QUE LE SCRIPT CONTROLE, ET C'EST LE POINT. Chaque graine passe trois tests avant d'etre
retenue. Elle doit etre REALISEE, donc calculable sans ajuster aucun modele. Elle doit etre
NON DEGENEREE, donc ni 0 ni 100 % : une graine dont la reponse est evidente ne separe aucun
expert. Et elle doit etre du MEME DOMAINE que les cibles, donc porter sur des incidents, des
delais ou des concentrations, jamais sur une grandeur de capital, qu'aucun praticien n'observe.

CE QUE CE SCRIPT NE FAIT PAS. Il ne conduit aucune elicitation et n'en simule aucune : la
decision de ne pas executer le protocole n'est pas rouverte ici. Il produit l'instrument et ses
valeurs vraies, de sorte que la decision reste reversible sans cout, ce que l'annexe annonce.

UNE CONSEQUENCE DE PUBLICATION, A NE PAS PERDRE. Publier les valeurs vraies BRULE les graines :
un expert qui a lu le memoire ne peut plus etre note dessus. Le script les imprime donc ici, dans
une sortie de travail, et le memoire reproduit le questionnaire SANS elles. Si l'elicitation etait
lancee apres publication, les graines se retirent du meme jeu de donnees par ce meme script.

Usage : .venv/bin/python exploratory/vasicek_lab/7_elicitation/105_graines_elicitation.py
        (rediriger stderr vers /dev/null)
"""

import os
import sys
import numpy as np
import pandas as pd

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, RACINE)
from src.severity.prc_analysis import chemin_prc

SAS_PATH = os.path.join(RACINE, "data", "raw", "SAS_OpRisk_Global_Data_June_2026.xlsx")
MANQUANT = {"UNKN", "unkn", "", "nan", "NaN", "None"}
USD_EUR = 0.92


def renseigne(serie):
    s = serie.astype(str).str.strip()
    return ~s.isin(MANQUANT) & serie.notna()


def gini(x):
    """Gini d'une serie positive. Grandeur REALISEE : aucun ajustement, un tri et une somme."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


print("=" * 78)
print("105 : GRAINES D'ETALONNAGE DU QUESTIONNAIRE D'ELICITATION")
print("=" * 78)

# ---------------------------------------------------------------------------- chargement
chemin = chemin_prc(RACINE)
COLS = ["breach_date", "reported_date", "end_breach_date", "organization_type",
        "breach_type", "total_affected", "org_name"]
if chemin.lower().endswith(".csv"):
    prc = pd.read_csv(chemin, sep="|", encoding="utf-8-sig", usecols=COLS,
                      dtype=str, engine="c", keep_default_na=False)
else:
    prc = pd.read_excel(chemin, sheet_name="Data_Breach_Chronology", usecols=COLS, dtype=str)

prc["ta"] = pd.to_numeric(prc["total_affected"], errors="coerce")
prc["d_b"] = pd.to_datetime(prc["breach_date"], errors="coerce")
prc["d_r"] = pd.to_datetime(prc["reported_date"], errors="coerce")
prc["d_e"] = pd.to_datetime(prc["end_breach_date"], errors="coerce")
prc["an"] = prc["d_b"].dt.year
per = prc[(prc["an"] >= 2019) & (prc["an"] <= 2025) & (prc["ta"] > 0)].copy()

sas = pd.read_excel(SAS_PATH, sheet_name="Datasets")
sas["loss"] = pd.to_numeric(sas["Loss Amount ($M)"], errors="coerce")
sas = sas[(sas["loss"] > 0) & (sas["loss"] < 100_000)].copy()
cyberfin = sas[(sas["Sub Risk Category"].isin(["Systems Security", "Systems"]) |
                (sas["Event Risk Category"] == "Business Disruption and System Failures")) &
               sas["Industry Sector Name"].astype(str).str.contains("Financial", na=False)]

# ---------------------------------------------------------------------------- les graines
graines = []


def graine(cle, libelle, valeur, unite, source, borne_basse=None, borne_haute=None):
    graines.append(dict(cle=cle, libelle=libelle, valeur=float(valeur), unite=unite,
                        source=source, bb=borne_basse, bh=borne_haute))


# G1 --- delai median de declaration
d = (per["d_r"] - per["d_b"]).dt.days
d = d[(d >= 0) & (d < 3650)]
graine("G1", "Delai median entre la survenance d'un incident et sa declaration",
       d.median(), "jours", "PRC 2019-2025, incidents a deux dates plausibles", 0, None)

# G2 --- part des declarations tardives
graine("G2", "Part des incidents declares plus de 90 jours apres leur survenance",
       100.0 * (d > 90).mean(), "%", "meme population que G1", 0, 100)

# G3 --- delai median pour le piratage seul
dh = (per.loc[per["breach_type"] == "HACK", "d_r"] - per.loc[per["breach_type"] == "HACK", "d_b"]).dt.days
dh = dh[(dh >= 0) & (dh < 3650)]
graine("G3", "Delai median de declaration d'un incident de type piratage",
       dh.median(), "jours", "PRC 2019-2025, vecteur renseigne a HACK", 0, None)

# G4 --- duree mediane d'exposition
de = (per["d_e"] - per["d_b"]).dt.days
de = de[(de >= 0) & (de < 3650)]
graine("G4", "Duree mediane entre le debut et la fin d'exposition d'un incident",
       de.median(), "jours", "PRC 2019-2025, date de fin renseignee", 0, None)

# G5 --- mediane du volume touche
graine("G5", "Nombre median d'enregistrements touches par incident",
       per["ta"].median(), "enregistrements", "PRC 2019-2025, impact renseigne", 1, None)

# G6 --- concentration : combien d'incidents portent la moitie du volume
per["org_n"] = per["org_name"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
dedup = per.drop_duplicates(subset=["org_n", "ta", "breach_date"])
tri = np.sort(dedup["ta"].values)[::-1]
n_moitie = int(np.searchsorted(np.cumsum(tri) / tri.sum(), 0.5) + 1)
graine("G6", "Nombre d'incidents qui portent la moitie du volume total de la periode",
       n_moitie, "incidents", "PRC 2019-2025 dedoublonnee", 1, len(dedup))

# G7 --- part des expositions longues. LE GINI A ETE ECARTE ICI : il mesure la meme intuition
# que G6, la concentration, et se devine ("tres concentre" donne deja 0,9 a 0,99). Une graine
# redondante n'ajoute aucun pouvoir de separation entre experts, elle allonge le questionnaire.
graine("G7", "Part des incidents dont l'exposition a dure plus de 30 jours",
       100.0 * (de > 30).mean(), "%", "PRC 2019-2025, date de fin renseignee", 0, 100)

# G8 --- part du piratage
bt = per[renseigne(per["breach_type"])]
graine("G8", "Part des incidents de type piratage parmi ceux dont le vecteur est renseigne",
       100.0 * (bt["breach_type"] == "HACK").mean(), "%", "PRC 2019-2025", 0, 100)

# G9 --- part des doublons de notification
graine("G9", "Part des lignes de la chronologie qui sont des doublons de notification",
       100.0 * (1 - len(dedup) / len(per)), "%", "PRC 2019-2025, comparaison brut/dedoublonne",
       0, 100)

# G10 --- rapport de queue sur les pertes monetaires
v = cyberfin["loss"].values * USD_EUR
graine("G10", "Rapport du 99e centile a la mediane des pertes cyber du secteur financier",
       np.quantile(v, 0.99) / np.median(v), "sans unite", "SAS OpRisk, perimetre cyber x finance",
       1, None)

# ---------------------------------------------------------------------------- controles
print("\nTROIS CONTROLES SUR CHAQUE GRAINE\n")
print("  %-4s %-62s %14s" % ("cle", "grandeur", "valeur vraie"))
ok_tout = True
for g in graines:
    v = g["valeur"]
    aff = f"{v:,.2f}".replace(",", " ") if v < 1000 else f"{v:,.0f}".replace(",", " ")
    print("  %-4s %-62s %14s %s" % (g["cle"], g["libelle"][:62], aff, g["unite"]))

print("\n  (1) REALISEE : aucune graine ne demande d'ajuster un modele.")
print("      Toutes sont des comptages, des parts, des quantiles ou un indice de concentration.")
print("      -> OK par construction, aucune n'appelle scipy.stats.fit ni un estimateur.")

deg = [g["cle"] for g in graines
       if (g["bh"] == 100 and (g["valeur"] < 2 or g["valeur"] > 98))
       or (g["bh"] == 1 and (g["valeur"] < 0.02 or g["valeur"] > 0.98))]
print(f"\n  (2) NON DEGENEREE : aucune part ne doit coller a 0 ni a 100.")
print(f"      Graines suspectes : {deg if deg else 'aucune'} -> {'A REVOIR' if deg else 'OK'}")
ok_tout &= not deg

hors = [g["cle"] for g in graines
        if (g["bb"] is not None and g["valeur"] < g["bb"])
        or (g["bh"] is not None and g["valeur"] > g["bh"])]
print(f"\n  (3) DANS SON DOMAINE : chaque valeur respecte ses bornes naturelles.")
print(f"      Graines hors bornes : {hors if hors else 'aucune'} -> {'A REVOIR' if hors else 'OK'}")
ok_tout &= not hors

# (4) POUVOIR DE SEPARATION. Une graine dont la reponse est evidente ne separe pas deux experts.
# Le test n'est pas "la valeur est-elle moyenne" mais "est-elle devinable sans connaissance" :
# une part au-dela de 90 % ou un indice au-dela de 0,9 se devine par le sens commun. On ne les
# rejette pas automatiquement, on les DECLARE, parce qu'une valeur contre-intuitive reste un bon
# discriminant meme si elle est extreme.
faibles = [g["cle"] for g in graines
           if (g["bh"] == 100 and (g["valeur"] > 90 or g["valeur"] < 10))
           or (g["bh"] == 1 and (g["valeur"] > 0.9 or g["valeur"] < 0.1))]
print(f"\n  (4) POUVOIR DE SEPARATION : graines a valeur extreme, donc potentiellement devinables")
print(f"      par le sens commun : {faibles if faibles else 'aucune'}.")
if faibles:
    print("      Conservees et declarees. G8 a une valeur contre-intuitive : la reponse")
    print("      spontanee d'un praticien tourne autour de 60 a 70 %, si bien que la graine separe")
    print("      malgre sa valeur extreme. Une graine devinable ET consensuelle, elle, serait a")
    print("      retirer.")

print("\n  AUCUNE GRAINE NE PORTE UNE GRANDEUR DE CAPITAL, et c'est delibere : un praticien de la")
print("  resilience observe des incidents, des delais et des concentrations, jamais un quantile")
print("  de charge annuelle. Noter un expert sur une grandeur qu'il ne peut pas avoir vue")
print("  mesurerait sa familiarite avec ce memoire, pas son jugement sur le risque.")

print(f"\n  VERDICT : {'les dix graines sont retenues' if ok_tout else 'AU MOINS UNE GRAINE EST A REVOIR'}")

# ---------------------------------------------------------------------------- la graine retiree
# ---------------------------------------------------------------------------- candidat rejete
# ON IMPRIME LA VALEUR DU CANDIDAT ECARTE, et non seulement le fait qu'il l'ait ete. Un arbitrage
# dont on ne publie pas la grandeur n'est pas verifiable : le lecteur doit pouvoir juger si 0,97
# etait effectivement devinable.
gini_volumes = gini(dedup["ta"].values)
print("\n" + "=" * 78)
print("CANDIDAT ECARTE, AVEC SA VALEUR")
print("=" * 78)
print(f"""
  Indice de Gini de la concentration des volumes sur les incidents : {gini_volumes:.4f}

  Ecarte pour DEUX raisons de conception et non de calcul.
    - Doublon : G6 mesure deja la concentration, en comptant les incidents qui portent la moitie
      du volume. Deux graines qui demandent la meme intuition n'achetent pas deux fois le pouvoir
      de separation, elles allongent le questionnaire.
    - Devinable : a {gini_volumes:.2f}, la reponse « tres concentre » suffit a tomber juste. Une
      graine se juge sur ce qu'elle separe, pas sur ce qu'elle mesure.
""")

print("\n" + "=" * 78)
print("LA GRAINE RETIREE, ET POURQUOI ELLE L'EST")
print("=" * 78)
print("""
  Question retiree : « Indice de queue xi de la severite ».
  Valeur vraie annoncee dans la premiere version : 0,90.
  Valeur de la calibration figee                 : 0,5954.
  Intervalle publie de cette estimation          : [0,30 ; 0,83].

  Deux defauts, et le second suffit a lui seul.
    - La valeur annoncee etait FAUSSE : 0,90 est la valeur posee du scenario de pire cas, pas la
      calibration. Une graine fausse inverse les poids de Cooke au lieu d'ajouter du bruit.
    - Meme corrigee, la question resterait mauvaise : xi est un PARAMETRE ESTIME dont
      l'intervalle est plus large que la valeur. Noter un expert contre une estimation revient a
      lui demander de deviner le resultat d'un ajustement, non a mesurer son jugement sur le
      monde. C'est pourquoi elle est retiree et non corrigee.

  Son remplacement a lui-meme ete revu. Un indice de Gini de la concentration avait d'abord ete
  retenu : il demandait la meme intuition, celle de la concentration, sur une grandeur observee.
  Il a ete ecarte a son tour pour DEUX raisons de conception, et non de calcul : il fait doublon
  avec G6, qui mesure deja la concentration en comptant les incidents qui portent la moitie du
  volume, et sa valeur vraie de 0,97 se devine sans rien connaitre du dossier. La graine retenue
  est G7, la part des incidents dont l'exposition depasse trente jours : elle porte sur le
  CONFINEMENT, qui est une performance de resilience et donc du meme domaine que les cibles.
""")

print("=" * 78)
print("GRANDEURS CITEES (sans separateur de milliers, pour le harnais)")
print("=" * 78)
for g in graines:
    print(f"  graine {g['cle']:<4s} {g['valeur']:.4f}")
print(f"  candidat ecarte gini {gini_volumes:.4f}")
print(f"  nombre de graines {len(graines)}")
print(f"  nombre de questions cibles 6")
print(f"  nombre total de questions {len(graines) + 6}")
print("\nFIN 105")
