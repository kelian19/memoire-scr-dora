#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
104 : profilage des variables, complétude et tests statistiques bivariés sur les deux bases.

POURQUOI CE SCRIPT EXISTE. Le mémoire décrivait ses deux sources par leur périmètre et leurs
biais, jamais par leurs VARIABLES : ni le type de chacune, ni son taux de manquant, ni le moindre
test d'association. Un jury d'actuaires ouvre un chapitre de données en cherchant exactement cela.
Ce script produit les trois objets qui manquaient : la table des variables avec sa complétude, les
tests d'association sur les qualitatives, et les tests de forme et d'hétérogénéité sur les
quantitatives. Tout ce que la sous-section publie sort d'ici.

LE PIEGE DE COMPLETUDE DE LA PRC, ET IL EST GROSSIER. Dans l'export PRC, un champ non renseigné
ne vaut pas vide : il vaut la chaîne « UNKN ». Un `notna()` naïf rend donc 100 % de présence sur
les trente-sept colonnes, y compris sur `total_affected` qui n'est en réalité renseignée que pour
deux incidents sur cinq. Le taux publié ici compte donc « UNKN » comme un manquant, ce qui est le
seul comptage honnête, et le script imprime les deux nombres côte à côte pour que l'écart soit
visible plutôt que supposé.

CE QUE LES TESTS FONT, ET SURTOUT CE QU'ILS NE FONT PAS.
  - Le khi-deux et le V de Cramér mesurent une ASSOCIATION dans la base observée, jamais un lien
    causal, et jamais une propriété du risque : les deux bases sont des collectes, avec leurs
    seuils et leurs obligations de notification. Une association forte entre le type de brèche et
    le dépassement du seuil peut n'être qu'une différence de régime déclaratif.
  - Sur des effectifs de cet ordre, le khi-deux rejette presque toujours. La p-valeur n'est donc
    PAS le résultat : le résultat est le V de Cramér, qui est une taille d'effet et ne croît pas
    avec n. Publier la p-valeur seule serait un contresens, et c'est pour cela que les deux sont
    imprimées ensemble.
  - Shapiro-Wilk sur des sévérités cyber ne teste rien qu'on ignore : il est là pour CHIFFRER le
    rejet de la normalité, c'est-à-dire pour donner une mesure à l'affirmation de queue lourde que
    le mémoire fait partout. Son échantillon est plafonné à 5 000 tirages, graine fixée, parce que
    la statistique n'est pas fiable au-delà ; le script le déclare au lieu de le taire.
  - Kruskal-Wallis teste l'égalité des distributions de sévérité entre classes d'organisation.
    C'est un test non paramétrique, donc il ne suppose pas la normalité que le précédent rejette.

CE SCRIPT NE RECALIBRE RIEN. Il ne touche ni config.py, ni un paramètre publié, ni une sortie
existante. C'est un diagnostic, compatible avec le gel de la calibration du 7 août 2026, au même
titre que les scripts 89 à 94.

TROIS FIGURES, toutes à la charte via style_nexialog.
  D1 : matrice de présence croisée de la PRC (taux, %).
  D2 : sources de notification, double axe (barres = volume affecté, ligne = nombre d'incidents).
  D3 : types de brèche en bulles (taille = nombre d'incidents, couleur = volume compromis).

Usage : .venv/bin/python exploratory/vasicek_lab/2_donnees/104_profilage_variables.py
        (rediriger stderr vers /dev/null, jamais vers le fichier : matplotlib y écrit des
         findfont qui corrompraient la sortie versionnée)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "exploratory", "vasicek_lab"))

import style_nexialog as sn
from src.severity.prc_analysis import chemin_prc

FIGDIR = os.path.join(RACINE, "exploratory", "vasicek_lab", "figures")
SAS_PATH = os.path.join(RACINE, "data", "raw", "SAS_OpRisk_Global_Data_June_2026.xlsx")

MANQUANT_PRC = {"UNKN", "unkn", "", "nan", "NaN", "None"}
GRAINE = 20260919
N_SHAPIRO = 5000

sn.appliquer(10)
plt.rcParams["font.family"] = ["DejaVu Sans", "Segoe UI", "sans-serif"]


def titre(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def renseigne(serie):
    """Masque des valeurs REELLEMENT renseignees : 'UNKN' est un manquant, pas une modalite."""
    s = serie.astype(str).str.strip()
    return ~s.isin(MANQUANT_PRC) & serie.notna()


def cramer_v(table):
    """V de Cramer avec correction de Bergsma, qui retire le biais d'effectif.

    Sans correction, le V monte mecaniquement avec le nombre de modalites : une table
    8 x 2 sur 15 000 lignes rend un V non nul sur des donnees independantes.
    """
    chi2, p, ddl, att = stats.chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    v = np.sqrt(phi2c / denom) if denom > 0 else np.nan
    return chi2, p, ddl, v, n


def ligne_test(nom, chi2, p, ddl, v, n):
    ptxt = "< 1e-300" if p < 1e-300 else f"{p:.3e}"
    print(f"  {nom:<44s} khi2 = {chi2:10.1f}  ddl = {ddl:3d}  p {ptxt:>10s}  V = {v:.4f}  n = {n}")


# =====================================================================================
titre("0. PERIMETRES ET STATUT")
# =====================================================================================
print("""
  Deux bases, deux roles, et elles ne se substituent pas :
    PRC  (Privacy Rights Clearinghouse / Data Breach Chronology) : incidents de violation de
         donnees notifies aux autorites americaines. Elle porte le NOMBRE d'enregistrements
         touches, jamais un montant. Elle sert la structure et la frequence, jamais le niveau
         de perte en euros.
    SAS  (OpRisk Global Data) : pertes operationnelles AVEREES, en dollars, toutes categories
         balaises. Elle porte le montant, donc la severite du memoire en sort.
  Le seuil de severite significative employe dans les tests ci-dessous est le 75e percentile
  de la base concernee. Il n'est PAS le seuil de calibration publie (u = 20,03 M EUR) : celui-ci
  vit dans config.py et ne bouge pas. Le q75 sert uniquement a binariser une variable continue
  pour les tests d'association, et il est declare comme tel.
""")

# =====================================================================================
titre("1. VOLUMETRIE ET COMPLETUDE")
# =====================================================================================

chemin = chemin_prc(RACINE)
if not chemin:
    print("  PRC absente de ce poste : sections PRC sautees.")
    sys.exit(1)

COLS_PRC = ["breach_date", "reported_date", "end_breach_date", "incident_details",
            "information_affected", "organization_type", "breach_type",
            "total_affected", "residents_affected", "source", "org_name"]

if chemin.lower().endswith(".csv"):
    prc = pd.read_csv(chemin, sep="|", encoding="utf-8-sig", usecols=COLS_PRC,
                      dtype=str, engine="c", keep_default_na=False)
else:
    prc = pd.read_excel(chemin, sheet_name="Data_Breach_Chronology", usecols=COLS_PRC,
                        dtype=str)

n_prc = len(prc)
print(f"\n  PRC : {n_prc} incidents dans l'export brut.")
print("\n  Taux de presence, naif contre honnete (%) :")
print("    %-24s %8s %8s" % ("variable", "notna", "hors UNKN"))
presence = {}
for c in COLS_PRC:
    naif = 100.0 * prc[c].notna().mean()
    vrai = 100.0 * renseigne(prc[c]).mean()
    presence[c] = vrai
    print("    %-24s %8.2f %8.2f" % (c, naif, vrai))

print(f"""
  LECTURE. La colonne « notna » vaut 100,00 partout : c'est l'artefact annonce en tete de
  script. La colonne de droite est celle qui compte. total_affected, seule variable d'impact
  de cette base, n'est renseignee que dans {presence["total_affected"]:.2f} % des cas : c'est la
  lacune structurante du chapitre, et c'est elle qui interdit d'utiliser la PRC pour un niveau
  de perte. Le champ de DATE de survenance manque de son cote dans un cas sur deux
  ({presence["breach_date"]:.2f} % de presence), ce qui borne toute lecture temporelle fine.
""")

prc["ta"] = pd.to_numeric(prc["total_affected"], errors="coerce")
prc["ra"] = pd.to_numeric(prc["residents_affected"], errors="coerce")
prc["d_breach"] = pd.to_datetime(prc["breach_date"], errors="coerce")
prc["d_report"] = pd.to_datetime(prc["reported_date"], errors="coerce")
prc["annee"] = prc["d_breach"].dt.year

n_ta = int((prc["ta"] > 0).sum())
print(f"  total_affected strictement positif : {n_ta} incidents "
      f"({100.0 * n_ta / n_prc:.2f} % de l'export).")

per = prc[(prc["annee"] >= 2019) & (prc["annee"] <= 2025) & (prc["ta"] > 0)].copy()
print(f"  Perimetre retenu par le memoire (2019-2025, impact renseigne) : {len(per)} incidents.")

# CONTROLE D'IDENTITE, et il n'est pas decoratif. Ce script recharge la PRC par un chemin
# independant de celui des scripts 21, 22 et 62, avec son propre traitement du manquant. S'il
# ne retombait pas sur le meme effectif, l'un des deux lirait autre chose que ce qu'il annonce.
CIBLE_PERIMETRE = 15053
if len(per) == CIBLE_PERIMETRE:
    print(f"  CONTROLE : effectif identique aux sorties versionnees 21, 22 et 62 "
          f"({CIBLE_PERIMETRE}) -> OK")
else:
    print(f"  CONTROLE : ECART avec les sorties versionnees 21, 22 et 62 "
          f"({len(per)} contre {CIBLE_PERIMETRE}) -> A INSTRUIRE AVANT TOUTE PUBLICATION")

sas = pd.read_excel(SAS_PATH, sheet_name="Datasets")
sas["loss"] = pd.to_numeric(sas["Loss Amount ($M)"], errors="coerce")
n_sas_brut = len(sas)
sas = sas[(sas["loss"] > 0) & (sas["loss"] < 100_000)].copy()
n_sas = len(sas)
print(f"\n  SAS OpRisk : {n_sas_brut} lignes brutes, {n_sas} pertes retenues "
      f"(montant strictement positif, aberrations au-dela de 100 000 M$ retirees).")

COLS_SAS = ["Loss Amount ($M)", "Event Risk Category", "Sub Risk Category",
            "Basel Business Line - Level 1", "Industry Sector Name",
            "Country of Incident", "Revenue ($M)", "Assets ($M)", "# of Employees"]
print("\n  Taux de presence SAS (%) :")
for c in COLS_SAS:
    print("    %-32s %8.2f" % (c, 100.0 * sas[c].notna().mean()))

cyber = sas[sas["Sub Risk Category"].isin(["Systems Security", "Systems"]) |
            (sas["Event Risk Category"] == "Business Disruption and System Failures")]
fin = sas[sas["Industry Sector Name"].astype(str).str.contains("Financial", na=False)]
cyberfin = cyber[cyber["Industry Sector Name"].astype(str).str.contains("Financial", na=False)]
print(f"\n  Perimetre cyber/TIC : {len(cyber)} pertes ; secteur financier : {len(fin)} ; "
      f"intersection cyber x finance : {len(cyberfin)}.")

# =====================================================================================
titre("1bis. STATISTIQUES DESCRIPTIVES DES VARIABLES D'IMPACT")
# =====================================================================================
#
# POURQUOI CETTE TABLE EXISTE. Le memoire publie des quantiles de severite dans plusieurs
# chapitres, mais il ne les avait jamais RASSEMBLES : un lecteur qui veut juger la queue avant
# de lire l'ajustement devait les reconstituer de proche en proche. C'est la premiere table
# qu'un jury d'actuaires cherche dans un chapitre de donnees.
#
# QUATRE COLONNES, ET ELLES NE MESURENT PAS LA MEME CHOSE. Les deux premieres sont des
# ENREGISTREMENTS, les deux dernieres des EUROS : elles ne se comparent ni en niveau ni en
# quantile, et la table le dit dans son en-tete plutot que de laisser le lecteur le supposer.
# La quatrieme, cyber x finance, est celle qui porte la calibration publiee.

from src.severity.prc_analysis import jacobs_severity_eur_m

USD_EUR = 0.92
colonnes = {}
colonnes["PRC 2019-2025, enregistrements"] = per["ta"].values.astype(float)
colonnes["PRC, severite derivee Jacobs (M EUR)"] = jacobs_severity_eur_m(per["ta"].values)
colonnes["SAS toutes categories (M EUR)"] = sas["loss"].values * USD_EUR
colonnes["SAS cyber x finance (M EUR)"] = cyberfin["loss"].values * USD_EUR

lignes = [("n", lambda x: len(x)),
          ("moyenne", np.mean), ("ecart-type", lambda x: np.std(x, ddof=1)),
          ("minimum", np.min),
          ("q25", lambda x: np.quantile(x, .25)), ("mediane", np.median),
          ("q75", lambda x: np.quantile(x, .75)), ("q85", lambda x: np.quantile(x, .85)),
          ("q90", lambda x: np.quantile(x, .90)), ("q95", lambda x: np.quantile(x, .95)),
          ("q99", lambda x: np.quantile(x, .99)), ("maximum", np.max)]

print("\n  %-14s %22s %22s %22s %22s" % (("statistique",) + tuple(colonnes)))
desc = {}
for nom, f in lignes:
    vals = [f(v) for v in colonnes.values()]
    desc[nom] = vals
    fmt = lambda v: f"{v:22.0f}" if abs(v) >= 1000 or nom == "n" else f"{v:22.4f}"
    print("  %-14s %s" % (nom, "".join(fmt(v) for v in vals)))

# Deux rapports de forme, qui se lisent mieux que les niveaux eux-memes.
print("\n  Rapports de forme :")
for i, nom in enumerate(colonnes):
    med = desc["mediane"][i]; q99 = desc["q99"][i]; mx = desc["maximum"][i]
    moy = desc["moyenne"][i]; et = desc["ecart-type"][i]
    print(f"    {nom:<38s} q99/mediane = {q99/med:9.1f}   max/q99 = {mx/q99:8.1f}"
          f"   CV = {et/moy:7.2f}")

print("""
  LECTURE. La colonne qui compte est la derniere : c'est elle qui porte la calibration publiee.
  Trois traits s'y lisent d'un coup, et ils commandent tout le reste du memoire.
    - La MOYENNE depasse le q75 sur les quatre colonnes, et de loin. Une distribution dont la
      moyenne tombe au-dessus du troisieme quartile n'est pas decrite par sa moyenne : le
      niveau est porte par une minorite d'observations.
    - Le rapport du q99 a la MEDIANE vaut pres de 270 sur les trois colonnes monetaires et
      depasse 1 500 sur les enregistrements. Ce n'est pas une queue qui se prolonge, c'est une
      queue qui change d'echelle, et c'est ce qui rend un ajustement global inadapte.
    - Le coefficient de variation vaut 4,0 sur le perimetre calibre et jusqu'a 12,0 sur les
      enregistrements. Une loi exponentielle le vaut exactement un : la dispersion est donc
      bien au-dela du regime que les lois usuelles de duree savent porter.
  ET UNE ASYMETRIE ENTRE LES DEUX DERNIERES COLONNES, a ne pas lire de travers. Le rapport du
  maximum au q99 vaut 3,7 sur le perimetre cyber x finance contre 52,7 sur la base entiere : le
  perimetre retenu est le MOINS extreme des deux en ce sens, non le plus. Le sinistre le plus
  lourd de la base ne releve pas du cyber, et la queue calibree n'est donc pas celle de
  l'evenement le plus spectaculaire du fichier.
""")

# LE CONTROLE QUI COMPTE, ET IL N'EST PAS CELUI QU'ON ECRIRAIT SPONTANEMENT. On serait tente de
# verifier que le q85 de cet echantillon redonne le seuil publie u = 20,03 M EUR. Ce serait FAUX,
# et le projet le documente : le seuil publie est GELE au percentile 84,4 de l'echantillon de
# calibration, quand le q85 des donnees courantes vaut 22,03. C'est exactement l'ecart qui avait
# fait diverger les scripts 46, 47 et 51 tant qu'ils rederivaient leur propre seuil au lieu de le
# lire dans config.py. Le bon controle porte donc sur le NOMBRE D'EXCES au-dessus du seuil publie,
# qui doit redonner le n_excess de config.py.
q85_cf = float(np.quantile(colonnes["SAS cyber x finance (M EUR)"], .85))
U_PUBLIE = 20.03
N_EXCESS_PUBLIE = 91
n_exc = int((colonnes["SAS cyber x finance (M EUR)"] > U_PUBLIE).sum())
print(f"  Reperes : q85 des donnees courantes = {q85_cf:.2f} M EUR ; seuil publie et gele "
      f"u = {U_PUBLIE} M EUR (percentile 84,4). L'ecart est attendu, pas anormal.")
print(f"  CONTROLE : exces au-dessus du seuil publie = {n_exc} contre n_excess = "
      f"{N_EXCESS_PUBLIE} dans config.py -> {'OK' if n_exc == N_EXCESS_PUBLIE else 'ECART A INSTRUIRE'}")
print(f"  RAPPEL : p_u = 0,1509 de config.py correspondrait a {0.1509*len(colonnes['SAS cyber x finance (M EUR)']):.1f} "
      f"exces et non a {N_EXCESS_PUBLIE}. Cette incoherence est CONNUE, chiffree et publiee comme "
      f"limite au chapitre de robustesse ; elle n'est pas corrigee ici, le gel de la calibration "
      f"l'interdisant.")

# =====================================================================================
titre("1ter. LES PLUS GROS INCIDENTS, NOMMES")
# =====================================================================================
#
# POURQUOI NOMMER. Un quantile a 99,5 % reste une abstraction tant qu'on ne voit pas les
# evenements qui le portent. Cette table les montre, et elle repond a la premiere objection
# qu'un jury formule devant une queue lourde : « d'ou vient ce niveau ? ».
#
# POURQUOI LA PRC ET PAS SAS. Les incidents de la PRC sont des NOTIFICATIONS PUBLIQUES aux
# autorites americaines : les nommer ne fait que reprendre une information deja publique. La
# base SAS est une base commerciale sous licence, et le depot ne la republie pas ; ses plus
# grosses pertes sont donc decrites par leur annee, leur categorie et leur montant, jamais par
# le nom de la firme. La regle d'anonymisation des quatre assureurs SFCR du memoire est encore
# une autre question, et elle ne change pas : elle porte sur des entites dont l'etat de
# conformite est SUPPOSE, ce qui n'est pas le cas ici.

# DEDOUBLONNAGE AVANT LA TABLE, ET IL EST INDISPENSABLE. La PRC n'agrege pas les notifications :
# un meme incident notifie a quatre autorites y figure quatre fois, avec le meme volume. La table
# brute des quinze plus gros affichait ainsi Texas Dow quatre fois et CafePress trois fois. On
# dedoublonne donc sur le triplet (organisation normalisee, volume, date de survenance), ce qui
# est la signature d'un incident unique, et l'on chiffre a part ce que le dedoublonnage retire.
per["org_norm"] = per["org_name"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "",
                                                                     regex=True)
cle = ["org_norm", "ta", "breach_date"]
dedup = per.drop_duplicates(subset=cle)
n_retire = len(per) - len(dedup)
print(f"\n  Dedoublonnage sur (organisation, volume, date de survenance) : {len(per)} lignes "
      f"-> {len(dedup)} incidents distincts, soit {n_retire} doublons de notification "
      f"({100.0*n_retire/len(per):.1f} % des lignes).")

top = dedup.nlargest(15, "ta")[["org_name", "organization_type", "breach_type",
                                "d_breach", "d_report", "ta"]]
print("\n  Quinze plus gros incidents distincts de la PRC sur 2019-2025 :")
print("    %-42s %5s %5s %12s %12s %14s" %
      ("organisation", "type", "breche", "survenance", "notification", "enregistrements"))
for _, r in top.iterrows():
    nom = str(r["org_name"])[:42]
    ds = r["d_breach"].strftime("%Y-%m-%d") if pd.notna(r["d_breach"]) else "-"
    dr = r["d_report"].strftime("%Y-%m-%d") if pd.notna(r["d_report"]) else "-"
    print("    %-42s %5s %5s %12s %12s %14.0f" %
          (nom, str(r["organization_type"])[:5], str(r["breach_type"])[:5], ds, dr, r["ta"]))

part15 = 100.0 * top["ta"].sum() / dedup["ta"].sum()
part1 = 100.0 * top["ta"].iloc[0] / dedup["ta"].sum()
print(f"\n  Ces quinze incidents portent {part15:.1f} % du volume total de la periode, "
      f"et le premier a lui seul {part1:.1f} %.")

# CONCENTRATION, sur la base DEDOUBLONNEE : la calculer sur les doublons la surestimerait.
tri = np.sort(dedup["ta"].values)[::-1]
cum = np.cumsum(tri) / tri.sum()
conc = {}
for part in (0.5, 0.8, 0.9):
    n_needed = int(np.searchsorted(cum, part) + 1)
    conc[part] = (n_needed, 100.0 * n_needed / len(tri))
    print(f"  {100*part:.0f} % du volume tient dans {n_needed} incidents, soit "
          f"{conc[part][1]:.2f} % de la base dedoublonnee.")

print(f"""
  LECTURE, ET C'EST L'ARGUMENT DE QUEUE RENDU CONCRET. La moitie du volume de sept annees tient
  dans {conc[0.5][0]} incidents sur {len(dedup)}, soit {conc[0.5][1]:.2f} % de la base. Un modele qui
  decrirait cette distribution par son centre manquerait donc l'objet : le niveau n'est pas
  produit par le regime courant, il est produit par une poignee d'evenements. C'est exactement
  ce que l'ajustement par depassement de seuil formalise, et c'est aussi pourquoi le quantile a
  99,5 % du memoire est porte par un sinistre unique plutot que par une accumulation.

  LE DEDOUBLONNAGE EST CONSERVATEUR, ET LA TABLE LE MONTRE. Il exige une egalite EXACTE du
  triplet, si bien qu'il ne rapproche ni « CafePress, Inc. » de « CafePress Inc. », ni deux
  notifications du meme incident decalees d'un jour. Des doublons residuels subsistent donc, et
  ils sont visibles dans les quinze lignes ci-dessus. Les rapprocher demanderait un appariement
  approximatif sur les raisons sociales, dont le taux d'erreur serait lui-meme a estimer : le
  cout depasse le benefice pour un usage ou chaque ligne est une observation de severite et non
  un terme d'une somme. Le chiffre de doublons ci-dessus est donc une BORNE BASSE.

  ET UN DEFAUT DE BASE A DECLARER. Les {n_retire} lignes retirees ci-dessus ne sont pas des
  erreurs de saisie : ce sont des notifications multiples d'un meme incident a des autorites
  differentes, que la PRC ne deduplique pas. Sommer total_affected sans dedoublonner surestime
  donc le volume. Le memoire n'est PAS touche, parce qu'il ne somme jamais ces volumes pour
  produire un montant : il les utilise comme echantillon de severite, ou chaque ligne est une
  observation. La remarque vaut pour qui reprendrait la base a d'autres fins.
""")

# =====================================================================================
titre("2. VARIABLES QUALITATIVES : KHI-DEUX D'INDEPENDANCE ET V DE CRAMER")
# =====================================================================================

q75_prc = per["ta"].quantile(0.75)
per["severe"] = (per["ta"] >= q75_prc).map({True: "au-dessus q75", False: "sous q75"})
print(f"\n  PRC, seuil de binarisation q75 = {q75_prc:.0f} enregistrements touches.")
for var in ["breach_type", "organization_type"]:
    sub = per[renseigne(per[var])]
    tab = pd.crosstab(sub[var], sub["severe"])
    ligne_test(f"{var} x depassement du q75", *cramer_v(tab))

q75_sas = sas["loss"].quantile(0.75)
sas["severe"] = (sas["loss"] >= q75_sas).map({True: "au-dessus q75", False: "sous q75"})
print(f"\n  SAS, seuil de binarisation q75 = {q75_sas:.4f} M$.")
for var in ["Event Risk Category", "Basel Business Line - Level 1", "Industry Sector Name"]:
    sub = sas[sas[var].notna()]
    tab = pd.crosstab(sub[var], sub["severe"])
    ligne_test(f"{var} x depassement du q75", *cramer_v(tab))

print("""
  LECTURE. Toutes les p-valeurs s'effondrent, ce qui n'apprend rien : a quinze mille et
  quarante mille lignes, le khi-deux rejette l'independance pour un ecart sans portee. Le
  renseignement est dans le V, qui est une taille d'effet bornee par un. Les associations
  mesurees restent MODESTES : aucune n'atteint 0,3, seuil usuel d'une association moyenne.
  Le type de brèche et la ligne metier informent donc le depassement du seuil, mais loin de
  le determiner, ce qui est exactement l'argument que le memoire fait valoir contre une
  segmentation de la severite par categorie.
""")

# =====================================================================================
titre("3. VARIABLES QUANTITATIVES : NORMALITE, CORRELATION DE RANG, HETEROGENEITE")
# =====================================================================================

rng = np.random.default_rng(GRAINE)


def shapiro_plafonne(x, nom):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    n = len(x)
    ech = x if n <= N_SHAPIRO else rng.choice(x, N_SHAPIRO, replace=False)
    w_brut, p_brut = stats.shapiro(ech)
    w_log, p_log = stats.shapiro(np.log(ech))
    sk = stats.skew(x)
    ku = stats.kurtosis(x, fisher=False)
    print(f"\n  {nom} (n = {n}, echantillon de test = {len(ech)})")
    print(f"    Shapiro-Wilk, echelle brute      W = {w_brut:.4f}   p = {p_brut:.3e}")
    print(f"    Shapiro-Wilk, echelle log        W = {w_log:.4f}   p = {p_log:.3e}")
    print(f"    asymetrie = {sk:.2f}   aplatissement (non centre) = {ku:.2f}")
    return w_brut, p_brut, w_log, p_log, sk, ku


sw_prc = shapiro_plafonne(per["ta"], "PRC, enregistrements touches")
sw_sas = shapiro_plafonne(sas["loss"], "SAS, montant de perte (M$)")
sw_cyb = shapiro_plafonne(cyber["loss"], "SAS, perimetre cyber/TIC (M$)")

print("""
    LECTURE. La normalite est rejetee a l'echelle brute des deux cotes, et l'aplatissement
    se compte en milliers la ou une gaussienne vaut trois. Le passage au logarithme AMELIORE
    nettement la statistique sans la sauver : la queue reste plus lourde qu'une log-normale,
    ce qui est precisement ce qui justifie une loi de Pareto generalisee au-dela d'un seuil
    plutot qu'un ajustement global.
""")

print("\n  Correlations de Spearman (rang, donc insensibles a la queue) :")
sp = per[["ta", "ra"]].copy()
sp["delai_jours"] = (per["d_report"] - per["d_breach"]).dt.days
sp = sp.dropna()
sp = sp[(sp["delai_jours"] >= 0) & (sp["delai_jours"] < 3650)]
noms = {"ta": "total_affected", "ra": "residents_affected", "delai_jours": "delai de notification"}
print(f"    PRC (n = {len(sp)} incidents a triplet complet et delai plausible) :")
for i, a in enumerate(sp.columns):
    for b in list(sp.columns)[i + 1:]:
        r, p = stats.spearmanr(sp[a], sp[b])
        ptxt = "< 1e-300" if p < 1e-300 else f"{p:.3e}"
        print(f"      {noms[a]:<24s} x {noms[b]:<24s} rho = {r:+.4f}  p {ptxt}")

ss = sas[["loss", "Revenue ($M)", "Assets ($M)", "# of Employees"]].apply(
    pd.to_numeric, errors="coerce").dropna()
ss = ss[(ss > 0).all(axis=1)]
nsas_sp = len(ss)
print(f"\n    SAS (n = {nsas_sp} pertes a profil financier complet) :")
lib = {"loss": "perte", "Revenue ($M)": "chiffre d'affaires",
       "Assets ($M)": "actif total", "# of Employees": "effectif"}
rho_taille = {}
for b in ["Revenue ($M)", "Assets ($M)", "# of Employees"]:
    r, p = stats.spearmanr(ss["loss"], ss[b])
    rho_taille[b] = r
    ptxt = "< 1e-300" if p < 1e-300 else f"{p:.3e}"
    print(f"      {'perte':<24s} x {lib[b]:<24s} rho = {r:+.4f}  p {ptxt}")

print("""
    LECTURE, ET C'EST LE RESULTAT LE PLUS UTILE DE LA SECTION. La perte est correlee a la
    taille de l'entite, mais TRES FAIBLEMENT : les trois coefficients de rang tiennent entre
    0,07 et 0,16, quand une proportionnalite en donnerait un proche de un. C'est la meme
    conclusion que l'elasticite de 0,087 mesuree par ailleurs sur la severite, obtenue ici par
    un chemin non parametrique. Une entite dix fois plus petite ne subit pas une perte dix fois
    plus petite, et c'est ce qui fonde la borne inferieure de validite de la descente d'echelle.
    L'actif total est le PLUS FAIBLE des trois predicteurs de taille, sous le chiffre
    d'affaires et sous l'effectif : la mesure de bilan, qui est pourtant celle qui sert a
    l'echelle prudentielle, est la moins liee a la perte observee.
""")

print("\n  Kruskal-Wallis, egalite des distributions de severite entre classes :")


def kruskal_par(df, col_classe, col_val, nom, mini=30):
    g = [v[col_val].dropna().values for _, v in df.groupby(col_classe) if len(v) >= mini]
    k = len(g)
    if k < 2:
        print(f"    {nom} : moins de deux classes suffisamment peuplees.")
        return None
    h, p = stats.kruskal(*g)
    n = sum(len(x) for x in g)
    eps2 = (h - k + 1) / (n - k) if n > k else np.nan
    ptxt = "< 1e-300" if p < 1e-300 else f"{p:.3e}"
    print(f"    {nom:<40s} H = {h:9.1f}  k = {k:2d}  p {ptxt:>10s}  eps2 = {eps2:.4f}  n = {n}")
    return h, p, k, eps2, n


kw_prc = kruskal_par(per[renseigne(per["organization_type"])], "organization_type", "ta",
                     "PRC, impact par type d'organisation")
kw_prcb = kruskal_par(per[renseigne(per["breach_type"])], "breach_type", "ta",
                      "PRC, impact par type de breche")
kw_sas = kruskal_par(sas, "Event Risk Category", "loss",
                     "SAS, perte par categorie balaise")
kw_sasb = kruskal_par(sas, "Basel Business Line - Level 1", "loss",
                      "SAS, perte par ligne metier balaise")

print("""
    LECTURE. L'hypothese d'une severite homogene entre classes est rejetee partout, et la
    taille d'effet epsilon-carre dit de combien : elle reste petite, de l'ordre de quelques
    centiemes, donc la classe d'organisation explique une part FAIBLE de la dispersion des
    severites. Ce resultat va dans le meme sens que le V de Cramer de la section 2 et il
    porte la meme consequence de modelisation : segmenter la severite par categorie
    d'organisation couterait des degres de liberte sans acheter de pouvoir explicatif, et
    c'est pourquoi le memoire ajuste une queue commune puis declare ce choix comme une
    limite au lieu de le presenter comme une mesure.
""")

# =====================================================================================
titre("4. LES TROIS FIGURES")
# =====================================================================================

# --- D1 : matrice de presence croisee -------------------------------------------------
vars_mat = ["breach_date", "reported_date", "end_breach_date", "incident_details",
            "information_affected", "organization_type", "breach_type",
            "total_affected", "residents_affected"]
courts = {"breach_date": "breach\ndate", "reported_date": "reported\ndate",
          "end_breach_date": "end breach\ndate", "incident_details": "incident\ndetails",
          "information_affected": "information\naffected", "organization_type": "organization\ntype",
          "breach_type": "breach\ntype", "total_affected": "total\naffected",
          "residents_affected": "residents\naffected"}
masques = {v: renseigne(prc[v]).values for v in vars_mat}
k = len(vars_mat)
M = np.full((k, k), np.nan)
for i in range(k):
    for j in range(i + 1):
        M[i, j] = 100.0 * (masques[vars_mat[i]] & masques[vars_mat[j]]).mean()

print("\n  D1, taux de presence croisee (%) — triangle inferieur :")
for i in range(k):
    print("    %-22s %s" % (vars_mat[i],
          " ".join(f"{M[i, j]:6.1f}" if not np.isnan(M[i, j]) else "      " for j in range(k))))

# INTERSECTION TRIPLE. Les taux croises se lisent deux a deux, mais l'argument du chapitre porte
# sur trois variables a la fois : une analyse qui voudrait le volume, le vecteur ET la duree
# d'exposition ne travaille plus que sur ce qui reste. On l'imprime au lieu de le supposer.
triple = 100.0 * (masques["total_affected"] & masques["breach_type"] &
                  masques["end_breach_date"]).mean()
print(f"\n  Intersection des TROIS variables (volume, vecteur, fin d'exposition) : {triple:.1f} %")
print(f"  a comparer au plus faible des croisements deux a deux, "
      f"{100.0 * (masques['total_affected'] & masques['end_breach_date']).mean():.1f} %.")

from matplotlib.colors import LinearSegmentedColormap
cmap_seq = LinearSegmentedColormap.from_list("nexialog_seq", sn.SEQUENTIEL_6[::-1])

# DEUX PANNEAUX ET NON UN, et ce n'est pas un ornement. Un taux de presence croisee ne dit pas
# sur quel EFFECTIF il porte : 12,5 % de 74 783 lignes font encore neuf mille incidents, ce qui
# n'a pas la meme portee qu'un taux identique sur une base de quelques centaines. Les deux
# panneaux se lisent donc ensemble, taux a gauche, effectifs a droite.
N = np.full((k, k), np.nan)
for i in range(k):
    for j in range(i + 1):
        N[i, j] = (masques[vars_mat[i]] & masques[vars_mat[j]]).sum()

print("\n  D1b, effectifs croises :")
for i in range(k):
    print("    %-22s %s" % (vars_mat[i],
          " ".join(f"{N[i, j]:7.0f}" if not np.isnan(N[i, j]) else "       " for j in range(k))))


def matrice(mat, nom_fichier, titre_fig, etiquette_cb, fmt, vmax):
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(mat, cmap=cmap_seq, vmin=0, vmax=vmax)
    for i in range(k):
        for j in range(i + 1):
            v = mat[i, j]
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 * vmax else sn.ENCRE)
    ax.set_xticks(range(k))
    ax.set_xticklabels([courts[v] for v in vars_mat], fontsize=7.5, rotation=45, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels([courts[v] for v in vars_mat], fontsize=7.5)
    ax.set_title(titre_fig, fontsize=10, color=sn.ENCRE)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(etiquette_cb, fontsize=8)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, nom_fichier), dpi=150)
    plt.close(fig)


matrice(M, "D1a_presence_taux.png", "taux de présence croisée (%)",
        "taux de présence (%)", lambda v: f"{v:.1f}", 100.0)
matrice(N, "D1b_presence_effectifs.png", "effectifs croisés",
        "incidents renseignés", lambda v: f"{v/1000:.1f}k" if v >= 1000 else f"{v:.0f}",
        float(np.nanmax(N)))

# --- D2 : sources de notification, double axe ----------------------------------------
src = per[renseigne(per["source"])].groupby("source").agg(
    incidents=("ta", "size"), volume=("ta", "sum")).sort_values("volume", ascending=False).head(12)
print("\n  D2, sources de notification (12 premieres par volume) :")
print("    %-8s %10s %16s" % ("source", "incidents", "volume (M enr.)"))
for s, r in src.iterrows():
    print("    %-8s %10d %16.1f" % (s, r["incidents"], r["volume"] / 1e6))

fig, ax1 = plt.subplots(figsize=(7.6, 4.2))
x = np.arange(len(src))
ax1.bar(x, src["volume"].values / 1e6, color=sn.CATEGORIEL[2], width=0.68,
        label="volume d'enregistrements touchés")
ax1.set_ylabel("volume touché (millions d'enregistrements)", fontsize=9, color=sn.CATEGORIEL[2])
ax1.tick_params(axis="y", labelcolor=sn.CATEGORIEL[2])
ax1.set_xticks(x); ax1.set_xticklabels(src.index, fontsize=8)
ax1.set_xlabel("source de notification", fontsize=9)
ax2 = ax1.twinx()
ax2.plot(x, src["incidents"].values, color=sn.CATEGORIEL[0], marker="o", lw=1.8,
         label="nombre d'incidents")
ax2.set_ylabel("nombre d'incidents", fontsize=9, color=sn.CATEGORIEL[0])
ax2.tick_params(axis="y", labelcolor=sn.CATEGORIEL[0])
ax2.spines["right"].set_visible(True)
ax1.set_title("Sources de notification : volume touché et nombre d'incidents",
              fontsize=10, color=sn.ENCRE)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "D2_sources_notification.png"), dpi=150)
plt.close(fig)

# --- D3 : bulles par type de breche ---------------------------------------------------
bt = per[renseigne(per["breach_type"])].groupby("breach_type").agg(
    incidents=("ta", "size"), volume=("ta", "sum"), median=("ta", "median"))
bt = bt.sort_values("incidents", ascending=False)
print("\n  D3, types de breche :")
print("    %-8s %10s %16s %12s" % ("type", "incidents", "volume (M enr.)", "mediane"))
for s, r in bt.iterrows():
    print("    %-8s %10d %16.1f %12.0f" % (s, r["incidents"], r["volume"] / 1e6, r["median"]))

# Plan frequence x severite plutot qu'un cercle decoratif : l'axe des abscisses porte le
# NOMBRE d'incidents, celui des ordonnees la MEDIANE d'impact, la taille et la couleur portent
# le volume cumule. Un type de breche se lit donc par sa position autant que par sa taille, et
# les etiquettes tiennent a cote des bulles au lieu de deborder des plus petites.
fig, ax = plt.subplots(figsize=(7.4, 4.6))
xv = bt["incidents"].values.astype(float)
yv = bt["median"].values.astype(float)
vol = bt["volume"].values / 1e6
taille = 1500 * (vol / vol.max()) ** 0.5 + 120
sc = ax.scatter(xv, yv, s=taille, c=vol, cmap=cmap_seq, edgecolors="white",
                linewidths=1.3, zorder=3, norm=matplotlib.colors.LogNorm(
                    vmin=max(vol.min(), 1e-2), vmax=vol.max()))
for xi, yi, nom, v in zip(xv, yv, bt.index, vol):
    ax.annotate(f"{nom}", (xi, yi), textcoords="offset points", xytext=(0, -26),
                ha="center", fontsize=8.5, weight="bold", color=sn.ENCRE, zorder=4)
ax.set_xscale("log"); ax.set_yscale("log")
# Marges explicites : sans elles la plus grosse bulle est coupee par le cadre.
ax.set_xlim(xv.min() / 3.2, xv.max() * 3.2)
ax.set_ylim(yv.min() / 1.9, yv.max() * 1.9)
ax.set_xlabel("nombre d'incidents (échelle log)", fontsize=9)
ax.set_ylabel("impact médian, enregistrements (log)", fontsize=9)
ax.grid(True, which="major", alpha=0.5)
ax.set_title("Types de brèche : fréquence, impact médian et volume cumulé",
             fontsize=10, color=sn.ENCRE)
cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
cb.set_label("volume cumulé touché (millions, log)", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "D3_bulles_breche.png"), dpi=150)
plt.close(fig)

# --- D4 : composition temporelle, qui rend VISIBLE le biais de profondeur de collecte -------
#
# POURQUOI CETTE FIGURE. Le memoire DECLARE depuis longtemps que la profondeur de collecte
# borne la fenetre exploitable, et le backtest du script 89 choisit sa fenetre sur ce motif.
# L'affirmation n'etait jamais MONTREE. Ici elle se voit : la base est quasi vide avant 2015,
# monte jusqu'en 2023, et retombe en 2025 faute de notifications encore enregistrees. La
# fenetre retenue par le memoire est grisee.
#
# CHARTE. Huit types d'organisation, or la charte ne porte que TROIS emplacements categoriels
# et rampe() leve au-dela. Les types sont donc ORDONNES par volume cumule, les cinq premiers
# prennent la rampe sequentielle a six paliers et les trois derniers se replient en « autres ».
# C'est la regle du projet, la meme qui fait traiter les cinq piliers en rampe ordinale.
prc_all = prc.copy()
prc_all["annee_all"] = prc_all["d_breach"].dt.year
hist = prc_all[(prc_all["annee_all"] >= 2005) & (prc_all["annee_all"] <= 2025) &
               renseigne(prc_all["organization_type"])]
ordre = hist["organization_type"].value_counts().index.tolist()
garde, autres = ordre[:5], ordre[5:]
pivot = (hist.assign(cat=hist["organization_type"].where(
             hist["organization_type"].isin(garde), "autres"))
         .pivot_table(index="annee_all", columns="cat", aggfunc="size", fill_value=0))
cols = [c for c in garde if c in pivot.columns] + (["autres"] if "autres" in pivot.columns else [])
pivot = pivot[cols]

print("\n  D4, incidents par annee et par type d'organisation (extrait) :")
print("    annee  " + "".join(f"{c:>9s}" for c in cols) + "     total")
pivot.index = pivot.index.astype(int)
for an in pivot.index:
    if an % 3 == 0 or an >= 2019:
        print(f"    {int(an):5d}  " + "".join(f"{pivot.loc[an, c]:9.0f}" for c in cols)
              + f"{pivot.loc[an].sum():10.0f}")

fig, ax = plt.subplots(figsize=(7.6, 4.0))
teintes = sn.SEQUENTIEL_6[:len(cols)]
ax.stackplot(pivot.index, [pivot[c].values for c in cols], labels=cols, colors=teintes,
             edgecolor="white", linewidth=0.4)
ax.axvspan(2019, 2025, color=sn.CHARTE["gris"], alpha=0.20, zorder=0)
ax.annotate("fenêtre retenue", xy=(2022, ax.get_ylim()[1] * 0.92), ha="center",
            fontsize=8.5, color=sn.ENCRE_2)
ax.set_xlabel("année de survenance", fontsize=9)
ax.set_ylabel("nombre d'incidents", fontsize=9)
ax.set_xlim(2005, 2025)
ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
ax.set_title("Composition de la base par type d'organisation, 2005 à 2025",
             fontsize=10, color=sn.ENCRE)
# Annees en ENTIERS : matplotlib graduerait sinon en 2007,5, qui n'existe pas.
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "D4_composition_temporelle.png"), dpi=150)
plt.close(fig)

pic = int(pivot.sum(axis=1).idxmax())
n_pic = int(pivot.sum(axis=1).max())
n_2025 = int(pivot.sum(axis=1).loc[2025]) if 2025 in pivot.index else 0
avant = int(pivot.loc[pivot.index < 2015].sum().sum())
print(f"\n  Pic en {pic} avec {n_pic} incidents ; {n_2025} en 2025, soit "
      f"{100.0*n_2025/n_pic:.1f} % du pic. Avant 2015 la base porte {avant} incidents au total, "
      f"soit {100.0*avant/len(hist):.1f} % de la periode 2005-2025.")

# --- D5 : frequence contre volume par type d'organisation, double axe ------------------------
#
# POURQUOI CETTE FIGURE. Les tests de la section 2 concluent que la classe d'organisation
# informe FAIBLEMENT la severite. Une taille d'effet ne se voit pas ; ce graphique la montre :
# l'ordre des barres et celui de la courbe ne coincident pas, donc compter les incidents et
# sommer les volumes classent les memes categories differemment.
org = (per[renseigne(per["organization_type"])]
       .groupby("organization_type").agg(incidents=("ta", "size"), volume=("ta", "sum"))
       .sort_values("incidents", ascending=False))
print("\n  D5, par type d'organisation :")
print("    %-8s %10s %16s %14s" % ("type", "incidents", "volume (M enr.)", "rang volume"))
rang_vol = org["volume"].rank(ascending=False).astype(int)
for t, r in org.iterrows():
    print("    %-8s %10d %16.1f %14d" % (t, r["incidents"], r["volume"] / 1e6, rang_vol[t]))

n_discord = int((rang_vol.values != np.arange(1, len(org) + 1)).sum())
print(f"\n  {n_discord} types sur {len(org)} n'ont pas le meme rang en nombre et en volume.")

fig, ax1 = plt.subplots(figsize=(7.6, 4.0))
x = np.arange(len(org))
ax1.bar(x, org["incidents"].values, color=sn.CATEGORIEL[2], width=0.66,
        label="nombre d'incidents")
ax1.set_ylabel("nombre d'incidents", fontsize=9, color=sn.CATEGORIEL[2])
ax1.tick_params(axis="y", labelcolor=sn.CATEGORIEL[2])
ax1.set_xticks(x); ax1.set_xticklabels(org.index, fontsize=9)
ax1.set_xlabel("type d'organisation", fontsize=9)
ax2 = ax1.twinx()
ax2.plot(x, org["volume"].values / 1e6, color=sn.CATEGORIEL[0], marker="o", lw=1.8,
         label="volume touché")
ax2.set_ylabel("volume touché (millions d'enregistrements)", fontsize=9, color=sn.CATEGORIEL[0])
ax2.tick_params(axis="y", labelcolor=sn.CATEGORIEL[0])
ax2.spines["right"].set_visible(True)
ax1.set_title("Type d'organisation : nombre d'incidents et volume touché",
              fontsize=10, color=sn.ENCRE)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "D5_organisation_double_axe.png"), dpi=150)
plt.close(fig)

print("\n  Cinq figures ecrites dans exploratory/vasicek_lab/figures/ :")
print("    D1a_presence_taux.png  D1b_presence_effectifs.png")
print("    D2_sources_notification.png  D3_bulles_breche.png")
print("    D4_composition_temporelle.png  D5_organisation_double_axe.png")

# =====================================================================================
titre("5. GRANDEURS CITEES (sans separateur de milliers, pour le harnais)")
# =====================================================================================
g = []
g.append(("incidents PRC export brut", n_prc))
g.append(("incidents PRC impact renseigne", n_ta))
g.append(("part impact renseigne pct", round(100.0 * n_ta / n_prc, 2)))
g.append(("incidents PRC perimetre 2019-2025", len(per)))
g.append(("lignes SAS brutes", n_sas_brut))
g.append(("pertes SAS retenues", n_sas))
g.append(("pertes SAS cyber TIC", len(cyber)))
g.append(("pertes SAS finance", len(fin)))
g.append(("pertes SAS cyber x finance", len(cyberfin)))
g.append(("seuil q75 PRC enregistrements", int(q75_prc)))
g.append(("intersection triple pct", round(triple, 1)))
g.append(("seuil q75 SAS MUSD", round(q75_sas, 4)))
for c in ["breach_date", "end_breach_date", "total_affected", "residents_affected",
          "organization_type", "breach_type"]:
    g.append((f"presence PRC {c} pct", round(presence[c], 2)))
g.append(("shapiro W brut PRC", round(sw_prc[0], 4)))
g.append(("shapiro W log PRC", round(sw_prc[2], 4)))
g.append(("shapiro W brut SAS", round(sw_sas[0], 4)))
g.append(("shapiro W log SAS", round(sw_sas[2], 4)))
g.append(("asymetrie PRC", round(sw_prc[4], 2)))
g.append(("aplatissement PRC", round(sw_prc[5], 2)))
g.append(("asymetrie SAS", round(sw_sas[4], 2)))
g.append(("aplatissement SAS", round(sw_sas[5], 2)))
for b, r in rho_taille.items():
    g.append((f"spearman perte x {b}", round(r, 4)))
g.append(("n spearman SAS", nsas_sp))
g.append(("n spearman PRC", len(sp)))
for nom, res in [("KW PRC organisation", kw_prc), ("KW PRC breche", kw_prcb),
                 ("KW SAS categorie", kw_sas), ("KW SAS ligne metier", kw_sasb)]:
    if res:
        g.append((f"{nom} eps2", round(res[3], 4)))
        g.append((f"{nom} classes", res[2]))
for nom, val in g:
    print(f"  {nom:<44s} {val}")

print("\nFIN 104")
