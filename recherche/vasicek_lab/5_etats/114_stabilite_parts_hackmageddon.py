#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
114 : les parts par vecteur d'attaque, et donc le multiplicateur de frequence, dependent-elles de
la periode d'observation retenue dans Hackmageddon ?

POURQUOI CE SCRIPT EXISTE. Le multiplicateur qui fait passer la frequence de l'etat conforme a
l'etat non conforme (21,56 -> 53,62 incidents par an, script 99) pondere des multiplicateurs par
vecteur d'attaque par les parts pi_v lues dans Hackmageddon sur le premier semestre 2026. Ces parts
etaient une citation externe non recalculable, le jeu n'ayant pas ete conserve. Les fichiers
semestriels de Hackmageddon ont ete recuperes le 8 octobre 2026 (data/raw/hackmageddon/, sous
licence de leur auteur, non versionnes). Ce script les relit et repond a une seule question : le
multiplicateur bouge-t-il si l'on prend les parts d'une autre periode ?

CE QU'IL NE FAIT PAS. Il ne remplace aucune valeur publiee. Les parts du moteur restent celles de
src.frequency.negbin.HACKMAGEDDON_PROPORTIONS, et les decomptes publies (1 041 incidents, 840 a
vecteur identifie) restent des citations datees : le classement des vecteurs fait ici n'est pas
celui qui les a produits, et il ne les retrouve pas a l'unite.

METHODE. Un incident = meme jour de publication, meme cible, meme debut de description : les
fichiers se recouvrent (copies, consolidations), d'ou un dedoublonnage. Le champ « Initial Access »
n'existe que pour 2025 et 2026 ; il est classe en cinq familles par une grille de mots-cles ecrite
ci-dessous, y compris pour les codes MITRE ATT&CK adoptes par la source en 2026. Les multiplicateurs
par vecteur (centres des fourchettes S1 et S2) sont ceux du moteur, importes et non recopies.

ARRET DUR si la base relue ne redonne pas les effectifs attendus (base modifiee ou incomplete), ou
si les parts du moteur ne redonnent pas le multiplicateur publie.

Usage, depuis la racine du depot :
    .venv\Scripts\python.exe exploratory\vasicek_lab\5_etats\114_stabilite_parts_hackmageddon.py > sorties_verif\114.txt
"""

import os
import pathlib
import re
import sys
import warnings

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DEPOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, ROOT_DEPOT):
    if p not in sys.path:
        sys.path.insert(0, p)
import canaux_conformite as cx                                            # noqa: E402
from src.frequency.negbin import (HACKMAGEDDON_PROPORTIONS,               # noqa: E402
                                  MULTIPLICATEURS_DORA)

warnings.filterwarnings("ignore")
WID = 88
BASE = pathlib.Path(ROOT_DEPOT) / "data" / "raw" / "hackmageddon"
DOSSIERS = ("2025", "2026")          # seules annees portant le champ « Initial Access »

# Effectifs attendus de la base recuperee le 8 octobre 2026 (controle d'identite).
ATTENDU_S1_2026 = 1137
ATTENDU_S1_2026_IDENTIFIES = 897

VECTEURS = ("phishing_social_eng", "exploit_vuln", "supply_chain_tiers", "identifiants", "autres")
PERIODES = {
    "T1 2025":              ("2025-01-01", "2025-04-01"),
    "S1 2026":              ("2026-01-01", "2026-07-01"),
    "juillet-octobre 2026": ("2026-07-01", "2026-11-01"),
    "2026 entier":          ("2026-01-01", "2026-11-01"),
}
REN = {"date": "Date", "date reported": "Date", "date published": "Date",
       "target": "Target", "description": "Description", "attack": "Attack", "technique": "Attack",
       "initial access": "InitialAccess", "initial vector": "InitialAccess"}


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def norm(c):
    return re.sub(r"\s+", " ", str(c).replace("\n", " ")).strip().lower()


def charger():
    frames = []
    for d in DOSSIERS:
        for f in sorted((BASE / d).rglob("*.xls*")):
            x = pd.ExcelFile(f)
            for sh in x.sheet_names:
                raw = x.parse(sh, header=None)
                hdr = None
                for i in range(min(4, len(raw))):
                    row = [norm(v) for v in raw.iloc[i].tolist()]
                    if ("attack" in row or "technique" in row) and any(v.startswith("date") for v in row):
                        hdr = i
                        break
                if hdr is None:
                    continue
                df = raw.iloc[hdr + 1:].copy()
                df.columns = [norm(v) for v in raw.iloc[hdr].tolist()]
                keep = {}
                for c in df.columns:
                    if c in REN and REN[c] not in keep.values():
                        keep[c] = REN[c]
                df = df[list(keep)].rename(columns=keep).dropna(how="all")
                if "Date" not in df.columns:
                    continue
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                frames.append(df[df["Date"].notna()])
    b = pd.concat(frames, ignore_index=True)
    cle = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())[:60]           # noqa: E731
    b["kd"] = b["Description"].fillna("").map(cle)
    b["kt"] = b["Target"].fillna("").map(cle)
    b["jour"] = b["Date"].dt.normalize()
    return b.drop_duplicates(["jour", "kt", "kd"])


def famille(s):
    s = str(s).lower()
    if s in ("nan", "", "n/a", "none", "<na>") or "unknown" in s:
        return None
    if re.search(r"t1566|phish|social eng|spearphish|smish|vish|rogue website|malvertis|watering|"
                 r"drive-by|t1189|fake|lure|user execution|t1204|copy and paste", s):
        return "phishing_social_eng"
    if re.search(r"t1190|vulnerab|exploit|zero.day|0-day|t1133|external remote|t1210|t1203", s):
        return "exploit_vuln"
    if re.search(r"supply chain|t1195|t1199|trusted relationship|third|vendor|provider", s):
        return "supply_chain_tiers"
    if re.search(r"credential|t1078|valid account|brute|t1110|password|account takeover|stuffing|t1586", s):
        return "identifiants"
    return "autres"


def multiplicateur(parts, scen):
    return sum(parts.get(v, 0.0) * sum(MULTIPLICATEURS_DORA[scen][v]) / 2 for v in VECTEURS)


# =====================================================================================
titre("1. Controle des parts du moteur")
# =====================================================================================
m_pub = multiplicateur(HACKMAGEDDON_PROPORTIONS, "S2_non_conforme")
assert abs(cx.LAM_C * m_pub - cx.LAM_NC) < 1e-9, "les parts du moteur ne redonnent pas LAM_NC"
print(f"  parts publiees (S1 2026, citation) : multiplicateur S2 = {m_pub:.4f}, "
      f"lambda NC = {cx.LAM_C * m_pub:.2f} -> identique au moteur")

# =====================================================================================
titre("2. Base relue et controle d'identite")
# =====================================================================================
u = charger()
u["vec"] = u["InitialAccess"].map(famille)
s1 = u[(u.Date >= "2026-01-01") & (u.Date < "2026-07-01")]
n_s1, n_s1_id = len(s1), int(s1.vec.notna().sum())
print(f"  incidents uniques 2025-2026 : {len(u)}")
print(f"  S1 2026 : {n_s1} incidents, {n_s1_id} a vecteur identifie "
      f"(attendus {ATTENDU_S1_2026} et {ATTENDU_S1_2026_IDENTIFIES})")
if (n_s1, n_s1_id) != (ATTENDU_S1_2026, ATTENDU_S1_2026_IDENTIFIES):
    sys.exit("ARRET : la base relue ne redonne pas les effectifs attendus.")
print("  publie dans le memoire : 1 041 incidents et 840 identifies, autre grille et autre date de")
print("  consultation ; les decomptes publies restent des citations datees.")

# =====================================================================================
titre("3. Parts par vecteur et multiplicateur, periode par periode")
# =====================================================================================
print(f"  {'periode':<22}{'n':>6}{'ident.':>8}" + "".join(f"{v[:8]:>10}" for v in VECTEURS)
      + f"{'mult S1':>9}{'mult S2':>9}{'lam NC':>8}")
pub = {v: HACKMAGEDDON_PROPORTIONS[v] for v in VECTEURS}
print(f"  {'publie (moteur)':<22}{'':>6}{'':>8}" + "".join(f"{pub[v]:>10.3f}" for v in VECTEURS)
      + f"{multiplicateur(pub, 'S1_partiel'):>9.4f}{m_pub:>9.4f}{cx.LAM_C * m_pub:>8.2f}")
res = {}
for nom, (a, b) in PERIODES.items():
    s = u[(u.Date >= a) & (u.Date < b)]
    v = s.vec.dropna()
    p = v.value_counts(normalize=True).to_dict()
    m1, m2 = multiplicateur(p, "S1_partiel"), multiplicateur(p, "S2_non_conforme")
    res[nom] = (len(s), len(v), p, m1, m2)
    print(f"  {nom:<22}{len(s):>6}{len(v):>8}" + "".join(f"{p.get(k, 0):>10.3f}" for k in VECTEURS)
          + f"{m1:>9.4f}{m2:>9.4f}{cx.LAM_C * m2:>8.2f}")

# =====================================================================================
titre("4. Lecture")
# =====================================================================================
m2s = [r[4] for r in res.values()]
lo, hi = min(m2s), max(m2s)
borne_lo = sum(HACKMAGEDDON_PROPORTIONS[v] * MULTIPLICATEURS_DORA["S2_non_conforme"][v][0] for v in VECTEURS)
borne_hi = sum(HACKMAGEDDON_PROPORTIONS[v] * MULTIPLICATEURS_DORA["S2_non_conforme"][v][1] for v in VECTEURS)
ecart_max = max(abs(m - m_pub) for m in m2s) / m_pub
print(f"  multiplicateur S2 selon la periode : {lo:.4f} a {hi:.4f} (publie {m_pub:.4f})")
print(f"  ecart relatif maximal au publie : {100 * ecart_max:.1f} %")
print(f"  lambda NC correspondant : {cx.LAM_C * lo:.2f} a {cx.LAM_C * hi:.2f} (publie {cx.LAM_C * m_pub:.2f})")
print(f"  a comparer aux bornes posees des fourchettes : {borne_lo:.4f} a {borne_hi:.4f}")
print(f"  largeur due a la periode / largeur due aux bornes : {(hi - lo) / (borne_hi - borne_lo):.3f}")
id_t34 = res["juillet-octobre 2026"]
print(f"  part de vecteurs identifies, juillet-octobre 2026 : {100 * id_t34[1] / id_t34[0]:.1f} % "
      "(passage de la source aux codes MITRE ATT&CK)")

# =====================================================================================
titre("5. Grandeurs citees, sans separateur")
# =====================================================================================
print(f"  multiplicateur S2 min {lo:.3f} max {hi:.3f} ; lambda NC min {cx.LAM_C * lo:.1f} max {cx.LAM_C * hi:.1f}")
print(f"  ecart relatif maximal {100 * ecart_max:.1f} % ; bornes posees {borne_lo:.3f} a {borne_hi:.3f}")
print(f"  rapport des largeurs {(hi - lo) / (borne_hi - borne_lo):.3f}")
print(f"  S1 2026 relu : {n_s1} incidents, {n_s1_id} identifies ; T1 2025 : {res['T1 2025'][0]} incidents")
print(f"  part identifiee juillet-octobre 2026 {100 * id_t34[1] / id_t34[0]:.1f} %")
