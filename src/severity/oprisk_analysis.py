"""
severity/oprisk_analysis.py
----------------------------
Analyse exploratoire qualitative et quantitative de la base
SAS OpRisk Global Data.

Objectifs :
  1. Vue d'ensemble de la base (catégories Bâle, volumétrie)
  2. Isolation du périmètre cyber/ICT
  3. Statistiques de sévérité (cyber × finance)
  4. Concentration de la perte, évolution temporelle, géographie
  5. Diagnostic des biais (US-centré, seuil de collecte, aberrations)

Périmètre cyber/ICT retenu :
  - Sub Risk Category ∈ {Systems Security, Systems}
  - OU Event Risk Category = Business Disruption and System Failures

Usage : python notebooks/02_oprisk_exploratory.py
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

USD_EUR = 0.92
ABERRATION_THRESHOLD = 100_000  # M$ — pertes au-delà = erreurs de saisie

CYBER_SUB_CATS = ["Systems Security", "Systems"]
CYBER_EVENT_CATS = ["Business Disruption and System Failures"]


# ---------------------------------------------------------------------------
# 1. CHARGEMENT ET NETTOYAGE
# ---------------------------------------------------------------------------

def load_clean(path: str, sheet: str = "Datasets") -> pd.DataFrame:
    """
    Charge la base et nettoie les aberrations de saisie.

    Returns
    -------
    DataFrame nettoyé avec colonnes dérivées : loss, year
    """
    df = pd.read_excel(path, sheet_name=sheet)
    df["loss"] = pd.to_numeric(df["Loss Amount ($M)"], errors="coerce")
    df["year"] = pd.to_datetime(df["First Year of Event"], errors="coerce").dt.year

    n_before = len(df)
    df = df[(df["loss"] > 0) & (df["loss"] < ABERRATION_THRESHOLD)].copy()
    n_removed = n_before - len(df)

    print(f"Base chargée : {len(df):,} incidents valides "
          f"({n_removed} aberration(s) > {ABERRATION_THRESHOLD:,}M$ retirée(s))")
    return df


# ---------------------------------------------------------------------------
# 2. ISOLATION DU PÉRIMÈTRE CYBER/ICT
# ---------------------------------------------------------------------------

def filter_cyber(df: pd.DataFrame) -> pd.DataFrame:
    """Isole les incidents cyber/ICT selon les catégories Bâle."""
    mask = (
        df["Sub Risk Category"].isin(CYBER_SUB_CATS) |
        df["Event Risk Category"].isin(CYBER_EVENT_CATS)
    )
    return df[mask].copy()


def filter_finance(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre le secteur financier."""
    return df[df["Industry Sector Name"].apply(
        lambda v: "Financial" in str(v) if pd.notna(v) else False)].copy()


# ---------------------------------------------------------------------------
# 3. ANALYSE QUANTITATIVE
# ---------------------------------------------------------------------------

def severity_stats(losses: pd.Series, label: str = "", currency: str = "M$") -> dict:
    """Statistiques de sévérité d'une série de pertes."""
    pcts = [.1, .25, .5, .75, .9, .95, .99, .999]
    q = losses.quantile(pcts)
    stats = {
        "n": len(losses),
        "mean": losses.mean(),
        "median": losses.median(),
        "std": losses.std(),
        "max": losses.max(),
        "total": losses.sum(),
        "skewness": losses.skew(),
        "quantiles": {f"P{int(p*100)}": q.loc[p] for p in pcts},
    }
    if label:
        print(f"\n=== SÉVÉRITÉ — {label} ({currency}) ===")
        print(f"  n        = {stats['n']:,}")
        print(f"  médiane  = {stats['median']:.2f}")
        print(f"  moyenne  = {stats['mean']:.1f}")
        print(f"  P90      = {stats['quantiles']['P90']:.1f}")
        print(f"  P99      = {stats['quantiles']['P99']:.0f}")
        print(f"  max      = {stats['max']:.0f}")
        print(f"  total    = {stats['total']:,.0f}")
        print(f"  skewness = {stats['skewness']:.1f}")
    return stats


def concentration_analysis(losses: pd.Series) -> dict:
    """
    Analyse de concentration de la perte (signature de la queue lourde).
    """
    s = losses.sort_values(ascending=False)
    total = s.sum()
    n = len(s)

    top1 = s.head(max(1, int(n * 0.01))).sum() / total
    top10 = s.head(int(n * 0.10)).sum() / total
    top25 = s.head(int(n * 0.25)).sum() / total

    print(f"\n=== CONCENTRATION DE LA PERTE ({n} incidents) ===")
    print(f"  Top  1% incidents = {100*top1:.1f}% de la perte totale")
    print(f"  Top 10% incidents = {100*top10:.1f}% de la perte totale")
    print(f"  Top 25% incidents = {100*top25:.1f}% de la perte totale")
    print(f"  → forte concentration = signature d'une queue lourde (EVT justifié)")

    return {"top1": top1, "top10": top10, "top25": top25}


def temporal_evolution(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """Évolution de la sévérité par période temporelle."""
    if periods is None:
        periods = [(1990,1999),(2000,2009),(2010,2019),(2020,2026)]

    rows = []
    for lo, hi in periods:
        sub = df[(df["year"] >= lo) & (df["year"] <= hi)]
        if len(sub) > 0:
            rows.append({
                "période": f"{lo}-{hi}",
                "n": len(sub),
                "médiane": sub["loss"].median(),
                "moyenne": sub["loss"].mean(),
                "max": sub["loss"].max(),
            })

    res = pd.DataFrame(rows)
    print(f"\n=== ÉVOLUTION TEMPORELLE DE LA SÉVÉRITÉ ===")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"  → sévérité croissante : ne pas calibrer sur données trop anciennes")
    return res


def geographic_breakdown(df: pd.DataFrame, top: int = 8) -> pd.Series:
    """Répartition géographique des incidents."""
    geo = df["Country of Incident"].value_counts().head(top)
    print(f"\n=== RÉPARTITION GÉOGRAPHIQUE (top {top}) ===")
    for k, v in geo.items():
        print(f"  {v:4d}  {k}")
    us_share = 100 * (df["Country of Incident"] == "United States").sum() / len(df)
    print(f"  → biais US : {us_share:.0f}% des incidents (mal aligné avec périmètre DORA UE)")
    return geo


def basel_business_lines(df: pd.DataFrame, top: int = 8) -> pd.Series:
    """Répartition par ligne métier Bâle."""
    bl = df["Basel Business Line - Level 1"].value_counts().head(top)
    print(f"\n=== LIGNES MÉTIER BÂLE (top {top}) ===")
    for k, v in bl.items():
        print(f"  {v:4d}  {str(k)[:45]}")
    return bl


# ---------------------------------------------------------------------------
# 4. RAPPORT COMPLET
# ---------------------------------------------------------------------------

def full_report(path: str) -> dict:
    """
    Génère le rapport d'analyse complet de la base OpRisk.
    """
    print("="*60)
    print("  ANALYSE EXPLORATOIRE — SAS OpRisk Global Data")
    print("="*60)

    df = load_clean(path)

    # Vue d'ensemble Bâle
    print(f"\n=== CATÉGORIES DE RISQUE BÂLE (base complète) ===")
    erc = df["Event Risk Category"].value_counts()
    for k, v in erc.head(8).items():
        med = df[df["Event Risk Category"] == k]["loss"].median()
        print(f"  {v:6,d} ({100*v/len(df):4.1f}%)  méd={med:6.2f}M$  {k}")

    # Périmètre cyber
    cyber = filter_cyber(df)
    print(f"\n=== PÉRIMÈTRE CYBER/ICT ===")
    print(f"  Incidents cyber/ICT (tous secteurs) : {len(cyber):,}")
    print(f"  Top secteurs :")
    for k, v in cyber["Industry Sector Name"].value_counts().head(5).items():
        print(f"    {v:4d}  {str(k)[:40]}")

    # Cyber × Finance
    cf = filter_finance(cyber)
    severity_stats(cf["loss"], label="Cyber × Finance", currency="M$")
    concentration_analysis(cf["loss"])
    temporal_evolution(cf)
    geographic_breakdown(cf)
    basel_business_lines(cf)

    print("\n" + "="*60)
    print("  LIMITES IDENTIFIÉES (à documenter dans le mémoire)")
    print("="*60)
    print("  1. Seuil de collecte élevé : petites pertes absentes")
    print("  2. Biais de déclaration : seules pertes publiques/judiciarisées")
    print("  3. Biais géographique US-centré : ~46% des incidents cyber")
    print("  4. Qualité variable : aberrations de saisie à nettoyer")
    print("="*60 + "\n")

    return {
        "df": df, "cyber": cyber, "cyber_finance": cf,
        "n_total": len(df), "n_cyber": len(cyber), "n_cyber_finance": len(cf),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    OPRISK_PATH = "data/raw/SAS_OpRisk_Global_Data_June_2026.xlsx"

    if not os.path.exists(OPRISK_PATH):
        print(f"⚠ Fichier non trouvé : {OPRISK_PATH}")
        print("  Placer le fichier dans data/raw/ (gitignored)")
    else:
        results = full_report(OPRISK_PATH)
