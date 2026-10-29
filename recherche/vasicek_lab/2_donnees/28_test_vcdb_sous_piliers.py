#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28 : peut-on calibrer les SOUS-PILIERS sur donnee ouverte ? Test sur VCDB.

Phase 6 de la feuille de route, volet donnee ouverte. Un praticien : chaque pilier a des
sous-piliers, cherchons des bases en ligne. On teste l'idee sur la VCDB (VERIS Community
Database, ~10 600 incidents codes selon le schema VERIS), secteur FINANCE (NAICS 52).

Le schema VERIS porte une taxonomie fine qui MAPPE sur des sous-piliers DORA :
  - P2 (incidents)   : familles d'action (hacking / malware / erreur / abus / social / physique) ;
  - P3 (resilience)  : timelines de DECOUVERTE et de CONTAINMENT (vitesse de detection/reprise) ;
  - P4 (tiers)       : implication d'un acteur PARTENAIRE (actor.Partner).

QUESTION REELLE : jusqu'ou la sous-maille tient-elle avant de devenir trop creuse ? On
mesure donc AUSSI l'entonnoir de sparsite. Verdict attendu (et a assumer) : les
sous-piliers BIEN peuples (types d'action P2) tiennent ; la detection P3 tient
grossierement ; le TIERS P4 s'effondre vite sur donnee ouverte (et VERIS SOUS-CODE le
partenaire), ce qui justifie le recours aux donnees clients (registre d'externalisation).

Donnee : data/raw/vcdb.csv (VCDB aplati, non versionne ; source vz-risk/VCDB, GitHub).
Ne touche ni src/ ni memoire/. Sortie : diagnostics + figure X_vcdb_sous_piliers.png.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "data", "raw"))
CSV = os.path.join(RAW, "vcdb.csv")
if not os.path.exists(CSV):
    sys.exit(f"donnee absente : {CSV}\n(VCDB non versionne ; telecharger vz-risk/VCDB)")

W = 74
ACT_FAM = ["Hacking", "Malware", "Misuse", "Physical", "Error", "Social"]
UNITS_FAST = ["Seconds", "Minutes", "Hours", "Days"]     # <= jour
UNITS_SLOW = ["Weeks", "Months", "Years"]
FIN = "victim.industry2.52"                               # NAICS 52 = finance


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


allc = pd.read_csv(CSV, nrows=0).columns.tolist()
disc = [f"timeline.discovery.unit.{u}" for u in UNITS_FAST + UNITS_SLOW + ["Never"]]
cont = [f"timeline.containment.unit.{u}" for u in UNITS_FAST + UNITS_SLOW + ["Never"]]
use = [FIN, "actor.Partner", "actor.External", "actor.Internal",
       "timeline.incident.year"] + [f"action.{f}" for f in ACT_FAM] + disc + cont
use = [c for c in use if c in allc]
df = pd.read_csv(CSV, usecols=use, low_memory=False)


def b(col):
    return df[col].fillna(0).astype(bool) if col in df else pd.Series(False, index=df.index)


fin = b(FIN)
NF = int(fin.sum())
print(f"VCDB : {len(df)} incidents | finance (NAICS 52) : {NF}")


def speed_dist(prefix, mask):
    """Repartition rapide (<=jour) / semaines / mois+ / jamais, sur les incidents dates."""
    known = pd.Series(False, index=df.index)
    buckets = {"<= 1 jour": UNITS_FAST, "semaines": ["Weeks"],
               "mois et +": ["Months", "Years"], "jamais": ["Never"]}
    counts = {}
    for lab, units in buckets.items():
        cols = [f"{prefix}.{u}" for u in units if f"{prefix}.{u}" in df]
        m = pd.Series(False, index=df.index)
        for c in cols:
            m = m | b(c)
        counts[lab] = int((m & mask).sum())
        known = known | m
    ntot = int((known & mask).sum())
    return counts, ntot


# =====================================================================================
titre("P2 : sous-piliers = familles d'action (finance)")
# =====================================================================================
p2 = {f: int((b(f"action.{f}") & fin).sum()) for f in ACT_FAM}
for f in sorted(p2, key=p2.get, reverse=True):
    print(f"   action {f:<12} : {p2[f]}")
print("   -> plusieurs centaines par famille : sous-maille P2 exploitable.")

# =====================================================================================
titre("P3 : sous-piliers = detection et containment (finance)")
# =====================================================================================
disc_counts, n_disc = speed_dist("timeline.discovery.unit", fin)
cont_counts, n_cont = speed_dist("timeline.containment.unit", fin)
print(f"  DECOUVERTE (n={n_disc}) : {disc_counts}")
print(f"  CONTAINMENT (n={n_cont}) : {cont_counts}")
slow_disc = (disc_counts["mois et +"] + disc_counts["jamais"]) / max(1, n_disc)
print(f"  part de detection LENTE (mois et +, ou jamais) : {slow_disc:.0%}")
print(f"  -> detection P3 exploitable grossierement ({n_disc}), containment mince ({n_cont}).")

# =====================================================================================
titre("P4 : sous-pilier = tiers (actor.Partner, finance)")
# =====================================================================================
part = b("actor.Partner")
n_part_fin = int((part & fin).sum())
print(f"  incidents finance avec acteur PARTENAIRE : {n_part_fin} "
      f"({n_part_fin / max(1, NF):.1%} de la finance)")
print(f"  part globale actor.Partner : {part.mean():.1%}  <- VERIS SOUS-CODE le tiers")
print("  -> sous-maille P4 tres creuse ET sous-comptee : la donnee ouverte ne suffit pas.")

# =====================================================================================
titre("ENTONNOIR DE SPARSITE : ou la sous-maille se casse")
# =====================================================================================
hack = b("action.Hacking")
funnel = [
    ("VCDB (tous secteurs)", len(df)),
    ("finance", NF),
    ("finance + decouverte datee", n_disc),
    ("finance + containment date", n_cont),
    ("finance + tiers (partenaire)", n_part_fin),
    ("finance + tiers + hacking", int((part & fin & hack).sum())),
]
for lab, v in funnel:
    flag = "  <- sous le seuil exploitable (~30)" if v < 30 else ""
    print(f"   {lab:<34} {v:>6}{flag}")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("Le praticien a raison, avec une frontiere nette : la donnee ouverte (VCDB) peuple les")
print("sous-piliers BIEN representes (types d'action P2, plusieurs centaines) et, plus")
print("grossierement, la detection P3. Mais le TIERS P4 s'effondre (56 en finance) et")
print("VERIS le sous-code : deux crans plus bas, on tombe sous le seuil exploitable.")
print("Conclusion operationnelle : open source pour P2/P3 au niveau sous-pilier ; pour")
print("les sous-process de P4 (registre d'externalisation, chaines de sous-traitance),")
print("il faut les donnees clients d'une entite. La granularite ne se decrete pas, elle se")
print("verifie sur la volumetrie.")

# =====================================================================================
# figure
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 11,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
})
INK, INK2, MUTED, GRID = "#1b1e30", "#223e55", "#595959", "#f2f2f2"
ACCENT = "#a6002e"
BL = ["#7baafd", "#4c79c7", "#204993"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.8, 4.9),
                                    gridspec_kw={"width_ratios": [1, 1, 1.15]})

# (a) P2 : familles d'action en finance
order = sorted(ACT_FAM, key=lambda f: p2[f], reverse=True)
ax1.barh(range(len(order)), [p2[f] for f in order], color=BL[1], edgecolor="#fcfcfb",
         height=0.66)
for i, f in enumerate(order):
    ax1.text(p2[f] + 4, i, str(p2[f]), va="center", fontsize=8.4, color=INK2)
ax1.set_yticks(range(len(order))); ax1.set_yticklabels(order, fontsize=8.7)
ax1.invert_yaxis()
ax1.set_xlim(0, max(p2.values()) * 1.2)
ax1.set_xlabel("incidents finance", color=INK2)
ax1.set_title("(a)  P2 : types d'action, sous-maille robuste", fontsize=11, color=INK, pad=8)
for s in ("top", "right", "left"):
    ax1.spines[s].set_visible(False)
ax1.tick_params(axis="y", length=0)

# (b) P3 : vitesse de detection et containment
labs = ["<= 1 jour", "semaines", "mois et +", "jamais"]
dvals = [disc_counts[l] for l in labs]
cvals = [cont_counts[l] for l in labs]
x = np.arange(len(labs)); h = 0.38
ax2.bar(x - h / 2, dvals, h, color=BL[2], edgecolor="#fcfcfb", label=f"découverte (n={n_disc})")
ax2.bar(x + h / 2, cvals, h, color=ACCENT, edgecolor="#fcfcfb", label=f"containment (n={n_cont})")
ax2.set_xticks(x); ax2.set_xticklabels(labs, fontsize=8.3)
ax2.set_ylabel("incidents finance", color=INK2)
ax2.legend(frameon=False, fontsize=8.2)
ax2.set_title("(b)  P3 : détection/containment, plus mince", fontsize=11, color=INK, pad=8)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# (c) entonnoir de sparsite
flabs = [f[0] for f in funnel][::-1]
fvals = [f[1] for f in funnel][::-1]
cols = [ACCENT if v < 30 else (MUTED if v == len(df) else BL[1]) for v in fvals]
ax3.barh(range(len(flabs)), fvals, color=cols, edgecolor="#fcfcfb", height=0.7)
ax3.set_xscale("log")
for i, v in enumerate(fvals):
    ax3.text(v * 1.15, i, str(v), va="center", fontsize=8, color=INK2)
ax3.axvline(30, color=ACCENT, lw=1.2, ls="--")
ax3.text(30, len(flabs) - 0.4, "seuil ~30", color=ACCENT, fontsize=7.8, rotation=90,
         va="top", ha="right")
ax3.set_yticks(range(len(flabs))); ax3.set_yticklabels(flabs, fontsize=8)
ax3.set_xlabel("nombre d'incidents (log)", color=INK2)
ax3.set_xlim(1, len(df) * 2)
ax3.set_title("(c)  L'entonnoir : où la sous-maille se casse", fontsize=11, color=INK, pad=8)
for s in ("top", "right", "left"):
    ax3.spines[s].set_visible(False)
ax3.tick_params(axis="y", length=0)

fig.suptitle("X : calibrer les sous-piliers sur donnée ouverte (VCDB), et sa limite",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "X_vcdb_sous_piliers.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nfigure ecrite :", path)
