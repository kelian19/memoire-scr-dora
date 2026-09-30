#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08e : cas d'usage PILIER 2 (gestion des incidents), oriente KPI.

Origine : echange avec un praticien du cabinet. Deux directives structurantes.
  (1) On ne pourra PAS tester la contagion entre piliers sur donnee reelle (trop dur).
      -> c'est le verdict deja etabli par 05_faisabilite_donnees (W inidentifiable).
      La cascade reste le modele normatif ; le poids EMPIRIQUE se met la ou la donnee
      existe. Un praticien conseille : faire sauter P1, cibler P2/P3/P4, donnee open source.
  (2) Choix retenu : P2. Definir l'INCIDENT SIGNIFICATIF (impact financier OU
      reputationnel/media), puis TRANSFORMER LE SCORE DE CRITICITE EN QUELQUES KPI.

Ce script construit le socle empirique du cas P2 a partir de la Data Breach Chronology
(incidents rendus publics = significatifs par revelation). KPI central = le DELAI de
notification, seul KPI directement OPPOSABLE au texte DORA (art. 19 : notification
initiale, intermediaire, finale).

CAVEATS ASSUMES (a ecrire dans la note) :
  - reported_date = revelation PUBLIQUE / notification d'Etat US, PAS la notification au
    superviseur au sens DORA. Le delai mesure est donc un MAJORANT illustratif du delai
    reglementaire, pas sa mesure. Il reste valide pour comparer les TYPES entre eux.
  - donnee US (notifications d'Etats), proxy d'entites DORA, pas des entites DORA.
  - la tendance annuelle du delai est BIAISEE par la troncature a droite (un incident
    recent n'apparait que s'il est deja declare) : on ne la presente pas comme resultat.

Donnees attendues dans data/raw/ (non versionnees). Sortie : diagnostics + figure.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "data", "raw"))
SRC = os.path.join(RAW, "Data_Breach_Chronology.xlsx")
if not os.path.exists(SRC):
    sys.exit(f"donnee absente : {SRC}\n(les sources brutes ne sont pas versionnees)")

W = 78
# seuils de reference DORA (art. 19 du reglement UE 2022/2554), en jours
DORA = [(1, "initiale\n24 h"), (3, "intermediaire\n72 h"), (30, "finale\n1 mois")]
# libelles courts des types d'incident (codes Data Breach Chronology)
TYPE_LAB = {"HACK": "Piratage", "UNKN": "Non precise", "DISC": "Divulgation",
            "INSD": "Interne", "PORT": "Perte de support", "PHYS": "Physique",
            "CARD": "Carte", "STAT": "Statique"}


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# =====================================================================================
titre("Cas d'usage P2 : incidents du secteur financier avec les deux dates")
# =====================================================================================
d = pd.read_excel(SRC, sheet_name="Data_Breach_Chronology",
                  usecols=["breach_date", "reported_date", "breach_type",
                           "organization_type"])
d["bd"] = pd.to_datetime(d.breach_date.astype(str).str.strip(), errors="coerce")
d["rd"] = pd.to_datetime(d.reported_date.astype(str).str.strip(), errors="coerce")
bsf = d[(d.organization_type == "BSF") & d.bd.notna() & d.rd.notna()].copy()
bsf["lag"] = (bsf.rd - bsf.bd).dt.days
ok = bsf[(bsf.lag >= 0) & (bsf.lag <= 3650)].copy()      # 0 a 10 ans
L = ok.lag.to_numpy()
print(f"incidents BSF avec les deux dates, delai valide : {len(ok)}")
print(f"delai occurrence -> declaration (jours) :")
qs = {q: np.quantile(L, q) for q in (0.5, 0.9, 0.99)}
print(f"   mediane = {qs[0.5]:.0f} | q90 = {qs[0.9]:.0f} | q99 = {qs[0.99]:.0f} | "
      f"moyenne = {L.mean():.0f}")

# KPI 3 : conformite aux echeances DORA
print("\nKPI DELAI : part des incidents respectant chaque echeance DORA")
share = {}
prev = 0.0
for thr, lab in DORA:
    p = (L <= thr).mean()
    share[thr] = p
    print(f"   <= {lab.splitlines()[-1]:<6} : {p:6.2%}")
frac_over_month = (L > 30).mean()
print(f"   => {frac_over_month:.1%} des incidents sont declares APRES un mois, "
      "l'echeance FINALE de DORA.")

# KPI 3b : delai par type (difficulte de detection)
print("\nKPI DETECTION : delai median par type d'incident")
by_type = (ok.groupby("breach_type").lag
           .agg(["size", "median"]).query("size >= 30").sort_values("median"))
for t, row in by_type.iterrows():
    print(f"   {TYPE_LAB.get(t, t):<16} n={int(row['size']):>5}  "
          f"mediane={row['median']:6.0f} j")

# =====================================================================================
titre("Definition operationnelle de l'INCIDENT SIGNIFICATIF (double lecture)")
# =====================================================================================
print("Un incident est SIGNIFICATIF s'il declenche l'obligation de gestion P2. Deux")
print("lentilles complementaires, chacune adossee a une source :")
print("  - impact FINANCIER : perte en euros au-dessus d'un seuil        (SAS OpRisk)")
print("  - impact REPUTATIONNEL : revelation publique / mediatisation     (Data Breach)")
print("Elles ne selectionnent pas les memes incidents : un ransomware paye discretement")
print("pese en euros sans presse ; une fuite de donnees pese en presse sans perte directe.")
print("La maille P2 doit unir les deux, d'ou un score de criticite plutot qu'un montant.")

# =====================================================================================
titre("Traduction du SCORE DE CRITICITE en KPI (la demande du praticien)")
# =====================================================================================
print("Le modele qualitatif produit une criticite = f(probabilite, gravite). Pour le")
print("pilier P2, chacun de ses deux facteurs se lit comme un KPI MESURABLE :")
print("")
print("  criticite P2")
print("   |-- PROBABILITE  <->  KPI 1  frequence d'incidents significatifs")
print("   |                            0,10 a 0,21 / entite / an       (script 08b)")
print("   |-- GRAVITE      <->  KPI 2  impact financier, indice de queue xi ~ 0,90")
print("   |                                                            (script 05)")
print("                    <->  KPI 3  delai de notification, mediane 94 j (ce script)")
print("")
print("Le score abstrait devient ainsi un tableau de bord de 3 quantites observables,")
print("plus 1 KPI d'accumulation (part portee par les causes communes = 20 %, script 08d).")
print("C'est la reponse concrete a 'transformer le score en quelques KPI'.")

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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 4.8),
                                    gridspec_kw={"width_ratios": [1.15, 1, 1]})

# (a) distribution du delai + echeances DORA
bins = np.logspace(0, np.log10(3650), 40)
ax1.hist(L, bins=bins, color=BL[1], edgecolor="#fcfcfb")
ax1.set_xscale("log")
ymax = ax1.get_ylim()[1]
# etiquettes DORA decalees en hauteur pour ne pas se chevaucher (24 h et 72 h proches)
heights = {1: 0.97, 3: 0.80, 30: 0.97}
short = {1: "DORA 24 h", 3: "72 h", 30: "1 mois"}
for thr, _ in DORA:
    ax1.axvline(thr, color=ACCENT, lw=1.4, ls="--")
    ax1.text(thr * 1.08, ymax * heights[thr], short[thr], color=ACCENT, fontsize=8,
             ha="left", va="top")
ax1.axvline(qs[0.5], color=INK, lw=1.8)
ax1.text(qs[0.5] * 1.12, ymax * 0.55, f"mediane\n{qs[0.5]:.0f} j",
         color=INK, fontsize=8.5)
ax1.set_xlabel("delai occurrence $\\to$ declaration (jours, echelle log)", color=INK2)
ax1.set_ylabel("nombre d'incidents", color=INK2)
ax1.set_title(f"(a)  {frac_over_month:.0%} declares apres l'echeance finale DORA",
              fontsize=11, color=INK, pad=8)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# (b) delai median par type d'incident
labs = [TYPE_LAB.get(t, t) for t in by_type.index]
meds = by_type["median"].to_numpy()
yb = np.arange(len(labs))
cols = [ACCENT if t == "HACK" else BL[1] for t in by_type.index]
ax2.barh(yb, meds, color=cols, edgecolor="#fcfcfb", height=0.66)
for y, m, n in zip(yb, meds, by_type["size"]):
    ax2.text(m + 2, y, f"{m:.0f} j  (n={int(n)})", va="center", fontsize=8, color=INK2)
ax2.set_yticks(yb); ax2.set_yticklabels(labs, fontsize=8.7)
ax2.set_xlim(0, meds.max() * 1.32)
ax2.set_xlabel("delai median de declaration (jours)", color=INK2)
ax2.set_title("(b)  Plus furtif, plus tardif : le piratage traine", fontsize=11,
              color=INK, pad=8)
for s in ("top", "right", "left"):
    ax2.spines[s].set_visible(False)
ax2.tick_params(axis="y", length=0)

# (c) part respectant chaque echeance DORA (marches cumulees)
cats = ["<= 24 h", "<= 72 h", "<= 1 mois", "> 1 mois"]
vals = [share[1], share[3] - share[1], share[30] - share[3], 1 - share[30]]
cols3 = [BL[2], BL[1], BL[0], ACCENT]
left = 0.0
for c, v, col in zip(cats, vals, cols3):
    ax3.barh([0], [v * 100], left=left * 100, color=col, edgecolor="#fcfcfb", height=0.5)
    if v > 0.02:
        ax3.text((left + v / 2) * 100, 0, f"{c}\n{v:.0%}", ha="center", va="center",
                 fontsize=8.3, color="#ffffff" if col in (BL[2], ACCENT) else INK)
    left += v
ax3.set_xlim(0, 100); ax3.set_ylim(-0.8, 0.8)
ax3.set_yticks([])
ax3.set_xlabel("part des incidents (%)", color=INK2)
ax3.set_title("(c)  Conformite aux echeances de notification DORA", fontsize=11,
              color=INK, pad=8)
ax3.text(0.5, -0.62, "la quasi-totalite tombe hors des delais reglementaires :\n"
         "le KPI de delai est le plus directement opposable au texte",
         transform=ax3.transAxes, ha="center", fontsize=8.3, color=INK2, style="italic")
for s in ("top", "right", "left", "bottom"):
    ax3.spines[s].set_visible(False)

fig.suptitle("Q : cas d'usage P2, le delai de notification comme KPI central",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Q_cas_usage_p2.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
