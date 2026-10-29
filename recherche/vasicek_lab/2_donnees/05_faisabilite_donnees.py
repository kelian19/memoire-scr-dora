#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05 : Peut-on calibrer W sur les donnees disponibles ? Non. Voici les chiffres.

Ce script reproduit CHAQUE nombre de la section "Peut-on calibrer W" de la note. Il ne
produit pas un modele : il produit un VERDICT DE FAISABILITE, et une specification de la
donnee manquante.

Deux sources sont examinees. Elles echouent pour des raisons OPPOSEES, et c'est cette
opposition qui fait la valeur du chapitre.

  Source 1 : Data Breach Chronology (notifications d'Etats americains)
             -> manque de VOLUME et de TAXONOMIE, et la date piege des deux cotes.

  Source 2 : SAS OpRisk Global Data (pertes operationnelles, dollars, Bale)
             -> le VOLUME est la (4329 transitions), mais le pas temporel est ANNUEL
                et l'asymetrie tombe SOUS le placebo. Aucune direction n'est identifiable.

Ce que la source 2 permet en revanche : ancrer l'indice de queue xi sur des pertes en
DOLLARS (xi ~ 0,9), valeur desormais utilisee par les scripts 02 et 04.

CONCLUSION OPERATIONNELLE. Pour identifier W, un registre d'incidents doit horodater la
SURVENANCE au mois ou a la semaine, et categoriser par DOMAINE DE CONTROLE (pilier) et non
par vecteur d'attaque. Exigence chiffree, issue d'une analyse de puissance et d'un placebo.

Donnees attendues dans data/raw/ (non versionnees : voir .gitignore).
Sortie : diagnostics + figure M_faisabilite.png
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20260709)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "data", "raw"))
BREACH = os.path.join(RAW, "Data_Breach_Chronology.csv")
OPRISK = os.path.join(RAW, "SAS_OpRisk_Global_Data_June_2026.xlsx")

for p in (BREACH, OPRISK):
    if not os.path.exists(p):
        sys.exit(f"donnee absente : {p}\n(les sources brutes ne sont pas versionnees)")

W = 74  # largeur des separateurs


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def transitions(df, ent, cat, ycol, lag=1):
    """Compte les transitions k(t-lag) -> j(t) sur periodes CONSECUTIVES, k != j.

    Renvoie (matrice, categories). Ligne j = recoit, colonne k = emet.
    """
    cats = sorted(df[cat].dropna().unique())
    ix = {c: i for i, c in enumerate(cats)}
    M = np.zeros((len(cats), len(cats)))
    sets = df.groupby([ent, ycol])[cat].apply(set)
    for org, sub in sets.groupby(level=0):
        per = sorted(sub.index.get_level_values(1))
        for a, b in zip(per, per[1:]):
            if _period_gap(a, b) != lag:
                continue
            for k in sub.loc[(org, a)]:
                for j in sub.loc[(org, b)]:
                    if k != j:
                        M[ix[j], ix[k]] += 1
    return M, cats


def _period_gap(a, b):
    """Ecart entre deux periodes, que ce soient des Period pandas ou des entiers."""
    return (b - a).n if hasattr(b - a, "n") else int(b - a)


def asym(M):
    return float(np.abs(M - M.T).sum() / 2)


# =====================================================================================
titre("SOURCE 1 : Data Breach Chronology")
# =====================================================================================
d1 = pd.read_csv(BREACH, sep="|", quotechar='"', low_memory=False,
                 usecols=["normalized_org_name", "breach_date", "reported_date",
                          "breach_type", "organization_type"])
print(f"incidents : {len(d1)} | organisations : {d1.normalized_org_name.nunique()} "
      f"| types : {d1.breach_type.nunique()}")
print(f"part de la categorie UNKN : {(d1.breach_type == 'UNKN').mean():.1%}  "
      "<- non informative, et non attribuable a un pilier")

# --- dates : la survenance n'est exploitable qu'a moitie, et pas au hasard
s = d1.breach_date.astype(str).str.strip()
d1["bd"] = pd.to_datetime(s, errors="coerce").fillna(
    pd.to_datetime(s.where(s.str.fullmatch(r"\d{4}-\d{2}")) + "-15", errors="coerce"))
d1["rd"] = pd.to_datetime(d1.reported_date, errors="coerce")
print(f"\ndate de survenance exploitable (jour ou mois) : {d1.bd.notna().mean():.1%}")

miss = d1.assign(m=d1.bd.isna()).groupby("breach_type").m.mean().sort_values()
print("\nmanquant de la date de survenance, PAR TYPE (le manquant n'est pas aleatoire) :")
for t, v in miss.items():
    print(f"   {t:<6} {v:6.1%}")
print(f"   => amplitude {miss.min():.1%} -> {miss.max():.1%}. Ordonner par survenance "
      f"ELIMINE {miss.idxmax()} du panel.")

lag = (d1.rd - d1.bd).dt.days
lg = d1.assign(l=lag).dropna(subset=["l"]).groupby("breach_type").l.median().sort_values()
print("\ndelai MEDIAN de declaration, PAR TYPE (jours) :")
for t, v in lg.items():
    print(f"   {t:<6} {v:6.0f}")
print(f"   => de {lg.min():.0f} a {lg.max():.0f} jours, facteur {lg.max()/lg.min():.1f}. "
      "Ordonner par declaration MESURE LA DETECTION,")
print("      pas la contagion.")

# --- puissance : combien de transitions informatives ?
USABLE = ["HACK", "DISC", "INSD", "PORT", "PHYS"]
p1 = d1.dropna(subset=["bd"]).copy()
p1["q"] = p1.bd.dt.to_period("Q")
p1 = p1[(p1.q >= pd.Period("2010Q1")) & (p1.q <= pd.Period("2025Q1"))]
pu = p1[p1.breach_type.isin(USABLE)]
M1, C1 = transitions(pu, "normalized_org_name", "breach_type", "q")
n_cells = len(C1) * (len(C1) - 1)
print(f"\ntransitions k(t-1)->j(t) sur trimestres consecutifs, types informatifs : "
      f"{int(M1.sum())}")
print(f"   {n_cells} cellules hors-diagonale -> {M1.sum()/n_cells:.1f} obs/cellule")
bsf = pu[pu.organization_type == "BSF"]
M1f, _ = transitions(bsf, "normalized_org_name", "breach_type", "q")
print(f"   secteur financier (BSF) seul : {int(M1f.sum())} transitions")

# --- le piege de la date, a SELECTION CONSTANTE
both = d1.dropna(subset=["bd", "rd"])
both = both[both.breach_type.isin(USABLE)].copy()
res = {}
for name, col in [("survenance", "bd"), ("declaration", "rd")]:
    t = both.copy()
    t["q"] = t[col].dt.to_period("Q")
    t = t[(t.q >= pd.Period("2010Q1")) & (t.q <= pd.Period("2025Q1"))]
    res[name], _ = transitions(t, "normalized_org_name", "breach_type", "q")
Ab, Ar = res["survenance"] - res["survenance"].T, res["declaration"] - res["declaration"].T
iu = np.triu_indices(len(USABLE), 1)
nz = (Ab[iu] != 0) | (Ar[iu] != 0)
flip = (np.sign(Ab[iu]) != np.sign(Ar[iu]))[nz].sum()
corr_axes = np.corrcoef(Ab[iu], Ar[iu])[0, 1]
print(f"\nMEMES incidents ({len(both)}), on ne change QUE l'axe temporel :")
print(f"   sens de l'asymetrie inverse sur {flip}/{nz.sum()} paires")
print(f"   correlation des asymetries : {corr_axes:+.3f}")
print("   (avec si peu de transitions, ceci ne distingue pas 'la date detruit le reseau'")
print("    de 'le reseau est du bruit'. Les deux lectures condamnent l'exercice.)")

# =====================================================================================
titre("SOURCE 2 : SAS OpRisk Global Data")
# =====================================================================================
d2 = pd.read_excel(OPRISK, sheet_name="Datasets")
d2["year"] = pd.to_datetime(d2["First Year of Event"], errors="coerce").dt.year
d2 = d2[(d2.year >= 1990) & (d2.year <= 2025)]
print(f"evenements 1990-2025 : {len(d2)}")
print("granularite temporelle de la survenance : L'ANNEE (aucune date plus fine)")

fs_all = d2[d2["Basel Business Line - Level 1"] != "Non-FS"].copy()
# Deduplication pour les TRANSITIONS seulement : un meme evenement eclate en plusieurs
# pertes gonflerait artificiellement la co-occurrence. Elle NE DOIT PAS servir a estimer
# la severite, ou chaque perte est une observation legitime.
fs = fs_all.drop_duplicates(subset=["Firm Name", "year", "Event Risk Category"])
print(f"secteur financier : {len(fs_all)} pertes -> {len(fs)} couples "
      "(firme, annee, categorie) apres deduplication")
print("   la deduplication sert aux TRANSITIONS ; xi sera estime sur les pertes brutes")

M2, C2 = transitions(fs, "Firm Name", "Event Risk Category", "year")
cells2 = len(C2) * (len(C2) - 1)
print(f"\ntransitions annuelles consecutives : {int(M2.sum())} sur {len(C2)} categories Bale")
print(f"   {cells2} cellules -> {M2.sum()/cells2:.0f} obs/cellule  "
      "<- LE VOLUME EST ENFIN SUFFISANT")

# --- le placebo : permuter les annees DANS chaque firme
titre("Le placebo : detruire l'anteriorite, garder la co-occurrence")


def placebo_asym(df, reps=30):
    sets = df.groupby(["Firm Name", "year"])["Event Risk Category"].apply(set)
    grouped = []
    for org, sub in sets.groupby(level=0):
        per = sorted(sub.index.get_level_values(1))
        grouped.append((per, [sub.loc[(org, p)] for p in per]))   # ordres alignes
    ix = {c: i for i, c in enumerate(C2)}
    out = []
    for _ in range(reps):
        M = np.zeros((len(C2), len(C2)))
        for per, vals in grouped:
            v = list(RNG.permutation(np.array(vals, dtype=object)))
            for (a, va), (b, vb) in zip(zip(per, v), zip(per[1:], v[1:])):
                if b - a == 1:
                    for k in va:
                        for j in vb:
                            if k != j:
                                M[ix[j], ix[k]] += 1
        out.append(asym(M))
    return np.array(out)


a_real = asym(M2)
a_pl = placebo_asym(fs)
z = (a_real - a_pl.mean()) / a_pl.std()
print(f"asymetrie reelle          = {a_real:.0f}")
print(f"asymetrie sous placebo    = {a_pl.mean():.0f} +- {a_pl.std():.0f}")
print(f"z-score du signal reel    = {z:+.1f}")
print("=> l'asymetrie observee est SOUS le bruit. Aucun signal directionnel.")

# --- test binomial par paire
print("\nTest binomial par paire, H0 : p = 0,5 (aucune direction privilegiee)")
pv = []
for a in range(len(C2)):
    for b in range(a + 1, len(C2)):
        nab, nba = int(M2[b, a]), int(M2[a, b])
        if nab + nba < 10:
            continue
        pv.append(stats.binomtest(nab, nab + nba, 0.5).pvalue)
pv = np.array(sorted(pv))
print(f"   {len(pv)} paires testees | p-value minimale = {pv.min():.3f}")
print(f"   significatives a 5% SANS correction : {(pv < 0.05).sum()}")
print(f"   significatives apres Bonferroni ({0.05/len(pv):.4f}) : {(pv < 0.05/len(pv)).sum()}")
print("=> la matrice de transition est SYMETRIQUE. Le pas annuel a lessive la direction.")

print("\nAVERTISSEMENT. Le test du 'temps inverse' est DEGENERE sur un comptage brut :")
print("renverser la chronologie donne M_inv = M^T par construction, donc une correlation")
print("des asymetries de -1 quel que soit le contenu. Il ne prouve rien ici. Seul le")
print("placebo par permutation est valide sur un comptage.")

# --- sous-categories TIC : le mur de la taxonomie
ICT = ["Systems Security", "Systems", "Vendors & Suppliers",
       "Monitoring and Reporting", "Unauthorized Activity"]
sub = d2[d2["Sub Risk Category"].isin(ICT)]
Mi, Ci = transitions(sub.drop_duplicates(subset=["Firm Name", "year", "Sub Risk Category"]),
                     "Firm Name", "Sub Risk Category", "year")
print("\nSous-categories les plus proches du TIC :")
for c in ICT:
    print(f"   {c:<26} {(d2['Sub Risk Category'] == c).sum():>6} evenements")
print(f"   -> {int(Mi.sum())} transitions seulement : meme mur que la source 1.")

# =====================================================================================
titre("Ce que la source 2 permet : ancrer l'indice de queue xi")
# =====================================================================================
L = pd.to_numeric(fs_all["Loss Amount ($M)"], errors="coerce").dropna()   # PAS fs
L = L[L > 0]
print(f"pertes en dollars : n={len(L)} | plancher de collecte {L.min():.1f} M$ "
      "(troncature a gauche, comme la sous-declaration)")
big = L.nlargest(1)
print(f"\nplus grosse valeur : {big.iloc[0]:,.0f} M$ -- l'erreur de saisie Citigroup 2024")
print("(81 000 milliards credites au lieu de 280 dollars, operation ANNULEE : pas une perte)")

print(f"\n{'seuil (M$)':>12}{'n_exc':>8}{'xi brut':>10}{'xi nettoye':>13}")
xis = []
Lc = L[L < 1e6].values          # retrait des valeurs aberrantes
for q in [0.90, 0.95, 0.99]:
    u = np.quantile(Lc, q)
    xr = stats.genpareto.fit(L.values[L.values > u] - u, floc=0)[0]
    xc = stats.genpareto.fit(Lc[Lc > u] - u, floc=0)[0]
    xis.append((u, (Lc > u).sum(), xr, xc))
    print(f"{u:>12.1f}{(Lc>u).sum():>8}{xr:>10.3f}{xc:>13.3f}")
print("=> un SEUL point deplace xi de 0,2 au seuil q99. Nettoyage et analyse de")
print("   sensibilite obligatoires. Valeur retenue : xi ~ 0,90 (scripts 02 et 04).")

print("\nxi par categorie Bale (seuil q75 propre a chaque categorie) :")
xi_cat = {}
for c, g in fs_all.groupby("Event Risk Category"):
    v = pd.to_numeric(g["Loss Amount ($M)"], errors="coerce").dropna()
    v = v[(v > 0) & (v < 1e6)].values
    if len(v) < 150:
        continue
    u = np.quantile(v, 0.75)
    xi_cat[c] = stats.genpareto.fit(v[v > u] - u, floc=0)[0]
    print(f"   {c[:44]:<46} n={len(v):>5}  xi = {xi_cat[c]:.2f}")
print("=> l'heterogeneite de queue entre categories appuie un xi_j PAR PILIER.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("W n'est calibrable sur AUCUNE des deux sources, pour des raisons OPPOSEES :")
print(f"  source 1 : {int(M1.sum())} transitions, taxonomie de vecteurs d'attaque,")
print("             date piegee des deux cotes  -> manque de VOLUME et de TAXONOMIE")
print(f"  source 2 : {int(M2.sum())} transitions (volume suffisant), mais z = {z:+.1f}")
print("             contre le placebo           -> manque de RESOLUTION TEMPORELLE")
print("\nSPECIFICATION DE LA DONNEE MANQUANTE : un registre d'incidents doit horodater la")
print("SURVENANCE au mois ou a la semaine, et categoriser par DOMAINE DE CONTROLE.")
print("Sans quoi la structure dirigee restera inidentifiable, QUEL QUE SOIT le volume.")

# =====================================================================================
# figure M
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2, MUTED, GRID = "#1b1e30", "#223e55", "#595959", "#f2f2f2"
ACCENT = "#a6002e"
BL = ["#7baafd", "#4c79c7", "#204993"]

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Trois panneaux traces
# sur seize pouces puis imprimes sur 7,27 donnent 1,6 pouce par panneau et
# des etiquettes sous 5 points. Le rapport largeur sur hauteur est conserve.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.69, 2.96),
                                    gridspec_kw={"width_ratios": [1, 1.1, 1]})

# (a) puissance : transitions disponibles vs requises
labels = ["Breach\n(trimestre)", "Breach BSF\n(trimestre)", "OpRisk TIC\n(annee)",
          "OpRisk Bale\n(annee)"]
vals = [M1.sum(), M1f.sum(), Mi.sum(), M2.sum()]
cols = [ACCENT, ACCENT, ACCENT, BL[2]]
ax1.bar(range(4), vals, color=cols, edgecolor="#fcfcfb", width=0.66)
ax1.set_yscale("log")
for i, v in enumerate(vals):
    ax1.text(i, v * 1.15, f"{int(v)}", ha="center", fontsize=9, color=INK2)
ax1.axhline(1000, color=INK, ls="--", lw=1.1)
ax1.text(3.45, 1150, "ordre de grandeur\nrequis (fig. K2)", ha="right", fontsize=8, color=INK)
ax1.set_xticks(range(4)); ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("transitions $k(t\\!-\\!1)\\to j(t)$", color=INK2)
ax1.set_ylim(10, 20000)
ax1.set_title("(a)  Le volume : une seule source passe", fontsize=11, color=INK, pad=8)

# (b) le placebo tue la source 2
ax2.hist(a_pl, bins=12, color=BL[0], edgecolor="#fcfcfb", label="placebo (annees permutees)")
ax2.axvline(a_real, color=ACCENT, lw=2.4, label=f"asymetrie reelle = {a_real:.0f}")
ax2.axvline(a_pl.mean(), color=BL[2], lw=1.6, ls="--",
            label=f"moyenne placebo = {a_pl.mean():.0f}")
ax2.set_xlabel("asymetrie $\\|M - M^\\top\\|_1/2$", color=INK2)
ax2.set_ylabel("tirages", color=INK2)
ax2.legend(frameon=False, fontsize=8, loc="upper left")
ax2.set_title(f"(b)  SAS OpRisk : le signal est SOUS le bruit  ($z={z:+.1f}$)",
              fontsize=11, color=INK, pad=8)
ax2.text(0.97, 0.42, "pas annuel\n$\\Rightarrow$ direction lessivee",
         transform=ax2.transAxes, ha="right", fontsize=9, color=ACCENT, style="italic")

# (c) ce qui EST identifiable : xi
cats = sorted(xi_cat, key=xi_cat.get)
ax3.barh(range(len(cats)), [xi_cat[c] for c in cats], color=BL[1],
         edgecolor="#fcfcfb", height=0.62)
ax3.axvline(0.90, color=ACCENT, lw=2, label="$\\xi = 0{,}90$ retenu")
ax3.axvline(0.40, color=MUTED, lw=1.4, ls=":", label="$\\xi = 0{,}40$ postule (abandonne)")
ax3.axvline(1.0, color=INK, lw=1, ls="--")
ax3.text(1.01, len(cats) - 0.35, "$\\xi>1$ :\nmoyenne infinie", fontsize=7.5, color=INK,
         va="top")
ax3.set_yticks(range(len(cats)))
ax3.set_yticklabels([c[:22] for c in cats], fontsize=8)
ax3.set_xlabel("indice de queue $\\hat\\xi$ (pertes en dollars)", color=INK2)
ax3.set_xlim(0, 1.75)
ax3.set_ylim(-0.9, len(cats) - 0.2)
ax3.legend(frameon=False, fontsize=8, loc="lower right")
ax3.set_title("(c)  Ce que la donnee identifie : la queue", fontsize=11, color=INK, pad=8)

fig.suptitle("M : $W$ n'est calibrable sur aucune source ; $\\xi$ l'est",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "M_faisabilite.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
