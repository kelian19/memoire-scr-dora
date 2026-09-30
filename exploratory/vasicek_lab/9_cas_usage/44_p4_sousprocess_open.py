#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
44 : cadrage du sous-process P4 (risque tiers) sur donnee OUVERTE, avant le registre d'une entite.

Une entite devrait fournir le registre d'externalisation DORA (structure connue : Commission Implementing
Regulation (EU) 2024/2956, templates B_01.01..B_99.01, 80+ champs, chaine de sous-traitance,
criticite, substituabilite). En attendant, on regarde ce que la donnee OUVERTE dit deja du
sous-process tiers, et ce qu'on peut exploiter des maintenant.

DEUX CONSTATS, OPPOSES ET COMPLEMENTAIRES.

(1) LA TAXONOMIE FINE DU TIERS EST RARE PARTOUT (mur de sparsite, meme lecon que le VCDB, 28).
    - OpRisk finance, sous-categorie TIC "Vendors & Suppliers" (calcule ici) : ~20 incidents,
      ~1 % du TIC finance : sous le seuil d'estimation.
    - VCDB finance, actor.Partner : 56 incidents (VERIS SOUS-code le tiers), 0 deux crans plus
      bas (script 28).
    => le QUI (quel sous-traitant, quelle fonction critique, quelle substituabilite) n'est PAS
       resoluble sur donnee ouverte : il faut le registre DORA d'une entite. C'est la these
       "donnee manquante = specification de reporting", declinee au sous-process P4.

(2) L'ORIGINATION ET LA CONCENTRATION TIERS SONT, ELLES, ANCREES SUR DONNEE UE OFFICIELLE.
    - Origination : ESAs, 1er rapport d'incidents DORA (3 juin 2026) : 29 % des 3 383 incidents
      ICT MAJEURS de 2025 proviennent d'une defaillance de TIERS. Validation externe, sur donnee
      reglementaire UE, de P4 comme amorce majeure (classeur : ROOT[4]=0,90).
    - Concentration (exposition) : ECB 2024 (registres 2023) : >30 % du budget d'externalisation
      des grandes banques UE sur 10 fournisseurs, 50 % du budget critique sur 30. 19 fournisseurs
      ICT critiques designes par les ESAs (nov. 2025). Ancre le parametre d'accumulation gamma
      (0,68) du modele sur un chiffre officiel, la ou il n'etait qu'un proxy cloud.
    - Concentration (sinistralite) : Gini 0,69 des comptes journaliers (deja etabli, script 08g),
      1,7 % des jours portant 20 % des incidents (cause commune tierce, type MOVEit).

Sortie : diagnostics + figure Z14_p4_sousprocess_open.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WID = 82

# --- ancres et resultats etablis : source unique dans `resultats_partages` --------------------
import resultats_partages as rp                                 # noqa: E402

ESAS_TP_ORIGIN = rp.ESAS_PART_TIERCE      # ESAs, 1er rapport incidents DORA (3 juin 2026)
ECB_CONC = rp.ECB_CONCENTRATION           # BCE 2024 (registres 2023)
N_CTPP = rp.ESAS_N_CTPP                   # prestataires TIC critiques designes (nov. 2025)
GAMMA_MODELE = rp.GAMMA_ACCUMULATION      # parametre d'accumulation (scripts 38/42)
ROOT_P4 = rp.ROOT_P4                      # propension d'amorce de P4 (classeur qualitatif)
GINI_08G = rp.GINI_JOURNALIER             # Gini des comptes journaliers, finance (08g)
SHARE_08G = rp.PART_JOURS_CHARGES         # 1,7 % des jours portent 20 % des incidents (08g)
SEUIL_CELL = rp.SEUIL_ESTIMATION          # volume minimal par cellule pour estimer
VCDB_PARTNER = rp.VCDB_PARTNER_FINANCE    # actor.Partner finance (script 28)


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# =====================================================================================
titre("1. Le sous-process tiers : la taxonomie fine est rare sur donnee ouverte (calcul)")
# =====================================================================================
vs_n = vs_share = None
try:
    from src.severity.oprisk_analysis import load_clean, filter_finance
    d = load_clean(os.path.join(REPO, "data", "raw", "SAS_OpRisk_Global_Data_June_2026.xlsx"))
    fs = filter_finance(d)
    ict = ["Systems Security", "Systems", "Vendors & Suppliers",
           "Monitoring and Reporting", "Unauthorized Activity"]
    g = fs[fs["Sub Risk Category"].isin(ict)]
    vs = g[g["Sub Risk Category"] == "Vendors & Suppliers"]
    vs_n = int(len(vs))
    vs_share = len(vs) / len(g)
    vs_loss_share = vs["loss"].sum() / g["loss"].sum()
    print(f"  OpRisk 'Vendors & Suppliers' (finance) : n = {vs_n}  "
          f"({100*vs_share:.1f} % du TIC finance en compte, {100*vs_loss_share:.1f} % de la perte).")
except Exception as e:
    vs_n, vs_share = 20, 0.010
    print(f"  OpRisk indisponible ({type(e).__name__}) ; valeur de sonde n=20 (~1 %).")
print(f"  VCDB actor.Partner (finance) : n = {VCDB_PARTNER}, 0 deux crans plus bas (script 28).")
print(f"  Seuil d'estimation : ~{SEUIL_CELL} obs/cellule.")
print("  => la TAXONOMIE fine du tiers n'est PAS resoluble sur open data : il faut le registre")
print("     DORA d'une entite (ITS 2024/2956). Meme mur de sparsite que le VCDB.")

# =====================================================================================
titre("2. Concentration de la sinistralite tiers (deja etablie, script 08g)")
# =====================================================================================
print(f"  Data Breach Chronology, secteur financier : Gini des comptes journaliers = {GINI_08G},")
print(f"  et {100*SHARE_08G[0]:.1f} % des jours portent {100*SHARE_08G[1]:.0f} % des incidents.")
print("  C'est le proxy de cause commune tierce : un prestataire partage fait tomber N entites")
print("  le meme jour (type MOVEit). Structure SYMETRIQUE, donc partie identifiee du modele.")

# =====================================================================================
titre("3. Origination et concentration tiers : ancrage sur donnee europeenne officielle")
# =====================================================================================
# EFFECTIF SANS SEPARATEUR. Ecrit « 3 383 », le harnais y lisait deux nombres, 3 et 383, et le
# « 3 383 » du memoire ressortait non confirme alors que cette ligne l'imprime.
print(f"  ORIGINATION. ESAs, 1er rapport incidents DORA (3 juin 2026) : {100*ESAS_TP_ORIGIN:.0f} % "
      f"des 3383 incidents ICT")
print(f"    MAJEURS de 2025 d'origine TIERCE => validation externe de P4 amorce (ROOT[4]={ROOT_P4}).")
print(f"  CONCENTRATION (exposition). ECB 2024 (registres 2023) :")
for nprov, share in ECB_CONC:
    print(f"    {nprov:>2} fournisseurs = {100*share:.0f} % du budget d'externalisation.")
print(f"    {N_CTPP} fournisseurs ICT critiques designes (ESAs, nov. 2025 : AWS, Microsoft, ...).")
print(f"    => ancre le parametre d'accumulation gamma = {GAMMA_MODELE} sur un chiffre officiel.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. La TAXONOMIE fine du sous-process tiers est rare sur open data (OpRisk V&S ~ 20,")
print("     VCDB Partner 56 -> 0) : il faut le registre DORA d'une entite (these du memoire, au")
print("     niveau sous-process P4).")
print(f"  2. Mais l'ORIGINATION ({100*ESAS_TP_ORIGIN:.0f} %, ESAs) et la CONCENTRATION (>30 % sur 10 "
      f"fourn., ECB ;")
print(f"     Gini {GINI_08G}, notre donnee) sont ancrees : P4 amorce majeure et gamma valides.")
print("  3. On avance donc sans registre d'entite sur la partie IDENTIFIEE (origination, accumulation")
print("     symetrique) et on reserve au registre la partie non resoluble (taxonomie fine).")

# =====================================================================================
# figure Z14
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 11,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
})
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Trois panneaux traces
# sur seize pouces puis imprimes sur 7,27 donnent 1,6 pouce par panneau et
# des etiquettes sous 5 points. Le rapport largeur sur hauteur est conserve.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.66, 3.17))

# (a) mur de sparsite de la taxonomie tiers
labs = ["OpRisk\nVendors &\nSuppliers", "VCDB\nactor.\nPartner"]
vals = [vs_n, VCDB_PARTNER]
ax1.bar(labs, vals, color=[MUTED, BLUE], alpha=0.9, width=0.55)
ax1.axhline(SEUIL_CELL, color=ACCENT, ls="--", lw=1.6)
ax1.text(1.45, SEUIL_CELL - 6.0, f"seuil d'estimation ~{SEUIL_CELL}", fontsize=8.5,
         color=ACCENT, ha="right", va="center")
for i, v in enumerate(vals):
    ax1.text(i, v + 1.2, str(v), ha="center", fontsize=9.5, color=INK2)
ax1.text(1.0, SEUIL_CELL + 9.0, "→ 0 au sous-process\n(deux crans plus bas)",
         fontsize=8, color=ACCENT, ha="center", va="center")
ax1.set_ylabel("incidents tiers (finance, open data)", color=INK2, fontsize=9)
ax1.set_ylim(0, 70)
ax1.set_title("(a)  Taxonomie fine du tiers : rare.\nLe registre DORA reste requis",
              fontsize=10.5, color=INK, pad=8)

# (b) concentration : exposition (ECB) et sinistralite (notre donnee, 08g)
bars = [("10 fourn.\n(ECB)", ECB_CONC[0][1], ACCENT),
        ("30 fourn.\n(ECB)", ECB_CONC[1][1], ACCENT),
        (f"1,7 % jours\n(observé)", SHARE_08G[1], BLUE)]
for i, (lab, v, c) in enumerate(bars):
    ax2.bar(i, v, color=c, alpha=0.9, width=0.6)
    ax2.text(i, v + 0.015, f"{100*v:.0f} %", ha="center", fontsize=9, color=INK2)
ax2.set_xticks(range(len(bars))); ax2.set_xticklabels([b[0] for b in bars], fontsize=8.5)
ax2.set_ylim(0, 0.62)
ax2.set_ylabel("part concentrée", color=INK2, fontsize=9)
ax2.text(0.5, 0.55, "exposition", ha="center", fontsize=8, color=ACCENT, style="italic")
ax2.text(2.0, 0.30, "sinistralité", ha="center", fontsize=8, color=BLUE, style="italic")
ax2.set_title("(b)  Concentration tiers : peu de fournisseurs\net de jours portent l'essentiel",
              fontsize=10.5, color=INK, pad=8)

# (c) ancrage des parametres P4 sur donnee europeenne
ax3.axis("off"); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.set_title("(c)  Les paramètres P4, ancrés sur\ndonnée UE officielle", fontsize=10.5,
              color=INK, pad=8)
rows = [
    ("Origination tiers", f"{100*ESAS_TP_ORIGIN:.0f} % des incidents majeurs", "ESAs, rapport DORA 2026"),
    ("  → P4 amorce (modèle)", f"ROOT[4] = {ROOT_P4}", "validé qualitativement"),
    ("Concentration budget", ">30 % sur 10 fournisseurs", "ECB 2024"),
    ("  → accumulation γ (modèle)", f"γ = {GAMMA_MODELE}", "ancré, non plus proxy"),
    ("Fournisseurs critiques", f"{N_CTPP} désignés (CTPP)", "ESAs, nov. 2025"),
]
y = 0.94
for lab, val, src_ in rows:
    is_arrow = lab.strip().startswith("→")
    x0 = 0.08 if is_arrow else 0.02
    ax3.text(x0, y, lab.strip(), fontsize=8.5, color=GREEN if is_arrow else INK,
             fontweight="normal" if is_arrow else "bold", va="center")
    ax3.text(x0 + 0.03, y - 0.075, f"{val}  ({src_})", fontsize=7.5, color=INK2,
             va="center", style="italic")
    y -= 0.195

fig.tight_layout(w_pad=1.8)
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z14_p4_sousprocess_open.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
