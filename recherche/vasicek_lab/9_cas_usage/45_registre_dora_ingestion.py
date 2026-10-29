#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
45 : squelette d'ingestion du registre d'information DORA, pret pour la livraison d'un registre reel.

Le registre d'externalisation DORA est normalise par le Reglement d'execution (UE) 2024/2956
(ITS des ESAs) : une quinzaine de templates lies (B_01..B_99). Ce script pre-cartographie les
templates qui portent le signal du SOUS-PROCESS P4, calcule les KPI correspondants, et les mappe
aux parametres du modele (gamma, phi_cs, ROOT[4]). Il tourne ici sur un registre SYNTHETIQUE
(clairement etiquete tel quel) pour demontrer le pipeline ; il suffira de pointer READER sur le
classeur reel d'une entite.

TEMPLATES ITS UTILES POUR P4 (les autres portent l'identification et le contractuel) :
  B_05.01  prestataires TIC tiers          -> identite, LEI, pays, type de prestataire
  B_05.02  chaine d'approvisionnement TIC   -> sous-traitance en chaine (rang)
  B_06.01  fonctions                        -> fonction metier, criticite/importance
  B_07.01  services TIC (evaluation)        -> service, fonction supportee, SUBSTITUABILITE

KPI DU SOUS-PROCESS P4 (ce que le registre permet, que l'open data ne permettait pas) :
  (K1) concentration par prestataire  = part des fonctions critiques portee par le top-k    -> gamma
  (K2) largeur de co-declenchement    = nb de fonctions critiques par prestataire partage   -> phi_cs
  (K3) substituabilite                = part des fonctions critiques sans substitut aise     -> severite bornee
  (K4) profondeur de sous-traitance   = rang max de la chaine (concentration cachee)         -> queue

NB : les codes de colonnes exacts suivent l'ITS ; a confirmer contre le fichier livre (la
cartographie COLS ci-dessous est le seul point a ajuster).
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WID = 82

# --- cartographie ITS -> noms internes (SEUL point a ajuster contre le fichier d'une entite) -----
COLS = {
    "provider": "b_05.01_provider_lei",       # identifiant du prestataire (LEI)
    "provider_name": "b_05.01_provider_name",
    "service": "b_07.01_ict_service_type",
    "function": "b_06.01_function_name",
    "critical": "b_06.01_criticality",         # 'critique'/'importante'/'autre'
    "substitutable": "b_07.01_substitutability",  # 'facile'/'difficile'/'non substituable'
    "chain_rank": "b_05.02_supply_chain_rank",    # 1 = direct, 2+ = sous-traitance
}
REGISTER_PATH = os.environ.get("DORA_REGISTER")   # pointer ici le fichier reel d'une entite


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# =====================================================================================
# LECTEUR : fichier reel si fourni, sinon registre SYNTHETIQUE de demonstration
# =====================================================================================
def load_register():
    if REGISTER_PATH and os.path.exists(REGISTER_PATH):
        df = pd.read_excel(REGISTER_PATH)
        df = df.rename(columns={v: k for k, v in COLS.items()})
        return df, "REEL (registre d'une entite)"
    return synthetic_register(), "SYNTHETIQUE (gabarit de demonstration)"


def synthetic_register(n_arr=320, n_prov=60, seed=20260727):
    """Registre synthetique respectant la CONCENTRATION documentee (ECB : ~30% sur 10 fourn.)."""
    rng = np.random.default_rng(seed)
    # concentration : poids de choix des prestataires en loi de puissance (peu portent beaucoup)
    w = 1.0 / np.arange(1, n_prov + 1) ** 0.75
    w = w / w.sum()
    prov = rng.choice(np.arange(n_prov), size=n_arr, p=w)
    crit = rng.choice(["critique", "importante", "autre"], size=n_arr, p=[0.30, 0.35, 0.35])
    # les fonctions critiques sont plus souvent difficilement substituables
    subs = np.where(crit == "critique",
                    rng.choice(["facile", "difficile", "non substituable"], n_arr, p=[0.25, 0.45, 0.30]),
                    rng.choice(["facile", "difficile", "non substituable"], n_arr, p=[0.60, 0.30, 0.10]))
    rank = rng.choice([1, 2, 3], size=n_arr, p=[0.70, 0.22, 0.08])
    return pd.DataFrame({
        "provider": [f"LEI{p:03d}" for p in prov],
        "provider_name": [f"Prestataire {p}" for p in prov],
        "service": rng.choice(["cloud", "paiement", "data", "cybersec", "telecom"], n_arr),
        "function": [f"F{i}" for i in range(n_arr)],
        "critical": crit, "substitutable": subs, "chain_rank": rank,
    })


df, source = load_register()
print(f"Registre charge : {source} --- {len(df)} accords, {df['provider'].nunique()} prestataires.")

crit_mask = df["critical"].isin(["critique", "importante"])
dc = df[crit_mask]

# =====================================================================================
titre("K1 - Concentration par prestataire (fonctions critiques) -> ancre gamma")
# =====================================================================================
by_prov = dc.groupby("provider").size().sort_values(ascending=False)
share = by_prov.cumsum() / by_prov.sum()
top10 = share.iloc[min(9, len(share) - 1)]
print(f"  Fonctions critiques/importantes : {len(dc)} sur {df['provider'].nunique()} prestataires.")
print(f"  Les 10 premiers prestataires portent {100*top10:.0f} % des fonctions critiques "
      f"(repere ECB : >30 %).")
print(f"  => lecture directe du parametre d'accumulation gamma (concentration reelle du livre).")

# =====================================================================================
titre("K2 - Largeur de co-declenchement : fonctions critiques par prestataire -> phi_cs")
# =====================================================================================
breadth = by_prov.mean()
top1 = int(by_prov.iloc[0])
print(f"  Un prestataire partage porte en moyenne {breadth:.1f} fonctions critiques ; le plus")
print(f"  concentre en porte {top1}. Sa defaillance emporte donc plusieurs fonctions d'un coup :")
print(f"  c'est la largeur du choc commun phi_cs, ici MESUREE et non plus bornee a l'aveugle.")

# =====================================================================================
titre("K3 - Substituabilite des fonctions critiques -> severite de l'accumulation")
# =====================================================================================
sub = dc["substitutable"].value_counts(normalize=True)
hard = sub.get("difficile", 0) + sub.get("non substituable", 0)
print(f"  Parmi les fonctions critiques : {100*sub.get('non substituable',0):.0f} % non "
      f"substituables, {100*hard:.0f} % difficilement ou pas substituables.")
print(f"  => la part non substituable borne par le haut la severite de l'accumulation (un choc")
print("     sur un prestataire non substituable ne se resorbe pas par bascule).")

# =====================================================================================
titre("K4 - Profondeur de sous-traitance en chaine -> concentration cachee")
# =====================================================================================
depth = df["chain_rank"].value_counts(normalize=True).sort_index()
deep = df[df["chain_rank"] >= 2]
print(f"  {100*(df['chain_rank']>=2).mean():.0f} % des accords passent par de la sous-traitance "
      f"(rang >= 2), {100*(df['chain_rank']>=3).mean():.0f} % au rang >= 3.")
print("  => concentration CACHEE : deux prestataires directs distincts peuvent partager un meme")
print("     sous-traitant en aval, ce que seul le registre (B_05.02) revele.")

# =====================================================================================
titre("VERDICT (pipeline pret)")
# =====================================================================================
print("  Le registre transforme les parametres P4 BORNES du modele en quantites MESUREES :")
print(f"    gamma      <- concentration top-10 = {100*top10:.0f} % (K1)")
print(f"    phi_cs     <- {breadth:.1f} fonctions critiques par prestataire (K2)")
print(f"    severite   <- {100*hard:.0f} % difficilement substituables (K3)")
print(f"    queue      <- {100*(df['chain_rank']>=2).mean():.0f} % de sous-traitance en chaine (K4)")
print("  Il suffit de definir la variable d'environnement DORA_REGISTER vers le fichier d'une entite")
print("  et d'ajuster la cartographie COLS ; tout le reste tourne a l'identique.")

# =====================================================================================
# figure Z15 (etiquetee SYNTHETIQUE si pas de fichier reel)
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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.9))

# (a) K1 courbe de concentration
xf = np.arange(1, len(share) + 1)
ax1.plot(xf, share.values, color=BLUE, lw=2.2, marker="o", ms=3)
ax1.axvline(10, color=MUTED, ls=":", lw=1)
ax1.axhline(top10, color=ACCENT, ls="--", lw=1.4)
ax1.text(10.5, 0.1, "top 10", fontsize=8, color=MUTED)
ax1.text(len(share) * 0.45, top10 + 0.03, f"{100*top10:.0f} % (repère ECB >30 %)",
         fontsize=8.5, color=ACCENT)
ax1.set_xlabel("prestataires (classés)", color=INK2)
ax1.set_ylabel("part cumulée des fonctions critiques", color=INK2, fontsize=9)
ax1.set_ylim(0, 1.02)
ax1.set_title("(a)  K1 concentration → γ", fontsize=10.5, color=INK, pad=8)

# (b) K3 substituabilite
order = ["facile", "difficile", "non substituable"]
vals = [sub.get(k, 0) for k in order]
cols = [GREEN, "#e0a458", ACCENT]
ax2.bar(range(3), vals, color=cols, alpha=0.9, width=0.6)
ax2.set_xticks(range(3)); ax2.set_xticklabels(["facile", "difficile", "non\nsubstituable"], fontsize=8.5)
for i, v in enumerate(vals):
    ax2.text(i, v + 0.01, f"{100*v:.0f} %", ha="center", fontsize=9, color=INK2)
ax2.set_ylabel("part des fonctions critiques", color=INK2, fontsize=9)
ax2.set_title("(b)  K3 substituabilité → sévérité", fontsize=10.5, color=INK, pad=8)

# (c) K4 profondeur de sous-traitance
dvals = [depth.get(r, 0) for r in (1, 2, 3)]
ax3.bar(range(3), dvals, color=[MUTED, BLUE, "#204993"], alpha=0.9, width=0.6)
ax3.set_xticks(range(3)); ax3.set_xticklabels(["direct\n(rang 1)", "rang 2", "rang 3"], fontsize=8.5)
for i, v in enumerate(dvals):
    ax3.text(i, v + 0.01, f"{100*v:.0f} %", ha="center", fontsize=9, color=INK2)
ax3.set_ylabel("part des accords", color=INK2, fontsize=9)
ax3.set_title("(c)  K4 sous-traitance en chaîne → queue", fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2, ax3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

tag = "" if REGISTER_PATH and os.path.exists(REGISTER_PATH) else "  [DONNÉES SYNTHÉTIQUES : gabarit, à remplacer par le registre d'une entite]"
fig.suptitle("Z15 : ingestion du registre DORA, les paramètres P4 bornés deviennent mesurés" + tag,
             fontsize=11.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z15_registre_dora_ingestion.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
