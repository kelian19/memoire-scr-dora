#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
102. La derive de severite est-elle de l'inflation ?

LA QUESTION. Le script 89 mesure une derive de l'echelle de severite : +13,0 % par an sur la
mediane annuelle des pertes, +4,00 % par an dans la queue. La chaine publiee lit la colonne
« Loss Amount ($M) » par load_clean. La base porte aussi « Current Value of Loss ($M) », le meme
montant reexprime en valeur courante. Si la premiere colonne est nominale, une part de la derive
mesuree est de l'inflation monetaire et non une aggravation du risque, et le memoire doit le dire.

CE QUE LE SCRIPT FAIT, SANS RIEN RECALIBRER.
  1. il etablit la nature des deux colonnes par leur rapport annee par annee ;
  2. il rejoue la tendance de la mediane (section 1bis du 89) sur les deux colonnes ;
  3. il rejoue l'echelle de queue derivee (section 1ter du 89, module derive_severite) sur les
     deux colonnes, au meme seuil publie.
Contrôle : la colonne nominale doit redonner exactement les +13,0 % et +4,00 % du script 89.
Aucun parametre publie ne bouge ; le script ne produit aucune figure.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(LAB, "..", ".."))
for p in (LAB, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import derive_severite as ds  # noqa: E402
from src.severity.oprisk_analysis import (  # noqa: E402
    USD_EUR, filter_cyber, filter_finance, load_clean,
)

AN_DEBUT, AN_FIN = 2004, 2025   # fenetre de la section 1bis du script 89
COL_NOM = "Loss Amount ($M)"
COL_COUR = "Current Value of Loss ($M)"


def titre(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


d = filter_finance(filter_cyber(load_clean(
    os.path.join(REPO, "data", "raw", ds.FICHIER))))
d = d.dropna(subset=["year"]).copy()
d["year"] = d["year"].astype(int)
d["cour"] = pd.to_numeric(d[COL_COUR], errors="coerce")
d["loss_eur"] = d["loss"] * USD_EUR

titre("1. LA NATURE DES DEUX COLONNES")
n_tot = len(d)
n_cour = int(d["cour"].notna().sum())
print(f"Perimetre de la chaine publiee : {n_tot} incidents ; valeur courante renseignee : {n_cour}.")
r = (d["cour"] / d["loss"]).where(d["cour"].notna())
med_r = r.groupby(d["year"]).median()
print()
print("  annee   rapport median valeur courante / montant")
for an in (1990, 2000, 2004, 2010, 2015, 2020, 2023, 2024, 2025):
    if an in med_r.index:
        print(f"  {an}    {med_r.loc[an]:.3f}")
m = (med_r.index >= AN_DEBUT) & (med_r.index <= AN_FIN)
sl_r, _, _, pv_r, _ = stats.linregress(med_r.index[m], np.log(med_r.values[m]))
infl = -sl_r
print()
print(f"  pente du log du rapport sur {AN_DEBUT}-{AN_FIN} : {100*sl_r:+.2f} % par an (p = {pv_r:.2e})")
print(f"  soit un taux d'inflation implicite de {100*infl:.2f} % par an")
print()
print("LECTURE. Le rapport vaut un pour les annees recentes et croit vers le passe : la colonne")
print("utilisee par la chaine publiee est NOMINALE, la valeur courante est corrigee de l'inflation.")

titre("2. LA TENDANCE DE LA MEDIANE, NOMINALE ET REELLE (section 1bis du script 89)")
dw = d[(d.year >= AN_DEBUT) & (d.year <= AN_FIN)]
med_n = dw.groupby("year")["loss_eur"].median()
sl_n, _, _, pv_n, se_n = stats.linregress(med_n.index.values, np.log(med_n.values))
dwc = dw.dropna(subset=["cour"])
med_c = (dwc["cour"] * USD_EUR).groupby(dwc["year"]).median()
sl_c, _, _, pv_c, se_c = stats.linregress(med_c.index.values, np.log(med_c.values))
print(f"  nominale (colonne publiee) : {100*sl_n:+.1f} % par an, p = {pv_n:.4f}")
print(f"  reelle (valeur courante)   : {100*sl_c:+.1f} % par an, p = {pv_c:.4f}")
assert abs(100 * sl_n - 13.0) < 0.05, "la tendance nominale ne reproduit pas le script 89"

titre("3. L'ECHELLE DE QUEUE DERIVEE, NOMINALE ET REELLE (section 1ter du script 89)")
fit_n = ds.ajuster(d)
assert abs(100 * fit_n["b"] - 4.00) < 0.005, "la derive de queue nominale ne reproduit pas le 89"
dc = d.dropna(subset=["cour"]).copy()
dc["loss_eur"] = dc["cour"] * USD_EUR
fit_c = ds.ajuster(dc)
print(f"  nominale : b = {100*fit_n['b']:+.2f} % par an (ecart-type {100*fit_n['se_b']:.2f}), "
      f"p = {fit_n['p_lr']:.4f}, {fit_n['n_exc']} exces")
print(f"  reelle   : b = {100*fit_c['b']:+.2f} % par an (ecart-type {100*fit_c['se_b']:.2f}), "
      f"p = {fit_c['p_lr']:.4f}, {fit_c['n_exc']} exces")
print()
print("LECTURE, ECRITE APRES LES NOMBRES.")
print(f"  - l'inflation implicite vaut {100*infl:.2f} % par an : elle explique "
      f"{100*infl/sl_n:.0f} % de la derive du corps")
print(f"    ({100*sl_n:.1f} contre {100*sl_c:.1f} % une fois deflatee), et la derive reelle du corps")
print(f"    reste {'significative' if pv_c < 0.05 else 'NON significative'} ;")
print(f"  - dans la queue, la derive deflatee vaut {100*fit_c['b']:+.2f} % par an, "
      f"{'significative' if fit_c['p_lr'] < 0.05 else 'NON significative'} (p = {fit_c['p_lr']:.3f}).")
print("  Le seuil publie est garde fixe dans les deux branches : en valeur courante, davantage")
print("  de pertes anciennes le franchissent, d'ou un nombre d'exces different.")

titre("GRANDEURS CITEES (sans separateur ni signe)")
print(f"inflation_implicite_pct {100*infl:.2f}")
print(f"derive_mediane_nominale_pct {100*sl_n:.1f}")
print(f"derive_mediane_reelle_pct {100*sl_c:.1f}")
print(f"p_mediane_reelle {pv_c:.4f}")
print(f"derive_queue_nominale_pct {100*fit_n['b']:.2f}")
print(f"derive_queue_reelle_pct {100*fit_c['b']:.2f}")
print(f"p_queue_reelle {fit_c['p_lr']:.4f}")
print(f"exces_reels {fit_c['n_exc']}")
print(f"part_inflation_corps_pct {100*infl/sl_n:.0f}")
