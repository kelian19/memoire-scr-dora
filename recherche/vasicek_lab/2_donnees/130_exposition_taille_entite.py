#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
130 : la descente d'echelle tient-elle compte de la taille d'une entite ?

LA QUESTION. Le chapitre resultats descend le capital du secteur a une entite par deux
elasticites estimees sur le panel OpRisk (b_lambda = 0.074 pour la frequence, b = 0.087 pour la
severite). Elles sont si faibles qu'un facteur 134 sur les actifs ne multiplie le besoin que par
2.6 : un groupe bancaire et un petit assureur recevraient pratiquement le meme niveau. Hugo
Rapior le conteste, et il a raison sur le principe : une entite dix fois plus grande ne porte pas
le meme besoin.

CE QUE LES DONNEES DISENT, ET POURQUOI LA BASE NE PEUT PAS TRANCHER (section 1). Sur les 521
pertes dont les actifs sont connus, la severite ne croit pas avec la taille : la pente est nulle ou
negative a tous les quantiles, y compris parmi les grandes institutions. La part de pertes au-dela
du seuil ne croit pas non plus. Les plus gros evenements rapportes aux actifs viennent
d'acteurs crypto de moins de 200 M USD d'actifs. La base est un registre d'evenements publics et
materiels : elle mesure ce qui est declare, pas l'exposition. Un exposant estime dessus vaut pour le
registre, il ne dit rien de ce qu'une entite plus grande risque.

NOTATION. Dans ce script l'exposant s'appelle beta ; le memoire l'ecrit b_lambda, le symbole de
l'elasticite de frequence dont la formule ci-dessous est la generalisation.

LA METHODE : LE PARTAGE DE L'EXPOSITION DU SECTEUR.
  lambda_i(beta) = Lambda * A_i^beta / sum_j A_j^beta ,
ou Lambda est le nombre annuel d'incidents du panel et la somme court sur les firmes actives. Le
secteur ne gagne ni ne perd d'incident : la somme des lambda_i vaut Lambda quel que soit beta
(controle). A beta = b_lambda, la formule redonne la descente actuelle a moins de 1 % : elle
n'est donc pas une autre methode, c'est la meme avec son exposant libere. A beta = 1, la
frequence est proportionnelle a l'exposition, la convention de toute tarification a l'exposition.
La severite garde la loi du secteur et son multiplicateur d'echelle m(A) inchange.

QUEL BETA RETENIR. Le registre en donne un (0.074) et le principe d'exposition un autre (1). Aucune
donnee du depot ne separe les deux. Nous retenons le milieu geometrique, beta_c = sqrt(b_lambda) =
0.27, et publions les deux bornes avec lui : la valeur centrale n'est pas estimee, elle est posee
entre deux lectures dont chacune a sa raison.

CE QUE LE SCRIPT NE FAIT PAS. Il ne modifie aucune constante gelee, ne recalibre pas le secteur
et ne plafonne pas la severite (c'est l'objet du script 78). Les montants sont des besoins ORSA
sous un etat de conformite SUPPOSE ; aucune de ces entites ne publie le sien.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(HERE)
import partial_id as pid                                            # noqa: E402
import descente as dsc                                              # noqa: E402

TAUX_USD = 1.04        # meme cours que le script 65
NY = 600_000           # meme nombre d'annees simulees que le script 65
SEED = 20260721
ALPHA = 0.995
U_MEUR = 20.03         # seuil de la calibration gelee
W_ = 92

# Les quatre assureurs du script 65 (SFCR 2024), en M EUR : actifs, SCR publie.
ASSUREURS = [("Assureur non-vie A", 2319.0, 425.0),
             ("Assureur non-vie B", 5305.0, 812.5),
             ("Assureur vie C", 37845.0, 1587.0),
             ("Assureur vie D", 309800.0, 14800.0)]
# Deux tailles POSEES (aucune n'est une entite observee) pour lire les deux extremes.
PETITE_MEUR = 500.0
GROUPE_MEUR = 2_400_000.0
NOTIONNELLE_MUSD = 20_000.0           # script 60


def titre(s):
    print("\n" + "=" * W_ + f"\n{s}\n" + "=" * W_)


# =====================================================================================
titre("1. Ce que la base dit de la taille, et ce qu'elle ne peut pas dire")
# =====================================================================================
from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR  # noqa: E402
import statsmodels.api as sm                                                       # noqa: E402

df = filter_finance(filter_cyber(load_clean(os.path.join(REPO, "data", "raw",
                                                         "SAS_OpRisk_Global_Data_June_2026.xlsx"))))
d = df[df.year.between(2000, 2026)].copy()
if len(d) != 570:
    sys.exit(f"population de calibration differente du protocole : {len(d)} pertes au lieu de 570")
d["A"] = pd.to_numeric(d["Assets ($M)"], errors="coerce")
d["L"] = d["loss"] * USD_EUR
m = d.A.gt(0) & d.L.gt(0)
dm = d[m].copy()
print(f"pertes cyber x finance 2000-2026 : {len(d)} ; avec actifs connus : {len(dm)}")
print(f"{'sous-population':<26}{'n':>5}{'MCO':>14}{'q50':>14}{'q75':>14}{'q90':>14}")
for lab, sel in (("toutes tailles", dm.A > 0), ("actifs >= 10 Md USD", dm.A >= 1e4),
                 ("actifs >= 50 Md USD", dm.A >= 5e4)):
    s = dm[sel]
    x = sm.add_constant(np.log(s.A.to_numpy()))
    y = np.log(s.L.to_numpy())
    cols = []
    ols = sm.OLS(y, x).fit()
    cols.append(f"{ols.params[1]:+.2f} ({ols.bse[1]:.2f})")
    for q in (0.5, 0.75, 0.9):
        r = sm.QuantReg(y, x).fit(q=q, max_iter=5000)
        cols.append(f"{r.params[1]:+.2f} ({r.bse[1]:.2f})")
    print(f"{lab:<26}{len(s):>5}" + "".join(f"{c:>14}" for c in cols))
print("pente de log(perte) sur log(actifs), entre parentheses l'erreur-type.")
dm["exces"] = dm.L > U_MEUR
dm["quintile"] = pd.qcut(dm.A, 5, labels=False)
print("\npart des pertes au-dela du seuil u par cinquieme d'actifs (du plus petit au plus grand) :")
parts = dm.groupby("quintile").exces.mean()
print("  " + "  ".join(f"{100*v:.0f} %" for v in parts.to_numpy()))
petites = dm[dm.A < 1e3]
print(f"\npertes de firmes a moins de 1 Md USD d'actifs : {len(petites)} sur {len(dm)} ; "
      f"{int(petites.exces.sum())} des {int(dm.exces.sum())} pertes au-dela de u")
dm["rel"] = dm.L / (dm.A * USD_EUR)
top = dm.nlargest(5, "rel")
print("les cinq pertes les plus grandes rapportees aux actifs viennent de firmes aux actifs de "
      f"{', '.join(f'{a:.0f}' for a in top.A)} M USD")
print("Lecture : aucune pente positive. La base ne permet pas d'estimer un effet de taille sur la")
print("severite, et l'exposant de frequence qu'elle donne vaut pour ce qu'elle enregistre.")

# =====================================================================================
titre("2. Le panel, l'exposant du registre, et l'identite de partage")
# =====================================================================================
D = dsc.Descente()
pa = D.pa
parannee = pa.groupby("year")
LAMBDA = float(parannee.n.sum().mean())
NFIRMES = float(parannee.firm.nunique().mean())
print(f"panel firme-annee avec actifs : {len(pa):,} lignes ; {NFIRMES:.0f} firmes actives par an en moyenne")
print(f"incidents du panel par an (Lambda) : {LAMBDA:.1f}")
B_REG = float(D.b_lam)
print(f"exposant de frequence du registre b_lambda = {B_REG:.4f} (ET {D.se_b:.4f}, z = {D.z_b:.2f})")
B_C = float(np.sqrt(B_REG))
LECTURES = [("registre", B_REG), ("central", B_C), ("proportionnel", 1.0)]
print(f"exposant central beta_c = sqrt(b_lambda) = {B_C:.4f} ; borne haute beta = 1")


def somme_pool(beta):
    """Moyenne annuelle de la somme des A_j^beta sur les firmes actives."""
    return float(np.mean([(g.actifs.to_numpy() ** beta).sum() for _, g in parannee]))


def lam_partage(a_musd, beta):
    return LAMBDA * a_musd ** beta / somme_pool(beta)


print("\nIdentite de partage : la somme des lambda_i sur les firmes actives vaut Lambda.")
for nom, beta in LECTURES:
    tot = float(np.mean([sum(lam_partage(a, beta) for a in g.actifs) for _, g in parannee]))
    print(f"  beta = {beta:.4f} : somme des lambda_i = {tot:.4f} (Lambda = {LAMBDA:.4f})")
    if abs(tot - LAMBDA) > 1e-6:
        sys.exit("le partage ne conserve pas la frequence du secteur : arret.")
print("\nControle d'emboitement : a beta = b_lambda, le partage redonne la descente actuelle.")
ecart_max = 0.0
for nom, a_eur, _ in ASSUREURS:
    a = a_eur * TAUX_USD
    r = lam_partage(a, B_REG) / D.lam(a)
    ecart_max = max(ecart_max, abs(r - 1))
    print(f"  {nom:<22} lambda partage {lam_partage(a, B_REG):.4f} contre {D.lam(a):.4f} (rapport {r:.4f})")
print(f"  ecart maximal : {100*ecart_max:.2f} %")
if ecart_max > 0.01:
    sys.exit("le partage ne redonne pas la descente actuelle a 1 % : arret.")

# =====================================================================================
titre("3. Le capital par entite et par lecture de la taille")
# =====================================================================================
Wexp = pid.expert_matrix()
S, _ = pid.decompose(Wexp)
A_ALL = pid.all_vertices(S, 1.0)
CARD_SOMMETS = []
for row in A_ALL:
    A = np.zeros((pid.NP_, pid.NP_))
    A[pid.IU] = row
    Wv = S + (A - A.T)
    if pid.admissible(Wv):
        CARD_SOMMETS.append(pid.card_dist_all(Wv)[0])
CARD_EXP = pid.card_dist_all(Wexp)[0]
print(f"sommets admissibles : {len(CARD_SOMMETS)} / {len(A_ALL)}")


def besoin(lam, mult):
    """(point, bande basse, bande haute) du besoin d'une entite, a nombres communs."""
    ev = pid.Evaluator(lam=lam, n_years=NY, alpha=ALPHA, seed=SEED)
    ev.cum = ev.cum * mult
    vals = [ev.from_card(c)[0] for c in CARD_SOMMETS]
    pt, _ = ev.from_card(CARD_EXP)
    return pt, min(vals), max(vals)


ENTITES = [("Petite entite (posee)", PETITE_MEUR * TAUX_USD, None)]
ENTITES += [(n, a * TAUX_USD, scr) for n, a, scr in ASSUREURS[:3]]
ENTITES += [("Entite notionnelle", NOTIONNELLE_MUSD, None),
            (ASSUREURS[3][0], ASSUREURS[3][1] * TAUX_USD, ASSUREURS[3][2]),
            ("Grand groupe (pose)", GROUPE_MEUR * TAUX_USD, None)]
ENTITES.sort(key=lambda e: e[1])

# controle de reproduction de la descente actuelle (script 60 / chapitre resultats)
pt_not, lo_not, hi_not = besoin(D.lam(NOTIONNELLE_MUSD), D.mult(NOTIONNELLE_MUSD))
print(f"\ncontrole : entite notionnelle, descente actuelle : {pt_not:.1f} et [{lo_not:.1f} ; {hi_not:.1f}]"
      " (publie : 169.0 et [131.5 ; 183.3])")
if abs(pt_not - 169.0) > 0.6 or abs(lo_not - 131.5) > 0.6 or abs(hi_not - 183.3) > 0.6:
    sys.exit("la descente actuelle n'est plus reproduite : arret.")

RES = {}
print(f"\n{'entite':<24}{'actifs M EUR':>14}", end="")
for nom, beta in LECTURES:
    print(f"{nom + ' (lam)':>18}{'point':>8}{'bande':>16}", end="")
print()
for nom_e, a_musd, scr in ENTITES:
    ligne = f"{nom_e:<24}{a_musd/TAUX_USD:>14.0f}"
    mult = D.mult(a_musd)
    for nom, beta in LECTURES:
        lam = D.lam(a_musd) if nom == "registre" else lam_partage(a_musd, beta)
        pt, lo, hi = besoin(lam, mult)
        RES[(nom_e, nom)] = (lam, pt, lo, hi)
        ligne += f"{lam:>18.4f}{pt:>8.1f}{'[' + format(lo, '.0f') + ' ; ' + format(hi, '.0f') + ']':>16}"
    print(ligne)
print("\nbesoin ORSA en M EUR, point a la matrice d'expert au gain non conforme et bande sur les")
print("sommets admissibles ; lambda en incidents par an ; severite multipliee par m(A) comme au script 65.")

# =====================================================================================
titre("4. Ce que la lecture centrale change, et ses proprietes")
# =====================================================================================
noms = [e[0] for e in ENTITES]
for nom, beta in LECTURES:
    pts = [RES[(n, nom)][1] for n in noms]
    mono = all(b >= a - 1e-9 for a, b in zip(pts, pts[1:]))
    print(f"lecture {nom:<14}: croissance du besoin avec les actifs : {mono}")
    if not mono:
        sys.exit("le besoin n'est pas croissant avec la taille : arret.")
print()
for nom, beta in LECTURES:
    pa_, pg_ = RES[("Assureur non-vie A", nom)][1], RES[("Grand groupe (pose)", nom)][1]
    pd_ = RES[("Assureur vie D", nom)][1]
    pp_ = RES[("Petite entite (posee)", nom)][1]
    rapport = (f"{pg_/pa_:.1f}" if pa_ > 0 else "non defini (besoin nul)")
    print(f"lecture {nom:<14}: grand groupe / assureur non-vie A = {rapport} ; "
          f"grand groupe / assureur vie D = {pg_/pd_:.2f} ; petite entite = {pp_:.1f}")
print(f"facteur sur les actifs, grand groupe / assureur non-vie A : "
      f"{GROUPE_MEUR/ASSUREURS[0][1]:.0f} ; grand groupe / petite entite : {GROUPE_MEUR/PETITE_MEUR:.0f}")
for nom, beta in LECTURES:
    g_ = RES[("Grand groupe (pose)", nom)][1]
    for cible in ("Assureur non-vie B", "Assureur vie C"):
        c_ = RES[(cible, nom)][1]
        quoi = f"{g_/c_:.1f}" if c_ > 0 else "non defini (besoin nul)"
        print(f"lecture {nom:<14}: grand groupe / {cible.lower()} = {quoi} "
              f"(facteur sur les actifs {GROUPE_MEUR/dict((n, a) for n, a, _ in ASSUREURS)[cible]:.0f})")

print("\nRapport du besoin central au SCR publie (repere, pas une part) :")
for nom_e, a_musd, scr in ENTITES:
    if scr:
        print(f"  {nom_e:<24} {RES[(nom_e, 'central')][1]:>7.1f} / {scr:>8.1f} = "
              f"{100*RES[(nom_e, 'central')][1]/scr:>5.1f} %   (registre : "
              f"{100*RES[(nom_e, 'registre')][1]/scr:>5.1f} %)")

# taille de bascule : sous quelle taille le quantile a 99.5 % de la charge annuelle est-il nul ?
print("\nTaille de bascule : plus petite taille dont le besoin au niveau 99.5 % n'est pas nul")


def positif(a_musd, beta):
    lam = lam_partage(a_musd, beta)
    ev = pid.Evaluator(lam=lam, n_years=NY, alpha=ALPHA, seed=SEED)
    ev.cum = ev.cum * D.mult(a_musd)
    return ev.from_card(CARD_EXP)[0] > 0


for nom, beta in LECTURES[1:]:
    lo_, hi_ = 100.0, 5.0e6
    if positif(lo_, beta):
        print(f"  lecture {nom:<14}: positif des 100 M USD")
        continue
    if not positif(hi_, beta):
        print(f"  lecture {nom:<14}: nul sur toute la plage")
        continue
    for _ in range(18):
        mid = float(np.sqrt(lo_ * hi_))
        lo_, hi_ = (lo_, mid) if positif(mid, beta) else (mid, hi_)
    print(f"  lecture {nom:<14}: {hi_/TAUX_USD:.0f} M EUR d'actifs")
print("Sous cette taille, la probabilite annuelle d'une charge de queue strictement positive est inferieure")
print("a 0.5 % : le quantile a 99.5 % est nul, comme tout quantile de niveau inferieur. Un incident compte")
print("peut avoir une charge nulle, la severite retenue etant nulle sous le seuil : cette probabilite est donc")
print("inferieure a celle d'au moins un incident (tableau ci-dessous). Le risque se lit a un niveau plus")
print("eleve, ou par une mesure de queue. Taille obtenue sur une seule graine : ordre de grandeur.")


def probas_annuelles(lam, mult):
    """(P(au moins un incident), P(charge annuelle de queue > 0)), meme tirage que `besoin`."""
    ev = pid.Evaluator(lam=lam, n_years=NY, alpha=ALPHA, seed=SEED)
    ev.cum = ev.cum * mult
    p_inc = np.unique(ev.year_of).size / NY
    card = pid.card_dist_all(Wexp)[0]
    K = np.empty(ev.T, dtype=np.int64)
    for a in range(pid.NP_):
        idx = ev.idx_by_am[a]
        if idx.size:
            cdf = np.cumsum(card[a][1:])
            cdf[-1] = 1.0
            K[idx] = np.searchsorted(cdf, ev.u[idx], side="right") + 1
    np.clip(K, 1, pid.NP_, out=K)
    annual = np.bincount(ev.year_of, weights=ev.cum[ev._ar, K - 1], minlength=NY)
    return p_inc, float((annual > 0).mean())


print("\nProbabilite annuelle d'au moins un incident, et d'une charge de queue strictement positive")
print("(etat suppose, lecture centrale puis proportionnelle) :")
for nom_e, a_musd, _ in ENTITES:
    morceaux = []
    for nom_l, beta in LECTURES[1:]:
        p_inc, p_pos = probas_annuelles(lam_partage(a_musd, beta), D.mult(a_musd))
        morceaux.append(f"{nom_l} : {100*p_inc:.2f} % / {100*p_pos:.2f} %")
    print(f"  {nom_e:<24} " + " ; ".join(morceaux))

# =====================================================================================
titre("5. Grandeurs citees")
# =====================================================================================
print(f"exposant du registre {B_REG:.3f}")
print(f"exposant central {B_C:.3f}")
for nom_e, a_musd, scr in ENTITES:
    if nom_e.startswith("Entite notionnelle"):
        continue
    for nom, beta in LECTURES:
        lam, pt, lo, hi = RES[(nom_e, nom)]
        print(f"{nom_e} lecture {nom} : lambda {lam:.4f} point {pt:.0f} bande {lo:.0f} {hi:.0f}")
