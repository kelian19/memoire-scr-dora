#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Moteur des QUATRE CANAUX que la conformite DORA deplace, extrait du script 43.

POURQUOI CE MODULE EXISTE. Le script 43 attribue l'ecart de capital DORA canal par canal et
imprime l'interaction en RESIDU. Le script 68 decompose ce residu, et il doit le faire sur
EXACTEMENT le meme moteur, les memes graines et le meme nombre d'annees, faute de quoi la
reconciliation qu'il annonce ne serait qu'une coincidence de tirages. Recopier le moteur d'un
script a l'autre est precisement la dette que le projet a deja identifiee (constantes recopiees
entre 44/49/50/55/56). Il vit donc ici, une fois, et 43 comme 68 le lisent.

CONTRAT. Ce module ne fait AUCUN calcul a l'import : il expose des constantes et deux
fonctions. Toute modification ici deplace la sortie des deux scripts, donc du memoire : la
calibration est gelee depuis le 7 aout 2026, seules les corrections d'erreur sont recevables.

LES QUATRE CANAUX (etats C = cible conforme -> NC = non conforme, scripts 16/38/39/42) :
  frequence    lam      21,6 -> 53,6    calibrable (KPI d'entree, script 08b)
  detection    p_u      x0,85 -> x1,20  calibrable (delai P2, confinement P3)
  propagation  g        0,45 -> 0,90    borne (le canal NON identifie : W)
  accumulation phi_cs   0 -> 0,68       borne (concentration des tiers P4, script 38/42)
"""

import os
import sys
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                                 # noqa: E402
from euro_cascade_model import PARAMS, var                      # noqa: E402
from src.aggregation.lda import simulate_remediation_severity   # noqa: E402
import scr_engine as eng                                        # noqa: E402

sp = PARAMS["OPRISK"]
PIL = eng.PIL
P4 = 4
_col = {j: c for c, j in enumerate(PIL)}
_others = [j for j in PIL if j != P4]

# LES TROIS CONSTANTES DE SIMULATION SONT PARTIE DU CHIFFRE PUBLIE, pas un reglage.
# Les changer deplace les 14 139 M du memoire.
NY = 40_000
NSEED = 4
SEED0 = 20260727

# etats DORA par canal (C = cible conforme, NC = non conforme)
LAM_C = sp["lam_ref"]
try:
    LAM_NC = ec.lambda_scenario("OPRISK", "S2_non_conforme")
except Exception:
    LAM_NC = LAM_C * 1.30                                       # facteur_recalibration config
P_U0 = sp["p_u"]
PU_C, PU_NC = 0.85 * P_U0, 1.20 * P_U0                          # multiplicateur detection (16/39)
G_C, G_NC = 0.45, 0.90                                          # gain propagation (G_PROP C/NC)
PHICS_NC = 0.68                                                 # accumulation P4 (gamma, script 38)


# ------------------------------------------------------------------ tables de cascade
def table_amorce(j, g):
    dist = eng.cascade_set_dist(j, g)
    sets = list(dist.keys())
    probs = np.array([dist[s] for s in sets])
    ind = np.zeros((len(sets), 5))
    for r, s in enumerate(sets):
        for p in s:
            ind[r, _col[p]] = 1.0
    return ind, probs / probs.sum()


def table_p4_choc(phi_cs):
    """P4 en choc commun : P4 + chaque autre pilier inclus indep. avec proba phi_cs."""
    sets, probs = [], []
    for r in range(len(_others) + 1):
        for combo in combinations(_others, r):
            p = 1.0
            for k in _others:
                p *= phi_cs if k in combo else (1.0 - phi_cs)
            ind = np.zeros(5)
            for x in (P4,) + combo:
                ind[_col[x]] = 1.0
            sets.append(ind)
            probs.append(p)
    return np.array(sets), np.array(probs)


def pertes_annuelles(lam, g, p_u, phi_cs, rng, ny=NY):
    """Vecteur des ny pertes annuelles agregees d'une configuration de canaux.

    L'ORDRE DES TIRAGES EST PARTIE DU RESULTAT. Comptage, amorce, uniformes de cascade, puis
    severite : c'est l'ordre du script 43, donc celui des 14 139 M publies. Le reordonner, ou
    inserer un tirage, redonnerait des nombres tout aussi valides et tout aussi differents.
    """
    r = lam / (ec.PHI - 1.0)
    counts = rng.negative_binomial(r, r / (r + lam), size=ny)
    T = int(counts.sum())
    if T == 0:
        return np.zeros(ny)
    year_of = np.repeat(np.arange(ny), counts)
    w = np.array([eng.LAMBDA[j] for j in PIL], float)
    amorce = rng.choice(5, size=T, p=w / w.sum())
    U = rng.random(T)
    SEV = simulate_remediation_severity(T * 5, sp["xi"], sp["sigma"], sp["u"], p_u,
                                        sp["cap"], rng).reshape(T, 5)
    tables = {j: table_amorce(j, g) for j in PIL}
    if phi_cs is not None:
        tables[P4] = table_p4_choc(phi_cs)
    annual = np.zeros(ny)
    for c, j in enumerate(PIL):
        idx = np.where(amorce == c)[0]
        if idx.size == 0:
            continue
        ind, probs = tables[j]
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        sel = np.searchsorted(cdf, U[idx], side="right")
        np.clip(sel, 0, len(probs) - 1, out=sel)
        loss = (SEV[idx] * ind[sel]).sum(axis=1)
        annual += np.bincount(year_of[idx], weights=loss, minlength=ny)
    return annual


def metriques_par_graine(lam, g, p_u, phi_cs=None, nseed=NSEED, ny=NY, seed0=SEED0):
    """(VaR 99,5 %, perte moyenne) graine par graine. phi_cs=None -> P4 noeud de cascade.

    NOMBRES COMMUNS, ET CE QU'ILS COUVRENT. La graine k est la meme pour toutes les
    configurations, donc deux configurations de MEME lam partagent leurs tirages de comptage,
    d'amorce et de severite : leur difference est alors quasi exempte de bruit de simulation.
    Deux configurations de lam DIFFERENT ne les partagent pas, la loi de comptage changeant :
    c'est la que le bruit entre, et c'est pourquoi le script 68 le mesure au lieu de le
    supposer negligeable.

    LA SECONDE METRIQUE N'EST PAS UN ORNEMENT. Une interaction est une difference de
    differences, l'estimateur le plus bruite du projet : le script 20b a montre que son SIGNE
    n'est pas resolu en VaR d'une graine a l'autre, alors qu'il l'est en perte moyenne. Toute
    decomposition de l'interaction doit donc etre lue dans les deux metriques.
    """
    out = np.empty((nseed, 2))
    for k in range(nseed):
        rng = np.random.default_rng(seed0 + k)
        annual = pertes_annuelles(lam, g, p_u, phi_cs, rng, ny)
        out[k] = (var(annual), float(annual.mean()))
    return out


def scr_par_graine(lam, g, p_u, phi_cs=None, nseed=NSEED, ny=NY):
    """SCR (VaR 99,5 %) graine par graine."""
    return metriques_par_graine(lam, g, p_u, phi_cs, nseed, ny)[:, 0]


def scr_config(lam, g, p_u, phi_cs=None, nseed=NSEED, ny=NY):
    """SCR moyen sur nseed graines. Signature et valeur identiques a celles du script 43."""
    return float(np.mean(scr_par_graine(lam, g, p_u, phi_cs, nseed, ny)))
