# -*- coding: utf-8 -*-
"""Source UNIQUE du jugement d'expert et du bareme de scores de la cascade
qualitative DORA.

Auparavant ROOT / TRANS / GBASE etaient copies-colles dans quatre scripts
(build_cascade_workbook, build_figures, build_tree_figure, sensitivity_analysis),
avec un risque de desynchronisation silencieuse. Tout est desormais defini ici,
une seule fois ; les scripts importent depuis ce module.
"""

PILIERS = {
    1: "Gouvernance & gestion du risque TIC",
    2: "Gestion / classification / notification des incidents",
    3: "Tests de résilience opérationnelle (TLPT)",
    4: "Gestion du risque lié aux tiers ICT",
    5: "Partage d'informations sur les cybermenaces",
}
N = len(PILIERS)

# --- jugement ancré DORA / mémoire ----------------------------------------
ROOT = {1: 1.00, 4: 0.90, 2: 0.60, 3: 0.50, 5: 0.30}     # amorce d'une cascade
TRANS = {                                                # « i entraîne j » dirigé
    1: {2: 0.80, 3: 0.70, 4: 0.70, 5: 0.40},
    2: {1: 0.20, 3: 0.40, 4: 0.30, 5: 0.60},
    3: {1: 0.30, 2: 0.50, 4: 0.30, 5: 0.30},
    4: {1: 0.20, 2: 0.80, 3: 0.30, 5: 0.40},
    5: {1: 0.10, 2: 0.20, 3: 0.20, 4: 0.20},
}
GBASE = {4: 7, 1: 6, 2: 4, 3: 3, 5: 1}                    # gravité de base d'un pilier

# --- constantes de forme du barème (les moins ancrées : choix de modélisation) ---
KFAC, AMPB, AMPS = 0.40, 0.50, 0.80   # étendue (part des autres piliers) ; amplification (base, pente)

# --- labels de sortie -------------------------------------------------------
PROBA_LAB = ["Très rare", "Rare", "Possible", "Probable", "Très probable"]
GRAV_LAB = ["Négligeable", "Mineure", "Modérée", "Majeure", "Critique"]
CRIT_LAB = ["Faible", "Modérée", "Élevée", "Majeure", "Extrême"]


def lvl(score):
    """Niveau 0..4 (bornes 2/4/6/8) pour l'échelle de couleurs et les labels."""
    if score is None:
        return None
    return 0 if score <= 2 else 1 if score <= 4 else 2 if score <= 6 else 3 if score <= 8 else 4


def proba_score(order):
    """Cotation ordinale /10 de la vraisemblance de la chaîne ordonnée (dépend de l'ordre).

    NB : cotation ordinale qui classe les ordres, pas une probabilité normalisée.
    """
    if not order:
        return None
    prop = ROOT[order[0]]
    for a, b in zip(order, order[1:]):
        prop *= TRANS[a][b]
    return max(1, min(10, round(10 * prop)))


def gravite_score(order):
    """Étendue des dégâts amplifiée/amortie par la cohérence de l'ordre (dépend de l'ordre)."""
    if not order:
        return None
    bases = [GBASE[p] for p in order]
    etendue = min(10, max(bases) + KFAC * (sum(bases) - max(bases)))
    if len(order) <= 1:
        raw = etendue
    else:
        links = [TRANS[a][b] for a, b in zip(order, order[1:])]
        raw = etendue * (AMPB + AMPS * (sum(links) / len(links)))
    return max(1, min(10, round(raw)))


def crit_score(p, g):
    """Criticité = round(sqrt(p*g)).

    La racine ramène le produit p*g (le RPN de l'AMDEC) sur l'échelle 1..10 ; le
    CLASSEMENT est celui du produit (racine monotone), seule l'échelle change.
    """
    return None if (p is None or g is None) else round((p * g) ** 0.5)
