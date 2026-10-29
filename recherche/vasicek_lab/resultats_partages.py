# -*- coding: utf-8 -*-
"""Resultats PARTAGES entre scripts, avec leur provenance.

RAISON D'ETRE. Plusieurs scripts avaient besoin de valeurs PRODUITES par d'autres scripts
(l'ecart de capital DORA du 43, les bornes du chapitre 10, le corpus de post-mortems du 53).
Elles y etaient recopiees a la main : correctes, mais susceptibles de devenir perimees en
silence si le script amont etait relance avec d'autres parametres. Ce module est la source
unique : chaque valeur y figure UNE fois, avec le script qui la produit.

REGLE. On ne met ici que des resultats PRODUITS PAR UN AUTRE SCRIPT ou des ANCRES EXTERNES
sourcees. Les parametres de modele (xi, sigma, seuils, lambda) restent dans `src.utils.config`
et `euro_cascade_model` : ce module ne les duplique pas.

REGLE 2. Ce qui peut etre DERIVE l'est, plutot que recopie : les signes directionnels et les
comptes par paire sont calcules depuis les transitions brutes du corpus, de sorte qu'ils ne
peuvent pas divergerdu codage d'origine.
"""

from itertools import combinations

PIL = [1, 2, 3, 4, 5]
PAIR_LAB = [f"P{a}-P{b}" for a, b in combinations(PIL, 2)]      # ordre de np.triu_indices(5,1)

# =============================================================================================
# 1. SOCLE ET BORNES DE CONTAGION  (chapitre 10 ; scripts 30 et 55, graine 20260721)
# =============================================================================================
SCR_SOCLE = 5275.0          # SCR sans contagion (W = 0), M EUR
BORNES_CONTAGION = (6858.0, 8697.0)   # bornes sur l'ensemble admissible, ignorance totale (t=1)

# =============================================================================================
# 2. ATTRIBUTION PAR CANAL / KPI  (script 43, source OpRisk)
# =============================================================================================
SCR_CONFORME = 6049.0       # etat conforme (cible DORA), M EUR
SCR_NON_CONFORME = 20188.0  # etat non conforme, M EUR
DELTA_DORA = 14139.0        # ecart total (inclut l'interaction entre canaux)
INTERACTION = 5001.0        # part non additive des canaux

# capital en jeu par canal, canal ISOLE : (libelle, Delta SCR en M EUR, statut).
# Les libelles servent a l'AFFICHAGE (figures, tableaux) : ils sont donc accentues.
LEVIERS = [
    ("fréquence (entrée)", 4328.0, "calibrable"),
    ("accumulation P4",    1728.0, "borné"),
    ("propagation W",      1633.0, "borné"),
    ("détection P2/P3",    1448.0, "calibrable"),
]

# =============================================================================================
# 3. CORPUS DE POST-MORTEMS  (script 53) : les transitions BRUTES, tout le reste en derive
# =============================================================================================
# (pilier source, pilier cible) par incident. Source unique : voir le script 53 pour le
# protocole de codage et les references des rapports officiels.
POSTMORTEM_EDGES = {
    "Microsoft Exchange Online 2023": [(1, 3), (3, 2), (1, 4)],
    "TSB Bank 2018":                  [(1, 4), (1, 3), (3, 2), (4, 2)],
    "Equifax 2017":                   [(1, 5), (5, 3), (3, 2), (1, 3)],
    "Knight Capital 2012":            [(3, 2), (1, 3)],
    "Capital One 2019":               [(1, 4), (1, 3)],
    "ION Cleared Derivatives 2023":   [(4, 2)],
    "MOVEit 2023":                    [(4, 2)],
}


def postmortem_counts():
    """(n_lo, n_hi) par paire : nb de chaines du pilier de plus PETIT numero vers l'autre, et
    l'inverse. Derive de POSTMORTEM_EDGES : ne peut pas diverger du codage d'origine."""
    c = {lab: [0, 0] for lab in PAIR_LAB}
    for edges in POSTMORTEM_EDGES.values():
        for (k, j) in edges:
            lo, hi = min(k, j), max(k, j)
            c[f"P{lo}-P{hi}"][0 if k == lo else 1] += 1
    return {lab: tuple(v) for lab, v in c.items()}


def postmortem_signs():
    """Signe directionnel par paire DOCUMENTEE : +1 si le pilier de plus petit numero precede.

    Convention du chapitre 10 : a_jk > 0 <=> W[lo,hi] > W[hi,lo] <=> lo precede hi.
    Les paires non documentees (ou a comptes egaux) sont ABSENTES du dictionnaire.
    """
    out = {}
    for lab, (n_lo, n_hi) in postmortem_counts().items():
        if n_lo > n_hi:
            out[lab] = +1
        elif n_hi > n_lo:
            out[lab] = -1
    return out


# =============================================================================================
# 4. RESULTATS EMPIRIQUES ETABLIS  (scripts 08g, 28)
# =============================================================================================
GINI_JOURNALIER = 0.69      # concentration des comptes journaliers, secteur financier (08g)
PART_JOURS_CHARGES = (0.017, 0.20)   # 1,7 % des jours portent 20 % des incidents (08g)
VCDB_PARTNER_FINANCE = 56   # incidents actor.Partner, finance (script 28) ; 0 deux crans plus bas
SEUIL_ESTIMATION = 30       # volume minimal par cellule pour estimer (regle du memoire)

# =============================================================================================
# 5. ANCRES EXTERNES SOURCEES  (ne proviennent pas de nos calculs)
# =============================================================================================
# Formule Standard, charge de risque operationnel pour l'entite notionnelle (chapitre 12, sc. 27)
SF_OP_CHARGE = 450.0

# ESAs, 1er rapport annuel d'incidents DORA (3 juin 2026)
ESAS_PART_TIERCE = 0.29     # part des 3 383 incidents TIC majeurs 2025 d'origine tierce
ESAS_N_INCIDENTS = 3383
ESAS_N_CTPP = 19            # prestataires TIC critiques designes (nov. 2025)

# BCE, analyse des registres d'externalisation (2024, donnees 2023)
ECB_CONCENTRATION = [(10, 0.30), (30, 0.50)]   # (nb de prestataires, part du budget)

# Marche cyber mondial : primes brutes 2025 (Business Insurance / Triple-I)
MARCHE_CYBER_USD = 16300.0
USD_EUR = 0.92

# Grandes pertes cyber reelles, en M USD (presse specialisee, rapports d'entreprise)
PERTES_REELLES_USD = {
    "Merck (NotPetya)": 1400.0,
    "Equifax 2017": 1400.0,
    "Change Healthcare 2024": 2400.0,
    "NotPetya total 2017": 10000.0,
}

# Cout de mise en conformite DORA par grande entite, M EUR (estimations de place, McKinsey 2025)
COUT_DORA_ENTITE = (25.0, 150.0)

# Cout du capital : marge de risque Solvabilite II (reglement delegue 2015/35)
COUT_DU_CAPITAL = 0.06

# =============================================================================================
# 6. PARAMETRES DE MODELE RAPPELES POUR LISIBILITE  (source : euro_cascade_model / classeur)
# =============================================================================================
GAMMA_ACCUMULATION = 0.68   # borne haute de phi_cs (concentration cloud), scripts 38/42
ROOT_P4 = 0.90              # propension d'amorce de P4 (classeur qualitatif)
