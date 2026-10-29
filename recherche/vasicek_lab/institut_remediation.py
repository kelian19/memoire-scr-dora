"""Actions par pilier pour les expériences Institut de janvier.

Les actions modifient des lignes de W, sans renormaliser la matrice. Deux noyaux
de cascade restent distincts : marche auto-évitante et branchement indépendant.
Le choc commun P4 remplace la cascade lorsqu'il est actif, comme dans 119.
Les actions sont des hypothèses mécaniques, pas des effets causaux mesurés.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

import canaux_conformite as cx
import partial_id as pid


NOYAUX = ("marche", "branchement")
FAMILLES = ("emission", "emission_et_tiers")
PILIERS = tuple(cx.PIL)
POIDS = np.array([cx.eng.LAMBDA[j] for j in PILIERS], dtype=float)
POIDS /= POIDS.sum()


def actions(budget=5):
    """Tous les sous-ensembles, avec coût unitaire conventionnel par pilier."""
    return [tuple(c) for n in range(budget+1) for c in combinations(PILIERS, n)]


def nom_action(action):
    return "+".join(f"P{j}" for j in action) or "aucune"


def verifier_matrice(w, noyau):
    w = np.asarray(w, dtype=float)
    if noyau not in NOYAUX or w.shape != (5, 5):
        raise ValueError("Noyau ou dimensions inadmissibles")
    if not np.isfinite(w).all() or (w < 0).any() or (w > 1).any():
        raise ValueError("Probabilités de transmission inadmissibles")
    if not np.allclose(np.diag(w), 0, rtol=0, atol=1e-14):
        raise ValueError("La diagonale doit être nulle")
    if pid.rho(w) >= 1:
        raise ValueError("Matrice hors de l'ensemble sous-critique retenu")
    if noyau == "marche" and (w.sum(axis=1) > 1+1e-13).any():
        raise ValueError("Une marche exige des sommes de lignes au plus égales à un")
    return w


def ensembles_marche(w, amorce):
    """Marche à au plus un successeur ; revisiter un pilier termine l'incident."""
    verifier_matrice(w, "marche")
    if amorce not in PILIERS:
        raise ValueError("Amorce inconnue")
    dist = {}

    def rec(cur, visites, poids):
        nouveaux = [k for k in range(5) if k not in visites]
        arret = 1-float(w[cur, nouveaux].sum())
        key = frozenset(j+1 for j in visites)
        dist[key] = dist.get(key, 0.) + poids*arret
        for k in nouveaux:
            if w[cur, k] > 0:
                rec(k, visites | {k}, poids*w[cur, k])

    rec(amorce-1, {amorce-1}, 1.)
    if min(dist.values()) < -1e-13 or abs(sum(dist.values())-1) > 1e-12:
        raise ArithmeticError("Loi des ensembles de la marche invalide")
    return dist


def appliquer(w, action=(), efficacite=.5, phi=0., famille="emission"):
    """Réduire les émissions de e ; option : réduire aussi phi si P4 est traité.

    On ne change ni fréquence, ni poids d'amorce, ni probabilité de dépassement.
    Une action n'est pas supposée atteindre un état de conformité observé.
    """
    action = tuple(action)
    if (not 0 <= efficacite <= 1 or not 0 <= phi <= 1
            or famille not in FAMILLES or len(set(action)) != len(action)
            or any(j not in PILIERS for j in action)):
        raise ValueError("Action, efficacité ou choc tiers inadmissible")
    v = np.array(w, dtype=float, copy=True)
    for j in action:
        v[j-1] *= 1-efficacite
    if famille == "emission_et_tiers" and 4 in action:
        phi *= 1-efficacite
    return v, float(phi)


def appliquer_ciblee(w, pilier, efficacite_emission, phi=None, efficacite_tiers=0.):
    """Séparer les efficacités d'émission et d'inclusion par le choc commun P4."""
    if (not np.isfinite(efficacite_tiers) or not 0 <= efficacite_tiers <= 1
            or (pilier != 4 and efficacite_tiers != 0)):
        raise ValueError("L'efficacité sur les tiers doit appartenir à [0,1] et concerner P4")
    v, _ = appliquer(w, (pilier,), efficacite_emission, phi or 0., 'emission')
    return v, None if phi is None else float(phi*(1-efficacite_tiers))


def economie_capital(q0, q1, q4, taux=.06):
    """Seuils de coût annuel tirés uniquement du financement du capital.

    q : M EUR de capital ; taux : coût de capital par an. Le coût direct fourni
    ensuite doit être annuel ou annualisé séparément. Aucun taux d'actualisation
    ni montant de perte moyenne n'est implicite dans ce score.
    """
    if (not np.isfinite([q0, q1, q4, taux]).all() or min(q0, q1, q4) < 0
            or not 0 < taux < 1):
        raise ValueError("Capitaux ou taux de financement inadmissibles")
    b1, b4 = taux*(q0-q1), taux*(q0-q4)
    return {'plafond_cout_P1': b1, 'plafond_cout_P4': b4,
            'surcout_P4_vs_P1': taux*(q1-q4)}


def choisir_cout(q0, q1, q4, cout1, cout4, taux=.06, tolerance=0.):
    """Choix parmi abstention/P1/P4, sur financement du capital + coûts directs."""
    e = economie_capital(q0, q1, q4, taux)
    if (not np.isfinite([cout1, cout4, tolerance]).all()
            or min(cout1, cout4, tolerance) < 0):
        raise ValueError("Les coûts et la tolérance doivent être finis et non négatifs")
    gains = {'aucune': 0., 'P1': e['plafond_cout_P1']-cout1,
             'P4': e['plafond_cout_P4']-cout4}
    maximum = max(gains.values())
    return [a for a, v in gains.items() if v >= maximum-tolerance]


def encadrer_bascule(avantage, tolerance=0., precision=.002):
    """Zone de bascule d'un avantage supposé non décroissant sur [0,1].

    L'appelant doit justifier la monotonie. La tolérance représente la résolution
    du calcul de capital, pas une incertitude statistique sur l'efficacité.
    """
    if (not np.isfinite([tolerance, precision]).all() or tolerance < 0
            or not 0 < precision < 1):
        raise ValueError("Précision ou tolérance inadmissible")
    cache = {}

    def a(t):
        if t not in cache:
            cache[t] = float(avantage(t))
            if not np.isfinite(cache[t]):
                raise ArithmeticError("Avantage non fini")
        return cache[t]

    a0, a1 = a(0.), a(1.)
    if a1 < a0-2*tolerance:
        raise ValueError("Avantage incompatible avec la monotonie supposée")
    base = {'intervalle': None, 'tolerance_capital': tolerance,
            'avantage_extremites': [a0, a1]}
    if a0 > tolerance:
        return dict(base, statut='P4_sur_toute_la_grille')
    if a1 < -tolerance:
        return dict(base, statut='P1_sur_toute_la_grille')
    if max(abs(a0), abs(a1)) <= tolerance:
        return dict(base, statut='egalite_numerique_sur_toute_la_grille')

    def franchissement(cible):
        if a0 >= cible:
            return [0., 0.]
        if a1 < cible:
            return [1., 1.]
        lo, hi = 0., 1.
        while hi-lo > precision:
            m = (lo+hi)/2
            if a(m) >= cible:
                hi = m
            else:
                lo = m
        return [lo, hi]

    gauche, droite = franchissement(-tolerance), franchissement(tolerance)
    return dict(base, statut='bascule_encadree', intervalle=[gauche[0], droite[1]],
                pas_dichotomie_max=precision, evaluations_supplementaires=len(cache))


def cardinal(w, noyau="marche", phi=None):
    """Loi exacte de K. phi=None : P4 suit la cascade ; phi=0 : choc P4 singleton.

    Cette distinction suit le moteur historique : désactiver le mécanisme de
    choc commun restaure la cascade P4, ce qui diffère d'un choc actif de force nulle.
    """
    w = verifier_matrice(w, noyau)
    if phi is not None and not 0 <= phi <= 1:
        raise ValueError("Choc tiers hors de [0,1]")
    if noyau == "branchement":
        cards = pid.card_dist_all(w)[0]
    else:
        cards = np.zeros((5, 6))
        for j in PILIERS:
            for ens, p in ensembles_marche(w, j).items():
                cards[j-1, len(ens)] += p
    if phi is not None:
        ind, probs = cx.table_p4_choc(phi)
        cards[3] = np.bincount(ind.sum(axis=1).astype(int), weights=probs, minlength=6)
    loi = POIDS @ cards
    if loi.min() < -1e-12 or abs(loi.sum()-1) > 1e-12 or loi[0] != 0:
        raise ArithmeticError("Loi de cardinal invalide")
    return loi


def directions():
    """Cinq matrices sur le segment expert/transposée, pas tout le pavé admissible."""
    expert = pid.expert_matrix(cx.G_NC)
    s, a = pid.decompose(expert)
    return [(f"direction_{t:+.1f}", float(t), s+t*a) for t in (-1., -.5, 0., .5, 1.)]


def decision(capitaux, liste_actions, budget, relatif=False, tolerance=0.):
    """Regret maximal sur un ensemble fini, avec ensemble complet des ex aequo.

    Chaque ligne est un scénario, chaque colonne une action. Le décideur peut
    s'abstenir. Normaliser par le capital sans action définit un autre objectif.
    La tolérance traite la résolution numérique ; ce n'est pas une bande statistique.
    """
    q = np.asarray(capitaux, dtype=float)
    if q.ndim != 2 or q.shape[1] != len(liste_actions) or not np.isfinite(q).all():
        raise ValueError("Tableau de capitaux invalide")
    admissibles = [i for i, a in enumerate(liste_actions) if len(a) <= budget]
    if not admissibles or () not in liste_actions:
        raise ValueError("Budget ou action nulle absent")
    regret = q[:, admissibles]-q[:, admissibles].min(axis=1, keepdims=True)
    if relatif:
        base = q[:, liste_actions.index(())]
        if (base <= 0).any():
            raise ValueError("Capital de référence non positif")
        regret = regret/base[:, None]
    maximum = regret.max(axis=0)
    best = float(maximum.min())
    return {
        "budget_unites": budget, "objectif": "relatif" if relatif else "M_EUR",
        "tolerance": tolerance, "regret_minimal": best,
        "actions_minimax": [nom_action(liste_actions[admissibles[k]])
                            for k in range(len(admissibles)) if maximum[k] <= best+tolerance],
        "regrets_maximaux": {nom_action(liste_actions[i]): float(v)
                              for i, v in zip(admissibles, maximum)},
    }
