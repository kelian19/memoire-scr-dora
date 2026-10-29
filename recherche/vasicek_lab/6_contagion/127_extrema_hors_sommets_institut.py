#!/usr/bin/env python3
"""127 : recherche d'extrema du capital hors des sommets du pavé admissible des directions.

Le pavé |A_jk| <= S_jk a 2^10 = 1024 sommets, évalués de façon exhaustive aux scripts 30 et 111. Rien
n'établit que les extrema du capital y vivent : augmenter A_jk augmente W_jk et diminue W_kj. Ce script
cherche des contre-exemples à l'intérieur du pavé, sans prétendre certifier l'absence d'extremum intérieur.

Deux objectifs, évalués aux mêmes tirages communs que les scripts 30 et 111 (40 000 années, graine
20260721) :
  - la moyenne annuelle LISSÉE : la loi du nombre de piliers touchés est remplacée par son espérance exacte
    (aucun seuil d'inversion), ce qui rend l'objectif régulier ; il est, par construction, un polynôme de
    degré au plus deux en chaque coordonnée antisymétrique (chaque arête n'apparaît qu'une fois, W_jk = S+a
    et W_kj = S-a), ce que le script vérifie numériquement ;
  - le capital (quantile à 99,5 %), constante par morceaux en W : recherche sans dérivée, et mesure du
    bruit de discrétisation pour ne pas appeler contre-exemple une fluctuation.
Aucune recalibration ; montants en M EUR comme aux scripts 30 et 111.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
for chemin in (REPO, LAB):
    sys.path.insert(0, str(chemin))
import partial_id as pid

SEED_EVAL = 20260721
SEED_RECHERCHE = 20261022
N_ALEATOIRES = 150
N_SOMMETS_DEPART = 20
N_POINTS_COURBURE = 2000


class Objectifs:
    def __init__(self):
        self.ev = pid.Evaluator(n_years=40_000, seed=SEED_EVAL)
        self.P = pid.expert_matrix()
        self.S, self.A_exp = pid.decompose(self.P)
        self.smax = self.S[pid.IU]
        self.sak = np.zeros((pid.NP_, pid.NP_+1))
        for a in range(pid.NP_):
            self.sak[a, 1:] = self.ev.cum[self.ev.idx_by_am[a], :].sum(axis=0)
        self.n_eval = 0

    def W(self, a):
        return pid.build_W(self.S, a)

    def moyenne(self, a):
        """Moyenne annuelle lissée : espérance exacte de la perte cumulée sur les tirages communs."""
        card, _ = pid.card_dist_all(self.W(a))
        self.n_eval += 1
        return float((card*self.sak).sum()/self.ev.n_years)

    def capital(self, a):
        self.n_eval += 1
        return float(self.ev.scr(self.W(a)))

    def admissible(self, a):
        return pid.admissible(self.W(a))


def sommets(smax):
    n = 1 << pid.NFREE
    signes = np.array([[(m >> k & 1)*2-1 for k in range(pid.NFREE)] for m in range(n)], dtype=float)
    return signes*smax


def coefficients_axe(f, a, i, smax):
    """f est un polynôme de degré 2 en a_i : coefficients exacts par trois évaluations aux bouts et au centre."""
    xs = np.array([-smax[i], 0.0, smax[i]])
    ys = []
    for x in xs:
        b = a.copy()
        b[i] = x
        ys.append(f(b))
    c2 = (ys[0]+ys[2]-2*ys[1])/(2*smax[i]**2)
    c1 = (ys[2]-ys[0])/(2*smax[i])
    return ys[1], c1, c2


def maximiser_axe(c0, c1, c2, s, signe):
    """Optimum de signe*(c0+c1 x+c2 x^2) sur [-s, s]."""
    candidats = [-s, s]
    if c2*signe < 0:
        x = -c1/(2*c2)
        if -s < x < s:
            candidats.append(x)
    return max(candidats, key=lambda x: signe*(c0+c1*x+c2*x*x))


def ascension_coordonnees(obj, f, depart, signe, balayages=60):
    a = depart.copy()
    valeur = f(a)
    for _ in range(balayages):
        avant = valeur
        for i in np.random.default_rng(SEED_RECHERCHE).permutation(pid.NFREE):
            c0, c1, c2 = coefficients_axe(f, a, i, obj.smax)
            x = maximiser_axe(c0, c1, c2, obj.smax[i], signe)
            b = a.copy()
            b[i] = x
            if obj.admissible(b):
                v = f(b)
                if signe*v >= signe*valeur-1e-12:
                    a, valeur = b, v
        if abs(valeur-avant) <= 1e-10*max(1.0, abs(valeur)):
            break
    return a, valeur


def est_sommet(a, smax, tol=1e-9):
    return bool(np.all(np.abs(np.abs(a)-smax) <= tol*np.maximum(1.0, smax)))


def main():
    debut = time.monotonic()
    obj = Objectifs()
    smax = obj.smax
    rng = np.random.default_rng(SEED_RECHERCHE)
    print('127 — EXTREMA DU CAPITAL HORS DES SOMMETS DU PAVÉ ADMISSIBLE, VERSION INSTITUT')
    print('Montants en M EUR ; mêmes tirages communs que les scripts 30 et 111 (40000 années, graine 20260721).')
    print('Recherche de contre-exemples, pas une certification : aucune preuve d’extremum global n’est revendiquée.')

    V = sommets(smax)
    adm_v = np.array([obj.admissible(v) for v in V])
    v_cap = np.array([obj.capital(v) if ok else np.nan for v, ok in zip(V, adm_v)])
    v_moy = np.array([obj.moyenne(v) if ok else np.nan for v, ok in zip(V, adm_v)])
    print(f'\nSOMMETS : {int(adm_v.sum())} admissibles sur {len(V)}.')
    print(f'  capital aux sommets : minimum {np.nanmin(v_cap):.0f}, maximum {np.nanmax(v_cap):.0f} '
          '(contrôle : 6858 et 8697 aux scripts 30 et 111).')
    assert round(np.nanmin(v_cap)) == 6858 and round(np.nanmax(v_cap)) == 8697
    print(f'  moyenne lissée aux sommets : minimum {np.nanmin(v_moy):.1f}, maximum {np.nanmax(v_moy):.1f}.')

    # Structure : degré deux en chaque coordonnée, et signe de la courbure.
    residus, courbures = [], []
    for _ in range(N_POINTS_COURBURE//20):
        a = (rng.random(pid.NFREE)*0.8-0.4)*smax
        i = int(rng.integers(pid.NFREE))
        xs = np.array([-0.5, -0.2, 0.2, 0.5])*smax[i]
        ys = []
        for x in xs:
            b = a.copy()
            b[i] = x
            ys.append(obj.moyenne(b))
        co = np.polyfit(xs[:3], ys[:3], 2)
        residus.append(abs(np.polyval(co, xs[3])-ys[3])/abs(ys[3]))
    print(f'\nSTRUCTURE : la moyenne lissée est un polynôme de degré deux en chaque coordonnée ; '
          f'résidu relatif maximal sur {len(residus)} contrôles {max(residus):.1e}.')
    assert max(residus) < 1e-9
    courb_par_axe = [[] for _ in range(pid.NFREE)]
    for _ in range(N_POINTS_COURBURE):
        a = (rng.random(pid.NFREE)*2-1)*smax
        i = int(rng.integers(pid.NFREE))
        courb_par_axe[i].append(coefficients_axe(obj.moyenne, a, i, smax)[2] > 0)
    fractions = [float(np.mean(c)) for c in courb_par_axe if c]
    tous = np.concatenate([np.array(c, dtype=float) for c in courb_par_axe])
    part_convexe = float(tous.mean())
    print(f'  courbure positive en {100*part_convexe:.0f} % des {len(tous)} points tirés dans le pavé ; '
          f'par coordonnée de {100*min(fractions):.0f} % à {100*max(fractions):.0f} %.')
    print('  Une courbure négative permet un maximum intérieur le long d’un axe : la structure seule ne démontre rien.')

    # Recherche : ascension exacte par coordonnées, depuis des sommets et des points intérieurs.
    resultats = {}
    for nom, f, v in (('moyenne', obj.moyenne, v_moy),):
        resultats[nom] = {}
        for sens, signe in (('maximum', 1), ('minimum', -1)):
            ordre = np.argsort(-signe*np.nan_to_num(v, nan=-signe*np.inf))
            departs = [V[j] for j in ordre[:N_SOMMETS_DEPART]]
            departs += [(rng.random(pid.NFREE)*2-1)*smax for _ in range(N_ALEATOIRES)]
            departs += [np.zeros(pid.NFREE), obj.A_exp[pid.IU].copy()]
            extremes = np.nanmax(v) if signe == 1 else np.nanmin(v)
            meilleur, meilleur_a, n_sommets, n_meilleurs, n_ameliore = None, None, 0, 0, 0
            for a0 in departs:
                if not obj.admissible(a0):
                    continue
                a, val = ascension_coordonnees(obj, f, a0, signe)
                if est_sommet(a, smax):
                    n_sommets += 1
                if meilleur is None or signe*val > signe*meilleur:
                    meilleur, meilleur_a = val, a
                if signe*val > signe*extremes+1e-9:
                    n_ameliore += 1
            resultats[nom][sens] = dict(sommet_extreme=float(extremes), meilleur_trouve=float(meilleur),
                                         depart=len(departs), finit_sur_sommet=n_sommets,
                                         ameliorent_le_sommet=n_ameliore,
                                         meilleur_est_sommet=bool(est_sommet(meilleur_a, smax)),
                                         coordonnees_interieures=int(np.sum(np.abs(np.abs(meilleur_a)-smax) > 1e-9*np.maximum(1, smax))))
            r = resultats[nom][sens]
            print(f'\nASCENSION PAR COORDONNÉES, moyenne lissée, {sens} : {r["depart"]} départs '
                  f'({N_SOMMETS_DEPART} sommets, {N_ALEATOIRES} points intérieurs, point nul, point d’expert).')
            print(f'  extremum aux sommets {r["sommet_extreme"]:.3f} ; meilleur trouvé {r["meilleur_trouve"]:.3f} ; '
                  f'{r["finit_sur_sommet"]} départs finissent sur un sommet ; {r["ameliorent_le_sommet"]} dépassent '
                  f'l’extremum des sommets ; le meilleur trouvé a {r["coordonnees_interieures"]} coordonnée(s) intérieure(s).')

    # Condition du premier ordre aux sommets extrêmes de la moyenne : la dérivée pointe-t-elle vers l'extérieur ?
    premier_ordre = {}
    for sens, signe, idx in (('maximum', 1, int(np.nanargmax(v_moy))), ('minimum', -1, int(np.nanargmin(v_moy)))):
        a = V[idx]
        derivees_exterieur = []
        for i in range(pid.NFREE):
            c0, c1, c2 = coefficients_axe(obj.moyenne, a, i, smax)
            derivee = c1+2*c2*a[i]
            derivees_exterieur.append(signe*derivee*np.sign(a[i]))
        ok = int(sum(d >= 0 for d in derivees_exterieur))
        premier_ordre[sens] = dict(coordonnees_ok=ok, sur=pid.NFREE)
        print(f'\nCONDITION DU PREMIER ORDRE au sommet {sens} de la moyenne : {ok} coordonnées sur {pid.NFREE} '
              'où la dérivée pointe vers le bord (l’extremum ne peut pas être amélioré en entrant par cette coordonnée).')

    # Capital : recherche sans dérivée, bruit de discrétisation, évolution différentielle semée de sommets.
    pas_bruit = 1e-3
    ecarts = []
    for _ in range(200):
        a = (rng.random(pid.NFREE)*1.8-0.9)*smax
        b = np.clip(a+rng.normal(0, pas_bruit, pid.NFREE)*smax, -smax, smax)
        ecarts.append(abs(obj.capital(b)-obj.capital(a)))
    bruit = float(np.std(ecarts))
    print(f'\nCAPITAL (quantile à 99,5 %) : bruit de discrétisation, écart moyen après une perturbation de '
          f'{100*pas_bruit:.1f} % de la demi-largeur : {np.mean(ecarts):.2f}, maximum {np.max(ecarts):.2f}.')
    resultats['capital'] = dict(bruit_discretisation_moyen=float(np.mean(ecarts)), bruit_maximum=float(np.max(ecarts)))
    for sens, signe in (('maximum', 1), ('minimum', -1)):
        extremes = np.nanmax(v_cap) if signe == 1 else np.nanmin(v_cap)
        ordre = np.argsort(-signe*np.nan_to_num(v_cap, nan=-signe*np.inf))[:10]
        pop = np.array([V[j]/smax for j in ordre]+[rng.random(pid.NFREE)*2-1 for _ in range(70)])
        res = differential_evolution(lambda x: -signe*obj.capital(x*smax), [(-1, 1)]*pid.NFREE, seed=SEED_RECHERCHE,
                                     maxiter=40, popsize=8, init=pop, tol=0, polish=False, updating='immediate')
        meilleur = -signe*res.fun
        a_best = res.x*smax
        resultats['capital'][sens] = dict(sommet_extreme=float(extremes), meilleur_trouve=float(meilleur),
                                          evaluations=int(res.nfev),
                                          meilleur_est_sommet=bool(est_sommet(a_best, smax, tol=1e-6)),
                                          admissible=bool(obj.admissible(a_best)),
                                          depasse_le_bruit=bool(signe*(meilleur-extremes) > bruit*3))
        r = resultats['capital'][sens]
        print(f'  {sens} : sommets {extremes:.1f} ; évolution différentielle semée de 10 sommets et de 70 points '
              f'intérieurs ({r["evaluations"]} évaluations) {meilleur:.1f} ; écart signé {signe*(meilleur-extremes):+.1f} ; '
              f'au-delà de trois bruits : {"oui" if r["depasse_le_bruit"] else "non"}.')

    cites = dict(part_courbure_positive_pct=100*part_convexe,
                 amelioration_max_moyenne=resultats['moyenne']['maximum']['meilleur_trouve']-resultats['moyenne']['maximum']['sommet_extreme'],
                 amelioration_min_moyenne=resultats['moyenne']['minimum']['sommet_extreme']-resultats['moyenne']['minimum']['meilleur_trouve'])
    print('\nGRANDEURS CITÉES')
    print(f'courbure positive en {cites["part_courbure_positive_pct"]:.0f} pour cent des points ; amélioration maximale de la moyenne '
          f'au-delà des sommets {cites["amelioration_max_moyenne"]:.3f}, minimale {cites["amelioration_min_moyenne"]:.3f}.')
    sortie = dict(experience='127 extrema hors des sommets', graine_evaluateur=SEED_EVAL, graine_recherche=SEED_RECHERCHE,
                  sommets_admissibles=int(adm_v.sum()), capital_sommets=[float(np.nanmin(v_cap)), float(np.nanmax(v_cap))],
                  moyenne_sommets=[float(np.nanmin(v_moy)), float(np.nanmax(v_moy))],
                  residu_degre_deux=float(max(residus)), part_courbure_positive=part_convexe,
                  courbure_par_coordonnee=fractions, resultats=resultats, premier_ordre=premier_ordre,
                  grandeurs_citees=cites, evaluations=obj.n_eval, duree_secondes=time.monotonic()-debut)
    json.dump(sortie, open(REPO/'resultats/institut_approfondissements/127.json', 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)
    print('\nLimites : recherches locales et évolution différentielle ne certifient pas l’absence d’extremum intérieur ;')
    print('le capital est constant par morceaux et la moyenne lissée n’est pas le capital ; un seul jeu de tirages communs ;')
    print('aucun encadrement par subdivision ni calcul d’intervalles n’a été tenté.')
    print(f'Durée {sortie["duree_secondes"]:.0f} s ; {obj.n_eval} évaluations ; résultats agrégés : 127.json')


if __name__ == '__main__':
    main()
