#!/usr/bin/env python3
"""125 : rééchantillonnage annuel conjoint des nombres et des pertes.

On tire des années complètes 2000 à 2025 avec remise, en conservant pour chaque année tirée
son nombre de pertes et ses pertes. À chaque tirage :
  - le taux de dépassement et les paramètres de queue sont réestimés ;
  - la fréquence est ancrée par un facteur relatif : lambda_b = lambda_ref * n_b / n_ref,
    où n_b est le nombre de pertes du tirage et n_ref celui des 26 années. La référence
    gelée 582/27 est donc conservée comme niveau de départ.
Deux schémas : années échangeables, et blocs de trois années consécutives (dépendance
temporelle). Les capitaux sont calculés avec et sans le facteur de fréquence sur les mêmes
tirages, ce qui isole l'apport de la fréquence.
2026, année incomplète, est exclue. La dispersion, les multiplicateurs entre états et la
cascade d'expert restent posés ; l'incertitude de fréquence mesurée ici est celle des
nombres annuels de pertes enregistrées, pas celle d'une fréquence par entité.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import genpareto

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
for chemin in (REPO, LAB):
    sys.path.insert(0, str(chemin))
import institut_incertitude as ii
from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR

U = ii.REFERENCE.u
DERNIERE_ANNEE = 2025
LONGUEUR_BLOC = 3


def capital_facteur(moteur, mask, regle, facteur, alpha=.995):
    """Même calcul que FourierIncidents.capital, avec la fréquence multipliée par un facteur."""
    cx = ii.cx
    lam = (cx.LAM_NC if mask & 1 else cx.LAM_C)*facteur
    pu = moteur.cal.pu*(1.20 if mask & 2 else .85)
    loi_k = ii.cardinal(bool(mask & 4), bool(mask & 8))
    incident = moteur.transformee_incident(pu, loi_k, regle)
    masses = np.fft.irfft(ii.pgf_nb(incident, lam), n=moteur.n)/moteur.attenuation
    if not np.isfinite(masses).all() or masses.min() < -1e-8:
        raise ArithmeticError("Inversion instable")
    cdf = np.cumsum(masses)
    idx = int(np.searchsorted(cdf, alpha))
    if idx >= moteur.n or idx > moteur.n*.25 or cdf[-1] < .995:
        raise ArithmeticError("Quantile trop près de la portée")
    return idx*moteur.step


def capitaux(cal, facteurs):
    """Capitaux aux deux états, quatre règles, pour chaque facteur de fréquence."""
    base = ii.grille_adaptee(cal)
    for essai in range(5):
        moteur = ii.FourierIncidents(cal, step=base*2**essai)
        try:
            return {nom: {r: {e: capital_facteur(moteur, m, r, f) for e, m in (('conforme', 0), ('non_conforme', 15))}
                          for r in ii.REGLES} for nom, f in facteurs.items()}
        except ArithmeticError:
            if essai == 4:
                raise
    raise AssertionError


def ajuster(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xi, loc, sigma = genpareto.fit(x, floc=0)
    return float(xi), float(sigma)


def tirage_annees(rng, annees, schema):
    n = len(annees)
    if schema == 'annees':
        return rng.integers(0, n, n)
    blocs = int(np.ceil(n/LONGUEUR_BLOC))
    debuts = rng.integers(0, n-LONGUEUR_BLOC+1, blocs)
    return np.concatenate([np.arange(d, d+LONGUEUR_BLOC) for d in debuts])[:n]


def centiles(valeurs):
    v = np.array(valeurs)
    return dict(p5=float(np.percentile(v, 5)), p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)),
                moyenne=float(v.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replications', type=int, default=800)
    parser.add_argument('--graine', type=int, default=20261022)
    parser.add_argument('--sortie', type=Path, default=REPO/'resultats/institut_approfondissements/125.json')
    args = parser.parse_args()
    debut = time.monotonic()
    df = filter_finance(filter_cyber(load_clean(REPO/'data/raw/SAS_OpRisk_Global_Data_June_2026.xlsx')))
    df = df[df['year'].between(2000, DERNIERE_ANNEE)]
    pertes = df['loss'].to_numpy(dtype=float)*USD_EUR
    annees = df['year'].to_numpy(dtype=int)
    liste_annees = np.arange(2000, DERNIERE_ANNEE+1)
    par_annee = [pertes[annees == a] for a in liste_annees]
    n_ref = int(sum(len(p) for p in par_annee))
    comptes = np.array([len(p) for p in par_annee])
    print('125 — RÉÉCHANTILLONNAGE ANNUEL CONJOINT, VERSION INSTITUT')
    print(f'{len(liste_annees)} années complètes 2000-{DERNIERE_ANNEE} ; {n_ref} pertes ; '
          f'{int(sum((p > U).sum() for p in par_annee))} excès au-dessus de {U:.2f} M EUR ; aucune année sans perte : '
          f'{bool((comptes > 0).all())}.')
    print(f'Nombre annuel de pertes : moyenne {comptes.mean():.2f}, écart-type {comptes.std(ddof=1):.2f}, '
          f'minimum {comptes.min()}, maximum {comptes.max()}.')
    print('Fréquence ancrée par lambda_b = lambda_ref * n_b / n_ref ; la référence gelée 582/27 est conservée.')
    print('Montants en M EUR du périmètre financier étudié ; bandes percentiles exploratoires.', flush=True)

    # Contrôle d'identité : facteur 1 reproduit le moteur sur le calage des 26 années.
    x_all = np.concatenate(par_annee)
    xi0, sg0 = ajuster(x_all[x_all > U]-U)
    cal0 = ii.Calibration(xi0, sg0, float((x_all > U).mean()), U)
    ref = capitaux(cal0, {'sans': 1.0})
    res0, _ = ii.calculer_configurations(cal0, masks=(0, 15))
    for r in ii.REGLES:
        assert abs(ref['sans'][r]['conforme']-res0[r]['0']) < 1e-9 and abs(ref['sans'][r]['non_conforme']-res0[r]['15']) < 1e-9
    print('Contrôle : le facteur 1 reproduit le moteur aux huit cellules du calage sur les 26 années.')
    print(f"Calage sur les 26 années : xi {xi0:.3f}, sigma {sg0:.1f}, taux de dépassement {cal0.pu:.4f} ; "
          f"capital additif {ref['sans']['additive']['conforme']:.0f} et {ref['sans']['additive']['non_conforme']:.0f}.", flush=True)

    sortie = dict(experience='125 rééchantillonnage annuel conjoint', n_annees=len(liste_annees), n_pertes=n_ref,
                  nombre_annuel=dict(moyenne=float(comptes.mean()), ecart_type=float(comptes.std(ddof=1)),
                                     minimum=int(comptes.min()), maximum=int(comptes.max())),
                  calage_26_annees=dict(xi=xi0, sigma=sg0, pu=cal0.pu, capitaux=ref['sans']),
                  longueur_bloc=LONGUEUR_BLOC, schemas={})
    for schema in ('annees', 'blocs'):
        rng = np.random.default_rng(args.graine+(0 if schema == 'annees' else 1))
        lignes, echecs, facteurs_n = [], 0, []
        for b in range(args.replications):
            idx = tirage_annees(rng, liste_annees, schema)
            x = np.concatenate([par_annee[i] for i in idx])
            n_b = len(x)
            exces = x[x > U]-U
            if len(exces) < 10:
                echecs += 1
                continue
            xi, sg = ajuster(exces)
            cal = ii.Calibration(xi, sg, len(exces)/n_b, U)
            try:
                cap = capitaux(cal, {'sans': 1.0, 'avec': n_b/n_ref})
            except ArithmeticError:
                echecs += 1
                continue
            lignes.append(dict(xi=xi, sigma=sg, pu=cal.pu, facteur=n_b/n_ref, capitaux=cap))
            facteurs_n.append(n_b/n_ref)
            if (b+1) % 200 == 0:
                print(f'  {schema} : {b+1}/{args.replications} ; {time.monotonic()-debut:.0f} s', flush=True)
        print(f'\nSCHÉMA {"ANNÉES ÉCHANGEABLES" if schema == "annees" else "BLOCS DE TROIS ANNÉES"} : '
              f'{len(lignes)} tirages réussis, {echecs} échecs journalisés ; '
              f'{sum(1 for l in lignes if l["xi"] >= 1)} ajustements avec xi >= 1 conservés.')
        f = np.array(facteurs_n)
        print(f'  facteur de fréquence n_b/n_ref : 5 % {np.percentile(f, 5):.3f}, médiane {np.percentile(f, 50):.3f}, '
              f'95 % {np.percentile(f, 95):.3f}')
        res_schema = dict(tirages=len(lignes), echecs=echecs,
                          xi_ge_1=int(sum(1 for l in lignes if l['xi'] >= 1)),
                          facteur_frequence=centiles(f), regles={})
        for r in ii.REGLES:
            res_schema['regles'][r] = {}
            for variante in ('sans', 'avec'):
                c = [l['capitaux'][variante][r]['conforme'] for l in lignes]
                n = [l['capitaux'][variante][r]['non_conforme'] for l in lignes]
                rap = [b_/a_ for a_, b_ in zip(c, n)]
                res_schema['regles'][r][variante] = dict(conforme=centiles(c), non_conforme=centiles(n),
                                                          rapport=centiles(rap), ecart=centiles([b_-a_ for a_, b_ in zip(c, n)]))
        print(f"  {'règle':10s} {'fréquence':10s} {'rapport : 5 %':>14s} {'médiane':>8s} {'95 %':>7s} {'  non conforme : 5 %':>20s} {'médiane':>8s} {'95 %':>8s}")
        for r in ii.REGLES:
            for variante, nom in (('sans', 'fixée'), ('avec', 'propagée')):
                v = res_schema['regles'][r][variante]
                print(f"  {r:10s} {nom:10s} {v['rapport']['p5']:14.2f} {v['rapport']['p50']:8.2f} {v['rapport']['p95']:7.2f} "
                      f"{v['non_conforme']['p5']:20.0f} {v['non_conforme']['p50']:8.0f} {v['non_conforme']['p95']:8.0f}")
        a = res_schema['regles']['additive']
        larg = lambda v, k: (v[k]['p95']-v[k]['p5'])
        res_schema['elargissement_additive'] = dict(
            rapport=larg(a['avec'], 'rapport')/larg(a['sans'], 'rapport')-1,
            non_conforme=larg(a['avec'], 'non_conforme')/larg(a['sans'], 'non_conforme')-1)
        print(f"  Largeur de la bande à 90 %, règle additive, due à la fréquence : rapport "
              f"{100*res_schema['elargissement_additive']['rapport']:+.0f} %, état non conforme "
              f"{100*res_schema['elargissement_additive']['non_conforme']:+.0f} %.", flush=True)
        sortie['schemas'][schema] = res_schema

    sortie['duree_secondes'] = time.monotonic()-debut
    json.dump(sortie, open(args.sortie, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\nLimites : un seul calage de référence ; années échangeables ou en blocs de trois, sans tendance ;')
    print('fréquence relative aux pertes enregistrées, pas par entité, avec dispersion et multiplicateurs posés ;')
    print('la hausse du nombre annuel de pertes dans le temps n’est pas modélisée ici.')
    print(f'Durée {sortie["duree_secondes"]:.1f} s ; résultats agrégés : {args.sortie.name}')


if __name__ == '__main__':
    main()
