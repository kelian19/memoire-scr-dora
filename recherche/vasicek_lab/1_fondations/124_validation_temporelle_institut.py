#!/usr/bin/env python3
"""124 : validation chronologique de trois lois de sévérité des excès, et effet sur le capital.

Modèles, au-dessus du seuil fixé u, sur les années complètes 2000 à 2025 :
  S : loi de Pareto généralisée stationnaire ;
  D : échelle variable, log sigma_t = a + b (t - centre), forme commune ;
  R : deux régimes d'échelle séparés par une date choisie sur l'apprentissage, forme commune.
Évaluation en origine glissante : pour chaque année notée, ajustement sur les années
strictement antérieures, puis score logarithmique et PIT des excès observés cette année-là.
Les excès d'une même année sont traités comme indépendants, ce qui est une approximation.
2026, année incomplète, est exclue de l'évaluation et de l'ajustement.
Le script ne propage ni la fréquence, ni la dispersion, ni la cascade d'expert.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import genpareto, kstest

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
for chemin in (REPO, LAB):
    sys.path.insert(0, str(chemin))
import institut_incertitude as ii
from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR

U = ii.REFERENCE.u
PREMIERE_ANNEE_NOTEE = 2014
DERNIERE_ANNEE_COMPLETE = 2025
MIN_PAR_REGIME = 8


def nll(xi, sigmas, x):
    """Vraisemblance négative de la loi de Pareto généralisée, échelle propre à chaque excès."""
    if np.any(sigmas <= 0):
        return 1e12
    z = xi*x/sigmas
    if np.any(1+z <= 0):
        return 1e12
    if abs(xi) < 1e-8:
        return float(np.sum(np.log(sigmas)+x/sigmas))
    return float(np.sum(np.log(sigmas)+(1+1/xi)*np.log1p(z)))


def minimiser(f, x0):
    meilleur = None
    for depart in (x0, x0+0.1, x0-0.1):
        r = minimize(f, depart, method='Nelder-Mead',
                     options=dict(xatol=1e-7, fatol=1e-9, maxiter=4000))
        if meilleur is None or r.fun < meilleur.fun:
            meilleur = r
    return meilleur


def ajuster_stationnaire(x):
    xi, _, sigma = genpareto.fit(x, floc=0)
    return dict(modele='S', xi=float(xi), sigma=float(sigma), nll=nll(xi, np.full(len(x), sigma), x))


def ajuster_derive(x, t):
    centre = float(np.mean(t))
    s0 = ajuster_stationnaire(x)
    def f(p):
        return nll(p[0], np.exp(p[1]+p[2]*(t-centre)), x)
    r = minimiser(f, np.array([s0['xi'], np.log(s0['sigma']), 0.0]))
    return dict(modele='D', xi=float(r.x[0]), log_sigma0=float(r.x[1]), pente=float(r.x[2]),
                centre=centre, nll=float(r.fun))


def ajuster_regimes(x, t):
    s0 = ajuster_stationnaire(x)
    meilleur = None
    for c in sorted(set(t))[:-1]:
        avant, apres = t <= c, t > c
        if avant.sum() < MIN_PAR_REGIME or apres.sum() < MIN_PAR_REGIME:
            continue
        def f(p):
            return nll(p[0], np.where(avant, np.exp(p[1]), np.exp(p[2])), x)
        r = minimiser(f, np.array([s0['xi'], np.log(s0['sigma']), np.log(s0['sigma'])]))
        if meilleur is None or r.fun < meilleur['nll']:
            meilleur = dict(modele='R', xi=float(r.x[0]), sigma_avant=float(np.exp(r.x[1])),
                            sigma_apres=float(np.exp(r.x[2])), rupture=int(c), nll=float(r.fun))
    return meilleur


def ajuster_logit(y, t):
    """Probabilité de dépassement du seuil : régression logistique sur l'année."""
    centre = float(np.mean(t))
    p0 = min(max(float(np.mean(y)), 1e-3), 1-1e-3)
    def f(p):
        eta = p[0]+p[1]*(t-centre)
        return float(np.sum(np.logaddexp(0, eta)-y*eta))
    r = minimiser(f, np.array([np.log(p0/(1-p0)), 0.0]))
    return dict(a=float(r.x[0]), b=float(r.x[1]), centre=centre)


def p_logit(a, annee):
    return float(1/(1+np.exp(-(a['a']+a['b']*(annee-a['centre'])))))


def echelle(ajustement, annee):
    m = ajustement['modele']
    if m == 'S':
        return ajustement['sigma']
    if m == 'D':
        return float(np.exp(ajustement['log_sigma0']+ajustement['pente']*(annee-ajustement['centre'])))
    return ajustement['sigma_apres']


def ajuster_tous(x, t):
    out = {'S': ajuster_stationnaire(x), 'D': ajuster_derive(x, t)}
    r = ajuster_regimes(x, t)
    out['R'] = r if r is not None else {**ajuster_stationnaire(x), 'modele': 'R', 'rupture': None,
                                         'sigma_avant': None, 'sigma_apres': ajuster_stationnaire(x)['sigma']}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sortie', type=Path, default=REPO/'resultats/institut_approfondissements/124.json')
    parser.add_argument('--bootstrap', type=int, default=4000)
    args = parser.parse_args()
    debut = time.monotonic()
    df = filter_finance(filter_cyber(load_clean(REPO/'data/raw/SAS_OpRisk_Global_Data_June_2026.xlsx')))
    df = df[df['year'].between(2000, 2026)]
    pertes = df['loss'].to_numpy(dtype=float)*USD_EUR
    annees = df['year'].to_numpy(dtype=int)
    if len(pertes) != 570:
        raise ValueError(f'Échantillon différent du protocole : {len(pertes)} pertes')
    complet = annees <= DERNIERE_ANNEE_COMPLETE
    exces_mask = pertes > U
    print('124 — VALIDATION CHRONOLOGIQUE DE LA SÉVÉRITÉ DES EXCÈS, VERSION INSTITUT')
    print(f'{len(pertes)} pertes 2000-2026 ; {int((complet).sum())} sur les années complètes 2000-{DERNIERE_ANNEE_COMPLETE} ; '
          f'{int((exces_mask & complet).sum())} excès au-dessus de {U:.2f} M EUR sur ces années.')
    print('Années notées : %d à %d. Ajustements sur les années antérieures seulement.' %
          (PREMIERE_ANNEE_NOTEE, DERNIERE_ANNEE_COMPLETE))
    print('Les excès d’une même année sont traités comme indépendants : approximation.', flush=True)

    scores = {m: [] for m in 'SDR'}
    pits = {m: [] for m in 'SDR'}
    par_annee = []
    exces_test = []
    for t in range(PREMIERE_ANNEE_NOTEE, DERNIERE_ANNEE_COMPLETE+1):
        app = (annees < t) & exces_mask
        test = (annees == t) & exces_mask
        x_app, t_app = pertes[app]-U, annees[app].astype(float)
        x_test = pertes[test]-U
        ajust = ajuster_tous(x_app, t_app)
        ligne = {'annee': t, 'n_apprentissage': int(app.sum()), 'n_test': int(test.sum())}
        for m, a in ajust.items():
            sig = echelle(a, t)
            ls = genpareto.logpdf(x_test, c=a['xi'], scale=sig)
            pit = genpareto.cdf(x_test, c=a['xi'], scale=sig)
            scores[m].extend(ls.tolist())
            pits[m].extend(pit.tolist())
            ligne[m] = dict(xi=a['xi'], sigma=sig, log_score=float(ls.sum()))
        exces_test.extend([t]*int(test.sum()))
        par_annee.append(ligne)
        print(f"  {t} : apprentissage {ligne['n_apprentissage']:3d} excès, test {ligne['n_test']:2d} ; "
              f"sigma S {ligne['S']['sigma']:6.1f} D {ligne['D']['sigma']:6.1f} R {ligne['R']['sigma']:6.1f} ; "
              f"xi S {ligne['S']['xi']:.2f} D {ligne['D']['xi']:.2f} R {ligne['R']['xi']:.2f}", flush=True)
    n_test = len(scores['S'])
    ls = {m: np.array(v) for m, v in scores.items()}
    rng = np.random.default_rng(20261022)
    print(f'\nSCORE LOGARITHMIQUE MOYEN PAR EXCÈS NOTÉ ({n_test} excès notés ; plus grand = mieux)')
    resume = {'n_notes': n_test, 'modeles': {}, 'differences': {}}
    for m in 'SDR':
        pit = np.array(pits[m])
        ks = kstest(pit, 'uniform')
        resume['modeles'][m] = dict(score_moyen=float(ls[m].mean()), pit_moyen=float(pit.mean()),
                                    ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))
        print(f"  {m} : score {ls[m].mean():.4f} ; PIT moyen {pit.mean():.3f} ; Kolmogorov-Smirnov {ks.statistic:.3f}, p {ks.pvalue:.3f}")
    print('\nDIFFÉRENCES DE SCORE MOYEN PAR EXCÈS (bootstrap des excès notés, exploratoire)')
    for m in 'DR':
        diff = ls[m]-ls['S']
        b = np.array([diff[rng.integers(0, n_test, n_test)].mean() for _ in range(args.bootstrap)])
        gagne = int(sum(1 for a in par_annee if a[m]['log_score'] > a['S']['log_score']))
        perd = int(sum(1 for a in par_annee if a[m]['log_score'] < a['S']['log_score']))
        resume['differences'][m] = dict(moyenne=float(diff.mean()), p5=float(np.percentile(b, 5)),
                                        p95=float(np.percentile(b, 95)), annees_gagnees=gagne, annees_perdues=perd)
        print(f"  {m} moins S : {diff.mean():+.4f} ; intervalle bootstrap à 90 % [{np.percentile(b, 5):+.4f} ; "
              f"{np.percentile(b, 95):+.4f}] ; années où {m} fait mieux : {gagne}, moins bien : {perd}")

    print('\nPROBABILITÉ DE DÉPASSEMENT DU SEUIL : constante contre tendance logistique')
    sc_const, sc_trend, gagnes, perdus = [], [], 0, 0
    for t in range(PREMIERE_ANNEE_NOTEE, DERNIERE_ANNEE_COMPLETE+1):
        app, test = annees < t, annees == t
        y_app = exces_mask[app].astype(float)
        y_test = exces_mask[test].astype(float)
        pc = float(y_app.mean())
        pt = p_logit(ajuster_logit(y_app, annees[app].astype(float)), t)
        lc = y_test*np.log(pc)+(1-y_test)*np.log(1-pc)
        lt = y_test*np.log(pt)+(1-y_test)*np.log(1-pt)
        sc_const.extend(lc.tolist()); sc_trend.extend(lt.tolist())
        gagnes += int(lt.sum() > lc.sum()); perdus += int(lt.sum() < lc.sum())
        print(f"  {t} : {int(test.sum()):3d} pertes notées, {int(y_test.sum()):2d} excès ; p constante {pc:.3f}, tendance {pt:.3f}")
    sc_const, sc_trend = np.array(sc_const), np.array(sc_trend)
    diff_p = sc_trend-sc_const
    bp = np.array([diff_p[rng.integers(0, len(diff_p), len(diff_p))].mean() for _ in range(args.bootstrap)])
    resume['taux_depassement'] = dict(n_pertes_notees=int(len(diff_p)), score_constant=float(sc_const.mean()),
                                      score_tendance=float(sc_trend.mean()), difference=float(diff_p.mean()),
                                      p5=float(np.percentile(bp, 5)), p95=float(np.percentile(bp, 95)),
                                      annees_gagnees=gagnes, annees_perdues=perdus)
    print(f"  score moyen par perte notée ({len(diff_p)} pertes) : constante {sc_const.mean():.4f}, tendance {sc_trend.mean():.4f} ; "
          f"différence {diff_p.mean():+.4f}, intervalle bootstrap à 90 % [{np.percentile(bp, 5):+.4f} ; {np.percentile(bp, 95):+.4f}] ; "
          f"années où la tendance fait mieux : {gagnes}, moins bien : {perdus}")

    # Ajustement sur toutes les années complètes et capital pour l'année suivante.
    x_all = pertes[complet & exces_mask]-U
    t_all = annees[complet & exces_mask].astype(float)
    pu = float(len(x_all)/complet.sum())
    ajust = ajuster_tous(x_all, t_all)
    annee_prevue = DERNIERE_ANNEE_COMPLETE+1
    print(f'\nAJUSTEMENT SUR 2000-{DERNIERE_ANNEE_COMPLETE} ({len(x_all)} excès, taux de dépassement {pu:.4f}) ; '
          f'échelle prévue pour {annee_prevue}')
    capitaux = {}
    a_logit = ajuster_logit(exces_mask[complet].astype(float), annees[complet].astype(float))
    pu_prevu = p_logit(a_logit, annee_prevue)
    resume['logit_complet'] = dict(a_logit, pente_par_an=a_logit['b'], p_prevu=pu_prevu, p_constant=pu)
    print(f"  tendance logistique sur 2000-{DERNIERE_ANNEE_COMPLETE} : pente {a_logit['b']:+.4f} par an ; "
          f"probabilité prévue pour {annee_prevue} : {pu_prevu:.4f} contre {pu:.4f} constante")
    modeles = [('S', 'S', pu), ('D', 'D', pu), ('R', 'R', pu), ('P', 'S', pu_prevu), ('DP', 'D', pu_prevu)]
    for m, base, pu_m in modeles:
        a = ajust[base]
        sig = echelle(a, annee_prevue)
        cal = ii.Calibration(a['xi'], sig, pu_m, U)
        res, _ = ii.calculer_configurations(cal, masks=(0, 15))
        capitaux[m] = {r: dict(conforme=v['0'], non_conforme=v['15'], rapport=v['15']/v['0']) for r, v in res.items()}
        extra = ''
        if m == 'D':
            extra = f" ; pente {a['pente']:+.4f} par an, soit {100*(np.exp(a['pente'])-1):+.1f} % par an"
        if m == 'R':
            extra = f" ; rupture après {a['rupture']}, échelles {a['sigma_avant']:.1f} puis {a['sigma_apres']:.1f}"
        print(f"  {m} : xi {a['xi']:.3f}, sigma {sig:.1f}{extra}")
    print("  (P : sévérité stationnaire et taux de dépassement en tendance ; DP : échelle variable et tendance)")
    print('\nCAPITAL ANNUEL À 99,5 % (M EUR, périmètre financier étudié) SELON LA LOI DE SÉVÉRITÉ')
    print(f"{'règle':10s} {'modèle':7s} {'conforme':>9s} {'non conforme':>13s} {'rapport':>8s}")
    for r in ii.REGLES:
        for m in ('S', 'D', 'R', 'P', 'DP'):
            c = capitaux[m][r]
            print(f"{r:10s} {m:7s} {c['conforme']:9.0f} {c['non_conforme']:13.0f} {c['rapport']:8.3f}")
    print('\nGRANDEURS CITÉES, RÈGLE ADDITIVE : variation relative au modèle stationnaire')
    sortie_rel = {}
    for m in ('D', 'R', 'P', 'DP'):
        c, s = capitaux[m]['additive'], capitaux['S']['additive']
        sortie_rel[m] = dict(conforme=c['conforme']/s['conforme']-1, non_conforme=c['non_conforme']/s['non_conforme']-1,
                             rapport=c['rapport']/s['rapport']-1)
        print(f"  {m} : conforme {100*sortie_rel[m]['conforme']:+.0f} %, non conforme {100*sortie_rel[m]['non_conforme']:+.0f} %, "
              f"rapport {100*sortie_rel[m]['rapport']:+.1f} %")
    sortie = dict(experience='124 validation chronologique de la sévérité', seuil=U,
                  annees_notees=[PREMIERE_ANNEE_NOTEE, DERNIERE_ANNEE_COMPLETE], par_annee=par_annee, resume=resume,
                  ajustements_complets=ajust, taux_depassement_complet=pu, annee_prevue=annee_prevue,
                  capitaux=capitaux, variation_relative_additive=sortie_rel,
                  duree_secondes=time.monotonic()-debut)
    json.dump(sortie, open(args.sortie, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\nLimites : peu d’excès par année (un à onze), apprentissage de 24 excès à la première année notée ;')
    print('forme commune aux régimes ; rupture choisie sur l’apprentissage ; excès supposés indépendants ;')
    print('ni fréquence, ni dispersion, ni cascade d’expert ne sont réestimées.')
    print(f'Durée {sortie["duree_secondes"]:.1f} s ; résultats agrégés : {args.sortie.name}')


if __name__ == '__main__':
    main()
