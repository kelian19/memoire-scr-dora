#!/usr/bin/env python3
"""131 : les caractéristiques observées améliorent-elles la prévision des pertes importantes ?

Question posée par la proposition de complément ML du 27 octobre 2026. Le script 124 compare déjà,
en origine glissante, un taux de dépassement constant et une tendance logistique sur l'année. Ce
script demande si les caractéristiques de la base SAS ajoutent une information hors période
d'apprentissage, au-delà de cette tendance.

Partie 1, dépassement du seuil. Cible Y = 1{perte > u}, u = 20,03 M EUR, parmi les PERTES
ENREGISTRÉES : la probabilité est conditionnelle à l'enregistrement, ce n'est ni une probabilité
annuelle d'incident ni une probabilité de non-conformité. Quatre modèles, hyperparamètres fixés
AVANT toute évaluation (aucun réglage sur les années notées) :
  M0 taux constant ; M1 logistique sur l'année (le modèle du script 124) ;
  M2 logistique pénalisée (L2) sur l'année et les caractéristiques ;
  M3 boosting d'arbres de profondeur 2 sur les mêmes caractéristiques.
Caractéristiques : log des actifs (indicateur de manque), ligne métier Bâle regroupée en quatre
modalités, catégorie d'événement (fraude externe ou interruption d'activité), région regroupée en
quatre modalités. Imputation et regroupements appris sur l'apprentissage seul.

Partie 2, sévérité des excès. Échelle de la loi de Pareto généralisée log sigma = a + b(t - centre)
(modèle D du script 124) contre la même échelle augmentée du log des actifs et de la catégorie
d'événement (modèle DX), forme commune. Score logarithmique hors apprentissage.

Évaluation : origine glissante 2014 à 2025, apprentissage sur les années strictement antérieures,
2026 incomplète exclue. Incertitude par bootstrap des années notées (blocs annuels). Les actifs
sont ceux de l'année d'information financière déclarée par la base, dont la part postérieure à
l'événement est mesurée et traitée en sensibilité. Diagnostic : aucune valeur publiée n'est modifiée.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import genpareto

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
for chemin in (REPO, LAB):
    sys.path.insert(0, str(chemin))
import institut_incertitude as ii
from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR

import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

U = ii.REFERENCE.u
PREMIERE_ANNEE_NOTEE = 2014
DERNIERE_ANNEE_COMPLETE = 2025
GRAINE = 20261027
# Hyperparamètres déclarés avant évaluation, jamais réglés sur les années notées.
LOGIT_C = 1.0
BOOST = dict(max_depth=2, max_iter=100, learning_rate=0.05, min_samples_leaf=20,
             l2_regularization=1.0, random_state=GRAINE)
METIERS = ('Retail Banking', 'Payment and Settlement', 'Insurance')
REGIONS = ('North America', 'Europe', 'Asia')
MODELES = ('M0', 'M1', 'M2', 'M3')
NOMS = dict(M0='taux constant', M1='tendance logistique', M2='logistique pénalisée + caractéristiques',
            M3='boosting + caractéristiques')


def signaux_covariables(variantes):
    """Conserve chaque sensibilité au lieu d'écraser les modèles de mêmes noms."""
    return {
        nom: {m: resultats[m]['contre_M1']['p5'] > 0 for m in ('M2', 'M3')}
        for nom, resultats in variantes.items()
    }


def charger():
    df = filter_finance(filter_cyber(load_clean(REPO/'data/raw/SAS_OpRisk_Global_Data_June_2026.xlsx')))
    df = df[df['year'].between(2000, 2026)].copy()
    if len(df) != 570:
        raise ValueError(f'Échantillon différent du protocole : {len(df)} pertes')
    df['perte'] = df['loss'].astype(float)*USD_EUR
    df['y'] = (df['perte'] > U).astype(int)
    actifs = pd.to_numeric(df['Assets ($M)'], errors='coerce')
    df['log_actifs'] = np.log(actifs.where(actifs > 0))
    # La colonne est une date de clôture, pas une année : lue en nombre, elle donnait des nanosecondes.
    annee_fin = pd.to_datetime(df['Year of Financial Information'], errors='coerce').dt.year
    df['fin_posterieure'] = (annee_fin > df['year']).astype(int)
    df['annee_fin'] = annee_fin
    # Actifs connus au plus tard l'année de l'événement, manquants sinon : seule version sans fuite.
    df['log_actifs_ex_ante'] = df['log_actifs'].where(annee_fin <= df['year'])
    df['metier'] = df['Basel Business Line - Level 1'].where(
        df['Basel Business Line - Level 1'].isin(METIERS), 'Autre')
    df['bdsf'] = (df['Event Risk Category'] == 'Business Disruption and System Failures').astype(int)
    df['region'] = df['Region of Domicile'].where(df['Region of Domicile'].isin(REGIONS), 'Autre')
    df['firme'] = df['Firm Name'].astype(str)
    return df


def matrice(app, test, col_actifs):
    """Caractéristiques standardisées sur l'apprentissage ; imputation par la médiane d'apprentissage."""
    centre_t = app['year'].mean()
    med = app[col_actifs].median() if col_actifs else None
    def construire(d):
        cols = {'annee': d['year']-centre_t, 'bdsf': d['bdsf']}
        if col_actifs:
            cols['log_actifs'] = d[col_actifs].fillna(med)
            cols['actifs_manquants'] = d[col_actifs].isna().astype(float)
        for m in METIERS:
            cols[f'metier_{m}'] = (d['metier'] == m).astype(float)
        for r in REGIONS:
            cols[f'region_{r}'] = (d['region'] == r).astype(float)
        return pd.DataFrame(cols).astype(float)
    xa, xt = construire(app), construire(test)
    mu, sd = xa.mean(), xa.std().replace(0, 1.0)
    return (xa-mu)/sd, (xt-mu)/sd, list(xa.columns)


def logit_annee(y, t):
    """Modèle M1, identique au script 124 : logistique non pénalisée sur l'année."""
    centre = float(np.mean(t))
    p0 = min(max(float(np.mean(y)), 1e-3), 1-1e-3)
    def f(p):
        eta = p[0]+p[1]*(t-centre)
        return float(np.sum(np.logaddexp(0, eta)-y*eta))
    r = minimize(f, np.array([np.log(p0/(1-p0)), 0.0]), method='Nelder-Mead',
                 options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
    return lambda tt: 1/(1+np.exp(-(r.x[0]+r.x[1]*(tt-centre))))


def predire(app, test, col_actifs):
    y = app['y'].to_numpy(float)
    xa, xt, cols = matrice(app, test, col_actifs)
    p = {'M0': np.full(len(test), y.mean()),
         'M1': logit_annee(y, app['year'].to_numpy(float))(test['year'].to_numpy(float))}
    lr = LogisticRegression(C=LOGIT_C, max_iter=5000).fit(xa, y)
    p['M2'] = lr.predict_proba(xt)[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gb = HistGradientBoostingClassifier(**BOOST).fit(xa, y)
    p['M3'] = gb.predict_proba(xt)[:, 1]
    return {m: np.clip(v, 1e-6, 1-1e-6) for m, v in p.items()}, lr, cols


def pente_calibration(y, p):
    """Pente de la régression logistique de y sur logit(p) : 1 si calibré, < 1 si trop dispersé."""
    z = np.log(p/(1-p))
    def f(b):
        eta = b[0]+b[1]*z
        return float(np.sum(np.logaddexp(0, eta)-y*eta))
    return float(minimize(f, np.array([0.0, 1.0]), method='Nelder-Mead').x[1])


def bootstrap_annees(valeurs, annees, rng, b):
    """Moyenne par perte notée, années tirées avec remise (blocs annuels)."""
    uniques = np.unique(annees)
    par_annee = {a: valeurs[annees == a] for a in uniques}
    out = np.empty(b)
    for k in range(b):
        tir = rng.choice(uniques, len(uniques), replace=True)
        v = np.concatenate([par_annee[a] for a in tir])
        out[k] = v.mean()
    return out


def nll_gpd(xi, sig, x):
    if np.any(sig <= 0) or np.any(1+xi*x/sig <= 0):
        return 1e12
    return float(np.sum(np.log(sig)+(1+1/xi)*np.log1p(xi*x/sig)))


def ajuster_echelle(x, z):
    """GPD de forme commune, log sigma = z @ beta (z contient la constante)."""
    xi0, _, s0 = genpareto.fit(x, floc=0)
    depart = np.concatenate([[xi0, np.log(s0)], np.zeros(z.shape[1]-1)])
    meilleur = None
    for d in (depart, depart+np.r_[0.1, np.zeros(len(depart)-1)]):
        r = minimize(lambda p: nll_gpd(p[0], np.exp(z@p[1:]), x), d, method='Nelder-Mead',
                     options=dict(xatol=1e-7, fatol=1e-9, maxiter=20000, maxfev=20000))
        if meilleur is None or r.fun < meilleur.fun:
            meilleur = r
    return meilleur.x


def partie_severite(df, rng, b):
    ex = df[(df['y'] == 1) & (df['year'] <= DERNIERE_ANNEE_COMPLETE)].copy()
    ex['x'] = ex['perte']-U
    scores = {'D': [], 'DX': []}
    annees_notees = []
    for t in range(PREMIERE_ANNEE_NOTEE, DERNIERE_ANNEE_COMPLETE+1):
        app, test = ex[ex['year'] < t], ex[ex['year'] == t]
        if len(test) == 0:
            continue
        c_t = app['year'].mean()
        med = app['log_actifs'].median()
        mu_a = app['log_actifs'].fillna(med).mean()
        def z(d, avec):
            cols = [np.ones(len(d)), d['year'].to_numpy(float)-c_t]
            if avec:
                cols += [d['log_actifs'].fillna(med).to_numpy(float)-mu_a, d['bdsf'].to_numpy(float)]
            return np.column_stack(cols)
        for m, avec in (('D', False), ('DX', True)):
            p = ajuster_echelle(app['x'].to_numpy(float), z(app, avec))
            sig = np.exp(z(test, avec)@p[1:])
            scores[m].extend(genpareto.logpdf(test['x'].to_numpy(float), c=p[0], scale=sig).tolist())
        annees_notees.extend([t]*len(test))
    sD, sDX, an = np.array(scores['D']), np.array(scores['DX']), np.array(annees_notees)
    nuls = {m: int(np.sum(~np.isfinite(v))) for m, v in (('D', sD), ('DX', sDX))}
    fini = np.isfinite(sD) & np.isfinite(sDX)
    d = sDX[fini]-sD[fini]
    bt = bootstrap_annees(d, an[fini], rng, b)
    # Ajustement complet, coefficients descriptifs.
    c_t = ex['year'].mean()
    med = ex['log_actifs'].median()
    la = ex['log_actifs'].fillna(med)
    zc = np.column_stack([np.ones(len(ex)), ex['year']-c_t, la-la.mean(), ex['bdsf']]).astype(float)
    pc = ajuster_echelle(ex['x'].to_numpy(float), zc)
    return dict(n_notes=len(sD), densite_nulle=nuls, n_communs=int(fini.sum()),
                score_D=float(sD[fini].mean()), score_DX=float(sDX[fini].mean()),
                difference=float(d.mean()), p5=float(np.percentile(bt, 5)), p95=float(np.percentile(bt, 95)),
                complet=dict(n_exces=len(ex), xi=float(pc[0]), pente_annee=float(pc[2]),
                             effet_log_actifs=float(pc[3]), effet_bdsf=float(pc[4])))


def evaluer(df, col_actifs, rng, b, verbeux):
    preds = {m: [] for m in MODELES}
    ys, ans, firmes_vues = [], [], 0
    for t in range(PREMIERE_ANNEE_NOTEE, DERNIERE_ANNEE_COMPLETE+1):
        app, test = df[df['year'] < t], df[df['year'] == t]
        p, _, _ = predire(app, test, col_actifs)
        for m in MODELES:
            preds[m].extend(p[m].tolist())
        ys.extend(test['y'].tolist())
        ans.extend([t]*len(test))
        firmes_vues += int(test['firme'].isin(set(app['firme'])).sum())
        if verbeux:
            print(f"  {t} : {len(test):3d} pertes, {int(test['y'].sum()):2d} dépassements ; p moyen "
                  + ' ; '.join(f"{m} {p[m].mean():.3f}" for m in MODELES))
    y, ans = np.array(ys, float), np.array(ans)
    res = {}
    for m in MODELES:
        p = np.array(preds[m])
        ls = y*np.log(p)+(1-y)*np.log(1-p)
        res[m] = dict(log_score=float(ls.mean()), brier=float(np.mean((p-y)**2)),
                      pr_auc=float(average_precision_score(y, p)), p_moyen=float(p.mean()),
                      taux_observe=float(y.mean()), pente_calibration=pente_calibration(y, p), _ls=ls)
    for m in ('M0', 'M2', 'M3'):
        d = res[m]['_ls']-res['M1']['_ls']
        bt = bootstrap_annees(d, ans, rng, b)
        res[m]['contre_M1'] = dict(difference=float(d.mean()), p5=float(np.percentile(bt, 5)),
                                   p95=float(np.percentile(bt, 95)),
                                   annees_gagnees=int(sum(d[ans == a].sum() > 0 for a in np.unique(ans))),
                                   annees=int(len(np.unique(ans))))
    for m in MODELES:
        res[m].pop('_ls')
    return res, dict(n_notes=int(len(y)), n_depassements=int(y.sum()), pertes_firme_deja_vue=firmes_vues)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sortie', type=Path, default=REPO/'resultats/institut_approfondissements/131.json')
    parser.add_argument('--bootstrap', type=int, default=4000)
    args = parser.parse_args()
    debut = time.monotonic()
    rng = np.random.default_rng(GRAINE)
    df = charger()
    complet = df[df['year'] <= DERNIERE_ANNEE_COMPLETE]
    print('131 — LES CARACTÉRISTIQUES OBSERVÉES AMÉLIORENT-ELLES LA PRÉVISION DES PERTES IMPORTANTES ?')
    print(f'{len(df)} pertes 2000-2026, {int(df["y"].sum())} au-dessus de {U:.2f} M EUR ; '
          f'2026 incomplète exclue ({int((df["year"] == 2026).sum())} pertes).')
    print('Cible : dépassement du seuil PARMI LES PERTES ENREGISTRÉES, probabilité conditionnelle à l’enregistrement.')
    print(f'scikit-learn {sklearn.__version__} ; hyperparamètres fixés avant évaluation : logistique L2 C = {LOGIT_C} ; '
          f'boosting profondeur {BOOST["max_depth"]}, {BOOST["max_iter"]} itérations, pas {BOOST["learning_rate"]}, '
          f'feuilles d’au moins {BOOST["min_samples_leaf"]}.')

    n_act = int(complet['log_actifs'].notna().sum())
    n_post = int((complet['fin_posterieure'] == 1).sum())
    print(f'\nDISPONIBILITÉ DES ACTIFS : renseignés pour {n_act} pertes sur {len(complet)} ; '
          f'information financière postérieure à l’année de l’événement pour {n_post} pertes '
          f'({100*n_post/len(complet):.0f} %). Sensibilités : modèles sans les actifs, et actifs connus '
          f'au plus tard l’année de l’événement ({int(complet["log_actifs_ex_ante"].notna().sum())} pertes).')
    ecart = (complet['annee_fin']-complet['year']).dropna()
    print(f'  écart entre année d’information financière et année de l’événement : médiane {ecart.median():.0f} ans, '
          f'de {ecart.min():.0f} à {ecart.max():.0f} ; même année {100*(ecart == 0).mean():.0f} %, '
          f'antérieure {100*(ecart < 0).mean():.0f} %, postérieure {100*(ecart > 0).mean():.0f} %.')
    taux = complet.groupby('y')['log_actifs'].median()
    print(f'  médiane du log des actifs : {taux.loc[0]:.2f} sous le seuil, {taux.loc[1]:.2f} au-dessus '
          f'(soit {np.exp(taux.loc[0]):,.0f} et {np.exp(taux.loc[1]):,.0f} M USD).')

    print(f'\nPARTIE 1. ORIGINE GLISSANTE {PREMIERE_ANNEE_NOTEE}-{DERNIERE_ANNEE_COMPLETE}, actifs tels que déclarés')
    res, info = evaluer(complet, 'log_actifs', rng, args.bootstrap, True)
    res_sens, _ = evaluer(complet, None, rng, args.bootstrap, False)
    print(f"\n{info['n_notes']} pertes notées, {info['n_depassements']} dépassements ; "
          f"{info['pertes_firme_deja_vue']} pertes notées viennent d’une firme déjà présente dans l’apprentissage.")
    print('Scores par perte notée (log-score : plus grand = mieux ; Brier : plus petit = mieux)')
    print(f"{'modèle':42s} {'log-score':>10s} {'Brier':>8s} {'PR-AUC':>7s} {'p moyen':>8s} {'pente cal.':>10s}")
    for m in MODELES:
        r = res[m]
        print(f"{m+' '+NOMS[m]:42s} {r['log_score']:10.4f} {r['brier']:8.4f} {r['pr_auc']:7.3f} "
              f"{r['p_moyen']:8.3f} {r['pente_calibration']:10.2f}")
    print(f"  taux observé sur les années notées : {res['M0']['taux_observe']:.3f}")
    print('\nDIFFÉRENCE DE LOG-SCORE CONTRE M1 (bootstrap des années notées, intervalle à 90 %)')
    for nom, rr in (('actifs déclarés', res), ('sans actifs', res_sens)):
        for m in ('M0', 'M2', 'M3'):
            c = rr[m]['contre_M1']
            print(f"  {nom:28s} {m} moins M1 : {c['difference']:+.4f} [{c['p5']:+.4f} ; {c['p95']:+.4f}] ; "
                  f"années où {m} fait mieux : {c['annees_gagnees']} sur {c['annees']}")

    # Coefficients descriptifs de M2 sur toutes les années complètes.
    _, lr, cols = predire(complet, complet.iloc[:1], 'log_actifs')
    print('\nM2 AJUSTÉ SUR 2000-2025, coefficients standardisés (descriptifs, pas une inférence)')
    coefs = dict(zip(cols, lr.coef_[0].tolist()))
    for c, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f'  {c:32s} {v:+.3f}')

    print(f'\nPARTIE 2. SÉVÉRITÉ DES EXCÈS : échelle en tendance (D) contre tendance + log actifs + catégorie (DX)')
    sev = partie_severite(df, rng, args.bootstrap)
    print(f"  {sev['n_notes']} excès notés ; densité nulle attribuée à un excès observé : D {sev['densite_nulle']['D']}, "
          f"DX {sev['densite_nulle']['DX']} (forme estimée négative, queue bornée sous l’excès observé).")
    print(f"  sur les {sev['n_communs']} excès où les deux densités sont positives : log-score moyen D {sev['score_D']:.4f}, "
          f"DX {sev['score_DX']:.4f} ; DX moins D {sev['difference']:+.4f} [{sev['p5']:+.4f} ; {sev['p95']:+.4f}]")
    c = sev['complet']
    print(f"  ajustement complet ({c['n_exces']} excès) : xi {c['xi']:.3f} ; log sigma : année {c['pente_annee']:+.4f}, "
          f"log actifs {c['effet_log_actifs']:+.3f}, interruption d’activité {c['effet_bdsf']:+.3f}")

    # Calculée après la partie 2 pour ne pas déplacer les tirages des résultats précédents.
    res_ex, _ = evaluer(complet, 'log_actifs_ex_ante', rng, args.bootstrap, False)
    print('\nSENSIBILITÉ : ACTIFS CONNUS AU PLUS TARD L’ANNÉE DE L’ÉVÉNEMENT (manquants sinon)')
    for m in ('M2', 'M3'):
        c = res_ex[m]['contre_M1']
        print(f"  {m} moins M1 : {c['difference']:+.4f} [{c['p5']:+.4f} ; {c['p95']:+.4f}] ; "
              f"années où {m} fait mieux : {c['annees_gagnees']} sur {c['annees']}")

    signaux = signaux_covariables(dict(actifs_declares=res, sans_actifs=res_sens, actifs_ex_ante=res_ex))
    signal = any(gain for modele in signaux.values() for gain in modele.values())
    print('\nCONCLUSION DÉRIVÉE DES INTERVALLES')
    if signal:
        print('  au moins un modèle à caractéristiques bat la tendance seule avec un intervalle à 90 % au-dessus de zéro.')
    else:
        print('  aucun modèle à caractéristiques ne bat la tendance seule avec un intervalle à 90 % au-dessus de zéro.')
    sev_signal = sev['p5'] > 0 and sev['densite_nulle']['DX'] == 0
    print('  sévérité : ' + ('les covariables améliorent le score hors apprentissage.' if sev_signal else
                             'les covariables n’améliorent pas le score hors apprentissage de façon résolue.'))

    sortie = dict(experience='131 benchmark covariables', seuil=U, graine=GRAINE, sklearn=sklearn.__version__,
                  hyperparametres=dict(logit_C=LOGIT_C, boosting=BOOST), disponibilite_actifs=dict(
                      renseignes=n_act, posterieurs=n_post, total=len(complet)),
                  partie1=dict(info=info, actifs_declares=res, sans_actifs=res_sens, actifs_ex_ante=res_ex,
                                coefficients_M2=coefs),
                  partie2=sev, signal_depassement=signal, signal_depassement_par_variante=signaux,
                  signal_severite=sev_signal,
                  duree_secondes=time.monotonic()-debut)
    json.dump(sortie, open(args.sortie, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\nLimites : registre d’événements déclarés, pas d’exposition ; probabilité conditionnelle à l’enregistrement ;')
    print('douze années notées, donc douze blocs de bootstrap ; firmes répétées entre apprentissage et test ;')
    print('aucune modification du moteur de capital ni des valeurs publiées.')
    print(f'Durée {sortie["duree_secondes"]:.1f} s ; résultats agrégés : {args.sortie.name}')


if __name__ == '__main__':
    main()
