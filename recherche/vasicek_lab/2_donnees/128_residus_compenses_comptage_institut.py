#!/usr/bin/env python3
"""128 : résidus compensés d'un processus de comptage, et structure temporelle non expliquée.

Pour un comptage journalier N_t d'intensité prévisible lambda_t (conditionnelle au passé strict), le
processus compensé M_t = N_t - lambda_t est une différence de martingale si l'intensité est bien spécifiée.
Les résidus compensés servent alors de diagnostic : autocorrélation résiduelle, dispersion, dérive. Mesure
physique ; aucun prix, aucun actif répliqué.

Modèles comparés sur la chronologie PRC du secteur financier (BSF), 2016-2023, dates au jour :
  M0 : tendance log-linéaire seule ;
  M1 : tendance et excitation interne exponentielle des jours antérieurs (pas de même jour) ;
  M2 : tendance et excitation externe, nombre de vulnérabilités ajoutées au catalogue CISA KEV pendant les
       sept jours antérieurs (fenêtre 2022-2023 seulement, après le lot initial de novembre 2021).
Les paramètres étant estimés sur les mêmes données, la loi des statistiques de diagnostic change : les
seuils sont recalibrés par bootstrap paramétrique avec réestimation à chaque tirage (comptes binomiaux
négatifs de dispersion estimée). La date d'ajout au catalogue KEV n'est pas la date de première
exploitation ; elle est auditée ici contre le pic MOVEit.
Un diagnostic compatible avec une martingale ne prouve ni causalité ni identification d'une contagion.
"""
import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.signal import lfilter

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
RAW = REPO/'data/raw'
Y0, Y1 = 2016, 2023
MOVEIT = (pd.Timestamp('2023-05-25'), pd.Timestamp('2023-06-05'))
KEV_DEBUT_EXPLOITABLE = pd.Timestamp('2022-01-01')
LAGS = 14
FENETRE_KEV = 7
GRAINE = 20261022


def lire_comptes():
    csv = RAW/'Data_Breach_Chronology.csv'
    if csv.exists():
        d = pd.read_csv(csv, sep='|', usecols=['breach_date', 'organization_type'], encoding='utf-8-sig',
                        low_memory=False)
    else:
        d = pd.read_excel(RAW/'Data_Breach_Chronology.xlsx', sheet_name='Data_Breach_Chronology',
                          usecols=['breach_date', 'organization_type'])
    d['bd'] = pd.to_datetime(d.breach_date.astype(str).str.strip(), errors='coerce')
    b = d[(d.organization_type == 'BSF') & d.bd.notna() & d.bd.dt.year.between(Y0, Y1)]
    return b


def comptes_journaliers(b, sans_moveit=False):
    if sans_moveit:
        b = b[~((b.bd >= MOVEIT[0]) & (b.bd <= MOVEIT[1]))]
    idx = pd.date_range(f'{Y0}-01-01', f'{Y1}-12-31', freq='D')
    return b.groupby(b.bd.dt.normalize()).size().reindex(idx, fill_value=0).to_numpy().astype(float)


def excitation(k, beta):
    """E_t = somme_{u>=1} exp(-beta u) k_{t-u}, strictement prévisible."""
    c = np.exp(-beta)
    return lfilter([0.0, c], [1.0, -c], k)


def intensite(modele, p, t, k, x=None):
    base = np.exp(p[0]+p[1]*t)
    if modele == 'M0':
        return base
    if modele == 'M1':
        return base+np.exp(p[2])*excitation(k, np.exp(p[3]))
    return np.exp(p[0]+p[1]*t+p[2]*x)


def nll(p, modele, t, k, x=None):
    lam = intensite(modele, p, t, k, x)
    if not np.isfinite(lam).all() or (lam <= 0).any():
        return 1e12
    return float(-(k*np.log(lam)-lam).sum())


def ajuster(modele, t, k, x=None, departs=None):
    nb = {'M0': 2, 'M1': 4, 'M2': 3}[modele]
    if departs is None:
        base = [np.log(max(k.mean(), 1e-3)), 0.0]
        departs = {'M0': [base], 'M1': [base+[np.log(.2), lb] for lb in (np.log(.7), np.log(2.), np.log(8.))],
                   'M2': [base+[0.0]]}[modele]
    meilleur = None
    for x0 in departs:
        r = optimize.minimize(nll, x0, args=(modele, t, k, x), method='Nelder-Mead',
                              options={'maxiter': 4000*nb, 'xatol': 1e-6, 'fatol': 1e-7})
        if meilleur is None or r.fun < meilleur.fun:
            meilleur = r
    return meilleur.x, float(meilleur.fun)


def diagnostics(modele, p, t, k, x=None):
    lam = intensite(modele, p, t, k, x)
    eps = k-lam
    npar = len(p)
    phi = float(np.sum(eps**2/lam)/(len(k)-npar))
    r = eps/np.sqrt(max(phi, 1e-9)*lam)
    acf = np.array([np.sum(r[:-l]*r[l:])/np.sum(r*r) for l in range(1, LAGS+1)])
    n = len(k)
    q = float(n*(n+2)*np.sum(acf**2/(n-np.arange(1, LAGS+1))))
    out = dict(phi=phi, Q=q, acf=acf.tolist(), derive=float(eps.sum()), lam_total=float(lam.sum()))
    if modele == 'M1':
        out['branchement'] = float(np.exp(p[2])/(np.exp(np.exp(p[3]))-1))
        out['demi_vie_jours'] = float(np.log(2)/np.exp(p[3]))
    if modele == 'M2':
        out['gamma'] = float(p[2])
    return out


def simuler(modele, p, t, phi, rng, x=None, k_init=None):
    n = len(t)
    phi = max(phi, 1.05)
    if modele in ('M0', 'M2'):
        lam = intensite(modele, p, t, np.zeros(n), x)
        r = lam/(phi-1)
        return rng.negative_binomial(r, 1/phi).astype(float)
    k = np.zeros(n)
    base = np.exp(p[0]+p[1]*t)
    alpha, c = np.exp(p[2]), np.exp(-np.exp(p[3]))
    e = 0.0
    for i in range(n):
        lam = base[i]+alpha*e
        k[i] = rng.negative_binomial(lam/(phi-1), 1/phi)
        e = c*(e+k[i])
    return k


def p_bootstrap(modele, p_hat, t, k, x, B, rng, observe, phi):
    q = []
    echecs = 0
    for _ in range(B):
        ks = simuler(modele, p_hat, t, phi, rng, x)
        try:
            ps, _ = ajuster(modele, t, ks, x, departs=[p_hat])
            q.append(diagnostics(modele, ps, t, ks, x)['Q'])
        except Exception:
            echecs += 1
    q = np.array(q)
    return dict(p=float((1+np.sum(q >= observe))/(1+len(q))), q95=float(np.percentile(q, 95)),
                mediane=float(np.median(q)), tirages=int(len(q)), echecs=echecs)


def rapport_vraisemblance_bootstrap(alt, p0, nll0, nll1, t, k, x, B, rng, phi):
    """Rapport de vraisemblance du modèle alt contre M0, loi nulle par bootstrap sous M0 avec réestimation."""
    obs = 2*(nll0-nll1)
    lr, branch = [], []
    for _ in range(B):
        ks = simuler('M0', p0, t, phi, rng)
        a0, n0 = ajuster('M0', t, ks)
        if alt == 'M1':
            deps = [np.r_[a0, np.log(.05), lb] for lb in (np.log(.7), np.log(3.))]
        else:
            deps = [np.r_[a0, 0.0]]
        a1, n1 = ajuster(alt, t, ks, x, departs=deps)
        lr.append(2*(n0-n1))
        if alt == 'M1':
            branch.append(float(np.exp(a1[2])/(np.exp(np.exp(a1[3]))-1)))
    lr = np.array(lr)
    out = dict(observe=float(obs), p=float((1+np.sum(lr >= obs))/(1+len(lr))), q95=float(np.percentile(lr, 95)))
    if branch:
        out['branchement_nul_q50'] = float(np.median(branch))
        out['branchement_nul_q95'] = float(np.percentile(branch, 95))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tirages', type=int, default=200)
    parser.add_argument('--sortie', type=Path, default=REPO/'resultats/institut_approfondissements/128.json')
    args = parser.parse_args()
    debut = time.monotonic()
    warnings.simplefilter('ignore')
    rng = np.random.default_rng(GRAINE)
    b = lire_comptes()
    print('128 — RÉSIDUS COMPENSÉS D’UN PROCESSUS DE COMPTAGE, VERSION INSTITUT')
    print(f'Chronologie PRC, secteur financier (BSF), dates au jour, {Y0}-{Y1} : {len(b)} événements, '
          f'{b.bd.dt.normalize().nunique()} jours distincts.')
    print('Résidu compensé : N_t moins l’intensité prévisible ; différence de martingale si l’intensité est bien spécifiée.')
    print(f'Seuils recalibrés par bootstrap paramétrique avec réestimation ({args.tirages} tirages, comptes binomiaux négatifs).')
    sortie = dict(experience='128 résidus compensés', evenements=int(len(b)), jours=int(b.bd.dt.normalize().nunique()),
                  tirages=args.tirages, graine=GRAINE, lags=LAGS, variantes={})

    # Contrôles de l'estimateur et du test sur données simulées.
    t = np.arange((pd.Timestamp(f'{Y1}-12-31')-pd.Timestamp(f'{Y0}-01-01')).days+1)/365.25
    p0 = np.array([np.log(.9), .25])
    nuls = []
    for _ in range(20):
        k_poisson = simuler('M0', p0, t, 1.5, rng)
        p1, _ = ajuster('M1', t, k_poisson, departs=[np.r_[ajuster('M0', t, k_poisson)[0], np.log(.05), lb] for lb in (np.log(.7), np.log(3.))])
        nuls.append(diagnostics('M1', p1, t, k_poisson)['branchement'])
    n_nul = float(np.median(nuls))
    pm1 = np.array([np.log(.7), .15, np.log(.45), np.log(.8)])
    k_h = simuler('M1', pm1, t, 1.5, rng)
    ph1, _ = ajuster('M1', t, k_h)
    n_vrai = float(np.exp(pm1[2])/(np.exp(np.exp(pm1[3]))-1))
    n_est = diagnostics('M1', ph1, t, k_h)['branchement']
    print('\nCONTRÔLES SUR DONNÉES SIMULÉES')
    print(f'  tendance seule simulée : branchement estimé par M1 médian {n_nul:.3f}, maximum {max(nuls):.3f} sur 20 simulations (biais de l’estimateur sous la loi nulle).')
    print(f'  excitation interne simulée de branchement {n_vrai:.3f} : estimé {n_est:.3f}.')
    sortie['controles'] = dict(branchement_nul_median=n_nul, branchement_nul_max=float(max(nuls)), branchement_vrai=n_vrai, branchement_estime=n_est)
    assert abs(n_est-n_vrai) < 0.15

    kev = None
    kev_chemin = RAW/'cisa_kev.json'
    for variante, sans in (('toutes les dates', False), ('sans la semaine MOVEit', True)):
        k = comptes_journaliers(b, sans)
        res = {}
        print(f'\nVARIANTE : {variante} ; {int(k.sum())} événements sur {len(k)} jours, maximum journalier {int(k.max())}.')
        print(f"  {'modèle':6s} {'dispersion':>10s} {'Q(14)':>8s} {'seuil 95 %':>11s} {'p bootstrap':>12s} "
              f"{'rho(1)':>7s} {'rho(2)':>7s} {'branchement':>12s}")
        for modele in ('M0', 'M1'):
            p, nl = ajuster(modele, t, k)
            if modele == 'M0':
                nll_m0 = nl
            d = diagnostics(modele, p, t, k)
            bs = p_bootstrap(modele, p, t, k, None, args.tirages, rng, d['Q'], d['phi'])
            res[modele] = dict(params=p.tolist(), **d, bootstrap=bs)
            if modele == 'M1':
                res['rapport_vraisemblance'] = rapport_vraisemblance_bootstrap(
                    'M1', np.array(res['M0']['params']), nll_m0, nl, t, k, None, args.tirages, rng, res['M0']['phi'])
            print(f"  {modele:6s} {d['phi']:10.2f} {d['Q']:8.1f} {bs['q95']:11.1f} {bs['p']:12.3f} "
                  f"{d['acf'][0]:+7.3f} {d['acf'][1]:+7.3f} {d.get('branchement', float('nan')):12.3f}", flush=True)
        lr = res['rapport_vraisemblance']
        print(f"  rapport de vraisemblance M0 contre M1 : {lr['observe']:.1f}, seuil bootstrap à 95 % {lr['q95']:.1f}, "
              f"p = {lr['p']:.3f} ; branchement estimé sous la loi nulle : médiane {lr['branchement_nul_q50']:.2f}, "
              f"95 % {lr['branchement_nul_q95']:.2f}.")
        sortie['variantes'][variante] = res

    # Excitation externe : vulnérabilités du catalogue KEV, fenêtre 2022-2023.
    print('\nEXCITATION EXTERNE : catalogue CISA KEV, fenêtre 2022-2023')
    if kev_chemin.exists():
        brut = kev_chemin.read_bytes()
        v = pd.DataFrame(json.loads(brut)['vulnerabilities'])
        v['d'] = pd.to_datetime(v.dateAdded)
        sortie['kev'] = dict(md5=hashlib.md5(brut).hexdigest()[:8], total=int(len(v)),
                             premier_lot=int((v.d == v.d.min()).sum()), debut=str(v.d.min().date()))
        print(f'  catalogue lu : {len(v)} entrées, premier lot du {v.d.min().date()} : {int((v.d == v.d.min()).sum())} '
              f'entrées d’un seul jour (reprise de stock, écartée) ; fenêtre retenue à partir du {KEV_DEBUT_EXPLOITABLE.date()}.')
        mv = v[v.cveID == 'CVE-2023-34362']
        if len(mv):
            retard = int((mv.d.iloc[0]-pd.Timestamp('2023-05-29')).days)
            sortie['kev']['moveit_retard_jours'] = retard
            print(f'  audit de date : CVE-2023-34362 (MOVEit) ajoutée le {mv.d.iloc[0].date()}, soit {retard} jours après le '
                  'pic du 29 mai 2023 : la date d’ajout suit l’événement, elle ne le précède pas.')
        idx = pd.date_range(f'{Y0}-01-01', f'{Y1}-12-31', freq='D')
        ajouts = v.groupby(v.d.dt.normalize()).size().reindex(idx, fill_value=0).to_numpy().astype(float)
        passe = np.array([ajouts[max(0, i-FENETRE_KEV):i].sum() for i in range(len(idx))])
        x_all = np.log1p(passe)
        fen = (idx >= KEV_DEBUT_EXPLOITABLE)
        for variante, sans in (('toutes les dates', False), ('sans la semaine MOVEit', True)):
            kf = comptes_journaliers(b, sans)[fen]
            tf = np.arange(fen.sum())/365.25
            xf = x_all[fen]
            res = {}
            print(f'  VARIANTE {variante} : {int(kf.sum())} événements sur {len(kf)} jours.')
            print(f"    {'modèle':6s} {'dispersion':>10s} {'Q(14)':>8s} {'seuil 95 %':>11s} {'p bootstrap':>12s} {'gamma':>7s} {'écart de log-vraisemblance':>28s}")
            p0f, nll0 = ajuster('M0', tf, kf)
            p2f, nll2 = ajuster('M2', tf, kf, xf)
            for modele, p, x_, nl in (('M0', p0f, None, nll0), ('M2', p2f, xf, nll2)):
                d = diagnostics(modele, p, tf, kf, x_)
                bs = p_bootstrap(modele, p, tf, kf, x_, args.tirages, rng, d['Q'], d['phi'])
                res[modele] = dict(params=p.tolist(), **d, bootstrap=bs, nll=nl)
                print(f"    {modele:6s} {d['phi']:10.2f} {d['Q']:8.1f} {bs['q95']:11.1f} {bs['p']:12.3f} "
                      f"{d.get('gamma', float('nan')):7.3f} {nll0-nl:28.2f}", flush=True)
            # Test du rapport de vraisemblance par bootstrap sous M0 (gamma réestimé).
            res['rapport_vraisemblance'] = rapport_vraisemblance_bootstrap(
                'M2', p0f, nll0, nll2, tf, kf, xf, args.tirages, rng, res['M0']['phi'])
            lrt_obs = res['rapport_vraisemblance']['observe']
            print(f"    rapport de vraisemblance M0 contre M2 : {lrt_obs:.2f}, seuil bootstrap à 95 % "
                  f"{res['rapport_vraisemblance']['q95']:.2f}, p = {res['rapport_vraisemblance']['p']:.3f}.")
            sortie['variantes'][f'KEV, {variante}'] = res
    else:
        print(f'  fichier {kev_chemin.name} absent de data/raw : volet externe non exécuté.')
        sortie['kev'] = dict(absent=True)

    sortie['duree_secondes'] = time.monotonic()-debut
    cle = sortie['variantes']
    cites = {}
    for var in ('toutes les dates', 'sans la semaine MOVEit'):
        cites[var] = dict(branchement=cle[var]['M1']['branchement'], p_m0=cle[var]['M0']['bootstrap']['p'],
                          p_m1=cle[var]['M1']['bootstrap']['p'], rho1_m0=cle[var]['M0']['acf'][0],
                          rho1_m1=cle[var]['M1']['acf'][0])
    sortie['grandeurs_citees'] = cites
    print('\nGRANDEURS CITÉES')
    for var, c in cites.items():
        print(f"{var} : branchement {c['branchement']:.2f} ; autocorrélation du premier retard {c['rho1_m0']:+.2f} sans excitation, "
              f"{c['rho1_m1']:+.2f} avec ; p bootstrap {c['p_m0']:.3f} puis {c['p_m1']:.3f}.")
    json.dump(sortie, open(args.sortie, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\nLimites : comptes de brèches déclarées au PRC, non une fréquence d’entités ; couverture déclarative croissante')
    print('absorbée par une tendance log-linéaire ; comptes binomiaux négatifs de dispersion constante comme loi nulle ;')
    print('puissance faible sur 730 jours pour l’excitation externe ; la date d’ajout au KEV n’est pas la date de première')
    print('exploitation ; un diagnostic compatible avec une martingale ne prouve ni causalité ni contagion.')
    print(f'Durée {sortie["duree_secondes"]:.0f} s ; résultats agrégés : {args.sortie.name}')


if __name__ == '__main__':
    main()
