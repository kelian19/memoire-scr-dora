#!/usr/bin/env python3
"""129 : KS/AD réestimés et estimateurs avec IC95 sur les 570 pertes de calibration.

Complément après le point Hugo du 22 octobre ; aucune constante gelée modifiée.
Les résultats du script 119_seuil_ks... concernent la population étendue 582/91.
Le numéro 129 évite aussi de confondre ce complément avec 119_unite_perte.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genpareto

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
for p in (REPO, LAB):
    sys.path.insert(0, str(p))
import institut_seuils as s
from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gof', type=int, default=1000)
    p.add_argument('--estimations', type=int, default=2000)
    args = p.parse_args(); debut = time.monotonic()
    df = filter_finance(filter_cyber(load_clean(REPO/'data/raw/SAS_OpRisk_Global_Data_June_2026.xlsx')))
    pertes = np.sort(df.loc[df.year.between(2000, 2026), 'loss'].to_numpy()*USD_EUR)
    u = 20.03; exces = pertes[pertes > u]-u
    if len(pertes) != 570 or len(exces) != 86:
        raise ValueError('Population de calibration différente du protocole')
    out = dict(experience='129', statut='en_cours', population='OpRisk finance cyber, années 2000–2026',
               n=570, n_exces=86, u=u, p_u_empirique=86/570, p_u_gele=.1509,
               B_gof=args.gof, B_estimateurs=args.estimations, graine=20261022,
               balayage=[], estimateurs={}, echecs=[], avertissements=[])
    print('129 — SEUIL POT ET ESTIMATEURS, POPULATION DE CALIBRATION 2000–2026', flush=True)
    print('570 pertes ; 86 excès ; u=20.03 M EUR. Base étendue de 119 : 582 pertes et 91 excès.')
    print(f"Protocole sauvegardé dans 129.json : B_gof={args.gof} ; B_estimateurs={args.estimations} ; graine=20261022.")
    print('KS bilatéral et AD : bootstrap paramétrique avec réestimation à seuil fixé.')
    print('Les tests écartent des seuils ; ils ne prouvent pas un optimum ni la validité du quantile annuel.')
    print('Seuil   n_exc  xi     sigma    IC95_profil_xi     D_KS   p_KS   p_AD', flush=True)
    rng = np.random.default_rng(20261022)
    seuils = sorted(set([u]+[float(np.percentile(pertes, q)) for q in (75, 77.5, 80, 82.5, 87.5, 90, 92.5, 95)]))
    for seuil in seuils:
        e = pertes[pertes > seuil]-seuil; xi, sig = s.emv(e)
        d, a = s.ks_stat(e, xi, sig), s.ad_stat(e, xi, sig)
        ds, ads = [], []
        with warnings.catch_warnings(record=True) as avis:
            warnings.simplefilter('always')
            for b in range(args.gof):
                sim = genpareto.rvs(xi, scale=sig, size=len(e), random_state=rng)
                try:
                    xb, sb = s.emv(sim)
                    ds.append(s.ks_stat(sim, xb, sb)); ads.append(s.ad_stat(sim, xb, sb))
                except (ValueError, ArithmeticError, FloatingPointError) as err:
                    out['echecs'].append(dict(etape='gof', u=seuil, replication=b, raison=str(err)))
            ic = s.profil_xi(e)
        out['avertissements'] += sorted(set(str(w.message) for w in avis))
        row = dict(u=seuil, n_exces=len(e), xi=xi, sigma=sig, ic95_profil=ic, D=d,
                   p_ks=(1+int((np.array(ds) >= d).sum()))/(len(ds)+1),
                   p_ad=(1+int((np.array(ads) >= a).sum()))/(len(ads)+1), succes=len(ds))
        out['balayage'].append(row)
        print(f"{seuil:.2f} {len(e)} {xi:.6f} {sig:.6f} [{ic[0]:.6f};{ic[1]:.6f}] {d:.6f} {row['p_ks']:.6f} {row['p_ad']:.6f}", flush=True)
    fonctions = {'EMV': s.emv, 'PWM': s.pwm,
                 'Hill': lambda e: (s.hill(e, u), None), 'Moments': lambda e: (s.moments(e, u), None)}
    indices = rng.integers(0, len(exces), size=(args.estimations, len(exces)))
    for nom, f in fonctions.items():
        xi, sig = f(exces); valeurs = []; echec = []
        for b, idx in enumerate(indices):
            try:
                xb, sb = f(exces[idx])
                if not np.isfinite(xb):
                    raise ArithmeticError('Indice non fini')
                valeurs.append([xb, sb, s.quantile(u, xb, sb, .1509) if sb is not None else None])
            except (ValueError, ArithmeticError, FloatingPointError) as err:
                echec.append(dict(etape=nom, replication=b, raison=str(err)))
        out['echecs'] += echec
        row = dict(xi=xi, sigma=sig, xi_ic95=np.quantile([v[0] for v in valeurs], [.025,.975]).tolist(),
                   sigma_ic95=np.quantile([v[1] for v in valeurs], [.025,.975]).tolist() if sig is not None else None,
                   q=s.quantile(u, xi, sig, .1509) if sig is not None else None,
                   q_ic95=np.quantile([v[2] for v in valeurs], [.025,.975]).tolist() if sig is not None else None,
                   succes=len(valeurs), xi_ge_un=sum(v[0] >= 1 for v in valeurs))
        out['estimateurs'][nom] = row
        print(f"{nom} xi={xi:.6f} IC95=[{row['xi_ic95'][0]:.6f};{row['xi_ic95'][1]:.6f}] sigma={sig} q={row['q']} sigmaIC={row['sigma_ic95']} qIC={row['q_ic95']} xi>=1={row['xi_ge_un']}", flush=True)
    out['avertissements'] = sorted(set(out['avertissements']))
    out['statut'] = 'termine' if not out['echecs'] else 'termine_avec_echecs'
    out['duree_secondes'] = time.monotonic()-debut
    dest = REPO/'resultats/institut_approfondissements/129.json'
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    rows = out['balayage']; us = np.array([r['u'] for r in rows])
    axes[0].plot(us, [r['xi'] for r in rows], 'o-', color='#223e55')
    axes[0].fill_between(us, [r['ic95_profil'][0] for r in rows], [r['ic95_profil'][1] for r in rows], alpha=.15)
    axes[0].set(xlabel='Seuil u (M EUR)', ylabel='Indice de queue ξ', title='EMV et intervalle profilé à 95 %')
    for cle, label in [('p_ks', 'KS'), ('p_ad', 'AD')]:
        axes[1].plot(us, [r[cle] for r in rows], 'o-', label=label)
    axes[1].axhline(.05, ls=':', color='grey'); axes[1].legend()
    axes[1].set(xlabel='Seuil u (M EUR)', ylabel='p-value calibrée', ylim=(0,1.02), title='Tests à seuil fixé avec réestimation')
    for ax in axes:
        ax.axvline(u, color='#b10031', ls='--', lw=1); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(LAB/'figures_inst/G129_seuils_hugo.png', dpi=190); plt.close(fig)
    print(f"Statut {out['statut']} ; échecs {len(out['echecs'])} ; durée {out['duree_secondes']:.1f} s", flush=True)
    if out['echecs']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
