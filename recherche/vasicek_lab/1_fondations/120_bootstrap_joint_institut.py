#!/usr/bin/env python3
"""120 : bootstrap conjoint des pertes entières, conditionnel aux hypothèses du moteur."""
import argparse
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from scipy.stats import genpareto

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
for chemin in (REPO, LAB):
    sys.path.insert(0, str(chemin))
import institut_incertitude as ii
from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR


def ajuster(pertes, u):
    exces = pertes[pertes > u]-u
    if len(exces) < 10:
        raise ValueError('Moins de dix excès : ajustement non exploitable')
    with warnings.catch_warnings(record=True) as avertissements:
        warnings.simplefilter('always')
        xi, loc, sigma = genpareto.fit(exces, floc=0)
    cal = ii.Calibration(float(xi), float(sigma), len(exces)/len(pertes), u)
    cal.verifier()
    if loc != 0 or not np.isfinite(genpareto.logpdf(exces, c=xi, scale=sigma)).all():
        raise ArithmeticError('Support ou vraisemblance inadmissible')
    return cal, len(exces), sorted(set(str(w.message) for w in avertissements))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replications', type=int, default=400)
    parser.add_argument('--graine', type=int, default=20261021)
    parser.add_argument('--sortie', type=Path, default=REPO/'resultats/institut_approfondissements/120.json')
    args = parser.parse_args()
    debut = time.monotonic()
    df = filter_finance(filter_cyber(load_clean(REPO/'data/raw/SAS_OpRisk_Global_Data_June_2026.xlsx')))
    df = df[df['year'].between(2000, 2026)]
    pertes = df['loss'].to_numpy(dtype=float)*USD_EUR
    cal, n_exces, avis = ajuster(pertes, ii.REFERENCE.u)
    if len(pertes) != 570 or n_exces != 86:
        raise ValueError(f'Échantillon différent du protocole : {len(pertes)} pertes, {n_exces} excès')
    central, central_diag = ii.calculer_configurations(cal, masks=(0, 15))
    print('120 — BOOTSTRAP CONJOINT CONDITIONNEL, VERSION INSTITUT', flush=True)
    print(f'Échantillon 2000–2026 : {len(pertes)} pertes, {n_exces} excès ; seuil {cal.u:.2f} M EUR.')
    print(f'Ajustement local : xi {cal.xi:.6f}, sigma {cal.sigma:.6f}, pu {cal.pu:.6f}.')
    print(f'{args.replications} réplications prévues ; graine {args.graine} ; percentile 95 % et 90 %.')
    print('Rééchantillonnage des pertes entières, puis ajustement conjoint xi, sigma et pu.')
    print('Chaque ajustement est partagé entre états et règles ; différences et rapports calculés par réplication.')
    print('Fréquence 582/27, dispersion 9,2, multiplicateurs et tables d’expert maintenus fixes.')
    print('Aucun écrêtage de xi, aucune moyenne des intervalles marginaux.')
    print('Ces bandes sont conditionnelles au seuil, à la stationnarité et aux choix de modèle.')
    print('La fréquence, sa discordance temporelle et l’incertitude d’expert ne sont pas propagées.', flush=True)
    out = dict(experience='120', methode='bootstrap iid des pertes entières ; percentiles conditionnels',
               n=len(pertes), n_exces=n_exces, replications_prevues=args.replications,
               graine=args.graine, calibration_locale=cal.__dict__, avertissements_fit_central=avis,
               central={r: ii.contrastes(v) for r, v in central.items()},
               central_diagnostics=central_diag, replications=[], echecs=[])
    args.sortie.parent.mkdir(parents=True, exist_ok=True)

    def enregistrer():
        out['duree_secondes'] = time.monotonic()-debut
        args.sortie.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    # Les graines par réplication permettent de rejouer isolément un cas extrême.
    for b in range(args.replications):
        rng = np.random.default_rng(np.random.SeedSequence([args.graine, b]))
        echantillon = rng.choice(pertes, size=len(pertes), replace=True)
        etape = 'ajustement'
        try:
            cb, nu, avis = ajuster(echantillon, cal.u)
            etape = 'agregation'
            valeurs, diagnostics = ii.calculer_configurations(cb, masks=(0, 15))
            out['replications'].append(dict(replication=b, calibration=cb.__dict__, n_exces=nu,
                avertissements=avis, resultats={r: ii.contrastes(v) for r, v in valeurs.items()},
                pas=diagnostics[0]['step'], portee=diagnostics[0]['portee'],
                cdf_portee_min=min(d['cdf_portee'] for d in diagnostics),
                masse_negative_max=max(d['masse_negative'] for d in diagnostics)))
        except (ValueError, ArithmeticError, FloatingPointError) as erreur:
            out['echecs'].append(dict(replication=b, etape=etape, erreur=str(erreur)))
        if (b+1) % 25 == 0 or b+1 == args.replications:
            enregistrer()
            print(f'Réplications {b+1}/{args.replications} ; succès {len(out["replications"])} ;'
                  f' échecs {len(out["echecs"])} ; {time.monotonic()-debut:.1f} s', flush=True)
    if not out['replications']:
        raise RuntimeError('Aucune réplication exploitable ; consulter le journal')
    params = np.array([[r['calibration'][p] for p in ('xi', 'sigma', 'pu')]
                       for r in out['replications']])
    out['correlation_parametres'] = np.corrcoef(params.T).tolist()
    out['ajustements_xi_superieur_egal_un'] = int((params[:, 0] >= 1).sum())
    out['resume'] = {}
    print('\nINTERVALLES PERCENTILES CONDITIONNELS, M EUR POUR LE PÉRIMÈTRE FINANCIER ÉTUDIÉ')
    for regle in ii.REGLES:
        out['resume'][regle] = {}
        for metrique in ('conforme', 'non_conforme', 'ecart', 'rapport'):
            v = np.array([r['resultats'][regle][metrique] for r in out['replications']])
            quantiles = np.quantile(v, [.025, .05, .5, .95, .975])
            out['resume'][regle][metrique] = dict(zip(('p025', 'p05', 'mediane', 'p95', 'p975'), quantiles.tolist()))
            print(f'{regle:12} {metrique:12} central={out["central"][regle][metrique]:.4f}'
                  f' médiane={quantiles[2]:.4f} 95%=[{quantiles[0]:.4f};{quantiles[4]:.4f}]'
                  f' 90%=[{quantiles[1]:.4f};{quantiles[3]:.4f}]')
    print(f'Corrélation bootstrap xi/sigma : {out["correlation_parametres"][0][1]:.4f}.')
    print(f'Ajustements xi >= 1 conservés : {out["ajustements_xi_superieur_egal_un"]}.')
    print(f'Échecs journalisés : {len(out["echecs"])}. Aucun tirage rejeté sur la taille du capital.')

    # Trois ajustements couvrant la plage de xi : sensibilité numérique hors bootstrap.
    ordre = np.argsort(params[:, 0])
    controles = []
    for index in sorted(set([int(ordre[0]), int(ordre[len(ordre)//2]), int(ordre[-1])])):
        rep = out['replications'][index]
        cb = ii.Calibration(**rep['calibration'])
        fin, diagnostics = ii.calculer_configurations(cb, masks=(0, 15),
            step=rep['pas']/2, n=1 << 17, tilt=18)
        relatif = max(abs(fin[r][str(m)]/rep['resultats'][r]['conforme' if m == 0 else 'non_conforme']-1)
                      for r in ii.REGLES for m in (0, 15))
        controles.append(dict(replication=rep['replication'], xi=cb.xi, ecart_relatif_max=relatif,
                             diagnostics=diagnostics))
        print(f'Contrôle numérique réplication {rep["replication"]}, xi={cb.xi:.4f} :'
              f' déplacement maximal {100*relatif:.4f} %.')
        if relatif > .005:
            raise ArithmeticError('Erreur numérique trop grande devant la bande bootstrap')
    out['controles_numeriques'] = controles
    out['statut'] = 'termine' if not out['echecs'] else 'termine_avec_echecs_a_examiner'
    enregistrer()
    print(f'Durée {out["duree_secondes"]:.1f} s ; données individuelles non exportées.', flush=True)


if __name__ == '__main__':
    main()
