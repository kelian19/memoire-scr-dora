#!/usr/bin/env python3
"""119 : unité de perte et capital annuel ; expérience distincte pour l'Institut."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import nbinom

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
for chemin in (REPO, LAB):
    sys.path.insert(0, str(chemin))
import institut_incertitude as ii
import canaux_conformite as cx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sortie', type=Path, default=REPO/'resultats/institut_approfondissements/119.json')
    args = parser.parse_args()
    debut = time.monotonic()
    valeurs, diagnostics = ii.calculer_configurations(ii.REFERENCE, step=2)
    resultats = {regle: ii.contrastes(v) for regle, v in valeurs.items()}
    print('119 — UNITÉ DE PERTE ET CAPITAL ANNUEL, VERSION INSTITUT', flush=True)
    print('Montants en M EUR du périmètre financier étudié. Paramètres historiques inchangés.')
    print('Comparaison de scénarios de modèle ; aucune règle de perte n’est validée par la donnée.')
    print('Calcul par Fourier incliné ; pas 2 M EUR ; 65536 cases ; inclinaison 16.')
    print('Aucune renormalisation, aucun dépôt de queue dans la dernière case.')
    print('règle           conforme     non conforme       écart        rapport      interaction')
    for regle, r in resultats.items():
        print(f'{regle:14} {r["conforme"]:10.1f} {r["non_conforme"]:14.1f}'
              f' {r["ecart"]:12.1f} {r["rapport"]:12.4f} {r["interaction"]:14.1f}')
        print('  Shapley : ' + ', '.join(f'{k} {v:.1f}' for k, v in r['shapley'].items()))
    print('\nDÉFINITION : Y vaut zéro ou u+GPD, K est le nombre de piliers touchés.')
    print('additive : somme de K Y indépendants ; moyenne : somme/K ; unique : un seul Y ;')
    print('amplifiee : K*Y unique (exposant 1 posé). Ces deux dernières partagent une seule Bernoulli.')
    print('La moyenne conserve K Bernoulli et peut être sous u ; elle ne rétablit pas l’unité observée.')
    print('L’incident unique neutralise financièrement propagation et accumulation par construction.')
    print('Il ne réfute pas ces canaux : leur effet exige un lien documenté entre coût et piliers touchés.')
    print('\nPROBABILITÉ DE PERTE POSITIVE, MOYENNES ANALYTIQUES')
    for regle in ii.REGLES:
        for mask, etat in ((0, 'C'), (15, 'NC')):
            loi = ii.cardinal(bool(mask & 4), bool(mask & 8))
            pu = ii.REFERENCE.pu*(1.2 if mask & 2 else .85)
            lam = cx.LAM_NC if mask & 1 else cx.LAM_C
            positive = ii.probabilite_positive(loi, pu, regle)
            print(f'{regle:14} {etat:2} P(perte>0)={positive:.6f}'
                  f' incidents positifs/an={lam*positive:.4f}'
                  f' moyenne annuelle={lam*ii.moyenne_incident(ii.REFERENCE, pu, loi, regle):.4f}')
    print('Une perte positive n’implique pas une perte supérieure à u pour la moyenne.')

    print('\nCONTRÔLES NUMÉRIQUES', flush=True)
    controles = []
    for titre, step, n, tilt in [('pas divisé par deux', 1, 1 << 17, 16),
                                  ('portée doublée', 2, 1 << 17, 16),
                                  ('inclinaison 12', 2, 1 << 16, 12),
                                  ('inclinaison 18', 2, 1 << 16, 18)]:
        v, d = ii.calculer_configurations(ii.REFERENCE, masks=(0, 15), step=step, n=n, tilt=tilt)
        ecart_max = max(abs(v[r][m]-valeurs[r][m]) for r in ii.REGLES for m in ('0', '15'))
        print(f'{titre} : déplacement maximal {ecart_max:.3f} M EUR', flush=True)
        if ecart_max > 12:
            raise ArithmeticError('Convergence de grille insuffisante')
        controles.append(dict(controle=titre, ecart_max=ecart_max, diagnostics=d))

    # K=1 : les quatre constructions doivent avoir la même loi annuelle.
    seul = np.array([0., 1., 0., 0., 0., 0.])
    moteur = ii.FourierIncidents()
    temoins = [moteur.capital(0, r, loi_k=seul)[0] for r in ii.REGLES]
    np.testing.assert_allclose(temoins, temoins[0], atol=1e-9)
    # Moments à la référence : intégrale tronquée <= moment total connu.
    for d in diagnostics:
        mask, regle = d['mask'], d['regle']
        loi = ii.cardinal(bool(mask & 4), bool(mask & 8))
        pu = ii.REFERENCE.pu*(1.2 if mask & 2 else .85)
        lam = cx.LAM_NC if mask & 1 else cx.LAM_C
        analytique = lam*ii.moyenne_incident(ii.REFERENCE, pu, loi, regle)
        if not 0 < d['moyenne_tronquee'] < analytique*1.02:
            raise ArithmeticError('Moment tronqué incompatible avec le moment analytique')
    print(f'K=1 : les quatre constructions donnent {temoins[0]:.1f} M EUR.')
    # Limite déterministe : une perte non nulle vaut u, donc NB amincie exacte.
    limite = ii.Calibration(0., 1e-10, .15, 20.)
    moteur_limite = ii.FourierIncidents(limite)
    observe = moteur_limite.capital(0, 'unique', loi_k=seul)[0]
    r = cx.LAM_C/(cx.ec.PHI-1)
    p = 1/cx.ec.PHI
    pu = .85*limite.pu
    attendu = 20*nbinom.ppf(.995, r, p/(p+(1-p)*pu))
    np.testing.assert_allclose(observe, attendu, atol=1e-8)
    print(f'Limite de sévérité déterministe : Fourier {observe:.1f}, NB amincie exacte {attendu:.1f}.')
    print('Moments tronqués compatibles avec les moyennes analytiques finies.')
    print('Le contrôle de moyenne doit laisser subsister la contribution de la queue hors portée.')

    mc = {}
    for mask, etat in ((0, 'conforme'), (15, 'non conforme')):
        q = cx.scr_par_graine(cx.LAM_NC if mask else cx.LAM_C,
                             cx.G_NC if mask else cx.G_C,
                             cx.PU_NC if mask else cx.PU_C,
                             cx.PHICS_NC if mask else None)
        moyenne, sd = float(q.mean()), float(q.std(ddof=1))
        fft = valeurs['additive'][str(mask)]
        mc[etat] = dict(moyenne=moyenne, ecart_type_graines=sd, fft=fft,
                       ecart_en_erreurs_types=(fft-moyenne)/(sd/np.sqrt(len(q))))
        print(f'Référence additive {etat} : Fourier {fft:.1f}, MC {moyenne:.1f},'
              f' écart-type entre graines {sd:.1f}, écart {mc[etat]["ecart_en_erreurs_types"]:.3f} erreurs types.')
    sortie = dict(experience='119', calibration=ii.REFERENCE.__dict__, configurations=valeurs,
                  resultats=resultats, diagnostics=diagnostics, controles=controles, monte_carlo=mc,
                  duree_secondes=time.monotonic()-debut)
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    args.sortie.write_text(json.dumps(sortie, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Durée {sortie["duree_secondes"]:.1f} s ; résultats agrégés : {args.sortie.name}')


if __name__ == '__main__':
    main()
