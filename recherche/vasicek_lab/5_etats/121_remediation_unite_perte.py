#!/usr/bin/env python3
"""121 : pilote de remédiation par pilier sous quatre constructions de perte.

32 sous-ensembles d'actions, cinq directions sur le segment expert/transposée,
deux noyaux, avec/sans choc tiers, deux périmètres d'action. Les calibrations
bootstrap optionnelles sont des scénarios choisis par rang de xi, pas des bandes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
for chemin in (REPO, LAB):
    sys.path.insert(0, str(chemin))
import canaux_conformite as cx
import institut_incertitude as ii
import institut_remediation as ir


def calibrations(bootstrap):
    cas = [('historique', ii.REFERENCE, None)]
    if bootstrap:
        source = REPO/'resultats/institut_approfondissements/120.json'
        objet = json.loads(source.read_text(encoding='utf-8'))
        if objet['statut'] != 'termine':
            raise ValueError('Le bootstrap source doit être terminé sans échecs')
        ordre = sorted(objet['replications'], key=lambda r: r['calibration']['xi'])
        for p in (.05, .5, .95):
            r = ordre[round(p*(len(ordre)-1))]
            cas.append((f'bootstrap_rang_xi_{p:g}', ii.Calibration(**r['calibration']),
                        r['replication']))
    return cas


def calculer_contexte(moteur, w, noyau, phi, famille, efficacite, actes):
    valeurs = {r: [] for r in ii.REGLES}
    diags = []
    for action in actes:
        v, phi_action = ir.appliquer(w, action, efficacite, phi or 0., famille)
        loi = ir.cardinal(v, noyau, phi_action if phi is not None else None)
        for regle in ii.REGLES:
            capital, diag = moteur.capital(15, regle, loi_k=loi)
            valeurs[regle].append(capital)
            diags.append(diag)
    return valeurs, {
        'step': moteur.step, 'portee': moteur.step*moteur.n,
        'cdf_portee_min': min(d['cdf_portee'] for d in diags),
        'masse_negative_max': max(d['masse_negative'] for d in diags),
    }


def decisions(contextes, actes):
    groupes = {'tous_scenarios': [(c, r) for c in contextes for r in ii.REGLES]}
    for famille in ir.FAMILLES:
        groupes[f'famille_{famille}'] = [(c, r) for c in contextes for r in ii.REGLES
                                        if c['famille'] == famille]
    for regle in ii.REGLES:
        groupes[f'regle_{regle}'] = [(c, regle) for c in contextes]
        for famille in ir.FAMILLES:
            groupes[f'{regle}_{famille}'] = [(c, regle) for c in contextes if c['famille'] == famille]
    resultat = {}
    for nom, lignes in groupes.items():
        tableau = np.array([c['capitaux'][r] for c, r in lignes])
        resultat[nom] = [ir.decision(tableau, actes, budget, relatif)
                         for budget in (1, 2) for relatif in (False, True)]
    return resultat


def figure(contextes, actes, sortie):
    """Gains unitaires de référence ; pas de mélange des noyaux dans la figure."""
    noms = {'additive': 'Additive', 'moyenne': 'Moyenne', 'unique': 'Unique', 'amplifiee': 'Amplifiée'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, regle in zip(axes.flat, ii.REGLES):
        for f, marque, style in [('emission', 'o', '-'), ('emission_et_tiers', 's', '--')]:
            c = next(c for c in contextes if c['calibration'] == 'historique'
                     and c['direction_t'] == 1 and c['noyau'] == 'marche'
                     and c['accumulation'] and c['famille'] == f)
            q = np.array(c['capitaux'][regle]); gains = [q[0]-q[actes.index((j,))] for j in ir.PILIERS]
            ax.plot(ir.PILIERS, gains, marker=marque, ls=style, lw=1.5,
                    label='Émission seule' if f == 'emission' else 'Émission + choc tiers pour P4')
        ax.axhline(0, color='#444444', lw=.7)
        ax.set_title(noms[regle], loc='left', fontweight='bold')
        ax.set_xticks(ir.PILIERS, [f'P{j}' for j in ir.PILIERS])
        ax.set_ylabel('Capital libéré (M EUR)')
        ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Réduire de moitié l’émission d’un pilier : le gain dépend de la perte', fontsize=13)
    fig.text(.5, .015, 'Scénario : direction expert, marche, choc tiers actif, calage historique.\n'
             'Fréquence et détection fixes ; actions supposées, coûts non estimés.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95))
    fig.savefig(sortie, dpi=170)
    plt.close(fig)


def rapport(objet, actes, dossier):
    lignes = [
        '# Première expérience de la feuille de route : remédiation par pilier', '',
        'Résultats expérimentaux du script 121. Montants en M EUR pour le périmètre étudié.',
        'Ce pilote ne fournit pas une recommandation calibrée pour une entité réelle.', '',
        '## Protocole exécuté', '',
        f"- {len(objet['contextes'])} contextes ; quatre constructions de perte ; 32 sous-ensembles d'actions.",
        '- Cinq directions : S+tA, t dans {-1 ; -0,5 ; 0 ; 0,5 ; 1}, à S expert fixé.',
        '- Marche et branchement évalués séparément ; choc tiers absent ou actif.',
        '- Chaque action réduit de moitié la ligne émettrice du pilier, sans renormalisation.',
        '- Une variante réduit aussi phi de moitié si P4 est traité ; ce lien est supposé.',
        '- Fréquence non conforme, détection et poids d’amorce fixes.',
        '- Coût conventionnel : une unité par pilier. Budgets de une ou deux unités, abstention autorisée.',
        '- Les scénarios bootstrap, si présents, sont sélectionnés par rang de xi ; ils ne forment pas un intervalle.', '',
        '## Gains unitaires au point de référence', '',
        'Direction expert, marche et choc tiers actif. Un gain négatif signifie que le capital augmente.', '',
        '| Perte | Action | P1 | P2 | P3 | P4 | P5 |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for regle in ii.REGLES:
        for famille in ir.FAMILLES:
            c = next(c for c in objet['contextes'] if c['calibration'] == 'historique'
                     and c['direction_t'] == 1 and c['noyau'] == 'marche'
                     and c['accumulation'] and c['famille'] == famille)
            q = np.array(c['capitaux'][regle])
            gains = [q[0]-q[actes.index((j,))] for j in ir.PILIERS]
            lignes.append('| '+regle+' | '+famille+' | '+' | '.join(f'{v:,.0f}'.replace(',', ' ') for v in gains)+' |')
    lignes += ['', '## Décision minimax sur les scénarios examinés', '',
               'Le regret compare chaque action au meilleur choix sous le même budget dans chaque scénario.',
               'Le regret relatif est divisé par le capital sans action du scénario : c’est un autre objectif.',
               'Les égalités sont celles de la grille ; la précision est examinée ci-dessous.', '',
               '| Ensemble | Budget | Minimax absolu | Regret max (M EUR) | Minimax relatif | Regret max (%) |',
               '|---|---:|---|---:|---|---:|']
    def montrer_choix(choix):
        return ', '.join(choix) if len(choix) <= 4 else f'Toutes ({len(choix)} actions ex aequo)'

    for groupe, choix in objet['decisions'].items():
        for budget in (1, 2):
            a = next(d for d in choix if d['budget_unites'] == budget and d['objectif'] == 'M_EUR')
            r = next(d for d in choix if d['budget_unites'] == budget and d['objectif'] == 'relatif')
            lignes.append(f"| {groupe} | {budget} | {montrer_choix(a['actions_minimax'])} | "
                          f"{a['regret_minimal']:.1f} | {montrer_choix(r['actions_minimax'])} | {100*r['regret_minimal']:.2f} |")
    ctrl = objet['controles']
    lignes += ['', '## Contrôles et limites', '',
               f"- Référence 119 : écart maximal {ctrl['ecart_reference_119_max']:.4g} M EUR.",
               f"- Raffinement de grille sur les interventions unitaires : écart maximal {ctrl['raffinement_max_M_EUR']:.4g} M EUR.",
               f"- Inversions : masse annuelle conservée minimale {ctrl['cdf_portee_min']:.8f} ; masse négative maximale {ctrl['masse_negative_max']:.3g}.",
               '- La perte unique donne exactement le même capital après toutes ces actions.',
               '- Le raffinement porte sur les actions nulles et unitaires aux deux orientations extrêmes ; les paires restent sur la grille principale.',
               '- Les trois calibrations sélectionnées par rang de xi ne comprennent pas tous les ajustements extrêmes, notamment ceux avec xi supérieur ou égal à un.',
               '- Les tests indépendants retrouvent les lois historiques des ensembles, une bifurcation analytique et un minimax connu.',
               '- Cinq directions ne couvrent pas le pavé continu de dix paramètres : aucune robustesse globale n’est revendiquée.',
               '- Le traitement de P4 dans le choc commun est une hypothèse d’intervention distincte, qui doit être examinée séparément.',
               '- Une action sur la propagation ne traite pas les autres mécanismes de conformité : un résultat nul ne réfute pas l’utilité du pilier.',
               '- Les coûts et efficacités réels ne sont pas mesurés ; aucune probabilité n’est attribuée aux scénarios.', '',
               '## Suite', '',
               'Définir des coûts et effets d’intervention documentés, élargir les directions admissibles,',
               'puis propager davantage de calibrations conjointes et analyser les changements de décision.',
               'Les ordres de remédiation sur un horizon et la prédiction annuelle sont les étapes suivantes.', '']
    (dossier/'121_rapport.md').write_text('\n'.join(lignes), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', action='store_true', help='Ajouter trois calibrations de 120, choisies par rang de xi')
    parser.add_argument('--restituer', action='store_true', help='Régénérer décisions, rapport et figure depuis 121.json existant')
    parser.add_argument('--dossier', type=Path, default=REPO/'resultats/institut_janvier')
    args = parser.parse_args()
    debut = time.monotonic()
    actes = ir.actions()
    if args.restituer:
        if args.bootstrap:
            parser.error('--restituer et --bootstrap ne se combinent pas')
        objet = json.loads((args.dossier/'121.json').read_text(encoding='utf-8'))
        if objet.get('statut') != 'termine' or objet['actions'] != [list(a) for a in actes]:
            raise ValueError('Résultat incomplet ou actions différentes')
        objet['decisions'] = decisions(objet['contextes'], actes)
        (args.dossier/'121.json').write_text(json.dumps(objet, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        figure(objet['contextes'], actes, args.dossier/'121_priorites.png')
        rapport(objet, actes, args.dossier)
        print('Décisions, rapport et figure restitués depuis les capitaux enregistrés.')
        return
    objet = dict(experience='121', statut='en_cours', efficacite=.5,
                 actions=[list(a) for a in actes], contextes=[], calibrations=[],
                 perimetre='cinq directions sur segment expert/transposée ; scénarios, pas une loi de probabilité')
    args.dossier.mkdir(parents=True, exist_ok=True)
    grille = []
    for nom_cal, cal, rep in calibrations(args.bootstrap):
        step = 2. if rep is None else max(2., ii.grille_adaptee(cal))
        moteur = ii.FourierIncidents(cal, step=step)
        fin = ii.FourierIncidents(cal, step=step/2, n=1 << 17, tilt=18)
        objet['calibrations'].append(dict(nom=nom_cal, calibration=cal.__dict__, replication_source=rep))
        for nom_direction, t, w in ir.directions():
            for noyau in ir.NOYAUX:
                for accumulation in (False, True):
                    phi = cx.PHICS_NC if accumulation else None
                    for famille in ir.FAMILLES:
                        caps, diag = calculer_contexte(moteur, w, noyau, phi, famille, .5, actes)
                        c = dict(calibration=nom_cal, direction=nom_direction, direction_t=t,
                                 noyau=noyau, accumulation=accumulation, famille=famille,
                                 capitaux=caps, diagnostic=diag)
                        objet['contextes'].append(c)
                        if np.ptp(caps['unique']) != 0:
                            raise AssertionError('La propagation ne doit pas agir sur la perte unique')
                        for regle in ('additive', 'amplifiee'):
                            if max(caps[regle]) > caps[regle][0]+2*step:
                                raise ArithmeticError('Augmentation inattendue sous une action réductrice')
                        # Tester les actions unitaires à chaque calage, noyau et famille
                        # aux deux orientations extrêmes du segment, sans refaire les paires.
                        if abs(t) == 1:
                            petit = [(), *((j,) for j in ir.PILIERS)]
                            valeurs_fin, _ = calculer_contexte(fin, w, noyau, phi, famille, .5, petit)
                            erreur = max(abs(valeurs_fin[r][k]-caps[r][actes.index(a)])
                                         for r in ii.REGLES for k, a in enumerate(petit))
                            if erreur > max(3*step, .002*max(caps['amplifiee'])):
                                raise ArithmeticError('Raffinement insuffisant pour les gains unitaires')
                            grille.append(dict(calibration=nom_cal, noyau=noyau, accumulation=accumulation,
                                               famille=famille, direction_t=t, ecart_max=erreur))
            print(f'{nom_cal} {nom_direction} : {len(objet["contextes"])} contextes ; '
                  f'{time.monotonic()-debut:.1f} s', flush=True)
    source119 = REPO/'resultats/institut_approfondissements/119.json'
    ref = json.loads(source119.read_text(encoding='utf-8'))
    ecart_ref = 0.
    for c in objet['contextes']:
        if c['calibration'] == 'historique' and c['direction_t'] == 1 and c['noyau'] == 'marche':
            mask = '15' if c['accumulation'] else '7'
            for r in ii.REGLES:
                ecart_ref = max(ecart_ref, abs(c['capitaux'][r][0]-ref['configurations'][r][mask]))
    if ecart_ref > 1e-8:
        raise ArithmeticError('Le pilote ne retrouve pas les références de 119')
    objet['decisions'] = decisions(objet['contextes'], actes)
    objet['controles'] = dict(ecart_reference_119_max=ecart_ref, raffinement=grille,
        raffinement_max_M_EUR=max(c['ecart_max'] for c in grille),
        cdf_portee_min=min(c['diagnostic']['cdf_portee_min'] for c in objet['contextes']),
        masse_negative_max=max(c['diagnostic']['masse_negative_max'] for c in objet['contextes']))
    objet['sources'] = {'119_sha256': hashlib.sha256(source119.read_bytes()).hexdigest()}
    if args.bootstrap:
        source120 = REPO/'resultats/institut_approfondissements/120.json'
        objet['sources']['120_sha256'] = hashlib.sha256(source120.read_bytes()).hexdigest()
    objet['statut'] = 'termine'
    objet['duree_secondes'] = time.monotonic()-debut
    (args.dossier/'121.json').write_text(json.dumps(objet, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    figure(objet['contextes'], actes, args.dossier/'121_priorites.png')
    rapport(objet, actes, args.dossier)
    print(f'Terminé : {len(objet["contextes"])} contextes, {len(actes)} actions, '
          f'{objet["duree_secondes"]:.1f} s ; rapport et figure produits.', flush=True)


if __name__ == '__main__':
    main()
