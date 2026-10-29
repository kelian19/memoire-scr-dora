#!/usr/bin/env python3
"""122 : frontières P1/P4 en efficacité et coût annuel de financement du capital."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
for chemin in (REPO, LAB):
    sys.path.insert(0, str(chemin))
import canaux_conformite as cx
import institut_incertitude as ii
import institut_remediation as ir

EMISSION = np.linspace(0, 1, 5)
TIERS = np.linspace(0, 1, 11)
TAUX = (.03, .06, .10)


def sources():
    fichiers = {n: REPO/f'resultats/{d}/{n}.json' for n, d in
                [('120', 'institut_approfondissements'), ('121', 'institut_janvier')]}
    objets = {n: json.loads(p.read_text(encoding='utf-8')) for n, p in fichiers.items()}
    if any(o.get('statut') != 'termine' for o in objets.values()):
        raise ValueError('Sources incomplètes')
    if objets['121']['sources']['120_sha256'] != hashlib.sha256(fichiers['120'].read_bytes()).hexdigest():
        raise ValueError('Le bootstrap ne correspond plus à la source de 121')
    cas = list(objets['121']['calibrations'])
    ordre = sorted(objets['120']['replications'], key=lambda r: r['calibration']['xi'])
    for nom, r in [('bootstrap_xi_minimal', ordre[0]), ('bootstrap_xi_maximal', ordre[-1])]:
        cas.append(dict(nom=nom, calibration=r['calibration'], replication_source=r['replication']))
    return cas, objets['121'], {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in fichiers.items()}


class Evaluateur:
    def __init__(self, moteur):
        self.moteur = moteur
        self.cache = {}
        self.diags = []

    def capital(self, loi, regle):
        # La règle unique est indépendante de K : une inversion suffit par calage.
        key = (regle, b'' if regle == 'unique' else loi.tobytes())
        if key not in self.cache:
            q, d = self.moteur.capital(15, regle, loi_k=loi)
            self.cache[key] = q
            self.diags.append(d)
        return self.cache[key]


def capital_action(ev, w, noyau, phi, pilier, e_emission, e_tiers, regle):
    v, p = ir.appliquer_ciblee(w, pilier, e_emission, phi, e_tiers)
    return ev.capital(ir.cardinal(v, noyau, p), regle)


def frontiere_fine(ev, w, noyau, phi, regle, principal):
    """Encadrer une bascule à coûts égaux ; zone de résolution en capital explicite."""
    if regle == 'unique':
        return {'statut': 'egalite_structurelle', 'intervalle': None}
    if regle == 'moyenne':
        return {'statut': 'grille_seulement', 'intervalle': None}
    avantage = np.asarray(principal['Q1'])[2]-np.asarray(principal['Q4'])[2]
    tolerance = 2*ev.moteur.step
    if (np.diff(avantage) < -4*ev.moteur.step).any():
        return {'statut': 'non_monotone_sur_grille', 'intervalle': None}
    q1 = capital_action(ev, w, noyau, phi, 1, .5, 0., regle)
    f = ir.encadrer_bascule(lambda t: q1-capital_action(ev, w, noyau, phi, 4, .5, t, regle),
                           tolerance=tolerance, precision=.002)
    f['monotonie'] = 'couplage du choc tiers pour ces deux règles ; contrôle sur grille'
    return f


def seuils(objet):
    # Mettre aussi à jour cette qualification lors d'une restitution : le couplage
    # concerne phi à W fixé, et ne prouve rien sur une variation de direction de W.
    for c in objet['contextes']:
        for regle in ('additive', 'amplifiee'):
            f = c['resultats'][regle]['frontiere']
            if 'monotonie' in f:
                f['monotonie'] = 'couplage du choc tiers pour ces deux règles ; contrôle sur grille'
    index = {(c['calibration'], c['direction_t'], c['noyau'], c['accumulation']): c
             for c in objet['contextes']}
    for d in objet['controles']['raffinement']:
        c = index[(d['calibration'], d['direction_t'], d['noyau'], d['accumulation'])]
        d['ecart_relatif_au_capital_initial'] = d['ecart_max']/c['resultats'][d['regle']]['Q0']
    objet['controles']['raffinement_relatif_max'] = max(
        d['ecart_relatif_au_capital_initial'] for d in objet['controles']['raffinement'])
    objet['controles']['raffinement_historique_max'] = max(
        d['ecart_max'] for d in objet['controles']['raffinement'] if d['calibration'] == 'historique')
    for c in objet['contextes']:
        for regle, r in c['resultats'].items():
            if r['Q1'][0] != r['Q0'] or r['Q4'][0][0] != r['Q0']:
                raise AssertionError('Une action d’efficacité nulle change le capital')
            if regle in ('additive', 'amplifiee'):
                if (np.diff(r['Q1']) > 2*c['step']).any() or (np.diff(r['Q4'], axis=1) > 2*c['step']).any():
                    raise AssertionError('Augmentation du capital sous une réduction supposée monotone')
    resume = {}
    for regle in ii.REGLES:
        cas = [c for c in objet['contextes'] if c['accumulation']]
        f = [c['resultats'][regle]['frontiere'] for c in cas]
        intervalles = [v['intervalle'] for v in f if v['intervalle'] is not None]
        resume[regle] = {'statuts': {s: sum(v['statut'] == s for v in f) for s in sorted({v['statut'] for v in f})},
            'enveloppe_intervalles': [min(i[0] for i in intervalles), max(i[1] for i in intervalles)] if intervalles else None,
            'surcout_a_6pct_minmax': [float(op([.06*(c['resultats'][regle]['Q1'][2]
                -c['resultats'][regle]['Q4'][2][5]) for c in cas])) for op in (min, max)]}
    objet['resume'] = resume


def figures(objet, dossier):
    noms = dict(additive='Additive', moyenne='Moyenne', unique='Unique', amplifiee='Amplifiée')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, regle in zip(axes.flat, ii.REGLES):
        for noyau, couleur in [('marche', '#2865a8'), ('branchement', '#b35228')]:
            cas = [c for c in objet['contextes'] if c['calibration'] == 'historique'
                   and c['accumulation'] and c['noyau'] == noyau]
            valeurs = np.array([.06*(np.asarray(c['resultats'][regle]['Q1'])[2]
                                    -np.asarray(c['resultats'][regle]['Q4'])[2]) for c in cas])
            expert = next(c for c in cas if c['direction_t'] == 1)
            ligne = .06*(np.asarray(expert['resultats'][regle]['Q1'])[2]
                         -np.asarray(expert['resultats'][regle]['Q4'])[2])
            ax.fill_between(TIERS*100, valeurs.min(axis=0), valeurs.max(axis=0), color=couleur, alpha=.12)
            ax.plot(TIERS*100, ligne, color=couleur, marker='o', markersize=3, label=noyau.capitalize())
        ax.axhline(0, color='#444444', lw=.8)
        ax.set_title(noms[regle], loc='left', fontweight='bold')
        ax.set_ylabel('Surcoût admissible de P4 (M EUR/an)')
        ax.set_xlabel('Réduction de la probabilité d’inclusion par le choc tiers (%)')
        ax.grid(alpha=.2)
        if regle == 'unique':
            ax.set_ylim(-1, 1)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('P4 face à P1 : efficacité sur le choc tiers et différence de coût acceptable', fontsize=12)
    fig.text(.5, .012, 'Émissions réduites de 50 % pour chaque action ; taux de financement posé à 6 %/an.\n'
             'Ligne : direction expert. Bande : cinq directions au calage historique, sans probabilité. Comparaison hors abstention.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .075, 1, .95))
    fig.savefig(dossier/'122_frontieres.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    couleurs = ['#eeeeee', '#2865a8', '#d97732']
    for ax, regle in zip(axes, ('additive', 'amplifiee')):
        c = next(c for c in objet['contextes'] if c['calibration'] == 'historique'
                 and c['accumulation'] and c['noyau'] == 'marche' and c['direction_t'] == 1)
        r = c['resultats'][regle]
        e = ir.economie_capital(r['Q0'], r['Q1'][2], r['Q4'][2][5])
        borne = max(e['plafond_cout_P1'], e['plafond_cout_P4'])*1.3
        x = np.linspace(0, borne, 301); y = np.linspace(0, borne, 301)
        xx, yy = np.meshgrid(x, y)
        nets = np.stack([np.zeros_like(xx), e['plafond_cout_P1']-xx, e['plafond_cout_P4']-yy])
        choix = nets.argmax(axis=0)
        ax.pcolormesh(x, y, choix, cmap=ListedColormap(couleurs), vmin=0, vmax=2, shading='auto')
        ax.set_xlabel('Coût annuel de P1 (M EUR/an)'); ax.set_ylabel('Coût annuel de P4 (M EUR/an)')
        ax.set_title(noms[regle], loc='left', fontweight='bold')
    axes[0].legend(handles=[Patch(color=c, label=n) for c, n in zip(couleurs, ['Abstention', 'P1', 'P4'])], fontsize=8)
    fig.suptitle('Choix sur le financement du capital et les coûts directs', fontsize=12)
    fig.text(.5, .012, 'Scénario expert, marche et choc actif. Émissions et inclusion par le choc réduites de 50 %.\n'
             'Taux posé à 6 %/an. Coûts annuels hypothétiques ; pertes évitées et obligations réglementaires hors score.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, .94))
    fig.savefig(dossier/'122_couts.png', dpi=170)
    plt.close(fig)


def rapport(objet, dossier):
    def pourcent_intervalle(intervalle):
        # Arrondir vers l'extérieur pour ne pas resserrer artificiellement la zone.
        lo = np.floor(1000*intervalle[0])/10
        hi = np.ceil(1000*intervalle[1])/10
        return f'[{lo:.1f} ; {hi:.1f}]'

    lignes = ['# 122 : seuils d’efficacité et de coût entre P1 et P4', '',
        'Ces seuils sont des scénarios sur le financement du capital, pas des coûts observés.', '',
        '## Protocole', '',
        f"{len(objet['contextes'])} contextes : six calibrations (dont xi minimal et maximal du bootstrap),",
        'cinq directions, deux noyaux de cascade et choc tiers absent/actif. Quatre constructions de perte.',
        'Les efficacités sur les lignes P1 et P4 varient indépendamment de 0 à 100 % par pas de 25 points ;',
        'l’efficacité sur l’inclusion par le choc tiers varie par pas de 10 points.', '',
        'Les coûts C1/C4 sont annuels ou annualisés séparément. Le taux h est un coût de financement, posé,',
        'et non un taux d’actualisation. Le plafond de coût de chaque action est h(Q0-Qa).',
        'Le surcoût admissible de P4 face à P1 est h(Q1-Q4). Comparer aussi à l’abstention.', '',
        '## Au point de référence', '',
        'Direction expert, marche, choc actif ; réduction de 50 % de chaque ligne et de l’inclusion par le choc tiers.', '',
        '| Perte | h (%/an) | Plafond P1 | Plafond P4 | Surcoût P4 face à P1 |',
        '|---|---:|---:|---:|---:|']
    expert = next(c for c in objet['contextes'] if c['calibration'] == 'historique'
                 and c['direction_t'] == 1 and c['noyau'] == 'marche' and c['accumulation'])
    for regle in ii.REGLES:
        r = expert['resultats'][regle]
        for taux in TAUX:
            e = ir.economie_capital(r['Q0'], r['Q1'][2], r['Q4'][2][5], taux)
            lignes.append(f"| {regle} | {100*taux:.0f} | {e['plafond_cout_P1']:.2f} | {e['plafond_cout_P4']:.2f} | {e['surcout_P4_vs_P1']:.2f} |")
    lignes += ['', 'Montants en M EUR/an pour le périmètre financier étudié. Un plafond négatif signifie',
               'que l’action augmente déjà le coût de financement avant tout coût direct positif.', '',
               '## Efficacité sur le choc tiers nécessaire à coûts égaux', '',
               'Les deux lignes sont réduites de 50 %. Les intervalles ci-dessous encadrent une zone',
               'de résolution numérique sur le scénario expert, pas une incertitude d’estimation. Les bornes sont arrondies vers l’extérieur.', '',
               '| Perte | Statut | Zone de bascule en efficacité (%) |', '|---|---|---|']
    for regle in ii.REGLES:
        f = expert['resultats'][regle]['frontiere']; intervalle = f['intervalle']
        valeur = pourcent_intervalle(intervalle) if intervalle is not None else 'sans seuil chiffré'
        lignes.append(f"| {regle} | {f['statut']} | {valeur} |")
    lignes += ['', '## Sensibilité sur les scénarios avec choc tiers', '',
               'Les six calibrations, cinq directions et deux noyaux sont examinés ; ils ne forment pas une loi de probabilité.', '',
               '| Perte | Statuts des frontières (nombre de cas) | Enveloppe des bascules intérieures (%) | Surcoût P4 à 6 % : min / max (M EUR/an) |',
               '|---|---|---|---|']
    for regle, r in objet['resume'].items():
        intervalle = r['enveloppe_intervalles']; a, b = r['surcout_a_6pct_minmax']
        valeur = pourcent_intervalle(intervalle) if intervalle is not None else 'non défini'
        lignes.append(f"| {regle} | {r['statuts']} | {valeur} | {a:.2f} / {b:.2f} |")
    ctrl = objet['controles']
    lignes += ['', '## Contrôles et portée', '',
        f"- Cellules communes avec 121 : écart maximal {ctrl['ecart_121_max']:.6g} M EUR.",
        f"- Raffinement : au plus {ctrl['raffinement_historique_max']:.3g} M EUR au calage historique ; {ctrl['raffinement_max']:.3g} M EUR sur les stress extrêmes.",
        f"- Écart maximal de raffinement rapporté au capital initial : {100*ctrl['raffinement_relatif_max']:.4f} %.",
        f"- Masse annuelle minimale conservée {ctrl['cdf_portee_min']:.8f} ; masse négative maximale {ctrl['masse_negative_max']:.3g}.",
        '- Les frontières additive/amplifiée sont raffinées sur la grille fine ; le couplage du choc tiers justifie la monotonie dans ce mécanisme précis.',
        '- Les plateaux de quantile sont traités par une tolérance de deux pas fins ; les bornes en efficacité ne sont pas des paramètres estimés.',
        '- Les autres points des surfaces utilisent la grille principale ; aucun optimum continu en efficacité n’est revendiqué.',
        '- Les six calibrations incluent xi >= 1, sans recours à une perte moyenne infinie dans le score.',
        '- Sans choc commun actif, l’efficacité sur ce mécanisme reste sans effet.',
        '- La règle unique est neutre par construction. La règle moyenne peut rendre les actions défavorables.',
        '- Une préférence P4/P1 ne suffit pas à préférer une action à l’abstention.',
        '- Les coûts, efficacités, taux et états sont posés ; les cinq directions ne couvrent pas le pavé admissible entier.',
        '- Le score exclut pertes moyennes évitées, bénéfices opérationnels et obligations de conformité.', '',
        'La prochaine étape est de documenter les coûts et effets opérationnels des interventions, puis d’élargir',
        'l’ensemble directionnel. Le mélange prédictif annuel reste un axe distinct de la feuille de route.', '']
    (dossier/'122_rapport.md').write_text('\n'.join(lignes), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dossier', type=Path, default=REPO/'resultats/institut_janvier')
    p.add_argument('--restituer', action='store_true')
    args = p.parse_args(); args.dossier.mkdir(parents=True, exist_ok=True)
    fichier = args.dossier/'122.json'
    if args.restituer:
        objet = json.loads(fichier.read_text(encoding='utf-8'))
        if objet['statut'] != 'termine':
            raise ValueError('Calcul incomplet')
        seuils(objet)
        fichier.write_text(json.dumps(objet, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        figures(objet, args.dossier); rapport(objet, args.dossier)
        return
    debut = time.monotonic(); cas, source121, empreintes = sources()
    objet = dict(experience='122', statut='en_cours', sources=empreintes,
                 calibrations=cas, grille_emission=EMISSION.tolist(), grille_tiers=TIERS.tolist(),
                 taux_financement=list(TAUX), contextes=[])
    controle121 = 0.; raffinement = []; diagnostics = []
    for cal in cas:
        calibration = ii.Calibration(**cal['calibration'])
        step = 2. if cal['nom'] == 'historique' else max(2., ii.grille_adaptee(calibration))
        ev = Evaluateur(ii.FourierIncidents(calibration, step=step))
        fin = Evaluateur(ii.FourierIncidents(calibration, step=step/2, n=1 << 17, tilt=18))
        for nom, t, w in ir.directions():
            for noyau in ir.NOYAUX:
                for accumulation in (False, True):
                    phi = cx.PHICS_NC if accumulation else None
                    base = ir.cardinal(w, noyau, phi)
                    c = dict(calibration=cal['nom'], direction_t=t, direction=nom, noyau=noyau,
                             accumulation=accumulation, step=step, resultats={})
                    for regle in ii.REGLES:
                        q0 = ev.capital(base, regle)
                        q1 = [capital_action(ev, w, noyau, phi, 1, e, 0., regle) for e in EMISSION]
                        q4 = [[capital_action(ev, w, noyau, phi, 4, e, et, regle) for et in TIERS] for e in EMISSION]
                        r = dict(Q0=q0, Q1=q1, Q4=q4)
                        c['resultats'][regle] = r
                        if regle == 'unique' and (np.ptp(q1) or np.ptp(q4)):
                            raise AssertionError('La règle unique doit rester neutre')
                        if not accumulation and (np.ptp(q4, axis=1) > 1e-9).any():
                            raise AssertionError('Efficacité tiers active alors que le choc est absent')
                        # Références témoin et extrémités : tous les contextes, y compris xi >= 1.
                        candidats = [(ev.capital(base, regle), fin.capital(base, regle)),
                                      (q1[2], capital_action(fin, w, noyau, phi, 1, .5, 0., regle))]
                        candidats += [(q4[2][k], capital_action(fin, w, noyau, phi, 4, .5, float(TIERS[k]), regle))
                                       for k in (0, 5, 10)]
                        erreur = max(abs(a-b) for a, b in candidats)
                        if erreur > max(3*step, .002*q0):
                            raise ArithmeticError('Convergence numérique insuffisante')
                        raffinement.append(dict(calibration=cal['nom'], direction_t=t, noyau=noyau,
                            accumulation=accumulation, regle=regle, ecart_max=erreur))
                        r['frontiere'] = frontiere_fine(fin, w, noyau, phi, regle, r)
                        reference = next((v for v in source121['contextes'] if v['calibration'] == cal['nom']
                            and v['direction_t'] == t and v['noyau'] == noyau and v['accumulation'] == accumulation
                            and v['famille'] == 'emission_et_tiers'), None)
                        if reference is not None:
                            actes = [tuple(a) for a in source121['actions']]
                            qref = reference['capitaux'][regle]
                            controle121 = max(controle121, abs(q0-qref[0]), abs(q1[2]-qref[actes.index((1,))]),
                                              abs(q4[2][5]-qref[actes.index((4,))]))
                    objet['contextes'].append(c)
            objet['duree_secondes'] = time.monotonic()-debut
            fichier.write_text(json.dumps(objet, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            print(f"{cal['nom']} {nom} : {len(objet['contextes'])} contextes ; {objet['duree_secondes']:.1f} s", flush=True)
        diagnostics += ev.diags+fin.diags
    if controle121 > 1e-8:
        raise AssertionError('La référence 121 n’est pas retrouvée')
    objet['controles'] = dict(ecart_121_max=controle121, raffinement=raffinement,
        raffinement_max=max(v['ecart_max'] for v in raffinement),
        cdf_portee_min=min(d['cdf_portee'] for d in diagnostics),
        masse_negative_max=max(d['masse_negative'] for d in diagnostics),
        inversions_fourier=len(diagnostics))
    objet['statut'] = 'termine'; objet['duree_secondes'] = time.monotonic()-debut
    seuils(objet)
    fichier.write_text(json.dumps(objet, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    figures(objet, args.dossier); rapport(objet, args.dossier)
    print(f"Terminé : {len(objet['contextes'])} contextes ; {len(diagnostics)} inversions ; {objet['duree_secondes']:.1f} s.", flush=True)


if __name__ == '__main__':
    main()
