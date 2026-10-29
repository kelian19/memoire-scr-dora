#!/usr/bin/env python3
"""126 : dépendance entre les coûts d'un même incident, à marges fixées.

Un incident qui touche K piliers a K composantes de coût Y_1..Y_K, chacune nulle ou égale à
u+GPD, avec la même marge (Bernoulli de paramètre p_u, puis excès GPD). Le script 119 compare
quatre constructions qui changent à la fois le nombre de composantes et leur dépendance :
l'additive somme des Y indépendants, l'amplifiée vaut K fois un seul Y (comonotonie).
Ici on relie les deux extrêmes par deux familles de dépendance qui conservent exactement les
marges et se calculent sans simulation (transformée de Fourier de la perte d'incident) :
  - mélange de copules : avec probabilité rho les K composantes sont comonotones (choc commun),
    sinon indépendantes ; la loi de la somme est le mélange des deux lois, donc rho = 0 redonne
    l'additive et rho = 1 l'amplifiée à exposant un ;
  - blocs : les K composantes sont réparties en blocs de b composantes comonotones, les blocs
    étant indépendants ; b = 1 redonne l'additive, b = K l'amplifiée.
On sépare ainsi l'effet du nombre de composantes (incident unique contre K indépendants), celui
de leur dépendance (additive contre amplifiée) et celui d'une amplification au-delà de la
comonotonie (exposant posé supérieur à un). Aucune de ces constructions n'est validée par la
donnée : la marge observée porte sur l'incident complet, la réutiliser comme marge de chaque
composante est une hypothèse. Montants en M EUR du périmètre financier étudié.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
for chemin in (REPO, LAB):
    sys.path.insert(0, str(chemin))
import institut_incertitude as ii

RHOS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
RHOS_FINS = tuple(np.round(np.linspace(0, 1, 21), 2))
BLOCS = (2, 3, 4)
ETAS = (1.25, 1.5)
MASQUES = tuple(range(16))
PAS_DE_BASE = 2.0


def tailles_blocs(k, b):
    """K composantes en blocs de b au plus : (b, ..., b, reste)."""
    return [b]*(k//b) + ([k % b] if k % b else [])


def facteur_composante(moteur, pu, s):
    """Transformée d'une composante dont la perte est multipliée par s."""
    return 1-pu + pu*moteur.transformee_severite(s)


def incident_melange(moteur, pu, loi_k, rho):
    z1 = facteur_composante(moteur, pu, 1)
    return sum(loi_k[k]*((1-rho)*z1**k + rho*facteur_composante(moteur, pu, k)) for k in range(1, 6))


def incident_blocs(moteur, pu, loi_k, b):
    total = 0
    for k in range(1, 6):
        terme = 1
        for taille in tailles_blocs(k, b):
            terme = terme*facteur_composante(moteur, pu, taille)
        total = total + loi_k[k]*terme
    return total


def incident_exposant(moteur, pu, loi_k, eta):
    return sum(loi_k[k]*facteur_composante(moteur, pu, k**eta) for k in range(1, 6))


def capital_incident(moteur, mask, construire, alpha=.995):
    """Même calcul que FourierIncidents.capital, avec une transformée d'incident quelconque."""
    cx = ii.cx
    lam = cx.LAM_NC if mask & 1 else cx.LAM_C
    pu = moteur.cal.pu*(1.20 if mask & 2 else .85)
    loi_k = ii.cardinal(bool(mask & 4), bool(mask & 8))
    masses = np.fft.irfft(ii.pgf_nb(construire(moteur, pu, loi_k), lam), n=moteur.n)/moteur.attenuation
    if not np.isfinite(masses).all() or masses.min() < -1e-8:
        raise ArithmeticError("Inversion instable")
    cdf = np.cumsum(masses)
    idx = int(np.searchsorted(cdf, alpha))
    if idx >= moteur.n or idx > moteur.n*.25 or cdf[-1] < .995 or -masses[masses < 0].sum() > 1e-5:
        raise ArithmeticError("Quantile trop près de la portée")
    return idx*moteur.step


def scenarios():
    s = {}
    for rho in RHOS_FINS:
        s[f'rho_{rho:.2f}'] = (lambda m, pu, lk, r=rho: incident_melange(m, pu, lk, r))
    for b in BLOCS:
        s[f'blocs_{b}'] = (lambda m, pu, lk, b=b: incident_blocs(m, pu, lk, b))
    for eta in ETAS:
        s[f'exposant_{eta}'] = (lambda m, pu, lk, e=eta: incident_exposant(m, pu, lk, e))
    return s


def calculer(cal=ii.REFERENCE, pas=PAS_DE_BASE):
    """Capitaux de toutes les constructions ; le pas double par scénario si la portée est insuffisante."""
    construits = scenarios()
    principaux = {f'rho_{r:.2f}' for r in RHOS} | {f'blocs_{b}' for b in BLOCS} | {f'exposant_{e}' for e in ETAS}
    moteurs, capitaux, pas_utilises = {}, {}, {}

    def moteur_pour(essai):
        if essai not in moteurs:
            moteurs[essai] = ii.FourierIncidents(cal, step=pas*2**essai)
        return moteurs[essai]

    for nom, f in construits.items():
        masques = MASQUES if nom in principaux else (0, 15)
        for essai in range(5):
            try:
                capitaux[nom] = {str(m): capital_incident(moteur_pour(essai), m, f) for m in masques}
                pas_utilises[nom] = pas*2**essai
                break
            except ArithmeticError:
                if essai == 4:
                    raise
    for regle in ('unique', 'moyenne', 'additive', 'amplifiee'):
        capitaux[regle] = {str(m): moteur_pour(0).capital(m, regle)[0] for m in MASQUES}
        pas_utilises[regle] = pas
    return moteur_pour(0), capitaux, pas_utilises


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sortie', type=Path, default=REPO/'resultats/institut_approfondissements/126.json')
    args = parser.parse_args()
    debut = time.monotonic()
    moteur, cap, pas_utilises = calculer()
    cal = moteur.cal
    cx = ii.cx
    print('126 — DÉPENDANCE ENTRE COÛTS D’UN MÊME INCIDENT, VERSION INSTITUT')
    print('Montants en M EUR du périmètre financier étudié. Mêmes marges pour toutes les dépendances.')
    print(f'Calcul par Fourier incliné ; pas {moteur.step:g} M EUR ; {moteur.n} cases ; inclinaison {moteur.tilt:g}.')
    doubles = sorted(nom for nom, p in pas_utilises.items() if p != moteur.step)
    if doubles:
        print('Pas doublé faute de portée pour : ' + ', '.join(f'{n} ({pas_utilises[n]:g})' for n in doubles) + '.')
    print('Aucune de ces constructions n’est validée par la donnée : la marge observée porte sur l’incident,')
    print('la réutiliser comme marge de chaque composante est une hypothèse.')
    assert abs(ii.cardinal(False, False)[0]) < 1e-15 and abs(ii.cardinal(True, True)[0]) < 1e-15

    # Contrôles d'identité : les extrêmes des familles redonnent les règles du script 119.
    ecarts = []
    for m in MASQUES:
        ecarts += [abs(cap['rho_0.00'][str(m)]-cap['additive'][str(m)]),
                   abs(cap['rho_1.00'][str(m)]-cap['amplifiee'][str(m)])]
    biais = max(ecarts)
    assert biais < 1e-9, biais
    for b, ext in ((1, 'additive'), (5, 'amplifiee')):
        for m in (0, 15):
            valeur = capital_incident(moteur, m, lambda mt, pu, lk, b=b: incident_blocs(mt, pu, lk, b))
            assert abs(valeur-cap[ext][str(m)]) < 1e-9
    print('Contrôle : le mélange à rho = 0 et à rho = 1 redonne l’additive et l’amplifiée aux seize configurations ;')
    print('des blocs de taille un et cinq les redonnent aussi aux deux états de référence.')
    seul = np.array([0., 1., 0., 0., 0., 0.])
    uniques = {capital_incident(moteur, 0, lambda mt, pu, lk, f=f: f(mt, pu, seul)) for f in
               [lambda mt, pu, lk: incident_melange(mt, pu, lk, .5), lambda mt, pu, lk: incident_blocs(mt, pu, lk, 3),
                lambda mt, pu, lk: incident_exposant(mt, pu, lk, 1.5)]}
    assert len(uniques) == 1
    print(f'Contrôle : avec K = 1, toutes les dépendances donnent le même capital, {uniques.pop():.1f}.')
    moy = {e: cx_lam*ii.moyenne_incident(cal, cal.pu*pm, ii.cardinal(e == 'NC', e == 'NC'), 'additive')
           for e, cx_lam, pm in (('C', cx.LAM_C, .85), ('NC', cx.LAM_NC, 1.20))}
    print(f'La perte annuelle moyenne ne dépend pas de la dépendance : {moy["C"]:.1f} (conforme), '
          f'{moy["NC"]:.1f} (non conforme) ; seule la forme de la loi change.')

    sortie = dict(experience='126 dépendance entre coûts d’un même incident', pas=moteur.step, n=moteur.n,
                  tilt=moteur.tilt, pas_par_scenario=pas_utilises, moyenne_annuelle=moy, capitaux=cap, contrastes={})
    print('\nCHEMIN DE DÉPENDANCE : MÉLANGE DE COPULES (rho = probabilité du choc commun)')
    print(f"{'rho':>5s} {'conforme':>10s} {'non conforme':>13s} {'écart':>9s} {'rapport':>8s} {'interaction':>12s}")
    for rho in RHOS:
        c = ii.contrastes(cap[f'rho_{rho:.2f}'])
        sortie['contrastes'][f'rho_{rho:.2f}'] = c
        print(f"{rho:5.2f} {c['conforme']:10.1f} {c['non_conforme']:13.1f} {c['ecart']:9.1f} {c['rapport']:8.3f} {c['interaction']:12.1f}")
    print('\nCHEMIN DE DÉPENDANCE : BLOCS DE COMPOSANTES COMONOTONES, BLOCS INDÉPENDANTS')
    print(f"{'bloc':>5s} {'conforme':>10s} {'non conforme':>13s} {'écart':>9s} {'rapport':>8s} {'interaction':>12s}")
    lignes_blocs = [('1', 'additive')] + [(str(b), f'blocs_{b}') for b in BLOCS] + [('K', 'amplifiee')]
    for lib, nom in lignes_blocs:
        c = ii.contrastes(cap[nom])
        sortie['contrastes'][nom] = c
        print(f"{lib:>5s} {c['conforme']:10.1f} {c['non_conforme']:13.1f} {c['ecart']:9.1f} {c['rapport']:8.3f} {c['interaction']:12.1f}")
    print('\nAMPLIFICATION AU-DELÀ DE LA COMONOTONIE : exposant posé eta, K**eta fois un seul Y')
    print(f"{'eta':>5s} {'conforme':>10s} {'non conforme':>13s} {'écart':>9s} {'rapport':>8s} {'interaction':>12s}")
    for lib, nom in [('1', 'amplifiee')] + [(f'{e:g}', f'exposant_{e}') for e in ETAS]:
        c = ii.contrastes(cap[nom])
        sortie['contrastes'][nom] = c
        print(f"{lib:>5s} {c['conforme']:10.1f} {c['non_conforme']:13.1f} {c['ecart']:9.1f} {c['rapport']:8.3f} {c['interaction']:12.1f}")
    for nom in ('unique', 'moyenne', 'additive'):
        sortie['contrastes'][nom] = ii.contrastes(cap[nom])

    # Monotonie, mesurée sur une grille fine de rho et non supposée.
    fins = [(rho, cap[f'rho_{rho:.2f}']['0'], cap[f'rho_{rho:.2f}']['15']) for rho in RHOS_FINS]
    rho_f, c_f, n_f = (np.array(x) for x in zip(*fins))
    r_f = n_f/c_f
    mono = {nom: dict(croissant=bool((np.diff(v) >= -1e-9).all()), plus_forte_baisse=float(min(0, np.diff(v).min())))
            for nom, v in (('conforme', c_f), ('non_conforme', n_f), ('rapport', r_f))}
    sortie['monotonie'] = dict(grille=[float(r) for r in rho_f], **mono)
    print('\nMONOTONIE SUR UNE GRILLE FINE DE rho (21 valeurs de 0 à 1, pas 0,05)')
    for nom, v in mono.items():
        print(f"  {nom:13s} croissant en rho : {'oui' if v['croissant'] else 'non'} ; plus forte baisse entre deux pas "
              f"{v['plus_forte_baisse']:.3f}")

    # Décomposition en trois effets, à chaque état.
    def etat(nom, m):
        return cap[nom][str(m)]
    dec = {}
    for lib, m in (('conforme', 0), ('non_conforme', 15)):
        dec[lib] = dict(unique=etat('unique', m), nombre=etat('additive', m)-etat('unique', m),
                        dependance=etat('amplifiee', m)-etat('additive', m),
                        amplification=etat('exposant_1.25', m)-etat('amplifiee', m))
    rap = lambda nom: etat(nom, 15)/etat(nom, 0)
    dec['rapport'] = dict(unique=rap('unique'), additive=rap('additive'), amplifiee=rap('amplifiee'),
                          exposant_1_25=rap('exposant_1.25'))
    sortie['decomposition'] = dec
    print('\nDÉCOMPOSITION DE LA DIFFÉRENCE ENTRE L’INCIDENT UNIQUE ET L’AMPLIFICATION POSÉE (M EUR)')
    print(f"{'':14s} {'incident unique':>16s} {'+ nombre':>9s} {'+ dépendance':>13s} {'+ amplification 1,25':>21s} {'= total':>9s}")
    for lib in ('conforme', 'non_conforme'):
        d = dec[lib]
        total = d['unique']+d['nombre']+d['dependance']+d['amplification']
        print(f"{lib:14s} {d['unique']:16.1f} {d['nombre']:9.1f} {d['dependance']:13.1f} {d['amplification']:21.1f} {total:9.1f}")
    print(f"rapport entre états : incident unique {dec['rapport']['unique']:.3f}, K indépendants {dec['rapport']['additive']:.3f}, "
          f"comonotonie {dec['rapport']['amplifiee']:.3f}, exposant 1,25 {dec['rapport']['exposant_1_25']:.3f}.")

    # Grandeurs citées par le manuscrit, sans séparateur de milliers.
    c0, c1 = ii.contrastes(cap['rho_0.00']), ii.contrastes(cap['rho_1.00'])
    demi = ii.contrastes(cap['rho_0.50'])
    cites = dict(
        hausse_conforme_pct=100*(c1['conforme']/c0['conforme']-1),
        hausse_non_conforme_pct=100*(c1['non_conforme']/c0['non_conforme']-1),
        hausse_rapport_pct=100*(c1['rapport']/c0['rapport']-1),
        part_dependance_conforme_pct=100*dec['conforme']['dependance']/(dec['conforme']['nombre']+dec['conforme']['dependance']),
        part_dependance_non_conforme_pct=100*dec['non_conforme']['dependance']/(dec['non_conforme']['nombre']+dec['non_conforme']['dependance']),
        rapport_mi_chemin=demi['rapport'],
        interaction_hausse_pct=100*(c1['interaction']/c0['interaction']-1),
        rapport_unique=dec['rapport']['unique'])
    sortie['grandeurs_citees'] = cites
    print('\nGRANDEURS CITÉES')
    print(f"de l'indépendance à la comonotonie : conforme {cites['hausse_conforme_pct']:+.0f} %, "
          f"non conforme {cites['hausse_non_conforme_pct']:+.0f} %, rapport entre états {cites['hausse_rapport_pct']:+.0f} %.")
    print(f"part de la dépendance dans l'écart entre l'incident unique et la comonotonie : "
          f"{cites['part_dependance_conforme_pct']:.0f} % (conforme), {cites['part_dependance_non_conforme_pct']:.0f} % (non conforme).")
    print(f"rapport entre états à mi-chemin (rho 0,5) : {cites['rapport_mi_chemin']:.2f} ; interaction entre canaux : "
          f"{cites['interaction_hausse_pct']:+.0f} % de l'indépendance à la comonotonie.")

    sortie['duree_secondes'] = time.monotonic()-debut
    json.dump(sortie, open(args.sortie, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\nLimites : deux familles de dépendance parmi d’autres ; marges des composantes posées égales à celle de l’incident ;')
    print('aucune dépendance n’est estimée sur la donnée, qui ne contient que le coût de l’incident complet ;')
    print('le choc commun et les blocs sont des scénarios, et la monotonie en rho est un constat numérique, pas un théorème.')
    print(f'Durée {sortie["duree_secondes"]:.1f} s ; résultats agrégés : {args.sortie.name}')


if __name__ == '__main__':
    main()
