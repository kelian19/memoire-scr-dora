#!/usr/bin/env python3
"""132 : les descriptions d'incidents SAS permettent-elles d'estimer la force des liens S ?

Pilote sur 50 descriptions codées en piliers DORA (pilote_codage_sas.py). Le script mesure ce que
le codage rend observable, avant tout investissement : part des incidents où au moins un pilier
est identifiable, part des incidents multi-piliers (seuls à renseigner S), loi marginale des
piliers, et co-occurrences par paire comparées à l'ordre de la matrice d'expert S.
Le codage est celui d'un assistant IA, non validé par un second codeur : les comptes ci-dessous
décrivent la faisabilité, ils n'estiment rien. Aucune valeur publiée n'est modifiée.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
for chemin in (REPO, LAB, Path(__file__).resolve().parent):
    sys.path.insert(0, str(chemin))
import partial_id as pid
from pilote_codage_sas import CODES

PAIRES = list(itertools.combinations(range(1, 6), 2))


def resume(niveau, sous_ensemble):
    ens = [c[2] if niveau == 'strict' else c[3] for c in sous_ensemble]
    n = len(ens)
    ident = sum(1 for e in ens if e)
    multi = sum(1 for e in ens if len(e) >= 2)
    return n, ident, multi, ens


def main():
    if len(CODES) != 50 or len({c[0] for c in CODES}) != 50:
        raise ValueError('Le pilote doit porter sur 50 identifiants distincts.')
    for c in CODES:
        if not c[2] <= c[3]:
            raise ValueError(f'Code strict non inclus dans le code large : {c[0]}')
    print('132 — PILOTE : CODER LES DESCRIPTIONS SAS EN PILIERS DORA')
    print('50 descriptions (30 fraudes externes, 20 interruptions d’activité), codeur unique : assistant IA, non validé.')

    for niveau in ('strict', 'large'):
        print(f'\nNIVEAU {niveau.upper()}')
        for nom, sel in (('fraude externe', [c for c in CODES if c[1] == 'EF']),
                         ('interruption d’activité', [c for c in CODES if c[1] == 'BDSF']),
                         ('ensemble', CODES)):
            n, ident, multi, _ = resume(niveau, sel)
            print(f'  {nom:24s} : {n:2d} incidents ; au moins un pilier {ident:2d} ({100*ident/n:.0f} %) ; '
                  f'au moins deux piliers {multi:2d} ({100*multi/n:.0f} %)')
        _, _, _, ens = resume(niveau, CODES)
        marg = {p: sum(1 for e in ens if p in e) for p in range(1, 6)}
        print('  incidents touchant chaque pilier : ' + ' ; '.join(f'P{p} {marg[p]}' for p in range(1, 6)))
        co = {pq: sum(1 for e in ens if set(pq) <= e) for pq in PAIRES}
        print('  co-occurrences par paire : ' + ' ; '.join(f'P{a}-P{b} {co[(a, b)]}' for a, b in PAIRES))
        tailles = [len(e) for e in ens if e]
        print(f'  piliers par incident identifiable : moyenne {np.mean(tailles):.2f}')

    S, _ = pid.decompose(pid.expert_matrix())
    s_exp = np.array([S[a-1, b-1] for a, b in PAIRES])
    _, _, _, ens = resume('large', CODES)
    co = np.array([sum(1 for e in ens if {a, b} <= e) for a, b in PAIRES], dtype=float)
    rho = spearmanr(s_exp, co).statistic
    print('\nCOMPARAISON À LA MATRICE D’EXPERT S (niveau large, ordre des dix paires)')
    print('  S posée : ' + ' ; '.join(f'P{a}-P{b} {v:.3f}' for (a, b), v in zip(PAIRES, s_exp)))
    print(f'  corrélation de rang entre S posée et co-occurrences codées : {rho:+.2f} sur dix paires, '
          f'dont {int(np.sum(co == 0))} paires jamais observées')
    print('\nLECTURE DÉRIVÉE DES COMPTES')
    _, ident_s, multi_s, _ = resume('strict', CODES)
    _, ident_l, multi_l, _ = resume('large', CODES)
    print(f'  incidents multi-piliers, seuls informatifs pour S : {multi_s} au niveau strict, {multi_l} au niveau large, sur 50.')
    poids = {'EF': 471, 'BDSF': 99}  # effectifs des strates parmi les 570 pertes
    extra = {}
    for niveau in ('strict', 'large'):
        total = 0.0
        for cat, effectif in poids.items():
            n, _, multi, _ = resume(niveau, [c for c in CODES if c[1] == cat])
            total += effectif*multi/n
        extra[niveau] = total
    print(f'  extrapolé aux 570 pertes en pondérant les strates (471 fraudes, 99 interruptions) : environ '
          f'{extra["strict"]:.0f} incidents multi-piliers au niveau strict, {extra["large"]:.0f} au niveau large.')
    print('  les co-occurrences brutes mêlent fréquence marginale et dépendance : P1 figure dans la plupart des '
          'incidents, ce qui suffit à ordonner les paires qui le contiennent.')
    print('\nLimites : un seul codeur, non humain ; descriptions de presse et de procédures, biais de narration ;')
    print('échantillon stratifié qui surreprésente les interruptions d’activité ; aucune direction n’est codée.')


if __name__ == '__main__':
    main()
