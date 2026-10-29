#!/usr/bin/env python3
"""133 : codage en piliers DORA des 570 descriptions SAS, et ce qu'il dit de la force des liens S.

Réunit le pilote (pilote_codage_sas.py, 50 descriptions) et le complément (codage_sas_complet.py,
520 descriptions), soit les 570 pertes cyber x finance 2000-2026. Les textes sont sous licence et ne
sont pas versionnés : seuls l'identifiant SAS et les codes le sont.

Le script mesure ce que le codage rend observable : part des incidents à au moins un pilier, part
des multi-piliers, marges, co-occurrences par paire, puis une mesure de dépendance corrigée de la
fréquence marginale (rapport observé sur attendu sous indépendance, coefficient phi, test exact de
Fisher), comparée à l'ordre des dix paires de la matrice d'expert S.
Codeur unique, assistant IA, non validé par un second codeur : rien ici n'estime S ni ne remplace
une valeur publiée.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, spearmanr

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
for chemin in (REPO, LAB, Path(__file__).resolve().parent):
    sys.path.insert(0, str(chemin))
import partial_id as pid
from codage_sas_complet import CODES as COMPLEMENT
from pilote_codage_sas import CODES as PILOTE

PAIRES = list(itertools.combinations(range(1, 6), 2))
CAT = {'EF': 'E', 'BDSF': 'B', 'E': 'E', 'B': 'B'}
NOM = {'E': 'fraude externe', 'B': 'interruption d’activité'}


def reunir():
    codes = [(i, CAT[c], s, l) for i, c, s, l in PILOTE + COMPLEMENT]
    ids = [c[0] for c in codes]
    if len(codes) != 570 or len(set(ids)) != 570:
        raise ValueError(f'{len(codes)} codes pour {len(set(ids))} identifiants distincts, 570 attendus.')
    for i, _, s, l in codes:
        if not s <= l:
            raise ValueError(f'Code strict non inclus dans le code large : {i}')
        if not (s | l) <= set(range(1, 6)):
            raise ValueError(f'Pilier hors de 1 à 5 : {i}')
    n_e = sum(1 for c in codes if c[1] == 'E')
    if (n_e, 570 - n_e) != (471, 99):
        raise ValueError(f'Strates {n_e} et {570 - n_e}, 471 et 99 attendues.')
    return codes


def ensembles(codes, niveau):
    return [c[2] if niveau == 'strict' else c[3] for c in codes]


def dependance(ens):
    n = len(ens)
    lignes = []
    for a, b in PAIRES:
        n11 = sum(1 for e in ens if a in e and b in e)
        n10 = sum(1 for e in ens if a in e and b not in e)
        n01 = sum(1 for e in ens if a not in e and b in e)
        n00 = n - n11 - n10 - n01
        attendu = (n11 + n10)*(n11 + n01)/n
        den = np.sqrt((n11 + n10)*(n01 + n00)*(n11 + n01)*(n10 + n00))
        phi = (n11*n00 - n10*n01)/den if den > 0 else np.nan
        p = fisher_exact([[n11, n10], [n01, n00]]).pvalue
        lignes.append(((a, b), n11, attendu, n11/attendu if attendu > 0 else np.nan, phi, p))
    return lignes


def main():
    codes = reunir()
    print('133 — CODAGE DES 570 DESCRIPTIONS SAS EN PILIERS DORA')
    print('570 descriptions (471 fraudes externes, 99 interruptions d’activité), dont les 50 du pilote.')
    print('Codeur unique : assistant IA, grille fixée avant lecture, non validé par un second codeur.')

    S, _ = pid.decompose(pid.expert_matrix())
    s_exp = np.array([S[a-1, b-1] for a, b in PAIRES])
    resultats = {}
    for niveau in ('strict', 'large'):
        print(f'\nNIVEAU {niveau.upper()}')
        for cat in ('E', 'B', None):
            sel = [c for c in codes if cat is None or c[1] == cat]
            ens = ensembles(sel, niveau)
            n = len(ens)
            ident = sum(1 for e in ens if e)
            multi = sum(1 for e in ens if len(e) >= 2)
            nom = NOM[cat] if cat else 'ensemble'
            print(f'  {nom:24s} : {n:3d} incidents ; au moins un pilier {ident:3d} ({100*ident/n:.1f} %) ; '
                  f'au moins deux piliers {multi:3d} ({100*multi/n:.1f} %)')
        ens = ensembles(codes, niveau)
        marg = {p: sum(1 for e in ens if p in e) for p in range(1, 6)}
        print('  incidents touchant chaque pilier : ' + ' ; '.join(f'P{p} {marg[p]}' for p in range(1, 6)))
        tailles = [len(e) for e in ens if e]
        print(f'  piliers par incident identifiable : moyenne {np.mean(tailles):.2f}')
        dep = dependance(ens)
        print('  paire   observé  attendu  obs/att    phi   p Fisher')
        for (a, b), n11, att, lift, phi, p in dep:
            print(f'  P{a}-P{b}   {n11:5d}  {att:7.1f}  {lift:7.2f}  {phi:+.3f}   {p:.4f}')
        co = np.array([d[1] for d in dep], dtype=float)
        phis = np.array([d[4] for d in dep])
        ok = ~np.isnan(phis)
        rho_co = spearmanr(s_exp, co).statistic
        rho_phi = spearmanr(s_exp[ok], phis[ok]).statistic
        sig = sum(1 for d in dep if d[5] < 0.05)
        print(f'  corrélation de rang avec S posée : co-occurrences brutes {rho_co:+.2f} ; phi {rho_phi:+.2f} '
              f'sur {int(ok.sum())} paires ; paires jamais observées {int(np.sum(co == 0))} ; '
              f'paires à p Fisher < 0,05 : {sig}')
        resultats[niveau] = (rho_co, rho_phi)

    print('\nS POSÉE (dix paires) : ' + ' ; '.join(f'P{a}-P{b} {v:.3f}' for (a, b), v in zip(PAIRES, s_exp)))
    print('\nLECTURE DÉRIVÉE DES COMPTES')
    print('  les co-occurrences brutes ordonnent surtout les paires par la fréquence de P1 ; la mesure qui')
    print('  compare à S est phi, corrigée des marges. Son signe et son ordre ne valent que pour ce codage.')
    print('\nLimites : un seul codeur, non humain ; descriptions de presse et de procédures, biais de narration ;')
    print('le niveau large ajoute des défaillances supposées ; aucune direction n’est codée, donc A reste hors')
    print('de portée ; la co-occurrence dans un texte n’est pas la co-occurrence de défaillances du modèle.')
    print('\nGRANDEURS CITÉES')
    for niveau, (rc, rp) in resultats.items():
        print(f'  rho_{niveau}_brut {rc:.2f} rho_{niveau}_phi {rp:.2f}')


if __name__ == '__main__':
    main()
