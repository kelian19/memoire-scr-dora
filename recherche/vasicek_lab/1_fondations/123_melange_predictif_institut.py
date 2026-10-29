#!/usr/bin/env python3
"""123 : mélange prédictif annuel sur les calibrations du bootstrap conjoint (120).

Relit les 2 000 calibrations de 120.json (xi, sigma, taux de dépassement), recalcule pour
chacune la loi annuelle complète aux états conforme et non conforme, sous les quatre
règles de perte, puis compare trois objets distincts :
  - le quantile au calage central ;
  - la distribution des quantiles conditionnels (120) ;
  - le quantile du mélange des lois annuelles (ce script).
Le mélange est une construction prédictive empirique, pas une loi prédictive bayésienne.
Il ne propage pas l'incertitude sur la fréquence, la dispersion, les multiplicateurs
entre états ni la cascade d'expert, qui restent conditionnants.
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

NIVEAUX = (0.95, 0.99, 0.995, 0.999)
ETATS = {"conforme": 0, "non_conforme": 15}
GRILLE = np.geomspace(1.0, 5e6, 6000)


def cdf_annuelle(moteur, mask, regle):
    """Même calcul que FourierIncidents.capital, mais renvoie la fonction de répartition."""
    cx = ii.cx
    lam = cx.LAM_NC if mask & 1 else cx.LAM_C
    pu = moteur.cal.pu * (1.20 if mask & 2 else .85)
    loi_k = ii.cardinal(bool(mask & 4), bool(mask & 8))
    incident = moteur.transformee_incident(pu, loi_k, regle)
    masses = np.fft.irfft(ii.pgf_nb(incident, lam), n=moteur.n)/moteur.attenuation
    if not np.isfinite(masses).all() or masses.min() < -1e-8:
        raise ArithmeticError("Inversion instable")
    return np.cumsum(masses)


def quantile_grille(cdf, step, alpha):
    idx = int(np.searchsorted(cdf, alpha))
    return idx*step


def tracer_figure(sortie, quantiles):
    """Trois objets de risque, règle additive, niveau 99,5 % : le calage central, la
    distribution des quantiles conditionnels (120) et le quantile du mélange."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import style_nexialog as sn
    sn.appliquer(9)
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI', 'sans-serif']
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.55), sharey=True)
    titres = {'conforme': 'État conforme', 'non_conforme': 'État non conforme'}
    for ax, e in zip(axes, ETATS):
        q = np.sort(np.array(quantiles[('additive', e)][.995]))
        y = np.arange(1, len(q)+1)/len(q)
        r = sortie['resultats']['additive'][e]
        ax.plot(q, y, color=sn.ENCRE_2, lw=1.8)
        ax.axvline(r['central'], color=sn.ENCRE, lw=1.4, ls='--')
        ax.axvline(r['melange']['0.995'], color=sn.CATEGORIEL[0], lw=1.8)
        ax.set_xscale('log')
        ax.set_xlim(r['conditionnels_p5_p95'][0]*0.55, np.percentile(q, 99.5)*1.2)
        bas, haut = ax.get_xlim()
        graduations = [t for t in (2000, 5000, 10000, 20000, 50000,
                                   100000, 200000, 300000) if bas <= t <= haut]
        ax.set_xticks(graduations)
        ax.set_xticks([], minor=True)
        ax.set_xticklabels([f'{t:,}'.replace(',', ' ') for t in graduations], fontsize=7.5)
        ax.set_title(titres[e], fontsize=9.5, color=sn.ENCRE)
        ax.set_xlabel('Quantile annuel à 99,5 % (M€, échelle log)', fontsize=8.5)
        ax.text(r['central']*0.97, 0.06, f"calage central\n{r['central']:,.0f}".replace(',', ' '),
                color=sn.ENCRE, fontsize=7.5, ha='right', va='bottom')
        ax.text(r['melange']['0.995'], 0.55, f" mélange\n {r['melange']['0.995']:,.0f}".replace(',', ' '),
                color=sn.CATEGORIEL[0], fontsize=7.5, ha='left', va='center')
        ax.grid(True, axis='y', alpha=.5)
    axes[0].set_ylabel('Part des 2 000 calibrations\ndont le quantile est inférieur', fontsize=8.5)
    fig.tight_layout()
    chemin = REPO/'recherche/vasicek_lab/figures_inst/melange_predictif_annuel.png'
    fig.savefig(chemin, dpi=200)
    plt.close(fig)
    return chemin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entree', type=Path, default=REPO/'resultats/institut_approfondissements/120.json')
    parser.add_argument('--sortie', type=Path, default=REPO/'resultats/institut_approfondissements/123.json')
    parser.add_argument('--limite', type=int, default=0, help='nombre de réplications (0 = toutes)')
    parser.add_argument('--figure-seule', action='store_true',
                        help='retrace la figure depuis le JSON existant, sans recalcul')
    args = parser.parse_args()
    if args.figure_seule:
        sortie = json.load(open(args.sortie, encoding='utf-8'))
        quantiles = {('additive', e): {.995: v} for e, v in sortie['quantiles_conditionnels_additive_995'].items()}
        print(f'Figure : {tracer_figure(sortie, quantiles).name}')
        return
    debut = time.monotonic()
    d = json.load(open(args.entree, encoding='utf-8'))
    reps = d['replications'][:args.limite] if args.limite else d['replications']
    print('123 — MÉLANGE PRÉDICTIF ANNUEL, VERSION INSTITUT', flush=True)
    print(f'{len(reps)} calibrations lues dans 120.json (bootstrap conjoint, seuil fixé).')
    print('Montants en M EUR du périmètre financier étudié. Mélange empirique, conditionnel aux')
    print('hypothèses du moteur ; ni loi prédictive bayésienne ni intervalle de confiance.', flush=True)

    # Contrôle d'identité : le quantile lu sur la fonction de répartition retombe sur le moteur.
    cal0 = ii.Calibration(**d['calibration_locale'])
    for essai in range(5):
        step0 = ii.grille_adaptee(cal0)*2**essai
        moteur0 = ii.FourierIncidents(cal0, step=step0)
        try:
            for regle in ii.REGLES:
                for mask in (0, 15):
                    v, _ = moteur0.capital(mask, regle)
                    q = quantile_grille(cdf_annuelle(moteur0, mask, regle), step0, .995)
                    assert abs(v-q) < 1e-9, (regle, mask, v, q)
            break
        except ArithmeticError:
            if essai == 4:
                raise
    print('Contrôle : le quantile lu sur la répartition reproduit le moteur aux 8 cellules centrales.')

    cles = [(r_, e) for r_ in ii.REGLES for e in ETATS]
    somme = {c: np.zeros(len(GRILLE)) for c in cles}
    somme_bornee = {c: np.zeros(len(GRILLE)) for c in cles}
    n_bornee = 0
    quantiles = {c: {a: [] for a in NIVEAUX} for c in cles}
    n_ok = 0
    for rep in reps:
        cal = ii.Calibration(**rep['calibration'])
        moteur = ii.FourierIncidents(cal, step=rep['pas'])
        try:
            lot = {}
            for regle in ii.REGLES:
                for etat, mask in ETATS.items():
                    cdf = cdf_annuelle(moteur, mask, regle)
                    lot[(regle, etat)] = cdf
        except ArithmeticError:
            continue
        xs = np.arange(moteur.n)*moteur.step
        borne = rep['calibration']['xi'] < 1
        n_bornee += int(borne)
        for c, cdf in lot.items():
            interp = np.interp(GRILLE, xs, cdf)
            somme[c] += interp
            if borne:
                somme_bornee[c] += interp
            for a in NIVEAUX:
                quantiles[c][a].append(quantile_grille(cdf, moteur.step, a))
        n_ok += 1
        if n_ok % 250 == 0:
            print(f'  {n_ok}/{len(reps)} ; {time.monotonic()-debut:.0f} s', flush=True)
    print(f'Calibrations traitées : {n_ok}/{len(reps)}.')

    def q_melange(c, a):
        f = somme[c]/n_ok
        i = int(np.searchsorted(f, a))
        if i >= len(GRILLE):
            return float('inf')
        return float(GRILLE[i])

    central = {r_: {e: float(moteur0.capital(m, r_)[0]) for e, m in ETATS.items()} for r_ in ii.REGLES}
    sortie = {'experience': '123 mélange prédictif annuel', 'n_calibrations': n_ok,
              'resolution_relative_grille': float(GRILLE[1]/GRILLE[0]-1), 'niveaux': list(NIVEAUX),
              'resultats': {}}
    print('\nQUANTILE À 99,5 % : central, médiane des quantiles conditionnels, moyenne des quantiles, mélange')
    print(f"{'règle':10s} {'état':13s} {'central':>9s} {'médiane':>9s} {'moyenne':>9s} {'mélange':>9s}  mélange/central")
    for r_ in ii.REGLES:
        sortie['resultats'][r_] = {}
        for e in ETATS:
            c = (r_, e)
            q = np.array(quantiles[c][.995])
            ligne = {'central': central[r_][e], 'mediane_conditionnels': float(np.median(q)),
                     'moyenne_conditionnels': float(q.mean()),
                     'melange': {str(a): q_melange(c, a) for a in NIVEAUX},
                     'conditionnels_p5_p95': [float(np.percentile(q, 5)), float(np.percentile(q, 95))]}
            sortie['resultats'][r_][e] = ligne
            print(f"{r_:10s} {e:13s} {ligne['central']:9.0f} {ligne['mediane_conditionnels']:9.0f} "
                  f"{ligne['moyenne_conditionnels']:9.0f} {ligne['melange']['0.995']:9.0f}  "
                  f"{ligne['melange']['0.995']/ligne['central']:.3f}")
        c_, n_ = sortie['resultats'][r_]['conforme'], sortie['resultats'][r_]['non_conforme']
        ecart_m = n_['melange']['0.995']-c_['melange']['0.995']
        rapport_m = n_['melange']['0.995']/c_['melange']['0.995']
        sortie['resultats'][r_]['contraste_melange'] = {'ecart': ecart_m, 'rapport': rapport_m}
        print(f"{r_:10s} contraste du mélange : écart {ecart_m:.0f}, rapport {rapport_m:.3f} "
              f"(central {n_['central']-c_['central']:.0f}, {n_['central']/c_['central']:.3f})")

    print('\nÉCART RELATIF DU MÉLANGE AU CALAGE CENTRAL À 99,5 %, RÈGLE ADDITIVE (grandeurs citées)')
    for e in ETATS:
        r = sortie['resultats']['additive'][e]
        print(f"  {e:13s} {100*(r['melange']['0.995']/r['central']-1):.0f} % ; "
              f"{100*(r['melange']['0.995']/r['central']-1):.1f} %")
    print(f"  calibrations avec xi >= 1 : {sum(1 for r_ in reps if r_['calibration']['xi'] >= 1)} sur {len(reps)}")
    print('\nSENSIBILITÉ AU NIVEAU, RÈGLE ADDITIVE : mélange / central')
    for e in ETATS:
        m = sortie['resultats']['additive'][e]['melange']
        cdf0 = cdf_annuelle(moteur0, ETATS[e], 'additive')
        ctr = {a: quantile_grille(cdf0, moteur0.step, a) for a in NIVEAUX}
        print(f"  {e:13s} " + "  ".join(f"{100*a:g} %: {m[str(a)]:.0f} / {ctr[a]:.0f}" for a in NIVEAUX))
        sortie['resultats']['additive'][e]['central_par_niveau'] = {str(a): float(v) for a, v in ctr.items()}

    print('\nVARIANTE SANS LES AJUSTEMENTS xi >= 1 (moyenne théorique infinie), mélange à 99,5 % :')
    sortie['variante_xi_inferieur_un'] = {'n_calibrations': n_bornee}
    for r_ in ii.REGLES:
        v = {}
        for e in ETATS:
            f = somme_bornee[(r_, e)]/n_bornee
            i = int(np.searchsorted(f, .995))
            v[e] = float(GRILLE[i]) if i < len(GRILLE) else float('inf')
        sortie['variante_xi_inferieur_un'][r_] = {**v, 'rapport': v['non_conforme']/v['conforme']}
        print(f"  {r_:10s} conforme {v['conforme']:.0f} ; non conforme {v['non_conforme']:.0f} ; "
              f"rapport {v['non_conforme']/v['conforme']:.3f}")
    print(f'  ({n_bornee} calibrations sur {n_ok}.)')
    sortie['quantiles_conditionnels_additive_995'] = {
        e: [round(float(v), 3) for v in quantiles[('additive', e)][.995]] for e in ETATS}
    figure = tracer_figure(sortie, quantiles)
    print(f'Figure : {figure.name}')
    sortie['duree_secondes'] = time.monotonic()-debut
    json.dump(sortie, open(args.sortie, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f"\nRésolution de la grille de lecture du mélange : {sortie['resolution_relative_grille']*100:.2f} % en relatif.")
    print('Le mélange ignore les masses au-delà de la portée de chaque calibration (cdf bornée à son dernier point).')
    print(f'Durée {sortie["duree_secondes"]:.1f} s ; résultats agrégés : {args.sortie.name}')


if __name__ == '__main__':
    main()
