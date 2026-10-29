import importlib.util
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parents[1]))
import institut_incertitude as ii

spec = importlib.util.spec_from_file_location(
    "reechant125", LAB / "1_fondations" / "125_reechantillonnage_annuel_institut.py")
reechant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reechant)


def test_tirage_par_annees_a_la_bonne_longueur_et_reste_dans_les_bornes():
    rng = np.random.default_rng(1)
    idx = reechant.tirage_annees(rng, np.arange(2000, 2026), 'annees')
    assert len(idx) == 26 and idx.min() >= 0 and idx.max() < 26


def test_tirage_par_blocs_est_fait_de_blocs_consecutifs():
    rng = np.random.default_rng(2)
    n = 26
    idx = reechant.tirage_annees(rng, np.arange(2000, 2026), 'blocs')
    assert len(idx) == n and idx.min() >= 0 and idx.max() < n
    b = reechant.LONGUEUR_BLOC
    for d in range(0, n-b, b):
        bloc = idx[d:d+b]
        assert np.all(np.diff(bloc) == 1)


def test_facteur_un_reproduit_le_moteur_et_le_capital_croit_avec_la_frequence():
    cal = ii.REFERENCE
    base = ii.grille_adaptee(cal)
    for essai in range(5):
        moteur = ii.FourierIncidents(cal, step=base*2**essai)
        try:
            v1 = reechant.capital_facteur(moteur, 0, 'additive', 1.0)
            v_moteur, _ = moteur.capital(0, 'additive')
            assert v1 == v_moteur
            assert reechant.capital_facteur(moteur, 0, 'additive', 1.3) > v1
            assert reechant.capital_facteur(moteur, 0, 'additive', 0.8) < v1
            return
        except ArithmeticError:
            continue
    raise AssertionError('aucune grille admissible')


def test_centiles_ordonnes():
    c = reechant.centiles(np.arange(1, 101))
    assert c['p5'] < c['p50'] < c['p95'] and abs(c['moyenne']-50.5) < 1e-9
