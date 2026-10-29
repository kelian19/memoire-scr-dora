import importlib.util
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parents[1]))

spec = importlib.util.spec_from_file_location(
    "residus128", LAB / "2_donnees" / "128_residus_compenses_comptage_institut.py")
res = importlib.util.module_from_spec(spec)
spec.loader.exec_module(res)

T = np.arange(1500)/365.25


def test_excitation_est_strictement_previsible():
    k = np.zeros(10)
    k[3] = 5.0
    e = res.excitation(k, 0.7)
    assert np.all(e[:4] == 0) and abs(e[4]-5*np.exp(-0.7)) < 1e-12 and abs(e[5]-5*np.exp(-1.4)) < 1e-12


def test_intensite_sans_excitation_ne_depend_pas_des_comptes():
    p = np.array([np.log(.8), .1])
    a = res.intensite('M0', p, T, np.zeros(len(T)))
    b = res.intensite('M0', p, T, np.ones(len(T))*50)
    assert np.allclose(a, b)


def test_ajustement_retrouve_la_tendance_et_le_branchement():
    rng = np.random.default_rng(3)
    p = np.array([np.log(.8), .15, np.log(.4), np.log(.9)])
    k = res.simuler('M1', p, T, 1.3, rng)
    est, _ = res.ajuster('M1', T, k)
    vrai = np.exp(p[2])/(np.exp(np.exp(p[3]))-1)
    n_est = res.diagnostics('M1', est, T, k)['branchement']
    assert abs(n_est-vrai) < 0.2


def test_residus_compenses_sans_derive_sous_le_bon_modele():
    rng = np.random.default_rng(4)
    p = np.array([np.log(.9), .1])
    k = res.simuler('M0', p, T, 1.4, rng)
    est, _ = res.ajuster('M0', T, k)
    d = res.diagnostics('M0', est, T, k)
    assert abs(d['derive']) < 0.03*d['lam_total'] and max(abs(np.array(d['acf']))) < 0.1
