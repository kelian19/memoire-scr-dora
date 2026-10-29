import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.stats import genpareto

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parents[1]))

spec = importlib.util.spec_from_file_location(
    "temporel124", LAB / "1_fondations" / "124_validation_temporelle_institut.py")
temporel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(temporel)


def echantillon(annees, sigma_par_annee, xi=0.5, par_annee=60, graine=7):
    rng = np.random.default_rng(graine)
    x, t = [], []
    for a in annees:
        x.extend(genpareto.rvs(c=xi, scale=sigma_par_annee(a), size=par_annee, random_state=rng))
        t.extend([a]*par_annee)
    return np.array(x), np.array(t, dtype=float)


def test_stationnaire_retrouve_les_parametres():
    x, _ = echantillon(range(2000, 2020), lambda a: 50.0)
    aj = temporel.ajuster_stationnaire(x)
    assert abs(aj['xi']-0.5) < 0.08 and abs(aj['sigma']/50-1) < 0.12


def test_derive_retrouve_une_pente_et_une_pente_nulle():
    x, t = echantillon(range(2000, 2020), lambda a: 40*np.exp(0.05*(a-2010)))
    assert abs(temporel.ajuster_derive(x, t)['pente']-0.05) < 0.015
    x, t = echantillon(range(2000, 2020), lambda a: 40.0)
    assert abs(temporel.ajuster_derive(x, t)['pente']) < 0.015


def test_regimes_retrouve_la_rupture():
    x, t = echantillon(range(2000, 2021), lambda a: 20.0 if a <= 2009 else 80.0)
    aj = temporel.ajuster_regimes(x, t)
    assert abs(aj['rupture']-2009) <= 1
    assert aj['sigma_apres'] > 2.5*aj['sigma_avant']


def test_logit_detecte_une_tendance_croissante():
    rng = np.random.default_rng(3)
    t = np.repeat(np.arange(2000, 2025), 40).astype(float)
    p = 1/(1+np.exp(-(-2.0+0.1*(t-2012))))
    y = (rng.random(len(t)) < p).astype(float)
    aj = temporel.ajuster_logit(y, t)
    assert 0.05 < aj['b'] < 0.16
    assert temporel.p_logit(aj, 2024) > temporel.p_logit(aj, 2000)


def test_echelle_selon_le_modele():
    assert temporel.echelle({'modele': 'S', 'sigma': 30.0}, 2030) == 30.0
    assert temporel.echelle({'modele': 'R', 'sigma_apres': 70.0}, 2030) == 70.0
    d = {'modele': 'D', 'log_sigma0': np.log(40.0), 'pente': 0.1, 'centre': 2010.0}
    assert abs(temporel.echelle(d, 2010)-40.0) < 1e-9
