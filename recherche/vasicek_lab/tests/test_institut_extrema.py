import importlib.util
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parents[1]))
import partial_id as pid

spec = importlib.util.spec_from_file_location(
    "extrema127", LAB / "6_contagion" / "127_extrema_hors_sommets_institut.py")
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)


def test_sommets_sont_1024_et_aux_bords():
    S = pid.decompose(pid.expert_matrix())[0]
    smax = S[pid.IU]
    V = ext.sommets(smax)
    assert V.shape == (1024, 10) and all(ext.est_sommet(v, smax) for v in V[:50])
    assert len({tuple(np.sign(v)) for v in V}) == 1024


def test_maximiser_axe_quadratique_concave_et_convexe():
    assert ext.maximiser_axe(0, 0, -1, 2, 1) == 0.0
    assert ext.maximiser_axe(0, 0, 1, 2, 1) in (-2, 2)
    assert abs(ext.maximiser_axe(0, 1, 1, 2, -1)+0.5) < 1e-12
    assert abs(ext.maximiser_axe(0, 1, -1, 2, 1)-0.5) < 1e-12


def test_coefficients_axe_exacts_pour_un_polynome_de_degre_deux():
    smax = np.array([.5, 1.0])
    f = lambda a: 3+2*a[0]-4*a[0]**2+a[1]*7
    c0, c1, c2 = ext.coefficients_axe(f, np.array([.1, .2]), 0, smax)
    assert abs(c1-2) < 1e-12 and abs(c2+4) < 1e-12 and abs(c0-(3+1.4)) < 1e-12
