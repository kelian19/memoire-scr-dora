import importlib.util
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parents[1]))
import institut_incertitude as ii

spec = importlib.util.spec_from_file_location(
    "dependance126", LAB / "8_benchmarks" / "126_dependance_couts_incident_institut.py")
dep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dep)

MOTEUR = ii.FourierIncidents(ii.REFERENCE, step=2.0)


def test_blocs_couvrent_les_composantes():
    assert dep.tailles_blocs(5, 2) == [2, 2, 1]
    assert dep.tailles_blocs(5, 5) == [5]
    assert dep.tailles_blocs(4, 1) == [1, 1, 1, 1]
    assert all(sum(dep.tailles_blocs(k, b)) == k for k in range(1, 6) for b in range(1, 6))


def test_extremes_du_melange_redonnent_additive_et_amplifiee():
    for mask in (0, 15):
        for rho, regle in ((0.0, 'additive'), (1.0, 'amplifiee')):
            v = dep.capital_incident(MOTEUR, mask, lambda m, pu, lk, r=rho: dep.incident_melange(m, pu, lk, r))
            assert abs(v - MOTEUR.capital(mask, regle)[0]) < 1e-9


def test_blocs_extremes_redonnent_additive_et_amplifiee():
    for b, regle in ((1, 'additive'), (5, 'amplifiee')):
        v = dep.capital_incident(MOTEUR, 15, lambda m, pu, lk, b=b: dep.incident_blocs(m, pu, lk, b))
        assert abs(v - MOTEUR.capital(15, regle)[0]) < 1e-9


def test_un_seul_pilier_rend_toutes_les_dependances_equivalentes():
    seul = np.array([0., 1., 0., 0., 0., 0.])
    valeurs = {dep.capital_incident(MOTEUR, 0, lambda m, pu, lk, f=f: f(m, pu, seul)) for f in (
        lambda m, pu, lk: dep.incident_melange(m, pu, lk, .3),
        lambda m, pu, lk: dep.incident_blocs(m, pu, lk, 2),
        lambda m, pu, lk: dep.incident_exposant(m, pu, lk, 1.0))}
    assert len(valeurs) == 1


def test_capital_croit_avec_la_probabilite_de_choc_commun():
    caps = [dep.capital_incident(MOTEUR, 15, lambda m, pu, lk, r=r: dep.incident_melange(m, pu, lk, r))
            for r in (0, .25, .5, .75, 1)]
    assert all(b >= a for a, b in zip(caps, caps[1:]))


def test_transformee_d_incident_est_une_sous_probabilite():
    pu = ii.REFERENCE.pu*1.2
    loi_k = ii.cardinal(True, True)
    for t in (dep.incident_melange(MOTEUR, pu, loi_k, .4), dep.incident_blocs(MOTEUR, pu, loi_k, 2)):
        # Le terme nul est la masse sur la portée, amortie par l'inclinaison : inférieure à un, proche de un.
        assert .95 < t[0].real <= 1 and np.abs(t).max() <= 1 + 1e-12
