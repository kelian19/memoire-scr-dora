import importlib.util
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parents[1]))
import institut_incertitude as ii

spec = importlib.util.spec_from_file_location(
    "melange123", LAB / "1_fondations" / "123_melange_predictif_institut.py")
melange = importlib.util.module_from_spec(spec)
spec.loader.exec_module(melange)


def moteur_central():
    cal = ii.REFERENCE
    for essai in range(5):
        step = ii.grille_adaptee(cal)*2**essai
        moteur = ii.FourierIncidents(cal, step=step)
        try:
            moteur.capital(15, "amplifiee")
            return moteur
        except ArithmeticError:
            continue
    raise AssertionError("aucune grille admissible")


def test_cdf_croissante_et_quantile_identique_au_moteur():
    moteur = moteur_central()
    for regle in ii.REGLES:
        for mask in (0, 15):
            cdf = melange.cdf_annuelle(moteur, mask, regle)
            assert np.all(np.diff(cdf) > -1e-8)
            assert .995 <= cdf[-1] <= 1.00001
            valeur, _ = moteur.capital(mask, regle)
            assert melange.quantile_grille(cdf, moteur.step, .995) == valeur


def test_melange_de_calibrations_identiques_redonne_le_quantile_central():
    moteur = moteur_central()
    xs = np.arange(moteur.n)*moteur.step
    cdf = melange.cdf_annuelle(moteur, 0, "additive")
    f = np.mean([np.interp(melange.GRILLE, xs, cdf) for _ in range(5)], axis=0)
    i = int(np.searchsorted(f, .995))
    central, _ = moteur.capital(0, "additive")
    assert abs(melange.GRILLE[i]/central-1) < 0.003  # résolution de la grille de lecture
