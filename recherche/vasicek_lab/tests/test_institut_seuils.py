"""Propriétés utiles des diagnostics POT, indépendantes de la donnée sous licence."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import institut_seuils as s


class DiagnosticsPOT(unittest.TestCase):
    def test_ks_utilise_les_deux_cotes_du_saut(self):
        # Deux probabilités théoriques .8 et .9 : D- = .8, D+ = .1.
        exces = -np.log([.2, .1])
        self.assertAlmostEqual(s.ks_stat(exces, 0, 1), .8)

    def test_quantile_limite_exponentielle(self):
        self.assertAlmostEqual(s.quantile(20, 0, 3, .15), 20+3*np.log(30))

    def test_pwm_et_hill_sont_equivariants_en_echelle(self):
        exces = np.array([1, 2, 3, 4, 7, 12, 25.])
        xi, sigma = s.pwm(exces)
        xi2, sigma2 = s.pwm(100*exces)
        self.assertAlmostEqual(xi, xi2)
        self.assertAlmostEqual(sigma2, 100*sigma)
        self.assertAlmostEqual(s.hill(exces, 20), s.hill(100*exces, 2000))

    def test_donnees_hors_support_refusees(self):
        with self.assertRaises(ValueError):
            s.emv([1, -1, 2])


if __name__ == '__main__':
    unittest.main()
