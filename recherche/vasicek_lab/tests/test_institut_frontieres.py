"""Contrôles de l'action découplée et de la conversion capital/coût annuel."""
from pathlib import Path
import sys
import unittest

import numpy as np

LAB = Path(__file__).resolve().parents[1]
for p in (LAB.parents[1], LAB):
    sys.path.insert(0, str(p))
import institut_remediation as ir
import partial_id as pid


class FrontieresTests(unittest.TestCase):
    def test_les_deux_efficacites_sont_independantes(self):
        w = pid.expert_matrix()
        v, phi = ir.appliquer_ciblee(w, 4, .25, .68, .75)
        np.testing.assert_allclose(v[3], w[3]*.75)
        self.assertAlmostEqual(phi, .17)
        v, phi = ir.appliquer_ciblee(w, 4, 0., .68, 1.)
        np.testing.assert_array_equal(v, w)
        self.assertEqual(phi, 0.)

    def test_retrouve_les_actions_de_121(self):
        w = pid.expert_matrix()
        for j in (1, 4):
            for f in ir.FAMILLES:
                a, p = ir.appliquer(w, (j,), .5, .68, f)
                b, q = ir.appliquer_ciblee(w, j, .5, .68,
                    .5 if j == 4 and f == 'emission_et_tiers' else 0.)
                np.testing.assert_array_equal(a, b)
                self.assertEqual(p, q)

    def test_absence_du_mecanisme_tiers_reste_absente(self):
        w, phi = ir.appliquer_ciblee(pid.expert_matrix(), 4, .5, None, .75)
        self.assertIsNone(phi)
        np.testing.assert_allclose(ir.cardinal(w, phi=phi), ir.cardinal(w))

    def test_conversion_stock_flux_et_signe_du_surcout(self):
        # P1 libère 100 M, P4 150 M ; à 6 %, 6 et 9 M par an.
        e = ir.economie_capital(1000, 900, 850, .06)
        self.assertEqual(e['plafond_cout_P1'], 6.)
        self.assertEqual(e['plafond_cout_P4'], 9.)
        self.assertEqual(e['surcout_P4_vs_P1'], 3.)
        self.assertEqual(ir.choisir_cout(1000, 900, 850, 2, 4), ['P4'])
        self.assertEqual(ir.choisir_cout(1000, 900, 850, 2, 6), ['P1'])
        self.assertEqual(set(ir.choisir_cout(1000, 900, 850, 2, 5)), {'P1', 'P4'})

    def test_abstention_si_couts_excessifs_ou_capital_en_hausse(self):
        self.assertEqual(ir.choisir_cout(1000, 900, 850, 7, 10), ['aucune'])
        self.assertEqual(ir.choisir_cout(1000, 1100, 1050, 0, 0), ['aucune'])
        self.assertEqual(set(ir.choisir_cout(1000, 1000, 1000, 0, 0)), {'aucune', 'P1', 'P4'})

    def test_effets_et_couts_invalides_rejetes(self):
        w = pid.expert_matrix()
        for kwargs in ({'pilier': 1, 'efficacite_tiers': .2},
                       {'pilier': 4, 'efficacite_tiers': 1.1}):
            with self.assertRaises(ValueError):
                ir.appliquer_ciblee(w, efficacite_emission=.5, **kwargs)
        with self.assertRaises(ValueError):
            ir.economie_capital(100, 90, 80, 0.)
        with self.assertRaises(ValueError):
            ir.choisir_cout(100, 90, 80, -1, 0)

    def test_frontiere_analytique_et_zone_de_resolution(self):
        r = ir.encadrer_bascule(lambda t: 100*(t-.3), tolerance=1., precision=.001)
        self.assertEqual(r['statut'], 'bascule_encadree')
        lo, hi = r['intervalle']
        self.assertLessEqual(lo, .29)
        self.assertGreaterEqual(hi, .31)
        self.assertLess(.29-lo, .0011)
        self.assertLess(hi-.31, .0011)

    def test_frontieres_absentes_et_plateau_ne_creent_pas_de_seuil(self):
        self.assertEqual(ir.encadrer_bascule(lambda t: 2+t)['statut'], 'P4_sur_toute_la_grille')
        self.assertEqual(ir.encadrer_bascule(lambda t: -2+t)['statut'], 'P1_sur_toute_la_grille')
        r = ir.encadrer_bascule(lambda t: 0., tolerance=1.)
        self.assertIsNone(r['intervalle'])
        self.assertEqual(r['statut'], 'egalite_numerique_sur_toute_la_grille')

    def test_avantage_decroissant_rejete_par_dichotomie(self):
        with self.assertRaises(ValueError):
            ir.encadrer_bascule(lambda t: 1-2*t)


if __name__ == '__main__':
    unittest.main()
