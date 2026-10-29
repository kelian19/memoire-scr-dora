"""Contrôles scientifiques du pilote de remédiation, indépendants du manuscrit."""
from pathlib import Path
import sys
import unittest

import numpy as np

LAB = Path(__file__).resolve().parents[1]
for p in (LAB.parents[1], LAB):
    sys.path.insert(0, str(p))
import canaux_conformite as cx
import institut_incertitude as ii
import institut_remediation as ir
import partial_id as pid


class RemediationTests(unittest.TestCase):
    def test_marche_retrouve_les_ensembles_historiques(self):
        for g in (cx.G_C, cx.G_NC):
            w = pid.expert_matrix(g)
            for j in ir.PILIERS:
                reference = cx.eng.cascade_set_dist(j, g)
                observe = ir.ensembles_marche(w, j)
                self.assertEqual(set(reference), set(observe))
                for ens in reference:
                    self.assertAlmostEqual(reference[ens], observe[ens], places=13)

    def test_cardinaux_retrouvent_119(self):
        for propagation in (False, True):
            w = pid.expert_matrix(cx.G_NC if propagation else cx.G_C)
            for accumulation in (False, True):
                np.testing.assert_allclose(
                    ir.cardinal(w, phi=cx.PHICS_NC if accumulation else None),
                    ii.cardinal(propagation, accumulation), atol=1e-13, rtol=0)

    def test_branchement_retrouve_les_tables_historiques(self):
        for g in (cx.G_C, cx.G_NC):
            expected = np.zeros(6)
            for i, j in enumerate(ir.PILIERS):
                for ens, proba in cx.eng.cascade_set_dist_branching(j, g).items():
                    expected[len(ens)] += ir.POIDS[i]*proba
            np.testing.assert_allclose(ir.cardinal(pid.expert_matrix(g), "branchement"),
                                       expected, atol=1e-12, rtol=0)

    def test_bifurcation_analytique_distingue_les_noyaux(self):
        w = np.zeros((5, 5)); w[0, 1:3] = [.2, .3]
        dist = ir.ensembles_marche(w, 1)
        self.assertAlmostEqual(dist[frozenset([1])], .5)
        self.assertAlmostEqual(dist[frozenset([1, 2])], .2)
        self.assertAlmostEqual(dist[frozenset([1, 3])], .3)
        self.assertNotIn(frozenset([1, 2, 3]), dist)
        cards, _ = pid.card_dist_all(w)
        self.assertAlmostEqual(cards[0, 3], .2*.3)

    def test_action_nulle_et_lignes_non_renormalisees(self):
        w = pid.expert_matrix(); copy = w.copy()
        v, phi = ir.appliquer(w, (1, 4), 0., .68, "emission_et_tiers")
        np.testing.assert_array_equal(v, w); self.assertEqual(phi, .68)
        v, phi = ir.appliquer(w, (1, 4), .5, .68, "emission_et_tiers")
        np.testing.assert_allclose(v[[0, 3]], w[[0, 3]]*.5)
        np.testing.assert_allclose(v[[1, 2, 4]], w[[1, 2, 4]])
        np.testing.assert_array_equal(w, copy); self.assertEqual(phi, .34)

    def test_choc_zero_ne_signifie_pas_cascade_p4(self):
        w = pid.expert_matrix()
        a = ir.cardinal(w, phi=0)
        b = ir.cardinal(w, phi=None)
        self.assertGreater(a[1], b[1])

    def test_tous_les_arcs_coupes_sans_choc_donne_k_un(self):
        for noyau in ir.NOYAUX:
            v, _ = ir.appliquer(pid.expert_matrix(), ir.PILIERS, 1.)
            np.testing.assert_allclose(ir.cardinal(v, noyau), [0, 1, 0, 0, 0, 0], atol=1e-12)

    def test_perte_unique_ne_depend_pas_des_actions_de_propagation(self):
        moteur = ii.FourierIncidents()
        for noyau in ir.NOYAUX:
            w = pid.expert_matrix()
            a = moteur.capital(15, "unique", loi_k=ir.cardinal(w, noyau, .68))[0]
            v, phi = ir.appliquer(w, ir.PILIERS, 1., .68, "emission_et_tiers")
            b = moteur.capital(15, "unique", loi_k=ir.cardinal(v, noyau, phi))[0]
            self.assertEqual(a, b)

    def test_minimax_connu_avec_abstention(self):
        q = [[100, 80, 95], [100, 95, 80]]
        acts = [(), (1,), (2,)]
        d = ir.decision(q, acts, 1)
        self.assertEqual(set(d['actions_minimax']), {'P1', 'P2'})
        self.assertEqual(d['regret_minimal'], 15)
        d = ir.decision([[100, 110], [100, 105]], [(), (1,)], 1)
        self.assertEqual(d['actions_minimax'], ['aucune'])

    def test_probabilites_et_actions_invalides_rejetees(self):
        w = np.zeros((5, 5)); w[0, 1:3] = [.8, .8]
        with self.assertRaises(ValueError):
            ir.cardinal(w, "marche")
        with self.assertRaises(ValueError):
            ir.appliquer(w, (1, 1))
        with self.assertRaises(ValueError):
            ir.appliquer(w, (1,), efficacite=1.1)

    def test_objectif_relatif_peut_changer_le_choix(self):
        # Risque élevé : A évite 100, B 30. Risque faible : A évite 0, B 50.
        # Le pire regret absolu est 50 pour A, 70 pour B ; relatif : 50 % et 7 %.
        q = [[1000, 900, 970], [100, 100, 50]]
        acts = [(), (1,), (2,)]
        self.assertEqual(ir.decision(q, acts, 1)['actions_minimax'], ['P1'])
        self.assertEqual(ir.decision(q, acts, 1, relatif=True)['actions_minimax'], ['P2'])


if __name__ == '__main__':
    unittest.main()
