"""Cas vérifiables du calcul pédagogique, indépendants des bases et du manuscrit."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exemples.calcul_synthetique import Etat, mesures, probabilites_ensembles, simuler


def test_moyenne_analytique_et_somme_des_incidents():
    # Sans propagation ni choc commun, un incident touche exactement un pilier.
    # E[L] = lambda * p_u * (u + sigma/(1-xi)).
    etat = Etat("cas analytique", 2, 1, 0, 0)
    resultat = simuler(12_000, 2026, (etat,))[0]
    attendu = 2 * (1 + .5 / (1 - .3))
    assert resultat["moyenne_theorique"] == pytest.approx(attendu)
    assert abs(resultat["mesures"]["moyenne"] - attendu) < 6 * resultat["mesures"]["erreur_type_moyenne"]
    annee = resultat["exemple_annee"]
    assert sum(i["perte"] for i in annee["incidents"]) == pytest.approx(annee["perte_annuelle"])
    assert all(i["piliers_atteints"] == [i["amorce"]] for i in annee["incidents"])


def test_etats_identiques_partagent_exactement_les_tirages():
    etat = Etat("identique", 2, .25, .4, .3)
    gauche, droite = simuler(500, 41, (etat, etat))
    assert gauche == droite


@pytest.mark.parametrize("etat", [Etat("sans perte", 3, 0, .5, .4), Etat("sans incident", 0, 1, .5, .4)])
def test_absence_de_perte_ou_incident(etat):
    resultat = simuler(100, 11, (etat,))[0]
    assert resultat["moyenne_theorique"] == 0
    assert all(x == 0 for x in resultat["mesures"].values())


def test_choc_commun_certain_et_amorce_conservee():
    ensembles = probabilites_ensembles(Etat("choc certain", 2, .1, .4, 1))
    assert ensembles[3, 31] == 1  # Les cinq piliers, masque binaire 11111.
    for j in range(5):
        assert ensembles[j, [m for m in range(32) if not (m >> j & 1)]].sum() == 0
    np.testing.assert_allclose(ensembles.sum(axis=1), 1)


def test_es_integree_avec_masse_au_quantile():
    # Les 25 % pires observations contiennent 20 et la moitié du 10, soit 50/3.
    resultat = mesures(np.array([0., 0., 0., 0., 10., 20.]), alpha=.75)
    assert resultat["var"] == 10
    assert resultat["es"] == pytest.approx(50 / 3)
