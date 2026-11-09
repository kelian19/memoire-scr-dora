"""Régressions de l'ancien moteur : horizon annuel, Poisson et sens de dépendance."""
from unittest.mock import patch

import numpy as np
import pytest

from src.aggregation import lda
from src.frequency.negbin import compute_lambda_scenario, simulate_frequency


@pytest.mark.parametrize("scenario", ["S0_conforme", "S1_partiel", "S2_non_conforme"])
def test_frequence_utilise_la_moyenne_annuelle_du_scenario(scenario):
    # La calibration est mensuelle ; le scénario demande une fréquence annuelle.
    params = {"mu": 10.0, "r": 20.0}
    for annuelle in (120.0, 1440.0):
        cible = compute_lambda_scenario(annuelle, scenario)["lambda_global"]
        counts = simulate_frequency(params, scenario, annuelle, n_sim=100_000)
        erreur_type = np.sqrt(cible * (1 + cible / params["r"]) / len(counts))
        assert abs(counts.mean() - cible) < 5 * erreur_type


PARAMS = {"xi": .2, "sigma": 1., "u": 1., "p_u": 1., "dispersion_factor": 1.}


def test_limite_poisson_a_une_variance_egale_a_la_moyenne():
    # Une perte unitaire par incident permet d'observer les comptages annuels.
    with patch.object(lda, "simulate_remediation_severity", side_effect=lambda n, *a: np.ones(n)), \
            patch.dict(lda.BRIQUE_PARAMS["stress_systemique"], {"max_loading": 0}):
        counts = lda.simulate_year_3_briques(10., PARAMS, 0., 50_000)["remediation"]
    assert abs(counts.mean() - 10) < .1
    assert abs(counts.var() - 10) < .3


def test_frequence_nulle_ne_produit_aucune_remediation_ni_perte_tiers():
    res = lda.simulate_year_3_briques(0., PARAMS, 0., 10)
    np.testing.assert_array_equal(res["total"], np.zeros(10))


@pytest.mark.parametrize("dispersion", [.9, float("nan"), float("inf")])
def test_dispersion_inadmissible_refusee(dispersion):
    with pytest.raises(ValueError, match="dispersion_factor"):
        lda.simulate_year_3_briques(10., dict(PARAMS, dispersion_factor=dispersion), 0., 10)


def test_sanction_accompagne_le_stress_eleve_et_conserve_sa_probabilite():
    # Copule comonotone : la moitié haute doit subir les sanctions.
    u = np.tile(np.array([.1, .3, .7, .9])[:, None], (1, 3))
    with patch.object(lda, "gumbel_copula_uniforms", return_value=u):
        res = lda.simulate_year_3_briques(0., PARAMS, .5, 4)
    np.testing.assert_array_equal(res["sanction"] > 0, [False, False, True, True])
