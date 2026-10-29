"""Controles mathematiques : credibilite et non-identification par amincissement."""
import importlib.util
from pathlib import Path

import numpy as np
import unittest
from scipy.integrate import quad
from scipy.stats import binom, gamma, nbinom, poisson

spec = importlib.util.spec_from_file_location("observation138",
    Path(__file__).resolve().parents[1] / "2_donnees/138_descente_observation.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def test_posterior_against_numerical_bayes_integration():
    a, b, count, exposure, q = 2.0, 3.0, 4, 12.0, 0.4
    ap, bp = model.gamma_poisson_posterior(a, b, count, exposure, q)
    density = lambda eta: gamma.pdf(eta, a, scale=1/b) * poisson.pmf(count, q*exposure*eta)
    normalizer = quad(density, 0, np.inf)[0]
    mean = quad(lambda eta: eta*density(eta), 0, np.inf)[0] / normalizer
    pred_zero = quad(lambda eta: np.exp(-eta)*density(eta), 0, np.inf)[0] / normalizer
    assert np.isclose(ap/bp, mean, rtol=1e-8)
    assert np.isclose((bp/(bp+1))**ap, pred_zero, rtol=1e-8)


def test_thinning_preserves_entire_observed_count_distribution():
    r, recorded_mean = 0.594, 0.021
    for q in (1.0, 0.5, 0.25, 0.1):
        latent_mean = recorded_mean/q
        n = np.arange(1000)
        latent = nbinom.pmf(n, r, r/(r+latent_mean))
        for observed in range(8):
            actual = np.sum(latent * binom.pmf(observed, n, q))
            expected = nbinom.pmf(observed, r, r/(r+recorded_mean))
            assert np.isclose(actual, expected, rtol=1e-9, atol=1e-13)


def test_exposure_unit_change_does_not_change_predictive_mean():
    a, b = model.gamma_poisson_posterior(2, 3, 4, 12, 0.4)
    # New exposure unit = one hundred old units: rate and future exposure scale together.
    a2, b2 = model.gamma_poisson_posterior(2, 300, 4, 1200, 0.4)
    assert np.isclose(10*a/b, 1000*a2/b2)


def test_invalid_information_is_rejected():
    for args in ((0, 1, 0, 0, 1), (1, 1, 1, 0, 1), (1, 1, 0.5, 2, 1), (1, 1, 0, 2, 0)):
        with unittest.TestCase().assertRaises(ValueError):
            model.gamma_poisson_posterior(*args)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(test) for name, test in globals().items()
                              if name.startswith("test_") and callable(test))


if __name__ == "__main__":
    unittest.main()
