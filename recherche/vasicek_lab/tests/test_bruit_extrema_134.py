"""Le diagnostic doit re-sélectionner les matrices extrémales à chaque graine."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "6_contagion/134_bruit_extrema_institut.py"
spec = importlib.util.spec_from_file_location("bruit_extrema_134", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_reselection_des_extrema_et_dispersion_de_largeur():
    # Les extrema changent de colonne ; conserver les premières matrices donnerait
    # des largeurs de 3 et -3 et masquerait le résultat du balayage exhaustif.
    out = mod.resumer([[1., 2., 4.], [6., 5., 3.]])
    assert out["indices_minimum"] == [0, 2]
    assert out["indices_maximum"] == [2, 0]
    assert out["largeur"] == [3., 3.]
    assert out["ecarts_types"][2] == 0.
    assert np.isclose(out["ecarts_types"][0], np.sqrt(2.))


@pytest.mark.parametrize("values", [[[1., 2.]], [[1., 2.], [3., float('nan')]]])
def test_refuse_une_seule_grille_ou_un_resultat_invalide(values):
    with pytest.raises(ValueError):
        mod.resumer(values)
