"""Le calcul d'accord ne doit pas transformer une réponse manquante en absence de pilier."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "double_codage", Path(__file__).resolve().parents[3] / "outils/validation/double_codage_sas.py")
codage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codage)


def test_accord_avec_marges_differentes_et_degenerescence():
    mesure = codage.accord_binaire([True, True, False, False], [True, False, False, False])
    assert mesure["accord"] == 0.75 and mesure["kappa"] == 0.5
    assert codage.accord_binaire([False] * 4, [False] * 4)["kappa"] is None


def test_reponse_inachevee_refusee_meme_avec_codes_vides():
    cle = {"cas": [{"cas": "V001", "strate": "E", "strict": [], "large": []}]}
    reponses = {"codeur": "test", "aveugle_declare": True, "cas": [
        {"cas": "V001", "termine": False, "strict": [], "large": [], "justification": "sans information"}]}
    with pytest.raises(ValueError, match="non terminé"):
        codage.evaluer(reponses, cle)


def test_inclusion_strict_large_controlee():
    cle = {"cas": [{"cas": "V001", "strate": "E", "strict": [], "large": []}]}
    reponses = {"codeur": "test", "aveugle_declare": True, "cas": [
        {"cas": "V001", "termine": True, "strict": [1], "large": [], "justification": "exemple"}]}
    with pytest.raises(ValueError, match="inclus"):
        codage.evaluer(reponses, cle)
