"""Régression : une sensibilité antérieure ne doit pas être écrasée par la dernière."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "benchmark131", Path(__file__).resolve().parents[1] / "1_fondations/131_benchmark_covariables_depassement.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_signal_premiere_variante_conserve_quand_derniere_negative():
    def resultat(m2, m3):
        return {"M2": {"contre_M1": {"p5": m2}}, "M3": {"contre_M1": {"p5": m3}}}

    signaux = benchmark.signaux_covariables({
        "actifs_declares": resultat(-0.01, 0.02),
        "sans_actifs": resultat(0.01, -0.01),
        "actifs_ex_ante": resultat(-0.02, -0.03),
    })
    assert signaux["actifs_declares"]["M3"]
    assert signaux["sans_actifs"]["M2"]
    assert not any(signaux["actifs_ex_ante"].values())
    assert any(signal for modele in signaux.values() for signal in modele.values())


def test_borne_nulle_ne_prouve_pas_un_gain_positif():
    assert benchmark.signaux_covariables({"v": {
        "M2": {"contre_M1": {"p5": 0.0}}, "M3": {"contre_M1": {"p5": -0.1}},
    }}) == {"v": {"M2": False, "M3": False}}
