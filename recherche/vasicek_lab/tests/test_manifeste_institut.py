"""Une nouvelle exécution conserve les réserves et les attestations antérieures."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "lot_institut", Path(__file__).resolve().parents[3] / "outils/ci/lancer_lot_institut.py")
lot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lot)


def test_mise_a_jour_conserve_notes_et_lignes_non_rejouees(tmp_path):
    mf = tmp_path / "MANIFESTE.md"
    mf.write_text(
        "# Manifeste\n\nDernier lancement le hier, code du commit `ancien` plus les\n"
        "modifications non committees du depot a cet instant.\n\n"
        "Les expériences 119 et 120 restent conditionnelles.\n\n"
        "| script | fichier | code | Traceback | duree | octets | md5 |\n", encoding="utf8")
    lignes = {"119": "| 119 | ancien.py | 0 | non | 3 s | 12 | abcdef12 |",
              "131": "| 131 | nouveau.py | 0 | non | 2 s | 24 | abcdef34 |"}
    lot.enregistrer_manifeste(mf, lignes, "nouveau", ["131"], "journaux/essai")
    texte = mf.read_text(encoding="utf8")
    assert "Les expériences 119 et 120 restent conditionnelles." in texte
    assert lignes["119"] in texte and lignes["131"] in texte
    assert "`nouveau`" in texte and "`ancien`" not in texte
    assert texte.count("| script |") == 1
