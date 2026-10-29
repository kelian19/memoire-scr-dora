"""Contrats des chemins apres le deplacement physique des dossiers."""

from pathlib import Path
import re
import tempfile
import unittest

import harnais_strict
from verifier_structure import DOSSIERS, verifier_arborescence

RACINE = Path(__file__).resolve().parents[2]


class CheminsTest(unittest.TestCase):
    def test_harnais_lit_les_chapitres_et_sorties_deplaces(self):
        self.assertEqual(harnais_strict.MEMOIRE, RACINE / "memoire")
        self.assertEqual(harnais_strict.SORTIES, RACINE / "resultats" / "sorties_verif")
        self.assertTrue(list((harnais_strict.MEMOIRE / "chapitres_sfe").glob("*.tex")))
        self.assertTrue((harnais_strict.SORTIES / "68.txt").is_file())

    def test_repertoires_des_figures_latex_existent(self):
        texte = (RACINE / "memoire" / "preambule_v2.tex").read_text(encoding="utf-8")
        bloc = re.search(r"\\graphicspath\{((?:\{[^{}]+\})+)\}", texte)
        self.assertIsNotNone(bloc)
        for relatif in re.findall(r"\{([^{}]+)\}", bloc.group(1)):
            with self.subTest(chemin=relatif):
                self.assertTrue((RACINE / "memoire" / relatif).is_dir())

    def test_ancien_dossier_de_sorties_refuse(self):
        with tempfile.TemporaryDirectory() as temporaire:
            racine = Path(temporaire)
            for nom in DOSSIERS:
                (racine / nom).mkdir()
            self.assertEqual(verifier_arborescence(racine), [])
            (racine / "sorties_verif").mkdir()
            self.assertTrue(any("sorties_verif" in e for e in verifier_arborescence(racine)))


if __name__ == "__main__":
    unittest.main()
