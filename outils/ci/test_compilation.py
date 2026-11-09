"""La version Institut de travail doit être contrôlée avant toute copie de dépôt."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

CHEMIN = Path(__file__).resolve().parents[2] / "memoire" / "refaire_tout.py"
SPEC = importlib.util.spec_from_file_location("chaine_memoire", CHEMIN)
chaine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chaine)


class CompilationTest(unittest.TestCase):
    def test_version_courante_presente_avec_limite_et_copie_distincte(self):
        versions = {base: (copie, limite) for base, copie, limite in chaine.VERSIONS}
        copie, limite = versions["main_institut_150"]
        self.assertEqual(limite, 150)
        self.assertNotEqual(copie, versions["main_institut"][0])

    def test_pdf_institut_au_dela_de_150_pages_bloque_le_controle(self):
        class Page:
            rotation = 0

            def get_text(self):
                return ""

            def get_images(self):
                return []

        class Document(list):
            @property
            def page_count(self):
                return len(self)

            def close(self):
                pass

        for pages, attendu in ((150, True), (151, False), (158, False)):
            with self.subTest(pages=pages), \
                    patch.dict("sys.modules", {"fitz": SimpleNamespace(
                        open=lambda _: Document([Page()] * pages))}), \
                    patch.object(chaine.os.path, "isfile", return_value=True), \
                    patch.object(chaine, "defauts", []), \
                    patch.object(chaine, "VERSIONS", [
                        ("main_institut_150", "copie.pdf", 150)]):
                self.assertEqual(chaine.etape_pdf(), attendu)

    def test_selection_institut150_ne_selectionne_pas_la_version_figee(self):
        with patch("sys.argv", ["refaire_tout.py", "--version", "institut150"]), \
                patch.object(chaine, "VERSIONS", list(chaine.VERSIONS)), \
                patch.object(chaine, "prerequis", return_value=(False, None)):
            self.assertEqual(chaine.main(), 1)
            self.assertEqual([v[0] for v in chaine.VERSIONS], ["main_institut_150"])

    def test_echec_pdf_empeche_toute_copie_de_depot(self):
        with patch("sys.argv", ["refaire_tout.py", "--sans-harnais"]), \
                patch.object(chaine, "defauts", []), \
                patch.object(chaine, "prerequis", return_value=(True, "tectonic")), \
                patch.object(chaine, "etape_compilation", return_value=True), \
                patch.object(chaine, "etape_pdf", return_value=False), \
                patch.object(chaine, "institut_fige", return_value=True), \
                patch.object(chaine, "etape_depot") as depot:
            self.assertEqual(chaine.main(), 1)
            depot.assert_not_called()
