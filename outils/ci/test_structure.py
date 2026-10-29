"""Les garde-fous doivent refuser les erreurs, y compris sous Windows."""

from pathlib import Path
import tempfile
import unittest

from verifier_structure import verifier_fichiers_suivis, verifier_lien


class StructureTest(unittest.TestCase):
    def setUp(self):
        self.temporaire = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporaire.cleanup)
        self.racine = Path(self.temporaire.name).resolve()
        (self.racine / "docs").mkdir()
        self.document = self.racine / "docs" / "guide.md"
        (self.racine / "README.md").write_text("Accueil", encoding="utf-8")

    def test_lien_relatif_et_ancre(self):
        self.assertIsNone(verifier_lien(self.racine, self.document, "../README.md#titre"))

    def test_lien_casse(self):
        self.assertIsNotNone(verifier_lien(self.racine, self.document, "absent.md"))

    def test_casse_incorrecte_meme_sous_windows(self):
        self.assertIsNotNone(verifier_lien(self.racine, self.document, "../readme.md"))

    def test_lien_exterieur_au_depot(self):
        self.assertIn("hors du depot", verifier_lien(self.racine, self.document, "../../ailleurs"))

    def test_espaces_et_accents_encodes(self):
        (self.racine / "docs" / "données locales.md").write_text("", encoding="utf-8")
        self.assertIsNone(verifier_lien(self.racine, self.document, "donn%C3%A9es%20locales.md"))

    def test_liens_internet_et_ancres_ignores(self):
        for lien in ("https://example.org/absent", "mailto:a@example.org", "#titre"):
            with self.subTest(lien=lien):
                self.assertIsNone(verifier_lien(self.racine, self.document, lien))

    def test_donnee_brute_signature_et_gros_fichier_refuses(self):
        fichiers = ["data/raw/source.xlsx", "logos/signature_a.png", "gros.pdf"]
        for nom in fichiers:
            chemin = self.racine / nom
            chemin.parent.mkdir(parents=True, exist_ok=True)
            chemin.touch()
        with (self.racine / "gros.pdf").open("wb") as fichier:
            fichier.truncate(20 * 1024 * 1024 + 1)
        erreurs = verifier_fichiers_suivis(self.racine, fichiers)
        self.assertEqual(len(erreurs), 3)
        self.assertTrue(all(any(nom in erreur for erreur in erreurs) for nom in fichiers))

    def test_fichier_suivi_supprime_refuse(self):
        self.assertTrue(verifier_fichiers_suivis(self.racine, ["absent.py"]))


if __name__ == "__main__":
    unittest.main()
