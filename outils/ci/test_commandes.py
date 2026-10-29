"""Une verification en echec doit empecher la preparation des PDF de depot."""

import unittest
from unittest.mock import patch

import projet


class DepotTest(unittest.TestCase):
    def test_structure_en_echec_arrete_avant_harnais_et_compilation(self):
        with patch("sys.argv", ["projet.py", "depot"]), \
                patch.object(projet, "executer", return_value=1) as executer:
            self.assertEqual(projet.main(), 1)
        self.assertEqual(executer.call_count, 1)

    def test_harnais_en_echec_empeche_la_compilation(self):
        with patch("sys.argv", ["projet.py", "depot"]), \
                patch.object(projet, "executer", side_effect=[0, 2]) as executer:
            self.assertEqual(projet.main(), 2)
        self.assertEqual(executer.call_count, 2)


if __name__ == "__main__":
    unittest.main()
