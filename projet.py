#!/usr/bin/env python3
"""Point d'entree des commandes du depot, sous Windows, Linux et macOS."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

RACINE = Path(__file__).resolve().parent
MANUSCRITS = Path("memoire")


def executer(script: Path, *arguments: str) -> int:
    """Conserver l'interpreteur choisi, les chemins stables et le code de sortie."""
    return subprocess.run(
        [sys.executable, str(RACINE / script), *arguments],
        cwd=RACINE,
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        check=False,
    ).returncode


def etat() -> int:
    resultat = subprocess.run(
        ["git", "status", "--short", "--branch"], cwd=RACINE,
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if resultat.returncode:
        print(resultat.stderr, file=sys.stderr, end="")
        return resultat.returncode
    print(resultat.stdout, end="")
    try:
        import fitz
    except ImportError:
        fitz = None
    for nom in ("main_ensae", "main_institut", "main_institut_150"):
        source = RACINE / MANUSCRITS / (nom + ".tex")
        pdf = source.with_suffix(".pdf")
        info = "PDF present" if pdf.is_file() else "PDF absent"
        if fitz is not None and pdf.is_file():
            try:
                with fitz.open(pdf) as document:
                    info = f"{document.page_count} pages"
            except (OSError, RuntimeError) as erreur:
                print(f"{nom} : PDF illisible : {erreur}", file=sys.stderr)
                return 1
        print(f"{nom} : source {'presente' if source.is_file() else 'absente'}, {info}")
    if fitz is None:
        print("Pagination facultative : installer PyMuPDF pour l'afficher.")
    return 0


def main() -> int:
    commandes = {
        "etat": "afficher la branche, les changements locaux et les manuscrits",
        "structure": "controler la navigation et les fichiers suivis (lecture seule)",
        "verifier": "controler la structure puis le harnais strict (lecture seule)",
        "compiler": "compiler les deux versions, controler et recopier les PDF, sans harnais",
        "depot": "passer les controles stricts puis compiler et recopier les PDF",
        "audit": "auditer les correspondances du harnais avec les sorties enregistrees",
    }
    analyseur = argparse.ArgumentParser(description=__doc__)
    sous_commandes = analyseur.add_subparsers(dest="commande", required=True)
    for nom, description in commandes.items():
        sous_commandes.add_parser(nom, help=description, description=description)
    commande = analyseur.parse_args().commande
    if commande == "etat":
        return etat()
    if commande in {"structure", "verifier", "depot"}:
        resultat = executer(Path("outils/ci/verifier_structure.py"))
        if resultat or commande == "structure":
            return resultat
    if commande in {"verifier", "depot"}:
        resultat = executer(Path("outils/ci/harnais_strict.py"))
        if resultat or commande == "verifier":
            return resultat
    if commande in {"compiler", "depot"}:
        return executer(MANUSCRITS / "refaire_tout.py", "--sans-harnais")
    return executer(MANUSCRITS / "audit_sources.py", "resultats/sorties_verif")


if __name__ == "__main__":
    raise SystemExit(main())
