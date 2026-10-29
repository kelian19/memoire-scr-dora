#!/usr/bin/env python3
"""Controle sans dependance des chemins, guides maintenus et fichiers suivis."""

import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

RACINE = (Path(__file__).resolve().parent / '../..').resolve()
GUIDES = ('README.md', 'CONTRIBUTING.md', 'docs/DEMARRAGE.md', 'docs/VERSIONS.md', 'docs/ARCHITECTURE.md', 'memoire/README.md', 'memoire/overleaf/README.md', '.github/pull_request_template.md')
INDISPENSABLES = (
    "projet.py", "requirements.txt", 'outils/ci/harnais_strict.py',
    'memoire/main_ensae.tex',
    'memoire/main_institut.tex',
    'memoire/chapitres_sfe',
    'memoire/chapitres',
    'memoire/preambule_v2.tex',
    'memoire/refaire_tout.py',
    'memoire/verif_chiffres.py',
    'recherche/vasicek_lab/scr_engine.py',
    'recherche/vasicek_lab/partial_id.py', "resultats/sorties_verif",
    "docs/ARCHITECTURE.md", "docs/VERSIONS.md",
)
LIEN = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
TAILLE_MAX = 20 * 1024 * 1024
DOSSIERS = ("data", "docs", "memoire", "outils", "recherche", "resultats", "src")
ANCIENS_DOSSIERS = ("audit", "ci", "consignes", "exploratory", "notebooks", "outputs",
                   "reduction", "sorties_verif", "style", "00_ACCES_RAPIDE")


def verifier_arborescence(racine: Path) -> list[str]:
    erreurs = [f"dossier de travail absent : {nom}" for nom in DOSSIERS
               if not (racine / nom).is_dir()]
    erreurs.extend(f"ancien dossier recree a la racine : {nom}" for nom in ANCIENS_DOSSIERS
                   if (racine / nom).exists())
    return erreurs


def chemin_exact(racine: Path, cible: Path) -> bool:
    """Controler aussi la casse sur les systemes de fichiers permissifs."""
    courant = racine
    for morceau in cible.relative_to(racine).parts:
        if not courant.is_dir() or morceau not in {p.name for p in courant.iterdir()}:
            return False
        courant = courant / morceau
    return courant.exists()


def verifier_lien(racine: Path, document: Path, lien: str) -> str | None:
    adresse = urlsplit(lien.strip().strip("<>"))
    if adresse.scheme or adresse.netloc or not adresse.path:
        return None
    chemin = unquote(adresse.path)
    cible = (racine / chemin.lstrip("/") if chemin.startswith("/")
             else document.parent / chemin)
    # abspath normalise les '..' sans effacer la casse des composants sous Windows.
    cible = Path(os.path.abspath(cible))
    if not cible.is_relative_to(racine) or not cible.resolve().is_relative_to(racine.resolve()):
        return f"lien hors du depot : {lien}"
    if not chemin_exact(racine, cible):
        return f"cible absente ou casse incorrecte : {lien}"
    return None


def verifier_fichiers_suivis(racine: Path, fichiers: list[str]) -> list[str]:
    erreurs = []
    for nom in fichiers:
        normalise = nom.lower()
        if normalise.startswith("data/raw/") or "logos/signature_" in normalise:
            erreurs.append(f"fichier interdit dans Git : {nom}")
        chemin = racine / nom
        if not chemin.exists():
            erreurs.append(f"fichier suivi absent : {nom}")
        elif chemin.is_file() and chemin.stat().st_size > TAILLE_MAX:
            erreurs.append(f"fichier suivi de plus de 20 Mio : {nom}")
    return erreurs


def main() -> int:
    erreurs = verifier_arborescence(RACINE)
    for nom in INDISPENSABLES:
        if not chemin_exact(RACINE, RACINE / nom):
            erreurs.append(f"chemin indispensable absent ou casse incorrecte : {nom}")
    guides = sorted(set(GUIDES) | {
        chemin.relative_to(RACINE).as_posix() for chemin in (RACINE / "docs").rglob("*.md")
        if "audits" not in chemin.relative_to(RACINE / "docs").parts
        and not chemin.is_relative_to(RACINE / "docs/collaboration/local")
    })
    nombre_liens = 0
    for nom in guides:
        document = RACINE / nom
        if not chemin_exact(RACINE, document):
            erreurs.append(f"guide absent ou casse incorrecte : {nom}")
            continue
        for numero, ligne in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
            for lien in LIEN.findall(ligne):
                nombre_liens += 1
                erreur = verifier_lien(RACINE, document, lien)
                if erreur:
                    erreurs.append(f"{nom}:{numero} : {erreur}")
    try:
        resultat = subprocess.run(
            ["git", "ls-files", "-z"], cwd=RACINE, capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as erreur:
        print(f"Impossible de lire l'index Git : {erreur}", file=sys.stderr)
        return 1
    fichiers = [nom for nom in resultat.stdout.decode("utf-8").split("\0") if nom]
    erreurs.extend(verifier_fichiers_suivis(RACINE, fichiers))
    if erreurs:
        for erreur in erreurs:
            print(f"ERREUR : {erreur}", file=sys.stderr)
        return 1
    print(f"Structure OK : {len(guides)} guides, {nombre_liens} liens examines, "
          f"{len(fichiers)} fichiers suivis controles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
