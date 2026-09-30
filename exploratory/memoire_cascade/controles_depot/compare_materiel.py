# -*- coding: utf-8 -*-
"""Le materiel complementaire reconstruit differe-t-il de la version publiee sur Zenodo ?

Usage, depuis exploratory/memoire_cascade/, APRES reconstruction :
    ..\\..\\.venv\\Scripts\\python.exe materiel_complementaire\\build_matcomp.py
    <tectonic> -X compile _tmp_materiel.tex --keep-intermediates
    ..\\..\\.venv\\Scripts\\python.exe materiel_complementaire\\build_matcomp.py --extract
    ..\\..\\.venv\\Scripts\\python.exe controles_depot\\compare_materiel.py

La reference est la version COMMITTEE du PDF (git show HEAD:...). Depuis la publication de la v1.2
le 29 septembre 2026 (records/23040521), elle est IDENTIQUE a la v1.2 publiee, md5
9daff1103a1f5c103569aac7f031e4b0, verifie sur l'API Zenodo. La v1.1 (md5 932463dd...) portait
encore la section du dispositif de verification, retiree depuis. Comparaison page par page, texte
integral puis titres courants retires. Zero page differente : aucune nouvelle version Zenodo n'est
due, et il faut restaurer le fichier committe par `git checkout` pour garder l'identite avec Zenodo.
Sinon : le signaler a Kelian, une version 1.3 est due (procedure dans ZENODO.md).
"""
import difflib
import hashlib
import os
import re
import subprocess
import sys
import unicodedata

import fitz

MEMOIRE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMIN = os.path.join(MEMOIRE, "materiel_complementaire", "materiel_complementaire.pdf")
RACINE = os.path.dirname(os.path.dirname(MEMOIRE))
REF = "HEAD:exploratory/memoire_cascade/materiel_complementaire/materiel_complementaire.pdf"
TITRE_COURANT = re.compile(r"[A-ZÉÈÀÊÂÎÔÛÇ][A-ZÉÈÀÊÂÎÔÛÇ ’'\-.]{4,}\s+\d{3}")

publie = subprocess.run(["git", "show", REF], cwd=RACINE, capture_output=True).stdout
if not publie:
    raise SystemExit("version committee introuvable")
with open(CHEMIN, "rb") as f:
    reconstruit = f.read()
print(f"md5 publie      : {hashlib.md5(publie).hexdigest()}")
meme = hashlib.md5(reconstruit).digest() == hashlib.md5(publie).digest()
print(f"md5 reconstruit : {hashlib.md5(reconstruit).hexdigest()}  "
      f"({'identique : fichier non reconstruit ou deja restaure' if meme else 'differe, normal apres reconstruction : dates du PDF'})")


def pages(brut, sans_titres):
    out = []
    for p in fitz.open(stream=brut, filetype="pdf"):
        t = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", p.get_text())).strip()
        out.append(TITRE_COURANT.sub(" ", t).strip() if sans_titres else t)
    return out


total = 0
for sans in (False, True):
    a, b = pages(publie, sans), pages(reconstruit, sans)
    diff = [i for i in range(min(len(a), len(b))) if a[i] != b[i]]
    total += len(diff) + abs(len(a) - len(b))
    print(f"{'titres retires' if sans else 'texte integral'} : publie {len(a)} p., "
          f"reconstruit {len(b)} p., {len(diff)} page(s) differente(s)")
    for i in diff[:4]:
        wa, wb = a[i].split(" "), b[i].split(" ")
        sm = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "equal":
                print(f"   p.{i + 1} [{op}] publie : ...{' '.join(wa[max(0, i1 - 6):i2 + 6])[:120]}...")
                print(f"          reconstruit : ...{' '.join(wb[max(0, j1 - 6):j2 + 6])[:120]}...")
if total:
    print("\nLE MATERIEL A CHANGE : une version Zenodo 1.3 est due, a signaler a Kelian.")
    sys.exit(1)
print("\nIdentique a la version publiee : restaurer le fichier committe (git checkout).")
