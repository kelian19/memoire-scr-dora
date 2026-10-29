#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
construire.py : compile les schemas LaTeX du support de soutenance et les rend en PNG.

Usage, depuis la racine du depot :
    .venv\\Scripts\\python.exe exploratory\\slides\\figures_soutenance\\construire.py [nom ...]

Sans argument, construit tous les .tex du dossier sauf le preambule. Chaque schema
<nom>.tex donne <nom>.png a 300 points par pouce, fond blanc, a cote de sa source.
La compilation se fait dans un dossier temporaire : aucun fichier intermediaire ne
reste dans le depot.

POURQUOI UN PNG ET PAS UN PDF DANS LE SUPPORT. python-pptx n'insere que des images
matricielles ; 300 points par pouce a la taille finale suffisent a une projection.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

ICI = os.path.dirname(os.path.abspath(__file__))
TECTONIC = os.environ.get("TECTONIC") or shutil.which("tectonic") or "tectonic"
DPI = 300


def construire(nom):
    source = os.path.join(ICI, nom + ".tex")
    if not os.path.exists(source):
        raise SystemExit(f"schema absent : {source}")
    tmp = tempfile.mkdtemp(prefix="schema_")
    try:
        shutil.copy(source, tmp)
        shutil.copy(os.path.join(ICI, "preambule_schemas.tex"), tmp)
        r = subprocess.run([TECTONIC, "-X", "compile", nom + ".tex"], cwd=tmp,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        journal = (r.stdout or "") + (r.stderr or "")
        erreurs = [l for l in journal.splitlines() if l.startswith("error")]
        debords = [l for l in journal.splitlines() if "Overfull" in l]
        if r.returncode != 0 or erreurs:
            print("\n".join(erreurs) or journal[-3000:])
            raise SystemExit(f"{nom} : compilation en echec")
        import pymupdf
        doc = pymupdf.open(os.path.join(tmp, nom + ".pdf"))
        page = doc[0]
        page.get_pixmap(dpi=DPI, alpha=False).save(os.path.join(ICI, nom + ".png"))
        l_in, h_in = page.rect.width / 72, page.rect.height / 72
        print(f"{nom}.png  {l_in:.2f} x {h_in:.2f} pouces"
              + (f"  ({len(debords)} debordement(s) de boite)" if debords else ""))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    noms = sys.argv[1:] or sorted(os.path.splitext(os.path.basename(p))[0]
                                  for p in glob.glob(os.path.join(ICI, "*.tex"))
                                  if not p.endswith("preambule_schemas.tex"))
    for n in noms:
        construire(n)
