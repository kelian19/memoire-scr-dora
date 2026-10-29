"""Repérage des nombres du Beamer dans le rapport ou le complément cité.

Usage, depuis la racine du depot :
    .venv\\Scripts\\python.exe recherche\\slides\\beamer\\controle_nombres.py

Lit le texte du PDF compile (soutenance_ensae.pdf), en extrait les nombres ecrits a la francaise
et les cherche dans le rapport par chercher_rapport.present (deux lectures du PDF, milliers fusionnes
ou non). Imprime ceux qu'il ne trouve pas, avec leur diapositive : ils se relisent un par un
(numeros de page, exemples illustratifs, annees, numeros de chapitre ne sont pas des resultats).
Sur les pages explicitement sourcées 129, accepte aussi les valeurs du calcul JSON,
à leur précision d'affichage. Une concordance numérique ne valide pas son sens.
"""
import json
import os
import re
import sys
from pathlib import Path

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, ".."))
from chercher_rapport import normaliser, present  # noqa: E402

import pymupdf  # noqa: E402

NOMBRE = re.compile(r"(?<![\w,.])\d{1,3}(?:[   ]\d{3})+(?:,\d+)?|(?<![\w,.])\d+(?:,\d+)?")


def nombres_complement():
    chemin = Path(ICI).parents[2]/'resultats/institut_approfondissements/129.json'
    resultat = json.loads(chemin.read_text(encoding='utf-8'))
    if resultat['statut'] != 'termine' or resultat['echecs']:
        raise ValueError('Complément 129 incomplet')
    valeurs = []

    def parcourir(x):
        if isinstance(x, dict):
            for v in x.values():
                parcourir(v)
        elif isinstance(x, list):
            for v in x:
                parcourir(v)
        elif isinstance(x, (int, float)):
            valeurs.append(x)
    parcourir(resultat)
    return valeurs


def present_complement(nombre, valeurs):
    n = normaliser(nombre).replace(' ', '').replace(',', '.')
    decimales = len(n.split('.')[1]) if '.' in n else 0
    return any(f'{v:.{decimales}f}' == f'{float(n):.{decimales}f}' for v in valeurs)


def main():
    """Sans argument : soutenance_ensae.pdf contre main_ensae.pdf.
    Avec --deck D.pdf et --rapport R.pdf : un autre support contre un autre rapport
    (soutenance Institut contre main_institut_150.pdf compilé)."""
    args = sys.argv[1:]
    deck = os.path.join(ICI, "soutenance_ensae.pdf")
    if "--rapport" in args:
        import chercher_rapport
        chercher_rapport.RAPPORT = os.path.abspath(args[args.index("--rapport") + 1])
        chercher_rapport._cache.clear()
    if "--deck" in args:
        deck = os.path.abspath(args[args.index("--deck") + 1])
    doc = pymupdf.open(deck)
    complement = nombres_complement()
    vus, absents = set(), []
    for i, page in enumerate(doc, 1):
        texte = normaliser(page.get_text(), fusion=False)
        for m in NOMBRE.finditer(texte):
            n = m.group(0)
            if (i, n) in vus:
                continue
            vus.add((i, n))
            cite_129 = bool(re.search(r'script\s+129', texte, flags=re.IGNORECASE))
            if not present(n) and not (cite_129 and present_complement(n, complement)):
                absents.append((i, n, texte[max(0, m.start() - 40):m.end() + 25]))
    print(f"{len(vus)} nombres lus, {len(absents)} absents du rapport ou du complément 129 cité :")
    for i, n, ctx in absents:
        print(f"  p.{i:>2}  {n:>10}   … {ctx} …")


if __name__ == "__main__":
    main()
