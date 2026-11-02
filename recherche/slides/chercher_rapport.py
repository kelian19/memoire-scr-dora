#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chercher_rapport.py : cherche un texte ou un nombre dans le rapport ENSAE compile.

Usage, depuis la racine du depot :
    .venv\\Scripts\\python.exe exploratory\\slides\\chercher_rapport.py "6 049" "facteur 3,34"

Pour chaque requete, imprime le folio (numero imprime en pied de page) et le contexte
de chaque occurrence dans exploratory/memoire_cascade/main_ensae.pdf.

POURQUOI CET OUTIL. Le jury « essaie de retrouver dans le texte du memoire ce que
l'etudiant presente a l'oral » : un nombre du support doit donc se lire dans le
rapport, au meme sens. La recherche se fait sur le texte NORMALISE du PDF, parce que
ce texte porte des ligatures (« fi » sort en un seul glyphe) et des espaces fines
entre les milliers : une recherche brute y echoue en silence.

Le module expose aussi texte_rapport() et present(), que build_soutenance_ppt.py
emploie pour verifier chaque nombre du support.
"""
import os
import re
import sys
import unicodedata

ICI = os.path.dirname(os.path.abspath(__file__))
RAPPORT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../memoire/main_ensae.pdf'))
_cache = {}


def normaliser(t, fusion=True):
    """Ligatures, espaces insecables et fines ramenees a une espace simple, et, avec
    fusion=True, separateurs de milliers retires entre chiffres (6 049 -> 6049)."""
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[    ]", " ", t)
    t = t.replace("−", "-")                       # signe moins typographique
    t = re.sub(r"-\n", "", t)                          # cesure en fin de ligne
    t = re.sub(r"\s+", " ", t)
    if fusion:
        t = re.sub(r"(?<=\d) (?=\d{3}(?!\d))", "", t)
    return t


def texte_rapport(fusion=True):
    """Liste de (folio, texte normalise) pour chaque page du rapport."""
    cle = "pages" if fusion else "pages_brutes"
    if cle not in _cache:
        import pymupdf
        if not os.path.exists(RAPPORT):
            raise SystemExit(f"rapport absent : {RAPPORT}")
        doc = pymupdf.open(RAPPORT)
        _cache[cle] = [(i, normaliser(p.get_text(), fusion)) for i, p in enumerate(doc)]
    return _cache[cle]


def present(nombre):
    """Vrai si le nombre (ecrit a la francaise) figure tel quel dans le rapport, borne
    a gauche et a droite par autre chose qu'un chiffre ou une virgule decimale.

    DEUX LECTURES DU PDF. Avec les milliers fusionnes, « 6 049 » se retrouve meme
    coupe en fin de ligne. Mais dans un tableau, le texte extrait colle les cellules
    voisines (« 2 319 425 112 » pour 2 319, 425 et 112), que la fusion soude en un seul
    nombre : la seconde lecture, sans fusion, retrouve ces cellules."""
    n = normaliser(nombre).strip()
    motif = re.compile(r"(?<![\d,])" + re.escape(n) + r"(?![\d]|,\d)")
    if any(motif.search(t) for _, t in texte_rapport()):
        return True
    brut = normaliser(nombre, fusion=False).strip()
    motif = re.compile(r"(?<![\d,])" + re.escape(brut) + r"(?![\d]|,\d)")
    return any(motif.search(t) for _, t in texte_rapport(fusion=False))


def chercher(requete, contexte=70):
    q = normaliser(requete).strip()
    res = []
    for folio, t in texte_rapport():
        for m in re.finditer(re.escape(q), t):
            a, b = max(0, m.start() - contexte), min(len(t), m.end() + contexte)
            res.append((folio, t[a:b]))
    return res


if __name__ == "__main__":
    for r in sys.argv[1:]:
        trouves = chercher(r)
        print(f"== « {r} » : {len(trouves)} occurrence(s)")
        for folio, ctx in trouves[:12]:
            print(f"   folio {folio:3d} : ...{ctx}...")
