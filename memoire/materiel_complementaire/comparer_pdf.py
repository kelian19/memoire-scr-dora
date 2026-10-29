r"""Compare deux PDF page par page, titres courants et folios retires.

    python materiel_complementaire/comparer_pdf.py ancien.pdf nouveau.pdf

C'est le controle que ZENODO.md impose avant toute nouvelle version : si le PDF reconstruit
differe du fichier publie autrement que par les folios, une nouvelle version est due. Le texte
est normalise (ligatures, espaces) ; chaque difference est imprimee avec son contexte.
"""
import difflib
import re
import sys
import unicodedata

import pymupdf


def pages(chemin):
    out = []
    for p in pymupdf.open(chemin):
        t = unicodedata.normalize("NFKC", p.get_text())
        lignes = [l.strip() for l in t.splitlines() if l.strip()]
        # titres courants et folios : tete et pied portent le nom du chapitre et un nombre seul
        lignes = [l for l in lignes if not re.fullmatch(r"\d{1,3}", l)]
        if lignes and re.match(r"^(Chapitre|Annexe|[A-Z]\.|Bibliographie|Mat)", lignes[0]) and len(lignes[0]) < 90:
            lignes = lignes[1:]
        out.append(" ".join(lignes))
    return out


def main():
    a, b = pages(sys.argv[1]), pages(sys.argv[2])
    print(f"pages : {len(a)} contre {len(b)}")
    mots_a = " ".join(a).split()
    mots_b = " ".join(b).split()
    sm = difflib.SequenceMatcher(None, mots_a, mots_b, autojunk=False)
    n = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        avant = " ".join(mots_a[i1:i2])
        apres = " ".join(mots_b[j1:j2])
        # folios et titres courants qui se glissent dans le flux : nombres seuls
        if re.fullmatch(r"[\d\s]*", avant) and re.fullmatch(r"[\d\s]*", apres):
            continue
        n += 1
        ctx = " ".join(mots_a[max(0, i1 - 8):i1])
        print(f"[{n}] ... {ctx}\n     - {avant[:300]}\n     + {apres[:300]}\n")
    print(f"{n} difference(s) hors folios")


if __name__ == "__main__":
    main()
