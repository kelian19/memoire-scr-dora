r"""Regenere materiel_complementaire.pdf.

METHODE, et c'est celle que le README impose : on compile le memoire ENTIER avec
les trois annexes deportees retablies, puis on extrait les pages. Compiler ces
trois fichiers a part imprimerait des ?? sur tous leurs renvois vers le corps.

Les \input sont places APRES les annexes du memoire (A a G) pour que le lettrage
du document depose reste intact : les trois restaurees prennent H, I et J.

Usage, depuis exploratory/memoire_cascade/ :

    python materiel_complementaire/build_matcomp.py
    <tectonic> -X compile _tmp_materiel.tex --keep-intermediates
    python materiel_complementaire/build_matcomp.py --extract

Le master temporaire _tmp_materiel.* n'est pas versionne : le supprimer apres.
"""

import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # exploratory/memoire_cascade

TMP = "_tmp_materiel"
OUT = os.path.join("materiel_complementaire", "materiel_complementaire.pdf")

PAGE_DE_TITRE = r"""
% MASTER TEMPORAIRE -- materiel complementaire. NE PAS VERSIONNER.
% La page ci-dessous est la premiere page du PDF extrait.
\clearpage
\thispagestyle{empty}
\vspace*{3cm}
\begin{center}
{\LARGE\bfseries Matériel complémentaire}\\[10pt]
{\large Quantification du SCR lié à la non-conformité\\ au règlement DORA}\\[24pt]
{\large Kélian Kaddouri}\\[6pt]
{ENSAE Paris --- Nexialog Consulting --- 2026}
\end{center}
\vspace{24pt}
\begin{quote}
\noindent Ce document rassemble les trois annexes sorties du mémoire pour en alléger la
lecture. Elles en conservent la numérotation : le mémoire porte les annexes~A à~G, et le
présent document prend la suite avec les annexes~H, I et~J. Les renvois qu'elles font vers
les chapitres du mémoire portent les numéros de celui-ci.

\medskip
\noindent \textbf{Annexe~H, démonstrations du socle.} Les théorèmes standards de la théorie des
valeurs extrêmes mobilisés au chapitre du socle mécaniste. Les démonstrations propres à la
contribution du mémoire, celles du modèle de cascade, restent dans le document déposé, à
l'annexe~A.

\medskip
\noindent \textbf{Annexe~I, pièces justificatives.} Les réconciliations entre sources, les corrections d'instrument et le registre des limites.

\medskip
\noindent \textbf{Annexe~J, compléments aux chapitres du corps.} Les développements dont le
raisonnement du mémoire ne dépend pas : lecture de marché détaillée, diagnostics de queue,
backtests complémentaires, sensibilités.
\end{quote}
\clearpage

\input{materiel_complementaire/15_demonstrations}
\input{materiel_complementaire/17_pieces_justificatives}
\input{materiel_complementaire/21_complements_corps}

"""


def ecrire_master():
    src = io.open("main_ensae.tex", encoding="utf-8").read()
    marker = "\\bibliographystyle"
    if src.count(marker) != 1:
        raise SystemExit("marqueur \\bibliographystyle trouve %d fois" % src.count(marker))
    io.open(TMP + ".tex", "w", encoding="utf-8").write(
        src.replace(marker, PAGE_DE_TITRE + marker)
    )
    print("master temporaire ecrit :", TMP + ".tex")
    print("compiler, puis relancer avec --extract")


def extraire():
    import pymupdf

    d = pymupdf.open(TMP + ".pdf")
    debut = [p.number for p in d if p.get_text().lstrip().startswith("Matériel complémentaire")]
    if not debut:
        raise SystemExit("page de titre introuvable dans " + TMP + ".pdf")
    out = pymupdf.open()
    out.insert_pdf(d, from_page=debut[0], to_page=d.page_count - 1)
    out.set_metadata(
        {
            "title": "Materiel complementaire -- Quantification du SCR lie a la "
            "non-conformite au reglement DORA",
            "author": "Kelian Kaddouri",
        }
    )
    out.save(OUT, deflate=True, garbage=4)

    import re

    c = pymupdf.open(OUT)
    manques = sum(len(re.findall(r"\?\?", p.get_text())) for p in c)
    print("%s : %d pages, %d renvoi(s) non resolu(s)" % (OUT, c.page_count, manques))
    if manques:
        raise SystemExit("des renvois ne sont pas resolus : ne pas deposer ce PDF en l'etat")


if __name__ == "__main__":
    os.chdir(ROOT)
    extraire() if "--extract" in sys.argv else ecrire_master()
