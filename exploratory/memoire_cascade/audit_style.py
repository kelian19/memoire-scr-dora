#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_style.py : mesure les marqueurs de registre rhetorique du memoire.

POURQUOI CE FICHIER EXISTE. Le registre d'un memoire d'actuariat de l'Institut
est sobre : voix impersonnelle, titres nominaux, gras quasi absent du corps. Le
memoire s'en ecartait par un petit nombre de tournures qui reviennent, et qui
sont exactement celles qui font reconnaitre un texte ecrit avec assistance :
l'antithese << ce n'est pas X, c'est Y >>, la clause morale finale, le titre bati
sur le moule << X, et ce que Y >>. Une tournure isolee ne se voit pas ; c'est la
repetition qui marque, donc c'est une FREQUENCE qu'il faut mesurer, pas une
presence.

CE SCRIPT NE MODIFIE RIEN. Il compte et il localise. La reecriture se fait
ensuite a la main, phrase par phrase : une substitution automatique sur de la
prose detruirait le sens dans un cas sur trois, et le harnais ne le verrait pas
puisqu'il ne verifie que les nombres.

Lancement, depuis exploratory/memoire_cascade/ :
    ..\\..\\.venv\\Scripts\\python.exe audit_style.py            # tableau de bord
    ..\\..\\.venv\\Scripts\\python.exe audit_style.py --detail   # chaque occurrence
"""

import glob
import io
import os
import re
import sys
import unicodedata

ICI = os.path.dirname(os.path.abspath(__file__))
CHAPITRES = os.path.join(ICI, "chapitres")

# ---------------------------------------------------------------------------
# LES MARQUEURS, et le motif de chacun.
#
# Chaque entree : (cle, libelle, expression, ou chercher).
# Le champ « ou » vaut "texte" pour le corps, "titre" pour les seuls titres de
# section. Un moule de titre n'est pas un defaut dans une phrase.
# ---------------------------------------------------------------------------
MARQUEURS = [
    ("antithese",
     "antithese « ce n'est pas X, c'est Y »",
     r"n[e'’]\s*(?:est|sont|s[e'’]agit|tient|vient|porte|mesure|dit|s[e'’]explique)"
     r"\b[^.;:]{0,80}?\bpas\b[^.;:]{0,90}?,\s*(?:c[e'’]est|mais|elle est|il est|ils sont)\b",
     "texte"),

    ("moule_titre",
     "titre au moule « X, et ce que Y » ou « X, et pourquoi Y »",
     r",\s*(?:et\s+)?(?:ce\s+qu|pourquoi|ce\s+que|ce\s+qui|comment|de\s+combien|"
     r"et\s+celle\s+que|et\s+ce\s+qu)",
     "titre"),

    ("clause_morale",
     "clause morale finale (« et c'est ce qui compte », « il ne faut pas le perdre »)",
     r"(?:et\s+)?c[e'’]est\s+(?:la|le|ce|tout|precisement|exactement|bien)\s+"
     r"[^.;]{0,60}?(?:qui\s+compte|l[e'’]interet|le\s+point|tout\s+l[e'’]interet|"
     r"le\s+resultat|ce\s+qui\s+tient)|il\s+ne\s+faut\s+pas\s+(?:le|la|les|l[e'’])\s*"
     r"(?:perdre|oublier|confondre)|et\s+c[e'’]est\s+ce\s+qui\s+(?:compte|rend|fait)",
     "texte"),

    ("cest_precisement",
     "insistance « et c'est précisément / exactement ce que »",
     r"c[e'’]est\s+(?:precisement|exactement)\b",
     "texte"),

    ("ce_qui_compte",
     "emphase « ce qui compte / la vraie question / le vrai sujet »",
     r"\b(?:ce\s+qui\s+compte|la\s+vraie\s+question|le\s+vrai\s+(?:sujet|point|enjeu)|"
     r"ce\s+qui\s+est\s+en\s+jeu)\b",
     "texte"),

    ("loin_de",
     "tournure « loin de X, Y » et « bien au contraire »",
     r"\bloin\s+d[e'’]\s*(?:etre|constituer|resoudre)|bien\s+au\s+contraire\b",
     "texte"),

    ("non_seulement",
     "gradation « non seulement X, mais Y »",
     r"\bnon\s+seulement\b[^.;]{0,120}\bmais\b",
     "texte"),

    ("il_convient",
     "formule de remplissage (« il convient de », « force est de constater »)",
     r"\bil\s+convient\s+de\b|\bforce\s+est\s+de\s+constater\b|\bil\s+importe\s+de\b|"
     r"\bil\s+est\s+a\s+noter\b|\bnotons\s+que\b",
     "texte"),

    ("question_rhet",
     "question rhetorique dans le corps",
     r"(?<!\\)\?(?!\})",
     "texte"),

    ("en_realite",
     "« en realite », « en verite », « a vrai dire »",
     r"\ben\s+realite\b|\ben\s+verite\b|\ba\s+vrai\s+dire\b",
     "texte"),
]

SANS_ACCENT = str.maketrans("àâäéèêëïîôöùûüçÀÂÄÉÈÊËÏÎÔÖÙÛÜÇ",
                            "aaaeeeeiioouuucAAAEEEEIIOOUUUC")


def nettoyer(texte):
    """Retire commentaires, maths et commandes, pour ne compter que de la prose."""
    texte = re.sub(r"(?<!\\)%.*", "", texte)
    texte = re.sub(r"\$[^$]*\$", " ", texte)
    texte = re.sub(r"\\begin\{(equation|align|tabular|table|figure|verbatim)\*?\}"
                   r".*?\\end\{\1\*?\}", " ", texte, flags=re.S)
    texte = re.sub(r"\\(?:label|ref|eqref|cite[tp]?|includegraphics|texttt|"
                   r"vspace|hspace|newcommand)\{[^}]*\}", " ", texte)
    texte = re.sub(r"\\[a-zA-Z]+\*?", " ", texte)
    return texte


def titres(texte):
    return re.findall(r"\\(?:chapter|section|subsection|paragraph)\*?\{(.+?)\}\s*$",
                      texte, flags=re.M)


def analyse(chemin, detail=False):
    brut = io.open(chemin, encoding="utf-8", errors="replace").read()
    prose = nettoyer(brut)
    sans_acc = prose.translate(SANS_ACCENT)
    mots = len(re.findall(r"[A-Za-zÀ-ÿ]{2,}", prose))
    liste_titres = titres(brut)
    titres_acc = [t.translate(SANS_ACCENT) for t in liste_titres]

    comptes, occurrences = {}, []
    for cle, libelle, motif, ou in MARQUEURS:
        n = 0
        if ou == "titre":
            for t, ta in zip(liste_titres, titres_acc):
                if re.search(motif, ta, re.I):
                    n += 1
                    occurrences.append((cle, t.strip()[:110]))
        else:
            for m in re.finditer(motif, sans_acc, re.I):
                n += 1
                d, f = max(0, m.start() - 55), min(len(prose), m.end() + 55)
                extrait = re.sub(r"\s+", " ", prose[d:f]).strip()
                occurrences.append((cle, "..." + extrait + "..."))
        comptes[cle] = n

    total = sum(comptes.values())
    if detail and occurrences:
        print(f"\n--- {os.path.basename(chemin)} ---")
        for cle, ex in occurrences:
            print(f"  [{cle}] {ex}")
    return mots, comptes, total


def principal():
    detail = "--detail" in sys.argv
    fichiers = sorted(glob.glob(os.path.join(CHAPITRES, "*.tex")))
    tot_mots = tot_marq = 0
    lignes = []
    for f in fichiers:
        mots, comptes, total = analyse(f, detail)
        if mots == 0:
            continue
        tot_mots += mots
        tot_marq += total
        lignes.append((os.path.basename(f), mots, total, 1000.0 * total / mots, comptes))

    if not detail:
        print("=" * 96)
        print("MARQUEURS DE REGISTRE RHETORIQUE, PAR CHAPITRE")
        print("=" * 96)
        print(f"{'chapitre':<36}{'mots':>7}{'marq.':>7}{'/1000 mots':>12}   principaux")
        print("-" * 96)
        for nom, mots, total, taux, comptes in sorted(lignes, key=lambda x: -x[3]):
            top = ", ".join(f"{k} {v}" for k, v in
                            sorted(comptes.items(), key=lambda kv: -kv[1])[:3] if v)
            print(f"{nom:<36}{mots:>7}{total:>7}{taux:>12.2f}   {top}")
        print("-" * 96)
        print(f"{'TOTAL':<36}{tot_mots:>7}{tot_marq:>7}"
              f"{1000.0 * tot_marq / tot_mots:>12.2f}")
        print()
        print("Lancer avec --detail pour voir chaque occurrence avec son contexte.")


if __name__ == "__main__":
    principal()
