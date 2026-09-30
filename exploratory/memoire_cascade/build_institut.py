r"""Genere main_institut.tex a partir de main_ensae.tex.

POURQUOI UN SCRIPT ET NON UN SECOND FICHIER ECRIT A LA MAIN. Le projet a supprime
main.tex et main_v2.tex le 19 septembre 2026 parce que plusieurs fichiers maitres
tenus en parallele divergent en une semaine. Les deux versions ne different ici
que par UNE ligne, le chapitre du stage, et la regle du projet reste intacte :
aucun chapitre n'est duplique, et main_ensae.tex demeure la source unique.

CE QUI DIFFERE, ET RIEN D'AUTRE :
  - la version Institut n'appelle pas chapitres/19_enseignements_stage, ni la
    partie qui l'introduit. L'Institut ne demande nulle part de chapitre de recul
    sur le stage ; l'ecole, si.

CE QUI NE DIFFERE PAS, ET C'EST UNE DECISION DE KELIAN DU 21 SEPTEMBRE :
  - la page de garde est le GABARIT OFFICIEL (page_de_garde.tex), commun aux
    deux versions depuis le 28 septembre 2026. C'est desormais
    page_de_garde_ensae.tex, la couverture libre dessinee pour l'ancien rapport
    de stage, qui est orpheline : la liste de controle de l'ecole interdit d'en
    modifier la structure et d'y ajouter un logo que le gabarit ne prevoit pas ;
  - le format reste Times 12 / interligne 1,5 via \formatensae. Le § 5.2 des
    Recommandations Jury laisse la typographie libre, donc rien ne s'y oppose ;
  - tout le reste du corps, des annexes et de la bibliographie est identique.

Usage, depuis exploratory/memoire_cascade/ :

    python build_institut.py
    <tectonic> -X compile main_institut.tex --keep-intermediates

main_institut.tex est REGENERABLE : ne pas l'editer a la main, editer
main_ensae.tex puis relancer ce script.
"""

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = "main_ensae.tex"
DST = "main_institut.tex"

ENTETE = r"""% =============================================================================
%  VERSION INSTITUT DES ACTUAIRES
%
%  FICHIER GENERE. Ne pas editer a la main : editer main_ensae.tex, puis
%  relancer  python build_institut.py  depuis ce dossier.
%
%  Il ne differe de la version ENSAE que par UN point : il n'appelle pas le
%  chapitre de recul sur le stage, que l'Institut ne demande nulle part. La page
%  de garde, le format et tout le reste du corps sont identiques, sur decision
%  du 21 septembre 2026.
%
%  FICHIER A DEPOSER : KADDOURI_Kelian_institut.pdf
% =============================================================================

"""


def main():
    os.chdir(ROOT)
    src = io.open(SRC, encoding="utf-8").read()

    # On retire le bloc de la partie « Le stage », de son separateur d'ouverture
    # jusqu'a la ligne d'\input incluse.
    motif = re.compile(
        r"% =+\n\\part\{Le stage\}.*?\\input\{chapitres/19_enseignements_stage\}[^\n]*\n",
        re.S,
    )
    out, n = motif.subn("", src)
    if n != 1:
        sys.exit("le bloc de la partie « Le stage » a ete trouve %d fois, attendu 1" % n)

    # On remplace l'en-tete de la version ENSAE par celui de la version Institut.
    # A faire AVANT le controle ci-dessous : l'ancien en-tete cite le chapitre du
    # stage dans sa liste de ce qui est propre a la version ecole.
    fin_entete = out.index("\\documentclass")
    out = ENTETE + out[fin_entete:]

    # Session de l'Institut : mars 2027. La couverture (page_de_garde.tex) porte
    # novembre 2026 par defaut, celle de l'ENSAE.
    if out.count("\\input{page_de_garde}") != 1:
        sys.exit("\\input{page_de_garde} attendu une fois")
    out = out.replace("\\input{page_de_garde}",
                      "\\newcommand*{\\pgdatesoutenance}{mars 2027}\n\\input{page_de_garde}")

    if "19_enseignements_stage" in out:
        sys.exit("le chapitre du stage subsiste dans la sortie")

    io.open(DST, "w", encoding="utf-8").write(out)
    print("%s ecrit (%d caracteres, %d de moins que %s)"
          % (DST, len(out), len(src) - len(out), SRC))
    print("chapitres appeles : %d" % len(re.findall(r"\\input\{chapitres/", out)))


if __name__ == "__main__":
    main()
