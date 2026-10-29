r"""Construit supplement_rapport_ENSAE.pdf, le supplement S1 a S4 du rapport de stage ENSAE.

(S5, les demonstrations du modele de cascade, est redevenu l'annexe A du rapport le 1er octobre
2026 tard le soir : demande de Kelian d'enrichir le rapport jusqu'a 100 pages.)

    python materiel_complementaire/build_supplement.py        (depuis exploratory/memoire_cascade/)

METHODE, celle du materiel complementaire : on ne compile pas le supplement seul, qui imprimerait
des ?? sur ses renvois vers le rapport. On compile le rapport ENTIER (main_ensae.tex) avec S1 a S4
inseres avant la bibliographie, chapitres numerotes S1 a S4 comme dans le calcul des etiquettes
relais (reduction/relais_supplement.py), puis on extrait de la page de titre du supplement a la
fin de la bibliographie. Les renvois du supplement vers le rapport portent ainsi les numeros du
rapport, et ceux du rapport vers le supplement (S1.2, S4.63...) se retrouvent dans ce PDF.

Le master temporaire est ecrit puis efface ; la compilation se fait dans un dossier temporaire.

CONTROLES, et le script s'arrete au moindre echec :
  - aucun renvoi non resolu (??) dans le PDF extrait ;
  - chaque etiquette relais de supplement/relais.tex porte dans ce PDF le numero que le rapport
    imprime ;
  - aucune chaine sensible (noms d'entites ou de personnes, adresses, chemins de poste), avec le
    motif de ci/vitrine/exporter.py quand il est present.
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # exploratory/memoire_cascade
DEPOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
OUT = os.path.join(HERE, "supplement_rapport_ENSAE.pdf")
NOM = "zz_supplement_tmp"
FICHIERS = [
    "S1_adaptations_pilier",
    "S2_pistes_futurs_memoires",
    "S3_questionnaire_elicitation",
    "S4_diagnostics_secondaires",
]

PAGE_DE_TITRE = r"""
% MASTER TEMPORAIRE -- supplement du rapport ENSAE. NE PAS VERSIONNER.
% La page ci-dessous est la premiere page du PDF extrait.
\clearpage
\thispagestyle{empty}
\vspace*{3cm}
\begin{center}
{\LARGE\bfseries Supplément au rapport de stage}\\[10pt]
{\large Quantification du SCR lié à la non-conformité\\ au règlement DORA}\\[24pt]
{\large Kélian Kaddouri}\\[6pt]
{ENSAE Paris, Nexialog Consulting, 2026}
\end{center}
\vspace{24pt}
\begin{quote}
\noindent Ce document accompagne la version du mémoire déposée à l'ENSAE comme rapport de stage
de fin d'études. Il rassemble les contenus que ce rapport, ramené à moins de cent pages, ne
reprend pas. Ses chapitres sont numérotés S1 à S4, comme dans les renvois du rapport ; ses
renvois vers les chapitres du rapport portent les numéros de celui-ci.

\medskip
\noindent \textbf{S1, adaptations par pilier.} Queue de sévérité par pilier, pilier des tiers
comme nœud d'accumulation, indicateurs pilotables et test de résistance inversé.

\medskip
\noindent \textbf{S2, pistes pour de futurs mémoires d'actuariat.}

\medskip
\noindent \textbf{S3, questionnaire d'élicitation.} Le questionnaire diffusé, sans les valeurs des
questions d'étalonnage.

\medskip
\noindent \textbf{S4, compléments retirés du corps du rapport.} Les passages des chapitres 1,
3 à 7, 9 et 10 et de la conclusion que le rapport résume, recopiés à l'identique, ainsi que le
protocole d'élicitation et les relectures de praticiens.

\medskip
\noindent La version complète du mémoire, déposée à l'Institut des actuaires, porte S1 et S2
comme annexes B et F, S3 comme section C.3, et les passages de S4 dans ses chapitres et son
annexe C. Son matériel complémentaire, annexes H à J, est déposé dans le même dépôt Zenodo.
\end{quote}
\clearpage
\renewcommand{\thechapter}{S\arabic{chapter}}
% ancres hyperref propres au supplement : sans elles, chapter.1, section.1.1... doublonnent celles
% du rapport compile avec lui, et signets comme liens internes pointent vers le rapport
\providecommand{\theHchapter}{}
\renewcommand{\theHchapter}{S\arabic{chapter}}
\setcounter{chapter}{0}
"""


def tectonic():
    for p in (shutil.which("tectonic"),):
        if p and os.path.isfile(p):
            return p
    sys.exit("tectonic introuvable")


def master():
    src = open(os.path.join(ROOT, "main_ensae.tex"), encoding="utf-8").read().replace("\r\n", "\n")
    # les etiquettes relais redefiniraient les etiquettes du supplement : on les retire
    src = re.sub(r"^\\input\{supplement/relais\}.*$", "", src, flags=re.M)
    marqueur = "\\bibliographystyle"
    if src.count(marqueur) != 1:
        sys.exit("marqueur \\bibliographystyle trouve %d fois" % src.count(marqueur))
    ajout = PAGE_DE_TITRE + "".join("\\input{supplement/%s}\n" % f for f in FICHIERS)
    return src.replace(marqueur, ajout + marqueur)


def relais_attendus():
    t = open(os.path.join(ROOT, "supplement", "relais.tex"), encoding="utf-8").read()
    return dict(re.findall(r"\\etiquettesupplement\{([^}]+)\}\{([^}]+)\}", t))


def motif_sensible():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../outils/ci/vitrine')))
    try:
        from exporter import CHEMIN_POSTE, SENSIBLE  # noqa: E402
        return SENSIBLE, CHEMIN_POSTE
    except Exception:  # dans la vitrine, ci/vitrine/ n'existe pas
        return None, re.compile(r"[A-Za-z]:[\\/]+Users[\\/]|/" + "Users/|/" + "home/")


def main():
    import pymupdf

    sys.path.insert(0, HERE)
    from build_matcomp import signets  # noqa: E402

    tmp = tempfile.mkdtemp(prefix="supplement_")
    chemin_master = os.path.join(ROOT, NOM + ".tex")
    with open(chemin_master, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(master())
    try:
        r = subprocess.run([tectonic(), "-X", "compile", NOM + ".tex", "--keep-intermediates",
                            "--outdir", tmp], cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode:
            sys.exit("compilation en echec :\n" + r.stderr[-3000:])
        aux = open(os.path.join(tmp, NOM + ".aux"), encoding="utf-8", errors="replace").read()
        d = pymupdf.open(os.path.join(tmp, NOM + ".pdf"))
        debut = [p.number for p in d if p.get_text().lstrip().startswith("Supplément au rapport")]
        # le sommaire cite aussi la notice : on prend sa premiere page apres le titre du supplement
        notice = [p.number for p in d if "Organisation des annexes" in p.get_text()[:300]
                  and debut and p.number > debut[0]]
        if not debut or not notice or notice[0] <= debut[0]:
            garde = os.path.join(tempfile.gettempdir(), NOM + "_echec.pdf")
            shutil.copy(os.path.join(tmp, NOM + ".pdf"), garde)
            sys.exit("page de titre (%s) ou notice des annexes (%s) introuvable ; PDF garde : %s"
                     % (debut[:3], notice[:3], garde))
        fin = notice[0] - 1
        out = pymupdf.open()
        out.insert_pdf(d, from_page=debut[0], to_page=fin)
        out.set_toc(signets(d, debut[0], fin, "Supplément au rapport de stage"))
        out.set_metadata({
            "title": "Supplement au rapport de stage ENSAE -- Quantification du SCR lie a la "
                     "non-conformite au reglement DORA",
            "author": "Kelian Kaddouri",
        })
        out.save(OUT, deflate=True, garbage=4)
    finally:
        os.remove(chemin_master)
        shutil.rmtree(tmp, ignore_errors=True)

    arret = []
    c = pymupdf.open(OUT)
    texte = unicodedata.normalize("NFKC", "".join(p.get_text() for p in c))
    manques = len(re.findall(r"\?\?", texte))
    if manques:
        arret.append("%d renvoi(s) non resolu(s)" % manques)
    numeros = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{(.*?)\}\{", aux))
    for lab, num in relais_attendus().items():
        if numeros.get(lab) != num:
            arret.append("relais %s : le rapport imprime %s, le supplement porte %s"
                         % (lab, num, numeros.get(lab)))
    sensible, chemin = motif_sensible()
    if sensible is None:
        print("avertissement : motif des chaines sensibles indisponible, balayage partiel")
    else:
        for m in sorted(set(sensible.findall(texte))):
            arret.append("chaine sensible dans le PDF : %s" % m)
    if chemin.search(texte):
        arret.append("chemin de poste dans le PDF")

    octets = open(OUT, "rb").read()
    print("%s : %d pages, %d signets, %d octets, md5 %s" % (
        os.path.relpath(OUT, ROOT), c.page_count, len(c.get_toc()), len(octets),
        hashlib.md5(octets).hexdigest()))
    print("relais verifies : %d ; renvois non resolus : %d" % (len(relais_attendus()), manques))
    if arret:
        print("ARRET :\n  " + "\n  ".join(arret))
        sys.exit(1)
    print("controles passes")


if __name__ == "__main__":
    main()
