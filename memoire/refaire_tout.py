#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TOUTE LA CHAINE EN UNE COMMANDE : generer, compiler, controler, recopier.

    python exploratory/memoire_cascade/refaire_tout.py

MOTIF. Cette chaine a ete faite a la main des dizaines de fois, et chaque etape
oubliee a deja coute une erreur reelle, consignee dans CLAUDE.md :

  - OUBLIER DE REDIRIGER STDERR. tectonic ecrit TOUS ses avertissements sur
    stderr. Un << > log.txt >> ne capture donc rien des debordements, et le
    controle passe a vide en donnant zero partout. Deux controles ont ete
    annonces faux le 9 septembre pour cette raison ;
  - COMPTER LES DEBORDEMENTS PAR LIGNE D'AVERTISSEMENT. Les deux passes de
    tectonic ne donnent pas toujours la meme largeur au dernier chiffre, et le
    meme paragraphe ressort alors sous deux valeurs. On dedoublonne donc sur
    l'emplacement, en retirant la valeur en points ;
  - CHERCHER LES REFERENCES NON RESOLUES DANS LE JOURNAL. tectonic n'emet
    AUCUN avertissement pour un \\ref casse : le grep renvoie zero quoi qu'il
    arrive. Cinq << cite au chapitre ?? >> ont ainsi ete publies. Le seul test
    fiable compte les << ?? >> sur le PDF produit ;
  - OUBLIER DE RECOPIER LES FICHIERS DEPOSES. Ils se periment a la moindre
    recompilation, et l'on a deja demande si le fichier etait pret a envoyer
    alors que la copie avait une heure douze de retard.

LE SCRIPT NE CORRIGE RIEN. Il fait, il mesure, et il dit ce qui cloche. Son code
de sortie vaut 0 si tous les controles passent, 1 sinon, de sorte qu'il puisse
servir de garde-fou avant un depot.

LES DEUX LIGNES DE BASE SONT DES CONSTANTES NOMMEES ci-dessous, et elles bougent
avec le document : les relire ici plutot que de les chercher dans un journal.
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SORTIES = os.path.abspath(os.path.join(os.path.dirname(__file__), '../resultats/sorties_verif'))

# --- ce qui bouge avec le document -----------------------------------------
# LIGNE DE BASE DES Overfull \hbox, PAR VERSION. Institut : 2 depuis le 22 septembre 2026
# (02b_cadre_cyber_dora:266 et 16_notations:79). ENSAE : 3 depuis la phase 3 de la reduction
# (1er octobre 2026), les deux memes plus 09_resultats:210 a 0,14 pt, ou « tableau 10.x »
# devenu « tableau 9.x » decale une coupure de ligne ; ce paragraphe se reecrit en phase 4.
# La ligne de base ENSAE se revoit a chaque phase de la reduction, jamais pour masquer un
# debordement neuf non explique.
DEBORDEMENTS_BASE = {"main_ensae": 3, "main_institut": 2}
PAGES_MAX_ENSAE = 100        # limite de la version ENSAE finalisee
PAGES_CIBLE_ENSAE = 100

VERSIONS = [
    ("main_ensae", "KADDOURI_Kelian_MS25.pdf", PAGES_MAX_ENSAE),
    ("main_institut", "KADDOURI_Kelian_institut.pdf", None),
]

INTERMEDIAIRES = (".aux", ".bbl", ".blg", ".log", ".out", ".toc")

VERT, ROUGE, GRAS, FIN = "\033[32m", "\033[31m", "\033[1m", "\033[0m"
if platform.system() == "Windows" and not os.environ.get("WT_SESSION"):
    VERT = ROUGE = GRAS = FIN = ""

defauts = []


def titre(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def verdict(libelle, ok, detail=""):
    marque = (VERT + "OK  " + FIN) if ok else (ROUGE + "ECHEC" + FIN)
    print("  [%s] %-44s %s" % (marque, libelle, detail))
    if not ok:
        defauts.append(libelle)
    return ok


# ---------------------------------------------------------------------------
# 0. Les prerequis, verifies AVANT de perdre trois minutes de compilation
# ---------------------------------------------------------------------------

def trouver_tectonic():
    """tectonic n'est pas dans le PATH sur les deux postes, et son nom differe."""
    noms = ["tectonic.exe", "tectonic"] if platform.system() == "Windows" else ["tectonic"]
    pistes = [os.path.join(REPO, "outils", "tectonic", n) for n in noms]
    pistes += [r"C:\Users\kelia\miniconda3\Library\bin\tectonic.exe"]
    for p in pistes:
        if os.path.isfile(p):
            return p
    trouve = shutil.which("tectonic")
    return trouve


def prerequis():
    titre("0. PREREQUIS")
    ok = True

    tec = trouver_tectonic()
    ok &= verdict("tectonic", tec is not None,
                  tec or "introuvable : le placer dans memoire/ ou dans le PATH")

    try:
        import fitz  # noqa: F401
        pdf_ok = True
    except ImportError:
        pdf_ok = False
    ok &= verdict("pymupdf (lecture des PDF)", pdf_ok,
                  "" if pdf_ok else "pip install pymupdf")

    nb = len([f for f in os.listdir(SORTIES) if f.endswith(".txt")]) \
        if os.path.isdir(SORTIES) else 0
    ok &= verdict("sorties versionnees", nb > 100, "%d fichiers" % nb)

    # la donnee brute n'est PAS necessaire pour compiler le memoire : elle ne
    # l'est que pour REJOUER un script d'analyse. On le signale sans echouer.
    brut = os.path.join(REPO, "data", "raw")
    a_brut = os.path.isdir(brut) and bool(os.listdir(brut))
    print("  [%s] %-44s %s"
          % ("info ", "data/raw (sous licence, gitignore)",
             "present" if a_brut else "ABSENT : les scripts d'analyse ne "
                                      "tourneront pas, le memoire compile quand meme"))
    return ok, tec


# ---------------------------------------------------------------------------
# 1. La version Institut est FIGEE : elle doit redonner le fichier depose
# ---------------------------------------------------------------------------
# DEPUIS LE 1er OCTOBRE 2026, LES DEUX VERSIONS NE SONT PLUS LIEES. main_institut.tex
# n'est plus genere depuis main_ensae.tex (build_institut.py est retire) : il est la
# version complete et figee du memoire, et seule la version ENSAE se condense, demande
# de Caroline Hillairet et decision de Kelian. Ce controle verifie apres compilation
# que la version Institut a exactement le contenu du PDF committe (texte, images et
# liens, page par page). tectonic horodatant chaque PDF, l'egalite des octets ne
# prouverait rien. Si le contenu est identique, le PDF committe est restaure, pour
# qu'une compilation de controle ne salisse pas le depot. Changer volontairement la
# version Institut, c'est committer son nouveau PDF : ce controle compare au dernier
# commit.

def empreinte_page(page, avec_images):
    liens = sorted((l.get("uri") or "", l.get("page", -1)) for l in page.get_links())
    images = sorted((x[2], x[3]) for x in page.get_images(full=True)) if avec_images else []
    return page.get_text(), images, liens


def institut_fige():
    titre("1. VERSION INSTITUT FIGEE : MEME CONTENU QUE LE FICHIER DEPOSE")
    import fitz
    rel = 'memoire/main_institut.pdf'
    r = subprocess.run(["git", "show", "HEAD:" + rel], capture_output=True, cwd=REPO)
    if r.returncode or not r.stdout:
        print("  [info ] %-44s %s" % ("version Institut", "aucun PDF committe : rien a comparer"))
        return True
    ref = fitz.open(stream=r.stdout, filetype="pdf")
    new = fitz.open(os.path.join(ICI, "main_institut.pdf"))
    # Sans les griffes (poste neuf, integration continue), la page 1 perd ses deux
    # signatures : on n'y compare alors que le texte et les liens.
    griffes = all(os.path.isfile(os.path.join(ICI, "logos", f))
                  for f in ("signature_kelian.png", "signature_hugo.png"))
    if ref.page_count != new.page_count:
        return verdict("version Institut inchangee", False,
                       "%d pages au lieu de %d" % (new.page_count, ref.page_count))
    ecarts = [i + 1 for i in range(ref.page_count)
              if empreinte_page(ref[i], griffes or i > 0) != empreinte_page(new[i], griffes or i > 0)]
    n_pages = ref.page_count
    # Fermer AVANT de restaurer : sous Windows, git ne peut pas ecraser un PDF encore ouvert
    # par PyMuPDF, et l'echec passait en silence le 1er octobre 2026.
    ref.close()
    new.close()
    if ecarts:
        return verdict("version Institut inchangee", False,
                       "%d page(s) modifiee(s), la premiere p.%d" % (len(ecarts), ecarts[0]))
    r = subprocess.run(["git", "checkout", "--", rel], cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        print("  [info ] %-44s %s" % ("PDF committe non restaure", (r.stderr or "").strip()[:60]))
    return verdict("version Institut inchangee", True,
                   "%d pages identiques au fichier committe%s"
                   % (n_pages, "" if griffes else " (page 1 sans les griffes)"))


# ---------------------------------------------------------------------------
# 2. Compiler, EN CAPTURANT LES DEUX FLUX
# ---------------------------------------------------------------------------

def compiler(tec, base):
    r = subprocess.run([tec, "-X", "compile", base + ".tex", "--keep-intermediates"],
                       capture_output=True, text=True, cwd=ICI)
    journal = (r.stdout or "") + (r.stderr or "")   # tectonic ecrit sur STDERR

    # dedoublonnage PAR EMPLACEMENT : on retire la valeur en points
    hbox = {re.sub(r": Overfull.*", "", l)
            for l in journal.split("\n") if "Overfull \\hbox" in l}
    vbox = sum(1 for l in journal.split("\n") if "Overfull \\vbox" in l)
    hors = sum(1 for l in journal.split("\n") if "out of page boundary" in l)
    cite = sum(1 for l in journal.split("\n") if "Citation" in l and "undefined" in l)
    return r.returncode, sorted(hbox), vbox, hors, cite


def etape_compilation(tec):
    titre("2. COMPILATION DES VERSIONS SELECTIONNEES")
    tout_ok = True
    for base, _, _ in VERSIONS:
        for ext in (".bbl",):                 # sinon l'ancien rendu de biblio est repris
            p = os.path.join(ICI, base + ext)
            if os.path.exists(p):
                os.remove(p)
        code, hbox, vbox, hors, cite = compiler(tec, base)
        print("\n  %s%s%s" % (GRAS, base, FIN))
        tout_ok &= verdict("compilation sans erreur", code == 0, "code %d" % code)
        tout_ok &= verdict("debordements (ligne de base %d)" % DEBORDEMENTS_BASE[base],
                           len(hbox) <= DEBORDEMENTS_BASE[base], "%d" % len(hbox))
        if len(hbox) > DEBORDEMENTS_BASE[base]:
            for h in hbox:
                print("        " + h.strip())
        tout_ok &= verdict("Overfull \\vbox", vbox == 0, "%d" % vbox)
        tout_ok &= verdict("annotations hors page", hors == 0, "%d" % hors)
        tout_ok &= verdict("citations non resolues", cite == 0, "%d" % cite)
    return tout_ok


# ---------------------------------------------------------------------------
# 3. Controles SUR LE PDF PRODUIT
# ---------------------------------------------------------------------------

def prose_hors_legende(page):
    """La page porte-t-elle du texte courant, hors tete, pied et legende ?

    C'est ce qui distingue une page composee d'une page de flottant. Un simple
    comptage de caracteres ne le distingue pas : une page dense a deux figures
    porte peu de texte et n'a rien d'une page vide.
    """
    import unicodedata
    cm = 72 / 2.54
    haut = page.rect.height
    for _, y0, _, y1, txt, *_ in page.get_text("blocks"):
        t = " ".join(txt.split())
        if len(t) < 90:
            continue
        if y1 < 2.2 * cm + 22 or y0 > haut - 2.2 * cm - 22:   # tete et pied
            continue
        sans = "".join(c for c in unicodedata.normalize("NFKD", t)
                       if not unicodedata.combining(c))
        if sans.lower().startswith(("figure", "table", "tableau")):
            continue
        return True
    return False


def etape_pdf():
    titre("3. CONTROLES SUR LE PDF PRODUIT")
    import fitz
    tout_ok = True
    for base, _, pages_max in VERSIONS:
        chemin = os.path.join(ICI, base + ".pdf")
        if not os.path.isfile(chemin):
            tout_ok &= verdict("%s existe" % base, False, "PDF absent")
            continue
        d = fitz.open(chemin)
        textes = [p.get_text() for p in d]
        print("\n  %s%s%s" % (GRAS, base, FIN))
        n_pages = d.page_count

        if pages_max:
            tout_ok &= verdict("pages (limite imperative %d)" % pages_max,
                               n_pages <= pages_max,
                               "%d, marge %d" % (n_pages, pages_max - n_pages))
        if base == "main_ensae" and PAGES_CIBLE_ENSAE < pages_max:
            print("  [info ] %-44s %s" % ("objectif de condensation %d pages" % PAGES_CIBLE_ENSAE,
                                          "%d pages, %+d" % (n_pages, n_pages - PAGES_CIBLE_ENSAE)))
        else:
            print("  [info ] %-44s %d" % ("pages", n_pages))

        tout_ok &= verdict("aucun << ?? >> (compte SUR LE PDF)",
                           sum(t.count("??") for t in textes) == 0,
                           "%d" % sum(t.count("??") for t in textes))
        tout_ok &= verdict("aucune page tournee",
                           all(p.rotation == 0 for p in d),
                           "%d" % sum(1 for p in d if p.rotation))
        # UNE PAGE DE FLOTTANT : une page qui ne porte QUE des figures et leurs
        # legendes. Le critere n'est pas un nombre de caracteres, et le seuil de
        # 1 300 employe d'abord etait faux : une page dense a deux figures en
        # porte 1 018 sans rien avoir d'une page vide. Le critere est la
        # PRESENCE de prose, c'est-a-dire d'un bloc de texte qui ne soit ni la
        # tete, ni le pied, ni une legende.
        seules = [i + 1 for i in range(1, d.page_count)
                  if d[i].get_images() and not prose_hors_legende(d[i])]
        tout_ok &= verdict("aucune figure seule sur sa page",
                           not seules, ("pages " + str(seules)) if seules else "")
        d.close()
    return tout_ok


# ---------------------------------------------------------------------------
# 4. Harnais, tous les chapitres
# ---------------------------------------------------------------------------

def etape_harnais():
    titre("4. HARNAIS DE VERIFICATION DES NOMBRES")
    total = confirmes = 0
    hors_controle = []
    print("  %-44s%8s%10s%9s" % ("chapitre", "verif.", "confirmes", "taux"))
    print("  " + "-" * 71)
    # la version ENSAE lit ses chapitres dans chapitres_sfe/, et ce qu'elle retire vit dans
    # supplement/ (reduction SFE, phase 3, 1er octobre 2026) : les deux se verifient aussi
    fichiers = [(d, f) for d in ("chapitres", "chapitres_sfe", "supplement")
                if os.path.isdir(os.path.join(ICI, d))
                for f in sorted(os.listdir(os.path.join(ICI, d))) if f.endswith(".tex")]
    for d, f in fichiers:
        dossier = os.path.join(ICI, d)
        etiquette = f[:-4] if d == "chapitres" else "%s/%s" % (d, f[:-4])
        # encodage fixe des deux cotes : lancee sous PYTHONIOENCODING=utf-8, la sortie du
        # harnais etait relue dans l'encodage du poste, et « vérifiables » n'etait plus reconnu
        r = subprocess.run([sys.executable, os.path.join(ICI, "verif_chiffres.py"),
                            SORTIES, os.path.join(dossier, f)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        s = (r.stdout or "") + (r.stderr or "")
        m = re.search(r"(\d+) nombres v.rifiables, (\d+) confirm", s)
        h = re.search(r"Hors controle non declare\s*:\s*(\d+)", s)
        if h and int(h.group(1)) > 0:
            hors_controle.append("%s (%s)" % (etiquette, h.group(1)))
        if m:
            v, c = int(m.group(1)), int(m.group(2))
            total += v
            confirmes += c
            print("  %-44s%8d%10d%8.1f %%" % (etiquette, v, c, 100 * c / v if v else 0))
        else:
            print("  %-44s%8s%10s%9s" % (etiquette, "-", "-", "sans script"))
    print("  " + "-" * 71)
    print("  %-44s%8d%10d%8.1f %%"
          % ("TOTAL", total, confirmes, 100 * confirmes / total if total else 0))
    print()
    ok = verdict("confirmation >= 97 %", total and confirmes / total >= 0.97,
                 "%.1f %%" % (100 * confirmes / total if total else 0))
    ok &= verdict("hors controle non declare = 0", not hors_controle,
                  ", ".join(hors_controle) if hors_controle else "")
    return ok


# ---------------------------------------------------------------------------
# 5. Recopier les fichiers deposes, puis nettoyer
# ---------------------------------------------------------------------------

def etape_depot():
    titre("5. FICHIERS DEPOSES ET NETTOYAGE")
    import hashlib
    ok = True
    for base, depose, _ in VERSIONS:
        src = os.path.join(ICI, base + ".pdf")
        dst = os.path.join(ICI, depose)
        if not os.path.isfile(src):
            ok &= verdict(depose, False, "source absente")
            continue
        shutil.copyfile(src, dst)
        h = [hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (src, dst)]
        ok &= verdict(depose, h[0] == h[1], h[0][:16])
    n = 0
    for base, _, _ in VERSIONS:
        for ext in INTERMEDIAIRES:
            p = os.path.join(ICI, base + ext)
            if os.path.exists(p):
                os.remove(p)
                n += 1
    print("  [info ] %-44s %d fichiers" % ("intermediaires nettoyes", n))
    return ok


# ---------------------------------------------------------------------------

def main():
    global VERSIONS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sans-harnais", action="store_true",
                    help="saute l'etape 4, qui prend une minute")
    ap.add_argument("--version", choices=("ensae", "institut", "toutes"), default="toutes",
                    help="version a compiler et recopier (defaut : toutes)")
    a = ap.parse_args()
    if a.version != "toutes":
        VERSIONS = [v for v in VERSIONS if v[0] == "main_" + a.version]

    print(GRAS + "CHAINE COMPLETE DU MEMOIRE" + FIN)
    print("depot : %s" % REPO)
    print("poste : %s, Python %s" % (platform.system(), platform.python_version()))

    ok, tec = prerequis()
    if not ok:
        print("\n" + ROUGE + "Prerequis manquants : on s'arrete avant de compiler." + FIN)
        return 1

    ok = etape_compilation(tec)
    ok &= etape_pdf()
    if any(v[0] == "main_institut" for v in VERSIONS):
        ok &= institut_fige()
    if not a.sans_harnais:
        ok &= etape_harnais()
    if ok and not defauts:
        ok &= etape_depot()
    else:
        print("\nFichiers de depot non remplaces : des controles ont echoue.")

    titre("VERDICT")
    if ok and not defauts:
        print(VERT + "  Tous les controles passent. Les fichiers deposes sont a jour." + FIN)
        return 0
    print(ROUGE + "  %d controle(s) en echec :" % len(defauts) + FIN)
    for d in defauts:
        print("    - " + d)
    return 1


if __name__ == "__main__":
    sys.exit(main())
