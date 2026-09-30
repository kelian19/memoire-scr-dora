#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Carte des figures imprimees par main_ensae.pdf : appel, script, page, taille.

Pour chaque image appelee par un chapitre de main_ensae.tex, imprime
  - la page du PDF ou elle tombe et le folio imprime en tete de page ;
  - sa taille imprimee en centimetres ;
  - le facteur << police x >> : une police de s points dans matplotlib
    s'imprime a s fois ce facteur (largeur imprimee / largeur de la figure) ;
  - le fichier .tex et la ligne de l'appel, avec la macro employee ;
  - le ou les scripts qui ecrivent l'image.

La page est retrouvee par les dimensions en pixels de l'image, puis, quand
plusieurs figures ont les memes dimensions, par comparaison des pixels. Une
image regeneree apres la compilation ne se retrouve donc plus dans le PDF :
recompiler avant de lire la carte.

Usage :
  python carte_figures.py [--pdf CHEMIN] [--image NOM ...] [--rendu DOSSIER] [--dpi N]

  --pdf     PDF a lire (defaut : main_ensae.pdf a cote de ce fichier). Un agent
            qui compile dans un dossier prive passe ici son propre PDF.
  --image   ne traiter que ces images (nom avec ou sans extension) ; repetable.
  --rendu   rendre en PNG chaque page portant une des figures retenues, avec la
            page qui precede et celle qui suit, pour juger la mise en page.
  --dpi     resolution du rendu (defaut 110).

Utilise par l'agent .claude/agents/verificateur-figures.md.
"""
import argparse
import glob
import io
import os
import re
import sys

import fitz
import numpy as np
from PIL import Image

MEM = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(MEM))
LAB = os.path.join(RACINE, "exploratory", "vasicek_lab")
# meme ordre que le \graphicspath du preambule, plus le dossier du memoire
CHEMINS = [os.path.join(LAB, "figures"),
           os.path.join(RACINE, "exploratory", "cascade_qualitative", "figures"),
           os.path.join(RACINE, "outputs"), MEM]
APPEL = re.compile(r"\\(figover|figcle|figcap|includegraphics)(\[[^\]]*\])?\{([^}]+?)\}")


def resout(nom):
    for c in CHEMINS:
        for ext in ("", ".png", ".jpg", ".pdf"):
            p = os.path.join(c, nom + ext)
            if os.path.isfile(p):
                return p
    return None


def sans_commentaires(texte):
    return "\n".join("" if l.lstrip().startswith("%") else l for l in texte.split("\n"))


def appels_du_memoire():
    maitre = sans_commentaires(io.open(os.path.join(MEM, "main_ensae.tex"), encoding="utf-8").read())
    appels = []
    for c in re.findall(r"\\input\{(chapitres/[^}]+)\}", maitre):
        p = os.path.join(MEM, c if c.endswith(".tex") else c + ".tex")
        if not os.path.exists(p):
            continue
        lignes = sans_commentaires(io.open(p, encoding="utf-8").read()).split("\n")
        for k, l in enumerate(lignes, 1):
            for m in APPEL.finditer(l):
                appels.append({"tex": "chapitres/" + os.path.basename(p), "ligne": k,
                               "macro": m.group(1) + (m.group(2) or ""), "image": m.group(3),
                               "chemin": resout(m.group(3))})
    return appels


def scripts_producteurs(appels):
    candidats = glob.glob(os.path.join(LAB, "**", "*.py"), recursive=True) + \
        glob.glob(os.path.join(RACINE, "notebooks", "*.py"))
    textes = {}
    for s in candidats:
        t = io.open(s, encoding="utf-8-sig", errors="replace").read()
        if "savefig" in t:
            textes[s] = t
    sources = {}
    for a in appels:
        base = os.path.splitext(os.path.basename(a["image"]))[0]
        motif = re.compile(re.escape(base) + r"(\.png|\.jpg|['\"])")
        for s, t in textes.items():
            if motif.search(t):
                rel = os.path.relpath(s, RACINE).replace("\\", "/")
                sources.setdefault(a["image"], [])
                if rel not in sources[a["image"]]:
                    sources[a["image"]].append(rel)
    return sources


def signature(im):
    return np.asarray(im.convert("L").resize((48, 16)), dtype=float)


def signature_pdf(doc, xref):
    pix = fitz.Pixmap(doc, xref)
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    if pix.n > 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    mode = "L" if pix.n == 1 else "RGB"
    return signature(Image.frombytes(mode, (pix.width, pix.height), pix.samples))


def principal():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=os.path.join(MEM, "main_ensae.pdf"))
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--rendu", default=None)
    ap.add_argument("--dpi", type=int, default=110)
    arg = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    tous = appels_du_memoire()
    appels = tous
    if arg.image:
        voulus = {os.path.splitext(os.path.basename(i))[0] for i in arg.image}
        appels = [a for a in tous if os.path.splitext(os.path.basename(a["image"]))[0] in voulus]
    sources = scripts_producteurs(appels)

    # dimensions et signature de TOUTES les images du memoire, pour lever les ambiguites
    par_taille, sig, dpi = {}, {}, {}
    for a in tous:
        if a["chemin"] and a["chemin"].lower().endswith((".png", ".jpg")):
            with Image.open(a["chemin"]) as im:
                a["pixels"] = im.size
                dpi[a["image"]] = float(im.info.get("dpi", (200, 200))[0]) or 200.0
                if a["image"] not in par_taille.setdefault(im.size, []):
                    par_taille[im.size].append(a["image"])
                sig[a["image"]] = signature(im)

    doc = fitz.open(arg.pdf)
    pages = {}
    for i in range(doc.page_count):
        page = doc[i]
        for img in page.get_images(full=True):
            xref, w, h = img[0], img[2], img[3]
            cands = par_taille.get((w, h), [])
            if not cands:
                continue
            nom = cands[0]
            if len(cands) > 1:
                s = signature_pdf(doc, xref)
                nom = min(cands, key=lambda n: float(((sig[n] - s) ** 2).mean()))
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            pages.setdefault(nom, [])
            if all(p != i + 1 for p, _ in pages[nom]):
                pages[nom].append((i + 1, rects[0] if rects else None))

    a_rendre = set()
    lignes = []
    for a in appels:
        occ = pages.get(a["image"], [])
        folio, impr, police = "?", "", ""
        if occ:
            p0, r = occ[0]
            tete = [t.strip() for t in doc[p0 - 1].get_text().strip().split("\n")[:4]]
            folio = next((t for t in tete if re.fullmatch(r"\d{1,3}", t)), "?")
            if r is not None and a.get("pixels"):
                impr = "%.1fx%.1f" % (r.width / 72 * 2.54, r.height / 72 * 2.54)
                police = "%.2f" % ((r.width / 72) / (a["pixels"][0] / dpi[a["image"]]))
            for p, _ in occ:
                a_rendre.update(q for q in (p - 1, p, p + 1) if 1 <= q <= doc.page_count)
        pix = "%dx%d" % a["pixels"] if a.get("pixels") else ""
        lignes.append((",".join(str(p) for p, _ in occ) or "-", folio, a["image"], pix, impr, police,
                       "%s:%d" % (a["tex"], a["ligne"]), a["macro"],
                       " ; ".join(sources.get(a["image"], ["? (aucun script trouve)"]))))

    lignes.sort(key=lambda l: int(l[0].split(",")[0]) if l[0][0].isdigit() else 99999)
    gabarit = "%-7s %-5s %-38s %-10s %-10s %-7s %-46s %-24s %s"
    print(gabarit % ("pagePDF", "folio", "image", "pixels", "impr. cm", "police", "appel", "macro", "script"))
    for l in lignes:
        print(gabarit % l)
    print("\n%d appel(s) d'image ; PDF lu : %s (%d pages)" % (len(lignes), arg.pdf, doc.page_count))
    print("police = facteur d'impression : une police de s pt dans matplotlib s'imprime a s x police pt")
    manquants = [l[2] for l in lignes if l[0] == "-"]
    if manquants:
        print("introuvables dans ce PDF (image regeneree depuis la compilation ?) : " + ", ".join(manquants))

    if arg.rendu:
        os.makedirs(arg.rendu, exist_ok=True)
        for p in sorted(a_rendre):
            doc[p - 1].get_pixmap(dpi=arg.dpi).save(os.path.join(arg.rendu, "page_%03d.png" % p))
        print("%d page(s) rendue(s) dans %s : %s" % (len(a_rendre), arg.rendu,
                                                    ", ".join("page_%03d.png" % p for p in sorted(a_rendre))))


if __name__ == "__main__":
    principal()
