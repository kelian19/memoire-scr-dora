#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Les controles de fin de tache, passes en une commande sur le PDF produit.

Ce que CLAUDE.md exige avant de dire qu'une tache est finie, plus les deux
controles de mise en page nes du reproche de Kelian du 15 septembre 2026,
<< l'image est trop grande >> :

  1. zero << ?? >> dans le PDF. Ce controle NE SE LIT PAS dans la sortie de
     tectonic, qui n'emet aucun avertissement pour une reference non resolue :
     il a passe a vide pendant des semaines et a laisse cinq << au chapitre ?? >>
     dans un PDF publie. Il se compte sur le PDF ;
  2. zero annotation hors page, zero Overfull \\vbox, et les Overfull \\hbox
     comptes PAR EMPLACEMENT et non par ligne d'avertissement : les deux passes
     de tectonic ne donnent pas toujours la meme largeur au dernier chiffre, et
     le meme paragraphe ressort alors deux fois ;
  3. zero page tournee ;
  4. aucune figure plus haute que la limite (11 cm par defaut) ;
  5. aucune page dont une figure occupe presque tout et dont le texte tient en
     quelques lignes, c'est-a-dire aucune page d'image seule.

Usage :
  python controles_memoire.py [--pdf CHEMIN] [--log CHEMIN] [--hbox N] [--hauteur CM]

  --pdf      defaut main_ensae.pdf a cote de ce fichier.
  --log      journal de compilation (les deux flux rediriges). Sans lui, les
             controles 2 sont sautes et signales comme tels.
  --hbox     ligne de base des debordements horizontaux (defaut 2 pour
             main_ensae) : au-dessus, il y en a de nouveaux.
  --hauteur  hauteur imprimee maximale d'une figure, en centimetres (defaut 11).

Sortie : une ligne par controle, prefixee OK ou A VOIR, et le detail dessous.
Code de retour 1 si un controle echoue.
"""
import argparse
import io
import os
import re
import sys

import fitz

MEM = os.path.dirname(os.path.abspath(__file__))


def lignes_log(chemin, motif):
    if not chemin or not os.path.exists(chemin):
        return None
    txt = io.open(chemin, encoding="utf-8", errors="replace").read()
    return [l for l in txt.split("\n") if motif in l]


def principal():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=os.path.join(MEM, "main_ensae.pdf"))
    ap.add_argument("--log", default=None)
    # LIGNE DE BASE A 2 DEPUIS LE 22 SEPTEMBRE 2026, et plus 6 : le passage des
    # marges a 2,2 cm a absorbe les quatre debordements de la table des parametres.
    # Comparer a 6 ferait passer quatre debordements nouveaux pour normaux.
    ap.add_argument("--hbox", type=int, default=2)
    ap.add_argument("--hauteur", type=float, default=11.0)
    arg = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    doc = fitz.open(arg.pdf)
    defauts = 0

    def dire(ok, titre, detail=None):
        nonlocal defauts
        print("%-7s %s" % ("OK" if ok else "A VOIR", titre))
        if not ok:
            defauts += 1
        for d in (detail or []):
            print("          " + d)

    print("PDF : %s, %d pages\n" % (arg.pdf, doc.page_count))

    # 1. les ?? du PDF
    pages_pb = []
    for i in range(doc.page_count):
        n = len(re.findall(r"\?\?", doc[i].get_text()))
        if n:
            pages_pb.append("page %d : %d" % (i + 1, n))
    dire(not pages_pb, "aucun << ?? >> dans le PDF", pages_pb)

    # 2. le journal de compilation
    if not arg.log or not os.path.exists(arg.log):
        print("%-7s journal de compilation absent : debordements et annotations non controles"
              % "SAUTE")
    else:
        hbox = lignes_log(arg.log, "Overfull \\hbox")
        lieux = sorted({re.sub(r": Overfull.*", "", l) for l in hbox})
        dire(len(lieux) <= arg.hbox,
             "debordements horizontaux : %d emplacement(s), ligne de base %d" % (len(lieux), arg.hbox),
             lieux if len(lieux) > arg.hbox else [])
        vbox = lignes_log(arg.log, "Overfull \\vbox")
        dire(not vbox, "aucun Overfull \\vbox", vbox[:10])
        ann = lignes_log(arg.log, "Annotation out of page")
        dire(not ann, "aucune annotation hors page", ann[:10])
        err = [l for l in lignes_log(arg.log, "error") or [] if "Fontconfig" not in l]
        dire(not err, "aucune erreur de compilation", err[:10])

    # 3. pages tournees
    tournees = ["page %d" % (i + 1) for i in range(doc.page_count) if doc[i].rotation]
    dire(not tournees, "aucune page tournee", tournees)

    # 4 et 5. la place des figures sur la page
    trop_hautes, seules = [], []
    for i in range(doc.page_count):
        page = doc[i]
        images = page.get_images(full=True)
        if not images:
            continue
        aire_texte = len(page.get_text().strip())
        hauteur_totale = 0.0
        for img in images:
            for r in page.get_image_rects(img[0]):
                h = r.height / 72 * 2.54
                hauteur_totale += h
                if h > arg.hauteur:
                    trop_hautes.append("page %d : une figure de %.1f cm de haut" % (i + 1, h))
        # une page d'image seule : la figure domine et il n'y a presque pas de texte
        if hauteur_totale > 14 and aire_texte < 900:
            seules.append("page %d : %.1f cm de figure pour %d caracteres de texte"
                          % (i + 1, hauteur_totale, aire_texte))
    dire(not trop_hautes, "aucune figure de plus de %.0f cm de haut" % arg.hauteur, trop_hautes)
    dire(not seules, "aucune page d'image seule", seules)

    print("\n%d controle(s) a voir" % defauts)
    return 1 if defauts else 0


if __name__ == "__main__":
    sys.exit(principal())
