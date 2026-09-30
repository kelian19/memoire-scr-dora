#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_figures.py : quelle taille fait vraiment le texte d'une figure, imprimé ?

LE DÉFAUT QU'IL MESURE. Une figure tracée sur 16 pouces de large puis imprimée
sur 7,3 subit une réduction de 2,2. Une étiquette posée à 10 points dans le
script s'imprime alors à 4,5 points, sous la plus petite taille lisible. Le
défaut ne se voit pas en regardant le PNG, qui est net et grand : il se voit en
rapportant la largeur de tracé à la largeur d'impression.

LA MESURE EST PRISE SUR LE PNG, PAS SUR LE figsize DU SCRIPT. Les figures sont
enregistrées avec bbox_inches="tight", donc le fichier est recadré et sa largeur
réelle diffère du figsize demandé. Le PNG porte sa résolution dans ses
métadonnées : largeur physique = pixels / dpi. C'est la seule mesure juste.

SEUILS. Sous 6,5 points effectifs une étiquette est illisible à la lecture
normale ; entre 6,5 et 8 elle est petite. Le corps du texte est 11.

CE SCRIPT NE MODIFIE RIEN. La correction se fait dans le script qui produit la
figure, en RÉDUISANT la largeur de tracé, jamais en agrandissant l'image.

Lancement, depuis exploratory/memoire_cascade/ :
    ..\\..\\.venv\\Scripts\\python.exe audit_figures.py
    ..\\..\\.venv\\Scripts\\python.exe audit_figures.py --consignes
"""

import glob
import io
import os
import re
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
CHAP = os.path.join(ICI, "chapitres")
LAB = os.path.join(ICI, "..", "vasicek_lab")
FIG = os.path.join(LAB, "figures")

TEXTWIDTH_IN = (21.0 - 2 * 2.4) / 2.54     # A4 moins deux marges de 2,4 cm
LARGEURS = {"figover": 1.14, "figcle": 1.10}
SEUIL_ILLISIBLE = 6.5
SEUIL_PETIT = 8.0
CIBLE_PT = 8.5                              # ce qu'on veut obtenir apres correction


def figures_du_memoire():
    usages = {}
    for f in sorted(glob.glob(os.path.join(CHAP, "*.tex"))):
        nom = os.path.basename(f)[:2]
        texte = re.sub(r"(?<!\\)%.*", "",
                       io.open(f, encoding="utf-8", errors="replace").read())
        for macro, facteur in LARGEURS.items():
            for m in re.finditer(r"\\" + macro + r"\{([^}]+?\.png)\}", texte):
                png = os.path.basename(m.group(1))
                usages.setdefault(png, [TEXTWIDTH_IN * facteur, set(), macro])
                usages[png][1].add(nom)
        for m in re.finditer(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]+?\.png)\}", texte):
            opts, png = m.group(1) or "", os.path.basename(m.group(2))
            w = re.search(r"width\s*=\s*([\d.]+)\s*\\(?:text|line)width", opts)
            facteur = float(w.group(1)) if w else 1.0
            usages.setdefault(png, [TEXTWIDTH_IN * facteur, set(), "graphics"])
            usages[png][1].add(nom)
    return usages


def producteurs():
    """{nom_png: (script, corps de police que le script applique)}."""
    out = {}
    for py in glob.glob(os.path.join(LAB, "**", "*.py"), recursive=True):
        src = io.open(py, encoding="utf-8", errors="replace").read()
        m = re.search(r"appliquer\(\s*(?:taille\s*=\s*)?([\d.]+)\s*\)", src)
        if m:
            taille = float(m.group(1))
        else:
            m2 = re.search(r"[\"']font\.size[\"']\s*:\s*([\d.]+)", src)
            taille = float(m2.group(1)) if m2 else 10.0
        for m3 in re.finditer(r"[\"']([A-Za-z0-9_]+\.png)[\"']", src):
            out.setdefault(m3.group(1), (os.path.basename(py), taille))
    return out


def dimensions(png):
    chemin = os.path.join(FIG, png)
    if not os.path.exists(chemin):
        return None
    from PIL import Image
    im = Image.open(chemin)
    dpi = im.info.get("dpi", (100.0, 100.0))[0] or 100.0
    return im.size[0] / dpi, im.size[1] / dpi


def principal():
    consignes = "--consignes" in sys.argv
    usages = figures_du_memoire()
    prods = producteurs()

    lignes = []
    for png, (larg_impr, chapitres, mode) in usages.items():
        dims = dimensions(png)
        script, taille = prods.get(png, (None, 10.0))
        if dims is None:
            lignes.append((99.0, png, mode, script, None, None, larg_impr, taille,
                           sorted(chapitres)))
            continue
        lt, ht = dims
        # une figure plus haute que large se dimensionne sur la HAUTEUR de la
        # page, pas sur la largeur : la reduction effective est alors plus forte.
        effectif = taille * larg_impr / lt
        lignes.append((effectif, png, mode, script, lt, ht, larg_impr, taille,
                       sorted(chapitres)))
    lignes.sort(key=lambda t: t[0])

    if consignes:
        print("CONSIGNES DE RETAILLE, figure par figure")
        print("Dans le script indiqué, remplacer la largeur de figsize par la valeur")
        print(f"« largeur cible ». La hauteur se réduit du MÊME facteur, sans quoi les")
        print("proportions changent et les axes s'écrasent.\n")
        print(f"{'figure':<36}{'script':<34}{'figsize actuel':>16}{'largeur cible':>15}")
        print("-" * 101)
        for eff, png, mode, script, lt, ht, li, taille, ch in lignes:
            if lt is None or eff >= SEUIL_PETIT:
                continue
            cible = li * taille / CIBLE_PT
            fact = cible / lt
            print(f"{png:<36}{str(script):<34}{lt:>7.1f} x{ht:>6.1f}"
                  f"{cible:>9.1f} x{ht * fact:>5.1f}")
        return

    print(f"justification : {TEXTWIDTH_IN:.2f} pouces   seuils {SEUIL_ILLISIBLE} "
          f"et {SEUIL_PETIT} pt   cible après correction {CIBLE_PT} pt\n")
    print("=" * 106)
    print(f"{'figure':<36}{'effectif':>9}{'tracé (po)':>13}{'impr.':>7}{'réduc':>7}"
          f"{'corps':>7}  chap.  script")
    print("=" * 106)
    n_ill = n_pet = 0
    for eff, png, mode, script, lt, ht, li, taille, ch in lignes:
        if lt is None:
            print(f"{png:<36}{'absente du dossier figures/':>40}")
            continue
        marque = "  ILLISIBLE" if eff < SEUIL_ILLISIBLE else (
            "  petite" if eff < SEUIL_PETIT else "")
        n_ill += eff < SEUIL_ILLISIBLE
        n_pet += SEUIL_ILLISIBLE <= eff < SEUIL_PETIT
        print(f"{png:<36}{eff:>9.1f}{lt:>7.1f} x{ht:>5.1f}{li:>7.2f}{lt / li:>7.2f}"
              f"{taille:>7.0f}  {','.join(ch):<7}{script}{marque}")
    print("=" * 106)
    print(f"{len(lignes)} figures employées par le mémoire")
    print(f"  {n_ill} sous {SEUIL_ILLISIBLE} pt effectifs, illisibles")
    print(f"  {n_pet} entre {SEUIL_ILLISIBLE} et {SEUIL_PETIT} pt, petites")
    print("\nLancer avec --consignes pour la liste des retailles à appliquer.")


if __name__ == "__main__":
    principal()
