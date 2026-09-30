#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit de SOURCE, complement du harnais, et non son doublon.

CE QUE LE HARNAIS REPOND : << ce nombre existe-t-il quelque part dans le pool des
scripts que sa section cite ? >>. CE QU'IL NE REPOND PAS : << quel script le produit,
et ce script parle-t-il du meme objet ? >>. C'est la seconde question qui compte, et
c'est elle qui a fait tomber le paragraphe VERIS le 24 septembre 2026 : quatre nombres
publies depuis des mois, confirmes a 100 %, et imprimes par aucun script. Le 126 se
confirmait contre un montant en euros, le 0,12 contre un ecart de forme de Dirichlet.

POURQUOI UN AUDIT PUREMENT NUMERIQUE NE PEUT PAS TRANCHER, ET IL FAUT LE DIRE AVANT DE
LIRE SA SORTIE. Une source arrondie et une coincidence sont indistinguables par le
calcul : 0,118 arrondi a deux decimales donne 0,12, exactement comme le ferait une vraie
source. On ne peut donc que REDUIRE a un sous-ensemble relisable, et lire.

LA REDUCTION, SUR L'EMPREINTE DU CAS VERIS. Un nombre est signale quand les deux
conditions sont reunies :
  1. SUPPORT UNIQUE : une seule occurrence, dans une seule sortie. Un nombre appuye par
     plusieurs scripts est robuste ; une coincidence l'est rarement ;
  2. LA TOLERANCE FAIT LE TRAVAIL : la valeur trouvee n'est pas celle publiee, elle
     n'entre dans la fenetre que par l'arrondi. Mesure par le rapport ecart/tolerance,
     qui vaut 0 quand la source imprime exactement le nombre et approche 1 quand la
     correspondance a tout juste franchi la barre.
Le rapport imprime LA LIGNE d'ou vient chaque correspondance : c'est elle qu'on lit, et
c'est le seul juge. Une ligne qui parle d'autre chose que la phrase du memoire est une
fausse confirmation, et le taux du harnais ne la distingue pas d'une vraie.

L'ANGLE MORT, DECLARE. Une coincidence qui tombe EXACTEMENT sur la valeur publiee est
invisible a toute methode numerique, et elle existe : le 13 de VERIS se confirmait par un
13 exact d'un triangle chain-ladder sans rapport. Cet audit ne l'aurait pas signale. Il
reduit le risque, il ne l'annule pas.

ETAT AU 24 SEPTEMBRE 2026 : 1 912 nombres balayes, 68 signales, 68 relus a la main,
tous des arrondis de lecture legitimes. Aucun second cas VERIS dans le memoire.

Usage, depuis exploratory/memoire_cascade/ :
    python audit_sources.py ../../sorties_verif
"""

import os
import re
import sys

import verif_chiffres as vc

ICI = os.path.dirname(os.path.abspath(__file__))
CHAPITRES = os.path.join(ICI, "chapitres")


def charge_lignes(dossier):
    """Pool par script, en gardant la LIGNE d'origine de chaque valeur."""
    par_script = {}
    for f in sorted(os.listdir(dossier)):
        if not f.endswith(".txt"):
            continue
        entries = []
        with open(os.path.join(dossier, f), encoding="utf-8", errors="replace") as fh:
            for ligne in fh:
                for v in vc.nombres(ligne, latex=False):
                    entries.append((v, ligne.rstrip()))
        par_script[f[:-4]] = entries
    return par_script


def sections_de(brut):
    coupe = re.compile(r"\\(?:sub)?section\*?\{(.+?)\}")
    sections, cur, titre = [], [], "(preambule)"
    for l in brut.split("\n"):
        m = coupe.match(l.strip())
        if m:
            sections.append((titre, cur))
            titre, cur = m.group(1), []
        else:
            cur.append(l)
    sections.append((titre, cur))
    return sections


def main():
    dossier = sys.argv[1] if len(sys.argv) > 1 else None
    if not dossier or not os.path.isdir(dossier):
        print(__doc__)
        sys.exit(1)

    par_script = charge_lignes(dossier)
    suspects, total = [], 0

    for chap in sorted(os.listdir(CHAPITRES)):
        if not chap.endswith(".tex"):
            continue
        brut = open(os.path.join(CHAPITRES, chap), encoding="utf-8").read()
        for titre, corps in sections_de(brut):
            txt = "\n".join(corps)
            cites = sorted(
                {c for c in re.findall(r"\\texttt\{([0-9]+[a-z]?)(?:\\?_[^}]*)?\}", txt)
                 if c in par_script}
                | {c for lg in re.findall(r"%\s*SOURCES-SCRIPTS\s*:\s*(.+)", txt)
                   for c in re.findall(r"[0-9]+[a-z]?", lg) if c in par_script})
            if not cites:
                continue
            for v in [x for x in vc.extrait(txt) if vc.exemption(x) is None]:
                total += 1
                tol = max(0.006 * abs(v), 0.5 * 10 ** (-vc.decimales(v)))
                appuis = [(c, w, lg) for c in cites
                          for (w, lg) in par_script[c] if abs(v - w) <= tol]
                if len(appuis) != 1:
                    continue
                c, w, lg = appuis[0]
                if w == v:
                    continue
                suspects.append((abs(v - w) / tol, chap, titre, v, c, w, lg.strip()))

    suspects.sort(reverse=True)
    print(f"nombres sous controle balayes : {total}")
    print(f"signales (support unique ET tolerance determinante) : {len(suspects)}")
    print("classes du plus suspect au moins suspect ; LIRE LA LIGNE, c'est le seul juge.")
    print("=" * 100)
    for ratio, chap, titre, v, c, w, lg in suspects:
        print(f"\n[{ratio:.2f}] {chap}  ->  publie {v:g}   appuye par le seul {w:g} "
              f"du script {c}")
        print(f"       section : {titre[:76]}")
        print(f"       ligne   : {lg[:96]}")


if __name__ == "__main__":
    main()
