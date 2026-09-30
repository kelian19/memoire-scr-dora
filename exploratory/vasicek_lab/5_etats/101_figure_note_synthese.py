#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
101 : la figure des notes de synthese, lisible par un lecteur non specialiste.

POURQUOI CE SCRIPT EXISTE. Les deux notes de synthese reprenaient la figure S24 du script 68 : trois
panneaux, dont une cascade par ordres de Mobius, imprimes sur 16 cm de large et 5 de haut, avec un
texte d'environ 5 points. C'est la figure du specialiste, pas celle d'une note. Celle-ci ne dit
qu'une chose, le message de gestion : pour chacun des quatre canaux, ce que rapporte sa correction
a une entite deja conforme ailleurs (effet isole) et a une entite defaillante partout (effet de
fermeture). Deux series, quatre barres chacune, les valeurs ecrites, et la version anglaise pour
l'executive summary.

CE QUE LE SCRIPT LIT. Les effets isoles et de fermeture de sorties_verif/68.txt, section 2, sans rien
simuler ; il s'arrete si leurs sommes ne redonnent pas 9 138 et 19 141 M.

Sortie : figures N1_canaux_note.png et N1_canaux_note_en.png.
"""

import os
import re
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import style_nexialog as st                                     # noqa: E402

st.appliquer(taille=10)
mpl.rcParams["font.family"] = ["DejaVu Sans", "Segoe UI", "sans-serif"]

CANAUX = ["frequence", "detection", "propagation", "accumulation"]
iso, ferm = {}, {}
with open(os.path.join(DEPOT, "sorties_verif", "68.txt"), encoding="utf-8") as f:
    for ligne in f:
        m = re.match(r"\s+(frequence|detection|propagation|accumulation)\s+(\d+) \+-\d+\s+(\d+) \+-\d+",
                     ligne)
        if m:
            iso[m.group(1)], ferm[m.group(1)] = int(m.group(2)), int(m.group(3))
assert len(iso) == 4, "section 2 de 68.txt illisible"
# les valeurs imprimees sont arrondies au million : leurs sommes peuvent differer d'un ou deux
assert abs(sum(iso.values()) - 9138) <= 2 and abs(sum(ferm.values()) - 19141) <= 2, "les effets publies ont derive"

LIBELLES = {
    "fr": dict(canaux=["Fréquence des incidents", "Détection", "Propagation entre piliers",
                       "Accumulation chez un prestataire"],
               iso="Corriger ce seul canal, l'entité étant conforme ailleurs",
               ferm="Corriger ce seul canal, l'entité étant défaillante partout",
               x="capital libéré (M€)", fichier="N1_canaux_note.png"),
    "en": dict(canaux=["Incident frequency", "Detection", "Propagation across pillars",
                       "Accumulation at a provider"],
               iso="Fixing this channel only, the firm compliant elsewhere",
               ferm="Fixing this channel only, the firm failing everywhere",
               x="capital released (EUR m)", fichier="N1_canaux_note_en.png"),
}

for langue, lib in LIBELLES.items():
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    y = range(len(CANAUX))
    h = 0.36
    for k, c in enumerate(CANAUX):
        ax.barh(k - h / 2, iso[c], height=h, color=st.CATEGORIEL[2],
                label=lib["iso"] if k == 0 else None)
        ax.barh(k + h / 2, ferm[c], height=h, color=st.CATEGORIEL[0],
                label=lib["ferm"] if k == 0 else None)
        sep = " " if langue == "fr" else ","
        ax.text(iso[c] + 120, k - h / 2, f"{iso[c]:,}".replace(",", sep), va="center", fontsize=8.5,
                color=st.ENCRE)
        ax.text(ferm[c] + 120, k + h / 2, f"{ferm[c]:,}".replace(",", sep), va="center", fontsize=8.5,
                color=st.ENCRE)
    ax.set_yticks(list(y))
    ax.set_yticklabels(lib["canaux"], fontsize=9.5, color=st.ENCRE)
    ax.invert_yaxis()
    ax.set_xlim(0, max(ferm.values()) * 1.18)
    ax.set_xlabel(lib["x"], fontsize=9, color=st.ENCRE_2)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    sep_axe = " " if langue == "fr" else ","
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", sep_axe)))
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncol=1, fontsize=8.5, frameon=False,
              handlelength=1.2)
    fig.tight_layout()
    chemin = os.path.join(HERE, "figures", lib["fichier"])
    fig.savefig(chemin)
    plt.close(fig)
    print(f"figure ecrite : {lib['fichier']}")

print("\neffets isoles  : " + " ; ".join(f"{c} {iso[c]}" for c in CANAUX))
print("effets de fermeture : " + " ; ".join(f"{c} {ferm[c]}" for c in CANAUX))
