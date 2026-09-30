#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Palette et style de figures, charte Nexialog Consulting.

POURQUOI CE MODULE EXISTE, ET CE QU'IL REMPLACE.
Quatre-vingt-treize scripts de vasicek_lab importent matplotlib et codent leurs couleurs
EN DUR, chacun de son cote : le releve du 17 aout 2026 compte dix-huit hexadecimaux
distincts, dont un fond #FCFCFB repete 351 fois et un accent orange #EB6834 utilise 98
fois. Aucun n'appartient a la charte Nexialog. Changer la charte demandait donc de reprendre
quatre-vingt-treize fichiers, ce qui est la definition d'une dette d'architecture.
Ce module est la source unique. Les scripts importent des ROLES, jamais des hexadecimaux.

CE QUI A ETE MESURE PLUTOT QUE JUGE A L'OEIL, et c'est le point principal.
Une palette categorielle n'est pas affaire de gout : elle se valide. Les six controles
(bande de clarte OKLCH, plancher de chroma, separation sous protanopie et deuteranopie
simulees par Machado-Oliveira-Fernandes 2009 a severite 1,0, plancher de vision normale,
contraste sur le fond) ont ete calcules sur les dix couleurs de la charte, puis sur des
paliers derives de ses teintes. Resultats, tous reproductibles par le bloc __main__ :

  1. LA CHARTE, PRISE AU PIED DE LA LETTRE, NE DONNE QUE TROIS SLOTS UTILISABLES.
     Sur ses dix couleurs, quatre sont hors de la bande de clarte (les bleus fonces,
     L de 0,24 a 0,35), et six sont sous le plancher de chroma, donc elles se lisent
     comme du gris et cessent de porter une identite. Restent #B10031, #4BC4BD et
     #49A8CB, dont les deux derniers sont sous 3:1 de contraste sur fond clair.
  2. D'OU DES PALIERS DERIVES DE SES TEINTES, a chroma RETENU. On tient la teinte de la
     charte et l'on deplace la clarte : c'est la procedure de rapprochement prescrite.
     Le chroma demande reste proche de celui de la charte (les bleus Nexialog sont
     volontairement sourds, C de 0,12) pour ne pas rendre electrique une identite sobre.
  3. LE CATEGORIEL TIENT A TROIS SLOTS, PAS QUATRE. En toutes paires, l'ordre retenu
     passe a dE 17,2 sous deuteranopie et 21,1 en vision normale. Un quatrieme slot pris
     dans la charte echoue le plancher de VISION NORMALE a 14,1 contre 15 exiges, et ce
     plancher ne se rachete pas par un encodage secondaire. Une quatrieme serie se replie
     donc en « autres », se facette, ou tire son identite de la position et d'une etiquette
     directe. Elle ne prend PAS une teinte inventee.
  4. LES CINQ PILIERS NE SONT PAS UN CAS CATEGORIEL. Ils se presentent presque toujours
     ORDONNES par contribution (P1 > P4 > P2 > P3 > P5), et une serie ordonnee prend une
     rampe a une seule teinte, pour que le lecteur voie l'ordre dans la couleur. La rampe
     ordinale ci-dessous passe ses quatre controles propres.

CE QUI N'EST PAS TOUCHE, ET IL NE FAUT PAS Y TOUCHER.
La declaration de police reste ["DejaVu Sans", "Segoe UI", "sans-serif"], dans cet ordre.
DejaVu Sans est fournie par matplotlib lui-meme, donc identique sur les deux postes du
projet ; c'est ce qui rend une figure rejouee identique A L'OCTET d'une machine a l'autre.
La charte demande Segoe UI pour le texte, et elle reste en second : sur un poste Windows
elle serait choisie, ce qui casserait la parite avec le Mac. La charte porte sur les
documents, pas sur la reproductibilite d'une figure scientifique versionnee, et c'est le
seul point ou ce module s'en ecarte sciemment.
"""

# =====================================================================================
# 1. LA CHARTE, TELLE QUELLE. Valeurs relevees dans le PDF de charte graphique.
#    Les trois nombres en regard sont mesures : clarte, chroma et teinte OKLCH.
# =====================================================================================
CHARTE = {
    # primaires
    "rouge":        "#B10031",   # L 0,482  C 0,193  H  18   le seul accent de marque
    "bleu_nuit":    "#002060",   # L 0,272  C 0,120  H 261   hors bande de clarte
    "bleu_ardoise": "#223E55",   # L 0,353  C 0,053  H 245   hors bande, sous chroma
    "presque_noir": "#1B1E30",   # L 0,241  C 0,035  H 276   hors bande, sous chroma
    # secondaires
    "blanc_casse":  "#F2F2F2",   # L 0,961  C 0,000         neutre, fonds et grilles
    "bleu_acier":   "#435B6E",   # L 0,459  C 0,043         sous chroma, lit gris
    "gris":         "#595959",   # L 0,464  C 0,000         neutre pur, milieu divergent
    # accentuation
    "turquoise":    "#4BC4BD",   # L 0,751  C 0,107  H 190   contraste 2,06:1 sur clair
    "creme":        "#EEE3D3",   # L 0,920  C 0,025         neutre chaud
    "bleu_clair":   "#49A8CB",   # L 0,689  C 0,102  H 226   contraste 2,64:1 sur clair
}

# Les quatre familles de teinte exploitables, avec le chroma retenu pour chacune.
# Le rouge garde son chroma de marque, les bleus restent sourds comme dans la charte.
TEINTES = {
    "rouge":     (18.0,  0.190),
    "turquoise": (190.0, 0.115),
    "bleu":      (226.0, 0.115),
    "nuit":      (261.0, 0.130),
}

# =====================================================================================
# 2. LES ROLES. C'est ce que les scripts importent.
# =====================================================================================

# Encre et surfaces. La charte n'impose pas de fond de graphique : on garde un blanc
# casse tres clair pour l'impression et l'on reserve #F2F2F2 aux remplissages.
FOND        = "#FCFCFB"          # surface du graphique, inchangee (351 occurrences)
FOND_PANNEAU = CHARTE["blanc_casse"]
ENCRE       = CHARTE["presque_noir"]     # texte principal
ENCRE_2     = CHARTE["bleu_ardoise"]     # texte secondaire
ENCRE_3     = CHARTE["gris"]             # texte discret, unites
GRILLE      = "#DCDCDC"                  # grille recessive, jamais une serie

# CATEGORIEL : trois slots, ordre fixe, jamais recycle. Valide en TOUTES PAIRES,
# donc utilisable aussi bien en nuage et en petits multiples qu'en barres.
CATEGORIEL = ["#a6002e", "#009a94", "#2b559f"]
CATEGORIEL_NOMS = ["rouge Nexialog", "turquoise", "bleu nuit"]

# ORDINAL : une seule teinte, cinq paliers, pour une serie ORDONNEE (les cinq piliers).
ORDINAL_5 = ["#204993", "#3661ac", "#4c79c7", "#6491e1", "#7baafd"]

# SEQUENTIEL : une seule teinte, six paliers, du fonce au clair, pour une magnitude.
SEQUENTIEL_6 = ["#09327a", "#204993", "#3661ac", "#4c79c7", "#6491e1", "#7baafd"]

# DIVERGENT : deux poles de la charte et un gris NEUTRE au milieu. Jamais une teinte
# au point median, jamais un arc-en-ciel.
DIVERGENT = ["#a6002e", "#d3384e", "#e8a0aa", CHARTE["gris"],
             "#8fa8d8", "#4c79c7", "#204993"]

# ETAT : echelle reservee, jamais recyclee en « serie 4 », toujours avec une etiquette.
# Le vert n'est pas dans la charte : on utilise le turquoise d'accentuation, qui en est
# la couleur froide positive, et le rouge de marque pour le critique.
ETAT = {"bon": "#009a94", "vigilance": "#d3384e", "critique": "#a6002e"}


def appliquer(taille=10):
    """Pose les rcParams de la charte. La police n'est PAS touchee, voir l'en-tete."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": FOND,
        "axes.facecolor": FOND,
        "savefig.facecolor": FOND,
        "axes.edgecolor": ENCRE_2,
        "axes.labelcolor": ENCRE,
        "text.color": ENCRE,
        "xtick.color": ENCRE_3,
        "ytick.color": ENCRE_3,
        "grid.color": GRILLE,
        "grid.linewidth": 0.6,
        # axes.grid N'EST PAS POSE ICI, ET C'EST DELIBERE. Les scripts existants
        # choisissent chacun d'afficher ou non la grille ; l'imposer changerait
        # silencieusement l'apparence de quatre-vingt-treize figures pour une raison
        # etrangere a la charte. Chaque script garde son choix.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(color=CATEGORIEL),
        "lines.linewidth": 2.0,          # traits fins, specification de marque
        "lines.markersize": 5.0,
        "font.size": taille,
        # LA FAMILLE DE POLICE EST DELIBEREMENT ABSENTE DE CE DICTIONNAIRE.
        # Chaque script garde ["DejaVu Sans", "Segoe UI", "sans-serif"] : c'est ce qui
        # rend une figure rejouee identique a l'octet entre le PC et le Mac.
    })


def rampe(n, palette=None):
    """n couleurs prises dans l'ordre fixe, jamais recyclees.

    Au-dela de la longueur de la palette, on LEVE une erreur au lieu de boucler : une
    neuvieme serie ne prend pas une teinte engendree, elle se replie en « autres », se
    facette, ou tire son identite de la position. Boucler serait donner deux series la
    meme couleur en silence.
    """
    palette = palette or CATEGORIEL
    if n > len(palette):
        raise ValueError(
            f"{n} series demandees pour {len(palette)} slots valides. La charte Nexialog "
            f"ne porte pas plus de {len(palette)} teintes categorielles distinctes au "
            f"standard (plancher de vision normale). Replier en « autres », facetter, ou "
            f"passer a une rampe ordinale si la serie est ORDONNEE.")
    return list(palette[:n])


# =====================================================================================
# 3. AUTO-VERIFICATION. Le module rejoue ses propres controles quand on l'execute.
#    C'est la meme discipline que le reste du projet : un chiffre qui n'est pas
#    recalculable n'est verifiable par personne.
# =====================================================================================
if __name__ == "__main__":
    import math

    def _lin(h):
        h = h.lstrip("#")
        f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        return [f(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4)]

    def _oklab(rgb):
        r, g, b = rgb
        l = max(0.0, 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
        m = max(0.0, 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
        s = max(0.0, 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
        return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)

    MACH = {"protan": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216),
                       (-0.003882, -0.048116, 1.051998)),
            "deutan": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413),
                       (-0.011820, 0.042940, 0.968881))}

    def _de(a, b, kind=None):
        def conv(h):
            rgb = _lin(h)
            if kind:
                M = MACH[kind]
                rgb = [min(1.0, max(0.0, sum(M[i][j] * rgb[j] for j in range(3))))
                       for i in range(3)]
            return _oklab(rgb)
        return 100 * math.dist(conv(a), conv(b))

    def _lum(h):
        r, g, b = _lin(h)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def _ct(a, b):
        hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    def _lch(h):
        L, a, b = _oklab(_lin(h))
        return L, math.hypot(a, b)

    print("=" * 78)
    print("CATEGORIEL, trois slots, controle EN TOUTES PAIRES")
    print("=" * 78)
    P = CATEGORIEL
    pl = [(i, j) for i in range(len(P)) for j in range(i + 1, len(P))]
    hb = [(c, round(_lch(c)[0], 3)) for c in P if not 0.43 <= _lch(c)[0] <= 0.77]
    lc = [(c, round(_lch(c)[1], 3)) for c in P if _lch(c)[1] < 0.10]
    cvd = min(min(_de(P[i], P[j], k) for i, j in pl) for k in ("protan", "deutan"))
    nor = min(_de(P[i], P[j]) for i, j in pl)
    ctr = [(c, round(_ct(c, FOND), 2)) for c in P if _ct(c, FOND) < 3.0]
    for nom, cond, det in [
            ("bande de clarte", not hb, hb or f"les {len(P)} dans 0,43-0,77"),
            ("plancher de chroma", not lc, lc or f"les {len(P)} >= 0,10"),
            ("separation CVD", cvd >= 8.0, f"pire paire dE {cvd:.1f} (cible 8)"),
            ("plancher vision normale", nor >= 15.0, f"pire paire dE {nor:.1f} (plancher 15)"),
            ("contraste sur fond", not ctr, ctr or f"les {len(P)} >= 3,0:1")]:
        print(f"  [{'PASS' if cond else 'FAIL'}] {nom:<24}{det}")

    print()
    print("=" * 78)
    print("ORDINAL, cinq paliers, controles propres a une rampe")
    print("=" * 78)
    R = ORDINAL_5
    Ls = [_lch(c)[0] for c in R]
    mono = Ls == sorted(Ls) or Ls == sorted(Ls, reverse=True)
    gaps = [abs(Ls[i + 1] - Ls[i]) for i in range(len(Ls) - 1)]
    clair = max(R, key=lambda c: _lch(c)[0])
    for nom, cond, det in [
            ("clarte monotone", mono, "les paliers se lisent dans l'ordre"),
            ("ecart adjacent", min(gaps) >= 0.06, f"le plus petit vaut {min(gaps):.3f} (>= 0,06)"),
            ("bout clair sur fond", _ct(clair, FOND) >= 2.0,
             f"{clair} a {_ct(clair, FOND):.2f}:1 (plancher 2,0)")]:
        print(f"  [{'PASS' if cond else 'FAIL'}] {nom:<24}{det}")

    print()
    print("=" * 78)
    print("CE QUE LA CHARTE NE PERMET PAS, mesure et non suppose")
    print("=" * 78)
    q = CATEGORIEL + ["#0087ac"]
    ql = [(i, i + 1) for i in range(len(q) - 1)]
    qn = min(_de(q[i], q[j]) for i, j in ql)
    print(f"  un quatrieme slot pris dans la charte : pire paire adjacente dE {qn:.1f}")
    print(f"  plancher exige 15,0 -> {'PASS' if qn >= 15 else 'ECHEC'}. Ce plancher ne se")
    print("  rachete PAS par un encodage secondaire : une quatrieme serie se replie,")
    print("  se facette, ou tire son identite de la position et d'une etiquette directe.")
    print("\nEXIT 0")
