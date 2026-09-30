#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""LE MEME INCIDENT, DEUX ENTITES : rendu video de la cascade dirigee.

CE QUE LA VIDEO MONTRE, ET POURQUOI C'EST UN APPORT ET NON UNE FIGURE ANIMEE. Le memoire
publie que 62,31 % des sinistres touchent plus d'un pilier a l'etat non conforme contre
31,15 % a l'etat conforme. Dans un tableau, c'est une affirmation. A l'ecran, deux entites
recoivent LE MEME incident, tire avec LES MEMES aleas, et l'une l'eteint quand l'autre le
propage : le lecteur voit le mecanisme au lieu de lire son resultat. C'est la seule chose
qu'une figure fixe ne sait pas faire, et c'est pourquoi cette video existe.

CE QU'ELLE ANIME EST LA MARCHE, PAS LA CASCADE INDEPENDANTE, et la distinction n'est pas
academique : le script 98 a etabli que le moteur publie est la MARCHE AUTO-EVITANTE, non la
cascade par branchement de la definition 6.1. Depuis le pilier courant j, on continue avec
probabilite e_j = g * s_j / max_s, le successeur est tire proportionnellement a TRANS[j],
et un pilier DEJA TOUCHE eteint la cascade. C'est mot pour mot le noyau de
scr_engine.cascade_set_dist, et le script l'importe au lieu de le reecrire.

NOMBRES ALEATOIRES COMMUNS, et c'est ce qui rend la comparaison honnete. Pour chaque
incident on tire une fois pour toutes la suite des uniformes, et les DEUX panneaux la
consomment dans le meme ordre, au meme rang. Les deux entites ont donc exactement la meme
chance : ce qui les separe a l'ecran est le seul gain de propagation, 0,45 contre 0,90.

UN SEUL CANAL BOUGE ICI, ET LA VIDEO LE DIT. La frequence est tenue FIXE, sans quoi les deux
panneaux ne recevraient pas les memes incidents et la phrase << memes aleas >> serait fausse.
Le carton final rappelle donc que l'ecart de capital publie, 6 049 vers 20 188 M EUR, met en
jeu les QUATRE canaux et non la seule propagation, dont l'effet isole ne vaut que 1 633 M EUR.
Laisser croire que la propagation porte tout l'ecart serait le defaut le plus grave que cette
video puisse commettre.

DEUX CONTROLES, ET ARRET DUR SI L'UN CEDE.
  1. la loi EXACTE du moteur, agregee sur les amorces, doit redonner les valeurs publiees
     par le script 74 : 31,15 % et 62,31 % de sinistres multi-piliers, 1,380 et 1,931 piliers
     en moyenne. Si le moteur a derive, la video ne se rend pas ;
  2. les deux capitaux du carton final doivent redonner ceux du script 68.

AUCUNE COULEUR ET AUCUN SEUIL DANS CE FICHIER : les couleurs viennent de style_nexialog, les
parametres du modele de scr_engine, les grandeurs affichees de la loi exacte, et le rendu de
config.yaml. Voir l'en-tete de config.yaml.

Usage, depuis exploratory/video_cascade/ :

    python render_cascade.py --apercu     # apercu 480p, pour valider le rendu
    python render_cascade.py              # version finale
"""

import argparse
import os
import re
import sys

import matplotlib as mpl
mpl.use("Agg")
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.animation import FFMpegWriter, FuncAnimation

# matplotlib cherche un executable ffmpeg, pas le paquet : on le lui designe.
mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

ICI = os.path.dirname(os.path.abspath(__file__))
DEPOT = os.path.dirname(os.path.dirname(ICI))
LAB = os.path.join(DEPOT, "exploratory", "vasicek_lab")
for _p in (DEPOT, LAB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scr_engine as eng                                          # noqa: E402
import style_nexialog as st                                       # noqa: E402

mpl.rcParams["font.family"] = ["DejaVu Sans", "Segoe UI", "sans-serif"]

G_CONFORME, G_NON_CONFORME = 0.45, 0.90       # G_PROP du moteur, etats C et NC
WID = 88


def titre(t):
    print()
    print("=" * WID)
    print(t)
    print("=" * WID)


# =====================================================================================
# 1. Configuration
# =====================================================================================

with open(os.path.join(ICI, "config.yaml"), encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

ap = argparse.ArgumentParser()
ap.add_argument("--apercu", action="store_true",
                help="rendu basse resolution et court, pour valider avant la version finale")
ARGS = ap.parse_args()

titre("VIDEO DE LA CASCADE : LE MEME INCIDENT, DEUX ENTITES")

FPS = int(CFG["video"]["fps"])
if ARGS.apercu:
    LARG, HAUT = int(CFG["apercu"]["largeur"]), int(CFG["apercu"]["hauteur"])
    SORTIE = os.path.join(ICI, CFG["apercu"]["fichier"])
else:
    LARG, HAUT = int(CFG["video"]["largeur"]), int(CFG["video"]["hauteur"])
    SORTIE = os.path.join(ICI, CFG["video"]["fichier"])

SEQ = {k: float(v) for k, v in CFG["sequences"].items()}
N_INC = int(CFG["incidents"]["nombre"])
PAS_S = float(CFG["incidents"]["pas_par_seconde"])
GRAINE = int(CFG["incidents"]["graine"])

# echelle typographique : une figure de video se lit de loin, pas a la loupe
K = HAUT / 1080.0

print(f"  sortie              : {os.path.relpath(SORTIE, DEPOT)}")
print(f"  resolution          : {LARG} x {HAUT} a {FPS} images par seconde")
print(f"  sequences (s)       : " + ", ".join(f"{k} {v:g}" for k, v in SEQ.items()))


# =====================================================================================
# 2. Le moteur, sa loi exacte, et les controles
# =====================================================================================

PIL = list(eng.PIL)
TRANS, SROW, MAXS = eng.TRANS, eng._SROW, eng._MAXS
LAM = eng.LAMBDA
P_AMORCE = np.array([LAM[j] for j in PIL], dtype=float)
P_AMORCE /= P_AMORCE.sum()


def loi_exacte(g):
    """Part de sinistres multi-piliers et taille moyenne, agregees sur les amorces."""
    part, moyenne = 0.0, 0.0
    for pj, j in zip(P_AMORCE, PIL):
        d = eng.cascade_set_dist(j, g)
        part += pj * sum(p for s, p in d.items() if len(s) > 1)
        moyenne += pj * sum(len(s) * p for s, p in d.items())
    return part, moyenne


PART_C, MOY_C = loi_exacte(G_CONFORME)
PART_NC, MOY_NC = loi_exacte(G_NON_CONFORME)


def lire(chemin, motifs):
    with open(os.path.join(DEPOT, "sorties_verif", chemin), encoding="utf-8") as f:
        texte = f.read()
    return {cle: re.search(m, texte) for cle, m in motifs.items()}


# valeurs publiees par le script 74 (loi exacte du nombre de piliers touches)
m74 = lire("74.txt", {
    "parts": r"PLUS D'UN pilier\s*:\s*([\d.]+)\s*%.*?et\s*\n?\s*([\d.]+)\s*%",
    "moyennes": r"moyenne\s+([\d.]+)\s+([\d.]+)",
})
# capitaux publies par le script 68
m68 = lire("68.txt", {
    "conforme": r"\n\s+conforme\s+(\d+)\s+0\s+0",
    "non_conforme": r"\n\s+freq\+det\+prop\+accum\s+(\d+)\s+",
})

arrets = []
if not m74["parts"] or not m74["moyennes"]:
    arrets.append("74.txt illisible : la loi exacte publiee n'a pas ete retrouvee")
if not m68["conforme"] or not m68["non_conforme"]:
    arrets.append("68.txt illisible : les deux capitaux publies n'ont pas ete retrouves")

print()
print("  Controles :")
if not arrets:
    pub_part_c, pub_part_nc = (float(v) / 100 for v in m74["parts"].groups())
    pub_moy_c, pub_moy_nc = (float(v) for v in m74["moyennes"].groups())
    SCR_C = int(m68["conforme"].group(1))
    SCR_NC = int(m68["non_conforme"].group(1))
    tol_p = float(CFG["controle"]["tolerance_part"])
    tol_m = float(CFG["controle"]["tolerance_moyenne"])

    for nom, calc, pub, tol in (
            ("part multi, conforme    ", PART_C, pub_part_c, tol_p),
            ("part multi, non conforme", PART_NC, pub_part_nc, tol_p),
            ("moyenne, conforme       ", MOY_C, pub_moy_c, tol_m),
            ("moyenne, non conforme   ", MOY_NC, pub_moy_nc, tol_m)):
        ecart = abs(calc - pub)
        print(f"    {nom} : moteur {calc:.4f}  publie {pub:.4f}  ecart {ecart:.5f}")
        if ecart > tol:
            arrets.append(f"{nom.strip()} : le moteur donne {calc:.4f} pour {pub:.4f} publie")
    print(f"    capitaux publies         : {SCR_C} et {SCR_NC} M EUR")
    if SCR_C != 6049 or SCR_NC != 20188:
        arrets.append(f"les capitaux publies valent {SCR_C} et {SCR_NC}, non 6049 et 20188")

if arrets:
    print()
    print("  ARRET DUR. La video ne se rend pas :")
    for a in arrets:
        print("    - " + a)
    sys.exit(1)
print("    => controles passes, le rendu commence.")


# =====================================================================================
# 3. Les incidents, a nombres aleatoires COMMUNS
# =====================================================================================

def successeur(cur, u):
    """Successeur tire ~ TRANS[cur] normalise, par inversion. Meme loi que le moteur."""
    acc, s = 0.0, SROW[cur]
    for k, w in TRANS[cur].items():
        acc += w / s
        if u < acc:
            return k
    return list(TRANS[cur])[-1]


def marche(amorce, g, tirages):
    """Marche auto-evitante du moteur. Rend (piliers touches, aretes parcourues)."""
    visites, chemin, cur = [amorce], [], amorce
    for u_stop, u_suiv in tirages:
        if u_stop >= g * SROW[cur] / MAXS:          # arret pur
            break
        cible = successeur(cur, u_suiv)
        if cible in visites:                         # un pilier deja touche eteint
            break
        chemin.append((cur, cible))
        visites.append(cible)
        cur = cible
    return visites, chemin


rng = np.random.default_rng(GRAINE)
INCIDENTS = []
for _ in range(N_INC):
    amorce = int(rng.choice(PIL, p=P_AMORCE))
    tirages = [(float(rng.random()), float(rng.random())) for _ in range(len(PIL) - 1)]
    INCIDENTS.append({
        "amorce": amorce,
        "C": marche(amorce, G_CONFORME, tirages),
        "NC": marche(amorce, G_NON_CONFORME, tirages),
    })

# statistiques courantes apres chaque incident, pour les compteurs a l'ecran
CUMUL = {"C": [], "NC": []}
for etat in ("C", "NC"):
    n_multi, total = 0, 0
    for i, inc in enumerate(INCIDENTS, start=1):
        taille = len(inc[etat][0])
        n_multi += taille > 1
        total += taille
        CUMUL[etat].append((n_multi / i, total / i))

print()
print(f"  {N_INC} incidents tires, graine {GRAINE}.")
print(f"    amorces               : " + " ".join(f"P{i['amorce']}" for i in INCIDENTS))
print(f"    part multi observee   : conforme {CUMUL['C'][-1][0]:.3f}, "
      f"non conforme {CUMUL['NC'][-1][0]:.3f}")
print(f"    (loi exacte           : conforme {PART_C:.3f}, non conforme {PART_NC:.3f} ; "
      f"l'ecart est celui d'un petit echantillon, il est NORMAL et la video l'affiche)")


# =====================================================================================
# 4. Geometrie et primitives de dessin
# =====================================================================================

ANG = {j: np.pi / 2 + 2 * np.pi * i / len(PIL) for i, j in enumerate(PIL)}
POS = {j: (np.cos(a), np.sin(a)) for j, a in ANG.items()}
NOMS = {1: "P1 gouvernance", 2: "P2 incidents", 3: "P3 tests",
        4: "P4 tiers", 5: "P5 partage"}

COUL_C = st.CATEGORIEL[1]        # turquoise : l'entite conforme
COUL_NC = st.CATEGORIEL[0]       # rouge Nexialog : l'entite non conforme
PANNEAUX = [("C", "ENTITÉ CONFORME", G_CONFORME, COUL_C),
            ("NC", "ENTITÉ NON CONFORME", G_NON_CONFORME, COUL_NC)]

# LES DEUX TEINTES D'ACCENT VIENNENT DE LA CHARTE ET NE CHANGENT PAS. Seuls les NEUTRES
# basculent, parce qu'une video ne se lit pas comme une page : projetee ou vue sur un
# ecran, un fond sombre tient mieux le contraste qu'un fond de papier. La palette du
# memoire est donc conservee, son fond seul est choisi dans config.yaml.
_SOMBRE = str(CFG.get("theme", {}).get("fond", "clair")).lower() == "sombre"
if _SOMBRE:
    COUL_FOND, COUL_ENCRE = "#10161D", "#F2F4F6"
    COUL_ENCRE_2, COUL_ENCRE_3, COUL_GRILLE = "#AFBAC6", "#7C8794", "#2B3540"
else:
    COUL_FOND, COUL_ENCRE = st.FOND, st.ENCRE
    COUL_ENCRE_2, COUL_ENCRE_3, COUL_GRILLE = st.ENCRE_2, st.ENCRE_3, st.GRILLE
print(f"  fond                : {'sombre' if _SOMBRE else 'clair'} "
      f"(les teintes d'accent restent celles de la charte)")


def fig_neuve():
    f = plt.figure(figsize=(LARG / 100, HAUT / 100), dpi=100)
    f.patch.set_facecolor(COUL_FOND)
    return f


def texte(ax, x, y, s, taille, couleur=None, gras=False, italique=False, ha="center"):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="center",
            fontsize=taille * K, color=couleur or COUL_ENCRE,
            fontweight="bold" if gras else "normal",
            style="italic" if italique else "normal")


def lissage(f):
    """Amortissement aux deux bouts. Un trait qui part et s'arrete net donne l'impression
    d'un diaporama ; la meme trajectoire amortie donne celle d'un mouvement."""
    f = min(1.0, max(0.0, f))
    return f * f * (3.0 - 2.0 * f)


def dessine_panneau(ax, etat, libelle, g, couleur, touches, aretes, prog):
    """Un panneau : le graphe des cinq piliers, les piliers touches, le chemin parcouru.

    `prog` est un avancement REEL et non un compte d'aretes : l'arete de rang k est
    entierement tracee si prog >= k+1, et partiellement si k <= prog < k+1. C'est ce qui
    fait avancer la fleche au lieu de la faire apparaitre.
    """
    # Plafond releve : a 1,75 l'etiquette du pilier du haut, << gouvernance >>, venait
    # toucher le sous-titre du panneau. Le contenu descend dans le cadre au lieu de
    # deplacer les etiquettes une par une.
    ax.set_xlim(-1.80, 1.80); ax.set_ylim(-1.55, 2.10)
    ax.axis("off")
    ax.set_facecolor("none")

    # le graphe de fond : les vingt arcs de TRANS, epaisseur proportionnelle au poids
    for j in PIL:
        for k, w in TRANS[j].items():
            x1, y1 = POS[j]; x2, y2 = POS[k]
            ax.plot([x1, x2], [y1, y2], color=COUL_GRILLE, lw=0.8 + 2.2 * w / 0.8,
                    zorder=1, solid_capstyle="round", alpha=0.55)

    # le chemin parcouru par l'incident en cours, trace progressivement
    for rang, (j, k) in enumerate(aretes):
        f = lissage(prog - rang)
        if f <= 0.0:
            break
        x1, y1 = POS[j]; x2, y2 = POS[k]
        xa, ya = x1 + f * (x2 - x1), y1 + f * (y2 - y1)
        ax.annotate("", xy=(xa, ya), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=couleur, lw=4.0 * K,
                                    shrinkA=26 * K, shrinkB=26 * K if f >= 1.0 else 0,
                                    mutation_scale=26 * K), zorder=4)

    n_atteints = 1 + int(np.floor(prog + 1e-9))
    vus = touches[:max(1, min(n_atteints, len(touches)))]
    for j in PIL:
        x, y = POS[j]
        atteint = j in vus
        # DOUBLE ENCODAGE : l'etat d'un pilier ne tient pas a la seule couleur. Un pilier
        # touche est PLEIN et cercle d'un anneau, un pilier intact est CREUX. La video se
        # lit donc en niveaux de gris et par un daltonien.
        ax.scatter([x], [y], s=(1500 if atteint else 900) * K ** 2,
                   facecolors=couleur if atteint else COUL_FOND,
                   edgecolors=couleur if atteint else COUL_ENCRE_3,
                   linewidths=(3.5 if atteint else 1.6) * K, zorder=5)
        if atteint:
            ax.scatter([x], [y], s=2600 * K ** 2, facecolors="none",
                       edgecolors=couleur, linewidths=1.4 * K, alpha=0.55, zorder=4)
            # ECLAT a l'instant ou le pilier tombe : un anneau qui s'ouvre et s'efface.
            rang_j = vus.index(j)
            age = prog - (rang_j - 1) if rang_j > 0 else prog
            if 0.0 <= age <= 0.9:
                e = age / 0.9
                ax.scatter([x], [y], s=(2600 + 9000 * e) * K ** 2, facecolors="none",
                           edgecolors=couleur, linewidths=2.6 * (1 - e) * K,
                           alpha=0.75 * (1 - e), zorder=3)
        ax.text(x, y, f"P{j}", ha="center", va="center", zorder=6,
                fontsize=17 * K, color=COUL_FOND if atteint else COUL_ENCRE_2,
                fontweight="bold" if atteint else "normal")
        ax.text(x * 1.46, y * 1.46, NOMS[j].split(" ", 1)[1], ha="center", va="center",
                fontsize=14 * K, color=COUL_ENCRE_3, zorder=6)

    texte(ax, 0.5, 0.985, libelle, 22, couleur, gras=True)
    # LE NOMBRE RESTE HORS DES MATHS. Dans une expression mathtext, la virgule est une
    # ponctuation et recoit une espace : $g = 0,45$ s'imprimait << g = 0, 45 >>.
    texte(ax, 0.5, 0.945, "gain de propagation  $g$ = " + f"{g:.2f}".replace(".", ","),
          17, COUL_ENCRE_2)


def dessine_compteurs(ax, i_inc):
    """Les deux compteurs, sous les panneaux : part multi-piliers et taille moyenne.

    Les deux grandeurs sont EMPILEES et non posees cote a cote : a l'apercu, la seconde
    debordait du cadre a droite et se lisait << 2,50 piliers par sini >>.
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor("none")
    for col, (etat, _, _, couleur) in enumerate(PANNEAUX):
        part, moy = CUMUL[etat][min(i_inc, N_INC - 1)]
        x0 = 0.055 + 0.5 * col
        ax.text(x0, 0.86, "sinistres touchant plus d'un pilier", fontsize=15 * K,
                color=COUL_ENCRE_3, transform=ax.transAxes, va="center")
        ax.text(x0, 0.50, f"{100 * part:.0f}".replace(".", ",") + " %", fontsize=46 * K,
                color=couleur, fontweight="bold", transform=ax.transAxes, va="center")
        # barre de progression, second encodage de la meme grandeur
        ax.add_patch(plt.Rectangle((x0 + 0.145, 0.44), 0.24, 0.12, transform=ax.transAxes,
                                   facecolor=COUL_GRILLE, edgecolor="none", zorder=2))
        ax.add_patch(plt.Rectangle((x0 + 0.145, 0.44), 0.24 * part, 0.12,
                                   transform=ax.transAxes, facecolor=couleur,
                                   edgecolor="none", zorder=3))
        ax.text(x0, 0.13, f"{moy:.2f}".replace(".", ",") + " piliers touchés par sinistre",
                fontsize=16 * K, color=COUL_ENCRE_2, transform=ax.transAxes, va="center")


def carton(ax, lignes):
    """Un ecran de texte plein cadre."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor("none")
    for y, s, taille, couleur, gras in lignes:
        ax.text(0.5, y, s, transform=ax.transAxes, ha="center", va="center",
                fontsize=taille * K, color=couleur, fontweight="bold" if gras else "normal")


# =====================================================================================
# 5. Le decoupage temporel
# =====================================================================================

if ARGS.apercu:
    # l'apercu garde le titre, deux incidents et le carton final : il valide le RENDU,
    # pas le montage complet.
    plan = [("titre", 1.4), ("mise_en_place", 1.4), ("incidents", 5.2), ("final", 2.0)]
else:
    plan = [(k, SEQ[k]) for k in ("titre", "mise_en_place", "incidents",
                                  "stabilisation", "final")]

BORNES, t0 = [], 0.0
for nom, duree in plan:
    BORNES.append((nom, t0, t0 + duree))
    t0 += duree
DUREE = t0
N_FRAMES = int(round(DUREE * FPS))
INC_VISIBLES = N_INC if not ARGS.apercu else 4
DUREE_INC = next(d for n, d in plan if n == "incidents") / INC_VISIBLES

print()
print(f"  duree totale        : {DUREE:.1f} s, soit {N_FRAMES} images")
print(f"  incidents montres   : {INC_VISIBLES}, {DUREE_INC:.2f} s chacun")


def scene_a(t):
    for nom, a, b in BORNES:
        if a <= t < b:
            return nom, (t - a), (b - a)
    return BORNES[-1][0], BORNES[-1][2] - BORNES[-1][1], BORNES[-1][2] - BORNES[-1][1]


FIG = fig_neuve()


def dessine(frame):
    FIG.clf()
    FIG.patch.set_facecolor(COUL_FOND)
    t = frame / FPS
    nom, dt, duree = scene_a(t)

    if nom == "titre":
        ax = FIG.add_axes([0, 0, 1, 1])
        f = lissage(dt / max(0.5, duree * 0.45))
        carton(ax, [
            (0.62, "Le même incident, deux entités", 54, COUL_ENCRE, True),
            (0.50, "à gauche une entité conforme, à droite la même non conforme",
             26, COUL_ENCRE_2, False),
            (0.43, "mêmes incidents, mêmes tirages aléatoires", 26, COUL_ENCRE_2, False),
        ])
        ax.text(0.5, 0.30, "cascade dirigée entre les cinq piliers du règlement DORA",
                transform=ax.transAxes, ha="center", va="center", fontsize=20 * K,
                color=COUL_ENCRE_3, alpha=f)

    elif nom in ("mise_en_place", "incidents", "stabilisation"):
        axs = [FIG.add_axes([0.02, 0.28, 0.46, 0.66]), FIG.add_axes([0.52, 0.28, 0.46, 0.66])]
        axc = FIG.add_axes([0.02, 0.05, 0.96, 0.21])

        if nom == "mise_en_place":
            i_inc = 0
            for ax, (etat, lib, g, coul) in zip(axs, PANNEAUX):
                dessine_panneau(ax, etat, lib, g, coul, [], [], -1.0)
            dessine_compteurs(axc, 0)
            axt = FIG.add_axes([0, 0.93, 1, 0.07]); axt.axis("off")
            axt.set_facecolor("none")
            texte(axt, 0.5, 0.62,
                  "un incident part d'un pilier, puis se propage ou s'éteint",
                  22, COUL_ENCRE_2, italique=True)
            # La regle du moteur, ecrite. mathtext rend la formule sans aucune
            # installation LaTeX, ce qui compte ici : tectonic ne fournit ni latex
            # ni dvisvgm, donc aucune chaine TeX externe n'est disponible sur ce poste.
            texte(axt, 0.5, 0.18,
                  r"il continue avec probabilité  $e_j = g\,s_j / \max_k s_k$",
                  19, COUL_ENCRE_3)
        else:
            if nom == "incidents":
                i_inc = min(int(dt / DUREE_INC), INC_VISIBLES - 1)
                prog = (dt - i_inc * DUREE_INC) * PAS_S
            else:
                i_inc, prog = INC_VISIBLES - 1, float(len(PIL))
            inc = INCIDENTS[i_inc]
            for ax, (etat, lib, g, coul) in zip(axs, PANNEAUX):
                touches, aretes = inc[etat]
                dessine_panneau(ax, etat, lib, g, coul, touches, aretes, prog)
            dessine_compteurs(axc, i_inc)
            axt = FIG.add_axes([0, 0.93, 1, 0.07]); axt.axis("off")
            axt.set_facecolor("none")
            if nom == "incidents":
                texte(axt, 0.5, 0.5,
                      f"incident {i_inc + 1} sur {INC_VISIBLES}   ·   amorcé au pilier "
                      f"P{inc['amorce']}", 22, COUL_ENCRE_2)
            else:
                texte(axt, 0.5, 0.5,
                      f"après {INC_VISIBLES} incidents, les mêmes des deux côtés",
                      22, COUL_ENCRE_2, italique=True)

        # barre de progression du film
        axp = FIG.add_axes([0, 0, 1, 0.009]); axp.axis("off")
        axp.set_facecolor("none")
        axp.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axp.transAxes,
                                    facecolor=COUL_GRILLE))
        axp.add_patch(plt.Rectangle((0, 0), t / DUREE, 1, transform=axp.transAxes,
                                    facecolor=COUL_ENCRE_3))

    else:  # final
        ax = FIG.add_axes([0, 0, 1, 1])
        p = lissage(dt / max(0.8, duree * 0.35))
        carton(ax, [
            (0.87, "Sur la loi exacte du modèle, tous incidents confondus",
             26, COUL_ENCRE_2, False),
            (0.75, "sinistres touchant plus d'un pilier", 24, COUL_ENCRE_3, False),
        ])
        for x, val, coul, lib in ((0.30, PART_C, COUL_C, "entité conforme"),
                                  (0.70, PART_NC, COUL_NC, "entité non conforme")):
            ax.text(x, 0.61, f"{100 * val:.2f}".replace(".", ",") + " %",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=62 * K * (0.65 + 0.35 * p), color=coul, fontweight="bold")
            ax.text(x, 0.505, lib, transform=ax.transAxes, ha="center",
                    fontsize=22 * K, color=COUL_ENCRE_2)
        ax.text(0.5, 0.39,
                f"{MOY_C:.2f}".replace(".", ",") + " contre "
                + f"{MOY_NC:.2f}".replace(".", ",") + " piliers touchés par sinistre",
                transform=ax.transAxes, ha="center", fontsize=26 * K, color=COUL_ENCRE)
        # LA RESERVE QUI DOIT RESTER, ET ELLE N'EST PAS UNE PRECAUTION DE FORME. La
        # propagation seule ne pese que 1 633 M EUR des 14 139 d'ecart : laisser croire
        # que ce qu'on vient de voir porte tout l'ecart serait le pire defaut possible.
        ax.text(0.5, 0.235, "Seule la propagation change ici.", transform=ax.transAxes,
                ha="center", fontsize=22 * K, color=COUL_ENCRE_2, fontweight="bold")
        ax.text(0.5, 0.165,
                "L'écart de capital publié, "
                + f"{SCR_C:,}".replace(",", " ") + " → "
                + f"{SCR_NC:,}".replace(",", " ")
                + " M€, met en jeu les quatre canaux.",
                transform=ax.transAxes, ha="center", fontsize=20 * K, color=COUL_ENCRE_3)
    return []


# =====================================================================================
# 6. Rendu
# =====================================================================================

os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
anim = FuncAnimation(FIG, dessine, frames=N_FRAMES, interval=1000 / FPS, blit=False)
# FILTRE D'ECHELLE OBLIGATOIRE, ET CE N'EST PAS UNE PRECAUTION DE STYLE. H.264 refuse une
# dimension impaire, et matplotlib rend 853 pixels pour 854 demandes : 854/100 n'est pas
# exact en binaire, donc la taille en pouces ne retombe pas sur un entier de pixels. Le
# premier rendu s'est arrete sur << width not divisible by 2 (853x480) >>. Le filtre force
# les deux dimensions au pair inferieur, quelle que soit la resolution demandee.
writer = FFMpegWriter(fps=FPS, bitrate=int(CFG["video"]["debit_kbps"]),
                      codec="libx264",
                      extra_args=["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                                  "-pix_fmt", "yuv420p", "-profile:v", "high"])
anim.save(SORTIE, writer=writer, dpi=100,
          savefig_kwargs={"facecolor": st.FOND})
plt.close(FIG)

taille = os.path.getsize(SORTIE) / 1e6
print()
print(f"  video ecrite : {os.path.relpath(SORTIE, DEPOT)}  ({taille:.1f} Mo)")

# images de controle
if not ARGS.apercu:
    dossier = os.path.join(ICI, CFG["frames_preview"]["dossier"])
    os.makedirs(dossier, exist_ok=True)
    FIG = fig_neuve()
    for inst in CFG["frames_preview"]["instants"]:
        if inst >= DUREE:
            continue
        dessine(int(round(float(inst) * FPS)))
        chemin = os.path.join(dossier, f"t{float(inst):05.1f}s.png".replace(".", "_", 1))
        FIG.savefig(chemin, dpi=100, facecolor=st.FOND)
        print(f"  image de controle : {os.path.relpath(chemin, DEPOT)}")
    plt.close(FIG)

print()
print("=" * WID)
print("FIN DU RENDU")
print("=" * WID)
