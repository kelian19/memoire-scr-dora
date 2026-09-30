#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10 : Le livrable. Le SCR n'est pas un nombre, c'est une SURFACE SCR(g, xi).

Le cas de base de 09 (VaR 99,5 %) est UN point. Ici on balaye les deux parametres
structurels qui commandent le capital :
  - g  : le gain de propagation de la cascade (etendue des incidents) ;
  - xi : l'indice de queue commun de la severite (poids des pertes extremes).
et on decompose la sensibilite (g, xi, charge systemique a) pour dire lequel deplace
le plus le SCR.

Lecture d'HONNETETE superposee (colonne vertebrale du memoire). Les deux axes n'ont
pas le meme statut epistemique :
  - g  gouverne la STRUCTURE de propagation : calibrable EN PRINCIPE avec un registre
       horodate (cf. resultat d'identifiabilite, note + script 05) ;
  - xi gouverne le REGIME DE QUEUE : importe de donnees externes (SAS OpRisk, ~0,9),
       NON calibrable sur l'entite, et la VaR y est hypersensible au-dela de 0,9.
Se deplacer selon g, c'est bouger dans ce que la donnee pourrait pincer ; se deplacer
selon xi, c'est bouger dans de l'hypothese pure. La surface rend cette asymetrie
visible.

PORTEE. Unites normalisees. Resolution Monte-Carlo reduite par point de grille
(NY_SURF) pour tenir le temps de calcul : la surface porte les TENDANCES et les ordres
de grandeur, pas le troisieme chiffre significatif de chaque cellule. Aucun point n'est
tronque en silence : la grille balayee est affichee.

Sortie : diagnostics (surface, tornado) + figure S4_sensibilite.png.
"""

import os
import sys

# AMORCE DE CHEMIN. Ce script importait les modules partages du lab en supposant que le
# repertoire courant etait la racine du lab : il echouait donc depuis la racine du depot,
# ou le harnais et les autres scripts sont lances. Meme amorce que les scripts 29 et 32.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scr_engine as eng
import frequence_model as freq   # scr_engine a deja insere le dossier dans sys.path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(20260715)
W = 74

NY_SURF = 120_000   # annees par point de grille (surface) : tendances, pas 3e decimale
NY_TORN = 150_000   # annees par extremite (tornado)

# cas de base (aligne sur 09)
G0, XI0, A0 = eng.G_BASE, 0.70, freq.A_LOAD


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def scr_point(g, xi, a, ny, rng):
    """VaR 99,5 % de la perte annuelle pour (g, xi, a)."""
    return eng.var(eng.simulate_annual_losses(ny, rng, g=g, xi=xi, a=a), 0.995)


# ============================================================ surface SCR(g, xi)
G_GRID = np.array([0.0, 0.3, 0.5, 0.7, 0.9, 0.95])
XI_GRID = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])

titre("Surface SCR(g, xi) : VaR 99,5 %, charge a fixee a la base")
print(f"  a = {A0} ; {NY_SURF:,} annees par cellule ; unites normalisees.")
print("  grille g  : " + ", ".join(f"{g:.2f}" for g in G_GRID))
print("  grille xi : " + ", ".join(f"{xi:.2f}" for xi in XI_GRID))
SURF = np.empty((len(XI_GRID), len(G_GRID)))
for iy, xi in enumerate(XI_GRID):
    for ix, g in enumerate(G_GRID):
        SURF[iy, ix] = scr_point(g, xi, A0, NY_SURF, RNG)

print("\n  SCR (lignes = xi croissant, colonnes = g croissant) :")
head = "   xi\\g " + "".join(f"{g:>9.2f}" for g in G_GRID)
print(head)
for iy, xi in enumerate(XI_GRID):
    print(f"  {xi:>5.2f} " + "".join(f"{SURF[iy, ix]:>9.0f}" for ix in range(len(G_GRID))))

# effet relatif de chaque axe, depuis le coin bas-gauche de la grille
base_cell = SURF[XI_GRID.tolist().index(0.70), G_GRID.tolist().index(0.90)]
print(f"\n  point de base (g=0,90 ; xi=0,70) : SCR = {base_cell:.0f}")
print(f"  a xi fixe (0,70), g de 0 a 0,95 : x{SURF[2, -1] / SURF[2, 0]:.2f}")
print(f"  a g fixe (0,90), xi de 0,5 a 0,95 : x{SURF[-1, 4] / SURF[0, 4]:.2f}")
print("  => la queue (xi) deplace le SCR bien plus que la propagation (g).")

# ============================================================ tornado de sensibilite
titre("Tornado : quel parametre deplace le plus le SCR ? (autres fixes a la base)")
base_scr = scr_point(G0, XI0, A0, NY_TORN, RNG)
factors = [
    ("g  (propagation)", "g", 0.50, 0.95),
    ("xi (queue)",       "xi", 0.50, 0.90),   # borne a 0,9 : au-dela, zone instable
    ("a  (systemique)",  "a", 0.30, 0.90),
]
print(f"  SCR de base (g={G0}, xi={XI0}, a={A0}) = {base_scr:.0f}")
print(f"  {'parametre':<20}{'bas':>10}{'SCR bas':>11}{'haut':>8}{'SCR haut':>11}{'amplitude':>11}")
tor = []
for name, key, lo, hi in factors:
    kw_lo = {"g": G0, "xi": XI0, "a": A0}
    kw_hi = dict(kw_lo)
    kw_lo[key], kw_hi[key] = lo, hi
    s_lo = scr_point(kw_lo["g"], kw_lo["xi"], kw_lo["a"], NY_TORN, RNG)
    s_hi = scr_point(kw_hi["g"], kw_hi["xi"], kw_hi["a"], NY_TORN, RNG)
    amp = abs(s_hi - s_lo)
    tor.append((name, lo, s_lo, hi, s_hi, amp))
    print(f"  {name:<20}{lo:>10.2f}{s_lo:>11.0f}{hi:>8.2f}{s_hi:>11.0f}{amp:>11.0f}")
tor.sort(key=lambda r: r[5])   # amplitude croissante pour l'affichage tornado

# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2 = "#1b1e30", "#223e55"
ACCENT, BLUE = "#a6002e", "#3661ac"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.6, 5.4),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# panneau A : carte de chaleur SCR(g, xi), couleur en echelle log
im = axA.imshow(SURF, origin="lower", aspect="auto", cmap="YlOrRd",
                norm=mpl.colors.LogNorm(vmin=SURF.min(), vmax=SURF.max()))
axA.set_xticks(range(len(G_GRID)))
axA.set_xticklabels([f"{g:.2f}" for g in G_GRID])
axA.set_yticks(range(len(XI_GRID)))
axA.set_yticklabels([f"{xi:.2f}" for xi in XI_GRID])
for iy in range(len(XI_GRID)):
    for ix in range(len(G_GRID)):
        axA.text(ix, iy, f"{SURF[iy, ix]:.0f}", ha="center", va="center",
                 fontsize=7.6, color="#222")
# bande instable xi > 0,9 (marquee par la ligne ; note explicite en bas de figure)
axA.axhline(len(XI_GRID) - 1.5, color=ACCENT, lw=1.2, ls="--")
axA.set_xlabel("g : propagation, calibrable si horodatage (identifiable)",
               fontsize=9, color=INK2)
axA.set_ylabel("xi : queue, importee du SAS (non identifiable ici)",
               fontsize=9, color=INK2)
axA.set_title("(A)  Le SCR est une surface, pas un nombre",
              fontsize=10.5, color=INK, pad=6)
cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.03)
cb.set_label("SCR = VaR 99,5 % (unites normalisees, echelle log)", fontsize=8.2)

# panneau B : tornado
ypos = np.arange(len(tor))
for i, (name, lo, s_lo, hi, s_hi, amp) in enumerate(tor):
    left, right = min(s_lo, s_hi), max(s_lo, s_hi)
    axB.barh(i, right - left, left=left, color=BLUE, alpha=0.75, height=0.55)
    axB.text(left, i, f"{lo:g} ", ha="right", va="center", fontsize=8, color=INK2)
    axB.text(right, i, f" {hi:g}", ha="left", va="center", fontsize=8, color=INK2)
axB.axvline(base_scr, color=ACCENT, lw=1.4, ls="--")
axB.text(base_scr, len(tor) - 0.35, f" base {base_scr:.0f}", color=ACCENT,
         fontsize=8.5, va="bottom")
# marge a gauche pour ne pas couper la barre la plus basse ni son etiquette
_lo_min = min(min(r[2], r[4]) for r in tor)
axB.set_xlim(left=_lo_min - 350)
axB.set_yticks(ypos)
axB.set_yticklabels([r[0] for r in tor])
axB.set_xlabel("SCR = VaR 99,5 % (unites normalisees)", fontsize=9, color=INK2)
axB.set_title("(B)  Quel parametre commande le capital ?",
              fontsize=10.5, color=INK, pad=6)
axB.grid(alpha=0.25, lw=0.5, axis="x")

fig.suptitle("La sensibilite du SCR : ce que la donnee pourrait pincer (g) vs "
             "l'hypothese pure (xi)",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0.03, 1, 0.94])
fig.text(0.01, 0.01, "(A) ligne pointillee : au-dela de xi = 0,9 la VaR 99,5 % "
         "devient numeriquement instable (queue tres lourde).",
         fontsize=8, color=ACCENT, ha="left", va="bottom")

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S4_sensibilite.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
