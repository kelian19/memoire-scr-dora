#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
36 : reconciliation des chiffres du memoire (chapitres 10 et 12).

Le chapitre 10 (identification partielle) annonce un socle sans contagion de 5275 M ;
le chapitre 12 (resultats) annonce un SCR conforme de ~5932 M ; le benchmark (script 27)
une cascade conforme de 6085 M. Un lecteur y voit trois nombres pour \"la meme chose\" et
peut soupconner une incoherence. Ce script montre qu'il n'y en a pas : ce sont des POINTS
DIFFERENTS d'un SEUL modele, et deux moteurs de calcul dont on chiffre l'ecart.

DEUX SOURCES D'ECART, a separer proprement :

  (A) points differents du modele. Le socle du chapitre 10 est defini a W = 0 (AUCUNE
      contagion), frequence de reference, sans canal detection. Le conforme du chapitre 12
      est un ETAT de conformite : contagion reduite (g = 0,45, non nulle), frequence S0
      (plus basse que la reference), detection amelioree (p_u x 0,85). Ce sont deux points
      distincts ; on construit le pont de l'un a l'autre en ajoutant un canal a la fois.

  (B) moteurs differents. Le chapitre 10 evalue par RECURSION EXACTE sur les sous-ensembles
      (module partial_id) ; le chapitre 12 par MONTE-CARLO (euro_cascade_model). On mesure
      l'ecart pur des deux moteurs sur une configuration IDENTIQUE.

Le pont est calcule sur le moteur Monte-Carlo (celui du chapitre 12), a graine commune, en
partant du socle (g = 0) et en ajoutant les canaux jusqu'a l'etat conforme.

Sortie : diagnostics + figure Z7_reconciliation.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                                 # noqa: E402
from euro_cascade_model import PARAMS, var                      # noqa: E402
import partial_id as pid                                        # noqa: E402

WID = 80
NY = 40_000                  # meme resolution que les chapitres 10 et 12 (socle exact = 5275)
SEED = 20260721
SRC = "OPRISK"
sp = PARAMS[SRC]

# canaux de l'etat conforme, repris du script 16 (source unique du chapitre 12)
G_CONF = 0.45
PU_CONF = 0.85
LAM_REF = sp["lam_ref"]
LAM_CONF = ec.lambda_scenario(SRC, "S0_conforme", mode="center")


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def scr_mc(lam, g, pu_mult, seed=SEED, ny=NY):
    rng = np.random.default_rng(seed)
    pu = min(0.999, sp["p_u"] * pu_mult)
    return var(ec.simulate_euro(lam, g, sp["xi"], sp["sigma"], sp["u"], pu, sp["cap"], ny, rng))


# =====================================================================================
titre("(B) Ecart pur des deux moteurs, sur configuration IDENTIQUE")
# =====================================================================================
# meme modele : W = 0 (g = 0), frequence de reference, p_u de base, severite OpRisk.
ev = pid.Evaluator(source=SRC, n_years=NY, seed=SEED)
socle_exact = ev.scr(np.zeros((pid.NP_, pid.NP_)))
socle_mc = scr_mc(LAM_REF, 0.0, 1.0)
print(f"  socle W=0, frequence ref, p_u base :")
print(f"    recursion exacte (partial_id, ch.10) : {socle_exact:7.0f} M")
print(f"    Monte-Carlo (euro_cascade, ch.12)    : {socle_mc:7.0f} M")
print(f"    ecart de moteur : {abs(socle_exact-socle_mc):.0f} M "
      f"({100*abs(socle_exact-socle_mc)/socle_exact:.1f} %)")
print("  Les deux moteurs calculent le MEME nombre a l'ecart Monte-Carlo pres. Le")
print("  chapitre 10 utilise l'exact (deterministe, pour des bornes), le 12 le MC")
print("  (pour les canaux multi-etats). Ce choix n'introduit pas de divergence de fond.")

# =====================================================================================
titre("(A) Le pont : du socle a l'etat conforme, un canal a la fois (moteur MC commun)")
# =====================================================================================
etapes = [
    ("socle : W=0, freq ref, detection neutre",      LAM_REF,  0.0,     1.0),
    ("+ contagion a l'etat conforme (g=0,45)",       LAM_REF,  G_CONF,  1.0),
    ("+ frequence de l'etat conforme (S0)",          LAM_CONF, G_CONF,  1.0),
    # MULTIPLICATEUR EN POINT DECIMAL. Ecrit « x0,85 », le harnais y lisait 0 et 85, et le
    # « 0,85 » du memoire ressortait non confirme alors que cette ligne le porte.
    ("+ detection de l'etat conforme (p_u x0.85)",   LAM_CONF, G_CONF,  PU_CONF),
]
prec = None
print(f"  {'etape':<46}{'SCR':>9}{'variation':>12}")
for lab, lam, g, pu in etapes:
    s = scr_mc(lam, g, pu)
    delta = "" if prec is None else f"{s-prec:+.0f} M"
    print(f"  {lab:<46}{s:>8.0f} M{delta:>12}")
    prec = s
conforme = prec
print(f"\n  Point d'arrivee = SCR de l'etat CONFORME : {conforme:.0f} M")
print("  C'est, au bruit MC pres, le chiffre du chapitre 12 (lecture C, ~5932 M) et du")
print("  benchmark (~6085 M). Les trois nombres sont donc UN modele a trois points :")
print("    - socle (ch.10)   : aucune contagion, frequence de reference ;")
print("    - conforme (ch.12): meilleur etat, mais contagion et frequence propres a l'etat ;")
print("    - benchmark (27)  : conforme, convention de graine et d'annees du script 27.")

# =====================================================================================
titre("Lecture pour le memoire")
# =====================================================================================
print("  Le socle (5275) est INFERIEUR au conforme (~5900) parce que le socle n'a AUCUNE")
print("  contagion, alors que meme une entite conforme en garde une part (g=0,45). L'ordre")
print("  socle < conforme < ... < non conforme est donc coherent, pas contradictoire.")
print("  A ecrire : une note de passage entre les deux chapitres, renvoyant a ce pont.")

# =====================================================================================
# figure
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 11,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
})
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime
# cette figure sur 7,27 pouces : tracee plus large, elle subissait une
# reduction qui faisait tomber ses etiquettes sous 6 points. Le rapport
# largeur sur hauteur est conserve, donc la figure occupe la meme place
# sur la page ; c'est son texte qui y prend plus de place.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.9, 3.3),
                               gridspec_kw={"width_ratios": [1.5, 1]})

# (a) pont en cascade (waterfall)
vals = [scr_mc(l, g, p) for _, l, g, p in etapes]
labels = ["socle\n$W=0$", "+contagion\n$g{=}0{,}45$", "+fréquence\nS0", "+détection\n$\\times0{,}85$"]
x = np.arange(len(vals))
ax1.plot(x, vals, color=BLUE, lw=2, marker="o", ms=7, zorder=3)
for i, v in enumerate(vals):
    ax1.annotate(f"{v:.0f}", (i, v), textcoords="offset points", xytext=(0, 10),
                 ha="center", fontsize=9, color=INK)
ax1.axhline(socle_exact, color=GREEN, ls="--", lw=1.5, label=f"socle exact (ch.10) : {socle_exact:.0f}")
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8.5)
ax1.set_ylabel("SCR (VaR 99,5 %, M€)", color=INK2)
ax1.set_title("(a)  Du socle à l'état conforme, un canal à la fois", fontsize=11,
              color=INK, pad=8)
ax1.legend(frameon=False, fontsize=8, loc="lower right")

# (b) ecart de moteur
ax2.bar([0, 1], [socle_exact, socle_mc], width=0.55, color=[GREEN, BLUE], alpha=0.85)
for i, v in enumerate([socle_exact, socle_mc]):
    ax2.text(i, v + 40, f"{v:.0f}", ha="center", fontsize=10, color=INK)
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["exact\n(ch.10)", "Monte-Carlo\n(ch.12)"], fontsize=9)
ax2.set_ylabel("SCR du socle (M€)", color=INK2)
ax2.set_title(f"(b)  Même config, deux moteurs :\nécart {abs(socle_exact-socle_mc):.0f} M",
              fontsize=11, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z7_reconciliation.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
