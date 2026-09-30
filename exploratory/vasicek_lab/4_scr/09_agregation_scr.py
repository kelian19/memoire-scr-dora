#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09 : Agregation. Frequence x cascade x severite -> perte annuelle -> SCR (VaR 99,5 %).

Assemble les trois briques (frequence_model, echantillonneur de cascade, severite_model)
via scr_engine. Une annee = tirer Y, tirer N_j | Y, cascader chaque incident (gain g),
sommer les severites. Le SCR est la VaR 99,5 % de la perte annuelle (Solvabilite II
art. 101). On reporte aussi la TVaR (finie car xi<1) a titre indicatif, pas comme mesure
retenue.

PORTEE. Resultat en UNITES NORMALISEES de severite (1 unite = perte mediane d'un incident
sur le pilier le moins grave). Le niveau absolu en euros n'est pas revendique : sans
donnees de perte de l'entite il n'est pas identifiable. Ce qui est porte, c'est la
STRUCTURE (contributions par pilier, part de la queue) et la SENSIBILITE (g, xi, a),
detaillee en 10. Le SCR chiffre ici est le point de base de cette surface.

Sortie : diagnostics (SCR, VaR/TVaR, IC bootstrap, decomposition) + figure
S3_agregation.png.
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
import severite_model as sev

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(20260715)
W = 74


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# ============================================================ verifs du moteur
titre("Cascade : la distribution des ensembles S est-elle propre ? (somme = 1)")
print(f"  gain de base g = {eng.G_BASE}")
tables = eng.build_cascade_tables(eng.G_BASE)
print(f"  {'amorce':<9}{'nb ensembles':>14}{'somme proba':>14}{'E[|S|]':>10}"
      f"{'P(|S|=1)':>11}")
for j in eng.PIL:
    ind, probs = tables[j]
    tailleS = ind.sum(axis=1)
    print(f"  P{j:<8}{len(probs):>14}{probs.sum():>14.6f}"
          f"{(probs*tailleS).sum():>10.3f}{probs[tailleS == 1].sum():>11.3f}")

titre("Effet du gain g sur la taille de cascade (E[|S|] moyenne sur les amorces)")
print("  g=0 => aucun incident ne se propage (|S|=1). g monte => cascades plus larges.")
sroot = sum(eng.LAMBDA.values())
for g in (0.0, 0.3, 0.6, 0.9, 1.0):
    tb = eng.build_cascade_tables(g)
    es = 0.0
    for j in eng.PIL:
        ind, probs = tb[j]
        es += (eng.LAMBDA[j] / sroot) * (probs * ind.sum(axis=1)).sum()
    print(f"  g = {g:.1f}   E[|S|] (pondere amorce) = {es:.3f}")

# ============================================================ SCR chiffre
titre("SCR chiffre (cas de base : g=0,90, xi=0,70, a=0,60)")
NY = 300_000
losses = eng.simulate_annual_losses(NY, RNG)
E = losses.mean()
v995 = eng.var(losses, 0.995)
t995 = eng.tvar(losses, 0.995)
lo, hi = eng.var_ci(losses, 0.995, n_boot=300, rng=RNG)
print(f"  annees simulees        : {NY:,}")
print(f"  perte annuelle moyenne : {E:10.2f}  (unites normalisees)")
print(f"  ecart-type             : {losses.std():10.2f}")
print(f"  VaR 99,5 %             : {v995:10.2f}   IC95% [{lo:.1f} ; {hi:.1f}]")
print(f"  TVaR 99,5 % (indicatif): {t995:10.2f}   (finie car xi<1 ; non retenue)")
print(f"  SCR = VaR - E[perte]   : {v995 - E:10.2f}   (capital economique)")
print(f"  P(perte = 0)           : {(losses == 0).mean():10.4f}")

titre("Quantiles de la perte annuelle")
for q in (0.50, 0.90, 0.99, 0.995, 0.999):
    print(f"  q{q:<6} = {eng.var(losses, q):10.2f}")

titre("Ce qui fait le SCR : contribution moyenne par pilier a la perte annuelle")
print("  contribution annuelle esperee = E[N_j] * E[severite d'un incident amorce en j].")
tb = eng.build_cascade_tables(eng.G_BASE)
tot = 0.0
rows = []
for j in eng.PIL:
    ind, probs = tb[j]
    espS = np.array([sum(sev.mean_pilier(p) for p in eng.PIL if ind[r, eng._COL[p]])
                     for r in range(len(probs))])
    esev = float((probs * espS).sum())          # severite esperee d'un incident amorce en j
    c = eng.LAMBDA[j] * esev                     # contribution annuelle esperee
    rows.append((j, eng.LAMBDA[j], esev, c))
    tot += c
for j, lam, esev, c in rows:
    print(f"  P{j} : lambda={lam:5.2f}  E[sev|amorce]={esev:8.2f}  "
          f"contrib={c:8.2f}  ({100*c/tot:4.1f} %)")
print(f"  total contributions (= E[perte]) : {tot:8.2f}   (MC : {E:.2f})")

# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2 = "#1b1e30", "#223e55"
ACCENT, BLUE, GREEN = "#a6002e", "#3661ac", "#008d87"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.2))

# panneau A : distribution de la perte annuelle avec VaR et TVaR
cap = np.quantile(losses, 0.9995)
axA.hist(losses[losses <= cap], bins=140, density=True, color=BLUE, alpha=0.55)
for x, lab, col in [(v995, "VaR 99,5 %", ACCENT), (t995, "TVaR 99,5 %", GREEN)]:
    axA.axvline(x, color=col, lw=1.6, ls="--")
    axA.text(x, axA.get_ylim()[1] * 0.9, f" {lab}\n {x:.0f}", color=col,
             fontsize=8.5, va="top")
axA.set_xlabel("perte annuelle agregee (unites normalisees)", fontsize=9.5, color=INK2)
axA.set_ylabel("densite", fontsize=9.5, color=INK2)
axA.set_title("(A)  Perte annuelle : la queue commande le capital",
              fontsize=10.5, color=INK, pad=6)
axA.grid(alpha=0.25, lw=0.5)

# panneau B : contributions par pilier
js = [r[0] for r in rows]
cs = [r[3] for r in rows]
PCOL = {1: "#204993", 2: "#4c79c7", 3: "#6491e1", 4: "#a6002e", 5: "#7d7d7d"}
axB.bar([f"P{j}" for j in js], cs, color=[PCOL[j] for j in js])
for i, (j, lam, esev, c) in enumerate(rows):
    axB.text(i, c, f"{100*c/tot:.0f} %", ha="center", va="bottom",
             fontsize=9, color=INK)
axB.set_ylabel("contribution a la perte annuelle esperee", fontsize=9.5, color=INK2)
axB.set_title("(B)  Qui porte le risque : amorce (ROOT) x gravite (GBASE)",
              fontsize=10.5, color=INK, pad=6)
axB.grid(alpha=0.25, lw=0.5, axis="y")

fig.suptitle("Agregation : de l'incident au capital (SCR = VaR 99,5 %)",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S3_agregation.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
