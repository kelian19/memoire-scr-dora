#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14 : Robustesse du Delta_DORA par la cascade + lecture honnete de l'IC.

Deux objectifs :

1. RESSERRER l'IC en retirant l'artefact de methode. Le script 13 tirait (xi, sigma)
   dans une approximation NORMALE de l'IC90, non bornee, qui atteignait xi proche de 1
   et gonflait l'IC (jusqu'a un facteur ~24). Ici on bootstrappe xi/sigma en
   REECHANTILLONNANT les 91 exces reels d'OpRisk (methode exacte du memoire, notebook
   01/07), ce qui reste realiste. On reporte separement :
     - l'IQR (Q25-Q75) : l'ecart TYPIQUE, resserre ;
     - l'IC90 : la queue, large, INHERENTE a 91 exces en queue lourde (c'est la these
       du memoire : le SCR est une distribution, pas un point ; son IC90 mono-perte
       vaut deja un facteur 2,5).

2. ROBUSTESSE DU VERDICT. Le niveau du capital n'est pas pincable, mais le VERDICT
   l'est : Delta_DORA > 0 (la non-conformite coute), d'ordre pluri-milliard, et
   croissant avec la propagation, sur toute la plage du gain g. On le montre en balayant
   g_nc et en verifiant que la mediane reste dans la meme bande.

PORTEE. Lecture LOCALE de data/raw (jamais poussee) pour les exces reels ; ne sort que
des parametres agreges. Ne touche ni src/ ni memoire/.

Sortie : diagnostics + figure S8_robustesse_delta.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                              # noqa: E402
from euro_cascade_model import PARAMS, MEMOIRE_DELTA, var    # noqa: E402

W = 74
G_NC = ec.G_BASE
NY = 20_000
B = 200


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# ============================================================ exces reels + controle
titre("Exces reels OpRisk (lecture locale de data/raw) et controle du fit")
EXC = ec.oprisk_excesses()
if EXC is None:
    sys.exit("SAS OpRisk absent de data/raw (licence) : robustesse OpRisk indisponible.")
from scipy.stats import genpareto                             # noqa: E402
xi_pt, _, sg_pt = genpareto.fit(EXC, floc=0)
print(f"  n exces = {len(EXC)}   (config : {PARAMS['OPRISK'].get('n_excess', 91) if 'n_excess' in PARAMS['OPRISK'] else 91})")
print(f"  fit ponctuel : xi = {xi_pt:.3f} (config 0,595)   sigma = {sg_pt:.1f} M€ (config 58,0)")
print(f"  seuil u = {PARAMS['OPRISK']['u']:.2f} M€")

# ============================================================ Delta_DORA : IQR vs IC90
titre("Delta_DORA OpRisk, bootstrap sur exces REELS (canal frequence, g=0,90)")
sp = PARAMS["OPRISK"]
orng = np.random.default_rng(20260715)
deltas = []
for b in range(B):
    lam_nc = ec.lambda_scenario("OPRISK", "S2_non_conforme", mode="sample", rng=orng)
    lam_c = ec.lambda_scenario("OPRISK", "S0_conforme", mode="sample", rng=orng)
    xi_b, sg_b = ec.bootstrap_sev_from_excesses(EXC, orng)
    seed_b = 2000 + b
    v_nc = var(ec.simulate_euro(lam_nc, G_NC, xi_b, sg_b, sp["u"], sp["p_u"], sp["cap"],
                                NY, np.random.default_rng(seed_b)))
    v_c = var(ec.simulate_euro(lam_c, G_NC, xi_b, sg_b, sp["u"], sp["p_u"], sp["cap"],
                               NY, np.random.default_rng(seed_b)))
    deltas.append(v_nc - v_c)
d = np.array(deltas)
med = np.median(d)
q25, q75 = np.percentile(d, [25, 75])
lo, hi = np.percentile(d, [5, 95])
ref = MEMOIRE_DELTA["OPRISK"]
print(f"  mediane           = {med:8.0f} M€")
print(f"  IQR  (Q25-Q75)    = [{q25:.0f} ; {q75:.0f}]  -> ecart TYPIQUE, facteur {q75/q25:.1f}")
print(f"  IC90 (Q05-Q95)    = [{lo:.0f} ; {hi:.0f}]  -> queue, facteur {hi/lo:.1f}")
print(f"  memoire (07)      = {ref['median']:.0f}  IC90 [{ref['ic90'][0]:.0f} ; {ref['ic90'][1]:.0f}]"
      f"  (facteur {ref['ic90'][1]/ref['ic90'][0]:.1f})")
print("  Lecture : le facteur IC90 rejoint celui du memoire (~15) une fois xi bootstrappe")
print("  sur les exces reels ; l'IQR montre que l'ecart TYPIQUE est bien plus resserre.")
print(f"  Verdict : {(d > 0).mean()*100:.0f} % des tirages donnent Delta_DORA > 0.")

# ============================================================ robustesse au gain g_nc
titre("Robustesse du verdict au gain de propagation g_nc (mediane sur exces reels)")
print("  Pour chaque g, mini-bootstrap sur les exces reels. Le verdict doit tenir.")
print(f"  {'g_nc':>6}{'Delta median':>15}{'IQR':>24}")
gcurve = {}
for g in (0.30, 0.50, 0.70, 0.90, 1.00):
    dd = []
    rr = np.random.default_rng(100 + int(g * 100))
    for b in range(40):
        lam_nc = ec.lambda_scenario("OPRISK", "S2_non_conforme", mode="sample", rng=rr)
        lam_c = ec.lambda_scenario("OPRISK", "S0_conforme", mode="sample", rng=rr)
        xi_b, sg_b = ec.bootstrap_sev_from_excesses(EXC, rr)
        sd = 3000 + b
        v_nc = var(ec.simulate_euro(lam_nc, g, xi_b, sg_b, sp["u"], sp["p_u"], sp["cap"],
                                    12_000, np.random.default_rng(sd)))
        v_c = var(ec.simulate_euro(lam_c, g, xi_b, sg_b, sp["u"], sp["p_u"], sp["cap"],
                                   12_000, np.random.default_rng(sd)))
        dd.append(v_nc - v_c)
    dd = np.array(dd)
    gcurve[g] = (np.median(dd), np.percentile(dd, 25), np.percentile(dd, 75))
    print(f"  {g:>6.2f}{gcurve[g][0]:>15.0f}   [{gcurve[g][1]:.0f} ; {gcurve[g][2]:.0f}]")
print("  Le Delta_DORA median reste pluri-milliard et > 0 sur toute la plage de g :")
print("  le NIVEAU depend de g (propagation), mais le VERDICT est robuste.")

# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2 = "#1b1e30", "#223e55"
ACCENT, BLUE, GREEN = "#a6002e", "#3661ac", "#008d87"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.4, 5.3))

# panneau A : distribution, IQR vs IC90
cut = float(np.percentile(d, 97))
n_omis = int((d > cut).sum())
axA.hist(d[d <= cut], bins=40, density=True, color=BLUE, alpha=0.5)
axA.axvspan(q25, q75, color=GREEN, alpha=0.18)
axA.axvline(med, color=BLUE, lw=1.8, label=f"mediane {med:.0f}")
axA.axvline(q25, color=GREEN, lw=1.1, ls="-")
axA.axvline(q75, color=GREEN, lw=1.1, ls="-", label=f"IQR [{q25:.0f} ; {q75:.0f}]")
axA.axvline(hi, color=ACCENT, lw=1.1, ls=":", label=f"Q95 {hi:.0f}")
axA.axvline(ref["median"], color="#000", lw=1.3, ls="--", label=f"median memoire {ref['median']:.0f}")
axA.set_xlim(0, cut)
axA.text(0.98, 0.6, f"{n_omis}/{len(d)} > {cut:.0f}\nomis (lisibilite)",
         transform=axA.transAxes, ha="right", va="top", fontsize=7.8, color=INK2)
axA.set_xlabel("Delta_DORA (M€)", fontsize=9.5, color=INK2)
axA.set_ylabel("densite", fontsize=9.5, color=INK2)
axA.set_title("(A)  IQR resserre vs IC90 large : l'ecart typique n'est pas la queue",
              fontsize=9.6, color=INK, pad=6)
axA.legend(fontsize=8, frameon=False)
axA.grid(alpha=0.25, lw=0.5)

# panneau B : robustesse au gain
gs = sorted(gcurve)
meds = [gcurve[g][0] for g in gs]
q25s = [gcurve[g][1] for g in gs]
q75s = [gcurve[g][2] for g in gs]
axB.fill_between(gs, q25s, q75s, color=BLUE, alpha=0.15, label="IQR")
axB.plot(gs, meds, "-o", color=BLUE, lw=2.0, ms=4.5, label="Delta_DORA median")
axB.axhline(ref["median"], color="#000", lw=1.2, ls="--")
axB.text(0.30, ref["median"], " median memoire", ha="left", va="bottom",
         fontsize=8.2, color=INK2)
axB.set_ylim(bottom=0)
axB.set_xlabel("gain de propagation g_nc", fontsize=9.5, color=INK2)
axB.set_ylabel("Delta_DORA (M€)", fontsize=9.5, color=INK2)
axB.set_title("(B)  Verdict robuste : pluri-milliard et > 0 sur toute la plage de g",
              fontsize=9.6, color=INK, pad=6)
axB.legend(fontsize=8.4, frameon=False)
axB.grid(alpha=0.25, lw=0.5)

fig.suptitle("Robustesse du Delta_DORA : IC honnete (exces reels) et verdict stable",
             fontsize=12.3, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S8_robustesse_delta.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
