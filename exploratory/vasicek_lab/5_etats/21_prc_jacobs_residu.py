#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21 : IC honnete du Delta_DORA PRC - reinjecter le RESIDU de la conversion Jacobs.

Le pendant PRC du script 14. L'IC90 PRC du memoire et du 16 est etroit (facteur ~1,3)
la ou l'IC OpRisk est large (facteur ~14). Ce n'est pas une vertu du perimetre : la
conversion Jacobs est un transport DETERMINISTE records -> euros (ln L = a + b ln X),
qui ecrase la dispersion de severite. La source primaire (Jacobs 2014, Data Driven
Security, sortie de regression reproduite dans le billet) donne pourtant :

    ln(L_usd) = 7.68 + 0.76 ln(X) + eps,   RSE = 0.523 (113 ddl),  R^2 = 0.512

soit, a volume de records donne, un cout reel qui varie d'un facteur ~2,4 autour de la
droite a 90 % (exp(+-1,645*0,523)). Ce script reinjecte ce residu et mesure ce que l'IC
PRC devient quand on cesse de faire comme si la conversion etait exacte.

PROTOCOLE (bootstrap 2 niveaux, lecture B comme le 16) : a chaque tirage b,
  1. reechantillonnage des 15 053 records PRC (avec remise) ;
  2. conversion Jacobs AVEC residu (eps ~ N(0, 0.523) par record) ET SANS residu
     (memes records reechantillonnes : l'ecart entre variantes isole le residu) ;
  3. refit GPD des exces au-dessus du seuil u FIXE de la config (4,176 M) ; p_u_b
     recalcule (part des pertes > u) ; frequences par etat en mode sample ;
  4. SCR par etat a graine MC commune -> Delta(NC vs C), Delta(PC vs C).

PORTEE. Les coefficients (a, b) restent ceux de Jacobs 2014 (non calibres sur l'entite,
axe de sensibilite, cf. plan) ; le residu est suppose gaussien homoscedastique comme dans
la regression d'origine. Cap de severite 40 M inchange (config). Ne touche ni src/ ni
memoire/.

Sortie : diagnostics (effet du residu sur la queue GPD ; Delta_DORA IC90 avec/sans
residu vs 16 et memoire) + figure S16_prc_jacobs_residu.png. Verifie exit 0.
"""

import os
import sys

import numpy as np
from scipy.stats import genpareto
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import euro_cascade_model as ec                              # noqa: E402
from euro_cascade_model import PARAMS, MEMOIRE_DELTA, var    # noqa: E402
from src.severity.prc_analysis import (load_prc, chemin_prc,           # noqa: E402
                                       JACOBS_A, JACOBS_B,
                                       USD_EUR)

W = 74
ETATS = ["C", "PC", "NC"]
SCENARIO = {"C": "S0_conforme", "PC": "S1_partiel", "NC": "S2_non_conforme"}
G_PROP = {"C": 0.45, "PC": 0.68, "NC": 0.90}     # lecture B (frequence + propagation)

JACOBS_RSE = 0.523        # ecart-type residuel ln, Jacobs 2014 (RSE, 113 ddl, R^2=0.512)
B = 60                    # tirages bootstrap (comme 16, PRC)
NY = 3_000                # annees simulees par etat et par tirage
XI_CLIP = (0.05, 2.5)     # PRC xi ~ 1,03 : ne pas ecraser xi > 1 (clip large)

REF_16 = {"NC": (3668, 3201, 4094), "PC": (1257, 1077, 1466)}   # 16, approx normale, lect. B


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def jacobs_eur_m(X, eps=None):
    """Conversion Jacobs records -> M EUR ; eps = residu ln (None = deterministe)."""
    ln_L = JACOBS_A + JACOBS_B * np.log(X)
    if eps is not None:
        ln_L = ln_L + eps
    return np.exp(ln_L) * USD_EUR / 1e6


def fit_tail(L, u):
    """(xi, sigma, p_u) : refit GPD des exces au-dessus du seuil fixe u."""
    exc = L[L > u] - u
    if exc.size < 30:
        return None
    xi, _, sg = genpareto.fit(exc, floc=0)
    return (float(np.clip(xi, *XI_CLIP)), max(0.05, float(sg)), float((L > u).mean()))


# ============================================================ donnees
PRC_PATH = chemin_prc(_REPO)
if not PRC_PATH:
    sys.exit("Data_Breach_Chronology absente de data/raw : script indisponible.")
X_ALL = load_prc(PRC_PATH)["total_affected"].values.astype(float)
sp = PARAMS["PRC"]
U = sp["u"]

# ============================================================ 1. effet du residu sur la queue
titre("1. Ce que le residu de Jacobs fait a la queue de severite PRC")
print(f"  ln(L) = {JACOBS_A} + {JACOBS_B} ln(X) + eps,  eps ~ N(0, {JACOBS_RSE})  (RSE Jacobs 2014)")
print(f"  facteur de cout a 90 % autour de la droite : x{np.exp(-1.645 * JACOBS_RSE):.2f} a "
      f"x{np.exp(1.645 * JACOBS_RSE):.2f}")
rng0 = np.random.default_rng(20260716)
L_det = jacobs_eur_m(X_ALL)
L_res = jacobs_eur_m(X_ALL, rng0.normal(0.0, JACOBS_RSE, X_ALL.size))
print(f"\n  {'':<26}{'sans residu':>14}{'avec residu':>14}")
f_det, f_res = fit_tail(L_det, U), fit_tail(L_res, U)
for lab, i in (("xi (queue)", 0), ("sigma (M EUR)", 1), ("p_u = P(L > u)", 2)):
    print(f"  {lab:<26}{f_det[i]:>14.3f}{f_res[i]:>14.3f}")
print(f"  (config figee : xi={sp['xi']:.3f}, sigma={sp['sigma']:.3f}, p_u={sp['p_u']:.3f} ; "
      f"seuil u={U:.3f} M fixe)")
print("  Lecture : le bruit lognormal multiplicatif epaissit la queue derivee ;")
print("  la conversion deterministe sous-estimait la dispersion de severite.")

# ============================================================ 2. bootstrap honnete avec/sans residu
titre("2. Delta_DORA PRC bootstrap 2 niveaux, residu ON/OFF (lecture B, CRN)")
print(f"  {B} tirages x {NY:,} annees x 3 etats x 2 variantes. Meme reechantillonnage de")
print("  records et memes frequences dans les deux variantes : l'ecart isole le residu.")
orng = np.random.default_rng(20260716)
acc = {(v, cible): [] for v in ("OFF", "ON") for cible in ("PC", "NC")}
pu_on = []
b_done = 0
for b in range(B):
    idx = orng.integers(0, X_ALL.size, X_ALL.size)          # records reechantillonnes
    Xb = X_ALL[idx]
    eps = orng.normal(0.0, JACOBS_RSE, Xb.size)
    lam = {e: ec.lambda_scenario("PRC", SCENARIO[e], mode="sample", rng=orng) for e in ETATS}
    fits = {"OFF": fit_tail(jacobs_eur_m(Xb), U), "ON": fit_tail(jacobs_eur_m(Xb, eps), U)}
    if fits["OFF"] is None or fits["ON"] is None:
        continue
    pu_on.append(fits["ON"][2])
    seed_b = 3000 + b
    for vlab, (xi_b, sg_b, pu_b) in fits.items():
        v = {}
        for e in ETATS:
            rng_e = np.random.default_rng(seed_b)            # CRN entre etats ET variantes
            v[e] = var(ec.simulate_euro(lam[e], G_PROP[e], xi_b, sg_b,
                                        U, pu_b, sp["cap"], NY, rng_e))
        acc[(vlab, "PC")].append(v["PC"] - v["C"])
        acc[(vlab, "NC")].append(v["NC"] - v["C"])
    b_done += 1

res = {}
for vlab, note in (("OFF", "records seuls (conversion exacte)"),
                   ("ON", "records + residu Jacobs (honnete)")):
    print(f"\n  variante {vlab} - {note}  ({b_done} tirages) :")
    for cible in ("PC", "NC"):
        d = np.array(acc[(vlab, cible)])
        med, (lo, hi) = np.median(d), np.percentile(d, [5, 95])
        res[(vlab, cible)] = (med, lo, hi, d)
        print(f"    Delta_DORA({cible} vs C) : median = {med:7.0f} M EUR   "
              f"IC90% [{lo:.0f} ; {hi:.0f}]  (facteur {hi / lo:.2f})   "
              f"part > 0 : {100.0 * (d > 0).mean():.0f}%")
print(f"\n  p_u avec residu : moyenne {np.mean(pu_on):.3f} (config {sp['p_u']:.3f}, seuil u fixe)")
print(f"  reference 16 (approx normale)   : NC {REF_16['NC'][0]} [{REF_16['NC'][1]} ; "
      f"{REF_16['NC'][2]}]   PC {REF_16['PC'][0]} [{REF_16['PC'][1]} ; {REF_16['PC'][2]}]")
ref = MEMOIRE_DELTA["PRC"]
print(f"  reference memoire (NC vs C, 07) : {ref['median']:.0f} "
      f"IC90 [{ref['ic90'][0]:.0f} ; {ref['ic90'][1]:.0f}]")
fac_on = res[("ON", "NC")][2] / res[("ON", "NC")][1]
fac_off = res[("OFF", "NC")][2] / res[("OFF", "NC")][1]
print(f"\n  Verdict : l'IC PRC s'elargit de facteur {fac_off:.2f} a facteur {fac_on:.2f} quand on")
print("  cesse de traiter la conversion Jacobs comme exacte. Le verdict directionnel")
print("  (Delta > 0) doit tenir ; le niveau PRC etait sur-precis, pas plus sur.")

# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2 = "#1b1e30", "#223e55"
ACCENT, BLUE, GREY = "#a6002e", "#3661ac", "#7d7d7d"

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime
# cette figure sur 7,27 pouces : tracee plus large, elle subissait une
# reduction qui faisait tomber ses etiquettes sous 6 points. Le rapport
# largeur sur hauteur est conserve, donc la figure occupe la meme place
# sur la page ; c'est son texte qui y prend plus de place.
fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.9, 3.52))

# panneau A : distributions du Delta (NC vs C), ON vs OFF
d_off, d_on = res[("OFF", "NC")][3], res[("ON", "NC")][3]
bins = np.linspace(min(d_off.min(), d_on.min()), max(d_off.max(), d_on.max()), 28)
axA.hist(d_off, bins=bins, color=BLUE, alpha=0.65, label="sans residu (conversion exacte)")
axA.hist(d_on, bins=bins, color=ACCENT, alpha=0.65, label="avec residu Jacobs (honnete)")
axA.axvline(REF_16["NC"][0], color=GREY, lw=1.4, ls="--", label="16 (approx normale)")
axA.set_xlabel("Delta_DORA NC vs C (M€, PRC, lecture B)", fontsize=9.3, color=INK2)
axA.set_ylabel("tirages bootstrap", fontsize=9.5, color=INK2)
axA.set_title("(A)  Le residu elargit la distribution du surcout", fontsize=10, color=INK, pad=6)
axA.legend(fontsize=8.2, frameon=False)
axA.grid(alpha=0.25, lw=0.5)

# panneau B : IC90 compares (barres d'intervalle)
rows = [("memoire (07)", ref["median"], ref["ic90"][0], ref["ic90"][1], GREY),
        ("16 approx normale", REF_16["NC"][0], REF_16["NC"][1], REF_16["NC"][2], GREY),
        ("21 records seuls", *[res[("OFF", "NC")][i] for i in (0, 1, 2)], BLUE),
        ("21 + residu Jacobs", *[res[("ON", "NC")][i] for i in (0, 1, 2)], ACCENT)]
for i, (lab, med, lo, hi, col) in enumerate(rows):
    axB.plot([lo, hi], [i, i], color=col, lw=3.2, alpha=0.85, solid_capstyle="round")
    axB.plot(med, i, "o", color=col, ms=7)
    axB.annotate(f"[{lo:.0f} ; {hi:.0f}]", (hi, i), textcoords="offset points",
                 xytext=(6, -3), fontsize=8, color=INK2)
axB.set_yticks(range(len(rows)))
axB.set_yticklabels([r[0] for r in rows], fontsize=9)
axB.invert_yaxis()
axB.set_xlabel("Delta_DORA NC vs C (M€, PRC) : mediane et IC90", fontsize=9.3, color=INK2)
axB.set_title("(B)  IC90 PRC : de sur-precis a honnete", fontsize=10, color=INK, pad=6)
axB.grid(alpha=0.25, lw=0.5, axis="x")

fig.tight_layout(w_pad=1.8)

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S16_prc_jacobs_residu.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
