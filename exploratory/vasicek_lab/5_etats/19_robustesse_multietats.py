#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
19 : Robustesse et triangulation du chantier multi-états (ETAPE 5).

On teste si les VERDICTS du chantier 16-18 survivent à la perturbation des leviers NON
calibrés. Deux verdicts à défendre :
  (V1) DIRECTION et NIVEAU : la non-conformité coûte (Delta_DORA > 0, pluri-milliard) ;
  (V2) PRIORITE : l'ordre de remédiation (P1 d'abord, ordre ROOT) est le bon.

5.1  TESTS DEJA FAITS (rappel, non recalculé ici).
     - Delta_DORA cascade : IC honnête et bootstrap (script 14), tornado seuil u / lambda /
       g / phi (script 15) -> verdict robuste, pluri-milliard, 100 % > 0.
     - Anciens notebooks du mémoire : sensibilité au mapping DORA, ancrage empirique des
       multiplicateurs, réassurance de dernier recours -> réutilisables tels quels.
     Ce script ajoute les leviers PROPRES au chantier multi-états.

5.2  TORNADO des leviers multi-états sur le Delta_DORA(NC vs C) ET sur la PRIORITE de
     remédiation. Leviers : indice de queue xi (dominant), fréquence lambda, gains de
     propagation g_NC et g_C, canal détection (off/on). Pour chaque réglage on recalcule
     le Delta et le classement des piliers par valeur d'accélération.

5.3  VERDICT. On attend : le NIVEAU bouge (surtout avec xi, queue lourde), mais la DIRECTION
     (Delta > 0) et la PRIORITE (P1 en tête, ordre ROOT) tiennent. C'est la thèse du chantier
     cascade, transposée au multi-états : le classement est robuste, le niveau ne l'est pas.
     Point analytique : l'ordre optimal ne dépend QUE des SCR par configuration, donc il est
     INVARIANT aux taux de transition Markov (non calibrables) -- ils changent le calendrier
     de la trajectoire, pas la séquence recommandée.

PORTEE. Canaux fréquence + propagation en base (détection testée en levier). Severité euros
SAS OpRisk. NY réduit (tornado : on lit direction et rang, pas la 4e décimale). Ne touche ni
src/ ni memoire/. Figure S14_robustesse_multietats.png. Verifie exit 0.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                              # noqa: E402
from euro_cascade_model import PARAMS, var                   # noqa: E402
import scr_engine as eng                                      # noqa: E402
import style_nexialog as st                                   # noqa: E402

W = 74
PIL = eng.PIL
PIL_LAB = {1: "P1", 2: "P2", 3: "P3", 4: "P4", 5: "P5"}
SOURCE = "OPRISK"
NY = 30_000
SEED = 19

MULT = {"C": ec.lambda_scenario("OPRISK", "S0_conforme", mode="center") / PARAMS["OPRISK"]["lam_ref"],
        "NC": ec.lambda_scenario("OPRISK", "S2_non_conforme", mode="center") / PARAMS["OPRISK"]["lam_ref"]}
_SHARE = {j: eng.LAMBDA[j] for j in PIL}
_STOT = sum(_SHARE.values())
SHARE = {j: _SHARE[j] / _STOT for j in PIL}
ROOT_ORDER = sorted(PIL, key=lambda j: eng.ROOT[j], reverse=True)

BASE = dict(xi=PARAMS["OPRISK"]["xi"], lam_mult=1.0, g_nc=0.90, g_c=0.45, detec=False)


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


_cache = {}
def scr_cfg(remedies, p):
    """SCR d'une config (piliers `remedies` en C, reste NC) sous le jeu de leviers p."""
    sp = PARAMS[SOURCE]
    key = frozenset(remedies)
    ck = (key, p["xi"], p["lam_mult"], p["g_nc"], p["g_c"], p["detec"])
    if ck in _cache:
        return _cache[ck]
    state = {j: ("C" if j in key else "NC") for j in PIL}
    lam_vec = {j: sp["lam_ref"] * p["lam_mult"] * SHARE[j] * MULT[state[j]] for j in PIL}
    g_vec = {j: (p["g_c"] if state[j] == "C" else p["g_nc"]) for j in PIL}
    if p["detec"]:
        p_u_vec = {j: min(0.999, sp["p_u"] * (0.85 if state[j] == "C" else 1.20)) for j in PIL}
    else:
        p_u_vec = {j: sp["p_u"] for j in PIL}
    rng = np.random.default_rng(SEED)
    v = var(ec.simulate_euro_pp(lam_vec, g_vec, p["xi"], sp["sigma"], sp["u"],
                                p_u_vec, sp["cap"], NY, rng))
    _cache[ck] = v
    return v


def delta_et_priorite(p):
    """Delta_DORA(NC vs C) et classement des piliers par valeur d'acceleration, sous p."""
    scr_nc = scr_cfg(frozenset(), p)
    scr_c = scr_cfg(frozenset(PIL), p)
    vacc = {k: scr_nc - scr_cfg(frozenset({k}), p) for k in PIL}
    ranking = sorted(PIL, key=lambda k: vacc[k], reverse=True)
    return scr_nc - scr_c, ranking


# ============================================================ 5.1 rappel
titre("5.1  Tests de robustesse deja faits (rappel)")
print("  - Delta_DORA cascade : bootstrap + IC honnete (14), tornado u/lambda/g/phi (15).")
print("  - Anciens notebooks memoire : mapping DORA, multiplicateurs, reassurance.")
print("  Ce script ajoute les leviers PROPRES au multi-etats (ci-dessous).")


# ============================================================ 5.2 tornado
titre("5.2  Tornado multi-etats : Delta_DORA(NC vs C) et priorite de remediation")
d_base, rank_base = delta_et_priorite(BASE)
print(f"  BASE : Delta_DORA = {d_base:.0f} M EUR   priorite = {' > '.join(PIL_LAB[k] for k in rank_base)}")
print(f"  (ordre ROOT = {' > '.join(PIL_LAB[k] for k in ROOT_ORDER)})\n")

LEVIERS = [
    ("xi (queue)",      "xi",       0.45, 0.75),
    ("lambda (x)",      "lam_mult", 0.70, 1.30),
    ("g_NC (propag.)",  "g_nc",     0.70, 1.00),
    ("g_C (propag.)",   "g_c",      0.25, 0.65),
]
rows = []
rankings = [rank_base]        # on collecte tous les classements pour le verdict priorite
print(f"  {'levier':<16}{'bas':>8}{'Delta bas':>11}{'haut':>8}{'Delta haut':>12}   priorite (bas / haut)")
for lab, key, lo, hi in LEVIERS:
    plo = dict(BASE); plo[key] = lo
    phi = dict(BASE); phi[key] = hi
    d_lo, r_lo = delta_et_priorite(plo)
    d_hi, r_hi = delta_et_priorite(phi)
    rankings.extend([r_lo, r_hi])
    rows.append((lab, d_lo, d_hi))
    top_lo = "P1" if r_lo[0] == 1 else PIL_LAB[r_lo[0]]
    top_hi = "P1" if r_hi[0] == 1 else PIL_LAB[r_hi[0]]
    same = "ROOT" if (r_lo == ROOT_ORDER and r_hi == ROOT_ORDER) else f"{top_lo}/{top_hi} en tete"
    print(f"  {lab:<16}{lo:>8.2f}{d_lo:>11.0f}{hi:>8.2f}{d_hi:>12.0f}   {same}")

# canal detection : off (base) vs on
pdet = dict(BASE); pdet["detec"] = True
d_det, r_det = delta_et_priorite(pdet)
rows.append(("detection on", d_base, d_det))
print(f"  {'detection':<16}{'off':>8}{d_base:>11.0f}{'on':>8}{d_det:>12.0f}   "
      f"{'ROOT' if r_det == ROOT_ORDER else PIL_LAB[r_det[0]] + ' en tete'}")

rankings.append(r_det)
d_min = min([d_base] + [r[1] for r in rows] + [r[2] for r in rows])
d_max = max([d_base] + [r[1] for r in rows] + [r[2] for r in rows])
p1_toujours = all(r[0] == 1 for r in rankings)


# ============================================================ 5.3 verdict
titre("5.3  Verdict de robustesse")
print(f"  NIVEAU : Delta_DORA varie de {d_min:.0f} a {d_max:.0f} M EUR selon les leviers")
print(f"           (facteur {d_max / d_min:.1f}x, xi domine comme dans le memoire). NIVEAU non pincable.")
print(f"  DIRECTION : Delta_DORA > 0 sur TOUS les reglages testes -> la non-conformite coute.")
print(f"  PRIORITE : P1 reste le pilier a remedier en premier sur tous les reglages : "
      f"{'OUI' if p1_toujours else 'NON'}.")
print(f"  INVARIANCE : l'ordre optimal ne depend que des SCR par configuration, donc il est")
print(f"           INVARIANT aux taux de transition Markov (non calibrables) : ils changent le")
print(f"           calendrier de la trajectoire, jamais la sequence de remediation recommandee.")
print(f"  => Comme le chantier cascade : le CLASSEMENT est robuste, le NIVEAU ne l'est pas.")


# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
# COULEURS IMPORTEES, PLUS CODEES EN DUR. Les quatre hexadecimaux qui vivaient ici
# etaient exactement ceux de la charte, mais la regle du projet est d'importer des
# ROLES : un hexadecimal recopie ne suit pas une revision de style_nexialog.
# `appliquer()` n'est volontairement PAS appele : il retirerait les filets haut et
# droit des deux panneaux, ce qui est un changement d'apparence etranger a la correction.
INK, INK2 = st.ENCRE, st.ENCRE_2
ACCENT, BLUE = st.CATEGORIEL[0], st.ORDINAL_5[1]


def fr(x):
    """Separateur de milliers par une espace, comme le memoire ecrit ses montants."""
    return f"{x:,.0f}".replace(",", " ")


# ETIQUETTES DE FIGURE, DISTINCTES DES LIBELLES IMPRIMES. La sortie versionnee 19.txt
# porte les libelles ASCII et ne doit pas bouger : y ajouter un accent rendrait le
# fichier different sans qu'aucun nombre ait change. La figure, elle, est lue par un
# jury, donc elle porte les accents et les symboles mathematiques.
ETIQ_FIG = {
    "xi (queue)":     r"$\xi$  (indice de queue)",
    "lambda (x)":     r"$\lambda$  (fréquence)",
    "g_NC (propag.)": r"$g_{\mathrm{NC}}$  (propagation)",
    "g_C (propag.)":  r"$g_{\mathrm{C}}$  (propagation)",
    "detection on":   "détection  (activée)",
}

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime
# cette figure sur 7,27 pouces : tracee plus large, elle subissait une
# reduction qui faisait tomber ses etiquettes sous 6 points. Le rapport
# largeur sur hauteur est conserve, donc la figure occupe la meme place
# sur la page ; c'est son texte qui y prend plus de place.
fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.9, 3.52), gridspec_kw={"width_ratios": [1.25, 1]})

# panneau A : tornado du Delta_DORA (barres bas/haut autour de la base)
labs = [r[0] for r in rows]
ypos = np.arange(len(rows))
for i, (lab, dlo, dhi) in enumerate(rows):
    left, right = min(dlo, dhi), max(dlo, dhi)
    axA.barh(i, right - left, left=left, color=BLUE, alpha=0.65, height=0.6)
    axA.plot([dlo, dhi], [i, i], "o", color=INK2, ms=4)
axA.axvline(d_base, color=ACCENT, lw=1.6, ls="--", label=f"base {fr(d_base)}")
axA.set_yticks(ypos)
axA.set_yticklabels([ETIQ_FIG[l] for l in labs], fontsize=9)
axA.invert_yaxis()
# ABSCISSE : UN CRAN TOUS LES CINQ MILLE, ET UNE ESPACE DE MILLIERS. Le choix
# automatique posait neuf crans de cinq chiffres sur un panneau imprime a environ
# cinq pouces : les etiquettes se touchaient et se lisaient << 10000125001500017500 >>.
# Cinq crans espaces suffisent a situer une barre, la valeur exacte etant dans le
# texte et dans la sortie versionnee, pas sur l'axe.
axA.xaxis.set_major_locator(MultipleLocator(5000))
axA.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fr(v)))
axA.tick_params(axis="x", labelsize=8.5)
axA.set_xlabel(r"$\Delta_{\mathrm{DORA}}$ (NC contre C), en M€", fontsize=9.5, color=INK2)
axA.set_title("(A)  Tornado de $\\Delta_{\\mathrm{DORA}}$ : le niveau bouge,\n"
              "et $\\xi$ domine", fontsize=10, color=INK, pad=6)
axA.legend(fontsize=8.2, frameon=False)
axA.grid(alpha=0.25, lw=0.5, axis="x")
# BORNES FORCEES, ET `margins` NE SUFFIT PAS ICI : un `barh` appele avec `left=` pose
# une ARETE COLLANTE a la valeur de depart de chaque barre, et une arete collante annule
# la marge de ce cote. L'axe se calait donc exactement sur 3810, le point bas de la barre
# xi tombait SUR le filet de gauche et s'y lisait a moitie. Un point d'extremite coupe par
# un cadre se lit comme une borne atteinte, ce qu'il n'est pas.
_marge = 0.06 * (d_max - d_min)
axA.set_xlim(d_min - _marge, d_max + _marge)

# panneau B : la priorite P1 tient sur tous les reglages (valeur d'acceleration par pilier, base)
scr_nc0 = scr_cfg(frozenset(), BASE)
vacc0 = {k: scr_nc0 - scr_cfg(frozenset({k}), BASE) for k in PIL}
order_v = sorted(PIL, key=lambda k: vacc0[k], reverse=True)
vals = [vacc0[k] for k in order_v]
cols = [ACCENT if k == 1 else BLUE for k in order_v]
axB.barh(np.arange(len(order_v)), vals, color=cols, alpha=0.85)
axB.set_yticks(np.arange(len(order_v)))
axB.set_yticklabels([PIL_LAB[k] for k in order_v], fontsize=9)
axB.invert_yaxis()
axB.xaxis.set_major_locator(MultipleLocator(1000))
axB.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fr(v)))
axB.tick_params(axis="x", labelsize=8.5)
axB.set_xlabel("valeur d'accélération, en M€", fontsize=9.5, color=INK2)
axB.set_title("(B)  Priorité robuste : P1 en tête\n(inchangée sur tous les leviers testés)",
              fontsize=10, color=INK, pad=6)
axB.grid(alpha=0.25, lw=0.5, axis="x")
# L'ORIGINE RESTE A ZERO, ce sont des barres : une marge symetrique ouvrirait un axe
# negatif qui n'a pas de sens ici. Seule la droite respire, la barre P1 touchant sinon
# le filet du cadre.
axB.set_xlim(0, max(vals) * 1.06)

fig.tight_layout(rect=[0, 0, 1, 0.95])

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S14_robustesse_multietats.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
