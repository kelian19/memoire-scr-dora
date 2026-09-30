#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
51 : le SCR robuste (pire-cas) sur l'ensemble d'ambiguite.

Reponse prudente a la remarque de Caroline (script 46) : plutot que le quantile d'un point
estime, on prend la BORNE HAUTE de la VaR sur un ensemble d'ambiguite, c'est-a-dire l'ensemble
des modeles plausibles compte tenu de l'incertitude. On la construit sur deux couches :
  - AMBIGUITE DE PARAMETRE : la region de confiance de (xi, sigma) (bootstrap OpRisk) ;
  - AMBIGUITE DE MODELE : la famille de queue (jusqu'a une GPD lourde xi=0,90, script 48).

VaR ROBUSTE au niveau de confiance beta = quantile beta de la loi bootstrap de la VaR
(= 'la VaR qu'on ne depasse pas avec probabilite beta, compte tenu du risque d'estimation').
C'est une VaR distributionnellement robuste sur la region de confiance des parametres.

L'echelle des postures : point < predictive < robuste(parametre) < robuste(parametre+modele).
Le niveau de robustesse beta est un CHOIX de politique prudentielle, a assumer explicitement.

Sortie : diagnostics + figure J7_scr_robuste.png.
"""

import os
import sys

import numpy as np
from scipy.stats import genpareto, norm
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR  # noqa: E402
from src.utils.config import OPRISK                                                        # noqa: E402

WID = 82
A = 0.995
B = 3000
N_SIM = 3_000_000
SEED = 20260727
rng = np.random.default_rng(SEED)


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def var_gpd(a, xi, sig, u, zu):
    return u + (sig / xi) * (((1 - a) / zu) ** (-xi) - 1)


# ------------------------------------------------------------------ calibration + ambiguite
d = filter_finance(filter_cyber(load_clean(
    os.path.join(REPO, "data", "raw", "SAS_OpRisk_Global_Data_June_2026.xlsx"))))
loss = np.sort(d["loss"].to_numpy() * USD_EUR)
# SEUIL PUBLIE, PAS SEUIL REDERIVE. Comme les scripts 46 et 47, celui-ci prenait le q85 des
# donnees courantes (22,03 M€, 88 exces) alors que le memoire publie la calibration figee
# (20,03 M€, 91 exces) : la VaR robuste etait donc construite sur un ensemble d'ambiguite
# autour d'un ajustement autre que celui du memoire.
u = float(OPRISK["seuil_u_eur"]); zu = float((loss > u).mean())
exc = loss[loss > u] - u; n = exc.size
xi_hat, _, sig_hat = genpareto.fit(exc, floc=0)

xb, sb = [], []
for _ in range(B):
    r = rng.choice(exc, size=n, replace=True)
    try:
        c, _, s = genpareto.fit(r, floc=0)
        if 0 < s < 1e6 and -0.5 < c < 3:
            xb.append(c); sb.append(s)
    except Exception:
        pass
xb, sb = np.array(xb), np.array(sb)

titre("Calibration et ensemble d'ambiguite (OpRisk cyber x finance)")
print(f"  u = {u:.1f} M€, {n} exces ; xi = {xi_hat:.3f}, sigma = {sig_hat:.1f}.")
print(f"  Bootstrap : {len(xb)} ajustements ; xi IC90 = [{np.percentile(xb,5):.2f} ; "
      f"{np.percentile(xb,95):.2f}].")

# ------------------------------------------------------------------ l'echelle des postures
var_point = var_gpd(A, xi_hat, sig_hat, u, zu)
idx = rng.integers(0, len(xb), N_SIM)
exc_pred = genpareto.rvs(c=xb[idx], scale=sb[idx], random_state=rng)
var_pred = u + float(np.quantile(exc_pred, 1 - (1 - A) / zu))
var_boot = var_gpd(A, xb, sb, u, zu)                       # loi de la VaR sur l'ambiguite param.
rob = {b: float(np.percentile(var_boot, 100 * b)) for b in (0.90, 0.95, 0.99)}
var_heavy = var_gpd(A, 0.90, sig_hat, u, zu)               # pire-cas famille (queue lourde 48)

titre("L'echelle des postures : du point au pire-cas")
# POURQUOI CES DEFINITIONS SONT IMPRIMEES ICI. La figure J7 affiche SIX barres et le memoire
# n'en expliquait que quatre : les niveaux beta = 90 % et 99 % n'etaient cites nulle part dans
# le texte. Un jury lit la figure avant la prose, et six etiquettes sans definition ne sont pas
# une echelle, ce sont six nombres. La definition de chaque posture est donc imprimee avec sa
# valeur, pour que le chapitre puisse la citer au lieu de la paraphraser.
print("  CE QUE CHAQUE POSTURE INTEGRE, ET CE QU'ELLE N'INTEGRE PAS")
print("    point        : quantile d'un ajustement PONCTUEL (xi_hat, sigma_hat). N'integre")
print("                   aucune incertitude. C'est le plug-in, et il ne se cite jamais seul.")
print("    predictive   : quantile de la loi MELANGE des GPD sur le bootstrap de (xi, sigma).")
print("                   L'incertitude entre AVANT le quantile. Point honnete, non prudent.")
print("    robuste beta : borne haute de la VaR sur l'ambiguite de PARAMETRE, au niveau beta.")
print("                   C'est le quantile beta de la loi bootstrap de la VaR. Le niveau beta")
print("                   est un CHOIX de politique prudentielle, pas un resultat de calcul :")
print("                   90 % dit 'je couvre le risque d'estimation neuf fois sur dix'.")
print("    pire-cas     : on ajoute l'ambiguite de MODELE, en poussant la famille de queue")
print("                   jusqu'a xi = 0,90 (script 48). Borne, pas estimation.")
print()
print(f"  {'posture':<40}{'VaR 99,5 % (M€)':>16}")
print(f"  {'point (branchement xi_hat)':<40}{var_point:>16.0f}")
# L'ETIQUETTE DISAIT « melange, script 46 » ET C'ETAIT TROMPEUR : rien n'est lu du script 46,
# la predictive est RECALCULEE ici, sur un ensemble bootstrap de taille differente (B = 3000
# contre 2000 dans le 46). D'ou 645 ici et 648 la-bas, pour la meme grandeur. Voir la note de
# bruit de simulation imprimee juste apres le tableau.
print(f"  {'predictive (recalculee ici, B = %d)' % B:<40}{var_pred:>16.0f}")
for b, v in rob.items():
    print(f"  {'robuste parametre (beta=%.0f%%)' % (100*b):<40}{v:>16.0f}")
print(f"  {'robuste parametre+modele (xi=0,90)':<40}{var_heavy:>16.0f}")
print(f"\n  La VaR robuste a 95 % ({rob[0.95]:.0f} M€) est {rob[0.95]/var_point:.1f}x le point")
print(f"  ({var_point:.0f}) : c'est la marge de prudence pour le risque d'ESTIMATION. Le pire-cas")
print(f"  de MODELE (queue lourde, {var_heavy:.0f} M€) ajoute la prudence sur le CHOIX de famille.")

# =====================================================================================
titre("Combien de chiffres chaque posture supporte-t-elle ? (bruit de simulation)")
# =====================================================================================
# POURQUOI CE BLOC. Le memoire publiait la predictive a 648 M EUR (script 46) et la figure de
# ce script l'affichait a 645 : deux valeurs pour la MEME grandeur, et un lecteur y voit un
# desaccord. Il n'y en a pas. Les deux scripts appliquent le meme estimateur sur des ensembles
# bootstrap de tailles differentes (2000 contre 3000), et le troisieme chiffre significatif de
# cette grandeur n'est pas stable. On le MESURE au lieu de l'affirmer, en rejouant l'echelle
# complete sur plusieurs graines.
#
# CE QUE CE BLOC N'EST PAS. Ce n'est pas une recalibration : aucune entree du modele ne change,
# les valeurs publiees restent celles de la graine de reference. On mesure la reproductibilite
# d'un nombre publie, ce qui est un controle et non un recalage.
#
# COUT. K-1 ensembles bootstrap supplementaires, soit environ une minute par graine. Le tirage
# du melange, lui, est reduit dans ce diagnostic : son bruit propre vaut moins d'un M EUR quand
# celui de l'ensemble vaut plusieurs, donc il ne domine pas et n'a pas besoin de 3 millions de
# tirages pour etre negligeable.
K_GRAINES = 4
N_SIM_DIAG = 400_000
_rep = {k: [] for k in ("predictive", "rob90", "rob95", "rob99")}
for _j in range(K_GRAINES):
    _rg = np.random.default_rng(SEED + _j)
    _xb, _sb = [], []
    for _ in range(B):
        _r = _rg.choice(exc, size=n, replace=True)
        try:
            _c, _, _s = genpareto.fit(_r, floc=0)
            if 0 < _s < 1e6 and -0.5 < _c < 3:
                _xb.append(_c); _sb.append(_s)
        except Exception:
            pass
    _xb, _sb = np.array(_xb), np.array(_sb)
    _idx = _rg.integers(0, len(_xb), N_SIM_DIAG)
    _e = genpareto.rvs(c=_xb[_idx], scale=_sb[_idx], random_state=_rg)
    _rep["predictive"].append(u + float(np.quantile(_e, 1 - (1 - A) / zu)))
    _vb = var_gpd(A, _xb, _sb, u, zu)
    for _b, _k in ((90, "rob90"), (95, "rob95"), (99, "rob99")):
        _rep[_k].append(float(np.percentile(_vb, _b)))

print(f"  {K_GRAINES} graines, B = {B} rechantillons chacune. Les deux extremites de l'echelle")
print("  sont DETERMINISTES et n'apparaissent donc pas ici : le point est le quantile d'un")
print("  ajustement unique, le pire-cas celui d'un xi POSE a 0,90. Tout le bruit est au milieu.\n")
print(f"  {'posture':<14}{'moyenne':>10}{'ecart-type':>12}{'etendue':>10}{'CV':>8}")
_cv = {}
for _k, _v in _rep.items():
    _a = np.array(_v)
    _cv[_k] = 100 * _a.std(ddof=1) / _a.mean()
    print(f"  {_k:<14}{_a.mean():>10.1f}{_a.std(ddof=1):>12.2f}"
          f"{_a.max()-_a.min():>10.1f}{_cv[_k]:>7.1f}%")

_sd = {_k: float(np.std(_v, ddof=1)) for _k, _v in _rep.items()}
print("\n  CE QU'IL FAUT EN RETENIR, ET C'EST UN RESULTAT.")
print(f"  1. La posture la plus prudente est la plus BRUITEE : l'ecart-type vaut {_sd['rob99']:.0f} M EUR")
print(f"     a beta = 99 % contre {_sd['rob90']:.0f} et {_sd['rob95']:.0f} a beta = 90 et 95 %, soit environ le")
print("     double. C'est structurel : un percentile a 99 % de la loi bootstrap est estime sur")
print("     bien moins de tirages utiles que sa mediane, et la queue est lourde.")
print("     A NE PAS SURINTERPRETER : avec quatre graines, un ecart-type est lui-meme connu a")
print(f"     40 % pres environ. L'ordre entre beta = 90 % ({_cv['rob90']:.1f} % de CV) et beta = 95 %")
print(f"     ({_cv['rob95']:.1f} %) n'est donc PAS resolu ici, et il n'a pas a l'etre : c'est le saut vers")
print("     beta = 99 % qui est net, pas le classement des deux premiers.")
print("  2. Donc CHOISIR LA POSTURE LA PLUS PRUDENTE COUTE DE LA PRECISION. Prendre beta = 99 %")
print("     achete de la prudence au prix d'un chiffre dont le troisieme digit ne tient pas.")
print("     Cela ne disqualifie pas la posture, cela interdit de la citer a trois chiffres.")
print(f"  3. La predictive vaut {np.mean(_rep['predictive']):.0f} M EUR a "
      f"{np.std(_rep['predictive'], ddof=1):.0f} pres. Le 648 du script 46 et le")
print("     645 de la figure sont donc LE MEME NOMBRE : leur ecart de 3 M EUR vaut moins d'un")
print("     ecart-type. Le memoire doit publier deux chiffres significatifs, pas trois.")
print("  4. Comparee au point (deterministe), la predictive en differe de "
      f"{abs(var_point - np.mean(_rep['predictive'])):.0f} M EUR, soit")
print("     le meme ordre que le bruit lui-meme. « Quasi egale au plug-in » n'est donc pas une")
print("     approximation de redaction : c'est tout ce que la mesure permet d'affirmer.")

titre("VERDICT")
print("  1. Le SCR robuste = borne haute de la VaR sur l'ensemble d'ambiguite (parametre, puis")
print("     modele). Il repond a Caroline par la PRUDENCE : on ne parie pas sur un point.")
print(f"  2. Echelle : point {var_point:.0f} -> predictive {var_pred:.0f} -> robuste 95 % "
      f"{rob[0.95]:.0f} -> pire-cas modele {var_heavy:.0f} M€.")
print("  3. Le niveau de robustesse beta est un CHOIX prudentiel explicite, pas un calcul : on")
print("     l'affiche, on ne le cache pas. C'est l'exact complement de la lecture en bande.")

# ------------------------------------------------------------------ figure J7
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 10.5,
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.9, 3.09))

# (a) loi de la VaR sur l'ambiguite de parametre, avec les postures
ax1.hist(var_boot, bins=60, color=BLUE, alpha=0.5, edgecolor="#fcfcfb")
# LE POINT ET LA PREDICTIVE SONT A DOUZE M EUR L'UN DE L'AUTRE, soit moins de deux ecarts-types
# de simulation : leurs deux lignes se superposent presque, et l'etiquette de la predictive,
# posee A DROITE de sa ligne comme les autres, se faisait couper par la ligne du point. On la
# renvoie A GAUCHE. C'est un defaut de lisibilite, pas de contenu, mais il portait sur la
# valeur meme que ce panneau doit faire lire.
for lab, v, c, cote in [("point", var_point, INK, "left"),
                        ("prédictive", var_pred, GREEN, "right"),
                        ("robuste 95 %", rob[0.95], ACCENT, "left")]:
    ax1.axvline(v, color=c, lw=1.8, ls="--" if lab != "point" else "-")
    _h = ax1.get_ylim()[1] * (0.9 if lab == "point" else 0.72 if lab == "prédictive" else 0.6)
    _pad = " " if cote == "left" else "  "
    ax1.text(v, _h, f"{_pad}{lab}\n{_pad}{v:.0f}", fontsize=8, color=c,
             ha=cote, linespacing=1.35)
ax1.set_xlabel("VaR 99,5 % (M€) sur l'ambiguïté de paramètre", color=INK2)
ax1.set_ylabel("fréquence (bootstrap)", color=INK2)
ax1.set_title("(a)  La VaR est elle-même incertaine :\nle robuste en prend la borne haute",
              fontsize=11, color=INK, pad=8)

# (b) l'echelle des postures, AVEC LE BRUIT DE SIMULATION DE CHACUNE.
# Sans barre d'erreur, six barres etiquetees a trois chiffres suggerent six mesures egalement
# precises. Elles ne le sont pas : les deux extremites sont deterministes (aucune simulation),
# les quatre du milieu portent un bruit qui double a beta = 99 %. La barre d'erreur est donc
# ici une correction de lecture, pas un ornement.
labs = ["point", "prédictive", "robuste\n90 %", "robuste\n95 %", "robuste\n99 %", "pire-cas\nmodèle"]
vals = [var_point, var_pred, rob[0.90], rob[0.95], rob[0.99], var_heavy]
errs = [0.0, _sd["predictive"], _sd["rob90"], _sd["rob95"], _sd["rob99"], 0.0]
cols = [INK, GREEN, "#9dc3e6", BLUE, "#204993", ACCENT]
xp = np.arange(len(labs))
ax2.bar(xp, vals, color=cols, alpha=0.9,
        yerr=errs, ecolor=INK2, capsize=4, error_kw={"lw": 1.1})
for x_, v, e in zip(xp, vals, errs):
    txt = f"{v:.0f}" if e == 0 else f"{v:.0f} ± {e:.0f}"
    ax2.text(x_, v + e + 34, txt, ha="center", fontsize=8.5, color=INK2)
ax2.set_xticks(xp); ax2.set_xticklabels(labs, fontsize=8.5)
ax2.set_ylabel("VaR 99,5 % (M€)", color=INK2)
ax2.set_title("(b)  L'échelle des postures : du point au pire-cas\n"
              "(± écart-type de simulation ; les extrémités sont exactes)",
              fontsize=11, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(w_pad=1.8)
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "J7_scr_robuste.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
