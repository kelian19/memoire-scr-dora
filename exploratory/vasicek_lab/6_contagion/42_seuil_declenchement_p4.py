#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
42 : le seuil de declenchement de P4, et l'accumulation bornee par la dependance de queue.

Question de Hugo : a partir de quel MOMENT un pilier est-il declenche, et quel est le seuil
mathematique pour P4 ? Et remarque de Kelian : une loi NORMALE pour le franchissement ?
Ce script formalise le declenchement dans le modele PAR PILIER (une entite, cinq piliers) et
traite les deux points, en unifiant le seuil latent K_j (scripts 02/04) et le choc commun de
P4 (script 38).

--------------------------------------------------------------------------------------------
DEUX NIVEAUX A DISTINGUER (c'est la reponse a la remarque sur la normale).

(1) LA MARGE : quand un pilier bascule-t-il ? Franchissement  X_j >= K_j,  K_j = F^-1(1 - p_j).
    Ici la loi F de la latente est une NORMALISATION : K_j = F^-1(1 - p_j) reproduit la
    probabilite marginale p_j quelle que soit F. Prendre F normale ne coute rien sur la marge,
    et surtout NE reintroduit PAS de queue fine dans le capital : la queue lourde vit dans la
    SEVERITE (GPD, xi ~ 0,60), objet SEPARE du declenchement (scripts 02/04). p_4 = 0,151
    donne K_4 = 1,03.

(2) LA DEPENDANCE : combien de piliers basculent ENSEMBLE ? C'est la l'accumulation (MOVEit),
    et c'est la que la normale est un MAUVAIS choix. Une copule GAUSSIENNE a une dependance de
    queue superieure NULLE (lambda_U = 0) : elle sous-estime les basculements simultanes de
    plusieurs piliers, exactement le phenomene qu'on veut modeliser. Elle contredit d'ailleurs
    la copule que le memoire a retenue partout ailleurs : une GUMBEL (theta = 1,8, config),
    choisie pour sa dependance de queue superieure (lambda_U = 2 - 2^{1/theta} = 0,53).

REPONSE, DANS L'ESPRIT DU MEMOIRE : BORNER. On ne tranche pas la copule, on lit l'accumulation
en BANDE, a dependance moyenne EGALE (meme tau de Kendall = 1 - 1/theta = 0,444) :
    plancher = copule gaussienne (lambda_U = 0),   haut = copule de Gumbel (lambda_U = 0,53).
La MARGE (frequence de declenchement d'un pilier = p_4) est INVARIANTE d'une copule a l'autre :
la copule ne change pas COMBIEN de fois un pilier bascule seul, elle change s'ils basculent
ENSEMBLE. L'accumulation (nombre de piliers emportes avec P4) est donc encadree entre le
plancher gaussien et le haut de Gumbel. La borne phi_cs <= gamma = 0,68 du 38 sert de repere.

PERIMETRE. Tout est PAR PILIER, a une entite : le nombre de piliers co-declenches, pas une
fraction de portefeuille. La vue inter-clients (masse d'entites victimes d'un meme prestataire,
vraie maille MOVEit) serait une loi de Vasicek en dimension ENTITE, HORS SCR par entite.

Sortie : diagnostics + figure Z12_seuil_declenchement_p4.png.
"""

import os
import sys

import numpy as np
from scipy.stats import norm
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from euro_cascade_model import PARAMS                            # noqa: E402

WID = 82
sp = PARAMS["OPRISK"]
P4_MARG = sp["p_u"]                    # proba marginale de declenchement (materialite OpRisk)
K4 = float(norm.ppf(1.0 - P4_MARG))    # seuil latent (marge)
THETA = 1.8                            # copule de Gumbel (config, dependance de queue sup.)
TAU = 1.0 - 1.0 / THETA                # tau de Kendall commun aux deux copules
RHO_G = float(np.sin(np.pi * TAU / 2)) # correlation gaussienne de MEME tau
LAMBDA_U = 2.0 - 2.0 ** (1.0 / THETA)  # dependance de queue sup. de la Gumbel
GAMMA = 0.68                           # borne phi_cs du script 38 (repere)
NPIL = 5
P4_IDX = 3
N_MC = 500_000
SEED = 20260727


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# ------------------------------------------------------------------ copules exchangeables
def gaussian_uniforms(n, d, rho, rng):
    """Copule gaussienne equicorrelee (facteur commun), lambda_U = 0."""
    Z = rng.standard_normal(n)
    eps = rng.standard_normal((n, d))
    X = np.sqrt(rho) * Z[:, None] + np.sqrt(1.0 - rho) * eps
    return norm.cdf(X)


def _rstable_pos(alpha, n, rng):
    """Stable positive de transformee de Laplace exp(-s^alpha) (Kanter), alpha dans (0,1)."""
    U = np.pi * rng.random(n)
    W = rng.exponential(1.0, n)
    a = (np.sin((1 - alpha) * U) * np.sin(alpha * U) ** (alpha / (1 - alpha))
         / np.sin(U) ** (1 / (1 - alpha)))
    return (a / W) ** ((1 - alpha) / alpha)


def gumbel_uniforms(n, d, theta, rng):
    """Copule de Gumbel exchangeable (frailty de Marshall-Olkin), lambda_U = 2 - 2^{1/theta}."""
    alpha = 1.0 / theta
    V = _rstable_pos(alpha, n, rng)
    E = rng.exponential(1.0, (n, d))
    return np.exp(-(E / V[:, None]) ** alpha)


def cotrigger_dist(uniforms):
    """Loi du nombre de piliers declenches SACHANT P4 declenche. trigger : U_j >= 1 - p_4."""
    trig = uniforms >= (1.0 - P4_MARG)
    M = trig[trig[:, P4_IDX]].sum(axis=1)
    cnt = np.bincount(M, minlength=NPIL + 1)[1:]
    return cnt / cnt.sum()


def chi_curve(u1, u2, qs):
    """Fonction de dependance de queue chi(q) = P(U2>q | U1>q)."""
    return np.array([((u1 > q) & (u2 > q)).mean() / (1 - q) for q in qs])


rng = np.random.default_rng(SEED)

# =====================================================================================
titre("1. La marge : franchissement K_j (la normale y est une simple normalisation)")
# =====================================================================================
print(f"  Pilier j declenche  <=>  X_j >= K_j,  K_j = F^-1(1 - p_j).")
print(f"  p_4 = {P4_MARG:.4f}  =>  K_4 = {K4:.3f}. La loi F fixe l'echelle, pas p_4 :")
print("  K_j = F^-1(1-p_j) reproduit p_j quelle que soit F. La normale ne coute rien ICI,")
print("  et ne remet pas de queue fine dans le capital : la queue lourde est dans la SEVERITE")
print("  (GPD xi ~ 0,60), objet separe du declenchement (scripts 02/04).")

# =====================================================================================
titre("2. La dependance : la ou la normale est un MAUVAIS choix (accumulation)")
# =====================================================================================
print(f"  A dependance moyenne EGALE (tau de Kendall = 1 - 1/theta = {TAU:.3f}) :")
print(f"    copule gaussienne : correlation rho = {RHO_G:.3f},  lambda_U = 0        (plancher)")
print(f"    copule de Gumbel  : theta = {THETA},         lambda_U = {LAMBDA_U:.3f}   (haut)")
qs = np.array([0.90, 0.95, 0.99, 0.995])
ug2 = gaussian_uniforms(N_MC, 2, RHO_G, rng)
uk2 = gumbel_uniforms(N_MC, 2, THETA, rng)
chi_g = chi_curve(ug2[:, 0], ug2[:, 1], qs)
chi_k = chi_curve(uk2[:, 0], uk2[:, 1], qs)
print(f"\n  Co-depassement P(U2>q | U1>q), a tau egal :")
print(f"  {'q':>8}{'gaussienne':>14}{'Gumbel':>10}")
for q, a, b in zip(qs, chi_g, chi_k):
    print(f"  {q:>8.3f}{a:>14.3f}{b:>10.3f}")
print("  La gaussienne s'effondre vers 0 dans la queue (asymptote lambda_U = 0) ; la Gumbel")
print(f"  plafonne vers {LAMBDA_U:.2f}. A dependance moyenne identique, l'extreme joint differe du")
print("  tout au tout : c'est precisement la ou l'accumulation se joue.")

# =====================================================================================
titre("3. L'accumulation bornee : nombre de piliers co-declenches, plancher a haut")
# =====================================================================================
pmf_g = cotrigger_dist(gaussian_uniforms(N_MC, NPIL, RHO_G, rng))
pmf_k = cotrigger_dist(gumbel_uniforms(N_MC, NPIL, THETA, rng))
freq_g = float((gaussian_uniforms(80_000, NPIL, RHO_G, rng) >= (1 - P4_MARG)).mean())
freq_k = float((gumbel_uniforms(80_000, NPIL, THETA, rng) >= (1 - P4_MARG)).mean())
print(f"  Frequence MARGINALE de declenchement d'un pilier : gaussienne {freq_g:.3f}, "
      f"Gumbel {freq_k:.3f}")
print(f"  (~ p_4 = {P4_MARG:.3f} pour les deux : la copule ne change PAS combien un pilier")
print("   bascule seul, elle change s'ils basculent ENSEMBLE.)")
print(f"\n  Loi du nombre de piliers declenches SACHANT P4 :")
print(f"  {'copule':>12}" + "".join(f"  P(={m})" for m in range(1, 6))
      + f"{'P(>=3)':>9}{'E[nb|P4]':>10}")
e_g = float((np.arange(1, 6) * pmf_g).sum())
e_k = float((np.arange(1, 6) * pmf_k).sum())
for lab, pmf, e in [("gaussienne", pmf_g, e_g), ("Gumbel", pmf_k, e_k)]:
    print(f"  {lab:>12}" + "".join(f"{v:7.3f}" for v in pmf)
          + f"{pmf[2:].sum():>9.3f}{e:>10.2f}")
phi_g = (e_g - 1) / (NPIL - 1)         # phi_cs = P(un autre pilier | P4)
phi_k = (e_k - 1) / (NPIL - 1)
print(f"\n  A tau egal, les deux copules s'accordent sur le co-declenchement MODERE (P(>=3|P4)")
print(f"  dans [{pmf_g[2:].sum():.2f} ; {pmf_k[2:].sum():.2f}]), mais divergent sur le CO-EXTREME :")
print(f"  P(les 5 piliers | P4) = {pmf_g[4]:.2f} (gaussien) contre {pmf_k[4]:.2f} (Gumbel), un")
print(f"  facteur {pmf_k[4]/pmf_g[4]:.1f}. C'est la que lambda_U mord : la queue gaussienne rate le")
print(f"  basculement simultane de tous les piliers, l'exact scenario MOVEit. phi_cs = P(un")
print(f"  autre pilier | P4) dans [{phi_g:.2f} ; {phi_k:.2f}], sous la borne gamma = {GAMMA} du 38.")

# =====================================================================================
titre("4. Reperes : le seuil, le statut, la borne du 38")
# =====================================================================================
print(f"  Le SEUIL de declenchement de P4 (le 'moment' de Hugo) est le franchissement")
print(f"  X_4 >= K_4 = {K4:.2f} : c'est la marge, family-free. Ce qui n'est PAS identifie,")
print(f"  c'est la DEPENDANCE de queue (la copule), donc l'accumulation : on la BORNE.")
print(f"  La borne phi_cs <= gamma = {GAMMA} du script 38 est un repere : le plancher gaussien")
print(f"  ({phi_g:.2f}) et le haut de Gumbel ({phi_k:.2f}) l'encadrent par en dessous, coherent.")
print("  Meme discipline que W et xi : ce qui est identifie (la marge p_4) est pose ; ce qui")
print("  ne l'est pas (la dependance de queue) est lu en bande, jamais en point.")
print("  Perimetre : nombre de piliers de l'entite, PAS une fraction de portefeuille ; la vue")
print("  inter-clients (Vasicek en dimension entite) est hors du SCR par entite.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. La MARGE (quand P4 bascule) = franchissement X_4 >= K_4 ; la loi normale y est une")
print("     normalisation inoffensive, la queue du capital etant dans la severite (GPD).")
print("  2. La DEPENDANCE (combien de piliers ensemble) NE doit pas etre gaussienne : lambda_U=0")
print("     sous-estime l'accumulation et contredit la Gumbel (theta=1,8) du memoire.")
print("  3. On BORNE l'accumulation a tau egal : le CO-EXTREME P(les 5 piliers | P4) va de "
      f"{pmf_g[4]:.2f} (gaussien) a {pmf_k[4]:.2f} (Gumbel), la ou lambda_U mord (MOVEit).")
print("  4. La marge est invariante d'une copule a l'autre ; seule la co-occurrence de queue")
print("     bouge. Meme discipline d'identification partielle que W et xi.")

# =====================================================================================
# figure Z12
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

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Trois panneaux traces
# sur seize pouces puis imprimes sur 7,27 donnent 1,6 pouce par panneau et
# des etiquettes sous 5 points. Le rapport largeur sur hauteur est conserve.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.66, 3.17))

# (a) la marge : franchissement de K_4 (la normale y est une normalisation)
xs = np.linspace(-4, 4, 400)
ax1.plot(xs, norm.pdf(xs), color=INK2, lw=1.8)
ax1.fill_between(xs[xs >= K4], norm.pdf(xs[xs >= K4]), color=ACCENT, alpha=0.28)
ax1.axvline(K4, color=INK, lw=1.4)
ax1.text(K4 + 0.1, 0.35, f"$K_4$={K4:.2f}", fontsize=9.5, color=INK)
ax1.text(2.15, 0.06, "déclenché\n$X_4\\geq K_4$", fontsize=8.5, color=ACCENT, ha="center")
ax1.text(0.02, 0.90, "marge : normalisation.\nla queue du capital est\ndans la sévérité (GPD),\npas ici.",
         transform=ax1.transAxes, fontsize=8, color=MUTED, style="italic", va="top")
ax1.set_xlabel("variable latente de P4", color=INK2)
ax1.set_ylabel("densité", color=INK2)
ax1.set_title("(a)  La marge : franchissement de $K_4$", fontsize=10.5, color=INK, pad=8)

# (b) fonction de dependance de queue chi(q), a tau egal
qg = np.linspace(0.80, 0.995, 40)
cg = chi_curve(ug2[:, 0], ug2[:, 1], qg)
ck = chi_curve(uk2[:, 0], uk2[:, 1], qg)
ax2.plot(qg, cg, color=BLUE, lw=2.2, label="gaussienne ($\\lambda_U=0$)")
ax2.plot(qg, ck, color=ACCENT, lw=2.2, label=f"Gumbel θ=1,8 ($\\lambda_U$={LAMBDA_U:.2f})")
ax2.axhline(LAMBDA_U, color=ACCENT, ls=":", lw=1)
ax2.set_ylim(0, 0.75)
ax2.set_xlabel("quantile  $q$", color=INK2)
ax2.set_ylabel("co-dépassement  $P(U_2>q\\,|\\,U_1>q)$", color=INK2, fontsize=9)
ax2.legend(frameon=False, fontsize=8, loc="upper center")
ax2.set_title("(b)  À dépendance moyenne égale (même τ),\nl'extrême joint diffère du tout au tout",
              fontsize=10.5, color=INK, pad=8)

# (c) l'accumulation bornee : nombre de piliers co-declenches, plancher vs haut
ms = np.arange(1, 6)
wd = 0.38
ax3.bar(ms - wd / 2, pmf_g, width=wd, color=BLUE, alpha=0.9, label="gaussienne (plancher)")
ax3.bar(ms + wd / 2, pmf_k, width=wd, color=ACCENT, alpha=0.9, label="Gumbel (haut)")
ax3.set_xticks(ms)
ax3.set_xlabel("nombre de piliers déclenchés (sachant P4)", color=INK2)
ax3.set_ylabel("probabilité", color=INK2)
ax3.legend(frameon=False, fontsize=8)
ax3.set_title("(c)  L'accumulation bornée : la queue lourde\ngroupe les piliers (MOVEit)",
              fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2, ax3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z12_seuil_declenchement_p4.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
