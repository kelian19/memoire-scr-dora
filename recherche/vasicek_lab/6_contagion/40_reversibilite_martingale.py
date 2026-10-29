#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
40 : la decomposition W = S + A, lue comme la REVERSIBILITE d'un processus.

Le script 32 (Prop. 3) etablit un FAIT algebrique : toute matrice se decompose de facon
unique en une partie symetrique S = (W+W^T)/2 et une partie antisymetrique A = (W-W^T)/2,
la donnee identifie S (co-occurrence) et pas A (direction), placebo z = -0,33. Ce script
donne a ce fait sa RAISON probabiliste, celle que demande Hugo : le parallele avec la
decomposition d'un processus de Markov en une partie reversible et un courant irreversible,
c'est-a-dire avec la martingalisation (partie fluctuation) et son compensateur (partie derive).

--------------------------------------------------------------------------------------------
LE PARALLELE, EN TROIS IDENTITES VERIFIABLES.

On construit sur les 5 piliers la chaine de Markov qui SAUTE de i vers j proportionnellement
au jugement dirige TRANS[i][j] (row-stochastique P). Elle porte exactement la direction posee.
Soit pi sa loi stationnaire (pi P = pi).

(1) COURANT ET REVERSIBILITE. Le courant de probabilite J_ij = pi_i P_ij - pi_j P_ji est
    antisymetrique. La chaine est REVERSIBLE (equilibre detaille) si et seulement si J = 0.
    Le renversement du temps donne la chaine adjointe Ptilde_ij = pi_j P_ji / pi_i, et
        P_s = (P + Ptilde)/2   est invariante par renversement (la partie reversible),
        P_a = (P - Ptilde)/2   change de SIGNE par renversement (le courant, la direction).
    C'est le miroir exact de S et A : S = co-occurrence, invariante au sens du temps ;
    A = direction, ce qui distingue l'avant de l'apres.

(2) LA FLUCTUATION NE VOIT QUE S (c'est la martingalisation). Pour toute fonction test f,
    l'increment f(X_1) - (Pf)(X_0) est une difference de martingale ; sa variation, mesuree
    par la forme de Dirichlet E(f,f) = (1/2) somme pi_i P_ij (f_j - f_i)^2, ne depend QUE de
    la partie reversible : E_P(f,f) = E_{P_s}(f,f), la partie antisymetrique P_a y contribue
    exactement 0. La fluctuation (la martingale) est aveugle a la direction ; seule la derive
    (le compensateur (Pf - f), le courant) la porte. On le verifie a la 12e decimale.

(3) LA DIRECTION = PRODUCTION D'ENTROPIE. L'irreversibilite se resume en un scalaire, le taux
    de production d'entropie sigma = (1/2) somme (pi_i P_ij - pi_j P_ji) log(pi_i P_ij / pi_j P_ji)
    >= 0, nul si et seulement si la chaine est reversible. La chaine symetrisee a sigma = 0 et
    courant nul. Le placebo directionnel du memoire (z = -0,33) se relit alors PROPREMENT :
    la donnee ne permet pas de distinguer sigma de 0, c'est-a-dire qu'elle est compatible avec
    l'equilibre detaille. Non pas "on n'a pas su calibrer la direction", mais "les observations
    de co-occurrence sont compatibles avec une dynamique reversible".

--------------------------------------------------------------------------------------------
FAISABILITE ET HONNETETE (consigne Hugo). Le parallele est FAISABLE et il RENFORCE la these :
il fonde la limite d'identification sur un theoreme standard (une statistique invariante par
renversement du temps ne peut pas identifier le courant), au lieu de la constater. Deux reserves
a assumer, exactement dans l'esprit "Vasicheck" :
  - la version propre du courant est PONDEREE par pi (produit scalaire de L2(pi)) ; le S = (W+W^T)/2
    du memoire est le cas particulier pi uniforme. On rapporte les deux, l'ecart est mesure ici.
  - W est un noyau de contagion, pas litteralement un generateur de Markov : on EMPRUNTE l'algebre
    de la reversibilite comme analogie eclairante, on ne pretend pas que la cascade EST une chaine
    reversible. C'est une lecture, pas une hypothese ajoutee.

Sortie : diagnostics + figure Z11_reversibilite_martingale.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_HERE = os.path.dirname(os.path.abspath(__file__))
_CASCADE = os.path.abspath(os.path.join(_HERE, "..", "..", "cascade_qualitative"))
if _CASCADE not in sys.path:
    sys.path.insert(0, _CASCADE)
from cascade_model import TRANS, PILIERS  # noqa: E402

WID = 82
PIL = [1, 2, 3, 4, 5]
NP_ = len(PIL)
COL = {j: c for c, j in enumerate(PIL)}


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# ============================================================ objets de base
def trans_matrix():
    """Matrice dirigee brute T_ij = TRANS[i][j] (0 sur la diagonale)."""
    T = np.zeros((NP_, NP_))
    for i in PIL:
        for j, w in TRANS[i].items():
            T[COL[i], COL[j]] = w
    return T


def markov_hop(T):
    """Chaine de saut inter-piliers : P_ij = T_ij / somme_j T_ij (row-stochastique)."""
    return T / T.sum(axis=1, keepdims=True)


def stationary(P):
    """Loi stationnaire pi : vecteur propre gauche de valeur propre 1, normalise."""
    w, V = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(V[:, k])
    pi = np.abs(pi)
    return pi / pi.sum()


def reversed_chain(P, pi):
    """Chaine adjointe (renversee dans le temps) Ptilde_ij = pi_j P_ji / pi_i."""
    return (pi[None, :] * P.T) / pi[:, None]


def current(P, pi):
    """Courant de probabilite J_ij = pi_i P_ij - pi_j P_ji (antisymetrique)."""
    F = pi[:, None] * P
    return F - F.T


def entropy_production(P, pi):
    """Taux de production d'entropie sigma >= 0 ; nul ssi equilibre detaille."""
    F = pi[:, None] * P
    s = 0.0
    for a in range(NP_):
        for b in range(NP_):
            if a != b and F[a, b] > 0 and F[b, a] > 0:
                s += 0.5 * (F[a, b] - F[b, a]) * np.log(F[a, b] / F[b, a])
    return s


def dirichlet(P, pi, f):
    """Forme de Dirichlet E(f,f) = (1/2) somme pi_i P_ij (f_j - f_i)^2 (energie de la martingale)."""
    tot = 0.0
    for a in range(NP_):
        for b in range(NP_):
            tot += 0.5 * pi[a] * P[a, b] * (f[b] - f[a]) ** 2
    return tot


# ============================================================ calculs
T = trans_matrix()
P = markov_hop(T)
pi = stationary(P)
Pt = reversed_chain(P, pi)
Ps = 0.5 * (P + Pt)                     # partie reversible (invariante au renversement)
Pa = 0.5 * (P - Pt)                     # courant (change de signe au renversement)
J = current(P, pi)
sigma = entropy_production(P, pi)

# version Euclidienne du memoire (pi uniforme), pour le pont avec S/A du script 32
G = 0.90
maxs = T.sum(axis=1).max()
W = G * T / maxs
S_eucl = 0.5 * (W + W.T)
A_eucl = 0.5 * (W - W.T)

titre("Chaine de saut inter-piliers construite sur le jugement dirige TRANS")
print("  P_ij = TRANS[i][j] / somme_j TRANS[i][j] (proba de sauter de i vers j).")
print(f"\n  {'':>6}" + "".join(f"{'P'+str(j):>9}" for j in PIL) + f"{'pi':>10}")
for a, i in enumerate(PIL):
    print(f"  {'P'+str(i):>6}" + "".join(f"{P[a, b]:>9.3f}" for b in range(NP_))
          + f"{pi[a]:>10.3f}")
ordre = [PIL[c] for c in np.argsort(-pi)]
print(f"\n  Loi stationnaire : le saut sejourne surtout en "
      + ", ".join(f"P{j} ({pi[COL[j]]:.0%})" for j in ordre[:3]) + ".")
print("  Lecture : pi est un profil de RECEPTACLE (ou la marche s'accumule), a ne pas")
print("  confondre avec l'ordre des SOURCES (ROOT : P1>P4). Voir 34, l'aplatissement a u=0.")

# =====================================================================================
# LES HYPOTHESES DE LA CHAINE, VERIFIEES ET NON SUPPOSEES. Caroline Hillairet a demande, au
# point du 13 aout, d'expliciter la quantite testee et les hypotheses sous-jacentes de la
# demarche dite de « martingalisation ». L'existence et l'unicite de pi, sur lesquelles tout
# repose ensuite, tiennent a l'irreductibilite et a l'aperiodicite : autant les mesurer.
# =====================================================================================
_hors_diag = P[~np.eye(NP_, dtype=bool)]
_irr = bool(np.all(np.linalg.matrix_power(np.eye(NP_) + P, NP_ - 1) > 0))
_ap2 = bool(np.all(np.diag(np.linalg.matrix_power(P, 2)) > 0))
_ap3 = bool(np.all(np.diag(np.linalg.matrix_power(P, 3)) > 0))
print("\n  HYPOTHESES DE LA CHAINE, mesurees :")
print(f"    diagonale nulle (pas de boucle sur place)        : {bool(np.all(np.diag(P) == 0))}")
print(f"    entrees hors diagonale > 0                       : "
      f"{int((_hors_diag > 0).sum())} sur {_hors_diag.size}")
print(f"    IRREDUCTIBLE (toutes entrees de (I+P)^4 > 0)     : {_irr}")
print(f"    APERIODIQUE (cycles de longueur 2 et 3 presents) : {_ap2 and _ap3}")
print("    => pi existe et est UNIQUE. Ce n'est donc pas une hypothese de commodite.")
print(f"\n  ET L'HYPOTHESE LA PLUS LOURDE, QU'IL FAUT NOMMER : cette chaine est SANS MEMOIRE,")
print("  alors que la cascade effectivement simulee est AUTO-EVITANTE, un pilier deja tombe")
print("  eteignant la propagation. Le processus reel n'est donc pas markovien sur les cinq")
print(f"  piliers : il l'est sur le couple (pilier courant, ensemble deja tombe), soit "
      f"{NP_} x {2 ** NP_} = {NP_ * 2 ** NP_} etats.")
print("  L'analyse de reversibilite porte donc sur une PROJECTION markovienne du processus.")
print("  Ce qu'elle capture : la direction des transferts. Ce qu'elle ne capture pas :")
print("  l'extinction et l'epuisement des piliers deja touches.")

titre("(1) Courant, reversibilite, renversement du temps")
rev_res = float(np.abs(J).max())
print(f"  Courant maximal max|J_ij| = {rev_res:.4f} > 0  =>  chaine IRREVERSIBLE :")
print("  le modele porte bien une direction (equilibre detaille NON satisfait).")
print("  Les trois plus forts courants nets (sens i -> j si J_ij > 0) :")
pairs = sorted(((J[a, b], PIL[a], PIL[b]) for a in range(NP_) for b in range(a + 1, NP_)),
               key=lambda x: -abs(x[0]))
for val, i, j in pairs[:3]:
    src, dst = (i, j) if val > 0 else (j, i)
    print(f"     P{src} -> P{dst}   |J| = {abs(val):.4f}")
err_s = float(np.abs(Ps - reversed_chain(Ps, pi)).max())     # P_s = son propre renverse
Pa_of_Pt = 0.5 * (Pt - reversed_chain(Pt, pi))               # courant de la chaine renversee
err_a = float(np.abs(Pa_of_Pt + Pa).max())                   # doit valoir -Pa
print(f"\n  Renversement du temps : partie reversible P_s invariante "
      f"(ecart {err_s:.2e}),")
print(f"  courant P_a change de signe sous renversement (ecart a -P_a : {err_a:.2e}).")
print(f"  Courant de la chaine symetrisee P_s : "
      f"{float(np.abs(current(Ps, pi)).max()):.2e} (nul, elle est reversible).")

titre("(2) La fluctuation ne voit que S : la forme de Dirichlet ignore le courant")
rng = np.random.default_rng(20260727)
worst = 0.0
for _ in range(2000):
    f = rng.standard_normal(NP_)
    e_full = dirichlet(P, pi, f)
    e_sym = dirichlet(Ps, pi, f)
    worst = max(worst, abs(e_full - e_sym))
print("  Pour 2000 fonctions test f tirees au hasard :")
print(f"  ecart maximal |E_P(f,f) - E_Ps(f,f)| = {worst:.2e}  (numeriquement nul).")
print("  L'energie de la martingale (la fluctuation, la variation quadratique) ne depend")
print("  QUE de la partie reversible S. La partie antisymetrique A = le courant = la")
print("  derive ; c'est exactement ce que la martingalisation retire. La donnee de")
print("  co-occurrence mesure une fluctuation : elle voit S, jamais A.")

titre("(3) La direction = production d'entropie ; le placebo teste sigma = 0")
sigma_sym = entropy_production(Ps, pi)
print(f"  Production d'entropie de la chaine posee   : sigma   = {sigma:.4f}  (> 0, dirigee)")
print(f"  Production d'entropie de la chaine symetrisee : sigma_s = {sigma_sym:.2e}  (= 0, reversible)")
print("  L'irreversibilite tient dans ce seul scalaire. Le placebo directionnel du memoire")
print("  (asymetrie 119 SOUS le nul de permutation 128 +/- 27, z = -0.33) se relit alors")
print("  comme un test de sigma = 0 : la co-occurrence observee est COMPATIBLE avec une")
print("  dynamique reversible. Ce n'est pas un echec de calibration, c'est une limite de")
print("  principe : une statistique invariante par renversement du temps ne peut pas")
print("  identifier un courant qui, lui, change de signe sous ce renversement.")

titre("Pont avec le S/A Euclidien du script 32 (cas pi uniforme)")
frac_eucl = np.linalg.norm(A_eucl) / np.linalg.norm(W)
print(f"  Part antisymetrique Euclidienne ||A||/||W|| = {frac_eucl:.1%} de la matrice de contagion.")
print(f"  Ecart entre pi et l'uniforme (1/5) : max |pi_i - 0,2| = {float(np.abs(pi - 0.2).max()):.3f}.")
print("  Le S = (W+W^T)/2 du memoire est la version pi-uniforme du courant pondere ci-dessus :")
print("  meme objet, meme conclusion, la version ponderee en donne la lecture probabiliste propre.")

titre("VERDICT (faisabilite pour Hugo)")
print("  Le parallele est FAISABLE et il RENFORCE la these d'identification partielle :")
print("   - S <-> partie reversible (fluctuation, martingale) : identifiee par la co-occurrence ;")
print("   - A <-> courant irreversible (derive, compensateur) : non identifie, car il est")
print("     precisement ce qui s'annule sous la symetrisation par renversement du temps ;")
print("   - le placebo z = -0.33 = donnee compatible avec sigma = 0 (equilibre detaille).")
print("  A assumer explicitement (esprit Vasicheck) : produit scalaire pondere par pi, et")
print("  emprunt d'analogie (W n'est pas un generateur de Markov). C'est une lecture qui")
print("  fonde la limite sur un theoreme, pas une hypothese de plus.")

# ============================================================ figure Z11
# TROIS PANNEAUX EN LIGNE, TRACES A LA TAILLE D'IMPRESSION. Empiles, ils remplissaient une
# page entiere du memoire (21,9 cm) et la matrice flottait dans un grand blanc. La figure
# s'imprime par \figover a 18,5 cm, soit 7,3 pouces : tracee a 7,4 pouces, une police de
# s points s'imprime a s points environ, donc les tailles ecrites ici sont celles de la page.
# Aucun titre general dans l'image : la legende LaTeX porte le titre.
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 8,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 8.5,
})
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"
T_PAN = 9.2                                # titres de panneau
# Carte divergente faite des valeurs de la charte (poles et paliers clairs de DIVERGENT), le
# fond de figure au point neutre : les courants proches de zero restent sur fond clair et
# leurs valeurs lisibles. Rouge pour un courant positif, bleu pour un negatif.
DIV = mpl.colors.LinearSegmentedColormap.from_list(
    "courant", ["#204993", "#8fa8d8", "#fcfcfb", "#e8a0aa", "#a6002e"])


def _signe(v):
    """Valeur signee a deux decimales, virgule decimale et vrai signe moins."""
    return f"{v:+.2f}".replace(".", ",").replace("-", "−")


_fr = mticker.FuncFormatter(lambda v, _p: _signe(v))

# DEUX PANNEAUX ET NON TROIS, DEPUIS LE 17 SEPTEMBRE 2026. L'ancien panneau (b) tracait des
# barres codees en dur (1 et 1, 1 et 0), et l'ancien (c) n'etait que du texte : aucun des deux
# ne portait une donnee. Le (b) trace desormais les energies CALCULEES sur des fonctions test.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), layout="constrained",
                               gridspec_kw=dict(width_ratios=[1.1, 1.0]))
fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.06)

# (a) le courant J : matrice divergente, valeurs dans les cases
vmax = np.abs(J).max()
im = ax1.imshow(J, cmap=DIV, vmin=-vmax, vmax=vmax)
# UNE MATRICE EST CARREE, PAS LE PANNEAU QUI LA PORTE : ancrage central.
ax1.set_anchor("C")
ax1.set_xticks(range(NP_)); ax1.set_yticks(range(NP_))
ax1.set_xticklabels([f"P{j}" for j in PIL]); ax1.set_yticklabels([f"P{j}" for j in PIL])
ax1.tick_params(length=0, pad=2)
for _sp in ax1.spines.values():
    _sp.set_visible(False)
ax1.set_ylabel("pilier source $j$")
ax1.set_xlabel("pilier cible $k$")
for a in range(NP_):
    for b in range(NP_):
        if a != b:
            ax1.text(b, a, _signe(J[a, b]), ha="center", va="center",
                     fontsize=7.2, color=INK if abs(J[a, b]) < 0.6 * vmax else "#fcfcfb")
ax1.set_title("(a)  Le courant $J_{jk}$ :\nla direction, non identifiée",
              fontsize=T_PAN, color=INK)
cb = fig.colorbar(im, ax=ax1, fraction=0.05, pad=0.03, aspect=16)
cb.ax.tick_params(labelsize=7.2, length=2, color="#dcdcdc")
cb.ax.yaxis.set_major_formatter(_fr)
cb.outline.set_edgecolor("#dcdcdc")
cb.outline.set_linewidth(0.6)

# (b) la fluctuation ne voit que S : energies calculees, chaine posee contre symetrisee.
# Generateur DISTINCT de celui des impressions, pour ne pas deplacer la sortie versionnee.
rng_fig = np.random.default_rng(20260917)
e_p, e_s = [], []
for _ in range(300):
    f = rng_fig.standard_normal(NP_)
    e_p.append(dirichlet(P, pi, f))
    e_s.append(dirichlet(Ps, pi, f))
e_p, e_s = np.array(e_p), np.array(e_s)
hi = 1.05 * max(e_p.max(), e_s.max())
ax2.plot([0, hi], [0, hi], color="#dcdcdc", lw=1.2, zorder=1)
ax2.scatter(e_s, e_p, s=9, color=BLUE, alpha=0.75, edgecolor="none", zorder=2,
            label="300 fonctions test $f$")
ax2.set_xlim(0, hi); ax2.set_ylim(0, hi)
_virg = mticker.FuncFormatter(lambda v, _p: f"{v:.1f}".replace(".", ","))
ax2.xaxis.set_major_formatter(_virg); ax2.yaxis.set_major_formatter(_virg)
ax2.set_xlabel("énergie, chaîne symétrisée $\\mathcal{E}_{S}(f,f)$")
ax2.set_ylabel("énergie, chaîne posée $\\mathcal{E}_{P}(f,f)$")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.text(0.04, 0.96,
         f"production d'entropie\nchaîne posée : {sigma:.4f}\nchaîne symétrisée : 0".replace(".", ","),
         transform=ax2.transAxes, ha="left", va="top", fontsize=7.8, color=INK2)
ax2.legend(frameon=False, fontsize=7.5, loc="lower right", handletextpad=0.3)
ax2.set_title("(b)  L'énergie ne voit que $S$ ;\nseule l'entropie voit la direction",
              fontsize=T_PAN, color=INK)

outdir = os.path.join(os.path.dirname(_HERE), "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z11_reversibilite_martingale.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
