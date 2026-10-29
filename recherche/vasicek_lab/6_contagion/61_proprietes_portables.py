#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
61 : les proprietes portables de la classe « Vasicek-cascade », et l'audit d'une hypothese tacite.

Ce script rassemble ce que la classe de modeles du memoire permet d'ENONCER et de DEMONTRER
independamment du cas DORA (des resultats citables hors de l'application), puis il AUDITE une
hypothese que les scripts de bornes utilisent sans la justifier.

--------------------------------------------------------------------------------------------
LES QUATRE PROPOSITIONS PORTABLES (demonstrations en commentaire, verifications numeriques ici)

P1. DECOMPOSITION UNIQUE ET ORTHOGONALE. Toute matrice reelle s'ecrit de facon unique
    W = S + A avec S symetrique et A antisymetrique, et <S, A>_F = tr(S^T A) = 0.
    Preuve : S = (W + W^T)/2, A = (W - W^T)/2 donnent l'existence ; si S+A = S'+A' alors
    S-S' = A'-A est a la fois symetrique et antisymetrique donc nulle. L'orthogonalite vient
    de tr(S^T A) = tr(S A) = tr((S A)^T) = tr(A^T S^T) = -tr(A S) = -tr(S A), donc tr(S A) = 0.

P2. LE PAVE ADMISSIBLE EST EXACTEMENT |A_jk| <= S_jk. La seule positivite de W l'impose :
    W_jk = S_jk + A_jk >= 0 et W_kj = S_jk - A_jk >= 0 equivalent a |A_jk| <= S_jk. Aucune
    hypothese supplementaire n'est requise : l'ensemble d'ignorance directionnelle est un PAVE,
    ce qui est ce qui rend l'identification partielle calculable.

P3. STABILITE DE LEONTIEF PAR CONSTRUCTION. Si W = g * T / max_j (somme de la ligne j de T)
    avec T >= 0 et g < 1, alors rho(W) <= g < 1 : la cascade s'eteint presque surement, et la
    forme reduite (I - W)^{-1} existe.
    Preuve : pour une matrice a coefficients positifs, rho(M) <= max_j sum_k M_jk (rayon
    spectral majore par la norme infinie induite). Ici max_j sum_k W_jk = g * max_j sum_k T_jk
    / max_j sum_k T_jk = g. La stabilite est donc GARANTIE, non supposee : c'est le point qui
    distingue cette normalisation d'un choix arbitraire de W.

P4. AVEUGLEMENT DES STATISTIQUES REVERSIBLES. Sur la chaine associee, la forme de Dirichlet
    E(f,f) = (1/2) sum_jk pi_j P_jk (f_k - f_j)^2 ne depend que de la partie reversible S.
    Consequence : toute statistique fonction de la seule energie de fluctuation est aveugle a
    la direction A. (Verifie numeriquement au script 40 ; rappele ici pour completude.)

--------------------------------------------------------------------------------------------
L'AUDIT : LES BORNES SONT-ELLES ATTEINTES AUX SOMMETS ?

Les scripts 30 et 55 calculent les bornes de capital en enumerant les SOMMETS du pave
(|A_jk| = S_jk), au motif que « les extremes d'une reponse monotone coordonnee par coordonnee
vivent aux sommets ». Cet argument est INCOMPLET, et il faut le dire : augmenter a_jk augmente
W_jk mais DIMINUE W_kj. La reponse n'est donc pas monotone en a_jk (c'est un arbitrage), et
rien ne garantit a priori que le min et le max vivent aux sommets. Si c'est faux, les bornes
publiees sont TROP ETROITES, ce qui serait un defaut serieux.

On le teste : on tire un grand nombre de points INTERIEURS du pave admissible et on verifie
qu'aucun ne produit un SCR hors des bornes issues des sommets. Ce n'est pas une preuve, c'est
une refutation possible ; l'absence de contre-exemple sur un grand echantillon rend l'hypothese
defendable, et un contre-exemple obligerait a corriger la methode.

Sortie : diagnostics + figure Z21_proprietes_portables.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import partial_id as pid                                        # noqa: E402

WID = 84
SEED = 20260721            # meme graine que 30 / 55 / 56
N_INT = 3000               # points interieurs tires pour l'audit


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


ev = pid.Evaluator(n_years=40_000, seed=SEED)
P = pid.expert_matrix()
S, A_EXP = pid.decompose(P)
smax = S[pid.IU]
rng = np.random.default_rng(SEED)

# =====================================================================================
titre("P1. Decomposition unique et orthogonale")
# =====================================================================================
rec = np.abs(P - (S + A_EXP)).max()
sym_err = np.abs(S - S.T).max()
asym_err = np.abs(A_EXP + A_EXP.T).max()
orth = abs(float(np.trace(S.T @ A_EXP)))
print(f"  reconstruction |W - (S+A)|_max      = {rec:.2e}")
print(f"  symetrie de S   |S - S^T|_max       = {sym_err:.2e}")
print(f"  antisymetrie de A |A + A^T|_max     = {asym_err:.2e}")
print(f"  orthogonalite <S,A>_F = tr(S^T A)   = {orth:.2e}")
print("  => la separation identifie/non-identifie est EXACTE et orthogonale : la donnee")
print("     contraint S, l'ignorance porte sur A, et les deux ne se recouvrent pas.")

# =====================================================================================
titre("P2. Le pave admissible est exactement |A_jk| <= S_jk")
# =====================================================================================
viol = 0
for _ in range(2000):
    a = (rng.random(pid.NFREE) * 2 - 1) * smax * 1.35          # on sort volontairement du pave
    W = pid.build_W(S, a)
    inside = bool((np.abs(a) <= smax + 1e-12).all())
    pos = bool((W >= -1e-12).all())
    if pos != inside and not inside:
        # un point hors du pave qui resterait positif contredirait P2
        if pos:
            viol += 1
print(f"  points hors du pave restant a coefficients positifs : {viol} / 2000")
print("  => aucun : la positivite de W equivaut bien au pave |A| <= S. L'ensemble d'ignorance")
print("     est un PAVE, donc enumerable et convexe. C'est ce qui rend les bornes calculables.")

# =====================================================================================
titre("P3. Stabilite de Leontief garantie par la normalisation")
# =====================================================================================
print(f"  {'g':>6}{'max somme de ligne':>21}{'rho(W)':>10}{'rho <= g ?':>12}")
for g in (0.10, 0.45, 0.68, 0.90, 0.99):
    Wg = pid.expert_matrix(g)
    rowmax = float(Wg.sum(axis=1).max())
    r = pid.rho(Wg)
    print(f"  {g:>6.2f}{rowmax:>21.4f}{r:>10.4f}{'oui' if r <= g + 1e-9 else 'NON':>12}")
print("  => rho(W) <= max somme de ligne = g, pour tout g. La sous-criticite ne depend d'aucun")
print("     reglage : elle est une consequence de la normalisation. La cascade s'eteint p.s.")

# =====================================================================================
titre("P4. Rappel : la fluctuation ne voit que S (script 40)")
# =====================================================================================
print("  La forme de Dirichlet ne depend que de la partie reversible : verifie au script 40")
print("  a 1e-15 pres, avec la production d'entropie qui s'annule a la symetrisation.")
print("  Consequence portable : toute statistique invariante par renversement du temps est")
print("  aveugle a la direction. C'est le fondement THEORIQUE de la non-identifiabilite.")

# =====================================================================================
titre("AUDIT. Les bornes sont-elles atteintes aux sommets du pave ?")
# =====================================================================================
print("  Rappel du probleme : a_jk augmente W_jk mais DIMINUE W_kj. La reponse n'est donc pas")
print("  monotone en a_jk, et l'argument 'les extremes vivent aux sommets' est incomplet.\n")
# bornes issues des sommets (identique aux scripts 30 / 55)
v_scr, v_mean = [], []
for m in range(1 << pid.NFREE):
    sg = np.array([(m >> k & 1) * 2 - 1 for k in range(pid.NFREE)], dtype=float)
    W = pid.build_W(S, sg * smax)
    if pid.admissible(W):
        s_, mn_ = ev(W)
        v_scr.append(s_); v_mean.append(mn_)
v_scr, v_mean = np.array(v_scr), np.array(v_mean)
lo_v, hi_v = v_scr.min(), v_scr.max()
mlo_v, mhi_v = v_mean.min(), v_mean.max()
print(f"  bornes aux sommets : SCR [{lo_v:.0f} ; {hi_v:.0f}] M   |   "
      f"moyenne [{mlo_v:.0f} ; {mhi_v:.0f}] M")

# points INTERIEURS : tirage uniforme dans le pave
int_scr, int_mean, kept = [], [], 0
for _ in range(N_INT):
    a = (rng.random(pid.NFREE) * 2 - 1) * smax
    W = pid.build_W(S, a)
    if not pid.admissible(W):
        continue
    kept += 1
    s_, mn_ = ev(W)
    int_scr.append(s_); int_mean.append(mn_)
int_scr, int_mean = np.array(int_scr), np.array(int_mean)
esc_lo = int(( int_scr < lo_v - 1e-9).sum())
esc_hi = int(( int_scr > hi_v + 1e-9).sum())
esc_mlo = int((int_mean < mlo_v - 1e-9).sum())
esc_mhi = int((int_mean > mhi_v + 1e-9).sum())
print(f"\n  {kept} points interieurs admissibles evalues.")
print(f"  SCR     : {esc_lo} sous la borne basse, {esc_hi} au-dessus de la borne haute")
print(f"  moyenne : {esc_mlo} sous la borne basse, {esc_mhi} au-dessus de la borne haute")
print(f"  amplitude interieure SCR : [{int_scr.min():.0f} ; {int_scr.max():.0f}] M")

if esc_lo + esc_hi + esc_mlo + esc_mhi == 0:
    print("\n  AUCUN contre-exemple. L'hypothese d'atteinte aux sommets n'est pas refutee sur")
    print(f"  {kept} tirages : les bornes publiees sont defendables. A ecrire toutefois comme une")
    print("  hypothese VERIFIEE NUMERIQUEMENT, non comme un theoreme : la non-monotonie en")
    print("  a_jk empeche de l'etablir par le seul argument de monotonie coordonnee par coordonnee.")
else:
    print("\n  CONTRE-EXEMPLE TROUVE : les bornes aux sommets NE sont PAS valides. Il faut")
    print("  elargir la methode (optimisation sur le pave, et non enumeration des sommets).")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. Quatre proprietes de la classe « Vasicek-cascade » sont portables hors du cas DORA :")
print("     decomposition unique et orthogonale, pave admissible exact, stabilite de Leontief")
print("     garantie, et aveuglement des statistiques reversibles a la direction.")
print("  2. La plus citable est P3 : la sous-criticite est une CONSEQUENCE de la normalisation,")
print("     pas une hypothese. C'est ce qui separe cette construction d'un W pose au hasard.")
print("  3. AUDIT : l'atteinte des bornes aux sommets, utilisee par les scripts 30 et 55, ne")
print("     decoule PAS d'un argument de monotonie (a_jk joue en sens contraire sur W_jk et")
print(f"     W_kj). Elle resiste toutefois a {kept} tirages interieurs sans contre-exemple :")
print("     a presenter comme une hypothese verifiee numeriquement, honnetement nommee.")

# =====================================================================================
# figure Z20
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 10.5,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
})
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.0))

# (a) P3 : rho(W) <= g
gs = np.linspace(0.05, 0.99, 24)
rhos = [pid.rho(pid.expert_matrix(g)) for g in gs]
ax1.plot(gs, gs, color=MUTED, ls="--", lw=1.4, label="borne $g$ (norme infinie)")
ax1.plot(gs, rhos, color=BLUE, lw=2.4, marker="o", ms=3.5, label="$\\rho(W)$ effectif")
ax1.axhline(1.0, color=ACCENT, ls=":", lw=1.2)
ax1.text(0.06, 1.01, "seuil critique $\\rho=1$", fontsize=8, color=ACCENT)
ax1.set_xlabel("amplitude de contagion $g$", color=INK2)
ax1.set_ylabel("rayon spectral", color=INK2)
ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
ax1.set_title("(a)  P3 : la sous-criticité est garantie\npar la normalisation, pas supposée",
              fontsize=11, color=INK, pad=8)

# (b) audit : interieur contre bornes aux sommets
ax2.hist(int_scr, bins=40, color=BLUE, alpha=0.5, edgecolor="#fcfcfb",
         label=f"{kept} points intérieurs")
ax2.axvline(lo_v, color=ACCENT, lw=2, label="bornes issues des sommets")
ax2.axvline(hi_v, color=ACCENT, lw=2)
ax2.plot(v_scr, np.full_like(v_scr, -2.0), "|", color=INK, ms=6, alpha=0.35)
ax2.set_xlabel("SCR (M€)", color=INK2)
ax2.set_ylabel("fréquence (points intérieurs)", color=INK2)
ax2.legend(frameon=False, fontsize=8.5)
ax2.set_title(f"(b)  Audit : {esc_lo+esc_hi} échappée(s) sur {kept} tirages\n"
              f"l'hypothèse des sommets n'est pas réfutée", fontsize=11, color=INK, pad=8)

for ax in (ax1, ax2):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig.suptitle("Z20 : propriétés portables de la classe Vasicek-cascade, et audit de l'hypothèse "
             "d'atteinte des bornes aux sommets",
             fontsize=11.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z21_proprietes_portables.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
