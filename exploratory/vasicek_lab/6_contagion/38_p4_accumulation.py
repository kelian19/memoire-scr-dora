#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
38 : le pilier tiers (P4) comme noeud d'ACCUMULATION, pas comme noeud de cascade.

Les cinq piliers ne sont pas cinq exemplaires du meme risque. P4 (prestataires tiers) est
qualitativement different : une defaillance d'un prestataire PARTAGE degrade plusieurs
piliers EN MEME TEMPS (dependance commune), au lieu de se propager de proche en proche.
C'est le mecanisme de MOVEit (191 victimes d'un seul choc fournisseur, script 08d).

DEUX MECANISMES A NE PAS CONFONDRE :
  - cascade dirigee : un incident sur P4 en ENTRAINE d'autres, sequentiellement, selon W ;
  - choc commun P4  : un prestataire partage tombe et frappe SIMULTANEMENT un ensemble de
    piliers, sans ordre. La severite est la somme des piliers touches, comme pour la
    cascade, mais l'ENSEMBLE atteint suit une loi differente.

CE QUI FAIT DE P4 UN CAS A PART, ET COHERENT AVEC LE FIL DU MEMOIRE. Le choc commun est une
CO-OCCURRENCE SYMETRIQUE (P4 et P_j tombent ensemble, sans direction). Or la co-occurrence
est la partie IDENTIFIEE du modele (le S de la decomposition W = S + A, chapitre 10), celle
que la donnee voit (MOVEit). P4 est donc le seul pilier ou la donnee soutient une structure
d'accumulation, par opposition a la direction, non identifiee.

PARAMETRE, ET SON STATUT. L'intensite du choc commun au sein d'UNE entite (combien de piliers
un prestataire partage fait tomber d'un coup) n'est pas identifiee : le batch de 08d est un
phenomene de portefeuille, pas d'entite. On l'ancre donc sur la concentration cloud
gamma = 0,68 (proxy AWS+Azure+GCP, deja dans le modele) et on BORNE le SCR sur phi_cs dans
[0, gamma], phi_cs etant la probabilite qu'un pilier soit emporte simultanement par le choc.

Sortie : diagnostics + figure Z9_p4_accumulation.png.
"""

import os
import sys
from itertools import combinations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                                 # noqa: E402
from euro_cascade_model import PARAMS, var                      # noqa: E402
from src.aggregation.lda import simulate_remediation_severity   # noqa: E402
import scr_engine as eng                                        # noqa: E402
import partial_id as pid                                        # noqa: E402

WID = 80
PIL = eng.PIL
P4 = 4
SRC = "OPRISK"
sp = PARAMS[SRC]
NY = 40_000
SEED = 20260721
GAMMA = 0.68                       # concentration cloud (proxy), borne haute de phi_cs
G_BASE = 0.90


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# attractivite symetrique (partie IDENTIFIEE S de W) : poids de co-occurrence des piliers
P_expert = pid.expert_matrix(G_BASE)
S_sym, _ = pid.decompose(P_expert)
_col = {j: c for c, j in enumerate(PIL)}
ATTR = {j: float(S_sym[:, _col[j]].sum()) for j in PIL}         # attractivite de la cible j
_others = [j for j in PIL if j != P4]
_wtot = sum(ATTR[j] for j in _others)
ATTR_N = {j: ATTR[j] / _wtot for j in _others}                  # poids normalises hors P4


def table_amorce(j, g):
    """(ind [nsets x 5], probs) de l'ensemble atteint depuis j par la cascade dirigee."""
    dist = eng.cascade_set_dist(j, g)
    sets = list(dist.keys())
    probs = np.array([dist[s] for s in sets])
    ind = np.zeros((len(sets), 5))
    for r, s in enumerate(sets):
        for p in s:
            ind[r, _col[p]] = 1.0
    return ind, probs / probs.sum()


def table_p4_choc_commun(phi_cs):
    """Ensemble atteint depuis P4 sous CHOC COMMUN : P4 + chaque autre pilier inclus
    INDEPENDAMMENT avec proba phi_cs (dependance partagee), simultanement. Loi exacte."""
    sets, probs = [], []
    for r in range(len(_others) + 1):
        for combo in combinations(_others, r):
            p = 1.0
            for k in _others:
                p *= phi_cs if k in combo else (1.0 - phi_cs)
            S = (P4,) + combo
            ind = np.zeros(5)
            for x in S:
                ind[_col[x]] = 1.0
            sets.append(ind)
            probs.append(p)
    return np.array(sets), np.array(probs)


# nombres aleatoires communs (frequence, amorces, severites) : comparaisons a bruit egal
_rng = np.random.default_rng(SEED)
lam = sp["lam_ref"]
r = lam / (ec.PHI - 1.0)
counts = _rng.negative_binomial(r, r / (r + lam), size=NY)
T = int(counts.sum())
year_of = np.repeat(np.arange(NY), counts)
w_am = np.array([eng.LAMBDA[j] for j in PIL], float)
amorce = _rng.choice(5, size=T, p=w_am / w_am.sum())
U_set = _rng.random(T)                                          # pour tirer l'ensemble
SEV = simulate_remediation_severity(T * 5, sp["xi"], sp["sigma"], sp["u"], sp["p_u"],
                                    sp["cap"], _rng).reshape(T, 5)
idx_by_am = [np.where(amorce == c)[0] for c in range(5)]
print(f"tirage commun : {NY} annees, {T} incidents, part d'amorce P4 = "
      f"{len(idx_by_am[_col[P4]])/T:.1%}")


def scr_with(tables_p4):
    """SCR (VaR 99,5 %), P1-P3,P5 en cascade, P4 selon la table fournie (cascade ou choc)."""
    base_tables = {j: table_amorce(j, G_BASE) for j in PIL}
    base_tables[P4] = tables_p4
    annual = np.zeros(NY)
    for c, j in enumerate(PIL):
        idx = idx_by_am[c]
        if idx.size == 0:
            continue
        ind, probs = base_tables[j]
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        sel = np.searchsorted(cdf, U_set[idx], side="right")
        np.clip(sel, 0, len(probs) - 1, out=sel)
        touched = ind[sel]                                      # (n_inc, 5) indicateurs
        loss = (SEV[idx] * touched).sum(axis=1)
        annual += np.bincount(year_of[idx], weights=loss, minlength=NY)
    return float(np.quantile(annual, 0.995))


# =====================================================================================
titre("Reference : P4 traite comme un noeud de cascade ordinaire")
# =====================================================================================
scr_cascade = scr_with(table_amorce(P4, G_BASE))
print(f"  SCR (P4 = noeud cascade) = {scr_cascade:.0f} M")

# =====================================================================================
titre("P4 en CHOC COMMUN : bornes du SCR sur l'intensite phi_cs dans [0, gamma]")
# =====================================================================================
phis = np.linspace(0.0, GAMMA, 13)
scr_cs = np.array([scr_with(table_p4_choc_commun(p)) for p in phis])
print(f"  {'phi_cs':>8}{'SCR':>10}{'ecart / cascade':>18}{'E[|S| | P4]':>14}")
for p, s in zip(phis, scr_cs):
    esize = 1 + len(_others) * p
    print(f"  {p:>8.2f}{s:>9.0f} M{100*(s/scr_cascade-1):>16.1f} %{esize:>14.2f}")
lo, hi = scr_cs.min(), scr_cs.max()
print(f"\n  Sur phi_cs dans [0 ; {GAMMA}] : SCR dans [{lo:.0f} ; {hi:.0f}] M.")
print(f"  A phi_cs = 0, le choc commun se reduit a P4 seul (aucune simultaneite) et le SCR")
print(f"  passe MEME SOUS la cascade dirigee ({scr_cs[0]:.0f} contre {scr_cascade:.0f}) :")
print(f"  la cascade, elle, propage vers d'autres piliers. A phi_cs = gamma, la dependance")
print(f"  partagee emporte en moyenne {1+len(_others)*GAMMA:.1f} piliers d'un coup, et le SCR")
print(f"  depasse la cascade de {100*(hi/scr_cascade-1):.0f} %.")

# =====================================================================================
titre("Ce que le choc commun ajoute que la cascade dirigee ne voit pas")
# =====================================================================================
# proba qu'un incident P4 touche >= 3 piliers, cascade vs choc commun a phi = gamma
def p_ge3(ind, probs):
    k = ind.sum(axis=1)
    return float(probs[k >= 3].sum())


ind_c, pr_c = table_amorce(P4, G_BASE)
ind_s, pr_s = table_p4_choc_commun(GAMMA)
print(f"  P(un incident P4 touche >= 3 piliers) :")
print(f"    cascade dirigee : {p_ge3(ind_c, pr_c):.3f}")
print(f"    choc commun (phi=gamma) : {p_ge3(ind_s, pr_s):.3f}")
print("  La simultaneite cree de la CO-OCCURRENCE multiple que la propagation sequentielle")
print("  produit rarement. C'est une structure SYMETRIQUE (pas de direction), donc dans la")
print("  partie IDENTIFIEE du modele (le S de W = S + A) : la donnee la voit (MOVEit).")

# =====================================================================================
titre("Verdict")
# =====================================================================================
print("  1. P4 n'est pas un noeud de cascade comme les autres : la bonne maille est le CHOC")
print("     COMMUN (dependance partagee), pas la propagation dirigee. Les deux mecanismes")
print("     coexistent, mais pour P4 l'accumulation domine.")
print(f"  2. Son intensite phi_cs n'est pas identifiee pour une entite isolee ; bornee sur")
print(f"     [0, gamma], elle place le SCR dans [{lo:.0f} ; {hi:.0f}] M. Encore une bande,")
print(f"     pas un point.")
print("  3. Contrairement a la direction, l'accumulation de P4 est SYMETRIQUE, donc")
print("     IDENTIFIEE : c'est le seul pilier ou la donnee (MOVEit, concentration cloud)")
print("     soutient une structure. A ecrire comme tel : la faiblesse d'identification")
print("     de W ne s'applique pas a l'accumulation de P4.")

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.9, 3.3))

ax1.plot(phis, scr_cs, color=BLUE, lw=2.2, marker="o", ms=4, zorder=3)
ax1.axhline(scr_cascade, color=ACCENT, lw=2, ls="--", label=f"P4 = nœud cascade : {scr_cascade:.0f}")
ax1.axvline(GAMMA, color=MUTED, lw=1, ls=":")
ax1.text(GAMMA, ax1.get_ylim()[0], " γ=0,68", fontsize=8, color=MUTED, va="bottom")
ax1.set_xlabel("intensité du choc commun $\\varphi_{cs}$", color=INK2)
ax1.set_ylabel("SCR (VaR 99,5 %, M€)", color=INK2)
ax1.set_title("(a)  P4 en accumulation : SCR borné\nsur l'intensité du choc partagé",
              fontsize=11, color=INK, pad=8)
ax1.legend(frameon=False, fontsize=8, loc="upper left")

ax2.bar([0, 1], [p_ge3(ind_c, pr_c), p_ge3(ind_s, pr_s)], width=0.55,
        color=[ACCENT, BLUE], alpha=0.85)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["cascade\ndirigée", "choc commun\n$\\varphi=\\gamma$"], fontsize=9)
ax2.set_ylabel("P(incident P4 touche ≥ 3 piliers)", color=INK2, fontsize=9.5)
ax2.set_title("(b)  La simultanéité crée une\nco-occurrence multiple (symétrique)",
              fontsize=11, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# PAS DE TITRE GENERAL : la legende LaTeX du memoire porte le titre de la figure.
fig.tight_layout()
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z9_p4_accumulation.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
