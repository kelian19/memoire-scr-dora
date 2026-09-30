# -*- coding: utf-8 -*-
"""Noyau d'IDENTIFICATION PARTIELLE de la contagion dirigee W.

Source unique du dispositif, partagee par les scripts 30 (bornes de capital) et
31 (comparaison des deux regimes). Rien ici ne depend d'une elicitation.

L'IDEE. Toute matrice se decompose de facon unique en
        W = S + A,   S = (W + W^T)/2 symetrique,   A = (W - W^T)/2 antisymetrique.
La donnee identifie la CO-OCCURRENCE (deux piliers touches ensemble), donc S. Elle
n'identifie PAS la DIRECTION, donc A (placebo z = -0,33, 0/18 aretes significatives).
L'ensemble admissible est alors

        W(A) = S + A,  A antisymetrique,  |A_jk| <= t * S_jk,  W sous-critique,

la borne |A_jk| <= S_jk venant de la seule positivite (W_jk >= 0 ET W_kj >= 0).
Le parametre t dans [0,1] mesure ce que la donnee ignore de la direction : t = 1 est
l'ignorance totale (etat actuel), t = 0 la direction connue nulle.

L'EVALUATION. A nombres aleatoires COMMUNS : le tirage des annees, des amorces, des
uniformes et des severites est fait UNE fois, puis reutilise pour toute matrice W.
Seule la loi du nombre de piliers touches change avec W. Les comparaisons entre W sont
donc quasi exemptes de bruit Monte-Carlo, ce qui est indispensable pour des bornes
(sinon min et max ramassent du bruit et les bornes sont artificiellement larges).

MISE EN GARDE DE LECTURE. A indice de queue eleve, la queue annuelle obeit au principe
de la perte unique dominante : la VaR est PEU sensible a la structure de cascade (le
script 29 l'avait deja constate entre marche et branchement). L'evaluateur renvoie donc
TOUJOURS le couple (VaR, moyenne), et la moyenne est l'estimateur a privilegier pour
lire un effet de structure.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CASCADE = os.path.abspath(os.path.join(_HERE, "..", "cascade_qualitative"))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_REPO, _CASCADE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cascade_model import TRANS                                 # noqa: E402
import euro_cascade_model as ec                                 # noqa: E402
from euro_cascade_model import PARAMS                           # noqa: E402
from src.aggregation.lda import simulate_remediation_severity   # noqa: E402

PIL = [1, 2, 3, 4, 5]
NP_ = len(PIL)
IU = np.triu_indices(NP_, k=1)          # 10 degres de liberte antisymetriques
NFREE = len(IU[0])
G_DEFAULT = 0.90                         # amplitude de contagion (le 25 balaie cet axe)


# ============================================================ la matrice et sa decomposition
def expert_matrix(g=G_DEFAULT):
    """P_jk = g * TRANS[j][k] / max_s : proba que j infecte DIRECTEMENT k."""
    srow = {j: sum(TRANS[j].values()) for j in PIL}
    maxs = max(srow.values())
    P = np.zeros((NP_, NP_))
    for a, j in enumerate(PIL):
        for b, k in enumerate(PIL):
            if j != k:
                P[a, b] = g * TRANS[j].get(k, 0.0) / maxs
    return P


def decompose(P):
    """(S identifiee, A non identifiee)."""
    return (P + P.T) / 2.0, (P - P.T) / 2.0


def build_W(S, a_vec):
    """W = S + A, A antisymetrique reconstruite depuis ses 10 coefficients libres."""
    A = np.zeros((NP_, NP_))
    A[IU] = a_vec
    return S + (A - A.T)


def rho(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def admissible(W):
    return bool((W >= -1e-12).all() and (W <= 1.0).all() and rho(W) <= 1.0)


def sample_vertices(S, t, n, rng):
    """Sommets TIRES du pave |A_jk| <= t*S_jk (echantillon, pour les analyses lourdes)."""
    smax = t * S[IU]
    return (rng.integers(0, 2, size=(n, NFREE)) * 2 - 1) * smax


def all_vertices(S, t):
    """TOUS les sommets du pave : 2^10 = 1024 configurations de signes.

    A privilegier pour les BORNES : le resultat est deterministe et reproductible, la
    ou un echantillon de sommets donne des bornes qui bougent d'un script a l'autre.
    Les extremes d'une reponse monotone coordonnee par coordonnee vivent aux sommets ;
    l'enumeration exhaustive les atteint donc tous, sans hypothese de tirage.
    """
    n = 1 << NFREE
    signs = np.array([[(m >> k & 1) * 2 - 1 for k in range(NFREE)] for m in range(n)],
                     dtype=float)
    return signs * (t * S[IU])


def sample_prior(S, A_centre, tau, n, rng):
    """Tirages d'un prior d'expert : gaussienne autour de A_centre, tronquee au pave."""
    smax = S[IU]
    return np.clip(rng.normal(A_centre[IU], tau * smax, size=(n, NFREE)), -smax, smax)


# ============================================================ cascade a matrice LIBRE
_BITS = [[k for k in range(NP_) if m >> k & 1] for m in range(1 << NP_)]
_POP = [len(b) for b in _BITS]


def card_dist_all(W):
    """Loi EXACTE du nombre de piliers touches, par pilier d'amorce (cascade independante).

    Depuis chaque pilier tombe j, chaque k est infecte independamment avec proba W[j,k] ;
    l'ensemble atteint est l'accessibilite dans le graphe d'aretes vivantes. Calcul exact
    par P(S=A) = R(A) * B(A), ou B(A) = prod_{j in A, k hors A} (1 - W_jk) (aucune fuite)
    et R(A) = proba que l'amorce atteigne TOUT A, par recursion sur les sous-ensembles.
    Renvoie (card[amorce, k] pour k = 0..5, setp[amorce, masque]).
    """
    L = np.log(np.clip(1.0 - W, 1e-15, 1.0))
    SL = np.zeros((NP_, 1 << NP_))
    for m in range(1 << NP_):
        if _BITS[m]:
            SL[:, m] = L[:, _BITS[m]].sum(axis=1)

    card = np.zeros((NP_, NP_ + 1))
    setp = np.zeros((NP_, 1 << NP_))
    for a in range(NP_):
        R = {}
        masks = sorted((m for m in range(1 << NP_) if m >> a & 1), key=lambda m: _POP[m])
        for m in masks:
            if _POP[m] == 1:
                R[m] = 1.0
                continue
            tot = 0.0
            sub = (m - 1) & m
            while sub:
                if sub >> a & 1:
                    tot += R[sub] * float(np.exp(SL[_BITS[sub], m & ~sub].sum()))
                sub = (sub - 1) & m
            R[m] = 1.0 - tot
        for m in masks:
            comp = ((1 << NP_) - 1) & ~m
            p = R[m] * (float(np.exp(SL[_BITS[m], comp].sum())) if comp else 1.0)
            setp[a, m] = p
            card[a, _POP[m]] += p
        s = card[a].sum()
        if s > 0:
            card[a] /= s
            setp[a] /= s
    return card, setp


# ============================================================ evaluateur a nombres communs
class Evaluator:
    """SCR et perte moyenne comme fonctions de W, a nombres aleatoires communs."""

    def __init__(self, source="OPRISK", n_years=40_000, seed=20260721, alpha=0.995,
                 lam=None, sev_source=None):
        """`source` fixe la frequence de reference, `sev_source` la loi de severite.

        Les dissocier permet d'isoler ce qui fait bouger le NIVEAU du capital (la loi de
        severite, la frequence) de ce qui fait bouger sa STRUCTURE (la cascade). C'est
        l'experience du script 33.
        """
        sp = PARAMS[sev_source or source]
        self.sp, self.alpha, self.n_years = sp, alpha, n_years
        self.lam = float(PARAMS[source]["lam_ref"] if lam is None else lam)
        rng = np.random.default_rng(seed)
        r = self.lam / (ec.PHI - 1.0)
        counts = rng.negative_binomial(r, r / (r + self.lam), size=n_years)
        self.T = int(counts.sum())
        self.year_of = np.repeat(np.arange(n_years), counts)
        w = np.array([ec.eng.LAMBDA[j] for j in PIL], float)
        self.amorce = rng.choice(NP_, size=self.T, p=w / w.sum())
        self.u = rng.random(self.T)
        sev = simulate_remediation_severity(self.T * NP_, sp["xi"], sp["sigma"], sp["u"],
                                            sp["p_u"], sp["cap"], rng).reshape(self.T, NP_)
        self.cum = np.cumsum(sev, axis=1)
        self.idx_by_am = [np.where(self.amorce == a)[0] for a in range(NP_)]
        self._ar = np.arange(self.T)

    def from_card(self, card):
        K = np.empty(self.T, dtype=np.int64)
        for a in range(NP_):
            idx = self.idx_by_am[a]
            if idx.size == 0:
                continue
            cdf = np.cumsum(card[a][1:])
            cdf[-1] = 1.0
            K[idx] = np.searchsorted(cdf, self.u[idx], side="right") + 1
        np.clip(K, 1, NP_, out=K)
        annual = np.bincount(self.year_of, weights=self.cum[self._ar, K - 1],
                             minlength=self.n_years)
        return float(np.quantile(annual, self.alpha)), float(annual.mean())

    def __call__(self, W):
        return self.from_card(card_dist_all(W)[0])

    def scr(self, W):
        return self(W)[0]

    def benefits(self, W):
        """Gain de SCR a remedier chaque pilier (couper ses aretes SORTANTES)."""
        base = self.scr(W)
        out = np.empty(NP_)
        for j in range(NP_):
            Wr = W.copy()
            Wr[j, :] = 0.0
            out[j] = base - self.scr(Wr)
        return out
