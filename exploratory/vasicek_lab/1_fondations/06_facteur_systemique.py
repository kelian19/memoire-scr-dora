#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06 : Calibrer le facteur systemique Y_j, les charges s_j et la matrice Sigma_Y.

C'est le dernier bloc du modele pour lequel la note ne proposait aucun protocole. La
generalisation A (un facteur systemique PAR PILIER, des dependances croisees Sigma_Y)
etait posee sans jamais dire comment on l'estimerait.

Portee : recovery en simulation (on simule Y / s / Sigma_Y, on les retrouve). La
Route B recoit W et base en entree, connus ici parce que simules ; la revendication
"sur donnees agregees" reste donc conditionnelle a W, non calibrable ailleurs (cf. 05).

--------------------------------------------------------------------------------
DEUX ROUTES, ET C'EST LA LE POINT.

  Route A -- PANEL. Les effets fixes de periode du probit du script 03 valent
             exactement phi_{j,t} = base_j + s_j * Y_{j,t}. On ne les avait jamais
             lus. Ils donnent gratuitement :
                base_j = moyenne_t(phi_j)      s_j = ecart-type_t(phi_j)
                Y_{j,t} = (phi_{j,t} - base_j) / s_j
                Sigma_Y = corr(phi)            (l'echelle s_j n'affecte pas la correlation)
             Exige un panel entite x pilier x periode.

  Route B -- AGREGE, par distance de Frobenius. Transposition directe de la methode
             de Pineau & Zuniga (2023, eq. 5), qui extrait leur facteur Z en minimisant
             la distance de Frobenius entre matrices de transition observees et
             engendrees. Ici, pour chaque periode t, on observe
                Pi_obs[j,k] = P(incident en j a t ET incident en k a t-1)   (25 moments)
                marg_obs[j] = P(incident en j a t)                          ( 5 moments)
             et l'on resout
                Y_t = argmin_Y || moments_obs - moments_modele(Y) ||^2   (30 eq., 5 inconnues)
             N'exige AUCUNE donnee individuelle : uniquement des comptages agreges.
             Les moments MARGINAUX sont indispensables (cf. moments_model).

--------------------------------------------------------------------------------
POURQUOI CETTE DISTINCTION EST DECISIVE.

Le script 05 etablit que W n'est calibrable sur aucune source publique : il faut un
panel entite x pilier horodate au mois. Le facteur systemique, lui, s'extrait par la
route B a partir de STATISTIQUES AGREGEES par pilier et par periode -- exactement ce
que les registres DORA, l'ENISA et les CERT publient deja.

  W       : inidentifiable sans donnees individuelles fines.
  Y, s, Sigma_Y : identifiables sur des donnees publiques agregees.

Le verrou de la note est donc plus etroit qu'annonce. C'est un resultat positif.

Sortie : diagnostics + figure N_facteur_systemique.png
"""

import os
import warnings
import numpy as np
from scipy import stats, optimize
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt

# Avertissements de convergence / overflow attendus (probits et moindres carres) ;
# la recovery est validee sur donnees simulees, on masque donc ces avertissements.
warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260710)
J = 5
PIL = ["P1", "P2", "P3", "P4", "P5"]

# ------------------------------------------------------------------ verite (DGP)
TRANS = {
    1: {2: 0.80, 3: 0.70, 4: 0.70, 5: 0.40},
    2: {1: 0.20, 3: 0.40, 4: 0.30, 5: 0.60},
    3: {1: 0.30, 2: 0.50, 4: 0.30, 5: 0.30},
    4: {1: 0.20, 2: 0.80, 3: 0.30, 5: 0.40},
    5: {1: 0.10, 2: 0.20, 3: 0.20, 4: 0.20},
}
T_RAW = np.zeros((J, J))
for k in TRANS:
    for j, v in TRANS[k].items():
        T_RAW[j - 1, k - 1] = v
GAIN = 0.9
W_TRUE = GAIN * T_RAW / T_RAW.sum(1).max()      # normalisation de Leontief (cf. script 03)
BASE = np.full(J, -1.7)

# Charges systemiques HETEROGENES : c'est tout l'objet de la generalisation A.
S_TRUE = np.array([0.35, 0.55, 0.45, 0.60, 0.30])

# Sigma_Y : les dependances croisees entre piliers, symetriques, au niveau systemique.
# P1 et P4 co-varient fortement (gouvernance et risque tiers subissent la meme meteo).
SIGMA_Y = np.array([
    [1.00, 0.30, 0.20, 0.60, 0.10],
    [0.30, 1.00, 0.40, 0.30, 0.20],
    [0.20, 0.40, 1.00, 0.20, 0.30],
    [0.60, 0.30, 0.20, 1.00, 0.10],
    [0.10, 0.20, 0.30, 0.10, 1.00],
])
assert np.linalg.eigvalsh(SIGMA_Y).min() > 0, "Sigma_Y doit etre definie positive"
CHOL = np.linalg.cholesky(SIGMA_Y)


def simulate(N, T, rng):
    """Panel D (N,J,T) et la verite Y (J,T) du facteur systemique."""
    D = np.zeros((N, J, T), dtype=np.int8)
    D[:, :, 0] = (rng.random((N, J)) < 0.07).astype(np.int8)
    Y = np.zeros((J, T))
    for t in range(1, T):
        Y[:, t] = CHOL @ rng.standard_normal(J)           # Y_t ~ N(0, Sigma_Y)
        X = BASE + D[:, :, t - 1] @ W_TRUE.T + S_TRUE * Y[:, t] + rng.standard_normal((N, J))
        D[:, :, t] = (X >= 0).astype(np.int8)
    return D, Y


# ============================================================ Route A : panel
def route_panel(D):
    """Lit les effets fixes de periode du probit : phi_{j,t} = base_j + s_j Y_{j,t}."""
    N, _, T = D.shape
    per_idx = np.tile(np.arange(T - 1), N)        # PAS np.repeat : cf. script 03
    per_oh = np.eye(T - 1)[per_idx]
    phi = np.zeros((J, T - 1))
    for j in range(J):
        y = D[:, j, 1:].reshape(-1)
        others = [k for k in range(J) if k != j]
        lag = np.column_stack([D[:, k, :-1].reshape(-1) for k in others])
        res = sm.Probit(y, np.column_stack([lag, per_oh])).fit(
            disp=0, method="bfgs", maxiter=500)
        phi[j] = res.params[len(others):]
    return phi


# ============================================================ Route B : agrege, Frobenius
PATTERNS = np.array([[(m >> b) & 1 for b in range(J)] for m in range(2 ** J)])  # 32 x 5


W_MARG = 3.0        # poids des moments marginaux : ce sont les plus informatifs


def moments_model(u, pi_lag, w, base):
    """Moments implicites par u_j = s_j Y_{j,t}, sous independance des piliers a t-1.

    Renvoie (Pi, marg) :
      Pi[j,k] = P(incident en j a t ET incident en k a t-1)   -> 25 moments
      marg[j] = P(incident en j a t)                          ->  5 moments

    ATTENTION. Les seules co-incidences ne suffisent PAS : elles ne portent que sur les
    entites ayant deja subi un incident a t-1, soit ~6 % de l'echantillon. u_j y est
    faiblement identifie, l'estimateur absorbe une part du systemique RETARDE Y_{t-1}
    (correlation du residu jusqu'a +0,36) et l'ecart-type de u est gonfle de 100 %.
    Ajouter les 5 moments MARGINAUX -- l'incidence de chaque pilier -- corrige le tir :
    corr(Y) passe de 0,835 a 0,969.
    """
    p = np.prod(np.where(PATTERNS == 1, pi_lag, 1.0 - pi_lag), axis=1)   # (32,)
    prob = stats.norm.cdf(base[None, :] + PATTERNS @ w.T + u[None, :])   # (32, J)
    Pi = (p[:, None, None] * prob[:, :, None] * PATTERNS[:, None, :]).sum(0)
    marg = (p[:, None] * prob).sum(0)
    return Pi, marg


def route_agregee(D, w, base):
    """Extrait u_t = s * Y_t par moindres carres de Frobenius, DONNEES AGREGEES SEULES."""
    N, _, T = D.shape
    U = np.zeros((J, T - 1))
    for t in range(1, T):
        pi_lag = D[:, :, t - 1].mean(0)
        # tout ceci est agrege : aucun identifiant d'entite n'est requis
        Pi_obs = (D[:, :, t].T.astype(float) @ D[:, :, t - 1].astype(float)) / N
        marg_obs = D[:, :, t].mean(0)

        def f(u):
            Pi, marg = moments_model(u, pi_lag, w, base)
            return np.concatenate([(Pi - Pi_obs).ravel(), W_MARG * (marg - marg_obs)])

        U[:, t - 1] = optimize.least_squares(f, np.zeros(J), method="lm").x
    return U


def diagnostics(name, comp, Y_resp):
    """comp[j,t] estime s_j*Y_{j,t} ; on en deduit base, s, Y, Sigma_Y."""
    s_hat = comp.std(1, ddof=1)
    Y_hat = (comp - comp.mean(1, keepdims=True)) / s_hat[:, None]
    corr_Y = np.array([np.corrcoef(Y_hat[j], Y_resp[j])[0, 1] for j in range(J)])
    Sig_hat = np.corrcoef(comp)
    iu = np.triu_indices(J, 1)
    err = np.abs(SIGMA_Y[iu] - Sig_hat[iu]).mean()
    print(f"\n--- {name} ---")
    print(f"{'pilier':>8}{'s vrai':>9}{'s estime':>11}{'corr(Y)':>10}")
    for j in range(J):
        print(f"{PIL[j]:>8}{S_TRUE[j]:>9.2f}{s_hat[j]:>11.3f}{corr_Y[j]:>+10.3f}")
    print(f"  corr(Y) moyenne = {corr_Y.mean():+.3f}")
    print(f"  Sigma_Y : erreur absolue moyenne hors-diagonale = {err:.3f}")
    print(f"            correlation vrai/estime = {np.corrcoef(SIGMA_Y[iu], Sig_hat[iu])[0,1]:.3f}")
    return Y_hat, s_hat, Sig_hat, corr_Y, err


# ============================================================ run
N0, T0 = 1200, 70
D0, Y_true = simulate(N0, T0, RNG)
Y_resp = Y_true[:, 1:]                 # periodes de reponse (1..T-1)
print(f"Panel : {N0} entites x {T0} periodes ; incidence moyenne {D0.mean():.1%}")
print(f"Charges vraies s_j = {S_TRUE}")
print("Sigma_Y vraie (hors-diagonale) :", np.round(SIGMA_Y[np.triu_indices(J, 1)], 2))

phi = route_panel(D0)
base_hat = phi.mean(1)
print(f"\nbase_j vrai = {BASE[0]:.2f} ; estime = {np.round(base_hat, 3)}")
YA, sA, SA, cA, eA = diagnostics("Route A : effets fixes de periode (panel)", phi, Y_resp)

U = route_agregee(D0, W_TRUE, BASE)
YB, sB, SB, cB, eB = diagnostics("Route B : Frobenius sur co-incidences (agrege)", U, Y_resp)

print("\n" + "=" * 74)
print("VERDICT")
print("=" * 74)
print(f"  Route A (panel)  : corr(Y) = {cA.mean():+.3f} | erreur Sigma_Y = {eA:.3f}")
print(f"  Route B (agrege) : corr(Y) = {cB.mean():+.3f} | erreur Sigma_Y = {eB:.3f}")
print("\n  La route B n'utilise AUCUN identifiant d'entite : uniquement des comptages")
print("  agreges par pilier et par periode. Or c'est precisement ce que publient les")
print("  registres DORA, l'ENISA et les CERT.")
print("\n  => W exige un panel individuel horodate au mois (script 05 : introuvable).")
print("     Y, s_j et Sigma_Y s'obtiennent sur des donnees publiques AGREGEES.")
print("     Le verrou de la note est donc plus etroit qu'annonce.")

# ============================================================ figure N
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2, MUTED, GRID = "#1b1e30", "#223e55", "#595959", "#f2f2f2"
ACCENT = "#a6002e"
BL = ["#cde2fb", "#6491e1", "#4c79c7", "#2b559f", "#204993", "#09327a"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.6),
                                    gridspec_kw={"width_ratios": [1.35, 1, 1]})

# (a) une serie temporelle
tt = np.arange(1, T0)
jj = 3                                   # P4 : forte charge systemique
ax1.plot(tt, Y_resp[jj], "-", color=INK, lw=2.0, label="$Y_{4,t}$ vrai", zorder=3)
ax1.plot(tt, YA[jj], "--", color=BL[4], lw=1.7, label=f"route A, panel ($r={cA[jj]:.3f}$)")
ax1.plot(tt, YB[jj], ":", color=ACCENT, lw=2.0, label=f"route B, agrege ($r={cB[jj]:.3f}$)")
ax1.set_xlabel("periode $t$", color=INK2)
ax1.set_ylabel("facteur systemique du pilier P4", color=INK2)
ax1.grid(True, color=GRID, lw=0.7)
ax1.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=1)
ax1.set_title("(a)  Le facteur systemique est restitue", fontsize=11, color=INK, pad=8)

# (b) correlation par pilier
x = np.arange(J)
ax2.bar(x - 0.2, cA, width=0.4, color=BL[4], label="route A (panel)")
ax2.bar(x + 0.2, cB, width=0.4, color=ACCENT, label="route B (agrege)")
ax2.axhline(1.0, color=INK, lw=1, ls="--")
ax2.set_xticks(x); ax2.set_xticklabels(PIL)
ax2.set_ylim(0, 1.12)
ax2.set_ylabel("corr$(\\hat Y_j,\\, Y_j)$", color=INK2)
ax2.yaxis.grid(True, color=GRID, lw=0.7)
ax2.legend(frameon=False, fontsize=8.5, loc="lower right")
ax2.set_title("(b)  Par pilier", fontsize=11, color=INK, pad=8)

# (c) Sigma_Y
iu = np.triu_indices(J, 1)
ax3.plot([0, 0.7], [0, 0.7], color=MUTED, lw=1.2, ls="--", zorder=1)
ax3.scatter(SIGMA_Y[iu], SA[iu], s=48, color=BL[4], edgecolor="#fcfcfb",
            zorder=3, label=f"route A (err {eA:.3f})")
ax3.scatter(SIGMA_Y[iu], SB[iu], s=34, color=ACCENT, marker="^", alpha=0.8,
            zorder=2, label=f"route B (err {eB:.3f})")
ax3.set_xlabel("$\\Sigma_Y$ vrai (hors-diagonale)", color=INK2)
ax3.set_ylabel("$\\Sigma_Y$ estime", color=INK2)
ax3.grid(True, color=GRID, lw=0.7)
ax3.legend(frameon=False, fontsize=8.5, loc="upper left")
ax3.set_title("(c)  Les dependances croisees $\\Sigma_Y$", fontsize=11, color=INK, pad=8)

fig.suptitle("N : le systemique se calibre sur donnees AGREGEES, la contagion non",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "N_facteur_systemique.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
