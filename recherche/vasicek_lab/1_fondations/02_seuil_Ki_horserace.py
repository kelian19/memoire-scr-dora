#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02 : Banc d'essai du seuil K_i : Phi^-1(PD) vs EVT.

On simule des severites cyber REELLEMENT a queue lourde (Pareto generalisee) avec
SOUS-DECLARATION (les petits incidents sont rarement declares). On compare trois
facons d'estimer le capital de queue (VaR 99,5 %) et la stabilite du seuil.

  M1  monde gaussien/lognormal : on ajuste une loi lognormale (queue fine) et on
      lit son quantile 99,5 % : l'esprit de K = Phi^-1(PD).
  M2  quantile empirique 99,5 % des pertes observees.
  M3  EVT / POT : loi de Pareto generalisee (GPD) sur les depassements d'un seuil
      haut : n'utilise que les GROS incidents, fiablement declares.

Deux resultats :
  (J1)  la sous-declaration des PETITS incidents ne biaise PAS l'indice de queue
        (invariance par troncature a gauche) -> EVT robuste ; lognormal biaise.
  (J2)  le seuil classique Phi^-1(PD) DERIVE dans le temps (non-stationnarite) ;
        l'indice de queue EVT reste stable.

Sortie : diagnostics + figure J_seuil_Ki.png
"""

import os
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

rng = np.random.default_rng(20260708)

# ---------------------------------------------------------------- verite (DGP)
# xi ANCRE SUR DONNEES REELLES : SAS OpRisk Global Data, pertes en dollars du secteur
# financier, POT au seuil q95 -> xi_hat = 0,85 (ec-type 0,06) ; 0,94 au seuil q90.
# On retient 0,90. Consequence : moyenne finie (xi<1) mais VARIANCE INFINIE (xi>0,5).
XI, SCALE = 0.90, 1.0          # GPD : queue lourde (Frechet), variance INFINIE
P = 0.995                      # niveau de capital DORA-like
S0, AREP = 0.6, 1.3            # sous-declaration : q(s) croit avec la taille


def q_report(s):               # proba qu'un incident de taille s soit declare
    return 1.0 / (1.0 + np.exp(-AREP * (np.log(s + 1e-9) - np.log(S0))))


def draw_true(n):
    return stats.genpareto.rvs(c=XI, scale=SCALE, size=n, random_state=rng)


def thin(sev):                 # applique la sous-declaration
    keep = rng.random(sev.size) < q_report(sev)
    return sev[keep]


# verite = quantile 99,5 % de la loi DECLAREE (celle qu'on observe reellement)
big = draw_true(4_000_000)
big_obs = thin(big)
TRUTH = np.quantile(big_obs, P)
print(f"Verite : VaR99,5 de la severite declaree = {TRUTH:.2f} M EUR")
print(f"  (indice de queue vrai xi = {XI})\n")


# ---------------------------------------------------------------- estimateurs
def m1_lognormal(x):
    s, loc, sc = stats.lognorm.fit(x, floc=0)
    return stats.lognorm.ppf(P, s, loc=0, scale=sc)


def m2_empirique(x):
    return np.quantile(x, P)


def m3_evt_pot(x, u_q=0.80):
    u = np.quantile(x, u_q)
    exc = x[x > u] - u
    if exc.size < 30:
        return np.nan, np.nan
    c, _, beta = stats.genpareto.fit(exc, floc=0)
    n, nu = x.size, exc.size
    var = u + (beta / c) * ((n / nu * (1 - P)) ** (-c) - 1)
    return var, c                # renvoie aussi l'indice de queue estime


# ---------------------------------------------------------------- (J2) course au capital
NREP, NOBS = 500, 800
res = {"M1 lognormal": [], "M2 empirique": [], "M3 EVT-POT": []}
xis_full, xis_obs = [], []
for _ in range(NREP):
    true_sev = draw_true(NOBS)
    obs = thin(true_sev)
    if obs.size < 100:
        continue
    res["M1 lognormal"].append(m1_lognormal(obs) / TRUTH)
    res["M2 empirique"].append(m2_empirique(obs) / TRUTH)
    var3, xi3 = m3_evt_pot(obs)
    res["M3 EVT-POT"].append(var3 / TRUTH)
    xis_obs.append(xi3)
    # indice de queue sur l'echantillon COMPLET (sans sous-declaration)
    _, xif = m3_evt_pot(true_sev)
    xis_full.append(xif)

print("Course au capital (estimation / verite, sur %d tirages) :" % NREP)
print(f"{'methode':<16}{'biais median':>14}{'IQR':>10}{'RMSE':>10}")
for k, v in res.items():
    v = np.array(v)
    v = v[np.isfinite(v)]
    biais = np.median(v) - 1
    iqr = np.quantile(v, 0.75) - np.quantile(v, 0.25)
    rmse = np.sqrt(np.mean((v - 1) ** 2))
    print(f"{k:<16}{biais:>+13.1%}{iqr:>10.2f}{rmse:>10.2f}")
print("  Avec xi = 0,90 > 0,5 la VARIANCE de la severite est INFINIE : le RMSE d'un")
print("  estimateur de quantile n'a plus de contenu asymptotique et n'est donne que")
print("  pour memoire. Lire le BIAIS MEDIAN et l'ecart interquartile (IQR).")

print("\nInvariance de l'indice de queue a la sous-declaration (J1) :")
print(f"  xi estime sur donnees COMPLETES  = {np.nanmedian(xis_full):.3f}")
print(f"  xi estime sur donnees DECLAREES  = {np.nanmedian(xis_obs):.3f}")
print(f"  (vrai xi = {XI} : la troncature des petits ne biaise pas la queue)\n")

# ---------------------------------------------------------------- (J1bis) recalibrer ou non
# ATTENTION. Un seuil ne peut pas etre a la fois "fixe" et recalcule chaque periode.
# On compare donc DEUX pratiques, sur la meme menace croissante :
#   (a) RECALIBRER K a chaque periode sur la frequence de la periode -> il derive
#       mecaniquement, et la comparabilite entre periodes est detruite.
#   (b) ESTIMER K UNE FOIS sur une periode de reference, puis le tenir. La menace
#       croissante se manifeste alors comme une hausse de la proba de depassement,
#       qui est une SORTIE du modele. C'est ce que la note revendique.
T, RPER = 12, 8
# Barre de materialite fixe (DORA art. 18). On la place LA OU LA DECLARATION EST QUASI
# COMPLETE, conformement a la regle etablie par le script 04 : q(15) = 0,985. Une barre
# basse (p. ex. u=3, q=0,89) biaiserait K_j vers le haut, donc sous-provisionnerait.
U_REG = 15.0
K_recal, xi_time, pd_time = [], [], []
for t in range(T):
    scale_t = SCALE * (1 + 0.6 * t / (T - 1))      # menace qui monte
    ks, xs, ps = [], [], []
    for _ in range(RPER):                          # moyenne pour debruiter
        sev = stats.genpareto.rvs(c=XI, scale=scale_t, size=8000, random_state=rng)
        obs = thin(sev)
        pd_hat = min(max(np.mean(obs > U_REG), 1e-4), 0.5)
        ps.append(pd_hat)
        ks.append(stats.norm.ppf(1 - pd_hat))      # K recalibre : K = Phi^-1(1-PD_t)
        xs.append(m3_evt_pot(obs)[1])
    K_recal.append(np.mean(ks))
    xi_time.append(np.nanmean(xs))
    pd_time.append(np.mean(ps))

K_fixe = K_recal[0]                                # estime une fois, periode de reference
K_held = [K_fixe] * T

print("Recalibrer le seuil a chaque periode, ou l'estimer une fois ?")
print(f"  (a) K recalibre chaque periode : de {K_recal[0]:.2f} a {K_recal[-1]:.2f} "
      f"(amplitude {max(K_recal)-min(K_recal):.2f})  -> cible mouvante")
print(f"  (b) K estime une fois           : {K_fixe:.2f} partout            -> comparable")
print(f"      la menace passe alors dans la SORTIE : proba de depassement "
      f"{pd_time[0]:.1%} -> {pd_time[-1]:.1%}")
print(f"  indice de queue EVT : de {xi_time[0]:.2f} a {xi_time[-1]:.2f} "
      f"(amplitude {max(xi_time)-min(xi_time):.2f})  -> stable dans les deux cas")
print("\n  La derive de (a) n'est PAS une propriete de Phi^-1 : c'est ce qui arrive")
print("  quand on reinvertit une frequence qui monte. Le vrai message reglementaire")
print("  est qu'un seuil recalibre chaque trimestre detruit le benchmarking.")

# ---------------------------------------------------------------- figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2, MUTED, GRID = "#1b1e30", "#223e55", "#595959", "#f2f2f2"
ACCENT = "#a6002e"
BL = ["#7baafd", "#4c79c7", "#204993"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2),
                               gridspec_kw={"width_ratios": [1.1, 1]})

# panneau 1 : course au capital
data = [np.clip(np.array(res[k])[np.isfinite(res[k])], 0, 3)
        for k in res]
bp = ax1.boxplot(data, patch_artist=True, widths=0.6,
                 showfliers=False, medianprops=dict(color=INK, lw=1.6))
for patch, c in zip(bp["boxes"], BL):
    patch.set_facecolor(c); patch.set_edgecolor("#fcfcfb")
ax1.axhline(1.0, color=ACCENT, lw=2, zorder=0)
ax1.text(3.35, 1.0, "verite", color=ACCENT, va="center", fontsize=9, fontweight="bold")
ax1.set_xticklabels(list(res.keys()), fontsize=9.5)
ax1.set_ylabel("capital estime / capital vrai", color=INK2)
ax1.set_ylim(0, max(2.4, max(np.quantile(v, 0.9) for v in data) * 1.15))
ax1.set_title("(J2)  La course au capital  (VaR 99,5 %)", fontsize=11, color=INK, pad=8)
ax1.yaxis.grid(True, color=GRID, lw=0.8)
ax1.text(0.5, 0.965, "la lognormale est PRECISE et FAUSSE ;\nl'EVT est JUSTE et imprecise",
         transform=ax1.transAxes, ha="center", va="top", fontsize=8.5,
         color=MUTED, style="italic")

# panneau 2 : recalibrer chaque periode, ou estimer une fois
tt = np.arange(1, T + 1)
axb = ax2.twinx()
l1, = ax2.plot(tt, K_recal, "-o", color=ACCENT, lw=2, ms=5,
               label="$K$ recalibre a chaque periode")
l2, = ax2.plot(tt, K_held, "-", color=BL[2], lw=2.2,
               label="$K$ estime une fois, puis tenu")
l3, = axb.plot(tt, [100 * p for p in pd_time], "--s", color=INK2, lw=1.6, ms=4,
               label="proba de depassement (SORTIE)")
ax2.set_xlabel("periode (menace croissante →)", color=INK2)
ax2.set_ylabel("seuil $K$", color=INK)
axb.set_ylabel("proba de depassement (%)", color=INK2)
lo, hi = min(K_recal), max(K_recal)
ax2.set_ylim(lo - 0.35 * (hi - lo), hi + 0.30 * (hi - lo))
axb.set_ylim(0, 1.6 * max(pd_time) * 100)
axb.tick_params(axis="y", colors=INK2)
ax2.annotate("cible mouvante :\nbenchmarking detruit", xy=(11, K_recal[-2]),
             xytext=(6.0, lo - 0.10 * (hi - lo)), fontsize=8.5, color=ACCENT,
             arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))
ax2.set_title("(J1)  Un seuil ne peut pas etre a la fois fixe et recalcule",
              fontsize=11, color=INK, pad=8)
ax2.legend(handles=[l1, l2, l3], loc="upper left", frameon=False, fontsize=8.5)

fig.suptitle("Seuil K_j : la queue EVT tient le capital, la barre reglementaire tient le seuil",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])

outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "J_seuil_Ki.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
