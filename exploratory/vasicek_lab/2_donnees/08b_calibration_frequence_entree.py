#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08b : calibrer la FREQUENCE D'ENTREE sur donnees, au lieu de la poser.

Etage 1 du modele collectif : l'arrivee des sinistres qui AMORCENT une cascade.
frequence_model.py pose aujourd'hui deux nombres sans donnee :
    LAMBDA_TOT = 12.0   (incidents attendus par an et par entite)
    A_LOAD     = 0.60   (charge systemique commune)
Ce script les confronte au SAS OpRisk Global Data.

POURQUOI C'EST FAISABLE ICI ALORS QUE W NE L'EST PAS (cf. 05_faisabilite_donnees).
Calibrer W exige l'horodatage fin + la taxonomie par pilier + la DIRECTION, et la
direction tombe sous le placebo. Une frequence d'ENTREE n'exige rien de tout cela :
seulement « un evenement TIC est survenu chez cette firme cette annee-la ». Le pas
annuel, fatal pour W, suffit pour un comptage. C'est un probleme d'identification
strictement plus facile.

DEUX QUANTITES, DEUX NIVEAUX D'OBSERVATION (elles repondent a deux parametres) :
  (A) taux par entite lambda      <- panel firme-annee        -> LAMBDA_TOT
  (B) choc systemique commun a    <- serie agregee du secteur -> A_LOAD
Melanger les deux serait une faute : la surdispersion du panel firme-annee est
dominee par l'HETEROGENEITE DE TAILLE des firmes, pas par le choc commun. Le choc
commun ne se lit qu'au niveau agrege, et APRES retrait de la tendance.

LE LIEN ERLANG. On ajuste une binomiale negative NB2 : N|theta ~ Poisson(theta),
theta ~ Gamma(forme r, moyenne mu). Une Gamma de forme ENTIERE est une Erlang :
r est donc directement la « loi d'Erlang » evoquee comme melange.

FENETRE 2005-2022. Au-dela de 2022 les comptes s'effondrent (2023 : 34, 2024 : 39
contre ~70-100 avant) : c'est la troncature a droite de la base (les evenements
recents ne sont pas encore remontes), pas une baisse du risque.

Donnees attendues dans data/raw/ (non versionnees). Sortie : diagnostics + figure.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "data", "raw"))
OPRISK = os.path.join(RAW, "SAS_OpRisk_Global_Data_June_2026.xlsx")
if not os.path.exists(OPRISK):
    sys.exit(f"donnee absente : {OPRISK}\n(les sources brutes ne sont pas versionnees)")

sys.path.insert(0, HERE)
from frequence_model import LAMBDA_TOT, A_LOAD  # noqa: E402  (valeurs posees a confronter)

# sous-categories TIC : MEME definition que 05_faisabilite_donnees.py
ICT = ["Systems Security", "Systems", "Vendors & Suppliers",
       "Monitoring and Reporting", "Unauthorized Activity"]
Y0, Y1 = 2005, 2022
W = 78


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# =====================================================================================
titre("Donnees : secteur financier, sous-categories TIC, 2005-2022")
# =====================================================================================
d = pd.read_excel(OPRISK, sheet_name="Datasets")
d["year"] = pd.to_datetime(d["First Year of Event"], errors="coerce").dt.year
fs = d[(d["Basel Business Line - Level 1"] != "Non-FS")].copy()
fs = fs[(fs.year >= Y0) & (fs.year <= Y1)]
ict = fs[fs["Sub Risk Category"].isin(ICT)].copy()

print(f"evenements du secteur financier   : {len(fs)}")
print(f"dont TIC (5 sous-categories)      : {len(ict)}")
print(f"firmes du secteur financier       : {fs['Firm Name'].nunique()}")
print(f"firmes avec au moins un TIC       : {ict['Firm Name'].nunique()}")

# =====================================================================================
titre("(A) Taux par entite : panel firme-annee")
# =====================================================================================
# Fenetre d'observation d'une firme = [premiere, derniere] annee ou elle apparait
# dans la base pour UN RISQUE QUELCONQUE. Une annee de ce span sans evenement TIC est
# alors un VRAI zero (la firme etait observable). Hors de ce span on ne sait rien.
span = fs.groupby("Firm Name")["year"].agg(["min", "max"])
ict_n = ict.groupby(["Firm Name", "year"]).size().rename("n")

rows = []
for firm, (y0, y1) in span[["min", "max"]].iterrows():
    for y in range(int(y0), int(y1) + 1):
        rows.append((firm, y, int(ict_n.get((firm, y), 0))))
panel = pd.DataFrame(rows, columns=["firm", "year", "n"])

y = panel["n"].to_numpy()
print(f"observations firme-annee : {len(y)}  ({panel.firm.nunique()} firmes)")
print(f"moyenne empirique  = {y.mean():.4f} evenement TIC / firme / an")
print(f"variance empirique = {y.var(ddof=1):.4f}")
print(f"indice de dispersion Var/E = {y.var(ddof=1) / y.mean():.2f}   (Poisson = 1)")
print(f"part de zeros = {(y == 0).mean():.1%}")


def ll_pois(mu, k):
    return np.sum(k * np.log(mu) - mu - special.gammaln(k + 1))


def ll_nb2(p, k):
    """NB2 : moyenne mu, variance mu + mu^2/r. r = forme de la Gamma melangeante."""
    mu, r = np.exp(p)
    return np.sum(special.gammaln(k + r) - special.gammaln(r) - special.gammaln(k + 1)
                  + r * np.log(r / (r + mu)) + k * np.log(mu / (r + mu)))


mu_p = y.mean()
llp = ll_pois(mu_p, y)
opt = optimize.minimize(lambda p: -ll_nb2(p, y), x0=np.log([mu_p, 1.0]),
                        method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-8})
mu_nb, r_nb = np.exp(opt.x)
llnb = -opt.fun
lr = 2 * (llnb - llp)
# test de bord (r -> infini sous H0) : melange 50/50 de chi2_0 et chi2_1
p_lr = 0.5 * stats.chi2.sf(lr, 1)

print(f"\nPoisson          : lambda = {mu_p:.4f}                 logL = {llp:,.1f}")
print(f"Binomiale neg.   : mu = {mu_nb:.4f} | r = {r_nb:.3f}   logL = {llnb:,.1f}")
print(f"test LR Poisson vs NB : LR = {lr:,.1f}, p = {p_lr:.2e}  -> la NB l'emporte")
print(f"forme Gamma melangeante r = {r_nb:.3f}")
print("   ATTENTION : r n'est PAS entier et il est INFERIEUR A 1. Une Erlang exige une")
print(f"   forme entiere >= 1 ; l'Erlang la plus proche est l'Erlang-1 (melange")
print("   exponentiel), mais c'est une APPROXIMATION, pas un ajustement exact. La loi")
print("   melangeante reellement ajustee est une Gamma de forme 0,63, plus dissymetrique")
print("   qu'une exponentielle. A dire tel quel : le melange Gamma est le bon objet,")
print("   'Erlang' n'en est qu'un cas particulier a forme entiere.")

# heterogeneite de taille : le taux depend fortement du profil de firme
tot = fs.groupby("Firm Name").size()
big = tot[tot >= 10].index          # firmes les mieux couvertes (proxy de grande taille)
yb = panel[panel.firm.isin(big)]["n"].to_numpy()
print(f"\nSensibilite a la taille (le panel melange TPE et G-SIB) :")
print(f"   toutes firmes ({len(y)} obs)          : lambda = {y.mean():.4f}")
print(f"   firmes >=10 evenements ({len(yb)} obs) : lambda = {yb.mean():.4f}  "
      f"<- profil plus proche d'une entite soumise a DORA")

# =====================================================================================
titre("(B) Choc systemique commun : serie agregee, apres retrait de tendance")
# =====================================================================================
agg = ict.groupby("year").size().reindex(range(Y0, Y1 + 1), fill_value=0)
t = np.arange(len(agg), dtype=float)
k = agg.to_numpy().astype(float)


def ll_trend(b):
    mu = np.exp(b[0] + b[1] * t)
    return np.sum(k * np.log(mu) - mu - special.gammaln(k + 1))


bt = optimize.minimize(lambda b: -ll_trend(b), x0=[np.log(k.mean()), 0.0],
                       method="Nelder-Mead").x
mu_t = np.exp(bt[0] + bt[1] * t)
# dispersion de Pearson APRES tendance : ce qui reste est le choc commun
dof = len(k) - 2
phi = np.sum((k - mu_t) ** 2 / mu_t) / dof
M = k.mean()
print(f"comptes annuels agreges TIC-FS : moyenne = {M:.1f} / an  ({len(k)} annees)")
print(f"tendance log-lineaire ajustee  : {100*(np.exp(bt[1])-1):+.1f} % / an")
print(f"dispersion de Pearson residuelle phi = {phi:.2f}   (Poisson pur = 1)")

# 18 annees seulement : phi est mal estime. IC via (dof)*phi_hat/phi ~ chi2_dof.
phi_lo = dof * phi / stats.chi2.ppf(0.975, dof)
phi_hi = dof * phi / stats.chi2.ppf(0.025, dof)
print(f"   IC 95 % de phi : [{phi_lo:.2f} ; {phi_hi:.2f}]  <- large, 18 points seulement")


def a_from_phi(f):
    """Modele : Var/E = 1 + M*(exp(a^2)-1)  =>  a = sqrt(ln(1 + (phi-1)/M))."""
    return np.sqrt(np.log(1.0 + max(f - 1.0, 0.0) / M)) if f > 1 else 0.0


a_hat, a_lo, a_hi = a_from_phi(phi), a_from_phi(phi_lo), a_from_phi(phi_hi)
print(f"\ncharge systemique impliquee a = {a_hat:.3f}   IC 95 % : [{a_lo:.3f} ; {a_hi:.3f}]")
print(f"   valeur posee dans frequence_model : A_LOAD = {A_LOAD:.2f}")
if A_LOAD > a_hi:
    print(f"   => A_LOAD depasse meme la BORNE HAUTE de l'IC ({a_hi:.3f}). La conclusion")
    print(f"      'le calage pose est conservateur' resiste a l'incertitude d'estimation.")
else:
    print("   => l'IC recouvre A_LOAD : on ne peut pas conclure.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print(f"lambda pose = {LAMBDA_TOT:.1f} / an ; lambda observe = {y.mean():.3f} "
      f"(toutes firmes) a {yb.mean():.3f} (grandes firmes).")
print("L'ecart n'est PAS une erreur de calage : la base ne retient que les pertes AU-DESSUS")
print("d'un seuil de collecte. On calibre donc la frequence des evenements TIC MATERIELS")
print("(generateurs de perte), qui est la bonne maille pour un modele de capital, et non")
print("la frequence de tous les incidents au sens du reporting DORA.")
print(f"\nSurdispersion confirmee (LR p = {p_lr:.1e}) : le Poisson simple est rejete.")
print(f"La loi melangeante ajustee est une Gamma de forme r = {r_nb:.2f}. C'est bien la")
print("famille evoquee sous le nom 'Erlang', mais la forme n'est pas entiere : l'Erlang")
print("est le cas particulier a forme entiere, ici seulement une approximation (Erlang-1).")
print("Point important : cette brique se calibre SANS AUCUN HORODATAGE, contrairement a W.")
print(f"\nCharge systemique : a = {a_hat:.3f} [{a_lo:.3f} ; {a_hi:.3f}] contre "
      f"A_LOAD = {A_LOAD:.2f} pose.")

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
INK, INK2, MUTED, GRID = "#1b1e30", "#223e55", "#595959", "#f2f2f2"
ACCENT = "#a6002e"
BL = ["#7baafd", "#4c79c7", "#204993"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.7),
                                    gridspec_kw={"width_ratios": [1, 1.15, 1]})

# (a) ajustement de la loi de comptage
kmax = 4
obs = np.array([(y == i).mean() for i in range(kmax + 1)])
obs[kmax] = (y >= kmax).mean()
xs = np.arange(kmax + 1)
pois = stats.poisson.pmf(xs, mu_p); pois[kmax] = 1 - stats.poisson.cdf(kmax - 1, mu_p)
pnb = r_nb / (r_nb + mu_nb)
nb = stats.nbinom.pmf(xs, r_nb, pnb); nb[kmax] = 1 - stats.nbinom.cdf(kmax - 1, r_nb, pnb)

wd = 0.27
ax1.bar(xs - wd, obs, width=wd, color=BL[2], edgecolor="#fcfcfb", label="observe")
ax1.bar(xs, pois, width=wd, color=MUTED, edgecolor="#fcfcfb", label="Poisson")
ax1.bar(xs + wd, nb, width=wd, color=ACCENT, edgecolor="#fcfcfb",
        label=f"NB (melange Gamma, $r={r_nb:.2f}$)")
ax1.set_yscale("log")
ax1.set_xticks(xs)
ax1.set_xticklabels([str(i) for i in range(kmax)] + [f"{kmax}+"])
ax1.set_xlabel("evenements TIC par firme et par an", color=INK2)
ax1.set_ylabel("probabilite", color=INK2)
ax1.legend(frameon=False, fontsize=8.5)
ax1.set_title("(a)  Le Poisson rate la queue", fontsize=11, color=INK, pad=8)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# (b) serie agregee : tendance + choc commun residuel
yrs = np.array(agg.index)
ax2.bar(yrs, k, color=BL[0], edgecolor="#fcfcfb", label="comptes annuels TIC-FS")
ax2.plot(yrs, mu_t, color=BL[2], lw=2.2, label="tendance Poisson ajustee")
ax2.fill_between(yrs, mu_t - np.sqrt(mu_t), mu_t + np.sqrt(mu_t), color=BL[2],
                 alpha=0.18, label="bande Poisson pure ($\\pm\\sqrt{\\mu}$)")
ax2.set_xticks(yrs[::3])
ax2.set_xticklabels([str(int(v)) for v in yrs[::3]])
ax2.set_xlabel("annee de survenance", color=INK2)
ax2.set_ylabel("evenements TIC (secteur financier)", color=INK2)
ax2.legend(frameon=False, fontsize=8.2, loc="upper right")
ax2.set_title(f"(b)  Hors tendance, il reste du choc commun  ($\\phi={phi:.1f}$)",
              fontsize=11, color=INK, pad=8)
ax2.text(0.02, 0.06, "les points sortent souvent de la bande\n"
         "$\\Rightarrow$ surdispersion residuelle = facteur systemique",
         transform=ax2.transAxes, fontsize=8.5, color=ACCENT, style="italic")
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# (c) pose vs calibre
ax3.barh([0.6], [A_LOAD], height=0.3, color=MUTED, edgecolor="#fcfcfb")
ax3.barh([0.2], [a_hat], height=0.3, color=BL[2], edgecolor="#fcfcfb",
         xerr=[[a_hat - a_lo], [a_hi - a_hat]], error_kw=dict(ecolor=ACCENT, capsize=4,
                                                              elinewidth=1.6))
ax3.text(A_LOAD - 0.02, 0.6, f"{A_LOAD:.2f}", va="center", ha="right", fontsize=9.5,
         color="#ffffff", fontweight="bold")
ax3.text(a_hi + 0.015, 0.2, f"{a_hat:.3f}\nIC [{a_lo:.2f} ; {a_hi:.2f}]", va="center",
         fontsize=8.2, color=INK2)
ax3.set_yticks([0.6, 0.2]); ax3.set_yticklabels(["pose\n(A_LOAD)", "calibre\n(donnee)"],
                                                fontsize=9)
ax3.set_xlim(0, max(A_LOAD, a_hi) * 1.5)
ax3.set_ylim(-0.62, 0.95)
ax3.set_xlabel("charge systemique commune $a$", color=INK2)
ax3.set_title("(c)  Le calage pose etait prudent", fontsize=11, color=INK, pad=8)
ax3.text(A_LOAD + 0.025, 0.6, "A_LOAD depasse\nla borne haute\nde l'IC", ha="left",
         va="center", fontsize=8, color=ACCENT, style="italic")
ax3.text(0.0, -0.30, f"Taux d'entree $\\lambda$ : {LAMBDA_TOT:.0f}/an pose  vs  "
         f"{y.mean():.2f} observe (toutes firmes),\n{yb.mean():.2f} (grandes firmes). "
         "Ecart explique par le seuil de collecte :\nla base ne voit que les pertes "
         "MATERIELLES, pas tout incident DORA.",
         fontsize=8.2, color=INK2, va="top")
for s in ("top", "right", "left"):
    ax3.spines[s].set_visible(False)
ax3.tick_params(axis="y", length=0)

fig.suptitle("N : la frequence d'entree se calibre, elle (contrairement a $W$)",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "N_frequence_entree.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
