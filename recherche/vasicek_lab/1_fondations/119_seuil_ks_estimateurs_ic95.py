#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
119 : choix du seuil POT par un test de Kolmogorov-Smirnov calibre, et estimateurs de la
queue avec leurs intervalles a 95 %.

DEMANDE. Hugo Rapior, point du 22 octobre 2026 (diapositive 8 du support de soutenance) :
« Kolmogorov-Smirnov a faire pour le u optimal ; regarder le memoire de Franck sur
l'optimisation des parametres de la GPD ; ajouter les resultats en tableau de chaque
estimateur avec IC a 95 % ».

CE QUE L'ON REPREND DU MEMOIRE DE FRANCK DOUNTIO (Nexialog, 2025, chapitre 3). La structure :
une p-value de Kolmogorov-Smirnov tracee en fonction du seuil, un bootstrap non parametrique
de 1 000 replications sur les exces, et un tableau seuil / effectif / parametres / IC a 95 %.
CE QUE L'ON NE REPREND PAS. Ses valeurs : sa variable est un nombre de dossiers compromis, la
notre une perte en euros. Et sa p-value de KS : avec des parametres estimes sur l'echantillon
teste, la table usuelle de KS est trop indulgente. Ici la p-value est CALIBREE par bootstrap
parametrique, en reestimant la GPD sur chaque echantillon simule (meme protocole que le
script 47).

DEUX LIMITES, IMPRIMEES PLUS BAS.
  (1) Une p-value elevee a haut seuil peut traduire une faible puissance (peu d'exces), pas
      un meilleur ajustement. Le test sert a ECARTER des seuils, pas a designer « le » seuil
      optimal ; on le croise avec la stabilite de xi.
  (2) Chaque p-value est calibree a seuil fixe. La selection du seuil sur la meme donnee n'est
      pas prise en compte.

CE SCRIPT NE CHANGE AUCUNE VALEUR PUBLIEE. La calibration reste celle de config.py (gelee).
Les quantiles du tableau des estimateurs sont calcules au taux de depassement gele
(p_u = 0,1509), comme le quantile publie de 663 M EUR, pour que les lignes se comparent a lui.

Sortie : resultats/sorties_verif/119.txt (par redirection) et figures/G_seuil_ks_ic.png.
Duree : quelques minutes (bootstrap parametrique sur onze seuils).
"""

import os
import sys

import numpy as np
from scipy import optimize, stats
from scipy.stats import genpareto
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR  # noqa: E402
from src.utils.config import OPRISK                                                         # noqa: E402
import style_nexialog as sn                                                                 # noqa: E402

WID = 92
B_GOF = 1000         # bootstrap parametrique (p-values KS et AD calibrees), par seuil
B_EST = 2000         # bootstrap non parametrique des estimateurs
ALPHA = 0.995
SEED = 20261022
PCTS = [75.0, 77.5, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0]


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def fr(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


# =====================================================================================
titre("0. Donnees : la meme serie que le script 47 et que la calibration publiee")
# =====================================================================================
d = filter_finance(filter_cyber(load_clean(
    os.path.join(REPO, "data", "raw", "SAS_OpRisk_Global_Data_June_2026.xlsx"))))
loss = np.sort(d["loss"].to_numpy() * USD_EUR)
N = loss.size
U_PUB = float(OPRISK["seuil_u_eur"])
P_U_GEL = float(OPRISK["p_u"])
n_pub = int((loss > U_PUB).sum())
print(f"  {N} pertes cyber x finance (M EUR) ; seuil publie u = {U_PUB} ; {n_pub} exces.")
if N != OPRISK["n_incidents"] or n_pub != OPRISK["n_excess"]:
    sys.exit(f"ARRET : la serie ne redonne pas {OPRISK['n_incidents']} pertes et "
             f"{OPRISK['n_excess']} exces ; ce script ne parlerait pas de l'objet publie.")


# =====================================================================================
# Outils : statistiques d'adequation, estimateurs, quantile
# =====================================================================================
def ks_stat(x, c, s):
    z = genpareto.cdf(np.sort(x), c, scale=s)
    m = z.size
    return float(max(np.max(np.arange(1, m + 1) / m - z), np.max(z - np.arange(0, m) / m)))


def ad_stat(x, c, s):
    z = np.clip(genpareto.cdf(np.sort(x), c, scale=s), 1e-12, 1 - 1e-12)
    m = z.size
    i = np.arange(1, m + 1)
    return float(-m - np.sum((2 * i - 1) / m * (np.log(z) + np.log(1 - z[::-1]))))


def emv(e):
    c, _, s = genpareto.fit(e, floc=0)
    return float(c), float(s)


def pwm(e):
    """Moments ponderes de Hosking et Wallis (1987), ecrits dans la convention xi."""
    x = np.sort(e)
    m = x.size
    p = (np.arange(1, m + 1) - 0.35) / m
    a0 = x.mean()
    a1 = np.mean((1 - p) * x)
    xi = 2.0 - a0 / (a0 - 2 * a1)
    s = 2 * a0 * a1 / (a0 - 2 * a1)
    return float(xi), float(s)


def hill(e, u):
    """Hill au-dessus du seuil u : moyenne de ln(X/u) sur les exces."""
    return float(np.mean(np.log((u + e) / u)))


def moments_dedh(e, u):
    """Estimateur des moments de Dekkers, Einmahl et de Haan (1989), au-dessus de u."""
    l = np.log((u + e) / u)
    m1, m2 = np.mean(l), np.mean(l ** 2)
    return float(m1 + 1 - 0.5 / (1 - m1 ** 2 / m2))


def q_sev(u, xi, s, p_u, a=ALPHA):
    return float(u + s / xi * (((1 - a) / p_u) ** (-xi) - 1))


def ic_profil_xi(e, niveau=0.95):
    """IC du rapport de vraisemblance profilee pour xi (EMV)."""
    c0, s0 = emv(e)
    ll0 = float(np.sum(genpareto.logpdf(e, c0, scale=s0)))
    crit = stats.chi2.ppf(niveau, 1) / 2

    def prof(c):
        r = optimize.minimize_scalar(lambda ls: -np.sum(genpareto.logpdf(e, c, scale=np.exp(ls))),
                                     bounds=(np.log(s0) - 4, np.log(s0) + 4), method="bounded")
        return ll0 + r.fun - crit        # > 0 hors de l'IC

    lo = optimize.brentq(prof, c0 - 2.0, c0 - 1e-6) if prof(c0 - 2.0) > 0 else float("nan")
    hi = optimize.brentq(prof, c0 + 1e-6, c0 + 2.0) if prof(c0 + 2.0) > 0 else float("nan")
    return lo, hi


# =====================================================================================
titre("1. Balayage du seuil : effectif, xi (EMV, IC 95 %) et KS / AD calibres")
# =====================================================================================
print("  p-values par bootstrap parametrique : on simule une GPD aux parametres estimes, on")
print(f"  REESTIME la GPD sur chaque echantillon simule, et l'on compare la statistique. B = {B_GOF}")
print("  par seuil. L'IC de xi est celui du rapport de vraisemblance profilee.\n")
rng = np.random.default_rng(SEED)
seuils = [(f"q{fr(p, 1)}", float(np.percentile(loss, p))) for p in PCTS]
seuils.append(("publie", U_PUB))
seuils.sort(key=lambda t: t[1])
lignes = []
print(f"  {'seuil':<9}{'u (M EUR)':>10}{'pctl':>7}{'n_u':>6}{'xi':>8}{'IC 95 % xi':>20}"
      f"{'sigma':>9}{'D KS':>8}{'p KS':>7}{'p AD':>7}{'echecs':>8}")
for lab, u in seuils:
    e = np.sort(loss[loss > u] - u)
    m = e.size
    c, s = emv(e)
    lo, hi = ic_profil_xi(e)
    D, A = ks_stat(e, c, s), ad_stat(e, c, s)
    dn, an, ech = [], [], 0
    for _ in range(B_GOF):
        xb = genpareto.rvs(c, scale=s, size=m, random_state=rng)
        try:
            cb, sb = emv(xb)
            dn.append(ks_stat(xb, cb, sb)); an.append(ad_stat(xb, cb, sb))
        except Exception:
            ech += 1
    p_ks = float(np.mean(np.array(dn) >= D))
    p_ad = float(np.mean(np.array(an) >= A))
    pct = 100 * float((loss <= u).mean())
    lignes.append(dict(lab=lab, u=u, pct=pct, n=m, xi=c, lo=lo, hi=hi, s=s, D=D, p_ks=p_ks,
                       p_ad=p_ad, ech=ech))
    print(f"  {lab:<9}{u:>10.2f}{pct:>7.1f}{m:>6}{c:>8.3f}{'[' + fr(lo, 3) + ' ; ' + fr(hi, 3) + ']':>20}"
          f"{s:>9.2f}{D:>8.3f}{p_ks:>7.3f}{p_ad:>7.3f}{ech:>8}")
print(f"\n  Bruit Monte-Carlo d'une p-value : ecart-type au plus {0.5/np.sqrt(B_GOF):.3f} (p = 0,5).")

non_rej = [L for L in lignes if L["p_ks"] > 0.05 and L["p_ad"] > 0.05]
print("\n  LECTURE.")
print(f"  Seuils ou ni KS ni AD ne rejettent a 5 % : {len(non_rej)} sur {len(lignes)}"
      f" ({', '.join(L['lab'] for L in non_rej)}).")
pub = next(L for L in lignes if L["lab"] == "publie")
print(f"  Au seuil publie (u = {U_PUB}, {pub['n']} exces) : D = {pub['D']:.3f}, p KS = {pub['p_ks']:.3f},"
      f" p AD = {pub['p_ad']:.3f}.")
dedans = [L for L in lignes if L["u"] >= U_PUB and L["lo"] <= pub["xi"] <= L["hi"]]
print(f"  Au-dessus du seuil publie, xi publie ({pub['xi']:.3f}) est dans l'IC 95 % de "
      f"{len(dedans)} seuils sur {sum(1 for L in lignes if L['u'] >= U_PUB)}.")
print("  Le test ECARTE des seuils ; il ne designe pas un optimum. Une p-value haute a haut seuil")
print("  peut venir du faible nombre d'exces (puissance faible), et la selection du seuil sur la")
print("  meme donnee n'est pas corrigee. Le seuil se lit donc avec la stabilite de xi.")


# =====================================================================================
titre("2. Au seuil publie : quatre estimateurs de la queue, IC a 95 % par bootstrap")
# =====================================================================================
e_pub = np.sort(loss[loss > U_PUB] - U_PUB)
print(f"  Bootstrap non parametrique des {e_pub.size} exces, B = {B_EST} (protocole de Franck,")
print("  percentiles 2,5 et 97,5). Le quantile de severite d'un sinistre q99,5 est calcule au")
print(f"  taux de depassement GELE p_u = {P_U_GEL}, comme le 663 M EUR publie : son IC ne porte")
print("  donc que l'incertitude sur (xi, sigma). Hill et les moments ne donnent que xi.\n")

est = {
    "EMV": lambda e: emv(e),
    "Moments ponderes (PWM)": lambda e: pwm(e),
    "Hill": lambda e: (hill(e, U_PUB), float("nan")),
    "Moments (DEdH)": lambda e: (moments_dedh(e, U_PUB), float("nan")),
}
rng2 = np.random.default_rng(SEED + 1)
idx = rng2.integers(0, e_pub.size, size=(B_EST, e_pub.size))
tab = {}
for nom, f in est.items():
    xi0, s0 = f(e_pub)
    xs, ss, qs, ech = [], [], [], 0
    for b in range(B_EST):
        eb = e_pub[idx[b]]
        try:
            xb, sb = f(eb)
        except Exception:
            ech += 1
            continue
        if not np.isfinite(xb):
            ech += 1
            continue
        xs.append(xb)
        if np.isfinite(sb):
            ss.append(sb)
            qs.append(q_sev(U_PUB, xb, sb, P_U_GEL))
    xs, ss, qs = np.array(xs), np.array(ss), np.array(qs)
    q0 = q_sev(U_PUB, xi0, s0, P_U_GEL) if np.isfinite(s0) else float("nan")
    r = dict(xi=xi0, xi_ic=np.quantile(xs, [0.025, 0.975]), s=s0,
             s_ic=np.quantile(ss, [0.025, 0.975]) if ss.size else [np.nan, np.nan],
             q=q0, q_ic=np.quantile(qs, [0.025, 0.975]) if qs.size else [np.nan, np.nan],
             ech=ech, n_ge1=int(np.sum(xs >= 1)))
    tab[nom] = r

print(f"  {'estimateur':<24}{'xi':>7}{'IC 95 % xi':>18}{'sigma':>8}{'IC 95 % sigma':>18}"
      f"{'q99,5':>8}{'IC 95 % q99,5':>18}{'xi>=1':>7}{'ech.':>6}")
for nom, r in tab.items():
    sic = "-" if not np.isfinite(r["s"]) else f"[{r['s_ic'][0]:.1f} ; {r['s_ic'][1]:.1f}]"
    qic = "-" if not np.isfinite(r["q"]) else f"[{r['q_ic'][0]:.0f} ; {r['q_ic'][1]:.0f}]"
    sv = "-" if not np.isfinite(r["s"]) else f"{r['s']:.2f}"
    qv = "-" if not np.isfinite(r["q"]) else f"{r['q']:.0f}"
    print(f"  {nom:<24}{r['xi']:>7.3f}{'[' + fr(r['xi_ic'][0], 3) + ' ; ' + fr(r['xi_ic'][1], 3) + ']':>18}"
          f"{sv:>8}{sic:>18}{qv:>8}{qic:>18}{r['n_ge1']:>7}{r['ech']:>6}")
lo_p, hi_p = ic_profil_xi(e_pub)
print(f"\n  EMV, IC 95 % de xi par vraisemblance profilee : [{lo_p:.3f} ; {hi_p:.3f}].")
c_p = tab["EMV"]["xi"]
se = (1 + c_p) / np.sqrt(e_pub.size)
print(f"  EMV, IC 95 % asymptotique (Fisher) : [{c_p - 1.96*se:.3f} ; {c_p + 1.96*se:.3f}].")
print(f"  Controle : l'EMV redonne la calibration publiee (xi = {OPRISK['xi']}, sigma = "
      f"{OPRISK['sigma_eur']}) a {abs(c_p - OPRISK['xi']):.4f} et {abs(tab['EMV']['s'] - OPRISK['sigma_eur']):.2f} pres,")
print(f"  et le quantile {tab['EMV']['q']:.1f} contre {OPRISK['var_995']} publie.")
print("\n  LECTURE. Hill depasse nettement les trois autres : il suppose une queue de Pareto pure")
print("  au-dessus de u, et le script 47 montre que cet ecart est celui qu'un xi de 0,595 produit")
print("  sur cet echantillon. L'IC a 95 % de l'EMV est plus large que l'IC a 90 % publie")
print("  ([0,30 ; 0,83]) : seul le niveau change, pas l'estimation.")


# =====================================================================================
titre("Figure")
# =====================================================================================
sn.appliquer(10)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.9))
us = np.array([L["u"] for L in lignes])
a1.fill_between(us, [L["lo"] for L in lignes], [L["hi"] for L in lignes], color=sn.CATEGORIEL[2],
                alpha=0.18, lw=0, label="IC à 95 % (vraisemblance profilée)")
a1.plot(us, [L["xi"] for L in lignes], "o-", color=sn.CATEGORIEL[2], lw=1.8, ms=4, label="ξ, EMV")
a1.axvline(U_PUB, color=sn.CATEGORIEL[0], lw=1.2, ls="--")
a1.text(U_PUB, a1.get_ylim()[1], " seuil retenu", color=sn.CATEGORIEL[0], va="top", fontsize=8.5)
a1.set_xlabel("seuil u (M€)")
a1.set_ylabel("indice de queue ξ")
a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
a1.set_title("(a) Stabilité de ξ selon le seuil", fontsize=10.5, loc="left")
a2.plot(us, [L["p_ks"] for L in lignes], "o-", color=sn.CATEGORIEL[2], lw=1.8, ms=4,
        label="Kolmogorov-Smirnov")
a2.plot(us, [L["p_ad"] for L in lignes], "s--", color=sn.CATEGORIEL[1], lw=1.4, ms=3.5,
        label="Anderson-Darling")
a2.axhline(0.05, color=sn.ENCRE_3, lw=1, ls=":")
a2.text(60, 0.06, "seuil de rejet à 5 %", ha="center", va="bottom", fontsize=8, color=sn.ENCRE_3)
a2.axvline(U_PUB, color=sn.CATEGORIEL[0], lw=1.2, ls="--")
a2.set_ylim(0, 1.02)
a2.set_xlabel("seuil u (M€)")
a2.set_ylabel("p-value calibrée (bootstrap)")
a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
a2.set_title("(b) Adéquation de la GPD selon le seuil", fontsize=10.5, loc="left")
for ax in (a1, a2):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
fig.tight_layout()
out = os.path.join(HERE, "figures", "G_seuil_ks_ic.png")
fig.savefig(out, dpi=220)
print("  figure ecrite : figures/G_seuil_ks_ic.png")


# =====================================================================================
titre("Grandeurs citees (sans separateur)")
# =====================================================================================
for L in lignes:
    print(f"  seuil {L['lab']} u={L['u']:.2f} n={L['n']} xi={L['xi']:.3f} ic=[{L['lo']:.3f};{L['hi']:.3f}]"
          f" D={L['D']:.3f} pks={L['p_ks']:.3f} pad={L['p_ad']:.3f}")
for nom, r in tab.items():
    print(f"  {nom} xi={r['xi']:.3f} ic=[{r['xi_ic'][0]:.3f};{r['xi_ic'][1]:.3f}] sigma={r['s']:.2f}"
          f" ic=[{r['s_ic'][0]:.2f};{r['s_ic'][1]:.2f}] q={r['q']:.0f} ic=[{r['q_ic'][0]:.0f};{r['q_ic'][1]:.0f}]")
