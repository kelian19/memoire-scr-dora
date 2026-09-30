#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08g : cas d'usage PILIER 4 (risque lie aux tiers), oriente KPI de CONCENTRATION.

M. Cherkaoui : P4 est le point d'entree le plus critique. Registre d'externalisation,
mapping des sous-traitants ; un sous-traitant qui tombe impacte toutes les activites de
la banque. Le modele qualitatif dit la meme chose autrement : ROOT[4]=0,90 (2e amorce),
P4 en tete des cascades critiques.

La donnee le confirme par la CONCENTRATION. On reutilise la detection de causes communes
du 08d (jours dont le compte depasse ce qu'un fond de Poisson produit), mais on la lit ici
comme un risque TIERS : une cause commune = un prestataire partage qui tombe et fait
chuter N entites LE MEME JOUR. Le KPI P4 est donc la CONCENTRATION du portefeuille sur
peu d'evenements, et la QUEUE du nombre de victimes par cause.

Le group_uuid de la base NE relie PAS les organismes (0 groupe multi-orgs, verifie) : on
ne peut donc pas tracer le tiers nominativement. La detection par sur-comptage journalier
est le proxy retenu ; les tags (MOVEit, vendor, ransomware) la corroborent.

CAVEAT ASSUME. Un jour de sur-comptage melange vraies causes communes tierces et lots de
declaration groupee : le KPI de concentration est un MAJORANT du risque tiers reel. Donnee
US, proxy d'entites DORA.

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
SRC = os.path.join(RAW, "Data_Breach_Chronology.xlsx")
if not os.path.exists(SRC):
    sys.exit(f"donnee absente : {SRC}\n(les sources brutes ne sont pas versionnees)")

Y0, Y1 = 2016, 2023
MOVEIT = (pd.Timestamp("2023-05-25"), pd.Timestamp("2023-06-05"))
W = 78


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# =====================================================================================
titre("Cas d'usage P4 : concentration du portefeuille sur les causes communes")
# =====================================================================================
d = pd.read_excel(SRC, sheet_name="Data_Breach_Chronology",
                  usecols=["breach_date", "organization_type", "normalized_org_name",
                           "breach_type", "tags"])
d["bd"] = pd.to_datetime(d.breach_date.astype(str).str.strip(), errors="coerce")
bsf = d[(d.organization_type == "BSF") & d.bd.notna()].copy()
bsf = bsf[(bsf.bd.dt.year >= Y0) & (bsf.bd.dt.year <= Y1)]

idx = pd.date_range(f"{Y0}-01-01", f"{Y1}-12-31", freq="D")
cnt = bsf.groupby(bsf.bd.dt.normalize()).size().reindex(idx, fill_value=0)
k = cnt.to_numpy().astype(float)
nd = len(k)
n_years = Y1 - Y0 + 1

# concentration : courbe de Lorenz sur les comptes journaliers + Gini
ks = np.sort(k)[::-1]                       # jours du plus charge au moins charge
cum_days = np.arange(1, nd + 1) / nd
cum_inc = np.cumsum(ks) / ks.sum()
# Gini via la Lorenz classique (ordre croissant)
asc = np.sort(k)
lor = np.cumsum(asc) / asc.sum()
gini = 1 - 2 * np.trapezoid(lor, dx=1.0 / nd) + 1.0 / nd
print(f"jours observes : {nd} | evenements : {int(k.sum())}")
print(f"indice de Gini de la charge journaliere : {gini:.2f}  (0 = uniforme, 1 = tout un jour)")
for p in (0.01, 0.02, 0.05):
    share = cum_inc[int(p * nd) - 1]
    print(f"   les {p:.0%} de jours les plus charges portent {share:.0%} des sinistres")

# detection des causes communes (meme regle que 08d)
t = np.arange(nd, dtype=float) / 365.25


def nll(b):
    mu = np.exp(b[0] + b[1] * t)
    return -np.sum(k * np.log(mu) - mu - special.gammaln(k + 1))


b = optimize.minimize(nll, x0=[np.log(k.mean()), 0.0], method="Nelder-Mead",
                      options={"maxiter": 4000}).x
mu_t = np.exp(b[0] + b[1] * t)
thr = stats.poisson.ppf(1.0 - 1.0 / nd, mu_t)
is_cc = k > thr
K = k[is_cc].astype(int)
print(f"\ncauses communes detectees : {is_cc.sum()} ({is_cc.sum()/n_years:.1f}/an), "
      f"portant {K.sum()/k.sum():.0%} des sinistres")
print(f"victimes par cause : mediane {np.median(K):.0f} | max {K.max()}")

# queue du nombre de victimes par cause (Hill)
Ks = np.sort(K)[::-1]
nh = max(5, len(Ks) // 3)
hill = 1.0 / np.mean(np.log(Ks[:nh] / Ks[nh]))
print(f"indice de queue (Hill) des victimes par cause : alpha = {hill:.2f}")

# etude de cas : les plus gros jours
titre("Etude de cas : les plus grosses causes communes")
cc_days = cnt[is_cc].sort_values(ascending=False)
top = cc_days.head(8)
rows = []
for dt, n in top.items():
    sub = bsf[bsf.bd.dt.normalize() == dt]
    is_mv = MOVEIT[0] <= dt <= MOVEIT[1]
    tag_mv = sub.tags.dropna().astype(str).str.contains("moveit", case=False).any()
    ex = list(sub.normalized_org_name.dropna().unique()[:2])
    rows.append((dt, int(n), is_mv or tag_mv, ex))
    flag = " <- MOVEit (tiers)" if (is_mv or tag_mv) else ""
    print(f"   {dt.date()} : {int(n):>4} victimes{flag} | ex: {ex}")
mv_tag = bsf.tags.dropna().astype(str).str.contains("moveit", case=False).sum()
print(f"\nlignes taggees MOVEit : {mv_tag}. La vague fin mai 2023 = un seul prestataire")
print("(Progress Software / MOVEit) faisant chuter des dizaines d'entites financieres.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print(f"1. CONCENTRATION forte : Gini {gini:.2f} ; {is_cc.sum()/nd:.1%} des jours portent "
      f"{K.sum()/k.sum():.0%} des sinistres.")
print(f"2. La queue des victimes par cause est lourde (Hill alpha = {hill:.2f}) : un seul")
print("   tiers peut faire tomber un nombre non borne d'entites (MOVEit : 191 en un jour).")
print("3. C'est la traduction chiffree du 'registre d'externalisation' : le risque P4 est")
print("   un risque de CONCENTRATION sur des prestataires partages, pas une somme de")
print("   risques independants. KPI P4 : Gini de concentration + queue victimes/cause.")
print("4. Convergence : le modele qualitatif (ROOT[4]=0,90, P4 en tete des cascades) et la")
print("   donnee (accumulation) designent le meme pilier. Validation externe du jugement.")

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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 4.8),
                                    gridspec_kw={"width_ratios": [1, 1, 1.15]})

# (a) courbe de concentration (Lorenz), jours ordonnes du plus charge au moins charge
xx = np.concatenate([[0], cum_days])
yy = np.concatenate([[0], cum_inc])
ax1.plot(xx * 100, yy * 100, color=BL[2], lw=2.2)
ax1.plot([0, 100], [0, 100], color=MUTED, lw=1, ls="--", label="repartition uniforme")
ax1.fill_between(xx * 100, xx * 100, yy * 100, color=BL[0], alpha=0.35)
x2 = 2.0
y2 = cum_inc[int(0.02 * nd) - 1] * 100
ax1.scatter([x2], [y2], color=ACCENT, zorder=5)
ax1.annotate(f"2 % des jours\n= {y2:.0f} % des sinistres", xy=(x2, y2),
             xytext=(22, y2 - 6), fontsize=8.6, color=ACCENT,
             arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
ax1.set_xlabel("part des jours les plus charges (%)", color=INK2)
ax1.set_ylabel("part cumulee des sinistres (%)", color=INK2)
ax1.set_title(f"(a)  Forte concentration (Gini {gini:.2f})", fontsize=11, color=INK, pad=8)
ax1.legend(frameon=False, fontsize=8.3, loc="lower right")
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# (b) queue du nombre de victimes par cause commune
srt = np.sort(K)[::-1]
surv = np.arange(1, len(srt) + 1) / len(srt)
ax2.loglog(srt, surv, "o", ms=5, color=ACCENT, label="causes observees")
xs = np.linspace(srt.min(), srt.max(), 100)
ax2.loglog(xs, (xs / srt.min()) ** (-hill), color=BL[2], lw=1.8,
           label=f"Pareto $\\alpha$={hill:.2f}")
ax2.annotate("MOVEit\n191 victimes", xy=(srt[0], surv[0]),
             xytext=(srt[0] * 0.2, surv[0] * 6), fontsize=8.5, color=ACCENT,
             arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
ax2.set_xlabel("victimes par cause commune", color=INK2)
ax2.set_ylabel("P(victimes > K)", color=INK2)
ax2.legend(frameon=False, fontsize=8.3)
ax2.set_title("(b)  Un tiers, un nombre non borne de victimes", fontsize=11,
              color=INK, pad=8)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# (c) etude de cas : les plus grosses causes communes
labs = [f"{dt.date()}" for dt, _, _, _ in rows][::-1]
vals = [n for _, n, _, _ in rows][::-1]
cols = [ACCENT if mv else BL[1] for _, _, mv, _ in rows][::-1]
yb = np.arange(len(vals))
ax3.barh(yb, vals, color=cols, edgecolor="#fcfcfb", height=0.66)
for y, v, mv in zip(yb, vals, [m for _, _, m, _ in rows][::-1]):
    lab = f"{v}" + ("  MOVEit" if mv else "")
    ax3.text(v + 3, y, lab, va="center", fontsize=8, color=ACCENT if mv else INK2)
ax3.set_yticks(yb); ax3.set_yticklabels(labs, fontsize=8.3)
ax3.set_xlim(0, max(vals) * 1.3)
ax3.set_xlabel("victimes le meme jour", color=INK2)
ax3.set_title("(c)  Les plus grosses causes communes", fontsize=11, color=INK, pad=8)
ax3.text(0.97, 0.10, "fin mai 2023 : une seule faille tierce\n(Progress / MOVEit)",
         transform=ax3.transAxes, ha="right", fontsize=8.2, color=ACCENT, style="italic")
for s in ("top", "right", "left"):
    ax3.spines[s].set_visible(False)
ax3.tick_params(axis="y", length=0)

fig.suptitle("S : cas d'usage P4, le risque tiers est un risque de CONCENTRATION",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S_cas_usage_p4.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
