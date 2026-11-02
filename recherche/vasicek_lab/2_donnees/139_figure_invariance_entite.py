#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
139 : figure de l'invariance au gain de propagation, a l'echelle de l'entite, sous la descente du 136.

La figure S23 du script 66 trace le besoin d'entite dans la lecture du registre (116, 143, 169 M EUR).
La version Institut publie depuis le 29 octobre la descente du script 136 (24,1 M EUR a l'etat non
conforme). Ce script redessine les trois panneaux sous cette descente : (a) le besoin en fonction de g,
(b) les cinq correspondances etat -> g, (c) l'ecart en fonction de l'ecartement g_NC - g_C.

Il recopie le moteur non publie du 136 (meme ajustement NB2 sur les firmes de plus de 100 Md USD, meme
graine, memes tirages dans le meme ordre) et S'ARRETE s'il ne retrouve pas les valeurs imprimees par le
136 (section 4 et grandeurs citees). Aucune valeur n'est retapee : le controle lit 136.txt.
Exige SAS_OpRisk_Global_Data_June_2026.xlsx. Nom de figure sans chiffre (le harnais lit les noms).
"""

import os
import re
import sys
import warnings

import numpy as np
from scipy import optimize, special
from scipy.stats import genpareto
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(HERE)
import partial_id as pid                                            # noqa: E402
import descente as dsc                                              # noqa: E402
import euro_cascade_model as ec                                     # noqa: E402
from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR  # noqa: E402

TAUX_USD = 1.04
NY = 600_000
SEED = 20260721
ALPHA = 0.995
NOTION_USD = 20_000.0
SF_ENTITE = 450.0
SP = pid.PARAMS["OPRISK"]
U = SP["u"]
G_MEMOIRE = {"C": 0.45, "PC": 0.68, "NC": 0.90}

# --- ajustement de frequence du 136 (firmes de plus de 100 Md USD)
D = dsc.Descente()
sub = D.pa[D.pa.actifs >= 1.0e5]
x = np.log(sub.actifs.to_numpy())
k = sub.n.to_numpy().astype(float)
xb = float(x.mean())
xc = x - xb


def nll(p):
    a, b, lr = p
    r = np.exp(lr)
    mu = np.exp(np.clip(a + b * xc, -30, 30))
    return -np.sum(special.gammaln(k + r) - special.gammaln(r) - special.gammaln(k + 1)
                   + r * np.log(r / (r + mu)) + k * np.log(mu / (r + mu)))


o = optimize.minimize(nll, [np.log(k.mean()), 0.2, 0.0], method="Nelder-Mead",
                      options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 20000})
A_, B_, R_ = float(o.x[0]), float(o.x[1]), float(np.exp(o.x[2]))
if abs(B_ - 0.494) > 5e-4:
    sys.exit("l'exposant du 136 n'est pas reproduit : arret.")

df = filter_finance(filter_cyber(load_clean(os.path.join(REPO, "data", "raw",
                                                         "SAS_OpRisk_Global_Data_June_2026.xlsx"))))
d = df[df.year.between(2000, 2026)]
if len(d) != 570:
    sys.exit(f"population de calibration differente du protocole : {len(d)} pertes au lieu de 570")
L_ = (d["loss"] * USD_EUR).to_numpy()
CORPS = np.sort(L_[L_ <= U])
W_AMORCE = np.array([ec.eng.LAMBDA[j] for j in pid.PIL], float)

# --- moteur non publie du 136, tirages dans le meme ordre
a_n = NOTION_USD
lam = float(np.exp(A_ + B_ * (np.log(a_n) - xb)))
mult = D.mult(a_n)
rng = np.random.default_rng(SEED)
counts = rng.negative_binomial(R_, R_ / (R_ + lam), size=NY)
T = int(counts.sum())
year_of = np.repeat(np.arange(NY), counts)
amorce = rng.choice(pid.NP_, size=T, p=W_AMORCE / W_AMORCE.sum())
u = rng.random(T)
n = T * pid.NP_
queue = rng.random(n) < SP["p_u"]
uq = rng.uniform(0, 1, int(queue.sum()))
ic = rng.integers(0, CORPS.size, int((~queue).sum()))
sev = np.empty(n)
sev[queue] = U + genpareto.ppf(uq, c=SP["xi"], scale=SP["sigma"])
sev[~queue] = CORPS[ic]
cum = np.cumsum((sev * mult).reshape(T, pid.NP_), axis=1)
idx_by_am = [np.where(amorce == a)[0] for a in range(pid.NP_)]
ar = np.arange(T)


def besoin(g):
    card = pid.card_dist_all(pid.expert_matrix(g))[0]
    K = np.empty(T, dtype=np.int64)
    for a in range(pid.NP_):
        idx = idx_by_am[a]
        if idx.size:
            cdf = np.cumsum(card[a][1:])
            cdf[-1] = 1.0
            K[idx] = np.searchsorted(cdf, u[idx], side="right") + 1
    np.clip(K, 1, pid.NP_, out=K)
    return float(np.quantile(np.bincount(year_of, weights=cum[ar, K - 1], minlength=NY), ALPHA))


# --- controle contre la sortie du 136
txt = open(os.path.join(REPO, "resultats", "sorties_verif_inst", "136.txt"), encoding="utf-8").read()
CORR = []
for m in re.finditer(r"^correspondance (.+?) \(([\d,]+) / ([\d,]+) / ([\d,]+)\) : ([\d.]+) ([\d.]+) ([\d.]+)",
                     txt, re.M):
    gs = [float(v.replace(",", ".")) for v in m.group(2, 3, 4)]
    CORR.append((m.group(1).strip(), gs, [float(v) for v in m.group(5, 6, 7)]))
m = re.search(r"^correspondance conformite faible \(([\d,]+)/([\d,]+)/([\d,]+)\) : ([\d.]+) ([\d.]+) ([\d.]+)",
              txt, re.M)
CORR.insert(3, ("conformite faible", [float(v.replace(",", ".")) for v in m.group(1, 2, 3)],
                [float(v) for v in m.group(4, 5, 6)]))
if len(CORR) != 5:
    sys.exit("les cinq correspondances du 136 n'ont pas ete relues : arret.")
print(f"lambda {lam:.4f} ; multiplicateur {mult:.3f} ; exposant {B_:.3f}")
print(f"{'correspondance':<22}{'C':>7}{'PC':>7}{'NC':>7}   relu dans 136.txt")
VAL = {}
for lab, gs, pub in CORR:
    v = [besoin(g) for g in gs]
    VAL[lab] = (gs, v)
    print(f"{lab:<22}{v[0]:>7.1f}{v[1]:>7.1f}{v[2]:>7.1f}   {pub}")
    if any(abs(a - b) > 0.051 for a, b in zip(v, pub)):
        sys.exit(f"la correspondance {lab} du 136 n'est pas reproduite : arret.")
print("Les cinq correspondances du 136 sont reproduites : le moteur est le bon.")

# --- panneaux (a) et (c)
GRILLE = np.linspace(0.05, 1.0, 20)
se = np.array([besoin(g) for g in GRILLE])
if np.any(np.diff(se) < -1e-9):
    sys.exit("le besoin n'est pas croissant en g sur la grille : arret.")
b_nc = besoin(0.90)
ECART = [(gc, 100 * (b_nc / besoin(gc) - 1)) for gc in np.round(np.arange(0.10, 0.901, 0.05), 3)]
print("grille de g, besoin croissant :", " ".join(f"{v:.1f}" for v in se))
print("ecartement et ecart (%) :", " ".join(f"{0.90-g:.2f}:{e:.1f}" for g, e in ECART))
print(f"ecart retenu (g_C = 0,45) : {100*(b_nc/besoin(0.45)-1):.1f}")

# --- figure
import style_nexialog as nx                                       # noqa: E402

nx.appliquer(taille=10.5)
mpl.rcParams.update({"font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "axes.grid": False})
INK, INK2, MUTED = nx.ENCRE, nx.ENCRE_2, nx.ENCRE_3
ACCENT, BLUE, GREEN = nx.CATEGORIEL[0], nx.CATEGORIEL[2], nx.CATEGORIEL[1]
LEG = dict(frameon=True, facecolor=nx.FOND, edgecolor="none", framealpha=0.88, fontsize=8.2)
ETIQ = {"C": "conforme", "PC": "partiellement conforme", "NC": "non conforme"}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 3.5), gridspec_kw=dict(width_ratios=[1, 1.3, 1]))
top = 1.30 * max(se.max(), max(max(v) for _, v in VAL.values()))

# (a) le besoin croit avec g ; le forfait (450 M EUR) est hors echelle et ne depend pas de l'etat
ax1.plot(GRILLE, se, "-o", color=BLUE, lw=2, ms=4)
for e, c, dec, va in (("C", GREEN, (-7, 8), "bottom"), ("PC", MUTED, (8, -6), "top"),
                      ("NC", ACCENT, (-7, 8), "bottom")):
    g = G_MEMOIRE[e]
    y = besoin(g)
    ax1.plot([g], [y], "o", ms=8, color=c, zorder=4)
    ax1.annotate({"C": "conforme", "PC": "partiel", "NC": "non conforme"}[e], (g, y),
                 textcoords="offset points", xytext=dec, ha="left" if va == "top" else "right",
                 va=va, fontsize=8.8, color=c)
# Sous la courbe, a droite : la seule zone vide du panneau (en haut, l'etiquette NC la recouvrait).
ax1.text(1.0, top * 0.03, f"forfait de Formule Standard :\n{SF_ENTITE:.0f} M€ dans les trois états\n(hors échelle)",
         ha="right", va="bottom", fontsize=8.2, color=GREEN)
ax1.set_ylim(0, top)
ax1.set_xlabel("gain de propagation $g$", color=INK2)
ax1.set_ylabel("besoin à 99,5 % de l'entité (M€)", color=INK2)
ax1.set_title("(a)  Le capital est croissant en $g$", fontsize=11, color=INK, pad=8)

# (b) les cinq correspondances
AFF = {"etroite": "étroite", "retenue": "RETENUE", "large": "large",
       "conformite faible": "conformité faible", "tres etroite": "très étroite"}
labs = [AFF[l] for l in VAL]
xi_ = np.arange(len(VAL))
wd = 0.26
for kk, (e, c) in enumerate((("C", GREEN), ("PC", BLUE), ("NC", ACCENT))):
    ax2.bar(xi_ + (kk - 1) * wd, [v[kk] for _, v in VAL.values()], width=wd, color=c, alpha=0.9, label=ETIQ[e])
ax2.set_xticks(xi_)
ax2.set_xticklabels(labs, fontsize=9, rotation=22, ha="right", rotation_mode="anchor")
ax2.set_ylabel("besoin à 99,5 % (M€)", color=INK2)
ax2.set_ylim(0, top)
ax2.legend(loc="upper left", ncol=2, columnspacing=0.9, handlelength=1.4, handletextpad=0.45,
           borderpad=0.35, **LEG)
ax2.set_title("(b)  L'ordre survit aux cinq\ncorrespondances", fontsize=11, color=INK, pad=8)

# (c) l'ecart selon l'ecartement
gcs = np.array([g for g, _ in ECART])
ecs = np.array([e for _, e in ECART])
ax3.plot(0.90 - gcs, ecs, "-o", color=ACCENT, lw=2, ms=5)
ax3.axhline(10.0, color=INK, lw=1.3, ls="--")
ax3.text((0.90 - gcs).max(), 12.5, "écart de 10 %", ha="right", va="bottom", fontsize=8.8, color=INK2)
_gr = 0.90 - G_MEMOIRE["C"]
_ye = 100 * (b_nc / besoin(G_MEMOIRE["C"]) - 1)
ax3.plot([_gr], [_ye], "D", ms=9, color=BLUE, zorder=4)
ax3.annotate(f"écartement\nretenu : {_gr:.2f}".replace(".", ","), (_gr, _ye), textcoords="offset points",
             xytext=(9, -6), ha="left", va="top", fontsize=8.8, color=BLUE)
ax3.set_xlabel("écartement $g_{NC} - g_C$\n(à $g_{NC}=0{,}90$)", color=INK2)
ax3.set_ylabel("écart de capital (%)", color=INK2)
ax3.set_title("(c)  À $g_{NC}$ fixé, l'amplitude\nsuit l'écartement", fontsize=11, color=INK, pad=8)

for ax in (ax1, ax2, ax3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout(w_pad=1.8)
outdir = os.path.join(HERE, "figures_inst")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "invariance_conformite_entite.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("figure ecrite : figures_inst/invariance_conformite_entite.png")
