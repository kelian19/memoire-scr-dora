#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08c : le clustering temporel des sinistres TIC est-il de l'AUTO-EXCITATION (Hawkes) ?

Suite de 08b. 08b a calibre la frequence d'entree en STATIQUE (melange Gamma, choc
commun a). Restait la question dynamique : les sinistres s'appellent-ils les uns les
autres dans le TEMPS ? Un processus de Hawkes le formaliserait :

    lambda(t) = mu(t) + somme_{t_j < t} alpha * exp(-beta (t - t_j))

avec le RATIO DE BRANCHEMENT n = alpha/beta = nombre attendu de sinistres « enfants »
par sinistre. La stationnarite exige n < 1 : c'est le miroir temporel exact de la
stabilite de la cascade (rayon spectral rho(W) < 1). Le parallele est joli.

REPONSE : NON, et c'est le resultat. Le clustering observe est SIMULTANE (une cause
commune chez un TIERS frappe des dizaines de victimes le meme jour), pas SEQUENTIEL.
Un Hawkes ajuste ici ne mesure pas une contagion : il mesure MOVEit.

TROIS PIEGES, traites explicitement (les ignorer donnerait un n bidon proche de 1) :
  1. TENDANCE. Les comptes passent de 24 (2011) a 1040 (2023) : c'est l'extension de
     la couverture declarative, pas le risque. Un mu constant absorberait cette
     croissance en auto-excitation. On met donc une baseline log-lineaire mu(t).
  2. EX-AEQUO. Jusqu'a 191 evenements le MEME JOUR. On brise les ex-aequo par un
     jitter uniforme intra-journalier (graine fixe).
  3. TRONCATURE A DROITE. 2024-2025 s'effondrent (376 puis 25) : non remonte. Exclu.

LE TEST DECISIF est l'ACF des residus journaliers apres tendance. Une auto-excitation
donne une decroissance LISSE sur plusieurs jours (echelle 1/beta). Un batch de cause
commune donne un pic aux lags 0-2 puis RIEN. C'est le second cas qu'on observe, et il
disparait en retirant la seule semaine MOVEit.

Donnees attendues dans data/raw/ (non versionnees). Sortie : diagnostics + figure.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, special
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "data", "raw"))
SRC = os.path.join(RAW, "Data_Breach_Chronology.xlsx")
if not os.path.exists(SRC):
    sys.exit(f"donnee absente : {SRC}\n(les sources brutes ne sont pas versionnees)")

Y0, Y1 = 2016, 2023          # couverture exploitable, avant la troncature a droite
MOVEIT = (pd.Timestamp("2023-05-25"), pd.Timestamp("2023-06-05"))
RNG = np.random.default_rng(20260720)
W = 78


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# =====================================================================================
titre("Donnees : Data Breach Chronology, secteur financier (BSF), dates au jour")
# =====================================================================================
d = pd.read_excel(SRC, sheet_name="Data_Breach_Chronology",
                  usecols=["normalized_org_name", "breach_date", "breach_type",
                           "organization_type"])
d["bd"] = pd.to_datetime(d.breach_date.astype(str).str.strip(), errors="coerce")
bsf = d[(d.organization_type == "BSF") & d.bd.notna()].copy()
bsf = bsf[(bsf.bd.dt.year >= Y0) & (bsf.bd.dt.year <= Y1)].sort_values("bd")
print(f"evenements BSF avec date au jour, {Y0}-{Y1} : {len(bsf)}")
print(f"jours distincts : {bsf.bd.dt.normalize().nunique()}")

per_day = bsf.groupby(bsf.bd.dt.normalize()).size()
print(f"maximum sur un jour : {per_day.max()}")
print("\n5 plus gros jours :")
for dt, n in per_day.sort_values(ascending=False).head(5).items():
    print(f"   {dt.date()} : {n:4d} evenements")
mv = per_day[(per_day.index >= MOVEIT[0]) & (per_day.index <= MOVEIT[1])]
print(f"\n=> les plus gros jours sont tous fin mai 2023 : c'est la vague MOVEit")
print(f"   (faille d'un logiciel TIERS de transfert de fichiers, exploitation de masse).")
print(f"   fenetre {MOVEIT[0].date()} -> {MOVEIT[1].date()} : {int(mv.sum())} evenements "
      f"({mv.sum()/len(bsf):.1%} du total) en {len(mv)} jours.")


# =====================================================================================
# outils
# =====================================================================================
def daily_counts(df, y0, y1):
    idx = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="D")
    return df.groupby(df.bd.dt.normalize()).size().reindex(idx, fill_value=0)


def fit_trend(k):
    """Poisson a baseline log-lineaire. Renvoie (mu_t, params, phi)."""
    t = np.arange(len(k), dtype=float) / 365.25

    def nll(b):
        mu = np.exp(b[0] + b[1] * t)
        return -np.sum(k * np.log(mu) - mu - special.gammaln(k + 1))

    b = optimize.minimize(nll, x0=[np.log(max(k.mean(), 1e-3)), 0.0],
                          method="Nelder-Mead", options={"maxiter": 4000}).x
    mu = np.exp(b[0] + b[1] * t)
    phi = np.sum((k - mu) ** 2 / mu) / (len(k) - 2)
    return mu, b, phi


def acf(x, L):
    x = x - x.mean()
    den = np.sum(x * x)
    return np.array([np.sum(x[:-l] * x[l:]) / den for l in range(1, L + 1)])


def hawkes_fit(times, T):
    """Hawkes exponentiel a baseline log-lineaire mu(t)=exp(b0+b1 t). Renvoie n=alpha/beta.

    LL = somme log lambda(t_i) - integrale_0^T lambda
    Recursion O(N) : R_i = exp(-beta dt)(1 + R_{i-1}).
    """
    ts = np.sort(times)

    def nll(p):
        b0, b1, la, lb = p
        alpha, beta = np.exp(la), np.exp(lb)
        # partie excitation par recursion
        R = np.empty(len(ts))
        R[0] = 0.0
        dt = np.diff(ts)
        for i in range(1, len(ts)):
            R[i] = np.exp(-beta * dt[i - 1]) * (1.0 + R[i - 1])
        lam = np.exp(b0 + b1 * ts) + alpha * R
        if np.any(lam <= 0) or not np.isfinite(lam).all():
            return 1e12
        # compensateur
        if abs(b1) < 1e-10:
            int_mu = np.exp(b0) * T
        else:
            int_mu = (np.exp(b0 + b1 * T) - np.exp(b0)) / b1
        int_ex = (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - ts)))
        return -(np.sum(np.log(lam)) - int_mu - int_ex)

    best, bestval = None, np.inf
    for lb0 in (np.log(0.5), np.log(2.0), np.log(10.0)):      # plusieurs departs
        x0 = [np.log(max(len(ts) / T, 1e-4)), 0.0, lb0 - 0.7, lb0]
        r = optimize.minimize(nll, x0=x0, method="Nelder-Mead",
                              options={"maxiter": 8000, "fatol": 1e-6})
        if r.fun < bestval:
            best, bestval = r, r.fun
    b0, b1, la, lb = best.x
    alpha, beta = np.exp(la), np.exp(lb)
    return alpha / beta, alpha, beta, b1


def jittered_times(df):
    """Times en JOURS depuis le debut, ex-aequo brises dans la journee."""
    t0 = pd.Timestamp(f"{Y0}-01-01")
    base = (df.bd.dt.normalize() - t0).dt.days.to_numpy().astype(float)
    return np.sort(base + RNG.uniform(0, 1, len(base)))


# =====================================================================================
titre("TEST DECISIF : forme de la dependance (ACF des residus journaliers)")
# =====================================================================================
cnt_all = daily_counts(bsf, Y0, Y1)
mu_all, b_all, phi_all = fit_trend(cnt_all.to_numpy().astype(float))
res_all = (cnt_all.to_numpy() - mu_all) / np.sqrt(mu_all)

no_mv = bsf[~((bsf.bd >= MOVEIT[0]) & (bsf.bd <= MOVEIT[1]))]
cnt_nm = daily_counts(no_mv, Y0, Y1)
mu_nm, b_nm, phi_nm = fit_trend(cnt_nm.to_numpy().astype(float))
res_nm = (cnt_nm.to_numpy() - mu_nm) / np.sqrt(mu_nm)

L = 12
a_all, a_nm = acf(res_all, L), acf(res_nm, L)
band = 1.96 / np.sqrt(len(cnt_all))
print(f"tendance ajustee : {100*(np.exp(b_all[1])-1):+.1f} % / an "
      "(extension de la couverture declarative, pas du risque)")
print(f"dispersion phi : {phi_all:.1f} (tout) -> {phi_nm:.1f} (sans la semaine MOVEit)")
print(f"\nbande de bruit blanc a 95 % : +-{band:.3f}")
print("lag :        " + "".join(f"{l:>8d}" for l in range(1, 7)))
print("ACF tout :   " + "".join(f"{v:>+8.3f}" for v in a_all[:6]))
print("ACF sans MV: " + "".join(f"{v:>+8.3f}" for v in a_nm[:6]))
print("\nLecture : une AUTO-EXCITATION donnerait une decroissance LISSE sur plusieurs")
print("jours. On voit un pic aux lags 1-2 puis un effondrement immediat, et il DISPARAIT")
print("quand on retire une seule semaine. C'est la signature d'un batch de cause commune.")

# =====================================================================================
titre("Ajustement de Hawkes : le ratio de branchement est-il stable ?")
# =====================================================================================
specs = {}
T = (pd.Timestamp(f"{Y1}-12-31") - pd.Timestamp(f"{Y0}-01-01")).days + 1.0

specs["tout (2016-2023)"] = bsf
specs["sans semaine MOVEit"] = no_mv
specs["sans 2023"] = bsf[bsf.bd.dt.year <= 2022]
cap = bsf.groupby(bsf.bd.dt.normalize(), group_keys=False).head(5)   # <=5 evts/jour
specs["plafonne a 5 evts/jour"] = cap

rows = []
for name, df in specs.items():
    if name == "sans 2023":
        Tn = (pd.Timestamp("2022-12-31") - pd.Timestamp(f"{Y0}-01-01")).days + 1.0
    else:
        Tn = T
    n_br, al, be, b1 = hawkes_fit(jittered_times(df), Tn)
    rows.append((name, len(df), n_br, be, b1))
    print(f"{name:<26} N={len(df):>5} | n = {n_br:5.3f} | beta = {be:6.3f} /j "
          f"| demi-vie = {np.log(2)/be:5.2f} j | tendance {100*(np.exp(b1*365.25)-1):+.0f} %/an")

n_full, n_nomv = rows[0][2], rows[1][2]
betas = [r[3] for r in rows]
hl = np.log(2) / np.mean(betas)
print(f"\nLECTURE HONNETE. n reste dans une bande etroite ({min(r[2] for r in rows):.2f}"
      f"-{max(r[2] for r in rows):.2f}) : retirer la semaine MOVEit ne le fait bouger que de "
      f"{100*(n_nomv-n_full)/n_full:+.0f} %.")
print("Il n'est donc PAS 'instable', et ce n'est pas par la qu'on le disqualifie.")
print(f"\nCE QUI LE DISQUALIFIE : beta ~ {np.mean(betas):.1f} / jour dans TOUTES les")
print(f"specifications, soit une demi-vie de {hl:.2f} jour (~{24*hl:.0f} heures).")
print("L'excitation estimee vit donc ENTIEREMENT A L'INTERIEUR DE LA JOURNEE. C'est la")
print("signature mecanique des ex-aequo brises par jitter (tous les evenements d'un meme")
print("jour s'auto-excitent artificiellement), et non d'une contagion sur plusieurs jours.")
print("Le Hawkes ajuste ici du bruit intra-journalier, pas un mecanisme de propagation.")
print(f"\nA l'inverse, la SURDISPERSION, elle, s'effondre : phi = {phi_all:.1f} -> "
      f"{phi_nm:.1f} en retirant cette seule semaine.")
print("C'est bien une cause commune ponctuelle qui porte la dependance, pas un processus.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("Le clustering temporel est SIMULTANE, pas SEQUENTIEL. Trois faits concordants :")
print("  - les 5 plus gros jours sont consecutifs (27-31 mai 2023) = vague MOVEit ;")
print("  - l'ACF pique aux lags 1-2 puis tombe a zero, au lieu de decroitre lissement ;")
print(f"  - le Hawkes estime une demi-vie d'excitation de {hl:.2f} jour (~{24*hl:.0f} h),")
print("    donc INTRA-JOURNALIERE : il ne capte aucune dependance d'un jour a l'autre.")
print(f"  - et la surdispersion tombe de {phi_all:.1f} a {phi_nm:.1f} en retirant une semaine.")
print("\nCONSEQUENCE DE MODELISATION. Il ne faut PAS ajouter une couche de Hawkes. Non")
print("parce que son parametre serait instable (il ne l'est pas), mais parce qu'il ne")
print("mesure PAS ce qu'on croit : a cette echelle de temps il code du simultane, et il")
print("importerait dans le SCR une contagion sequentielle que la donnee n'atteste pas.")
print("Le bon objet est un CHOC COMMUN a arrivees groupees (une cause chez un tiers, N")
print("victimes le meme jour). C'est exactement ce que le melange de Poisson de 08b")
print("approxime deja ; l'extension naturelle est un Poisson COMPOSE (batch), pas un Hawkes.")
print("\nCORROBORATION DU MODELE QUALITATIF. MOVEit est un sinistre de PILIER 4 (risque")
print("lie aux tiers) qui frappe en masse. Le classeur des cascades place justement P4 en")
print("tete des amorces les plus critiques : la donnee appuie ce classement, obtenu lui")
print("par jugement d'expert. C'est une validation externe, independante du calage.")
print("\nDONNEE MANQUANTE pour trancher l'auto-excitation : il faudrait la date de")
print("SURVENANCE (et non de notification) horodatee, sur un registre ou une meme cause")
print("commune est identifiee comme UN evenement a N victimes. Sans cela, batch de")
print("declaration et contagion sequentielle restent confondus.")

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
                                    gridspec_kw={"width_ratios": [1.15, 1, 1]})

# (a) la serie journaliere : tout tient dans une semaine
ax1.plot(cnt_all.index, cnt_all.to_numpy(), lw=0.7, color=BL[1])
ax1.plot(cnt_all.index, mu_all, lw=2.0, color=BL[2], label="tendance ajustee")
ax1.axvspan(MOVEIT[0], MOVEIT[1], color=ACCENT, alpha=0.25)
ax1.annotate("MOVEit\n27-31 mai 2023\n388 evts en 5 j",
             xy=(MOVEIT[0], 191), xytext=(pd.Timestamp("2018-06-01"), 140),
             fontsize=8.5, color=ACCENT,
             arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3))
ax1.set_ylabel("sinistres TIC par jour (BSF)", color=INK2)
ax1.set_xlabel("date de survenance", color=INK2)
ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
ax1.set_title("(a)  Une cause commune domine la serie", fontsize=11, color=INK, pad=8)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# (b) ACF : la signature
lags = np.arange(1, L + 1)
ax2.axhspan(-band, band, color=GRID, alpha=0.75, label="bruit blanc (IC 95 %)")
ax2.plot(lags, a_all, "o-", color=ACCENT, lw=1.8, ms=5, label="tout")
ax2.plot(lags, a_nm, "s--", color=BL[2], lw=1.8, ms=4.5, label="sans la semaine MOVEit")
ax2.axhline(0, color=INK, lw=0.8)
ax2.set_xlabel("decalage (jours)", color=INK2)
ax2.set_ylabel("autocorrelation des residus", color=INK2)
ax2.legend(frameon=False, fontsize=8.5)
ax2.set_title("(b)  Un pic, puis rien : ce n'est pas du Hawkes", fontsize=11,
              color=INK, pad=8)
ax2.text(0.97, 0.62, "une auto-excitation\ndecroitrait lissement\nsur plusieurs jours",
         transform=ax2.transAxes, ha="right", fontsize=8.2, color=INK2, style="italic")
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# (c) instabilite du ratio de branchement
names = [r[0] for r in rows][::-1]
vals = [r[2] for r in rows][::-1]
hls = [np.log(2) / r[3] for r in rows][::-1]
ax3.barh(range(len(vals)), vals, color=BL[2], edgecolor="#fcfcfb", height=0.6)
for i, (v, h) in enumerate(zip(vals, hls)):
    ax3.text(v + 0.012, i, f"$n$={v:.2f}   demi-vie {24*h:.0f} h", va="center",
             fontsize=8.5, color=INK2)
ax3.set_yticks(range(len(names)))
ax3.set_yticklabels(names, fontsize=8.5)
ax3.set_xlim(0, max(vals) * 1.75)
ax3.set_ylim(-1.25, len(vals) - 0.4)
ax3.set_xlabel("ratio de branchement $n=\\alpha/\\beta$", color=INK2)
ax3.set_title("(c)  Une excitation qui ne dure pas la journee", fontsize=11,
              color=INK, pad=8)
ax3.text(0.0, -0.72, "$n$ est stable, mais sa demi-vie est INTRA-JOURNALIERE :\n"
         "le Hawkes ajuste les ex-aequo, pas une contagion.",
         fontsize=8.4, color=ACCENT, style="italic", va="top")
for s in ("top", "right", "left"):
    ax3.spines[s].set_visible(False)
ax3.tick_params(axis="y", length=0)

fig.suptitle("O : le clustering est SIMULTANE (cause commune), pas auto-excite",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "O_hawkes_verdict.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
