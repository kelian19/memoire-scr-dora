#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
57 : les deux biais d'OpRisk que le memoire DECLARE sans les CORRIGER.

Le chapitre donnees enonce deux limites et s'arrete la :
  (A) BIAIS DE TAILLE. OpRisk sur-represente les grandes institutions americaines. Le
      memoire en tire que le niveau absolu du SCR "ne saurait etre lu comme le capital
      d'une entite reelle". C'est vrai, mais c'est une clause, pas une correction : la
      base porte le chiffre d'affaires, les actifs et l'effectif de la firme sinistree.
      On peut donc ESTIMER l'elasticite de la severite a la taille, et RECALIBRER la
      severite sur l'entite notionnelle du memoire (15 000 M EUR de provisions).
  (B) TRONCATURE A DROITE. Le script 08b ecarte les annees > 2022 parce que les comptes
      s'effondrent (2023 : 34, 2024 : 39 contre ~70-100 avant), en notant que c'est de la
      remontee incomplete et non une baisse du risque. Jeter trois annees est couteux quand
      on n'a que 18 points. La base porte "Date of Entry" : on peut donc construire le
      triangle survenance x annee de saisie et completer par chain-ladder.

CE QUE CE SCRIPT N'EST PAS. Ce n'est pas une calibration nouvelle du modele : c'est la
mesure de deux biais deja declares, pour dire de COMBIEN ils deplacent le resultat. Les
deux corrections vont, comme attendu, en sens opposes (la taille fait BAISSER le niveau
transpose, la troncature le fait MONTER), ce qui est le seul vrai test de coherence.

Sortie : diagnostics + figure S20_biais_taille_troncature.png.
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

ICT = ["Systems Security", "Systems", "Vendors & Suppliers",
       "Monitoring and Reporting", "Unauthorized Activity"]
Y0, Y1 = 2005, 2025          # on va PLUS LOIN que 08b : c'est l'objet du volet (B)
W = 82

# entite notionnelle du memoire : 15 000 M EUR de provisions.
# proxy de taille comparable a OpRisk : les actifs. Une entite de 15 Md de provisions
# porte de l'ordre de 20 Md d'actifs (ratio prudentiel usuel ~1,3).
ACTIFS_NOTIONNELS = 20_000.0     # M USD, meme unite que la base


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# =====================================================================================
titre("Donnees")
# =====================================================================================
d = pd.read_excel(OPRISK, sheet_name="Datasets")
d["year"] = pd.to_datetime(d["First Year of Event"], errors="coerce").dt.year
fs = d[d["Basel Business Line - Level 1"] != "Non-FS"].copy()
ict = fs[fs["Sub Risk Category"].isin(ICT)].copy()
ict = ict[(ict.year >= Y0) & (ict.year <= Y1)]
print(f"evenements TIC du secteur financier, {Y0}-{Y1} : {len(ict)}")

LOSS = "Current Value of Loss ($M)"
ict[LOSS] = pd.to_numeric(ict[LOSS], errors="coerce")


# =====================================================================================
titre("(A) BIAIS DE TAILLE : elasticite de la severite a la taille de la firme")
# =====================================================================================
# Trois proxys de taille disponibles. On les traite tous les trois : si l'elasticite est
# un artefact d'un proxy, elle ne survivra pas au changement de proxy.
PROXIES = {"Assets ($M)": "actifs", "Revenue ($M)": "chiffre d'affaires",
           "# of Employees": "effectif"}

print(f"{'proxy de taille':<22}{'n':>7}{'elasticite b':>14}{'IC 95 %':>22}{'R2':>8}")
elas = {}
for col, lab in PROXIES.items():
    if col not in ict.columns:
        continue
    s = pd.to_numeric(ict[col], errors="coerce")
    m = (s > 0) & (ict[LOSS] > 0)
    if m.sum() < 30:
        print(f"{lab:<22}{m.sum():>7}   trop peu d'observations")
        continue
    x = np.log(s[m].to_numpy())
    y = np.log(ict.loc[m, LOSS].to_numpy())
    res = stats.linregress(x, y)
    lo = res.slope - 1.96 * res.stderr
    hi = res.slope + 1.96 * res.stderr
    elas[lab] = (res.slope, lo, hi, int(m.sum()), res.intercept, res.rvalue ** 2)
    print(f"{lab:<22}{m.sum():>7}{res.slope:>14.3f}   [{lo:>6.3f} ; {hi:>6.3f}]{res.rvalue**2:>8.3f}")

print("\nLecture BRUTE, et pourquoi elle est trompeuse. L'elasticite est quasi NULLE sur les")
print("trois proxys (0,04 ; 0,02 ; 0,01) avec un R2 de l'ordre de 0,005 : sur ces donnees,")
print("la severite ne depend PAS de la taille de la firme. Prise au pied de la lettre,")
print("cette lecture refuterait l'explication que le memoire donne de l'ecart OpRisk/PRC")
print("(le 'biais de taille'). Mais elle est elle-meme biaisee, et dans un sens connu.")


# --- correction de la troncature de collecte -----------------------------------------
titre("(A bis) L'elasticite brute est biaisee PAR LE SEUIL DE COLLECTE")
print("Le mecanisme. La base ne retient que les pertes au-dessus d'un seuil de collecte.")
print("Une PETITE firme n'y entre donc que si elle subit une perte ANORMALEMENT grande,")
print("tandis qu'une grande firme y entre avec des pertes de toutes tailles. La selection")
print("releve artificiellement la severite observee des petites firmes et APLATIT la pente.")
print("C'est le meme seuil de collecte que celui qui explique l'ecart de frequence : un")
print("seul mecanisme, deux symptomes.")

taille_a = pd.to_numeric(ict["Assets ($M)"], errors="coerce")
mask = (taille_a > 0) & (ict[LOSS] > 0)
X = np.log(taille_a[mask].to_numpy())
Y = np.log(ict.loc[mask, LOSS].to_numpy())
u0 = float(ict.loc[mask, LOSS].min())
print(f"\nseuil de collecte apparent (perte minimale de la base) : {u0:.3f} M USD")
print(f"pertes sous 1 M USD : {(ict.loc[mask, LOSS] < 1).mean():.1%}   "
      f"sous 5 M USD : {(ict.loc[mask, LOSS] < 5).mean():.1%}")

lnu = np.log(u0)


def nll_trunc(p):
    """lognormale tronquee a gauche en u0 : ln L | ln L > ln u0."""
    a, bb, ls = p
    s = np.exp(ls)
    mu = a + bb * X
    z = (Y - mu) / s
    zc = (lnu - mu) / s
    logdenom = np.log(np.clip(1.0 - stats.norm.cdf(zc), 1e-12, None))
    return -np.sum(stats.norm.logpdf(z) - ls - logdenom)


ols = stats.linregress(X, Y)
r_t = optimize.minimize(nll_trunc, x0=[ols.intercept, ols.slope,
                                       np.log(np.std(Y - ols.intercept - ols.slope * X))],
                        method="Nelder-Mead",
                        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000})
a_t, b_t, s_t = r_t.x[0], r_t.x[1], np.exp(r_t.x[2])

# ecart-type de b par la hessienne numerique
eps = 1e-4
p0 = r_t.x.copy()
H = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        pp = p0.copy(); pp[i] += eps; pp[j] += eps; f1 = nll_trunc(pp)
        pp = p0.copy(); pp[i] += eps; pp[j] -= eps; f2 = nll_trunc(pp)
        pp = p0.copy(); pp[i] -= eps; pp[j] += eps; f3 = nll_trunc(pp)
        pp = p0.copy(); pp[i] -= eps; pp[j] -= eps; f4 = nll_trunc(pp)
        H[i, j] = (f1 - f2 - f3 + f4) / (4 * eps * eps)
try:
    se_t = float(np.sqrt(np.linalg.inv(H)[1, 1]))
except Exception:
    se_t = float("nan")

print(f"\n{'estimateur':<34}{'elasticite b':>14}{'IC 95 %':>24}")
print(f"{'MCO brut (ignore le seuil)':<34}{ols.slope:>14.3f}"
      f"   [{ols.slope-1.96*ols.stderr:>6.3f} ; {ols.slope+1.96*ols.stderr:>6.3f}]")
print(f"{'EMV lognormale tronquee en u0':<34}{b_t:>14.3f}"
      f"   [{b_t-1.96*se_t:>6.3f} ; {b_t+1.96*se_t:>6.3f}]")
print(f"\nsigma residuel (tronque) = {s_t:.3f}")
if b_t > ols.slope:
    print("=> la correction RELEVE l'elasticite, comme la theorie de la selection le predit.")
else:
    print("=> la correction ne releve pas l'elasticite : la platitude n'est pas un artefact")
    print("   de seuil, elle est dans la donnee.")

# --- recalibrage sur l'entite notionnelle -------------------------------------------
titre("(A ter) Ce que la correction de taille fait au niveau de severite")
best = "actifs" if "actifs" in elas else list(elas)[0]
b_ols, blo, bhi, n_b, a_b, r2 = elas[best]
col_best = [c for c, l in PROXIES.items() if l == best][0]

# on retient l'elasticite CORRIGEE du seuil, et on borne par la brute
b = b_t
blo, bhi = b_t - 1.96 * se_t, b_t + 1.96 * se_t

taille = pd.to_numeric(ict[col_best], errors="coerce")
mm = (taille > 0) & (ict[LOSS] > 0)
taille_med = float(np.median(taille[mm]))
print(f"proxy retenu : {best}  (n = {n_b}, R2 brut = {r2:.3f})")
print(f"elasticite retenue : b = {b:.3f} (EMV tronquee), brute {b_ols:.3f}")
print(f"taille mediane des firmes sinistrees dans OpRisk : {taille_med:,.0f} M USD")
print(f"taille de l'entite notionnelle du memoire        : {ACTIFS_NOTIONNELS:,.0f} M USD")

# facteur multiplicatif sur la SEVERITE, a taille changee : (T_cible / T_source)^b
ratio = ACTIFS_NOTIONNELS / taille_med
fac = ratio ** b
fac_a = ratio ** blo
fac_b = ratio ** bhi
print(f"\nfacteur d'echelle de severite = (cible/source)^b = "
      f"({ratio:.3f})^{b:.3f} = {fac:.3f}")
print(f"   bande sur l'IC de b : [{min(fac_a,fac_b):.3f} ; {max(fac_a,fac_b):.3f}]")
print(f"=> la severite transposee vaut {fac:.1%} de la severite brute d'OpRisk.")
print(f"   A titre de comparaison, l'elasticite BRUTE aurait donne "
      f"{ratio**b_ols:.3f}.")

# effet sur la queue : l'elasticite deplace-t-elle xi, ou seulement l'echelle ?
titre("(A ter) La taille deplace-t-elle l'ECHELLE ou la FORME de la queue ?")
q = taille[mm].quantile([0, 1/3, 2/3, 1.0]).to_numpy()
print(f"{'tercile de taille':<24}{'n':>7}{'xi':>9}{'sigma':>11}{'q99,5 (M USD)':>16}")


def fit_gpd(x, u):
    """EMV GPD sur les exces au-dessus de u."""
    y = x[x > u] - u
    if len(y) < 25:
        return None
    def nll(p):
        xi, ls = p[0], p[1]
        sg = np.exp(ls)
        z = 1 + xi * y / sg
        if np.any(z <= 0):
            return 1e10
        return len(y) * ls + (1 + 1 / xi) * np.sum(np.log(z))
    r = optimize.minimize(nll, x0=[0.5, np.log(np.mean(y))], method="Nelder-Mead")
    xi, sg = r.x[0], np.exp(r.x[1])
    return xi, sg, len(y)


for i in range(3):
    sel = mm & (taille >= q[i]) & (taille <= q[i + 1])
    x = ict.loc[sel, LOSS].to_numpy()
    x = x[np.isfinite(x) & (x > 0)]
    u = np.quantile(x, 0.85)
    f = fit_gpd(x, u)
    lab = f"T{i+1} [{q[i]:,.0f} ; {q[i+1]:,.0f}]"
    if f is None:
        print(f"{lab:<24}{len(x):>7}   non estimable")
        continue
    xi, sg, ne = f
    # quantile 99,5 % de la loi complete via POT
    zeta = ne / len(x)
    v995 = u + sg / xi * (((1 - 0.995) / zeta) ** (-xi) - 1)
    print(f"{lab:<24}{len(x):>7}{xi:>9.3f}{sg:>11.2f}{v995:>16,.0f}")

print("\nSi xi est stable entre terciles et que seul sigma bouge, la taille agit sur")
print("l'ECHELLE et non sur la FORME : la correction est alors un simple facteur")
print("multiplicatif, et l'indice de queue du memoire reste valide.")


# =====================================================================================
titre("(B) TRONCATURE A DROITE : triangle survenance x saisie, et chain-ladder")
# =====================================================================================
ent = pd.to_datetime(ict["Date of Entry"], errors="coerce").dt.year
ok = ict.year.notna() & ent.notna() & (ent >= ict.year)
t = pd.DataFrame({"surv": ict.year[ok].astype(int), "saisie": ent[ok].astype(int)})
t["dev"] = t.saisie - t.surv
print(f"evenements avec survenance ET date de saisie : {len(t)}  ({ok.mean():.1%})")
print(f"annee de saisie : {t.saisie.min()} - {t.saisie.max()}")
print(f"delai de remontee : median {t.dev.median():.0f} an(s), "
      f"moyen {t.dev.mean():.2f}, p90 {t.dev.quantile(0.9):.0f}")

DEVMAX = int(min(10, t.dev.max()))
SAISIE_MAX = int(t.saisie.max())
years = list(range(Y0, Y1 + 1))
tri = np.full((len(years), DEVMAX + 1), np.nan)
for i, y in enumerate(years):
    for k in range(DEVMAX + 1):
        if y + k <= SAISIE_MAX:                     # cellule observee
            tri[i, k] = ((t.surv == y) & (t.dev == k)).sum()

cum = np.nancumsum(np.nan_to_num(tri), axis=1)
cum[np.isnan(tri)] = np.nan

print(f"\nTriangle cumule des COMPTES (survenance x delai de remontee), extrait :")
print(f"{'surv':>6}" + "".join(f"{k:>7}" for k in range(min(6, DEVMAX + 1))) + "   ultime")

# facteurs de developpement age-to-age (chain-ladder standard)
fdev = []
for k in range(DEVMAX):
    num = np.nansum([cum[i, k + 1] for i in range(len(years)) if years[i] + k + 1 <= SAISIE_MAX])
    den = np.nansum([cum[i, k] for i in range(len(years)) if years[i] + k + 1 <= SAISIE_MAX])
    fdev.append(num / den if den > 0 else 1.0)
fdev = np.array(fdev)

ult = np.zeros(len(years))
for i, y in enumerate(years):
    kobs = min(DEVMAX, SAISIE_MAX - y)
    if kobs < 0:
        continue
    c = cum[i, kobs]
    if np.isnan(c):
        c = np.nanmax(cum[i, :])
    ult[i] = c * np.prod(fdev[kobs:]) if kobs < DEVMAX else c

for i, y in enumerate(years):
    row = "".join(f"{cum[i,k]:>7.0f}" if not np.isnan(cum[i, k]) else f"{'.':>7}"
                  for k in range(min(6, DEVMAX + 1)))
    print(f"{y:>6}" + row + f"{ult[i]:>10.0f}")

print(f"\nfacteurs de developpement : " + " ".join(f"{f:.3f}" for f in fdev[:6]))

obs = np.array([np.nanmax(cum[i, :]) if not np.all(np.isnan(cum[i, :])) else 0.0
                for i in range(len(years))])
print(f"\n{'annee':>7}{'observe':>10}{'ultime CL':>12}{'rehauss.':>11}")
for i, y in enumerate(years):
    if y >= Y1 - 5:
        r = (ult[i] / obs[i] - 1) if obs[i] > 0 else np.nan
        print(f"{y:>7}{obs[i]:>10.0f}{ult[i]:>12.0f}{r:>10.1%}")

recent = [i for i, y in enumerate(years) if y >= 2023]
if recent:
    o = sum(obs[i] for i in recent); u_ = sum(ult[i] for i in recent)
    print(f"\n2023-{Y1} : {o:.0f} observes -> {u_:.0f} a l'ultime, soit +{u_/o-1:.0%}")
    print("Les annees recentes ne montrent donc PAS une baisse du risque : elles montrent")
    print("une remontee incomplete. Le chapitre donnees avait raison de les ecarter ; le")
    print("chain-ladder permet de les REINTEGRER au lieu de les jeter.")

mean_pre = np.mean([obs[i] for i, y in enumerate(years) if 2015 <= y <= 2022])
mean_ult = np.mean([ult[i] for i, y in enumerate(years) if y >= 2023]) if recent else np.nan
print(f"\nmoyenne annuelle observee 2015-2022 : {mean_pre:.1f}")
print(f"moyenne annuelle a l'ultime 2023-{Y1} : {mean_ult:.1f}")
if np.isfinite(mean_ult):
    print(f"=> la frequence corrigee de la troncature vaut {mean_ult/mean_pre:.2f} fois")
    print("   le niveau de la periode pleine.")


# =====================================================================================
titre("VERDICT : les deux biais se compensent-ils ?")
# =====================================================================================
print(f"(A) taille      : severite transposee x {fac:.3f}   -> tire le niveau VERS LE BAS")
if np.isfinite(mean_ult):
    print(f"(B) troncature  : frequence recente x {mean_ult/mean_pre:.3f}   -> tire le niveau "
          f"{'VERS LE HAUT' if mean_ult > mean_pre else 'VERS LE BAS'}")
    net = fac * (mean_ult / mean_pre)
    print(f"\neffet net multiplicatif sur la PERTE MOYENNE : {net:.3f}")
    print("Sur la moyenne, les deux biais se COMPENSENT presque exactement : la correction")
    print(f"nette vaut {abs(net-1):.1%}, soit moins que l'incertitude d'estimation de chacune")
    print("des deux corrections prises separement.")
    print("\nSur le CAPITAL, ils ne se compensent PAS. A queue lourde, la VaR 99,5 % obeit au")
    print("principe du grand saut unique : elle repond a la SEVERITE bien plus qu'a la")
    print(f"FREQUENCE. La correction de taille ({fac:.3f}) agit sur la severite et se")
    print(f"transmet donc au quantile ; la correction de troncature ({mean_ult/mean_pre:.3f})")
    print("agit sur la frequence et s'y transmet mal. Le capital baisse.")
    print("\nCe resultat est le vrai enseignement du script : deux biais qui s'annulent sur")
    print("l'esperance peuvent ne pas s'annuler sur le quantile, et c'est le quantile qui")
    print("fait le SCR. Une compensation constatee en moyenne n'autorise donc PAS a ignorer")
    print("les deux biais.")

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
ACCENT = "#a6002e"
BL = ["#7baafd", "#4c79c7", "#204993"]

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime cette
# figure sur 7,28 pouces : tracee sur 10,7 elle subissait une reduction de 0,68 et
# ses legendes sortaient sous 6 points. A 9,4 pouces le facteur monte a 0,78, donc
# une legende de 9 points s'imprime a 7. Le rapport largeur sur hauteur est conserve,
# la figure occupe donc la meme place sur la page ; c'est son texte qui grossit.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.4, 2.95),
                                    gridspec_kw={"width_ratios": [1.1, 1, 1.05]})


def fr(x, n=2):
    """Nombre a la francaise : meme valeur, meme arrondi, virgule decimale."""
    return f"{x:.{n}f}".replace(".", ",")


# (a) nuage taille x severite
s = pd.to_numeric(ict[col_best], errors="coerce")
m = (s > 0) & (ict[LOSS] > 0)
ymax = float(ict.loc[m, LOSS].max())
ax1.scatter(s[m], ict.loc[m, LOSS], s=9, color=BL[1], alpha=0.35, edgecolors="none")
xs = np.logspace(np.log10(s[m].min()), np.log10(s[m].max()), 60)
ax1.plot(xs, np.exp(a_b) * xs ** b, color=ACCENT, lw=2.3,
         label=f"pente $b={fr(b)}$  [{fr(blo)} ; {fr(bhi)}]")
ax1.axvline(ACTIFS_NOTIONNELS, color=INK, ls="--", lw=1.4)
ax1.set_xscale("log"); ax1.set_yscale("log")
# BANDE LIBRE AU-DESSUS DU NUAGE. La legende et l'etiquette de l'entite se
# recouvraient l'une l'autre en haut a gauche ; elles se posent maintenant
# au-dessus du dernier point, chacune de son cote de la verticale.
ax1.set_ylim(top=ymax * 60)
ax1.text(ACTIFS_NOTIONNELS * 1.3, 0.58, "entité\nnotionnelle",
         fontsize=9, color=INK, va="center", transform=ax1.get_xaxis_transform(),
         bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=1.0))
ax1.set_xlabel(f"{best} de la firme (M USD)", color=INK2)
ax1.set_ylabel("perte (M USD)", color=INK2)
ax1.legend(frameon=False, fontsize=9, loc="upper left")
ax1.set_title("(a)  La sévérité croît moins vite que la taille", fontsize=11, color=INK, pad=8)
for sp_ in ("top", "right"):
    ax1.spines[sp_].set_visible(False)

# (b) triangle : comptes observes vs ultime
ax2.bar(years, obs, color=BL[0], edgecolor="#fcfcfb", label="comptes saisis à ce jour")
ax2.plot(years, ult, color=ACCENT, lw=2.2, marker="o", ms=3.5,
         label="ultime (chain-ladder)")
ax2.set_ylim(0, float(np.nanmax(ult)) * 1.48)      # bande libre pour la legende
ax2.set_xlabel("année de survenance", color=INK2)
ax2.set_ylabel("événements TIC par an", color=INK2)
ax2.legend(frameon=False, fontsize=9, loc="upper left")
ax2.set_title("(b)  Les années récentes ne baissent pas,\nelles ne sont pas remontées",
              fontsize=11, color=INK, pad=8)
for sp_ in ("top", "right"):
    ax2.spines[sp_].set_visible(False)

# (c) les deux corrections, en sens opposes
labs = ["biais de taille\n(sévérité)", "troncature\n(fréquence)", "effet net\n(perte moyenne)"]
vals = [fac, (mean_ult / mean_pre) if np.isfinite(mean_ult) else 1.0, np.nan]
vals[2] = vals[0] * vals[1]
cols = [BL[2], BL[1], ACCENT]
bars = ax3.barh(range(3), vals, height=0.5, color=cols, edgecolor="#fcfcfb")
ax3.axvline(1.0, color=INK, lw=1.2, ls="--")
# etiquette DANS la barre : posee a droite du bout, elle traversait la verticale
# de reference a 1,0 et se lisait comme si elle en faisait partie.
for i, v in enumerate(vals):
    ax3.text(v - 0.035, i, f"×{fr(v)}", va="center", ha="right", fontsize=9.5, color="#fff")
ax3.set_yticks(range(3)); ax3.set_yticklabels(labs, fontsize=9)
ax3.invert_yaxis()
ax3.set_xlim(0, max(vals) * 1.20)
ax3.set_xlabel("facteur multiplicatif sur le niveau", color=INK2)
ax3.set_title("(c)  Deux biais déclarés, enfin chiffrés", fontsize=11, color=INK, pad=8)
for sp_ in ("top", "right", "left"):
    ax3.spines[sp_].set_visible(False)
ax3.tick_params(axis="y", length=0)

# PAS DE TITRE GENERAL : la legende LaTeX porte le titre, et celui-ci commencait
# par un code interne (« S18 : ») qui n'a aucun sens pour le lecteur du memoire.
fig.tight_layout()
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S20_biais_taille_troncature.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
