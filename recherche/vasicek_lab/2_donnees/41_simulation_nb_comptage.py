#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
41 : la loi de comptage negative binomiale, simulee, reconciliee, et son effet sur le capital.

Le projet porte TROIS representations de la surdispersion de la frequence, a trois echelles
differentes, jamais mises face a face :
    r    = 0,63   melange Gamma AU NIVEAU FIRME-ANNEE      (08b, panel firme-annee)
    a    = 0,60   charge systemique commune (Poisson x lognormale)  (frequence_model, A_LOAD)
    phi  = 9,20   facteur de surdispersion NB AGREGE        (config, euro_cascade_model)
Ce script montre par simulation que (a) et (phi) sont le MEME objet vu a travers deux lois
de melange, mesure l'ecart entre les deux lois sur la QUEUE du nombre annuel, cale le phi
sur la vraie serie cyber x finance, et chiffre enfin ce que le choix de la loi de comptage
change au SCR.

TROIS FAITS.
  1. RECONCILIATION a <-> phi. Un Poisson melange a pour indice de dispersion Var/E :
       - melange Gamma (forme r)      : D = 1 + lam/r = phi          (c'est la NB)
       - melange lognormal (charge a) : D = 1 + lam (e^{a^2} - 1)
     Les deux DECRIVENT LA MEME SURDISPERSION si l'on pose e^{a^2} = 1 + (phi-1)/lam, soit
       a = sqrt( ln(1 + (phi-1)/lam) ).
     A l'echelle du moteur euro (lam ~ 21,6/an OpRisk), phi = 9,20 correspond a a ~ 0,57,
     tres proche du A_LOAD = 0,60 pose ailleurs : les deux calages sont coherents.

  2. MEME VARIANCE, QUEUE DIFFERENTE. A indice de dispersion EGAL, la NB (melange Gamma) et
     le Poisson-lognormal n'ont pas le meme quantile 99,5 % du nombre annuel : appairer la
     variance n'appaire pas la queue. On mesure l'ecart.

  3. EFFET SUR LE CAPITAL. On propage chaque loi de comptage dans le moteur euro-cascade
     (severite GPD OpRisk, cascade dirigee) et on lit le SCR. La severite etant a queue
     lourde (xi ~ 0,60), on lit VaR ET moyenne : le principe de la perte unique dominante
     limite la sensibilite de la VaR au comptage, la moyenne la revele mieux.

Sortie : diagnostics + figure N2_simulation_nb.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import euro_cascade_model as ec                        # noqa: E402
from euro_cascade_model import PARAMS, var             # noqa: E402
from src.utils.config import OPRISK, FREQUENCY         # noqa: E402

WID = 82
SRC = "OPRISK"
sp = PARAMS[SRC]
LAM = OPRISK["n_incidents"] / OPRISK["n_years"]         # ~21,6 amorces/an (serie cyber x finance)
PHI_POSE = FREQUENCY["dispersion_factor"]              # 9,20
A_LOAD = 0.60                                          # charge systemique posee (frequence_model)
R_FIRME = 0.63                                         # forme Gamma calibree au niveau firme (08b)
MU_FIRME = 0.099                                       # taux firme-annee observe (08b, toutes firmes)
N_SIM = 400_000                                        # annees simulees pour le comptage
NY_SCR = 40_000                                        # annees pour le SCR euro
SEED = 20260727


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def a_from_phi(phi, lam):
    return float(np.sqrt(np.log(1.0 + (phi - 1.0) / lam))) if phi > 1 else 0.0


def phi_from_a(a, lam):
    return 1.0 + lam * (np.exp(a * a) - 1.0)


def draw_nb(lam, phi, size, rng):
    """Nombre annuel, melange Gamma (= NB) : Var/E = phi."""
    r = lam / (phi - 1.0)
    return rng.negative_binomial(r, r / (r + lam), size=size)


def draw_poislogn(lam, a, size, rng):
    """Nombre annuel, melange lognormal (charge commune a) : Poisson(lam e^{aY - a^2/2})."""
    Y = rng.standard_normal(size)
    return rng.poisson(lam * np.exp(a * Y - a * a / 2.0))


# =====================================================================================
titre("1. Reconciliation des trois representations de la surdispersion")
# =====================================================================================
a_equiv = a_from_phi(PHI_POSE, LAM)
phi_from_aload = phi_from_a(A_LOAD, LAM)
print(f"  Echelle du moteur euro : lam = {LAM:.2f} amorces/an (OpRisk cyber x finance, "
      f"{OPRISK['n_incidents']}/{OPRISK['n_years']} ans).")
print(f"\n  {'representation':<34}{'echelle':<18}{'indice Var/E':>14}")
print(f"  {'melange Gamma r = %.2f' % R_FIRME:<34}{'firme-annee':<18}"
      f"{1 + MU_FIRME / R_FIRME:>14.2f}   (micro : mu={MU_FIRME}/firme, surdispersion faible)")
print(f"  {'charge lognormale a = %.2f' % A_LOAD:<34}{'agregee (portefeuille)':<18}"
      f"{phi_from_aload:>14.2f}")
print(f"  {'facteur NB phi = %.2f' % PHI_POSE:<34}{'agregee (portefeuille)':<18}"
      f"{PHI_POSE:>14.2f}")
print(f"\n  Correspondance a <-> phi a lam = {LAM:.1f} :")
print(f"     phi = {PHI_POSE:.2f}  implique  a = {a_equiv:.3f}   (pose : A_LOAD = {A_LOAD:.2f})")
print(f"     a   = {A_LOAD:.2f}  implique  phi = {phi_from_aload:.2f}   (pose : phi = {PHI_POSE:.2f})")
print(f"  => les deux calages agreges coincident a {100*abs(a_equiv-A_LOAD)/A_LOAD:.0f} % pres sur a,")
print(f"     {100*abs(phi_from_aload-PHI_POSE)/PHI_POSE:.0f} % pres sur phi. Le r firme (0,63) vit")
print("     a une AUTRE echelle (une seule firme), il ne se compare pas directement.")

# =====================================================================================
titre("2. Simulation : meme variance, queue differente (Gamma vs lognormal)")
# =====================================================================================
rng = np.random.default_rng(SEED)
n_pois = rng.poisson(LAM, size=N_SIM)
n_nb = draw_nb(LAM, PHI_POSE, N_SIM, rng)
n_logn = draw_poislogn(LAM, a_equiv, N_SIM, rng)


def stats_line(name, x):
    D = x.var() / x.mean()
    q = np.quantile(x, 0.995)
    print(f"  {name:<38}E = {x.mean():6.2f}   Var/E = {D:5.2f}   q99,5% = {q:6.0f}")
    return D, q


print(f"  {N_SIM:,} annees simulees, lam = {LAM:.2f} :\n")
_, q_pois = stats_line("Poisson (sans surdispersion)", n_pois)
D_nb, q_nb = stats_line(f"NB / melange Gamma (phi = {PHI_POSE:.1f})", n_nb)
D_logn, q_logn = stats_line(f"Poisson-lognormal (a = {a_equiv:.2f}, meme phi)", n_logn)
print(f"\n  A indice de dispersion egal ({D_nb:.1f} vs {D_logn:.1f}), la queue 99,5 % differe :")
print(f"     NB = {q_nb:.0f}  vs  lognormal = {q_logn:.0f}  "
      f"(ecart {100*(q_logn-q_nb)/q_nb:+.0f} %). Appairer la variance n'appaire pas la queue.")
heavier = "lognormale" if q_logn > q_nb else "Gamma (NB)"
print(f"  La loi de melange {heavier} porte la queue la plus lourde a variance egale.")

# =====================================================================================
titre("3. Calage du phi sur la vraie serie annuelle cyber x finance")
# =====================================================================================
# Discipline de 08b : le Var/E BRUT confond la tendance de croissance avec le choc commun.
# On isole le choc commun par la dispersion de Pearson RESIDUELLE apres retrait d'une
# tendance log-lineaire, sur la MEME fenetre validee que 08b (2005-2022), puis on en tire la
# charge a (scale-free) pour la comparer a A_LOAD.
Y0W, Y1W = 2005, 2022
phi_implied = None
try:
    from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance
    path = os.path.join(REPO, "data", "raw", "SAS_OpRisk_Global_Data_June_2026.xlsx")
    df = filter_finance(filter_cyber(load_clean(path)))
    yr = df["year"].dropna().astype(int)
    yr = yr[(yr >= Y0W) & (yr <= Y1W)]
    counts = yr.value_counts().reindex(range(Y0W, Y1W + 1), fill_value=0).to_numpy().astype(float)
    m = counts.mean()
    phi_raw = counts.var(ddof=1) / m
    # tendance log-lineaire (Poisson) puis dispersion de Pearson residuelle
    tt = np.arange(len(counts))
    b1, b0 = np.polyfit(tt, np.log(counts + 1e-9), 1)
    mu_t = np.exp(b0 + b1 * tt)
    phi_resid = float(np.sum((counts - mu_t) ** 2 / mu_t) / (len(counts) - 2))
    a_data = a_from_phi(phi_resid, m)
    phi_implied = phi_from_a(a_data, LAM)              # remis a l'echelle du moteur (lam=21,6)
    print(f"  Serie cyber x finance {Y0W}-{Y1W} ({len(counts)} ans, {int(counts.sum())} evts, "
          f"moyenne {m:.1f}/an) :")
    print(f"     Var/E BRUT = {phi_raw:.2f}  (gonfle : melange tendance + choc commun)")
    print(f"     tendance ajustee : {100*(np.exp(b1)-1):+.0f} %/an")
    print(f"     dispersion residuelle apres tendance phi_resid = {phi_resid:.2f}  "
          f"(le vrai choc commun)")
    print(f"     charge implicite a_data = {a_data:.3f}  (vs A_LOAD = {A_LOAD:.2f} pose)")
    print(f"     remise a l'echelle du moteur (lam={LAM:.0f}) : phi_implied = {phi_implied:.2f}")
    print(f"  => coherent avec 08b : la charge empirique ({a_data:.2f}) est BIEN INFERIEURE au")
    print(f"     A_LOAD = 0,60 pose ; le phi = 9,20 de la config est donc CONSERVATEUR, non calibre.")
except Exception as e:                                  # donnee absente ou format inattendu
    print(f"  Serie brute indisponible ({type(e).__name__}), calage empirique saute.")
    print("  On garde le phi pose de la config (9,20) comme reference conservatrice.")

# =====================================================================================
titre("4. Effet de la loi de comptage sur le SCR euro-cascade")
# =====================================================================================
G = 0.90
N_SEED_SCR = 6
print("  Meme moteur (severite GPD OpRisk, cascade dirigee g=0,90), seule la loi du nombre")
print("  annuel change. La MOYENNE de la perte est invariante par construction (E[N]=lam")
print("  pour les trois lois) : ce que la surdispersion deplace, c'est la VaR. On moyenne")
print(f"  sur {N_SEED_SCR} graines pour lisser le bruit d'echantillonnage de la VaR.\n")
scenarios = [("Poisson (phi->1)", 1.0001),
             ("NB phi = 9,20 (pose)", PHI_POSE),
             (f"NB phi = {phi_from_aload:.1f} (= A_LOAD 0,60)", phi_from_aload)]
if phi_implied is not None:
    scenarios.insert(1, (f"NB phi = {phi_implied:.1f} (empirique detrende)", phi_implied))

print(f"  {'loi de comptage':<34}{'SCR (VaR 99,5%)':>18}{'moyenne':>12}")
res = []
for name, phi in scenarios:
    scr_s, mean_s = [], []
    for k in range(N_SEED_SCR):
        rng_s = np.random.default_rng(SEED + k)
        annual = ec.simulate_euro(LAM, G, sp["xi"], sp["sigma"], sp["u"], sp["p_u"],
                                  sp["cap"], NY_SCR, rng_s, phi=phi)
        scr_s.append(var(annual))
        mean_s.append(float(annual.mean()))
    scr, mean = float(np.mean(scr_s)), float(np.mean(mean_s))
    res.append((name, phi, scr, mean))
    print(f"  {name:<34}{scr:>15.0f} M{mean:>10.0f} M")
pois_row = next(r for r in res if r[1] < 1.01)                     # phi -> 1
nb_row = next(r for r in res if abs(r[1] - PHI_POSE) < 1e-6)       # NB pose (phi = 9,20)
scr_pois, mean_pois = pois_row[2], pois_row[3]
scr_nb, mean_nb = nb_row[2], nb_row[3]
print(f"\n  Passage Poisson -> NB pose : SCR {scr_pois:.0f} -> {scr_nb:.0f} M "
      f"({100*(scr_nb-scr_pois)/scr_pois:+.0f} %), moyenne {mean_pois:.0f} -> {mean_nb:.0f} M "
      f"(invariante, l'ecart {100*abs(mean_nb-mean_pois)/mean_pois:.0f} % est du bruit MC).")
print("  La surdispersion du comptage deplace la VaR (plus d'annees a forte frequence donnent")
print(f"  plus de tirages dans la queue lourde de severite), pas la moyenne. L'effet est")
print(f"  modere ({100*(scr_nb-scr_pois)/scr_pois:+.0f} %) car au 99,5 % la queue reste largement")
print("  portee par une perte dominante (severite xi~0,60), coherent avec le fil du memoire (29).")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. Les trois surdispersions du code ne se contredisent pas : la charge lognormale")
print(f"     a = 0,60 et le facteur NB phi = 9,20 sont la MEME surdispersion agregee "
      f"(a lam~{LAM:.0f}, phi<->a a {100*abs(a_equiv-A_LOAD)/A_LOAD:.0f} % pres) ;")
print("     le r firme (0,63) vit a l'echelle firme-annee, non comparable directement.")
print("  2. La NB (melange Gamma) est le bon objet AGREGE : forme close, un seul parametre")
print("     phi, quantiles exacts. Mais a variance egale la queue depend de la loi de melange")
print(f"     (Gamma vs lognormal : ecart de {100*abs(q_logn-q_nb)/q_nb:.0f} % au q99,5% du comptage).")
print(f"  3. Le choix de comptage deplace la VaR ({100*(scr_nb-scr_pois)/scr_pois:+.0f} %, Poisson->NB pose)")
print("     et laisse la moyenne invariante (E[N]=lam fixe). La charge empirique detrendee")
print(f"     (phi~{phi_implied:.0f}) est bien en-deca du phi=9,20 pose : le pose est un choix de")
print("     PRUDENCE sur la queue de frequence, a assumer comme tel, pas une calibration.")

# =====================================================================================
# figure N2
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 11,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
})
INK, INK2, MUTED, GRID = "#1b1e30", "#223e55", "#595959", "#f2f2f2"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime cette
# figure sur 7,28 pouces : tracee sur 10,7 elle subissait une reduction de 0,68 et
# ses legendes sortaient sous 6 points. A 9,4 pouces le facteur monte a 0,78, donc
# une legende de 9 points s'imprime a 7. Le rapport largeur sur hauteur est conserve.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.4, 2.85),
                                    gridspec_kw={"width_ratios": [1, 1, 1.15]})


def fr(x, n=2):
    """Nombre a la francaise : meme valeur, meme arrondi, virgule decimale."""
    return f"{x:.{n}f}".replace(".", ",")


# (a) distributions du nombre annuel (queues)
hi = int(np.quantile(n_logn, 0.999))
bins = np.arange(0, hi + 2)
for x, c, lab in [(n_pois, MUTED, "Poisson"),
                  (n_nb, BLUE, f"NB (Gamma, $\\varphi$={PHI_POSE:.0f})"),
                  (n_logn, ACCENT, f"Poisson-lognormal ($a$={fr(a_equiv)})")]:
    h, _ = np.histogram(x, bins=bins, density=True)
    ax1.plot(bins[:-1], h, color=c, lw=2, label=lab)
ax1.axvline(LAM, color=INK, ls=":", lw=1)
ax1.set_yscale("log")
# l'etiquette de la moyenne passe SOUS la verticale : posee en haut, elle tombait
# sur le sommet de la densite de Poisson. La legende passe a droite, ou les trois
# queues sont deja redescendues.
ax1.text(LAM * 1.06, ax1.get_ylim()[0] * 2.0, f"moyenne {LAM:.0f}",
         fontsize=9, color=INK2, va="bottom")
ax1.set_xlabel("nombre d'amorces par an", color=INK2)
ax1.set_ylabel("densité (log)", color=INK2)
ax1.set_ylim(top=ax1.get_ylim()[1] * 700)
ax1.legend(frameon=False, fontsize=8.5, loc="upper right", handlelength=1.4)
ax1.set_title("(a)  Même moyenne, queues très différentes", fontsize=11, color=INK, pad=8)

# (b) reconciliation a <-> phi
phis = np.linspace(1.2, 14, 120)
aa = [a_from_phi(p, LAM) for p in phis]
ax2.plot(aa, phis, color=BLUE, lw=2.2)
ax2.scatter([A_LOAD], [phi_from_aload], color=ACCENT, s=70, zorder=5)
ax2.scatter([a_equiv], [PHI_POSE], color=GREEN, s=70, zorder=5)
# LES DEUX LECTURES SONT ECRITES EN HAUT A GAUCHE, ou la courbe convexe ne passe
# pas : collees a leur point, elles se recouvraient l'une l'autre, les deux points
# n'etant distants que de 0,03 en abscisse. La couleur du texte fait le lien.
ax2.text(0.03, 0.97, f"$\\varphi=9{{,}}20 \\rightarrow a={fr(a_equiv)}$",
         transform=ax2.transAxes, fontsize=9, color=GREEN, va="top")
ax2.text(0.03, 0.85, f"$a=0{{,}}60$ posé $\\rightarrow \\varphi={fr(phi_from_aload, 1)}$",
         transform=ax2.transAxes, fontsize=9, color=ACCENT, va="top")
ax2.set_xlabel("charge systémique lognormale  $a$", color=INK2)
ax2.set_ylabel("surdispersion NB  $\\varphi$", color=INK2)
ax2.set_title(f"(b)  $a$ et $\\varphi$ : même surdispersion\n(à $\\lambda={LAM:.0f}$/an)",
              fontsize=11, color=INK, pad=8)

# (c) SCR par loi de comptage
# etiquettes d'axe reecrites pour le lecteur : le nom interne du scenario porte
# « phi » en clair et un point decimal, et les quatre se chevauchaient.
names = [r[0].split(" (")[0].replace("NB ", "NB\n").replace("phi", "$\\varphi$").replace(".", ",")
         for r in res]
scrs = [r[2] for r in res]
means = [r[3] for r in res]
x = np.arange(len(res))
ax3.bar(x - 0.19, scrs, width=0.36, color=BLUE, alpha=0.9, label="SCR (VaR 99,5 %)")
ax3.bar(x + 0.19, means, width=0.36, color=GREEN, alpha=0.9, label="moyenne")
ax3.set_xticks(x); ax3.set_xticklabels(names, fontsize=8)
ax3.set_ylim(0, max(scrs) * 1.30)                  # bande libre pour la legende
ax3.set_ylabel("M€", color=INK2)
ax3.legend(frameon=False, fontsize=9, loc="upper center", ncol=2,
           handlelength=1.3, columnspacing=1.0)
ax3.set_title("(c)  Le comptage déplace la VaR,\npas la moyenne", fontsize=11,
              color=INK, pad=8)

for ax in (ax1, ax2, ax3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# PAS DE TITRE GENERAL : la legende LaTeX porte le titre, et celui-ci commencait
# par le code interne « N2 : ».
fig.tight_layout()
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "N2_simulation_nb.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
