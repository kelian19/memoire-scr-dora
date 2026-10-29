#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08d : remplacer le choc commun par un POISSON COMPOSE (arrivees groupees).

Conclusion de 08c : le clustering est SIMULTANE (une cause chez un tiers frappe N
victimes le meme jour), pas sequentiel. Le bon objet n'est donc pas un Hawkes mais un
processus a ARRIVEES GROUPEES. On le construit et on le calibre ici.

STRUCTURE A DEUX COMPOSANTES (idiosyncratique / systemique) :

    N_annuel = Poisson(lambda_0)  +  somme_{i=1..Poisson(lambda_B)} K_i

  - composante de FOND : incidents ordinaires, independants, taux lambda_0 ;
  - composante de CAUSE COMMUNE : lambda_B evenements par an, chacun touchant
    K_i entites SIMULTANEMENT (K a queue lourde : MOVEit = 191 en un jour).

Pour un Poisson compose, la surdispersion se lit en fermeture analytique :
    Var(N)/E(N) = E(K^2)/E(K)
Elle vient donc ENTIEREMENT de la variabilite de la taille des paquets. Un seul
evenement a 191 victimes pese plus dans la queue que des centaines d'incidents isoles.

NIVEAU D'OBSERVATION. Le batch est un phenomene de PORTEFEUILLE : pour une entite
isolee, une cause commune n'est qu'un incident de plus. C'est quand on porte un
PORTEFEUILLE d'entites qu'un MOVEit frappe plusieurs assures d'un coup. C'est donc
exactement la maille du risque d'ACCUMULATION, celle qui fait le SCR.

CAVEAT ASSUME. Les jours de forte concentration melangent de vraies causes communes
et des LOTS DE DECLARATION (un Etat qui deverse ses notifications). La loi de taille
estimee est donc une BORNE HAUTE de l'accumulation reelle. On le dit.

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

sys.path.insert(0, HERE)
from frequence_model import A_LOAD  # noqa: E402  (choc commun pose, pour comparaison)

Y0, Y1 = 2016, 2023
A_CAL = 0.122          # charge systemique calibree en 08b (melange de Poisson)
NSIM = 400_000
RNG = np.random.default_rng(20260720)
W = 78


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# =====================================================================================
titre("Donnees et separation fond / cause commune")
# =====================================================================================
d = pd.read_excel(SRC, sheet_name="Data_Breach_Chronology",
                  usecols=["breach_date", "organization_type"])
d["bd"] = pd.to_datetime(d.breach_date.astype(str).str.strip(), errors="coerce")
bsf = d[(d.organization_type == "BSF") & d.bd.notna()]
bsf = bsf[(bsf.bd.dt.year >= Y0) & (bsf.bd.dt.year <= Y1)]

idx = pd.date_range(f"{Y0}-01-01", f"{Y1}-12-31", freq="D")
cnt = bsf.groupby(bsf.bd.dt.normalize()).size().reindex(idx, fill_value=0)
k = cnt.to_numpy().astype(float)
nd = len(k)
t = np.arange(nd, dtype=float) / 365.25


def nll(b):
    mu = np.exp(b[0] + b[1] * t)
    return -np.sum(k * np.log(mu) - mu - special.gammaln(k + 1))


b = optimize.minimize(nll, x0=[np.log(k.mean()), 0.0], method="Nelder-Mead",
                      options={"maxiter": 4000}).x
mu_t = np.exp(b[0] + b[1] * t)

# Un jour est « cause commune » si son compte depasse ce qu'un fond de Poisson peut
# produire : quantile 1 - 1/nd (on tolere ~1 faux positif sur toute la fenetre).
thr = stats.poisson.ppf(1.0 - 1.0 / nd, mu_t)
is_cc = k > thr
K_obs = k[is_cc].astype(int)          # tailles de paquet observees
n_years = (Y1 - Y0 + 1)

print(f"jours observes : {nd} | evenements : {int(k.sum())} | moyenne {k.mean():.2f}/j")
print(f"seuil de detection : {thr.min():.0f} a {thr.max():.0f} evts/jour "
      "(quantile de Poisson du fond)")
print(f"\njours de CAUSE COMMUNE detectes : {is_cc.sum()} "
      f"({is_cc.sum()/n_years:.1f} par an)")
print(f"evenements qu'ils portent : {int(K_obs.sum())} "
      f"({K_obs.sum()/k.sum():.1%} du total)")
print(f"tailles de paquet : min={K_obs.min()} | mediane={np.median(K_obs):.0f} "
      f"| max={K_obs.max()}")
print("\n10 plus gros paquets :", sorted(K_obs)[::-1][:10])

lam_B = is_cc.sum() / n_years                       # causes communes par an
bg = k[~is_cc]
lam_0 = bg.sum() / n_years                          # fond par an
EK, EK2 = K_obs.mean(), (K_obs ** 2).mean()
print(f"\nlambda_0 (fond)          = {lam_0:.1f} evts/an")
print(f"lambda_B (causes comm.)  = {lam_B:.2f} /an | E[K] = {EK:.1f} | "
      f"E[K^2]/E[K] = {EK2/EK:.1f}")
print(f"=> indice de dispersion apporte par la seule composante groupee : "
      f"{EK2/EK:.1f}")

# queue de la loi de taille : indice de Pareto discret (Hill)
Ks = np.sort(K_obs)[::-1]
nh = max(5, len(Ks) // 3)
hill = 1.0 / np.mean(np.log(Ks[:nh] / Ks[nh]))
print(f"indice de queue (Hill sur le tiers superieur) : alpha = {hill:.2f}  "
      f"({'queue lourde, variance non bornee' if hill < 2 else 'variance finie'})")

# =====================================================================================
titre("Ce que la structure par paquets fait a la queue annuelle")
# =====================================================================================
M = k.sum() / n_years                # meme moyenne annuelle pour les trois modeles
print(f"moyenne annuelle commune aux trois modeles : {M:.0f} evenements/an\n")


def sim_poisson(n):
    return RNG.poisson(M, n)


def sim_mixed(n, a):
    Y = RNG.standard_normal(n)
    return RNG.poisson(M * np.exp(a * Y - a * a / 2.0))


def sim_batch(n):
    """Fond + somme de paquets tires par bootstrap sur les tailles observees."""
    out = RNG.poisson(lam_0, n).astype(float)
    nb = RNG.poisson(lam_B, n)
    mx = nb.max()
    if mx > 0:
        draws = RNG.choice(K_obs, size=(n, mx))
        mask = np.arange(mx)[None, :] < nb[:, None]
        out += (draws * mask).sum(axis=1)
    return out


sims = {
    "Poisson (aucune dependance)": sim_poisson(NSIM).astype(float),
    f"melange de Poisson (a={A_CAL})": sim_mixed(NSIM, A_CAL).astype(float),
    f"melange de Poisson (a={A_LOAD} pose)": sim_mixed(NSIM, A_LOAD).astype(float),
    "Poisson COMPOSE (paquets)": sim_batch(NSIM),
}

print(f"{'modele':<36}{'moyenne':>9}{'Var/E':>9}{'q99,5':>9}{'ratio':>8}")
base_q = None
res = {}
for name, s in sims.items():
    q = np.quantile(s, 0.995)
    if base_q is None:
        base_q = q
    res[name] = (s.mean(), s.var() / s.mean(), q)
    print(f"{name:<36}{s.mean():>9.0f}{s.var()/s.mean():>9.1f}{q:>9.0f}"
          f"{q/base_q:>8.2f}")

q_pois = res["Poisson (aucune dependance)"][2]
q_batch = res["Poisson COMPOSE (paquets)"][2]
q_mixcal = res[f"melange de Poisson (a={A_CAL})"][2]
q_posed = res[f"melange de Poisson (a={A_LOAD} pose)"][2]

print("\nLECTURE. Deux enseignements, et le second corrige une intuition naturelle.")
print(f"\n(1) La simultaneite ajoute une vraie queue AU-DELA du melange calibre :")
print(f"    {q_batch:.0f} (paquets) contre {q_mixcal:.0f} (melange a={A_CAL}), soit "
      f"+{100*(q_batch/q_mixcal-1):.0f} % a moyenne egale.")
print("    Faire varier l'INTENSITE ne remplace pas la SIMULTANEITE : deux mecanismes.")
print(f"\n(2) MAIS le A_LOAD={A_LOAD} pose reste PLUS conservateur que le modele a paquets")
print(f"    ({q_posed:.0f} contre {q_batch:.0f}, soit x{q_posed/q_batch:.1f}). Il serait donc faux de dire")
print("    que la structure groupee 'charge plus qu'un facteur commun' : tout depend du")
print("    niveau du facteur. Le calage pose n'est pas justifiable comme ESTIMATION de la")
print("    variation d'intensite (08b : a = 0,122), mais il COUVRE l'accumulation observee.")
print("    A formuler comme un choix de prudence assume, et non comme une mesure.")
print(f"\nATTENTION a l'estimation de {q_batch:.0f} : le bootstrap retire les tailles OBSERVEES,")
print(f"donc plafonnees a {K_obs.max()}. Avec alpha = {hill:.2f} (variance non bornee), la vraie")
print("queue du modele a paquets est vraisemblablement PLUS lourde que simulee ici.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("1. La composante groupee porte l'essentiel du risque d'accumulation :")
print(f"   {is_cc.sum()} jours sur {nd} ({is_cc.sum()/nd:.1%}) concentrent "
      f"{K_obs.sum()/k.sum():.0%} des sinistres.")
print(f"2. La loi de taille de paquet a une queue lourde (Hill alpha = {hill:.2f}).")
if hill < 2:
    print("   alpha < 2 : la VARIANCE de la taille de paquet n'est pas bornee. Toute")
    print("   estimation de Var/E est alors instable par nature, et le SCR est pilote")
    print("   par le plus gros evenement observe. A dire explicitement.")
print("3. Le melange de Poisson CALIBRE sous-estime la queue a moyenne egale : il fait")
print("   varier l'INTENSITE, il ne cree pas de SIMULTANEITE. Deux mecanismes distincts.")
print(f"4. En revanche le A_LOAD={A_LOAD} pose enveloppe le modele a paquets. Le calage")
print("   actuel n'est donc pas a corriger a la baisse sans precaution : il n'est pas")
print("   justifie comme estimation, mais il est PRUDENT face a l'accumulation reelle.")
print("\nRECOMMANDATION. Garder le melange de Poisson pour la variation d'intensite, et")
print("lui AJOUTER une composante groupee pour l'accumulation. Le parametre qui compte")
print("n'est plus la charge a, mais la loi de TAILLE DE PAQUET, qui est, elle,")
print("observable et deja partiellement documentee (MOVEit, Lakeview, etc.).")
print("Interet pour la redaction : on remplace un parametre non observable (a) par une")
print("quantite qui a un SENS METIER et se discute avec un souscripteur (combien")
print("d'assures un meme prestataire critique peut-il faire tomber d'un coup ?).")
print("\nCAVEAT. Les jours de concentration melangent vraies causes communes et lots de")
print("declaration : la loi de taille est une BORNE HAUTE de l'accumulation reelle.")
print("Un registre identifiant la cause commune comme UN evenement a N victimes la")
print("corrigerait ; c'est la meme donnee manquante que pour W et pour le Hawkes.")

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
                                    gridspec_kw={"width_ratios": [1.1, 1, 1.1]})

# (a) separation fond / causes communes
ax1.scatter(cnt.index[~is_cc], k[~is_cc], s=5, color=BL[1], alpha=0.55, label="fond")
ax1.scatter(cnt.index[is_cc], k[is_cc], s=26, color=ACCENT, zorder=3,
            label="cause commune detectee")
ax1.plot(cnt.index, thr, color=BL[2], lw=1.4, ls="--", label="seuil de detection")
ax1.set_yscale("symlog", linthresh=10)
ax1.set_ylabel("sinistres par jour (BSF)", color=INK2)
ax1.set_xlabel("date de survenance", color=INK2)
ax1.legend(frameon=False, fontsize=8.2, loc="upper left")
ax1.set_title(f"(a)  {is_cc.sum()} jours portent {K_obs.sum()/k.sum():.0%} des sinistres",
              fontsize=11, color=INK, pad=8)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# (b) loi de taille de paquet, en log-log
srt = np.sort(K_obs)[::-1]
surv = np.arange(1, len(srt) + 1) / len(srt)
ax2.loglog(srt, surv, "o", ms=5, color=ACCENT, label="tailles observees")
xs = np.linspace(srt.min(), srt.max(), 100)
ax2.loglog(xs, (xs / srt.min()) ** (-hill), color=BL[2], lw=1.8,
           label=f"Pareto ajuste $\\alpha$={hill:.2f}")
ax2.set_xlabel("taille de paquet $K$ (victimes le meme jour)", color=INK2)
ax2.set_ylabel("P(taille > K)", color=INK2)
ax2.legend(frameon=False, fontsize=8.2)
ax2.set_title("(b)  Une queue lourde de tailles", fontsize=11, color=INK, pad=8)
ax2.annotate("MOVEit", xy=(srt[0], surv[0]), xytext=(srt[0] * 0.28, surv[0] * 5.0),
             fontsize=8.5, color=ACCENT,
             arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# (c) consequence sur la queue annuelle
order = ["Poisson (aucune dependance)", f"melange de Poisson (a={A_CAL})",
         "Poisson COMPOSE (paquets)", f"melange de Poisson (a={A_LOAD} pose)"]
labs = ["Poisson\n(independance)", f"melange\n$a$={A_CAL} (calibre)",
        "Poisson COMPOSE\n(paquets)", f"melange\n$a$={A_LOAD} (pose)"]
qs = [res[o][2] for o in order]
cols = [MUTED, BL[1], ACCENT, BL[0]]
ax3.bar(range(4), qs, color=cols, edgecolor="#fcfcfb", width=0.62)
ax3.axhline(M, color=INK, lw=1.1, ls=":", label=f"moyenne commune ({M:.0f}/an)")
for i, q in enumerate(qs):
    ax3.text(i, q + 12, f"{q:.0f}\n($\\times${q/qs[0]:.2f})", ha="center", fontsize=8.5,
             color=INK2)
ax3.set_xticks(range(4)); ax3.set_xticklabels(labs, fontsize=8.2)
ax3.set_ylabel("quantile a 99,5 % du nombre annuel", color=INK2)
ax3.set_ylim(0, max(qs) * 1.30)
ax3.legend(frameon=False, fontsize=8.2, loc="upper left")
ax3.set_title("(c)  A moyenne egale, la queue change", fontsize=11, color=INK, pad=8)
ax3.text(0.5, 0.62, "les paquets depassent le melange calibre,\n"
         "mais le calage pose les enveloppe", transform=ax3.transAxes, ha="center",
         fontsize=8.2, color=INK2, style="italic")
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)

fig.suptitle("P : intensite et simultaneite sont deux mecanismes distincts",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "P_poisson_compose.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
