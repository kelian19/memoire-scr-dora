#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23 : Benchmark copule. Ce qu'une copule symetrique ne peut structurellement pas faire.

Le memoire a REMPLACE le LDA a copule par la cascade dirigee. Ce script chiffre ce que ce
remplacement apporte, en confrontant la cascade a des jumeaux copule construits POUR lui
ressembler. Deux tests complementaires, perimetre OpRisk (le PRC plafonne compresse tout).

TEST 1, LA PHOTO. Marges par pilier EXACTEMENT celles de la cascade (pertes annuelles par
pilier touche, etat tous Conforme, simulate_euro_pp by_pillar) ; seule la structure de
dependance change : gaussienne / Student (nu=4) / independance, matrice R calee sur la
cascade (Spearman -> Pearson). Mesures : SCR de base, et CO-OCCURRENCE des extremes (nb de
piliers au-dela de leur propre quantile 99 % dans les annees de queue). Verdict rapporte tel
quel : en queue lourde (xi ~ 0,6) la queue ANNUELLE obeit au principe de la perte unique
dominante (subexponentialite), si bien que la gaussienne retombe sur le SCR cascade et que
la Student le SURESTIME en forcant des co-extremes que le mecanisme ne produit pas. Le
niveau de base n'identifie donc pas la dependance ; le choix de copule est a la fois non
identifiable et consequent, la ou la cascade produit sa dependance par mecanisme.

TEST 2, L'INTERVENTION. Le vrai discriminant : basculer un pilier en Non conforme est une
INTERVENTION, pas un conditionnement. Un LDA-copule par pilier (le monde de l'ancien
memoire) exprime les canaux frequence et detection (les marges dependent de l'etat du
pilier), mais :
  - le canal PROPAGATION est inexprimable (aucun parametre de marge ne porte g) ;
  - la dependance est FIGEE (la copule ne durcit pas quand la conformite se degrade) ;
  - les marges ne dependant que de l'etat de LEUR pilier et la copule etant fixe, la perte
    moyenne totale est EXACTEMENT additive : interaction moyenne = 0 par construction.
    L'interaction moyenne mesuree de la cascade (+528 M, script 20b) est donc une signature
    FALSIFIABLE qu'aucun modele copule-marges ne peut produire.
Mesures : Delta_k (un pilier NC a la fois), Delta total, interaction moyenne, et la
PRIORITE de remediation impliquee (classement des Delta_k) : la copule suit les marges
(profil receptacle), la cascade suit les sources (P1 en tete, = ROOT).

CRN partout (meme U de copule entre configurations ; meme graine cascade entre configs).
Choix non calibres identiques a 16b/20 (3 canaux ; nu = 4 en axe de stress comme au 11).
Ne touche ni src/ ni memoire/.

Sortie : diagnostics + figure S18_benchmark_copule.png. Verifie exit 0.
"""

import os
import sys

import numpy as np
from scipy.stats import norm, rankdata, t as student_t, chi2
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                              # noqa: E402
from euro_cascade_model import PARAMS, var                   # noqa: E402
from src.aggregation.lda import simulate_remediation_severity  # noqa: E402
import scr_engine as eng                                      # noqa: E402

W = 74
PIL = eng.PIL
PIL_LAB = {1: "P1 gouvernance", 2: "P2 incidents", 3: "P3 tests",
           4: "P4 tiers", 5: "P5 partage"}
SCENARIO = {"C": "S0_conforme", "NC": "S2_non_conforme"}
G_PROP = {"C": 0.45, "NC": 0.90}
PU_MULT = {"C": 0.85, "NC": 1.20}
MULT_ETAT = {st: ec.lambda_scenario("OPRISK", sc, mode="center") / PARAMS["OPRISK"]["lam_ref"]
             for st, sc in SCENARIO.items()}
_SHARE = {j: eng.LAMBDA[j] for j in PIL}
_STOT = sum(_SHARE.values())
SHARE = {j: _SHARE[j] / _STOT for j in PIL}

NY = 60_000
SEED = 909
NU = 4                     # ddl Student (non identifiable : axe de stress, comme au 11)
PHI = ec.PHI

sp = PARAMS["OPRISK"]


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def cascade_annual(state_map, by_pillar=False, ny=NY, seed=SEED):
    """Pertes annuelles de la cascade pour une configuration {pilier -> etat} (CRN)."""
    lam_vec = {j: sp["lam_ref"] * SHARE[j] * MULT_ETAT[state_map[j]] for j in PIL}
    g_vec = {j: G_PROP[state_map[j]] for j in PIL}
    p_u_vec = {j: min(0.999, sp["p_u"] * PU_MULT[state_map[j]]) for j in PIL}
    rng = np.random.default_rng(seed)
    return ec.simulate_euro_pp(lam_vec, g_vec, sp["xi"], sp["sigma"], sp["u"],
                               p_u_vec, sp["cap"], ny, rng, by_pillar=by_pillar)


def marge_lda(j, etat, ny=NY, seed_off=0):
    """Marge LDA du pilier j sous un etat : compound NegBin(lam_j) x GPD (p_u module).

    Le monde copule par pilier : frequence et detection dans la marge, la propagation
    n'a AUCUN parametre ou vivre. Graine propre a (j, etat) : reutilisee entre configs.
    """
    lam = sp["lam_ref"] * SHARE[j] * MULT_ETAT[etat]
    p_u = min(0.999, sp["p_u"] * PU_MULT[etat])
    rng = np.random.default_rng(10_000 + 100 * j + (0 if etat == "C" else 1) + seed_off)
    r = lam / (PHI - 1.0)
    p = r / (r + lam)
    counts = rng.negative_binomial(r, p, size=ny)
    annual = np.zeros(ny)
    T = int(counts.sum())
    if T == 0:
        return annual
    yrs = np.repeat(np.arange(ny), counts)
    sev = simulate_remediation_severity(T, sp["xi"], sp["sigma"], sp["u"], p_u,
                                        sp["cap"], rng)
    np.add.at(annual, yrs, sev)
    return annual


def couple(margins, U):
    """Somme annuelle sous copule : U (ny x 5) -> quantiles empiriques des marges."""
    total = np.zeros(U.shape[0])
    for c, j in enumerate(PIL):
        total += np.quantile(margins[j], U[:, c], method="linear")
    return total


# ============================================================ cascade : configs de reference
titre("Cascade de reference (OpRisk, canaux frequence + propagation + detection)")
M0 = cascade_annual({j: "C" for j in PIL}, by_pillar=True)      # tous C, par pilier
casc = {"C": M0.sum(axis=1)}
casc["NC"] = cascade_annual({j: "NC" for j in PIL})
for k in PIL:
    cfg = {j: "C" for j in PIL}
    cfg[k] = "NC"
    casc[f"solo{k}"] = cascade_annual(cfg)
scr_c = {name: var(v) for name, v in casc.items()}
mean_c = {name: float(v.mean()) for name, v in casc.items()}
print(f"  SCR tous C = {scr_c['C']:.0f} M   SCR tous NC = {scr_c['NC']:.0f} M   "
      f"Delta = {scr_c['NC'] - scr_c['C']:.0f} M")

# ============================================================ TEST 1 : la photo
titre("TEST 1  La photo : memes marges, seule la dependance change (tous C)")
ranks = np.column_stack([rankdata(M0[:, c]) for c in range(len(PIL))])
rho_s = np.corrcoef(ranks, rowvar=False)
R = 2.0 * np.sin(np.pi * rho_s / 6.0)                            # Spearman -> Pearson
np.fill_diagonal(R, 1.0)
Lch = np.linalg.cholesky(R)
rng_u = np.random.default_rng(777)
Z = rng_u.standard_normal((NY, len(PIL))) @ Lch.T
U_GAUSS = norm.cdf(Z)
w_chi = chi2.rvs(NU, size=NY, random_state=np.random.default_rng(778)) / NU
U_T = student_t.cdf(Z / np.sqrt(w_chi)[:, None], df=NU)
U_IND = np.random.default_rng(779).uniform(size=(NY, len(PIL)))

marges_photo = {j: M0[:, c] for c, j in enumerate(PIL)}          # marges cascade exactes
photo = {"cascade": casc["C"],
         "gaussienne": couple(marges_photo, U_GAUSS),
         f"Student nu={NU}": couple(marges_photo, U_T),
         "independance": couple(marges_photo, U_IND)}
q99 = {j: np.quantile(marges_photo[j], 0.99) for j in PIL}       # memes seuils pour tous
scr_photo_ref = var(photo["cascade"])
print(f"\n  {'modele':<16}{'SCR tous C':>12}{'ecart vs cascade':>18}{'co-extremes queue':>19}")
coexc = {}
for lab, tot in photo.items():
    v = var(tot)
    tail = tot >= v
    if lab == "cascade":
        n_ext = (M0[tail] >= np.array([q99[j] for j in PIL])).sum(axis=1).mean()
    else:
        Uu = {"gaussienne": U_GAUSS, f"Student nu={NU}": U_T, "independance": U_IND}[lab]
        n_ext = (Uu[tail] >= 0.99).sum(axis=1).mean()
    coexc[lab] = (v, n_ext)
    print(f"  {lab:<16}{v:>12.0f}{100.0 * (v - scr_photo_ref) / scr_photo_ref:>17.1f}%"
          f"{n_ext:>19.2f}")
print("  Lecture (rapportee telle quelle, contraire a l'intuition copule) : en queue")
print("  lourde la queue annuelle est portee par la perte unique dominante, si bien que")
print("  la dependance compte peu pour le NIVEAU de base : la gaussienne retombe sur le")
print("  SCR cascade. La Student nu=4 le surestime en forcant des co-extremes que le")
print("  mecanisme ne produit pas. Le niveau n'identifie pas la dependance ; le choix de")
print("  copule est non identifiable ET consequent. La cascade n'a pas ce degre de")
print("  liberte : sa dependance est produite par le mecanisme, pas choisie.")

# ============================================================ TEST 2 : l'intervention
titre("TEST 2  L'intervention : basculer un pilier, copule figee vs cascade")
marges = {(j, e): marge_lda(j, e) for j in PIL for e in ("C", "NC")}
cop = {}
for name, cfgmap in [("C", {j: "C" for j in PIL}), ("NC", {j: "NC" for j in PIL})] + \
        [(f"solo{k}", {j: ("NC" if j == k else "C") for j in PIL}) for k in PIL]:
    m = {j: marges[(j, cfgmap[j])] for j in PIL}
    cop[name] = couple(m, U_GAUSS)                                # MEME U : CRN
scr_g = {name: var(v) for name, v in cop.items()}
mean_g = {name: float(v.mean()) for name, v in cop.items()}

d_tot_c = scr_c["NC"] - scr_c["C"]
d_tot_g = scr_g["NC"] - scr_g["C"]
dk_c = {k: scr_c[f"solo{k}"] - scr_c["C"] for k in PIL}
dk_g = {k: scr_g[f"solo{k}"] - scr_g["C"] for k in PIL}
mi_c = (mean_c["NC"] - mean_c["C"]) - sum(mean_c[f"solo{k}"] - mean_c["C"] for k in PIL)
mi_g = (mean_g["NC"] - mean_g["C"]) - sum(mean_g[f"solo{k}"] - mean_g["C"] for k in PIL)
ord_c = sorted(PIL, key=lambda k: dk_c[k], reverse=True)
ord_g = sorted(PIL, key=lambda k: dk_g[k], reverse=True)

print(f"\n  {'pilier bascule en NC':<24}{'Delta_k cascade':>16}{'Delta_k copule':>16}")
for k in sorted(PIL, key=lambda k: dk_c[k], reverse=True):
    print(f"  {PIL_LAB[k]:<24}{dk_c[k]:>15.0f}M{dk_g[k]:>15.0f}M")
print(f"\n  Delta total (tous NC)        cascade {d_tot_c:>8.0f} M    copule {d_tot_g:>8.0f} M")
print(f"  interaction en perte MOYENNE cascade {mi_c:>+8.0f} M    copule {mi_g:>+8.0f} M")
print(f"  priorite impliquee  cascade : {' > '.join(f'P{k}' for k in ord_c)}  (= ROOT)")
print(f"  priorite impliquee  copule  : {' > '.join(f'P{k}' for k in ord_g)}")
print("\n  Lecture : trois manques structurels du monde copule.")
print("  1. La propagation n'a nulle part ou vivre : le Delta total est sous-estime")
print("     (le canal g, 0,45 -> 0,90, est inexprimable dans des marges).")
print("  2. L'interaction moyenne copule est ~0 PAR CONSTRUCTION (marges par etat propre,")
print("     copule figee) : l'interaction cascade mesuree est une signature falsifiable.")
print("  3. La dependance ne durcit pas avec la non-conformite, alors que la cascade")
print("     fait monter g : la copule fige precisement ce que DORA fait bouger.")

# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2 = "#1b1e30", "#223e55"
ACCENT, BLUE, GREEN, GREY = "#a6002e", "#3661ac", "#008d87", "#7d7d7d"

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime
# cette figure sur 7,27 pouces : tracee plus large, elle subissait une
# reduction qui faisait tomber ses etiquettes sous 6 points. Le rapport
# largeur sur hauteur est conserve, donc la figure occupe la meme place
# sur la page ; c'est son texte qui y prend plus de place.
fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.9, 3.52))

# panneau A : Delta_k cascade vs copule
order = sorted(PIL, key=lambda k: dk_c[k], reverse=True)
ypos = np.arange(len(order))
h = 0.38
axA.barh(ypos - h / 2, [dk_c[k] for k in order], h, color=ACCENT, alpha=0.9,
         label="cascade (source)")
axA.barh(ypos + h / 2, [dk_g[k] for k in order], h, color=BLUE, alpha=0.85,
         label="LDA-copule (marges)")
axA.set_yticks(ypos)
axA.set_yticklabels([PIL_LAB[k] for k in order], fontsize=9)
axA.invert_yaxis()
axA.set_xlabel("Delta_k : surcout si ce pilier seul est NC (M€)", fontsize=9.3, color=INK2)
axA.set_title("(A)  L'intervention : la copule suit les marges,\nla cascade suit les sources",
              fontsize=9.8, color=INK, pad=6)
axA.legend(fontsize=8.2, frameon=False, loc="lower right")
axA.grid(alpha=0.25, lw=0.5, axis="x")

# panneau B : co-occurrence d'extremes dans la queue (photo)
labs = list(photo.keys())
vals = [coexc[k][1] for k in labs]
cols = [ACCENT, BLUE, GREEN, GREY]
axB.bar(range(len(labs)), vals, color=cols, alpha=0.9)
for i, v in enumerate(vals):
    axB.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 4),
                 ha="center", fontsize=8.6, color=INK2)
axB.set_xticks(range(len(labs)))
axB.set_xticklabels(labs, fontsize=8.6)
axB.set_ylabel("piliers extremes (> q99 propre) par annee de queue", fontsize=9.0, color=INK2)
axB.set_title("(B)  La photo : perte unique dominante en queue ;\nla Student force des co-extremes que le mecanisme n'a pas",
              fontsize=9.8, color=INK, pad=6)
axB.grid(alpha=0.25, lw=0.5, axis="y")

fig.tight_layout(rect=[0, 0, 1, 0.94])

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S18_benchmark_copule.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
