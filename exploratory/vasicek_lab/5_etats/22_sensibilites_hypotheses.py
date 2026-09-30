#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22 : Sensibilites des HYPOTHESES restantes du chantier multi-etats.

Complete le tornado du 19 (xi, lambda, g, detection) et le residu du 21 par les trois
entrees non calibrees qui n'avaient pas encore leur axe de stress :

  A. COEFFICIENTS JACOBS (a, b). La severite PRC depend de ln L = a + b ln X. La source
     primaire (Jacobs 2014) publie la table des coefficients : a = 7,68 (SE 0,7013),
     b = 0,7584 (SE 0,0697), n = 115, RSE = 0,523. Stresser a et b INDEPENDAMMENT
     ignorerait leur correlation negative ; on stresse donc b a +-1,645 SE en PIVOTANT
     la droite au centroide (x_bar, y_bar) de la regression, comme l'OLS l'impose :
     a(b) = a_hat + (b_hat - b) * x_bar. Le centroide n'est pas publie mais il se DEDUIT
     des SE publies :  SE_a^2 = RSE^2 (1/n + x_bar^2/Sxx)  et  SE_b^2 = RSE^2/Sxx
     =>  x_bar = sqrt((SE_a^2 - RSE^2/n) / SE_b^2) ~ 10,04  (soit ~23 000 enregistrements,
     coherent avec l'echantillon Ponemon). La calibration aval suit la convention du
     pipeline (notebook 13) : seuil u re-derive au percentile 85, p_u = 0,15, refit GPD.

  B. GAMMA (charge systemique de la latente, 0,68 non calibre) : 0,50 / 0,85. N'affecte
     PAS le SCR par etat (les etats y sont conditionnes) ; affecte les probabilites d'etat
     conditionnelles a la crise et le SCR ESPERE (normal vs crise).

  C. ANCRAGE des probabilites d'etat (NC 35/PC 35/C 30, enquetes) : variante optimiste
     (25/40/35) et pessimiste (45/35/20). Meme canal que gamma : l'ancrage deplace le
     melange (SCR espere, point de depart de la trajectoire du 17), jamais le SCR par etat
     ni le Delta NC vs C. Dit explicitement pour eviter un faux debat de calibration.

Graine MC commune. Ne touche ni src/ ni memoire/.
Sortie : diagnostics + figure S17_sensibilites_hypotheses.png. Verifie exit 0.
"""

import os
import sys

import numpy as np
from scipy.stats import norm, genpareto
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import euro_cascade_model as ec                              # noqa: E402
from euro_cascade_model import PARAMS, var                   # noqa: E402
from src.severity.prc_analysis import load_prc, chemin_prc, USD_EUR   # noqa: E402

W = 74
ETATS = ["C", "PC", "NC"]
SCENARIO = {"C": "S0_conforme", "PC": "S1_partiel", "NC": "S2_non_conforme"}
G_PROP = {"C": 0.45, "PC": 0.68, "NC": 0.90}                  # lecture B

# Jacobs 2014, table des coefficients publiee (Data Driven Security)
A_HAT, SE_A = 7.68, 0.7013
B_HAT, SE_B = 0.7584, 0.0697
N_OBS, RSE = 115, 0.523
X_BAR = float(np.sqrt((SE_A ** 2 - RSE ** 2 / N_OBS) / SE_B ** 2))   # centroide deduit

# latente du 16
P_ETAT = {"NC": 0.35, "PC": 0.35, "C": 0.30}
GAMMA_BASE = 0.68
THETA_CRISE = -2.5

NY = 80_000
SEED = 4242


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def etat_probs(theta, gamma, p_etat):
    k_bas = norm.ppf(p_etat["NC"])
    k_haut = norm.ppf(p_etat["NC"] + p_etat["PC"])
    d = np.sqrt(1.0 - gamma ** 2)
    p_nc = norm.cdf((k_bas - gamma * theta) / d)
    p_ncpc = norm.cdf((k_haut - gamma * theta) / d)
    return {"NC": p_nc, "PC": p_ncpc - p_nc, "C": 1.0 - p_ncpc}


def scr_etat_prc(xi, sg, u, p_u, etat, seed=SEED, ny=NY):
    """SCR PRC d'un etat (lecture B), parametres de severite fournis, CRN."""
    lam = ec.lambda_scenario("PRC", SCENARIO[etat], mode="center")
    rng = np.random.default_rng(seed)
    return var(ec.simulate_euro(lam, G_PROP[etat], xi, sg, u, p_u,
                                PARAMS["PRC"]["cap"], ny, rng))


# ============================================================ A. coefficients Jacobs
titre("A. Coefficients Jacobs (a, b) : stress de b pivote au centroide")
print(f"  source primaire : a = {A_HAT} (SE {SE_A}), b = {B_HAT} (SE {SE_B}), "
      f"n = {N_OBS}, RSE = {RSE}")
print(f"  centroide deduit des SE publies : x_bar = {X_BAR:.2f} "
      f"(~{np.exp(X_BAR):,.0f} enregistrements, coherent Ponemon)")
X_ALL = load_prc(chemin_prc(_REPO))[
    "total_affected"].values.astype(float)

axes_b = {"bas (-1,645 SE)": B_HAT - 1.645 * SE_B,
          "central (2014)": B_HAT,
          "haut (+1,645 SE)": B_HAT + 1.645 * SE_B}
res_jacobs = {}
print(f"\n  {'axe':<20}{'b':>8}{'a(b)':>8}{'u=P85':>9}{'xi':>8}{'sigma':>8}"
      f"{'SCR(C)':>9}{'SCR(NC)':>10}{'Delta':>9}")
for lab, b in axes_b.items():
    a = A_HAT + (B_HAT - b) * X_BAR                      # pivot au centroide
    L = np.exp(a + b * np.log(X_ALL)) * USD_EUR / 1e6
    u = float(np.percentile(L, 85))                      # convention pipeline (nb 13)
    exc = L[L > u] - u
    xi, _, sg = genpareto.fit(exc, floc=0)
    xi = float(np.clip(xi, 0.05, 2.5))
    v = {e: scr_etat_prc(xi, sg, u, 0.15, e) for e in ETATS}
    res_jacobs[lab] = (b, a, u, xi, sg, v)
    print(f"  {lab:<20}{b:>8.4f}{a:>8.3f}{u:>9.3f}{xi:>8.3f}{sg:>8.3f}"
          f"{v['C']:>9.0f}{v['NC']:>10.0f}{v['NC'] - v['C']:>9.0f}")
d_lo = res_jacobs["bas (-1,645 SE)"][5]
d_hi = res_jacobs["haut (+1,645 SE)"][5]
d_c = res_jacobs["central (2014)"][5]
print("\n  Lecture : b est l'ELASTICITE cout-taille. Pivote au centroide, un b plus haut")
print("  appauvrit les petites breches et enrichit les grandes : la queue derivee")
print("  s'epaissit, le SCR et le Delta PRC montent. C'est l'axe d'hypothese dominant")
print("  du perimetre PRC (le residu par enregistrement, lui, se moyenne : script 21).")

# ============================================================ B. gamma
titre("B. Charge systemique gamma : probabilites de crise et SCR espere")
print("  gamma n'affecte PAS le SCR par etat (conditionnel) ; il commande la bascule")
print("  des probabilites d'etat en crise, donc le SCR espere conditionnel a Theta.")
# SCR par etat OpRisk, lecture B, protocole 16 (recalcule ici, CRN)
sp_o = PARAMS["OPRISK"]
scr_o = {}
for e in ETATS:
    lam = ec.lambda_scenario("OPRISK", SCENARIO[e], mode="center")
    rng = np.random.default_rng(SEED)
    scr_o[e] = var(ec.simulate_euro(lam, G_PROP[e], sp_o["xi"], sp_o["sigma"],
                                    sp_o["u"], sp_o["p_u"], sp_o["cap"], NY, rng))
print(f"  SCR par etat (OpRisk, lecture B, fixes) : C {scr_o['C']:.0f}  "
      f"PC {scr_o['PC']:.0f}  NC {scr_o['NC']:.0f} M EUR")
print(f"\n  {'gamma':<10}{'P(NC|crise)':>13}{'P(C|crise)':>12}{'SCR espere normal':>19}"
      f"{'SCR espere crise':>18}")
res_gamma = {}
for g in (0.50, GAMMA_BASE, 0.85):
    pc = etat_probs(THETA_CRISE, g, P_ETAT)
    esp_n = sum(P_ETAT[e] * scr_o[e] for e in ETATS)     # marginale = ancrage
    esp_c = sum(pc[e] * scr_o[e] for e in ETATS)
    res_gamma[g] = (pc, esp_n, esp_c)
    print(f"  {g:<10.2f}{pc['NC']:>12.1%}{pc['C']:>12.1%}{esp_n:>18.0f} M{esp_c:>17.0f} M")
print("  Lecture : le SCR espere NORMAL est invariant (la marginale redonne l'ancrage")
print("  par construction) ; gamma ne joue que sur la severite de la bascule en crise.")

# ============================================================ C. ancrage des probas d'etat
titre("C. Ancrage des probabilites d'etat : melange, jamais SCR par etat")
ANCRAGES = {"optimiste (25/40/35)": {"NC": 0.25, "PC": 0.40, "C": 0.35},
            "central  (35/35/30)": dict(P_ETAT),
            "pessimiste (45/35/20)": {"NC": 0.45, "PC": 0.35, "C": 0.20}}
print("  Le SCR par etat et le Delta NC vs C ne dependent pas de l'ancrage (ils sont")
print("  conditionnels a l'etat). L'ancrage deplace le MELANGE : SCR espere et point")
print("  de depart de la trajectoire du 17.")
print(f"\n  {'ancrage':<24}{'SCR espere normal':>19}{'SCR espere crise':>18}"
      f"{'surcout vs conforme':>21}")
res_anc = {}
for lab, pe in ANCRAGES.items():
    pc = etat_probs(THETA_CRISE, GAMMA_BASE, pe)
    esp_n = sum(pe[e] * scr_o[e] for e in ETATS)
    esp_c = sum(pc[e] * scr_o[e] for e in ETATS)
    res_anc[lab] = (esp_n, esp_c)
    print(f"  {lab:<24}{esp_n:>18.0f} M{esp_c:>17.0f} M{esp_n - scr_o['C']:>20.0f} M")
print("  Lecture : l'ancrage (enquetes, non calibre sur l'entite) borne le capital")
print("  ESPERE actuel ; la cible conforme et le surcout par etat restent inchanges.")

# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2 = "#1b1e30", "#223e55"
ACCENT, BLUE, GREEN, GREY = "#a6002e", "#3661ac", "#008d87", "#7d7d7d"

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Trois panneaux traces
# sur seize pouces puis imprimes sur 7,27 donnent 1,6 pouce par panneau et
# des etiquettes sous 5 points. Le rapport largeur sur hauteur est conserve.
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(9.69, 3.11))

# panneau A : Delta PRC selon b (pivote)
labs = list(axes_b.keys())
vals = [res_jacobs[k][5]["NC"] - res_jacobs[k][5]["C"] for k in labs]
cols = [BLUE, GREY, ACCENT]
axA.bar(range(3), vals, color=cols, alpha=0.9)
for i, v in enumerate(vals):
    axA.annotate(f"{v:,.0f}".replace(",", " "), (i, v), textcoords="offset points",
                 xytext=(0, 4), ha="center", fontsize=8.5, color=INK2)
axA.set_ylim(0, max(vals) * 1.16)
axA.set_xticks(range(3))
# Le prefixe << b = >> repete sur chaque barre faisait se chevaucher les trois etiquettes : la
# variable passe dans le libelle d'axe, les graduations ne portent plus que la valeur.
axA.set_xticklabels([f"{axes_b[k]:.3f}".replace(".", ",") for k in labs], fontsize=8.5)
axA.set_xlabel("coefficient $b$ (borne basse, centrale, haute)", fontsize=8.8, color=INK2)
axA.set_ylabel(r"$\Delta_{\mathrm{DORA}}$ NC contre C (M€, PRC)", fontsize=9.3, color=INK2)
axA.set_title("(A)  Coefficient $b$ de Jacobs (± 1,645 écart-type,\npivoté au centroïde)",
              fontsize=9.6, color=INK, pad=6)
axA.grid(alpha=0.25, lw=0.5, axis="y")

# panneau B : bascule de crise selon gamma
gammas = [0.50, GAMMA_BASE, 0.85]
pnc = [res_gamma[g][0]["NC"] for g in gammas]
esps = [res_gamma[g][2] for g in gammas]
axB2 = axB.twinx()
axB.bar([i - 0.19 for i in range(3)], pnc, 0.38, color=BLUE, alpha=0.85,
        label="P(NC | crise)")
axB2.bar([i + 0.19 for i in range(3)], esps, 0.38, color=ACCENT, alpha=0.85,
         label="SCR espéré en crise")
axB.set_xticks(range(3))
axB.set_xticklabels([f"{g:.2f}".replace(".", ",") for g in gammas], fontsize=8.5)
axB.set_xlabel(r"charge systémique $\gamma$", fontsize=8.8, color=INK2)
axB.set_ylabel("P(NC | crise)", fontsize=9.3, color=BLUE)
axB2.set_ylabel("SCR espéré en crise (M€)", fontsize=9.3, color=ACCENT)
axB.set_ylim(0, 1.15)
axB.set_title("(B)  $\\gamma$ : la sévérité de la bascule\n(SCR espéré en régime normal invariant)",
              fontsize=9.6, color=INK, pad=6)
axB.grid(alpha=0.25, lw=0.5, axis="y")

# panneau C : ancrage -> SCR espere normal
labs_c = list(ANCRAGES.keys())
vn = [res_anc[k][0] for k in labs_c]
axC.bar(range(3), vn, color=[GREEN, GREY, ACCENT], alpha=0.9)
axC.axhline(scr_o["C"], color=BLUE, lw=1.4, ls="--", label="SCR conforme (cible)")
for i, v in enumerate(vn):
    axC.annotate(f"{v:,.0f}".replace(",", " "), (i, v), textcoords="offset points",
                 xytext=(0, 4), ha="center", fontsize=8.5, color=INK2)
axC.set_ylim(0, max(vn) * 1.32)
axC.set_xticks(range(3))
axC.set_xticklabels(["25/40/35", "35/35/30", "45/35/20"], fontsize=8.5)
axC.set_xlabel("parts NC / PC / C (%)", fontsize=8.8, color=INK2)
axC.set_ylabel("SCR espéré, régime normal (M€, OpRisk)", fontsize=9.3, color=INK2)
axC.set_title("(C)  Ancrage NC/PC/C : le mélange bouge,\nle SCR par état jamais",
              fontsize=9.6, color=INK, pad=6)
axC.legend(fontsize=8.0, frameon=False, loc="upper left")
axC.grid(alpha=0.25, lw=0.5, axis="y")

fig.tight_layout(w_pad=1.8)

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S17_sensibilites_hypotheses.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
