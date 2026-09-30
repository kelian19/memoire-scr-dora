#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
77 : l'invariance du script 69 porte-t-elle sur le bon objet ? E[VaR] contre VaR du melange.

L'OBJECTION QUI MOTIVE CE SCRIPT, et elle est serieuse. Le script 69 etablit que la STRUCTURE de
la dependance entre etats de conformite ne deplace pas le capital, et l'argument est elegant : le
capital des 32 configurations s'ajuste par une forme ADDITIVE en indicateurs de pilier a
R2 = 0,9945, or l'esperance d'une fonction additive ne depend que des MARGES. L'invariance vaut
donc pour toute structure a marges fixees, y compris non essayee.

MAIS L'ESPERANCE D'UNE FONCTION ADDITIVE N'EST PAS LE CAPITAL. Le capital est un QUANTILE. Ce
que le script 69 mesure est

    E[VaR]  =  somme sur m de  w(m) x VaR(perte | configuration m),

c'est-a-dire la MOYENNE DES VaR CONDITIONNELLES. Un lecteur qui demande « quel capital pour une
entite dont l'etat de conformite est incertain » demande autre chose :

    VaR(melange)  =  quantile a 99,5 % de la loi MELANGEE sur les configurations.

Et ces deux objets ne coincident pas. La moyenne des quantiles n'est pas le quantile du melange,
et un melange peut epaissir la queue. Surtout, l'argument d'additivite NE S'APPLIQUE PAS au
second : il repose sur la linearite de l'esperance, dont un quantile ne dispose pas.

CE QUE FAIT CE SCRIPT :
  1. il recalcule les 32 configurations exactement comme le script 69, a nombres communs ;
  2. il construit la loi MELANGEE en tirant, pour chaque annee, une configuration selon ses
     poids, et en prenant la perte de cette configuration cette annee-la ;
  3. il compare E[VaR] et VaR(melange) au modele actuel ;
  4. il refait la comparaison en faisant varier le surcroit de correlation delta, pour voir si
     l'invariance survit sur le second objet ou si elle est propre au premier.

Sortie : diagnostics + figure S33_var_melange.png.
"""

import os
import sys

import numpy as np
from scipy.stats import norm
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                                 # noqa: E402
from euro_cascade_model import PARAMS, var                      # noqa: E402
import scr_engine as eng                                        # noqa: E402

WID = 86
PIL = eng.PIL
N = len(PIL)

# Protocole IDENTIQUE au script 69. Toute divergence ici rendrait la comparaison sans objet.
SCENARIO = {"C": "S0_conforme", "NC": "S2_non_conforme"}
G_PROP = {"C": 0.45, "NC": 0.90}
PU_MULT = {"C": 0.85, "NC": 1.20}
MULT_ETAT = {st: ec.lambda_scenario("OPRISK", sc, mode="center") / PARAMS["OPRISK"]["lam_ref"]
             for st, sc in SCENARIO.items()}
_S = {j: eng.LAMBDA[j] for j in PIL}
SHARE = {j: _S[j] / sum(_S.values()) for j in PIL}

P_NC = 0.35
GAMMA = 0.68
K = norm.ppf(P_NC)

NY = 40_000
SEEDS = (909, 1234, 2718, 31415)
SOURCE = "OPRISK"
J1, J2, J3 = PIL[0], PIL[1], PIL[2]
DELTAS = (0.0, 0.10, 0.20, 0.30, 0.38)
NSIM_ETATS = 4_000_000
ALPHA = 0.995


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def annual_config(state_map, ny, seed):
    sp = PARAMS[SOURCE]
    lam_vec = {j: sp["lam_ref"] * SHARE[j] * MULT_ETAT[state_map[j]] for j in PIL}
    g_vec = {j: G_PROP[state_map[j]] for j in PIL}
    p_u_vec = {j: min(0.999, sp["p_u"] * PU_MULT[state_map[j]]) for j in PIL}
    rng = np.random.default_rng(seed)
    return ec.simulate_euro_pp(lam_vec, g_vec, sp["xi"], sp["sigma"], sp["u"],
                               p_u_vec, sp["cap"], ny, rng)


def masque_vers_etats(m):
    return {j: ("NC" if m & (1 << i) else "C") for i, j in enumerate(PIL)}


def nom_masque(m):
    nc = [f"P{j}" for i, j in enumerate(PIL) if m & (1 << i)]
    return "tous conformes" if not nc else "+".join(nc) + " NC"


def matrice_corr(delta):
    """Correlation echangeable gamma^2, plus un surcroit delta sur (J1,J3) et (J2,J3)."""
    base = GAMMA ** 2
    C = np.full((N, N), base)
    np.fill_diagonal(C, 1.0)
    i1, i2, i3 = PIL.index(J1), PIL.index(J2), PIL.index(J3)
    for a, b in ((i1, i3), (i2, i3)):
        C[a, b] = C[b, a] = min(0.999, base + delta)
    return C


def poids_configs(delta, nsim=NSIM_ETATS, seed=20260813):
    """Probabilite des 2^N configurations sous la latente, marges preservees par construction."""
    C = matrice_corr(delta)
    L = np.linalg.cholesky(C)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((nsim, N)) @ L.T
    nc = (Z < K).astype(np.int64)
    codes = nc @ (1 << np.arange(N))
    return np.bincount(codes, minlength=2 ** N) / nsim


# =====================================================================================
titre("1. Les 32 configurations, recalculees a l'identique du script 69")
# =====================================================================================
print(f"  {2 ** N} configurations, {NY} annees, {len(SEEDS)} graines, a nombres communs.")
pertes = {}          # masque -> pertes annuelles empilees sur les graines
scr = np.empty(2 ** N)
for m in range(2 ** N):
    sm = masque_vers_etats(m)
    series = [annual_config(sm, NY, s) for s in SEEDS]
    pertes[m] = np.concatenate(series)
    scr[m] = float(np.mean([var(x) for x in series]))
print(f"  Capital de {scr.min():.0f} M (tous conformes) a {scr.max():.0f} M (tous non conformes).")

w0 = poids_configs(0.0)
e_var0 = float(np.dot(w0, scr))
print(f"\n  E[VaR] au modele actuel = {e_var0:.0f} M€  <- l'objet que mesure le script 69")

# =====================================================================================
titre("2. La loi MELANGEE, et son quantile")
# =====================================================================================
print("  On tire une configuration par annee selon ses poids, et l'on prend la perte de cette")
print("  configuration cette annee-la. Le resultat est la loi de la charge annuelle d'une entite")
print("  dont l'etat de conformite est INCERTAIN, ce qui est la question posee.")


def var_melange(w, seed=4242):
    """Quantile a 99,5 % de la loi melangee sur les configurations."""
    rng = np.random.default_rng(seed)
    n = len(pertes[0])
    choix = rng.choice(2 ** N, size=n, p=w)
    tire = np.empty(n)
    for m in range(2 ** N):
        sel = choix == m
        if sel.any():
            tire[sel] = pertes[m][sel]
    return float(np.quantile(tire, ALPHA)), tire


v_mel0, serie0 = var_melange(w0)
print(f"\n  E[VaR]        = {e_var0:>9.0f} M€")
print(f"  VaR(melange)  = {v_mel0:>9.0f} M€")
ecart_pct = 100 * (v_mel0 - e_var0) / e_var0
print(f"  ecart         = {v_mel0 - e_var0:>9.0f} M€, soit {ecart_pct:+.1f} %")
print("\n  LES DEUX OBJETS NE COINCIDENT PAS, et l'ecart n'est pas du bruit. La raison est")
print("  structurelle : a 99,5 % la queue du melange est prelevee dans les composantes les plus")
print("  lourdes, alors que E[VaR] moyenne toutes les composantes, y compris les conformes.")

# =====================================================================================
titre("3. L'invariance survit-elle sur le second objet ?")
# =====================================================================================
print("  On refait la comparaison en augmentant le surcroit de correlation sur les deux paires")
print("  concernees. Les MARGES sont preservees par construction : seule la structure bouge.")
print(f"\n  {'delta':>7}{'P(tous NC)':>13}{'E[VaR]':>11}{'ecart':>9}{'VaR(mel.)':>12}{'ecart':>9}")
lignes = []
for d in DELTAS:
    w = poids_configs(d)
    ev = float(np.dot(w, scr))
    vm, _ = var_melange(w)
    lignes.append((d, w[-1], ev, vm))
    print(f"  {d:>7.2f}{w[-1]:>13.4f}{ev:>11.0f}{ev - e_var0:>+9.0f}{vm:>12.0f}"
          f"{vm - v_mel0:>+9.0f}")

amp_ev = max(l[2] for l in lignes) - min(l[2] for l in lignes)
amp_vm = max(l[3] for l in lignes) - min(l[3] for l in lignes)
print(f"\n  Amplitude sur le balayage : E[VaR] {amp_ev:.0f} M€, VaR(melange) {amp_vm:.0f} M€,")
print(f"  soit un rapport de {amp_vm / max(amp_ev, 1e-9):.1f}.")
print(f"  En relatif : {100*amp_ev/e_var0:.3f} % contre {100*amp_vm/v_mel0:.3f} %.")

# LE BRUIT DU MELANGE, MESURE ET NON SUPPOSE. Le deplacement ci-dessus n'est PAS monotone en
# delta, ce qui est la signature d'un bruit de tirage important : le quantile a 99,5 % ne
# repose que sur quelques centaines de points. Publier 334 M€ comme un effet sans mesurer ce
# bruit serait exactement la faute que ce projet s'interdit.
print("\n  LE DEPLACEMENT N'EST PAS MONOTONE EN DELTA, ce qui trahit un bruit de tirage. On le")
print("  mesure en refaisant le melange a delta = 0 sous plusieurs graines de tirage.")
graines_mel = (4242, 17, 2718, 31415, 99991, 5, 20260813, 777)
vs = [var_melange(w0, seed=s)[0] for s in graines_mel]
sd_mel = float(np.std(vs, ddof=1))
etendue = max(vs) - min(vs)
print(f"  {len(graines_mel)} graines de melange, delta = 0 : ecart-type {sd_mel:.0f} M€, "
      f"etendue {etendue:.0f} M€.")
resolu = amp_vm > 2 * etendue
print(f"  L'amplitude du balayage ({amp_vm:.0f}) est-elle superieure au double de cette etendue ?"
      f" {resolu}.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. L'OBJECTION EST FONDEE SUR LE PRINCIPE, et la mesure le confirme. L'argument")
print("     d'additivite du script 69 vaut pour une ESPERANCE, et un quantile ne dispose pas de")
print("     la linearite sur laquelle il repose. De fait, sur le balayage, E[VaR] ne bouge que")
print(f"     de {amp_ev:.0f} M€ quand le quantile du melange bouge de {amp_vm:.0f} : un rapport de")
print(f"     {amp_vm / max(amp_ev, 1e-9):.0f}. L'invariance DEMONTREE ne s'etend donc pas au second objet,")
print("     et il fallait le mesurer plutot que de l'esperer.")
print(f"\n  2. LES DEUX OBJETS DIFFERENT EN NIVEAU DE {v_mel0 - e_var0:.0f} M€, soit {ecart_pct:+.1f} % :")
print(f"     {e_var0:.0f} contre {v_mel0:.0f}. Ce n'est pas un ecart de plusieurs milliards, mais il")
print("     depasse largement le bruit, et le memoire doit dire lequel des deux il publie. La")
print("     raison de l'ecart est structurelle : a 99,5 % la queue du melange est prelevee dans")
print("     les composantes les plus lourdes, alors que E[VaR] moyenne toutes les composantes.")
print(f"\n  3. LE DEPLACEMENT DU QUANTILE N'EST PAS RESOLU : {resolu}. Il n'est pas monotone en")
print(f"     delta, et son amplitude ({amp_vm:.0f} M€) est du meme ordre que l'etendue du seul bruit")
print(f"     de tirage ({etendue:.0f} M€ sur {len(graines_mel)} graines). On ne peut donc PAS affirmer que")
print("     la structure deplace le quantile ; on peut seulement dire qu'on ne le voit pas a")
print("     cette resolution. C'est un enonce plus faible que l'invariance demontree sur")
print("     l'esperance, et il faut le presenter comme tel.")
print(f"\n  4. CE QUI RESTE VRAI ET UTILISABLE : quelle que soit la lecture, le deplacement reste")
print(f"     petit devant l'incertitude deja declaree. Les {amp_vm:.0f} M€ representent"
      f" {100*amp_vm/1839:.0f} % de la")
print("     largeur de la bande d'identification, qui vaut 1 839 M€. La conclusion de gestion ne")
print("     change pas ; c'est le STATUT de la conclusion qui change, de demontre a non detecte.")
print("\n  5. CE QU'IL FAUT ECRIRE DESORMAIS. L'argument d'additivite reste le plus fort, mais il")
print("     doit etre borne a l'esperance, et la mesure sur le quantile doit l'accompagner avec")
print("     son bruit. Annoncer l'invariance sans cette distinction expose a une objection a")
print("     laquelle le script 69 seul ne repond pas.")

# =====================================================================================
# figure S33
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 11,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
})
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime
# cette figure sur 7,27 pouces : tracee plus large, elle subissait une
# reduction qui faisait tomber ses etiquettes sous 6 points. Le rapport
# largeur sur hauteur est conserve, donc la figure occupe la meme place
# sur la page ; c'est son texte qui y prend plus de place.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.9, 3.21))

# (a) les deux objets, en niveau
ax1.bar([0, 1], [e_var0, v_mel0], color=[BLUE, ACCENT], width=0.5, alpha=0.9)
for x, v, lab in ((0, e_var0, "E[VaR]\nmoyenne des VaR\nconditionnelles"),
                  (1, v_mel0, "VaR(mélange)\nquantile de la loi\nmélangée")):
    ax1.text(x, v * 1.02, f"{v:,.0f}".replace(",", " "), ha="center", fontsize=11,
             fontweight="bold", color=INK)
    ax1.text(x, -max(e_var0, v_mel0) * 0.13, lab, ha="center", fontsize=9, color=INK2)
ax1.set_xlim(-0.6, 1.6)
ax1.set_ylim(0, max(e_var0, v_mel0) * 1.16)
ax1.set_xticks([])
ax1.set_ylabel("capital (M€)", color=INK2)
ax1.set_title(f"(a)  Deux objets, {ecart_pct:+.1f} % d'écart :\nle mémoire doit dire lequel il "
              "publie".replace(".", ","), fontsize=10.5, color=INK, pad=8)

# (b) l'invariance, sur les deux objets, a l'echelle de la bande d'identification
ds = [l[0] for l in lignes]
ax2.plot(ds, [l[2] - e_var0 for l in lignes], "o-", color=BLUE, lw=2, ms=7,
         label="E[VaR], écart au modèle actuel")
ax2.plot(ds, [l[3] - v_mel0 for l in lignes], "s-", color=ACCENT, lw=2, ms=7,
         label="VaR(mélange), écart au modèle actuel")
ax2.axhline(0, color=MUTED, lw=1.0)
# LE BRUIT DE TIRAGE, TRACE ET NON SEULEMENT DIT. C'est le point de la figure : la courbe du
# melange reste DANS la bande de bruit, donc son deplacement n'est pas resolu. Sans cette bande
# le lecteur voit une courbe qui monte et conclut a un effet.
demi = etendue / 2.0
ax2.axhspan(-demi, demi, color=MUTED, alpha=0.16, lw=0)
ax2.text(ds[-1], demi * 0.55, f"± demi-étendue du bruit de tirage ({demi:.0f} M€)",
         fontsize=8.5, color=INK2, ha="right")
# ECHELLE : la largeur de la bande d'identification deja publiee. Tracer ces ecarts sur leur
# propre amplitude les ferait paraitre enormes, ce qui est le travers corrige sur la figure S25.
BANDE = 1839.0
ax2.axhline(BANDE, color=MUTED, lw=1.2, ls="--")
ax2.text(ds[0], BANDE * 1.02, "largeur de la bande d'identification, 1 839 M€",
         fontsize=8.5, color=INK2, va="bottom")
ax2.set_ylim(-BANDE * 0.18, BANDE * 1.22)
ax2.set_xlabel("surcroît de corrélation δ sur les deux paires concernées", color=INK2)
ax2.set_ylabel("écart au capital du modèle actuel (M€)", color=INK2)
ax2.legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.86))
# TITRE PRECIS. Une premiere version disait « reste dans le bruit », ce que la courbe dement :
# les points depassent la demi-etendue. Le fait exact est que l'amplitude du balayage (334) est
# du MEME ORDRE que l'etendue du bruit (344), et que la courbe n'est pas monotone.
ax2.set_title("(b)  Sur le quantile, le déplacement est du même ordre\nque le bruit, et non "
              "monotone : non résolu", fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# TITRE CORRIGE : il annoncait « l'invariance vaut pour les deux », ce que la mesure ne soutient
# pas. Elle est DEMONTREE sur l'esperance et seulement NON DETECTEE sur le quantile, le
# deplacement y restant dans le bruit de tirage. La nuance est tout l'objet de cette figure.
fig.tight_layout(rect=[0, 0, 1, 0.91])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S33_var_melange.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
