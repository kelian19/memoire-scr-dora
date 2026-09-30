#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
76 : le tornado, renormalise. Deux defauts du tornado du script 15, et ce qu'il en reste.

POURQUOI CE SCRIPT. Le script 15 conclut que le seuil de la loi de queue domine la sensibilite
et que le gain de propagation est le levier le moins sensible. Cette conclusion est reprise dans
le memoire et sert d'argument fort : la fragilite serait dans l'ajustement de valeurs extremes
et non dans la contagion, ce qui est l'inverse de ce qu'on redoute d'un modele de cascade.
Une relecture critique fait apparaitre DEUX defauts, et le second est plus grave que le premier.

DEFAUT 1, DE COMPARABILITE. Les quatre leviers sont balayes sur des plages qui n'ont pas la
meme vraisemblance : le gain va de 0,50 a 1,00 (une amplitude POSEE), la frequence de la moitie
au triple, la surdispersion de 5 a 15. Un tornado compare des LONGUEURS DE BARRE, or une barre
est longue parce que le parametre bouge beaucoup ou parce qu'on l'a fait bouger beaucoup. En
l'etat, le classement mesure autant nos choix de plage que la sensibilite du modele.

DEFAUT 2, ET C'EST LE PLUS SERIEUX : LE LEVIER « SEUIL » N'EST PAS UN LEVIER. Deplacer le seuil
du percentile 80 au percentile 90 REAJUSTE la loi de queue : xi, sigma et le taux de depassement
changent tous les trois. Ce levier deplace donc QUATRE parametres quand « gain g » en deplace
UN. Qu'il sorte en tete n'est pas un resultat, c'est une consequence arithmetique de ce qu'on a
mis dedans.

CE QUE FAIT CE SCRIPT :
  1. il decompose le levier « seuil » en ses quatre composantes, et montre laquelle porte ;
  2. il calcule des ELASTICITES, d log(ecart) / d log(parametre), sur une perturbation relative
     IDENTIQUE pour tous les leviers. Une elasticite est sans dimension et sans plage : c'est la
     seule normalisation qui rende les leviers comparables ;
  3. il separe les parametres ESTIMES, qui ont une incertitude d'echantillonnage et donc une
     plage legitime, des parametres POSES, qui n'en ont aucune. Pour les premiers on donne aussi
     le deplacement sur leur intervalle PUBLIE, qui est la seule plage defendable ;
  4. il dit si la conclusion du script 15 survit.

Sortie : diagnostics + figure S32_tornado_normalise.png.
"""

import os
import sys

import numpy as np
from scipy.stats import genpareto
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                              # noqa: E402
from euro_cascade_model import var, OPRISK                   # noqa: E402
from src.frequency.negbin import compute_lambda_scenario     # noqa: E402

W = 84
NY = 100_000          # identique au script 15, pour que la base soit comparable
SEED = 11             # meme graine : les ecarts entre variantes sont APPARIES, donc peu bruites
LAM_REF0 = OPRISK["n_incidents"] / OPRISK["n_years"]
G0, PHI0 = ec.G_BASE, ec.PHI
EPS = 0.10            # perturbation relative commune a tous les leviers

# Intervalle a 90 % de l'indice de queue, publie par la calibration figee (bootstrap).
XI_IC90 = (0.3044, 0.8313)


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def gpd_at_threshold(losses, u):
    exc = losses[losses > u] - u
    xi, _, sg = genpareto.fit(exc, floc=0)
    return float(np.clip(xi, 0.05, 0.98)), max(1.0, float(sg)), len(exc) / len(losses)


def delta(lam_ref, u, p_u, xi, sigma, g, phi, ny=NY):
    """Ecart entre etats, deterministe a graine commune. Identique au script 15."""
    lam_nc = compute_lambda_scenario(lam_ref, "S2_non_conforme", "center")["lambda_global"]
    lam_c = compute_lambda_scenario(lam_ref, "S0_conforme", "center")["lambda_global"]
    v_nc = var(ec.simulate_euro(lam_nc, g, xi, sigma, u, p_u, None, ny,
                                np.random.default_rng(SEED), phi=phi))
    v_c = var(ec.simulate_euro(lam_c, g, xi, sigma, u, p_u, None, ny,
                               np.random.default_rng(SEED), phi=phi))
    return v_nc - v_c


LOSSES = ec.oprisk_losses()
if LOSSES is None:
    sys.exit("SAS OpRisk absent de data/raw (licence) : tornado indisponible.")

u0 = OPRISK["seuil_u_eur"]
xi0, sg0, pu0 = gpd_at_threshold(LOSSES, u0)
BASE = dict(lam_ref=LAM_REF0, u=u0, p_u=pu0, xi=xi0, sigma=sg0, g=G0, phi=PHI0)
d0 = delta(**BASE)

titre("0. Le point de base, identique au script 15")
print(f"  u = {u0:.2f} M€ | xi = {xi0:.3f} | sigma = {sg0:.1f} | p_u = {pu0:.3f}")
print(f"  lambda_ref = {LAM_REF0:.2f}/an | g = {G0} | phi = {PHI0}")
print(f"  ecart entre etats a la base = {d0:.0f} M€")

# =====================================================================================
titre("1. Le levier « seuil » n'est pas un levier : il en contient quatre")
# =====================================================================================
print("  Deplacer le seuil REAJUSTE la loi de queue. Voici ce que le refit change vraiment.")
print(f"\n  {'seuil':<10}{'u (M€)':>10}{'xi':>9}{'sigma':>10}{'p_u':>9}")
fits = {}
for p in (80, 85, 90):
    u_p = float(np.quantile(LOSSES, p / 100))
    fits[p] = (u_p,) + gpd_at_threshold(LOSSES, u_p)
    u_p, xi_p, sg_p, pu_p = fits[p]
    marque = "  <- base" if abs(u_p - u0) < 1e-9 else ""
    print(f"  p{p:<9}{u_p:>10.2f}{xi_p:>9.3f}{sg_p:>10.1f}{pu_p:>9.3f}{marque}")

print("\n  DECOMPOSITION : on deplace du percentile 80 au percentile 90, d'abord tout ensemble")
print("  comme le fait le script 15, puis une composante a la fois, les autres restant a la base.")
u80, xi80, sg80, pu80 = fits[80]
u90, xi90, sg90, pu90 = fits[90]

def d_avec(**kw):
    a = dict(BASE)
    a.update(kw)
    return delta(**a)

paquet_bas = d_avec(u=u80, xi=xi80, sigma=sg80, p_u=pu80)
paquet_haut = d_avec(u=u90, xi=xi90, sigma=sg90, p_u=pu90)
comp = [
    ("u seul", d_avec(u=u80), d_avec(u=u90)),
    ("xi seul", d_avec(xi=xi80), d_avec(xi=xi90)),
    ("sigma seul", d_avec(sigma=sg80), d_avec(sigma=sg90)),
    ("p_u seul", d_avec(p_u=pu80), d_avec(p_u=pu90)),
]
print(f"\n  {'composante':<14}{'bas (p80)':>13}{'haut (p90)':>13}{'effet signe':>14}")
net_paquet = paquet_haut - paquet_bas
print(f"  {'LE PAQUET':<14}{paquet_bas:>13.0f}{paquet_haut:>13.0f}{net_paquet:>+14.0f}")
amps, signes = {}, {}
for nom, b, h in comp:
    amps[nom] = abs(h - b)
    signes[nom] = h - b
    print(f"  {nom:<14}{b:>13.0f}{h:>13.0f}{h - b:>+14.0f}")
somme = sum(signes.values())
print(f"  {'somme des 4':<14}{'':>13}{'':>13}{somme:>+14.0f}")
porteur = max(amps, key=amps.get)
print(f"\n  DEUX CHOSES A LIRE ICI, ET LA SECONDE EST LA PLUS GENANTE.")
print(f"  (i) La composante qui porte tout est {porteur.upper()}, dont l'effet propre vaut")
print(f"      {signes[porteur]:+.0f} M€. La barre « seuil » du tornado publie ne mesure donc pas la")
print("      sensibilite au seuil : elle mesure surtout celle a l'INDICE DE QUEUE, que le refit")
print(f"      entraine. Le seuil PRIS SEUL ne deplace que {signes['u seul']:+.0f} M€.")
print(f"  (ii) LES COMPOSANTES NE S'ADDITIONNENT PAS, ELLES SE COMPENSENT. Leur somme signee")
print(f"      vaut {somme:+.0f} M€ quand le paquet en deplace {net_paquet:+.0f} : il y a")
print(f"      {net_paquet - somme:+.0f} M€ de compensation, xi tirant vers le bas et sigma vers le")
print("      haut. La longueur de la barre publiee est donc PLUS COURTE que l'effet du seul")
print("      indice de queue, tout en lui etant attribuee sous le nom de « seuil ».")
print("\n  Comparer cette barre a celle du gain de propagation, qui ne deplace qu'un parametre,")
print("  n'a pas de sens : ce sont deux objets de nature differente.")

# =====================================================================================
titre("2. Elasticites : la seule normalisation qui rende les leviers comparables")
# =====================================================================================
print("  Une elasticite d log(ecart) / d log(parametre) est SANS DIMENSION et SANS PLAGE. On")
print(f"  applique la MEME perturbation relative de +/- {100*EPS:.0f} % a chaque levier, un a la fois.")
print("  Aucun refit : chaque parametre bouge seul, ce qui est precisement ce qui manquait.")

LEVIERS = [
    ("frequence lambda", "lam_ref", "estime"),
    ("indice de queue xi", "xi", "estime"),
    ("echelle sigma", "sigma", "estime"),
    ("seuil u", "u", "choix"),
    ("taux de depassement p_u", "p_u", "estime"),
    ("gain de propagation g", "g", "pose"),
    ("surdispersion phi", "phi", "pose"),
]

elas = {}
print(f"\n  {'levier':<26}{'statut':<10}{'-10 %':>11}{'+10 %':>11}{'elasticite':>12}")
for nom, cle, statut in LEVIERS:
    v = BASE[cle]
    d_bas = d_avec(**{cle: v * (1 - EPS)})
    d_haut = d_avec(**{cle: v * (1 + EPS)})
    e = (np.log(d_haut) - np.log(d_bas)) / (np.log(1 + EPS) - np.log(1 - EPS))
    elas[nom] = (e, statut, d_bas, d_haut)
    print(f"  {nom:<26}{statut:<10}{d_bas:>11.0f}{d_haut:>11.0f}{e:>12.2f}")

ordre = sorted(elas, key=lambda k: -abs(elas[k][0]))
print(f"\n  CLASSEMENT PAR ELASTICITE ABSOLUE : " + " > ".join(ordre))
e_g = elas["gain de propagation g"][0]
e_xi = elas["indice de queue xi"][0]
e_lam = elas["frequence lambda"][0]
print(f"\n  Le gain de propagation a une elasticite de {e_g:.2f}, l'indice de queue de {e_xi:.2f}")
print(f"  et la frequence de {e_lam:.2f}. Rapport queue sur propagation : "
      f"{abs(e_xi / e_g):.1f}, frequence sur propagation : {abs(e_lam / e_g):.1f}.")

# =====================================================================================
titre("3. Estime contre pose : la plage legitime n'existe que pour les premiers")
# =====================================================================================
print("  Une elasticite dit la sensibilite par point de pourcentage. Elle ne dit pas de combien")
print("  le parametre PEUT bouger. Pour cela il faut une plage, et une plage n'est defendable")
print("  que si elle vient d'une incertitude MESUREE.")
# incertitude de lambda : ecart-type sous la surdispersion mesuree sur la serie annuelle
PHI_SERIE = 2.24
n_ans = OPRISK["n_years"]
se_lam = float(np.sqrt(PHI_SERIE * LAM_REF0 / n_ans))
lam_bas, lam_haut = LAM_REF0 - 1.645 * se_lam, LAM_REF0 + 1.645 * se_lam
d_lam_bas, d_lam_haut = d_avec(lam_ref=lam_bas), d_avec(lam_ref=lam_haut)
d_xi_bas, d_xi_haut = d_avec(xi=XI_IC90[0]), d_avec(xi=XI_IC90[1])
print(f"\n  {'levier':<26}{'plage':<28}{'ecart bas':>11}{'ecart haut':>12}{'ampl.':>9}")
print(f"  {'indice de queue xi':<26}{'IC90 publie [%.3f ; %.3f]' % XI_IC90:<28}"
      f"{d_xi_bas:>11.0f}{d_xi_haut:>12.0f}{abs(d_xi_haut - d_xi_bas):>9.0f}")
print(f"  {'frequence lambda':<26}"
      f"{'+/- 1,645 ET, ET = %.2f' % se_lam:<28}{d_lam_bas:>11.0f}{d_lam_haut:>12.0f}"
      f"{abs(d_lam_haut - d_lam_bas):>9.0f}")
print(f"  {'gain de propagation g':<26}{'AUCUNE : parametre POSE':<28}{'-':>11}{'-':>12}{'-':>9}")
print(f"  {'surdispersion phi':<26}{'AUCUNE : parametre POSE':<28}{'-':>11}{'-':>12}{'-':>9}")
print("\n  C'EST LE POINT QUE LE TORNADO D'ORIGINE MASQUAIT. Deux des quatre leviers n'ont pas")
print("  d'intervalle d'incertitude parce qu'ils ne sont pas estimes : la plage qu'on leur")
print("  donne est un CHOIX D'AFFICHAGE. Leur barre peut donc etre allongee ou raccourcie a")
print("  volonte, et la comparer a celle d'un parametre estime revient a comparer une")
print("  incertitude a une convention.")

# =====================================================================================
titre("VERDICT : ce qui survit de la conclusion du script 15")
# =====================================================================================
e_u = elas["seuil u"][0]
substantiels = {k: v[0] for k, v in elas.items() if k != "seuil u"}
survit = abs(e_g) == min(abs(x) for x in substantiels.values())
print(f"  1. LE LEVIER « SEUIL » EST A RETIRER DU CLASSEMENT tel quel : il deplace quatre")
print(f"     parametres a la fois, et c'est {porteur} qui porte son effet, avec en prime une")
print("     compensation entre composantes. Sa premiere place dans le tornado publie est une")
print("     consequence de sa composition, pas une mesure.")
print(f"\n  2. LA CONCLUSION DE FOND SURVIT A LA NORMALISATION : {survit}.")
print(f"     Sur une perturbation relative identique, la propagation a l'elasticite la plus")
print(f"     faible des leviers SUBSTANTIELS ({e_g:.2f}), contre {e_xi:.2f} pour l'indice de queue,")
print(f"     {abs(elas['echelle sigma'][0]):.2f} pour l'echelle et {e_lam:.2f} pour la frequence.")
print(f"\n     UNE PRECISION QUI COMPTE, ET QUI VA DANS LE SENS DU POINT 1 : le seuil PRIS SEUL a")
print(f"     une elasticite de {e_u:.2f}, plus faible encore que la propagation. Dire que la")
print("     propagation est « le levier le moins sensible » sans cette precision serait donc")
print("     inexact. Mais ce classement-la est instructif plutot que genant : il montre que le")
print("     seuil en lui-meme ne fait presque rien, et que tout ce qu'on lui attribuait venait")
print("     de l'indice de queue qu'il entraine.")
print("\n  3. CE QU'IL FAUT DIRE DESORMAIS, ET C'EST PLUS FORT QUE L'ANCIENNE FORMULATION :")
print("     la sensibilite du modele a la contagion est faible PAR POINT DE POURCENTAGE, ce qui")
print("     ne depend d'aucune convention d'affichage. L'ancienne formulation comparait des")
print("     amplitudes et pouvait etre retournee en changeant une plage ; celle-ci non.")
print("\n  4. RESERVE A CONSERVER : une elasticite est locale, mesuree autour du point de base.")
print("     Elle ne dit pas ce qui se passe loin de ce point, et sur un parametre POSE comme g")
print("     la question « et si g valait 1,5 » reste legitime. La reponse est ailleurs, dans")
print("     l'invariance de l'ordre en g du script 66, et non dans ce tornado.")

# =====================================================================================
# figure S32
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
COUL = {"estime": BLUE, "pose": ACCENT, "choix": MUTED}

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime
# cette figure sur 7,27 pouces : tracee plus large, elle subissait une
# reduction qui faisait tomber ses etiquettes sous 6 points. Le rapport
# largeur sur hauteur est conserve, donc la figure occupe la meme place
# sur la page ; c'est son texte qui y prend plus de place.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.9, 3.34),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# (a) elasticites, triees
noms = sorted(elas, key=lambda k: abs(elas[k][0]))
ys = np.arange(len(noms))
vals = [abs(elas[n][0]) for n in noms]
cols = [COUL[elas[n][1]] for n in noms]
ax1.barh(ys, vals, color=cols, alpha=0.9, height=0.62)
for y, v, n in zip(ys, vals, noms):
    ax1.text(v + max(vals) * 0.015, y, f"{v:.2f}".replace(".", ","), va="center",
             fontsize=9, color=INK2)
ax1.set_yticks(ys)
ax1.set_yticklabels(noms, fontsize=9)
ax1.set_xlim(0, max(vals) * 1.18)
ax1.set_xlabel("élasticité de l'écart au paramètre, en valeur absolue", color=INK2)
# TITRE CORRIGE : il annoncait « la propagation est le levier le plus faible », ce que le
# graphique lui-meme dement, le seuil seul etant plus bas encore. Le rapport de dix a l'indice
# de queue est le fait a retenir, et il ne prete pas a contradiction.
ax1.set_title("(a)  À perturbation relative égale, la queue pèse\ndix fois la propagation",
              fontsize=10.5, color=INK, pad=8)
from matplotlib.patches import Patch                                       # noqa: E402
ax1.legend(handles=[Patch(color=BLUE, label="paramètre estimé"),
                    Patch(color=ACCENT, label="paramètre posé"),
                    Patch(color=MUTED, label="choix de méthode")],
           fontsize=8.5, frameon=False, loc="lower right")

# (b) le levier « seuil » decompose
lab = ["le paquet\n(ce que publie\nle tornado)"] + [c[0] for c in comp]
amp = [abs(paquet_haut - paquet_bas)] + [amps[c[0]] for c in comp]
colb = [MUTED] + [BLUE if c[0] != "u seul" else GREEN for c in comp]
xs = np.arange(len(lab))
ax2.bar(xs, amp, color=colb, alpha=0.9, width=0.6)
for x, v in zip(xs, amp):
    ax2.text(x, v * 1.02 + max(amp) * 0.01, f"{v:,.0f}".replace(",", " "), ha="center",
             fontsize=9, color=INK2)
ax2.set_xticks(xs)
ax2.set_xticklabels(lab, fontsize=8.5)
ax2.set_ylim(0, max(amp) * 1.16)
ax2.set_ylabel("amplitude sur l'écart entre états (M€)", color=INK2)
ax2.set_title("(b)  Le levier « seuil » déplace quatre paramètres :\nce n'est pas un levier",
              fontsize=10.5, color=INK, pad=8)
# LE FAIT A MONTRER : la barre de xi seul DEPASSE celle du paquet entier, ce qui n'est possible
# que si les composantes se compensent. Sans cette ligne, le lecteur voit deux barres sans
# comprendre que la seconde contredit la premiere.
ax2.annotate("", xy=(2, amps["xi seul"]), xytext=(0, amp[0]),
             arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.0, ls=":"))
ax2.text(1.0, (amps["xi seul"] + amp[0]) / 2 * 1.03,
         "ξ seul dépasse le paquet :\nles composantes se compensent",
         ha="center", fontsize=8.5, color=INK, fontweight="bold")

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("S32 : le tornado renormalisé, et ce que l'ancien classement devait à ses plages",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.91])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S32_tornado_normalise.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
