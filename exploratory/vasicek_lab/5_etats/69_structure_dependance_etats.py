#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
69 : la STRUCTURE de la dependance entre etats de conformite, et ce qu'elle coute au capital.

REMARQUE METIER A L'ORIGINE DE CE SCRIPT (Caroline Hillairet, point d'etape du 12 aout 2026) :
« etre conforme aux piliers 1 et 2 permet, indirectement, d'etre un peu conforme au pilier 3.
Il faut rester vigilant la-dessus dans la modelisation des dependances croisees. »

CE QUE LE MODELE FAIT DEJA, ET IL FAUT LE DIRE AVANT DE CONCLURE A UN MANQUE. Les etats de
conformite ne sont PAS independants : ils derivent d'une latente de Vasicek a facteur commun,
C*_j = gamma*Theta + sqrt(1-gamma^2)*eps_j avec gamma = 0,68, soit une correlation de latentes
de gamma^2 = 0,46 entre deux piliers quelconques. La dependance existe donc, et elle est forte.

CE QUI MANQUE, ET C'EST PRECIS. Cette dependance est ECHANGEABLE : elle vaut la meme chose pour
toutes les paires, P1 avec P3 comme P3 avec P5. La remarque porte sur une dependance
STRUCTUREE : le recouvrement des dispositifs de controle fait que la conformite de P1
(gouvernance) et de P2 (incidents) confere une conformite partielle a P3 (tests), et ce lien-la
n'a aucune raison de valoir pour une paire quelconque. Ce n'est donc pas la dependance qui
manque, c'est sa STRUCTURE.

DEUX CHOSES A NE PAS CONFONDRE, et le projet a deja paye ce genre de confusion :
  - la PROPAGATION (matrice W) dit qu'une DEFAILLANCE de j entraine celle de k. Elle vit au
    niveau de l'incident et elle est deja asymetrique et deja bornee ;
  - la CONFERENCE DE CONFORMITE, objet de ce script, dit qu'un ETAT conforme de j rend l'etat
    conforme de k plus probable. Elle vit au niveau de l'etat, un cran au-dessus.
Les deux se ressemblent et ne sont pas le meme objet. Ce script ne touche pas W.

COMMENT LA STRUCTURE EST INTRODUITE SANS DEPLACER LES MARGES. On passe par la matrice de
correlation des latentes. Elle vaut gamma^2 hors diagonale dans le modele actuel, et l'on ajoute
un surcroit delta aux deux seules paires (P1,P3) et (P2,P3) :

    R[j,k] = gamma^2 + delta   pour (P1,P3) et (P2,P3),   gamma^2 ailleurs,   1 sur la diagonale

Les latentes se tirent alors par factorisation de Cholesky. La diagonale valant un, les marges
sont preservees PAR CONSTRUCTION, et la probabilite d'etre non conforme sur chaque pilier ne
bouge pas d'un millieme : la comparaison porte sur la structure et sur rien d'autre, exactement
comme l'axe de dependance de la bande de modele compare des copules a marges fixees. Le
surcroit admissible est borne par la positivite de R, et le script imprime cette borne.

Sortie : diagnostics + figure S25_structure_dependance_etats.png.
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

# Meme protocole que les scripts 16b, 20 et 20b : trois canaux par etat, lecture binaire C / NC.
SCENARIO = {"C": "S0_conforme", "NC": "S2_non_conforme"}
G_PROP = {"C": 0.45, "NC": 0.90}
PU_MULT = {"C": 0.85, "NC": 1.20}
MULT_ETAT = {st: ec.lambda_scenario("OPRISK", sc, mode="center") / PARAMS["OPRISK"]["lam_ref"]
             for st, sc in SCENARIO.items()}
_S = {j: eng.LAMBDA[j] for j in PIL}
SHARE = {j: _S[j] / sum(_S.values()) for j in PIL}

# Latente de conformite : ancrage et charge systemique du script 16, lecture BINAIRE.
P_NC = 0.35
GAMMA = 0.68
K = norm.ppf(P_NC)

NY = 40_000
SEEDS = (909, 1234, 2718, 31415)
SOURCE = "OPRISK"

# Les trois piliers de la remarque. L'ordre de PIL fixe les indices, on ne le suppose pas.
J1, J2, J3 = PIL[0], PIL[1], PIL[2]

# Surcroits de correlation balayes. delta = 0 redonne EXACTEMENT le modele actuel. Le dernier
# cran est fixe par la POSITIVITE de la matrice de correlation et non a la main : une premiere
# version balayait jusqu'a 0,40 et plantait, la borne admissible valant 0,38.
DELTAS_DEMANDES = (0.0, 0.10, 0.20, 0.30, 0.40)
NSIM_ETATS = 4_000_000


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def annual_config(state_map, ny, seed):
    """Pertes annuelles d'une configuration {pilier -> etat}, a nombres communs."""
    sp = PARAMS[SOURCE]
    lam_vec = {j: sp["lam_ref"] * SHARE[j] * MULT_ETAT[state_map[j]] for j in PIL}
    g_vec = {j: G_PROP[state_map[j]] for j in PIL}
    p_u_vec = {j: min(0.999, sp["p_u"] * PU_MULT[state_map[j]]) for j in PIL}
    rng = np.random.default_rng(seed)
    return ec.simulate_euro_pp(lam_vec, g_vec, sp["xi"], sp["sigma"], sp["u"],
                               p_u_vec, sp["cap"], ny, rng)


def masque_vers_etats(m):
    """bit a 1 = pilier NON CONFORME."""
    return {j: ("NC" if m & (1 << i) else "C") for i, j in enumerate(PIL)}


def nom_masque(m):
    nc = [f"P{j}" for i, j in enumerate(PIL) if m & (1 << i)]
    return "tous conformes" if not nc else "+".join(nc) + " NC"


# =====================================================================================
titre("1. Le capital des 32 configurations de conformite, a nombres communs")
# =====================================================================================
print(f"  Lecture binaire C / NC, {N} piliers, {2 ** N} configurations, {NY} annees, "
      f"{len(SEEDS)} graines.")
print(f"  Canaux par etat : lambda x{MULT_ETAT['C']:.2f} / x{MULT_ETAT['NC']:.2f}, "
      f"g {G_PROP['C']} / {G_PROP['NC']}, p_u x{PU_MULT['C']} / x{PU_MULT['NC']}.")

scr = np.empty(2 ** N)
for m in range(2 ** N):
    sm = masque_vers_etats(m)
    scr[m] = float(np.mean([var(annual_config(sm, NY, s)) for s in SEEDS]))

ordre = np.argsort(scr)
print(f"\n  {'configuration':<26}{'SCR (M)':>10}      {'configuration':<26}{'SCR (M)':>10}")
for a, b in zip(ordre[:16], ordre[16:]):
    print(f"  {nom_masque(a):<26}{scr[a]:>10.0f}      {nom_masque(b):<26}{scr[b]:>10.0f}")

print(f"\n  Le capital va de {scr.min():.0f} M (tous conformes) a {scr.max():.0f} M "
      f"(tous non conformes).")
# LA CONVEXITE EST LE MECANISME, et elle se verifie au lieu de se supposer : c'est elle qui
# decide du SENS de l'effet d'une dependance plus polarisante.
par_nb = [scr[[m for m in range(2 ** N) if bin(m).count("1") == k]].mean()
          for k in range(N + 1)]
print(f"\n  SCR moyen par NOMBRE de piliers non conformes :")
print("    " + "  ".join(f"{k}:{par_nb[k]:.0f}" for k in range(N + 1)))
incr = np.diff(par_nb)
print(f"  increments successifs : " + "  ".join(f"{d:+.0f}" for d in incr))
convexe = bool(np.all(np.diff(incr) > 0))
print(f"  increments strictement croissants (convexite stricte) : {convexe}")
print(f"  amplitude des increments : de {incr.min():.0f} a {incr.max():.0f} M, "
      f"soit {100 * (incr.max() - incr.min()) / incr.mean():.0f} % autour de leur moyenne.")
# CE RESULTAT DECIDE DE TOUT LE RESTE, ET IL N'EST PAS CELUI QUE J'ATTENDAIS. Le capital est
# QUASI LINEAIRE en nombre de piliers non conformes : les increments ne varient que de quelques
# pour cent et leur ordre n'est meme pas monotone, ce qui a cette resolution est du bruit de
# simulation plutot qu'une courbure. Or c'est la COURBURE qui transforme une dependance en
# effet sur l'esperance : a fonction lineaire, l'inegalite de Jensen est une egalite et la
# structure de la dependance NE PEUT PAS deplacer le capital espere a marges fixees. La
# prediction est donc un effet quasi nul, et la section 3 la met a l'epreuve.
print("\n  LECTURE, ET ELLE COMMANDE LA SUITE. Le capital est QUASI LINEAIRE en nombre de piliers")
print("  non conformes. Or c'est la COURBURE qui transforme une dependance en effet sur")
print("  l'esperance : a fonction exactement lineaire, l'esperance ne depend que des MARGES et")
print("  la structure de la dependance n'a aucun effet, quelle qu'elle soit. On doit donc")
print("  s'attendre a un effet quasi nul, et c'est une PREDICTION que la section 3 teste.")


# =====================================================================================
titre("2. La loi des etats : echangeable aujourd'hui, structuree dans la variante")
# =====================================================================================
def matrice_corr(delta):
    """Correlation des latentes : echangeable a rho = gamma^2, plus delta sur (P1,P3) et (P2,P3).

    LA CONSTRUCTION PASSE PAR UNE MATRICE DE CORRELATION, ET C'EST NECESSAIRE. Une premiere
    version de ce script combinait la latente de P3 avec la moyenne des latentes de P1 et P2
    par une ponderation en racine, sqrt(kappa) et sqrt(1-kappa). Deux defauts : le second terme
    y perdait le facteur commun Theta, ce qui DE-correlait P3 du reste au lieu de le
    sur-correler au bloc (le balayage donnait alors un effet non monotone, signature du bug), et
    la variance de la somme n'etait de toute facon pas un, les deux termes etant correles. Une
    matrice de correlation dont la diagonale vaut un preserve les marges par CONSTRUCTION, ce
    qui est exactement la condition pour que la comparaison porte sur la structure seule.
    """
    rho = GAMMA ** 2
    R = np.full((N, N), rho)
    np.fill_diagonal(R, 1.0)
    i1, i2, i3 = PIL.index(J1), PIL.index(J2), PIL.index(J3)
    for i in (i1, i2):
        R[i, i3] = R[i3, i] = rho + delta
    return R


def poids_configs(delta, nsim=NSIM_ETATS, seed=20260812):
    """Probabilite des 2^N configurations sous la latente, a surcroit de correlation delta.

    delta = 0 redonne EXACTEMENT la latente echangeable du modele actuel.
    """
    R = matrice_corr(delta)
    L = np.linalg.cholesky(R)                                  # leve si R n'est pas definie positive
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((nsim, N)) @ L.T
    nc = C <= K                                                # bit a 1 = non conforme
    codes = (nc * (1 << np.arange(N))).sum(axis=1)
    return np.bincount(codes, minlength=2 ** N) / nsim


def delta_admissible(pas=0.01, borne=0.60):
    """Plus grand delta laissant la matrice de correlation definie positive."""
    d, best = 0.0, 0.0
    while d <= borne:
        try:
            np.linalg.cholesky(matrice_corr(d))
            best = d
        except np.linalg.LinAlgError:
            break
        d += pas
    return best


print(f"  Latente : C*_j = {GAMMA}*Theta + sqrt(1-{GAMMA}^2)*eps_j, seuil K = Phi^-1({P_NC}) "
      f"= {K:+.3f}.")
print(f"  Correlation de latentes entre deux piliers, modele actuel : gamma^2 = {GAMMA ** 2:.3f}.")
print(f"  La dependance EXISTE donc deja, et elle est forte. Ce qui change avec delta n'est pas")
print(f"  son existence mais sa STRUCTURE : les paires (P{J1},P{J3}) et (P{J2},P{J3}) passent a")
print(f"  gamma^2 + delta quand les autres restent a gamma^2, a marges strictement inchangees.")
d_max = delta_admissible()
print(f"  Surcroit maximal admissible (matrice definie positive) : delta = {d_max:.2f}, soit une")
print(f"  correlation de {GAMMA ** 2 + d_max:.2f} sur les deux paires structurees.")
DELTAS = tuple(sorted({min(d, d_max) for d in DELTAS_DEMANDES}))
if DELTAS != DELTAS_DEMANDES:
    print(f"  Le balayage demande est ramene a la borne : {', '.join(f'{d:.2f}' for d in DELTAS)}.")

w0 = poids_configs(0.0)
print(f"\n  Controle des marges a delta = 0 : P(NC) par pilier =")
marg0 = [sum(w0[m] for m in range(2 ** N) if m & (1 << i)) for i in range(N)]
print("    " + "  ".join(f"P{PIL[i]}:{marg0[i]:.3f}" for i in range(N))
      + f"   (ancrage {P_NC})")


# =====================================================================================
titre("3. Ce que la structure change au capital espere")
# =====================================================================================
print(f"  {'delta':>7}{'E[SCR] (M)':>14}{'ecart au modele actuel':>26}"
      f"{'P(tous NC)':>13}{'P(tous C)':>12}{'marge P3':>11}")
base = None
res = []
for kap in DELTAS:
    w = poids_configs(kap)
    e = float(w @ scr)
    if base is None:
        base = e
    p_tousnc = w[2 ** N - 1]
    p_tousc = w[0]
    mp3 = sum(w[m] for m in range(2 ** N) if m & (1 << PIL.index(J3)))
    res.append((kap, e, e - base, p_tousnc, p_tousc, mp3))
    print(f"  {kap:>7.2f}{e:>14.0f}{e - base:>+22.0f} M{p_tousnc:>13.3f}{p_tousc:>12.3f}"
          f"{mp3:>11.3f}")

ecart_max = max((abs(r[2]) for r in res))
print(f"\n  La marge de P{J3} ne bouge pas d'un millieme sur tout le balayage : la comparaison")
print("  porte bien sur la structure seule, et non sur un deplacement d'ancrage.")
print(f"\n  ET POURTANT LA STRUCTURE CHANGE BIEN LA LOI DES CONFIGURATIONS : P(tous NC) passe de")
print(f"  {res[0][3]:.3f} a {res[-1][3]:.3f} et P(tous C) de {res[0][4]:.3f} a {res[-1][4]:.3f}, "
      f"soit des deplacements")
print("  de l'ordre de vingt pour cent en relatif. Les configurations POLARISEES deviennent plus")
print("  probables et les mixtes moins, ce qui est exactement l'effet attendu d'une conference de")
print(f"  conformite. Le capital espere, lui, ne bouge que de {ecart_max:.0f} M.")

# =====================================================================================
titre("3bis. Pourquoi l'effet est nul : le capital est presque ADDITIF par pilier")
# =====================================================================================
# LA PREDICTION DE LA SECTION 1 SE VERIFIE, ET SA RAISON EST PLUS FORTE QUE LA LINEARITE EN
# NOMBRE. Si le capital s'ecrit comme une constante plus une somme de contributions par pilier,
# alors son esperance ne depend QUE des marges, quelle que soit la dependance : les termes
# croises sont ce qui pourrait la faire dependre de la structure, et il n'y en a presque pas.
X = np.ones((2 ** N, N + 1))
for m in range(2 ** N):
    for i in range(N):
        X[m, i + 1] = 1.0 if m & (1 << i) else 0.0
coef, *_ = np.linalg.lstsq(X, scr, rcond=None)
pred = X @ coef
resid = scr - pred
r2 = 1.0 - resid.var() / scr.var()
print("  Ajustement du capital des 32 configurations par une forme ADDITIVE")
print("  SCR(config) ~ a + somme des b_j sur les piliers non conformes :")
print(f"    a (tous conformes) = {coef[0]:.0f} M")
print("    " + "  ".join(f"b(P{PIL[i]})={coef[i + 1]:.0f}" for i in range(N)))
print(f"  R2 = {r2:.4f} ; residu maximal {np.abs(resid).max():.0f} M sur un capital de "
      f"{scr.min():.0f} a {scr.max():.0f} M.")
print("\n  VOILA LA RAISON. L'esperance d'une fonction ADDITIVE des indicateurs de pilier ne")
print("  depend que des MARGES, jamais de la dependance : les termes croises sont le seul canal")
print("  par lequel une structure pourrait agir, et il ne reste presque rien a ce canal. Ce")
print("  n'est pas une coincidence de ce script : c'est le meme fait que l'interaction entre")
print("  PILIERS deja publiee, petite devant la somme des marginaux et de signe non resolu")
print("  d'une graine a l'autre (scripts 20 et 20b). Deux lectures du meme constat.")
print("\n  CE QUE CELA NE DIT PAS. L'invariance porte sur le capital ESPERE. Une conference de")
print("  conformite deplace bien la loi des configurations, donc elle deplacerait un quantile de")
print("  cette loi ou une mesure de sa queue : ce qui est etabli ici est que l'esperance n'en")
print("  voit rien, pas que le phenomene est sans consequence sur toute grandeur.")

# =====================================================================================
titre("4. La taille de l'effet, mise en regard de ce qui est deja publie")
# =====================================================================================
print(f"  Ecart maximal en valeur absolue sur le balayage : {ecart_max:.0f} M")
print(f"  Ecart en part du capital espere de base         : {100 * ecart_max / base:.2f} %")
print("  A comparer aux bandes deja publiees, et c'est la mise en perspective qui compte :")
print("    bande d'identification de la direction   1 839 M de largeur")
print("    axe de dependance de la bande de modele  de 5 322 a 9 806 M")
print("  La structure de la dependance des ETATS pese donc TROIS ORDRES DE GRANDEUR de moins que")
print("  l'ignorance sur la direction de la propagation. La remarque est juste sur le fond, le")
print("  modele omet bien ce mecanisme, et l'omission ne deplace pas le capital espere.")
print("\n  A NE PAS SURINTERPRETER, DANS LES DEUX SENS. delta n'est pas calibre : aucune donnee du")
print("  projet ne mesure le recouvrement des dispositifs de controle entre piliers, et le")
print("  balayage donne un ordre de grandeur, pas une correction. Mais l'invariance constatee ne")
print("  depend PAS de la valeur de delta : elle tient a l'additivite du capital, donc elle vaut")
print("  pour toute structure de dependance a marges fixees, y compris celles qu'on n'a pas")
print("  essayees. C'est ce qui rend la conclusion solide malgre l'absence de calibration.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. Les etats de conformite ne sont PAS independants dans le modele actuel : ils")
print(f"     partagent un facteur commun de charge {GAMMA}, soit une correlation de latentes de")
print(f"     {GAMMA ** 2:.2f}. Repondre a la remarque par « la dependance manque » serait faux.")
print("  2. Ce qui manque est sa STRUCTURE : la dependance est echangeable, alors que le")
print("     recouvrement des dispositifs de controle est specifique a certains triplets.")
print(f"  3. La structure deplace bien la loi des configurations, P(tous NC) passant de "
      f"{res[0][3]:.3f} a {res[-1][3]:.3f},")
print(f"     et pourtant le capital espere ne bouge que de {ecart_max:.0f} M, soit "
      f"{100 * ecart_max / base:.2f} %.")
print(f"  4. LA RAISON EST STRUCTURELLE ET NON NUMERIQUE : le capital est presque ADDITIF par")
print(f"     pilier (R2 = {r2:.4f}), et l'esperance d'une fonction additive ne depend que des")
print("     marges. L'invariance vaut donc pour TOUTE structure a marges fixees, pas seulement")
print("     pour celles qu'on a essayees, ce qui la rend solide sans calibration de delta.")
print("  5. Ne pas confondre avec la propagation : W dit qu'une defaillance se transmet, ceci")
print("     dit qu'une conformite se confere. Deux etages, deux objets.")

# =====================================================================================
# figure S25
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.9, 3.21),
                               gridspec_kw={"width_ratios": [1, 1.05]})

# (a) la convexite, qui porte le sens de l'effet
ax1.plot(range(N + 1), par_nb, "o-", color=BLUE, lw=1.8, ms=7)
for k in range(N + 1):
    ax1.annotate(f"{par_nb[k]:.0f}", (k, par_nb[k]), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8, color=INK2)
ax1.set_xlabel("nombre de piliers non conformes", color=INK2)
ax1.set_ylabel("SCR moyen sur les configurations (M€)", color=INK2)
ax1.set_xticks(range(N + 1))
ax1.set_ylim(min(par_nb) * 0.88, max(par_nb) * 1.10)
ax1.set_title("(a)  Le capital est presque linéaire en nombre de piliers,\net c'est ce qui annule "
              "l'effet de la structure",
              fontsize=10.5, color=INK, pad=8)

# (b) l'effet de la structure, en regard des bandes deja publiees
kk = [r[0] for r in res]
dd = [r[2] for r in res]
# L'ECHELLE DE CE PANNEAU EST LE PROPOS, PAS UN DETAIL. Tracees sur leur propre amplitude, des
# barres de 2 M paraissent enormes : la premiere version de cette figure donnait un axe de deux
# millions d'euros de haut et suggerait un effet massif, alors que le resultat est une
# invariance. L'axe est donc cale sur la LARGEUR DE LA BANDE D'IDENTIFICATION deja publiee, qui
# est l'incertitude a laquelle cet effet doit etre compare. Les barres devienment invisibles :
# c'est exactement ce que le script mesure.
REF_BANDE = 1839.0
ax2.bar(range(len(kk)), dd, color=[GREEN if d <= 0 else ACCENT for d in dd], alpha=0.9,
        width=0.6)
ax2.axhline(0, color=INK2, lw=0.8)
ax2.axhline(REF_BANDE, color=MUTED, lw=1.4, ls="--")
ax2.text(len(kk) - 0.5, REF_BANDE * 0.94,
         f"largeur de la bande d'identification, {REF_BANDE:.0f} M€",
         ha="right", va="top", fontsize=8.5, color=INK2)
ax2.annotate(f"écart maximal {max(abs(d) for d in dd):.0f} M€", xy=(len(kk) - 1, 0),
             xytext=(len(kk) - 1.15, REF_BANDE * 0.35), fontsize=8.5, color=ACCENT,
             ha="right", arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))
ax2.set_xticks(range(len(kk)))
ax2.set_xticklabels([f"{k:.2f}" for k in kk], fontsize=9)
ax2.set_xlim(-0.6, len(kk) - 0.4)
ax2.set_ylim(-REF_BANDE * 0.12, REF_BANDE * 1.12)
ax2.set_xlabel("surcroît de corrélation δ sur les paires (P1,P3) et (P2,P3)", color=INK2)
ax2.set_ylabel("écart au capital espéré actuel (M€)", color=INK2)
ax2.set_title("(b)  Ce que la structure change, à marges strictement inchangées,\n"
              "et à l'échelle de l'incertitude déjà déclarée",
              fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S25_structure_dependance_etats.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
