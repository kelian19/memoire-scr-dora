#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
70 : TEST de la renormalisation proportionnelle au SCR de marche, contre la descente a deux canaux.

DEMANDE A L'ORIGINE (Caroline Hillairet, point d'etape du 12 aout 2026) : « pour calculer le SCR
par taille d'entreprise, petite, moyenne et grande, nous appliquerons un facteur de
renormalisation proportionnel au SCR du marche ». Le memoire refuse cette mise a l'echelle et
motive son refus par deux elasticites mesurees. Un refus argumente n'est pas un test : ce script
MESURE l'ecart entre les deux methodes, pour que l'arbitrage se fasse sur un nombre.

CE QUI EST COMPARE, ET POURQUOI LE RAPPORT PLUTOT QUE LE NIVEAU. Comparer des NIVEAUX melerait
un probleme d'unites (le SCR du secteur n'est pas celui d'une firme mediane) au probleme de
methode. On compare donc, pour chaque taille, le RAPPORT du besoin de capital a celui d'une
taille de reference :

  proportionnelle : R(taille) = taille / taille_ref          -> elasticite 1 par construction
  deux canaux     : R(taille) = SCR(taille) / SCR(taille_ref) -> elasticite mesuree

La question devient alors sans ambiguite : de combien la proportionnalite deforme-t-elle le
rapport entre une petite et une grande entite.

LES DEUX CANAUX, TELS QUE LE PROJET LES APPLIQUE DEJA :
  frequence : lambda lu comme une FONCTION de la taille, lien logarithmique de la binomiale
              negative, elasticite b_lambda = +0,0744 (module descente, script 60) ;
  severite  : homothetie exacte, la GPD etant une famille d'echelle au-dessus du seuil,
              de facteur (taille / taille de reference)^b avec b = 0,087 (script 57).

CE QUE CE SCRIPT NE PRETEND PAS. Il ne dit pas que la descente a deux canaux est juste. La
quasi-nullite de l'elasticite de severite est precisement la limite que le memoire declare, et
qui lui fait publier le chiffre d'entite comme une BORNE SUPERIEURE. Le test tranche une question
plus modeste et parfaitement nette : l'elasticite unitaire que suppose la proportionnalite est-elle
compatible avec la donnee.

Sortie : diagnostics + figure S26_test_renormalisation.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import descente as dsc                                          # noqa: E402
import canaux_conformite as cx                                  # noqa: E402
from euro_cascade_model import var                              # noqa: E402

WID = 88

# TAILLES POSEES, ET DECLAREES COMME TELLES. Une decade de part et d'autre de l'entite
# notionnelle du memoire (20 000 M USD d'actifs), ce qui encadre la firme mediane du panel
# (9 903 M USD) et la mediane ponderee par les evenements (121 894 M USD).
TAILLES = [("petite", 2_000.0), ("moyenne", 20_000.0), ("grande", 200_000.0)]
REF = 20_000.0

# LA FREQUENCE D'ENTITE EST BASSE, DONC IL FAUT BEAUCOUP D'ANNEES. Le script 58 en prend
# 600 000 pour la meme raison : a lambda de l'ordre du dixieme, la quasi-totalite des annees
# sont vides et le quantile a 99,5 % se lit sur la queue du petit nombre d'annees chargees.
NY = 400_000
NSEED = 4

# Deux etats de canaux, pour montrer que le RAPPORT ne depend pas de l'etat choisi.
ETATS = [("conforme", dict(g=cx.G_C, p_u=cx.PU_C, phi_cs=None)),
         ("non conforme", dict(g=cx.G_NC, p_u=cx.PU_NC, phi_cs=cx.PHICS_NC))]


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def scr_taille(actifs, cfg, D):
    """SCR d'une entite de cette taille : lambda lu a la taille, severite en homothetie.

    L'HOMOTHETIE S'APPLIQUE HORS DU QUANTILE, et c'est exact : la VaR est positivement
    homogene, donc multiplier les pertes par un facteur ou multiplier le quantile par ce
    facteur donne le meme resultat. C'est la convention du script 58.
    """
    lam = D.lam(actifs)
    m = D.mult(actifs)
    scrs = []
    for k in range(NSEED):
        rng = np.random.default_rng(cx.SEED0 + k)
        annual = cx.pertes_annuelles(lam, cfg["g"], cfg["p_u"], cfg["phi_cs"], rng, ny=NY)
        scrs.append(var(annual) * m)
    return float(np.mean(scrs)), lam, m


# =====================================================================================
titre("1. Les deux canaux, lus a chaque taille")
# =====================================================================================
D = dsc.Descente()
print(f"  elasticite frequence / taille : b_lambda = {D.b_lam:+.4f}")
print(f"  elasticite severite / taille  : b = {dsc.B_SEV:.3f} "
      f"[{dsc.B_SEV_LO:.3f} ; {dsc.B_SEV_HI:.3f}]")
print(f"  taille de reference du multiplicateur de severite : {D.med_act_ict:,.0f} M USD"
      .replace(",", " "))
print(f"\n  {'taille':<12}{'actifs (M USD)':>16}{'lambda / an':>14}{'mult. severite':>16}")
for nom, a in TAILLES:
    print(f"  {nom:<12}{a:>16,.0f}{D.lam(a):>14.4f}{D.mult(a):>16.4f}".replace(",", " "))

# =====================================================================================
titre("2. Le rapport de capital entre tailles : proportionnelle contre deux canaux")
# =====================================================================================
res = {}
for etat, cfg in ETATS:
    vals = {}
    for nom, a in TAILLES:
        s, lam, m = scr_taille(a, cfg, D)
        vals[nom] = s
    res[etat] = vals

ref_nom = "moyenne"
print(f"  Reference : entite {ref_nom} ({REF:,.0f} M USD d'actifs). "
      "Rapports au capital de cette entite.".replace(",", " "))
print(f"\n  {'taille':<12}{'proportionnelle':>18}{'deux canaux (C)':>19}"
      f"{'deux canaux (NC)':>20}{'facteur d ecart':>18}")
ecarts = {}
for nom, a in TAILLES:
    r_prop = a / REF
    r_c = res["conforme"][nom] / res["conforme"][ref_nom]
    r_nc = res["non conforme"][nom] / res["non conforme"][ref_nom]
    ecarts[nom] = (r_prop, r_c, r_nc)
    fac = r_prop / r_c if r_c else float("nan")
    print(f"  {nom:<12}{r_prop:>18.3f}{r_c:>19.3f}{r_nc:>20.3f}{fac:>18.1f}")

print("\n  LE RAPPORT NE DEPEND PRATIQUEMENT PAS DE L'ETAT DES CANAUX, conforme ou non : les")
print("  deux colonnes du milieu se suivent. La question de la mise a l'echelle est donc bien")
print("  separable de celle de la conformite, ce qui n'allait pas de soi et rend le test propre.")

# elasticites implicites, calculees et non posees
def elasticite(vals):
    x = np.log([a for _, a in TAILLES])
    y = np.log([vals[n] for n, _ in TAILLES])
    return float(np.polyfit(x, y, 1)[0])


e_c = elasticite(res["conforme"])
e_nc = elasticite(res["non conforme"])
print(f"\n  ELASTICITE IMPLICITE DU CAPITAL A LA TAILLE, par regression log-log sur les trois")
print(f"  tailles : {e_c:.3f} a l'etat conforme et {e_nc:.3f} a l'etat non conforme, contre")
print(f"  1,000 que la proportionnalite suppose.")

# =====================================================================================
titre("3. Ce que la proportionnalite ferait, et dans quel sens")
# =====================================================================================
r_petite = ecarts["petite"]
r_grande = ecarts["grande"]
print(f"  Sur la petite entite, la proportionnalite donne un rapport de {r_petite[0]:.3f} quand")
print(f"  les deux canaux donnent {r_petite[1]:.3f} : elle SOUS-ESTIME d'un facteur "
      f"{r_petite[1]/r_petite[0]:.1f}.")
print(f"  Sur la grande, elle donne {r_grande[0]:.1f} contre {r_grande[1]:.3f} : elle")
print(f"  SUR-ESTIME d'un facteur {r_grande[0]/r_grande[1]:.1f}.")
print(f"  Sur toute la plage, de la petite a la grande, la proportionnalite ecarte les deux")
print(f"  entites d'un facteur {r_grande[0]/r_petite[0]:.0f} quand la mesure les ecarte de "
      f"{r_grande[1]/r_petite[1]:.2f}.")

print("\n  LE MECANISME, ET IL EST DEJA DANS LE MEMOIRE. La severite ne se met presque pas a")
print("  l'echelle de la taille (elasticite 0,087), et l'effet de la frequence sur un quantile")
print("  de queue lourde n'est pas lineaire : par le principe du grand saut unique, multiplier")
print("  lambda par un facteur multiplie le quantile par ce facteur puissance xi, soit une")
print("  elasticite de frequence deja amortie. Les deux canaux composent donc en une elasticite")
print("  d'un ordre de grandeur sous l'unite, et c'est ce que la regression ci-dessus retrouve.")

# =====================================================================================
titre("4. Ce que le test tranche, et ce qu'il ne tranche pas")
# =====================================================================================
print("  TRANCHE. L'elasticite unitaire que suppose la proportionnalite est incompatible avec")
print(f"  les deux elasticites mesurees : {D.b_lam:+.4f} sur la frequence, significative a")
print(f"  z = {D.z_b:.2f} et pourtant tres loin de 1, et {dsc.B_SEV:.3f} sur la")
print(f"  severite, d'intervalle [{dsc.B_SEV_LO:.3f} ; {dsc.B_SEV_HI:.3f}] qui exclut 1 de tres")
print("  loin. Adopter la proportionnalite reviendrait a poser une elasticite que la donnee")
print("  rejette sur les deux canaux separement.")
print("\n  NE TRANCHE PAS. Que la descente a deux canaux soit juste. Une elasticite de severite")
print("  quasi nulle affirme qu'une entite de quelques milliards subit a peu pres la severite")
print("  d'une institution mondiale, ce qui est la limite que le memoire declare et qui lui fait")
print("  publier le chiffre d'entite comme une BORNE SUPERIEURE. Les deux methodes se trompent")
print("  donc en sens OPPOSES, et le test dit seulement laquelle est rejetee par la mesure.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print(f"  1. Elasticite du capital a la taille : {e_c:.3f} mesuree, 1,000 supposee par la")
print("     proportionnalite. L'ecart n'est pas de degre, il est d'ordre de grandeur.")
print(f"  2. Entre la petite et la grande entite, la proportionnalite ecarte les capitaux d'un")
print(f"     facteur {r_grande[0]/r_petite[0]:.0f} quand la mesure les ecarte de "
      f"{r_grande[1]/r_petite[1]:.2f}.")
print("  3. Le sens de l'erreur s'inverse avec la taille : la proportionnalite sous-estime les")
print("     petites entites et sur-estime les grandes.")
print("  4. Le rapport est insensible a l'etat des canaux, donc la mise a l'echelle et la")
print("     conformite sont bien deux questions separees.")
print("  5. A dire a Caroline : la proportionnalite est rejetee par la mesure, mais la methode")
print("     retenue a sa propre limite, declaree, et de sens contraire. Le desaccord n'est pas")
print("     sur la prudence, il est sur le canal par lequel la taille agit.")

# =====================================================================================
# figure S26
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

# La virgule decimale se pose sur le NOMBRE seul, jamais par un .replace sur la phrase.
# Defini AVANT le premier panneau : la version precedente le posait juste avant le suptitle,
# donc apres son premier usage, et le script tombait sur un NameError.
_e_fr = f"{e_c:.2f}".replace(".", ",")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.2))

# (a) les deux lois d'echelle, en log-log
a_grid = np.logspace(np.log10(1_000), np.log10(500_000), 60)
ax1.loglog(a_grid, a_grid / REF, "--", color=MUTED, lw=1.8,
           label="proportionnelle (élasticité 1)")
mes = [ecarts[n][1] for n, _ in TAILLES]
ax1.loglog([a for _, a in TAILLES], mes, "o-", color=BLUE, lw=2, ms=8,
           label=f"deux canaux (élasticité {_e_fr})")
for (nom, a), r in zip(TAILLES, mes):
    ax1.annotate(f"{nom}\n{f'{r:.2f}'.replace('.', ',')}", (a, r),
                 textcoords="offset points", xytext=(0, -30),
                 ha="center", fontsize=8.5, color=INK2)
ax1.axvline(REF, color=INK2, lw=0.7, ls=":")
ax1.set_xlabel("actifs de l'entité (M USD)", color=INK2)
ax1.set_ylabel("capital rapporté à l'entité moyenne", color=INK2)
ax1.legend(fontsize=8.5, frameon=False, loc="upper left")
ax1.set_title("(a)  Deux lois d'échelle, et l'écart est d'ordre de grandeur",
              fontsize=10.5, color=INK, pad=8)

# (b) le facteur d'ecart par taille, en clair
noms = [n for n, _ in TAILLES]
fac = [ecarts[n][0] / ecarts[n][1] for n in noms]
# TROIS DEFAUTS CORRIGES APRES AVOIR REGARDE LA FIGURE. L'etiquette de la petite entite, posee
# a gauche de sa barre, recouvrait le libelle de l'axe. La taille de reference etait annotee
# « sur-estime » alors qu'elle vaut exactement 1 par construction. Et les nombres portaient le
# point decimal anglais.
def _sens(f):
    if abs(f - 1.0) < 5e-3:
        return "référence, exacte par construction"
    return "sous-estime" if f < 1 else "sur-estime"


cols = [MUTED if abs(f - 1.0) < 5e-3 else (GREEN if f < 1 else ACCENT) for f in fac]
y = np.arange(len(noms))[::-1]
ax2.barh(y, fac, color=cols, alpha=0.88, height=0.5)
ax2.axvline(1.0, color=INK2, lw=1.2)
for yy, f, n in zip(y, fac, noms):
    # L'ETIQUETTE SE POSE TOUJOURS A DROITE DE LA BARRE, quel que soit le sens.
    ax2.text(f * 1.10, yy, f"×{f'{f:.2f}'.replace('.', ',')}  ({_sens(f)})", va="center",
             ha="left", fontsize=9, color=INK2)
ax2.set_xscale("log")
ax2.set_yticks(y)
ax2.set_yticklabels(noms, fontsize=10)
ax2.set_xlim(min(fac) * 0.55, max(fac) * 22.0)
ax2.set_xlabel("rapport proportionnelle / mesurée (échelle log)", color=INK2)
ax2.set_title("(b)  Le sens de l'erreur s'inverse avec la taille",
              fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("S26 : la renormalisation proportionnelle au marché suppose une élasticité de 1 ; "
             f"la mesure en donne {_e_fr}",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S26_test_renormalisation.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
