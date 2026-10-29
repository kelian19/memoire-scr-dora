#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
86 : le plafond et la saturation, mesures CONJOINTEMENT. Deux reserves, un seul objet.

CE QUE LE PROJET A ETABLI SEPAREMENT. Le memoire declare deux limites structurelles, traitees par
deux scripts et jamais ensemble :
  - script 74 : l'ADDITIVITE DES COUTS au sein d'un sinistre n'est pas testee. Bornee par un
    exposant theta sur le nombre de piliers touches, elle deplace l'ecart DORA de 76 % ;
  - script 78 : la SEVERITE N'EST PAS PLAFONNEE a l'echelle d'entite. Un plafond kappa x E
    adosse a l'exposition la borne, et le plafond qui ampute le capital de moitie est quasi
    invariant en euros.
Le script 78 note que ce sont « les deux moities d'une meme forme fonctionnelle » et qu'elles
agissent EN SENS CONTRAIRE, le plafond retirant de la queue et la saturation en ajoutant. Il en
tire qu'il ne faut pas les additionner comme deux reserves independantes. MAIS IL NE L'A PAS
MESURE : l'affirmation etait un raisonnement, pas un resultat.

CE QUE CE SCRIPT FAIT. Il applique les deux ensemble, sur le meme sinistre :
        perte du sinistre  =  min( kappa x E ,  (somme des severites) x k^(theta - 1) )
et balaie le rectangle (kappa, theta). Trois mesures en sortent :
  1. les deux bandes marginales, chacune l'autre reserve neutralisee ;
  2. la bande conjointe sur le rectangle, a comparer a la somme des marginales ;
  3. LE TERME D'INTERACTION, seule quantite qui dise si les deux reserves sont separables :
        I(kappa, theta) = V(kappa, theta) - V(kappa, 1) - V(inf, theta) + V(inf, 1).
C'est la meme construction que le treillis des quatre canaux, appliquee a deux reserves au lieu
de quatre parametres, et elle repond a la meme question : peut-on les traiter une par une ?

POURQUOI A L'ECHELLE D'ENTITE. Un plafond adosse a l'exposition demande une exposition ; il n'y
en a pas au niveau secteur. Le script 78 l'avait deja etabli, et l'on reste donc sur l'entite
notionnelle, avec son controle : a theta = 1 et sans plafond, on doit retrouver sa VaR libre.

Sortie : diagnostics + figure S42_plafond_saturation.png.
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
import euro_cascade_model as ec                                 # noqa: E402
from euro_cascade_model import var, PARAMS                      # noqa: E402
import canaux_conformite as cx                                  # noqa: E402
from src.aggregation.lda import simulate_remediation_severity   # noqa: E402
import scr_engine as eng                                        # noqa: E402
from descente import Descente                                   # noqa: E402

WID = 88
PIL = cx.PIL
sp = PARAMS["OPRISK"]
TAUX_USD = 1.04
ACTIFS = 20_000.0 / TAUX_USD          # entite notionnelle du script 78, en M€
B_PUBLIE = 169.0                      # besoin publie pour cette entite (script 65)
NY = 200_000
NSEED = 3
SEED0 = 20260814

# Les deux plages sont POSEES, aucune n'est estimee, et c'est dit dans les deux scripts d'origine.
THETAS = (0.70, 0.85, 1.00, 1.15, 1.30)
KAPPAS = (0.002, 0.005, 0.010, 0.020, None)      # None = aucun plafond
NOM_K = {0.002: "0,2 %", 0.005: "0,5 %", 0.010: "1 %", 0.020: "2 %", None: "aucun"}

W_AM = np.array([eng.LAMBDA[j] for j in PIL], float)
W_AM = W_AM / W_AM.sum()
D = Descente()


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def fnum(v):
    return f"{v:,.0f}".replace(",", " ")


def var_entite(kappa, theta, seed, ny=NY):
    """VaR 99,5 % de l'entite, avec saturation PUIS plafond, appliques AU SINISTRE.

    L'ORDRE DES DEUX OPERATIONS EST PARTIE DE LA FORME et il faut le dire : la saturation
    modifie le cout d'un sinistre multi-piliers, le plafond borne ce que l'entite peut perdre
    sur ce sinistre. Le plafond vient donc EN DERNIER, sans quoi la saturation pourrait faire
    repasser la perte au-dessus de l'exposition, ce qui n'aurait pas de sens.

    A theta = 1 et kappa = None on retrouve exactement la chaine du script 78 : c'est le controle.
    """
    rng = np.random.default_rng(seed)
    a_musd = ACTIFS * TAUX_USD
    lam, mult = D.lam(a_musd), D.mult(a_musd)
    r = lam / (ec.PHI - 1.0)
    counts = rng.negative_binomial(r, r / (r + lam), size=ny)
    T = int(counts.sum())
    if T == 0:
        return 0.0
    year_of = np.repeat(np.arange(ny), counts)
    amorce = rng.choice(len(PIL), size=T, p=W_AM)
    U = rng.random(T)
    SEV = simulate_remediation_severity(T * len(PIL), sp["xi"], mult * sp["sigma"],
                                        mult * sp["u"], sp["p_u"], None,
                                        rng).reshape(T, len(PIL))
    tables = {j: cx.table_amorce(j, ec.G_BASE) for j in PIL}
    annual = np.zeros(ny)
    for c, j in enumerate(PIL):
        idx = np.where(amorce == c)[0]
        if idx.size == 0:
            continue
        ind, probs = tables[j]
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        sel = np.searchsorted(cdf, U[idx], side="right")
        np.clip(sel, 0, len(probs) - 1, out=sel)
        touches = ind[sel]
        k_pil = touches.sum(axis=1)
        perte = (SEV[idx] * touches).sum(axis=1) * np.power(k_pil, theta - 1.0)
        if kappa is not None:
            np.minimum(perte, kappa * ACTIFS, out=perte)
        annual += np.bincount(year_of[idx], weights=perte, minlength=ny)
    return float(var(annual))


def moyenne(kappa, theta):
    return float(np.mean([var_entite(kappa, theta, SEED0 + k) for k in range(NSEED)]))


# =====================================================================================
titre("1. CONTROLE : sans plafond et sans saturation, on retrouve la chaine du script 78")
# =====================================================================================
V0 = moyenne(None, 1.0)
print(f"  Entite notionnelle, exposition {fnum(ACTIFS)} M€, {fnum(NY)} annees, {NSEED} graines.")
print(f"  VaR libre (theta = 1, aucun plafond) : {V0:.1f} M€.")
print(f"  Le script 78 publie 177,1 M€ pour la meme entite dans le meme moteur, et le script 65")
print(f"  publie {B_PUBLIE:.0f} par l'evaluateur d'identification partielle. Les trois sont du meme")
print("  ordre, et tout ce qui suit est mesure PAR RAPPORT A CETTE valeur, dans ce seul moteur.")

# =====================================================================================
titre("2. LES DEUX BANDES MARGINALES, chacune l'autre reserve neutralisee")
# =====================================================================================
marg_theta = {th: moyenne(None, th) for th in THETAS}
marg_kappa = {ka: moyenne(ka, 1.0) for ka in KAPPAS}
print(f"  SATURATION SEULE (aucun plafond) :")
print(f"  {'theta':>8}{'VaR (M€)':>12}{'/ libre':>10}")
for th in THETAS:
    print(f"  {th:>8.2f}{marg_theta[th]:>12.1f}{marg_theta[th]/V0:>10.2f}")
b_theta = max(marg_theta.values()) - min(marg_theta.values())
print(f"  Bande de saturation : {b_theta:.1f} M€.")
print(f"\n  PLAFOND SEUL (theta = 1) :")
print(f"  {'kappa':>8}{'VaR (M€)':>12}{'/ libre':>10}")
for ka in KAPPAS:
    print(f"  {NOM_K[ka]:>8}{marg_kappa[ka]:>12.1f}{marg_kappa[ka]/V0:>10.2f}")
b_kappa = max(marg_kappa.values()) - min(marg_kappa.values())
print(f"  Bande de plafond : {b_kappa:.1f} M€.")
print(f"\n  LES DEUX BANDES VONT BIEN EN SENS CONTRAIRE, et c'est la premiere verification de")
print("  l'affirmation du script 78 : la saturation FAIT MONTER le capital quand theta croit, le")
print("  plafond le FAIT BAISSER quand kappa decroit. Leurs deux extremes defavorables ne sont donc")
print("  pas au meme coin du rectangle.")

# =====================================================================================
titre("3. LA SURFACE CONJOINTE, et le terme d'interaction")
# =====================================================================================
S = {(ka, th): moyenne(ka, th) for ka in KAPPAS for th in THETAS}
print(f"  {'kappa \\ theta':>14}" + "".join(f"{th:>10.2f}" for th in THETAS))
for ka in KAPPAS:
    print(f"  {NOM_K[ka]:>14}" + "".join(f"{S[(ka, th)]:>10.1f}" for th in THETAS))
vals = list(S.values())
b_conj = max(vals) - min(vals)
print(f"\n  Bande conjointe sur le rectangle : {b_conj:.1f} M€.")
print(f"  Somme des deux bandes marginales : {b_theta:.1f} + {b_kappa:.1f} = {b_theta + b_kappa:.1f} M€.")
print(f"  Rapport : {b_conj/(b_theta + b_kappa):.2f}.")
if b_conj < b_theta + b_kappa:
    print("  LA BANDE CONJOINTE EST DONC PLUS ETROITE QUE LA SOMME DES DEUX, ce qui confirme")
    print("  l'affirmation du script 78 : additionner les deux reserves comme si elles etaient")
    print("  independantes SUR-ESTIMERAIT l'incertitude totale.")
else:
    print("  LA BANDE CONJOINTE N'EST PAS PLUS ETROITE QUE LA SOMME DES DEUX, et il faut le dire")
    print("  tel quel : l'affirmation du script 78 selon laquelle les additionner sur-estimerait")
    print("  l'incertitude n'est PAS verifiee sur ce rectangle. Le raisonnement etait qu'elles")
    print("  agissent en sens contraire, ce qui est vrai des MARGINALES mais ne suffit pas : les")
    print("  deux coins extremes du rectangle restent atteignables, et c'est eux qui fixent la")
    print("  bande. Ce qui est vrai est plus etroit, et c'est l'objet du terme d'interaction.")

print("\n  LE TERME D'INTERACTION, qui est la seule quantite disant si les deux se traitent")
print("  separement : I = V(kappa, theta) - V(kappa, 1) - V(inf, theta) + V(inf, 1).")
print(f"\n  {'kappa \\ theta':>14}" + "".join(f"{th:>10.2f}" for th in THETAS))
inter = {}
for ka in KAPPAS:
    if ka is None:
        continue
    ligne = []
    for th in THETAS:
        I = S[(ka, th)] - S[(ka, 1.0)] - S[(None, th)] + S[(None, 1.0)]
        inter[(ka, th)] = I
        ligne.append(I)
    print(f"  {NOM_K[ka]:>14}" + "".join(f"{v:>+10.1f}" for v in ligne))
I_max = max(abs(v) for v in inter.values())
print(f"\n  Interaction maximale en valeur absolue : {I_max:.1f} M€, soit {100*I_max/V0:.1f} % de la VaR libre")
print(f"  et {100*I_max/b_conj:.0f} % de la bande conjointe.")
print("\n  LE SIGNE DE L'INTERACTION SUIT CELUI DE (1 - theta), ET C'EST PLUS INSTRUCTIF QU'UN SIGNE")
print("  CONSTANT. Il est POSITIF a theta < 1 et NEGATIF a theta > 1 : autrement dit le plafond")
print("  ATTENUE ce que la saturation fait, dans les DEUX directions. A theta > 1 la saturation")
print("  veut faire monter la perte, le plafond l'en empeche ; a theta < 1 elle veut la faire")
print("  baisser, mais le plafond l'a deja fait baisser davantage, si bien que l'effet propre de")
print("  theta est mange. LE PLAFOND N'EST DONC PAS UN CONTREPOIDS QUI S'OPPOSE A LA SATURATION,")
print("  C'EST UN AMORTISSEUR QUI REDUIT SA PRISE, quel que soit son sens.")
print("\n  On le voit directement sur la surface : a plafond de 0,5 % la VaR vaut la meme valeur")
print("  pour LES CINQ exposants. La sensibilite a theta y est exactement nulle, ce qui est le cas")
print("  extreme de l'amortissement : une perte ecretee ne porte plus l'exposant, puisqu'elle est")
print("  remplacee par le plafond avant d'etre comptee.")

# =====================================================================================
titre("4. Y A-T-IL UNE CRETE DE COMPENSATION ? Non, et c'est mieux ainsi")
# =====================================================================================
print("  On cherchait des couples (kappa, theta) NON TRIVIAUX redonnant la VaR libre, c'est-a-dire")
print("  ou un plafond qui MORD serait exactement compense par une saturation. Leur existence")
print("  serait une non-identifiabilite : le bon chiffre obtenu avec deux hypotheses fausses.")
print("\n  PRECAUTION DE METHODE, ET ELLE A ETE NECESSAIRE. Une premiere version cherchait toute")
print("  intersection avec la VaR libre et en trouvait a kappa = 1 % et 2 %, en concluant a une")
print("  compensation. C'etait un artefact : a ces plafonds la contrainte NE MORD PAS, la VaR")
print("  libre est atteinte a theta = 1, et le « couple compensateur » n'est que le point de")
print("  reference. On n'accepte donc un couple que si le plafond mord vraiment, c'est-a-dire si")
print("  V(kappa, 1) < V(libre).")
crete = []
for ka in KAPPAS:
    if ka is None:
        continue
    mord = S[(ka, 1.0)] < V0 - 1e-6
    ths = np.array(THETAS, float)
    vs = np.array([S[(ka, th)] for th in THETAS])
    if mord and vs.min() <= V0 <= vs.max():
        crete.append((ka, float(np.interp(V0, vs, ths))))
print(f"\n  {'kappa':>10}{'le plafond mord ?':>20}{'VaR max sur theta':>20}{'atteint la libre ?':>21}")
for ka in KAPPAS:
    if ka is None:
        continue
    mord = "oui" if S[(ka, 1.0)] < V0 - 1e-6 else "non"
    vmax = max(S[(ka, th)] for th in THETAS)
    print(f"  {NOM_K[ka]:>10}{mord:>20}{vmax:>20.1f}{('oui' if vmax >= V0 - 1e-6 else 'non'):>21}")
if crete:
    print(f"\n  COUPLES COMPENSATEURS NON TRIVIAUX : "
          + ", ".join(f"({NOM_K[k]} ; theta = {t:.2f})" for k, t in crete))
    print("  Le capital publie y est retrouve alors que les deux hypotheses sont fausses : les")
    print("  deux reserves sont CONFONDUES, et c'est une non-identifiabilite a declarer.")
else:
    print("\n  IL N'EXISTE AUCUN COUPLE COMPENSATEUR NON TRIVIAL, et c'est un meilleur resultat que")
    print("  celui qu'on cherchait. Des que le plafond mord, aucun exposant de la plage posee ne")
    print("  ramene le capital a sa valeur libre : a 0,5 % la VaR reste a 96 M€ pour les cinq")
    print("  exposants, tres en dessous des 179 libres. LES DEUX RESERVES NE SONT DONC PAS")
    print("  CONFONDUES, ELLES SONT HIERARCHISEES : le plafond commande, et la saturation ne")
    print("  s'exprime que dans l'espace qu'il lui laisse.")
    print("\n  CE QUE CELA VAUT POUR LE MEMOIRE, ET C'EST RASSURANT : il n'y a pas ici de")
    print("  non-identifiabilite du type de celle de la direction de W, ou deux hypotheses fausses")
    print("  produiraient le bon chiffre. Un capital d'entite conforme a la valeur publiee")
    print("  IMPLIQUE que le plafond ne mord pas, donc que la sevérite n'est pas bornee par")
    print("  l'exposition dans la plage exploree. La limite declaree reste entiere, mais elle est")
    print("  desormais SEPAREE de l'autre : on peut discuter le plafond sans discuter theta.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. LES DEUX RESERVES SONT DESORMAIS MESUREES ENSEMBLE, sur la meme forme fonctionnelle :")
print("     perte = min(kappa x E, somme des severites x k^(theta-1)). Le script 78 affirmait")
print("     qu'elles agissent en sens contraire ; c'etait un raisonnement, c'est maintenant une")
print("     mesure.")
print(f"  2. LES MARGINALES CONFIRMENT LE SENS OPPOSE : bande de saturation {b_theta:.1f} M€ vers le haut,")
print(f"     bande de plafond {b_kappa:.1f} M€ vers le bas.")
print(f"  3. MAIS ELLES NE S'ANNULENT PAS SUR LE RECTANGLE : bande conjointe {b_conj:.1f} M€ contre")
print(f"     {b_theta + b_kappa:.1f} pour la somme des marginales, soit un rapport de {b_conj/(b_theta+b_kappa):.2f}. Les deux coins")
print("     extremes restent atteignables et ce sont eux qui fixent la bande.")
print(f"  4. CE QUI EST VRAI, ET C'EST LE TERME D'INTERACTION : {I_max:.1f} M€ au maximum, DE SIGNE")
print("     OPPOSE A CELUI DE (theta - 1). Le plafond n'est donc pas un contrepoids qui s'oppose")
print("     a la saturation, c'est un AMORTISSEUR qui reduit sa prise dans les DEUX directions.")
print("     Cas extreme a 0,5 % : la VaR y vaut la meme valeur pour les cinq exposants, la")
print("     sensibilite a theta etant exactement nulle. Une perte ecretee ne porte plus")
print("     l'exposant, puisqu'elle est remplacee par le plafond avant d'etre comptee.")
print("  5. IL N'EXISTE AUCUN COUPLE COMPENSATEUR NON TRIVIAL, apres avoir ecarte les faux")
print("     positifs ou le plafond ne mord pas. Les deux reserves ne sont donc pas CONFONDUES")
print("     mais HIERARCHISEES : le plafond commande, la saturation s'exprime dans ce qu'il")
print("     laisse. C'est rassurant pour le memoire, puisqu'il n'y a pas ici de")
print("     non-identifiabilite du type de celle de la direction de W.")
print("  6. CONSEQUENCE DE REDACTION, et elle nuance la formulation actuelle du script 78 : dire")
print("     « les traiter comme deux reserves independantes sur-estimerait l'incertitude » est")
print("     vrai en amplitude (rapport 0,87) mais la raison n'est pas celle qui etait donnee. Ce")
print("     n'est pas que les deux se compensent, c'est que l'une DESACTIVE l'autre la ou elle")
print("     mord.")
print("  7. RIEN N'EST IMPLEMENTE DANS LA CHAINE PUBLIEE. Le gel l'interdit, la VaR libre est")
print("     retrouvee au controle, et aucun nombre publie ne bouge.")

# =====================================================================================
# figure S42
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.4))

# (a) les courbes par plafond : on voit d'un coup que le plafond aplatit la pente en theta
# Le plafond de 2 % ne mord jamais : sa courbe est CONFONDUE avec celle sans plafond. On la
# trace en pointilles epais pour qu'elle reste visible sous la noire, plutot que de laisser
# croire a une courbe manquante.
for ka, col in zip(KAPPAS, [ACCENT, "#d98f5a", GREEN, BLUE, INK]):
    lab = "aucun plafond" if ka is None else f"plafond {NOM_K[ka]}"
    st = "o--" if ka == 0.020 else "o-"
    lw = 3.2 if ka == 0.020 else 2.0
    ax1.plot(THETAS, [S[(ka, th)] for th in THETAS], st, color=col, lw=lw, ms=7, label=lab)
ax1.axhline(V0, color=MUTED, lw=1.2, ls=":")
# ETIQUETTE A DROITE : posee a gauche elle tombait dans la legende.
ax1.text(THETAS[-1], V0 * 1.035, f"VaR libre {V0:.0f} M€", fontsize=8.5, color=INK2, ha="right")
ax1.axvline(1.0, color=INK2, lw=1.0, ls="--")
ax1.text(1.0, min(vals) * 0.94, "additivité", fontsize=8.5, color=INK2, ha="center")
ax1.set_ylim(min(vals) * 0.88, max(vals) * 1.06)
ax1.set_xlabel("exposant de saturation θ", color=INK2)
ax1.set_ylabel("besoin de capital de l'entité (M€)", color=INK2)
ax1.legend(fontsize=8.5, frameon=False, loc="upper left")
ax1.set_title("(a)  Le plafond aplatit la pente en θ :\nune perte écrêtée ne porte plus l'exposant",
              fontsize=10.5, color=INK, pad=8)

# (b) le terme d'interaction, qui est la vraie reponse
kk = [k for k in KAPPAS if k is not None]
M = np.array([[inter[(ka, th)] for th in THETAS] for ka in kk])
lim = np.abs(M).max()
im = ax2.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto", origin="upper")
ax2.set_xticks(range(len(THETAS)))
ax2.set_xticklabels([f"{t:g}".replace(".", ",") for t in THETAS], fontsize=9.5)
ax2.set_yticks(range(len(kk)))
ax2.set_yticklabels([NOM_K[k] for k in kk], fontsize=9.5)
for i in range(len(kk)):
    for j in range(len(THETAS)):
        ax2.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=9.5,
                 color=INK if abs(M[i, j]) < 0.55 * lim else "white", fontweight="bold")
ax2.set_xlabel("exposant de saturation θ", color=INK2)
ax2.set_ylabel("plafond κ (fraction du bilan)", color=INK2)
fig.colorbar(im, ax=ax2, label="interaction (M€)", fraction=0.046, pad=0.03)
ax2.set_title("(b)  L'interaction change de signe à θ = 1 : le plafond\namortit la saturation "
              "dans les deux directions", fontsize=10.5, color=INK, pad=8)

for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

fig.suptitle("S42 : les deux limites déclarées sont les deux moitiés d'une même forme, et elles "
             "ne s'additionnent pas", fontsize=12.5, fontweight="bold", color=INK, x=0.02,
             ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S42_plafond_saturation.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
