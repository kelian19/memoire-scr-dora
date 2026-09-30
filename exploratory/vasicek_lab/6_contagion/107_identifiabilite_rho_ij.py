# -*- coding: utf-8 -*-
"""107 - RHO_IJ EST-IL IDENTIFIABLE, ET QUE VAUT LE COMPROMIS RHO_IJ = RHO_J + U_IJ ?

LA PHRASE QUE CE SCRIPT MET A L'EPREUVE est celle de la generalisation A du chapitre de la
cascade : << sur des donnees cyber rares, on ne calibre pas un rho_ij par entite (environ 5N
parametres, non identifiable) : on retient le compromis hierarchique rho_ij = rho_j + u_ij >>.
Elle affirme deux choses, et aucune des deux n'etait mesuree :

  - que le rho_ij libre n'est pas identifiable. C'etait un COMPTE DE PARAMETRES, donc un
    argument de forme : un modele peut avoir plus de parametres que d'observations et rester
    utile, comme il peut en avoir moins et n'etre pas identifie. Ce qui se mesure est
    l'INFORMATION que la donnee porte sur rho_ij, et elle se mesure en annees d'observation ;
  - que le compromis hierarchique s'y substitue. Or rho_ij libre et rho_j + u_ij sont DEUX
    OBJETS : le premier est un parametre par cellule, le second une restriction qui ramene
    chaque cellule vers la moyenne de son pilier. Ecrire l'un pour l'autre suppose que u_ij se
    laisse estimer. C'est la seconde chose que ce script mesure.

CE QUI EST TENU FIXE. Les ancrages sont ceux du pipeline publie et ils ne bougent pas : charge
systemique 0,68, donc rho = 0,4624 (la correlation de 0,462 publiee au chapitre des etats de
conformite), et ancrage marginal P(non conforme) = 0,35, soit le seuil K du script 69. Le
script ne recalibre rien, n'ecrit pas config.py et ne deplace aucune valeur publiee : c'est un
diagnostic, au meme titre que les scripts 89 a 94 et 106.

CONVENTION DE SIGNE. Le chapitre de la cascade ecrit X = stress et l'incident au DEPASSEMENT,
X_ij >= K. Le script 69 ecrit la latente en sante, C* <= Phi^-1(0,35). Les deux lois sont
identiques par X = -C*, et le controle d'identite le verifie sur la marge.

AUCUN TIRAGE ALEATOIRE, AUCUNE SIMULATION. Tout est calcule en loi : la loi exacte du nombre
d'entites non conformes s'obtient par quadrature sur le facteur, et la limite de population
admet une forme fermee (la transformee probit de la part non conforme est GAUSSIENNE, de
moyenne -K/sqrt(1-rho) et de variance rho/(1-rho)). Le resultat ne depend donc d'aucune graine.

CE QUE << IDENTIFIABLE >> VEUT DIRE ICI, ET C'EST LA SEULE CONVENTION DU SCRIPT. On demande le
nombre de periodes d'observation qu'il faudrait pour separer deux valeurs de la sensibilite par
un test du rapport de vraisemblance a 5 %, avec 80 % de puissance. C'est la convention du
script 91, qui chiffre en annees la testabilite du quantile a 99,5 %. La non-centralite se
lit 2 T KL, approximation locale : les comptes donnent un ORDRE DE GRANDEUR, pas une decimale.

AUCUNE FIGURE. Trois tables courtes, et le memoire doit tenir sous 200 pages.
"""

import math
import os
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp
from scipy.stats import chi2, ncx2, norm

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (REPO, LAB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WID = 88

# --- ancrages du pipeline publie -------------------------------------------------------
# Charge systemique de la latente de conformite, script 69 ligne 75 (et script 16). La part
# du systemique au sens de Vasicek est son CARRE : c'est le rho du chapitre de la cascade.
GAMMA_LATENTE = 0.68
RHO_PUB = GAMMA_LATENTE ** 2          # 0,4624 ; publie arrondi a 0,462 au chapitre des etats
P_NC = 0.35                           # ancrage marginal, script 69
K_STRESS = norm.ppf(1.0 - P_NC)       # seuil en convention de stress : NC = {X >= K}
N_ENTITES = 4                         # les quatre bilans SFCR du script 65

# Ecart de sensibilite a detecter, en points de rho. C'est l'amplitude de u_ij.
DELTA_REF = 0.10
DELTAS = (0.05, 0.10, 0.20, 0.30)
TAILLES = (4, 20, 100, 1000)

NIV_TEST = 0.05
PUISSANCE = 0.80

# Quadrature de Gauss-Legendre sur le facteur, ramenee a [-12 ; 12]. Preferee a Gauss-Hermite
# parce que l'integrande devient tres pique quand le nombre d'entites monte, et qu'on controle
# ici la precision par deux identites exactes (somme a un, marge invariante).
_NQ = 3000
_xg, _wg = np.polynomial.legendre.leggauss(_NQ)
Y_NODES = 12.0 * _xg
Y_LOGW = np.log(_wg * 12.0) + norm.logpdf(Y_NODES)


def titre(t):
    print()
    print("=" * WID)
    print(t)
    print("=" * WID)


def log_p_nc(y, rho):
    """log P(non conforme | facteur = y) et son complement, en convention de stress."""
    if rho <= 0.0:
        z = np.full_like(np.asarray(y, dtype=float), -K_STRESS)
    else:
        z = (math.sqrt(rho) * np.asarray(y, dtype=float) - K_STRESS) / math.sqrt(1.0 - rho)
    return norm.logcdf(z), norm.logsf(z)


def loi_compte(n, rho):
    """Loi EXACTE du nombre d'entites non conformes d'un pilier, sur une periode.

    Les n entites d'un pilier partagent le facteur Y_j : c'est la generalisation A, ou le
    groupe est le PILIER et non l'entite. La loi s'obtient en integrant le binomial conditionnel
    contre la loi du facteur. Rendue en logarithmes, sans quoi n = 1000 deborde.
    """
    ks = np.arange(n + 1)
    lp, lq = log_p_nc(Y_NODES, rho)
    lcomb = gammaln(n + 1) - gammaln(ks + 1) - gammaln(n - ks + 1)
    # matrice (k, noeud) : le produit conditionnel, pondere par la loi du facteur
    m = (Y_LOGW[None, :] + ks[:, None] * lp[None, :] + (n - ks)[:, None] * lq[None, :])
    return lcomb + logsumexp(m, axis=1)


def kl_compte(n, rho0, rho1):
    """Divergence de Kullback-Leibler par periode, sur le nombre d'entites non conformes."""
    l0, l1 = loi_compte(n, rho0), loi_compte(n, rho1)
    p0 = np.exp(l0)
    return float(np.sum(p0 * (l0 - l1)))


def kl_bernoulli_oracle(rho0, rho1):
    """KL par periode sur UNE entite, le facteur de son pilier etant SUPPOSE CONNU.

    C'est le cas le plus favorable imaginable a l'identification de rho_ij : il faudrait une
    infinite d'entites dans le pilier pour reconstituer Y_j exactement. Le compte d'annees
    qui en sort est donc une BORNE BASSE de ce qu'il faudrait reellement.
    """
    lp0, lq0 = log_p_nc(Y_NODES, rho0)
    lp1, lq1 = log_p_nc(Y_NODES, rho1)
    integrande = np.exp(lp0) * (lp0 - lp1) + np.exp(lq0) * (lq0 - lq1)
    return float(np.sum(np.exp(Y_LOGW) * integrande))


def kl_population(rho0, rho1, seuil_connu=True):
    """KL par periode dans la limite de population, en forme fermee.

    Quand le pilier compte une infinite d'entites, la part non conforme observee vaut
    exactement Phi((sqrt(rho) Y - K)/sqrt(1-rho)). Sa transformee probit est donc GAUSSIENNE,
    de moyenne -K/sqrt(1-rho) et de variance rho/(1-rho) : une periode d'observation, si large
    que soit la population, vaut UN tirage gaussien. Si le seuil K est lui aussi estime, la
    moyenne s'ajuste librement et il ne reste que l'ecart de variance.
    """
    v0, v1 = rho0 / (1.0 - rho0), rho1 / (1.0 - rho1)
    d = 0.5 * (math.log(v1 / v0) + v0 / v1 - 1.0)
    if seuil_connu:
        m0 = -K_STRESS / math.sqrt(1.0 - rho0)
        m1 = -K_STRESS / math.sqrt(1.0 - rho1)
        d += 0.5 * (m0 - m1) ** 2 / v1
    return d


def lam_puissance(niv=NIV_TEST, puiss=PUISSANCE, ddl=1):
    """Non-centralite du chi-deux qui donne la puissance demandee. Calculee, pas recitee."""
    crit = chi2.ppf(1.0 - niv, ddl)
    return brentq(lambda lam: ncx2.sf(crit, ddl, lam) - puiss, 1e-9, 1e6, xtol=1e-10)


LAM_80 = lam_puissance()
CRIT_95 = chi2.ppf(1.0 - NIV_TEST, 1)


def periodes(kl):
    """Nombre de periodes pour atteindre la puissance demandee, a KL par periode donne."""
    if kl <= 0.0:
        return float("inf")
    return LAM_80 / (2.0 * kl)


def fmt_T(t):
    if not np.isfinite(t):
        return "infini"
    if t >= 1e6:
        return f"{t:.2e}"
    return f"{t:,.0f}".replace(",", " ")


# ---------------------------------------------------------------------------------------
# 0. Controle d'identite, et ARRET DUR
# ---------------------------------------------------------------------------------------

titre("107 - IDENTIFIABILITE DE RHO_IJ ET STATUT DU COMPROMIS HIERARCHIQUE")

print("  Modele teste (generalisation A, chapitre de la cascade) :")
print("    X_ij = sqrt(rho_ij) Y_j + sqrt(1 - rho_ij) eps_ij,  non conforme si X_ij >= K.")
print("  Le facteur Y_j est celui du PILIER : dans cette generalisation le groupe qui partage")
print("  un alea commun est le pilier, pas l'entite. C'est ce qui commande tout ce qui suit.")
print()
print(f"  charge systemique gamma            : {GAMMA_LATENTE:.4f}   (script 69, script 16)")
print(f"  part du systemique rho = gamma^2   : {RHO_PUB:.4f}   (publie : 0.462)")
print(f"  ancrage marginal P(non conforme)   : {P_NC:.4f}")
print(f"  seuil en convention de stress K    : {K_STRESS:+.4f}   (= -Phi^-1(0.35) du script 69)")
print(f"  entites du perimetre reel          : {N_ENTITES}        (bilans SFCR, script 65)")

ecart_pub = abs(RHO_PUB - 0.462)
l4 = loi_compte(N_ENTITES, RHO_PUB)
somme4 = float(np.sum(np.exp(l4)))
marge4 = float(np.sum(np.arange(N_ENTITES + 1) * np.exp(l4))) / N_ENTITES
# la marge ne doit dependre d'AUCUN rho : c'est le mecanisme, autant que le controle
marges = {r: float(np.sum(np.arange(N_ENTITES + 1) * np.exp(loi_compte(N_ENTITES, r))))
          / N_ENTITES for r in (0.0, 0.20, RHO_PUB, 0.80, 0.95)}
ecart_marge = max(abs(v - P_NC) for v in marges.values())

print()
print("  Controles :")
print(f"    ecart a la correlation publiee   : {ecart_pub:.6f}   (tolere 0.001)")
print(f"    somme de la loi du compte a un   : {abs(somme4 - 1.0):.2e}   (tolere 1e-9)")
print(f"    marge recalculee a rho publie    : {marge4:.9f}")
print(f"    ecart max de la marge sur rho    : {ecart_marge:.2e}   (tolere 1e-9)")

_stop = []
if ecart_pub > 1e-3:
    _stop.append(f"rho = gamma^2 = {RHO_PUB:.4f} ne redonne plus la correlation publiee 0.462")
if abs(somme4 - 1.0) > 1e-9:
    _stop.append(f"la loi du compte ne somme pas a un (ecart {abs(somme4 - 1.0):.2e}) : quadrature")
if ecart_marge > 1e-9:
    _stop.append(f"la marge depend de rho (ecart {ecart_marge:.2e}) : quadrature insuffisante")
if _stop:
    print()
    print("  ARRET DUR. Le script refuse de produire un resultat :")
    for m in _stop:
        print("    - " + m)
    sys.exit(1)
print("    => controles passes, le script continue.")

print()
print(f"  Convention du test : rapport de vraisemblance a {NIV_TEST:.0%}, puissance "
      f"{PUISSANCE:.0%}, 1 ddl.")
print(f"    valeur critique {CRIT_95:.4f}, non-centralite requise {LAM_80:.4f} (calculee).")

# ---------------------------------------------------------------------------------------
# 1. Le compte de parametres, et pourquoi il ne suffit pas
# ---------------------------------------------------------------------------------------

titre("1. CE QUE LE COMPTE DE PARAMETRES DIT, ET CE QU'IL NE DIT PAS")

for n in TAILLES:
    obs = 5 * n
    libres = 5 * n + 5
    print(f"  {n:5d} entites : {obs:6d} bits observes par periode, "
          f"{libres:6d} parametres libres (5N sensibilites + 5 seuils)")

print()
print("  Le compte est donc perdant a toute taille, ce que le memoire ecrivait deja. Mais il")
print("  est perdant PAR CONSTRUCTION, quelle que soit la duree d'observation : ajouter des")
print("  periodes ajoute des bits sans ajouter de parametres, donc le compte finit par passer")
print("  alors que le probleme, lui, demeure. Le compte n'est pas le bon instrument.")

# ---------------------------------------------------------------------------------------
# 2. Le mecanisme : la marge ne porte aucune information sur rho
# ---------------------------------------------------------------------------------------

titre("2. LE MECANISME : LA MARGE EST AVEUGLE A LA SENSIBILITE")

print("  P(entite non conforme) recalcule a plusieurs sensibilites, tout le reste fixe :")
print()
print("    rho       P(non conforme)")
for r, v in marges.items():
    print(f"    {r:5.4f}    {v:.9f}")
print()
print("  C'est une identite, pas une coincidence : integrer le facteur redonne la marge quel")
print("  que soit son poids. Une sensibilite ne se lit donc JAMAIS sur un taux de conformite,")
print("  elle ne se lit que sur la CO-VARIATION entre cellules qui partagent un facteur. D'ou")
print("  la question qui suit : combien de fois faut-il voir le systeme pour la lire ?")

# ---------------------------------------------------------------------------------------
# 3. rho_ij d'une entite : ce que vaut une cellule
# ---------------------------------------------------------------------------------------

titre("3. RHO_IJ D'UNE ENTITE, FACTEUR SUPPOSE CONNU (CAS LE PLUS FAVORABLE)")

print("  Une entite, un pilier : une cellule rend UN BIT par periode. On accorde en plus au")
print("  test le facteur Y_j exactement connu, ce qu'aucune donnee ne donne. Les annees ci-")
print("  dessous sont donc une BORNE BASSE.")
print()
print("    u_ij     rho_ij     KL par periode    periodes pour 80 % de puissance")
lignes_cell = []
for d in DELTAS:
    r1 = RHO_PUB + d
    kl = kl_bernoulli_oracle(r1, RHO_PUB)
    T = periodes(kl)
    lignes_cell.append((d, r1, kl, T))
    print(f"    {d:+.2f}     {r1:.4f}     {kl:.3e}         {fmt_T(T):>12s}")

# et le sens inverse : que separerait UNE seule observation transversale ?
def sep_une_periode(fn_kl, borne_haute):
    """Plus petit ecart de sensibilite separable en UNE periode, s'il existe."""
    f = lambda d: 2.0 * fn_kl(RHO_PUB + d, RHO_PUB) - CRIT_95
    if f(borne_haute) < 0:
        return None
    return brentq(f, 1e-6, borne_haute, xtol=1e-9)

BORNE_U = 0.99 - RHO_PUB       # u_ij ne peut pas depasser 1 - rho_j : rho reste une part
d_cell = sep_une_periode(kl_bernoulli_oracle, BORNE_U)
kl_max_cell = kl_bernoulli_oracle(RHO_PUB + BORNE_U, RHO_PUB)
print()
print(f"  Lecture inverse, sur UNE periode : l'ecart maximal admissible u_ij = {BORNE_U:+.4f}")
print(f"  (rho_ij = 0.99, le bord du domaine) ne porte que 2 KL = {2 * kl_max_cell:.4f}, contre")
print(f"  {CRIT_95:.4f} requis.")
if d_cell is None:
    print("  AUCUNE valeur de u_ij dans le domaine admissible n'est separable de zero sur une")
    print("  seule observation transversale. Le rho_ij libre n'est donc pas estime avec peu de")
    print("  precision : il n'est pas estime du tout.")
else:
    print(f"  Le plus petit ecart separable vaut u_ij = {d_cell:+.4f}.")

# ---------------------------------------------------------------------------------------
# 4. rho_j d'un pilier : ce que la mise en commun rachete
# ---------------------------------------------------------------------------------------

titre("4. RHO_J D'UN PILIER, ET CE QUE LA MISE EN COMMUN RACHETE")

print(f"  Meme ecart a detecter, u = {DELTA_REF:+.2f}, mais porte par TOUTES les entites du")
print("  pilier a la fois. Le pilier est le groupe : ses entites partagent le facteur Y_j.")
print()
print("    entites du pilier    KL par periode    periodes pour 80 % de puissance")
lignes_pil = []
for n in TAILLES:
    kl = kl_compte(n, RHO_PUB + DELTA_REF, RHO_PUB)
    T = periodes(kl)
    lignes_pil.append((n, kl, T))
    print(f"    {n:9d}            {kl:.3e}         {fmt_T(T):>12s}")

kl_pop = kl_population(RHO_PUB + DELTA_REF, RHO_PUB, seuil_connu=True)
kl_pop_prof = kl_population(RHO_PUB + DELTA_REF, RHO_PUB, seuil_connu=False)
T_pop, T_pop_prof = periodes(kl_pop), periodes(kl_pop_prof)
print(f"    {'population':>9s}            {kl_pop:.3e}         {fmt_T(T_pop):>12s}"
      "   (forme fermee)")
print(f"    {'idem, seuil estime':>9s}    {kl_pop_prof:.3e}         {fmt_T(T_pop_prof):>12s}")
print()
kl_gd = kl_compte(TAILLES[-1], RHO_PUB + DELTA_REF, RHO_PUB)
print(f"  Controle de la forme fermee : la quadrature a {TAILLES[-1]} entites donne "
      f"{kl_gd:.4f},")
print(f"  la limite de population {kl_pop:.4f}, soit {100 * kl_gd / kl_pop:.1f} % de la limite.")
print()
print("  LA SATURATION EST LE RESULTAT. Elargir le pilier ne rachete que jusqu'a un plafond,")
print("  parce qu'une periode ne livre qu'UN tirage du facteur, si large que soit la")
print("  population. Ce qui identifie une sensibilite n'est donc pas le nombre d'entites,")
print("  c'est la REPETITION DANS LE TEMPS. C'est le meme enonce que celui du script 03 sur")
print("  l'asymetrie de W, atteint ici par un autre chemin.")

# ---------------------------------------------------------------------------------------
# 4bis. L'autre sens de regroupement, celui du modele publie
# ---------------------------------------------------------------------------------------

titre("4bis. CE QUI CHANGE QUAND LE GROUPE EST L'ENTITE ET NON LE PILIER")

print("  Le modele publie au chapitre des etats de conformite ne regroupe pas dans le meme")
print("  sens : sa latente est C*_j = gamma Theta_i + sqrt(1 - gamma^2) eps_ij, ou Theta_i est")
print("  propre a l'ENTITE et partage par ses cinq piliers. Une entite y forme donc un groupe")
print("  de cinq, et deux entites sont independantes. La consequence est tranchee :")
print()
kl_entite5 = kl_compte(5, RHO_PUB + DELTA_REF, RHO_PUB)
n_star = LAM_80 / (2.0 * kl_entite5)
print(f"    KL par entite et par periode (groupe de 5 piliers) : {kl_entite5:.3e}")
print(f"    entites necessaires sur UNE seule periode          : {fmt_T(n_star)}")
print()
print("  Sous ce regroupement, la sensibilite s'identifie donc EN COUPE, par le nombre")
print("  d'entites, et l'ordre de grandeur est celui d'un marche national. Sous la")
print("  generalisation A, elle ne s'identifie QUE par le temps. Le sens du regroupement")
print("  n'est pas un detail d'ecriture : il decide de ce qui serait observable.")

# ---------------------------------------------------------------------------------------
# 5. Le compromis hierarchique est-il le meme objet ?
# ---------------------------------------------------------------------------------------

titre("5. RHO_IJ LIBRE ET RHO_J + U_IJ NE SONT PAS LE MEME OBJET")

T_cell_ref = periodes(kl_bernoulli_oracle(RHO_PUB + DELTA_REF, RHO_PUB))
T_pil_ref = periodes(kl_compte(N_ENTITES, RHO_PUB + DELTA_REF, RHO_PUB))
rapport = T_cell_ref / T_pil_ref
print(f"  Pour le meme ecart u = {DELTA_REF:+.2f}, et en accordant a la cellule le facteur")
print(f"  connu, il faut {fmt_T(T_cell_ref)} periodes au niveau de l'entite contre "
      f"{fmt_T(T_pil_ref)} au niveau")
print(f"  du pilier a {N_ENTITES} entites, soit un rapport de {rapport:.1f}.")
print()
print("  Le terme u_ij du compromis n'est donc pas un parametre estime avec peu de precision :")
print("  c'est un parametre que la donnee laisse a sa valeur a priori, zero. Sur tout panel")
print("  realiste, rho_j + u_ij se CONFOND numeriquement avec rho_j, et le compromis")
print("  hierarchique n'est pas une approximation du rho_ij libre : c'en est la seule version")
print("  que la donnee soutienne. Ecrire l'un pour l'autre n'est donc pas une commodite, c'est")
print("  ce que l'information disponible impose.")

# --- et la forme du terme aleatoire, qui est un defaut d'ecriture ---
print()
print("  UNE ECRITURE A SURVEILLER, ET LE POINT EST DE NATURE AVANT D'ETRE DE DEGRE. Une")
print("  sensibilite est une PART : rho_ij doit rester dans [0 ; 1], sans quoi sqrt(1 - rho_ij)")
print("  n'existe pas. Un terme additif gaussien en sort avec probabilite strictement positive,")
print("  quelle que soit sa dispersion, et le degre se lit sur cette dispersion :")
print()
print("    ecart-type de u_ij    P(rho_ij hors de [0 ; 1])")
for tau in (0.05, 0.10, 0.20, 0.30):
    p_out = norm.cdf(-RHO_PUB / tau) + norm.sf((1.0 - RHO_PUB) / tau)
    print(f"    {tau:14.2f}        {p_out:.3e}")
print()
print("  La variante qui ferme le point porte le terme aleatoire sur une echelle transformee,")
print("  logit(rho_ij) = logit(rho_j) + u_ij, admissible par construction. LE MEMOIRE GARDE")
print("  L'ECRITURE ADDITIVE ET PUBLIE LA VARIANTE A COTE : le pipeline posant une sensibilite")
print("  unique, le choix entre les deux ne deplace aucun chiffre publie, et le declarer vaut")
print("  mieux que le corriger en silence.")

# ---------------------------------------------------------------------------------------
# 6. Grandeurs citees
# ---------------------------------------------------------------------------------------

titre("6. GRANDEURS CITEES PAR LE MEMOIRE (sans separateur de milliers)")

print(f"  rho publie (gamma au carre)                        : {RHO_PUB:.4f}")
print(f"  ancrage marginal, invariant a rho                  : {P_NC:.4f}")
print(f"  ecart de sensibilite de reference u                : {DELTA_REF:.2f}")
print(f"  2 KL au bord du domaine, une cellule, une periode  : {2 * kl_max_cell:.4f}")
print(f"  valeur critique du test a 5 pourcent               : {CRIT_95:.4f}")
print(f"  non-centralite pour 80 pourcent de puissance       : {LAM_80:.4f}")
print(f"  periodes, une entite, facteur connu, u = 0.10      : {T_cell_ref:.0f}")
print(f"  periodes, un pilier de 4 entites, u = 0.10         : {T_pil_ref:.0f}")
print(f"  periodes, un pilier de 100 entites, u = 0.10       : "
      f"{periodes(kl_compte(100, RHO_PUB + DELTA_REF, RHO_PUB)):.0f}")
print(f"  periodes, population entiere, u = 0.10             : {T_pop:.0f}")
print(f"  periodes, population entiere, seuil estime         : {T_pop_prof:.0f}")
print(f"  rapport entite sur pilier de 4 entites             : {rapport:.1f}")
print(f"  part de la limite atteinte a 1000 entites          : {100 * kl_gd / kl_pop:.1f}")
print(f"  entites pour identifier en coupe, groupe = entite  : {n_star:.0f}")
_p_out20 = norm.cdf(-RHO_PUB / 0.20) + norm.sf((1.0 - RHO_PUB) / 0.20)
print(f"  P(rho hors de 0 1), u additif d'ecart-type 0.20    : {_p_out20:.4f}")
print(f"  la meme, en pourcent                               : {100 * _p_out20:.2f}")

print()
print("=" * WID)
print("FIN 107")
print("=" * WID)
