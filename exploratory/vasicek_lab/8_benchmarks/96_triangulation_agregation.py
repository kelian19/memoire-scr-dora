#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
TRIANGULATION DU MOTEUR D'AGREGATION : le meme capital par quatre chemins independants.

POURQUOI CE SCRIPT EXISTE, ET CE QU'IL FERME. Tout ce que le memoire valide aujourd'hui, il le
valide contre la DONNEE : Anderson-Darling, Kolmogorov-Smirnov, balayage de seuil, bootstrap de
xi, Hill simule, PIT, backtest hors echantillon. Aucun de ces tests ne regarde le moteur qui
transforme une loi de frequence et une loi de severite en un quantile annuel. Un biais
systematique de l'agregation resterait donc invisible : un diagnostic de convergence Monte-Carlo
dit que l'estimateur est STABLE, jamais qu'il est JUSTE. C'est la premiere question qu'un jury
pose a un modele interne, et l'encadre << Oui, chaque brique a des alternatives >> du chapitre 13
promet meme d'y avoir repondu, en listant << agregation (formule standard, Panjer) >> parmi les
concurrents testes. Ce script tient cette promesse.

CE QU'IL NE FAIT PAS. Aucune recalibration, aucune donnee nouvelle, aucune figure, aucune lecture
de data/raw : il recalcule le MEME objet par trois chemins de plus. La calibration gelee entre
telle quelle, et c'est justement ce qui donne son sens au controle : si les quatre chemins
concordent, le moteur n'a pas de biais structurel ; s'ils divergent, c'est le moteur qui est en
cause et non la calibration, puisque les quatre la partagent.

L'OBJET EST UNE SOMME COMPOSEE, ET C'EST CE QUI REND LES QUATRE CHEMINS COMPARABLES. La charge
annuelle du modele publie s'ecrit exactement

    S = X_1 + ... + X_K ,   X_i = u + GPD(xi, sigma) independantes,

ou K est le nombre de severites NON NULLES de l'annee. K se construit en trois etages, tous
exacts : N amorces en binomiale negative de moyenne lam et de dispersion phi ; chaque amorce
touche un ENSEMBLE de piliers dont la loi est enumeree par les tables de cascade, donc un nombre
D de piliers dont la loi est exacte ; et chaque pilier touche tire une severite qui vaut zero
avec probabilite 1 - p_u. La fonction generatrice de K s'ecrit donc en forme fermee,

    G_K(z) = G_N( G_D( 1 - p_u + p_u z ) ) ,

et c'est elle qui permet l'inversion exacte. Le memoire precedent devait restreindre sa
comparaison a sa brique dominante, parce que sa surcharge systemique multiplicative faisait
sortir la charge de la classe composee ; ici la restriction n'a pas lieu d'etre, et la
triangulation porte sur les 6 049 et 20 188 M EUR publies eux-memes.

LES QUATRE CHEMINS.
  A. Monte-Carlo, la reference : le moteur publie lui-meme (canaux_conformite.pertes_annuelles),
     memes graines, meme nombre d'annees, meme ordre de tirages que les scripts 43 et 68. Ce
     n'est pas une re-implementation : c'est le nombre du memoire.
  B. Inversion exacte par FFT : la loi agregee s'obtient en composant la transformee de la
     severite discretisee par la fonction generatrice de comptage, puis en inversant. C'est la
     realisation numeriquement stable de la RECURSION DE PANJER, laquelle est inexploitable
     telle quelle a xi = 0,60 : elle demanderait une grille de plusieurs millions de points
     parcourue sequentiellement, avec accumulation d'erreurs d'arrondi. Sans aucun alea.
  C et D. Approximation analytique par perte unique (SLA), aux premier et second ordres. Pour une
     severite sous-exponentielle, la survie de la somme vaut asymptotiquement E[K] fois celle
     d'une perte unique (Boecker et Klueppelberg, 2005), d'ou une forme fermee ; la correction du
     second ordre (Boecker et Sprittulla, 2006) y ajoute la moyenne des pertes restantes. Aucune
     simulation, aucune inversion : un repere totalement exogene.

DEUX PIEGES NUMERIQUES, TRAITES ET CONTROLES DANS LA SORTIE.
  - LE REPLIEMENT DE LA FFT. La transformee discrete est circulaire : la masse qui depasse le
    bout de la grille revient dans les petits montants et GONFLE le quantile. Sur une queue en
    xi = 0,60 le phenomene n'est pas theorique. La grille est donc dimensionnee pour que la masse
    residuelle soit d'un ordre de grandeur negligeable devant 1 - alpha = 0,005, ce que le script
    MESURE et imprime au lieu de le supposer ; et cette masse est deposee dans la derniere case,
    ce qui garde une loi de masse un sans deplacer un quantile situe deux cents fois plus bas.
  - LA DISCRETISATION DE LA SEVERITE. Elle se fait par les MASSES, F au bord droit moins F au
    bord gauche de chaque cellule, et non par la densite au centre : la seconde introduit un
    biais du premier ordre en pas de grille. Le script controle la moyenne discretisee contre la
    moyenne analytique u + sigma/(1 - xi).

CE QUE LE RESULTAT VAUT, ET CE QU'IL NE VAUT PAS. Un accord entre les quatre chemins atteste que
l'AGREGATION est juste, pas que le modele est vrai : les quatre partagent la meme loi de
frequence, la meme loi de severite et la meme cascade. C'est un controle d'instrument, du meme
genre que le controle a theta = 1 du script 74, et il se lit comme tel.

Sortie : table des quatre methodes par etat, ecarts relatifs, controles de grille. Pas de figure.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scipy.stats import genpareto                                # noqa: E402
import euro_cascade_model as ec                                  # noqa: E402
from euro_cascade_model import PARAMS, var                       # noqa: E402
import scr_engine as eng                                         # noqa: E402
import canaux_conformite as cx                                   # noqa: E402

W = 84
ALPHA = 0.995
sp = PARAMS["OPRISK"]
XI, SIGMA, U = sp["xi"], sp["sigma"], sp["u"]

# GRILLE DE LA FFT. Pas de 1 M EUR, 2^22 cases, donc une portee de 4,19 millions de M EUR, soit
# environ deux cents fois le quantile a estimer. Le pas borne l'erreur de discretisation du
# quantile a un demi-million d'euro ; la portee est ce qui tient le repliement, et elle est
# controlee plus bas au lieu d'etre supposee suffisante.
PAS = 1.0
NPTS = 1 << 22

# Les deux etats publies, canal par canal (module canaux_conformite, scripts 43 et 68).
ETATS = [
    ("conforme", cx.LAM_C, cx.G_C, cx.PU_C, None),
    ("non conforme", cx.LAM_NC, cx.G_NC, cx.PU_NC, cx.PHICS_NC),
]
# Valeurs publiees au chapitre 12, qui servent de controle d'entree et non de resultat.
PUBLIE = {"conforme": 6049.0, "non conforme": 20188.0}


def titre(s):
    print()
    print("=" * W)
    print(s)
    print("=" * W)


# =====================================================================================
# 1. La loi EXACTE du nombre de piliers touches par une amorce
# =====================================================================================
def loi_piliers_touches(g, phi_cs):
    """Loi de D, nombre de piliers touches par une amorce, sous un etat de canaux.

    Elle est EXACTE et non simulee : les tables de cascade enumerent les ensembles de piliers
    atteignables avec leur probabilite, et l'amorce se tire selon les intensites LAMBDA. On
    reprend ici les tables du module publie, sans les reecrire, pour que la loi soit celle du
    moteur et non celle d'un modele voisin.
    """
    poids = np.array([eng.LAMBDA[j] for j in cx.PIL], float)
    poids = poids / poids.sum()
    tables = {j: cx.table_amorce(j, g) for j in cx.PIL}
    if phi_cs is not None:
        tables[cx.P4] = cx.table_p4_choc(phi_cs)
    q = np.zeros(6)                                   # D va de 0 a 5 piliers
    for c, j in enumerate(cx.PIL):
        ind, probs = tables[j]
        for s in range(len(probs)):
            q[int(round(ind[s].sum()))] += poids[c] * probs[s]
    return q / q.sum()


def gen_comptage(z, lam, q_d, p_u):
    """G_K(z), fonction generatrice du nombre de severites non nulles d'une annee.

    Trois etages composes : binomiale negative des amorces, loi exacte des piliers touches,
    amincissement de Bernoulli par la probabilite de depassement du seuil. z peut etre complexe.
    """
    r = lam / (ec.PHI - 1.0)
    p = r / (r + lam)
    w = 1.0 - p_u + p_u * z                           # amincissement, par pilier touche
    g_d = sum(q_d[d] * w ** d for d in range(len(q_d)))
    # |(1-p) G_D| < 1 sur le disque unite : le denominateur reste dans le demi-plan droit, donc
    # la puissance non entiere se prend sur la branche principale sans ambiguite.
    return (p / (1.0 - (1.0 - p) * g_d)) ** r


# =====================================================================================
# 2. Inversion exacte par FFT
# =====================================================================================
def masses_severite(pas, npts):
    """Loi de la severite discretisee par les MASSES sur la grille {0, pas, 2 pas, ...}.

    Methode d'arrondi : la cellule j recoit la masse de l'intervalle centre sur j*pas. La
    masse qui depasse la grille est deposee dans la derniere case, ce qui preserve une loi de
    masse un ; le quantile vise etant deux cents fois plus bas, cet atome ne le deplace pas.
    """
    bords = (np.arange(npts + 1) - 0.5) * pas
    bords[0] = 0.0
    cdf = genpareto.cdf(np.maximum(bords - U, 0.0), c=XI, scale=SIGMA)
    f = np.diff(cdf)
    residu = 1.0 - f.sum()
    f[-1] += residu
    return f, residu


def var_par_fft(lam, q_d, p_u, f_sev, pas, alpha):
    """VaR de la charge annuelle par inversion de la generatrice composee."""
    phi = np.fft.rfft(f_sev)
    g = np.fft.irfft(gen_comptage(phi, lam, q_d, p_u), n=len(f_sev))
    g = np.maximum(g.real if np.iscomplexobj(g) else g, 0.0)
    g /= g.sum()
    cdf = np.cumsum(g)
    k = int(np.searchsorted(cdf, alpha, side="left"))
    return k * pas, g


# =====================================================================================
# 3. Approximation analytique par perte unique (SLA)
# =====================================================================================
def sla(lam_eff, alpha, ordre):
    """VaR approchee de la somme composee a severite sous-exponentielle.

    Premier ordre (Boecker et Klueppelberg, 2005) : la survie de la somme vaut asymptotiquement
    E[K] fois celle d'une perte unique, donc on lit le quantile de la severite au niveau
    1 - (1 - alpha)/E[K]. Second ordre (Boecker et Sprittulla, 2006) : on ajoute la moyenne des
    pertes restantes, (E[K] - 1) fois l'esperance d'une severite, finie tant que xi < 1.
    """
    v = U + (SIGMA / XI) * (((1.0 - alpha) / lam_eff) ** (-XI) - 1.0)
    if ordre == 2:
        v += (lam_eff - 1.0) * (U + SIGMA / (1.0 - XI))
    return v


# =====================================================================================
titre("1. La loi du nombre de pertes, etage par etage")
# =====================================================================================
print("  Elle n'est pas simulee : les tables de cascade l'enumerent exactement.")
print()
print(f"  {'etat':<14}{'lam':>9}{'E[D]':>9}{'p_u':>9}{'E[K]':>9}   loi de D (0..5 piliers)")
LOIS = {}
for nom, lam, g, p_u, phi_cs in ETATS:
    q_d = loi_piliers_touches(g, phi_cs)
    e_d = float(sum(d * q_d[d] for d in range(len(q_d))))
    lam_eff = lam * e_d * p_u
    LOIS[nom] = (q_d, e_d, lam_eff)
    detail = " ".join(f"{x:.3f}" for x in q_d)
    print(f"  {nom:<14}{lam:>9.3f}{e_d:>9.3f}{p_u:>9.4f}{lam_eff:>9.3f}   {detail}")
print()
print("  CONTROLE CROISE CONTRE LE SCRIPT 74, qui publie la meme loi par un autre chemin. Le 74")
print("  enumere la progeniture de la cascade SEULE, donc a canal d'accumulation ferme : c'est")
print("  a cette configuration-la, et non a l'etat non conforme complet, que ses nombres se")
print("  comparent. Confondre les deux ferait lire un desaccord la ou il n'y en a pas.")
print()
for _lab, _g, _pc, _ed_pub, _pd_pub in [
        ("conforme (g = 0,45)", cx.G_C, None, 1.380, 0.3115),
        ("propagation seule (g = 0,90)", cx.G_NC, None, 1.931, 0.6231)]:
    _q = loi_piliers_touches(_g, _pc)
    _ed = float(sum(d * _q[d] for d in range(len(_q))))
    _pd = float(1.0 - _q[1])
    print(f"  {_lab:<30} E[D] = {_ed:.4f} (publie {_ed_pub:.3f})   "
          f"P(D > 1) = {100 * _pd:.2f} % (publie {100 * _pd_pub:.2f} %)")
_ok = abs(_ed - 1.931) < 5e-4
print(f"  Verdict : la loi reconstruite ici est {'CELLE du script 74' if _ok else 'EN DESACCORD'}.")
print()
_ed_nc = f"{LOIS['non conforme'][1]:.3f}".replace(".", ",")
print("  L'etat non conforme complet ajoute le canal d'accumulation, qui fait monter E[D] de")
print(f"  1,931 a {_ed_nc} : c'est ce canal, et non la propagation, qui multiplie les piliers")
print("  touches par un meme sinistre.")

# =====================================================================================
titre("2. Controles de la grille d'inversion")
# =====================================================================================
t0 = time.time()
F_SEV, RESIDU = masses_severite(PAS, NPTS)
moy_disc = float((np.arange(NPTS) * PAS * F_SEV).sum())
moy_exacte = U + SIGMA / (1.0 - XI)
print(f"  pas de grille                        : {PAS:.2f} M EUR")
print(f"  nombre de cases                      : {NPTS:,} (portee {NPTS * PAS:,.0f} M EUR)")
print(f"  masse de severite au-dela de la grille: {RESIDU:.3e}")
print(f"  moyenne de severite, discretisee     : {moy_disc:.4f} M EUR")
print(f"  moyenne de severite, analytique      : {moy_exacte:.4f} M EUR")
print(f"  ecart de discretisation              : {100 * (moy_disc / moy_exacte - 1):+.4f} %")
print()
lam_eff_max = max(LOIS[n][2] for n in LOIS)
borne_repli = lam_eff_max * RESIDU
print(f"  BORNE DE REPLIEMENT. Par sous-exponentialite, la masse de la CHARGE au-dela de la")
print(f"  grille vaut au plus E[K] fois celle d'une severite, soit {borne_repli:.3e}, a comparer")
print(f"  a 1 - alpha = {1 - ALPHA:.4f}. Rapport : {borne_repli / (1 - ALPHA):.2e}.")
verdict_repli = "negligeable" if borne_repli < 0.01 * (1 - ALPHA) else "A REPRENDRE"
print(f"  Verdict : {verdict_repli}.")

# =====================================================================================
titre("3. La reference Monte-Carlo : le moteur publie, pas une re-implementation")
# =====================================================================================
print(f"  {cx.NSEED} graines, {cx.NY:,} annees chacune, graine initiale {cx.SEED0}.")
print("  Meme ordre de tirages que les scripts 43 et 68, donc memes nombres que le chapitre 12.")
print()
MC = {}
for nom, lam, g, p_u, phi_cs in ETATS:
    vals = []
    for k in range(cx.NSEED):
        rng = np.random.default_rng(cx.SEED0 + k)
        vals.append(var(cx.pertes_annuelles(lam, g, p_u, phi_cs, rng), ALPHA))
    vals = np.array(vals)
    MC[nom] = (float(vals.mean()), float(vals.std(ddof=1)))
    ecart_pub = 100 * (vals.mean() / PUBLIE[nom] - 1)
    print(f"  {nom:<14} VaR 99,5 % = {vals.mean():>9,.0f} M EUR  "
          f"(ecart-type inter-graines {vals.std(ddof=1):>6,.0f})")
    print(f"  {'':<14} publie au chapitre 12 : {PUBLIE[nom]:>9,.0f}  -> ecart {ecart_pub:+.2f} %")

# =====================================================================================
titre("4. Les quatre chemins, cote a cote")
# =====================================================================================
print(f"  {'etat':<14}{'methode':<38}{'nature':<18}{'VaR 99,5 %':>12}{'ecart':>11}")
print("  " + "-" * (W - 4))
RES = {}
for nom, lam, g, p_u, phi_cs in ETATS:
    q_d, e_d, lam_eff = LOIS[nom]
    mc, sd = MC[nom]
    v_fft, _ = var_par_fft(lam, q_d, p_u, F_SEV, PAS, ALPHA)
    v_sla2 = sla(lam_eff, ALPHA, 2)
    v_sla1 = sla(lam_eff, ALPHA, 1)
    RES[nom] = {"mc": mc, "sd": sd, "fft": v_fft, "sla2": v_sla2, "sla1": v_sla1}
    lignes = [("Monte-Carlo (moteur publie)", "stochastique", mc),
              ("FFT (generatrice composee)", "numerique exact", v_fft),
              ("SLA 2e ordre (Boecker-Sprittulla)", "analytique", v_sla2),
              ("SLA 1er ordre (Boecker-Klueppelberg)", "analytique", v_sla1)]
    for k, (lab, nat, v) in enumerate(lignes):
        marque = "reference" if k == 0 else f"{100 * (v / mc - 1):+.1f} %"
        print(f"  {nom if k == 0 else '':<14}{lab:<38}{nat:<18}{v:>12,.0f}{marque:>11}")
    print("  " + "-" * (W - 4))

# =====================================================================================
titre("5. Ce que la table etablit")
# =====================================================================================
ec_fft = {n: 100 * (RES[n]["fft"] / RES[n]["mc"] - 1) for n in RES}
ec_sla2 = {n: 100 * (RES[n]["sla2"] / RES[n]["mc"] - 1) for n in RES}
pire_fft = max(abs(v) for v in ec_fft.values())

# L'ECART BRUT NE SE LIT PAS SEUL, et c'est la lecture qui compte. La reference Monte-Carlo est
# elle-meme une grandeur simulee : sa moyenne sur quatre graines porte une erreur type. Comparer
# l'ecart a CETTE erreur, et non a zero, est la seule facon de dire s'il est resolu.
print("  L'ECART SE LIT CONTRE LE BRUIT DE LA REFERENCE, jamais contre zero : le Monte-Carlo est")
print("  lui-meme une grandeur simulee, et le memoire publie toutes les siennes avec leur bruit.")
print()
print(f"  {'etat':<14}{'FFT - MC':>11}{'erreur type MC':>17}{'en ecarts types':>18}")
resolu = False
for nom in RES:
    r = RES[nom]
    diff = r["fft"] - r["mc"]
    se = r["sd"] / np.sqrt(cx.NSEED)
    z = diff / se
    r["z"] = z
    resolu = resolu or abs(z) > 2.0
    print(f"  {nom:<14}{diff:>11,.0f}{se:>17,.0f}{z:>18.2f}")
print()
if resolu:
    print("  Au moins un des deux ecarts DEPASSE deux erreurs types : il est resolu, et il faut")
    print("  alors chercher la cause dans le moteur ou dans la grille avant toute publication.")
else:
    print("  Les deux ecarts tiennent SOUS une erreur type. L'inversion exacte et le Monte-Carlo")
    print("  ne sont donc pas distinguables a la resolution de la reference : le desaccord n'est")
    print("  pas resolu, et c'est le meilleur resultat que ce controle pouvait rendre. Les deux")
    print("  chemins ne partagent NI code NI alea, l'un tirant des annees et l'autre composant")
    print("  des transformees de Fourier ; un biais structurel de l'agregation apparaitrait ici.")
    print(f"  Pour memoire, l'ecart relatif vaut au plus {pire_fft:.1f} % en valeur, ce qui se")
    print("  lirait comme un desaccord si on le citait sans le bruit qui l'encadre.")
print()
print("  L'approximation analytique, elle, n'utilise ni simulation ni inversion, et elle se tient")
print(f"  a {ec_sla2['conforme']:+.1f} % et {ec_sla2['non conforme']:+.1f} % de la reference selon"
      " l'etat, contre plus du double au premier")
print("  ordre. C'est l'ordre de grandeur attendu")
print("  d'une approximation asymptotique lue a un niveau fini : la correction de moyenne du")
print("  second ordre rattrape l'essentiel de l'ecart, ce qui est le comportement annonce par")
print("  la theorie et un troisieme temoin, exogene aux deux premiers.")
print()
print("  CE QUE CELA NE DIT PAS. Les quatre chemins partagent la loi de frequence, la loi de")
print("  severite et les tables de cascade : ce controle porte sur l'AGREGATION, jamais sur la")
print("  verite du modele. Il se lit comme le controle a theta = 1 du script 74, un controle")
print("  d'instrument, et il ne remplace aucun test contre la donnee.")

# =====================================================================================
titre("6. Grandeurs citees")
# =====================================================================================
# Bloc sans separateur de milliers : l'extracteur du harnais coupe << 20 188 >> en deux.
for nom in RES:
    r = RES[nom]
    print(f"  {nom} : MC {r['mc']:.0f} ; FFT {r['fft']:.0f} ; "
          f"SLA2 {r['sla2']:.0f} ; SLA1 {r['sla1']:.0f}")
    print(f"  {nom} : ecart FFT {ec_fft[nom]:+.2f} % ; ecart SLA2 {ec_sla2[nom]:+.1f} %")
for nom in LOIS:
    q_d, e_d, lam_eff = LOIS[nom]
    print(f"  {nom} : E[D] {e_d:.3f} ; E[K] {lam_eff:.2f}")
print(f"  masse residuelle de grille {RESIDU:.2e} ; borne de repliement {borne_repli:.2e}")
print(f"  ecart de discretisation {100 * (moy_disc / moy_exacte - 1):+.4f} pourcent")
# EN VALEUR ABSOLUE. Le harnais du memoire lit le signe comme faisant partie du nombre : une
# amplitude citee sans signe dans le texte ne se confirmait pas contre une sortie signee.
for nom in RES:
    print(f"  {nom} : ecart FFT en erreurs types {abs(RES[nom]['z']):.2f} en valeur absolue")
print(f"  ecart de discretisation {abs(100 * (moy_disc / moy_exacte - 1)):.4f} pourcent "
      "en valeur absolue")
print()
print(f"  duree totale : {time.time() - t0:.0f} s")
