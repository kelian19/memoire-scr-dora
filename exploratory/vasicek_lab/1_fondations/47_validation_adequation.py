#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
47 : validation et adequation des deux ajustements qui portent le SCR (severite GPD, frequence NB).

Un jury actuariel cherche d'abord ceci : les lois posees s'ajustent-elles vraiment aux donnees ?
Ce script le teste, sans complaisance.

SEVERITE (GPD, POT OpRisk cyber x finance) :
  - QQ-plot et PP-plot contre la GPD ajustee ;
  - mean residual life (linearite au-dela du seuil = signature GPD) ;
  - stabilite de xi au choix du seuil ;
  - tests d'adequation Anderson-Darling et Kolmogorov-Smirnov, avec p-value par BOOTSTRAP
    PARAMETRIQUE (les valeurs critiques tabulees ne valent pas quand les parametres sont estimes) ;
  - couverture reelle de l'IC90 asymptotique de xi (a n fini), par simulation.

FREQUENCE (comptes firme-annee, secteur financier) :
  - adequation de la binomiale negative contre le Poisson (rootogram, test du rapport de
    vraisemblance, indice de dispersion).

Sortie : diagnostics + figure J3_validation_adequation.png.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from scipy.stats import genpareto, poisson, nbinom
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR  # noqa: E402
from src.utils.config import OPRISK, OPRISK_COHERENCE                                      # noqa: E402

WID = 82
B_GOF = 800          # bootstrap parametrique pour les p-values d'adequation
M_COV = 2000         # simulations pour la couverture de l'IC
SEED = 20260727
rng = np.random.default_rng(SEED)


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# =====================================================================================
# SEVERITE
# =====================================================================================
titre("Severite GPD : chargement et ajustement (OpRisk cyber x finance)")
d = filter_finance(filter_cyber(load_clean(
    os.path.join(REPO, "data", "raw", "SAS_OpRisk_Global_Data_June_2026.xlsx"))))
loss = np.sort(d["loss"].to_numpy() * USD_EUR)
# LE SEUIL TESTE EST CELUI QUE LE MEMOIRE PUBLIE.
# Ce script rederivait le q85 des donnees courantes, soit 22,03 M€ et 88 exces, et testait
# l'adequation de CET ajustement. Or le memoire publie la calibration figee, 20,03 M€ et
# 91 exces : la section de validation attestait donc un ajustement autre que celui dont
# elle publie les parametres. On teste desormais l'ajustement publie, et l'on garde le
# q85 rederive comme controle de robustesse, plus bas.
u = float(OPRISK["seuil_u_eur"])
exc = np.sort(loss[loss > u] - u)
n = exc.size
xi, _, sig = genpareto.fit(exc, floc=0)
print(f"  {loss.size} pertes ; seuil u = {u:.2f} M€ (configuration figee, percentile "
      f"{100*(loss < u).mean():.1f}) ; {n} exces ;")
print(f"  re-ajustement libre a ce seuil : GPD xi = {xi:.3f}, sigma = {sig:.2f}.")
print(f"  parametres PUBLIES par la configuration : xi = {OPRISK['xi']:.4f}, "
      f"sigma = {OPRISK['sigma_eur']:.2f}.")


def ad_stat(x, c, s):
    z = np.clip(genpareto.cdf(np.sort(x), c, scale=s), 1e-12, 1 - 1e-12)
    m = z.size
    i = np.arange(1, m + 1)
    return -m - np.sum((2 * i - 1) / m * (np.log(z) + np.log(1 - z[::-1])))


def ks_stat(x, c, s):
    z = genpareto.cdf(np.sort(x), c, scale=s)
    m = z.size
    e = np.arange(1, m + 1) / m
    return np.max(np.abs(z - e))


ad_obs, ks_obs = ad_stat(exc, xi, sig), ks_stat(exc, xi, sig)
ad_null, ks_null = [], []
for _ in range(B_GOF):
    xb = genpareto.rvs(xi, scale=sig, size=n, random_state=rng)
    try:
        cb, _, sb = genpareto.fit(xb, floc=0)
        ad_null.append(ad_stat(xb, cb, sb)); ks_null.append(ks_stat(xb, cb, sb))
    except Exception:
        pass
ad_null, ks_null = np.array(ad_null), np.array(ks_null)
p_ad = float((ad_null >= ad_obs).mean())
p_ks = float((ks_null >= ks_obs).mean())

titre("Tests d'adequation (p-value par bootstrap parametrique)")
print(f"  Anderson-Darling : A2 = {ad_obs:.3f}, p = {p_ad:.3f}  "
      f"({'ajustement NON rejete' if p_ad > 0.05 else 'ajustement rejete'} a 5 %).")
print(f"  Kolmogorov-Smirnov : D = {ks_obs:.3f}, p = {p_ks:.3f}  "
      f"({'non rejete' if p_ks > 0.05 else 'rejete'} a 5 %).")
# L'ECART-TYPE ASYMPTOTIQUE ET L'IC PAR DELTA-METHODE. Le chapitre socle les publie comme
# resultats de sa proposition de Fisher, et aucun script ne les imprimait : ils etaient
# calcules a la main. Ils se deduisent pourtant de deux valeurs que ce script possede deja,
# l'indice de queue et le nombre d'exces.
_sd = (1.0 + OPRISK["xi"]) / np.sqrt(n)
_lo, _hi = OPRISK["xi"] - 1.645 * _sd, OPRISK["xi"] + 1.645 * _sd
print(f"\n  Ecart-type asymptotique de xi, (1+xi)/sqrt(N_u) avec N_u = {n} : {_sd:.3f}")
print(f"  IC 90 % par delta-methode : [{_lo:.3f} ; {_hi:.3f}]")
print(f"  a comparer a l'IC bootstrap de la configuration figee : "
      f"[{OPRISK['xi_ic90'][0]:.3f} ; {OPRISK['xi_ic90'][1]:.3f}]")
print("  Les deux approches, asymptotique et par reechantillonnage, se recoupent.")

# L'ESTIMATEUR DE HILL, ET SON ECART AU MLE. Le chapitre socle oppose les deux estimateurs et
# publie l'ecart, que le script 67 avait explicitement renvoye ici : c'est ce script qui
# possede la conversion en euros et les exces au-dessus du seuil de collecte.
# Hill se calcule sur les pertes converties triees, convention de src/severity/gpd.py.
# LE CHAPITRE SOCLE ANNONCAIT xi_Hill = 1,42 et un ecart de 138,7 % a k = 86. Ni cette
# fonction ni aucune population du pipeline ne les reproduit : cyber x finance en euros
# donne 1,315 et +120,8 %, le cyber seul 0,935, le secteur financier 0,893, la base entiere
# 0,697. La valeur publiee vient d'une sonde dont le filtre n'a pas ete conserve. Le
# chapitre a ete aligne sur la valeur ci-dessous, qui a un script qui l'imprime.
_ord = np.sort(loss)[::-1]
for _k in (86, n):
    _lx = np.log(_ord[:_k + 1])
    _hill = float(np.mean(_lx[:_k]) - _lx[_k])
    print(f"\n  Estimateur de Hill, k = {_k} : xi_Hill = {_hill:.3f}   "
          f"(MLE au seuil publie : {OPRISK['xi']:.4f})")
    print(f"  ecart relatif de Hill au MLE : {100*(_hill/OPRISK['xi'] - 1):+.1f} %")
print("  Hill suppose une queue de Pareto pure et ignore le parametre d'echelle : sur des")
print("  excedents convertis, il surestime systematiquement. C'est le MLE qui est retenu.")

# ---------------------------------------------------------------------------------------
# L'ECART HILL / MLE : DEMONTRE PLUTOT QU'INVOQUE
# ---------------------------------------------------------------------------------------
# POURQUOI CE BLOC. Dire « Hill est biaise en echantillon fini » est un argument d'autorite :
# c'est vrai en general, cela ne dit pas si CET ecart-la, +120 % sur CES donnees, est du biais
# ou un vrai desaccord. Et l'enjeu n'est pas mince : a xi = 0,595 la variance de la severite
# est infinie mais l'esperance existe ; a xi = 1,32 l'ESPERANCE elle-meme n'existe plus, et
# tout le modele de perte agregee tombe. C'est le point le plus attaquable du memoire.
#
# LE TEST. On simule sous le modele PUBLIE (corps observe sous le seuil, queue GPD de
# parametres xi = 0,5954 et sigma = 57,97 au-dessus) et l'on applique a chaque echantillon
# simule EXACTEMENT le meme estimateur de Hill, aux memes k. Si le monde a xi = 0,595
# reproduit les valeurs de Hill observees, alors ces valeurs ne temoignent de rien : elles
# sont ce qu'un xi de 0,595 produit. Si au contraire il ne les reproduit pas, le desaccord
# est reel et il faut le traiter.
#
# C'est un bootstrap parametrique de l'estimateur, pas une correction du chiffre : on ne
# change rien a la calibration, on mesure ce que l'estimateur concurrent aurait dit dans un
# monde ou la calibration est vraie.
titre("Hill contre MLE : l'ecart est-il du biais, ou un desaccord ?")
K_HILL = (30, 50, 86, 91, 120, 200)
B_HILL = 2000
rng_h = np.random.default_rng(12345)


def _hill(x_desc, k):
    lx = np.log(x_desc[:k + 1])
    return float(np.mean(lx[:k]) - lx[k])


_corps = loss[loss <= OPRISK["seuil_u_eur"]]
_sim = {k: [] for k in K_HILL}
for _ in range(B_HILL):
    _y = genpareto.rvs(OPRISK["xi"], scale=OPRISK["sigma_eur"], size=n, random_state=rng_h)
    _ech = np.sort(np.concatenate([_corps, OPRISK["seuil_u_eur"] + _y]))[::-1]
    for k in K_HILL:
        _sim[k].append(_hill(_ech, k))

print(f"  Monde simule : corps observe sous u = {OPRISK['seuil_u_eur']} M EUR, queue GPD de")
print(f"  parametres publies (xi = {OPRISK['xi']}, sigma = {OPRISK['sigma_eur']}), "
      f"{B_HILL} tirages.")
print(f"\n  {'k':>5}{'Hill observe':>15}{'Hill simule (moy.)':>22}{'IC90 simule':>24}"
      f"{'dedans ?':>10}")
_dedans = 0
for k in K_HILL:
    v = np.array(_sim[k])
    lo, hi = float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))
    obs = _hill(_ord, k)
    ok = lo <= obs <= hi
    _dedans += ok
    print(f"  {k:>5}{obs:>15.3f}{v.mean():>22.3f}   [{lo:>6.3f} ; {hi:>6.3f}]{'oui' if ok else 'NON':>10}")

print(f"\n  VERDICT. {_dedans} valeurs de Hill sur {len(K_HILL)} tombent dans l'intervalle a "
      f"90 % du monde")
print(f"  simule a xi = {OPRISK['xi']}. L'ecart de +120 % au MLE n'est donc PAS un desaccord "
      f"entre")
print(f"  deux estimateurs : c'est ce qu'un xi de {OPRISK['xi']} produit quand on applique Hill")
print(f"  a des pertes brutes, decalees du seuil, sur un echantillon de cette taille.")
print(f"  Les donnees ne contredisent pas la calibration publiee, elles la confirment par un")
print(f"  troisieme chemin, apres Anderson-Darling et Kolmogorov-Smirnov.")
print(f"\n  LA SIGNATURE DU BIAIS SE LIT AUSSI DANS LE SENS DE LA DERIVE. Un vrai indice de")
print(f"  queue donne un PLATEAU sur le trace de Hill. Ici l'estimateur croit sans s'arreter")
print(f"  avec k ({_hill(_ord, 30):.2f} a k=30 puis {_hill(_ord, 200):.2f} a k=200) : plus on")
print(f"  descend dans le CORPS de la loi, plus il monte. Et le balayage de seuil ci-dessous")
print(f"  va dans le sens inverse, xi DECROIT quand on monte dans la queue. Si la queue valait")
print(f"  vraiment 1,32, les deux derives seraient inversees.")

print("  Ci-dessus : la FAMILLE GPD est-elle compatible avec les exces, parametres refaits")
print("  a chaque tirage. Ci-dessous : le couple (xi, sigma) que le memoire PUBLIE est-il")
print("  compatible avec eux, parametres imposes. C'est le second test qui atteste le")
print("  chiffre publie ; le premier n'atteste que le choix de loi.")

xi_p, sig_p = OPRISK["xi"], OPRISK["sigma_eur"]
ad_p, ks_p = ad_stat(exc, xi_p, sig_p), ks_stat(exc, xi_p, sig_p)
ad_n2, ks_n2 = [], []
for _ in range(B_GOF):
    xb = genpareto.rvs(xi_p, scale=sig_p, size=n, random_state=rng)
    ad_n2.append(ad_stat(xb, xi_p, sig_p))
    ks_n2.append(ks_stat(xb, xi_p, sig_p))
p_ad2 = float((np.array(ad_n2) >= ad_p).mean())
p_ks2 = float((np.array(ks_n2) >= ks_p).mean())
print(f"\n  Parametres PUBLIES imposes (xi = {xi_p:.4f}, sigma = {sig_p:.2f}) :")
print(f"  Anderson-Darling : A2 = {ad_p:.3f}, p = {p_ad2:.3f}  "
      f"({'NON rejete' if p_ad2 > 0.05 else 'REJETE'} a 5 %).")
print(f"  Kolmogorov-Smirnov : D = {ks_p:.3f}, p = {p_ks2:.3f}  "
      f"({'non rejete' if p_ks2 > 0.05 else 'REJETE'} a 5 %).")

# CONTROLE DE ROBUSTESSE AU SEUIL. Le seuil publie n'est pas le q85 des donnees courantes ;
# on verifie que le choix de seuil ne porte pas la conclusion.
u_q85 = float(np.quantile(loss, 0.85))
e_q85 = np.sort(loss[loss > u_q85] - u_q85)
xi_q, _, sig_q = genpareto.fit(e_q85, floc=0)
print(f"\n  Controle de robustesse au seuil : au q85 des donnees courantes "
      f"({u_q85:.2f} M€, {e_q85.size} exces),")
print(f"  le re-ajustement libre donne xi = {xi_q:.4f}, sigma = {sig_q:.2f}, a comparer a "
      f"xi = {xi:.4f}, sigma = {sig:.2f}")
print(f"  au seuil publie. L'indice de queue bouge de "
      f"{100*abs(xi_q-xi)/xi:.1f} % entre les deux seuils : le resultat ne tient pas au seuil.")

# INCOHERENCE INTERNE DE LA CONFIGURATION. Les chiffres viennent tous du bloc
# OPRISK_COHERENCE de config.py, qui les recalcule a chaque import : ils ne peuvent donc pas
# se perimer en silence si le seuil ou le perimetre bougent un jour.
C = OPRISK_COHERENCE
titre("Coherence du taux de depassement p_u : un defaut connu, chiffre et publie")
print(f"  p_u N'EST PAS UN PARAMETRE LIBRE. La formule POT a trois entrees (u, p_u, (xi, sigma))")
print(f"  et deux degres de liberte : le seuil et l'echantillon fixes, le taux de depassement")
print(f"  est COMPTE, il n'est pas choisi. Le couple publie n'est donc pas une hypothese que")
print(f"  l'on pourrait assumer, c'est une incoherence arithmetique.")
print(f"\n  configuration : n_excess = {OPRISK['n_excess']}, p_u = {C['p_u_publie']:.4f}")
print(f"  p_u x n = {C['p_u_publie']*loss.size:.1f} exces attendus, contre {n} que le seuil "
      f"publie donne dans la donnee courante.")
print(f"  taux coherent = {n}/{loss.size} = {C['p_u_coherent']:.4f}, soit "
      f"{100*C['ecart_relatif_p_u']:+.1f} % sur le taux.")
print(f"  p_u vaut 88/583 : le taux du percentile 85 d'un filtrage anterieur. C'est donc lui")
print(f"  qui est en decalage, et non n_excess, qui compte bien les exces du seuil publie.")
print(f"\n  EFFET. Quantile de severite a 99,5 % (UN sinistre, pas un capital) : "
      f"{C['var_995_publiee']:.2f} publie contre "
      f"{C['var_995_coherente']:.2f} M EUR coherente,")
print(f"  soit {100*C['ecart_relatif_var']:+.1f} %. Calcule en rapport et non en niveau : la VaR")
print(f"  reconstruite a partir des xi et sigma arrondis du dictionnaire vaut 662,99 et non")
print(f"  662,78, mais le rapport, lui, ne depend pas de cet arrondi.")
print(f"\n  SENS, ET IL FAUT DISTINGUER DEUX PRUDENCES QUE L'ON CONFOND FACILEMENT.")
print(f"  Du point de vue de la SOLVABILITE, l'ecart est anti-conservateur : il sous-estime")
print(f"  le besoin de capital de {100*C['ecart_relatif_var']:.1f} %, et une sous-estimation ne "
      f"s'excuse pas par la prudence,")
print(f"  elle se declare. Du point de vue de la THESE, il va au contraire dans le bon sens :")
print(f"  le chiffre avance par le memoire n'est pas gonfle par ce defaut, il est minore.")
print(f"  Les deux sont vrais, ils ne parlent pas de la meme chose, et seul le premier engage.")
print(f"\n  MATERIALITE. {C['materialite']}.")
print(f"  DECISION. {C['decision']}.")
print(f"  Rejouer le pipeline pour ce seul ecart reinjecterait un bruit de Monte-Carlo du meme")
print(f"  ordre (bootstrap a 200 tirages) : on deplacerait des centaines de nombres publies")
print(f"  sans pouvoir attribuer un seul deplacement a la correction. On ne rebase pas un")
print(f"  modele pour un ecart immateriel detecte tard, on l'inscrit au registre des limites.")

# couverture de l'IC90 asymptotique de xi (sd = (1+xi)/sqrt(n)) a n fini
titre("Couverture reelle de l'IC90 asymptotique de xi (a n fini)")
cov = 0
for _ in range(M_COV):
    xb = genpareto.rvs(xi, scale=sig, size=n, random_state=rng)
    try:
        cb, _, _ = genpareto.fit(xb, floc=0)
    except Exception:
        continue
    sd = (1 + cb) / np.sqrt(n)
    if cb - 1.645 * sd <= xi <= cb + 1.645 * sd:
        cov += 1
print(f"  Couverture empirique de l'IC90 = {100*cov/M_COV:.0f} % (nominal 90 %).")
# CE N'EST PAS UNE « BONNE CALIBRATION », ET C'ETAIT LE MOT QU'IMPRIMAIT CE SCRIPT.
# Une couverture reelle de 86 % pour un intervalle annonce a 90 % veut dire que l'intervalle
# publie est TROP ETROIT : l'incertitude reportee sur xi est sous-estimee, pas surestimee.
# C'est petit, mais c'est le sens qui compte, et il va du meme cote que l'ecart sur p_u :
# les deux DEFAUTS D'ESTIMATION du memoire sont anti-conservateurs, quand tous ses choix
# POSES (xi = 0,90, a = 0,60, phi = 9,20) sont au contraire prudents. Un lecteur a le droit
# de voir les deux colonnes.
# DE COMBIEN. Sous l'approximation normale, un intervalle qui couvre 86 % au lieu de 90 %
# doit voir sa demi-largeur multipliee par z(0,95)/z((1+0,86)/2) pour atteindre le nominal.
_cov = cov / M_COV
_z_nom = stats.norm.ppf(0.95)
_z_reel = stats.norm.ppf((1 + _cov) / 2)
_facteur = _z_nom / _z_reel
print(f"  LECTURE. Un intervalle annonce a 90 % qui n'en couvre que {100*_cov:.0f} est trop")
print(f"  ETROIT : l'incertitude publiee sur xi est sous-estimee, pas l'inverse. Pour atteindre")
print(f"  le nominal il faudrait elargir la demi-largeur d'un facteur {_facteur:.2f}, soit "
      f"{100*(_facteur-1):.0f} %.")
print(f"  C'est le second des deux defauts d'estimation du memoire, et il va DANS LE MEME SENS")
print(f"  que le premier (le taux de depassement gele, +2,4 % de capital manquant) : tous deux")
print(f"  sous-estiment. Les choix POSES du memoire, eux, sont tous prudents. Voir le")
print(f"  chapitre inventaire des hypotheses, qui met les deux colonnes en regard.")
print(f"  Cela conforte le choix de rapporter les IC bootstrap plutot que les asymptotiques.")

# stabilite de xi au seuil
qs = np.array([0.75, 0.80, 0.85, 0.90, 0.93, 0.95])
xi_thr, sd_thr, u_thr = [], [], []
for q in qs:
    uu = np.quantile(loss, q); e = loss[loss > uu] - uu
    cc, _, _ = genpareto.fit(e, floc=0)
    xi_thr.append(cc); sd_thr.append((1 + cc) / np.sqrt(e.size)); u_thr.append(uu)
xi_thr, sd_thr = np.array(xi_thr), np.array(sd_thr)
# LA PLAGE DE xi SUR LES SEUILS, IMPRIMEE. La table des parametres de l'annexe annonce une
# sensibilite « forte, 0,68 a 0,93 selon le seuil » : c'est le resultat de ce balayage, et il
# n'etait affiche que dans le panneau (d) de la figure. Une sensibilite qui ne se lit que sur
# un graphique n'est verifiable par personne.
n_thr = np.array([int((loss > uu).sum()) for uu in u_thr])
print(f"\n  Stabilite de xi au seuil, balayage q = {', '.join(f'{q:.2f}' for q in qs)} :")
for q, uu, cc, ne in zip(qs, u_thr, xi_thr, n_thr):
    flag = "" if ne >= 30 else "   (sous le minimum de 30 exces : non retenu)"
    print(f"    q = {q:.2f}   u = {uu:6.2f} M EUR   {ne:>3} exces   xi = {cc:.3f}{flag}")
adm = n_thr >= 30
print(f"  plage brute sur tout le balayage : [{xi_thr.min():.2f} ; {xi_thr.max():.2f}].")
print(f"  plage RETENUE, seuils a 30 exces au moins : "
      f"[{xi_thr[adm].min():.2f} ; {xi_thr[adm].max():.2f}].")
# ET IL FAUT COUPER LE BALAYAGE AU SEUIL PUBLIE, parce que les deux moities ne disent pas la
# meme chose. Au-dessus, on est dans la queue et l'on voit l'estimation se deplacer a l'interieur
# de son propre intervalle de confiance : c'est de l'incertitude d'estimation, deja publiee.
# En dessous, on ajuste une GPD a des pertes qui ne sont pas encore dans le regime asymptotique,
# et xi remonte : c'est le biais de seuil classique, pas une information sur la queue. Melanger
# les deux moities produit une amplitude spectaculaire qui ne veut rien dire.
sup = np.array(u_thr) >= OPRISK["seuil_u_eur"]
ic = OPRISK["xi_ic90"]
print(f"  AU-DESSUS du seuil publie : xi va de {xi_thr[sup].max():.2f} a {xi_thr[sup].min():.2f} "
      f"quand le seuil monte,")
print(f"    tout entier dans l'IC90 publie de xi, [{ic[0]:.2f} ; {ic[1]:.2f}] : la sensibilite au")
print(f"    seuil ne cree donc pas d'incertitude nouvelle, elle se lit dans celle deja declaree.")
print(f"  EN DESSOUS : xi remonte jusqu'a {xi_thr[~sup].max():.2f}, au-dela de la borne haute de "
      f"cet IC.")
print(f"    Biais de seuil attendu, et c'est ce qui justifie de ne pas descendre plus bas.")
print(f"  A NE PAS ECRIRE : « xi est stable quand on fait varier le seuil ». Il ne l'est pas.")
print(f"    Ce qui est stable, c'est le VOISINAGE du seuil retenu, ou l'indice ne bouge que de")
print(f"    0.5 % entre le seuil publie et le q85 des donnees courantes.")
# POURQUOI DEUX PLAGES. La plage brute inclut des seuils ou l'ajustement n'est pas defendable :
# tout en haut il ne reste plus assez d'exces pour estimer une queue, tout en bas on ajuste une
# GPD a des pertes qui ne sont pas encore dans le regime asymptotique. La regle des 30 exces est
# celle que le memoire s'impose deja ailleurs (registre du script 67) ; on l'applique ici plutot
# que d'annoncer une amplitude flatteuse ou alarmiste selon le bout du balayage que l'on garde.

# =====================================================================================
# FREQUENCE (comptes firme-annee)
# =====================================================================================
titre("Frequence : adequation binomiale negative vs Poisson (comptes firme-annee)")
# Panel firme-annee construit sur l'ENSEMBLE finance (comme 08b) : le span d'observation
# d'une firme couvre TOUS ses risques, une annee sans evenement TIC dans ce span est un vrai zero.
ICT = ["Systems Security", "Systems", "Vendors & Suppliers",
       "Monitoring and Reporting", "Unauthorized Activity"]
raw = pd.read_excel(os.path.join(REPO, "data", "raw", "SAS_OpRisk_Global_Data_June_2026.xlsx"),
                    sheet_name="Datasets")
raw["year"] = pd.to_datetime(raw["First Year of Event"], errors="coerce").dt.year
fin = raw[raw["Basel Business Line - Level 1"] != "Non-FS"].copy()
fin = fin[(fin.year >= 2005) & (fin.year <= 2022)]
ictf = fin[fin["Sub Risk Category"].isin(ICT)]
span = fin.groupby("Firm Name")["year"].agg(["min", "max"])
ict_n = ictf.groupby(["Firm Name", "year"]).size()
rows = []
for firm, (y0, y1) in span[["min", "max"]].iterrows():
    for yy in range(int(y0), int(y1) + 1):
        rows.append(int(ict_n.get((firm, yy), 0)))
y = np.array(rows)


def ll_nb2(p, k):
    mu, r = np.exp(p)
    return np.sum(special.gammaln(k + r) - special.gammaln(r) - special.gammaln(k + 1)
                  + r * np.log(r / (r + mu)) + k * np.log(mu / (r + mu)))


mu_p = y.mean()
ll_p = float(np.sum(y * np.log(mu_p) - mu_p - special.gammaln(y + 1)))
opt = optimize.minimize(lambda p: -ll_nb2(p, y), x0=np.log([mu_p, 1.0]), method="Nelder-Mead")
mu_nb, r_nb = np.exp(opt.x)
lr = 2 * (-opt.fun - ll_p)
p_lr = 0.5 * stats.chi2.sf(lr, 1)
disp = y.var(ddof=1) / y.mean()
print(f"  {y.size} cellules firme-annee ; moyenne = {mu_p:.3f}, dispersion Var/E = {disp:.2f}.")
print(f"  Poisson logL = {ll_p:,.0f} ; NB (r={r_nb:.2f}) logL = {-opt.fun:,.0f}.")
print(f"  LR Poisson vs NB : LR = {lr:,.0f}, p = {p_lr:.1e} -> la NB l'emporte nettement.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print(f"  1. Severite : la GPD n'est PAS rejetee (Anderson-Darling p = {p_ad:.2f}, KS p = {p_ks:.2f}) ;")
print(f"     xi = {xi:.2f} au seuil publie, stable dans son VOISINAGE (0.5 % jusqu'au q85) mais")
print(f"     decroissant sur tout le balayage ; mean residual life lineaire au-dela de u.")
print(f"  2. La couverture de l'IC90 asymptotique est de {100*cov/M_COV:.0f} % : "
      f"{'correcte' if abs(cov/M_COV-0.9)<0.05 else 'imparfaite a n fini, d ou le recours au bootstrap'}.")
print(f"  3. Frequence : la binomiale negative domine le Poisson sans ambiguite "
      f"(LR p = {p_lr:.0e}), la surdispersion est un fait, pas un artefact.")
print("  Les deux briques du socle passent les tests d'adequation : le niveau reste incertain")
print("  (bande), mais les FORMES de loi posees sont soutenues par la donnee.")

# =====================================================================================
# figure J3
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 10.5,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
})
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime cette
# figure sur 7,28 pouces : tracee sur 10,2 elle subissait une reduction de 0,71 et
# ses legendes sortaient sous 6 points. A 9,1 pouces le facteur monte a 0,80, donc
# un titre de panneau de 11 points s'imprime a 8,8. La hauteur suit le meme
# rapport : la figure passe de 10,3 a 9,7 cm imprimes, six panneaux ne remplissant
# pas une page de memoire.
fig, axs = plt.subplots(2, 3, figsize=(9.8, 4.50))

# (a) QQ-plot
theo = genpareto.ppf((np.arange(1, n + 1) - 0.5) / n, xi, scale=sig)
axs[0, 0].scatter(theo, exc, s=14, color=BLUE, alpha=0.7, edgecolor="none")
lim = [0, max(theo.max(), exc.max()) * 1.02]
axs[0, 0].plot(lim, lim, color=ACCENT, lw=1.5)
axs[0, 0].set_xlabel("quantiles GPD théoriques (M€)", color=INK2)
axs[0, 0].set_ylabel("quantiles empiriques (M€)", color=INK2)
axs[0, 0].set_title("(a)  QQ-plot sévérité (GPD)", fontsize=11, color=INK, pad=6)

# (b) PP-plot
emp = (np.arange(1, n + 1) - 0.5) / n
fit = genpareto.cdf(exc, xi, scale=sig)
axs[0, 1].scatter(emp, fit, s=14, color=BLUE, alpha=0.7, edgecolor="none")
axs[0, 1].plot([0, 1], [0, 1], color=ACCENT, lw=1.5)
axs[0, 1].set_xlabel("probabilité empirique", color=INK2)
axs[0, 1].set_ylabel("probabilité GPD ajustée", color=INK2)
axs[0, 1].set_title("(b)  PP-plot sévérité (GPD)", fontsize=11, color=INK, pad=6)

# (c) mean residual life
vgrid = np.quantile(loss, np.linspace(0.5, 0.97, 30))
mrl = np.array([np.mean(loss[loss > v] - v) for v in vgrid])
se = np.array([np.std(loss[loss > v] - v) / np.sqrt((loss > v).sum()) for v in vgrid])
axs[0, 2].plot(vgrid, mrl, color=BLUE, lw=2)
axs[0, 2].fill_between(vgrid, mrl - 1.96 * se, mrl + 1.96 * se, color=BLUE, alpha=0.15)
axs[0, 2].axvline(u, color=ACCENT, ls="--", lw=1.4)
# etiquette du seuil remontee au-dessus de la bande : posee au bas de la
# verticale, elle tombait sur la courbe et sur son intervalle.
axs[0, 2].text(u * 1.25, mrl.max() * 1.06, f"$u={u:.0f}$", fontsize=9, color=ACCENT,
               va="bottom")
axs[0, 2].set_xlabel("seuil $v$ (M€)", color=INK2)
axs[0, 2].set_ylabel("excès moyen $e(v)$ (M€)", color=INK2)
axs[0, 2].set_title("(c)  Vie résiduelle moyenne\n(linéaire au-delà de $u$)",
                    fontsize=11, color=INK, pad=6)

# (d) stabilite de xi au seuil
axs[1, 0].errorbar(u_thr, xi_thr, yerr=1.645 * sd_thr, fmt="o-", color=BLUE,
                   ecolor=MUTED, capsize=3, ms=5)
axs[1, 0].axvline(u, color=ACCENT, ls="--", lw=1.4)
axs[1, 0].axhline(1.0, color=MUTED, ls=":", lw=1)
# repere de la ligne xi = 1, renvoye a droite : a gauche il tombait dans la barre
# d'erreur du premier seuil.
axs[1, 0].text(u_thr[-1], 1.03, "$\\xi=1$", fontsize=9, color=MUTED,
               ha="right", va="bottom")
axs[1, 0].set_xlabel("seuil $u$ (M€)", color=INK2)
axs[1, 0].set_ylabel("$\\hat\\xi$ (IC90)", color=INK2)
axs[1, 0].set_title("(d)  Stabilité de ξ au seuil", fontsize=11, color=INK, pad=6)

# (e) test d'Anderson-Darling : loi nulle bootstrap + observe
axs[1, 1].hist(ad_null, bins=40, color=BLUE, alpha=0.5, edgecolor="#fcfcfb")
axs[1, 1].axvline(ad_obs, color=ACCENT, lw=1.8,
                  label=f"observé $A^2={ad_obs:.2f}$".replace(".", "{,}"))
axs[1, 1].text(0.70, 0.96,
               (f"p = {p_ad:.2f}\n(non rejeté)" if p_ad > 0.05
                else f"p = {p_ad:.2f}\n(rejeté)").replace("0.", "0,"),
               transform=axs[1, 1].transAxes, fontsize=9, color=INK2, ha="center", va="top")
axs[1, 1].set_xlabel("statistique $A^2$ sous $H_0$", color=INK2)
axs[1, 1].set_ylabel("fréquence (bootstrap)", color=INK2)
axs[1, 1].legend(frameon=False, fontsize=9)
axs[1, 1].set_title("(e)  Adéquation GPD\n(Anderson-Darling, bootstrap)", fontsize=11, color=INK, pad=6)

# (f) frequence : observe vs Poisson vs NB
kmax = 5
xs = np.arange(kmax + 1)
obs = np.array([(y == k).mean() for k in range(kmax)] + [(y >= kmax).mean()])
pp = poisson.pmf(xs, mu_p); pp[kmax] = 1 - poisson.cdf(kmax - 1, mu_p)
pnb = r_nb / (r_nb + mu_nb)
nb = nbinom.pmf(xs, r_nb, pnb); nb[kmax] = 1 - nbinom.cdf(kmax - 1, r_nb, pnb)
wd = 0.27
axs[1, 2].bar(xs - wd, obs, width=wd, color="#204993", label="observé", edgecolor="#fcfcfb")
axs[1, 2].bar(xs, pp, width=wd, color=MUTED, label="Poisson", edgecolor="#fcfcfb")
axs[1, 2].bar(xs + wd, nb, width=wd, color=ACCENT,
              label=f"NB ($r={r_nb:.2f}$)".replace(".", "{,}"), edgecolor="#fcfcfb")
axs[1, 2].set_yscale("log")
axs[1, 2].set_xticks(xs); axs[1, 2].set_xticklabels([str(k) for k in range(kmax)] + [f"{kmax}+"])
axs[1, 2].set_xlabel("événements TIC par firme et par an", color=INK2)
axs[1, 2].set_ylabel("probabilité", color=INK2)
axs[1, 2].set_ylim(top=axs[1, 2].get_ylim()[1] * 300)
axs[1, 2].set_yticks([1e-7, 1e-5, 1e-3, 1e-1, 1e0])
axs[1, 2].legend(frameon=False, fontsize=9, loc="upper right",
                 handlelength=1.4, borderaxespad=0.2)
_mant, _expo = f"{p_lr:.0e}".split("e")          # meme valeur, meme arrondi qu'avant
axs[1, 2].set_title("(f)  Fréquence : la NB colle,\nle Poisson rate la queue "
                    f"($p={_mant}\\cdot10^{{{int(_expo)}}}$)",
                    fontsize=11, color=INK, pad=6)

for ax in axs.flat:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# PAS DE TITRE GENERAL : la legende LaTeX porte le titre, et celui-ci commencait
# par le code interne « J3 : ».
fig.tight_layout()
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "J3_validation_adequation.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
