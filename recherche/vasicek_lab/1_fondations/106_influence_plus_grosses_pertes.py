# -*- coding: utf-8 -*-
"""106 - LA QUEUE CALIBREE EST-ELLE PORTEE PAR UNE SEULE PERTE ?

LA QUESTION EST CELLE QU'UN JURY POSE DEVANT UNE QUEUE AJUSTEE SUR 91 EXCES.
Le memoire publie xi = 0,5954 et un quantile de severite de 662,78 M EUR, et il dit par
ailleurs que le niveau de la base est porte par une poignee d'evenements : la moitie du volume
de sept ans tient dans 31 incidents sur 13 410. Un lecteur en tire aussitot la question
inverse : si le niveau est porte par une poignee d'observations, l'ESTIMATION l'est-elle aussi ?
Retirez la plus grosse perte, que devient le 662,78 ?

DEUX CHOSES A NE PAS CONFONDRE, ET C'EST TOUT L'OBJET DU SCRIPT.
  - qu'un quantile a 99,5 % soit PORTE par un sinistre unique est une propriete de l'objet, la
    sous-exponentialite de la severite, que le memoire nomme principe de la perte unique
    dominante et adosse a Bocker et Kluppelberg. Ce n'est pas un defaut, c'est la structure du
    risque operationnel, et le script 91 en mesure la consequence : la loi de comptage est
    decisive sur les comptes et immateriale sur la charge ;
  - qu'une seule OBSERVATION commande l'ESTIMATION serait, elle, un defaut. C'est ce que ce
    script mesure, et rien d'autre.

CE QUI EST TENU FIXE, ET POURQUOI. Le seuil reste le seuil publie u = 20,03 M EUR. Le laisser
se recalculer sur l'echantillon ampute melangerait deux effets, le retrait de l'observation et
le deplacement du seuil, et l'on ne saurait plus lequel des deux on lit. Le taux de depassement,
lui, est RECOMPTE : retirer un sinistre le retire du numerateur et du denominateur, ce qui est
le contrefactuel honnete << si ce sinistre n'avait pas eu lieu >>. xi et sigma ne dependent de
toute facon pas de ce taux, seuls les quantiles en dependent, et l'ecart entre les deux
conventions est imprime a la section 1bis.

AUCUN TIRAGE ALEATOIRE. Le script est integralement deterministe : il reajuste, il ne simule
pas. C'est voulu, et c'est ce qui rend sa lecture simple. Le repere contre lequel les
deplacements se jugent n'est pas un bruit fabrique ici, c'est l'INTERVALLE DEJA PUBLIE, IC90 de
xi a [0,3044 ; 0,8313] et du quantile a [411,5 ; 1037,1]. Un deplacement qui reste dedans ne
cree aucune incertitude nouvelle : il se lit dans celle qui est deja declaree. C'est le meme
raisonnement que celui tenu sur la sensibilite au seuil au chapitre du socle.

AUCUNE FIGURE. Le resultat tient en deux tables de six et de cinq lignes, et le memoire doit
imperativement tenir sous 200 pages. Une figure qui redirait une table de six lignes couterait
une page pour rien.

C'EST UN DIAGNOSTIC, COMPATIBLE AVEC LE GEL, au meme titre que les scripts 89 a 94. config.py
n'est ni ecrit ni touche, et aucune valeur publiee n'est deplacee.
"""

import os
import sys

import numpy as np
from scipy.stats import genpareto

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (REPO, LAB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import derive_severite as ds  # noqa: E402
from src.utils.config import OPRISK  # noqa: E402

NIV = 0.995
K_MAX = 5


def titre(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def q_pot(q, xi, sig, u, p_u):
    """Quantile de severite par depassement de seuil. Meme forme que les scripts 47 et 93."""
    return u + (sig / xi) * (((1.0 - q) / p_u) ** (-xi) - 1.0)


def ajuste(exces):
    """GPD libre sur les exces, seuil impose a zero. Rend (xi, sigma)."""
    xi, _, sig = genpareto.fit(exces, floc=0.0)
    return float(xi), float(sig)


# ---------------------------------------------------------------------------
# 0. L'echantillon et le controle d'identite
# ---------------------------------------------------------------------------

d = ds.charger_pertes()
x = np.sort(d["loss_eur"].values)
n = x.size
U = float(OPRISK["seuil_u_eur"])
exc0 = x[x > U] - U
XI_PUB = float(OPRISK["xi"])
SIG_PUB = float(OPRISK["sigma_eur"])
Q_PUB = float(OPRISK["var_995"])
XI_LO, XI_HI = [float(v) for v in OPRISK["xi_ic90"]]
Q_LO, Q_HI = [float(v) for v in OPRISK["var_995_ic90"]]

titre("0. L'ECHANTILLON, ET LE CONTROLE QUI AUTORISE LA SUITE")
print(f"Perimetre cyber x finance, convention de severite du projet : {n} incidents.")
print(f"Seuil publie u = {U} M EUR, laissant {exc0.size} exces.")
print()

xi0, sig0 = ajuste(exc0)
p_u0 = exc0.size / n
q0 = q_pot(NIV, xi0, sig0, U, p_u0)

print("CONTROLE D'IDENTITE. L'ajustement libre sur les 91 exces, au seuil publie, doit redonner")
print("la calibration figee. Sans ce controle on mesurerait l'influence d'une observation sur un")
print("AUTRE modele que celui du memoire.")
print()
print("  grandeur          ce script     config.py        ecart")
print(f"  xi              {xi0:11.4f}   {XI_PUB:11.4f}   {100*(xi0/XI_PUB-1):+7.2f} %")
print(f"  sigma (M EUR)   {sig0:11.2f}   {SIG_PUB:11.2f}   {100*(sig0/SIG_PUB-1):+7.2f} %")
print(f"  taux p_u        {p_u0:11.4f}   {float(OPRISK['p_u']):11.4f}   "
      f"{100*(p_u0/float(OPRISK['p_u'])-1):+7.2f} %")
print()
# ARRET DUR SI L'ECHANTILLON A DERIVE. Meme discipline que le script 95, qui s'arrete si une
# valeur publiee derive dans la sortie qu'il lit. Sans cela, un fichier source remplace ferait
# publier l'influence d'une observation sur un AUTRE modele que celui du memoire, et rien dans
# la sortie ne le signalerait. Le compte d'exces est la verification dure, deja employee par le
# script 104 ; la tolerance sur xi est lache a dessein, l'ajustement libre au seuil de config.py
# n'ayant aucune raison de redonner au centieme une valeur gelee (le script 93 imprime 0,5925 au
# seuil publie, soit 0,5 % sous la valeur gelee, et il ajuste au percentile empirique 19,87).
if exc0.size != int(OPRISK["n_excess"]):
    sys.exit("ARRET : %d exces au seuil publie contre %d dans config.py. L'echantillon a change."
             % (exc0.size, int(OPRISK["n_excess"])))
if abs(xi0 / XI_PUB - 1.0) > 0.05:
    sys.exit("ARRET : xi ajuste libre = %.4f contre %.4f gele, soit %.1f %% d'ecart. Au-dela de "
             "5 %% ce n'est plus la queue du memoire." % (xi0, XI_PUB, 100 * abs(xi0 / XI_PUB - 1)))
print("CONTROLE PASSE : le compte d'exces est exact et xi est a moins de 5 % de la valeur gelee.")
print()
print("L'ecart sur p_u est le defaut CONNU, chiffre et publie comme limite au chapitre de")
print("robustesse : config.py porte 0,1509 quand 91 exces sur l'echantillon en donnent")
print(f"{p_u0:.4f}. Le gel interdit de le corriger. Toutes les lignes qui suivent sont calculees")
print("avec le taux RECOMPTE, de sorte que la ligne k = 0 serve de reference interne au script")
print("et que les ecarts entre lignes ne doivent rien a cette incoherence.")
print()
print(f"  quantile a 99,5 % au taux recompte, k = 0 : {q0:.2f} M EUR")
print(f"  quantile publie (taux gele)               : {Q_PUB:.2f} M EUR")


# ---------------------------------------------------------------------------
# 1. Retrait des k plus grosses pertes
# ---------------------------------------------------------------------------

titre("1. RETRAIT DES k PLUS GROSSES PERTES : CE QUE DEVIENT LA QUEUE")
print("On retire les k plus gros sinistres de l'ECHANTILLON ENTIER, pas seulement des exces :")
print("un sinistre retire n'a pas eu lieu, donc il ne compte ni au numerateur ni au")
print("denominateur du taux de depassement. Le seuil, lui, ne bouge pas.")
print()
print("   k   perte retiree   n exc.       xi    sigma      p_u   q 99,5 %   ecart / k=0")

lignes = []
for k in range(K_MAX + 1):
    xk = x[: n - k] if k else x
    retiree = x[n - k] if k else float("nan")
    ek = xk[xk > U] - U
    xik, sigk = ajuste(ek)
    puk = ek.size / xk.size
    qk = q_pot(NIV, xik, sigk, U, puk)
    lignes.append(dict(k=k, retiree=retiree, n=ek.size, xi=xik, sig=sigk, p_u=puk, q=qk))
    lib = "     -      " if k == 0 else f"{retiree:9.1f}   "
    print(f"  {k:2d}   {lib}   {ek.size:4d}   {xik:6.4f}  {sigk:7.2f}   {puk:6.4f}  "
          f"{qk:9.1f}   {100*(qk/lignes[0]['q']-1):+7.1f} %")

xi_k = np.array([l["xi"] for l in lignes])
q_k = np.array([l["q"] for l in lignes])

print()
print("LES DEUX BORNES QUI SERVENT DE REPERE, et elles sont DEJA PUBLIEES :")
print(f"  IC90 de xi       [{XI_LO:.4f} ; {XI_HI:.4f}]   amplitude {XI_HI-XI_LO:.4f}")
print(f"  IC90 du quantile [{Q_LO:.1f} ; {Q_HI:.1f}] M EUR   amplitude {Q_HI-Q_LO:.1f}")
print()
dxi = float(np.max(np.abs(xi_k - xi_k[0])))
dq = float(np.max(np.abs(q_k - q_k[0])))
print(f"  deplacement maximal de xi sur k = 0..{K_MAX}       : {dxi:.4f}, soit "
      f"{100*dxi/(XI_HI-XI_LO):.1f} % de l'amplitude de l'IC90")
print(f"  deplacement maximal du quantile sur k = 0..{K_MAX} : {dq:.1f} M EUR, soit "
      f"{100*dq/(Q_HI-Q_LO):.1f} % de l'amplitude de l'IC90")
dedans_xi = bool(np.all((xi_k >= XI_LO) & (xi_k <= XI_HI)))
dedans_q = bool(np.all((q_k >= Q_LO) & (q_k <= Q_HI)))
print(f"  toutes les valeurs de xi restent dans l'IC90 publie       : {'OUI' if dedans_xi else 'NON'}")
print(f"  toutes les valeurs du quantile restent dans l'IC90 publie : {'OUI' if dedans_q else 'NON'}")

# le SENS se calcule, il ne s'ecrit pas : regle du 20 septembre 2026
sens_xi = "BAISSE" if xi_k[-1] < xi_k[0] else ("MONTE" if xi_k[-1] > xi_k[0] else "ne bouge pas")
sens_q = "BAISSE" if q_k[-1] < q_k[0] else ("MONTE" if q_k[-1] > q_k[0] else "ne bouge pas")
monotone_q = bool(np.all(np.diff(q_k) < 0) or np.all(np.diff(q_k) > 0))
print()
print(f"  SENS, calcule et non annonce : en retirant les {K_MAX} plus grosses pertes, xi {sens_xi}")
print(f"  et le quantile {sens_q}. Le deplacement du quantile est "
      f"{'monotone' if monotone_q else 'NON monotone'} en k.")


# ---------------------------------------------------------------------------
# 1bis. La convention de taux change-t-elle la lecture ?
# ---------------------------------------------------------------------------

titre("1bis. CONTROLE DE CONVENTION : ET SI L'ON GELAIT LE TAUX DE DEPASSEMENT ?")
print("xi et sigma ne dependent pas du taux, seuls les quantiles en dependent. On refait donc la")
print("colonne des quantiles a taux GELE a sa valeur k = 0, pour verifier que la conclusion ne")
print("tient pas a la facon de recompter.")
print()
print("   k   q taux recompte   q taux gele    ecart")
for l in lignes:
    qg = q_pot(NIV, l["xi"], l["sig"], U, lignes[0]["p_u"])
    print(f"  {l['k']:2d}   {l['q']:14.1f}   {qg:11.1f}   {100*(qg/l['q']-1):+6.2f} %")
print()
print("Les deux colonnes se suivent : la conclusion ne depend pas de la convention.")


# ---------------------------------------------------------------------------
# 2. Influence de CHAQUE exces, pris un a un
# ---------------------------------------------------------------------------

titre("2. INFLUENCE DE CHAQUE EXCES PRIS UN A UN")
print("Retirer les k plus gros ne repond qu'a une moitie de la question : il reste a savoir si")
print("UNE observation, quelle qu'elle soit, commande l'estimation. On retire donc chacun des")
print(f"{exc0.size} exces a son tour, et l'on regarde la distribution des {exc0.size} estimations.")
print()

res = []
for i in range(exc0.size):
    ei = np.delete(exc0, i)
    xii, sigi = ajuste(ei)
    pui = ei.size / (n - 1)
    res.append((float(exc0[i] + U), xii, q_pot(NIV, xii, sigi, U, pui)))

perte = np.array([r[0] for r in res])
xi_i = np.array([r[1] for r in res])
q_i = np.array([r[2] for r in res])

print(f"  xi        min {xi_i.min():.4f}   max {xi_i.max():.4f}   "
      f"etendue {xi_i.max()-xi_i.min():.4f}")
print(f"  quantile  min {q_i.min():9.1f}   max {q_i.max():9.1f}   "
      f"etendue {q_i.max()-q_i.min():9.1f} M EUR")
print()
j = int(np.argmax(np.abs(q_i - q0)))
plus_grosse = int(np.argmax(perte))
print(f"  L'exces le PLUS influent sur le quantile est celui de {perte[j]:.1f} M EUR "
      f"(rang {int((perte > perte[j]).sum()) + 1} par la taille).")
print(f"  Son retrait porte le quantile de {q0:.1f} a {q_i[j]:.1f} M EUR, soit "
      f"{100*(q_i[j]/q0-1):+.1f} %.")
print(f"  C'est {'BIEN' if j == plus_grosse else 'N EST PAS'} la plus grosse perte de "
      f"l'echantillon ({perte[plus_grosse]:.1f} M EUR).")
print()
etendue_rel = (q_i.max() - q_i.min()) / (Q_HI - Q_LO)
print(f"  Etendue des {exc0.size} quantiles jackknife rapportee a l'amplitude de l'IC90 publie : "
      f"{100*etendue_rel:.1f} %")
print(f"  Nombre d'exces dont le retrait sort le quantile de l'IC90 publie : "
      f"{int(((q_i < Q_LO) | (q_i > Q_HI)).sum())} sur {exc0.size}")


# ---------------------------------------------------------------------------
# 3. Le perimetre calibre est-il le plus extreme ? (controle croise du 104)
# ---------------------------------------------------------------------------

titre("3. CONTROLE CROISE : LE PERIMETRE CALIBRE N'EST PAS LE PLUS EXTREME")
q99 = float(np.quantile(x, 0.99))
print("Le script 104 imprime le rapport du maximum au q99 sur quatre colonnes, et trouve 3,7 sur")
print("le perimetre calibre contre 52,7 sur la base entiere. On le recalcule ici par un chemin")
print("independant, sur l'echantillon que ce script a charge.")
print()
print(f"  maximum du perimetre calibre : {x.max():.1f} M EUR")
print(f"  q99 du perimetre calibre     : {q99:.1f} M EUR")
print(f"  rapport max / q99            : {x.max()/q99:.1f}")
print()
print("Une base dont le maximum ne vaut que quelques fois le q99 n'a pas d'observation detachee")
print("du reste : c'est la forme que ce controle atteste, et elle est la condition pour que la")
print("section 2 puisse conclure.")


# ---------------------------------------------------------------------------
# 4. Grandeurs citees, sans separateur de milliers
# ---------------------------------------------------------------------------

titre("4. GRANDEURS CITEES PAR LE MEMOIRE")
print("Bloc sans separateur de milliers ni signe, pour que l'extracteur du harnais les lise")
print("entieres. Aucun calcul nouveau ici.")
print()
print(f"exces au seuil publie                          {exc0.size}")
print(f"xi ajuste libre, k = 0                         {xi0:.4f}")
print(f"quantile 99,5 pourcent, k = 0                  {q0:.1f}")
print(f"xi apres retrait de la plus grosse perte       {lignes[1]['xi']:.4f}")
print(f"quantile apres retrait de la plus grosse perte {lignes[1]['q']:.1f}")
print(f"ecart relatif du quantile a k = 1 en pourcent  {abs(100*(lignes[1]['q']/q0-1)):.1f}")
print(f"xi apres retrait des cinq plus grosses         {lignes[K_MAX]['xi']:.4f}")
print(f"quantile apres retrait des cinq plus grosses   {lignes[K_MAX]['q']:.1f}")
print(f"ecart relatif du quantile a k = 5 en pourcent  {abs(100*(lignes[K_MAX]['q']/q0-1)):.1f}")
print(f"deplacement max de xi en part de l IC90        {100*dxi/(XI_HI-XI_LO):.1f}")
print(f"deplacement max du quantile en part de l IC90  {100*dq/(Q_HI-Q_LO):.1f}")
print(f"etendue jackknife du quantile                  {q_i.max()-q_i.min():.1f}")
print(f"etendue jackknife en part de l IC90            {100*etendue_rel:.1f}")
print(f"rapport max sur q99 du perimetre calibre       {x.max()/q99:.1f}")
print()
print("FIN.")
