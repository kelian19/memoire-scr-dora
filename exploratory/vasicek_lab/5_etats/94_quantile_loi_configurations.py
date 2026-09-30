#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""94 - LE QUANTILE DE LA LOI DES CONFIGURATIONS, ou le trou que le script 69 declare.

CE SCRIPT FERME UNE LIMITE QUE LE PROJET AVAIT ECRITE LUI-MEME. Le script 69 montre que le
capital des 32 configurations de conformite s'ajuste par une forme ADDITIVE en indicateurs de
pilier, a R^2 = 0,9945, et en tire une invariance : l'esperance d'une fonction additive ne
depend que des MARGES, jamais de la dependance, si bien qu'un surcroit de correlation entre
piliers ne deplace le capital espere que de 2 M EUR, soit 0,02 %. Sa propre conclusion pose la
reserve : << l'invariance porte sur l'ESPERANCE. Une conference de conformite deplacerait un
quantile de la loi des configurations ou une mesure de sa queue. >> Ce quantile n'a jamais ete
calcule. Il l'est ici.

POURQUOI LE RESULTAT NE PEUT PAS ETRE LE MEME. L'argument d'additivite est un argument sur
l'esperance et il est exact. Il ne transporte pas a un quantile : la loi d'une somme
d'indicateurs dependants n'est pas determinee par ses marges, meme quand son esperance l'est.
Une correlation plus forte polarise les configurations, donc elle epaissit les DEUX extremites
de la loi des configurations a esperance inchangee. Ce script mesure de combien.

UN AVERTISSEMENT DE VOCABULAIRE, ET IL N'EST PAS COSMETIQUE. La grandeur calculee ici est un
quantile de la loi des CONFIGURATIONS, c'est-a-dire de l'incertitude sur l'ETAT de conformite de
l'entite. Ce n'est PAS un quantile de la charge annuelle, donc ce n'est pas une mesure de
capital, et le memoire n'a aucune macro pour elle : \VaR est reservee a la charge agregee et
\qsev a la severite d'un sinistre. Confondre les deux ferait lire une incertitude d'etat comme
une exigence de solvabilite. On l'appelle donc ici << quantile de la loi des configurations >>,
jamais VaR.

CE QU'IL TROUVE, EN TROIS LIGNES. La reserve du script 69 etait justifiee et elle est desormais
chiffree : l'esperance bouge de 0,02 % quand le quantile a 75 % bouge de 5,5 % et la moyenne de
queue a 90 % de 1,1 %. L'ordre de grandeur reste petit, deux ordres au-dessus de l'esperance mais
loin des incertitudes declarees ailleurs. Et l'objet a un plafond : au-dela d'environ 93 % le
quantile de la loi des configurations EST la configuration integralement non conforme, donc les
colonnes hautes du tableau sont SATUREES et non invariantes.

C'EST UN DIAGNOSTIC. Aucun parametre publie n'est touche, et le controle est que l'esperance a
surcroit nul reproduit celle du script 69.
"""

import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                                 # noqa: E402
from euro_cascade_model import PARAMS, var                      # noqa: E402
import scr_engine as eng                                        # noqa: E402

WID = 88
PIL = eng.PIL
N = len(PIL)

# Protocole IDENTIQUE au script 69, recopie et non importe : le script 69 n'est pas un module.
# Le controle de la section 1 est ce qui atteste que la recopie est fidele.
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
NSIM_ETATS = 4_000_000
DELTAS = (0.0, 0.10, 0.20, 0.30, 0.38)
NIVEAUX = (0.50, 0.75, 0.90, 0.95, 0.99)


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def fnum(v):
    return f"{v:,.0f}".replace(",", " ")


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


def matrice_corr(delta):
    rho = GAMMA ** 2
    R = np.full((N, N), rho)
    np.fill_diagonal(R, 1.0)
    i1, i2, i3 = PIL.index(J1), PIL.index(J2), PIL.index(J3)
    for i in (i1, i2):
        R[i, i3] = R[i3, i] = rho + delta
    return R


def poids_configs(delta, nsim=NSIM_ETATS, seed=20260812):
    R = matrice_corr(delta)
    L = np.linalg.cholesky(R)
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((nsim, N)) @ L.T
    nc = C <= K
    codes = (nc * (1 << np.arange(N))).sum(axis=1)
    return np.bincount(codes, minlength=2 ** N) / nsim


def quantile_discret(val, poids, q):
    """Quantile de niveau q d'une loi discrete, convention de la fonction de repartition.

    Le plus petit atome dont la masse cumulee atteint q. Un quantile discret est PAR NATURE
    en escalier : il reste sur le meme atome tant que la masse cumulee ne franchit pas q, puis
    saute. Ce n'est pas un artefact de calcul et le script le signale plutot que de le lisser.
    """
    o = np.argsort(val)
    v, w = val[o], poids[o]
    c = np.cumsum(w)
    i = int(np.searchsorted(c, q, side="left"))
    return float(v[min(i, v.size - 1)])


def cte_discret(val, poids, q):
    """Moyenne des atomes au-dela du quantile de niveau q, ponderee et renormalisee."""
    seuil = quantile_discret(val, poids, q)
    m = val >= seuil
    return float(np.sum(val[m] * poids[m]) / np.sum(poids[m]))


# ---------------------------------------------------------------------------
# 1. Les 32 configurations, et le controle
# ---------------------------------------------------------------------------

titre("1. LE CAPITAL DES 32 CONFIGURATIONS, ET LE CONTROLE CONTRE LE SCRIPT 69")
print(f"Lecture binaire C / NC sur {N} piliers, {2 ** N} configurations, {NY} annees, "
      f"{len(SEEDS)} graines,")
print("a nombres communs. Protocole recopie du script 69 ; le controle ci-dessous est ce qui")
print("atteste que la recopie est fidele, et sans lui le reste porterait sur un autre modele.")
print()
scr = np.empty(2 ** N)
for m in range(2 ** N):
    sm = masque_vers_etats(m)
    scr[m] = float(np.mean([var(annual_config(sm, NY, s)) for s in SEEDS]))

w0 = poids_configs(0.0)
e0 = float(np.sum(scr * w0))
print(f"  capital minimal (tous conformes)   : {fnum(scr.min()):>7} M EUR")
print(f"  capital maximal (tous non conf.)   : {fnum(scr.max()):>7} M EUR")
print(f"  ESPERANCE a surcroit nul           : {fnum(e0):>7} M EUR   "
      "<- le script 69 imprime 9736")
print()
marges = [float(np.sum(w0[[m for m in range(2 ** N) if m & (1 << i)]])) for i in range(N)]
print("  controle des marges a surcroit nul, P(non conforme) par pilier :")
print("    " + "  ".join(f"P{PIL[i]} {marges[i]:.4f}" for i in range(N))
      + f"   (ancrage {P_NC})")


# ---------------------------------------------------------------------------
# 2. La loi des configurations : esperance contre quantiles
# ---------------------------------------------------------------------------

titre("2. ESPERANCE CONTRE QUANTILES, LE LONG DU SURCROIT DE CORRELATION")
print("Le surcroit delta s'ajoute aux deux seules paires structurees, par une matrice de")
print("correlation dont la diagonale vaut un, ce qui preserve les marges par CONSTRUCTION. La")
print("comparaison porte donc sur la structure de dependance seule, a marges strictement fixees.")
print()
en_tete = "  delta   esperance" + "".join(f"   q {100*q:4.0f} %" for q in NIVEAUX) + "    CTE 90 %"
print(en_tete)
res = {}
for dl in DELTAS:
    w = poids_configs(dl)
    e = float(np.sum(scr * w))
    qs = [quantile_discret(scr, w, q) for q in NIVEAUX]
    cte = cte_discret(scr, w, 0.90)
    res[dl] = dict(e=e, qs=qs, cte=cte, w=w)
    print(f"  {dl:5.2f}   {fnum(e):>9}" + "".join(f"   {fnum(q):>7}" for q in qs)
          + f"   {fnum(cte):>9}")

print()
print("DEPLACEMENT ENTRE LE MODELE ACTUEL ET LE SURCROIT MAXIMAL ADMISSIBLE :")
a, b = res[DELTAS[0]], res[DELTAS[-1]]
print(f"  esperance : {fnum(a['e'])} -> {fnum(b['e'])}, soit "
      f"{100*(b['e']/a['e'] - 1):+.2f} %")
for i, q in enumerate(NIVEAUX):
    print(f"  q {100*q:4.0f} %  : {fnum(a['qs'][i])} -> {fnum(b['qs'][i])}, soit "
          f"{100*(b['qs'][i]/a['qs'][i] - 1):+.2f} %")
print(f"  CTE 90 %  : {fnum(a['cte'])} -> {fnum(b['cte'])}, soit "
      f"{100*(b['cte']/a['cte'] - 1):+.2f} %")
print()
print("UN QUANTILE DISCRET EST EN ESCALIER, et il faut le dire plutot que de le lisser. La loi")
print(f"des configurations ne porte que {2 ** N} atomes : un quantile reste sur le meme atome tant")
print("que la masse cumulee ne franchit pas son niveau, puis saute. Une absence de deplacement")
print("d'un quantile n'est donc PAS une invariance, seulement l'absence de franchissement.")


# ---------------------------------------------------------------------------
# 2bis. Au-dela d'un certain niveau, le quantile NE PEUT PAS bouger
# ---------------------------------------------------------------------------

titre("2bis. LE PLAFOND DE L'EXERCICE : AU-DELA D'UN NIVEAU, LE QUANTILE EST SATURE")
print("Les colonnes a 95 % et 99 % ci-dessus ne bougent pas d'un iota, et il serait faux d'y lire")
print("une invariance. L'atome le plus haut, la configuration ou les cinq piliers sont non")
print("conformes, porte a lui seul une masse importante : des que le complement du niveau tombe")
print("sous cette masse, le quantile EST cet atome et il ne peut plus bouger par construction.")
print()
print("  delta   P(tous non conformes)   niveau au-dela duquel le quantile est sature")
for dl in DELTAS:
    p_max = float(res[dl]["w"][2 ** N - 1])
    print(f"  {dl:5.2f}   {100*p_max:19.2f} %   {100*(1.0 - p_max):>8.2f} %")
print()
print("CONSEQUENCE, ET C'EST UNE LIMITE DE L'OBJET ET NON DU CALCUL. Un << quantile a 99,5 % de la")
print("loi des configurations >> n'existe pas comme grandeur informative : il vaut la")
print("configuration integralement non conforme, quelle que soit la structure de dependance et")
print("quelles que soient les marges tant qu'elles restent a leur ancrage. La loi des")
print(f"configurations ne porte que {2 ** N} atomes ; on ne peut pas lui demander la resolution qu'on")
print("demande a une loi de perte. C'est une raison de plus de ne jamais lui appliquer la macro")
print("\\VaR, qui est reservee a la charge annuelle agregee.")


# ---------------------------------------------------------------------------
# 3. La mesure qui bouge continument : les probabilites de depassement
# ---------------------------------------------------------------------------

titre("3. LA MESURE QUI NE SAUTE PAS : LES PROBABILITES DE DEPASSEMENT")
print("Pour eviter l'escalier, on lit la loi a NIVEAUX de capital fixes plutot qu'a probabilites")
print("fixees. La probabilite de depasser un niveau donne varie continument avec delta, et c'est")
print("la statistique a citer quand on veut parler de la queue de la loi des configurations.")
print()
seuils = [e0 * f for f in (1.0, 1.25, 1.50, 1.75)]
print("  niveau de capital " + "".join(f"   delta {d:.2f}" for d in DELTAS))
for s in seuils:
    ps = [float(np.sum(res[d]["w"][scr >= s])) for d in DELTAS]
    print(f"  {fnum(s):>7} M EUR (x{s/e0:.2f})" + "".join(f"   {100*p:9.2f} %" for p in ps))
print()
print("ET LES DEUX EXTREMITES DE LA LOI DES CONFIGURATIONS, qui sont la lecture la plus directe")
print("de ce que fait la correlation :")
m_tous_c, m_tous_nc = 0, 2 ** N - 1
for etiq, m in (("tous conformes    ", m_tous_c), ("tous non conformes", m_tous_nc)):
    ps = [100 * float(res[d]["w"][m]) for d in DELTAS]
    print(f"  P({etiq})" + "".join(f"   {p:9.2f} %" for p in ps))


# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------

titre("VERDICT")
print("Ecrit APRES lecture des sorties.")
print()
p75 = 100 * (b["qs"][1] / a["qs"][1] - 1)
p90 = 100 * (b["qs"][2] / a["qs"][2] - 1)
pcte = 100 * (b["cte"] / a["cte"] - 1)
pe = 100 * (b["e"] / a["e"] - 1)
pc0 = 100 * float(res[DELTAS[0]]["w"][2 ** N - 1])
pc1 = 100 * float(res[DELTAS[-1]]["w"][2 ** N - 1])
q0 = 100 * float(np.sum(res[DELTAS[0]]["w"][scr >= e0 * 1.25]))
q1 = 100 * float(np.sum(res[DELTAS[-1]]["w"][scr >= e0 * 1.25]))
print("1. LA RESERVE DU SCRIPT 69 ETAIT JUSTIFIEE, ET ELLE EST MAINTENANT CHIFFREE. L'esperance")
print(f"   du capital ne bouge que de {pe:+.2f} % sur tout le balayage, ce qui reproduit le")
print("   resultat publie ; les quantiles de la loi des configurations, eux, bougent :")
print(f"   {p75:+.2f} % au niveau de 75 %, {p90:+.2f} % a 90 %, et {pcte:+.2f} % sur la moyenne "
      "de queue a 90 %.")
print("   L'argument d'additivite ne transporte donc PAS a un quantile, et il ne faut pas")
print("   presenter l'invariance mesuree sur l'esperance comme une invariance du modele.")
print()
print("2. MAIS L'ORDRE DE GRANDEUR RESTE PETIT, ET C'EST LE SECOND RESULTAT. Quelques pour cent,")
print("   contre deux centiemes de pour cent pour l'esperance : l'ecart entre les deux objets est")
print("   de deux ordres de grandeur, et pourtant les deux restent loin des incertitudes que le")
print("   memoire declare par ailleurs. La reserve du script 69 se referme donc en un chiffre")
print("   plutot qu'en une inquietude.")
print()
print("3. LE MECANISME EST LA POLARISATION, ET C'EST LA MESURE LA PLUS ROBUSTE DU SCRIPT. A")
print("   esperance inchangee, plus de correlation rend les configurations extremes plus")
print(f"   probables des DEUX cotes : P(tous conformes) passe de "
      f"{100*float(res[DELTAS[0]]['w'][0]):.2f} % a "
      f"{100*float(res[DELTAS[-1]]['w'][0]):.2f} % et")
print(f"   P(tous non conformes) de {pc0:.2f} % a {pc1:.2f} %. La probabilite de depasser "
      "1,25 fois le capital")
print(f"   espere monte de {q0:.2f} % a {q1:.2f} %, soit {100*(q1/q0 - 1):+.1f} % en relatif. Ces")
print("   grandeurs varient CONTINUMENT, contrairement aux quantiles, et ce sont donc elles")
print("   qu'il faut citer pour parler de la queue de la loi des configurations.")
print()
print("4. ET UNE LIMITE DE L'OBJET, TROUVEE EN CHEMIN. Au-dela du niveau imprime en section 2bis,")
print("   le quantile de la loi des configurations EST la configuration integralement non")
print("   conforme et ne peut plus bouger. Les colonnes a 95 % et 99 % du tableau sont donc")
print("   SATUREES et non invariantes, et un quantile a 99,5 % de cette loi n'aurait aucun")
print("   contenu. Une loi a 32 atomes ne se lit pas comme une loi de perte.")
print()
print("CE QUE CE SCRIPT NE DIT PAS.")
print("  - le surcroit de correlation n'est pas CALIBRE : aucune donnee du projet ne l'estime, et")
print("    le balayage donne un ordre de grandeur, pas une correction. Ce qui est robuste est que")
print("    la borne de positivite de la matrice ferme l'exercice par le haut ;")
print("  - le deplacement d'un quantile discret est un CHANGEMENT D'ATOME, donc un changement de")
print("    la configuration qui occupe ce rang, et non la variation continue d'une valeur. Sa")
print("    direction suit de la polarisation, qui est mesuree directement au point 3 ;")
print("  - la grandeur calculee ici est un quantile de l'incertitude sur l'ETAT de conformite,")
print("    JAMAIS une mesure de capital. Elle ne porte ni \\VaR ni \\TVaR, et la confondre avec")
print("    l'exigence de solvabilite ferait lire une incertitude d'etat comme une exigence.")

print()
print("GRANDEURS CITEES PAR LE MEMOIRE, SANS SEPARATEUR DE MILLIERS")
print("Aucun calcul nouveau : les memes valeurs, sans l'espace de milliers que l'extracteur du")
print("harnais coupe en deux.")
print(f"  esperance a surcroit nul                {a['e']:.0f}")
print(f"  quantile 75 %, surcroit nul             {a['qs'][1]:.0f}")
print(f"  quantile 75 %, surcroit maximal         {b['qs'][1]:.0f}")
print(f"  quantile 90 %, surcroit nul             {a['qs'][2]:.0f}")
print(f"  quantile 90 %, surcroit maximal         {b['qs'][2]:.0f}")
print(f"  moyenne de queue a 90 %, surcroit nul   {a['cte']:.0f}")
print(f"  moyenne de queue a 90 %, maximal        {b['cte']:.0f}")
print(f"  capital de la configuration extreme     {scr.max():.0f}")
print(f"  capital de la configuration conforme    {scr.min():.0f}")

print()
print("EXIT 0")
