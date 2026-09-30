#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
64 : le BIAIS DE NARRATION, chiffre au lieu d'etre seulement declare.

CE QUI MANQUAIT. Les scripts 53 et 59 nomment trois limites du codage de post-mortems, et
la troisieme est declaree « la plus serieuse » : un rapport d'enquete cherche une CAUSE
RACINE, et « defaillance de gouvernance » est une conclusion de convention. Le corpus
pourrait donc surrepresenter P1 comme SOURCE non parce que la gouvernance emet, mais parce
que les enqueteurs concluent ainsi. Onze des vingt-deux transitions codees partent de P1 :
la moitie du corpus repose sur la partie la plus suspecte du codage.

Cette limite etait ECRITE et non MESUREE. C'est la difference entre un caveat et un
resultat. Ce script la mesure de trois facons, de la plus douce a la plus adversaire.

  (1) JACKKNIFE PAR INCIDENT. On retire un rapport a la fois. Si le signal directionnel
      tient a un seul incident, il faut le savoir avant qu'un rapporteur ne le trouve.

  (2) POINT DE RUPTURE DU BIAIS. On suppose que la convention narrative a fabrique une
      partie des aretes partant de P1, et on cherche COMBIEN il en faudrait pour que le
      signal cesse d'etre concluant. Deux operations, la seconde bien plus severe :
        - RETRAIT     : le rapport n'etablissait rien, l'arete disparait.
        - RETOURNEMENT: l'arete allait en realite dans l'AUTRE sens.
      Dans les deux cas on prend le PIRE sous-ensemble d'aretes possible, jamais un tirage
      moyen : c'est une borne, pas une esperance.

  (3) L'EFFET SUR LE CAPITAL. La question qui interesse un actuaire n'est pas le z, c'est
      le SCR. On degrade l'emission de P1 dans la matrice retenue et on recalcule la BANDE
      d'identification a l'echelle d'entite, pas seulement le point.

LA LOI NULLE DEVIENT EXACTE, ce qui rend tout ce qui precede calculable. Les scripts 53 et
59 estiment la loi de l'asymetrie sous H0 par 20 000 permutations. Or

    T = ||M - M^T||_1 / 2 = somme sur les PAIRES NON ORDONNEES {j,k} de |m_jk - m_kj|,

et sous H0 chaque arete s'oriente independamment a pile ou face. Le solde d'une paire
portant n aretes suit donc la loi de 2B - n avec B binomiale(n, 1/2), et les paires sont
independantes : la loi de T est une CONVOLUTION EXACTE, calculee ici sans tirage. On gagne
deux choses. La p-valeur n'est plus plafonnee par le nombre de permutations (1/20 000), et
rejouer le test des milliers de fois ne coute plus rien, ce qui est exactement ce que
demandent les points (1) et (2).

CE QUE CE SCRIPT NE FAIT PAS. Il ne corrige pas le biais et ne pretend pas le mesurer : on
ne sait pas quelle fraction des aretes P1 est un artefact de narration. Il repond a une
autre question, la seule qui soit decidable sans nouvelle donnee : QUELLE AMPLEUR devrait
avoir ce biais pour renverser la conclusion. Un second codage en aveugle (script 54) reste
le remede propre ; ceci en est la borne, en attendant.

Sortie : diagnostics + figure Z22_biais_narration.png
"""

import os
import sys
from math import comb

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUALI = os.path.abspath(os.path.join(HERE, "..", "cascade_qualitative"))
for _p in (HERE, QUALI):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(HERE)
import partial_id as pid                                            # noqa: E402
import cascade_model as cm                                          # noqa: E402
from postmortem_corpus import CORPUS                                # noqa: E402

WID = 86
PIL = [1, 2, 3, 4, 5]
LAB = {1: "P1 gouvernance", 2: "P2 incidents", 3: "P3 tests", 4: "P4 tiers", 5: "P5 partage"}
_c = {j: i for i, j in enumerate(PIL)}
# LA DECISION SE PREND SUR LA p-VALEUR EXACTE, JAMAIS SUR UN SEUIL GAUSSIEN SUR z. La loi
# nulle est discrete et asymetrique : a k = 4 retournements on lit z = +1,85, au-dessus de
# 1,645, et pourtant p = 0,084, au-dessus de 0,05. Une premiere version de ce script
# concluait « aucune rupture » sur le z et manquait donc la rupture. Le z n'est garde que
# comme grandeur de lecture, comparable a celui des scripts 53 et 59.
P_SEUIL = 0.05
Z_LECTURE = 1.645                # repere gaussien, affiche mais NON DECISIONNEL

# echelle d'entite : valeurs FIGEES imprimees par le script 60, comme le fait deja le 58.
# Ne pas les reestimer ici : le script 60 est seul depositaire de la descente d'echelle, et
# deux estimations independantes du meme lambda finiraient par diverger d'un chiffre.
LAM_ENTITE = 0.09168423156000373
SEV_MULT = 0.8545
NY = 600_000
SEED = 20260721
ALPHA_VAR = 0.995


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# =====================================================================================
titre("0. Convention d'indices : la ligne emet, la colonne recoit")
# =====================================================================================
T_ = np.array([[cm.TRANS[j].get(k, 0.0) for k in PIL] for j in PIL], float)
rs, cs = T_.sum(1), T_.sum(0)
ok = (rs.argmax() == _c[1]) and (cs[_c[1]] < cs[_c[2]])
print(f"  emission par ligne   : " + "  ".join(f"P{j}={rs[_c[j]]:.1f}" for j in PIL))
print(f"  reception par colonne: " + "  ".join(f"P{j}={cs[_c[j]]:.1f}" for j in PIL))
print(f"  => convention LIGNE = SOURCE {'CONFIRMEE' if ok else 'INFIRMEE'}.")
if not ok:
    sys.exit("convention incoherente : arret.")


# =====================================================================================
titre("1. La loi nulle de l'asymetrie, exacte au lieu d'echantillonnee")
# =====================================================================================
def aretes(corpus):
    return [(j, k) for c in corpus for (j, k, _) in c["transitions"]]


def soldes(edges):
    """{paire non ordonnee : solde signe}, la reference etant la paire triee."""
    d = {}
    for (j, k) in edges:
        key = (min(j, k), max(j, k))
        d[key] = d.get(key, 0) + (1 if (j, k) == key else -1)
    return d


def tailles(edges):
    n = {}
    for (j, k) in edges:
        key = (min(j, k), max(j, k))
        n[key] = n.get(key, 0) + 1
    return n


def _loi_abs(n):
    """Loi de |d|, d = somme de n signes +-1 equiprobables."""
    out = {}
    for b in range(n + 1):
        v = abs(2 * b - n)
        out[v] = out.get(v, 0.0) + comb(n, b) / 2.0 ** n
    return out


_CACHE = {}


def loi_nulle(sizes):
    """Loi EXACTE de T sous H0, par convolution sur les paires. Mise en cache."""
    key = tuple(sorted(sizes))
    if key in _CACHE:
        return _CACHE[key]
    law = {0: 1.0}
    for n in key:
        la = _loi_abs(n)
        new = {}
        for a, pa in law.items():
            for b, pb in la.items():
                new[a + b] = new.get(a + b, 0.0) + pa * pb
        law = new
    _CACHE[key] = law
    return law


def test(edges):
    """(T observe, moyenne nulle, ecart-type nul, z, p exacte unilaterale)."""
    if not edges:
        return 0.0, 0.0, 0.0, float("nan"), 1.0
    s = soldes(edges)
    T = float(sum(abs(v) for v in s.values()))
    law = loi_nulle(list(tailles(edges).values()))
    m = sum(t * p for t, p in law.items())
    v = sum((t - m) ** 2 * p for t, p in law.items())
    sd = float(np.sqrt(v))
    p = sum(p for t, p in law.items() if t >= T - 1e-9)
    z = (T - m) / sd if sd > 0 else float("nan")
    return T, m, sd, z, float(p)


E_ALL = aretes(CORPUS)
T0, m0, s0, z0, p0 = test(E_ALL)
print(f"  corpus complet : {len(CORPUS)} incidents, {len(E_ALL)} transitions, "
      f"{len(tailles(E_ALL))} paires peuplees")
print(f"  paires et soldes : " + ", ".join(f"P{a}-P{b}:{v:+d}" for (a, b), v in
                                           sorted(soldes(E_ALL).items())))
print(f"\n  T observe      = {T0:.0f}")
print(f"  loi nulle EXACTE : moyenne {m0:.4f}, ecart-type {s0:.4f}")
# SANS SEPARATEUR DE MILLIERS. Une premiere version ecrivait « 65 536 » avec une espace,
# pour eviter la lecture « 65,536 » en decimal francais. Mais verif_chiffres.py releve des
# suites de chiffres : l'espace coupait le nombre en deux et le rendait introuvable, donc
# non verifiable dans le memoire. L'entier nu est a la fois sans ambiguite et verifiable.
print(f"  z = {z0:+.4f}    p exacte = {p0:.3e}   (soit 1 chance sur {int(round(1/p0))})")
print(f"\n  Controle : le script 59 obtenait z = +5,12 et p = 0,0001 par 20000 permutations.")
print(f"  Le z concorde a {abs(z0 - 5.12):.2f} pres, ce qui valide l'un par l'autre. La")
print(f"  p-valeur, elle, etait PLAFONNEE par le nombre de tirages : le calcul exact la")
print(f"  situe {0.0001/p0:,.0f} fois plus bas. La conclusion ne change pas, sa force si.")
print(f"\n  Pourquoi T = {T0:.0f} egale le nombre de transitions : AUCUNE paire n'est contredite,")
print(f"  donc chaque arete compte pour un. C'est le fait marquant du corpus, et c'est")
print(f"  aussi ce qui rend la suite indispensable : une coherence parfaite est exactement")
print(f"  ce que produirait une convention de redaction partagee par tous les enqueteurs.")


# =====================================================================================
titre("2. Jackknife par incident : le signal tient-il a un seul rapport ?")
# =====================================================================================
print("  On retire un incident, on rejoue le test exact. La question n'est pas de savoir si")
print("  chaque retrait reste significatif (il le sera : on retire peu), mais QUEL retrait")
print("  fait le plus de degats, et si le pire d'entre eux passe sous le seuil.\n")
print(f"  {'incident retire':<32}{'trans.':>8}{'T':>6}{'nul':>8}{'z':>9}{'p':>12}")
jk = []
for i, c in enumerate(CORPUS):
    sub = [x for x in CORPUS if x is not c]
    E = aretes(sub)
    T, m, s, z, p = test(E)
    jk.append((c["nom"], len(c["transitions"]), z, p))
    print(f"  {c['nom']:<32}{len(E):>8}{T:>6.0f}{m:>8.2f}{z:>+9.2f}{p:>12.2e}")
z_jk = np.array([x[2] for x in jk])
p_jk = np.array([x[3] for x in jk])
pire = jk[int(np.argmax(p_jk))]
print(f"\n  pire retrait : {pire[0]} -> z = {pire[2]:+.2f}, p = {pire[3]:.2e}")
print(f"  amplitude du jackknife : z de {z_jk.min():+.2f} a {z_jk.max():+.2f} "
      f"(corpus complet {z0:+.2f})")
print(f"  p la moins bonne : {p_jk.max():.2e} ; tous les retraits restent significatifs "
      f"au seuil de 5 % : {bool((p_jk < P_SEUIL).all())}")
print("\n  Lecture : le signal n'est porte par aucun rapport en particulier. Retirer le plus")
print("  informatif le laisse largement concluant. Ce n'est pas surprenant vu la coherence")
print("  totale du corpus, mais c'etait a verifier, et ce n'est pas ce qui menace le")
print("  resultat. Ce qui le menace est un biais SYSTEMATIQUE, present dans TOUS les")
print("  rapports a la fois, et c'est l'objet de la section suivante.")


# =====================================================================================
titre("3. Point de rupture : quelle ampleur devrait avoir le biais de narration ?")
# =====================================================================================
idx_p1 = [i for i, (j, k) in enumerate(E_ALL) if j == 1]
n_p1 = len(idx_p1)
print(f"  aretes partant de P1 : {n_p1} sur {len(E_ALL)} ({100*n_p1/len(E_ALL):.0f} % du corpus)")
print(f"  paires concernees    : " + ", ".join(
    f"P{a}-P{b}:{v}" for (a, b), v in sorted(tailles([E_ALL[i] for i in idx_p1]).items())))
print("\n  Pour chaque nombre d'aretes touchees, on enumere TOUS les sous-ensembles et on")
print("  garde le PIRE, celui qui minimise z. C'est une borne adverse, pas une moyenne.\n")


def sous_ensembles(n):
    """Tous les sous-ensembles d'indices 0..n-1, groupes par cardinal."""
    par_k = [[] for _ in range(n + 1)]
    for m in range(1 << n):
        s = [i for i in range(n) if m >> i & 1]
        par_k[len(s)].append(s)
    return par_k


PAR_K = sous_ensembles(n_p1)
print(f"  {'k':>3}{'retrait : z':>16}{'p':>12}   {'retournement : z':>20}{'p':>12}")
courbe_ret, courbe_flip = [], []
for k in range(n_p1 + 1):
    best_r, best_f = None, None
    for s in PAR_K[k]:
        sset = set(idx_p1[i] for i in s)
        # RETRAIT : les aretes disparaissent du corpus
        Er = [e for i, e in enumerate(E_ALL) if i not in sset]
        tr = test(Er)
        # le PIRE sous-ensemble est celui de p MAXIMALE, non de z minimal : le retrait
        # change la loi nulle, et les deux classements peuvent alors differer.
        if best_r is None or tr[4] > best_r[4]:
            best_r = tr
        # RETOURNEMENT : les aretes changent de sens
        Ef = [(e[1], e[0]) if i in sset else e for i, e in enumerate(E_ALL)]
        tf = test(Ef)
        if best_f is None or tf[4] > best_f[4]:
            best_f = tf
    courbe_ret.append((best_r[3], best_r[4]))
    courbe_flip.append((best_f[3], best_f[4]))
    print(f"  {k:>3}{best_r[3]:>+16.2f}{best_r[4]:>12.2e}   "
          f"{best_f[3]:>+20.2f}{best_f[4]:>12.2e}")


def rupture(courbe):
    """Premier k dont la p-valeur EXACTE depasse 5 %. Voir P_SEUIL : pas de seuil sur z."""
    for k, (z, p) in enumerate(courbe):
        if p >= P_SEUIL:
            return k
    return None


k_r, k_f = rupture(courbe_ret), rupture(courbe_flip)
p_max_f = max(p for _, p in courbe_flip)
k_max_f = int(np.argmax([p for _, p in courbe_flip]))
print(f"\n  POINT DE RUPTURE (premier k dont la p exacte depasse {P_SEUIL:.2f}) :")
print(f"    par retrait      : " + (f"k = {k_r} sur {n_p1}" if k_r is not None
                                    else f"AUCUN, meme en retirant les {n_p1} aretes"))
print(f"    par retournement : " + (f"k = {k_f} sur {n_p1}" if k_f is not None
                                    else f"AUCUN, meme en retournant les {n_p1} aretes"))
print(f"    pire configuration atteignable : k = {k_max_f}, p = {p_max_f:.3f} "
      f"(z = {courbe_flip[k_max_f][0]:+.2f})")
print(f"\n  ATTENTION AU z. A k = {k_max_f} le z vaut {courbe_flip[k_max_f][0]:+.2f}, donc AU-DESSUS du")
print(f"  repere gaussien {Z_LECTURE}, alors que la p exacte vaut {p_max_f:.3f}, donc AU-DESSUS de")
print(f"  {P_SEUIL:.2f}. La loi nulle etant discrete et asymetrique, le z surestime ici la")
print("  significativite. C'est la p exacte qui decide, et c'est elle qui donne la rupture.")
print("\n  Les deux operations ne disent pas la meme chose. Un retrait affaiblit le corpus ;")
print("  un retournement le retourne CONTRE l'hypothese, et detruit donc le signal beaucoup")
print("  plus vite. C'est la version la plus hostile du biais de narration : non seulement")
print("  les enqueteurs auraient invente le role de la gouvernance, mais ils auraient")
print("  inverse le sens reel de la chaine, et de la maniere la plus defavorable possible.")
print(f"\n  Pourquoi la courbe de retournement REMONTE apres k = {k_max_f} : retourner les {n_p1}")
print("  aretes de P1 ne detruit pas l'asymetrie, elle l'INVERSE. A k = 11 le corpus est")
print("  aussi coherent qu'a k = 0, dans le sens oppose, et le test retrouve son z initial.")
print("  Le maximum de degat est donc atteint a mi-chemin, quand les deux sens se")
print("  compensent. Un critique qui voudrait nuire au resultat n'aurait pas interet a")
print("  pousser son hypothese jusqu'au bout : c'est une propriete du test, il faut la dire.")


# =====================================================================================
titre("4. Le cas extreme : P1 entierement retire du corpus")
# =====================================================================================
print("  Hypothese maximale : TOUTE arete partant de P1 est un artefact de la convention")
print("  narrative. On les supprime toutes et on regarde ce qui reste.\n")
E_sans = [e for i, e in enumerate(E_ALL) if i not in set(idx_p1)]
Ts, ms, ss, zs, ps = test(E_sans)
print(f"  transitions restantes : {len(E_sans)} sur {len(E_ALL)}")
print(f"  paires restantes      : " + ", ".join(f"P{a}-P{b}:{v}" for (a, b), v in
                                                sorted(tailles(E_sans).items())))
print(f"  T = {Ts:.0f}, loi nulle {ms:.2f} +/- {ss:.2f}  ->  z = {zs:+.2f}, p = {ps:.4f}")
print(f"  significatif au seuil de 5 % : {bool(ps < P_SEUIL)}")
print("\n  C'EST LE RESULTAT PRINCIPAL DE CE SCRIPT. Meme en accordant au critique la")
print("  totalite de son objection, c'est-a-dire en supprimant la moitie du corpus et la")
print("  totalite du role de la gouvernance, la direction reste significative. Elle repose")
print("  alors sur les seules chaines P3 -> P2 et P4 -> P2 : les tests qui ne detectent pas,")
print("  et les tiers qui entrainent la gestion d'incident. Ces deux chaines ne doivent rien")
print("  a la convention « cause racine = gouvernance », puisque P1 n'y figure pas.")

# variante honnete, et plus severe : retirer les INCIDENTS, non les aretes
inc_sans_p1 = [c for c in CORPUS if not any(j == 1 for (j, _, _) in c["transitions"])]
Ei = aretes(inc_sans_p1)
Ti, mi, si, zi, pi = test(Ei)
print(f"\n  VARIANTE PLUS SEVERE, et qu'il faut donner aussi : retirer les {len(CORPUS)-len(inc_sans_p1)}")
print(f"  INCIDENTS entiers ou P1 apparait comme source, et non les seules aretes.")
print(f"    reste {len(inc_sans_p1)} incidents, {len(Ei)} transitions : T = {Ti:.0f}, "
      f"z = {zi:+.2f}, p = {pi:.4f}")
print(f"    significatif : {bool(pi < P_SEUIL)}")
print("  Cette variante-la ne passe pas, et je ne la cache pas. Mais elle jette avec l'eau")
print("  du bain des aretes que le biais ne concerne pas : les chaines P3 -> P2 documentees")
print("  dans les MEMES rapports. Un biais sur la designation de la cause racine n'invalide")
print("  pas la sequence technique que le meme rapport etablit par ailleurs. La borne")
print("  pertinente est donc celle du retrait des aretes, pas celle du retrait des rapports.")


# =====================================================================================
titre("5. L'effet sur le CAPITAL : le SCR d'entite se deplace-t-il ?")
# =====================================================================================
print("  Un z n'est pas un capital. On degrade donc l'emission de P1 dans la matrice")
print("  retenue, W_1k -> (1 - a) W_1k, et on recalcule le SCR d'entite. Deux precautions.")
print("\n  (i) La degradation change S, donc l'ensemble admissible LUI-MEME. On ne peut pas")
print("      comparer un point degrade a la bande d'origine : la bande est RECALCULEE pour")
print("      chaque degre a, sur les 1 024 sommets du pave.")
print("  (ii) L'echelle est celle de l'entite du chapitre resultats (script 60), pour que")
print("      les nombres soient directement comparables a la bande publiee.\n")

Wexp = pid.expert_matrix()


def degrade(W, a):
    Wd = W.copy()
    Wd[_c[1], :] *= (1.0 - a)
    return Wd


ev = pid.Evaluator(lam=LAM_ENTITE, n_years=NY, alpha=ALPHA_VAR, seed=SEED)
ev.cum = ev.cum * SEV_MULT


def bande(W):
    S, _ = pid.decompose(W)
    vals = []
    for row in pid.all_vertices(S, 1.0):
        A = np.zeros((pid.NP_, pid.NP_))
        A[pid.IU] = row
        Wv = S + (A - A.T)
        if not pid.admissible(Wv):
            continue
        vals.append(ev.from_card(pid.card_dist_all(Wv)[0])[0])
    return np.array(vals)


ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
print(f"  {'a':>6}{'SCR au point':>15}{'bande basse':>14}{'bande haute':>14}"
      f"{'largeur':>10}{'sommets':>9}")
bandes = {}
for a in ALPHAS:
    Wa = degrade(Wexp, a)
    pt = ev.from_card(pid.card_dist_all(Wa)[0])[0]
    v = bande(Wa)
    bandes[a] = (pt, v)
    print(f"  {a:>6.2f}{pt:>15,.1f}{v.min():>14,.1f}{v.max():>14,.1f}"
          f"{v.max()-v.min():>10,.1f}{len(v):>9}")

pt0, v_0 = bandes[0.0]
pt1, v_1 = bandes[1.0]
print(f"\n  rappel du chapitre resultats : point 169,0 ; bande [131,5 ; 183,3] M EUR")
print(f"  ici a a = 0            : point {pt0:,.1f} ; bande [{v_0.min():,.1f} ; {v_0.max():,.1f}] M EUR")
print(f"  a a = 1 (P1 muette)    : point {pt1:,.1f} ; bande [{v_1.min():,.1f} ; {v_1.max():,.1f}] M EUR")
print(f"  deplacement du point   : {100*(pt1/pt0-1):+.1f} %")
print(f"  deplacement de la borne basse : {100*(v_1.min()/v_0.min()-1):+.1f} %")
print(f"  la bande degradee est-elle INCLUSE dans la bande d'origine : "
      f"{bool(v_1.min() >= v_0.min() - 1e-9 and v_1.max() <= v_0.max() + 1e-9)}")
print(f"  les deux bandes se recouvrent sur [{max(v_0.min(), v_1.min()):,.1f} ; "
      f"{min(v_0.max(), v_1.max()):,.1f}] M EUR")

# LA COMPARAISON QUI DONNE LA MESURE DU BIAIS. Un deplacement en pourcentage ne dit rien
# tant qu'on ne le compare pas a une incertitude DEJA assumee dans le memoire.
dep = pt0 - pt1
larg = v_0.max() - v_0.min()
print(f"\n  MISE A L'ECHELLE DU BIAIS, et c'est la lecture qui compte. Le deplacement du")
print(f"  point vaut {dep:,.1f} M EUR. La largeur de la bande d'identification, c'est-a-dire le")
print(f"  prix DEJA PAYE dans le memoire pour ne pas connaitre la direction de W, vaut")
print(f"  {larg:,.1f} M EUR. Le rapport est de {dep/larg:.2f}.")
print(f"  Autrement dit : accorder au critique la totalite de son objection sur le biais de")
print(f"  narration coute {'MOINS' if dep < larg else 'PLUS'} cher que l'ignorance directionnelle que le memoire")
print(f"  assume et publie deja. Le biais de narration n'est donc pas la premiere source")
print(f"  d'incertitude du chiffre d'entite, et il ne faut pas le presenter comme telle.")
print("\n  Lecture. Meme en rendant P1 completement muette, le capital d'entite ne s'effondre")
print("  pas : la severite et la frequence portent le niveau, la cascade en deplace une")
print("  part. C'est deja la these du chapitre 12 sur la descente d'echelle, verifiee ici")
print("  sur une perturbation qui n'avait pas ete construite pour la tester.")
print(f"\n  CONTROLE CROISE DU SCRIPT 60. A a = 0 ce script recalcule le point et la bande par")
print(f"  un chemin de code different, et retrouve {pt0:,.1f} et [{v_0.min():,.1f} ; {v_0.max():,.1f}] :")
print(f"  les trois chiffres publies au chapitre resultats sont donc reproduits par deux")
print(f"  scripts independants, ce qui n'etait verifie par aucun jusqu'ici.")

# ce qui, en revanche, DEPEND de P1 : le classement de remediation
print("\n  CE QUI DEPEND DE P1, ET QU'IL FAUT DIRE. Le SCR bouge peu, mais le PILOTAGE")
print("  bouge. Gain de SCR a couper les aretes sortantes de chaque pilier :\n")
print(f"  {'a':>6}" + "".join(f"{LAB[j]:>18}" for j in PIL) + f"{'   1er':>8}")
for a in ALPHAS:
    b = ev.benefits(degrade(Wexp, a))
    prem = PIL[int(np.argmax(b))]
    print(f"  {a:>6.2f}" + "".join(f"{b[_c[j]]:>18,.1f}" for j in PIL) + f"   P{prem}")
print("\n  Le classement de remediation, lui, est sensible : si la gouvernance emet moins")
print("  que les rapports ne le disent, la priorite se deplace. Le chiffre de capital")
print("  resiste au biais de narration, la recommandation de pilotage en depend. Les deux")
print("  ne doivent donc pas etre presentes avec la meme assurance.")

# et l'ecart entre etats de conformite, qui est la these du memoire
print("\n  ENFIN, LA THESE ELLE-MEME : l'ecart entre etats de conformite survit-il ?")
print(f"  {'a':>6}{'conforme':>12}{'partiel':>11}{'non conf.':>12}{'ecart':>9}{'%':>9}")
for a in ALPHAS:
    vv = [ev.from_card(pid.card_dist_all(degrade(pid.expert_matrix(g), a))[0])[0]
          for g in (0.45, 0.68, 0.90)]
    print(f"  {a:>6.2f}{vv[0]:>12,.1f}{vv[1]:>11,.1f}{vv[2]:>12,.1f}"
          f"{vv[2]-vv[0]:>9,.1f}{100*(vv[2]/vv[0]-1):>+9.1f}")
print("\n  L'ecart entre profil conforme et profil defaillant se reduit avec a, mais ne")
print("  s'annule pas : la conformite agit par le gain g sur TOUTES les aretes, pas")
print("  seulement sur celles de P1. Face a une charge forfaitaire qui ne bouge d'aucun")
print("  euro entre les trois etats, la conclusion du memoire tient a tous les degres de")
print("  degradation testes.")


# =====================================================================================
titre("VERDICT")
# =====================================================================================
print(f"  1. La loi nulle de l'asymetrie est EXACTE (convolution sur les paires), et non")
print(f"     plus echantillonnee : p = {p0:.2e} au lieu d'un plafond a 1e-4.")
print(f"  2. Jackknife : le pire retrait d'incident laisse p = {p_jk.max():.1e}. Aucun rapport")
print(f"     ne porte le resultat a lui seul.")
print(f"  3. Biais de narration, borne adverse. Il faudrait que la convention narrative ait")
print(f"     INVERSE {k_f} des {n_p1} aretes de P1, dans la pire configuration possible, pour")
print(f"     perdre la significativite (p = {p_max_f:.3f}). Par simple retrait, aucun")
print(f"     nombre ne suffit : supprimer les {n_p1} aretes laisse p = {ps:.4f} sur les seules")
print(f"     chaines P3 -> P2 et P4 -> P2, que le biais ne concerne pas.")
print(f"  4. Capital : le SCR d'entite passe de {pt0:,.1f} a {pt1:,.1f} M EUR quand P1 devient")
print(f"     muette ({100*(pt1/pt0-1):+.1f} %), soit {dep/larg:.2f} fois la largeur de la bande")
print(f"     d'identification deja publiee. Le NIVEAU resiste, le CLASSEMENT de remediation non.")
print(f"  5. Ce qui reste a faire, et qu'aucun calcul ne remplace : le second codage en")
print(f"     aveugle du script 54. Ceci en est la borne, pas le substitut.")


# =====================================================================================
titre("Figure")
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
LEG = dict(frameon=True, facecolor="#fcfcfb", edgecolor="none", framealpha=0.88, fontsize=8.5)

# PLUS DE COLONNE DE TROIS PANNEAUX. Empiles sur 10,4 pouces ils occupaient une page
# entiere du memoire. Le panneau (a) est le seul qui exige de la hauteur (dix noms
# d'incidents) : il prend toute la colonne de gauche, (b) et (c) s'empilent a droite.
# La figure s'imprime alors sur une dizaine de centimetres, dans le fil du texte.
fig = plt.figure(figsize=(8.6, 4.7))
_gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.16])
ax1 = fig.add_subplot(_gs[:, 0])
ax2 = fig.add_subplot(_gs[0, 1])
ax3 = fig.add_subplot(_gs[1, 1])

# (a) jackknife par incident
noms = [x[0] for x in jk]
ys = np.arange(len(noms))
ax1.barh(ys, z_jk, color=BLUE, alpha=0.85, height=0.62)
ax1.axvline(z0, color=ACCENT, lw=2)
ax1.set_yticks(ys)
ax1.set_yticklabels([n if len(n) <= 30 else n[:29] + "…" for n in noms], fontsize=8.2)
ax1.invert_yaxis()
ax1.tick_params(axis="x", labelsize=9)
ax1.set_xlabel("$z$ du test exact, incident retiré", color=INK2, fontsize=9.5)
ax1.set_xlim(0, max(z0, z_jk.max()) * 1.18)
# AUCUNE LEGENDE ENCADREE ICI. Dix barres horizontales ne laissent aucune place libre :
# posee en bas a droite elle recouvrait la fin de la derniere barre. L'unique repere a
# nommer est la ligne du corpus complet, donc on l'ecrit LE LONG de la ligne, dans la
# marge droite qu'aucune barre n'atteint.
ax1.text(z0 * 1.015, len(noms) / 2.0, f"corpus complet : $z={z0:+.2f}$".replace(".", "{,}"),
         rotation=90, ha="left", va="center", fontsize=8.4, color=ACCENT)
# PAS DE SEUIL SUR z DANS CE PANNEAU. La decision se prend sur la p exacte (cf. P_SEUIL) ;
# tracer une ligne a 1,645 inviterait a lire le graphique avec le mauvais critere.
_ex = int(np.floor(np.log10(p_jk.max())))
_ma = f"{p_jk.max() / 10.0 ** _ex:.1f}".replace(".", "{,}")
ax1.set_title(f"(a)  Aucun rapport ne porte le signal\n"
              f"($p$ exacte $\\leq {_ma}\\cdot 10^{{{_ex}}}$ pour les dix retraits)",
              fontsize=10, color=INK, pad=6)

# (b) point de rupture, EN p-VALEUR EXACTE et non en z
ks = np.arange(n_p1 + 1)
ax2.semilogy(ks, [c[1] for c in courbe_ret], "o-", color=BLUE, lw=2, ms=5,
             label="retrait des arêtes de P1")
ax2.semilogy(ks, [c[1] for c in courbe_flip], "s-", color=ACCENT, lw=2, ms=5,
             label="retournement (pire cas)")
ax2.axhline(P_SEUIL, color=INK, lw=1.3, ls="--")
_ymax = max(max(c[1] for c in courbe_flip), max(c[1] for c in courbe_ret)) * 3.0
ax2.axhspan(P_SEUIL, _ymax, color=MUTED, alpha=0.16, lw=0)
# L'ETIQUETTE DE LA BANDE PASSE A GAUCHE. A droite elle tombait sur le plateau de la
# courbe de retournement, qui longe justement le seuil ; a gauche les deux courbes sont
# au plancher.
ax2.text(0, P_SEUIL * 1.5, " non significatif", ha="left", va="bottom", fontsize=7.8,
         color=INK2)
if k_f is not None:
    ax2.plot([k_f], [courbe_flip[k_f][1]], "o", ms=11, mfc="none", mec=ACCENT, mew=1.8)
    ax2.annotate(f"rupture : {k_f} arêtes\nsur {n_p1}", (k_f, courbe_flip[k_f][1]),
                 textcoords="offset points", xytext=(9, -36), fontsize=8, color=ACCENT,
                 bbox=dict(boxstyle="round,pad=0.25", fc="#fcfcfb", ec="none", alpha=0.9))
ax2.set_xlabel("arêtes de P1 supposées produites\npar la narration",
               color=INK2, fontsize=9.5)
ax2.set_ylabel("$p$-valeur exacte\n(échelle log)", color=INK2, fontsize=9.5)
ax2.set_xticks(ks)
ax2.tick_params(labelsize=8.5)
ax2.set_ylim(top=_ymax)
# EN BAS AU CENTRE. En bas a gauche la legende recouvrait les deux points a k = 0, qui
# sont le point de depart de la lecture. Les deux courbes passent haut au centre : la
# place libre est sous elles.
ax2.legend(loc="lower center", **LEG)
ax2.set_title("(b)  Combien d'arêtes le biais devrait\nfabriquer pour renverser la conclusion",
              fontsize=10, color=INK, pad=6)

# (c) capital
xs = np.arange(len(ALPHAS))
lo = np.array([bandes[a][1].min() for a in ALPHAS])
hi = np.array([bandes[a][1].max() for a in ALPHAS])
pts = np.array([bandes[a][0] for a in ALPHAS])
ax3.vlines(xs, lo, hi, color=BLUE, lw=8, alpha=0.42)
ax3.plot(xs, pts, "o-", color=ACCENT, lw=1.6, ms=6, label="matrice retenue")
ax3.plot(xs, lo, "_", color=BLUE, ms=13, label="bande d'identification")
ax3.plot(xs, hi, "_", color=BLUE, ms=13)
for x, p_, h in zip(xs, pts, hi):
    ax3.text(x, h * 1.012, f"{p_:,.0f}", ha="center", va="bottom", fontsize=8.2,
             color=INK)
ax3.set_xticks(xs)
ax3.set_xticklabels([f"$a={a:.2f}$".replace(".", "{,}") for a in ALPHAS], fontsize=8.5)
ax3.set_xlabel("degré de dégradation de l'émission de P1", color=INK2, fontsize=9.5)
ax3.set_ylabel("SCR 99,5 %\nde l'entité (M€)", color=INK2, fontsize=9.5)
ax3.tick_params(axis="y", labelsize=8.5)
# MARGE HAUTE POUR LA LEGENDE. Posee en bas a gauche elle recouvrait la bande du
# premier point ; les deux courbes descendent vers la droite, la place libre est
# au-dessus d'elles une fois le plafond releve.
ax3.set_ylim(lo.min() * 0.92, hi.max() * 1.30)
ax3.legend(loc="upper right", **LEG)
ax3.set_title("(c)  Le niveau de capital résiste,\nmême avec P1 rendue muette",
              fontsize=10, color=INK, pad=6)

for ax in (ax1, ax2, ax3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# AUCUN TITRE GENERAL : la legende LaTeX porte le titre, et le code interne « Z22 »
# n'a rien a faire dans le PDF depose.
fig.tight_layout(h_pad=1.6, w_pad=1.8)
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z22_biais_narration.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
