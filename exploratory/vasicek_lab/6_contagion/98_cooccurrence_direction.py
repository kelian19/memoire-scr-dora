#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
98 : ce que la co-occurrence voit de la direction de W, et ce qu'elle n'en voit pas.

POURQUOI CE SCRIPT EXISTE. La proposition 6.7 affirmait que la co-occurrence ne depend de W que
par sa partie symetrique S = (W + W^T)/2, donc que W et sa transposee sont indiscernables pour la
donnee. La preuve ne valait que pour les transferts DIRECTS : des qu'un chemin passe par un
troisieme pilier, ou que l'amorce n'est pas uniforme, la loi de l'ensemble atteint depend de la
partie antisymetrique A. Ce script le mesure sur la matrice publiee, pour les deux cascades du
projet :
  - la MARCHE auto-evitante a un successeur par pas, qui est le moteur publie
    (scr_engine.cascade_set_dist) ;
  - la cascade INDEPENDANTE de la definition 6.1 (scr_engine.cascade_set_dist_branching).

CE QUI RESTE VRAI, ET CE QUI NE L'EST PAS.
  (i)  au premier ordre en W, et sous amorce uniforme, la probabilite qu'un sinistre touche une
       paire {j, k} vaut r (W_jk + W_kj) = 2 r S_jk : elle ne voit que S (section 3) ;
  (ii) au-dela du premier ordre, ou sous amorce non uniforme, la loi de l'ensemble atteint
       distingue W de sa transposee (sections 2 et 4).
La direction echappe donc aux statistiques d'ordre un que les sources agregees permettent de
construire, pas a toute donnee : une source qui consignerait l'ensemble des piliers touches par
incident porterait de l'information sur A.

CONTROLE. Les deux lois exactes sont recalculees ici pour une matrice QUELCONQUE, et l'appel a la
matrice publiee doit redonner les fonctions du moteur a 1e-12 pres. Sans ce controle, on
demontrerait une propriete d'un autre modele.

Aucune simulation, aucune donnee, aucune figure : le script tourne sur les deux postes.
"""

import os
import sys
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import scr_engine as eng                                        # noqa: E402

WID = 88
PIL = eng.PIL                       # [1, 2, 3, 4, 5]
J = len(PIL)
G = eng.G_BASE                      # 0,90 : gain de l'etat non conforme, celui de la figure H1
TOL = 1e-12


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# ------------------------------------------------------------------ la matrice publiee
# W_jk = g TRANS[j][k] / max_j s_j, la ligne j decrivant ce que j EMET (convention du chapitre 6).
T = np.zeros((J, J))
for j in PIL:
    for k, v in eng.TRANS[j].items():
        T[j - 1, k - 1] = v
SMAX = float(T.sum(axis=1).max())
W = G * T / SMAX
WT = W.T.copy()


# ------------------------------------------------------------------ les deux lois exactes
def loi_marche(Wm, a):
    """Loi exacte de l'ensemble atteint par la marche auto-evitante, amorce a (indice 0..4).

    Depuis le pilier courant c, la marche passe a k avec la probabilite Wm[c, k] ; elle s'arrete
    avec la probabilite 1 - somme_k Wm[c, k], ou si k est deja tombe. C'est la regle de
    scr_engine.cascade_set_dist, ecrite pour une matrice quelconque.
    """
    dist = {}

    def rec(cur, visited, p):
        ligne = Wm[cur]
        fin = 1.0 - ligne.sum() + sum(ligne[k] for k in visited)
        cle = frozenset(visited)
        dist[cle] = dist.get(cle, 0.0) + p * fin
        for k in range(J):
            if k not in visited and ligne[k] > 0.0:
                rec(k, visited | {k}, p * ligne[k])

    rec(a, frozenset({a}), 1.0)
    return dist


def loi_branchement(Wm, a):
    """Loi exacte de l'ensemble atteint par la cascade independante (definition 6.1).

    P(S_a = A) = R(A) B(A), B(A) = prod_{j dans A, k hors A} (1 - W_jk), et R(A) par recursion
    sur les sous-ensembles propres contenant l'amorce : proposition 6.4.
    """
    cache = {}

    def R(A):
        if A in cache:
            return cache[A]
        if len(A) == 1:
            cache[A] = 1.0
            return 1.0
        reste = [x for x in A if x != a]
        tot = 0.0
        for r in range(len(reste)):
            for combo in combinations(reste, r):
                Ap = frozenset((a,) + combo)
                pr = 1.0
                for jj in Ap:
                    for kk in A - Ap:
                        pr *= 1.0 - Wm[jj, kk]
                tot += R(Ap) * pr
        cache[A] = 1.0 - tot
        return cache[A]

    dist = {}
    autres = [x for x in range(J) if x != a]
    for r in range(len(autres) + 1):
        for combo in combinations(autres, r):
            A = frozenset((a,) + combo)
            B = 1.0
            for jj in A:
                for kk in range(J):
                    if kk not in A:
                        B *= 1.0 - Wm[jj, kk]
            dist[A] = R(A) * B
    return dist


CASCADES = (("marche (moteur publie)", loi_marche, eng.cascade_set_dist),
            ("branchement (definition 6.1)", loi_branchement, eng.cascade_set_dist_branching))


def cooccurrences(fn, Wm, amorce):
    """P({j,k} inclus dans S) par paire, E|S| et P(|S| >= 2), l'amorce etant un vecteur de poids."""
    co = np.zeros((J, J))
    taille = multi = 0.0
    for a in range(J):
        d = fn(Wm, a)
        tot = sum(d.values())
        for s, p in d.items():
            p = amorce[a] * p / tot
            taille += p * len(s)
            if len(s) >= 2:
                multi += p
            for j, k in combinations(sorted(s), 2):
                co[j, k] += p
    return co, taille, multi


PAIRES = list(combinations(range(J), 2))
w_modele = np.array([eng.LAMBDA[j] for j in PIL], float)
w_modele = w_modele / w_modele.sum()
w_unif = np.full(J, 1.0 / J)
AMORCES = (("amorce du modele (proportionnelle a ROOT)", w_modele), ("amorce uniforme", w_unif))

# =====================================================================================
titre("1. Controle : les lois recalculees ici sont celles du moteur")
# =====================================================================================
print(f"  matrice publiee : g = {G:.2f}, diviseur d'emission max_s = {SMAX:.2f}, "
      f"rho(W) = {max(abs(np.linalg.eigvals(W))):.3f}")
for nom, fn, fn_moteur in CASCADES:
    err = 0.0
    for a in range(J):
        mine = {frozenset(PIL[i] for i in s): v for s, v in fn(W, a).items()}
        ref = fn_moteur(PIL[a], G)
        for s in set(mine) | set(ref):
            err = max(err, abs(mine.get(s, 0.0) - ref.get(s, 0.0)))
    assert err < TOL, f"{nom} : la loi recalculee s'ecarte du moteur de {err:.2e}"
    print(f"  {nom:<32} ecart maximal au moteur sur les cinq amorces : {err:.1e}  -> OK")
print("  Les deux fonctions ecrites pour une matrice quelconque reproduisent le moteur : ce qui")
print("  suit est une propriete du modele publie, pas d'un autre.")

# =====================================================================================
titre("2. La loi complete distingue W de sa transposee")
# =====================================================================================
print("  W et W^T ont la meme partie symetrique S. Si la co-occurrence ne dependait que de S,")
print("  les deux colonnes de chaque table seraient egales.")
resultats = {}
for nom_c, fn, _ in CASCADES:
    for nom_a, am in AMORCES:
        coW, tW, mW = cooccurrences(fn, W, am)
        coT, tT, mT = cooccurrences(fn, WT, am)
        resultats[(nom_c, nom_a)] = (coW, coT, tW, tT, mW, mT)
        print(f"\n  {nom_c}, {nom_a}")
        print(f"  {'paire':<10}{'P(co-occ | W)':>16}{'P(co-occ | W^T)':>18}{'ecart relatif':>16}")
        for j, k in PAIRES:
            print(f"  P{j+1}-P{k+1}{'':<5}{coW[j, k]:>16.4f}{coT[j, k]:>18.4f}"
                  f"{100 * (coT[j, k] / coW[j, k] - 1):>+15.1f} %")
        ecarts = [abs(coT[j, k] / coW[j, k] - 1) for j, k in PAIRES]
        print(f"  taille moyenne de l'ensemble atteint : {tW:.3f} sous W, {tT:.3f} sous W^T")
        print(f"  part des sinistres touchant plus d'un pilier : {100*mW:.2f} % sous W, "
              f"{100*mT:.2f} % sous W^T")
        print(f"  plus grand ecart relatif sur une paire : {100*max(ecarts):.1f} %")

print("\n  LECTURE. Sous l'amorce du modele, les deux cascades distinguent W de sa transposee sur")
print("  toutes les paires, et jusque dans la taille moyenne des sinistres et la part des")
print("  sinistres multi-piliers. Sous l'amorce uniforme, la taille moyenne ne bouge pour aucune")
print("  des deux cascades (pour la cascade independante, 'a atteint k' sous W^T equivaut a")
print("  'k atteint a' sous W, et la somme sur les couples est invariante) ; la loi des paires,")
print("  elle, bouge encore. La proposition 6.7 sous sa forme d'origine est donc fausse pour les")
print("  deux cascades du projet.")

# =====================================================================================
titre("3. Ce qui reste vrai : au premier ordre et sous amorce uniforme, seul S compte")
# =====================================================================================
# On contracte la matrice publiee, W_eps = eps W, et l'on suit l'ecart relatif maximal entre
# W_eps et sa transposee. S'il ne depend que de S au premier ordre, cet ecart tend vers zero
# avec eps sous amorce uniforme, et il ne tend PAS vers zero sous l'amorce du modele, ou le terme
# d'ordre un vaut r_j W_jk + r_k W_kj et non r (W_jk + W_kj).
EPS = (1.0, 0.5, 0.1, 0.02)
print(f"  {'eps':<8}", end="")
for nom_c, _, _ in CASCADES:
    for nom_a, _ in AMORCES:
        court = ("marche" if nom_c.startswith("marche") else "branch.") + \
                (" / modele" if nom_a.startswith("amorce du") else " / uniforme")
        print(f"{court:>22}", end="")
print()
premier_ordre = {}
for eps in EPS:
    print(f"  {eps:<8.2f}", end="")
    for nom_c, fn, _ in CASCADES:
        for nom_a, am in AMORCES:
            coW, _, _ = cooccurrences(fn, eps * W, am)
            coT, _, _ = cooccurrences(fn, eps * WT, am)
            e = max(abs(coT[j, k] / coW[j, k] - 1) for j, k in PAIRES)
            premier_ordre[(eps, nom_c, nom_a)] = e
            print(f"{100 * e:>20.2f} %", end="")
    print()
print("\n  LECTURE. Sous amorce uniforme l'ecart s'efface proportionnellement a eps : la")
print("  co-occurrence d'ordre un ne voit que S, et c'est ce qui subsiste de la proposition.")
print("  Sous l'amorce du modele il ne s'efface pas, le terme d'ordre un etant deja asymetrique.")

# =====================================================================================
titre("4. Le contre-exemple minimal, a la main")
# =====================================================================================
w3 = 0.5
F = np.zeros((3, 3))
F[0, 1] = F[0, 2] = w3            # fourche : P1 entraine P2 et P3
print(f"  Trois piliers, W12 = W13 = {w3} et rien d'autre ; la transposee est un collisionneur")
print("  (P2 et P3 entrainent P1). Meme partie symetrique. Cascade independante, amorce uniforme.")
for nom, M in (("fourche W", F), ("collisionneur W^T", F.T.copy())):
    co = np.zeros((3, 3))
    for a in range(3):
        # enumeration directe des 2^6 graphes d'aretes vivantes
        aretes = [(x, y) for x in range(3) for y in range(3) if x != y]
        for masque in range(1 << len(aretes)):
            p = 1.0
            vivantes = []
            for b, (x, y) in enumerate(aretes):
                if masque >> b & 1:
                    p *= M[x, y]
                    vivantes.append((x, y))
                else:
                    p *= 1.0 - M[x, y]
            if p == 0.0:
                continue
            atteint, pile = {a}, [a]
            while pile:
                x = pile.pop()
                for (u, v) in vivantes:
                    if u == x and v not in atteint:
                        atteint.add(v)
                        pile.append(v)
            if 1 in atteint and 2 in atteint:
                co[1, 2] += p / 3.0
    print(f"  {nom:<20} P(P2 et P3 touches ensemble) = {co[1, 2]:.4f}")
print(f"  Sous la fourche, une amorce en P1 touche P2 et P3 avec la probabilite {w3}^2 ; sous le")
print("  collisionneur, P2 et P3 ne sont jamais touches par un meme sinistre. C'est la structure")
print("  en V de l'identification des graphes causaux (Verma et Pearl, 1990).")

# =====================================================================================
titre("5. Grandeurs citees, sans separateur")
# =====================================================================================
cm = resultats[("marche (moteur publie)", "amorce du modele (proportionnelle a ROOT)")]
cb = resultats[("branchement (definition 6.1)", "amorce du modele (proportionnelle a ROOT)")]
cu = resultats[("branchement (definition 6.1)", "amorce uniforme")]
mu = resultats[("marche (moteur publie)", "amorce uniforme")]


def paire_max(res):
    coW, coT = res[0], res[1]
    j, k = max(PAIRES, key=lambda p: abs(coT[p] / coW[p] - 1))
    return j, k, coW[j, k], coT[j, k], 100 * (coT[j, k] / coW[j, k] - 1)


for nom, res in (("marche, amorce du modele", cm), ("branchement, amorce du modele", cb),
                 ("branchement, amorce uniforme", cu), ("marche, amorce uniforme", mu)):
    j, k, a, b, e = paire_max(res)
    print(f"  {nom:<32} paire P{j+1}-P{k+1} : {a:.4f} sous W, {b:.4f} sous W^T, ecart {e:+.1f} %")
    print(f"  {'':<32} taille moyenne {res[2]:.3f} sous W, {res[3]:.3f} sous W^T")
for nom_c, _, _ in CASCADES:
    court = "marche" if nom_c.startswith("marche") else "branchement"
    e_unif = premier_ordre[(0.02, nom_c, "amorce uniforme")]
    e_mod = premier_ordre[(0.02, nom_c, "amorce du modele (proportionnelle a ROOT)")]
    print(f"  {court:<12} a eps = 0.02 : ecart maximal {100*e_unif:.2f} % sous amorce uniforme, "
          f"{100*e_mod:.2f} % sous l'amorce du modele")
