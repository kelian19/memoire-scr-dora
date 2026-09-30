#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
100 : les quatre canaux se composent de facon quasi MULTIPLICATIVE, et c'est ce qui fait
l'interaction de +5 001 M.

POURQUOI CE SCRIPT EXISTE. Le script 68 decompose l'ecart de 14 139 M entre etats en quinze termes
de Mobius, en euros, et lit dans les ordres 2 a 4 une interaction de +5 001 M, soit 35 % de
l'ecart. Une decomposition ADDITIVE d'effets qui se MULTIPLIENT produit mecaniquement une
interaction positive : (1 + a)(1 + b) - 1 depasse a + b. La question est donc de savoir quelle part
de l'interaction publiee n'est que cette arithmetique, et quelle part serait un effet propre de la
cascade. On la tranche en refaisant la decomposition sur le LOGARITHME du capital, ou des effets
multiplicatifs deviennent additifs.

DEUX FORMES FERMEES, qui expliquent les deux canaux calibrables. Par l'approximation de la perte
unique, a queue de Pareto generalisee d'indice xi, le quantile annuel croit comme
(lambda p_u)^xi. Le facteur du canal de frequence doit donc valoir (lambda_NC / lambda_C)^xi, et
celui du canal de detection (p_u,NC / p_u,C)^xi. On les confronte aux facteurs simules.

CE QUE LE SCRIPT LIT. Les seize capitaux de sorties_verif/68.txt, sans rien simuler : il s'arrete si
les deux coins ne redonnent pas 6 049 et 20 188 M. Les parametres viennent du moteur
(canaux_conformite, euro_cascade_model), jamais d'une recopie.

Aucune simulation, aucune donnee sous licence, aucune figure : le script tourne sur les deux postes.
"""

import math
import os
import re
import sys
from itertools import combinations

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import canaux_conformite as cx                                  # noqa: E402

WID = 88
CANAUX = ("freq", "det", "prop", "accum")
NOMS = {"freq": "frequence", "det": "detection", "prop": "propagation", "accum": "accumulation"}


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# ------------------------------------------------------------------ lecture du treillis
chemin = os.path.join(DEPOT, "sorties_verif", "68.txt")
scr = {}
with open(chemin, encoding="utf-8") as f:
    for ligne in f:
        m = re.match(r"\s+(conforme|[a-z]+(?:\+[a-z]+)*)\s+(\d+)\s+\d+\s+\d\s*$", ligne)
        if m and (m.group(1) == "conforme" or all(c in CANAUX for c in m.group(1).split("+"))):
            cle = frozenset() if m.group(1) == "conforme" else frozenset(m.group(1).split("+"))
            scr[cle] = float(m.group(2))
assert len(scr) == 16, f"treillis incomplet dans 68.txt : {len(scr)} configurations lues"
C0, C4 = scr[frozenset()], scr[frozenset(CANAUX)]
assert (C0, C4) == (6049.0, 20188.0), f"les coins publies ont derive : {C0}, {C4}"

# =====================================================================================
titre("1. Le produit des facteurs isoles contre le facteur publie")
# =====================================================================================
f_iso = {c: scr[frozenset([c])] / C0 for c in CANAUX}
for c in CANAUX:
    print(f"  {NOMS[c]:<14} facteur isole {f_iso[c]:.4f}  ({scr[frozenset([c])]:.0f} M / {C0:.0f} M)")
prod = math.prod(f_iso.values())
fact = C4 / C0
print(f"\n  produit des quatre facteurs isoles : {prod:.4f}")
print(f"  facteur publie (non conforme / conforme) : {fact:.4f}")
print(f"  rapport publie / produit : {fact/prod:.4f}, soit {100*(fact/prod-1):+.1f} %")
somme_iso = sum(scr[frozenset([c])] - C0 for c in CANAUX)
inter_pub = (C4 - C0) - somme_iso
inter_mult = C0 * (prod - 1) - somme_iso
print(f"\n  somme des effets isoles : {somme_iso:.0f} M ; interaction publiee : {inter_pub:.0f} M")
print(f"  interaction qu'un modele PUREMENT multiplicatif produirait : {inter_mult:.0f} M")
print(f"  l'interaction publiee vaut, en part de cette interaction multiplicative : "
      f"{100*inter_pub/inter_mult:.0f} %")

# =====================================================================================
titre("2. La decomposition de Mobius en logarithme : l'interaction propre")
# =====================================================================================
L = {S: math.log(v) for S, v in scr.items()}
mob = {}
for r in range(1, 5):
    for S in combinations(CANAUX, r):
        S = frozenset(S)
        mob[S] = sum((-1) ** (len(S) - len(T)) * L[frozenset(T)]
                     for k in range(len(S) + 1) for T in combinations(sorted(S), k))
ecart_log = L[frozenset(CANAUX)] - L[frozenset()]
par_ordre = {r: sum(v for S, v in mob.items() if len(S) == r) for r in range(1, 5)}
print(f"  ecart en log : ln({fact:.4f}) = {ecart_log:.4f}")
for r in range(1, 5):
    print(f"  ordre {r} : {par_ordre[r]:+.4f}  soit {100*par_ordre[r]/ecart_log:+.1f} % de l'ecart en log")
inter_log = sum(par_ordre[r] for r in (2, 3, 4))
print(f"\n  interaction propre (ordres 2 a 4 en log) : {inter_log:+.4f}, "
      f"soit un facteur {math.exp(inter_log):.4f}")
print("\n  croises de paires en log :")
for S, v in sorted(((S, v) for S, v in mob.items() if len(S) == 2), key=lambda t: -t[1]):
    a, b = sorted(S, key=CANAUX.index)
    print(f"    {NOMS[a]} x {NOMS[b]:<14} {v:+.4f}  (facteur {math.exp(v):.3f})")

# =====================================================================================
titre("3. Les seize configurations contre la prediction multiplicative")
# =====================================================================================
print(f"  {'configuration':<26}{'publie':>9}{'produit':>10}{'rapport':>10}")
ecarts = []
for S in sorted(scr, key=lambda s: (len(s), sorted(CANAUX.index(c) for c in s))):
    pred = C0 * math.prod(f_iso[c] for c in S)
    rap = scr[S] / pred
    ecarts.append(abs(rap - 1))
    nom = "conforme" if not S else "+".join(sorted(S, key=CANAUX.index))
    print(f"  {nom:<26}{scr[S]:>9.0f}{pred:>10.0f}{rap:>10.3f}")
print(f"\n  ecart maximal a la prediction multiplicative : {100*max(ecarts):.1f} %")
print("  LECTURE. L'arithmetique du produit explique toute l'interaction publiee et au-dela : le")
print("  residu propre est une legere SOUS-multiplicativite, portee par la paire propagation x")
print("  accumulation, les deux canaux qui font tomber plusieurs piliers sur un meme sinistre.")

# =====================================================================================
titre("4. Les deux canaux calibrables en forme fermee")
# =====================================================================================
xi = cx.sp["xi"]
f_freq_sla = (cx.LAM_NC / cx.LAM_C) ** xi
f_det_sla = (cx.PU_NC / cx.PU_C) ** xi
print(f"  indice de queue xi = {xi:.4f}")
print(f"  frequence : (lambda_NC / lambda_C)^xi = ({cx.LAM_NC:.2f} / {cx.LAM_C:.2f})^{xi:.4f} = "
      f"{f_freq_sla:.4f} ; simule {f_iso['freq']:.4f} ; ecart {100*(f_iso['freq']/f_freq_sla-1):+.1f} %")
print(f"  detection : (p_u,NC / p_u,C)^xi = ({cx.PU_NC:.4f} / {cx.PU_C:.4f})^{xi:.4f} = "
      f"{f_det_sla:.4f} ; simule {f_iso['det']:.4f} ; ecart {100*(f_iso['det']/f_det_sla-1):+.1f} %")
print("\n  LECTURE. Le canal de frequence et celui de detection suivent la loi de puissance de la")
print("  perte unique a moins de 1 % pres : ils deplacent le quantile comme ils deplacent le nombre")
print("  de sinistres qui depassent le seuil, eleve a la puissance xi. Leur effet ne doit presque rien a")
print("  la cascade ; les canaux de propagation et d'accumulation en dependent, eux, directement.")

# =====================================================================================
titre("5. Grandeurs citees, sans separateur")
# =====================================================================================
print(f"  produit des facteurs isoles {prod:.2f} ; facteur publie {fact:.2f} ; rapport {fact/prod:.3f}")
print(f"  interaction multiplicative {inter_mult:.0f} ; interaction publiee {inter_pub:.0f} ; "
      f"part {100*inter_pub/inter_mult:.0f} %")
print(f"  interaction propre en log {inter_log:.4f} ; facteur {math.exp(inter_log):.3f} ; "
      f"soit {100*(math.exp(inter_log)-1):.1f} %")
print(f"  ecart maximal a la prediction multiplicative {100*max(ecarts):.1f} %")
print(f"  frequence forme fermee {f_freq_sla:.3f} simule {f_iso['freq']:.3f} ; "
      f"detection forme fermee {f_det_sla:.3f} simule {f_iso['det']:.3f}")
