#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
REMPLACE PAR LE SCRIPT 117 (19 octobre 2026, soir) : les ensembles compares ici ne sont pas
emboites, d'ou des resserrements negatifs. Le memoire cite le 117 ; ce script reste pour la trace.

116 : la valeur de l'information, recalculee sur l'ensemble que les signes contraignent vraiment.

LE DEFAUT DU SCRIPT 55, signale par une relecture du 19 octobre 2026. Le 55 mesure ce que
rapporte la connaissance du SENS d'une dependance en FILTRANT les sommets du pave : une paire
de signe connu positif y est fixee a A_jk = +S_jk. Or connaitre le sens impose seulement
A_jk dans [0, S_jk] : l'amplitude reste libre, et la valeur nulle reste admissible. Filtrer les
sommets retire donc des configurations que l'information ne retire pas, et surestime le
resserrement. Le 55 en portait l'indice : a dix signes connus, il trouve une bande de 656 M EUR
sur le bon ensemble, PLUS LARGE que les 630 M EUR qu'il annonce a six signes sur les sommets.

CE QUE CE SCRIPT CALCULE, sur la meme convention que le 55 (coins du pave, meme moteur, meme
graine, memes 40 000 annees) :
  - paire de sens connu : coefficient a 0 ou a son maximum du cote du signe ;
  - paire de sens inconnu : coefficient a -max ou +max, comme dans le 55.
  1. la bande a six sens connus et le resserrement corrige ;
  2. son bruit de simulation, sur quatre graines ;
  3. le classement des paires, chacune documentee seule, sur le meme ensemble corrige,
     lui aussi sur quatre graines ;
  4. la figure valeur_information_contrainte.png est tracee par 116b depuis cette sortie.

CONTROLES ET ARRET DUR : a la graine du 55, l'ignorance totale doit redonner sa bande, le filtrage
des sommets ses 16 sommets et sa bande, et les dix signes connus sa bande d'amplitude, lus dans
sorties_verif/55.txt. Sinon le script s'arrete : il demontrerait une propriete d'un autre calcul.

Ne touche ni partial_id.py, ni config.py, ni une sortie existante. Duree : une vingtaine de minutes.
Sortie : sorties_verif/116.txt.
"""

import os
import re
import sys
import time
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import partial_id as pid                                        # noqa: E402
import resultats_partages as rp                                 # noqa: E402

WID = 84
T = 1.0
SEED = 20260721                     # graine du 55 (et du 30)
GRAINES_BRUIT = [SEED, SEED + 1, SEED + 2, SEED + 3]
N_YEARS = 40_000
PIL = pid.PIL
PAIRS = list(combinations(range(len(PIL)), 2))
PAIR_LAB = [f"P{PIL[a]}-P{PIL[b]}" for a, b in PAIRS]
SIGNES_PM = rp.postmortem_signs()
SORTIE_55 = os.path.abspath(os.path.join(HERE, "../../resultats/sorties_verif/55.txt"))


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def lire_55():
    """Bandes publiees par le 55, pour le controle."""
    txt = open(SORTIE_55, encoding="utf-8-sig").read()
    bandes = re.findall(r"bande(?: a signes TOUS connus :)? \[(\d+) ; (\d+)\] M", txt)
    if len(bandes) != 3:
        sys.exit(f"ARRET : 3 bandes attendues dans 55.txt, {len(bandes)} lues.")
    m = re.search(r"Sommets restants : (\d+)", txt)
    return [(int(a), int(b)) for a, b in bandes], int(m.group(1))


S, A_EXP = pid.decompose(pid.expert_matrix())
SMAX = T * S[pid.IU]
SIGNE_CLASSEUR = np.sign(A_EXP[pid.IU])
CONNUES = {m: float(SIGNES_PM[lab]) for m, lab in enumerate(PAIR_LAB) if lab in SIGNES_PM}
TOUS_SIGNES = {m: (CONNUES.get(m) or float(SIGNE_CLASSEUR[m]) or 1.0) for m in range(pid.NFREE)}


class Capital:
    """Capital par configuration de coefficients, chaque coefficient code -1, 0 ou +1 (x SMAX)."""

    def __init__(self, seed):
        self.ev = pid.Evaluator(n_years=N_YEARS, seed=seed)
        self.cache = {}

    def __call__(self, code):
        if code not in self.cache:
            W = pid.build_W(S, np.array(code, float) * SMAX)
            self.cache[code] = self.ev.scr(W) if pid.admissible(W) else None
        return self.cache[code]


def coins(choix):
    """Produit cartesien : choix[m] est la liste des codes possibles pour la paire m."""
    out = [()]
    for c in choix:
        out = [o + (v,) for o in out for v in c]
    return out


def bande(cap, choix):
    v = [x for x in (cap(c) for c in coins(choix)) if x is not None]
    return (min(v), max(v), len(v)) if v else None


LIBRE = [-1, 1]


def choix_contraint(connus):
    """Sens connu : {0, signe} ; sens inconnu : {-1, +1}."""
    return [[0, int(connus[m])] if m in connus else LIBRE for m in range(pid.NFREE)]


def choix_filtre(connus):
    """Convention du 55 : sens connu fixe au sommet du cote du signe."""
    return [[int(connus[m])] if m in connus else LIBRE for m in range(pid.NFREE)]


t0 = time.time()
publie, n_sommets_pub = lire_55()

titre("116 : valeur de l'information sur l'ensemble contraint par les signes")
print("Meme moteur, meme graine, memes 40 000 annees que le script 55. Un sens connu laisse le")
print("coefficient dans [0, S] du cote du signe : on enumere les coins 0 et maximum ; un sens")
print("inconnu garde les coins -S et +S, comme dans le 55.")

cap = Capital(SEED)
b_tot = bande(cap, [LIBRE] * pid.NFREE)
b_filtre = bande(cap, choix_filtre(CONNUES))
b_dix = bande(cap, choix_contraint(TOUS_SIGNES))
attendu = [(b_tot[0], b_tot[1]), (b_filtre[0], b_filtre[1]), (b_dix[0], b_dix[1])]
for (lo, hi), (plo, phi), nom in zip(attendu, publie,
                                     ("ignorance totale", "six signes, sommets filtres",
                                      "dix signes connus")):
    if round(lo) != plo or round(hi) != phi:
        sys.exit(f"ARRET : {nom} donne [{lo:.0f} ; {hi:.0f}] contre [{plo} ; {phi}] publie par le 55.")
if b_filtre[2] != n_sommets_pub:
    sys.exit(f"ARRET : {b_filtre[2]} sommets filtres contre {n_sommets_pub} dans le 55.")
print("\n  controle : les trois bandes du 55 et ses 16 sommets sont retrouves a l'unite.")

titre("1. Six sens connus : sommets filtres (55) contre ensemble contraint")
b_six = bande(cap, choix_contraint(CONNUES))
w0 = b_tot[1] - b_tot[0]
w_f = b_filtre[1] - b_filtre[0]
w_c = b_six[1] - b_six[0]
w_d = b_dix[1] - b_dix[0]
print(f"  paires de sens connu : {', '.join(PAIR_LAB[m] for m in CONNUES)}")
print(f"  {'ensemble':<36}{'bande (M EUR)':>20}{'largeur':>10}{'configurations':>16}{'resserrement':>14}")
for nom, b in (("ignorance totale", b_tot), ("six sens, sommets filtres (55)", b_filtre),
               ("six sens, ensemble contraint", b_six), ("dix sens, ensemble contraint", b_dix)):
    w = b[1] - b[0]
    print(f"  {nom:<36}{'[' + f'{b[0]:.0f} ; {b[1]:.0f}' + ']':>20}{w:>10.0f}{b[2]:>16d}"
          f"{100 * (1 - w / w0):>13.1f} %")
print(f"\n  Le resserrement publie ({100 * (1 - w_f / w0):.0f} %) tombe a {100 * (1 - w_c / w0):.1f} %"
      f" une fois l'amplitude laissee libre.")
print(f"  Les ensembles sont desormais emboites : six sens ({w_c:.0f} M) puis dix sens ({w_d:.0f} M).")

titre("2. Bruit de simulation du resserrement corrige (quatre graines)")
CAPS = {SEED: cap}
red, larg_tot = [], []
for g in GRAINES_BRUIT:
    c = CAPS.setdefault(g, cap if g == SEED else Capital(g))
    bt = bande(c, [LIBRE] * pid.NFREE)
    bs = bande(c, choix_contraint(CONNUES))
    r = 100 * (1 - (bs[1] - bs[0]) / (bt[1] - bt[0]))
    red.append(r)
    larg_tot.append(bt[1] - bt[0])
    print(f"  graine {g} : totale {bt[1] - bt[0]:.0f} M, six sens {bs[1] - bs[0]:.0f} M, "
          f"resserrement {r:.1f} %")
red = np.array(red)
larg_tot = np.array(larg_tot)
print(f"  resserrement : moyenne {red.mean():.1f} %, ecart-type {red.std(ddof=1):.1f} point, "
      f"etendue [{red.min():.1f} ; {red.max():.1f}]")
print(f"  largeur de la bande totale : de {larg_tot.min():.0f} a {larg_tot.max():.0f} M selon la graine "
      f"(ecart-type {larg_tot.std(ddof=1):.0f} M).")
print("  La graine du 55 donne le resserrement le plus fort des quatre. A 40 000 annees, la largeur")
print("  d'une bande de quantiles extremes porte un bruit du meme ordre que l'effet mesure : le")
print("  resserrement se cite en moyenne avec son etendue, jamais a la seule graine de reference.")

titre("3. Quelle dependance documenter en priorite ? (ensemble contraint, quatre graines)")
print("  Pour chaque paire, son seul sens est connu (post-mortem s'il existe, classeur sinon) :")
print("  son coefficient vit dans [0, S] du cote du signe, les neuf autres restent libres.\n")
par_paire = {}
for g, c in CAPS.items():
    bt = bande(c, [LIBRE] * pid.NFREE)
    wt = bt[1] - bt[0]
    for m, lab in enumerate(PAIR_LAB):
        b = bande(c, choix_contraint({m: TOUS_SIGNES[m]}))
        par_paire.setdefault(lab, []).append(100 * (1 - (b[1] - b[0]) / wt))
rows = []
for m, lab in enumerate(PAIR_LAB):
    v = np.array(par_paire[lab])
    rows.append((lab, v.mean(), "post-mortem" if m in CONNUES else "classeur", v.min(), v.max(), v[0]))
rows.sort(key=lambda r: -r[1])
print(f"  {'paire':<8}{'moyenne':>9}{'etendue':>18}{'graine du 55':>14}   source du signe")
for lab, moy, src, lo, hi, v0 in rows:
    print(f"  {lab:<8}{moy:>8.1f} %{'[' + f'{lo:.1f} ; {hi:.1f}' + ']':>18}{v0:>13.1f} %   {src}")
rangs = {lab: [] for lab in PAIR_LAB}
for i in range(len(GRAINES_BRUIT)):
    ordre = sorted(PAIR_LAB, key=lambda l: -par_paire[l][i])
    for k, l in enumerate(ordre):
        rangs[l].append(k + 1)
print(f"\n  Rangs de {rows[0][0]} sur les quatre graines : {rangs[rows[0][0]]} ; "
      f"de {rows[1][0]} : {rangs[rows[1][0]]}.")
nuls = [r[0] for r in rows if r[4] < 0.05]
print(f"  Paire(s) sans aucune valeur sur les quatre graines : {', '.join(nuls) if nuls else 'aucune'}.")
print("  Les etendues des paires de tete se recouvrent : leur ordre n'est pas resolu. Ce qui")
print("  tient est la separation entre les paires de tete et celles qui ne levent presque rien.")

titre("LECTURE")
print(f"  La connaissance du sens de six dependances reduit la bande de {red.mean():.0f} % en moyenne,")
print(f"  de {red.min():.0f} a {red.max():.0f} % selon la graine, et non de {100 * (1 - w_f / w0):.0f} %.")
print("  C'est un ordre de grandeur conditionnel : au S d'expert maintenu, a la convention des")
print("  coins, et sans rien dire de ce qu'un registre futur etablirait vraiment.")

titre("GRANDEURS CITEES (sans separateur de milliers, pour le harnais)")
print(f"  bande six sens contraint {b_six[0]:.0f} {b_six[1]:.0f} largeur {w_c:.0f}")
print(f"  resserrement corrige {100 * (1 - w_c / w0):.1f} publie 55 {100 * (1 - w_f / w0):.1f}")
print(f"  bruit resserrement moyenne {red.mean():.1f} ecart-type {red.std(ddof=1):.1f} "
      f"minimum {red.min():.1f} maximum {red.max():.1f}")
print(f"  largeur totale selon la graine minimum {larg_tot.min():.0f} maximum {larg_tot.max():.0f}")
for lab, moy, _, lo, hi, v0 in rows:
    print(f"  paire {lab} moyenne {moy:.1f} minimum {lo:.1f} maximum {hi:.1f} graine55 {v0:.1f}")

# La figure est tracee par 116b, qui relit cette sortie : la retoucher ne demande pas de
# relancer les vingt minutes de calcul.
print(f"\nduree {time.time() - t0:.0f} s", file=sys.stderr)
