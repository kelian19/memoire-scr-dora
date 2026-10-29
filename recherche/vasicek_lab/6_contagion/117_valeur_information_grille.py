#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
117 : la valeur de l'information sur une grille COMMUNE, pour que les ensembles soient emboites.

LE DEFAUT DU SCRIPT 116, signale par une relecture du 19 octobre 2026. Le 116 compare la bande
d'ignorance totale, prise sur les coins {-S, +S}, a une bande contrainte prise sur d'autres points,
les coins {0, signe x S}. Les deux ensembles de points ne sont pas emboites : la bande contrainte
peut etre plus large que la reference, d'ou des resserrements negatifs (jusqu'a -4,1 %), alors que
la valeur d'information definie au chapitre 10 est positive par construction.

CE QUE CE SCRIPT FAIT. Il evalue UNE FOIS la grille commune {-S, 0, +S} sur les dix paires
(3^10 = 59 049 configurations, meme moteur, memes 40 000 annees), puis chaque information est un
FILTRE de cette grille :
  - ignorance totale : toute la grille ;
  - six sens connus : coefficient de chaque paire connue dans {0, signe x S}, les autres libres ;
  - dix sens connus, et chaque paire seule : meme regle.
Les ensembles filtres sont des sous-ensembles de la reference, donc chaque resserrement est
compris entre 0 et 1. Quatre graines, calculees en parallele.

CE QUE CA NE DIT PAS. La grille a trois points par coefficient : ses bandes minorent la bande
continue, sans en demontrer les extrema. Le resserrement est un indicateur numerique sur cette
grille, conditionnel au S d'expert, et non la reduction d'incertitude d'un registre futur.

CONTROLE ET ARRET DUR : a la graine du 55, les points de la grille dont tous les codes valent -1 ou
+1 sont les sommets du 55 ; ils doivent redonner sa bande [6858 ; 8697] et ses 16 sommets filtres
[7801 ; 8431].

Duree : une quarantaine de minutes sur quatre coeurs. Sortie : sorties_verif/117.txt.
"""

import os
import re
import sys
import time
from itertools import combinations, product
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import partial_id as pid                                        # noqa: E402
import resultats_partages as rp                                 # noqa: E402

WID = 84
SEED = 20260721
GRAINES = [SEED, SEED + 1, SEED + 2, SEED + 3]
N_YEARS = 40_000
PIL = pid.PIL
PAIRS = list(combinations(range(len(PIL)), 2))
PAIR_LAB = [f"P{PIL[a]}-P{PIL[b]}" for a, b in PAIRS]
SORTIE_55 = os.path.abspath(os.path.join(HERE, "../../resultats/sorties_verif/55.txt"))

S, A_EXP = pid.decompose(pid.expert_matrix())
SMAX = S[pid.IU]
SIGNES_PM = rp.postmortem_signs()
CONNUES = {m: int(SIGNES_PM[lab]) for m, lab in enumerate(PAIR_LAB) if lab in SIGNES_PM}
SIGNE_CLASSEUR = np.sign(A_EXP[pid.IU])
TOUS = {m: (CONNUES.get(m) or int(SIGNE_CLASSEUR[m]) or 1) for m in range(pid.NFREE)}
CODES = np.array(list(product((-1, 0, 1), repeat=pid.NFREE)), dtype=np.int8)


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def evaluer(seed):
    ev = pid.Evaluator(n_years=N_YEARS, seed=seed)
    out = np.full(len(CODES), np.nan)
    for i, c in enumerate(CODES):
        W = pid.build_W(S, c.astype(float) * SMAX)
        if pid.admissible(W):
            out[i] = ev.scr(W)
    return seed, out


def masque(connus):
    """Configurations compatibles : pour chaque paire connue, code dans {0, signe}."""
    m = np.ones(len(CODES), bool)
    for k, s in connus.items():
        m &= (CODES[:, k] == 0) | (CODES[:, k] == s)
    return m


def largeur(v, m):
    x = v[m & ~np.isnan(v)]
    return x.min(), x.max(), x.max() - x.min(), x.size


def main():
    t0 = time.time()
    with Pool(len(GRAINES)) as pool:
        res = dict(pool.map(evaluer, GRAINES))

    titre("117 : valeur de l'information sur la grille commune {-S, 0, +S}")
    print("3^10 = 59 049 configurations par graine, evaluees une fois ; chaque information est un")
    print("filtre de la grille, donc les ensembles compares sont emboites et le resserrement >= 0.")

    v0 = res[SEED]
    sommets = np.all(CODES != 0, axis=1)
    txt55 = open(SORTIE_55, encoding="utf-8-sig").read()
    pub = [(int(a), int(b)) for a, b in re.findall(r"bande \[(\d+) ; (\d+)\] M", txt55)[:2]]
    lo, hi, _, n = largeur(v0, sommets)
    lof, hif, _, nf = largeur(v0, sommets & masque(CONNUES))
    if (round(lo), round(hi)) != pub[0] or (round(lof), round(hif)) != pub[1] or nf != 16:
        sys.exit(f"ARRET : sommets [{lo:.0f} ; {hi:.0f}] et filtre [{lof:.0f} ; {hif:.0f}] ({nf}) "
                 f"contre {pub} dans le 55.")
    print(f"\n  controle : les {n} sommets redonnent [{lo:.0f} ; {hi:.0f}] et les {nf} sommets filtres "
          f"[{lof:.0f} ; {hif:.0f}], comme le 55.")

    titre("1. Bandes sur la grille, par graine")
    print(f"  {'graine':<10}{'admissibles':>12}{'reference':>18}{'largeur':>9}{'six sens':>18}"
          f"{'largeur':>9}{'resserrement':>14}{'dix sens':>10}")
    m6, m10 = masque(CONNUES), masque(TOUS)
    red6, red10, wref = [], [], []
    for g in GRAINES:
        v = res[g]
        a, b, w, n = largeur(v, np.ones(len(CODES), bool))
        a6, b6, w6, _ = largeur(v, m6)
        _, _, w10, _ = largeur(v, m10)
        red6.append(100 * (1 - w6 / w)); red10.append(100 * (1 - w10 / w)); wref.append(w)
        print(f"  {g:<10}{n:>12d}{'[' + f'{a:.0f} ; {b:.0f}' + ']':>18}{w:>9.0f}"
              f"{'[' + f'{a6:.0f} ; {b6:.0f}' + ']':>18}{w6:>9.0f}{red6[-1]:>13.1f} %{red10[-1]:>9.1f} %")
    red6, red10, wref = np.array(red6), np.array(red10), np.array(wref)
    print(f"\n  six sens : resserrement moyen {red6.mean():.1f} %, ecart-type {red6.std(ddof=1):.1f} point, "
          f"etendue [{red6.min():.1f} ; {red6.max():.1f}]")
    print(f"  dix sens : resserrement moyen {red10.mean():.1f} %, etendue [{red10.min():.1f} ; {red10.max():.1f}]")
    print(f"  largeur de reference sur la grille : de {wref.min():.0f} a {wref.max():.0f} M selon la graine")

    titre("2. Chaque paire documentee seule (meme grille, quatre graines)")
    rows = []
    for k, lab in enumerate(PAIR_LAB):
        r = []
        for g in GRAINES:
            v = res[g]
            w = largeur(v, np.ones(len(CODES), bool))[2]
            r.append(100 * (1 - largeur(v, masque({k: TOUS[k]}))[2] / w))
        r = np.array(r)
        rows.append((lab, r.mean(), r.min(), r.max(), "post-mortem" if k in CONNUES else "classeur"))
    rows.sort(key=lambda x: -x[1])
    print(f"  {'paire':<8}{'moyenne':>9}{'etendue':>18}   source du signe")
    for lab, mo, a, b, src in rows:
        print(f"  {lab:<8}{mo:>8.1f} %{'[' + f'{a:.1f} ; {b:.1f}' + ']':>18}   {src}")
    neg = min(r[2] for r in rows)
    print(f"\n  plus petit resserrement observe : {neg:.1f} % (positif ou nul par construction).")

    titre("LECTURE")
    print(f"  Sur la grille commune, six sens connus resserrent la plage de {red6.mean():.1f} % en moyenne")
    print(f"  ({red6.min():.1f} a {red6.max():.1f} % selon la graine). C'est un indicateur numerique,")
    print("  conditionnel au S d'expert et a la grille, pas la reduction garantie par un registre.")

    titre("GRANDEURS CITEES (sans separateur de milliers, pour le harnais)")
    print(f"  six sens moyenne {red6.mean():.1f} ecart-type {red6.std(ddof=1):.1f} minimum {red6.min():.1f} "
          f"maximum {red6.max():.1f}")
    print(f"  dix sens moyenne {red10.mean():.1f} minimum {red10.min():.1f} maximum {red10.max():.1f}")
    print(f"  largeur reference grille minimum {wref.min():.0f} maximum {wref.max():.0f}")
    for lab, mo, a, b, _ in rows:
        print(f"  paire {lab} moyenne {mo:.1f} minimum {a:.1f} maximum {b:.1f}")
    print(f"\nduree {time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
