#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
55 : la valeur de l'information manquante, chiffree paire par paire.

Le chapitre 9 conclut que « la donnee manquante est un livrable » et specifie le registre
d'incidents qui identifierait W. Il manque le CHIFFRAGE : que vaut cette information ? Le
script 30 le mesure comme fonction d'un parametre d'ignorance scalaire t. Ce script le rend
CONCRET, de deux facons nouvelles.

(1) CE QUE LE CORPUS DE POST-MORTEMS A DEJA ACHETE. Le script 53 etablit le SENS de la direction
    pour 6 des 10 paires de piliers (P1-P3, P1-P4, P1-P5, P2-P3, P2-P4, P3-P5). Connaitre un
    signe ne fixe pas l'amplitude, mais il RESTREINT l'ensemble admissible : les sommets a
    contre-sens sortent. On compare donc la bande AVANT (ignorance totale, 1024 sommets) et
    APRES (signes contraints, 2^4 = 16 sommets libres). L'ecart est la valeur, en capital, de
    l'information documentaire acquise a l'item 1.

(2) QUELLE DEPENDANCE DOCUMENTER EN PRIORITE. Pour chacune des 10 paires, on fixe le signe de
    CETTE SEULE paire et on mesure le retrecissement de la bande. On obtient un CLASSEMENT des
    paires par valeur d'information : si le registre DORA ne pouvait documenter qu'une seule
    dependance inter-piliers, laquelle faudrait-il choisir ? C'est une recommandation
    reglementaire actionnable, et elle ne depend d'aucun prior.

LECTURE. Tout est exprime en LARGEUR DE BANDE et en pourcentage, jamais comme un niveau de
capital : c'est la largeur qui mesure l'ignorance, et c'est elle que l'information reduit.

Sortie : diagnostics + figure Z18_valeur_information.png.
"""

import os
import sys
from itertools import combinations, product

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import partial_id as pid                                        # noqa: E402
import resultats_partages as rp                                 # noqa: E402

WID = 84
T = 1.0                    # ignorance directionnelle maximale (|A| <= S), meme convention que le 30
SEED = 20260721            # MEME graine que le script 30 : les bandes sont comparables
PIL = pid.PIL
PAIRS = list(combinations(range(len(PIL)), 2))                  # ordre de np.triu_indices(5,1)
PAIR_LAB = [f"P{PIL[a]}-P{PIL[b]}" for a, b in PAIRS]

# --- ce que le corpus de post-mortems etablit (script 53) -------------------------------------
# Convention : a_vec[m] > 0  <=>  le pilier de PLUS PETIT numero de la paire PRECEDE l'autre
# (car W[lo,hi] = S + a, et W[j,k] = proba que j entraine k).
# Les signes sont DERIVES des transitions brutes du corpus, via `resultats_partages` : ils ne
# peuvent donc pas diverger du codage du script 53.
POSTMORTEM_SIGNS = rp.postmortem_signs()


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


ev = pid.Evaluator(n_years=40_000, seed=SEED)
P = pid.expert_matrix()
S, A_EXP = pid.decompose(P)
SCR_SOCLE, _ = ev(np.zeros((pid.NP_, pid.NP_)))
smax = T * S[pid.IU]                                            # demi-amplitude par paire


# --- CACHE : les 1024 sommets du pave sont evalues UNE SEULE FOIS ---------------------------
# Toutes les bandes de ce script sont des sous-ensembles de ces sommets (on ne fait que fixer
# des signes) : on les evalue donc une fois, puis chaque bande devient un simple filtre. C'est
# exact (aucune approximation) et environ six fois plus rapide que de reevaluer par bande.
print("evaluation des 1024 sommets (une seule fois)...", flush=True)
_SIGNS, _SCR = [], []
for m in range(1 << pid.NFREE):
    sg = np.array([(m >> k & 1) * 2 - 1 for k in range(pid.NFREE)], dtype=float)
    W = pid.build_W(S, sg * smax)
    if pid.admissible(W):
        _SIGNS.append(sg)
        _SCR.append(ev.scr(W))
_SIGNS = np.array(_SIGNS)
_SCR = np.array(_SCR)
print(f"  {len(_SCR)}/1024 sommets admissibles, evalues.", flush=True)


def band(fixed):
    """Bande de SCR sur les sommets admissibles dont les signes respectent `fixed`.

    `fixed` : dict index_de_paire -> signe (+1/-1). Les autres paires balaient les deux signes.
    """
    mask = np.ones(len(_SCR), dtype=bool)
    for m, s in fixed.items():
        mask &= (_SIGNS[:, m] == np.sign(s))
    if not mask.any():
        return None
    v = _SCR[mask]
    return float(v.min()), float(v.max()), int(mask.sum())


# =====================================================================================
titre("1. La bande d'ignorance directionnelle, avant toute information")
# =====================================================================================
lo0, hi0, n0 = band({})
w0 = hi0 - lo0
print(f"  Socle sans contagion (reference) : {SCR_SOCLE:.0f} M")
print(f"  Ignorance TOTALE sur les 10 paires : {n0}/1024 sommets admissibles")
print(f"  bande [{lo0:.0f} ; {hi0:.0f}] M, largeur {w0:.0f} M "
      f"({100*w0/SCR_SOCLE:.1f} % du socle)")

# =====================================================================================
titre("2. Ce que le corpus de post-mortems a achete (item 1)")
# =====================================================================================
idx = {lab: m for m, lab in enumerate(PAIR_LAB)}
fixed_pm = {idx[lab]: s for lab, s in POSTMORTEM_SIGNS.items()}
lo1, hi1, n1 = band(fixed_pm)
w1 = hi1 - lo1
print(f"  6 paires sur 10 ont un SENS etabli par les rapports officiels :")
for lab, s in POSTMORTEM_SIGNS.items():
    a, b = lab.split("-")
    print(f"    {lab} : {a if s > 0 else b} precede {b if s > 0 else a}")
print(f"\n  Sommets restants : {n1} (4 paires libres) contre {n0} auparavant.")
print(f"  bande [{lo1:.0f} ; {hi1:.0f}] M, largeur {w1:.0f} M "
      f"({100*w1/SCR_SOCLE:.1f} % du socle)")
print(f"\n  RETRECISSEMENT : {100*(1-w1/w0):.0f} % de la largeur de bande, soit {w0-w1:.0f} M")
print(f"  d'ambiguite de capital levee par la seule lecture de 7 rapports publics.")
print(f"  C'est la valeur, chiffree, de l'information documentaire acquise a l'item 1.")

# =====================================================================================
titre("3. Quelle dependance documenter en priorite ? (valeur marginale par paire)")
# =====================================================================================
print("  Pour chaque paire, on fixe le signe de CETTE SEULE paire (au sens des post-mortems")
print("  quand il est connu, au sens du classeur sinon) et on mesure le retrecissement.\n")
rows = []
A_sign_exp = np.sign(A_EXP[pid.IU])
for m, lab in enumerate(PAIR_LAB):
    s = POSTMORTEM_SIGNS.get(lab)
    src = "post-mortem" if s is not None else "classeur"
    if s is None:
        s = float(A_sign_exp[m]) or 1.0
    r = band({m: s})
    if r is None:
        continue
    lo, hi, _ = r
    w = hi - lo
    rows.append((lab, 100 * (1 - w / w0), src, smax[m]))
rows.sort(key=lambda t: -t[1])
print(f"  {'paire':<8}{'retrecissement':>16}{'demi-amplitude':>17}   source du signe")
for lab, red, src, sm in rows:
    print(f"  {lab:<8}{red:>15.1f} %{sm:>17.3f}   {src}")
print(f"\n  Lecture : la paire {rows[0][0]} porte l'essentiel de l'ambiguite "
      f"({rows[0][1]:.0f} % a elle seule).")
print(f"  C'est LA dependance que le registre DORA devrait documenter en premier. La")
print(f"  recommandation reglementaire du chapitre 9 cesse d'etre generique : elle se")
print(f"  hierarchise, et cette hierarchie ne depend d'aucun prior.")

# =====================================================================================
titre("4. Ce qui resterait a gagner, et ce qu'aucune information de SENS ne donnera")
# =====================================================================================
SIGNS_ALL = {m: (POSTMORTEM_SIGNS.get(lab) or float(A_sign_exp[m]) or 1.0)
             for m, lab in enumerate(PAIR_LAB)}
lo_v, hi_v, _ = band(SIGNS_ALL)
print(f"  Sur les SOMMETS (|A| = S, la convention du chapitre 10), fixer les 10 signes ne laisse")
print(f"  qu'UN seul sommet : la bande y tombe mecaniquement a {hi_v-lo_v:.0f} M. C'est un ARTEFACT")
print(f"  de la convention, pas une identification : les sommets sont les configurations")
print(f"  directionnelles EXTREMES, et il n'en reste qu'une quand tous les signes sont poses.")
print(f"\n  La bonne lecture : a signes connus, l'AMPLITUDE reste inconnue, chaque coefficient")
print(f"  vivant dans [0, S] du cote de son signe. On enumere donc les coins de CE pave")
print(f"  (chaque coefficient a 0 ou a son maximum) : 1024 configurations d'amplitude.")
vals_amp = []
for m in range(1 << pid.NFREE):
    frac = np.array([(m >> k & 1) for k in range(pid.NFREE)], dtype=float)   # 0 ou 1
    a = np.array([SIGNS_ALL[k] for k in range(pid.NFREE)]) * smax * frac
    W = pid.build_W(S, a)
    if pid.admissible(W):
        vals_amp.append(ev.scr(W))
vals_amp = np.array(vals_amp)
lo_amp, hi_amp = float(vals_amp.min()), float(vals_amp.max())
w_amp = hi_amp - lo_amp
print(f"  bande a signes TOUS connus : [{lo_amp:.0f} ; {hi_amp:.0f}] M, largeur {w_amp:.0f} M "
      f"({100*w_amp/SCR_SOCLE:.1f} % du socle)")
print(f"\n  Autrement dit : meme si le registre DORA etablissait les DIX directions, il resterait")
print(f"  une bande de {w_amp:.0f} M, soit {100*w_amp/w0:.0f} % de l'ambiguite directionnelle initiale,")
print(f"  imputable a la seule AMPLITUDE. Connaitre le sens ne donne pas la force du lien :")
print(f"  l'identification resterait PARTIELLE, et la lecture en bande n'est pas provisoire.")
print(f"\n  ATTENTION a ne pas soustraire ces deux bandes : la bande de SENS ({w1:.0f} M) est calculee")
print(f"  sur les SOMMETS (amplitudes extremes, convention du chapitre 10), la bande d'AMPLITUDE")
print(f"  ({w_amp:.0f} M) sur les coins du pave a signes fixes. Les deux conventions explorent des")
print(f"  ensembles differents et ne s'additionnent pas ; elles repondent a deux questions")
print(f"  distinctes : « quel sens ? » et « quelle force ? ».")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print(f"  1. La valeur de l'information se chiffre : lire 7 rapports publics a leve")
print(f"     {100*(1-w1/w0):.0f} % de la largeur de bande d'ambiguite directionnelle.")
print(f"  2. Elle se HIERARCHISE : la paire {rows[0][0]} vaut a elle seule {rows[0][1]:.0f} %, la")
print(f"     derniere {rows[-1][0]} n'en vaut que {rows[-1][1]:.0f} %. Le registre DORA devrait")
print(f"     documenter en priorite les dependances a forte valeur, pas toutes a egalite.")
print(f"  3. Meme les DIX sens etablis, une bande de {w_amp:.0f} M subsiste ({100*w_amp/w0:.0f} % de")
print(f"     l'ambiguite initiale) : elle est imputable a l'AMPLITUDE, que nul rapport ne donne.")
print(f"     La lecture en bande n'est donc pas provisoire, elle est structurelle.")
print("  4. Cela transforme « la donnee manquante est un livrable » en une specification")
print("     CHIFFREE et HIERARCHISEE, opposable a un superviseur.")

# =====================================================================================
# figure Z18
# =====================================================================================
# Une LIGNE de deux panneaux, et non une colonne : la figure s'imprime dans le fil du
# texte au lieu d'occuper seule une page. Aucun titre general dans l'image : la
# legende LaTeX le porte. HAUTEUR DE 2,5 POUCES, et elle a un motif : imprimee a
# environ 6 cm, la figure et sa legende de six lignes tiennent sous le titre de leur
# section, en fin de chapitre ; a 3,1 pouces le flottant etait renvoye seul sur une page.
from matplotlib.patches import Patch                            # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator    # noqa: E402

mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 8.5,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"
ESP = " "                    # espace fine insecable, separateur de milliers


def fr(x, nd=0):
    """Nombre ecrit a la francaise : virgule decimale, espace fine pour les milliers."""
    return f"{x:,.{nd}f}".replace(",", ESP).replace(".", ",")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.3, 2.5),
                               gridspec_kw={"width_ratios": [1, 1.35]})

# (a) retrecissement de la bande : avant / apres post-mortems / signes complets
stages = [("ignorance\ntotale", lo0, hi0, MUTED),
          ("après les 7\npost-mortems", lo1, hi1, BLUE),
          ("les 10 sens\nconnus", lo_amp, hi_amp, GREEN)]
for i, (lab, lo, hi, c) in enumerate(stages):
    ax1.plot([i, i], [lo, hi], color=c, lw=15, solid_capstyle="butt", alpha=0.85)
    ax1.text(i, hi + 70, f"largeur\n{fr(hi - lo)} M€", ha="center", va="bottom",
             fontsize=7.5, color=INK2)
ax1.text(1, lo1 - 90, f"{100*(1-w1/w0):.0f} % acquis\npar 7 rapports", ha="center",
         va="top", fontsize=7.5, color=BLUE)
ax1.text(2, lo_amp - 90, "reste\nl'amplitude", ha="center", va="top", fontsize=7.5,
         color=GREEN)
ax1.axhline(SCR_SOCLE, color=ACCENT, ls=":", lw=1.1)
ax1.text(-0.5, SCR_SOCLE + 60, "socle sans contagion", fontsize=7.5, color=ACCENT,
         va="bottom")
ax1.set_xticks(range(3)); ax1.set_xticklabels([s[0] for s in stages])
ax1.set_xlim(-0.55, 2.55)
ax1.set_ylim(5000, 9600)
ax1.yaxis.set_major_locator(MultipleLocator(1000))
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fr(v)))
ax1.set_ylabel("bande de SCR (M€)")
ax1.set_title("(a)  L'information rétrécit la bande", fontsize=9.5, color=INK, loc="left")

# (b) valeur marginale par paire, classee ; la couleur dit la source du signe
labs = [r[0] for r in rows]
reds = [r[1] for r in rows]
cols = [BLUE if r[2] == "post-mortem" else MUTED for r in rows]
yp = np.arange(len(rows))[::-1]
ax2.barh(yp, reds, height=0.72, color=cols, alpha=0.9)
for y_, r in zip(yp, rows):
    ax2.text(r[1] + 0.5, y_, f"{fr(r[1], 1)} %", va="center", fontsize=7.5, color=INK2)
ax2.set_yticks(yp); ax2.set_yticklabels(labs)
ax2.set_ylim(-0.6, len(rows) - 0.4)
ax2.set_xlim(0, max(reds) * 1.25)
ax2.set_xlabel("rétrécissement si cette seule paire est documentée (%)")
ax2.set_title("(b)  Quelle dépendance documenter en priorité", fontsize=9.5, color=INK,
              loc="left")
ax2.legend(handles=[Patch(facecolor=BLUE, alpha=0.9, label="sens établi par un post-mortem"),
                    Patch(facecolor=MUTED, alpha=0.9, label="sens tiré du classeur d'expert")],
           loc="lower right", fontsize=7.5, frameon=False, handlelength=1.2)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(w_pad=1.8)
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z18_valeur_information.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
