#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31 : les DEUX regimes face a face, AVANT toute elicitation.

La question tranchee ici : puisque W n'est pas calibrable, que vaut-il mieux faire ?

  REGIME A, \"on ne calibre pas W\" (statu quo). On POSE W au classeur qualitatif et on
    publie un SCR ponctuel et un ordre de remediation. C'est simple, lisible, et
    entierement suspendu a un jugement qui n'a pas encore ete elicite.

  REGIME B, le modele EN COUCHES. Un socle calibre (frequence, severite, accumulation)
    qui ne depend d'aucun parametre non identifiable, plus une couche de contagion
    PARTIELLEMENT IDENTIFIEE : on ne choisit pas W, on caracterise son ensemble
    admissible et on publie des BORNES de capital et une priorite robuste.

Le test est fait AVANT l'elicitation, et c'est tout l'interet : on veut savoir ce que
chaque regime peut affirmer aujourd'hui, ce que l'elicitation pourra encore changer, et
surtout ce qui se passe si le panel est BIAISE. Un dispositif qui ne survit qu'a une
elicitation reussie n'est pas un dispositif, c'est un pari.

Le test de biais du panel est le point decisif. On fait varier la direction elicitee de
\"le panel confirme le classeur\" a \"le panel dit l'inverse\", et on regarde bouger la
reponse de chaque regime.

Sortie : diagnostics + figure Z2_comparaison_regimes.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import partial_id as pid                                        # noqa: E402

WID = 78
N_VERT = 200          # sommets pour les bornes
N_SET = 150           # matrices pour l'analyse de decision
N_BIAS = 21           # points du balayage de biais du panel
SEED = 20260721


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


ev = pid.Evaluator(n_years=40_000, seed=SEED)
P = pid.expert_matrix()
S, A_exp = pid.decompose(P)
print(f"tirage commun : {ev.n_years} annees, {ev.T} incidents, lambda={ev.lam:.2f}/an")

W0 = np.zeros((pid.NP_, pid.NP_))
scr_socle, mean_socle = ev(W0)
scr_exp, mean_exp = ev(P)
scr_sym, mean_sym = ev(S)

# =====================================================================================
titre("1. Ce que chaque regime peut affirmer AUJOURD'HUI, sans elicitation")
# =====================================================================================
rng = np.random.default_rng(SEED + 1)
vals, mns, Wsets = [], [], []
for a_vec in pid.all_vertices(S, 1.0):      # enumeration EXHAUSTIVE : bornes reproductibles
    W = pid.build_W(S, a_vec)
    if not pid.admissible(W):
        continue
    v, m = ev(W)
    vals.append(v)
    mns.append(m)
    Wsets.append(W)
vals, mns = np.array(vals), np.array(mns)
lo, hi = float(vals.min()), float(vals.max())
mlo, mhi = float(mns.min()), float(mns.max())

ben_exp = ev.benefits(P)
prio_exp = int(np.argmax(ben_exp))

print(f"  REGIME A (W pose) : SCR = {scr_exp:.0f} M, priorite = P{pid.PIL[prio_exp]}.")
print("    Un point et un ordre. Aucun enonce sur ce que ce point aurait pu etre.")
print(f"  REGIME B (couches) : socle = {scr_socle:.0f} M (ne depend d'AUCUN parametre non")
print(f"    identifiable), contagion bornee, SCR dans [{lo:.0f} ; {hi:.0f}] M.")
print(f"    En perte moyenne : socle {mean_socle:.0f} M, bornes [{mlo:.0f} ; {mhi:.0f}] M.")
print(f"  Le point du regime A tombe dans la bande du regime B "
      f"({'oui' if lo <= scr_exp <= hi else 'NON'}), a "
      f"{100*(scr_exp-lo)/(hi-lo):.0f} % de la borne basse.")

# =====================================================================================
titre("2. Quelle part du resultat est deja fixee sans elicitation ?")
# =====================================================================================
part_socle = 100.0 * scr_socle / hi
part_bande = 100.0 * (hi - lo) / hi
print(f"  Sur la borne haute ({hi:.0f} M) :")
print(f"    - {part_socle:5.1f} %  socle calibre, insensible a W et donc a l'elicitation")
print(f"    - {100-part_socle-part_bande:5.1f} %  contagion CERTAINE (le plancher de la bande "
      f"au-dessus du socle)")
print(f"    - {part_bande:5.1f} %  bande d'ignorance directionnelle, la SEULE part que "
      f"l'elicitation peut deplacer")
print(f"  Autrement dit, {100-part_bande:.0f} % du capital est deja determine avant que le")
print("  moindre expert ait repondu. L'elicitation ne joue que sur le solde.")

# =====================================================================================
titre("3. L'elicitation peut-elle trancher la DECISION ? (priorite de remediation)")
# =====================================================================================
idx = np.random.default_rng(SEED + 5).choice(len(Wsets), size=min(N_SET, len(Wsets)),
                                             replace=False)
sub = [Wsets[i] for i in idx]
B = np.array([ev.benefits(W) for W in sub])
top = B.argmax(axis=1)
freq = np.bincount(top, minlength=pid.NP_) / len(sub)
ordre = np.argsort(-freq)
print(f"  Sur {len(sub)} matrices admissibles tirees de l'ensemble, pilier arrivant en tete :")
for j in ordre:
    print(f"    P{pid.PIL[j]} : {100*freq[j]:5.1f} %")
# Deux enonces distincts, a ne pas confondre :
#   - ce que l'ensemble EXCLUT deja (acquis sans elicitation) ;
#   - ce qu'il laisse ouvert (le travail de l'elicitation).
jamais = [j for j in range(pid.NP_) if freq[j] == 0.0]
rares = [j for j in range(pid.NP_) if 0.0 < freq[j] < 0.10]
tete = [j for j in ordre if freq[j] >= 0.20]
print(f"\n  ACQUIS SANS ELICITATION :")
if jamais:
    print(f"    - {', '.join('P'+str(pid.PIL[j]) for j in jamais)} n'arrive JAMAIS en tete, "
          f"sur aucune matrice admissible.")
if rares:
    print(f"    - {', '.join('P'+str(pid.PIL[j]) for j in rares)} y arrive dans moins de "
          f"10 % des cas.")
print(f"  OUVERT, ET C'EST LE TRAVAIL DE L'ELICITATION :")
print(f"    - departager {' et '.join('P'+str(pid.PIL[j]) for j in tete)} "
      f"({100*sum(freq[j] for j in tete):.0f} % des cas a eux deux)")
resid = [j for j in ordre if 0.10 <= freq[j] < 0.20]
if resid:
    print(f"    - avec un residu pour {', '.join('P'+str(pid.PIL[j]) for j in resid)} "
          f"({100*sum(freq[j] for j in resid):.0f} %)")

# =====================================================================================
titre("4. TEST DE BIAIS DU PANEL : que devient chaque regime si le panel derape ?")
# =====================================================================================
# b = facteur d'echelle applique a la direction du classeur.
#   b = +1 : le panel confirme le classeur ;  b = 0 : le panel ne voit aucune direction ;
#   b = -1 : le panel dit l'inverse.  On va jusqu'au bord du pave admissible.
nz = np.abs(A_exp[pid.IU]) > 1e-12
bmax = float(np.min(S[pid.IU][nz] / np.abs(A_exp[pid.IU][nz])))
bs = np.linspace(-bmax, bmax, N_BIAS)
scr_b, prio_b = [], []
for b in bs:
    W = pid.build_W(S, b * A_exp[pid.IU])
    scr_b.append(ev.scr(W))
    prio_b.append(int(np.argmax(ev.benefits(W))))
scr_b = np.array(scr_b)
prio_b = np.array(prio_b)

amp = scr_b.max() - scr_b.min()
print(f"  Balayage du biais de b = {-bmax:.2f} a b = {bmax:.2f} (bord du pave admissible).")
print(f"  REGIME A : le SCR publie suit le panel de {scr_b.min():.0f} a {scr_b.max():.0f} M,")
print(f"    soit une amplitude de {amp:.0f} M ({100*amp/scr_exp:.0f} % du point actuel), et")
print(f"    la priorite bascule entre {len(set(prio_b.tolist()))} pilier(s) : "
      f"{', '.join('P'+str(pid.PIL[j]) for j in sorted(set(prio_b.tolist())))}.")
print(f"    Rien dans le regime A ne signale ce deplacement au lecteur.")
print(f"  REGIME B : la bande [{lo:.0f} ; {hi:.0f}] ne bouge pas d'un euro, puisqu'elle ne")
print(f"    depend d'aucun dire d'expert. Tous les paniques du panel y sont contenus "
      f"({'verifie' if scr_b.min() >= lo - 1 and scr_b.max() <= hi + 1 else 'NON VERIFIE'}).")
print("  C'est l'argument decisif : le regime B est publiable AVANT l'elicitation et")
print("  reste vrai APRES, quel que soit son resultat. Le regime A, lui, est un pari")
print("  sur une seance qui n'a pas encore eu lieu.")

# =====================================================================================
titre("5. Verdict compare")
# =====================================================================================
rows = [
    ("Sortie sur le capital", f"un point ({scr_exp:.0f} M)",
     f"socle {scr_socle:.0f} + bande [{lo:.0f} ; {hi:.0f}]"),
    ("Opposable sans elicitation", "non, tout repose sur le classeur",
     f"oui, {100-part_bande:.0f} % du capital est deja fixe"),
    ("Sensibilite a un panel biaise", f"deplace le SCR de {amp:.0f} M",
     "aucune, la bande est invariante"),
    ("Priorite de remediation", f"P{pid.PIL[prio_exp]}, sans marge d'erreur",
     f"groupe de tete {', '.join('P'+str(pid.PIL[j]) for j in tete)}, ordre interne ouvert"),
    ("Role de l'elicitation", "fournit LE resultat",
     "resserre la bande et departage la tete"),
    ("Si l'elicitation echoue", "le modele n'a plus de parametre",
     "le resultat tient, en bornes"),
]
print(f"  {'':<32}{'REGIME A (W pose)':<38}{'REGIME B (couches)'}")
for a, b, c in rows:
    print(f"  {a:<32}{b:<38}{c}")

# =====================================================================================
# figure
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 11,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
})
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"

# GEOMETRIE : DEUX RANGEES, PAS QUATRE COLONNES. Le memoire imprime cette figure
# sur 7,27 pouces. Tracee sur 20,5, elle subissait une reduction de 2,8 : une
# etiquette de 9 points s'imprimait a 3,2, illisible. En 2 x 2 sur 9,4 pouces la
# reduction tombe a 1,29 et la meme etiquette sort a 7 points. La largeur de
# PANNEAU passe de 1,8 pouce imprime a 3,6, ce qui est le vrai gain.
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9.0, 4.3))

# (a) les deux regimes cote a cote
ax1.bar([0], [scr_exp], width=0.5, color=ACCENT, alpha=0.85)
ax1.bar([1], [scr_socle], width=0.5, color=GREEN, alpha=0.85, label="socle calibré")
ax1.bar([1], [hi - scr_socle], width=0.5, bottom=[scr_socle], color=BLUE, alpha=0.30,
        label="contagion, bornée")
ax1.plot([1, 1], [lo, hi], color=INK, lw=2.4, zorder=5)
ax1.scatter([1, 1], [lo, hi], marker="_", s=420, color=INK, zorder=6)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["A : $W$ posé", "B : en couches"])
ax1.set_ylabel("SCR (VaR 99,5 %, M€)", color=INK2)
ax1.set_title("(a)  Un point, ou un socle et des bornes", fontsize=10, color=INK, pad=6)
ax1.set_ylim(0, ax1.get_ylim()[1] * 1.34)
ax1.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# (b) ce qui depend de l'elicitation
parts = [scr_socle, lo - scr_socle, hi - lo]
labs = ["socle calibré", "contagion certaine", "bande d'ignorance"]
cols = [GREEN, BLUE, ACCENT]
bot = 0
for p, l, c in zip(parts, labs, cols):
    ax2.bar([0], [p], bottom=[bot], width=0.45, color=c, alpha=0.85, label=l)
    ax2.text(0.28, bot + p / 2, f"{100*p/hi:.0f} %", va="center", fontsize=9, color=INK2)
    bot += p
ax2.set_xticks([])
ax2.set_ylabel("SCR (M€)", color=INK2)
ax2.set_title("(b)  Seule la part orange dépend\nde l'élicitation", fontsize=11,
              color=INK, pad=8)
ax2.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# (c) test de biais du panel
ax3.axhspan(lo, hi, color=BLUE, alpha=0.16, label="bande du régime B")
ax3.plot(bs, scr_b, color=ACCENT, lw=2.4, label="SCR publié par le régime A")
ax3.axvline(1.0, color=MUTED, lw=0.9, ls=":")
ax3.text(1.0, scr_b.min(), " classeur\n actuel", fontsize=8.5, color=MUTED, va="bottom")
ax3.axvline(0.0, color=MUTED, lw=0.9, ls=":")
ax3.set_xlabel("biais du panel  (1 = confirme le classeur, 0 = aucune direction,\n"
               "négatif = dit l'inverse)", color=INK2, fontsize=9)
ax3.set_ylabel("SCR (M€)", color=INK2)
ax3.set_title("(c)  Le régime A suit le panel,\nla bande B ne bouge pas", fontsize=11,
              color=INK, pad=8)
ax3.legend(frameon=False, fontsize=8, loc="upper left")
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)

# (d) la decision
xs = np.arange(pid.NP_)
ax4.bar(xs, 100 * freq, color=[BLUE if freq[j] > 0.05 else MUTED for j in xs], alpha=0.85)
for j in xs:
    if freq[j] > 0.005:
        ax4.text(j, 100 * freq[j] + 1.5, f"{100*freq[j]:.0f}%", ha="center", fontsize=8.5,
                 color=INK2)
ax4.scatter([prio_exp], [100 * freq[prio_exp] + 11], marker="v", s=90, color=ACCENT, zorder=6)
ax4.set_ylim(0, ax4.get_ylim()[1] * 1.32)
ax4.text(prio_exp, 100 * freq[prio_exp] + 14, "choix A", ha="center", fontsize=8.5,
         color=ACCENT)
ax4.set_xticks(xs)
ax4.set_xticklabels([f"P{p}" for p in pid.PIL])
ax4.set_ylabel("% des $W$ admissibles", color=INK2, fontsize=9)
ax4.set_title("(d)  Ce que l'élicitation doit\ndépartager, et rien de plus", fontsize=11,
              color=INK, pad=8)
for s in ("top", "right"):
    ax4.spines[s].set_visible(False)

fig.tight_layout()
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z2_comparaison_regimes.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
