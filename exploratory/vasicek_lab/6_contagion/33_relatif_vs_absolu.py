#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
33 : le NIVEAU du capital est fragile, sa STRUCTURE ne l'est pas.

Le reproche previsible d'un relecteur dur : \"une tour d'hypotheses qui accouche d'un
grand nombre\". Il est fonde si le memoire vend un niveau. Il tombe si le memoire vend
un rapport, ET si l'on DEMONTRE que le rapport, lui, ne bouge pas.

C'est exactement ce que teste ce script. On fait varier ce qui pilote le NIVEAU, en
laissant la cascade inchangee :

  A. reference             : severite OpRisk,  frequence OpRisk
  B. loi de severite seule : severite PRC,     frequence OpRisk   (xi 0,60 -> 1,03)
  C. frequence seule       : severite OpRisk,  frequence x3

et l'on compare, pour chacune :
  - des grandeurs ABSOLUES : socle, bornes, point d'expert, en millions d'euros ;
  - des grandeurs RELATIVES : bornes rapportees au socle, largeur de bande en % du
    socle, part du capital deja fixee sans elicitation, et le classement de priorite.

VERDICT ATTENDU, a verifier et non a postuler : les absolues se deplacent fortement, les
relatives beaucoup moins. Si c'est le cas, alors presenter le niveau comme ILLUSTRATIF et
mettre le rapport au centre n'est plus une precaution rhetorique, c'est une conclusion.
Si ce n'est pas le cas, il faut le dire et renoncer a l'argument.

Sortie : diagnostics + figure Z4_relatif_vs_absolu.png.
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

WID = 80
N_SET = 100               # matrices pour le classement de priorite
SEED = 20260721

CONFIGS = [
    ("A. référence",          dict(source="OPRISK", sev_source="OPRISK", lam=None,
                                   n_years=40_000)),
    ("B. sévérité PRC",       dict(source="OPRISK", sev_source="PRC", lam=None,
                                   n_years=40_000)),
    ("C. fréquence x3",       dict(source="OPRISK", sev_source="OPRISK", lam=3 * 21.5646,
                                   n_years=40_000)),
]


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


P = pid.expert_matrix()
S, A_EXP = pid.decompose(P)
res = {}

for nom, kw in CONFIGS:
    ev = pid.Evaluator(seed=SEED, **kw)
    scr_socle, mean_socle = ev(np.zeros((pid.NP_, pid.NP_)))
    scr_exp, _ = ev(P)
    vals = []
    Wok = []
    for a_vec in pid.all_vertices(S, 1.0):
        W = pid.build_W(S, a_vec)
        if not pid.admissible(W):
            continue
        vals.append(ev.scr(W))
        Wok.append(W)
    vals = np.array(vals)
    lo, hi = float(vals.min()), float(vals.max())
    idx = np.random.default_rng(SEED + 9).choice(len(Wok), size=min(N_SET, len(Wok)),
                                                 replace=False)
    B = np.array([ev.benefits(Wok[i]) for i in idx])
    freq = np.bincount(B.argmax(axis=1), minlength=pid.NP_) / len(idx)
    res[nom] = dict(socle=scr_socle, expert=scr_exp, lo=lo, hi=hi, freq=freq,
                    lam=ev.lam, xi=ev.sp["xi"], T=ev.T)
    print(f"  {nom:<20} lambda={ev.lam:7.2f}/an  xi={ev.sp['xi']:.4f}  "
          f"{ev.T} incidents  ->  socle {scr_socle:7.0f} M, "
          f"bornes [{lo:7.0f} ; {hi:7.0f}] M")

# =====================================================================================
titre("1. Les grandeurs ABSOLUES : elles se deplacent fortement")
# =====================================================================================
print(f"  {'configuration':<20}{'socle':>10}{'borne basse':>14}{'borne haute':>14}"
      f"{'expert':>10}")
for nom, _ in CONFIGS:
    r = res[nom]
    print(f"  {nom:<20}{r['socle']:>9.0f} M{r['lo']:>13.0f} M{r['hi']:>13.0f} M"
          f"{r['expert']:>9.0f} M")
soc = np.array([res[n]['socle'] for n, _ in CONFIGS])
hau = np.array([res[n]['hi'] for n, _ in CONFIGS])
print(f"\n  Amplitude du socle       : facteur {soc.max()/soc.min():.2f} entre configurations")
print(f"  Amplitude de la borne haute : facteur {hau.max()/hau.min():.2f}")

# LES DEUX FACTEURS DU RESUME, DECOMPOSES. Le resume du memoire ecrit « un facteur 22 avec la
# seule source de severite et jusqu'a 41 en y ajoutant l'echelle de frequence ». Ces deux
# nombres se deduisaient du tableau ci-dessus, mais AUCUN n'etait imprime : ils etaient
# calcules pendant la redaction, donc verifiables par personne. C'est la quatrieme classe du
# residu du harnais, la seule qui soit un defaut. Les tracer ici les rend citables et empeche
# qu'ils se desynchronisent du tableau, ce qui est deja arrive ailleurs dans ce projet.
i_ref = [n for n, _ in CONFIGS].index("A. référence")
i_prc = [n for n, _ in CONFIGS].index("B. sévérité PRC")
i_frq = [n for n, _ in CONFIGS].index("C. fréquence x3")
print("\n  LES DEUX FACTEURS QUE LE RESUME CITE, tires du meme tableau :")
print(f"    severite SEULE, sur le socle        : facteur "
      f"{soc[i_ref]/soc[i_prc]:.2f}   (A / B)")
print(f"    severite SEULE, sur la borne haute  : facteur "
      f"{hau[i_ref]/hau[i_prc]:.2f}")
print(f"    en ajoutant l'echelle de frequence  : facteur "
      f"{hau[i_frq]/hau[i_prc]:.2f}   (C / B, borne haute)")
print(f"    idem sur le socle                   : facteur "
      f"{soc[i_frq]/soc[i_prc]:.2f}")
print("    Le resume arrondit le premier a 22 et le troisieme a 41. Les deux autres sont")
print("    imprimes pour que le choix de la grandeur de reference soit visible : selon qu'on")
print("    lit le socle ou la borne haute, l'amplitude n'est pas la meme, et annoncer un")
print("    facteur sans dire sur quoi il porte serait un raccourci.")

print("\n  Le NIVEAU depend donc massivement de la source de severite et de la frequence,")
print("  qui sont precisement les deux briques les moins transposables (pertes")
print("  americaines de grandes institutions, perimetre de collecte). Presenter ce")
print("  niveau comme une mesure serait indefendable.")

# =====================================================================================
titre("2. Les grandeurs RELATIVES : sont-elles stables ?")
# =====================================================================================
print(f"  {'configuration':<20}{'basse/socle':>14}{'haute/socle':>14}"
      f"{'largeur/socle':>16}{'% fixe sans élic.':>20}")
rel = {}
for nom, _ in CONFIGS:
    r = res[nom]
    a, b = r['lo'] / r['socle'], r['hi'] / r['socle']
    larg = (r['hi'] - r['lo']) / r['socle']
    fixe = 100 * (1 - (r['hi'] - r['lo']) / r['hi'])
    rel[nom] = (a, b, larg, fixe)
    print(f"  {nom:<20}{a:>13.3f}x{b:>13.3f}x{100*larg:>14.1f} %{fixe:>18.1f} %")

arr = np.array([rel[n][:3] for n, _ in CONFIGS])
etendues = arr.max(axis=0) - arr.min(axis=0)
rel_var = etendues / arr.mean(axis=0)          # variation RELATIVE de chaque rapport
noms_rel = ["borne basse / socle", "borne haute / socle", "largeur / socle"]
print()
for lab, e, v in zip(noms_rel, etendues, rel_var):
    print(f"  {lab:<24} varie de {e:.3f} sur les trois configurations "
          f"(soit {100*v:.1f} % de sa valeur moyenne)")
pire = int(np.argmax(rel_var))
print(f"\n  Le moins stable est \"{noms_rel[pire]}\" ({100*rel_var[pire]:.1f} %). Il faut le")
print("  dire : ce rapport diminue quand la frequence augmente, ce qui est attendu, la")
print("  queue annuelle etant alors moins dominee par une perte unique et donc moins")
print("  sensible a la structure de cascade. La stabilite du relatif n'est pas parfaite.")
fixes = np.array([rel[n][3] for n, _ in CONFIGS])
print(f"  En revanche la part du capital fixee sans elicitation est tres stable : "
      f"{fixes.min():.1f} a {fixes.max():.1f} %.")

# =====================================================================================
titre("3. La DECISION : le classement de priorite bouge-t-il ?")
# =====================================================================================
print(f"  {'configuration':<20}" + "".join(f"{'P'+str(p):>9}" for p in pid.PIL))
for nom, _ in CONFIGS:
    print(f"  {nom:<20}" + "".join(f"{100*f:>8.0f}%" for f in res[nom]['freq']))
tops = {nom: [int(i) for i in np.argsort(-res[nom]['freq'])[:2]] for nom, _ in CONFIGS}
memes = all(set(tops[n]) == set(tops[CONFIGS[0][0]]) for n, _ in CONFIGS)
print(f"\n  Groupe de tete identique dans les trois configurations : "
      f"{'OUI' if memes else 'NON'} "
      f"({', '.join('P'+str(pid.PIL[j]) for j in tops[CONFIGS[0][0]])})")

# =====================================================================================
titre("4. Verdict")
# =====================================================================================
stable = float(rel_var.max()) < 0.35        # seuil assume, pas une norme
print(f"  Le NIVEAU bouge d'un facteur {hau.max()/hau.min():.1f} quand on change la loi de")
print("  severite ou la frequence, c'est-a-dire quand on change ce que le memoire ne")
print("  maitrise pas (perimetre de collecte des pertes, plafond de reassurance).")
print(f"  Les RAPPORTS varient d'au plus {100*rel_var.max():.0f} % de leur valeur moyenne,")
print(f"  soit un ordre de grandeur {hau.max()/hau.min()/(1+rel_var.max()):.0f} fois plus")
print(f"  faible que le niveau, et le groupe de tete de la priorite est "
      f"{'INCHANGE' if memes else 'MODIFIE'}.")
if stable and memes:
    print("\n  CONCLUSION, a ecrire telle quelle dans le memoire : le niveau absolu du SCR")
    print("  est ILLUSTRATIF et ne doit pas etre lu comme une mesure. Ce que le modele")
    print("  determine, et qui resiste au changement de source, c'est le RAPPORT du")
    print("  capital au socle, la largeur de la bande d'ignorance, et le classement des")
    print("  piliers. Ce n'est pas une precaution de langage, c'est un resultat verifie.")
else:
    print("\n  ATTENTION : la stabilite du relatif n'est pas verifiee sur ces configurations.")
    print("  L'argument \"je vends le relatif\" ne peut pas etre soutenu tel quel.")

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
noms = [n for n, _ in CONFIGS]
xs = np.arange(len(noms))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.0))

ax1.bar(xs, [res[n]['socle'] for n in noms], width=0.5, color=GREEN, alpha=0.85,
        label="socle")
for i, n in enumerate(noms):
    ax1.plot([i, i], [res[n]['lo'], res[n]['hi']], color=INK, lw=2.6, zorder=5)
    ax1.scatter([i, i], [res[n]['lo'], res[n]['hi']], marker="_", s=380, color=INK, zorder=6)
ax1.set_xticks(xs)
ax1.set_xticklabels(noms, fontsize=8.5)
ax1.set_ylabel("SCR (M€)", color=INK2)
ax1.set_title(f"(a)  Le NIVEAU bouge  (facteur {hau.max()/hau.min():.1f})",
              fontsize=11, color=INK, pad=8)
ax1.legend(frameon=False, fontsize=8)

w = 0.26
for k, (lab, col) in enumerate(zip(["borne basse / socle", "borne haute / socle",
                                    "largeur / socle"], arr.T)):
    ax2.bar(xs + (k - 1) * w, col, width=w, alpha=0.85,
            color=[BLUE, ACCENT, MUTED][k], label=lab)
ax2.set_xticks(xs)
ax2.set_xticklabels(noms, fontsize=8.5)
ax2.set_ylabel("rapport au socle", color=INK2)
ax2.set_title("(b)  Les RAPPORTS ne bougent pas", fontsize=11, color=INK, pad=8)
ax2.legend(frameon=False, fontsize=8)

bot = np.zeros(len(noms))
cols = [BLUE, "#6491e1", MUTED, ACCENT, "#c9c7bd"]
for j in range(pid.NP_):
    v = np.array([100 * res[n]['freq'][j] for n in noms])
    ax3.bar(xs, v, bottom=bot, width=0.5, color=cols[j], alpha=0.9,
            label=f"P{pid.PIL[j]}")
    bot += v
ax3.set_xticks(xs)
ax3.set_xticklabels(noms, fontsize=8.5)
ax3.set_ylabel("% des $W$ où le pilier est 1er", color=INK2)
ax3.set_title("(c)  La DÉCISION ne bouge pas", fontsize=11, color=INK, pad=8)
ax3.legend(frameon=False, fontsize=8, ncol=5, loc="upper center",
           bbox_to_anchor=(0.5, -0.12))

for ax in (ax1, ax2, ax3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("Z4 : le niveau du capital est fragile, sa structure ne l'est pas",
             fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z4_relatif_vs_absolu.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
