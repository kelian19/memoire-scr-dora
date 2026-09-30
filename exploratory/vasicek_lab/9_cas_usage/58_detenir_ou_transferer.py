#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
58 : detenir ou transferer ? Le SCR compare au PRIX du meme risque sur le marche cyber.

CE QUE LE MEMOIRE COMPARE DEJA, ET CE QUI MANQUE. Le chapitre resultats confronte le SCR
a trois choses : la charge de Formule Standard, les grandes pertes cyber reelles, et le
volume du marche cyber mondial. Trois comparaisons de NIVEAU. Il en manque une de PRIX.

L'IDEE. Un SCR n'est pas une perte, c'est du capital IMMOBILISE, et le capital coute.
Solvabilite II chiffre ce cout explicitement, par le taux de cout du capital de la marge
de risque : 6 % historiquement, ramene a 4,75 % par la directive (UE) 2025/2 (transposition
janvier 2027). Detenir le risque coute donc CoC x SCR par an. En face, le TRANSFERER coute
une prime, que le marche cyber facture avec un chargement : a un ratio de sinistralite de
l'ordre de 48 %, l'assureur encaisse environ 2,1 fois la prime pure.

LE VRAI PROBLEME N'EST PAS BINAIRE. On ne peut pas tout transferer : la capacite du marche
cyber est bornee par risque. La question actuarielle est donc le DIMENSIONNEMENT d'un
traite en exces de perte annuel (retention R, portee L) qui minimise le cout total

    cout(R, L) = prime(R, L) + CoC x SCR_retenu(R, L)

C'est un arbitrage classique retention/capital, applique ici a un risque dont la queue est
issue d'une cascade dirigee. Il donne une conclusion OPERATIONNELLE la ou le memoire ne
donnait qu'un niveau.

ECHELLE. Tout est calcule a l'echelle d'UNE ENTITE, pas du secteur : c'est la seule echelle
a laquelle la question du transfert se pose. La frequence est lue A LA TAILLE de l'entite
visee (lambda = 0,0917/an) et la severite transposee a la meme taille (x 0,8545), et non
prise dans un seau de firmes dix-neuf fois plus grandes comme dans une premiere version.
Voir le script 60, qui etablit les deux corrections et les compose.

Sortie : diagnostics + figure S19_detenir_ou_transferer.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)
import partial_id as pid                                          # noqa: E402

W_ = 84
# ECHELLE D'ENTITE, LECTURE COHERENTE (script 60). L'ancienne valeur 0,210 etait celle du
# seau "firmes a >= 10 evenements", dont les actifs medians valent 19 fois ceux de l'entite
# notionnelle, et qui est de surcroit defini par un filtre sur la grandeur meme qu'on estime.
# On lit desormais lambda A LA TAILLE de la cible (binomiale negative, elasticite 0,074) et
# l'on transpose la severite a la meme taille (elasticite 0,087, script 57).
# valeurs reportees a pleine precision depuis le script 60 : arrondir lambda a quatre
# decimales suffit a deplacer le SCR de 2 %, ce qui ferait diverger les deux scripts.
LAM_ENTITE = 0.09168423156000373
SEV_MULT = 0.8545
NY = 600_000                # a cette frequence la quasi-totalite des annees sont vides
ALPHA = 0.995

# --- parametres de marche, sources publiques -----------------------------------------
COC_ACTUEL = 0.0600         # marge de risque Solvabilite II, regime actuel
COC_REVISE = 0.0475         # directive (UE) 2025/2
LOSS_RATIO = 0.485          # ratio de sinistralite cyber (Beazley H1 2025)
COMBINED = 0.70             # ratio combine moyen 2022-2024


def titre(s):
    print("\n" + "=" * W_ + f"\n{s}\n" + "=" * W_)


# =====================================================================================
titre("La distribution de perte annuelle, a l'echelle d'une entite")
# =====================================================================================
W = pid.expert_matrix()
ev = pid.Evaluator(lam=LAM_ENTITE, n_years=NY, alpha=ALPHA)
# transposition de severite a la taille de l'entite : homothetie exacte (GPD = famille
# d'echelle au-dessus du seuil, source OpRisk non plafonnee). Cf. scripts 57 et 60.
ev.cum = ev.cum * SEV_MULT
card = pid.card_dist_all(W)[0]

# on rejoue from_card pour recuperer le VECTEUR des pertes annuelles, pas seulement
# ses deux resumes : l'analyse par tranche a besoin de toute la loi.
K = np.empty(ev.T, dtype=np.int64)
for a in range(pid.NP_):
    idx = ev.idx_by_am[a]
    if idx.size == 0:
        continue
    cdf = np.cumsum(card[a][1:])
    cdf[-1] = 1.0
    K[idx] = np.searchsorted(cdf, ev.u[idx], side="right") + 1
np.clip(K, 1, pid.NP_, out=K)
X = np.bincount(ev.year_of, weights=ev.cum[ev._ar, K - 1], minlength=ev.n_years)

scr0 = float(np.quantile(X, ALPHA))
esp0 = float(X.mean())
print(f"annees simulees            : {ev.n_years:,}")
print(f"annees sans aucun sinistre : {(X == 0).mean():.1%}")
print(f"perte moyenne E[X]         : {esp0:,.2f} M EUR")
print(f"SCR = VaR {ALPHA:.1%}          : {scr0:,.1f} M EUR")
print(f"multiple de capital SCR/E[X] : {scr0/esp0:,.1f}")
print(f"TVaR {ALPHA:.1%}               : {X[X >= scr0].mean():,.1f} M EUR")

titre("Le point de depart : tout detenir")
for lab, coc in (("regime actuel (6,00 %)", COC_ACTUEL),
                 ("directive (UE) 2025/2 (4,75 %)", COC_REVISE)):
    print(f"{lab:<34} cout annuel de detention = {coc*scr0:8,.1f} M EUR"
          f"   soit {coc*scr0/esp0:5.2f} x la sinistralite attendue")
print("\nAutrement dit : immobiliser le capital coute plusieurs fois ce que le risque")
print("coute en moyenne. C'est ce rapport qui rend le transfert economiquement pertinent,")
print("et c'est un rapport que le memoire ne calculait nulle part.")


# =====================================================================================
titre("Le traite en exces de perte annuel : cout total en fonction de la portee")
# =====================================================================================
def evalue(R, L, coc):
    """Retention R, portee L. Retourne (prime, SCR retenu, cout total, part cedee)."""
    cede = np.minimum(np.maximum(X - R, 0.0), L)
    retenu = X - cede
    prime = cede.mean() / LOSS_RATIO           # l'assureur charge 1/loss_ratio
    scr_r = float(np.quantile(retenu, ALPHA))
    return prime, scr_r, prime + coc * scr_r, cede.mean() / esp0


RETENTIONS = [0.0, 5.0, 10.0, 25.0]
PORTEES = np.array([0, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000], float)

print("Retention R = 0 (le traite prend tout des le premier euro), CoC = 4,75 %\n")
print(f"{'portee L':>10}{'prime':>10}{'SCR retenu':>13}{'cout capital':>15}"
      f"{'cout total':>13}{'% cede':>9}")
best = None
for L in PORTEES:
    p, s, c, part = evalue(0.0, L, COC_REVISE)
    flag = ""
    if best is None or c < best[1]:
        best, flag = (L, c), ""
    print(f"{L:>10,.0f}{p:>10,.2f}{s:>13,.1f}{COC_REVISE*s:>15,.2f}{c:>13,.2f}{part:>8.0%}{flag}")

# balayage fin pour l'optimum
grid = np.linspace(0, 1200, 481)
for lab, coc in (("6,00 %", COC_ACTUEL), ("4,75 %", COC_REVISE)):
    couts = np.array([evalue(0.0, L, coc)[2] for L in grid])
    i = int(np.argmin(couts))
    p, s, c, part = evalue(0.0, grid[i], coc)
    c0 = evalue(0.0, 0.0, coc)[2]
    print(f"\nCoC = {lab} : portee optimale L* = {grid[i]:,.0f} M EUR")
    print(f"   cout total {c:,.2f} M EUR/an contre {c0:,.2f} en tout-detention "
          f"({c/c0-1:+.0%})")
    if coc == COC_REVISE:
        gain_revise = c / c0 - 1.0     # relu plus bas, au lieu d'un -47 % code en dur
    print(f"   prime {p:,.2f} + cout du capital residuel {coc*s:,.2f} ; "
          f"{part:.0%} de la sinistralite cedee")
    # LE TAUX MARGINAL DE PRIME, ET LE SEUIL D'INEFFICACITE QUI EN DECOULE.
    # Le memoire publie kappa* = 1 - taux marginal / CoC comme seuil au-dela duquel detenir
    # redevient preferable, et ni le taux marginal ni kappa* n'etaient imprimes : ils
    # etaient calcules pendant la redaction, donc verifiables par personne.
    _h = grid[1] - grid[0]
    _tm = (evalue(0.0, grid[i] + _h, coc)[0] - evalue(0.0, max(grid[i] - _h, 0.0), coc)[0]) \
        / (2 * _h)
    print(f"   taux marginal de prime a l'optimum : {_tm:.4f} par euro de portee")
    print(f"   seuil d'inefficacite kappa* = 1 - {_tm:.4f}/{coc:.4f} = {1 - _tm/coc:.0%}")

titre("Le seuil d'indifference : a partir de quel chargement detenir redevient rationnel ?")
# transferer tout le risque jusqu'a L = SCR ; a quel loss ratio le cout s'egalise-t-il ?
cede_full = np.minimum(X, scr0)
esp_cede = cede_full.mean()
retenu_full = X - cede_full
scr_res = float(np.quantile(retenu_full, ALPHA))
for lab, coc in (("6,00 %", COC_ACTUEL), ("4,75 %", COC_REVISE)):
    # prime = esp_cede / lr ; indifference : prime + coc*scr_res = coc*scr0
    denom = coc * (scr0 - scr_res)
    lr_star = esp_cede / denom if denom > 0 else np.nan
    print(f"CoC = {lab} : ratio de sinistralite d'indifference = {lr_star:.1%}")
print(f"\nratio de sinistralite observe sur le marche cyber : {LOSS_RATIO:.1%}")
print("Si le ratio d'indifference est INFERIEUR au ratio de marche, le transfert est")
print("moins cher que la detention. L'ecart mesure la marge de negociation de l'entite.")

titre("La correction qui empeche de conclure trop vite : le transfert n'est pas gratuit")
print("Le resultat ci-dessus est MECANIQUE : la VaR du retenu vaut max(SCR - L, 0), donc")
print("elle s'annule exactement en L = SCR, et l'optimum tombe la par construction. Trois")
print("raisons interdisent de s'y arreter, et toutes trois vont dans le meme sens.")
print("  (i)  CAPACITE. Une couverture agregee de pres de 500 M EUR est au plafond de ce")
print("       que le marche cyber sait placer sur un seul risque.")
print("  (ii) RISQUE DE BASE. Les polices cyber excluent la guerre et les infrastructures")
print("       essentielles : une part de la queue modelisee n'est pas assurable.")
print("  (iii) CONTREPARTIE. Sous Solvabilite II, ceder ne supprime pas l'exigence, cela")
print("       SUBSTITUE au risque cyber un risque de DEFAUT DE CONTREPARTIE (module SCR_def")
print("       sur les creances de reassurance).")
print("\nOn resume les trois par un coefficient d'inefficacite kappa : le capital residuel")
print("devient max(SCR - L, 0) + kappa x L. Le transfert reste avantageux tant que le taux")
print("marginal de prime reste sous CoC x (1 - kappa).\n")


def evalue_k(L, coc, kappa):
    cede = np.minimum(X, L)
    prime = cede.mean() / LOSS_RATIO
    scr_r = float(np.quantile(X - cede, ALPHA)) + kappa * L
    return prime, scr_r, prime + coc * scr_r


print(f"{'kappa':>8}{'L* (M EUR)':>13}{'cout total':>13}{'vs tout detenir':>18}"
      f"{'% cede':>9}")
for kappa in (0.00, 0.02, 0.05, 0.10, 0.20, 0.30):
    cc = np.array([evalue_k(L, COC_REVISE, kappa)[2] for L in grid])
    i = int(np.argmin(cc))
    c0k = evalue_k(0.0, COC_REVISE, kappa)[2]
    part = np.minimum(X, grid[i]).mean() / esp0
    print(f"{kappa:>8.2f}{grid[i]:>13,.0f}{cc[i]:>13,.2f}{cc[i]/c0k-1:>17.0%}{part:>9.0%}")

# seuil critique : le transfert cesse d'etre marginalement rentable quand le taux
# marginal de prime depasse CoC x (1 - kappa).
# POURQUOI LE PAS EST RELATIF ET NON DE 50 M EUR. Ce bloc tourne a l'echelle d'ENTITE, ou
# scr0 vaut environ 170 M EUR : un pas de 50 y represente 30 % de la portee, si bien que la
# difference finie ne mesurait plus la pente EN L = SCR mais sa moyenne sur un large
# intervalle. La prime etant concave en L, ce biais etait haussier, et le script imprimait
# 0,0116 ici contre 0,0103 a l'optimum plus haut : deux valeurs de la MEME grandeur, d'ou le
# 76 % contre 78 % qui a circule. Le pas est desormais de 1 % de la portee, et les deux
# impressions concordent. C'est une correction d'artefact numerique, pas une recalibration :
# aucune entree du modele ne change.
dL = max(scr0 / 100.0, 1e-6)
p_hi = np.minimum(X, scr0).mean() / LOSS_RATIO
p_lo = np.minimum(X, scr0 - dL).mean() / LOSS_RATIO
taux_marginal = (p_hi - p_lo) / dL
kappa_star = 1.0 - taux_marginal / COC_REVISE
print(f"\ntaux marginal de prime pres de L = SCR (pas de {dL:.2f} M EUR, soit 1 % de la portee) :")
print(f"   {taux_marginal:.4f} par euro de portee")
print(f"   soit {taux_marginal:.2%} de taux sur ligne, contre un cout du capital economise")
print(f"   de {COC_REVISE:.2%} par euro.")
print(f"\nSEUIL CRITIQUE : kappa* = 1 - {taux_marginal:.4f}/{COC_REVISE:.4f} = {kappa_star:.0%}")
print("Il faudrait donc que plus des trois quarts de la couverture soient inefficaces")
print("(capacite indisponible, exclusion, ou defaut de contrepartie) pour que la detention")
# Le gain etait ecrit -47 % en dur : c'etait la valeur de l'ANCIENNE echelle (lambda du seau
# a 0,21). A l'echelle corrigee le script calcule -64 % plus haut. On le relit au lieu de le
# reciter, pour qu'il ne puisse plus se perimer.
print("redevienne preferable. La conclusion est robuste, mais le gain affiche"
      f" ({gain_revise:+.0%}) est")
print("une BORNE SUPERIEURE et doit etre cite comme telle.")

titre("Ce que la conformite DORA change a cet arbitrage")
# le SCR conforme vs non conforme : la remediation deplace-t-elle l'optimum ?
print("La remediation agit sur la FREQUENCE d'amorce. On rejoue l'arbitrage a lambda")
print("reduit, ce qui simule une entite devenue conforme.\n")
print(f"{'lambda':>9}{'SCR':>10}{'E[X]':>9}{'L* (4,75 %)':>14}{'cout total':>13}")
for lam in (LAM_ENTITE, LAM_ENTITE * 0.7, LAM_ENTITE * 0.5):
    e2 = pid.Evaluator(lam=lam, n_years=NY, alpha=ALPHA)
    e2.cum = e2.cum * SEV_MULT          # meme transposition de taille que plus haut
    K2 = np.empty(e2.T, dtype=np.int64)
    for a in range(pid.NP_):
        idx = e2.idx_by_am[a]
        if idx.size == 0:
            continue
        cdf = np.cumsum(card[a][1:]); cdf[-1] = 1.0
        K2[idx] = np.searchsorted(cdf, e2.u[idx], side="right") + 1
    np.clip(K2, 1, pid.NP_, out=K2)
    X2 = np.bincount(e2.year_of, weights=e2.cum[e2._ar, K2 - 1], minlength=e2.n_years)
    s2, m2 = float(np.quantile(X2, ALPHA)), float(X2.mean())

    def ev2(L):
        cede = np.minimum(X2, L)
        return cede.mean() / LOSS_RATIO + COC_REVISE * float(np.quantile(X2 - cede, ALPHA))
    cc = np.array([ev2(L) for L in grid])
    i = int(np.argmin(cc))
    print(f"{lam:>9.3f}{s2:>10,.0f}{m2:>9,.2f}{grid[i]:>14,.0f}{cc[i]:>13,.2f}")
print("\nLecture : la conformite ne fait pas que baisser le capital, elle deplace le")
print("point d'equilibre entre retention et transfert. C'est un resultat de PILOTAGE,")
print("pas seulement de mesure.")


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
ACCENT = "#a6002e"
BL = ["#7baafd", "#4c79c7", "#204993"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 3.3),
                                    gridspec_kw={"width_ratios": [1, 1.15, 1]})

# (a) detenir vs transferer, en tout ou rien
labs = ["détenir\n(CoC 6,00 %)", "détenir\n(CoC 4,75 %)", "sinistralité\nattendue"]
vals = [COC_ACTUEL * scr0, COC_REVISE * scr0, esp0]
cols = [BL[2], BL[1], MUTED]
ax1.bar(range(3), vals, color=cols, edgecolor="#fcfcfb", width=0.6)
for i, v in enumerate(vals):
    ax1.text(i, v * 1.03, f"{v:,.1f}", ha="center", fontsize=9.5, color=INK2)
ax1.set_xticks(range(3)); ax1.set_xticklabels(labs, fontsize=8)
ax1.set_ylabel("coût annuel (M€)", color=INK2)
ax1.set_title(f"(a)  Le capital coûte {COC_REVISE*scr0/esp0:.1f} fois\nla sinistralité attendue",
              fontsize=9.5, color=INK, pad=8)
for s_ in ("top", "right"):
    ax1.spines[s_].set_visible(False)

# (b) cout total en fonction de la portee
for lab, coc, col in (("CoC 6,00 %", COC_ACTUEL, BL[2]), ("CoC 4,75 %", COC_REVISE, BL[1])):
    cc = np.array([evalue(0.0, L, coc)[2] for L in grid])
    ax2.plot(grid, cc, color=col, lw=2.2, label=lab)
    i = int(np.argmin(cc))
    ax2.plot(grid[i], cc[i], "o", color=ACCENT, ms=7, zorder=5)
    ax2.annotate(f"$L^*={grid[i]:,.0f}$", (grid[i], cc[i]), textcoords="offset points",
                 xytext=(8, 10), fontsize=8.5, color=ACCENT)
ax2.axvline(scr0, color=INK, ls="--", lw=1.3)
ax2.text(scr0 * 1.05, ax2.get_ylim()[1] * 0.55, f"SCR = {scr0:,.0f}", fontsize=8.5, color=INK)
ax2.set_xlabel("portée du traité en excès de perte $L$ (M€)", color=INK2)
ax2.set_ylabel("coût annuel total (M€)", color=INK2)
ax2.legend(frameon=True, facecolor="#fcfcfb", edgecolor="none", framealpha=0.88, fontsize=9)
ax2.set_title("(b)  Prime + coût du capital résiduel :\nun optimum intérieur",
              fontsize=9.5, color=INK, pad=8)
for s_ in ("top", "right"):
    ax2.spines[s_].set_visible(False)

# (c) decomposition du cout a l'optimum
cc = np.array([evalue(0.0, L, COC_REVISE)[2] for L in grid])
i = int(np.argmin(cc))
p_, s_r, c_, part_ = evalue(0.0, grid[i], COC_REVISE)
p0_, s0_, c0_, _ = evalue(0.0, 0.0, COC_REVISE)
ax3.bar([0], [0], color="none")
ax3.bar([0], [c0_], color=BL[2], edgecolor="#fcfcfb", width=0.55, label="coût du capital")
ax3.bar([1], [COC_REVISE * s_r], color=BL[2], edgecolor="#fcfcfb", width=0.55)
ax3.bar([1], [p_], bottom=[COC_REVISE * s_r], color=ACCENT, edgecolor="#fcfcfb",
        width=0.55, label="prime de transfert")
# ESPACEMENT DES ETIQUETTES EN VALEUR ABSOLUE, PAS EN POURCENTAGE DE LA BARRE.
# Un decalage multiplicatif (c_ * 1.03 puis c_ * 1.14) place les deux etiquettes de la
# barre basse a 0,3 M d'ecart sur un axe qui monte a 8 : elles se chevauchaient. On les
# separe d'une fraction de la HAUTEUR DE L'AXE, ce qui est invariant a la valeur des barres.
_pas = 0.055 * c0_
ax3.text(0, c0_ + _pas, f"{c0_:,.1f}", ha="center", fontsize=9.5, color=INK2)
ax3.text(1, c_ + _pas, f"{c_:,.1f}", ha="center", fontsize=9.5, color=INK2)
ax3.text(1, c_ + 3 * _pas, f"{c_/c0_-1:+.0%}", ha="center", fontsize=10,
         color=ACCENT, fontweight="bold")
ax3.set_ylim(0, c0_ * 1.28)          # de la place pour les deux etiquettes empilees
ax3.set_xticks([0, 1])
ax3.set_ylim(0, max(c0_, c_) * 1.45)
ax3.set_xticklabels(["tout détenir", f"traité optimal\n$L^*={grid[i]:,.0f}$ M€"], fontsize=8)
ax3.set_ylabel("coût annuel (M€)", color=INK2)
# legende en HAUT A DROITE : la barre de gauche monte jusqu'au sommet de l'axe, la
# legende y recouvrait son etiquette de valeur.
ax3.legend(frameon=True, facecolor="#fcfcfb", edgecolor="none", framealpha=0.88, fontsize=8.5, loc="upper right")
ax3.set_title("(c)  Ce que le dimensionnement fait gagner", fontsize=9.5, color=INK, pad=8)
for s_ in ("top", "right"):
    ax3.spines[s_].set_visible(False)

# RESERVE EN POUCES, PAS EN FRACTION. Un rect a 0,90 reserve 10 % de la HAUTEUR au
# titre : correct sur une figure large de 5 pouces de haut, deux fois trop sur une
# figure empilee de 10 pouces, ou cela creait un bandeau blanc sous le titre. On
# reserve donc une hauteur FIXE de 0,42 pouce, quelle que soit la taille de la figure.
_top = 1.0 - 0.26 / fig.get_figheight()
fig.tight_layout(w_pad=1.8)
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S19_detenir_ou_transferer.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
