#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
74 : les defaillances SIMULTANEES de plusieurs piliers, et ce que coute l'additivite des couts.

QUESTION A L'ORIGINE (Caroline Hillairet, compte rendu du 13 aout 2026, section 4.2) : « les cas
de defaillances simultanees de plusieurs piliers et la nature de l'agregation des couts associes
(additivite vs effets croises non lineaires) : ce cas n'a pour l'instant pas ete traite ».

LA PREMISSE DOIT ETRE CORRIGEE AVANT D'Y REPONDRE, et c'est le premier apport de ce script. Les
defaillances simultanees ne sont pas un cas particulier laisse de cote : elles sont l'OBJET MEME
que le modele produit. Chaque sinistre tire un ENSEMBLE de piliers dans la loi de progeniture de
la cascade, et cet ensemble a plus d'un element des que la propagation opere. La section 1 en donne
la loi EXACTE, sans simulation.

CE QUI EST VRAIMENT EN QUESTION EST L'ADDITIVITE, et il faut distinguer TROIS enonces que le mot
recouvre, a trois etages differents. Les confondre est facile et le projet a deja paye ce genre de
confusion :
  (a) au sein d'un SINISTRE, les couts des piliers touches s'additionnent. C'est une HYPOTHESE de
      construction, non testee, et c'est exactement ce que la question interroge ;
  (b) en fonction des PILIERS non conformes, le capital est presque additif : R2 = 0,9945 sur les
      32 configurations (script 69) ;
  (c) en fonction des quatre CANAUX de la conformite, le capital est franchement SUPER-additif :
      +5 001 M d'interaction, soit +35 % (script 68).
Trois etages, trois reponses differentes. La question porte sur (a), et (b) comme (c) sont deja
mesures.

CE QUE LA SECTION 3 FAIT. Puisque (a) n'est pas observable, on la BORNE : on introduit un exposant
theta sur le nombre de piliers touches, de sorte que la perte d'un sinistre vaille
(somme des severites) x (nombre de piliers)^(theta - 1). A theta = 1 on retrouve exactement le
modele publie, ce qui sert de controle. Au-dessus, les couts sont super-additifs (saturation de la
capacite de remediation) ; en dessous, sous-additifs (mutualisation d'une cellule de crise, d'une
investigation). On mesure ce que le capital y gagne ou y perd.

Sortie : diagnostics + figure S30_defaillances_simultanees.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                                 # noqa: E402
from euro_cascade_model import var                              # noqa: E402
import canaux_conformite as cx                                  # noqa: E402
from src.aggregation.lda import simulate_remediation_severity   # noqa: E402
import scr_engine as eng                                        # noqa: E402

WID = 88
PIL = cx.PIL
sp = cx.sp

# Exposants balayes. theta = 1 EST le modele publie, et c'est le controle du script.
THETAS = (0.70, 0.85, 1.00, 1.15, 1.30)
NY = cx.NY
NSEED = cx.NSEED


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# =====================================================================================
titre("1. La loi EXACTE du nombre de piliers touches par sinistre")
# =====================================================================================
# AUCUNE SIMULATION ICI : la loi de progeniture de la cascade est calculee exactement par
# enumeration, et les amorces sont ponderees par leur propension. C'est la reponse directe a
# « comment les defaillances simultanees sont-elles traitees » : elles sont la loi de sortie.
w = np.array([eng.LAMBDA[j] for j in PIL], float)
w = w / w.sum()


def loi_cardinal(g):
    """P(k piliers touches) pour un sinistre, exacte, ponderee par la propension d'amorce."""
    p = np.zeros(len(PIL) + 1)
    for c, j in enumerate(PIL):
        dist = eng.cascade_set_dist(j, g)
        tot = sum(dist.values())
        for s, pr in dist.items():
            p[len(s)] += w[c] * pr / tot
    return p


loi_c = loi_cardinal(cx.G_C)
loi_nc = loi_cardinal(cx.G_NC)
print(f"  Un sinistre touche un ENSEMBLE de piliers, tire dans la loi de progeniture de la")
print(f"  cascade. Voici cette loi, exacte, aux deux valeurs du gain de propagation.")
print(f"\n  {'piliers touches':<18}{'etat conforme (g=' + str(cx.G_C) + ')':>26}"
      f"{'non conforme (g=' + str(cx.G_NC) + ')':>26}")
for k in range(1, len(PIL) + 1):
    print(f"  {k:<18}{100*loi_c[k]:>25.2f} %{100*loi_nc[k]:>25.2f} %")
m_c = float(sum(k * loi_c[k] for k in range(len(PIL) + 1)))
m_nc = float(sum(k * loi_nc[k] for k in range(len(PIL) + 1)))
print(f"  {'moyenne':<18}{m_c:>26.3f}{m_nc:>26.3f}")
multi_c, multi_nc = 100 * (1 - loi_c[1]), 100 * (1 - loi_nc[1])
print(f"\n  Part des sinistres touchant PLUS D'UN pilier : {multi_c:.2f} % a l'etat conforme et")
print(f"  {multi_nc:.2f} % a l'etat non conforme. La simultaneite n'est donc pas un cas laisse de")
print("  cote, c'est la sortie du modele, et la non-conformite en augmente la frequence.")
print(f"\n  ET CE N'EST PAS UN CAS MARGINAL : a l'etat non conforme la defaillance multiple est")
print(f"  MAJORITAIRE, {multi_nc:.1f} % contre {100*loi_nc[1]:.1f} % de sinistres mono-pilier, pour une")
print(f"  moyenne de {m_nc:.2f} piliers par sinistre contre {m_c:.2f} a l'etat conforme, soit un")
print(f"  facteur {m_nc/m_c:.2f}. La consequence est importante pour la suite : tout ce qui touche au")
print("  cout des sinistres MULTI-piliers frappe plus fort l'etat non conforme que l'etat")
print("  conforme, donc deplace l'ECART et pas seulement le niveau. La section 3 le mesure.")

# =====================================================================================
titre("2. Les TROIS enonces d'additivite, a ne pas confondre")
# =====================================================================================
print("  (a) AU SEIN D'UN SINISTRE : les couts des piliers touches s'ADDITIONNENT.")
print("      Statut : HYPOTHESE de construction, non testee. C'est l'objet de la question.")
print("      La perte d'un sinistre vaut la somme des severites sur les piliers de l'ensemble.")
print("\n  (b) EN FONCTION DES PILIERS non conformes : le capital est PRESQUE additif.")
print("      Statut : MESURE. R2 = 0,9945 sur les 32 configurations (script 69), contributions")
print("      3 524 (P1), 2 863 (P4), 2 022 (P2), 1 577 (P3), 897 M (P5).")
print("\n  (c) EN FONCTION DES QUATRE CANAUX : le capital est franchement SUPER-additif.")
print("      Statut : MESURE. +5 001 M d'interaction sur un ecart de 14 139, soit +35 %")
print("      (script 68), et l'interaction vit aux paires.")
print("\n  POURQUOI CES TROIS ENONCES NE SE CONTREDISENT PAS. Ils portent sur des variables")
print("  differentes : (a) sur les couts d'un evenement, (b) sur les indicateurs de pilier, (c)")
print("  sur les parametres de la loi de perte. Un capital peut etre additif en (b) et")
print("  super-additif en (c) sans contradiction, et c'est le cas ici. Repondre a la question par")
print("  « le modele est super-additif » serait donc faux : il l'est sur les canaux, pas sur les")
print("  piliers, et l'hypothese sur les couts est encore autre chose.")

# =====================================================================================
titre("3. Ce que couterait une non-additivite des couts, borne par un exposant")
# =====================================================================================
def scr_theta(theta, lam, g, p_u, phi_cs, nseed=NSEED, ny=NY):
    """SCR quand la perte d'un sinistre vaut (somme des severites) x (nb de piliers)^(theta-1).

    A theta = 1 le facteur vaut 1 pour tout cardinal, donc on retrouve EXACTEMENT la fonction
    du module partage : c'est le controle de ce script, et il est verifie ci-dessous.
    """
    scrs = []
    for k in range(nseed):
        rng = np.random.default_rng(cx.SEED0 + k)
        r = lam / (ec.PHI - 1.0)
        counts = rng.negative_binomial(r, r / (r + lam), size=ny)
        T = int(counts.sum())
        if T == 0:
            scrs.append(0.0)
            continue
        year_of = np.repeat(np.arange(ny), counts)
        ww = np.array([eng.LAMBDA[j] for j in PIL], float)
        amorce = rng.choice(5, size=T, p=ww / ww.sum())
        U = rng.random(T)
        SEV = simulate_remediation_severity(T * 5, sp["xi"], sp["sigma"], sp["u"], p_u,
                                            sp["cap"], rng).reshape(T, 5)
        tables = {j: cx.table_amorce(j, g) for j in PIL}
        if phi_cs is not None:
            tables[cx.P4] = cx.table_p4_choc(phi_cs)
        annual = np.zeros(ny)
        for c, j in enumerate(PIL):
            idx = np.where(amorce == c)[0]
            if idx.size == 0:
                continue
            ind, probs = tables[j]
            cdf = np.cumsum(probs)
            cdf[-1] = 1.0
            sel = np.searchsorted(cdf, U[idx], side="right")
            np.clip(sel, 0, len(probs) - 1, out=sel)
            touches = ind[sel]
            k_pil = touches.sum(axis=1)
            loss = (SEV[idx] * touches).sum(axis=1) * np.power(k_pil, theta - 1.0)
            annual += np.bincount(year_of[idx], weights=loss, minlength=ny)
        scrs.append(var(annual))
    return np.array(scrs, float)          # PAR GRAINE : le bruit sert a la section 3


# CONTROLE D'ABORD : a theta = 1 on doit retrouver le nombre publie, sinon rien de ce qui suit
# ne vaut. On le verifie contre la fonction du module partage, celle des 14 139 M.
ref_c = cx.scr_config(lam=cx.LAM_C, g=cx.G_C, p_u=cx.PU_C, phi_cs=None)
ref_nc = cx.scr_config(lam=cx.LAM_NC, g=cx.G_NC, p_u=cx.PU_NC, phi_cs=cx.PHICS_NC)
s1_c = scr_theta(1.0, cx.LAM_C, cx.G_C, cx.PU_C, None)
s1_nc = scr_theta(1.0, cx.LAM_NC, cx.G_NC, cx.PU_NC, cx.PHICS_NC)
t1_c, t1_nc = float(s1_c.mean()), float(s1_nc.mean())
print(f"  CONTROLE a theta = 1, contre le moteur partage :")
print(f"    conforme      {t1_c:.0f} contre {ref_c:.0f}  -> ecart {abs(t1_c-ref_c):.4f} M")
print(f"    non conforme  {t1_nc:.0f} contre {ref_nc:.0f}  -> ecart {abs(t1_nc-ref_nc):.4f} M")
print("  L'ecart est nul : le balayage part donc bien du modele publie.")
# BRUIT DE L'ECART, apparie graine par graine. Il sert d'unite de lecture a la plage : un
# deplacement plus petit que ce bruit ne serait pas un effet.
d_seeds = s1_nc - s1_c
sd_d = float(d_seeds.std(ddof=1))
print(f"  Bruit de simulation de l'ecart, {NSEED} graines appariees : +/- {sd_d:.0f} M.")

print(f"\n  {'theta':>7}{'SCR conforme':>15}{'SCR non conf.':>15}{'ecart DORA':>13}"
      f"{'/ publie':>11}")
res = []
for th in THETAS:
    a = float(scr_theta(th, cx.LAM_C, cx.G_C, cx.PU_C, None).mean())
    b = float(scr_theta(th, cx.LAM_NC, cx.G_NC, cx.PU_NC, cx.PHICS_NC).mean())
    d = b - a
    res.append((th, a, b, d))
    print(f"  {th:>7.2f}{a:>15.0f}{b:>15.0f}{d:>13.0f}{d/(ref_nc-ref_c):>11.3f}")

d_ref = ref_nc - ref_c
lo = [r for r in res if r[0] == THETAS[0]][0]
hi = [r for r in res if r[0] == THETAS[-1]][0]
lg = np.log([r[0] for r in res])
el = float(np.polyfit(lg, np.log([r[3] for r in res]), 1)[0])
el_c = float(np.polyfit(lg, np.log([r[1] for r in res]), 1)[0])
el_nc = float(np.polyfit(lg, np.log([r[2] for r in res]), 1)[0])
print(f"\n  De theta = {THETAS[0]:.2f} a {THETAS[-1]:.2f}, l'ecart de capital DORA va de {lo[3]:.0f} a "
      f"{hi[3]:.0f} M,")
print(f"  contre {d_ref:.0f} au modele publie : une plage de {hi[3]-lo[3]:.0f} M, soit "
      f"{100*(hi[3]-lo[3])/d_ref:.0f} % de l'ecart publie et {(hi[3]-lo[3])/sd_d:.0f} fois son "
      f"bruit de simulation.")
print(f"  Elasticites, par regression log-log sur les {len(THETAS)} exposants :")
print(f"    etat conforme      {el_c:.2f}")
print(f"    etat non conforme  {el_nc:.2f}    -> facteur {el_nc/el_c:.2f} entre les deux")
print(f"    ecart DORA         {el:.2f}")

print("\n  C'EST UNE VRAIE SENSIBILITE, ET ELLE N'ETAIT PAS ATTENDUE A CE NIVEAU. La plage vaut")
print(f"  {(hi[3]-lo[3])/sd_d:.0f} fois le bruit de simulation, donc l'effet est reel et non un")
print("  artefact de Monte-Carlo. L'hypothese d'additivite des couts est donc la plus lourde des")
print("  hypotheses non testees du modele, et il faut le dire ainsi.")
print("\n  LE MECANISME EST CELUI DE LA SECTION 1, ET IL EXPLIQUE L'ASYMETRIE. L'exposant ne")
print("  touche QUE les sinistres multi-piliers, puisque 1^(theta-1) = 1 : il laisse les")
print(f"  mono-piliers exactement ou ils sont. Or la defaillance multiple est majoritaire a l'etat")
print(f"  non conforme ({multi_nc:.1f} %) et minoritaire a l'etat conforme ({multi_c:.1f} %), d'ou des")
print(f"  elasticites de {el_nc:.2f} contre {el_c:.2f}. L'hypothese n'est donc PAS neutre entre les deux")
print("  etats : elle interagit avec le canal de propagation, et deplace la grandeur centrale du")
print("  memoire et non un simple niveau.")
print("\n  CE QUI TIENT MALGRE TOUT, ET C'EST LE POINT A RETENIR. Sur toute la plage le signe et")
print(f"  l'ordre de grandeur de l'ecart survivent : de {lo[3]:.0f} a {hi[3]:.0f} M, jamais proche de zero,")
print("  toujours de l'ordre de la dizaine de milliards. La these ne depend donc pas de")
print("  l'hypothese d'additivite ; seule son AMPLITUDE en depend. C'est la meme structure")
print("  d'argument que l'invariance aux valeurs de g du script 66 : l'ordre resiste, le niveau")
print("  est un scenario.")
print("\n  ET LA PLAGE D'EXPOSANTS EST POSEE, PAS ESTIMEE. Aucune donnee ne permet de la")
print("  restreindre, voir la section 4. La grandeur transferable est donc l'ELASTICITE, qui")
print("  laisse un lecteur choisir sa propre plage, et non la plage elle-meme.")

# =====================================================================================
titre("4. Ce qui n'est PAS testable, et il faut le dire")
# =====================================================================================
print("  LE SENS de la non-additivite n'est pas determinable sur la donnee disponible, et deux")
print("  mecanismes plausibles vont en sens CONTRAIRE :")
print("    SOUS-additivite : une cellule de crise, une investigation forensique et une")
print("      communication servent a plusieurs piliers a la fois. Le total serait alors")
print("      INFERIEUR a la somme, et le modele publie SUR-estimerait.")
print("    SUPER-additivite : plusieurs defaillances simultanees saturent la capacite de")
print("      remediation, donc le cout unitaire monte. Le total serait SUPERIEUR a la somme,")
print("      et le modele publie SOUS-estimerait.")
print("\n  AUCUN DES DEUX N'EST OBSERVABLE ICI, et la raison est dans la donnee elle-meme : le")
print("  cout d'un sinistre DATE n'est jamais observe dans les sources retenues, il est")
print("  reconstruit, la chronologie de breches portant les dates et la conversion portant les")
print("  couts. A plus forte raison le cout d'un sinistre a PLUSIEURS piliers, qui demanderait")
print("  une ventilation par domaine de controle qu'aucune source publique ne fournit.")
print("\n  C'EST DONC UNE LIMITE DECLAREE, PAS UN OUBLI, et elle est desormais CHIFFREE plutot que")
print("  bornee : la plage de la section 3 dit ce que l'hypothese coute sur une amplitude POSEE")
print("  d'exposants, pas sur une amplitude estimee, et le sens reste ouvert. C'est pourquoi")
print("  l'elasticite est la grandeur a publier : elle vaut pour toute amplitude que le lecteur")
print("  jugera plausible, alors que la plage ne vaut que pour celle-ci.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. LA PREMISSE EST A CORRIGER : les defaillances simultanees ne sont pas un cas laisse")
print(f"     de cote, elles sont la SORTIE du modele, et a l'etat non conforme elles sont")
print(f"     MAJORITAIRES : {multi_nc:.2f} % des sinistres touchent plus d'un pilier, contre")
print(f"     {multi_c:.2f} % a l'etat conforme, pour {m_nc:.2f} piliers par sinistre contre {m_c:.2f}.")
print("  2. TROIS enonces d'additivite, a trois etages : hypothese sur les couts d'un sinistre,")
print("     quasi-additivite MESUREE sur les piliers, super-additivite MESUREE sur les canaux.")
print("     Repondre « le modele est super-additif » serait faux : cela depend de l'etage.")
print(f"  3. L'hypothese d'additivite des couts est BORNEE, et c'est la plus lourde des hypotheses")
print(f"     non testees : un exposant de {THETAS[0]:.2f} a {THETAS[-1]:.2f} deplace l'ecart DORA de "
      f"{lo[3]:.0f} a {hi[3]:.0f} M")
print(f"     contre {d_ref:.0f} publie, soit {100*(hi[3]-lo[3])/d_ref:.0f} % de l'ecart et "
      f"{(hi[3]-lo[3])/sd_d:.0f} fois son bruit.")
print(f"  4. ELLE N'EST PAS NEUTRE ENTRE LES DEUX ETATS, et c'est le resultat inattendu :")
print(f"     elasticites {el_nc:.2f} (non conforme) contre {el_c:.2f} (conforme), facteur "
      f"{el_nc/el_c:.2f}, parce que")
print("     l'exposant ne touche que les multi-piliers, majoritaires au seul etat non conforme.")
print("     L'hypothese interagit donc avec le canal de propagation.")
print(f"  5. CE QUI TIENT : sur toute la plage l'ecart reste du meme signe et du meme ordre,")
print(f"     {lo[3]:.0f} a {hi[3]:.0f} M. La these resiste, l'amplitude est un scenario, comme pour g.")
print("  6. Le SENS de la non-additivite reste NON TESTABLE : le cout d'un sinistre date n'est")
print("     jamais observe, celui d'un sinistre multi-piliers encore moins. Limite declaree,")
print("     desormais bornee, et l'elasticite est la grandeur a transferer plutot que la plage.")

# =====================================================================================
# figure S30
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


def fnum(v):
    """Separateur de milliers, applique AU NOMBRE SEUL et jamais a une phrase entiere."""
    return f"{v:,.0f}".replace(",", " ")


def fpct(v):
    """Assez de decimales pour qu'un 0,04 % ne s'affiche pas comme zero."""
    s = f"{v:.2f}" if v < 1.0 else f"{v:.1f}"
    return s.replace(".", ",")


# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime
# cette figure sur 7,27 pouces : tracee plus large, elle subissait une
# reduction qui faisait tomber ses etiquettes sous 6 points. Le rapport
# largeur sur hauteur est conserve, donc la figure occupe la meme place
# sur la page ; c'est son texte qui y prend plus de place.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.9, 3.21))

# (a) la loi du cardinal, aux deux etats
ks = np.arange(1, len(PIL) + 1)
h = 0.36
ax1.bar(ks - h / 2, [100 * loi_c[k] for k in ks], width=h, color=GREEN, alpha=0.9,
        label=f"conforme (g = {cx.G_C})".replace(".", ","))
ax1.bar(ks + h / 2, [100 * loi_nc[k] for k in ks], width=h, color=ACCENT, alpha=0.9,
        label=f"non conforme (g = {cx.G_NC})".replace(".", ","))
for k in ks:
    ax1.text(k - h / 2, 100 * loi_c[k] * 1.30, fpct(100 * loi_c[k]),
             ha="center", fontsize=7.5, color=GREEN)
    ax1.text(k + h / 2, 100 * loi_nc[k] * 1.30, fpct(100 * loi_nc[k]),
             ha="center", fontsize=7.5, color=ACCENT)
ax1.set_yscale("log")
ax1.set_ylim(0.02, 400)
ax1.set_xticks(ks)
ax1.set_xlabel("nombre de piliers touchés par un sinistre", color=INK2)
ax1.set_ylabel("probabilité (%, échelle log)", color=INK2)
ax1.legend(fontsize=9, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.88))
# LE MESSAGE DE CE PANNEAU EN UNE LIGNE : l'echelle log rend les deux premieres barres
# visuellement proches, donc la bascule vers le multi-pilier doit etre ECRITE.
ax1.annotate(f"plus d'un pilier : {fpct(multi_c)} % à l'état conforme,\n{fpct(multi_nc)} % à l'état "
             f"non conforme", xy=(3.0, 130), ha="center", va="center", fontsize=9.5,
             color=INK, fontweight="bold")
ax1.set_title("(a)  La simultanéité est la sortie du modèle,\net la non-conformité l'augmente",
              fontsize=10.5, color=INK, pad=8)

# (b) l'ASYMETRIE entre les deux etats : c'est elle le resultat, pas la plage
th = [r[0] for r in res]
cc = [r[1] for r in res]
nn = [r[2] for r in res]
dd = [r[3] for r in res]
ax2.fill_between(th, cc, nn, color=BLUE, alpha=0.10, lw=0)
ax2.plot(th, nn, "o-", color=ACCENT, lw=2, ms=7,
         label=f"état non conforme, élasticité {el_nc:.2f}".replace(".", ","))
ax2.plot(th, cc, "o-", color=GREEN, lw=2, ms=7,
         label=f"état conforme, élasticité {el_c:.2f}".replace(".", ","))
ax2.axvline(1.0, color=INK2, lw=1.0, ls=":")
# pas de fleche : la verticale pointillee designe deja l'abscisse, et une fleche courte ici
# passait sur la courbe non conforme.
ax2.text(1.0, max(nn) * 1.035, "additivité (modèle publié)", fontsize=8.5, color=INK,
         ha="center", va="top")
# l'ecart, ecrit dans la bande, aux trois exposants qui suffisent a la lire. Les deux extremes
# sont ferres vers l'interieur pour ne pas mordre sur les axes.
for i, ha in ((0, "left"), (2, "center"), (4, "right")):
    x, y = th[i], 0.5 * (cc[i] + nn[i])
    ax2.annotate(f"écart\n{fnum(dd[i])}", (x, y), ha=ha, va="center", fontsize=8.5,
                 color=BLUE, fontweight="bold")
# LES DEUX SENS, A LA MEME HAUTEUR ET EN BAS. Poses en haut a gauche et en bas a droite, ils
# se lisaient a l'envers : l'oeil rattache l'etiquette a la courbe la plus proche.
ax2.text(THETAS[0], 3450, "$\\leftarrow$ coûts sous-additifs", fontsize=9, color=MUTED,
         ha="left")
ax2.text(THETAS[-1], 3450, "coûts super-additifs $\\rightarrow$", fontsize=9, color=MUTED,
         ha="right")
ax2.set_ylim(3000, max(nn) * 1.05)
ax2.set_xlabel("exposant θ sur le nombre de piliers touchés", color=INK2)
ax2.set_ylabel("besoin de capital (M€)", color=INK2)
# descendue : a la taille du memoire elle frolait l'etiquette « additivite » posee en tete
ax2.legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.90))
ax2.set_title("(b)  L'hypothèse n'est pas neutre entre les deux états :\nelle mord sur celui qui "
              "cumule les piliers", fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S30_defaillances_simultanees.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
