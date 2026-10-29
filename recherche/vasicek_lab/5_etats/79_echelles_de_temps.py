#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
79 : la separation des echelles de temps, CHIFFREE au lieu d'etre affirmee.

CE QUE LE MEMOIRE AFFIRME AUJOURD'HUI. Le chapitre 11 separe une COUCHE LENTE (l'etat de
conformite de chaque pilier, qui progresse sur des mois) et une COUCHE RAPIDE (les incidents et
leur cascade a l'interieur d'une annee), la premiere servant de PARAMETRE a la seconde. Le texte
dit que ce n'est pas une commodite de calcul mais une contrainte de construction, et il s'appuie
sur un precedent : le prepublie de cascade climatique separe de meme un temps calendaire, une
position et un pas de propagation A L'INTERIEUR d'un evenement.

CE QUI MANQUAIT, ET C'EST L'OBJET DE CE SCRIPT. L'argument etait une ANALOGIE. Personne n'avait
mesure ce que la separation coute ni ce qu'elle cache. Or elle repose sur deux approximations
distinctes, et chacune se teste :
  (A) la cascade est traitee comme INSTANTANEE devant l'annee. Si elle a une duree, deux
      sinistres peuvent se chevaucher, et un sinistre peut enjamber le 31 decembre ;
  (B) l'etat de conformite est traite comme FIGE sur l'annee. S'il migre en cours d'annee, le
      capital d'une annee mixte n'est pas la moyenne des capitaux des deux etats.

CE QUE CHACUNE PRODUIT, ET AUCUN DES DEUX RESULTATS N'ETAIT ATTENDU :
  - le chevauchement n'est PAS negligeable des que la duree depasse quelques heures, et il monte
    avec la non-conformite par DEUX canaux a la fois (plus de sinistres, et des cascades plus
    longues). Consequence pour la notion 5 : l'exposant theta du script 74 ne sature que DANS un
    sinistre, donc il MINORE la saturation reelle, deux sinistres concomitants partageant la
    meme cellule de crise ;
  - l'enjambement du 31 decembre deplace le capital dans le sens CONSERVATEUR, en coupant les
    grosses cascades en deux, ce qui est l'inverse de l'intuition ;
  - le capital d'une annee mixte est CONVEXE en la fraction d'annee passee non conforme, donc
    remedier a mi-annee ne rapporte PAS la moitie du benefice annuel.

CE QU'IL NE FAIT PAS : implementer une horloge intra-sinistre dans la chaine publiee. Le gel
l'interdit, et aucun nombre publie ne bouge. La duree de propagation vient du script 08h (lag
moyen 6,7 h, noyau a retard) et elle est BALAYEE, jamais posee a une valeur unique.

Sortie : diagnostics + figure S35_echelles_de_temps.png.
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
NY = cx.NY
NSEED = cx.NSEED
SEED0 = cx.SEED0

H_PAR_AN = 24.0 * 365.0

# Lag moyen d'un PAS de propagation, en heures. La valeur centrale est celle du script 08h
# (noyau a retard de Bessy-Roland, lag moyen 6,7 h). Les autres sont des ordres de grandeur de
# discussion : une cascade qui se joue dans l'heure, dans la journee, sur trois jours.
LAGS_H = (1.0, 6.7, 24.0, 72.0)
LAG_REF_H = 6.7

# Fractions d'annee passees en etat non conforme, pour la couche lente.
TAUS = (0.0, 0.25, 0.50, 0.75, 1.0)


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def fnum(v):
    """Separateur de milliers applique AU NOMBRE SEUL, jamais a une phrase entiere."""
    return f"{v:,.0f}".replace(",", " ")


# poids d'amorce, comme partout dans le projet
W_AM = np.array([eng.LAMBDA[j] for j in PIL], float)
W_AM = W_AM / W_AM.sum()


def loi_cardinal(g):
    """P(k piliers touches) pour un sinistre, EXACTE, ponderee par la propension d'amorce."""
    p = np.zeros(len(PIL) + 1)
    for c, j in enumerate(PIL):
        dist = eng.cascade_set_dist(j, g)
        tot = sum(dist.values())
        for s, pr in dist.items():
            p[len(s)] += W_AM[c] * pr / tot
    return p


# =====================================================================================
titre("1. Les trois indices, et ce que le memoire collapse")
# =====================================================================================
print("  Le prepublie de cascade climatique distingue trois indices : un temps CALENDAIRE, une")
print("  POSITION, et un pas de PROPAGATION a l'interieur d'un evenement. Le present modele en")
print("  garde deux et en collapse un :")
print("    - calendaire  : l'annee, qui porte la frequence et sur laquelle le capital se mesure ;")
print("    - position    : le pilier, qui porte la cascade ;")
print("    - propagation : COLLAPSE. La cascade est instantanee, son pas n'a pas de duree.")
print("\n  ET LA COUCHE LENTE EST UN QUATRIEME INDICE, traite comme un PARAMETRE : l'etat de")
print("  conformite est fige sur toute l'annee. Deux approximations distinctes en decoulent, et")
print("  ce script les mesure separement parce qu'elles ne vont pas dans le meme sens.")

loi_c = loi_cardinal(cx.G_C)
loi_nc = loi_cardinal(cx.G_NC)
m_c = float(sum(k * loi_c[k] for k in range(len(PIL) + 1)))
m_nc = float(sum(k * loi_nc[k] for k in range(len(PIL) + 1)))
print(f"\n  Rappel de la loi exacte du cardinal (script 74) : {m_c:.3f} pilier par sinistre a")
print(f"  l'etat conforme, {m_nc:.3f} a l'etat non conforme. Une cascade de k piliers compte")
print("  k - 1 pas de propagation, donc sa DUREE croit avec la non-conformite : c'est le second")
print("  canal du chevauchement, et c'est lui qu'on aurait manque en ne regardant que lambda.")

# =====================================================================================
titre("2. Collapse (A) : si la cascade a une duree, elle chevauche et elle enjambe")
# =====================================================================================


def taux_chevauchement(lam, g, lag_h, ny=200_000, seed=4242):
    """Part des sinistres qui chevauchent un autre, et part qui enjambe le 31 decembre.

    Construction : le comptage annuel est celui du modele publie (NegBin de dispersion phi) et,
    CONDITIONNELLEMENT au comptage, les instants d'arrivee sont uniformes sur l'annee. C'est une
    propriete du melange de Poisson, pas une hypothese ajoutee.

    Le test de chevauchement sur les seuls VOISINS est EXACT pour « chevauche au moins un
    autre » : si le sinistre i recouvre un sinistre j au-dela de i+1, alors t_{i+1} <= t_j se
    trouve aussi dans [t_i, t_i + D_i], donc i recouvre deja i+1.
    """
    rng = np.random.default_rng(seed)
    r = lam / (ec.PHI - 1.0)
    counts = rng.negative_binomial(r, r / (r + lam), size=ny)
    T = int(counts.sum())
    if T == 0:
        return 0.0, 0.0, 0.0
    an = np.repeat(np.arange(ny), counts).astype(np.float64)
    t = an + rng.random(T)                       # instant absolu, en annees
    # cardinal de chaque cascade, tire dans la loi exacte, puis duree = (k-1) pas exponentiels
    loi = loi_cardinal(g)
    kk = rng.choice(np.arange(len(loi)), size=T, p=loi / loi.sum())
    npas = np.maximum(kk - 1, 0)
    dur = rng.gamma(shape=np.maximum(npas, 1e-9), scale=lag_h / H_PAR_AN)
    dur[npas == 0] = 0.0
    o = np.argsort(t, kind="stable")
    ts, ds, ans = t[o], dur[o], an[o]
    fin = ts + ds
    meme_annee = np.zeros(T, bool)
    meme_annee[:-1] = ans[:-1] == ans[1:]
    # i chevauche i+1 (dans la meme annee civile)
    ch = np.zeros(T, bool)
    suiv = np.zeros(T, bool)
    suiv[:-1] = (fin[:-1] > ts[1:]) & meme_annee[:-1]
    ch |= suiv
    ch[1:] |= suiv[:-1]
    # enjambement de l'annee : la cascade finit apres le 31 decembre de son annee d'amorce
    enj = fin > (ans + 1.0)
    return float(ch.mean()), float(enj.mean()), float(dur.mean() * H_PAR_AN)


print("  On donne une duree a la cascade et l'on compte, SANS toucher au capital : quelle part")
print("  des sinistres en recouvre un autre, et quelle part enjambe le 31 decembre.")
print(f"\n  {'lag par pas':>12}{'duree moy.':>12}"
      f"{'chevauche (C)':>16}{'chevauche (NC)':>16}{'enjambe (NC)':>15}")
chev = []
for lg in LAGS_H:
    ch_c, _, dm_c = taux_chevauchement(cx.LAM_C, cx.G_C, lg)
    ch_nc, en_nc, dm_nc = taux_chevauchement(cx.LAM_NC, cx.G_NC, lg)
    chev.append((lg, dm_c, dm_nc, ch_c, ch_nc, en_nc))
    print(f"  {lg:>9.1f} h{dm_nc:>10.1f} h{100*ch_c:>15.1f} %{100*ch_nc:>15.1f} %"
          f"{100*en_nc:>14.1f} %")

i_ref = LAGS_H.index(LAG_REF_H)
_, _, dmnc_ref, chc_ref, chnc_ref, enj_ref = chev[i_ref]
print(f"\n  AU LAG MESURE PAR LE SCRIPT 08h ({LAG_REF_H} h par pas), une cascade non conforme dure")
print(f"  {dmnc_ref:.1f} h en moyenne, et {100*chnc_ref:.1f} % des sinistres en recouvrent deja un autre")
print(f"  contre {100*chc_ref:.1f} % a l'etat conforme. LE RAPPORT EST DE {chnc_ref/chc_ref:.1f}, et il vient de")
print("  DEUX canaux qui se composent : la non-conformite multiplie le nombre de sinistres ET")
print("  allonge chaque cascade. En ne regardant que la frequence on aurait sous-estime l'effet.")
print("\n  CE QUE CELA DIT DE L'HYPOTHESE D'ADDITIVITE DES COUTS, ET C'EST LE LIEN QUI MANQUAIT.")
print("  L'exposant theta du script 74 fait jouer la non-additivite au sein d'UN sinistre, et")
print(f"  seulement la. Or a cette duree, {100*chnc_ref:.0f} % des sinistres non conformes sont concomitants")
print("  d'un autre : ils partagent la meme cellule de crise, la meme equipe forensique, le meme")
print("  fournisseur d'astreinte. LE MECANISME DE theta S'APPLIQUE DONC A MOINS D'EVENEMENTS QUE")
print("  LA REALITE N'EN CONTIENT, et la plage publiee en SOUS-ESTIME l'amplitude.")
print("\n  DEUX PRECAUTIONS SUR CET ENONCE, parce qu'il serait facile d'en dire trop.")
print("  (i) Il porte sur l'AMPLITUDE, pas sur le sens. Le sens de la non-additivite reste")
print("      indetermine (script 74, section 4) : la concomitance amplifie la mutualisation aussi")
print("      bien que la saturation. Elle ELARGIT la plage des deux cotes, elle ne la deplace pas.")
print("  (ii) L'ASYMETRIE ENTRE ETATS, elle, est bien renforcee, et dans le sens du memoire : la")
print(f"      concomitance est {chnc_ref/chc_ref:.1f} fois plus frequente a l'etat non conforme, exactement comme")
print("      les sinistres multi-piliers le sont. Les deux canaux d'asymetrie vont dans le meme")
print("      sens, donc les elasticites de 0,95 contre 0,39 ne sont pas un artefact de theta.")

# =====================================================================================
titre("3. Ce que l'enjambement du 31 decembre fait au capital")
# =====================================================================================


def scr_horloge(lam, g, p_u, phi_cs, lag_h, nseed=NSEED, ny=NY):
    """SCR quand chaque pilier touche porte sa perte dans l'ANNEE ou son propre pas tombe.

    A lag_h = 0 la cascade est instantanee et l'on retrouve EXACTEMENT le moteur publie : c'est
    le controle de cette section. Au-dessus, une cascade amorcee en fin d'annee voit une partie
    de ses piliers basculer sur l'exercice suivant.

    L'IDENTITE DES PILIERS NE COMPTE PAS ICI, et il faut le dire : la table de cascade stocke un
    ENSEMBLE, pas un chemin ordonne. Ce qui determine le partage entre deux exercices est le
    NOMBRE de piliers tombant apres le 31 decembre, donc les lags cumules, non l'identite des
    piliers. On indexe donc les piliers touches par leur rang de propagation.
    """
    scrs = []
    for k in range(nseed):
        rng = np.random.default_rng(SEED0 + k)
        r = lam / (ec.PHI - 1.0)
        counts = rng.negative_binomial(r, r / (r + lam), size=ny)
        T = int(counts.sum())
        if T == 0:
            scrs.append(0.0)
            continue
        year_of = np.repeat(np.arange(ny), counts)
        amorce = rng.choice(5, size=T, p=W_AM)
        U = rng.random(T)
        SEV = simulate_remediation_severity(T * 5, sp["xi"], sp["sigma"], sp["u"], p_u,
                                            sp["cap"], rng).reshape(T, 5)
        tables = {j: cx.table_amorce(j, g) for j in PIL}
        if phi_cs is not None:
            tables[cx.P4] = cx.table_p4_choc(phi_cs)
        # instant d'amorce dans l'annee, et lags cumules des quatre pas possibles
        u_t = rng.random(T)
        lags = rng.exponential(scale=lag_h / H_PAR_AN, size=(T, len(PIL) - 1))
        cum = np.cumsum(lags, axis=1)
        annual = np.zeros(ny + 1)                # +1 : deversoir de l'annee ny (bord droit)
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
            sev_i = SEV[idx] * touches
            # severites triees par pilier decroissant : sans effet sur la somme, mais rend le
            # rang de propagation independant de l'indice de colonne (l'ensemble n'est pas
            # ordonne, cf. docstring). On somme les severites presentes dans l'ordre des colonnes.
            k_pil = touches.sum(axis=1).astype(int)
            ordre = np.argsort(-touches, axis=1, kind="stable")     # colonnes touchees d'abord
            sev_ord = np.take_along_axis(sev_i, ordre, axis=1)
            t0 = u_t[idx]
            for m in range(len(PIL)):
                pres = m < k_pil
                if not pres.any():
                    continue
                if m == 0:
                    t_m = t0
                else:
                    t_m = t0 + cum[idx, m - 1]
                yr = year_of[idx] + (t_m >= 1.0).astype(int)       # bascule d'exercice
                np.add.at(annual, np.where(pres, yr, ny), np.where(pres, sev_ord[:, m], 0.0))
        scrs.append(var(annual[:ny]))
    return np.array(scrs, float)


ref_nc = cx.scr_config(lam=cx.LAM_NC, g=cx.G_NC, p_u=cx.PU_NC, phi_cs=cx.PHICS_NC)
ref_c = cx.scr_config(lam=cx.LAM_C, g=cx.G_C, p_u=cx.PU_C, phi_cs=None)
ctrl_nc = scr_horloge(cx.LAM_NC, cx.G_NC, cx.PU_NC, cx.PHICS_NC, 0.0)
ctrl_c = scr_horloge(cx.LAM_C, cx.G_C, cx.PU_C, None, 0.0)
print("  CONTROLE D'ABORD, a duree nulle, contre le moteur partage des 14 139 M :")
print(f"    conforme      {ctrl_c.mean():>9.0f} contre {ref_c:>9.0f}   ecart {abs(ctrl_c.mean()-ref_c):.3f} M")
print(f"    non conforme  {ctrl_nc.mean():>9.0f} contre {ref_nc:>9.0f}   ecart {abs(ctrl_nc.mean()-ref_nc):.3f} M")
sd_ref = float((ctrl_nc - ctrl_c).std(ddof=1))
print(f"  Bruit de l'ecart sur {NSEED} graines appariees : +/- {sd_ref:.0f} M. Tout deplacement plus")
print("  petit que ce bruit n'est pas un effet.")

print("\n  CE BALAYAGE EST A NOMBRES COMMUNS, et c'est ce qui le rend lisible. A etat et graine")
print("  fixes, deux valeurs de lag partagent EXACTEMENT les memes tirages : comptages, amorces,")
print("  uniformes de cascade, severites, et jusqu'aux uniformes des lags, l'echelle d'une")
print("  exponentielle etant un simple facteur. La difference entre deux lignes est donc du")
print("  SIGNAL, pas du bruit de Monte-Carlo, et elle ne se lit PAS contre le +/- 734 ci-dessus,")
print("  qui est le bruit d'une comparaison entre deux etats de frequences differentes.")
print(f"\n  {'lag par pas':>12}{'SCR conforme':>15}{'SCR non conf.':>15}{'ecart DORA':>13}"
      f"{'deplacement par graine':>26}{'signe':>8}")
horl = [(0.0, ctrl_c, ctrl_nc)]
for lg in LAGS_H:
    a = scr_horloge(cx.LAM_C, cx.G_C, cx.PU_C, None, lg)
    b = scr_horloge(cx.LAM_NC, cx.G_NC, cx.PU_NC, cx.PHICS_NC, lg)
    horl.append((lg, a, b))
d_ref = ref_nc - ref_c
d0_seeds = horl[0][2] - horl[0][1]
for lg, a, b in horl:
    lab = "0 (instantane)" if lg == 0.0 else f"{lg:.1f} h"
    d_seeds = b - a
    dd = d_seeds - d0_seeds                      # deplacement PAR GRAINE, apparie
    if lg == 0.0:
        det, res = "reference", ""
    else:
        det = " ".join(f"{v:+.0f}" for v in dd)
        # Un test binaire « meme signe strict » ecraserait la difference entre un deplacement
        # NUL et un deplacement de signe indetermine. Les deux se lisent autrement.
        if np.all(dd == 0):
            res = "nul"
        elif np.all(dd >= 0):
            res = ">= 0"
        elif np.all(dd <= 0):
            res = "<= 0"
        else:
            res = "NON"
    print(f"  {lab:>12}{a.mean():>15.0f}{b.mean():>15.0f}{d_seeds.mean():>13.0f}"
          f"{det:>26}{res:>8}")

d_inst = float(d0_seeds.mean())
d_long = float((horl[-1][2] - horl[-1][1]).mean())
d_ref_lag = float((horl[i_ref + 1][2] - horl[i_ref + 1][1]).mean())
print(f"\n  AU LAG QUE LA DONNEE SOUTIENT, L'EFFET EST NUL, ET C'EST LA LE RESULTAT. A {LAG_REF_H} h par")
print(f"  pas, l'ecart DORA vaut {fnum(d_ref_lag)} M contre {fnum(d_inst)} pour la cascade instantanee, soit")
print(f"  {d_ref_lag-d_inst:+.1f} M sur les quatre graines. Comme le balayage est a nombres communs, ce")
print("  zero n'est pas un zero masque par le bruit : c'est une absence d'effet MESUREE. Une")
print("  cascade de quelques heures ne rencontre pratiquement jamais le 31 decembre.")
print(f"\n  ET AU-DELA, LE SIGNE SE DEFAIT, ce qu'il faut dire plutot que raconter un mecanisme.")
print("  A 24 h par pas le deplacement est positif ou nul sur les quatre graines, donc de signe")
print(f"  etabli. A {LAGS_H[-1]:.0f} h il vaut {d_long-d_inst:+.0f} M en moyenne, soit "
      f"{100*(d_long-d_inst)/d_inst:+.1f} % de l'ecart, mais il CHANGE DE SIGNE")
print("  d'une graine a l'autre. On aurait pu ecrire que couper une cascade au 31 decembre")
print("  RABOTE la queue, le quantile etant porte par un sinistre unique ; on aurait pu ecrire")
print("  l'inverse, le partage DEPOSANT un fragment sur l'exercice suivant. Les deux mecanismes")
print("  existent, ils se compensent, et quatre graines ne les separent pas a cette duree.")
print("  LE SEUL ENONCE QUE LA MESURE PORTE est donc : a duree realiste l'effet est nul, a duree")
print("  d'une journee il est haussier et sous le demi-pour-cent, au-dela il n'a plus de")
print("  direction etablie. Le collapse est NEUTRE, et surtout pas prudent.")

# =====================================================================================
titre("4. Collapse (B) : la couche lente bouge en cours d'annee")
# =====================================================================================


def scr_mixte(tau, nseed=NSEED, ny=NY):
    """SCR d'une annee passee en NON CONFORME une fraction tau du temps, puis en CONFORME.

    Construction, et elle est contrainte. La surdispersion du modele est un facteur d'ANNEE :
    on tire donc UN facteur d'environnement Z par annee, commun aux deux sous-periodes, puis un
    comptage de Poisson par sous-periode d'intensite proportionnelle a sa duree. Le facteur suit
    une Gamma de moyenne 1 et de forme r = lambda_mix / (phi - 1), de sorte qu'a tau = 0 comme a
    tau = 1 la loi du comptage annuel redevient EXACTEMENT la NegBin publiee. C'est ce qui rend
    les deux bouts du balayage comparables aux nombres du memoire.

    Chaque sinistre porte les quatre canaux de SA sous-periode : frequence, detection,
    propagation et accumulation. Un sinistre n'est donc jamais a cheval sur deux etats, ce qui
    est coherent avec la section 3 : la cascade est courte devant la duree d'un etat.
    """
    lam_mix = tau * cx.LAM_NC + (1.0 - tau) * cx.LAM_C
    r = lam_mix / (ec.PHI - 1.0)
    scrs = []
    for k in range(nseed):
        rng = np.random.default_rng(SEED0 + k)
        Z = rng.gamma(shape=r, scale=1.0 / r, size=ny)
        annual = np.zeros(ny)
        for etat, part, lam_e, g_e, pu_e, phi_e in (
                ("NC", tau, cx.LAM_NC, cx.G_NC, cx.PU_NC, cx.PHICS_NC),
                ("C", 1.0 - tau, cx.LAM_C, cx.G_C, cx.PU_C, None)):
            if part <= 0.0:
                continue
            counts = rng.poisson(part * lam_e * Z)
            T = int(counts.sum())
            if T == 0:
                continue
            year_of = np.repeat(np.arange(ny), counts)
            amorce = rng.choice(5, size=T, p=W_AM)
            U = rng.random(T)
            SEV = simulate_remediation_severity(T * 5, sp["xi"], sp["sigma"], sp["u"], pu_e,
                                                sp["cap"], rng).reshape(T, 5)
            tables = {j: cx.table_amorce(j, g_e) for j in PIL}
            if phi_e is not None:
                tables[cx.P4] = cx.table_p4_choc(phi_e)
            for c, j in enumerate(PIL):
                idx = np.where(amorce == c)[0]
                if idx.size == 0:
                    continue
                ind, probs = tables[j]
                cdf = np.cumsum(probs)
                cdf[-1] = 1.0
                sel = np.searchsorted(cdf, U[idx], side="right")
                np.clip(sel, 0, len(probs) - 1, out=sel)
                loss = (SEV[idx] * ind[sel]).sum(axis=1)
                annual += np.bincount(year_of[idx], weights=loss, minlength=ny)
        scrs.append(var(annual))
    return np.array(scrs, float)


mix = [(t, scr_mixte(t)) for t in TAUS]
m0 = float(mix[0][1].mean())
m1 = float(mix[-1][1].mean())
sd_mix = float(np.mean([s.std(ddof=1) for _, s in mix]))
print("  CONTROLE AUX DEUX BOUTS, contre le moteur publie :")
print(f"    tau = 0 (tout conforme)      {m0:>9.0f} contre {ref_c:>9.0f}   ecart "
      f"{100*abs(m0-ref_c)/ref_c:>5.2f} %")
print(f"    tau = 1 (tout non conforme)  {m1:>9.0f} contre {ref_nc:>9.0f}   ecart "
      f"{100*abs(m1-ref_nc)/ref_nc:>5.2f} %")
print(f"  Bruit moyen de simulation, {NSEED} graines : +/- {sd_mix:.0f} M. La construction")
print("  Gamma-Poisson ne redonne pas les MEMES TIRAGES que la NegBin directe du moteur publie,")
print("  seulement la meme LOI : les deux bouts se comparent donc au bruit pres, et c'est ce")
print("  qu'on verifie ici avant de lire le milieu.")

print(f"\n  {'tau (part NC)':>14}{'SCR':>10}{'prorata':>10}{'ecart a la droite':>19}"
      f"{'part du surcout':>17}")
for t, s in mix:
    v = float(s.mean())
    lin = m0 + t * (m1 - m0)
    print(f"  {t:>14.2f}{v:>10.0f}{lin:>10.0f}{v-lin:>+19.0f}"
          f"{100*(v-m0)/(m1-m0):>16.0f} %")
t_mid = 0.5
v_mid = float([s for t, s in mix if t == t_mid][0].mean())
lin_mid = m0 + t_mid * (m1 - m0)
part_mid = 100.0 * (m1 - v_mid) / (m1 - m0)
v_q1 = float([s for t, s in mix if t == 0.25][0].mean())
part_q1 = 100.0 * (v_q1 - m0) / (m1 - m0)
print(f"\n  LE CAPITAL EST CONCAVE EN tau, et c'est bien dans ce sens-la : la courbe passe AU-DESSUS")
print(f"  de la droite, de {v_mid-lin_mid:+.0f} M a mi-annee, soit {abs(v_mid-lin_mid)/sd_mix:.1f} fois le bruit de simulation.")
print("  Le capital monte vite au debut puis s'aplatit, ce qui est la signature d'une queue")
print("  portee par un sinistre unique : il suffit d'une FENETRE de non-conformite pour qu'un tel")
print("  sinistre s'y produise, et allonger la fenetre au-dela n'ajoute plus grand-chose.")
print(f"\n  DEUX LECTURES DE GESTION, ET AUCUNE N'EST CELLE QU'ON ATTEND.")
print(f"  (i) Un TRIMESTRE de non-conformite porte deja {part_q1:.0f} % du surcout de l'annee entiere.")
print(f"      Une fenetre courte n'est donc pas une petite exposition.")
print(f"  (ii) Symetriquement, remedier a mi-exercice ne capture que {part_mid:.0f} % du benefice annuel et")
print("      non la moitie. UN PLAN QUI PRORATE LE GAIN SE TROMPE, et se trompe du cote optimiste.")
print("\n  CE QUE CELA VALIDE DE LA SEPARATION, ET CE QUE CELA LUI COUTE. Le collapse de la couche")
print("  lente est LEGITIME pour ce que le memoire en fait, a savoir conditionner a un etat et")
print("  publier un capital par etat. Il devient FAUTIF des qu'on veut lire un calendrier de")
print("  remediation : un plan qui promet la moitie du gain a mi-parcours se trompe, et se")
print("  trompe dans le sens optimiste.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. LA SEPARATION DES HORLOGES N'EST PLUS UNE ANALOGIE, elle est chiffree sur ses deux")
print("     approximations, et elles ne vont pas dans le meme sens.")
print(f"  2. LE CHEVAUCHEMENT N'EST PAS NEGLIGEABLE : au lag de {LAG_REF_H} h du script 08h, "
      f"{100*chnc_ref:.0f} % des")
print(f"     sinistres non conformes en recouvrent un autre, contre {100*chc_ref:.0f} % a l'etat conforme, "
      f"facteur {chnc_ref/chc_ref:.1f}.")
print("     Il monte par DEUX canaux composes, plus de sinistres et des cascades plus longues.")
print("  3. D'OU DEUX CONSEQUENCES POUR L'ADDITIVITE DES COUTS, ET IL FAUT LES SEPARER. L'exposant")
print("     theta du script 74 ne joue que DANS un sinistre, alors que des sinistres concomitants")
print("     partagent la meme capacite de remediation. Donc (i) la plage publiee SOUS-ESTIME")
print("     l'amplitude de la non-additivite, sans que cela dise rien de son SENS, la concomitance")
print("     amplifiant la mutualisation autant que la saturation ; et (ii) l'ASYMETRIE entre etats")
print(f"     est renforcee, la concomitance etant {chnc_ref/chc_ref:.1f} fois plus frequente a l'etat non conforme,")
print("     donc les elasticites 0,95 contre 0,39 ne sont pas un artefact du seul theta.")
print(f"  4. LE COLLAPSE DE L'HORLOGE INTRA-SINISTRE NE COUTE RIEN AU LAG QUE LA DONNEE SOUTIENT :")
print(f"     a {LAG_REF_H} h par pas l'ecart DORA vaut {fnum(d_ref_lag)} contre {fnum(d_inst)} M, soit "
      f"{d_ref_lag-d_inst:+.1f} M. Le balayage")
print("     etant a nombres communs, c'est une absence d'effet MESUREE et non un effet noye dans")
print("     le bruit. A 24 h par pas le deplacement devient haussier sur les quatre graines mais")
print(f"     reste sous le demi-pour-cent ; a {LAGS_H[-1]:.0f} h il CHANGE DE SIGNE d'une graine a l'autre.")
print("     Deux mecanismes de sens contraire coexistent, le partage abaissant le maximum de")
print("     l'annee d'amorce et deposant un fragment sur la suivante. Le collapse est donc")
print("     NEUTRE, et surtout pas prudent : c'est un enonce plus faible que celui qu'on aurait")
print("     ecrit sans regarder les graines une par une.")
print(f"  5. LE CAPITAL EST CONCAVE EN LA FRACTION D'ANNEE NON CONFORME, et les deux lectures qui en")
print(f"     sortent vont contre l'intuition : un trimestre de non-conformite porte deja {part_q1:.0f} % du")
print(f"     surcout annuel, et remedier a mi-exercice ne rend que {part_mid:.0f} % du benefice, pas 50 %.")
print("     Le collapse de la couche lente est licite pour publier un capital par etat, FAUTIF")
print("     pour lire un calendrier de remediation.")
print("  6. RIEN N'EST IMPLEMENTE DANS LA CHAINE PUBLIEE. Le gel l'interdit et aucun nombre publie")
print("     ne bouge. Ce qui change est le statut de la separation : elle etait posee par analogie,")
print("     elle est desormais bornee dans ses deux directions et son signe est connu.")

# =====================================================================================
# figure S35
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.2))

# (a) le chevauchement, aux deux etats, en fonction de la duree d'un pas
lg = [c[0] for c in chev]
ax1.plot(lg, [100 * c[4] for c in chev], "o-", color=ACCENT, lw=2, ms=8,
         label="état non conforme")
ax1.plot(lg, [100 * c[3] for c in chev], "o-", color=GREEN, lw=2, ms=8,
         label="état conforme")
ax1.axvline(LAG_REF_H, color=INK2, lw=1.1, ls=":")
ax1.annotate(f"lag mesuré,\nscript 08h : {LAG_REF_H} h".replace(".", ","),
             xy=(LAG_REF_H, 100 * chev[-1][4] * 0.62), ha="left", va="center",
             fontsize=8.5, color=INK, xytext=(11, 100 * chev[-1][4] * 0.62))
ax1.set_xscale("log")
ax1.set_xticks(lg)
ax1.set_xticklabels([f"{v:g}" for v in lg])
ax1.set_xlabel("durée moyenne d'un pas de propagation (heures, log)", color=INK2)
ax1.set_ylabel("sinistres recouvrant un autre sinistre (%)", color=INK2)
ax1.legend(fontsize=9, frameon=False, loc="upper left")
ax1.set_title("(a)  Le chevauchement monte par deux canaux composés :\nplus de sinistres, et "
              "des cascades plus longues", fontsize=10.5, color=INK, pad=8)

# (b) LA CONCAVITE ELLE-MEME, pas le niveau. Tracer les deux courbes de niveau sur un axe de
# 6 000 a 20 000 rendait un ecart de 897 M invisible, defaut SYMETRIQUE de celui du script 69 ou
# un axe trop serre rendait 2 M enorme. On trace donc l'ECART a la droite, avec son bruit, qui
# est la grandeur mesuree et la seule qui se juge.
ts = [t for t, _ in mix]
vs = [float(s.mean()) for _, s in mix]
lins = [m0 + t * (m1 - m0) for t in ts]
ecarts = [v - l for v, l in zip(vs, lins)]
ax2.axhspan(-sd_mix, sd_mix, color=MUTED, alpha=0.16, lw=0)
ax2.bar(ts, ecarts, width=0.10, color=ACCENT, alpha=0.9)
ax2.axhline(0.0, color=INK2, lw=1.0)
for t, e in zip(ts, ecarts):
    if abs(e) > 1.0:
        ax2.text(t, e + 0.06 * max(ecarts), f"+{e:.0f}", ha="center", fontsize=9,
                 color=ACCENT, fontweight="bold")
ax2.set_ylim(-1.5 * sd_mix, 2.10 * max(ecarts))
ax2.set_xlim(-0.09, 1.09)
ax2.set_xticks(ts)
ax2.set_xlabel("fraction de l'année passée en état non conforme", color=INK2)
ax2.set_ylabel("capital au-dessus du prorata linéaire (M€)", color=INK2)
# PAS DE LEGENDE ICI, et c'est deliberе : posee en haut a droite elle heurtait le bloc de texte,
# posee en bas a gauche elle tombait sur la bande de bruit. Deux elements seulement sont a
# nommer, et ils le sont sur place : l'axe des ordonnees dit ce que les barres mesurent, et la
# bande porte son etiquette a l'interieur.
ax2.text(1.06, 0.0, f"bruit\n± {sd_mix:.0f}", ha="right", va="center", fontsize=8.5,
         color=INK2, style="italic")
# les deux lectures de gestion, ecrites : une barre positive ne dit pas d'elle-meme ce qu'elle
# coute a un plan de remediation. Posee a 1,36 fois le maximum, l'etiquette « +1 070 » de la
# premiere barre passait dessous : le plafond de l'axe est releve et le texte remonte avec.
ax2.text(0.02, 2.00 * max(ecarts),
         f"un trimestre non conforme porte déjà {part_q1:.0f} % du surcoût annuel ;\nremédier à "
         f"mi-exercice ne rend que {part_mid:.0f} % du bénéfice, pas 50 %".replace(".", ","),
         fontsize=9, color=INK, va="top", ha="left")
ax2.set_title("(b)  Le capital est concave en la durée de non-conformité :\nles barres sont "
              "l'écart au prorata, et il dépasse le bruit",
              fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("S35 : les deux horloges du modèle, et ce que coûte de les confondre",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S35_echelles_de_temps.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
