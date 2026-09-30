#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""90 - LA DERIVE D'ECHELLE DE SEVERITE EST-ELLE NEUTRE SUR L'ECART ENTRE ETATS ?

CE SCRIPT FERME UNE AFFIRMATION DU MEMOIRE, IL N'EN OUVRE PAS UNE NOUVELLE.
Le script 89 mesure une derive de l'echelle de severite et chiffre ce qu'elle coute sur le
QUANTILE de severite (+24,7 %). Le memoire ecrit ensuite, au chapitre 06 et au chapitre 13, que
cette derive ne touche pas la THESE, parce que celle-ci porte sur un RAPPORT entre etats a
severite de base commune et qu'une derive commune aux deux etats s'y simplifie.

C'ETAIT UN ARGUMENT, PAS UNE MESURE, et il n'est pas gratuit. Les quatre canaux ne transforment
pas la severite de la meme facon selon l'etat : le canal de detection entre dans la
TRANSFORMATION de severite (taux de depassement), et les canaux de propagation et d'accumulation
changent le NOMBRE de piliers touches par sinistre, donc le nombre de severites tirees. Une
severite plus lourde peut donc, a priori, se propager differemment dans les deux etats. C'est
exactement ce que ce script mesure.

CE QU'IL TROUVE, EN UNE LIGNE, POUR QU'UN RELECTEUR NE PARTE PAS SUR L'ANCIENNE FORMULATION.
L'argument est juste mais il porte sur l'ECHELLE et sur elle seule : une variation de la seule
echelle de severite est neutre sur le rapport a 0,16 % pres, alors qu'elle deplace les niveaux de
46 %. La derive mesuree, elle, n'est pas un pur changement d'echelle, puisque la modeliser fait
BAISSER l'indice de queue : le facteur entre etats tombe de 3,344 a 3,124, un deplacement resolu
a 4,4 ecarts-types, entierement porte par la composante de FORME. L'ecart en euros, lui, ne bouge
que de -1,9 %, mais par COMPENSATION de deux mouvements de sens contraires, et non par
insensibilite. Le memoire doit donc citer l'invariance d'echelle, jamais l'invariance a la derive.

CE N'EST PAS UNE RECALIBRATION. Aucun parametre publie n'est modifie : config.py n'est ni lu pour
etre ecrit ni touche, et le module canaux_conformite.py n'est pas modifie. Le script recalcule les
deux etats sous une severite ALTERNATIVE et compare des rapports. Compatible avec le gel du
7 aout.

DEUX CONTROLES, ET SANS EUX LE SCRIPT DEMONTRERAIT UNE PROPRIETE D'UN AUTRE MODELE.
  1. la fonction de perte locale, appelee avec la severite PUBLIEE, doit reproduire
     canaux_conformite.pertes_annuelles TIRAGE POUR TIRAGE. Meme patron que le controle du
     script 81 ;
  2. la branche publiee doit redonner 6 049 et 20 188 M EUR, donc le facteur 3,34 et l'ecart
     14 139.

POURQUOI LA COMPARAISON SE FAIT CONTRE LA BRANCHE STATIONNAIRE ET NON CONTRE LA PUBLIEE. Les
parametres publies (xi 0,5954, sigma 57,97) et l'ajustement stationnaire libre du script 89
(0,5979, 55,75) ne coincident pas exactement. Comparer la branche derivee a la branche publiee
melangerait donc la derive avec ce petit ecart d'ajustement. La derive se lit contre la
stationnaire ; la publiee est la, elle, pour attester que le moteur est bien celui du memoire.
Meme discipline que la section 1ter du script 89.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
if LAB not in sys.path:
    sys.path.insert(0, LAB)

import canaux_conformite as cx                                   # noqa: E402
import derive_severite as ds                                     # noqa: E402
import euro_cascade_model as ec                                   # noqa: E402
from euro_cascade_model import var                                # noqa: E402
from src.aggregation.lda import simulate_remediation_severity     # noqa: E402
import scr_engine as eng                                          # noqa: E402

WID = 88
PIL = cx.PIL
sp = cx.sp
NY, NSEED, SEED0 = cx.NY, cx.NSEED, cx.SEED0

ETAT_C = dict(lam=cx.LAM_C, g=cx.G_C, p_u=cx.PU_C, phi_cs=None)
ETAT_NC = dict(lam=cx.LAM_NC, g=cx.G_NC, p_u=cx.PU_NC, phi_cs=cx.PHICS_NC)


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def fnum(v):
    """Separateur de milliers applique AU NOMBRE SEUL, jamais a une phrase entiere."""
    return f"{v:,.0f}".replace(",", " ")


# ---------------------------------------------------------------------------
# Fonction de perte, identique au module a la severite pres
# ---------------------------------------------------------------------------

def pertes_annuelles(lam, g, p_u, phi_cs, rng, xi, sigma, ny=NY):
    """Copie de canaux_conformite.pertes_annuelles, xi et sigma rendus parametrables.

    L'ORDRE DES TIRAGES EST CELUI DU MODULE PARTAGE : comptage, amorce, uniformes de cascade,
    severite. Le reordonner donnerait des nombres tout aussi valides et tout aussi differents,
    et le controle 1 ne passerait plus.
    """
    r = lam / (ec.PHI - 1.0)
    counts = rng.negative_binomial(r, r / (r + lam), size=ny)
    T = int(counts.sum())
    if T == 0:
        return np.zeros(ny)
    year_of = np.repeat(np.arange(ny), counts)
    w = np.array([eng.LAMBDA[j] for j in PIL], float)
    amorce = rng.choice(5, size=T, p=w / w.sum())
    U = rng.random(T)
    SEV = simulate_remediation_severity(T * 5, xi, sigma, sp["u"], p_u,
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
        loss = (SEV[idx] * ind[sel]).sum(axis=1)
        annual += np.bincount(year_of[idx], weights=loss, minlength=ny)
    return annual


def scr_par_graine(etat, xi, sigma):
    """SCR (VaR 99,5 %) graine par graine, a severite (xi, sigma) donnee."""
    out = np.empty(NSEED)
    for k in range(NSEED):
        rng = np.random.default_rng(SEED0 + k)
        out[k] = var(pertes_annuelles(etat["lam"], etat["g"], etat["p_u"],
                                      etat["phi_cs"], rng, xi, sigma))
    return out


# ---------------------------------------------------------------------------
# 0. Les trois branches de severite, et le controle du moteur
# ---------------------------------------------------------------------------

titre("0. LES TROIS BRANCHES DE SEVERITE")
print("La derive est estimee par le module partage derive_severite.py, celui-la meme que lit la")
print("section 1ter du script 89 : les deux scripts ne peuvent donc pas s'ecarter d'une decimale.")
print()

d = ds.charger_pertes()
fit = ds.ajuster(d)
print(f"Exces : {fit['n_exc']} au-dessus du seuil publie de {fit['u']} M EUR, "
      f"sur {fit['n']} incidents.")
print(f"Derive de l'echelle de queue : b = {100*fit['b']:+.2f} % par an "
      f"(ecart-type {100*fit['se_b']:.2f} %, rapport de vraisemblance p = {fit['p_lr']:.4f}).")
print()

BRANCHES = [
    ("publiee     ", sp["xi"], sp["sigma"], "controle : doit redonner le chiffre du memoire"),
    ("stationnaire", fit["xi_stat"], fit["sigma_stat"], "reference de la comparaison"),
    ("derivee     ", fit["xi_derive"], fit["sigma_derive"],
     f"echelle evaluee a {fit['an_ref']}"),
]
print("branche          xi      sigma   role")
for nom, xi, sig, role in BRANCHES:
    print(f"  {nom}  {xi:.4f}  {sig:7.2f}   {role}")

titre("0bis. CONTROLE 1 : LA FONCTION DE PERTE LOCALE EST CELLE DU MODULE")
rng_a = np.random.default_rng(SEED0)
a = pertes_annuelles(cx.LAM_NC, cx.G_NC, cx.PU_NC, cx.PHICS_NC, rng_a, sp["xi"], sp["sigma"])
rng_b = np.random.default_rng(SEED0)
b_ref = cx.pertes_annuelles(cx.LAM_NC, cx.G_NC, cx.PU_NC, cx.PHICS_NC, rng_b, NY)
ecart_max = float(np.max(np.abs(a - b_ref)))
print(f"ecart maximal sur les {NY} annees simulees de l'etat non conforme : {ecart_max:.2e}")
print("lecture : " + ("identique a la precision machine, le moteur est bien celui du memoire"
                      if ecart_max < 1e-9 else
                      "DIVERGENT, le reste du script ne mesure PAS le modele publie"))
assert ecart_max < 1e-9, "la fonction de perte locale n'est pas celle du module partage"


# ---------------------------------------------------------------------------
# 1. Les deux etats sous chaque branche
# ---------------------------------------------------------------------------

titre("1. LES DEUX ETATS DE CONFORMITE SOUS CHAQUE BRANCHE DE SEVERITE")
print("Seule la severite de base change d'une ligne a l'autre. Les quatre canaux, les graines, le")
print("nombre d'annees et l'ordre des tirages sont ceux du chiffre publie.")
print()
print("branche          SCR conforme      SCR non conforme     facteur   ecart DORA")

res = {}
for nom, xi, sig, _role in BRANCHES:
    c = scr_par_graine(ETAT_C, xi, sig)
    nc = scr_par_graine(ETAT_NC, xi, sig)
    fac = nc.mean() / c.mean()
    fac_g = nc / c                       # facteur graine par graine
    res[nom.strip()] = dict(c=c, nc=nc, fac=fac, fac_g=fac_g)
    print(f"  {nom}  {fnum(c.mean()):>7} +- {fnum(c.std(ddof=1)):<5} "
          f"{fnum(nc.mean()):>8} +- {fnum(nc.std(ddof=1)):<6} "
          f"{fac:8.3f}   {fnum(nc.mean() - c.mean()):>7}")

pub, sta, der = res["publiee"], res["stationnaire"], res["derivee"]

print()
print("CONTROLE 2 : la branche publiee doit redonner 6 049 et 20 188 M EUR, donc 3,34 et 14 139.")
print(f"  elle donne {fnum(pub['c'].mean())} et {fnum(pub['nc'].mean())}, "
      f"facteur {pub['fac']:.2f}, ecart {fnum(pub['nc'].mean() - pub['c'].mean())}.")


# ---------------------------------------------------------------------------
# 2. La derive se simplifie-t-elle dans le rapport ?
# ---------------------------------------------------------------------------

titre("2. LA DERIVE SE SIMPLIFIE-T-ELLE DANS LE RAPPORT ?")
print("Si la derive agit sur les deux etats par le MEME facteur multiplicatif, alors le rapport")
print("entre etats ne bouge pas et l'affirmation du memoire est exacte. Les deux facteurs :")
print()
r_c = der["c"].mean() / sta["c"].mean()
r_nc = der["nc"].mean() / sta["nc"].mean()
print(f"  effet de la derive sur l'etat CONFORME      : x{r_c:.4f}  "
      f"({100*(r_c-1):+.1f} %)")
print(f"  effet de la derive sur l'etat NON CONFORME  : x{r_nc:.4f}  "
      f"({100*(r_nc-1):+.1f} %)")
print(f"  rapport des deux effets                     : {r_nc / r_c:.4f}  "
      f"(1 = simplification exacte)")
print()
d_fac = der["fac"] - sta["fac"]
print(f"  facteur entre etats, branche stationnaire : {sta['fac']:.3f}")
print(f"  facteur entre etats, branche derivee      : {der['fac']:.3f}")
print(f"  deplacement du facteur                    : {d_fac:+.3f} "
      f"({100*d_fac/sta['fac']:+.1f} %)")
print()
print("ET IL FAUT LE COMPARER AU BRUIT, sans quoi un deplacement non nul ne veut rien dire.")
print("Le facteur est calcule graine par graine, les deux etats partageant la graine :")
et_sta = float(sta["fac_g"].std(ddof=1))
et_der = float(der["fac_g"].std(ddof=1))
print(f"  ecart-type du facteur entre graines : {et_sta:.3f} (stationnaire), "
      f"{et_der:.3f} (derivee)")
print(f"  deplacement en ecarts-types         : {abs(d_fac) / max(et_sta, 1e-12):.2f}")
print()
print("MEME LECTURE SUR L'ECART EN EUROS, qui est la grandeur que le memoire publie :")
e_sta = sta["nc"].mean() - sta["c"].mean()
e_der = der["nc"].mean() - der["c"].mean()
eg_sta = sta["nc"] - sta["c"]
eg_der = der["nc"] - der["c"]
print(f"  ecart, branche stationnaire : {fnum(e_sta)} +- {fnum(eg_sta.std(ddof=1))} M EUR")
print(f"  ecart, branche derivee      : {fnum(e_der)} +- {fnum(eg_der.std(ddof=1))} M EUR")
print(f"  deplacement                 : {fnum(e_der - e_sta)} M EUR, "
      f"soit {100*(e_der/e_sta - 1):+.1f} %, et "
      f"{abs(e_der - e_sta) / max(float(eg_sta.std(ddof=1)), 1e-12):.2f} ecart-type(s)")


# ---------------------------------------------------------------------------
# 3. D'ou vient le deplacement : de la forme ou de l'echelle ?
# ---------------------------------------------------------------------------

titre("3. LE DEPLACEMENT VIENT-IL DE LA FORME OU DE L'ECHELLE ?")
print("L'ajustement derive ne fait pas que monter l'echelle : il fait aussi BAISSER l'indice de")
print(f"queue, de {fit['xi_stat']:.4f} a {fit['xi_derive']:.4f}. Les deux mouvements vont en sens "
      "contraire sur le capital, et")
print("un seul des deux peut expliquer le deplacement du facteur. On les separe au lieu de le")
print("supposer : deux branches intermediaires, un parametre a la fois.")
print()
print("branche            xi      sigma  |  SCR conf.   SCR non conf.  facteur   ecart")

for nom, xi, sig in (("forme seule  ", fit["xi_derive"], fit["sigma_stat"]),
                     ("echelle seule", fit["xi_stat"], fit["sigma_derive"])):
    c = scr_par_graine(ETAT_C, xi, sig)
    nc = scr_par_graine(ETAT_NC, xi, sig)
    res[nom.strip()] = dict(c=c, nc=nc, fac=nc.mean() / c.mean(), fac_g=nc / c)
    print(f"  {nom}   {xi:.4f}  {sig:7.2f}  |  {fnum(c.mean()):>7}   {fnum(nc.mean()):>8}    "
          f"{nc.mean()/c.mean():7.3f}  {fnum(nc.mean() - c.mean()):>7}")

print()
print("DEPLACEMENT DU FACTEUR PAR RAPPORT A LA BRANCHE STATIONNAIRE, en ecarts-types de graine :")
for cle in ("forme seule", "echelle seule", "derivee"):
    dd = res[cle]["fac"] - sta["fac"]
    print(f"  {cle:<14} {dd:+.3f}  soit {100*dd/sta['fac']:+6.1f} %  "
          f"({abs(dd)/et_sta:5.2f} ecart-type(s))")
print()
print("EFFET MULTIPLICATIF SUR CHAQUE ETAT, qui est la lecture qui dit si le rapport survit :")
for cle in ("forme seule", "echelle seule", "derivee"):
    rc = res[cle]["c"].mean() / sta["c"].mean()
    rn = res[cle]["nc"].mean() / sta["nc"].mean()
    print(f"  {cle:<14} conforme x{rc:.4f}   non conforme x{rn:.4f}   "
          f"rapport {rn/rc:.4f}")
print()
print("SOMME DES DEUX EFFETS SEPARES CONTRE L'EFFET CONJOINT, pour savoir si les deux")
print("mouvements interagissent :")
s_sep = (res["forme seule"]["fac"] - sta["fac"]) + (res["echelle seule"]["fac"] - sta["fac"])
print(f"  somme des deplacements separes : {s_sep:+.3f}")
print(f"  deplacement conjoint           : {d_fac:+.3f}")
print(f"  residu d'interaction           : {d_fac - s_sep:+.3f} "
      f"({abs(d_fac - s_sep)/et_sta:.2f} ecart-type(s))")


# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------

titre("VERDICT")
print("Ecrit APRES lecture des sorties, et non en meme temps que le code qui les produit. La")
print("premiere redaction de ce bloc annoncait l'inverse des mesures sur les deux premiers")
print("points : c'est la sortie qui tranche.")
print()

f_ech = res["echelle seule"]
r_ech_c = f_ech["c"].mean() / sta["c"].mean()
r_ech_nc = f_ech["nc"].mean() / sta["nc"].mean()
f_frm = res["forme seule"]
e_frm = f_frm["nc"].mean() - f_frm["c"].mean()
e_ech = f_ech["nc"].mean() - f_ech["c"].mean()

print("1. L'ARGUMENT DU MEMOIRE EST JUSTE, MAIS IL PORTE SUR L'ECHELLE ET SUR ELLE SEULE, ET")
print("   IL EST DESORMAIS MESURE. Une variation de la seule ECHELLE de severite multiplie les")
print(f"   deux etats par le meme facteur, x{r_ech_c:.4f} en conforme et x{r_ech_nc:.4f} en non")
print(f"   conforme : le rapport des deux effets vaut {r_ech_nc/r_ech_c:.4f}, donc le facteur "
      "entre etats est")
print(f"   invariant d'echelle a {100*abs(r_ech_nc/r_ech_c - 1):.2f} % pres, alors que les "
      "niveaux montent de "
      f"{100*(r_ech_c-1):.0f} %.")
print("   C'est la forme forte de l'affirmation, et elle tient.")
print()
print("2. MAIS LA DERIVE MESUREE N'EST PAS UN PUR CHANGEMENT D'ECHELLE, ET C'EST CE QUI DEPLACE")
print("   LE FACTEUR. Modeliser la derive REATTRIBUE a l'echelle ce que l'ajustement")
print(f"   stationnaire lisait comme de la forme : l'indice de queue tombe de "
      f"{fit['xi_stat']:.4f} a {fit['xi_derive']:.4f}.")
print(f"   Or c'est la forme qui compte : la composante de forme deplace seule le facteur de")
print(f"   {100*(f_frm['fac']-sta['fac'])/sta['fac']:+.1f} % "
      f"({abs(f_frm['fac']-sta['fac'])/et_sta:.2f} ecarts-types) quand la composante d'echelle "
      f"ne le deplace que de")
print(f"   {100*(f_ech['fac']-sta['fac'])/sta['fac']:+.1f} % "
      f"({abs(f_ech['fac']-sta['fac'])/et_sta:.2f} ecart-type, non resolu). Les deux sont "
      "additives, le residu")
print(f"   d'interaction valant {abs(d_fac - s_sep)/et_sta:.2f} ecart-type. Au total le facteur "
      f"passe de {sta['fac']:.3f} a {der['fac']:.3f},")
print(f"   soit {100*d_fac/sta['fac']:+.1f} % et {abs(d_fac)/et_sta:.2f} ecarts-types : "
      "le deplacement est RESOLU.")
print()
print("3. L'ECART EN EUROS BOUGE A PEINE, ET IL NE FAUT SURTOUT PAS Y LIRE UNE ROBUSTESSE.")
print(f"   Il passe de {fnum(e_sta)} a {fnum(e_der)} M EUR, soit {100*(e_der/e_sta-1):+.1f} % "
      f"pour un bruit de {fnum(float(eg_sta.std(ddof=1)))}, donc")
print("   non resolu. Mais cette quasi-invariance est une COMPENSATION et non une insensibilite :")
print(f"   la composante de forme seule donnerait {fnum(e_frm)} et celle d'echelle seule "
      f"{fnum(e_ech)} M EUR.")
print("   L'ecart est donc tres sensible a la severite prise composante par composante, et")
print("   presque insensible a la seule combinaison que la derive produit. L'annoncer comme")
print("   robuste a la severite serait faux.")
print()
print("4. CE QUE LE MEMOIRE DOIT EN RETENIR, ET LE SENS COMPTE. Le signe et l'ordre de l'ecart")
print("   survivent avec une marge large. La direction est favorable : sous la severite derivee")
print(f"   l'ecart est PLUS PETIT ({100*(e_der/e_sta-1):+.1f} %) et le facteur AUSSI "
      f"({100*d_fac/sta['fac']:+.1f} %), donc le chiffre publie")
print("   n'est pas gonfle par la derive. C'est la meme structure a deux sens que le p_u gele :")
print("   la derive rend le NIVEAU anti-conservateur (+24,7 % sur le quantile, script 89) et")
print("   rend la THESE legerement conservatrice. Seul le premier engage la solvabilite.")
print()
print("CE QUE CE SCRIPT NE DIT PAS, ET IL Y A DEUX RESERVES.")
print("  - il teste UNE severite alternative, celle que la derive mesuree produit a l'annee de")
print("    reference, non toute la famille des severites possibles. L'invariance d'echelle du")
print("    point 1, elle, vaut pour toute amplitude d'echelle, ce qui est plus fort ;")
print("  - la separation forme / echelle est une REPARAMETRISATION, non deux mecanismes")
print("    physiques independants. Elle dit d'ou vient le deplacement dans le modele, elle ne")
print("    dit pas que la queue s'est allegee dans le monde reel.")


# ---------------------------------------------------------------------------
# Grandeurs citees par le memoire, sans mise en forme
# ---------------------------------------------------------------------------

titre("GRANDEURS CITEES PAR LE MEMOIRE, SANS SEPARATEUR NI SIGNE")
print("Les tables ci-dessus impriment les montants avec une espace de milliers, lisible pour un")
print("humain et coupee en deux par l'extracteur du harnais de verification. Les memes valeurs")
print("sont donc reprises ici en clair, ainsi que les amplitudes que le memoire cite en valeur")
print("absolue. Ce bloc n'ajoute aucun calcul : il rend citables des nombres deja imprimes.")
print()
print(f"  ecart branche stationnaire            {e_sta:.0f}")
print(f"  ecart branche derivee                 {e_der:.0f}")
print(f"  ecart composante de forme seule       {e_frm:.0f}")
print(f"  ecart composante d'echelle seule      {e_ech:.0f}")
print(f"  bruit de graine sur l'ecart           {float(eg_sta.std(ddof=1)):.0f}")
print(f"  baisse du facteur, en valeur absolue  {abs(100*d_fac/sta['fac']):.1f} %")
print(f"  part de forme, en valeur absolue      "
      f"{abs(100*(f_frm['fac']-sta['fac'])/sta['fac']):.1f} %")
print(f"  part d'echelle, en valeur absolue     "
      f"{abs(100*(f_ech['fac']-sta['fac'])/sta['fac']):.1f} %")
print(f"  baisse de l'ecart, en valeur absolue  {abs(100*(e_der/e_sta - 1)):.1f} %")

print()
print("EXIT 0")
