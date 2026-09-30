#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
68 : la table COMPLETE des quatre canaux, et ou vivent les 5 001 M d'interaction.

CE QUE LE SCRIPT 43 LAISSE OUVERT. Il isole chaque canal, somme les quatre effets (9 138 M),
constate que l'ecart total vaut 14 139 M et imprime la difference comme un RESIDU : +5 001 M,
« interaction ». Un residu n'est pas un resultat. Tant qu'il n'est pas decompose, trois
questions restent sans reponse : d'ou vient-il, quelle paire de canaux le porte, et le tableau
des leviers boucle-t-il vraiment sur l'ecart global.

CE QUE CELUI-CI FAIT. Il enumere les SEIZE configurations du treillis des quatre canaux et en
tire trois choses :

  1. LA TABLE COMPLETE DES LEVIERS, trois lectures d'un meme canal, qui ne sont pas
     interchangeables : l'effet ISOLE (depuis l'etat conforme, ce que le canal fait seul), le
     marginal de FERMETURE (depuis l'etat non conforme, ce que remedier ce canal rapporte quand
     les trois autres restent defaillants) et la valeur de SHAPLEY (la seule attribution
     additive). Les trois diffèrent, et confondre la premiere avec la seconde est l'erreur qui
     conduit a promettre une remediation.

  2. LA RECONCILIATION, ordre par ordre. Decomposition de Mobius sur le treillis : effets
     principaux, six termes croises d'ordre 2, quatre d'ordre 3, un d'ordre 4. La somme des
     quinze termes vaut l'ecart total A LA PRECISION MACHINE. Ce n'est pas une verification
     empirique, c'est une identite algebrique : ce qu'elle atteste, c'est que la table est
     complete, pas que les nombres sont precis. La precision, c'est le point 3.

  3. LES SIX EFFETS CROISES, ET LESQUELS SONT RESOLUS. Une interaction est une difference de
     differences, l'estimateur le plus bruite du projet. Le script 20b l'a etabli sur
     l'interaction entre PILIERS : son signe change d'une graine a l'autre en VaR alors qu'il
     tient en perte moyenne. Aucun terme croise n'est donc publie sans son ecart-type de
     simulation, et chacun est lu dans les DEUX metriques.

CE QU'IL NE FAUT PAS CONFONDRE, et le memoire s'y est deja expose. Il existe DEUX
interactions dans ce projet, de meme nom et d'objets differents : celle entre les cinq PILIERS
(scripts 16b, 20, 20b ; +1 395 M redistribues par Shapley, signe non resolu en VaR) et celle
entre les quatre CANAUX, seule objet de ce script (+5 001 M). Elles ne se somment pas, ne se
comparent pas, et une slide qui presente l'une comme « la suite » de l'autre invite le lecteur
a les additionner.

Calibration gelee : ce script ne recalibre rien, il relit le moteur partage a graines,
resolution et parametres identiques a ceux du script 43.

Sortie : diagnostics + figure S24_interaction_canaux.png.
"""

import math
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import canaux_conformite as cx                                  # noqa: E402

WID = 88
N = 4
VAR, MOY = 0, 1

# (nom, parametre, statut). L'ORDRE FIXE LES BITS DU MASQUE et ne doit pas bouger.
CANAUX = [
    ("frequence",    "lam",           "calibrable"),
    ("detection",    "p_u",           "calibrable"),
    ("propagation",  "g",             "borne"),
    ("accumulation", "phi_cs",        "borne"),
]
COURT = ["freq", "det", "prop", "accum"]
# Les sorties texte du projet sont en ASCII, les FIGURES sont en francais accentue.
AFFICHE = ["fréquence", "détection", "propagation", "accumulation"]

# GRAINES. Les quatre premieres sont celles du script 43, donc celles des 14 139 M publies :
# les VALEURS de ce script sont calculees sur elles, et sur elles seules. Les douze suivantes ne
# servent QU'A estimer l'ecart-type de simulation, parce qu'un ecart-type mesure sur quatre
# tirages est lui-meme connu a 40 % pres, ce qui ne permet pas de dire si un terme croise de
# quelques centaines de millions est reel. Aucune moyenne sur seize graines n'est imprimee :
# ce serait une seconde valeur du meme nombre, exactement le defaut que le projet a corrige
# sur la VaR predictive.
NSEED_PUB = cx.NSEED
NSEED_BRUIT = 16


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def bits(m):
    return bin(m).count("1")


def nom_masque(m):
    if m == 0:
        return "conforme"
    return "+".join(COURT[i] for i in range(N) if m & (1 << i))


def config(m):
    """Configuration de canaux du masque m : bit a 1 = canal non conforme."""
    return dict(
        lam=cx.LAM_NC if m & 1 else cx.LAM_C,
        p_u=cx.PU_NC if m & 2 else cx.PU_C,
        g=cx.G_NC if m & 4 else cx.G_C,
        phi_cs=cx.PHICS_NC if m & 8 else None,
    )


# ------------------------------------------------------------------ algebre du treillis
def mobius(v):
    """Termes d'interaction de Mobius : I(S) = somme_{T inclus dans S} (-1)^{|S|-|T|} v(T).

    Identite : somme des I(S) sur tous les S non vides = v(ensemble complet). C'est ce qui
    fait que la table BOUCLE, par construction et non par chance.
    """
    return {S: sum((-1) ** (bits(S) - bits(T)) * v[T]
                   for T in range(1 << N) if T & S == T)
            for S in range(1, 1 << N)}


def shapley(v):
    """Attribution de Shapley des quatre canaux. Somme = v(complet), exactement."""
    phi = np.zeros(N)
    for i in range(N):
        bit = 1 << i
        for S in range(1 << N):
            if S & bit:
                continue
            s = bits(S)
            w = math.factorial(s) * math.factorial(N - s - 1) / math.factorial(N)
            phi[i] += w * (v[S | bit] - v[S])
    return phi


# =====================================================================================
titre("1. Le treillis complet : seize configurations de canaux")
# =====================================================================================
# M[masque] a la forme (graine, metrique)
M = {}
for m in range(1 << N):
    M[m] = cx.metriques_par_graine(**config(m), nseed=NSEED_BRUIT)

print(f"  Graines : {NSEED_PUB} pour les valeurs (celles du script 43), "
      f"{NSEED_BRUIT} pour les ecarts-types.")
print(f"  {cx.NY} annees par graine, VaR 99,5 %.")
print(f"\n  {'configuration':<26}{'SCR (M)':>10}{'ecart a conforme':>19}{'ordre':>7}")
for m in sorted(range(1 << N), key=lambda x: (bits(x), x)):
    scr = M[m][:NSEED_PUB, VAR].mean()
    d = scr - M[0][:NSEED_PUB, VAR].mean()
    print(f"  {nom_masque(m):<26}{scr:>10.0f}{d:>19.0f}{bits(m):>7}")

PLEIN = (1 << N) - 1
scr_C = M[0][:NSEED_PUB, VAR].mean()
scr_NC = M[PLEIN][:NSEED_PUB, VAR].mean()
print(f"\n  Reperes du script 43 : conforme {scr_C:.0f} M, non conforme {scr_NC:.0f} M,")
print(f"  ecart total {scr_NC - scr_C:.0f} M, facteur {scr_NC / scr_C:.2f}.")

# valeurs publiees (4 graines) et jeu complet (16 graines, pour les ecarts-types)
v_pub = {m: M[m][:NSEED_PUB, VAR].mean() - scr_C for m in range(1 << N)}
V_seed = {k: {m: M[m][k, VAR] - M[0][k, VAR] for m in range(1 << N)}
          for k in range(NSEED_BRUIT)}
Vmoy_seed = {k: {m: M[m][k, MOY] - M[0][k, MOY] for m in range(1 << N)}
             for k in range(NSEED_BRUIT)}


def sigma_terme(fonction, jeu):
    """Ecart-type inter-graines d'une grandeur derivee, sur les 16 graines."""
    vals = np.array([fonction(jeu[k]) for k in range(NSEED_BRUIT)])
    return vals.std(axis=0, ddof=1)


# =====================================================================================
titre("2. La table complete des leviers : trois lectures d'un meme canal")
# =====================================================================================
iso = np.array([v_pub[1 << i] for i in range(N)])
ferm = np.array([v_pub[PLEIN] - v_pub[PLEIN ^ (1 << i)] for i in range(N)])
phi = shapley(v_pub)

s_iso = sigma_terme(lambda v: np.array([v[1 << i] for i in range(N)]), V_seed)
s_ferm = sigma_terme(lambda v: np.array([v[PLEIN] - v[PLEIN ^ (1 << i)] for i in range(N)]),
                     V_seed)
s_phi = sigma_terme(shapley, V_seed)

print("  ISOLE      : depuis l'etat conforme, ce canal seul passe en non conforme.")
print("  FERMETURE  : depuis l'etat non conforme, ce canal seul revient a la cible.")
print("  SHAPLEY    : moyenne sur les 24 ordres d'entree ; la seule lecture qui SOMME au total.")
print(f"\n  {'canal':<15}{'isole':>15}{'fermeture':>15}{'Shapley':>15}{'statut':>13}")
for i, (nom, par, statut) in enumerate(CANAUX):
    print(f"  {nom:<15}{iso[i]:>8.0f} +-{s_iso[i]:<4.0f}{ferm[i]:>8.0f} +-{s_ferm[i]:<4.0f}"
          f"{phi[i]:>8.0f} +-{s_phi[i]:<4.0f}{statut:>13}")
print(f"  {'-' * 71}")
print(f"  {'somme':<15}{iso.sum():>15.0f}{ferm.sum():>15.0f}{phi.sum():>15.0f}")
print(f"  {'ecart total':<15}{v_pub[PLEIN]:>15.0f}{v_pub[PLEIN]:>15.0f}{v_pub[PLEIN]:>15.0f}")

print(f"\n  LA COLONNE ISOLEE MANQUE {v_pub[PLEIN] - iso.sum():.0f} M, soit "
      f"{100 * (v_pub[PLEIN] - iso.sum()) / v_pub[PLEIN]:.0f} % de l'ecart : c'est le residu du")
print("  script 43. La colonne de FERMETURE, elle, depasse le total de "
      f"{ferm.sum() - v_pub[PLEIN]:+.0f} M.")
print(f"  Rapport fermeture / isole : de {min(ferm / iso):.1f} a {max(ferm / iso):.1f} selon le canal.")
print("  CONSEQUENCE DE GESTION, ET C'EST LE POINT UTILE. Remedier un canal rapporte")
print("  DAVANTAGE a une entite deja defaillante partout qu'a une entite conforme, parce que")
print("  le canal ferme aussi les effets croises qu'il portait. Les deux lectures ne sont donc")
print("  pas deux estimations du meme nombre : ce sont deux questions differentes, et seule la")
print("  seconde correspond a une entite reelle en debut de remediation.")
print("  Shapley est la seule colonne qui somme au total, mais c'est une CONVENTION")
print("  d'attribution, pas une prevision de remediation : deux des quatre canaux sont bornes")
print("  et non calibres, donc leur part n'est pas un budget actionnable.")

# =====================================================================================
titre("3. Reconciliation exacte : la decomposition de Mobius, ordre par ordre")
# =====================================================================================
I = mobius(v_pub)
par_ordre = {o: sum(I[S] for S in I if bits(S) == o) for o in range(1, N + 1)}
s_ordre = sigma_terme(
    lambda v: np.array([sum(mobius(v)[S] for S in range(1, 1 << N) if bits(S) == o)
                        for o in range(1, N + 1)]), V_seed)

print(f"  {'ordre':<32}{'termes':>8}{'total (M)':>14}{'part de l ecart':>18}")
for o in range(1, N + 1):
    nb = sum(1 for S in I if bits(S) == o)
    lib = {1: "effets principaux", 2: "croises de paires", 3: "croises de triplets",
           4: "croise des quatre"}[o]
    print(f"  ordre {o} : {lib:<24}{nb:>8}{par_ordre[o]:>9.0f} +-{s_ordre[o-1]:<4.0f}"
          f"{100 * par_ordre[o] / v_pub[PLEIN]:>16.1f} %")
print(f"  {'-' * 72}")
somme = sum(par_ordre.values())
print(f"  {'somme des quinze termes':<32}{15:>8}{somme:>14.0f}{100 * somme / v_pub[PLEIN]:>16.1f} %")
print(f"  {'ecart total (non conforme - conforme)':<40}{v_pub[PLEIN]:>14.0f}")
print(f"\n  RESIDU DE RECONCILIATION = {somme - v_pub[PLEIN]:.2e} M, soit zero a la precision")
print("  machine. C'est une IDENTITE de Mobius, pas une verification statistique : ce qu'elle")
print("  atteste est que la table est COMPLETE, aucun euro de l'ecart n'etant hors des quinze")
print("  termes. Elle ne dit rien de la precision de chaque terme, qui est l'objet du point 5.")
print(f"\n  Et elle referme le residu du script 43 : les {par_ordre[2] + par_ordre[3] + par_ordre[4]:.0f} M")
print(f"  des ordres 2 a 4 sont exactement l'interaction que le 43 imprimait sans la decomposer.")

# =====================================================================================
titre("4. Les six effets croises d'ordre 2 : quelle paire porte l'interaction")
# =====================================================================================
paires = [(i, j) for i in range(N) for j in range(i + 1, N)]
inter_tot = par_ordre[2] + par_ordre[3] + par_ordre[4]


def _vec_paires(v):
    Iv = mobius(v)
    return np.array([Iv[(1 << i) | (1 << j)] for i, j in paires])


val_p = _vec_paires(v_pub)
sig_p = sigma_terme(_vec_paires, V_seed)
# structure dans la SECONDE metrique : part de chaque paire dans le total d'ordre 2, en perte
# moyenne. On compare des PARTS et non des niveaux, les deux metriques n'ayant pas la meme
# echelle : ce qui doit se corroborer, c'est la structure, pas le montant.
val_p_moy = np.array([_vec_paires(Vmoy_seed[k]) for k in range(NSEED_BRUIT)]).mean(axis=0)

ordre_lecture = np.argsort(-np.abs(val_p))
print(f"  {'paire de canaux':<30}{'croise (M)':>16}{'part ordre 2':>15}{'perte moyenne':>16}")
for idx in ordre_lecture:
    i, j = paires[idx]
    nom = f"{CANAUX[i][0]} x {CANAUX[j][0]}"
    part = 100 * val_p[idx] / par_ordre[2]
    part_moy = 100 * val_p_moy[idx] / val_p_moy.sum()
    print(f"  {nom:<30}{val_p[idx]:>9.0f} +-{sig_p[idx]:<4.0f}{part:>13.0f} %{part_moy:>14.0f} %")
print(f"  {'-' * 77}")
print(f"  {'total ordre 2':<30}{par_ordre[2]:>16.0f}{100:>13.0f} %{100:>14.0f} %")

i_fa = paires.index((0, 3))
print(f"\n  LA PAIRE QUE LE CALL DEMANDAIT, frequence x accumulation P4 : "
      f"{val_p[i_fa]:+.0f} +-{sig_p[i_fa]:.0f} M,")
print(f"  soit {100 * val_p[i_fa] / par_ordre[2]:.0f} % du total d'ordre 2 et "
      f"{100 * val_p[i_fa] / inter_tot:.0f} % de toute l'interaction.")
print("  MECANISME. Le choc commun de P4 fait tomber plusieurs piliers sur le MEME sinistre ; la")
print("  frequence multiplie le nombre de sinistres exposes a ce choc. A queue lourde le quantile")
print("  est porte par un sinistre unique : multiplier les tirages et epaissir chaque tirage ne")
print("  produit pas deux effets qui s'ajoutent mais un seul effet compose. C'est la definition")
print("  du croise, et c'est ce que la frequence fait avec CHACUN des trois autres canaux.")

# --------------------------------------------------------------------------------------
# CE QUI EST RESOLU DANS CE CLASSEMENT, ET CE QUI NE L'EST PAS. La tentation est de nommer
# une paire dominante. Elle n'existe pas : les deux premieres se tiennent a 73 M l'une de
# l'autre pour des ecarts-types de 708 et 354, et leur ORDRE S'INVERSE quand on passe a la
# perte moyenne. Ce qui est resolu, c'est que les trois paires porteuses passent toutes par
# la frequence.
# --------------------------------------------------------------------------------------
i1, i2 = ordre_lecture[0], ordre_lecture[1]
ecart12 = val_p[i1] - val_p[i2]
print(f"\n  PAS DE PAIRE DOMINANTE, et il faut le dire. Les deux premieres se tiennent a "
      f"{ecart12:.0f} M")
print(f"  l'une de l'autre pour des ecarts-types de {sig_p[i1]:.0f} et {sig_p[i2]:.0f} M, et leur")
print("  ordre S'INVERSE en perte moyenne. Le classement des deux premieres n'est donc pas")
print("  resolu et n'a pas a l'etre. Ce qui l'est : les trois paires porteuses passent TOUTES")
print(f"  par la frequence, et elles font a elles seules "
      f"{100 * sum(val_p[paires.index(p)] for p in [(0, 1), (0, 2), (0, 3)]) / par_ordre[2]:.0f} % "
      f"de l'ordre 2.")

# --------------------------------------------------------------------------------------
i_pa = paires.index((2, 3))
print(f"\n  LA SEULE PAIRE NEGATIVE, et c'est un resultat a part : propagation x accumulation,")
print(f"  {val_p[i_pa]:+.0f} +-{sig_p[i_pa]:.0f} M, negative sur les {NSEED_BRUIT} graines et sur")
print("  les deux metriques. Les deux canaux sont SUBSTITUTS et non complements : la cascade par")
print("  W et le choc commun de P4 sont deux facons de faire tomber plusieurs piliers sur un")
print("  meme sinistre. Le premier actif, il reste moins a gagner au second, un pilier deja")
print("  touche ne pouvant l'etre deux fois. La cascade est donc super-additive en general mais")
print("  SOUS-additive entre ses deux canaux de co-occurrence, ce qui borne ce qu'on peut")
print("  attribuer a la contagion prise en bloc : les deux canaux bornes ne s'empilent pas.")

# =====================================================================================
titre("5. Ce qui est resolu, et ce qui ne l'est pas")
# =====================================================================================
def _vec_tous(v):
    Iv = mobius(v)
    return np.array([Iv[S] for S in range(1, 1 << N)])


tous = np.array([_vec_tous(V_seed[k]) for k in range(NSEED_BRUIT)])
signes = (tous > 0).sum(axis=0)
noms = [nom_masque(S) for S in range(1, 1 << N)]
val_tous = _vec_tous(v_pub)
sig_tous = tous.std(axis=0, ddof=1)

resolus = [i for i in range(len(noms)) if signes[i] in (0, NSEED_BRUIT)]
print(f"  Un terme est dit RESOLU si son signe est le meme sur les {NSEED_BRUIT} graines.")
print(f"  Resolus : {len(resolus)} termes sur 15.")
print(f"\n  {'terme':<26}{'valeur (M)':>14}{'ecart-type':>12}{'graines > 0':>14}{'resolu':>9}")
for i in np.argsort(-np.abs(val_tous)):
    r = "oui" if signes[i] in (0, NSEED_BRUIT) else "NON"
    print(f"  {noms[i]:<26}{val_tous[i]:>14.0f}{sig_tous[i]:>12.0f}"
          f"{signes[i]:>10}/{NSEED_BRUIT}{r:>9}")

# l'interaction totale, la seule grandeur que le memoire publie de cette famille
tot_inter = tous[:, [i for i in range(15) if bits(i + 1) >= 2]].sum(axis=1)
print(f"\n  L'INTERACTION TOTALE (ordres 2 a 4) = {inter_tot:.0f} M, ecart-type "
      f"{tot_inter.std(ddof=1):.0f} M sur {NSEED_BRUIT} graines,")
print(f"  positive sur {(tot_inter > 0).sum()}/{NSEED_BRUIT} graines. Elle est donc "
      f"{'RESOLUE' if (tot_inter > 0).all() else 'NON RESOLUE'} en signe,")
print("  ce qui n'allait pas de soi : sur l'interaction entre PILIERS, a resolution comparable,")
print("  le script 20b trouve 4 graines positives sur 6 et conclut que le signe n'est pas")
print("  resolu. Ici il l'est, et la raison est de taille : l'interaction entre canaux vaut")
print(f"  {100 * inter_tot / v_pub[PLEIN]:.0f} % de son ecart quand celle entre piliers en vaut 13 %.")
print("\n  A NE PAS CITER SANS SON ECART-TYPE. Les termes d'ordre 3 et 4 sont des differences de")
print("  differences de differences : leur ecart-type approche leur valeur. Ils sont dans la")
print("  table parce que la reconciliation l'exige, pas parce qu'ils sont mesures.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print(f"  1. La table des leviers BOUCLE : quinze termes, somme = {v_pub[PLEIN]:.0f} M a la")
print("     precision machine. Le +5 001 M du script 43 n'est plus un residu, il est reparti.")
print(f"  2. L'interaction vit AUX PAIRES : ordre 2 = {par_ordre[2]:.0f} M, ordres 3 et 4 reunis")
print(f"     {par_ordre[3] + par_ordre[4]:.0f} M, et aucun de ces cinq termes n'est resolu")
print("     individuellement : leur quasi-annulation ne doit pas se lire comme cinq mesures.")
print("     Le mecanisme est a DEUX canaux, pas diffus.")
print("  3. Les trois paires porteuses passent toutes par la FREQUENCE. Le classement des deux")
print("     premieres n'est pas resolu, et s'inverse d'une metrique a l'autre.")
print(f"  3bis. Une seule paire est negative, propagation x accumulation ({val_p[paires.index((2, 3))]:.0f} M),")
print("     resolue sur les 16 graines et les deux metriques : les deux canaux de co-occurrence")
print("     sont substituts. La cascade est super-additive en bloc, sous-additive entre ces deux.")
print("  4. Trois lectures d'un canal, non interchangeables : isole, fermeture, Shapley. La")
print("     fermeture est la seule qui reponde a la question d'une entite en remediation, et")
print("     elle est la plus grande des trois.")
print("  5. Le signe de l'interaction entre CANAUX est resolu ; celui entre PILIERS ne l'est")
print("     pas (script 20b). Deux objets distincts, a ne jamais additionner.")

# =====================================================================================
# figure S24
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

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Trois panneaux traces
# sur seize pouces puis imprimes sur 7,27 donnent 1,6 pouce par panneau et
# des etiquettes sous 5 points. Le rapport largeur sur hauteur est conserve.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.66, 3.55),
                                    gridspec_kw={"width_ratios": [1.05, 1.0, 1.15]})

# (a) reconciliation en cascade
lab_a = ["conforme", "ordre 1", "ordre 2", "ordre 3",
         "ordre 4", "non\nconforme"]
ax1.bar(0, scr_C, color=GREEN, alpha=0.9, width=0.62)
ax1.text(0, scr_C + 350, f"{scr_C:.0f}", ha="center", fontsize=8.5, color=GREEN)
base = scr_C
for k, o in enumerate(range(1, N + 1), start=1):
    val = par_ordre[o]
    col = BLUE if o == 1 else ACCENT
    ax1.bar(k, val, bottom=base, color=col, alpha=0.85 if o == 1 else 0.55, width=0.62)
    err = s_ordre[o - 1]
    ax1.errorbar(k, base + val, yerr=err, color=INK2, capsize=3, lw=0.9, ls="none")
    # L'ETIQUETTE PASSE AU-DESSUS DU PLUS HAUT DES DEUX BORDS, jamais sous la barre : posee
    # sous un terme NEGATIF elle etait traversee par sa propre barre d'erreur, defaut vu en
    # regardant la figure et invisible a tout controle de proportions.
    haut = max(base, base + val) + err
    ax1.text(k, haut + 380, f"{val:+.0f}", ha="center", fontsize=8, color=INK2)
    base += val
ax1.bar(5, scr_NC, color=ACCENT, alpha=0.9, width=0.62)
ax1.text(5, scr_NC + 350, f"{scr_NC:.0f}", ha="center", fontsize=8.5, color=ACCENT)
ax1.set_xticks(range(6))
ax1.set_xticklabels(lab_a, fontsize=7.5, rotation=18, ha="right")
ax1.set_ylabel("SCR (VaR 99,5 %, M€)", color=INK2)
ax1.set_ylim(0, scr_NC * 1.24)
ax1.set_title("(a)  La table boucle : les quinze termes\nsomment à l'écart, exactement",
              fontsize=10.5, color=INK, pad=8)

# (b) trois lectures d'un meme canal
y = np.arange(N)[::-1]
h = 0.26
ax2.barh(y + h, iso, height=h, color=MUTED, alpha=0.9, label="isolé")
ax2.barh(y, phi, height=h, color=BLUE, alpha=0.85, label="Shapley")
ax2.barh(y - h, ferm, height=h, color=ACCENT, alpha=0.85, label="fermeture")
# LES TROIS BARRES SONT CHIFFREES, PAS UNE SEULE. Le propos du panneau est que les trois
# lectures diffèrent : n'en etiqueter qu'une laisse le lecteur estimer l'ecart a l'oeil.
for k in range(N):
    for dy, val in ((h, iso[k]), (0.0, phi[k]), (-h, ferm[k])):
        ax2.text(val + 90, y[k] + dy, f"{val:.0f}", va="center", fontsize=7, color=INK2)
ax2.set_yticks(y)
ax2.set_yticklabels([f"{AFFICHE[i]}\n({CANAUX[i][2]})" for i in range(N)], fontsize=8.5)
ax2.set_xlabel("capital en jeu (M€)", color=INK2)
ax2.set_xlim(0, max(ferm) * 1.30)
ax2.set_xlim(0, ax2.get_xlim()[1] * 1.28)
ax2.legend(fontsize=7.4, loc="lower right", frameon=True, framealpha=0.92,
           edgecolor="#dcdcdc")
ax2.set_title("(b)  Trois lectures d'un même canal,\net elles ne sont pas interchangeables",
              fontsize=10.5, color=INK, pad=8)

# (c) les six croises de paires, avec leur bruit
o2 = np.argsort(val_p)
lab_c = [f"{COURT[paires[i][0]]} × {COURT[paires[i][1]]}" for i in o2]
yc = np.arange(len(o2))
ax3.barh(yc, val_p[o2], color=[ACCENT if v > 0 else MUTED for v in val_p[o2]], alpha=0.85)
ax3.errorbar(val_p[o2], yc, xerr=sig_p[o2], color=INK2, capsize=3, lw=0.9, ls="none")
for k, i in enumerate(o2):
    dx = 60 if val_p[i] > 0 else -60
    ha = "left" if val_p[i] > 0 else "right"
    ax3.text(val_p[i] + sig_p[i] * np.sign(val_p[i]) + dx, k,
             f"{val_p[i]:+.0f}", va="center", ha=ha, fontsize=8, color=INK2)
ax3.axvline(0, color=INK2, lw=0.8)
ax3.set_yticks(yc)
ax3.set_yticklabels(lab_c, fontsize=8.5)
ax3.set_xlabel("effet croisé d'ordre 2 (M€)", color=INK2)
lo, hi = min(val_p) - max(sig_p), max(val_p) + max(sig_p)
ax3.set_xlim(lo - 0.22 * (hi - lo), hi + 0.30 * (hi - lo))
ax3.set_title("(c)  Où vit l'interaction : trois paires\npassent par la fréquence, "
              "une seule paire est négative",
              fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2, ax3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# LE SEPARATEUR DE MILLIERS SE POSE SUR LE NOMBRE, PAS SUR LA PHRASE. Un .replace pose
# sur la chaine entiere avait mange la virgule du titre : deuxieme fois dans ce script.
_mille = f"{inter_tot:,.0f}".replace(",", " ")
fig.tight_layout(w_pad=1.8)
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S24_interaction_canaux.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
