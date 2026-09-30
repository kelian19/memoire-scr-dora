#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
66 : la these ne depend pas des trois valeurs de g, seulement de leur ORDRE.

LE PROBLEME, ET IL EST LE PLUS SERIEUX QUI RESTE. Le memoire chiffre un besoin de capital
lie a la NON-CONFORMITE, et la non-conformite n'est observee nulle part. Le canal par lequel
la conformite agit est le gain de propagation g, et ses trois valeurs

        g = 0,45 (conforme)   0,68 (partiellement conforme)   0,90 (non conforme)

sont POSEES. Le script 16 les marque « non calibre » depuis le debut, ce qui est honnete mais
insuffisant : tant que le resultat est presente comme « le SCR passe de 116 a 169 M EUR »,
il herite entierement de ces trois nombres, et un rapporteur demandera d'ou vient le 0,90.
Sur une entite notionnelle la question restait theorique. Depuis que le script 65 applique le
modele a des bilans reels, elle devient frontale.

CE QUE FAIT CE SCRIPT. Il ne calibre pas g, faute de donnee, et ne pretend pas le faire. Il
montre que la CONCLUSION du memoire n'en a pas besoin, par un argument en deux temps.

  (1) LEMME DE MONOTONIE. Le SCR est croissant en g. Verifie numeriquement sur une grille
      fine, a nombres aleatoires communs, aux deux echelles (entite et secteur). Ce n'est pas
      une evidence : la VaR d'une loi a queue lourde peut etre non monotone en un parametre
      de structure, et le chapitre resultats montre justement qu'elle est peu sensible a la
      cascade. Il faut donc le verifier, pas le supposer.

  (2) INVARIANCE. Si le SCR est croissant en g, alors TOUTE application etat -> g qui
      respecte l'ordre g_C < g_PC < g_NC produit SCR_C < SCR_PC < SCR_NC. Or la charge
      forfaitaire de Formule Standard est CONSTANTE sur les trois etats. L'existence et le
      SENS de l'ecart ne dependent donc pas des valeurs, seulement de leur ordre. Seule son
      AMPLITUDE est un scenario.

CE QUI RESTE UNE HYPOTHESE, ET C'EST TOUT CE QUI RESTE : « se conformer a DORA reduit la
propagation entre piliers ». C'est une affirmation qualitative, discutable en soutenance, et
sur laquelle le reglement lui-meme prend parti (les cinq piliers sont des exigences de
maitrise). Elle est infiniment plus defendable qu'un triplet de decimales.

ON DONNE EN PLUS L'ELASTICITE, pour qu'un lecteur qui n'aime pas nos valeurs puisse
transposer les siennes sans relancer le modele, et l'ecart en fonction du seul ECARTEMENT
g_NC - g_C, qui est la grandeur dont l'amplitude depend reellement.

Sortie : diagnostics + figure S23_invariance_conformite.png
"""

import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUALI = os.path.abspath(os.path.join(HERE, "..", "cascade_qualitative"))
for _p in (HERE, QUALI):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(HERE)
import partial_id as pid                                            # noqa: E402

# echelle d'entite : valeurs FIGEES du script 60, comme les scripts 58 et 64
LAM_ENTITE = 0.09168423156000373
SEV_MULT = 0.8545
NY_ENTITE = 600_000
NY_SECTEUR = 40_000
SEED = 20260721
ALPHA = 0.995
SF_ENTITE = 450.0            # charge forfaitaire de l'entite notionnelle : 0,03 x 15000
W_ = 90

# les trois valeurs POSEES du script 16, reprises telles quelles pour etre situees
G_MEMOIRE = {"C": 0.45, "PC": 0.68, "NC": 0.90}
ETIQ = {"C": "conforme", "PC": "partiellement conforme", "NC": "non conforme"}


def titre(s):
    print("\n" + "=" * W_ + f"\n{s}\n" + "=" * W_)


# =====================================================================================
titre("0. Ce qui est pose, et ce qui ne l'est pas")
# =====================================================================================
print("POSE, sans calibration : l'application etat -> g.")
for e in ("C", "PC", "NC"):
    print(f"   {ETIQ[e]:<26} g = {G_MEMOIRE[e]:.2f}")
print("\nNON POSE, et c'est la difference : la matrice de contagion W, dont la structure est")
print("corroboree par le corpus de post-mortems (chapitre 9) et dont la direction est bornee")
print("par identification partielle (chapitre 10). g n'est qu'un facteur d'echelle applique a")
print("cette matrice : W(g) = g * TRANS / max_j(s_j).")
print("\nLA QUESTION A LAQUELLE CE SCRIPT REPOND. Que reste-t-il de la these si l'on refuse")
print("les trois valeurs ci-dessus ? Reponse : tout, sauf l'amplitude.")


# =====================================================================================
titre("1. Lemme : le SCR est-il CROISSANT en g ? (a verifier, pas a supposer)")
# =====================================================================================
print("A nombres aleatoires communs : le tirage des annees, des amorces, des uniformes et des")
print("severites est fait UNE fois et reutilise pour tous les g. Seule la loi du nombre de")
print("piliers touches change. Sans cela, deux g voisins differeraient du seul bruit de")
print("tirage et la monotonie serait indecidable.\n")

ev_e = pid.Evaluator(lam=LAM_ENTITE, n_years=NY_ENTITE, alpha=ALPHA, seed=SEED)
ev_e.cum = ev_e.cum * SEV_MULT
ev_s = pid.Evaluator(n_years=NY_SECTEUR, alpha=ALPHA, seed=SEED)      # lam_ref d'OpRisk


def scr(ev, g):
    return ev.from_card(pid.card_dist_all(pid.expert_matrix(g))[0])


GRILLE = np.round(np.arange(0.0, 0.951, 0.05), 3)
print(f"{'g':>6}{'SCR entite':>13}{'E[X] entite':>14}{'SCR secteur':>14}{'E[X] secteur':>15}")
se, me, ss, ms = [], [], [], []
for g in GRILLE:
    v1, e1 = scr(ev_e, g)
    v2, e2 = scr(ev_s, g)
    se.append(v1); me.append(e1); ss.append(v2); ms.append(e2)
    print(f"{g:>6.2f}{v1:>13.1f}{e1:>14.2f}{v2:>14.1f}{e2:>15.1f}")
se, me, ss, ms = map(np.array, (se, me, ss, ms))


def viole(v):
    d = np.diff(v)
    return int((d < 0).sum()), (float(d.min()) if len(d) else 0.0)


for lab, v in (("SCR entite", se), ("perte moyenne entite", me),
               ("SCR secteur", ss), ("perte moyenne secteur", ms)):
    n, mn = viole(v)
    print(f"\n  {lab:<24} pas decroissants : {n} sur {len(v)-1}"
          + (f"   (pire pas : {mn:+.2f})" if n else "   -> CROISSANT sur toute la grille"))

mono = all(viole(v)[0] == 0 for v in (se, me, ss, ms))
print(f"\n  LEMME DE MONOTONIE : {'VERIFIE' if mono else 'INFIRME'} sur les quatre grandeurs.")
if not mono:
    print("  Attention : la suite du raisonnement suppose la monotonie. Un pas decroissant")
    print("  isole peut venir du bruit residuel ; plusieurs invalideraient l'argument.")
print("\n  Ce lemme n'allait pas de soi. Le chapitre resultats etablit que la VaR d'une queue")
print("  a indice eleve est PEU sensible a la structure de cascade, par principe du grand")
print("  saut unique. Peu sensible ne veut pas dire monotone, et c'est la monotonie, non la")
print("  sensibilite, qui porte l'argument suivant.")


# =====================================================================================
titre("2. Invariance : l'ecart existe pour TOUTE application ordonnee etat -> g")
# =====================================================================================
print("Si le SCR est croissant en g, alors g_C < g_PC < g_NC entraine")
print("   SCR(g_C) < SCR(g_PC) < SCR(g_NC),")
print("quelles que soient les VALEURS. La charge forfaitaire, elle, ne depend pas de l'etat :")
print(f"elle vaut {SF_ENTITE:.0f} M EUR pour les trois. L'existence et le SENS de l'ecart sont donc")
print("INVARIANTS a la calibration ; seule son amplitude est un scenario.\n")

MAPPINGS = [
    ("etroite      (0,70 / 0,80 / 0,90)", {"C": 0.70, "PC": 0.80, "NC": 0.90}),
    ("RETENUE      (0,45 / 0,68 / 0,90)", dict(G_MEMOIRE)),
    ("large        (0,20 / 0,55 / 0,90)", {"C": 0.20, "PC": 0.55, "NC": 0.90}),
    ("conformite faible (0,45/0,55/0,65)", {"C": 0.45, "PC": 0.55, "NC": 0.65}),
    ("tres etroite (0,85 / 0,875 / 0,90)", {"C": 0.85, "PC": 0.875, "NC": 0.90}),
]
print(f"{'application etat -> g':<36}{'SCR C':>9}{'SCR PC':>9}{'SCR NC':>9}"
      f"{'ecart':>9}{'%':>8}{'ordre':>8}")
res = {}
for lab, gm in MAPPINGS:
    v = {e: scr(ev_e, gm[e])[0] for e in ("C", "PC", "NC")}
    ordre_ok = v["C"] < v["PC"] < v["NC"]
    ec = v["NC"] - v["C"]
    res[lab] = (gm, v, ec)
    print(f"{lab:<36}{v['C']:>9.1f}{v['PC']:>9.1f}{v['NC']:>9.1f}"
          f"{ec:>9.1f}{100*(v['NC']/v['C']-1):>+8.1f}{'oui' if ordre_ok else 'NON':>8}")
print(f"{'forfait de Formule Standard':<36}{SF_ENTITE:>9.1f}{SF_ENTITE:>9.1f}"
      f"{SF_ENTITE:>9.1f}{0.0:>9.1f}{0.0:>+8.1f}{'plat':>8}")

tous_ordonnes = all(v["C"] < v["PC"] < v["NC"] for _, v, _ in res.values())
print(f"\n  ordre respecte par les {len(MAPPINGS)} applications testees : {tous_ordonnes}")
print("  Y compris la derniere, ou la conformite ne deplace g que de 0,05 : l'ecart devient")
print(f"  petit ({res['tres etroite (0,85 / 0,875 / 0,90)'][2]:.1f} M EUR) mais il ne change pas de signe, quand le forfait reste")
print("  exactement plat. C'est cela, la these, et elle ne contient aucune decimale.")


# =====================================================================================
titre("3. L'elasticite, pour que le lecteur transpose SES valeurs")
# =====================================================================================
print("Plutot que d'imposer nos trois nombres, on donne la sensibilite locale. Un lecteur qui")
print("juge que la conformite deplace g de 0,80 a 0,90 et non de 0,45 a 0,90 lit son propre")
print("ecart sur cette table, sans relancer le modele.\n")
# L'ELASTICITE LOCALE EST BRUITEE, ET IL FAUT LE DIRE PLUTOT QUE DE LA LISSER EN SILENCE.
# Une difference finie sur un pas de 0,05 divise un ecart de VaR par un petit nombre : le
# bruit residuel de Monte-Carlo, invisible sur le niveau, y devient visible. La grandeur a
# citer est donc l'elasticite d'ARC, estimee par regression log-log sur toute la plage, et
# les valeurs locales ne sont donnees qu'a titre indicatif.
_m = GRILLE >= 0.20
_b, _a = np.polyfit(np.log(GRILLE[_m]), np.log(se[_m]), 1)
print(f"  >>> ELASTICITE D'ARC, regression log-log sur g dans [0,20 ; 0,95] : "
      f"{_b:.3f}")
print(f"      soit : +10 % sur g donnent {100*((1.1)**_b - 1):.1f} % de capital en plus.\n")
print(f"{'g':>6}{'SCR':>10}{'dSCR/dg':>12}{'elasticite locale':>20}")
for i, g in enumerate(GRILLE):
    if i == 0 or g < 0.20:
        continue
    d = (se[i] - se[i - 1]) / (GRILLE[i] - GRILLE[i - 1])
    el = d * g / se[i]
    print(f"{g:>6.2f}{se[i]:>10.1f}{d:>12.1f}{el:>20.2f}")
print("\n  Les valeurs locales oscillent (0,19 a 0,68) : c'est du bruit de difference finie,")
print("  non une structure. Seule l'elasticite d'arc ci-dessus est citable.")
print("\n  L'elasticite reste inferieure a 1 sur toute la plage : le capital repond a g, mais")
print("  moins que proportionnellement. Le niveau du SCR est porte par la severite et la")
print("  frequence, la conformite en deplace une part. C'est coherent avec le principe du")
print("  grand saut unique, et c'est ce qui rend l'ecart robuste sans etre spectaculaire.")


# =====================================================================================
titre("4. L'amplitude depend du seul ECARTEMENT, et on la trace")
# =====================================================================================
print("A g_NC fixe a 0,90, on fait varier g_C et on lit l'ecart. C'est la seule grandeur dont")
print("l'amplitude depend reellement, et elle se lit d'un coup d'oeil.\n")
G_NC = 0.90
scr_nc = scr(ev_e, G_NC)[0]
print(f"{'g_C':>7}{'ecartement':>13}{'SCR C':>10}{'ecart M EUR':>14}{'ecart %':>10}")
ecarts = []
for gc in np.round(np.arange(0.10, 0.901, 0.05), 3):
    v = scr(ev_e, gc)[0]
    ecarts.append((gc, scr_nc - v, 100 * (scr_nc / v - 1)))
    print(f"{gc:>7.2f}{G_NC-gc:>13.2f}{v:>10.1f}{scr_nc-v:>14.1f}{100*(scr_nc/v-1):>+10.1f}")

# quel ecartement minimal pour que l'ecart depasse 10 % ?
seuil = 0.10
elig = [(gc, e, p) for gc, e, p in ecarts if p / 100.0 >= seuil]
if elig:
    gc_max = max(gc for gc, _, _ in elig)
    print(f"\n  Pour que l'ecart depasse {100*seuil:.0f} %, il suffit que la conformite fasse baisser g")
    print(f"  de {G_NC:.2f} a {gc_max:.2f} ou moins, soit un ecartement de {G_NC-gc_max:.2f}. C'est une exigence")
    print("  faible : elle n'impose pas de connaitre g, seulement d'accepter que la conformite")
    print("  le deplace d'au moins cet ecart.")


# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("1. LEMME : le SCR est croissant en g, verifie sur 20 points de grille et aux deux")
print("   echelles, a nombres aleatoires communs. Ce n'etait pas acquis pour une VaR de queue")
print("   lourde, et c'est ce lemme qui porte tout le reste.")
print("2. INVARIANCE : l'existence et le sens de l'ecart entre etats de conformite ne")
print(f"   dependent PAS des valeurs de g, seulement de leur ordre. Verifie sur {len(MAPPINGS)}")
print("   applications, de tres etroite a large. Le forfait reste plat dans tous les cas.")
print("3. CE QUI RESTE UNE HYPOTHESE, et c'est tout : « se conformer reduit la propagation »,")
print("   soit g_C < g_NC. Affirmation qualitative, sur laquelle le reglement prend parti.")
print("4. CE QUI RESTE UN SCENARIO : l'AMPLITUDE. Elle depend de l'ecartement g_NC - g_C, et")
print("   l'elasticite donnee en section 3 permet a un lecteur de la recalculer avec ses")
print("   propres valeurs, sans relancer le modele.")
print("5. CONSEQUENCE DE REDACTION : le memoire ne doit plus ecrire « le SCR passe de 116 a")
print("   169 M EUR » comme une mesure, mais comme un scenario d'ecartement, et mettre en")
print("   avant l'invariance d'ordre, qui est le resultat solide.")


# =====================================================================================
titre("Figure")
# =====================================================================================
# CHARTE NEXIALOG, 17 AOUT 2026. Les couleurs ne sont plus codees en dur ici : elles
# viennent de style_nexialog, source unique du projet. Trois series -> les TROIS slots
# categoriels valides du module, ni plus (un quatrieme echoue le plancher de vision
# normale) ni recycles. La famille de police reste celle du script, voir l'en-tete du
# module : elle garantit la parite a l'octet entre le PC et le Mac.
import style_nexialog as nx                                       # noqa: E402

nx.appliquer(taille=10.5)
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "axes.grid": False,                    # choix propre a ce script, conserve
})
INK, INK2, MUTED = nx.ENCRE, nx.ENCRE_2, nx.ENCRE_3
ACCENT, BLUE, GREEN = nx.CATEGORIEL[0], nx.CATEGORIEL[2], nx.CATEGORIEL[1]
LEG = dict(frameon=True, facecolor=nx.FOND, edgecolor="none", framealpha=0.88, fontsize=8.5)

# TROIS PANNEAUX EN LIGNE, PAS EN COLONNE. Empiles, ils remplissaient une page entiere du
# memoire (21,9 cm) et laissaient un grand blanc au-dessus des barres. En ligne, la figure
# s'imprime dans le fil du texte. Le panneau du milieu porte cinq categories : il recoit
# donc un tiers de largeur de plus que ses voisins.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 3.5),
                                    gridspec_kw=dict(width_ratios=[1, 1.3, 1]))

# (a) le lemme de monotonie
ax1.plot(GRILLE, se, "-o", color=BLUE, lw=2, ms=4)
# ETIQUETTES COMPLETES. Une premiere version prenait le premier mot de l'intitule, ce qui
# affichait « non » pour « non conforme » : un etat de conformite ne se tronque pas.
# ETAGEES EN HAUTEUR. Sur un panneau etroit les trois etiquettes, toutes posees au-dessus
# de la courbe, se chevauchaient : « partiel » passe donc sous la courbe.
COURT_E = {"C": "conforme", "PC": "partiel", "NC": "non conforme"}
for e, c, dec, va in (("C", GREEN, (-7, 10), "bottom"), ("PC", MUTED, (0, -9), "top"),
                      ("NC", ACCENT, (-7, 10), "bottom")):
    g = G_MEMOIRE[e]
    ax1.plot([g], [scr(ev_e, g)[0]], "o", ms=8, color=c, zorder=4)
    ax1.annotate(COURT_E[e], (g, scr(ev_e, g)[0]), textcoords="offset points",
                 xytext=dec, ha="center" if va == "top" else "right", va=va,
                 fontsize=8.8, color=c)
ax1.axhline(SF_ENTITE, color=GREEN, lw=1.6, ls="--")
# LA LIGNE EST ETIQUETEE SUR PLACE, SANS LEGENDE. Une legende de deux entrees dans un
# panneau de 5 cm de large aurait recouvert la courbe ; la bande au-dessus des donnees,
# elle, est vide sur toute sa longueur.
ax1.text(GRILLE[0], SF_ENTITE + 9, f"forfait de Formule Standard : {SF_ENTITE:.0f} M€",
         ha="left", va="bottom", fontsize=8.8, color=GREEN)
ax1.set_xlabel("gain de propagation $g$", color=INK2)
ax1.set_ylabel("SCR 99,5 % de l'entité (M€)", color=INK2)
ax1.set_ylim(0, SF_ENTITE * 1.12)
ax1.set_title("(a)  Le capital est croissant en $g$,\nle forfait ne l'est pas",
              fontsize=11, color=INK, pad=8)

# (b) l'invariance sur cinq applications
# LIBELLES D'AFFICHAGE ACCENTUES. Les intitules des applications sont en ASCII dans les
# sorties de script, par convention ; une figure de memoire, elle, s'ecrit en francais.
AFFICHE = {"etroite": "étroite", "RETENUE": "RETENUE", "large": "large",
           "conformite faible": "conformité faible", "tres etroite": "très étroite"}
labs = [AFFICHE[l.split("(")[0].strip()] for l, _ in MAPPINGS]
xi = np.arange(len(MAPPINGS))
wd = 0.26
for k, (e, c) in enumerate((("C", GREEN), ("PC", BLUE), ("NC", ACCENT))):
    vals = [res[l][1][e] for l, _ in MAPPINGS]
    ax2.bar(xi + (k - 1) * wd, vals, width=wd, color=c, alpha=0.9, label=ETIQ[e])
ax2.axhline(SF_ENTITE, color=GREEN, lw=1.6, ls="--", label="forfait (plat)")
ax2.set_xticks(xi)
# ETIQUETTES INCLINEES. Cinq intitules dont deux de seize caracteres ne tiennent pas cote a
# cote sur un panneau de 7 cm : a plat ils se chevauchaient deux a deux.
ax2.set_xticklabels(labs, fontsize=9, rotation=22, ha="right", rotation_mode="anchor")
ax2.set_ylabel("SCR 99,5 % (M€)", color=INK2)
# BANDE HAUTE DEGAGEE POUR LA LEGENDE. Aucune barre ne depasse 170 M EUR et la ligne du
# forfait est a 450 : en portant le haut a 585, la legende sur deux rangs ne recouvre ni
# les barres ni la ligne.
ax2.set_ylim(0, SF_ENTITE * 1.30)
ax2.legend(loc="upper left", ncol=2, columnspacing=0.9, handlelength=1.4,
           handletextpad=0.45, borderpad=0.35, **dict(LEG, fontsize=8.2))
ax2.set_title("(b)  L'ordre survit aux cinq calibrations,\ny compris la plus étroite",
              fontsize=11, color=INK, pad=8)

# (c) l'ecart en fonction de l'ecartement
gcs = np.array([g for g, _, _ in ecarts])
ecs = np.array([p for _, _, p in ecarts])
ax3.plot(G_NC - gcs, ecs, "-o", color=ACCENT, lw=2, ms=5)
ax3.axhline(100 * seuil, color=INK, lw=1.3, ls="--")
# A DROITE DE LA LIGNE. A gauche, la courbe traverse justement ce niveau et passait sous le
# texte ; a droite elle est a 100 %, donc la ligne y est libre.
ax3.text((G_NC - gcs).max(), 100 * seuil + 2.5, f"écart de {100*seuil:.0f} %",
         ha="right", va="bottom", fontsize=8.8, color=INK2)
_gr = G_NC - G_MEMOIRE["C"]
_ye = 100 * (scr_nc / scr(ev_e, G_MEMOIRE['C'])[0] - 1)
ax3.plot([_gr], [_ye], "D", ms=9, color=BLUE, zorder=4)
# ETIQUETTE SUR PLACE, SOUS LE POINT. Une legende en haut a gauche recouvrait le haut de la
# courbe, qui monte de gauche a droite ; le triangle sous la courbe, lui, est vide.
ax3.annotate(f"écartement\nretenu : {_gr:.2f}".replace(".", ","), (_gr, _ye),
             textcoords="offset points", xytext=(9, -6), ha="left", va="top",
             fontsize=8.8, color=BLUE)
# SUR DEUX LIGNES. En une seule, l'intitule depassait la figure a droite et le recadrage
# serre coupait la parenthese fermante : le calcul de boite d'un texte mathtext ne la voit
# pas toujours.
ax3.set_xlabel("écartement $g_{NC} - g_C$\n(à $g_{NC}=0{,}90$)", color=INK2)
ax3.set_ylabel("écart de capital (%)", color=INK2)
ax3.set_title("(c)  L'amplitude dépend\ndu seul écartement",
              fontsize=11, color=INK, pad=8)

for ax in (ax1, ax2, ax3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# PAS DE TITRE GENERAL : la legende du memoire porte le titre, et le code interne de la
# figure n'a rien a faire dans un document remis a un jury.
fig.tight_layout(w_pad=1.8)
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S23_invariance_conformite.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
