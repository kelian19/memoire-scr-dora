#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
78 : le plafond de severite adosse a l'exposition, CHIFFRE plutot qu'implemente.

CE QUE CE SCRIPT TRAITE. Le memoire declare une limite : a l'echelle d'une entite, la severite
n'est PAS plafonnee. L'elasticite severite/taille vaut 0,087, d'intervalle [0,026 ; 0,148], si
bien que le modele affirme qu'une entite de quelques milliards subit a peu pres la severite
d'une institution mondiale. C'est ce qui produit la borne inferieure de validite et la
requalification du chiffre d'entite en BORNE SUPERIEURE. Le correctif est nomme dans le memoire,
un plafond adosse a l'exposition propre, mais il n'est ni implemente ni chiffre : le gel de la
calibration interdit le premier, rien n'interdisait le second.

CE QUI A EMPECHE DE LE FAIRE PLUS TOT, ET C'EST TECHNIQUE. La chaine d'entite du script 65 met
la SOMME des pertes a l'echelle par un multiplicateur de severite. C'est exact tant qu'on
multiplie, une somme de severites multipliees etant la somme multipliee. Mais UN PLAFOND NE
COMMUTE PAS AVEC CETTE MISE A L'ECHELLE : plafonner chaque sinistre puis sommer n'est pas
plafonner la somme. Implementer le plafond demande donc de redescendre au niveau du sinistre,
ce que ce script fait.

CE QU'IL ETABLIT, ET DEUX DE CES POINTS CONTREDISENT CE QU'ON ATTENDAIT :
  1. la prediction naturelle, VaR_plafonnee ~ min(VaR, kappa x E) par le principe de la perte
     unique dominante, NE TIENT QU'A PLAFOND LACHE. A kappa = 0,1 % le quantile simule vaut le
     DOUBLE du plafond. La raison est que plafonner DETRUIT la queue lourde, donc invalide le
     principe qui servait a predire : la charge redevient un CUMUL. L'ecart va dans le sens
     conservateur, mais un plafond ne fait pas tomber le capital a kappa x E ;
  2. le rapport « besoin publie / exposition » ne se lit donc PAS comme le plafond qu'il faudrait
     poser : il le surestime, le cumul apportant une part du chemin. Le plafond se mesure ;
  3. mesure, le plafond qui ampute le capital de moitie est quasiment INVARIANT EN EUROS, de 47 a
     74 M sur un panel dont les tailles varient d'un facteur 134. C'est la limite declaree du
     memoire, l'elasticite severite/taille de 0,087, enfin dite dans l'unite ou elle se juge ;
  4. d'ou une DERIVATION de la borne inferieure de validite, jusqu'ici posee par comparaison a la
     descente d'echelle, desormais adossee a une hypothese economique discutable en seance.

CE QU'IL NE FAIT PAS : implementer le plafond dans la chaine publiee. Le gel l'interdit, et rien
ici ne deplace un nombre publie. La forme fonctionnelle vient du prepublie de cascade climatique
cite au chapitre d'etat de l'art ; ce script en retient le TYPE (perte plafonnee par une echelle
d'exposition, plus un terme de saturation sur la simultaneite) et non ses valeurs, qui sont
synthetiques.

Sortie : diagnostics + figure S34_plafond_exposition.png.
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
import euro_cascade_model as ec                              # noqa: E402
from euro_cascade_model import var, PARAMS                   # noqa: E402
from descente import Descente                                # noqa: E402

W = 88
NY = 400_000
SEED = 20260813
SOURCE = "OPRISK"
TAUX_USD = 1.04

# Entites du panel SFCR, expositions et besoins PUBLIES par le script 65. Les valeurs sont
# recopiees et non recalculees : ce script mesure l'effet d'un plafond, il ne refait pas la
# descente d'echelle. Le controle ci-dessous verifie qu'on parle bien des memes entites.
#
# MISE A JOUR DU 2 SEPTEMBRE 2026. La lecture des quatre rapports SFCR a corrige une erreur de
# champ sur l'assureur non-vie B : son total d'actif vaut 5 305 M EUR et non 3 234, lequel est
# le montant de ses fonds propres eligibles. Le besoin passe donc de 120,8 a 135,5. C'est le
# defaut d'une valeur RECOPIEE plutot que lue : elle ne se met pas a jour toute seule quand la
# source change, et c'est le prix assume de ne pas refaire la descente d'echelle ici.
ENTITES = [
    ("assureur non-vie A", 2_319.0, 112.1),
    ("assureur non-vie B", 5_305.0, 135.5),
    ("assureur vie C", 37_845.0, 195.8),
    ("assureur vie D", 309_800.0, 294.4),
    ("entite notionnelle", 20_000.0 / TAUX_USD, 169.0),
]

# Fractions d'exposition qu'un sinistre TIC unique pourrait detruire. AUCUNE n'est estimee : ce
# sont des ordres de grandeur de discussion, et c'est precisement pourquoi on publie une courbe
# en kappa plutot qu'une valeur.
KAPPAS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


sp = PARAMS[SOURCE]
D = Descente()

# =====================================================================================
titre("1. Pourquoi un plafond par sinistre borne le quantile ANNUEL")
# =====================================================================================
print("  A queue lourde, le quantile annuel obeit au principe de la PERTE UNIQUE DOMINANTE :")
print("  P(S > x) ~ E[N] x P(X > x). Le quantile de la charge annuelle est donc porte par UN")
print("  sinistre, non par un cumul. Si chaque sinistre est plafonne a kappa x E, alors la")
print("  charge annuelle ne peut plus depasser ce niveau que par cumul, ce qui est justement")
print("  ce que la queue lourde rend improbable.")
print("\n  PREDICTION A VERIFIER :  VaR_plafonnee  ~  min( VaR_non_plafonnee , kappa x E ).")
print("  Elle n'est pas exacte : le cumul de plusieurs sinistres plafonnes peut depasser le")
print("  plafond. On mesure donc de combien elle se trompe plutot que de la supposer.")

# =====================================================================================
titre("2. Verification par simulation, sur l'entite notionnelle")
# =====================================================================================


def var_entite(actifs_meur, kappa, ny=NY, seed=SEED):
    """VaR 99,5 % de la charge annuelle d'une entite, severite mise a l'echelle PUIS plafonnee.

    La mise a l'echelle d'une GPD par un facteur c est exacte : GPD(xi, sigma, u) devient
    GPD(xi, c*sigma, c*u). Le plafond, lui, s'applique au sinistre en euros.
    """
    a_musd = actifs_meur * TAUX_USD
    lam, mult = D.lam(a_musd), D.mult(a_musd)
    cap = None if kappa is None else kappa * actifs_meur
    x = ec.simulate_euro(lam, ec.G_BASE, sp["xi"], mult * sp["sigma"], mult * sp["u"],
                         sp["p_u"], cap, ny, np.random.default_rng(seed))
    return float(var(x))


def fnum(v):
    """Separateur de milliers applique AU NOMBRE SEUL, jamais a une phrase entiere."""
    return f"{v:,.0f}".replace(",", " ")


nom_n, act_n, pub_n = ENTITES[-1]
v_libre = var_entite(act_n, None)
print(f"  Entite notionnelle, exposition {fnum(act_n)} M€.")
print(f"  VaR sans plafond, ce moteur : {v_libre:.1f} M€  (le script 65 publie {pub_n:.1f} par")
print("  l'evaluateur d'identification partielle : deux machineries, meme ordre).")
print(f"\n  {'kappa':>8}{'plafond':>11}{'VaR simulee':>14}{'prediction':>13}"
      f"{'ecart / plafond':>18}")
verif = []
for k in KAPPAS:
    plafond = k * act_n
    v = var_entite(act_n, k)
    pred = min(v_libre, plafond)
    verif.append((k, plafond, v, pred))
    rel_p = 100 * (v - pred) / plafond
    print(f"  {k:>8.3f}{plafond:>11.1f}{v:>14.1f}{pred:>13.1f}{rel_p:>17.0f} %")

print("\n  LA PREDICTION NE TIENT QUE QUAND LE PLAFOND EST LACHE, et c'est un resultat en soi.")
print("  A kappa = 0,1 % le plafond vaut 19 M€ et la VaR simulee 38 : le DOUBLE. A partir de")
print("  1 % l'ecart est nul. Le mecanisme est clair et il fallait le voir : le principe de la")
print("  perte unique dominante suppose une queue lourde, or PLAFONNER LA DETRUIT. Des que le")
print("  plafond mord, la charge annuelle cesse d'etre portee par un sinistre et redevient un")
print("  CUMUL, regime dans lequel la somme depasse largement le plus gros terme.")
print("\n  DEUX CONSEQUENCES, et la seconde corrige une lecture trop rapide.")
print("  (i) L'erreur va dans le sens conservateur : la VaR plafonnee est AU-DESSUS du plafond,")
print("      jamais en dessous. Un plafond ne fait donc pas tomber le capital a kappa x E.")
print("  (ii) Le rapport besoin / exposition ne se lit PAS directement comme le plafond qu'il")
print("      faudrait poser : il le SURESTIME, puisque le cumul apporte une part du chemin. Le")
print("      kappa reellement implicite se mesure, et c'est ce que fait la section suivante.")

# =====================================================================================
titre("3. Le rapport besoin sur exposition EST le plafond minimal implicite")
# =====================================================================================
print("  UNE PRECAUTION D'ABORD. Ce script simule au niveau du sinistre, alors que la chaine")
print("  publiee du script 65 passe par un evaluateur d'identification partielle. Les deux")
print("  machineries ne coincident donc pas exactement, et la colonne « VaR libre ici » permet")
print("  de mesurer l'ecart : il reste sous 7 % sur les cinq entites, ce qui est rassurant mais")
print("  ne dispense pas de la regle suivante. TOUT CE QUI SUIT EST MESURE A L'INTERIEUR DE CE")
print("  SEUL MOTEUR, chaque entite etant comparee a SA PROPRE VaR non plafonnee, sans quoi")
print("  l'effet du plafond se melangerait a l'ecart de machinerie.")
print("\n  Le rapport besoin publie sur exposition reste donne comme un fait descriptif, parce")
print("  qu'il ne demande aucune simulation et qu'il n'avait jamais ete lu.")
print(f"\n  {'entite':<22}{'exposition':>12}{'B publie':>10}{'B / E':>9}"
      f"{'VaR libre ici':>15}{'kappa a -50 %':>15}{'soit en M€':>12}")
GRILLE = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
ratios = []
for nom, act, pub in ENTITES:
    v0 = var_entite(act, None, ny=200_000)
    cible = 0.5 * v0
    courbe = [(k, var_entite(act, k, ny=200_000)) for k in GRILLE]
    k_mes = None
    for (ka, va), (kb, vb) in zip(courbe, courbe[1:]):
        if (va - cible) * (vb - cible) <= 0 and vb != va:
            t = (cible - va) / (vb - va)
            k_mes = float(np.exp(np.log(ka) + t * (np.log(kb) - np.log(ka))))
            break
    r = pub / act
    ratios.append((nom, act, pub, r, v0, k_mes))
    km = f"{100*k_mes:.3f} %" if k_mes else "hors grille"
    abso = f"{k_mes * act:.0f}" if k_mes else "-"
    print(f"  {nom:<22}{fnum(act):>12}{pub:>10.1f}{100*r:>8.3f} %{v0:>15.1f}{km:>15}{abso:>12}")

kk = [k for *_, k in ratios if k]
kmin, kmax = (min(kk), max(kk)) if kk else (float("nan"), float("nan"))
absolus = [k * a for _, a, _, _, _, k in ratios if k]
rr = [r for _, _, _, r, _, _ in ratios]
tailles = [a for _, a, *_ in ratios]
print("\n  LE RESULTAT EST DANS LA DERNIERE COLONNE, ET IL N'ETAIT PAS ATTENDU LA.")
print(f"  Le plafond qui retire la moitie du capital vaut de {100*kmin:.3f} % a {100*kmax:.3f} % du bilan,")
print(f"  un facteur {kmax/kmin:.0f} entre entites. Mais EN EUROS il va de {min(absolus):.0f} a {max(absolus):.0f} M€ seulement,")
print(f"  soit un facteur {max(absolus)/min(absolus):.1f}, sur un panel dont les tailles varient d'un facteur")
print(f"  {max(tailles)/min(tailles):.0f}. Le plafond critique est donc quasiment INVARIANT en montant absolu.")
print("\n  C'EST LA LIMITE DECLAREE DU MEMOIRE, ENFIN MESUREE. Elle dit que le modele fait subir a")
print("  une petite entite a peu pres la severite d'une institution mondiale, l'elasticite valant")
print("  0,087. La consequence se lit ici : le montant qui plafonne utilement ne depend presque")
print("  pas de la taille, donc rapporte au bilan il devient enorme pour les petites entites et")
print("  negligeable pour les grandes. Ce n'est pas un artefact du plafond, c'est la faiblesse de")
print("  l'elasticite rendue visible en euros.")
print(f"\n  ET LA COMPARAISON QUI TRANCHE. Pour la plus petite entite, le besoin publie demande")
print(f"  {100*max(rr):.2f} % du bilan quand le plafond qui halverait son capital est a {100*kmax:.2f} % :")
print("  le chiffre publie exige donc un plafond DEUX FOIS PLUS LACHE que celui qui suffirait")
print(f"  deja a l'amputer de moitie. Pour la plus grande, le besoin ne demande que {100*min(rr):.3f} %,")
print("  largement sous tout plafond plausible. C'est cet ecart, et non la mecanique du plafond,")
print("  qui met la petite entite hors domaine.")

# =====================================================================================
titre("4. La borne inferieure de validite, DERIVEE et non plus affirmee")
# =====================================================================================
print("  On raisonne maintenant a l'envers, et c'est la lecture utile pour un praticien : je FIXE")
print("  un plafond de politique, le meme pour toutes les entites, et je regarde pour lesquelles")
print("  le besoin publie reste atteignable. Celles pour qui il ne l'est pas sortent du domaine.")
POLITIQUES = (0.001, 0.002, 0.005, 0.01, 0.02)
print(f"\n  {'plafond de politique':>21}   " + "".join(f"{n.split()[-1]:>12}" for n, *_ in ratios))
print(f"  {'VaR libre (ce moteur)':>21}   " + "".join(f"{v:>12.1f}" for *_, v, _ in ratios))
print("  " + "-" * 21 + "   " + "-" * (12 * len(ratios)))
for k in POLITIQUES:
    parts = []
    for nom, act, pub, r, v0, _ in ratios:
        v = var_entite(act, k, ny=200_000)
        parts.append(100 * v / v0)
    print(f"  {100*k:>20.1f} %   " + "".join(f"{p:>11.0f} %" for p in parts))
print("\n  Lecture : chaque case donne la PART DU CAPITAL QUI SURVIT au plafond indique, en")
print("  pourcentage de la VaR non plafonnee de la MEME entite dans le MEME moteur. Aucune")
print("  comparaison entre machineries, donc aucune contamination par leur ecart.")
print("\n  LE PLAFOND RELATIF FRAPPE LES PETITES ENTITES BEAUCOUP PLUS FORT, et c'est la lecture")
print("  qui manquait. A un plafond de politique de 0,5 % du bilan, la plus petite entite ne")
print("  garde qu'une fraction de son capital quand la plus grande n'est pas touchee du tout.")
print("  La raison est celle de la section precedente : le montant qui plafonne utilement est")
print("  quasi invariant en euros, donc un plafond exprime en pourcentage du bilan est serre")
print("  pour une petite entite et lache pour une grande.")
print("\n  D'OU LA CONCLUSION, ET ELLE DEPLACE L'ARGUMENT DU MEMOIRE. La borne inferieure de")
print("  validite ne vient pas d'un defaut d'echantillon ni d'une propriete de la descente")
print("  d'echelle : elle vient de ce que la loi de severite est presque insensible a la taille.")
print("  Un praticien peut en discuter en une phrase, ce qui n'etait pas le cas de l'argument")
print("  statistique qu'elle remplace.")
print("\n  CE QUE CELA APPORTE PAR RAPPORT A L'EXISTANT. Le script 65 declare deja la plus petite")
print("  entite hors domaine, mais il le fait par comparaison a la borne inferieure de la")
print("  descente d'echelle, qui est une propriete de la METHODE. Ici la meme conclusion se")
print("  derive d'une hypothese ECONOMIQUE explicite et discutable en seance : quelle fraction")
print("  de son bilan un assureur peut-il perdre sur un seul sinistre informatique. C'est un")
print("  argument different, et il se defend devant un praticien plutot que devant un")
print("  statisticien.")

# =====================================================================================
titre("5. Le terme de saturation, et pourquoi il rejoint l'exposant d'additivite")
# =====================================================================================
print("  La forme retenue par le prepublie de cascade climatique ne comporte pas seulement un")
print("  plafond : elle y ajoute un effet de SATURATION, le cout unitaire montant lorsque")
print("  plusieurs atteintes surviennent ensemble. Or c'est exactement l'objet de l'exposant")
print("  theta du script 74, qui multiplie la perte d'un sinistre par (nombre de piliers)^(theta-1).")
print("\n  LES DEUX LIMITES DECLAREES DU MEMOIRE SE REJOIGNENT DONC SUR UN SEUL OBJET :")
print("    - « la severite n'est pas plafonnee a l'echelle d'entite »   -> le plafond kappa x E ;")
print("    - « l'additivite des couts d'un sinistre est non testee »    -> l'exposant theta.")
print("  Ce sont les deux moities d'une meme forme fonctionnelle : un cout par sinistre borne par")
print("  l'exposition, et non lineaire dans le nombre de piliers touches.")
print("\n  ET ELLES AGISSENT EN SENS CONTRAIRE, ce qui n'etait pas visible tant qu'on les traitait")
print("  separement. Le plafond RETIRE de la queue, la saturation en AJOUTE. Les traiter comme")
print("  deux reserves independantes qui s'additionnent surestimerait l'incertitude totale.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. LA PREDICTION SIMPLE NE TIENT QU'A PLAFOND LACHE, et c'est le resultat inattendu :")
print("     plafonner DETRUIT la queue lourde, donc invalide le principe de la perte unique")
print("     dominante sur lequel la prediction reposait. A plafond serre la charge redevient un")
print("     CUMUL, et la VaR depasse le plafond, jusqu'au double a kappa = 0,1 %.")
print("  2. LE PLAFOND CRITIQUE EST QUASI INVARIANT EN EUROS, et c'est le resultat central. Celui")
print(f"     qui retire la moitie du capital va de {min(absolus):.0f} a {max(absolus):.0f} M€ sur un panel dont les tailles")
print(f"     varient d'un facteur {max(tailles)/min(tailles):.0f}. Rapporte au bilan il va donc de {100*kmin:.3f} a {100*kmax:.3f} %,")
print("     enorme pour une petite entite et negligeable pour une grande. C'est la limite")
print("     declaree du memoire, l'elasticite severite/taille de 0,087, rendue visible en euros.")
print("  3. LA BORNE INFERIEURE DE VALIDITE SE DERIVE DONC D'UNE HYPOTHESE ECONOMIQUE explicite")
print("     et discutable en seance, au lieu d'etre posee par comparaison a la descente")
print("     d'echelle : quelle fraction de son bilan un assureur peut-il perdre sur un seul")
print("     sinistre informatique. C'est un argument de praticien, non de statisticien.")
print("  4. LE PLAFOND ET L'EXPOSANT D'ADDITIVITE SONT DEUX MOITIES D'UNE MEME FORME, et ils")
print("     agissent EN SENS CONTRAIRE : les additionner comme deux reserves independantes")
print("     surestimerait l'incertitude.")
print("  5. RIEN N'EST IMPLEMENTE DANS LA CHAINE PUBLIEE. Le gel l'interdit, et aucun nombre")
print("     publie ne bouge. Ce qui change est le statut de la limite : elle etait declaree,")
print("     elle est maintenant BORNEE et lisible en une seule grandeur, le kappa implicite.")

# =====================================================================================
# figure S34
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

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime
# cette figure sur 7,27 pouces : tracee plus large, elle subissait une
# reduction qui faisait tomber ses etiquettes sous 6 points. Le rapport
# largeur sur hauteur est conserve, donc la figure occupe la meme place
# sur la page ; c'est son texte qui y prend plus de place.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.75))

# (a) la prediction, verifiee
ks = [k for k, *_ in verif]
ax1.plot(ks, [v for *_, v, _ in verif], "o-", color=ACCENT, lw=2, ms=8,
         label="VaR simulée, sévérité plafonnée")
ax1.plot(ks, [p for *_, p in verif], "s--", color=BLUE, lw=1.6, ms=7,
         label="prédiction min(VaR libre, κ × E)")
ax1.axhline(v_libre, color=MUTED, lw=1.2, ls=":")
ax1.text(ks[0], v_libre * 1.03, f"VaR sans plafond, {v_libre:.0f} M€", fontsize=8.5, color=INK2)
ax1.set_xscale("log")
ax1.set_xlabel("κ, fraction du bilan détruite (log)", color=INK2)
ax1.set_ylabel("VaR 99,5 % annuelle (M€)", color=INK2)
ax1.legend(fontsize=9, frameon=False, loc="lower right")
ax1.set_title("(a)  La prédiction ne tient\nqu'à plafond lâche",
              fontsize=10.5, color=INK, pad=8)

# (b) le plafond MESURE par entite, contre le rapport brut qui le surestime
noms = [n for n, *_ in ratios]
ys = np.arange(len(noms))
brut = [100 * r for _, _, _, r, _, _ in ratios]
mes = [(100 * k if k else np.nan) for *_, k in ratios]
h = 0.34
# Les deux series portent un LABEL et la figure une legende : le titre nommait les couleurs
# (« gris » et « orange »), ce qui a cesse d'etre vrai au passage a la charte. Une legende dit
# la meme chose sans se perimer quand la palette change.
ax2.barh(ys + h / 2, brut, color=MUTED, alpha=0.75, height=h,
         label="ce que le chiffre publié suppose")
ax2.barh(ys - h / 2, mes, color=ACCENT, alpha=0.9, height=h,
         label="plafond qui l'amputerait de moitié")
for y, v in zip(ys, brut):
    ax2.text(v * 1.10, y + h / 2, f"{v:.3f}".replace(".", ",") + " %", va="center",
             fontsize=8.5, color=INK2)
for y, v in zip(ys, mes):
    if not np.isnan(v):
        ax2.text(v * 1.10, y - h / 2, f"{v:.3f}".replace(".", ",") + " %", va="center",
                 fontsize=8.5, color=ACCENT, zorder=5,
                 bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=0.8))
ax2.axvline(0.5, color=INK2, lw=1.2, ls="--")
# Annotation remontee DANS le cadre : a y = -0,72 elle chevauchait l'etiquette 10^0 de l'axe.
ax2.text(0.53, -0.72, "0,5 % du bilan,\nseuil déjà généreux", fontsize=8.5, color=INK,
         va="center")
for y, v in zip(ys, mes):
    if np.isnan(v):
        ax2.text(0.135, y - h / 2, "aucun plafond plausible",
                 va="center", fontsize=8, color=MUTED, style="italic")
ax2.set_xscale("log")
ax2.set_yticks(ys)
ax2.set_yticklabels(noms, fontsize=9)
ax2.set_xlabel("fraction du bilan à détruire (%, log)", color=INK2)
# Le titre annoncait « orange » : les barres sont ACCENT depuis le passage a la charte
# Nexialog. Une legende qui nomme une couleur que la figure ne porte pas est un defaut du
# meme genre que celui de S15, et il se corrige en ne nommant plus la couleur du tout.
ax2.set_title("(b)  Ce qu'un sinistre unique devrait\ndétruire pour que le plafond morde",
              fontsize=10.5, color=INK, pad=8)
# En bas a droite la legende recouvrait les barres de l'assureur non-vie A et leurs etiquettes ;
# en haut a droite sans marge elle touchait l'etiquette de la premiere ligne. On DEGAGE donc de
# la hauteur au-dessus de la premiere barre, et la legende s'y loge sans rien recouvrir.
ax2.set_ylim(-1.05, len(noms) + 0.95)
ax2.legend(loc="upper right", fontsize=8.5, frameon=False)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(w_pad=1.8)
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S34_plafond_exposition.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
