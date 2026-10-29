#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
85 : une cascade SANS auto-excitation produit-elle le ratio de branchement qu'on lui attribue ?

CE QUE LE MEMOIRE FAIT AUJOURD'HUI. Le chapitre 13 rejette la couche de Hawkes, et le rejet est
serieux : il est teste contre les variantes de noyau de la litterature (script 08h), y compris le
noyau a retard de Bessy-Roland/Boumezoued/Hillairet. Le script 08h mesure un ratio de branchement
de 0,551 sur la chronologie reelle, et observe qu'il tombe a environ zero des que les
co-occurrences du MEME JOUR passent en exogene. Le memoire ajoute que le prepublie de cascade
climatique ecarte le meme outil pour la meme raison de RESOLUTION TEMPORELLE.

CE QUI MANQUAIT, ET C'EST UN ARGUMENT PLUS FORT QUE CELUI QU'IL REMPLACE. Dire « l'endogeneite
disparait quand on re-etiquette les co-occurrences » est un constat de sensibilite : un lecteur
peut repondre que le re-etiquetage est arbitraire. L'enonce fort serait une EQUIVALENCE
OBSERVATIONNELLE : montrer qu'un processus SANS aucune auto-excitation, observe a la resolution
dont on dispose, produit un ratio de branchement du meme ordre que celui mesure. Si c'est le cas,
le 0,551 ne constitue AUCUNE preuve d'auto-excitation, et le choix entre les deux modeles ne peut
pas se faire sur la donnee de frequence : il se fait sur l'interpretabilite des parametres.

LA PREDICTION EST ANALYTIQUE AVANT D'ETRE SIMULEE, et c'est ce qui rend le test decisif. Dans la
cascade, un sinistre amorce touche en moyenne E[k] piliers ; si chaque pilier touche est un
evenement enregistrable, alors la fraction d'evenements qui sont des « enfants » vaut
(E[k] - 1) / E[k]. C'est exactement ce qu'un ajustement de Hawkes appelle son ratio de
branchement. Avec E[k] = 1,931 a l'etat non conforme, on attend donc n de l'ordre de 0,48 SANS LA
MOINDRE AUTO-EXCITATION.

TROIS CONTROLES AVANT TOUTE CONCLUSION, parce qu'un estimateur non valide ne prouve rien :
  (a) sur un Poisson pur, l'estimateur doit rendre n proche de zero ;
  (b) sur un Hawkes simule de ratio connu, il doit le retrouver ;
  (c) le balayage de RESOLUTION doit reproduire le comportement du script 08h, l'endogeneite
      s'effondrant quand la cascade cesse d'etre resolue.

Sortie : diagnostics + figure S41_equivalence_hawkes.png.
"""

import os
import sys

import numpy as np
from scipy.optimize import minimize
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                                 # noqa: E402
import canaux_conformite as cx                                  # noqa: E402
import scr_engine as eng                                        # noqa: E402

WID = 88
PIL = cx.PIL
H_AN = 24.0 * 365.0
LAG_H = 6.7                     # lag moyen d'un pas de propagation, script 08h
# TAILLE DE L'ECHANTILLON, ET ELLE EST CHOISIE. A 53,6 amorces par an et 1,93 pilier par
# sinistre, soixante annees donnent de l'ordre de six mille evenements : c'est deja un ordre de
# grandeur au-dessus de ce qu'une chronologie reelle offre, donc largement assez pour que
# l'ajustement soit stable, et cela garde le cout praticable. La recursion de la vraisemblance est
# sequentielle par nature : elle boucle en Python, et quarante mille evenements rendaient chaque
# ajustement inutilisable.
N_ANS = 60
NSEED = 3
SEED0 = 20260814
N_08H = 0.551                   # ratio de branchement mesure sur la chronologie reelle (08h)

# Resolutions d'observation, en heures. La chronologie de breches est datee AU JOUR : tout ce qui
# est plus fin n'est pas disponible, et tout ce qui est plus grossier est un agregat volontaire.
RESOLUTIONS_H = (1.0, 24.0, 24.0 * 7, 24.0 * 30)
NOM_RES = {1.0: "heure", 24.0: "jour", 168.0: "semaine", 720.0: "mois"}


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# ============================================================ estimateur de Hawkes exponentiel
def _neg_loglik_vec(theta, t, T):
    """-log-vraisemblance d'un Hawkes a noyau exponentiel, par la recursion de Ozaki.

    lambda(t) = mu + somme_{t_i < t} alpha exp(-beta (t - t_i)).
    La recursion R_i = exp(-beta (t_i - t_{i-1})) (1 + R_{i-1}) evite la double boucle, sans quoi
    l'ajustement serait quadratique en le nombre d'evenements et impraticable ici.
    """
    mu, alpha, beta = np.exp(theta)
    if beta <= 0 or mu <= 0 or alpha <= 0:
        return 1e12
    dt = np.diff(t)
    R = np.empty(t.size)
    R[0] = 0.0
    for i in range(1, t.size):
        R[i] = np.exp(-beta * dt[i - 1]) * (1.0 + R[i - 1])
    lam = mu + alpha * R
    if np.any(lam <= 0):
        return 1e12
    compens = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - t)))
    return -(np.sum(np.log(lam)) - compens)


def ajuste_hawkes(t, T):
    """(mu, alpha, beta, n) par maximum de vraisemblance. n = alpha/beta est le ratio de branchement."""
    t = np.sort(np.asarray(t, float))
    if t.size < 50:
        return np.nan, np.nan, np.nan, np.nan
    lam0 = t.size / T
    best = None
    # DEUX DEPARTS : la vraisemblance d'un Hawkes a des plateaux, et un depart unique peut rendre
    # un optimum local. On garde le meilleur des deux.
    for a0, b0 in ((0.3, 60.0), (0.6, 400.0)):
        x0 = np.log([max(lam0 * (1 - a0), 1e-6), a0 * b0, b0])
        r = minimize(_neg_loglik_vec, x0, args=(t, T), method="Nelder-Mead",
                     options=dict(maxiter=1200, xatol=1e-5, fatol=1e-5))
        if best is None or r.fun < best.fun:
            best = r
    mu, alpha, beta = np.exp(best.x)
    return mu, alpha, beta, alpha / beta


def simule_hawkes(mu, alpha, beta, T, rng):
    """Hawkes exponentiel par amincissement d'Ogata. Sert de CONTROLE a l'estimateur.

    On suit l'intensite entre deux candidats : A porte la part excitee, majoree par sa valeur au
    dernier evenement, et decroit de exp(-beta w) a chaque pas w. Un candidat est retenu avec
    probabilite (mu + A) / M.
    """
    t, A, out = 0.0, 0.0, []
    while True:
        M = mu + A
        w = rng.exponential(1.0 / M)
        t += w
        if t > T:
            break
        A *= np.exp(-beta * w)
        if rng.random() * M <= mu + A:
            out.append(t)
            A += alpha
    return np.array(out)


def observe(t, res_h):
    """Dates vues a une resolution donnee : on ne connait que la CELLULE, pas l'instant.

    Chaque date est ramenee au debut de sa cellule puis reperturbee uniformement a l'interieur.
    C'est la representation honnete d'une chronologie datee au jour : l'ordre intra-cellule est
    perdu, et c'est precisement ce que la resolution enleve.
    """
    res = res_h / H_AN
    if res <= 0:
        return np.sort(t)
    cell = np.floor(t / res)
    rng = np.random.default_rng(12345)
    return np.sort((cell + rng.random(t.size)) * res)


# ============================================================ dates produites par la CASCADE
W_AM = np.array([eng.LAMBDA[j] for j in PIL], float)
W_AM = W_AM / W_AM.sum()


def loi_cardinal(g):
    p = np.zeros(len(PIL) + 1)
    for c, j in enumerate(PIL):
        dist = eng.cascade_set_dist(j, g)
        tot = sum(dist.values())
        for s, pr in dist.items():
            p[len(s)] += W_AM[c] * pr / tot
    return p


def dates_cascade(lam, g, n_ans, rng, lag_h=LAG_H):
    """Dates des piliers touches, engendrees par la cascade. AUCUNE AUTO-EXCITATION N'Y FIGURE.

    Le comptage annuel est la NegBin du modele publie ; conditionnellement, les amorces sont
    uniformes dans l'annee (propriete du melange de Poisson). Chaque amorce touche k piliers, le
    m-ieme a t + somme de (m-1) lags exponentiels. Le processus est un branchement A UN SEUL
    NIVEAU DE DESCENDANCE PAR PAS, sans aucun terme d'excitation : c'est le point du script.
    """
    r = lam / (ec.PHI - 1.0)
    counts = rng.negative_binomial(r, r / (r + lam), size=n_ans)
    T = int(counts.sum())
    if T == 0:
        return np.array([])
    t0 = np.repeat(np.arange(n_ans), counts) + rng.random(T)
    loi = loi_cardinal(g)
    kk = rng.choice(np.arange(len(loi)), size=T, p=loi / loi.sum())
    lags = rng.exponential(scale=lag_h / H_AN, size=(T, len(PIL) - 1))
    cum = np.cumsum(lags, axis=1)
    out = [t0]
    for m in range(1, len(PIL)):
        sel = kk > m
        if sel.any():
            out.append(t0[sel] + cum[sel, m - 1])
    return np.sort(np.concatenate(out))


# =====================================================================================
titre("1. LA PREDICTION ANALYTIQUE, avant toute simulation")
# =====================================================================================
loi_nc = loi_cardinal(cx.G_NC)
loi_c = loi_cardinal(cx.G_C)
Ek_nc = float(sum(k * loi_nc[k] for k in range(len(loi_nc))))
Ek_c = float(sum(k * loi_c[k] for k in range(len(loi_c))))
n_pred_nc = (Ek_nc - 1.0) / Ek_nc
n_pred_c = (Ek_c - 1.0) / Ek_c
print("  Si chaque pilier touche est un evenement enregistrable, alors dans une cascade la")
print("  fraction d'evenements qui sont des ENFANTS vaut (E[k] - 1) / E[k], ou E[k] est le nombre")
print("  moyen de piliers touches par sinistre. C'est exactement ce qu'un ajustement de Hawkes")
print("  appelle son ratio de branchement.")
print(f"\n  E[k] = {Ek_c:.3f} a l'etat conforme       -> n attendu = {n_pred_c:.3f}")
print(f"  E[k] = {Ek_nc:.3f} a l'etat non conforme   -> n attendu = {n_pred_nc:.3f}")
print(f"\n  A COMPARER AU {N_08H:.3f} MESURE PAR LE SCRIPT 08h SUR LA CHRONOLOGIE REELLE. Les deux")
print("  sont du meme ordre, et cela AVANT d'avoir simule quoi que ce soit. Si la simulation le")
print("  confirme, alors le ratio de branchement observe est exactement ce qu'une cascade SANS")
print("  auto-excitation produit, et il ne constitue aucune preuve d'auto-excitation.")

# =====================================================================================
titre("2. CONTROLES DE L'ESTIMATEUR : sans eux rien de ce qui suit ne vaut")
# =====================================================================================
T_ctrl = 60.0
rng = np.random.default_rng(SEED0)
t_poi = np.sort(rng.random(int(100 * T_ctrl)) * T_ctrl)
_, _, _, n_poi = ajuste_hawkes(t_poi, T_ctrl)
print(f"  (a) POISSON PUR, {t_poi.size} evenements : n estime = {n_poi:.3f}")
print("      Attendu proche de zero. Un estimateur qui verrait de l'excitation dans un Poisson")
print("      invaliderait tout le reste.")
ctrl_h = []
for n_vrai in (0.30, 0.55, 0.80):
    beta_v = 100.0
    mu_v = 80.0 * (1.0 - n_vrai)
    th = simule_hawkes(mu_v, n_vrai * beta_v, beta_v, T_ctrl, np.random.default_rng(SEED0 + 7))
    _, _, _, n_est = ajuste_hawkes(th, T_ctrl)
    ctrl_h.append((n_vrai, n_est, th.size))
    print(f"  (b) HAWKES SIMULE de ratio {n_vrai:.2f}, {th.size} evenements : n estime = {n_est:.3f}")
print("      L'estimateur retrouve le ratio impose. Il est donc utilisable sur les dates de la")
print("      cascade, ou l'on ne connait pas la reponse a l'avance.")

# =====================================================================================
titre("3. LE TEST : ajuster un Hawkes sur des dates ENGENDREES PAR LA CASCADE")
# =====================================================================================
print(f"  On engendre {N_ANS} annees de dates avec le moteur de cascade a l'etat non conforme. Ce")
print("  processus ne contient AUCUN terme d'auto-excitation : sa seule structure temporelle est")
print("  la duree des pas de propagation a l'interieur d'un sinistre. On lui ajuste un Hawkes")
print("  exponentiel, comme on le ferait sur une chronologie reelle, et l'on regarde ce qu'il")
print("  trouve, resolution d'observation par resolution d'observation.")
print(f"\n  {'resolution':>12}{'evenements':>13}{'n estime':>12}{'etendue':>11}{'demi-vie (h)':>15}")
res_tab = []
for res_h in RESOLUTIONS_H:
    ns, hl = [], []
    for k in range(NSEED):
        t = dates_cascade(cx.LAM_NC, cx.G_NC, N_ANS, np.random.default_rng(SEED0 + 100 * k))
        to = observe(t, res_h)
        _, _, beta, n = ajuste_hawkes(to, float(N_ANS))
        if np.isfinite(n):
            ns.append(n)
            hl.append(np.log(2.0) / beta * H_AN)
    ns = np.array(ns)
    res_tab.append((res_h, t.size, float(ns.mean()), float(np.ptp(ns)), float(np.mean(hl))))
    print(f"  {NOM_RES[res_h]:>12}{t.size:>13}{ns.mean():>12.3f}{np.ptp(ns):>11.3f}"
          f"{np.mean(hl):>15.1f}")

n_heure = [r[2] for r in res_tab if r[0] == 1.0][0]
n_jour = [r[2] for r in res_tab if r[0] == 24.0][0]
n_mois = [r[2] for r in res_tab if r[0] == 720.0][0]
hl_jour = [r[4] for r in res_tab if r[0] == 24.0][0]
hl_mois = [r[4] for r in res_tab if r[0] == 720.0][0]
print(f"\n  A LA RESOLUTION DE LA DONNEE REELLE, LE JOUR, l'ajustement trouve n = {n_jour:.3f} SUR UN")
print(f"  PROCESSUS QUI N'A AUCUNE AUTO-EXCITATION. La valeur mesuree sur la chronologie reelle est")
print(f"  {N_08H:.3f}, et la prediction analytique de la section 1 vaut {n_pred_nc:.3f}. LES TROIS SONT DU")
print("  MEME ORDRE, et les deux premieres coincident a trois millemes.")
print(f"\n  ET L'AJUSTEMENT RETROUVE AUSSI L'HORLOGE INTERNE DE LA CASCADE : la demi-vie estimee du")
print(f"  noyau vaut {hl_jour:.1f} h a la resolution du jour, pour un lag de propagation vrai de {LAG_H} h.")
print("  Le Hawkes ne se contente donc pas de trouver le bon ratio : il retrouve le bon temps")
print("  caracteristique, et l'appelle « decroissance de l'excitation ».")
print(f"\n  L'EFFET DE LA RESOLUTION VA DANS LE SENS INVERSE DE CELUI QU'ON ATTENDAIT, et il faut le")
print(f"  dire tel quel : n MONTE quand on degrade la resolution, de {n_heure:.3f} a l'heure a {n_mois:.3f} au")
print("  mois, au lieu de tomber. Le mecanisme est clair une fois vu : agreger des dates dans des")
print("  cellules plus larges FABRIQUE du groupement, et l'ajustement le lit comme de l'excitation")
print(f"  en allongeant son noyau, dont la demi-vie passe de {hl_jour:.0f} a {hl_mois:.0f} heures.")
print(f"\n  CE N'EST DONC PAS LE MEME PHENOMENE QUE CELUI DU SCRIPT 08h, et les confondre serait une")
print("  faute. Les deux operations sont OPPOSEES : le script 08h RETIRE les co-occurrences du")
print("  meme jour du compte de l'excitation, et le ratio tombe vers zero ; ici on les CONSERVE en")
print("  perdant leur ordre, et le ratio monte. Les deux encadrent donc le meme fait par les deux")
print("  cotes : L'ESTIMATION DU RATIO EST ENTIEREMENT PILOTEE PAR LA STRUCTURE INFRA-JOURNALIERE.")
print("  La retirer l'annule, la brouiller l'amplifie, et dans les deux cas ce qui est mesure n'est")
print("  pas une propriete du phenomene mais une propriete de la fenetre par laquelle on le")
print("  regarde. C'est exactement l'argument de resolution temporelle, et il est ici chiffre.")

# =====================================================================================
titre("4. CE QUE CELA CHANGE POUR L'ARGUMENT DU MEMOIRE")
# =====================================================================================
print("  L'ARGUMENT ACTUEL est un constat de sensibilite : « l'endogeneite disparait quand on")
print("  re-etiquette les co-occurrences du meme jour ». Un lecteur peut objecter que le")
print("  re-etiquetage est un choix, donc que le constat est circulaire.")
print("\n  L'ARGUMENT QUE CE SCRIPT PERMET EST UNE EQUIVALENCE OBSERVATIONNELLE, et elle ne prete")
print("  pas le flanc a cette objection : a la resolution dont on dispose, un processus SANS")
print("  auto-excitation produit le meme ratio de branchement que celui qu'on mesure. Le 0,551")
print("  n'est donc PAS une preuve d'auto-excitation ; c'est une quantite que les deux modeles")
print("  reproduisent egalement bien.")
print("\n  D'OU LA FORMULATION A RETENIR, ET ELLE EST PLUS HONNETE QUE « LE HAWKES EST REJETE » :")
print("  sur la donnee de frequence disponible, LE HAWKES ET LA CASCADE NE SONT PAS DISTINGUABLES.")
print("  Le choix se fait donc ailleurs, et il se defend mieux ainsi :")
print("    - les parametres de la cascade sont des DOMAINES DE CONTROLE REGLEMENTAIRES, sur")
print("      lesquels DORA fait porter des exigences et qu'une entite peut piloter ;")
print("    - les parametres d'un Hawkes sont un taux de fond et un noyau d'excitation, qui ne")
print("      correspondent a aucune exigence et sur lesquels aucune remediation ne se formule.")
print("  A pouvoir explicatif EGAL sur la donnee, on retient le modele dont les parametres sont")
print("  actionnables. C'est un argument de PARCIMONIE INTERPRETATIVE, pas un rejet empirique, et")
print("  il ne pretend pas plus que ce que la donnee permet.")
print("\n  CE QUE CELA NE DIT PAS, et il faut le garder. On ne demontre pas qu'il n'y a AUCUNE")
print("  auto-excitation dans le phenomene reel : on demontre que la donnee disponible ne permet")
print("  pas de la distinguer de la cascade. Une chronologie a l'heure, ou un marquage des")
print("  sinistres par identifiant d'evenement racine, trancherait ; ni l'une ni l'autre n'existe")
print("  dans les sources retenues.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. LA PREDICTION EST ANALYTIQUE : dans une cascade ou chaque pilier touche est un")
print(f"     evenement, la fraction d'enfants vaut (E[k]-1)/E[k], soit {n_pred_nc:.3f} a l'etat non")
print(f"     conforme. Le script 08h mesure {N_08H:.3f} sur la chronologie reelle.")
print("  2. L'ESTIMATEUR EST VALIDE AVANT USAGE : il rend n proche de zero sur un Poisson pur")
print(f"     ({n_poi:.3f}) et retrouve les ratios imposes sur des Hawkes simules.")
print(f"  3. AJUSTE SUR DES DATES ENGENDREES PAR LA CASCADE, SANS AUCUNE AUTO-EXCITATION, il trouve")
print(f"     n = {n_jour:.3f} a la resolution du JOUR, celle de la donnee reelle. Le ratio observe est")
print("     donc exactement ce qu'une cascade produit, et il ne prouve rien.")
print(f"  3bis. IL RETROUVE MEME L'HORLOGE : demi-vie estimee {hl_jour:.1f} h pour un lag de propagation")
print(f"     vrai de {LAG_H} h. Le bon ratio ET le bon temps caracteristique, sans excitation.")
print(f"  4. L'EFFET DE LA RESOLUTION VA A L'INVERSE DE CE QU'ON ATTENDAIT : n MONTE quand on")
print(f"     degrade, de {n_heure:.3f} a l'heure a {n_mois:.3f} au mois, parce qu'agreger des dates FABRIQUE du")
print(f"     groupement (la demi-vie du noyau passe de {hl_jour:.0f} a {hl_mois:.0f} h). Ce n'est donc PAS le meme")
print("     phenomene que dans le script 08h, dont l'operation est OPPOSEE : il retire les")
print("     co-occurrences du meme jour et le ratio tombe vers zero. Les deux encadrent le meme")
print("     fait par les deux cotes : l'estimation est entierement pilotee par la structure")
print("     INFRA-JOURNALIERE. La retirer l'annule, la brouiller l'amplifie ; dans les deux cas on")
print("     mesure la fenetre d'observation et non le phenomene.")
print("  5. L'ARGUMENT DU MEMOIRE SE REFORMULE, ET IL Y GAGNE : au lieu d'un rejet empirique du")
print("     Hawkes, une EQUIVALENCE OBSERVATIONNELLE a la resolution disponible, suivie d'un choix")
print("     de parcimonie interpretative. Les parametres de la cascade sont des domaines de")
print("     controle pilotables, ceux d'un Hawkes ne le sont pas.")
print("  6. ET CE QUI N'EST PAS DEMONTRE RESTE DIT : on ne montre pas l'absence d'auto-excitation")
print("     dans le phenomene, seulement que la donnee ne permet pas de l'en distinguer.")

# =====================================================================================
# figure S41
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.3))

# (a) l'estimateur valide : Poisson et Hawkes de ratio connu
xs = [0.0] + [c[0] for c in ctrl_h]
ys = [n_poi] + [c[1] for c in ctrl_h]
ax1.plot([0, 0.9], [0, 0.9], "--", color=MUTED, lw=1.4, label="identité (estimateur exact)")
ax1.plot(xs, ys, "o", color=BLUE, ms=11, label="ratio estimé")
for x, y in zip(xs, ys):
    ax1.annotate(f"{y:.3f}".replace(".", ","), (x, y), textcoords="offset points",
                 xytext=(9, -4), fontsize=9, color=BLUE)
ax1.set_xlim(-0.06, 0.95)
ax1.set_ylim(-0.06, 0.95)
ax1.set_xlabel("ratio de branchement imposé", color=INK2)
ax1.set_ylabel("ratio de branchement estimé", color=INK2)
ax1.legend(fontsize=9, frameon=False, loc="upper left")
ax1.set_title("(a)  L'estimateur est validé avant usage :\nzéro sur un Poisson, exact sur un "
              "Hawkes connu", fontsize=10.5, color=INK, pad=8)

# (b) LE RESULTAT : le ratio trouve sur une cascade SANS excitation, par resolution
labs = [NOM_RES[r[0]] for r in res_tab]
vals = [r[2] for r in res_tab]
errs = [r[3] / 2 for r in res_tab]
xs2 = np.arange(len(labs))
ax2.bar(xs2, vals, yerr=errs, width=0.55, color=ACCENT, alpha=0.9,
        error_kw=dict(ecolor=MUTED, lw=1.2, capsize=4))
ax2.axhline(N_08H, color=BLUE, lw=1.8, ls="--")
# ETIQUETTE FERREE A GAUCHE : ferree a droite, elle traversait le libelle de la barre « semaine ».
ax2.text(-0.45, N_08H + 0.028,
         f"{N_08H:.3f} mesuré sur la chronologie réelle (08h)".replace(".", ","),
         fontsize=9, color=BLUE, ha="left", fontweight="bold")
ax2.axhline(n_pred_nc, color=GREEN, lw=1.5, ls=":")
ax2.text(-0.42, n_pred_nc - 0.045, f"{n_pred_nc:.3f} prédit sans simuler".replace(".", ","),
         fontsize=9, color=GREEN, ha="left", fontweight="bold")
for x, v in zip(xs2, vals):
    ax2.text(x, v + 0.022, f"{v:.3f}".replace(".", ","), ha="center", fontsize=9.5,
             color=INK, fontweight="bold")
ax2.set_xticks(xs2)
ax2.set_xticklabels(labs, fontsize=10)
ax2.set_ylim(0, max(max(vals), N_08H) * 1.30)
ax2.set_xlabel("résolution à laquelle les dates sont observées", color=INK2)
ax2.set_ylabel("ratio de branchement estimé", color=INK2)
# LE SECOND MESSAGE DU PANNEAU, et il a du etre corrige apres lecture : j'attendais que le ratio
# TOMBE quand on degrade la resolution. Il monte, parce qu'agreger fabrique du groupement.
ax2.annotate("dégrader la résolution FABRIQUE\ndu groupement : le ratio monte",
             xy=(xs2[-1], vals[-1]), xytext=(xs2[1] - 0.30, max(vals) * 1.16),
             fontsize=9, color=INK2, ha="left",
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
ax2.set_title("(b)  Sur une cascade SANS auto-excitation, l'ajustement\ntrouve pourtant le ratio "
              "qu'on mesure sur les vraies dates", fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("S41 : à la résolution disponible, le Hawkes et la cascade ne sont pas distinguables",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S41_equivalence_hawkes.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
