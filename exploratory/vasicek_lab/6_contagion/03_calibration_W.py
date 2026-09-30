#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03 : Protocole de calibration de la matrice de contagion dirigee W.

Objectif : montrer COMMENT on estimera W sur des donnees reelles, et VALIDER la
methode sur des donnees simulees dont on connait la verite. Pret a brancher sur
un vrai panel d'incidents (entite x pilier x periode) le jour venu.

Portee : ce script VALIDE L'ESTIMATEUR (on simule depuis le modele, on le
retrouve), il ne prouve PAS que le modele decrit le monde. Et l'asymetrie n'est
identifiable ici que parce que la simulation fournit un ordre temporel propre ;
sur les donnees reelles disponibles elle ne l'est pas (cf. script 05).

Idee cle : l'ASYMETRIE de W (qui entraine qui) n'est identifiable que par
l'information TEMPORELLE (qui declenche AVANT qui). La co-occurrence transversale
seule ne donne qu'une dependance symetrique. Le modele est donc dynamique :

  latent  X[i,j,t] = base_j + somme_k W[j,k] * D[i,k,t-1] + s_j * Y[j,t] + eps
  incident D[i,j,t] = 1 si X[i,j,t] >= 0            (X = STRESS, pas sante)

  D[i,k,t-1] : le pilier k a-t-il eu un incident a la periode precedente
  W[j,k]     : force dirigee de k vers j (ce qu'on veut estimer)
  Y[j,t]     : facteur systemique du pilier j a la periode t (choc commun)

Estimateur : PROBIT par pilier. Le modele generateur pose eps gaussien et
D = 1{X >= 0} : c'est LITTERALEMENT un probit. Une regression logistique estimerait
les memes coefficients gonfles d'environ un facteur 2 (unites logit), d'ou une
correlation correcte mais des MAGNITUDES fausses. Le probit rend W dans ses
propres unites. Effets fixes de periode pour absorber le systemique Y[j,t].

NORMALISATION DE W : la lecon de Leontief. Dans le modele entrees-sorties, la matrice
technique A a pour entree A[i,j] = x_ij / x_j : c'est une matrice de PARTS. Ses colonnes
somment a moins de 1 parce que la valeur ajoutee est strictement positive. C'est
exactement cela qui garantit rho(A) < 1, donc l'existence de l'inverse de Leontief
B = (I-A)^-1 = somme des A^k (Pineau & Zuniga 2023, Remarque 1).

TRANS n'est pas une matrice de parts : ses lignes somment jusqu'a 2,3 et
rho(TRANS) = 1,461 > 1. La brancher telle quelle dans (I-W)^-1 est une ERREUR DE
CATEGORIE, pas un probleme de calibration. On applique donc la meme discipline :

  W = g * TRANS / max_k( EMISSION du pilier k )

  g            = part de contagion du pilier LE PLUS PROLIFIQUE, dans (0, 1]
  contrainte   = decomposition de variance de Vasicek : contagion + systemique + idio = 1

LE DIVISEUR EST L'EMISSION MAXIMALE, ET C'EST LA CONVENTION DU PROJET : la ligne emet, la
colonne recoit. Une version anterieure de ce script divisait par la RECEPTION maximale (2,3),
ce qui donnait rho(W) = 0,572 et un g critique de 1,57, en desaccord avec les 0,506 et 1,78
publies par le chapitre cascade. Ce qui tranche n'est pas une preference : la normalisation
existe pour garantir que le pilier le plus prolifique engendre au plus g descendants directs,
e_k = g * s_k / c avec s_k son EMISSION. Avec c = 2,3 on obtenait e_P1 = 1,02, donc PLUS que
g : la borne etait franchie et g cessait de designer ce qu'il designe. Avec c = 2,60 on a
e_P1 = 0,90 exactement. La contrainte de reception reste satisfaite au passage, la reception
maximale (2,3) etant sous l'emission maximale.

Alors rho(W) <= g < 1 : la STABILITE EST GARANTIE PAR LA DECOMPOSITION DE VARIANCE, comme la
valeur ajoutee la garantit chez Leontief. Ici rho(W) = 0,562 * g. L'asymetrie est preservee
d'un bout a l'autre : on ne normalise pas ligne a ligne, on divise par un scalaire unique.

Lecture epidemique. Le modele dynamique est un PROCESSUS DE BRANCHEMENT multitype.
La matrice de generation suivante M donne le nombre attendu d'incidents de type j
engendres par un incident de type k :

  M[j,k] = Phi(base_j + W[j,k]) - Phi(base_j)        (effet marginal, pas W lui-meme)

  R0 = rayon spectral de M           cascade s'eteint ssi R0 < 1
  (I - M)^-1 = progeniture totale attendue (nombre de descendants par source)

ATTENTION : rho(W) et R0 = rho(M) sont deux objets differents. rho(W) gouverne le
systeme latent SIMULTANE (celui qui donne (I-W)^-1) ; R0 gouverne la cascade
d'INCIDENTS. Sur le domaine admissible g <= 1, LES DEUX sont stables.

Sortie : diagnostics + figures K1 (verite vs estimation), K2 (taille d'echantillon,
sous-declaration), K3 (R0 et progeniture).
"""

import os
import warnings
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt

# Avertissements de convergence / overflow attendus des probits sur cellules peu
# peuplees ; les echecs reels sont comptes dans estimate_W, on les masque donc ici.
warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260708)
J = 5
PIL = ["P1", "P2", "P3", "P4", "P5"]

# structure dirigee de reference (celle de TRANS : "k entraine j")
TRANS = {
    1: {2: 0.80, 3: 0.70, 4: 0.70, 5: 0.40},
    2: {1: 0.20, 3: 0.40, 4: 0.30, 5: 0.60},
    3: {1: 0.30, 2: 0.50, 4: 0.30, 5: 0.30},
    4: {1: 0.20, 2: 0.80, 3: 0.30, 5: 0.40},
    5: {1: 0.10, 2: 0.20, 3: 0.20, 4: 0.20},
}
# ROOT : "qui amorce une cascade", jugement d'expert du chantier qualitatif.
# On le RETROUVE ici comme progeniture du branchement : il est donc redondant.
ROOT = {1: 1.00, 4: 0.90, 2: 0.60, 3: 0.50, 5: 0.30}

T_RAW = np.zeros((J, J))              # TRANS brut, ligne j = ce que j RECOIT
for k in TRANS:                       # k -> j
    for j, v in TRANS[k].items():
        T_RAW[j - 1, k - 1] = v
RECEPTION = T_RAW.sum(1)              # ligne de T_RAW = ce que le pilier RECOIT
EMISSION = T_RAW.sum(0)               # colonne de T_RAW = ce que le pilier EMET
ROWSUM = RECEPTION                    # conserve pour la lecture d'exposition, plus bas
MAXROW = RECEPTION.max()              # 2,3 : le pilier le plus EXPOSE (P2)
# LE DIVISEUR DE LEONTIEF EST L'EMISSION MAXIMALE, ET C'EST UNE CORRECTION.
# Ce script divisait par MAXROW, la RECEPTION maximale (2,3), quand tout le reste du pipeline
# divise par l'EMISSION maximale (2,60). Le rayon spectral etant invariant par transposition,
# seul le diviseur separait les deux lectures, et elles coexistaient dans une meme phrase du
# chapitre cascade : rho(W) = 0,506 et g critique = 1,78 venaient de la convention d'emission,
# R0 = 0,062 de celle-ci.
#
# CE QUI TRANCHE N'EST PAS UNE PREFERENCE. La normalisation existe pour garantir que le pilier
# le plus prolifique engendre au plus g descendants directs, e_j = g * s_j / c avec s_j son
# EMISSION. Avec c = 2,3, le plus prolifique donne e_P1 = 0,9 x 2,60 / 2,3 = 1,02 descendants,
# donc PLUS que g : la borne que la normalisation est censee poser est franchie, et g cesse de
# vouloir dire « part de contagion du pilier le plus expose ». Avec c = 2,60 on a e_P1 = 0,90
# exactement, soit g. La contrainte de reception reste satisfaite au passage, la reception
# maximale (2,3) etant inferieure a l'emission maximale.
MAXEM = EMISSION.max()                # 2,60 : le pilier le plus PROLIFIQUE (P1)
DIVISEUR = MAXEM
RHO_RAW = float(max(abs(np.linalg.eigvals(T_RAW))))   # 1,461 : INADMISSIBLE

GAIN = 0.9                            # g : part de contagion du pilier le plus expose
W_TRUE = GAIN * T_RAW / DIVISEUR      # normalisation de Leontief : matrice de PARTS
BASE = np.full(J, -1.7)               # cale l'incidence de base : Phi(-1,7) = 4,5 % et NON 7 %,
#                                       comme le disait ce commentaire. Le chapitre cascade
#                                       publie bien 4,5 % ; c'est ici que la valeur etait fausse,
#                                       et elle n'etait imprimee nulle part, donc invisible.
SLOAD = 0.5                           # charge du facteur systemique


def simulate(N, T, rng, w=W_TRUE, under=0.0):
    """Panel d'incidents D (N,J,T). under = proba de NON-declaration d'un petit incident."""
    D = np.zeros((N, J, T), dtype=np.int8)
    D[:, :, 0] = (rng.random((N, J)) < 0.08).astype(np.int8)
    for t in range(1, T):
        Y = rng.standard_normal(J)                        # systemique par pilier
        contagion = D[:, :, t - 1] @ w.T                  # (N,J) : somme_k W[j,k] D[i,k,t-1]
        X = BASE + contagion + SLOAD * Y + rng.standard_normal((N, J))
        D[:, :, t] = (X >= 0).astype(np.int8)
    if under > 0:                                         # sous-declaration (troncature)
        keep = rng.random((N, J, T)) >= under
        D = (D & keep).astype(np.int8)
    return D


def estimate_W(D, link="probit"):
    """W_hat (J,J, diagonale nulle) et base_hat, par regression par pilier.

    link='probit' : le lien du modele generateur -> W_hat dans les unites de W.
    link='logit'  : conserve pour montrer le biais d'echelle (~ x1,6).
    """
    N, _, T = D.shape
    # ATTENTION. D[:, j, 1:] est de forme (N, T-1) ; reshape(-1) l'aplatit en ordre
    # LIGNE PAR LIGNE : l'observation m est (entite m//(T-1), periode m%(T-1)).
    # La periode de chaque observation est donc np.tile, PAS np.repeat. Avec np.repeat
    # les indicatrices sont affectees aux mauvaises observations, n'absorbent rien, et
    # le systemique Y reste dans l'erreur : le probit est alors ATTENUE d'un facteur
    # 1/sqrt(1 + s^2).
    per_idx = np.tile(np.arange(T - 1), N)                # periode de chaque obs
    per_oh = np.eye(T - 1)[per_idx]                       # effets fixes de periode
    W_hat = np.zeros((J, J))
    base_hat = np.zeros(J)
    n_skip = 0                                            # piliers non estimes (degeneres / non convergents)
    for j in range(J):
        y = D[:, j, 1:].reshape(-1)                       # reponse (N*(T-1))
        others = [k for k in range(J) if k != j]
        lag = np.column_stack([D[:, k, :-1].reshape(-1) for k in others])
        Xd = np.column_stack([lag, per_oh])               # retards + effets periode
        if y.min() == y.max():                            # pilier degenere (aucune variation)
            n_skip += 1
            continue
        model = sm.Probit(y, Xd) if link == "probit" else sm.Logit(y, Xd)
        try:
            res = model.fit(disp=0, method="bfgs", maxiter=500)
        except Exception as exc:                          # non convergence : on compte, on ne masque pas
            n_skip += 1
            print(f"    [estimate_W] pilier {j} non estime : {type(exc).__name__}")
            continue
        for c, k in enumerate(others):
            W_hat[j, k] = res.params[c]
        base_hat[j] = res.params[len(others):].mean()     # niveau moyen des effets periode
    if n_skip:
        print(f"    [estimate_W] {n_skip}/{J} piliers non estimes (laisses a zero)")
    return W_hat, base_hat


def next_generation(w, base):
    """M[j,k] = nb attendu d'incidents en j engendres par un incident en k.

    Effet marginal sur la PROBABILITE, pas le coefficient latent : c'est M, et non W,
    qui gouverne l'extinction de la cascade.
    """
    M = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j != k:
                M[j, k] = stats.norm.cdf(base[j] + w[j, k]) - stats.norm.cdf(base[j])
    return M


def R0(M):
    return float(max(abs(np.linalg.eigvals(M))))


def total_progeny(M):
    """(I-M)^-1 : colonne k = descendants attendus d'un incident initial en k."""
    return np.linalg.inv(np.eye(J) - M)


def offdiag(M):
    return M[~np.eye(J, dtype=bool)]


def direction_recovery(w_true, w_hat):
    """% de paires dont le SENS de l'asymetrie (j->k vs k->j) est bien retrouve."""
    ok = tot = 0
    for a in range(J):
        for b in range(a + 1, J):
            dt = w_true[a, b] - w_true[b, a]
            dh = w_hat[a, b] - w_hat[b, a]
            if abs(dt) < 1e-9:
                continue
            ok += (np.sign(dt) == np.sign(dh))
            tot += 1
    return ok / tot


def slope(w_true, w_hat):
    """Pente de la regression W_hat ~ W_true : 1 = bonnes unites, 1,6 = unites logit."""
    return float(np.polyfit(offdiag(w_true), offdiag(w_hat), 1)[0])


# ============================================================ run principal
N0, T0 = 800, 60
D0 = simulate(N0, T0, RNG)
inc = D0.mean()
Wp, bp = estimate_W(D0, "probit")
Wl, _ = estimate_W(D0, "logit")

corr_p = np.corrcoef(offdiag(W_TRUE), offdiag(Wp))[0, 1]
dir_p = direction_recovery(W_TRUE, Wp)
sl_p, sl_l = slope(W_TRUE, Wp), slope(W_TRUE, Wl)

print(f"Panel : {N0} entites x {T0} periodes x {J} piliers ; incidence moyenne {inc:.1%}\n")
print("Recuperation de W (hors diagonale) :")
print(f"  probit : correlation {corr_p:.3f} | pente {sl_p:.3f} | sens de l'asymetrie {dir_p:.0%}")
print(f"  logit  : correlation {np.corrcoef(offdiag(W_TRUE), offdiag(Wl))[0,1]:.3f} | pente {sl_l:.3f}")
print(f"  (pente ~1 = W dans ses propres unites ; {sl_l:.2f} en unites logit, soit "
      f"{100*(sl_l/sl_p-1):.0f} % de trop)\n")

# --------------------------------------------------- normalisation et stabilite
rho_W = float(max(abs(np.linalg.eigvals(W_TRUE))))
print("Normalisation de W (la lecon de Leontief) :")
# LES DEUX VECTEURS SONT IMPRIMES, ET ETIQUETES. La version anterieure n'en imprimait qu'un,
# sous le nom neutre de « sommes des lignes » : comme ce script travaille sur TRANS transposee,
# c'etait la RECEPTION, de maximum 2,3, affichee deux lignes au-dessus de l'annonce d'un diviseur
# de 2,6 presente comme l'emission maximale. Le lecteur qui verifiait la coherence des deux
# trouvait 2,3 contre 2,6 et concluait a une contradiction qui n'existait pas.
print(f"  TRANS brut, RECEPTION par pilier (ce que chacun recoit) {np.round(ROWSUM, 2)}, "
      f"max {ROWSUM.max():.1f}")
print(f"  TRANS brut, EMISSION par pilier  (ce que chacun envoie) {np.round(EMISSION, 2)}, "
      f"max {EMISSION.max():.1f}  <- le diviseur")
print(f"  rho de TRANS brut = {RHO_RAW:.3f} (invariant par transposition)")
print("     -> ce n'est PAS une matrice de parts. (I-TRANS)^-1 aurait 21 entrees")
print("        negatives sur 25 : erreur de categorie, pas de calibration.")
print(f"  W = g * TRANS / {DIVISEUR:.1f}  avec g = {GAIN} ; le diviseur est l'EMISSION maximale")
print(f"     parts recues s_j = {np.round(W_TRUE.sum(1), 3)}  (<= g : asymetrie preservee)")
print(f"     rho(W) = {rho_W:.3f} = {RHO_RAW/DIVISEUR:.3f} x g   -> stable pour tout g <= 1")
print(f"  incidence de fond Phi(base) = {100*stats.norm.cdf(BASE[0]):.1f} %  (base = {BASE[0]})")
# DIVISEUR DE LEONTIEF : TRANCHE LE 7 AOUT 2026, C'EST L'EMISSION MAXIMALE (2,60). NE PAS
# ROUVRIR. Ce bloc etait, jusqu'au 12 aout, redige au present et decrivait l'etat ANTERIEUR a
# cette decision : il annoncait que « ce script divise par la RECEPTION maximale, 2,3 » et que la
# question restait « a trancher ». Les deux affirmations etaient fausses depuis l'alignement,
# comme la sortie de ce script l'imprime elle-meme quelques lignes plus bas. Un commentaire
# perime a cet endroit est plus dangereux qu'ailleurs : c'est le seul point du projet ou une
# confusion de convention a deja coute des jours, et un lecteur qui suit le commentaire plutot
# que le code rouvre une decision close.
#
# CE QUI EST FIXE. Ce script transpose TRANS (ligne j = ce que j RECOIT, cf. T_RAW plus haut) ;
# le reste du pipeline garde la convention du projet, la ligne emet et la colonne recoit. Le
# rayon spectral etant invariant par transposition, seul le DIVISEUR separait les deux lectures :
#     / 2,3   -> rapport 0,635 ; rho(W) = 0,572 ; rho(W) = 1 a g = 1,57   (ABANDONNE)
#     / 2,60  -> rapport 0,562 ; rho(W) = 0,506 ; rho(W) = 1 a g = 1,78   (RETENU)
# Le chapitre cascade publie 0,562 / 0,506 / 1,78, et R0 = 0,054 depuis la correction du 7 aout.
# L'argument qui tranche n'est pas un gout de normalisation : la normalisation de Leontief existe
# pour garantir que le pilier le plus prolifique engendre AU PLUS g descendants directs, et avec
# le diviseur de reception il en engendrait 1,02 pour g = 0,9, donc plus que g. Il est reimprime
# ci-dessous a chaque execution pour qu'il ne puisse plus se perdre dans un commentaire.
print(f"\n  DIVISEUR DE NORMALISATION : l'EMISSION maximale ({DIVISEUR:.1f}), convention du projet,")
print(f"  la ligne emet et la colonne recoit. Une version anterieure de ce script divisait par")
print(f"  la RECEPTION maximale ({MAXROW:.1f}), ce qui donnait rho(W) = 0.572 et un g critique de")
print(f"  1.57, en desaccord avec les 0.506 et 1.78 publies par le chapitre cascade.")
print(f"  CE QUI TRANCHE. La normalisation garantit que le pilier le plus prolifique engendre au")
print(f"  plus g descendants directs. Avec l'ancien diviseur il en engendrait "
      f"{GAIN*EMISSION.max()/MAXROW:.2f}, donc PLUS")
print(f"  que g = {GAIN} : la borne etait franchie et g cessait de designer ce qu'il designe.")
print(f"  Avec celui-ci il en engendre exactement {GAIN*EMISSION.max()/DIVISEUR:.2f}. La contrainte de "
      f"reception reste")
print(f"  satisfaite au passage, la reception maximale ({MAXROW:.1f}) etant sous l'emission "
      f"maximale.\n")

# --------------------------------------------------- lecture epidemique
M_true = next_generation(W_TRUE, BASE)
M_hat = next_generation(Wp, bp)
R0_t, R0_h = R0(M_true), R0(M_hat)
P_true = total_progeny(M_true)
prog_true = P_true.sum(0) - 1.0                # descendants attendus par pilier source

print("Lecture epidemique (processus de branchement multitype) :")
print(f"  rho(W), systeme latent SIMULTANE = {rho_W:.3f}  (stable)")
print(f"  R0 = rho(M), cascade d'INCIDENTS = {R0_t:.3f}  "
      f"{'sous-critique : la cascade s eteint' if R0_t < 1 else 'SUR-CRITIQUE'}")
print(f"  R0 estime sur le panel simule    = {R0_h:.3f}\n")

print("Progeniture totale (I-M)^-1 : descendants attendus d'un incident initial")
order_prog = np.argsort(-prog_true)
order_root = np.argsort(-np.array([ROOT[j] for j in range(1, J + 1)]))
for j in range(J):
    print(f"  {PIL[j]} : {prog_true[j]:.3f} descendants   (ROOT expert = {ROOT[j+1]:.2f})")
rho_s = stats.spearmanr([ROOT[j] for j in range(1, J + 1)], prog_true).statistic
print(f"\n  classement progeniture : {' > '.join(PIL[i] for i in order_prog)}")
print(f"  classement ROOT expert : {' > '.join(PIL[i] for i in order_root)}")
print(f"  correlation de rang (Spearman) = {rho_s:.3f}")
print("  => ROOT se deduit de TRANS par coherence interne (Spearman=1.00) : reduction de parametres, non validation externe.\n")

# --------------------------------------------------- R0 en fonction du gain
# g est BORNE par la decomposition de variance : g <= 1. On prolonge au-dela pour
# localiser les seuils critiques et montrer qu'ils sont HORS du domaine admissible.
gains = np.linspace(0.02, 9.0, 120)
Wg = lambda g: g * T_RAW / DIVISEUR
R0_by_gain = np.array([R0(next_generation(Wg(g), BASE)) for g in gains])
rho_by_gain = np.array([float(max(abs(np.linalg.eigvals(Wg(g))))) for g in gains])
g_crit_rho = np.interp(1.0, rho_by_gain, gains)
g_crit_R0 = np.interp(1.0, R0_by_gain, gains)
print("Seuils critiques, dans la parametrisation admissible g dans (0, 1] :")
print(f"  rho(W) = 1 a g = {g_crit_rho:.2f}   (systeme latent simultane)  HORS DOMAINE")
print(f"  R0     = 1 a g = {g_crit_R0:.1f}    (cascade d'incidents)       HORS DOMAINE")
print(f"  ratio {g_crit_R0/g_crit_rho:.0f}x : si l'on forcait le modele hors de son domaine,")
print("  la forme simultanee crierait a l'emballement bien avant la cascade reelle.")
print(f"  A g = 1 (contagion maximale admissible) : rho(W) = {RHO_RAW/DIVISEUR:.3f}, "
      f"R0 = {R0(next_generation(Wg(1.0), BASE)):.3f}")
print("  => sur tout le domaine admissible, les DEUX lectures sont stables.\n")

# ============================================================ courbe taille d'echantillon
sizes = [50, 100, 200, 400, 800]
REP, TSW = 6, 50
corr_by_n, dir_by_n = [], []
for N in sizes:
    cs, ds = [], []
    for _ in range(REP):
        D = simulate(N, TSW, RNG)
        Wh, _ = estimate_W(D)
        cs.append(np.corrcoef(offdiag(W_TRUE), offdiag(Wh))[0, 1])
        ds.append(direction_recovery(W_TRUE, Wh))
    corr_by_n.append(np.mean(cs)); dir_by_n.append(np.mean(ds))
    print(f"  N={N:4d} : corr {np.mean(cs):.2f} ; sens asymetrie {np.mean(ds):.0%}")

# ============================================================ robustesse sous-declaration
unders = [0.0, 0.2, 0.4, 0.6]
dir_by_u, corr_by_u = [], []
for u in unders:
    cs, ds = [], []
    for _ in range(REP):
        D = simulate(600, TSW, RNG, under=u)
        Wh, _ = estimate_W(D)
        cs.append(np.corrcoef(offdiag(W_TRUE), offdiag(Wh))[0, 1])
        ds.append(direction_recovery(W_TRUE, Wh))
    corr_by_u.append(np.mean(cs)); dir_by_u.append(np.mean(ds))
    print(f"  sous-declaration {u:.0%} : corr {np.mean(cs):.2f} ; sens {np.mean(ds):.0%}")

# ============================================================ tests de falsification
def placebo_permute(D, rng):
    """Permute le temps DANS chaque entite : toute asymetrie doit s'effondrer."""
    Dp = D.copy()
    N, _, T = D.shape
    for i in range(N):
        Dp[i] = D[i][:, rng.permutation(T)]
    return Dp


def reverse_time(D):
    return D[:, :, ::-1].copy()


Dpl = placebo_permute(D0, RNG)
Wpl, _ = estimate_W(Dpl)
Drev = reverse_time(D0)
Wrev, _ = estimate_W(Drev)


def asym_strength(w):
    return float(np.abs(w - w.T).sum() / 2)


print("\nTests de falsification (l'asymetrie doit disparaitre / s'inverser) :")
print(f"  asymetrie sur le panel reel     = {asym_strength(Wp):.3f}")
print(f"  asymetrie apres permutation     = {asym_strength(Wpl):.3f}  (doit tendre vers 0)")
print(f"  asymetrie en temps inverse      = {asym_strength(Wrev):.3f}")
corr_rev = np.corrcoef(offdiag(Wp), offdiag(Wrev))[0, 1]
print(f"  correlation W_avant vs W_inverse = {corr_rev:.3f}  "
      f"(proche de 1 => on capte de la co-occurrence, pas de la direction)")

# ============================================================ figures
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2, MUTED, GRID = "#1b1e30", "#223e55", "#595959", "#f2f2f2"
ACCENT = "#a6002e"
BLUES = ["#cde2fb", "#6491e1", "#4c79c7", "#2b559f", "#204993", "#09327a"]
SEQ = mpl.colors.LinearSegmentedColormap.from_list("b", BLUES)
outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(outdir, exist_ok=True)

# --- figure K1 : verite vs estimation (probit contre logit)
fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6),
                         gridspec_kw={"width_ratios": [1, 1, 1.15]})


def heat(ax, M, title):
    vmax = max(M.max(), 1e-6)
    ax.imshow(M, cmap=SEQ, vmin=0, vmax=vmax)
    for a in range(J):
        for b in range(J):
            if a != b and M[a, b] > 1e-6:
                ax.text(b, a, f"{M[a,b]:.2f}", ha="center", va="center",
                        fontsize=8, color="#fff" if M[a, b] > 0.55 * vmax else INK)
    ax.set_xticks(range(J)); ax.set_xticklabels(PIL)
    ax.set_yticks(range(J)); ax.set_yticklabels(PIL)
    ax.set_xlabel("de ce pilier k", color=INK2, fontsize=9)
    ax.set_ylabel("vers ce pilier j", color=INK2, fontsize=9)
    ax.set_title(title, fontsize=10.5, color=INK, pad=8)


heat(axes[0], W_TRUE, "W vrai (simule)")
heat(axes[1], np.clip(Wp, 0, None), "W estime (probit)")
ax = axes[2]
xt = offdiag(W_TRUE)
ax.scatter(xt, offdiag(Wp), s=44, color=BLUES[3], edgecolor="#fcfcfb",
           zorder=3, label=f"probit (pente {sl_p:.2f})")
ax.scatter(xt, offdiag(Wl), s=30, color=ACCENT, alpha=0.65, marker="^",
           zorder=2, label=f"logit (pente {sl_l:.2f})")
lim = np.linspace(0, max(xt.max(), 0.75), 10)
ax.plot(lim, lim, color=MUTED, lw=1.2, ls="--", zorder=1, label="identite")
ax.set_xlabel("W vrai (hors diagonale)", color=INK2, fontsize=9)
ax.set_ylabel("W estime", color=INK2, fontsize=9)
ax.set_title(f"Probit : bonnes unites (corr {corr_p:.2f} ; sens {dir_p:.0%})",
             fontsize=10.5, color=INK, pad=8)
ax.grid(True, color=GRID, lw=0.7)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout(rect=[0, 0, 1, 0.94])
p1 = os.path.join(outdir, "K1_calibration_W.png")
fig.savefig(p1, dpi=200, bbox_inches="tight"); print("\nfigure ecrite :", p1)

# --- figure K2 : taille d'echantillon et sous-declaration
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
ax1.plot(sizes, corr_by_n, "-o", color=BLUES[4], lw=2, label="correlation W")
ax1.plot(sizes, dir_by_n, "-s", color=ACCENT, lw=2, label="sens de l'asymetrie")
ax1.set_xscale("log")
ax1.set_xticks(sizes); ax1.set_xticklabels([str(s) for s in sizes])
ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax1.tick_params(axis="x", which="minor", bottom=False)
ax1.set_xlabel("nombre d'entites N (x 50 periodes)", color=INK2)
ax1.set_ylabel("qualite de recuperation", color=INK2)
ax1.set_ylim(0, 1.05); ax1.grid(True, color=GRID, lw=0.7)
ax1.legend(frameon=False, fontsize=9, loc="lower right")
ax1.set_title("(a) Combien de donnees faut-il ?", fontsize=11, color=INK, pad=8)

ax2.plot([u * 100 for u in unders], corr_by_u, "-o", color=BLUES[4], lw=2, label="correlation W")
ax2.plot([u * 100 for u in unders], dir_by_u, "-s", color=ACCENT, lw=2, label="sens de l'asymetrie")
ax2.set_xlabel("taux de sous-declaration (%)", color=INK2)
ax2.set_ylabel("qualite de recuperation", color=INK2)
ax2.set_ylim(0, 1.05); ax2.grid(True, color=GRID, lw=0.7)
ax2.legend(frameon=False, fontsize=9, loc="lower left")
ax2.set_title("(b) Robustesse a la sous-declaration", fontsize=11, color=INK, pad=8)
fig.tight_layout(rect=[0, 0, 1, 0.94])
p2 = os.path.join(outdir, "K2_calibration_diagnostics.png")
fig.savefig(p2, dpi=200, bbox_inches="tight"); print("figure ecrite :", p2)

# --- figure K3 : lecture epidemique
# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime cette
# figure sur 7,28 pouces : tracee sur 9,5 elle subissait une reduction de 0,77 et
# ses etiquettes sortaient a 6 points. A 8,6 pouces le facteur monte a 0,85. La
# hauteur perd la bande du titre general, retire plus bas.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.62))

ax1.axvspan(0, 1.0, color=BLUES[0], alpha=0.35, zorder=0)
ax1.text(0.52, 2.35, "domaine\nadmissible\n$g\\leq 1$", ha="center", va="center",
         fontsize=8.5, color=BLUES[5], fontweight="bold")
ax1.plot(gains, rho_by_gain, "-", color=ACCENT, lw=2.2,
         label="$\\rho(W)$ : système latent simultané")
ax1.plot(gains, R0_by_gain, "-", color=BLUES[4], lw=2.2,
         label="$R_0=\\rho(M)$ : cascade d'incidents")
ax1.axhline(1.0, color=INK, lw=1.1, ls="--")
ax1.text(3.2, 0.84, "seuil critique", ha="center", va="top", fontsize=9, color=INK)
for gc, col, lab in [(g_crit_rho, ACCENT, f"$\\rho(W)=1$\n$g={g_crit_rho:.2f}$".replace(".", "{,}")),
                     (g_crit_R0, BLUES[4], f"$R_0=1$\n$g={g_crit_R0:.1f}$".replace(".", "{,}"))]:
    ax1.axvline(gc, color=col, lw=1, ls=":")
    # a gauche de sa ligne pour le seuil de R0, sinon coupe par le bord droit
    ax1.text(gc + 0.15 if gc < 5 else gc - 0.15, 3.0, lab, fontsize=8.5, color=col,
             ha="left" if gc < 5 else "right")
ax1.annotate(f"$\\rho(\\mathrm{{TRANS}})={RHO_RAW:.2f}$".replace(".", "{,}")
             + "\n(non normalisé :\nerreur de catégorie)",
             xy=(1.0, RHO_RAW), xytext=(5.9, 1.45), fontsize=8.5, color=MUTED,
             style="italic", ha="center",
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1,
                             connectionstyle="arc3,rad=0"))
ax1.plot([1.0], [RHO_RAW], "o", color=MUTED, ms=5, zorder=5)
ax1.set_xlabel("gain $g$ = part de contagion du pilier le plus exposé", color=INK2)
ax1.set_ylabel("rayon spectral", color=INK2)
ax1.set_xlim(0, 9.0); ax1.set_ylim(0, 4.0)
ax1.grid(True, color=GRID, lw=0.7)
# sous le cadre : dans le cadre elle recouvrait l'etiquette du seuil de rho(W)
ax1.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1)
# TITRES SUR DEUX LIGNES, ET CE N'EST PAS UN GOUT. Poses sur une seule ligne a 11 points
# ils mesuraient 4,6 et 3,8 pouces pour un panneau qui en fait 3,4 : matplotlib centre un
# titre sur ses axes sans jamais le couper, donc chacun debordait de part et d'autre et le
# (a) se terminait sur le (b) au milieu de la figure. Vu a l'impression, page 78.
ax1.set_title("(a)  Normalisé, le modèle est stable\nsur tout son domaine",
              fontsize=10, color=INK, pad=6)

xpos = np.arange(J)
rootv = np.array([ROOT[j] for j in range(1, J + 1)])
axb = ax2.twinx()
b1 = ax2.bar(xpos - 0.2, prog_true, width=0.4, color=BLUES[4], label="progéniture $(I-M)^{-1}$")
b2 = axb.bar(xpos + 0.2, rootv, width=0.4, color=ACCENT, alpha=0.85, label="ROOT (expert)")
ax2.set_xticks(xpos); ax2.set_xticklabels(PIL)
ax2.set_ylabel("descendants attendus", color=BLUES[4])
axb.set_ylabel("ROOT, jugement d'expert", color=ACCENT)
ax2.tick_params(axis="y", colors=BLUES[4]); axb.tick_params(axis="y", colors=ACCENT)
ax2.set_title(f"(b)  ROOT se déduit de TRANS\n(Spearman = {rho_s:.2f})".replace(".", ","),
              fontsize=10, color=INK, pad=6)
# marge haute : la barre ROOT de P4 montait dans la legende
ax2.set_ylim(0, prog_true.max() * 1.45); axb.set_ylim(0, rootv.max() * 1.45)
ax2.legend(handles=[b1, b2], loc="upper right", frameon=False, fontsize=9)
# PAS DE TITRE GENERAL : la legende LaTeX porte le titre, et celui-ci commencait
# par le code interne « K3 : ».
fig.tight_layout()
p3 = os.path.join(outdir, "K3_branchement_R0.png")
fig.savefig(p3, dpi=200, bbox_inches="tight"); print("figure ecrite :", p3)
