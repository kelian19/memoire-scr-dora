#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17 : Markov + trajectoire SCR(t). La dimension TEMPS, empilee sur le moteur du 16.

Le 16 donne le SCR pour un ETAT de conformite fige. Ici on ajoute la DYNAMIQUE : l'etat
de conformite evolue dans le temps le long de la mise en conformite DORA, et on en deduit
la trajectoire du capital SCR(t). Deux couches, deux echelles de temps (fast-slow) :
  - LENTE  : une chaine de Markov fait progresser l'etat de conformite (mois, annees) ;
  - RAPIDE : pour un etat donne, le moteur du 16 chiffre le SCR (cascade intra-annee).

2.1  CHAINE DE MARKOV 3 etats NC -> PC -> C, temps continu, Conforme ABSORBANT en cas de
     base (une fois conforme, on le reste ; la regression est un axe de sensibilite).

2.2  SEJOURS DE TYPE PHASE (Erlang-k), pas exponentiels. Une exponentielle autorise une
     remediation quasi instantanee (densite maximale en 0), irrealiste : un projet DORA a
     une duree minimale. L'Erlang-k (somme de k exponentielles) a un mode > 0, "en cloche",
     qui capte cette duree de projet. On developpe chaque macro-etat transitoire en k phases.

2.3  EMBOITEMENT SEQUENTIEL. A chaque horizon t, la chaine donne la distribution des etats
     P(NC,t), P(PC,t), P(C,t) ; le SCR de chaque etat vient du 16 ; on agrege :
         SCR(t) = P(NC,t) SCR_NC + P(PC,t) SCR_PC + P(C,t) SCR_C.

2.4  COUPLAGE PAR LE MEME FACTEUR SYSTEMIQUE Theta. Un environnement degrade (Theta bas)
     fait DEUX choses a la fois, avec le meme tirage :
       - il RALENTIT la remediation : taux mu(Theta) = mu0 * exp(beta_rem * Theta) ;
       - il DURCIT la cascade : SCR_etat(Theta) = SCR_etat * exp(-beta_casc * Theta).
     Les deux poussent le SCR dans le meme sens quand Theta < 0 : la correlation entre
     conformite et sinistralite, perdue si les couches etaient independantes, est restauree.
     La dispersion de Theta ~ N(0,1) engendre la BANDE d'incertitude de SCR(t).

2.5  SORTIE : trajectoire SCR(t) (mediane + bande) et Delta_DORA(t) = SCR(t) - SCR_C, le
     surcout de non-conformite RESIDUEL a l'horizon t (tend vers 0 quand tout devient C).

PORTEE. Les taux de transition NE SE CALIBRENT PAS (pas de donnee : DORA applique depuis
2025). Ils sont ancres sur des durees de projet types (NC->PC et PC->C ~ 1,5 an chacun,
soit ~3 ans NC->C) et presentes en SENSIBILITE, comme beta_rem et beta_casc. Les 3 valeurs
de SCR par etat viennent du 16 (lecture frequence + propagation, sans le canal detection en
attente de validation). Ne touche ni src/ ni memoire/.

Sortie : diagnostics + figure S12_trajectoire_scr.png. Verifie exit 0.
"""

import os
import sys

import numpy as np
from scipy.linalg import expm
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                              # noqa: E402
from euro_cascade_model import PARAMS, var                   # noqa: E402

W = 74
ETATS = ["NC", "PC", "C"]
SCENARIO = {"C": "S0_conforme", "PC": "S1_partiel", "NC": "S2_non_conforme"}
G_PROP = {"C": 0.45, "PC": 0.68, "NC": 0.90}     # lecture freq + propagation (comme 16, sans detection)

# --- 2.2 sejours de type phase (Erlang-k), taux ancres sur durees de projet DORA ---
K_ERLANG = 2                                      # Erlang-2 : sejour "en cloche", duree minimale
MEAN_NC = 1.5                                     # duree moyenne en NC avant PC (annees)
MEAN_PC = 1.5                                     # duree moyenne en PC avant C (annees)
# --- 2.4 couplage systemique (non calibre, sensibilite) ---
BETA_REM = 0.35                                   # Theta<0 ralentit la remediation
BETA_CASC = 0.10                                  # Theta<0 durcit la cascade
N_THETA = 3000                                    # tirages de Theta pour la bande
INIT = {"NC": 0.35, "PC": 0.35, "C": 0.30}        # etat initial = marginale ancree du 16
HORIZONS = np.linspace(0.0, 5.0, 51)              # trajectoire sur 5 ans
TABLE_T = [0.0, 1.0, 2.0, 3.0, 5.0]
NY = 60_000
SEED = 17


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


# ============================================================ SCR par etat (moteur du 16)
def scr_state(source, etat, seed=SEED, ny=NY):
    sp = PARAMS[source]
    lam = ec.lambda_scenario(source, SCENARIO[etat], mode="center")
    rng = np.random.default_rng(seed)
    return var(ec.simulate_euro(lam, G_PROP[etat], sp["xi"], sp["sigma"],
                                sp["u"], sp["p_u"], sp["cap"], ny, rng))


# ============================================================ 2.1 + 2.2 generateur CTMC Erlang
def generator_base():
    """Generateur Q0 sur l'espace de phases : NC_1..k, PC_1..k, C absorbant. Taux a Theta=0."""
    k = K_ERLANG
    n = 2 * k + 1                                 # k phases NC, k phases PC, 1 etat C
    C = n - 1
    Q = np.zeros((n, n))
    rate_nc = k / MEAN_NC                          # Erlang-k : chaque phase au taux k*mu
    rate_pc = k / MEAN_PC
    for i in range(k):                             # phases NC
        nxt = i + 1 if i < k - 1 else k            # derniere phase NC -> premiere phase PC
        Q[i, nxt] += rate_nc
    for i in range(k, 2 * k):                      # phases PC
        nxt = i + 1 if i < 2 * k - 1 else C        # derniere phase PC -> C
        Q[i, nxt] += rate_pc
    for i in range(n):
        Q[i, i] = -Q[i].sum()                      # C : ligne nulle (absorbant)
    return Q


def init_vector():
    """Distribution initiale sur les phases : masse macro en phase 1 de chaque macro-etat."""
    k = K_ERLANG
    v = np.zeros(2 * k + 1)
    v[0] = INIT["NC"]                              # NC_1
    v[k] = INIT["PC"]                              # PC_1
    v[-1] = INIT["C"]                              # C
    return v


def macro_probs(v):
    """Somme les phases -> probabilites macro (NC, PC, C)."""
    k = K_ERLANG
    return {"NC": v[:k].sum(), "PC": v[k:2 * k].sum(), "C": v[-1]}


def state_probs_at(t, theta, Q0, v0):
    """P(etat) a l'horizon t sous Theta : taux ralentis par exp(beta_rem*theta)."""
    speed = np.exp(BETA_REM * theta)               # Theta>0 -> plus rapide ; Theta<0 -> plus lent
    P = expm(Q0 * speed * t)
    return macro_probs(v0 @ P)


# ============================================================ trajectoire
titre("Trajectoire SCR(t) : Markov (Erlang-2) empile sur le moteur du 16")
print(f"  Etats NC->PC->C, C absorbant. Sejours moyens : NC {MEAN_NC} an, PC {MEAN_PC} an.")
print(f"  Init (marginale du 16) : NC {INIT['NC']:.0%}  PC {INIT['PC']:.0%}  C {INIT['C']:.0%}")
print(f"  Couplage Theta : beta_rem={BETA_REM} (vitesse), beta_casc={BETA_CASC} (cascade).")

Q0 = generator_base()
v0 = init_vector()
rng_theta = np.random.default_rng(SEED)
thetas = rng_theta.standard_normal(N_THETA)

traj = {}     # traj[source] = dict(t, med, lo, hi, dmed, scr_state)
for source in ("OPRISK", "PRC"):
    scr = {e: scr_state(source, e) for e in ETATS}
    scr_vec = {e: scr[e] for e in ETATS}
    # pour chaque Theta et chaque horizon : SCR(t|Theta)
    curves = np.empty((N_THETA, len(HORIZONS)))
    for a, th in enumerate(thetas):
        harden = np.exp(-BETA_CASC * th)           # Theta<0 -> durcit la cascade
        for b, t in enumerate(HORIZONS):
            p = state_probs_at(t, th, Q0, v0)
            s = sum(p[e] * scr_vec[e] for e in ETATS) * harden
            curves[a, b] = s
    med = np.median(curves, axis=0)
    lo, hi = np.percentile(curves, [5, 95], axis=0)
    traj[source] = dict(med=med, lo=lo, hi=hi, scr=scr)
    print(f"\n  {source} : SCR_C={scr['C']:.0f}  SCR_PC={scr['PC']:.0f}  SCR_NC={scr['NC']:.0f} M EUR")
    print(f"  {'t (ans)':>8}{'SCR(t) med':>13}{'bande 90%':>20}{'Delta_DORA(t)':>16}")
    for t in TABLE_T:
        bi = int(np.argmin(np.abs(HORIZONS - t)))
        d = med[bi] - scr["C"]
        print(f"  {t:>8.0f}{med[bi]:>13.0f}{f'[{lo[bi]:.0f} ; {hi[bi]:.0f}]':>20}{d:>16.0f}")
    print("  Delta_DORA(t) = surcout residuel de non-conformite ; -> 0 quand tout devient C.")


# ============================================================ figure
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"],
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
INK, INK2 = "#1b1e30", "#223e55"
ACCENT, BLUE, GREEN, GREY = "#a6002e", "#3661ac", "#008d87", "#7d7d7d"

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime
# cette figure sur 7,27 pouces : tracee plus large, elle subissait une
# reduction qui faisait tomber ses etiquettes sous 6 points. Le rapport
# largeur sur hauteur est conserve, donc la figure occupe la meme place
# sur la page ; c'est son texte qui y prend plus de place.
fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.9, 3.52))

# panneau A : trajectoire SCR(t) OpRisk, mediane + bande + asymptote SCR_C
tr = traj["OPRISK"]
axA.fill_between(HORIZONS, tr["lo"], tr["hi"], color=BLUE, alpha=0.18,
                 label="bande 90% (alea Theta)")
axA.plot(HORIZONS, tr["med"], color=BLUE, lw=2.2, label="SCR(t) mediane")
axA.axhline(tr["scr"]["C"], color=GREEN, ls="--", lw=1.4,
            label=f"SCR conforme {tr['scr']['C']:.0f}")
axA.axhline(tr["scr"]["NC"], color=ACCENT, ls=":", lw=1.2,
            label=f"SCR non conforme {tr['scr']['NC']:.0f}")
axA.set_xlabel("horizon t (annees)", fontsize=9.5, color=INK2)
axA.set_ylabel("SCR_DORA(t) (M€)", fontsize=9.5, color=INK2)
# TITRES RACCOURCIS. A la largeur de trace corrigee, les deux titres etaient
# plus larges que leur panneau et se chevauchaient au milieu de la figure. Ce
# que le titre perd, la legende de la figure le porte deja dans le memoire.
axA.set_title("(A)  Trajectoire du capital le long de la remédiation",
              fontsize=10, color=INK, pad=6)
axA.legend(fontsize=8.0, frameon=False, loc="upper right")
axA.grid(alpha=0.25, lw=0.5)
axA.set_xlim(0, 5)

# panneau B : evolution des probabilites d'etat (Theta=0), la remediation en marche
pN, pP, pC = [], [], []
for t in HORIZONS:
    p = state_probs_at(t, 0.0, Q0, v0)
    pN.append(p["NC"]); pP.append(p["PC"]); pC.append(p["C"])
axB.stackplot(HORIZONS, pN, pP, pC, colors=[ACCENT, "#e0a53f", GREEN], alpha=0.85,
              labels=["Non conforme", "Partiellement conforme", "Conforme"])
axB.set_xlabel("horizon t (annees)", fontsize=9.5, color=INK2)
axB.set_ylabel("probabilite d'etat", fontsize=9.5, color=INK2)
axB.set_title("(B)  Les états de conformité au fil de la remédiation",
              fontsize=10, color=INK, pad=6)
axB.legend(fontsize=8.2, frameon=False, loc="center right")
axB.set_xlim(0, 5)
axB.set_ylim(0, 1)

fig.tight_layout(rect=[0, 0, 1, 0.95])

outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S12_trajectoire_scr.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
