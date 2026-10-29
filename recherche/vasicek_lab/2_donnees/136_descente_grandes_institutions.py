#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
136 : une variante de la descente d'echelle, estimee sur les grandes institutions.

LE DEFAUT DE LA DESCENTE PUBLIEE (scripts 60, 65, 130). L'elasticite de frequence du registre,
b_lambda = 0.074, fait qu'un facteur 134 sur les actifs ne multiplie le besoin que par 2.6, et
qu'une entite de 500 M EUR immobiliserait 17 % de son bilan. Hugo Rapior juge cette descente
incoherente. La section 1 mesure une difference d'observation : le panel suit une firme de la
premiere a la derniere annee ou elle apparait dans la base, pour un risque quelconque, et les petites
firmes y ont des fenetres plus courtes. C'est un biais de fenetre PLAUSIBLE, pas un mecanisme
demontre : une annee sans evenement enregistre n'est pas une annee sans incident, et rien ne garantit
que les grandes firmes soient observees exhaustivement. La restriction aux grandes institutions vise
a l'attenuer ; l'exposant y monte puis se stabilise. Diagnostics complementaires : script 137.

LE MODELE RETENU. Trois changements, la severite et la cascade restant celles du secteur :
  1. frequence : binomiale negative NB2 ajustee sur les firmes de plus de 100 Md USD d'actifs,
     lambda(A) = exp(a + beta (log A - xbar)), dispersion r de ce meme ajustement ;
  2. pertes sous le seuil : tirees dans les 484 pertes observees sous u, au lieu d'une perte nulle ;
  3. severite : multiplicateur m(A) inchange (b = 0.087), aucun plafond.
L'exposant est EXTRAPOLE sous 100 Md USD : c'est la reserve principale, imprimee avec ses bandes.

CONTROLE. Le script reproduit d'abord, avec le moteur publie (pid.Evaluator), les valeurs des
scripts 60, 64 et 66 a l'entite notionnelle, et s'arrete sinon.

CE QUE LE SCRIPT NE FAIT PAS. Il ne touche ni a config.py ni au moteur, et aucun resultat du secteur
ne bouge. Les etats de conformite restent supposes. Exige SAS_OpRisk_Global_Data_June_2026.xlsx.
"""

import os
import re
import sys
import warnings

import numpy as np
from scipy import optimize, special
from scipy.stats import genpareto

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(HERE)
import partial_id as pid                                            # noqa: E402
import descente as dsc                                              # noqa: E402
import euro_cascade_model as ec                                     # noqa: E402
from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR  # noqa: E402

TAUX_USD = 1.04            # meme cours que les scripts 65 et 130
NY = 600_000
SEED = 20260721
ALPHA = 0.995
SEUIL_GRANDES = 1.0e5      # M USD : firmes de plus de 100 Md USD d'actifs
COC, COC_HIST, LR = 0.0475, 0.06, 0.485
SP = pid.PARAMS["OPRISK"]
U = SP["u"]
W_ = 92
NOTION_USD = 20_000.0
LEVIER_SCR = 1932.0        # SCR de l'entite notionnelle au levier median (script 65)

ENT = [("Petite entite (posee)", 500.0, None), ("Assureur non-vie A", 2319.0, 425.0),
       ("Assureur non-vie B", 5305.0, 812.5), ("Entite notionnelle", NOTION_USD / TAUX_USD, None),
       ("Assureur vie C", 37845.0, 1587.0), ("Assureur vie D", 309800.0, 14800.0),
       ("Grand groupe (pose)", 2_400_000.0, None)]


def titre(s):
    print("\n" + "=" * W_ + f"\n{s}\n" + "=" * W_)


# =====================================================================================
titre("1. Pourquoi l'exposant du registre est aplati : la fenetre d'observation")
# =====================================================================================
D = dsc.Descente()
pa = D.pa
q = pa.groupby("firm").agg(A=("actifs", "first"), ans=("year", "size"), n=("n", "sum"))
classes = [("moins de 1 Md USD", q.A < 1e3), ("1 a 10 Md USD", (q.A >= 1e3) & (q.A < 1e4)),
           ("10 a 100 Md USD", (q.A >= 1e4) & (q.A < 1e5)), ("plus de 100 Md USD", q.A >= 1e5)]
print(f"{'classe d actifs':<22}{'firmes':>8}{'annees observees':>18}{'incidents/firme/an':>20}")
for lab, sel in classes:
    s = q[sel]
    print(f"{lab:<22}{len(s):>8}{s.ans.mean():>18.2f}{s.n.sum() / s.ans.sum():>20.3f}")
print("Les petites firmes ont des fenetres plus courtes : biais de fenetre plausible, non demontre.")


def ajuste(sub):
    x = np.log(sub.actifs.to_numpy())
    k = sub.n.to_numpy().astype(float)
    xb = float(x.mean())
    xc = x - xb

    def nll(p):
        a, b, lr = p
        r = np.exp(lr)
        mu = np.exp(np.clip(a + b * xc, -30, 30))
        return -np.sum(special.gammaln(k + r) - special.gammaln(r) - special.gammaln(k + 1)
                       + r * np.log(r / (r + mu)) + k * np.log(mu / (r + mu)))
    o = optimize.minimize(nll, [np.log(k.mean()), 0.2, 0.0], method="Nelder-Mead",
                          options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 20000})
    eps = 1e-4
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            P = [o.x.copy() for _ in range(4)]
            P[0][i] += eps; P[0][j] += eps; P[1][i] += eps; P[1][j] -= eps
            P[2][i] -= eps; P[2][j] += eps; P[3][i] -= eps; P[3][j] -= eps
            H[i, j] = (nll(P[0]) - nll(P[1]) - nll(P[2]) + nll(P[3])) / (4 * eps * eps)
    return dict(a=float(o.x[0]), b=float(o.x[1]), r=float(np.exp(o.x[2])), xb=xb,
                se=float(np.sqrt(np.linalg.inv(H)[1, 1])), n=len(sub), firmes=sub.firm.nunique())


print(f"\n{'panel restreint aux actifs >=':<32}{'beta':>8}{'ET':>8}{'lignes':>9}{'firmes':>8}")
FITS = {}
for s in (0.0, 1e3, 1e4, 5e4, 1e5, 3e5):
    F = ajuste(pa[pa.actifs >= s])
    FITS[s] = F
    print(f"{s:>20,.0f} M USD{'':>6}{F['b']:>8.3f}{F['se']:>8.3f}{F['n']:>9}{F['firmes']:>8}")
if abs(FITS[0.0]["b"] - D.b_lam) > 1e-3:
    sys.exit("l'ajustement sur tout le panel ne redonne pas l'exposant du registre : arret.")
F = FITS[SEUIL_GRANDES]
print(f"\nexposant retenu (firmes >= 100 Md USD) : beta = {F['b']:.3f} (ET {F['se']:.3f}), "
      f"dispersion r = {F['r']:.3f}, ancre exp(xbar) = {np.exp(F['xb'])/1e3:.0f} Md USD")


def lam_new(a_usd, b=None, fit=None):
    fit = fit or F
    return float(np.exp(fit["a"] + (fit["b"] if b is None else b) * (np.log(a_usd) - fit["xb"])))


# =====================================================================================
titre("2. Le moteur et son controle sur la methode publiee")
# =====================================================================================
df = filter_finance(filter_cyber(load_clean(os.path.join(REPO, "data", "raw",
                                                         "SAS_OpRisk_Global_Data_June_2026.xlsx"))))
d = df[df.year.between(2000, 2026)]
if len(d) != 570:
    sys.exit(f"population de calibration differente du protocole : {len(d)} pertes au lieu de 570")
L_ = (d["loss"] * USD_EUR).to_numpy()
CORPS = np.sort(L_[L_ <= U])
print(f"pertes sous le seuil u = {U:.2f} M EUR : {CORPS.size}, mediane {np.median(CORPS):.2f}, "
      f"moyenne {CORPS.mean():.2f} M EUR")
W_AMORCE = np.array([ec.eng.LAMBDA[j] for j in pid.PIL], float)


class Moteur:
    """Annees simulees a nombres communs ; `card` fixe la cascade, comme pid.Evaluator."""

    def __init__(self, lam, mult, publie=False, xi=None, cap=None, seed=SEED, r=None):
        if publie:     # moteur publie, a l'identique des scripts 60, 64 et 130
            ev = pid.Evaluator(lam=lam, n_years=NY, alpha=ALPHA, seed=seed)
            self.T, self.year_of, self.u = ev.T, ev.year_of, ev.u
            self.idx_by_am, cum = ev.idx_by_am, ev.cum * mult
        else:
            rng = np.random.default_rng(seed)
            r = F["r"] if r is None else r
            counts = rng.negative_binomial(r, r / (r + lam), size=NY)
            self.T = int(counts.sum())
            self.year_of = np.repeat(np.arange(NY), counts)
            amorce = rng.choice(pid.NP_, size=self.T, p=W_AMORCE / W_AMORCE.sum())
            self.u = rng.random(self.T)
            n = self.T * pid.NP_
            queue = rng.random(n) < SP["p_u"]
            uq = rng.uniform(0, 1, int(queue.sum()))
            ic = rng.integers(0, CORPS.size, int((~queue).sum()))
            sev = np.empty(n)
            sev[queue] = U + genpareto.ppf(uq, c=SP["xi"] if xi is None else xi, scale=SP["sigma"])
            sev[~queue] = CORPS[ic]
            sev = sev * mult
            if cap is not None:
                sev = np.minimum(sev, cap)
            cum = np.cumsum(sev.reshape(self.T, pid.NP_), axis=1)
            self.idx_by_am = [np.where(amorce == a)[0] for a in range(pid.NP_)]
        self.cum, self._ar = cum, np.arange(self.T)

    def annuel(self, card):
        K = np.empty(self.T, dtype=np.int64)
        for a in range(pid.NP_):
            idx = self.idx_by_am[a]
            if idx.size:
                cdf = np.cumsum(card[a][1:])
                cdf[-1] = 1.0
                K[idx] = np.searchsorted(cdf, self.u[idx], side="right") + 1
        np.clip(K, 1, pid.NP_, out=K)
        return np.bincount(self.year_of, weights=self.cum[self._ar, K - 1], minlength=NY)

    def var(self, card):
        return float(np.quantile(self.annuel(card), ALPHA))


def tvar(x):
    s = np.sort(x)
    return float(s[-int(np.ceil((1 - ALPHA) * s.size)):].mean())


def degrade(Wm, a):
    Wd = Wm.copy()
    Wd[0, :] *= (1.0 - a)          # P1 est le premier pilier de pid.PIL
    return Wd


def cartes_sommets(Wm):
    S, _ = pid.decompose(Wm)
    out = []
    for row in pid.all_vertices(S, 1.0):
        A = np.zeros((pid.NP_, pid.NP_))
        A[pid.IU] = row
        Wv = S + (A - A.T)
        if pid.admissible(Wv):
            out.append(pid.card_dist_all(Wv)[0])
    return out


if pid.PIL[0] != 1:
    sys.exit("P1 n'est plus le premier pilier : la degradation viserait un autre pilier.")
WEXP = pid.expert_matrix()
CEXP = pid.card_dist_all(WEXP)[0]
CSOM = cartes_sommets(WEXP)
CG = {g: pid.card_dist_all(pid.expert_matrix(g))[0] for g in (0.45, 0.68, 0.90)}

a_n = NOTION_USD
m_n = D.mult(a_n)
mp = Moteur(D.lam(a_n), m_n, publie=True)
pub = dict(point=mp.var(CEXP), lo=min(mp.var(c) for c in CSOM), hi=max(mp.var(c) for c in CSOM),
           g45=mp.var(CG[0.45]), g68=mp.var(CG[0.68]))
Wd1 = degrade(WEXP, 1.0)
pub["a1"] = mp.var(pid.card_dist_all(Wd1)[0])
attendu = dict(point=169.0, lo=131.5, hi=183.3, g45=116.4, g68=143.2, a1=127.1)
for k_, v_ in attendu.items():
    print(f"controle, methode publiee, {k_:<6}: {pub[k_]:7.1f}  (publie {v_})")
    if abs(pub[k_] - v_) > 0.6:
        sys.exit(f"la methode publiee n'est plus reproduite ({k_}) : arret.")
print("Les valeurs des scripts 60, 64 et 66 sont reproduites : le moteur est le bon.")

LAM_SECT = 21.56           # valeur du tableau des echelles (script 60)
ms = Moteur(LAM_SECT, 1.0, publie=True)
xs = ms.annuel(CEXP)
B_SECT, E_SECT = float(np.quantile(xs, ALPHA)), float(xs.mean())
print(f"secteur, meme protocole : besoin {B_SECT:.0f} M EUR, perte moyenne {E_SECT:.0f} M EUR")

# =====================================================================================
titre("3. L'entite notionnelle sous le modele retenu")
# =====================================================================================
lam_n = lam_new(a_n)
mn = Moteur(lam_n, m_n)
xn = mn.annuel(CEXP)
Bn, En = float(np.quantile(xn, ALPHA)), float(xn.mean())
som_n = [mn.var(c) for c in CSOM]
b_lo, b_hi = F["b"] - 1.96 * F["se"], F["b"] + 1.96 * F["se"]
vb = sorted([Moteur(lam_new(a_n, b_lo), m_n).var(CEXP), Moteur(lam_new(a_n, b_hi), m_n).var(CEXP)])
B300 = Moteur(lam_new(a_n, fit=FITS[3e5]), m_n).var(CEXP)        # fonction seule, r du seuil 100
B300c = Moteur(lam_new(a_n, fit=FITS[3e5]), m_n, r=FITS[3e5]["r"]).var(CEXP)   # reestimation complete
xi_lo, xi_hi = SP["xi_ic90"]
Mxi = [Moteur(lam_n, m_n, xi=x).annuel(CEXP) for x in (xi_lo, xi_hi)]
Bxi = [float(np.quantile(x, ALPHA)) for x in Mxi]
Txi = [tvar(x) for x in Mxi]
p0 = float(np.mean(np.bincount(mn.year_of, minlength=NY) == 0))
print(f"actifs {a_n:,.0f} M USD, soit {a_n/TAUX_USD:,.0f} M EUR ; multiplicateur de severite {m_n:.3f}")
print(f"lambda {lam_n:.4f} par an (registre : {D.lam(a_n):.4f}) ; annees sans incident {100*p0:.1f} %")
print(f"besoin a 99,5 % : {Bn:.1f} M EUR ; sommets [{min(som_n):.1f} ; {max(som_n):.1f}], "
      f"largeur {max(som_n)-min(som_n):.1f}")
print(f"sensibilite a la pente seule, beta a +-1,96 ET [{b_lo:.3f} ; {b_hi:.3f}], intercept et "
      f"dispersion fixes : besoin [{vb[0]:.1f} ; {vb[1]:.1f}] (pas un IC du capital, voir 137)")
print(f"frequence estimee sur les firmes >= 300 Md USD (beta {FITS[3e5]['b']:.3f}) : besoin {B300:.1f} "
      f"a dispersion du seuil 100 (r {F['r']:.3f}) ; {B300c:.1f} en reestimation complete "
      f"(r {FITS[3e5]['r']:.3f})")
print(f"xi sur son IC 90 % [{xi_lo:.3f} ; {xi_hi:.3f}] : besoin {Bxi[0]:.1f} a {Bxi[1]:.1f} ; "
      f"TVaR {Txi[0]:.1f} a {Txi[1]:.1f}")
print("A cette frequence, P(perte > 0) est de l'ordre de 2 % : le quantile a 99,5 % se lit pres du seuil")
print("de queue et depend peu de xi ; la moyenne des pires annees porte la sensibilite a la queue.")
print(f"moyenne des 0,5 % pires annees (TVaR) {tvar(xn):.1f} ; perte moyenne {En:.2f} ; "
      f"P(perte > 0) {100*(xn > 0).mean():.2f} %")
print(f"publie : 169.0 ; nouveau / publie = {Bn/169.0:.3f}")
print(f"secteur / entite : lambda / {LAM_SECT/lam_n:,.0f}, perte moyenne / {E_SECT/En:,.0f}, "
      f"besoin / {B_SECT/Bn:,.0f}")
print(f"au levier median, SCR {LEVIER_SCR:.0f} M EUR : besoin = {100*Bn/LEVIER_SCR:.1f} % du SCR")
print(f"facteur entre notifications PRC (341/an) et lambda d'entite : {341/lam_n:,.0f}")

# --- le gain de propagation et ses correspondances
titre("4. Le gain de propagation a l'echelle de l'entite")
grille = np.linspace(0.05, 1.0, 20)
vals = [mn.var(pid.card_dist_all(pid.expert_matrix(g))[0]) for g in grille]
mono = all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
print(f"grille de vingt valeurs de g, nombres communs : croissance {mono}")
if not mono:
    sys.exit("le besoin d'entite n'est pas croissant en g : arret.")
Bg = {}
for g in (0.20, 0.45, 0.55, 0.65, 0.68, 0.70, 0.80, 0.85, 0.875, 0.90):
    Bg[g] = mn.var(pid.card_dist_all(pid.expert_matrix(g))[0])
B020, B095 = (mn.var(pid.card_dist_all(pid.expert_matrix(g))[0]) for g in (0.20, 0.95))
EL_ARC = float(np.log(B095 / B020) / np.log(0.95 / 0.20))
fine = np.round(np.arange(0.50, 0.9001, 0.01), 2)
Bfine = {g: mn.var(pid.card_dist_all(pid.expert_matrix(g))[0]) for g in fine}
G10 = max(g for g in fine if Bg[0.90] / Bfine[g] - 1 >= 0.10)
print(f"elasticite d'arc du besoin au gain sur [0,20 ; 0,95] : {EL_ARC:.2f}")
print(f"plus grand g (pas de 0,01) donnant un ecart d'au moins 10 % avec g = 0,90 : {G10:.2f}")
CORR = [("etroite (0,70 / 0,80 / 0,90)", 0.70, 0.80, 0.90), ("retenue (0,45 / 0,68 / 0,90)", 0.45, 0.68, 0.90),
        ("large (0,20 / 0,55 / 0,90)", 0.20, 0.55, 0.90), ("conformite faible (0,45/0,55/0,65)", 0.45, 0.55, 0.65),
        ("tres etroite (0,85 / 0,875 / 0,90)", 0.85, 0.875, 0.90)]
print(f"{'application':<38}{'C':>8}{'PC':>8}{'NC':>8}{'ecart':>9}  ordre")
for lab, c, p_, nc in CORR:
    e = Bg[nc] / Bg[c] - 1
    print(f"{lab:<38}{Bg[c]:>8.1f}{Bg[p_]:>8.1f}{Bg[nc]:>8.1f}{100*e:>+8.1f}%  {Bg[c] < Bg[p_] < Bg[nc]}")

# --- la degradation de l'emission de P1 (objection du biais de narration, chapitre 7)
titre("5. L'emission de P1 degradee, a l'echelle de l'entite")
print(f"{'a':>6}{'point':>9}{'bande':>18}{'largeur':>9}  premier pilier")
DEG = {}
for a in (0.0, 0.25, 0.50, 0.75, 1.0):
    Wa = degrade(WEXP, a)
    pt = mn.var(pid.card_dist_all(Wa)[0])
    sv = [mn.var(c) for c in cartes_sommets(Wa)]
    gains = []
    for j in range(pid.NP_):
        Wr = Wa.copy()
        Wr[j, :] = 0.0
        gains.append(pt - mn.var(pid.card_dist_all(Wr)[0]))
    premier = f"P{pid.PIL[int(np.argmax(gains))]}"
    DEG[a] = (pt, min(sv), max(sv), premier)
    print(f"{a:>6.2f}{pt:>9.1f}{'[' + format(min(sv), '.1f') + ' ; ' + format(max(sv), '.1f') + ']':>18}"
          f"{max(sv)-min(sv):>9.1f}  {premier}")
for a in (0.0, 1.0):
    bc = mn.var(pid.card_dist_all(degrade(pid.expert_matrix(0.45), a))[0])
    bnc = mn.var(pid.card_dist_all(degrade(pid.expert_matrix(0.90), a))[0])
    print(f"a = {a:.0f} : ecart entre etats (g 0,45 -> 0,90) {bc:.1f} -> {bnc:.1f}, {100*(bnc/bc-1):+.1f} %")
d_a = DEG[0.0][0] - DEG[1.0][0]
larg = DEG[0.0][2] - DEG[0.0][1]
print(f"P1 muette : {DEG[0.0][0]:.1f} -> {DEG[1.0][0]:.1f} M EUR ({100*(DEG[1.0][0]/DEG[0.0][0]-1):+.1f} %), "
      f"deplacement {d_a:.1f} = {d_a/larg:.2f} fois la largeur de bande {larg:.1f}")

# --- detenir ou transferer
titre("6. Detenir ou transferer, a l'echelle corrigee")
grid = np.linspace(0.0, 3.0 * Bn, 601)


def couts(X):
    cede = [np.minimum(X, L) for L in grid]
    fin = np.array([c.mean() / LR + COC * float(np.quantile(X - c, ALPHA)) for c in cede])
    ret = np.array([float((X - c).mean()) for c in cede])
    return fin, fin + ret


fin, eco = couts(xn)
i_f, i_e = int(np.argmin(fin)), int(np.argmin(eco))
kap = np.linspace(0.0, 1.0, 10001)


def seuil_global(c_l, c0):
    for k in kap:
        if np.min(c_l[1:] + COC * k * grid[1:]) >= c0:
            return float(k)
    return float("nan")


k_fin, k_eco = seuil_global(fin, fin[0]), seuil_global(eco, eco[0])
print(f"besoin {Bn:.1f} ; perte moyenne {En:.2f} ; multiple besoin / perte moyenne {Bn/En:.1f} "
      f"(secteur {B_SECT/E_SECT:.1f})")
print(f"cout de detention : {COC*Bn:.2f} M EUR/an a 4,75 %, {COC_HIST*Bn:.2f} a 6 % ; "
      f"soit {COC*Bn/En:.2f} a {COC_HIST*Bn/En:.2f} fois la perte moyenne")
print(f"financement : tout detenir {fin[0]:.2f}, optimum L* = {grid[i_f]:.1f} cout {fin[i_f]:.2f} "
      f"({100*(fin[i_f]/fin[0]-1):+.0f} %) ; seuil global {100*k_fin:.1f} %")
print(f"economique : tout detenir {eco[0]:.2f}, optimum L* = {grid[i_e]:.1f} cout {eco[i_e]:.2f} "
      f"({100*(eco[i_e]/eco[0]-1):+.0f} %) ; seuil global {100*k_eco:.1f} %")
mh = Moteur(lam_n / 2.0, m_n)
xh = mh.annuel(CEXP)
Bh = float(np.quantile(xh, ALPHA))
grid_n = grid
_, eco_h = couts(xh)
print(f"lambda divise par deux ({lam_n/2:.4f}) : besoin {Bh:.1f} M EUR ; cout economique optimal "
      f"{100*(eco_h.min()/eco.min()-1):+.0f} %")

# =====================================================================================
titre("7. Les entites, modele retenu face a la lecture du registre")
# =====================================================================================
lignes = {}
print(f"{'entite':<23}{'actifs':>9}{'lambda':>8}{'besoin':>8}{'bande':>14}{'bande beta':>13}{'TVaR':>8}"
      f"{'/actifs':>9}{'/SCR':>7}{'registre':>9}{'C->NC':>8}")
prev = None
for nom, a_eur, scr in ENT:
    a = a_eur * TAUX_USD
    m = D.mult(a)
    me = Moteur(lam_new(a), m)
    x = me.annuel(CEXP)
    B = float(np.quantile(x, ALPHA))
    sv = [me.var(c) for c in CSOM]
    bb = sorted([Moteur(lam_new(a, b_lo), m).var(CEXP), Moteur(lam_new(a, b_hi), m).var(CEXP)])
    reg = Moteur(D.lam(a), m, publie=True).var(CEXP)
    bc = me.var(CG[0.45])
    lignes[nom] = dict(a=a_eur, scr=scr, lam=lam_new(a), B=B, lo=min(sv), hi=max(sv), bb=bb, tv=tvar(x),
                       E=float(x.mean()), reg=reg, bc=bc, pos=float((x > 0).mean()))
    el = f"  el. TVaR {np.log(tvar(x)/prev[1])/np.log(a_eur/prev[0]):.2f}" if prev else ""
    rs = f"{100*B/scr:>6.1f}%" if scr else "      -"
    ec_ = f"{100*(B/bc-1):>+7.0f}%" if bc > 0 else "      -"
    print(f"{nom:<23}{a_eur:>9,.0f}{lam_new(a):>8.4f}{B:>8.1f}"
          f"{'[' + format(min(sv), '.0f') + ' ; ' + format(max(sv), '.0f') + ']':>14}"
          f"{'[' + format(bb[0], '.0f') + ' ; ' + format(bb[1], '.0f') + ']':>13}{tvar(x):>8.1f}"
          f"{100*B/a_eur:>8.3f}%{rs}{reg:>9.1f}{ec_}{el}")
    prev = (a_eur, tvar(x))
pts = [lignes[n]["B"] for n, _, _ in ENT]
if not all(b >= a - 1e-9 for a, b in zip(pts, pts[1:])):
    sys.exit("le besoin n'est pas croissant avec la taille : arret.")
print("croissance du besoin avec les actifs : True")
la, ld = lignes["Assureur non-vie A"], lignes["Assureur vie D"]
print(f"assureur vie D / assureur non-vie A : besoin x{ld['B']/la['B']:.0f} (registre "
      f"x{ld['reg']/la['reg']:.1f}) pour des actifs x{ld['a']/la['a']:.0f} ; TVaR x{ld['tv']/la['tv']:.1f}")


def positif(a_eur):
    a = a_eur * TAUX_USD
    return Moteur(lam_new(a), D.mult(a)).var(CEXP) > 0


lo_, hi_ = 100.0, 2.0e5
for _ in range(16):
    mid = float(np.sqrt(lo_ * hi_))
    lo_, hi_ = (lo_, mid) if positif(mid) else (mid, hi_)
print(f"taille de bascule (plus petite taille a besoin positif a 99,5 %) : {hi_:,.0f} M EUR d'actifs")

# --- agregation au capital total (script 95), a l'echelle corrigee
titre("8. Agregation au capital total, a l'echelle corrigee")
txt95 = open(os.path.join(REPO, "resultats", "sorties_verif", "95.txt"), encoding="utf-8").read()
BSCR = {}
for nom in ("Assureur non-vie A", "Assureur non-vie B", "Assureur vie C", "Assureur vie D"):
    mm = re.search(re.escape(nom) + r"\s+[\d.]+\s+[\d.]+\s+\d+\s+(?:[\d.]+|non publie)\s+([\d.]+)", txt95)
    if not mm:
        sys.exit(f"BSCR de {nom} introuvable dans 95.txt : arret.")
    BSCR[nom] = float(mm.group(1))


def tau(rho, dd):
    return (np.sqrt(1 + 2 * rho * dd + dd * dd) - 1) / dd


print(f"{'entite':<23}{'BSCR >=':>9}{'ecart D':>9}{'d':>9}{'tau 0,25':>10}{'1/tau':>8}{'D/SCR':>8}"
      f"{'niveau 1/tau':>14}")
for nom in BSCR:
    l = lignes[nom]
    Dd = l["B"] - l["bc"]
    dd = Dd / BSCR[nom]
    dn = l["B"] / BSCR[nom]
    l["D"] = Dd
    print(f"{nom:<23}{BSCR[nom]:>9.1f}{Dd:>9.1f}{dd:>9.4f}{100*tau(0.25, dd):>9.1f}%{1/tau(0.25, dd):>8.2f}"
          f"{100*Dd/l['scr']:>7.2f}%{1/tau(0.25, dn):>14.2f}")
marge = 0.3 * BSCR["Assureur non-vie A"] - 59.0
print(f"assureur non-vie A, marge sous le plafond 0,3 BSCR : {marge:.1f} M EUR ; besoin = "
      f"{la['B']/marge:.2f} fois la marge, ecart = {la['D']/marge:.2f} fois")

# --- plafond par l'exposition (script 78), a l'echelle corrigee
titre("9. Plafond par l'exposition, a l'echelle corrigee")
GRILLE = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
print(f"{'entite':<23}{'besoin':>8}{'kappa a -50 %':>15}{'soit M EUR':>12}")
for nom, a_eur, _ in ENT[1:6]:
    a = a_eur * TAUX_USD
    v0 = lignes[nom]["B"]
    courbe = [(k, Moteur(lam_new(a), D.mult(a), cap=k * a_eur).var(CEXP)) for k in GRILLE]
    km = None
    for (ka, va), (kb, vb_) in zip(courbe, courbe[1:]):
        if v0 > 0 and (va - 0.5 * v0) * (vb_ - 0.5 * v0) <= 0 and vb_ != va:
            t = (0.5 * v0 - va) / (vb_ - va)
            km = float(np.exp(np.log(ka) + t * (np.log(kb) - np.log(ka))))
            break
    s1 = f"{100*km:.3f} %" if km else "hors grille"
    s2 = f"{km*a_eur:.0f}" if km else "-"
    print(f"{nom:<23}{v0:>8.1f}{s1:>15}{s2:>12}")

# =====================================================================================
titre("10. Grandeurs citees (sans separateur, pour le harnais)")
# =====================================================================================
print(f"beta {F['b']:.3f} ET {F['se']:.3f} beta300 {FITS[3e5]['b']:.3f} dispersion {F['r']:.3f}")
for s, Fs in FITS.items():
    print(f"exposant au seuil {s:.0f} : {Fs['b']:.3f} ET {Fs['se']:.3f}")
for lab, sel in classes:
    print(f"annees observees {lab} : {q[sel].ans.mean():.2f}")
print(f"notionnelle lambda {lam_n:.4f} besoin {Bn:.1f} bande {min(som_n):.1f} {max(som_n):.1f} "
      f"largeur {max(som_n)-min(som_n):.1f} bandebeta {vb[0]:.1f} {vb[1]:.1f} tvar {tvar(xn):.1f} "
      f"moyenne {En:.2f} sansincident {100*p0:.1f}")
print(f"notionnelle xi {Bxi[0]:.1f} {Bxi[1]:.1f} tvarxi {Txi[0]:.1f} {Txi[1]:.1f} beta300 {B300:.1f} "
      f"levier {100*Bn/LEVIER_SCR:.1f} pperte {100*(xn > 0).mean():.2f} beta300complet {B300c:.1f}")
print(f"secteur besoin {B_SECT:.0f} moyenne {E_SECT:.0f}")
print(f"rapports lambda {LAM_SECT/lam_n:.0f} moyenne {E_SECT/En:.0f} besoin {B_SECT/Bn:.0f} prc {341/lam_n:.0f}")
print(f"g 0.45 {Bg[0.45]:.1f} 0.68 {Bg[0.68]:.1f} 0.90 {Bg[0.90]:.1f} ecart {100*(Bg[0.90]/Bg[0.45]-1):.1f}")
print(f"elasticite d'arc {EL_ARC:.2f} ; g a 10 % d'ecart {G10:.2f}")
for lab, c, p_, nc in CORR:
    print(f"correspondance {lab} : {Bg[c]:.1f} {Bg[p_]:.1f} {Bg[nc]:.1f} {100*(Bg[nc]/Bg[c]-1):.1f}")
print(f"P1 muette {DEG[1.0][0]:.1f} bande {DEG[1.0][1]:.1f} {DEG[1.0][2]:.1f} variation "
      f"{100*(DEG[1.0][0]/DEG[0.0][0]-1):.1f} deplacement {d_a:.1f} rapport {d_a/larg:.2f}")
print(f"detention {COC*Bn:.2f} {COC_HIST*Bn:.2f} fois {COC*Bn/En:.2f} {COC_HIST*Bn/En:.2f} multiple {Bn/En:.1f}")
print(f"transfert financement {fin[0]:.2f} {fin[i_f]:.2f} {100*(fin[i_f]/fin[0]-1):.0f} seuil {100*k_fin:.1f}")
print(f"transfert economique {eco[0]:.2f} {eco[i_e]:.2f} {100*(eco[i_e]/eco[0]-1):.0f} seuil {100*k_eco:.1f}")
print(f"lambda moitie {lam_n/2:.4f} besoin {Bh:.1f} cout {100*(eco_h.min()/eco.min()-1):.0f}")
for nom, *_ in ENT:
    l = lignes[nom]
    print(f"{nom} : besoin {l['B']:.1f} bande {l['lo']:.0f} {l['hi']:.0f} bandebeta {l['bb'][0]:.0f} "
          f"{l['bb'][1]:.0f} tvar {l['tv']:.0f} registre {l['reg']:.1f} conforme {l['bc']:.1f} "
          f"pperte {100*l['pos']:.2f} ecartetats {100*(l['B']/l['bc']-1) if l['bc'] > 0 else 0:.0f}"
          + (f" ratio {100*l['B']/l['scr']:.1f}" if l["scr"] else ""))
print(f"bascule {hi_:.0f}")
