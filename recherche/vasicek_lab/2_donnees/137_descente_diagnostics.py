#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
137 : diagnostics de la descente d'echelle du script 136, demandes par la revue Codex du 29 octobre.

Le script 136 estime l'exposant de frequence sur les firmes de plus de 100 Md USD et donne 24,1 M EUR
a l'entite notionnelle, contre 169,0 par la methode publiee. Ce script ne remplace aucune de ses
sorties. Il repond a sept questions :

  1. la pente tient-elle a duree d'observation comparable, et avec un controle de l'annee ?
  2. d'ou vient le passage de 169 a 24 : frequence, dispersion ou pertes sous le seuil ?
     (plan factoriel 2 x 2 x 2, quatre graines) ;
  3. quelle incertitude sur le capital quand intercept, pente et dispersion sont reestimes
     ensemble par un bootstrap par firme ?
  4. le metier (assureur ou non) change-t-il la frequence a taille egale ?
  5. les populations de frequence et de severite se recouvrent-elles ?
  6. la taille de bascule simulee concorde-t-elle avec la forme fermee de P(N = 0) ?

Moteur. Les annees, les amorces, la cascade et les severites sont tirees a nombres communs : une
uniforme par annee fixe le nombre d'incidents par inversion de la loi binomiale negative, et chaque
incident de rang i lit les memes uniformes quelle que soit la configuration. Le couplage des comptes
est donc exact ; celui des severites l'est par rang d'incident, pas par annee, puisque changer la
frequence deplace les rangs. Le bruit de simulation est mesure sur quatre graines.

Controle. La configuration publiee (frequence du registre, dispersion posee, pertes sous le seuil
nulles) doit redonner 169,0 M EUR au bruit pres, et la configuration du script 136 doit redonner
24,1. Sinon le script s'arrete. Exige SAS_OpRisk_Global_Data_June_2026.xlsx.
"""

import os
import sys
import warnings
from itertools import permutations

import numpy as np
import pandas as pd
from scipy import optimize, special
from scipy.stats import genpareto, nbinom

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
import statsmodels.api as sm                                        # noqa: E402

TAUX_USD = 1.04
NY = 600_000
SEEDS = (20260721, 20260722, 20260723, 20260724)
ALPHA = 0.995
SEUIL = 1.0e5
SP = pid.PARAMS["OPRISK"]
U = SP["u"]
PHI = ec.PHI
POOL = 400_000
B_BOOT = 300
W_ = 92
NOTION_USD = 20_000.0
ENT = [("Assureur non-vie A", 2319.0), ("Assureur non-vie B", 5305.0),
       ("Entite notionnelle", NOTION_USD / TAUX_USD), ("Assureur vie C", 37845.0),
       ("Assureur vie D", 309800.0)]


def titre(s):
    print("\n" + "=" * W_ + f"\n{s}\n" + "=" * W_)


# ------------------------------------------------------------------ donnees
D = dsc.Descente()
pa = D.pa.copy()
win = pa.groupby("firm").size().rename("fen")
pa = pa.join(win, on="firm")
metier = D.fs.groupby("Firm Name")["Basel Business Line - Level 1"].agg(
    lambda s: s.value_counts().index[0])
pa["assur"] = pa.firm.map(metier).eq("Insurance").astype(float)

df = filter_finance(filter_cyber(load_clean(os.path.join(REPO, "data", "raw",
                                                         "SAS_OpRisk_Global_Data_June_2026.xlsx"))))
d = df[df.year.between(2000, 2026)]
if len(d) != 570:
    sys.exit(f"population de calibration differente du protocole : {len(d)} pertes au lieu de 570")
L_ = (d["loss"] * USD_EUR).to_numpy()
CORPS = np.sort(L_[L_ <= U])


def nll_nb(p, xc, k):
    a, b, lr = p
    r = np.exp(lr)
    mu = np.exp(np.clip(a + b * xc, -30, 30))
    return -np.sum(special.gammaln(k + r) - special.gammaln(r) - special.gammaln(k + 1)
                   + r * np.log(r / (r + mu)) + k * np.log(mu / (r + mu)))


def ajuste(x, k, x0=None):
    xb = float(x.mean())
    xc = x - xb
    x0 = np.array([np.log(k.mean()), 0.3, 0.0]) if x0 is None else x0
    o = optimize.minimize(nll_nb, x0, args=(xc, k), method="L-BFGS-B")
    meth = "L-BFGS-B"
    if not o.success:
        o = optimize.minimize(nll_nb, x0, args=(xc, k), method="Nelder-Mead",
                              options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 20000})
        meth = "Nelder-Mead"
    return dict(a=float(o.x[0]), b=float(o.x[1]), r=float(np.exp(o.x[2])), xb=xb,
                ok=bool(o.success), meth=meth)


def lam_of(fit, a_usd):
    return float(np.exp(fit["a"] + fit["b"] * (np.log(a_usd) - fit["xb"])))


big = pa[pa.actifs >= SEUIL]
F = ajuste(np.log(big.actifs.to_numpy()), big.n.to_numpy().astype(float))
if abs(F["b"] - 0.494) > 5e-4:
    sys.exit(f"l'ajustement de reference ne redonne pas beta = 0,494 du script 136 ({F['b']:.4f}) : arret.")

# =====================================================================================
titre("1. La pente a duree d'observation comparable, et avec un controle de l'annee")
# =====================================================================================
print("Fenetre d'observation par firme (annees), firmes de plus de 100 Md USD :")
fb = big.groupby("firm").fen.first()
print(f"  moyenne {fb.mean():.2f}, mediane {fb.median():.0f}, part observee un an seulement "
      f"{100*(fb == 1).mean():.1f} %")
print("Une fenetre d'un an n'est pas une observation longue : la restriction de taille ne garantit")
print("pas une observation exhaustive des grandes firmes.\n")


def nb_sm(sub, annee=False, assur=False):
    X = pd.DataFrame({"logA": np.log(sub.actifs.to_numpy())})
    X["logA"] -= X.logA.mean()
    if assur:
        X["assur"] = sub.assur.to_numpy()
    if annee:
        X = X.join(pd.get_dummies(sub.year.astype(int).to_numpy(), prefix="y", drop_first=True,
                                  dtype=float).set_index(X.index))
    X = sm.add_constant(X, has_constant="add")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = sm.NegativeBinomial(sub.n.to_numpy(), X, loglike_method="nb2")
        res = m.fit(method="bfgs", maxiter=2000, disp=0, cov_type="cluster",
                    cov_kwds={"groups": pd.factorize(sub.firm)[0]})
    conv = bool(res.mle_retvals.get("converged", False))
    return res, conv, len(w)


print(f"{'panel':<34}{'fenetre >=':>11}{'beta':>8}{'ET grappe':>11}{'lignes':>8}{'firmes':>8}"
      f"{'conv.':>7}{'avert.':>8}")
PENTES = {}
for lab, base in (("tout le panel", pa), (">= 100 Md USD", big)):
    for kmin in (1, 3, 5, 8, 12):
        sub = base[base.fen >= kmin]
        if sub.firm.nunique() < 20 or sub.n.sum() < 30:
            continue
        res, conv, nw = nb_sm(sub)
        PENTES[(lab, kmin, False)] = (res.params["logA"], res.bse["logA"])
        print(f"{lab:<34}{kmin:>11}{res.params['logA']:>8.3f}{res.bse['logA']:>11.3f}{len(sub):>8}"
              f"{sub.firm.nunique():>8}{str(conv):>7}{nw:>8}")
print("\navec effets fixes d'annee :")
for lab, base in (("tout le panel", pa), (">= 100 Md USD", big)):
    for kmin in (1, 5):
        sub = base[base.fen >= kmin]
        res, conv, nw = nb_sm(sub, annee=True)
        PENTES[(lab, kmin, True)] = (res.params["logA"], res.bse["logA"])
        print(f"{lab:<34}{kmin:>11}{res.params['logA']:>8.3f}{res.bse['logA']:>11.3f}{len(sub):>8}"
              f"{sub.firm.nunique():>8}{str(conv):>7}{nw:>8}")
res0, _, _ = nb_sm(big)
print(f"\n>= 100 Md USD, fenetre >= 1 : ET par hessienne naive (script 136) 0.043, ET par grappe "
      f"de firmes {res0.bse['logA']:.3f}")
b5, b1 = PENTES[("tout le panel", 5, False)][0], PENTES[("tout le panel", 1, False)][0]
print(f"Sur tout le panel, exiger une fenetre de 5 ans fait passer la pente de {b1:.3f} a {b5:.3f}.")
print("Ces restrictions ne fournissent pas un denominateur d'exposition externe : une annee sans")
print("evenement enregistre n'est pas necessairement une annee sans incident.")

# =====================================================================================
titre("2. Le moteur a nombres communs, et ses controles")
# =====================================================================================
W_AM = np.array([ec.eng.LAMBDA[j] for j in pid.PIL], float)
CDF_AM = np.cumsum(W_AM / W_AM.sum())
CEXP = pid.card_dist_all(pid.expert_matrix())[0]
CG45 = pid.card_dist_all(pid.expert_matrix(0.45))[0]


class Couple:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.Uy = rng.random(NY)
        self.Ua = rng.random(POOL)
        self.Uk = rng.random(POOL)
        self.Ut = rng.random((POOL, pid.NP_))
        self.Uq = rng.random((POOL, pid.NP_))
        self.Ub = rng.random((POOL, pid.NP_))

    def annuel(self, lam, r, corps, mult, card=None, xi=None):
        card = CEXP if card is None else card
        p = r / (r + lam)
        p0 = p ** r
        counts = np.zeros(NY, dtype=np.int64)
        hi = self.Uy > p0
        counts[hi] = nbinom.ppf(self.Uy[hi], r, p).astype(np.int64)
        T = int(counts.sum())
        if T > POOL:
            sys.exit(f"reserve de tirages insuffisante ({T} incidents) : arret.")
        year_of = np.repeat(np.arange(NY), counts)
        amorce = np.minimum(np.searchsorted(CDF_AM, self.Ua[:T], side="right"), pid.NP_ - 1)
        tail = self.Ut[:T] < SP["p_u"]
        sev = np.where(tail, U + genpareto.ppf(self.Uq[:T], c=SP["xi"] if xi is None else xi,
                                               scale=SP["sigma"]), 0.0)
        if corps:
            sev = np.where(tail, sev, CORPS[(self.Ub[:T] * CORPS.size).astype(np.int64)])
        cum = np.cumsum(sev * mult, axis=1)
        K = np.empty(T, dtype=np.int64)
        for a in range(pid.NP_):
            idx = np.where(amorce == a)[0]
            if idx.size:
                cdf = np.cumsum(card[a][1:])
                cdf[-1] = 1.0
                K[idx] = np.searchsorted(cdf, self.Uk[idx], side="right") + 1
        np.clip(K, 1, pid.NP_, out=K)
        return np.bincount(year_of, weights=cum[np.arange(T), K - 1], minlength=NY)


def stats_(x):
    s = np.sort(x)
    return dict(var=float(np.quantile(x, ALPHA)), tvar=float(s[-int(np.ceil((1 - ALPHA) * s.size)):].mean()),
                moy=float(x.mean()), pos=float((x > 0).mean()))


ENG = [Couple(s) for s in SEEDS]
a_n = NOTION_USD
m_n = D.mult(a_n)
lam_reg, lam_new = D.lam(a_n), lam_of(F, a_n)
FREQ = {"registre": lam_reg, "nouvelle": lam_new}
DISP = {"posee (phi 9,2)": lambda lam: lam / (PHI - 1.0), "estimee": lambda lam: F["r"]}
CORP = {"nulles": False, "empiriques": True}
print(f"pertes sous le seuil : {CORPS.size}, minimum {CORPS.min():.3f} M EUR (toutes strictement positives)")
print(f"entite notionnelle : lambda registre {lam_reg:.4f}, nouvelle {lam_new:.4f} ; "
      f"r pose {lam_reg/(PHI-1):.4f} (a lambda registre), r estime {F['r']:.3f}")

RES = {}
for fk, lam in FREQ.items():
    for dk, rf in DISP.items():
        for ck, cb in CORP.items():
            vals = [stats_(e.annuel(lam, rf(lam), cb, m_n)) for e in ENG]
            RES[(fk, dk, ck)] = {k: (np.mean([v[k] for v in vals]), np.std([v[k] for v in vals], ddof=1))
                                 for k in vals[0]}
ref_pub = RES[("registre", "posee (phi 9,2)", "nulles")]["var"]
ref_136 = RES[("nouvelle", "estimee", "empiriques")]["var"]
print(f"controle, configuration publiee : {ref_pub[0]:.1f} +- {ref_pub[1]:.1f} (publie 169,0)")
print(f"controle, configuration du script 136 : {ref_136[0]:.1f} +- {ref_136[1]:.1f} (script 136 : 24,1)")
if abs(ref_pub[0] - 169.0) > max(3 * ref_pub[1], 5.0) or abs(ref_136[0] - 24.1) > max(3 * ref_136[1], 1.0):
    sys.exit("le moteur a nombres communs ne retrouve pas les deux references : arret.")

# =====================================================================================
titre("3. Plan factoriel 2 x 2 x 2 a l'entite notionnelle (moyenne +- ET sur quatre graines)")
# =====================================================================================
print(f"{'frequence':<10}{'dispersion':<17}{'sous seuil':<12}{'VaR 99,5':>14}{'TVaR':>16}"
      f"{'perte moy.':>12}{'P(L>0)':>9}")
for (fk, dk, ck), v in RES.items():
    print(f"{fk:<10}{dk:<17}{ck:<12}{v['var'][0]:>8.1f} +-{v['var'][1]:>4.1f}"
          f"{v['tvar'][0]:>10.1f} +-{v['tvar'][1]:>4.1f}{v['moy'][0]:>12.3f}{100*v['pos'][0]:>8.2f}%")
print("\ncontraste additif de chaque facteur (moyenne des quatre differences nouvelle - ancienne,")
print("en M EUR ; configurations a VaR nulle incluses ; ces contrastes ne s'additionnent pas) :")
FACT = [("frequence", 0, ("registre", "nouvelle")), ("dispersion", 1, ("posee (phi 9,2)", "estimee")),
        ("sous seuil", 2, ("nulles", "empiriques"))]
EFF = {}
for nom, pos, (o, n) in FACT:
    for met in ("var", "tvar", "moy"):
        differences = []
        for k, v in RES.items():
            if k[pos] == n:
                ko = list(k)
                ko[pos] = o
                vo = RES[tuple(ko)][met][0]
                differences.append(v[met][0] - vo)
        if len(differences) != 4:
            sys.exit("plan factoriel incomplet : quatre contrastes attendus par facteur.")
        EFF[(nom, met)] = float(np.mean(differences))
    print(f"  {nom:<12} VaR {EFF[(nom, 'var')]:+.3f}   TVaR {EFF[(nom, 'tvar')]:+.3f}   "
          f"perte moyenne {EFF[(nom, 'moy')]:+.3f}")
print("\nles six chemins du publie au script 136, un facteur a la fois (VaR) :")
CONTRIB = {(nom, met): [] for nom, _, _ in FACT for met in ("var", "tvar", "moy")}
for ordre in permutations(range(3)):
    cur = ["registre", "posee (phi 9,2)", "nulles"]
    pas = [f"{RES[tuple(cur)]['var'][0]:.1f}"]
    for pos in ordre:
        avant = RES[tuple(cur)]
        cur[pos] = FACT[pos][2][1]
        apres = RES[tuple(cur)]
        for met in ("var", "tvar", "moy"):
            CONTRIB[(FACT[pos][0], met)].append(apres[met][0] - avant[met][0])
        pas.append(f"{FACT[pos][0]} {RES[tuple(cur)]['var'][0]:.1f}")
    print("  " + " -> ".join(pas))
ATTR = {k: float(np.mean(v)) for k, v in CONTRIB.items()}
print("\nattribution additive (moyenne des contributions marginales sur les six ordres, en M EUR) :")
for nom, _, _ in FACT:
    print(f"  {nom:<12} VaR {ATTR[(nom, 'var')]:+.3f}   TVaR {ATTR[(nom, 'tvar')]:+.3f}   "
          f"perte moyenne {ATTR[(nom, 'moy')]:+.3f}")
for met in ("var", "tvar", "moy"):
    delta = (RES[("nouvelle", "estimee", "empiriques")][met][0]
             - RES[("registre", "posee (phi 9,2)", "nulles")][met][0])
    if not np.isclose(sum(ATTR[(nom, met)] for nom, _, _ in FACT), delta,
                      rtol=1e-10, atol=1e-10):
        sys.exit(f"attribution additive incoherente pour {met} : arret.")
print("controle : les contributions s'additionnent a l'ecart total pour chaque mesure.")
p0_reg = (lambda r, l: (r / (r + l)) ** r)(lam_reg / (PHI - 1), lam_reg)
p0_new = (lambda r, l: (r / (r + l)) ** r)(F["r"], lam_new)
print(f"P(N >= 1) en forme fermee : registre et dispersion posee {100*(1-p0_reg):.2f} %, "
      f"nouvelle frequence et dispersion estimee {100*(1-p0_new):.2f} %")
print(f"incidents moyens par annee touchee : {lam_reg/(1-p0_reg):.2f} contre {lam_new/(1-p0_new):.2f}")
print("Le bruit de simulation est l'ecart-type entre graines ; l'ecart entre lignes est la variation")
print("de modele, a nombres communs.")

# =====================================================================================
titre("4. Bootstrap par firme : intercept, pente et dispersion reestimes ensemble")
# =====================================================================================
grp = {f: g for f, g in big.groupby("firm")}
firms = np.array(list(grp))
xs = {f: np.log(g.actifs.to_numpy()) for f, g in grp.items()}
ks = {f: g.n.to_numpy().astype(float) for f, g in grp.items()}
rng = np.random.default_rng(SEEDS[0])
e0 = ENG[0]
BOOT, ech_lbfgs, ech_tot = [], 0, 0
for _ in range(B_BOOT):
    tir = rng.choice(firms, size=firms.size, replace=True)
    x = np.concatenate([xs[f] for f in tir])
    k = np.concatenate([ks[f] for f in tir])
    fb_ = ajuste(x, k, x0=np.array([F["a"], F["b"], np.log(F["r"])]))
    ech_lbfgs += fb_["meth"] != "L-BFGS-B"
    if not fb_["ok"]:
        ech_tot += 1
        continue
    row = dict(b=fb_["b"], r=fb_["r"])
    for nom, a_eur in ENT:
        a = a_eur * TAUX_USD
        lam = lam_of(fb_, a)
        st = stats_(e0.annuel(lam, fb_["r"], True, D.mult(a)))
        row[nom] = (lam, st["var"], st["tvar"])
    BOOT.append(row)
print(f"{B_BOOT} tirages de firmes ; repli sur Nelder-Mead {ech_lbfgs} ; echecs de convergence "
      f"{ech_tot} (ecartes et comptes)")
bb = np.array([r_["b"] for r_ in BOOT])
rb = np.array([r_["r"] for r_ in BOOT])
print(f"beta : mediane {np.median(bb):.3f}, intervalle 2,5-97,5 % [{np.percentile(bb, 2.5):.3f} ; "
      f"{np.percentile(bb, 97.5):.3f}], ET {bb.std(ddof=1):.3f}")
print(f"r : mediane {np.median(rb):.3f}, intervalle [{np.percentile(rb, 2.5):.3f} ; {np.percentile(rb, 97.5):.3f}]")
print(f"\n{'entite':<22}{'lambda med.':>12}{'VaR med.':>10}{'VaR 2,5-97,5 %':>18}{'P(VaR=0)':>10}"
      f"{'TVaR med.':>11}{'TVaR 2,5-97,5 %':>18}")
BQ = {}
for nom, _ in ENT:
    lm = np.array([r_[nom][0] for r_ in BOOT])
    vv = np.array([r_[nom][1] for r_ in BOOT])
    tv = np.array([r_[nom][2] for r_ in BOOT])
    BQ[nom] = (np.percentile(vv, [2.5, 50, 97.5]), np.percentile(tv, [2.5, 50, 97.5]), (vv == 0).mean())
    print(f"{nom:<22}{np.median(lm):>12.4f}{np.median(vv):>10.1f}"
          f"{'[' + format(BQ[nom][0][0], '.1f') + ' ; ' + format(BQ[nom][0][2], '.1f') + ']':>18}"
          f"{100*BQ[nom][2]:>9.0f}%{np.median(tv):>11.1f}"
          f"{'[' + format(BQ[nom][1][0], '.0f') + ' ; ' + format(BQ[nom][1][2], '.0f') + ']':>18}")
print("\nCes bandes portent l'incertitude d'estimation de la frequence seule, a severite et graine")
print("fixees. Elles ne corrigent pas la selection du registre.")
print("\nsources d'incertitude separees, entite notionnelle :")
st_xi = [stats_(e0.annuel(lam_new, F["r"], True, m_n, xi=x)) for x in SP["xi_ic90"]]
mc = RES[("nouvelle", "estimee", "empiriques")]
print(f"  frequence (bootstrap par firme)  VaR [{BQ['Entite notionnelle'][0][0]:.1f} ; "
      f"{BQ['Entite notionnelle'][0][2]:.1f}]  TVaR [{BQ['Entite notionnelle'][1][0]:.0f} ; "
      f"{BQ['Entite notionnelle'][1][2]:.0f}]")
print(f"  severite (xi sur son IC 90 %)    VaR {st_xi[0]['var']:.1f} a {st_xi[1]['var']:.1f}  "
      f"TVaR {st_xi[0]['tvar']:.0f} a {st_xi[1]['tvar']:.0f}")
print(f"  simulation (quatre graines)      VaR +- {mc['var'][1]:.1f}  TVaR +- {mc['tvar'][1]:.1f}")

# =====================================================================================
titre("5. Le metier : assureur ou non, a taille egale")
# =====================================================================================
print("Variable : ligne de metier baloise la plus frequente parmi les evenements de la firme")
print("(Insurance ou non). C'est un indicateur imparfait du metier, pas une classification de l'entite.")
for lab, base in (("tout le panel", pa), (">= 100 Md USD", big)):
    res, conv, nw = nb_sm(base, assur=True)
    na = int(base[base.assur == 1].firm.nunique())
    print(f"  {lab:<16} firmes assureurs {na:>5} / {base.firm.nunique():>5} ; beta {res.params['logA']:.3f} ; "
          f"effet assureur {res.params['assur']:+.3f} (ET grappe {res.bse['assur']:.3f}), "
          f"rapport des frequences {np.exp(res.params['assur']):.2f} ; converge {conv}")
# validation croisee par firme, 5 plis, sur les grandes firmes
rng = np.random.default_rng(SEEDS[1])
pli = dict(zip(firms, rng.integers(0, 5, firms.size)))
big = big.assign(pli=big.firm.map(pli))
gain = []
for f_ in range(5):
    tr, te = big[big.pli != f_], big[big.pli == f_]
    sc = []
    for assur in (False, True):
        res, _, _ = nb_sm(tr, assur=assur)
        X = pd.DataFrame({"logA": np.log(te.actifs.to_numpy()) - np.log(tr.actifs.to_numpy()).mean()})
        if assur:
            X["assur"] = te.assur.to_numpy()
        X = sm.add_constant(X, has_constant="add")
        beta_ = res.params.drop("alpha").to_numpy()
        mu = np.exp(X.to_numpy() @ beta_)
        r_ = 1.0 / res.params["alpha"]
        k = te.n.to_numpy().astype(float)
        sc.append(float(np.sum(special.gammaln(k + r_) - special.gammaln(r_) - special.gammaln(k + 1)
                               + r_ * np.log(r_ / (r_ + mu)) + k * np.log(mu / (r_ + mu)))))
    gain.append(sc[1] - sc[0])
print(f"validation croisee par firme (5 plis, >= 100 Md USD) : gain de log-vraisemblance hors "
      f"echantillon de l'effet assureur {sum(gain):+.2f} nats au total, {sum(g > 0 for g in gain)} plis sur 5 positifs")
resA, _, _ = nb_sm(big.drop(columns="pli"), assur=True)
xbA = float(np.log(big.actifs.to_numpy()).mean())
rA = 1.0 / resA.params["alpha"]
print(f"\nSENSIBILITE, NON RETENUE : les entites lues comme assureurs (effet {resA.params['assur']:+.3f}, "
      f"r {rA:.3f})")
print(f"{'entite':<22}{'lambda':>9}{'VaR':>14}{'TVaR':>16}{'P(L>0)':>9}   sans effet metier")
METIER = {}
for nom, a_eur in ENT:
    a = a_eur * TAUX_USD
    lamA = float(np.exp(resA.params["const"] + resA.params["logA"] * (np.log(a) - xbA) + resA.params["assur"]))
    sts = [stats_(e.annuel(lamA, rA, True, D.mult(a))) for e in ENG]
    s0 = [stats_(e.annuel(lam_of(F, a), F["r"], True, D.mult(a))) for e in ENG]
    v, t = np.mean([s["var"] for s in sts]), np.mean([s["tvar"] for s in sts])
    METIER[nom] = (lamA, v, t, np.mean([s["var"] for s in s0]), np.mean([s["tvar"] for s in s0]))
    print(f"{nom:<22}{lamA:>9.4f}{v:>8.1f} +-{np.std([s['var'] for s in sts], ddof=1):>4.1f}"
          f"{t:>10.1f} +-{np.std([s['tvar'] for s in sts], ddof=1):>4.1f}"
          f"{100*np.mean([s['pos'] for s in sts]):>8.2f}%   VaR {METIER[nom][3]:.1f}, TVaR {METIER[nom][4]:.1f}")
print("L'indicateur de metier est celui des evenements enregistres : il peut traduire une moindre")
print("frequence reelle comme une moindre declaration. L'effet n'est pas integre a la variante du 136.")

# =====================================================================================
titre("6. Populations de frequence et de severite")
# =====================================================================================
ict = D.ict
idf = set(ict["Reference ID Code"])
d0522 = d[d.year.between(dsc.Y0, dsc.Y1)]
ids = set(d0522["Reference ID Code"])
print(f"frequence : {len(idf)} evenements TIC (5 sous-categories, secteur financier hors Non-FS, "
      f"{dsc.Y0}-{dsc.Y1})")
print(f"severite  : {len(ids)} pertes cyber-finance sur {dsc.Y0}-{dsc.Y1} ({len(d)} sur 2000-2026)")
print(f"recouvrement : {len(idf & ids)} evenements communs, soit {100*len(idf & ids)/len(idf):.1f} % "
      f"de la frequence et {100*len(idf & ids)/len(ids):.1f} % de la severite")
print(f"pertes sous le seuil GPD : {CORPS.size} sur {len(d)} ({100*CORPS.size/len(d):.1f} %) ; "
      f"part au-dessus du seuil {100*(1-CORPS.size/len(d)):.2f} % contre p_u = {100*SP['p_u']:.2f} %")
print(f"plus petite perte de la population de severite : {L_.min():.3f} M EUR. Le registre ne collecte")
print("que des pertes publiques : le corps empirique n'est pas l'ensemble des petites pertes reelles,")
print("et il est repris d'une population melangee puis mis a l'echelle par m(A), sans validation par entite.")

# =====================================================================================
titre("7. Taille de bascule : forme fermee contre simulation")
# =====================================================================================
r0 = F["r"]
lam_star = r0 * ((ALPHA) ** (-1.0 / r0) - 1.0)
a_star = np.exp(F["xb"] + (np.log(lam_star) - F["a"]) / F["b"]) / TAUX_USD
print("Toutes les pertes d'incident etant strictement positives, L > 0 si et seulement si N >= 1, et")
print("P(N = 0) = (r / (r + lambda))^r. Le besoin a 99,5 % devient positif quand P(N = 0) < 0,995,")
print(f"soit lambda > {lam_star:.5f} : taille de bascule {a_star:,.0f} M EUR d'actifs.")


def bascule(eng):
    lo_, hi_ = 100.0, 2.0e5
    for _ in range(20):
        mid = float(np.sqrt(lo_ * hi_))
        a = mid * TAUX_USD
        v = np.quantile(eng.annuel(lam_of(F, a), r0, True, D.mult(a)), ALPHA)
        lo_, hi_ = (lo_, mid) if v > 0 else (mid, hi_)
    return hi_


bas = [bascule(e) for e in ENG]
print(f"simulation, quatre graines : {', '.join(f'{b:,.0f}' for b in bas)} M EUR ; script 136 : 1 015")
bst = []
rng = np.random.default_rng(SEEDS[2])
for _ in range(B_BOOT):
    tir = rng.choice(firms, size=firms.size, replace=True)
    fb_ = ajuste(np.concatenate([xs[f] for f in tir]), np.concatenate([ks[f] for f in tir]),
                 x0=np.array([F["a"], F["b"], np.log(F["r"])]))
    if fb_["ok"]:
        ls = fb_["r"] * (ALPHA ** (-1.0 / fb_["r"]) - 1.0)
        bst.append(np.exp(fb_["xb"] + (np.log(ls) - fb_["a"]) / fb_["b"]) / TAUX_USD)
print(f"forme fermee sous le bootstrap par firme : mediane {np.median(bst):,.0f}, intervalle 2,5-97,5 % "
      f"[{np.percentile(bst, 2.5):,.0f} ; {np.percentile(bst, 97.5):,.0f}] M EUR")
print("La bascule ne se donne donc qu'en ordre de grandeur.")

# =====================================================================================
titre("8. Grandeurs citees (sans separateur, pour le harnais)")
# =====================================================================================
for (lab, kmin, an), (b_, s_) in PENTES.items():
    print(f"pente {lab} fenetre {kmin} annee {an} : {b_:.3f} ET {s_:.3f}")
print(f"grandes firmes fenetre moyenne {fb.mean():.2f} mediane {fb.median():.0f} un an {100*(fb == 1).mean():.1f}")
print(f"ET grappe {res0.bse['logA']:.3f}")
for (fk, dk, ck), v in RES.items():
    print(f"factoriel {fk} {dk} {ck} : var {v['var'][0]:.1f} et {v['var'][1]:.1f} tvar {v['tvar'][0]:.1f} "
          f"moy {v['moy'][0]:.3f} pos {100*v['pos'][0]:.2f}")
for (nom, met), e in EFF.items():
    print(f"contraste additif {nom} {met} {e:+.3f}")
for (nom, met), e in ATTR.items():
    print(f"attribution six ordres {nom} {met} {e:+.3f}")
print(f"pn1 registre {100*(1-p0_reg):.2f} nouvelle {100*(1-p0_new):.2f}")
print(f"boot beta {np.median(bb):.3f} {np.percentile(bb, 2.5):.3f} {np.percentile(bb, 97.5):.3f} echecs {ech_tot}")
for nom, (qv, qt, z) in BQ.items():
    print(f"boot {nom} var {qv[0]:.1f} {qv[1]:.1f} {qv[2]:.1f} tvar {qt[0]:.0f} {qt[1]:.0f} {qt[2]:.0f} zero {100*z:.0f}")
print(f"bascule fermee {a_star:.0f} simulee {min(bas):.0f} {max(bas):.0f} boot {np.median(bst):.0f} "
      f"{np.percentile(bst, 2.5):.0f} {np.percentile(bst, 97.5):.0f}")
print(f"metier effet {resA.params['assur']:.3f} ET {resA.bse['assur']:.3f} "
      f"rapport {np.exp(resA.params['assur']):.2f} inverse {np.exp(-resA.params['assur']):.1f} cv {sum(gain):.2f}")
for nom, (lamA, v, t, v0, t0) in METIER.items():
    print(f"metier {nom} lambda {lamA:.4f} var {v:.1f} tvar {t:.1f} sans {v0:.1f} {t0:.1f}")
print(f"recouvrement {len(idf & ids)} {100*len(idf & ids)/len(idf):.1f} {100*len(idf & ids)/len(ids):.1f}")
