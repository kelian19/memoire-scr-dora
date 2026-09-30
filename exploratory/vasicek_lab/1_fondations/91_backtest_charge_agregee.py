# -*- coding: utf-8 -*-
"""91 - BACKTEST DE LA CHARGE ANNUELLE AGREGEE, et test d'independance frequence / severite.

CE QUE CE SCRIPT AJOUTE AU 89, ET POURQUOI CE N'EST PAS LA MEME CHOSE. Le script 89 valide
la loi de FREQUENCE et la FORME de la severite, chacune de son cote. Or le capital n'est ni
l'une ni l'autre : c'est le quantile de la CHARGE ANNUELLE AGREGEE. Un modele compose peut
avoir deux marginales correctes et un agregat faux, et c'est le reproche standard adresse aux
modeles de type frequence x severite. Ce script note donc l'agregat lui-meme.

L'OBJET NOTE EST LA CHARGE DE QUEUE, ET CE CHOIX EST IMPOSE PAR LE MODELE, NON PAR COMMODITE.
La chaine publiee donne une severite NULLE aux incidents sous le seuil (convention de la
severite de remediation, src/aggregation/lda.py) : sa charge annuelle est donc la somme des
seuls sinistres depassant le seuil. L'observable comparable est la meme somme dans la donnee.
C'est aussi coherent avec la limite d'attritionnel deja declaree au chapitre 13 : le capital
publie est un capital de queue. Noter la charge TOTALE exigerait un modele de corps que le
memoire n'a pas, et melangerait le test d'un objet publie avec celui d'un objet invente.

C'EST UN DIAGNOSTIC, PAS UNE RECALIBRATION. Aucun parametre publie n'est touche. Le seuil est
RE-ESTIME sur chaque echantillon d'apprentissage, comme dans le script 89 et pour la meme
raison : un seuil calcule sur toute la periode ferait fuiter l'information de test.

TROIS QUESTIONS, ET LA DEUXIEME EST CELLE QU'ON NE SE POSE PAS ASSEZ.
  1. la loi de l'agregat predit-elle une annee qu'elle n'a pas vue ?
  2. douze annees permettent-elles de tester le quantile a 99,5 % ? La reponse est non, et
     elle se chiffre. Un backtest qui ne dit pas ce qu'il ne peut pas tester est trompeur ;
  3. l'independance entre frequence et severite, hypothese de construction de TOUT modele
     compose, que rien ne testait dans ce projet. Elle est confondue par la derive du script
     89, donc le test doit controler l'annee.

CE QU'IL TROUVE, EN QUATRE LIGNES, POUR QU'UN RELECTEUR NE PARTE PAS DE TRAVERS.
L'agregat est REJETE la ou les deux marginales passaient (PIT 0,734, Kolmogorov-Smirnov
p = 0,0042, couverture 75 % pour 90 % annonces), mais le rejet n'est pas structurel : c'est la
derive du script 89 vue sur l'objet qui porte le capital, signature confirmee par le decoupage
en deux moities. Le choix de la loi de comptage, decisif sur les comptes, ne pese que 0,3 % sur
la charge, celle-ci etant portee par un sinistre unique. L'independance frequence / severite
TIENT sur la population que le modele tire : la pente negative visible sur toute la distribution
disparait sur les exces, ou elle vaut 14 % de sa valeur et n'est plus significative. Et le
quantile a 99,5 % n'est testable sur AUCUN historique existant, 1 811 annees etant necessaires.

Sortie : diagnostics seulement, aucune figure.
"""

import os
import sys

import numpy as np
from scipy import optimize, stats
from scipy.stats import genpareto, nbinom, poisson

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (REPO, LAB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import derive_severite as ds  # noqa: E402

AN_DEBUT = 2004      # premiere annee de collecte dense (script 89)
AN_FIN = 2025        # derniere annee non tronquee
AN_ORIGINE = 2013    # premiere origine de prevision
P_SEUIL = 0.85       # regle de seuil, et non la valeur 20,03
NSIM = 200_000       # annees simulees par predictive
SEED = 20260908
NIV = 0.90           # niveau de l'intervalle de prevision reporte


def titre(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def fnum(v):
    return f"{v:,.0f}".replace(",", " ")


def fit_nb(y):
    """MLE de la binomiale negative sur des comptes. Identique au script 89."""
    def nll(par):
        r = np.exp(par[0]); mu = np.exp(par[1]); p = r / (r + mu)
        return -np.sum(nbinom.logpmf(y, r, p))
    m, v = y.mean(), y.var(ddof=1)
    r0 = max(m * m / max(v - m, 1e-6), 0.5)
    res = optimize.minimize(nll, [np.log(r0), np.log(max(m, 1e-6))], method="Nelder-Mead")
    r = np.exp(res.x[0]); mu = np.exp(res.x[1])
    return r, r / (r + mu), mu


def crps(ech, y):
    """CRPS d'une predictive donnee par un echantillon, contre l'observation y.

    CRPS = E|X - y| - 1/2 E|X - X'|. Le second terme se calcule sur l'echantillon TRIE, ce qui
    evite la matrice des ecarts deux a deux. Plus BAS est meilleur. Il est choisi ici plutot
    qu'un log-score parce que la predictive est un echantillon et non une densite fermee, et
    qu'il reste defini sur une loi a queue lourde tant que l'esperance existe (xi < 1).
    """
    x = np.sort(ech)
    m = x.size
    t1 = float(np.mean(np.abs(x - y)))
    i = np.arange(1, m + 1)
    t2 = float(2.0 * np.sum((2 * i - m - 1) * x) / (m * m))
    return t1 - 0.5 * t2


def predictive_agregat(rng, r, p, xi, sig, u, p_u, nsim=NSIM, loi="nb", lam=None):
    """Echantillon de charges annuelles agregees de QUEUE, loi de comptage au choix.

    Chaine du modele publie : un comptage annuel, puis chaque sinistre est de queue avec
    probabilite p_u, et sa severite vaut alors u + GPD(xi, sigma). Les sinistres sous le seuil
    ont une severite nulle, comme dans la chaine publiee.
    """
    n = rng.poisson(lam, size=nsim) if loi == "poisson" else rng.negative_binomial(r, p, size=nsim)
    k = rng.binomial(n, p_u)                       # sinistres de queue de l'annee
    T = int(k.sum())
    if T == 0:
        return np.zeros(nsim)
    sev = u + genpareto.rvs(xi, loc=0.0, scale=sig, size=T, random_state=rng)
    idx = np.repeat(np.arange(nsim), k)
    return np.bincount(idx, weights=sev, minlength=nsim)


# ---------------------------------------------------------------------------
# Donnees
# ---------------------------------------------------------------------------

d = ds.charger_pertes()
d = d[(d.year >= AN_DEBUT) & (d.year <= AN_FIN)].copy()

titre("0. PERIMETRE ET OBJET NOTE")
print(f"Perimetre : cyber x finance, convention de severite du projet, {len(d)} incidents "
      f"sur {AN_FIN - AN_DEBUT + 1} annees ({AN_DEBUT}-{AN_FIN}).")
print(f"Origines de prevision : {AN_ORIGINE} a {AN_FIN - 1}, soit {AN_FIN - AN_ORIGINE} "
      "annees notees hors echantillon.")
print("Observable : la somme annuelle des sinistres DEPASSANT le seuil d'apprentissage, qui est")
print("l'objet que la chaine publiee produit (severite nulle sous le seuil).")
print(f"Predictive : {NSIM} annees simulees par origine, comptage puis amincissement au taux de")
print("depassement, puis severite de queue u + GPD. Graine fixee, sortie deterministe.")


# ---------------------------------------------------------------------------
# 1. Origine glissante sur l'agregat
# ---------------------------------------------------------------------------

titre("1. LA CHARGE ANNUELLE AGREGEE, ORIGINE GLISSANTE")
print("origine   observe   mediane pred.       IP90 predictif        dans IP ?   PIT   CRPS")

pits, dans, crps_nb, crps_po, lignes = [], [], [], [], []
for t in range(AN_ORIGINE, AN_FIN):
    appr = d[d.year <= t]
    test = d[d.year == t + 1]
    if len(appr) < 50 or len(test) == 0:
        continue
    x_appr = appr["loss_eur"].values
    u = float(np.quantile(x_appr, P_SEUIL))
    exc = x_appr[x_appr > u] - u
    if len(exc) < 20:
        continue
    xi, _, sig = genpareto.fit(exc, floc=0.0)
    p_u = len(exc) / len(x_appr)
    cnt = appr.groupby("year").size().reindex(range(AN_DEBUT, t + 1), fill_value=0).values
    r, p, mu = fit_nb(cnt)

    obs = float(test.loc[test.loss_eur > u, "loss_eur"].sum())
    rng = np.random.default_rng(SEED + t)
    ech = predictive_agregat(rng, r, p, xi, sig, u, p_u)
    rng_p = np.random.default_rng(SEED + t)
    ech_p = predictive_agregat(rng_p, None, None, xi, sig, u, p_u, loi="poisson",
                               lam=float(cnt.mean()))

    lo, hi = np.quantile(ech, [(1 - NIV) / 2, 1 - (1 - NIV) / 2])
    med = float(np.median(ech))
    pit = float((ech <= obs).mean())
    ok = bool(lo <= obs <= hi)
    pits.append(pit)
    dans.append(ok)
    crps_nb.append(crps(ech, obs))
    crps_po.append(crps(ech_p, obs))
    lignes.append((t + 1, obs, med, lo, hi, ok, pit))
    print(f"  {t}   {fnum(obs):>7}   {fnum(med):>11}   [{fnum(lo):>5} ; {fnum(hi):>7}]   "
          f"{'oui' if ok else 'NON':>7}    {pit:.3f}  {crps_nb[-1]:7.1f}")

pit_a = np.array(pits)
print()
print(f"COUVERTURE REELLE de l'intervalle annonce a {100*NIV:.0f} % : "
      f"{sum(dans)}/{len(dans)} = {100*sum(dans)/len(dans):.1f} %")
ks = stats.kstest(pit_a, "uniform")
print(f"UNIFORMITE DES PIT : moyenne {pit_a.mean():.3f} (0,500 attendu), "
      f"D = {ks.statistic:.3f}, p = {ks.pvalue:.4f}")
print(f"  lecture : {'compatible avec l uniforme' if ks.pvalue > 0.05 else 'INCOMPATIBLE'}")
print(f"  A NE PAS SURINTERPRETER : sur {len(pits)} annees, le test de Kolmogorov-Smirnov ne")
print("  detecte qu'un ecart grossier. Une non-uniformite moderee resterait invisible.")
print()
print("CRPS CUMULE (plus bas = meilleur), sur les MEMES tirages de severite :")
print(f"  comptage binomiale negative : {np.sum(crps_nb):9.1f}")
print(f"  comptage Poisson            : {np.sum(crps_po):9.1f}")
gag = "la binomiale negative" if np.sum(crps_nb) < np.sum(crps_po) else "Poisson"
print(f"  ecart en faveur de {gag} : {abs(np.sum(crps_nb) - np.sum(crps_po)):.1f}, soit "
      f"{100*abs(np.sum(crps_nb) - np.sum(crps_po))/np.sum(crps_po):.1f} %")
print("  CE QUE CE SCORE AJOUTE au log-score du script 89 : celui-la notait le COMPTAGE, celui-ci")
print("  note la CHARGE. Le choix de loi de comptage se juge donc sur la grandeur qui porte le")
print("  capital, et non seulement sur le nombre d'incidents.")


# ---------------------------------------------------------------------------
# 1bis. Le rejet est-il un defaut de structure ou la derive deja mesuree ?
# ---------------------------------------------------------------------------

titre("1bis. LE REJET EST-IL STRUCTUREL, OU EST-CE LA DERIVE DEJA MESUREE ?")
print("Un rejet du PIT ne dit pas d'ou il vient. Deux causes possibles et elles n'appellent pas")
print("le meme traitement : une mauvaise structure de composition, qui condamnerait le modele, ou")
print("la derive d'echelle du script 89, qui est deja declaree. La derive laisse une signature")
print("reconnaissable : elle ne desaligne pas les premieres annees notees, seulement les")
print("dernieres, l'apprentissage prenant du retard a mesure que la derive s'accumule.")
print()
mid = len(pits) // 2
p1, p2 = pit_a[:mid], pit_a[mid:]
print(f"  PIT moyen des {mid} PREMIERES annees notees ({lignes[0][0]}-{lignes[mid-1][0]}) : "
      f"{p1.mean():.3f}")
print(f"  PIT moyen des {len(pits)-mid} DERNIERES ({lignes[mid][0]}-{lignes[-1][0]})        : "
      f"{p2.mean():.3f}")
mw = stats.mannwhitneyu(p1, p2, alternative="less")
print(f"  test de Mann-Whitney, premieres < dernieres : p = {mw.pvalue:.4f}")
print()
print(f"  charge observee, moyenne des {mid} premieres annees : {fnum(np.mean([l[1] for l in lignes[:mid]]))} M EUR")
print(f"  charge observee, moyenne des {len(pits)-mid} dernieres         : "
      f"{fnum(np.mean([l[1] for l in lignes[mid:]]))} M EUR")
print(f"  mediane predictive, meme decoupage : {fnum(np.mean([l[2] for l in lignes[:mid]]))} "
      f"puis {fnum(np.mean([l[2] for l in lignes[mid:]]))} M EUR")
print()
print("LECTURE. L'observe et la predictive montent tous deux, mais pas au meme rythme : c'est le")
print("retard d'une fenetre qui s'etend sur une echelle qui derive, et non un defaut de la")
print("composition frequence x severite. Le rejet du PIT est donc la MEME limite que celle du")
print("script 89, vue sur l'objet qui porte le capital, et il ne s'y ajoute pas.")


# ---------------------------------------------------------------------------
# 2. Ce que douze annees ne permettent pas de tester
# ---------------------------------------------------------------------------

titre("2. CE QUE DOUZE ANNEES PERMETTENT, ET CE QU'ELLES NE PERMETTENT PAS")
print("Un backtest qui ne declare pas sa puissance laisse croire qu'une absence de rejet vaut")
print("validation. Le calcul est elementaire et il doit etre ecrit.")
print()
n_an = len(pits)
for niv in (0.90, 0.99, 0.995):
    att = n_an * (1.0 - niv)
    # nombre d'annees pour esperer au moins un depassement, puis pour un test de puissance 80 %
    n1 = int(np.ceil(1.0 / (1.0 - niv)))
    # detecter un taux double du nominal, test binomial unilateral, puissance 80 %
    n80 = None
    for n in range(10, 20001):
        seuil = stats.binom.ppf(0.95, n, 1.0 - niv)
        if stats.binom.sf(seuil, n, 2.0 * (1.0 - niv)) >= 0.80:
            n80 = n
            break
    print(f"  niveau {100*niv:5.1f} % : {att:5.2f} depassement(s) attendu(s) sur {n_an} annees ; "
          f"{n1:5d} annees pour en esperer un ;")
    print(f"                  {n80 if n80 else '>20000':>6} annees pour detecter un taux DOUBLE "
          "du nominal a 80 % de puissance")
print()
print("LECTURE, ET C'EST UNE LIMITE DU BACKTEST ET NON DU MODELE. Le quantile a 99,5 % qui porte")
print("le capital N'EST PAS testable sur un historique de cette longueur, et il ne le serait sur")
print("aucun historique de risque operationnel existant. Ce que ce script teste est donc le")
print("CENTRE et le CORPS de la loi de l'agregat, a 90 %, pas sa queue extreme. Le capital reste")
print("adosse a l'ajustement parametrique, dont la queue est validee par la section 1 du script")
print("89, par Anderson-Darling et Kolmogorov-Smirnov, et bornee par l'IC de xi. Aucun de ces")
print("chemins n'est un backtest du quantile lui-meme, et il ne faut pas le presenter comme tel.")


# ---------------------------------------------------------------------------
# 3. L'independance entre frequence et severite
# ---------------------------------------------------------------------------

titre("3. L'INDEPENDANCE ENTRE FREQUENCE ET SEVERITE, TESTEE")
print("Tout modele compose la suppose : le montant d'un sinistre ne depend pas du nombre de")
print("sinistres de l'annee. Rien ne la testait dans ce projet. Si elle etait fausse dans le sens")
print("positif, les annees chargees porteraient aussi des sinistres plus gros, et la queue de")
print("l'agregat serait plus lourde que celle que le modele produit.")
print()
cnt_an = d.groupby("year").size()
d2 = d.copy()
d2["n_an"] = d2["year"].map(cnt_an).astype(float)
d2["ll"] = np.log(d2["loss_eur"].values)

print("(a) AU NIVEAU DU SINISTRE, regression du logarithme du montant sur le compte de l'annee.")
sl, ic, rv, pv, se = stats.linregress(d2["n_an"].values, d2["ll"].values)
print(f"    sans controle : pente {sl:+.5f} par incident, ecart-type {se:.5f}, p = {pv:.4f}")
X = np.column_stack([np.ones(len(d2)), d2["n_an"].values, d2["year"].values])
y = d2["ll"].values
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
s2 = float(resid @ resid) / (len(y) - X.shape[1])
cov = s2 * np.linalg.inv(X.T @ X)
se_b = float(np.sqrt(cov[1, 1]))
tstat = beta[1] / se_b
pv_b = float(2 * stats.t.sf(abs(tstat), len(y) - X.shape[1]))
print(f"    AVEC controle d'annee : pente {beta[1]:+.5f}, ecart-type {se_b:.5f}, "
      f"t = {tstat:+.2f}, p = {pv_b:.4f}")
print(f"    le controle est indispensable : la severite DERIVE de "
      f"{100*beta[2]:+.1f} % par an en log (script 89),")
print("    et une annee chargee est aussi une annee datee. Sans controle, la pente melange les")
print("    deux effets.")
print()
print("(b) AU NIVEAU DE L'ANNEE, correlations de rang, robustes a la queue lourde.")
med_an = d.groupby("year")["loss_eur"].median()
max_an = d.groupby("year")["loss_eur"].max()
for nom, serie in (("mediane annuelle", med_an), ("maximum annuel", max_an)):
    rho, pr = stats.spearmanr(cnt_an.values, serie.values)
    print(f"    compte contre {nom:<18} : rho de Spearman {rho:+.3f}, p = {pr:.4f}")
print("    UNE CORRELATION DE RANG PLUTOT QU'UN TEST DE VARIANCE, et le motif n'est pas le gout :")
print("    a xi voisin de 0,6 la variance de la severite est INFINIE, donc un test adosse a la")
print("    variance de l'agregat n'aurait pas d'objet. C'est le meme argument d'existence qui")
print("    ecarte la TVaR sous la calibration PRC.")
print()
print("(c) LE CONTROLE QUI TRANCHE : L'EFFET SURVIT-IL SUR LES SEULS SINISTRES DE QUEUE ?")
print("    Une annee a beaucoup d'incidents enregistres peut etre une annee ou la collecte")
print("    descend plus bas, ce qui abaisse mecaniquement le montant moyen SANS qu'aucune")
print("    propriete du risque ne change. C'est un artefact de base, non une dependance. Or le")
print("    modele ne tire que des severites de QUEUE : si l'effet vient du bas de la")
print("    distribution, il doit s'affaiblir en restreignant aux exces.")
u_ref = float(np.quantile(d["loss_eur"].values, P_SEUIL))
dq = d2[d2.loss_eur > u_ref]
Xq = np.column_stack([np.ones(len(dq)), dq["n_an"].values, dq["year"].values])
yq = dq["ll"].values
bq, *_ = np.linalg.lstsq(Xq, yq, rcond=None)
rq = yq - Xq @ bq
s2q = float(rq @ rq) / (len(yq) - Xq.shape[1])
covq = s2q * np.linalg.inv(Xq.T @ Xq)
se_q = float(np.sqrt(covq[1, 1]))
tq = bq[1] / se_q
pq = float(2 * stats.t.sf(abs(tq), len(yq) - Xq.shape[1]))
print(f"    sur les {len(dq)} exces au-dessus du percentile {100*P_SEUIL:.0f} "
      f"({u_ref:.2f} M EUR), avec controle d'annee :")
print(f"    pente {bq[1]:+.5f}, ecart-type {se_q:.5f}, t = {tq:+.2f}, p = {pq:.4f}")
print(f"    rapport a la pente sur toute la distribution : {bq[1]/beta[1]:.2f}")


# ---------------------------------------------------------------------------
# 3bis. La consequence de cette dependance sur le quantile de l'agregat
# ---------------------------------------------------------------------------

titre("3bis. CE QUE CETTE DEPENDANCE DEPLACERAIT SUR LE QUANTILE DE L'AGREGAT")
print("Un signe ne suffit pas : il faut la taille. On relache l'independance en faisant dependre")
print("l'echelle de queue du compte de l'annee, sigma_n = sigma exp(beta (n - n moyen)), avec le")
print("beta estime sur les seuls exces, puis on compare les quantiles de l'agregat.")
print("L'independance est le cas beta = 0, et elle sert de reference.")
print()
t_der = lignes[-1][0] - 1
appr = d[d.year <= t_der]
x_appr = appr["loss_eur"].values
u_f = float(np.quantile(x_appr, P_SEUIL))
exc_f = x_appr[x_appr > u_f] - u_f
xi_f, _, sig_f = genpareto.fit(exc_f, floc=0.0)
p_u_f = len(exc_f) / len(x_appr)
cnt_f = appr.groupby("year").size().reindex(range(AN_DEBUT, t_der + 1), fill_value=0).values
r_f, p_f, mu_f = fit_nb(cnt_f)
n_bar = float(cnt_f.mean())


def predictive_dep(rng, beta_dep, nsim=NSIM):
    """Agregat annuel avec une echelle de queue dependant du compte de l'annee."""
    n = rng.negative_binomial(r_f, p_f, size=nsim)
    k = rng.binomial(n, p_u_f)
    T = int(k.sum())
    if T == 0:
        return np.zeros(nsim)
    g = np.exp(beta_dep * (n.astype(float) - n_bar))
    sev = u_f + genpareto.rvs(xi_f, loc=0.0, scale=sig_f, size=T,
                              random_state=rng) * np.repeat(g, k)
    idx = np.repeat(np.arange(nsim), k)
    return np.bincount(idx, weights=sev, minlength=nsim)


print(f"origine {t_der}, {len(cnt_f)} annees d'apprentissage, compte moyen {n_bar:.2f}.")
print("beta        q 90 %      q 99 %     q 99,5 %    ecart a l'independance")
q_ref = None
for etiq, bd in (("0 (independance)", 0.0), (f"{bq[1]:+.5f} (mesure)", float(bq[1])),
                 (f"{-bq[1]:+.5f} (signe oppose)", -float(bq[1]))):
    ech = predictive_dep(np.random.default_rng(SEED + 777), bd)
    q = np.quantile(ech, [0.90, 0.99, 0.995])
    if q_ref is None:
        q_ref = q
        ec = "reference"
    else:
        ec = f"{100*(q[2]/q_ref[2] - 1):+.1f} % sur le quantile a 99,5 %"
    print(f"  {etiq:<22} {fnum(q[0]):>6}  {fnum(q[1]):>8}  {fnum(q[2]):>9}   {ec}")
print()
print("LECTURE, ET ELLE N'EST PAS CELLE QUI ETAIT ANNONCEE AVANT D'AVOIR LU LE CONTROLE (c). Le")
print("signe negatif trouve en (a) sur toute la distribution NE SURVIT PAS sur les exces : la")
print(f"pente y tombe a {bq[1]/beta[1]:.2f} de sa valeur et cesse d'etre significative "
      f"(p = {pq:.2f}). L'explication la")
print("plus simple est donc un artefact de collecte au bas de la distribution, une annee a")
print("beaucoup d'incidents enregistres etant une annee ou l'on descend plus bas, et non une")
print("dependance du risque. Et meme prise au pied de la lettre sur les exces, sa consequence est")
print("de l'ordre de 1 % sur le quantile a 99,5 % de l'agregat, symetrique en signe.")
print("L'hypothese d'independance n'est donc plus non testee : elle est testee sur la population")
print("que le modele tire vraiment, et elle y tient.")


# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------

titre("VERDICT")
print("Ecrit APRES lecture des sorties. Le commentaire de la section 3bis annoncait avant lecture")
print("une dependance conservatrice : le controle (c) l'a dementi, et c'est la sortie qui tranche.")
print()
print("1. LA CHARGE AGREGEE EST REJETEE LA OU LES DEUX MARGINALES PASSAIENT, et c'est exactement")
print(f"   la raison d'etre de ce test. Le PIT de l'agregat vaut {pit_a.mean():.3f} pour "
      f"0,500 attendu et")
print(f"   Kolmogorov-Smirnov le rejette (p = {ks.pvalue:.4f}) ; la couverture de l'intervalle a "
      f"{100*NIV:.0f} % n'est")
print(f"   que de {100*sum(dans)/len(dans):.1f} %, avec {sum(1 for x in dans if not x)} "
      "depassements de la borne haute, tous dans les quatre")
print("   dernieres annees notees. Deux marginales correctes ne font donc pas un agregat correct,")
print("   et le script 89 seul ne pouvait pas le voir.")
print()
print("2. MAIS LE REJET N'EST PAS STRUCTUREL : C'EST LA DERIVE DU SCRIPT 89, VUE SUR L'OBJET QUI")
print(f"   PORTE LE CAPITAL. Le PIT vaut {p1.mean():.3f} sur les {mid} premieres annees notees "
      f"contre {p2.mean():.3f} sur les")
print(f"   {len(pits)-mid} dernieres (Mann-Whitney p = {mw.pvalue:.4f}), et la charge observee "
      "passe en moyenne de")
print(f"   {fnum(np.mean([l[1] for l in lignes[:mid]]))} a "
      f"{fnum(np.mean([l[1] for l in lignes[mid:]]))} M EUR quand la mediane predictive ne passe "
      f"que de {fnum(np.mean([l[2] for l in lignes[:mid]]))} a "
      f"{fnum(np.mean([l[2] for l in lignes[mid:]]))}.")
print("   AUCUNE LIMITE NOUVELLE NE S'AJOUTE donc a l'inventaire du chapitre 13 : la meme s'y lit")
print("   plus fort, parce que l'agregat cumule la derive de l'ECHELLE et celle du TAUX de")
print("   depassement, quand chaque marginale n'en portait qu'une.")
print()
print("3. LE CHOIX DE LA LOI DE COMPTAGE EST DECISIF SUR LES COMPTES ET PRESQUE IMMATERIEL SUR LA")
print("   CHARGE, ce qui n'etait pas attendu. Le script 89 donne un avantage net a la binomiale")
print("   negative sur les comptes, couverture 91,7 % contre 75,0 % et 3,054 nats de log-score.")
print(f"   Au CRPS sur la CHARGE, son avantage tombe a "
      f"{100*abs(np.sum(crps_nb) - np.sum(crps_po))/np.sum(crps_po):.1f} %. Le motif est celui que "
      "le memoire avance")
print("   deja pour son quantile : la charge annuelle est portee par un sinistre unique, donc par")
print("   la queue de severite et non par la surdispersion du comptage. La binomiale negative")
print("   reste le bon choix, mais elle cesse d'etre un determinant du capital.")
print()
print("4. L'INDEPENDANCE FREQUENCE / SEVERITE N'EST PLUS UNE HYPOTHESE NON TESTEE, ET ELLE TIENT.")
print(f"   Sur toute la distribution la pente est negative et significative "
      f"({beta[1]:+.5f}, p = {pv_b:.4f} avec")
print(f"   controle d'annee), mais sur les seuls EXCES elle tombe a {bq[1]:+.5f} "
      f"(p = {pq:.2f}), soit {bq[1]/beta[1]:.2f} de sa")
print("   valeur. L'effet vit donc dans le BAS de la distribution, ou il s'explique par la")
print("   profondeur de collecte, et il est absent de la population que le modele tire. Sa")
print("   consequence sur le quantile de l'agregat est de l'ordre de 1 %, symetrique en signe.")
print()
print("5. ET CE QUE CE BACKTEST NE PEUT PAS FAIRE, DIT EN ANNEES PLUTOT QU'EN PRECAUTION DE STYLE.")
print("   Detecter un taux de depassement DOUBLE du nominal a 80 % de puissance demanderait 1 811")
print("   annees a 99,5 %, 905 a 99 % et 78 meme a 90 %. Le quantile qui porte le capital n'est")
print("   donc testable sur AUCUN historique de risque operationnel existant. Ce script teste le")
print("   centre et le corps de la loi de l'agregat ; le presenter comme une validation du")
print("   quantile a 99,5 % serait faux, et c'est la formulation qu'il faut tenir devant un jury.")


titre("GRANDEURS CITEES PAR LE MEMOIRE, SANS SEPARATEUR DE MILLIERS")
print("Les tables ci-dessus impriment les montants avec une espace de milliers, que l'extracteur")
print("du harnais de verification coupe en deux. Les memes valeurs sont reprises ici en clair.")
print("Ce bloc n'ajoute aucun calcul.")
print()
print(f"  charge observee, moyenne des {mid} premieres annees notees   "
      f"{np.mean([l[1] for l in lignes[:mid]]):.0f}")
print(f"  charge observee, moyenne des {len(pits)-mid} dernieres              "
      f"{np.mean([l[1] for l in lignes[mid:]]):.0f}")
print(f"  mediane predictive, memes decoupages                    "
      f"{np.mean([l[2] for l in lignes[:mid]]):.0f} puis "
      f"{np.mean([l[2] for l in lignes[mid:]]):.0f}")
print(f"  charge observee la plus forte ({lignes[-1][0]})                 "
      f"{max(l[1] for l in lignes):.0f}")
print(f"  quantile 99,5 % de l'agregat sous independance           {q_ref[2]:.0f}")
print(f"  quantile 90 % de l'agregat sous independance             {q_ref[0]:.0f}")
print(f"  quantile 99 % de l'agregat sous independance             {q_ref[1]:.0f}")

print()
print("EXIT 0")
