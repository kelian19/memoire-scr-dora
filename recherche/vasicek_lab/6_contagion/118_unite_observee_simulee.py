#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
118 : l'unite observee et l'unite simulee. De combien un sinistre simule pese-t-il plus qu'un
incident de la donnee ?

LA QUESTION, posee par une relecture du 19 octobre 2026. La loi de severite est calibree sur des
pertes d'INCIDENTS complets (base OpRisk). Le moteur tire ensuite, pour un sinistre qui touche k
piliers, k pertes independantes dans cette meme loi et les additionne. Si une perte historique
contient deja les consequences de plusieurs defaillances de controle, le sinistre simule est plus
lourd que l'incident observe. Le script 74 relache l'additivite par un exposant ; il ne dit pas de
combien la loi d'un sinistre simule s'ecarte de la loi calibree.

CE QUE CE SCRIPT MESURE, SANS RIEN RECALIBRER, ET SON PERIMETRE. Pour chaque etat (loi exacte du
nombre de piliers touches, reprise du script 74) et a taux de depassement p_u ELEMENTAIRE de base
(celui d'une severite par pilier touche, pas celui d'un incident complet), pour isoler l'effet du
cardinal :
  - les quantiles 99 %, 99,5 % et 99,9 % de la perte d'un sinistre simule, rapportes a ceux d'un
    incident unique tire dans la meme loi ;
  - la perte moyenne d'un sinistre, rapportee a celle d'un incident (elle vaut E[K] exactement) ;
  - la reference asymptotique d'une loi sous-exponentielle, E[K] puissance xi pour le quantile.

PERIMETRE : CASCADE SEULE, HORS CHOC COMMUN P4. La loi de K vient de `cascade_set_dist` aux deux
gains de propagation ; elle n'applique pas, a l'etat non conforme, le choc commun P4 du moteur des
quatre canaux (`table_p4_choc`, canaux_conformite.py). Les etats 'conforme' et 'non conforme' de ce
script ne sont donc pas ceux du chiffre de tete (E[K] y est plus grand, voir la revue du 20 octobre
2026) et ses rapports ne couvrent pas toute la multiplicite de piliers du moteur publie. Ils ne
servent pas a corriger le capital annuel.

CONTROLE ET ARRET DUR : le quantile a 99,5 % d'un incident unique simule doit redonner la forme
fermee u + GPD^{-1}(1 - 0,005 / p_u) a 2 % pres (a deux millions de tirages, le quantile
simule varie de 657 a 665 selon la graine : le bruit d'echantillon est de l'ordre du pour cent), et les moyennes de K celles du script 74.

Lecture : un ecart de NIVEAU entre sinistre simule et incident observe. Ce n'est pas un test contre
la donnee : la donnee ne dit pas combien de piliers chaque incident observe a touches.
Sortie : sorties_verif/118.txt. Aucune figure.
"""

import os
import re
import sys

import numpy as np
from scipy.stats import genpareto

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RACINE = os.path.abspath(os.path.join(HERE, "../.."))
for p in (HERE, RACINE):
    if p not in sys.path:
        sys.path.insert(0, p)
import canaux_conformite as cx                                  # noqa: E402
import scr_engine as eng                                        # noqa: E402
from src.aggregation.lda import simulate_remediation_severity   # noqa: E402

WID = 88
SEED = 20261019
N = 5_000_000                       # sinistres simules par etat
NIVEAUX = (0.99, 0.995, 0.999)
sp = cx.sp
PIL = cx.PIL
SORTIE_74 = os.path.join(RACINE, "resultats", "sorties_verif", "74.txt")


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


w = np.array([eng.LAMBDA[j] for j in PIL], float)
w = w / w.sum()


def loi_cardinal(g):
    """Recopie du script 74 : P(k piliers touches), exacte, ponderee par la propension d'amorce."""
    p = np.zeros(len(PIL) + 1)
    for c, j in enumerate(PIL):
        dist = eng.cascade_set_dist(j, g)
        tot = sum(dist.values())
        for s, pr in dist.items():
            p[len(s)] += w[c] * pr / tot
    return p


def loi_cardinal_choc(g, phi_cs):
    """Comme loi_cardinal, mais l'amorce P4 suit la table de choc commun du moteur des quatre canaux."""
    p = np.zeros(len(PIL) + 1)
    for c, j in enumerate(PIL):
        if j == cx.P4:
            ind, probs = cx.table_p4_choc(phi_cs)
            for r in range(len(probs)):
                p[int(ind[r].sum())] += w[c] * probs[r] / probs.sum()
        else:
            dist = eng.cascade_set_dist(j, g)
            tot = sum(dist.values())
            for s, pr in dist.items():
                p[len(s)] += w[c] * pr / tot
    return p


def main():
    lois = {"conforme": loi_cardinal(cx.G_C), "non conforme": loi_cardinal(cx.G_NC)}
    moy = {e: float(np.dot(np.arange(len(l)), l)) for e, l in lois.items()}
    t74 = open(SORTIE_74, encoding="utf-8-sig").read()
    m = re.search(r"moyenne\s+([\d.]+)\s+([\d.]+)", t74)
    if not m or abs(moy["conforme"] - float(m.group(1))) > 5e-4 or abs(moy["non conforme"] - float(m.group(2))) > 5e-4:
        sys.exit("ARRET : les moyennes de K ne redonnent pas celles du script 74.")

    rng = np.random.default_rng(SEED)
    p_u = sp["p_u"]
    unique = simulate_remediation_severity(N, sp["xi"], sp["sigma"], sp["u"], p_u, sp["cap"], rng)
    q_ferme = sp["u"] + genpareto.ppf(1 - 0.005 / p_u, c=sp["xi"], scale=sp["sigma"])
    q_sim = float(np.quantile(unique, 0.995))
    if abs(q_sim / q_ferme - 1) > 0.02:
        sys.exit(f"ARRET : quantile simule {q_sim:.1f} contre {q_ferme:.1f} en forme fermee.")

    titre("118 : sinistre simule contre incident unique, meme loi calibree (cascade seule, hors choc P4)")
    print(f"Loi calibree : xi = {sp['xi']}, sigma = {sp['sigma']}, u = {sp['u']}, p_u de base = {p_u}.")
    print(f"{N:_} sinistres par etat, graine {SEED}. Le taux de depassement ELEMENTAIRE (par pilier".replace("_", " "))
    print("touche) est tenu a sa valeur de base dans les deux etats, pour n'isoler que l'effet du nombre")
    print("de piliers touches. Cascade seule, hors choc commun P4 (le moteur des quatre canaux l'ajoute")
    print("a l'etat non conforme) : ces etats ne sont pas ceux du chiffre de tete.")
    print(f"\n  controle : quantile 99,5 % d'un incident unique {q_sim:.1f} contre {q_ferme:.1f} en forme")
    print(f"  fermee ; moyennes de K {moy['conforme']:.3f} et {moy['non conforme']:.3f}, comme le script 74.")

    titre("1. Rapports sinistre simule / incident unique (cascade seule, hors choc P4, p_u elementaire de base)")
    q_un = {a: float(np.quantile(unique, a)) for a in NIVEAUX}
    print(f"  {'etat':<14}{'E[K]':>7}{'E[K]^xi':>9}{'moyenne':>10}" + "".join(f"{'q ' + str(a):>11}" for a in NIVEAUX))
    res = {}
    for e, loi in lois.items():
        K = rng.choice(len(loi), size=N, p=loi / loi.sum())
        sev = simulate_remediation_severity(N * 5, sp["xi"], sp["sigma"], sp["u"], p_u, sp["cap"],
                                            rng).reshape(N, 5)
        cum = np.cumsum(sev, axis=1)
        X = cum[np.arange(N), K - 1]
        r_moy = X.mean() / unique.mean()
        r_q = {a: float(np.quantile(X, a)) / q_un[a] for a in NIVEAUX}
        res[e] = (r_moy, r_q)
        print(f"  {e:<14}{moy[e]:>7.3f}{moy[e] ** sp['xi']:>9.3f}{r_moy:>10.3f}"
              + "".join(f"{r_q[a]:>11.3f}" for a in NIVEAUX))
    print(f"\n  quantiles d'un incident unique : " + ", ".join(f"{100*a:g} % : {q_un[a]:.0f} M" for a in NIVEAUX))

    # CONTROLE SUPPLEMENTAIRE (revue du 20 octobre) : l'etat non conforme des quatre canaux ajoute le
    # choc commun P4. Les tirages ci-dessous viennent APRES tous les precedents : les nombres des
    # sections 1 et LECTURE ne changent pas.
    loi_choc = loi_cardinal_choc(cx.G_NC, cx.PHICS_NC)
    moy_choc = float(np.dot(np.arange(len(loi_choc)), loi_choc))
    K = rng.choice(len(loi_choc), size=N, p=loi_choc / loi_choc.sum())
    sev = simulate_remediation_severity(N * 5, sp["xi"], sp["sigma"], sp["u"], p_u, sp["cap"],
                                        rng).reshape(N, 5)
    X = np.cumsum(sev, axis=1)[np.arange(N), K - 1]
    r_choc = {a: float(np.quantile(X, a)) / q_un[a] for a in NIVEAUX}
    titre("1bis. CONTROLE : etat non conforme AVEC le choc commun P4 (comme le moteur des quatre canaux)")
    pk = {"conforme": float(lois["conforme"][2:].sum()), "non conforme (cascade seule)": float(lois["non conforme"][2:].sum()),
          "non conforme avec choc P4": float(loi_choc[2:].sum())}
    print("  Part des sinistres touchant plus d'un pilier (loi exacte) : " + " ; ".join(f"{k} {100 * v:.2f} %" for k, v in pk.items()))
    print(f"  E[K] = {moy_choc:.3f} piliers touches, contre {moy['non conforme']:.3f} sans le choc. Memes tirages de")
    print("  severite, meme p_u elementaire de base. Rapports sinistre simule / incident unique :")
    print("".join(f"  q {a} : {r_choc[a]:.3f}" for a in NIVEAUX))
    print("  Ce n'est pas un facteur de correction du capital annuel : il isole l'effet du nombre de")
    print("  piliers touches, a p_u elementaire fixe, et ne mesure ni un ecart aux pertes historiques")
    print("  ni l'effet sur le capital.")

    # INCIDENTS MATERIELS PAR AN (relecture du 20 octobre). Calcul exact, sans tirage : la
    # probabilite qu'un incident ait une perte positive est E[1 - (1 - p_u)^K], p_u s'appliquant
    # a chaque pilier touche. On la confronte au rythme observe dans la donnee de calibration.
    def p_incident(loi, pu):
        return float(sum(loi[k] * (1 - (1 - pu) ** k) for k in range(len(loi))))
    n_exc = cx.sp.get("n_exc", None)
    from src.utils.config import OPRISK
    obs = OPRISK["n_excess"] / OPRISK["n_years"]
    cas = [("conforme", cx.LAM_C, loi_cardinal(cx.G_C), cx.PU_C),
           ("parametres de base (p_u de base, g = 0,90)", cx.LAM_C, loi_cardinal(0.90), p_u),
           ("non conforme, choc P4 inclus", cx.LAM_NC, loi_choc, cx.PU_NC)]
    titre("1ter. INCIDENTS MATERIELS PAR AN : moteur contre donnee de calibration (calcul exact)")
    print(f"  donnee : {OPRISK['n_excess']} exces sur {OPRISK['n_years']} ans, soit {obs:.2f} incidents materiels par an")
    mat = {}
    for nom, lam, loi, pu in cas:
        q = p_incident(loi, pu)
        mat[nom] = (q, lam * q)
        print(f"  {nom:<44} P(incident materiel) {q:.4f}   {lam * q:.2f} par an   ({lam * q / obs:.2f} fois la donnee)")
    print("  Le taux p_u, calibre par incident, est applique a chaque pilier touche : le moteur produit")
    print("  plus d'incidents materiels par an que la donnee, des l'etat conforme. Sens : le niveau du")
    print("  capital est majore par rapport a la donnee ; ce n'est pas un facteur de correction.")

    titre("LECTURE")
    rc, rn = res["conforme"][1][0.995], res["non conforme"][1][0.995]
    print("  Perimetre : cascade seule, hors choc commun P4, a p_u elementaire de base fixe.")
    print(f"  Au quantile 99,5 %, un sinistre simule pese {rc:.2f} fois un incident unique a l'etat")
    print(f"  conforme et {rn:.2f} fois a l'etat non conforme de ce diagnostic. Si chaque perte")
    print("  observee couvrait deja toutes les consequences d'un incident, ces facteurs mesureraient")
    print("  le surplus de niveau du au changement d'unite. Le rapport des deux vaut")
    print(f"  {rn / rc:.2f} dans ce perimetre.")
    print("  Ce n'est pas un test contre la donnee : elle ne dit pas combien de piliers chaque")
    print("  incident observe a touches. Ce n'est pas non plus un facteur de correction du capital")
    print("  annuel, et il ne couvre pas le choc commun P4 de l'etat non conforme des quatre canaux.")
    print("  Aucune recalibration (gel du 7 aout 2026).")

    titre("GRANDEURS CITEES (sans separateur de milliers, pour le harnais)")
    for e, (r_moy, r_q) in res.items():
        print(f"  {e} moyenne {r_moy:.3f} " + " ".join(f"q{a} {r_q[a]:.3f}" for a in NIVEAUX))
    print(f"  rapport non conforme sur conforme a 99.5 {rn / rc:.3f}")
    print(f"  non conforme avec choc P4 moyenne K {moy_choc:.3f} " + " ".join(f"q{a} {r_choc[a]:.3f}" for a in NIVEAUX))
    print(f"  donnee incidents materiels par an {obs:.2f}")
    for nom, (q, n) in mat.items():
        print(f"  {nom} P(incident materiel) {q:.4f} par an {n:.2f} rapport {n / obs:.2f}")


if __name__ == "__main__":
    main()
