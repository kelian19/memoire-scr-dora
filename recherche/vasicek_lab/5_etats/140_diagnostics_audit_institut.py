#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
140 : trois diagnostics demandes par l'audit du 5 novembre 2026, et les citations externes.

Aucune constante du modele n'est modifiee. Le script lit canaux_conformite et partial_id,
les fait tourner sous d'autres conventions, et ne remplace aucun chiffre publie.

 1. ANCRAGE DES ETATS. Le moteur affecte la frequence observee a l'etat conforme et le taux
    de depassement observe a l'etat partiel. On deplace ces ancrages, multiplicateurs inchanges.
 2. PILIERS A QUATRE CANAUX. Shapley et ordre de remediation avec la frequence, la detection,
    la propagation et l'accumulation par pilier, amorces selon ROOT puis uniformes.
 3. DIRECTION A QUATRE CANAUX. Plage sur les 1 024 sommets, en cascade independante : la marche
    auto-evitante n'est pas definie quand l'emission d'un pilier depasse 1 a un sommet.
 4. Citations externes transcrites (statut : citation, non recalculable) et conversions.

Sortie : resultats/sorties_verif/140.txt. Duree : quelques minutes selon le nombre de coeurs.
"""

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import canaux_conformite as cx                                  # noqa: E402
import euro_cascade_model as ec                                 # noqa: E402
import partial_id as pid                                        # noqa: E402
import scr_engine as eng                                        # noqa: E402
from src.aggregation.lda import simulate_remediation_severity   # noqa: E402

sp = cx.sp
PIL = eng.PIL
XI = sp["xi"]
M_NC = cx.LAM_NC / cx.LAM_C
M_PC = ec.lambda_scenario("OPRISK", "S1_partiel", mode="center") / cx.LAM_C
NSEED_P = 8
NSEED_V = 4
NY = cx.NY
ROOT_W = np.array([eng.LAMBDA[j] for j in PIL], float)


def titre(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# ------------------------------------------------------------------ 2. moteur par pilier
def pertes_piliers(args):
    """Perte annuelle par pilier a quatre canaux. etat[c] vaut 1 si le pilier c est non conforme.

    Frequence : amorce du pilier j multipliee selon son etat. Propagation : gain du pilier source.
    Detection : p_u du pilier touche. Accumulation : si P4 est non conforme, un incident amorce en
    P4 devient un choc commun. Memes conventions que le script 16b, plus l'accumulation.
    """
    etat, part, seed = args
    lam_vec = np.array([cx.LAM_C * part[c] * (M_NC if etat[c] else 1.0) for c in range(5)])
    g_vec = {j: (cx.G_NC if etat[c] else cx.G_C) for c, j in enumerate(PIL)}
    pu = [cx.PU_NC if etat[c] else cx.PU_C for c in range(5)]
    rng = np.random.default_rng(seed)
    lam = float(lam_vec.sum())
    r = lam / (ec.PHI - 1.0)
    counts = rng.negative_binomial(r, r / (r + lam), size=NY)
    T = int(counts.sum())
    year_of = np.repeat(np.arange(NY), counts)
    amorce = rng.choice(5, size=T, p=lam_vec / lam)
    U = rng.random(T)
    SEV = np.column_stack([
        simulate_remediation_severity(T, XI, sp["sigma"], sp["u"], pu[c], sp["cap"], rng)
        for c in range(5)])
    tables = eng.build_cascade_tables(g_vec, mode="walk")
    annual = np.zeros(NY)
    for c, j in enumerate(PIL):
        idx = np.where(amorce == c)[0]
        if idx.size == 0:
            continue
        ind, probs = (cx.table_p4_choc(cx.PHICS_NC) if (j == cx.P4 and etat[c]) else tables[j])
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        sel = np.clip(np.searchsorted(cdf, U[idx], side="right"), 0, len(probs) - 1)
        annual += np.bincount(year_of[idx], weights=(SEV[idx] * ind[sel]).sum(axis=1), minlength=NY)
    return float(np.quantile(annual, 0.995))


def bits(m):
    return tuple((m >> c) & 1 for c in range(5))


def shapley(v):
    """v : dict masque -> capital. Shapley exact sur les cinq piliers."""
    phi = np.zeros(5)
    for c in range(5):
        for m in range(32):
            if m >> c & 1:
                continue
            s = bin(m).count("1")
            poids = math.factorial(s) * math.factorial(4 - s) / 120
            phi[c] += poids * (v[m | 1 << c] - v[m])
    return phi


def ordres(v):
    """Capital integre le long de chaque ordre : somme des capitaux avant chaque remediation."""
    out = {}
    for o in permutations(range(5)):
        m, tot = 31, 0.0
        for c in o:
            tot += v[m]
            m &= ~(1 << c)
        out[o] = tot
    return out


def nom(o):
    return ">".join(f"P{PIL[c]}" for c in o)


# ------------------------------------------------------------------ 3. evaluateur aux sommets
class Eval4:
    """Patron de pid.Evaluator, avec le p_u de l'etat. Cascade independante, cardinal seul."""

    def __init__(self, lam, p_u, seed):
        rng = np.random.default_rng(seed)
        r = lam / (ec.PHI - 1.0)
        counts = rng.negative_binomial(r, r / (r + lam), size=NY)
        self.T = int(counts.sum())
        self.year_of = np.repeat(np.arange(NY), counts)
        am = rng.choice(5, size=self.T, p=ROOT_W / ROOT_W.sum())
        self.u = rng.random(self.T)
        sev = simulate_remediation_severity(self.T * 5, XI, sp["sigma"], sp["u"], p_u, sp["cap"],
                                            rng).reshape(self.T, 5)
        self.cum = np.cumsum(sev, axis=1)
        self.idx = [np.where(am == a)[0] for a in range(5)]
        self.ar = np.arange(self.T)

    def var(self, card):
        K = np.empty(self.T, dtype=np.int64)
        for a in range(5):
            cdf = np.cumsum(card[a][1:])
            cdf[-1] = 1.0
            K[self.idx[a]] = np.searchsorted(cdf, self.u[self.idx[a]], side="right") + 1
        np.clip(K, 1, 5, out=K)
        an = np.bincount(self.year_of, weights=self.cum[self.ar, K - 1], minlength=NY)
        return float(np.quantile(an, 0.995))


def card_choc(card, phi):
    """Remplace la loi du cardinal d'un incident amorce en P4 par celle du choc commun."""
    out = np.array(card, float, copy=True)
    out[3] = 0.0
    for k in range(5):
        out[3][k + 1] = math.comb(4, k) * phi ** k * (1 - phi) ** (4 - k)
    return out


def sommets_graine(args):
    k, verts = args
    S9, _ = pid.decompose(pid.expert_matrix(cx.G_NC))
    S4, _ = pid.decompose(pid.expert_matrix(cx.G_C))
    enc, enn = Eval4(cx.LAM_C, cx.PU_C, cx.SEED0 + k), Eval4(cx.LAM_NC, cx.PU_NC, cx.SEED0 + k)
    res = []
    for a in verts:
        cc = pid.card_dist_all(pid.build_W(S4, a * cx.G_C / cx.G_NC))[0]
        cn = card_choc(pid.card_dist_all(pid.build_W(S9, a))[0], cx.PHICS_NC)
        res.append((enc.var(cc), enn.var(cn)))
    wc = pid.expert_matrix(cx.G_C)
    wn = pid.expert_matrix(cx.G_NC)
    exp = (enc.var(pid.card_dist_all(wc)[0]), enn.var(card_choc(pid.card_dist_all(wn)[0], cx.PHICS_NC)))
    return np.array(res), exp


XI_CAT = np.array([0.865, 1.031, 1.038, 1.355, 1.368])    # script 37, queues par categorie Bale


def scr37(xis):
    """Meme calcul que scr() du script 37, pour un vecteur de queues par pilier."""
    part = ROOT_W / ROOT_W.sum()
    lam_vec = {j: sp["lam_ref"] * part[c] for c, j in enumerate(PIL)}
    g_vec = {j: 0.90 for j in PIL}
    p_u_vec = {j: sp["p_u"] for j in PIL}
    xi_vec = {j: float(xis[c]) for c, j in enumerate(PIL)}
    rng = np.random.default_rng(909)
    return float(np.quantile(ec.simulate_euro_pp(lam_vec, g_vec, XI, sp["sigma"], sp["u"], p_u_vec,
                                                 sp["cap"], 30_000, rng, xi_vec=xi_vec), 0.995))


def main():
    # ====================================================== 1
    titre("1. Ancrage des etats, multiplicateurs inchanges (marche, memes graines que le 43)")
    print("controle : la ligne publiee doit redonner 6 049 et 20 188")
    cas = [
        ("publie : lambda obs. = C, p_u obs. = PC", cx.LAM_C, cx.LAM_NC, cx.PU_C, cx.PU_NC),
        ("lambda obs. = PC", cx.LAM_C / M_PC, cx.LAM_C * M_NC / M_PC, cx.PU_C, cx.PU_NC),
        ("lambda obs. = NC", cx.LAM_C / M_NC, cx.LAM_C, cx.PU_C, cx.PU_NC),
        ("p_u obs. = C", cx.LAM_C, cx.LAM_NC, cx.P_U0, cx.P_U0 * 1.20 / 0.85),
        ("p_u obs. = NC", cx.LAM_C, cx.LAM_NC, cx.P_U0 * 0.85 / 1.20, cx.P_U0),
        ("lambda obs. = NC, p_u obs. = NC", cx.LAM_C / M_NC, cx.LAM_C, cx.P_U0 * 0.85 / 1.20, cx.P_U0),
    ]
    print(f"multiplicateur partiel {M_PC:.4f}, non conforme {M_NC:.4f}")
    print(f"{'convention':<42}{'lam C':>7}{'lam NC':>7}{'B(C)':>8}{'B(NC)':>8}{'ecart':>8}{'fact.':>7}{'ET fact.':>9}")
    for lab, lc, ln, pc, pn in cas:
        a = cx.scr_par_graine(lc, cx.G_C, pc, None)
        b = cx.scr_par_graine(ln, cx.G_NC, pn, cx.PHICS_NC)
        if lab.startswith("publie") and (round(a.mean()), round(b.mean())) != (6049, 20188):
            raise RuntimeError("Le diagnostic ne reproduit pas les capitaux publiés 6 049 et 20 188.")
        print(f"{lab:<42}{lc:>7.2f}{ln:>7.2f}{a.mean():>8.0f}{b.mean():>8.0f}"
              f"{b.mean() - a.mean():>8.0f}{b.mean() / a.mean():>7.2f}{np.std(b / a, ddof=1):>9.2f}")

    # ====================================================== 2
    titre(f"2. Piliers a quatre canaux, {NSEED_P} graines de {NY} annees")
    parts = {"ROOT": ROOT_W / ROOT_W.sum(), "uniformes": np.full(5, 0.2)}
    with ProcessPoolExecutor() as pool:
        for lab, part in parts.items():
            jobs = [(bits(m), tuple(part), 909 + k) for k in range(NSEED_P) for m in range(32)]
            val = np.array(list(pool.map(pertes_piliers, jobs, chunksize=4))).reshape(NSEED_P, 32)
            phis = np.array([shapley({m: val[k, m] for m in range(32)}) for k in range(NSEED_P)])
            moy = {m: float(val[:, m].mean()) for m in range(32)}
            tot = moy[31] - moy[0]
            print(f"\namorces {lab} (parts {np.round(part, 3)})")
            print(f"  tous C {moy[0]:.0f}, tous NC {moy[31]:.0f}, ecart {tot:.0f}, facteur {moy[31] / moy[0]:.2f}")
            ph = phis.mean(axis=0)
            et = phis.std(axis=0, ddof=1)
            for c in np.argsort(-ph):
                print(f"  P{PIL[c]}  Shapley {ph[c]:>7.0f}  ET {et[c]:>5.0f}  part {100 * ph[c] / tot:>5.1f} %")
            tete = np.array([PIL[int(np.argmax(phis[k]))] for k in range(NSEED_P)])
            print("  pilier de tete par graine : " + " ".join(f"P{t}" for t in tete))
            print("  P1 - P4, moyenne et ET entre graines : "
                  f"{(phis[:, 0] - phis[:, 3]).mean():.0f} ; {(phis[:, 0] - phis[:, 3]).std(ddof=1):.0f}")
            od = ordres(moy)
            tri = sorted(od, key=od.get)
            print(f"  meilleur ordre {nom(tri[0])}, pire {nom(tri[-1])}, "
                  f"ecart {100 * (od[tri[-1]] / od[tri[0]] - 1):.1f} %")
            print("  cinq meilleurs : " + " ; ".join(nom(o) for o in tri[:5]))
            premiers = [int(np.argmin([min(v for o, v in ordres({m: val[k, m] for m in range(32)}).items()
                                           if o[0] == c) for c in range(5)])) for k in range(NSEED_P)]
            print("  premier pilier de l'ordre optimal par graine : " + " ".join(f"P{PIL[c]}" for c in premiers))

        # ================================================== 3
        titre(f"3. Direction a quatre canaux, cascade independante, {NSEED_V} graines")
        S9, _ = pid.decompose(pid.expert_matrix(cx.G_NC))
        verts = pid.all_vertices(S9, 1.0)
        rows = list(pool.map(sommets_graine, [(k, verts) for k in range(NSEED_V)]))
    C = np.mean([r[0][:, 0] for r in rows], axis=0)
    N = np.mean([r[0][:, 1] for r in rows], axis=0)
    ec_, en_ = np.mean([r[1] for r in rows], axis=0)
    print(f"sommets : {len(verts)}")
    print(f"point du bareme : B(C) {ec_:.0f}, B(NC) {en_:.0f}, ecart {en_ - ec_:.0f}, facteur {en_ / ec_:.2f}")
    print(f"B(C)  aux sommets : [{C.min():.0f} ; {C.max():.0f}]")
    print(f"B(NC) aux sommets : [{N.min():.0f} ; {N.max():.0f}]")
    print(f"ecart aux sommets : [{(N - C).min():.0f} ; {(N - C).max():.0f}]")
    print(f"facteur aux sommets, meme signe aux deux etats : [{(N / C).min():.2f} ; {(N / C).max():.2f}]")
    print(f"B(NC) rapporte au point : [{N.min() / en_:.3f} ; {N.max() / en_:.3f}]")
    larg = [r[0][:, 1].max() - r[0][:, 1].min() for r in rows]
    print(f"largeur de B(NC) par graine : {np.round(larg, 0)}, ET {np.std(larg, ddof=1):.0f}")

    # ====================================================== 4
    titre("4. Citations externes transcrites (non recalculables) et conversions")
    print("Deloitte, DORA survey wave 3, mars 2025, 36 entites, 28 pays : pleine conformite pilier I 25, "
          "pilier II 48, pilier III 8, tiers 8 ; 50 attendent la pleine conformite fin 2025")
    print("ESAs, dry run 2024 : 947 registres analyses, 6.5 passent tous les controles")
    print("Synergy, T3 2025 : 63 des depenses cloud pour Amazon, Microsoft et Google")
    print("IBM/Ponemon 2025 : formation -191893 USD sur le cout ; chaine tierce 15 des breches, 4.91 M USD")
    print("SecurityScorecard 2025 : 35.5 des breches de 2024 d'origine tierce, sur 1000 breches")
    print("ENISA 2025 : 4875 incidents ; 70 des cas de vulnerabilite et 27 du hameconnage menent a une intrusion")
    print("Meyer et al. 2023 : l'authentification multifacteur reduit la compromission de 99.22")
    print("McKinsey, 28 juin 2024 : 5 a 15 M EUR de pilotage, 5 a 10 fois en mise en oeuvre ; "
          "enveloppe calculee 25 a 150 M EUR, non observee")
    print(f"controle : {5 * 5} et {15 * 10}")
    print(f"conversions : 20000 M USD a 1.04 USD/EUR = {20000 / 1.04:.0f} M EUR ; a 0.92 EUR/USD = {20000 * 0.92:.0f} M EUR")
    print(f"1/1.04 = {1 / 1.04:.3f} EUR/USD")
    print(f"non pleinement conformes dans le melange pose : NC + PC = {0.35 + 0.35:.2f}")
    print("Deloitte, depense prevue : 64 des entites entre 2 et 5 M EUR")

    # ====================================================== 5
    titre("5. Queue par pilier (patron du script 37) : recentrage a 0,90 puis a la reference")
    print("Moyenne des queues fixee ; fréquence conforme, g = 0,90, p_u de base, 30000 annees, graine 909.")
    with ProcessPoolExecutor() as pool:
        for ref in (0.90, float(XI)):
            centre = XI_CAT - XI_CAT.mean() + ref
            commun = scr37(tuple([ref] * 5))
            vals = np.array(list(pool.map(scr37, list(permutations(centre)), chunksize=4)))
            print(f"reference {ref:.4f} : queues {np.round(centre, 3)}, max {centre.max():.3f}")
            print(f"  queue commune {commun:.0f} ; assignations [{vals.min():.0f} ; {vals.max():.0f}] ; "
                  f"ecart de {100 * (vals.min() / commun - 1):+.1f} % a {100 * (vals.max() / commun - 1):+.1f} %")


if __name__ == "__main__":
    main()
