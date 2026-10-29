"""Expériences Institut : unité de perte et bootstrap conjoint, hors calage publié.

Montants en millions d'euros. Les tables de cascade et le comptage sont ceux du
moteur historique. Aucune constante partagée n'est modifiée.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.stats import genpareto

import canaux_conformite as cx


REGLES = ("additive", "moyenne", "unique", "amplifiee")
CANAUX = ("frequence", "detection", "propagation", "accumulation")


@dataclass(frozen=True)
class Calibration:
    xi: float
    sigma: float
    pu: float
    u: float = float(cx.sp["u"])

    def verifier(self):
        if not np.isfinite([self.xi, self.sigma, self.pu, self.u]).all():
            raise ValueError("Paramètres non finis")
        if self.sigma <= 0 or not 0 < self.pu < 1 or self.u <= 0:
            raise ValueError("Échelle, seuil ou probabilité inadmissible")


REFERENCE = Calibration(float(cx.sp["xi"]), float(cx.sp["sigma"]), float(cx.P_U0))


@lru_cache(maxsize=4)
def cardinal(propagation: bool, accumulation: bool):
    """Loi exacte de K, y compris le choc P4 du moteur à quatre canaux."""
    g = cx.G_NC if propagation else cx.G_C
    poids = np.array([cx.eng.LAMBDA[j] for j in cx.PIL], dtype=float)
    poids /= poids.sum()
    loi = np.zeros(6)
    for poids_j, j in zip(poids, cx.PIL):
        ind, probs = (cx.table_p4_choc(cx.PHICS_NC)
                      if accumulation and j == cx.P4 else cx.table_amorce(j, g))
        for k, prob in zip(ind.sum(axis=1).astype(int), probs):
            loi[k] += poids_j * prob
    np.testing.assert_allclose(loi.sum(), 1, atol=1e-13)
    return loi


def probabilite_positive(loi_k, pu, regle):
    if regle in ("unique", "amplifiee"):
        return float(pu)
    return float(sum(loi_k[k] * (1 - (1-pu)**k) for k in range(1, 6)))


def moyenne_incident(cal, pu, loi_k, regle, eta=1.0):
    if cal.xi >= 1:
        return math.inf
    moyenne = pu * (cal.u + cal.sigma / (1-cal.xi))
    if regle == "additive":
        return moyenne * float(np.arange(6) @ loi_k)
    if regle == "amplifiee":
        return moyenne * float((np.arange(6)**eta) @ loi_k)
    return moyenne


def pgf_nb(z, lam):
    r = lam / (cx.ec.PHI - 1)
    p = 1 / cx.ec.PHI
    return (p / (1 - (1-p)*z))**r


class FourierIncidents:
    """Inversion de séries avec inclinaison exponentielle, sans renormalisation.

    Une perte élémentaire vaut zéro ou u+GPD. Pour chaque cardinal k :
      additive = somme de k pertes ; moyenne = somme/k ;
      unique = une perte ; amplifiee = k**eta fois une perte.
    On discrétise chaque perte élémentaire après changement d'échelle, puis
    compose les transformées par la loi exacte de K et la génératrice NB.
    Les masses de sévérité au-delà de la portée sont omises, et journalisées.
    L'inclinaison exp(-tilt*j/n) amortit le repliement ; la somme annuelle
    est donc une sous-probabilité sur la grille, jamais renormalisée à un.
    """

    def __init__(self, cal=REFERENCE, step=2.0, n=1 << 16, tilt=16.0, eta=1.0):
        cal.verifier()
        if n < 16 or n & (n-1) or step <= 0 or not 0 < tilt <= 20:
            raise ValueError("Grille ou inclinaison inadmissible")
        self.cal, self.step, self.n, self.tilt, self.eta = cal, step, n, tilt, eta
        self.attenuation = np.exp(-tilt*np.arange(n)/n)
        self.bords = (np.arange(n+1)-.5)*step
        self.bords[0] = 0
        self._transforms = {}
        self.tail_severite = {}

    def transformee_severite(self, facteur):
        facteur = float(facteur)
        if facteur not in self._transforms:
            exces = np.maximum(self.bords/facteur-self.cal.u, 0)
            survie = genpareto.sf(exces, c=self.cal.xi, scale=self.cal.sigma)
            masses = -np.diff(survie)
            self.tail_severite[facteur] = float(survie[-1])
            if masses.min() < -1e-14:
                raise ArithmeticError("Masse de sévérité négative")
            self._transforms[facteur] = np.fft.rfft(masses*self.attenuation)
        return self._transforms[facteur]

    def transformee_incident(self, pu, loi_k, regle):
        if regle not in REGLES:
            raise ValueError(regle)
        if not 0 < pu < 1:
            raise ValueError("Taux de dépassement transformé hors de (0,1)")
        if regle == "unique":
            return 1-pu + pu*self.transformee_severite(1)
        if regle == "additive":
            z = 1-pu + pu*self.transformee_severite(1)
            return sum(loi_k[k]*z**k for k in range(1, 6))
        if regle == "moyenne":
            return sum(loi_k[k]*(1-pu + pu*self.transformee_severite(1/k))**k
                       for k in range(1, 6))
        return sum(loi_k[k]*(1-pu + pu*self.transformee_severite(k**self.eta))
                   for k in range(1, 6))

    def capital(self, mask, regle, alpha=.995, loi_k=None):
        lam = cx.LAM_NC if mask & 1 else cx.LAM_C
        pu = self.cal.pu * (1.20 if mask & 2 else .85)
        loi_k = cardinal(bool(mask & 4), bool(mask & 8)) if loi_k is None else loi_k
        incident = self.transformee_incident(pu, loi_k, regle)
        masses = np.fft.irfft(pgf_nb(incident, lam), n=self.n)/self.attenuation
        # Aucun clipping : une masse négative sensible fait échouer le calcul.
        if not np.isfinite(masses).all() or masses.min() < -1e-8:
            raise ArithmeticError("Inversion instable : masse négative ou non finie")
        cdf = np.cumsum(masses)
        idx = int(np.searchsorted(cdf, alpha))
        if idx >= self.n or idx > self.n*.25:
            raise ArithmeticError("Quantile trop près de la portée : élargir la grille")
        cumul_avant = cdf[idx-1] if idx else 0
        # Le quantile sur la grille est retenu : aucune interpolation implicite.
        value = idx*self.step
        diag = {"step": self.step, "n": self.n, "portee": self.step*self.n,
                "tilt": self.tilt, "cdf_portee": float(cdf[-1]),
                "cdf_avant": float(cumul_avant), "cdf_quantile": float(cdf[idx]),
                "masse_negative": float(-masses[masses < 0].sum()),
                "masse_negative_avant_quantile": float(-masses[:idx+1][masses[:idx+1] < 0].sum()),
                "severite_hors_portee_max": max(self.tail_severite.values()),
                "moyenne_tronquee": float(np.sum(np.arange(self.n)*masses)*self.step)}
        if diag["masse_negative"] > 1e-5 or not .995 <= diag["cdf_portee"] <= 1.00001:
            raise ArithmeticError(f"Portée ou stabilité insuffisante : {diag}")
        return value, diag


def grille_adaptee(cal):
    """Le pas varie avec la queue ; la portée contient au moins quatre quantiles."""
    q = cardinal(True, True)
    n_effectif = cx.LAM_NC*1.20*cal.pu*float(np.arange(6) @ q)
    # L'amplification peut peser davantage que la somme en queue.
    n_effectif *= max(1.0, float((np.arange(6)**max(cal.xi, 0)) @ q)
                       / float(np.arange(6) @ q))
    niveau = max(1e-10, 1-.005/max(n_effectif, .01))
    repere = cal.u + genpareto.ppf(niveau, c=cal.xi, scale=cal.sigma)
    return max(1.0, float(repere)/10_000)


def calculer_configurations(cal, masks=range(16), regles=REGLES, step=None, n=1 << 16, tilt=16):
    step = grille_adaptee(cal) if step is None else step
    for essai in range(5):
        moteur = FourierIncidents(cal, step=step*2**essai, n=n, tilt=tilt)
        try:
            resultats, diagnostics = {}, []
            for regle in regles:
                valeurs = {}
                for mask in masks:
                    valeur, diag = moteur.capital(mask, regle)
                    valeurs[str(mask)] = valeur
                    diagnostics.append(dict(regle=regle, mask=mask, **diag))
                resultats[regle] = valeurs
            return resultats, diagnostics
        except ArithmeticError:
            if essai == 4:
                raise
    raise AssertionError("Boucle inaccessible")


def contrastes(valeurs):
    c, nc = valeurs["0"], valeurs["15"]
    resultat = {"conforme": c, "non_conforme": nc, "ecart": nc-c, "rapport": nc/c}
    if len(valeurs) == 16:
        isoles = sum(valeurs[str(1 << i)]-c for i in range(4))
        shapley = {}
        for i, nom in enumerate(CANAUX):
            shapley[nom] = sum(math.factorial(m.bit_count())*math.factorial(3-m.bit_count())/24
                               * (valeurs[str(m | (1 << i))]-valeurs[str(m)])
                               for m in range(16) if not m & (1 << i))
        resultat.update(effets_isoles=isoles, interaction=nc-c-isoles, shapley=shapley)
        np.testing.assert_allclose(sum(shapley.values()), nc-c, atol=1e-7)
    return resultat
