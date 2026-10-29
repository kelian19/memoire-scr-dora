# -*- coding: utf-8 -*-
"""DESCENTE D'ECHELLE secteur -> entite : le panel, l'elasticite, et les deux lectures.

SOURCE UNIQUE, partagee par les scripts 60 (qui l'a etablie et la documente) et 65 (qui
l'applique a des entites reelles). Tout ce qui suit vivait dans le corps du script 60 ; deux
scripts en ayant besoin, l'estimation devait devenir un module. Recopier une estimation de
maximum de vraisemblance dans deux fichiers, c'est se garantir qu'ils divergeront un jour.

CE QUE LE MODULE FAIT.
  - Reconstruit le panel firme-annee d'OpRisk selon le protocole du script 08b : la fenetre
    d'observation d'une firme est l'intervalle [premiere, derniere] annee ou elle apparait
    POUR UN RISQUE QUELCONQUE, et une annee de cet intervalle sans incident TIC est un VRAI
    zero. Le filtre annuel s'applique AVANT le calcul de la fenetre, sans quoi les fenetres
    sont artificiellement allongees et lambda s'effondre.
  - Estime une binomiale negative NB2 a log-lien sur ce panel, avec log(actifs) centre comme
    unique covariable : l'elasticite frequence / taille, avec son erreur-type par hessienne
    numerique.
  - Expose les deux lectures a une taille quelconque : `lam(actifs)` pour la frequence,
    `mult(actifs)` pour l'echelle de severite (elasticite du script 57).

CE QUE LE MODULE NE FAIT PAS. Il n'imprime rien et ne decide rien. Le script 60 reste seul
depositaire de la lecture, des reserves et des chiffres publies au chapitre resultats.
"""

import os

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(_HERE, "..", "..", "data", "raw"))
OPRISK_XLS = os.path.join(RAW, "SAS_OpRisk_Global_Data_June_2026.xlsx")

ICT = ["Systems Security", "Systems", "Vendors & Suppliers",
       "Monitoring and Reporting", "Unauthorized Activity"]
Y0, Y1 = 2005, 2022
LOSS = "Current Value of Loss ($M)"
ASSETS = "Assets ($M)"

# elasticite severite / taille, EMV lognormale tronquee au seuil de collecte (script 57)
B_SEV, B_SEV_LO, B_SEV_HI = 0.087, 0.026, 0.148


class Descente:
    """Le panel, l'ajustement NB2, et les deux lectures a la taille voulue."""

    def __init__(self, xls=None):
        xls = xls or OPRISK_XLS
        if not os.path.exists(xls):
            raise FileNotFoundError(
                f"donnee absente : {xls}\n(les sources brutes ne sont pas versionnees)")

        d = pd.read_excel(xls, sheet_name="Datasets")
        d["year"] = pd.to_datetime(d["First Year of Event"], errors="coerce").dt.year
        fs = d[d["Basel Business Line - Level 1"] != "Non-FS"].copy()
        fs = fs[(fs.year >= Y0) & (fs.year <= Y1)]
        ict = fs[fs["Sub Risk Category"].isin(ICT)].copy()

        span = fs.groupby("Firm Name")["year"].agg(["min", "max"])
        ict_n = ict.groupby(["Firm Name", "year"]).size().rename("n")

        fs[ASSETS] = pd.to_numeric(fs[ASSETS], errors="coerce")
        taille = fs.groupby("Firm Name")[ASSETS].median()

        rows = []
        for firm, (y0, y1) in span[["min", "max"]].iterrows():
            a = taille.get(firm, np.nan)
            for y in range(int(y0), int(y1) + 1):
                rows.append((firm, y, int(ict_n.get((firm, y), 0)), a))
        panel = pd.DataFrame(rows, columns=["firm", "year", "n", "actifs"])

        pa = panel.dropna(subset=["actifs"])
        pa = pa[pa.actifs > 0]

        self.fs, self.ict, self.panel, self.pa, self.taille = fs, ict, panel, pa, taille
        self.med_act = float(taille.dropna().median())
        self.med_act_ict = float(pd.to_numeric(ict[ASSETS], errors="coerce").median())

        self._ajuster()

    # ------------------------------------------------------------------ NB2 a log-lien
    def _ajuster(self):
        x = np.log(self.pa.actifs.to_numpy())
        k = self.pa.n.to_numpy().astype(float)
        self.xbar = x.mean()
        xc = x - self.xbar          # centrage : l'intercept devient log-lambda a la taille
                                    # MEDIANE, et le probleme est mieux conditionne
        self._xc, self._k = xc, k

        def nll(p):
            a, b, lr = p
            r = np.exp(lr)
            mu = np.exp(np.clip(a + b * xc, -30, 30))
            return -np.sum(special.gammaln(k + r) - special.gammaln(r) - special.gammaln(k + 1)
                           + r * np.log(r / (r + mu)) + k * np.log(mu / (r + mu)))

        self.nll = nll
        opt = optimize.minimize(nll, x0=[np.log(max(k.mean(), 1e-6)), 0.2, 0.0],
                                method="Nelder-Mead",
                                options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 20000})
        self.opt = opt
        self.a_hat, self.b_lam, self.lr_hat = opt.x

        eps = 1e-4
        H = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                pp, pm, mp, mm = (opt.x.copy() for _ in range(4))
                pp[i] += eps; pp[j] += eps
                pm[i] += eps; pm[j] -= eps
                mp[i] -= eps; mp[j] += eps
                mm[i] -= eps; mm[j] -= eps
                H[i, j] = (nll(pp) - nll(pm) - nll(mp) + nll(mm)) / (4 * eps * eps)
        cov = np.linalg.inv(H)
        self.se_b = float(np.sqrt(max(cov[1, 1], 0.0)))
        self.z_b = self.b_lam / self.se_b if self.se_b > 0 else np.nan
        self.p_b = 2 * stats.norm.sf(abs(self.z_b))

    # ------------------------------------------------------------------ les deux lectures
    def lam(self, actifs):
        """Frequence annuelle d'incidents TIC materiels, lue a la taille `actifs` (M USD)."""
        return float(np.exp(self.a_hat + self.b_lam * (np.log(actifs) - self.xbar)))

    def lam_bande(self, actifs):
        """Bande de lambda induite par l'IC 95 % sur la seule elasticite b_lambda."""
        lo = float(np.exp(self.a_hat + (self.b_lam + 1.96 * self.se_b)
                          * (np.log(actifs) - self.xbar)))
        hi = float(np.exp(self.a_hat + (self.b_lam - 1.96 * self.se_b)
                          * (np.log(actifs) - self.xbar)))
        return min(lo, hi), max(lo, hi)

    def mult(self, actifs, b=B_SEV):
        """Multiplicateur d'echelle de severite, a la taille `actifs` (M USD).

        La severite GPD est une famille d'ECHELLE au-dessus du seuil : multiplier toutes les
        severites par c multiplie le quantile par c EXACTEMENT. La source OpRisk n'etant pas
        plafonnee, rien ne vient tronquer cette homothetie.
        """
        return float((actifs / self.med_act_ict) ** b)

    def mult_bande(self, actifs):
        return (self.mult(actifs, B_SEV_HI), self.mult(actifs, B_SEV_LO))
