#!/usr/bin/env python3
"""138 : sensibilite d'observation, sans remplacer le capital publie.

Une NB observee de moyenne mu et forme r est compatible avec toutes les familles
NB(mu/q, r) amincies independamment avec probabilite q. La base ne choisit pas q.
On mesure cette non-identification, en gardant le corps, la queue et la cascade du 136.
q = 1, 0.5, 0.25, 0.1 sont des SCENARIOS, pas des taux de declaration estimes.
L'independance de la declaration et de la severite est une hypothese restrictive.
Le modele ne corrige ni le panel selectionne, ni les fenetres, ni les seuils de collecte.
Les rapports DORA et les cas d'entreprises ne sont pas injectes dans cette calibration.
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy import optimize, special
from scipy.stats import binom, genpareto, nbinom

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
for p in (REPO, LAB):
    sys.path.insert(0, str(p))
import descente as dsc
import partial_id as pid
import euro_cascade_model as ec
from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance, USD_EUR

YEARS = 600_000
SEEDS = (20260721, 20260722, 20260723, 20260724)
ALPHA = 0.995
SCENARIOS = (1.0, 0.5, 0.25, 0.1)
ENTITIES = (("Assureur non-vie A", 2319.0), ("Entite notionnelle", 20_000 / 1.04),
            ("Assureur vie D", 309800.0))


def gamma_poisson_posterior(shape, rate, count, exposure, capture=1.0):
    """Noyau de credibilite ; aucun a priori metier n'est invente par le programme.

    eta ~ Gamma(shape, rate), n | eta ~ Pois(capture * exposure * eta).
    exposure est une somme d'unites d'activite-annees REELLEMENT observees.
    Aucune fenetre du registre public n'est acceptee comme preuve de cette exposition.
    """
    if not all(np.isfinite(x) for x in (shape, rate, count, exposure, capture)):
        raise ValueError("parametres non finis")
    if shape <= 0 or rate <= 0 or count < 0 or int(count) != count or exposure < 0:
        raise ValueError("a priori, compte ou exposition invalides")
    if not 0 < capture <= 1 or (exposure == 0 and count > 0):
        raise ValueError("capture ou observation impossible")
    return shape + count, rate + capture * exposure


def fit_frequency(panel):
    sub = panel[panel.actifs >= 1e5]
    x = np.log(sub.actifs.to_numpy())
    xb = float(x.mean())
    xc, k = x - xb, sub.n.to_numpy().astype(float)

    def nll(theta):
        a, b, logr = theta
        r = np.exp(logr)
        mu = np.exp(np.clip(a + b * xc, -30, 30))
        return -np.sum(special.gammaln(k + r) - special.gammaln(r) - special.gammaln(k + 1)
                       + r * np.log(r / (r + mu)) + k * np.log(mu / (r + mu)))
    opt = optimize.minimize(nll, [np.log(k.mean()), 0.2, 0.0], method="Nelder-Mead",
                            options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-9})
    if not opt.success or not np.isfinite(opt.fun):
        raise RuntimeError(f"ajustement non converge : {opt.message}")
    a, b, logr = opt.x
    return dict(a=float(a), b=float(b), r=float(np.exp(logr)), xb=xb)


def risk(x):
    # Integral empirical ES: the largest m losses plus the fractional boundary loss.
    s = np.sort(x)
    mass = (1 - ALPHA) * len(s)
    m = int(np.floor(mass))
    frac = mass - m
    es = (s[-m:].sum() + frac * s[-m - 1]) / mass
    return dict(var=float(np.quantile(s, ALPHA)), es=float(es), mean=float(s.mean()),
                prob_loss=float(np.mean(s > 0)))


def main():
    print(f"Protocole : {YEARS} annees par graine, {len(SEEDS)} graines, alpha = {ALPHA}")
    print("Captures supposees, non estimees : " + ", ".join(f"{100*q:g} %" for q in SCENARIOS))
    d = dsc.Descente()
    f = fit_frequency(d.pa)
    if abs(f["b"] - 0.494) > 5e-4:
        raise RuntimeError("reference de frequence du 136 non reproduite")
    sp = pid.PARAMS["OPRISK"]
    raw = filter_finance(filter_cyber(load_clean(
        REPO / "data/raw/SAS_OpRisk_Global_Data_June_2026.xlsx")))
    raw = raw[raw.year.between(2000, 2026)]
    if len(raw) != 570:
        raise RuntimeError("population de severite differente du protocole")
    loss = raw.loss.to_numpy() * USD_EUR
    body = np.sort(loss[loss <= sp["u"]])
    if np.any(body <= 0):
        raise RuntimeError("pertes non positives dans le corps")
    card = pid.card_dist_all(pid.expert_matrix())[0]
    weights = np.array([ec.eng.LAMBDA[j] for j in pid.PIL], float)
    seeds_cdf = np.cumsum(weights / weights.sum())
    rows = []
    for name, assets_eur in ENTITIES:
        assets = assets_eur * 1.04
        mu_obs = float(np.exp(f["a"] + f["b"] * (np.log(assets) - f["xb"])))
        mult = d.mult(assets)
        for q in SCENARIOS:
            # Poisson-Gamma thinning preserves shape r, only the mean changes.
            mu_true, r = mu_obs / q, f["r"]
            # Amincir la loi latente par une binomiale doit redonner la loi enregistree.
            nn = np.arange(2000)
            latent = nbinom.pmf(nn, r, r / (r + mu_true))
            thinned = [np.sum(latent * binom.pmf(k, nn, q)) for k in range(8)]
            if not np.allclose(thinned, nbinom.pmf(np.arange(8), r, r / (r + mu_obs)),
                               rtol=1e-9, atol=1e-13):
                raise RuntimeError("l'amincissement ne redonne pas la loi enregistree")
            runs = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                uy = rng.random(YEARS)
                p = r / (r + mu_true)
                counts = np.zeros(YEARS, dtype=np.int64)
                active = uy > p ** r
                counts[active] = nbinom.ppf(uy[active], r, p).astype(np.int64)
                total = int(counts.sum())
                # Same seed couples counts by year; severity positions shift with total counts.
                rng_sev = np.random.default_rng(seed + 100)
                u = rng_sev.random((total, 2 + 3 * pid.NP_))
                origin = np.minimum(np.searchsorted(seeds_cdf, u[:, 0]), pid.NP_ - 1)
                n = pid.NP_
                tail = u[:, 2:2+n] < sp["p_u"]
                sev = np.where(tail, sp["u"] + genpareto.ppf(u[:, 2+n:2+2*n],
                               c=sp["xi"], scale=sp["sigma"]),
                               body[np.minimum((u[:, 2+2*n:] * len(body)).astype(int), len(body)-1)])
                cum = np.cumsum(sev * mult, axis=1)
                size = np.ones(total, dtype=np.int64)
                for j in range(n):
                    ix = np.flatnonzero(origin == j)
                    cdf = np.cumsum(card[j][1:])
                    cdf[-1] = 1.0
                    size[ix] = np.searchsorted(cdf, u[ix, 1], side="right") + 1
                annual = np.bincount(np.repeat(np.arange(YEARS), counts),
                                    weights=cum[np.arange(total), size-1], minlength=YEARS)
                runs.append(risk(annual))
            means = {key: float(np.mean([v[key] for v in runs])) for key in runs[0]}
            stds = {key: float(np.std([v[key] for v in runs], ddof=1)) for key in runs[0]}
            p_loss = float(1 - (r / (r + mu_true)) ** r)
            probability_tolerance = 5 * np.sqrt(p_loss * (1 - p_loss) / (YEARS * len(SEEDS)))
            if abs(means["prob_loss"] - p_loss) > probability_tolerance:
                raise RuntimeError("probabilite de perte simulee incompatible avec la NB")
            rows.append(dict(entity=name, assets_meur=assets_eur, q_scenario=q,
                             mean_count_recorded=mu_obs, mean_count_latent=mu_true,
                             prob_loss_exact=p_loss, estimates=means, mc_sd=stds))
            print(f"{name:<23} q={q:4.2f} lambda={mu_true:.4f} "
                  f"VaR={means['var']:.1f} +- {stds['var']:.1f} "
                  f"ES={means['es']:.1f} +- {stds['es']:.1f} "
                  f"P(perte)={100*p_loss:.2f}%")
        # Distribution checks of the reference; this does not validate its real-world calibration.
        ref = rows[-len(SCENARIOS)]
        expected = {"Assureur non-vie A": 1.0, "Entite notionnelle": 24.1,
                    "Assureur vie D": 192.6}[name]
        if abs(ref["estimates"]["var"] - expected) > max(4 * ref["mc_sd"]["var"], 0.2 * expected):
            raise RuntimeError("la reference du 136 n'est pas reproduite au bruit pres")
    output = REPO / "resultats/institut_approfondissements/138.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(status="scenarios_non_valides", fit=f, years=YEARS,
        seeds=SEEDS, q_estimated=False, external_sources_used_for_fit=False, rows=rows),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("Controle : meme distribution de comptes enregistres pour tous les q.")
    print("Les capitaux latents ne sont pas identifies par cette distribution observee.")
    print("Aucun q ni capital central nouveau n'est retenu. " + str(output.relative_to(REPO)))


if __name__ == "__main__":
    main()
