"""Compléments POT pour la population de calibration Institut ; seuil fixé en euros."""
import numpy as np
from scipy import optimize, stats
from scipy.stats import genpareto


def emv(exces):
    x = np.asarray(exces, dtype=float)
    if x.size < 3 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Excès non exploitables')
    xi, _, sigma = genpareto.fit(x, floc=0)
    if sigma <= 0 or not np.isfinite(genpareto.logpdf(x, xi, scale=sigma)).all():
        raise ArithmeticError('Ajustement GPD hors support')
    return float(xi), float(sigma)


def ks_stat(exces, xi, sigma):
    z = genpareto.cdf(np.sort(exces), xi, scale=sigma)
    n = len(z)
    return float(max(np.max(np.arange(1, n+1)/n-z), np.max(z-np.arange(n)/n)))


def ad_stat(exces, xi, sigma):
    z = np.clip(genpareto.cdf(np.sort(exces), xi, scale=sigma), 1e-12, 1-1e-12)
    n = len(z)
    return float(-n-np.sum((2*np.arange(1, n+1)-1)/n*(np.log(z)+np.log(1-z[::-1]))))


def pwm(exces):
    x = np.sort(exces); p = (np.arange(1, len(x)+1)-.35)/len(x)
    a0, a1 = x.mean(), np.mean((1-p)*x)
    xi = 2-a0/(a0-2*a1); sigma = 2*a0*a1/(a0-2*a1)
    if not np.isfinite([xi, sigma]).all() or sigma <= 0:
        raise ArithmeticError('PWM non admissible')
    return float(xi), float(sigma)


def hill(exces, u):
    return float(np.log((u+np.asarray(exces))/u).mean())


def moments(exces, u):
    z = np.log((u+np.asarray(exces))/u); m1, m2 = z.mean(), (z*z).mean()
    return float(m1+1-.5/(1-m1*m1/m2))


def quantile(u, xi, sigma, pu, alpha=.995):
    if not 0 < 1-alpha < pu < 1 or sigma <= 0:
        raise ValueError('Quantile hors queue ou paramètres invalides')
    return float(u+genpareto.ppf(1-(1-alpha)/pu, xi, scale=sigma))


def profil_xi(exces, niveau=.95):
    xi, sigma = emv(exces); ll = genpareto.logpdf(exces, xi, scale=sigma).sum()
    crit = stats.chi2.ppf(niveau, 1)/2

    def score(c):
        r = optimize.minimize_scalar(lambda z: -genpareto.logpdf(exces, c, scale=np.exp(z)).sum(),
            bounds=(np.log(sigma)-4, np.log(sigma)+4), method='bounded')
        return float(ll+r.fun-crit)

    return [float(optimize.brentq(score, xi-2, xi-1e-6)),
            float(optimize.brentq(score, xi+1e-6, xi+2))]
