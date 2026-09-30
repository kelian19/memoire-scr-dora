"""
aggregation/copule.py
----------------------
Copule de Gumbel pour la dépendance de QUEUE entre les 4 briques de perte
(aggravation, prestataire, remédiation, sanction).

Pourquoi Gumbel : c'est une copule archimédienne à dépendance de queue
SUPÉRIEURE asymétrique — deux briques ont peu de chances d'être corrélées
pour de petits sinistres, mais une forte chance de l'être simultanément en
cas de sinistre extrême (la contagion interne se manifeste surtout dans la
queue, pas au centre de la distribution). C'est l'hypothèse de dépendance la
plus défendable pour un incident DORA majeur : un incident qui dégénère a
toutes les chances d'entraîner remédiation lourde, recours à un prestataire,
ET déclenchement d'une procédure de sanction simultanément.

PARAMÈTRE θ (theta) :
  θ = 1   → indépendance
  θ → ∞   → dépendance parfaite
  θ = 1.8 → valeur retenue (cf. travaux antérieurs du mémoire), correspond à
            un tau de Kendall τ ≈ 1 - 1/θ ≈ 0.44 (dépendance modérée à forte)

ALTERNATIVE DE ROBUSTESSE : modèle à facteur commun (choc systémique
B ~ Bernoulli(p_sys)), cf. fonction common_factor_uniforms ci-dessous.
"""

import numpy as np
from scipy.stats import kendalltau


# ---------------------------------------------------------------------------
# 1. ÉCHANTILLONNAGE COPULE DE GUMBEL (algorithme de Marshall-Olkin)
# ---------------------------------------------------------------------------

def _sample_positive_stable(alpha: float, n: int, rng) -> np.ndarray:
    """
    Échantillonne une variable stable positive d'indice alpha ∈ (0,1),
    skewness totale (β=1), via l'algorithme de Chambers-Mallows-Stuck (CMS).
    Nécessaire pour générer une copule de Gumbel par la méthode de
    Marshall-Olkin (mélange de variables exponentielles).
    """
    w = rng.exponential(1.0, size=n)
    phi = rng.uniform(-np.pi / 2, np.pi / 2, size=n)

    a = alpha
    num = np.sin(a * (phi + np.pi / 2))
    den = np.cos(phi) ** (1.0 / a)
    term1 = num / den
    term2 = (np.cos(phi - a * (phi + np.pi / 2)) / w) ** ((1.0 - a) / a)
    return term1 * term2


def gumbel_copula_uniforms(n_sim: int, theta: float, dim: int = 4,
                            seed: int = 42) -> np.ndarray:
    """
    Génère n_sim tirages d'un vecteur de dimension `dim` suivant une copule
    de Gumbel de paramètre theta (marginales uniformes [0,1], dépendance de
    queue supérieure entre les dim composantes).

    Returns
    -------
    array (n_sim, dim) de valeurs dans [0,1], à transformer ensuite par les
    quantiles des lois marginales souhaitées (cf. lda.py).
    """
    if theta < 1.0:
        raise ValueError("theta doit être >= 1 (theta=1 -> indépendance)")

    rng = np.random.default_rng(seed)
    alpha = 1.0 / theta

    V = _sample_positive_stable(alpha, n_sim, rng)
    V = np.abs(V)  # garde-fou numérique (doit être positif par construction)

    E = rng.exponential(1.0, size=(n_sim, dim))  # iid Exp(1)
    U = np.exp(-((E / V[:, None]) ** (1.0 / theta)))

    return np.clip(U, 1e-10, 1 - 1e-10)  # évite les bornes exactes 0/1


def empirical_kendall_tau(theta: float, n_sim: int = 20_000, seed: int = 1) -> float:
    """
    Valide numériquement la calibration : pour une copule de Gumbel de
    paramètre theta, le tau de Kendall théorique est 1 - 1/theta.
    Cette fonction le retrouve empiriquement sur un tirage bivarié.
    """
    U = gumbel_copula_uniforms(n_sim, theta, dim=2, seed=seed)
    tau_empirical, _ = kendalltau(U[:, 0], U[:, 1])
    tau_theoretical = 1 - 1 / theta
    return tau_empirical, tau_theoretical


# ---------------------------------------------------------------------------
# 2. ALTERNATIVE DE ROBUSTESSE — MODÈLE À FACTEUR COMMUN
# ---------------------------------------------------------------------------

def clayton_tau(theta: float) -> float:
    """Tau de Kendall théorique d'une copule de Clayton de paramètre theta > 0."""
    return theta / (theta + 2.0)


def solve_clayton_theta(target_tau: float) -> float:
    """Paramètre Clayton reproduisant un tau de Kendall cible (theta > 0)."""
    if not (0 < target_tau < 1):
        raise ValueError("target_tau doit être dans (0,1)")
    return 2.0 * target_tau / (1.0 - target_tau)


def clayton_copula_uniforms(n_sim: int, theta: float, dim: int = 4,
                             seed: int = 42, rotated: bool = True) -> np.ndarray:
    """
    Génère n_sim tirages d'une copule de Clayton (méthode de la frailty Gamma,
    Marshall-Olkin). theta > 0. La copule de Clayton \"brute\" a une dépendance
    de queue INFÉRIEURE ; rotated=True (par défaut) applique la rotation à
    180° (copule de survie), qui restaure une dépendance de queue SUPÉRIEURE,
    directement comparable à celle de Gumbel — utile pour un test de
    robustesse sur la FAMILLE de copule plutôt que sur son seul paramètre.
    """
    if theta <= 0:
        raise ValueError("theta doit être > 0 pour Clayton")

    rng = np.random.default_rng(seed)
    V = rng.gamma(shape=1.0 / theta, scale=1.0, size=n_sim)
    E = rng.exponential(1.0, size=(n_sim, dim))
    U = (1.0 + E / V[:, None]) ** (-1.0 / theta)
    if rotated:
        U = 1.0 - U
    return np.clip(U, 1e-10, 1 - 1e-10)


def frank_tau(theta: float) -> float:
    """Tau de Kendall théorique d'une copule de Frank (intégration numérique
    de la fonction de Debye d'ordre 1)."""
    from scipy.integrate import quad
    if theta == 0:
        return 0.0
    debye, _ = quad(lambda t: t / (np.exp(t) - 1.0), 0, theta)
    debye /= theta
    return 1.0 + 4.0 / theta * (debye - 1.0)


def solve_frank_theta(target_tau: float) -> float:
    """Paramètre Frank reproduisant un tau de Kendall cible (theta > 0)."""
    from scipy.optimize import brentq
    return brentq(lambda th: frank_tau(th) - target_tau, 1e-6, 100.0)


def frank_copula_uniforms(n_sim: int, theta: float, dim: int = 4,
                           seed: int = 42) -> np.ndarray:
    """
    Génère n_sim tirages d'une copule de Frank (méthode de la frailty
    logarithmique, Marshall-Olkin). theta > 0. La copule de Frank n'a AUCUNE
    dépendance de queue (ni supérieure, ni inférieure) : c'est le cas de
    contraste naturel face à Gumbel pour mesurer combien la dépendance de
    queue supérieure, spécifiquement, pèse sur le capital simulé.
    """
    if theta == 0:
        raise ValueError("theta doit être != 0 pour Frank")

    from scipy.stats import logser
    rng = np.random.default_rng(seed)
    p = 1.0 - np.exp(-theta)
    V = logser.rvs(p, size=n_sim, random_state=rng.integers(0, 2**31 - 1))
    E = rng.exponential(1.0, size=(n_sim, dim))
    U = -1.0 / theta * np.log(1.0 + np.exp(-E / V[:, None]) * (np.exp(-theta) - 1.0))
    return np.clip(U, 1e-10, 1 - 1e-10)


def rho_from_kendall_tau(tau: float) -> float:
    """Corrélation de Pearson équivalente à un tau de Kendall cible, pour une
    copule elliptique (Gaussienne ou Student) : rho = sin(pi/2 * tau)."""
    return np.sin(np.pi / 2.0 * tau)


def student_t_copula_uniforms(n_sim: int, rho: float, df: float = 4.0,
                               dim: int = 4, seed: int = 42) -> np.ndarray:
    """
    Génère n_sim tirages d'une copule de Student (corrélation équicorrélée
    rho, degré de liberté df). Dépendance de queue SYMÉTRIQUE (haute et
    basse), contrairement à Gumbel (haute uniquement) — troisième famille de
    contraste pour le test de robustesse sur le choix de copule.
    """
    from scipy.stats import t as student_t
    rng = np.random.default_rng(seed)
    Sigma = np.full((dim, dim), rho)
    np.fill_diagonal(Sigma, 1.0)
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n_sim, dim)) @ L.T
    W = rng.chisquare(df, size=n_sim)
    T = Z * np.sqrt(df / W)[:, None]
    U = student_t.cdf(T, df=df)
    return np.clip(U, 1e-10, 1 - 1e-10)


def common_factor_uniforms(n_sim: int, p_sys: float, dim: int = 4,
                            seed: int = 42) -> np.ndarray:
    """
    Alternative à la copule de Gumbel : modèle à facteur commun, où un choc
    systémique B ~ Bernoulli(p_sys) affecte simultanément les `dim` briques.

    Si le choc survient (B=1, probabilité p_sys) : les dim composantes sont
    PARFAITEMENT corrélées (même tirage uniforme pour toutes).
    Si le choc ne survient pas (B=0) : les dim composantes sont indépendantes.

    C'est une représentation plus grossière mais plus interprétable de la
    contagion EXTERNE (cf. variable latente, paramètre γ) — utile en
    robustesse face à la copule de Gumbel, qui capture surtout la
    contagion INTERNE (dépendance entre briques d'un même incident).

    p_sys peut être directement ancré sur γ de la variable latente
    (cf. compliance/latent.py, ANCHORED_PARAMS['gamma'] ≈ 0.68).
    """
    rng = np.random.default_rng(seed)
    B = rng.random(n_sim) < p_sys

    U_common = rng.uniform(0, 1, size=n_sim)       # facteur commun
    U_idio = rng.uniform(0, 1, size=(n_sim, dim))  # chocs idiosyncratiques

    U = np.where(B[:, None], U_common[:, None], U_idio)
    return U


# ---------------------------------------------------------------------------
# MAIN — validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  VALIDATION — Copule de Gumbel (θ=1.8)")
    print("=" * 60)
    tau_emp, tau_theo = empirical_kendall_tau(theta=1.8, n_sim=50_000)
    print(f"  τ de Kendall théorique (1 - 1/θ) = {tau_theo:.4f}")
    print(f"  τ de Kendall empirique (50k sims) = {tau_emp:.4f}")
    print(f"  Écart = {abs(tau_emp - tau_theo):.4f} (doit être proche de 0)")

    print(f"\n  Échantillon dim=4, n=5 (aperçu) :")
    U = gumbel_copula_uniforms(5, theta=1.8, dim=4, seed=1)
    for row in U:
        print(f"    {row.round(3)}")

    print(f"\n{'='*60}")
    print(f"  VALIDATION — Modèle à facteur commun (p_sys=0.68, ancré γ)")
    print(f"{'='*60}")
    Uc = common_factor_uniforms(50_000, p_sys=0.68, dim=4, seed=1)
    corr = np.corrcoef(Uc.T)
    print(f"  Corrélation moyenne entre briques (hors diagonale) = "
          f"{(corr.sum()-4)/12:.4f}")
    print(f"  (attendu proche de p_sys={0.68} si choc binaire domine)")
