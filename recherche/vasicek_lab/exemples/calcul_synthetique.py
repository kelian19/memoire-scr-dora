"""Une année d'incidents jusqu'au capital, avec des paramètres entièrement fictifs.

Exemple pédagogique de cascade indépendante, pas une reproduction du chiffre principal
obtenu par marche auto-évitante. Seul le calcul des ensembles atteints est réutilisé.
Aucun fichier de données, de calibration ou de résultat n'est lu par cet exemple.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import nbinom

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
from partial_id import card_dist_all  # noqa: E402


@dataclass(frozen=True)
class Etat:
    nom: str
    frequence: float
    depassement: float
    propagation: float
    accumulation: float


# Conventions inventées pour l'exemple, en M€ fictifs. Aucun ajustement sur une base réelle.
ETATS = (
    Etat("Conforme fictif", 2.0, 0.10, 0.30, 0.0),
    Etat("Non conforme fictif", 3.5, 0.16, 0.60, 0.45),
)
DISPERSION = 2.0  # variance du nombre d'incidents / moyenne
XI, SIGMA, SEUIL = 0.30, 0.50, 1.0
ALPHA = 0.995
LIENS = np.array([
    [0, .35, .25, .25, .15],
    [.20, 0, .35, .30, .15],
    [.20, .30, 0, .30, .20],
    [.30, .25, .25, 0, .20],
    [.25, .25, .25, .25, 0],
])  # Chaque ligne somme à 1 ; le gain multiplie cette matrice fictive.
MASQUES = np.arange(32)
INDICATEURS = ((MASQUES[:, None] >> np.arange(5)) & 1).astype(bool)


def probabilites_ensembles(etat: Etat) -> np.ndarray:
    """Loi des ensembles atteints ; ligne j = pilier d'amorce j (indexé depuis 0)."""
    _, ensembles = card_dist_all(etat.propagation * LIENS)
    # Lors d'une amorce P4, le choc commun remplace la cascade : P4 plus chacun des
    # quatre autres piliers, indépendamment avec probabilité omega.
    autres = INDICATEURS.sum(axis=1) - 1
    avec_p4 = INDICATEURS[:, 3]
    ensembles[3] = 0.0
    ensembles[3, avec_p4] = (
        etat.accumulation ** autres[avec_p4]
        * (1.0 - etat.accumulation) ** (4 - autres[avec_p4])
    )
    # La récurrence exacte peut produire un arrondi négatif de l'ordre de 1e-16.
    ensembles = np.maximum(ensembles, 0.0)
    return ensembles / ensembles.sum(axis=1, keepdims=True)


def mesures(pertes: np.ndarray, alpha: float = ALPHA) -> dict:
    """VaR empirique et ES intégrée, y compris lorsque des pertes sont à égalité."""
    capital = float(np.quantile(pertes, alpha, method="inverted_cdf"))
    return {
        "moyenne": float(pertes.mean()),
        "var": capital,
        "es": capital + float(np.maximum(pertes - capital, 0).mean()) / (1 - alpha),
        "erreur_type_moyenne": float(pertes.std(ddof=1) / np.sqrt(len(pertes))),
    }


def simuler(annees: int = 20_000, graine: int = 42, etats=ETATS) -> list[dict]:
    """Uniformes communs par année, rang d'incident et pilier entre tous les états.

    Les incidents supplémentaires d'un état ont leurs propres rangs. L'appariement
    n'implique ni monotonie trajectoire par trajectoire, ni suppression du bruit MC.
    """
    if annees < 2 or not etats:
        raise ValueError("Au moins deux années et un état sont nécessaires.")
    for etat in etats:
        if not np.isfinite(etat.frequence) or etat.frequence < 0:
            raise ValueError("La fréquence doit être finie et positive ou nulle.")
        if not all(np.isfinite(x) and 0 <= x <= 1 for x in
                   (etat.depassement, etat.propagation, etat.accumulation)):
            raise ValueError("Les probabilités et le gain doivent être dans [0, 1].")

    rng = np.random.default_rng(graine)
    uniformes_n = np.clip(rng.random(annees), np.nextafter(0., 1.), np.nextafter(1., 0.))
    comptages = [
        nbinom.ppf(uniformes_n, e.frequence / (DISPERSION - 1), 1 / DISPERSION).astype(int)
        if e.frequence else np.zeros(annees, dtype=int) for e in etats
    ]
    rangs = max(1, max(int(n.max()) for n in comptages))
    forme = (annees, rangs)
    amorces = rng.integers(0, 5, size=forme)
    uniformes_cascade = rng.random(forme)
    uniformes_materiel = rng.random((*forme, 5))
    uniformes_perte = rng.random((*forme, 5))
    # Quantile GPD des excès : F^-1(v) = sigma/xi * ((1-v)^(-xi) - 1).
    severites = SEUIL + SIGMA / XI * np.expm1(-XI * np.log1p(-uniformes_perte))
    resultats = []
    for etat, nombres in zip(etats, comptages):
        ensembles = probabilites_ensembles(etat)
        cdf = ensembles.cumsum(axis=1)
        cdf[:, -1] = 1.0
        masques = np.zeros(forme, dtype=int)
        for j in range(5):
            selection = amorces == j
            masques[selection] = np.searchsorted(cdf[j], uniformes_cascade[selection], side="right")
        atteints = INDICATEURS[masques]
        actifs = np.arange(rangs)[None, :] < nombres[:, None]
        pertes_incident = (severites * (uniformes_materiel < etat.depassement) * atteints).sum(axis=2)
        pertes_incident *= actifs
        pertes_annuelles = pertes_incident.sum(axis=1)

        # E[L] calculable exactement pour ce cas fictif, distincte du capital VaR.
        taille_moyenne = float((ensembles @ INDICATEURS.sum(axis=1)).mean())
        moyenne_theorique = (etat.frequence * taille_moyenne * etat.depassement
                             * (SEUIL + SIGMA / (1 - XI)))
        # Une année avec perte positive est sélectionnée pour expliquer le calcul,
        # sans prétendre qu'elle soit représentative des années simulées.
        positives = np.flatnonzero(pertes_annuelles > 0)
        annee = int(positives[0]) if len(positives) else 0
        incidents = [
            {"rang": k + 1, "amorce": f"P{amorces[annee, k] + 1}",
             "piliers_atteints": [f"P{j + 1}" for j in np.flatnonzero(atteints[annee, k])],
             "perte": float(pertes_incident[annee, k])}
            for k in range(int(nombres[annee]))
        ]
        resultats.append({
            "etat": asdict(etat), "annees": annees, "graine": graine,
            "alpha": ALPHA, "unite": "M EUR fictifs", "statut": "pedagogique non calibre",
            "severite": {"xi": XI, "sigma": SIGMA, "seuil": SEUIL,
                         "sous_seuil": "perte nulle", "plafond": None},
            "dispersion": DISPERSION, "matrice_liens_fictive": LIENS.tolist(),
            "mesures": mesures(pertes_annuelles), "moyenne_theorique": moyenne_theorique,
            "exemple_annee": {"numero": annee + 1, "incidents": incidents,
                              "perte_annuelle": float(pertes_annuelles[annee])},
        })
    return resultats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annees", type=int, default=20_000)
    parser.add_argument("--graine", type=int, default=42)
    parser.add_argument("--sortie", type=Path, help="JSON facultatif ; aucun fichier écrit par défaut")
    args = parser.parse_args()
    if args.annees < 2 or args.graine < 0:
        parser.error("--annees >= 2 et --graine >= 0 sont requis")
    resultats = simuler(args.annees, args.graine)
    print("EXEMPLE FICTIF - cascade indépendante, aucun montant du mémoire reproduit")
    print(f"{args.annees:,} années ; graine {args.graine} ; VaR/ES à 99,5 % ; M EUR fictifs")
    for resultat in resultats:
        e, m = resultat["etat"], resultat["mesures"]
        print(f"\n{e['nom']} : lambda={e['frequence']}, p_u={e['depassement']}, "
              f"g={e['propagation']}, omega={e['accumulation']}")
        print(f"Moyenne = {m['moyenne']:.3f} (théorie {resultat['moyenne_theorique']:.3f}) ; "
              f"VaR = {m['var']:.3f} ; ES = {m['es']:.3f}")
        an = resultat["exemple_annee"]
        print(f"Année {an['numero']} choisie avec perte positive :")
        for incident in an["incidents"]:
            print(f"  Incident {incident['rang']} : {incident['amorce']} -> "
                  f"{', '.join(incident['piliers_atteints'])} ; perte {incident['perte']:.3f}")
        print(f"  Somme annuelle = {an['perte_annuelle']:.3f}")
    ecart = resultats[1]["mesures"]["var"] - resultats[0]["mesures"]["var"]
    print(f"\nDifférence des deux VaR = {ecart:.3f} M EUR fictifs.")
    print("Ce calcul ne représente ni une différence de pertes moyennes, ni une validation empirique.")
    if args.sortie:
        args.sortie.parent.mkdir(parents=True, exist_ok=True)
        args.sortie.write_text(json.dumps(resultats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
