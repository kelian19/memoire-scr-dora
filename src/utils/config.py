"""
utils/config.py
---------------
Configuration centrale du projet SCR_DORA.
Tous les paramètres calibrés sont centralisés ici.

MISE À JOUR : les résultats centraux (SCR_DORA, COPULE) ont été synchronisés
avec les vrais résultats du pipeline (grille bootstrap Δ_DORA + LDA 4 briques
+ allocation d'Euler). Les anciennes valeurs placeholder ont été retirées.
"""

import os

# ---------------------------------------------------------------------------
# PARAMÈTRES CALIBRÉS — PRC 2025
# ---------------------------------------------------------------------------

PRC = {
    "source": "Privacy Rights Clearinghouse 2025 — Data_Breach_Chronology.xlsx",
    "period": "2019–2025",
    "n_records": 15053,          # incidents avec total_affected > 0, période retenue
    "seuil_u_eur": 4.176,        # M€ (seuil POT après conversion Jacobs, percentile 85)
    "xi": 1.0328,                # paramètre de queue GPD (MLE, IC90% [0.971, 1.094])
    "xi_ic90": [0.9709, 1.0939],
    "sigma_eur": 6.507,          # M€ (paramètre d'échelle, IC90% [6.091, 6.985])
    "sigma_ic90": [6.091, 6.985],
    "n_excess": 2258,
    "p_u": 0.15,                 # P(X > u)
    "var_995": 209.18,           # M€
    "var_995_ic90": [182.75, 240.52],
    "jacobs_a": 7.68,            # ln(L_usd) = a + b*ln(X) — Jacobs (2014), base NATURELLE
    "jacobs_b": 0.76,            # paramètre conversion log-log
    "usd_eur": 0.92,             # taux de conversion
    "note": (
        "Calibration RECALCULÉE en propre sur les données PRC brutes (voir "
        "notebooks/13_prc_jacobs_calibration.py). Les valeurs précédentes "
        "(ξ=1.30, u=0.128 M€, σ=0.257 M€) provenaient d'une référence externe, "
        "jamais reproduites en interne à partir de ce fichier (n_records=None "
        "en attestait) : une validation empirique (log10 vs ln, coefficients "
        "Jacobs 2014 vs extension 2018 CODB, sur plusieurs périodes/seuils) n'a "
        "reproduit aucune des trois. Cette calibration-ci est reproductible et "
        "tracée de bout en bout : conversion ln(L_usd)=7.68+0.76*ln(X), période "
        "2019-2025, seuil = percentile 85 (p_u=0.15, même convention qu'OpRisk)."
    ),
}

# ---------------------------------------------------------------------------
# PARAMÈTRES CALIBRÉS — SAS OpRisk Global (juin 2026)
# ---------------------------------------------------------------------------

OPRISK = {
    "source": "SAS OpRisk Global Data, juin 2026",
    "perimetre": "Systems Security + Business Disruption — Finance (2000–2026)",
    "n_incidents": 582,          # périmètre cyber×finance reconstruit (filtrage validé)
    "n_excess": 91,              # excès au-dessus de seuil_u_eur dans la donnée courante
    "seuil_u_eur": 20.03,        # M€ — percentile 84,4 (et NON 85 : voir la note ci-dessous)
    "xi": 0.5954,                # paramètre de queue GPD
    "sigma_eur": 57.97,          # M€
    # TAUX DE DÉPASSEMENT GELÉ, ET IL NE CORRESPOND PAS AU SEUIL CI-DESSUS. Ne pas le
    # « corriger » sans avoir lu le bloc OPRISK_COHERENCE, en bas de fichier, qui recalcule
    # l'écart et son effet à chaque import.
    "p_u": 0.1509,
    "xi_ic90": [0.3044, 0.8313],
    "sigma_ic90": [41.88, 82.80],
    # ATTENTION AU NOM DE CE CHAMP. C'est le quantile à 99,5 % de la SÉVÉRITÉ d'un sinistre
    # unique, et non un besoin de capital : la mesure de capital est le quantile de la charge
    # ANNUELLE AGRÉGÉE, qui vaut plusieurs milliers de M€ au secteur. Le nom « var_995 » est
    # conservé parce que des scripts le lisent, mais VaR et TVaR sont réservées à l'agrégé
    # dans le mémoire. Le pont entre les deux échelles est imprimé par le script 67 (1bis).
    "var_995": 662.78,           # M€ (formule POT corrigée) — quantile de sévérité, pas un SCR
    "tvar_99": 1133.05,
    "n_years": 27,               # période 2000–2026 (fréquence propre OpRisk = 582/27 ≈ 21,6/an)          # M€
    "var_995_ic90": [411.5, 1037.1],
    "note": (
        "Source de montants de pertes RÉELS. Biais vers les grandes entités "
        "financières → queue possiblement sous-estimée. Complémentaire de la PRC "
        "(qui couvre un spectre de tailles plus large mais via sévérité dérivée Jacobs). "
        "Pas de hiérarchie 'primaire' en sévérité : deux estimations indépendantes. "
        "Fréquence propre OpRisk ≈ n_incidents/27 ≈ 21,6/an (2000-2026), "
        "à ne JAMAIS mélanger avec λ_ref PRC."
    ),
}

# ECHANTILLON DE CALIBRATION, 15 OCTOBRE 2026. La calibration gelee (u, p_u, xi, sigma ci-dessus)
# est celle de l'echantillon 2000-2026 : 570 pertes, 86 exces, p_u = 86/570 = 0,1509, xi = 0,5952,
# sigma = 57,98, quantile 662,8. Les comptes 582 et 91 ci-dessus sont ceux de la base courante, qui
# porte douze pertes d'avant 2000. Les versions deposees (ENSAE, Institut de 195 pages) gardent ces
# comptes. La version Institut a 150 pages rejoue ses tests de severite sur l'echantillon 2000-2026 :
# elle lance les scripts avec OPRISK_ANNEE_MIN=2000, ce qui restreint la periode (filter_finance) et
# aligne ici les comptes attendus. Sans la variable, rien ne change.
if os.environ.get("OPRISK_ANNEE_MIN") == "2000":
    # n_incidents reste a 582 : c'est le numerateur de la frequence gelee (lambda = 582/27 = 21,56),
    # lue par le moteur. Le compte de l'echantillon de severite est porte a part.
    OPRISK["n_pertes_echantillon"] = 570
    OPRISK["n_excess"] = 86

# ---------------------------------------------------------------------------
# CONTRÔLE DE COHÉRENCE DU TAUX DE DÉPASSEMENT OpRisk
# ---------------------------------------------------------------------------
# POURQUOI CE BLOC EXISTE. p_u N'EST PAS UN PARAMÈTRE LIBRE. La formule POT a trois entrées
# (u, p_u, (ξ, σ)) mais deux degrés de liberté seulement : une fois le seuil et l'échantillon
# fixés, le taux de dépassement est COMPTÉ, il n'est pas choisi. Publier le couple
# (u = 20,03 M€ ; p_u = 0,1509) n'est donc pas une hypothèse de modélisation que l'on pourrait
# assumer comme telle : il n'existe aucun état du monde où les deux sont vrais ensemble.
# C'est une incohérence arithmétique, et un lecteur qui divise 91 par 582 la trouve en trente
# secondes. Le 0,1509 vaut 88/583, le taux du percentile 85 d'une version antérieure du
# filtrage ; le seuil publié est au percentile 84,4 et donne 91 excès sur 582.
#
# CE QUE L'ON EN FAIT, ET POURQUOI PAS AUTRE CHOSE. Trois postures étaient possibles.
#   (1) Recalibrer. C'est juste en principe, et c'est ce qu'il faut faire si le temps le
#       permet. Mais le pipeline aval est stochastique (bootstrap n_boot = 200, n_sim = 20 000
#       pour la grille Δ_DORA) : le rejouer réinjecterait un bruit de Monte-Carlo du même
#       ordre, voire supérieur, à l'effet que l'on cherche à corriger. On déplacerait plusieurs
#       centaines de nombres publiés sans pouvoir attribuer un seul de ces déplacements à la
#       correction. On perdrait la piste d'audit pour gagner 2,4 % sur une grandeur dont
#       l'intervalle de calibration couvre un facteur 2,5.
#   (2) Requalifier le couple pour le rendre cohérent sur le papier. Refusé : c'est une
#       reformulation qui rend la réserve moins visible, exactement ce que ce mémoire s'interdit.
#   (3) Geler la valeur publiée, rendre l'écart calculable à chaque import, et le PUBLIER comme
#       une limite chiffrée et signée dans l'inventaire des hypothèses. C'est le traitement
#       standard d'une erreur de paramètre connue, quantifiée, immatérielle et détectée tard :
#       on ne rebase pas le modèle, on inscrit l'écart au registre des limites avec son sens et
#       sa taille. C'est la posture retenue.
#
# LE SENS COMPTE PLUS QUE LA TAILLE. L'écart va dans le sens de la SOUS-estimation du capital.
# On ne peut donc pas l'excuser par la prudence : une sous-estimation ne se couvre pas par un
# argument de prudence, elle se déclare. D'où sa présence dans l'inventaire du chapitre 13.


def _var_pot(u, xi, sigma, p_u, q=0.995):
    """VaR de niveau q par la formule POT, à seuil et taux de dépassement donnés."""
    return u + (sigma / xi) * (((1.0 - q) / p_u) ** (-xi) - 1.0)


def _coherence_oprisk():
    o = OPRISK
    p_pub = o["p_u"]
    p_coh = o["n_excess"] / o.get("n_pertes_echantillon", o["n_incidents"])
    args = (o["seuil_u_eur"], o["xi"], o["sigma_eur"])
    # On raisonne en RAPPORT, pas en niveau : la VaR recalculée à partir des ξ et σ arrondis
    # du dictionnaire vaut 662,99 et non les 662,78 publiés. Le rapport, lui, est insensible
    # à cet arrondi, et c'est lui qui porte l'effet du taux de dépassement.
    ratio = _var_pot(*args, p_coh) / _var_pot(*args, p_pub)
    # LA MATERIALITE SE MESURE CONTRE L'INCERTITUDE DEJA PUBLIEE, pas dans l'absolu. Le bon
    # denominateur est la largeur de l'IC90 de la meme VaR : c'est la precision que le memoire
    # revendique, et un ecart qui tient dans un quarantieme de cette largeur ne change aucune
    # lecture. Calcule et non ecrit a la main : une premiere version de ce commentaire portait
    # « un soixantieme », qui etait faux.
    ic = OPRISK["var_995_ic90"]
    ecart_abs = OPRISK["var_995"] * (ratio - 1.0)
    part_ic = ecart_abs / (ic[1] - ic[0])
    return {
        "p_u_publie": p_pub,
        "p_u_coherent": p_coh,
        "ecart_relatif_p_u": p_coh / p_pub - 1.0,
        "var_995_publiee": o["var_995"],
        "var_995_coherente": o["var_995"] * ratio,
        "ecart_relatif_var": ratio - 1.0,
        "sens": ("aucun ecart de taux" if abs(ratio - 1.0) < 1e-3 else "sous-estimation du capital"),
        "ecart_absolu": ecart_abs,
        "part_largeur_ic90": part_ic,
        "materialite": (f"{ecart_abs:.1f} M EUR, soit {100*part_ic:.1f} % de la largeur de "
                        f"l'IC90 de cette meme VaR ([{ic[0]:.1f} ; {ic[1]:.1f}] M EUR)"),
        "decision": ("taux compte egal au taux publie" if abs(ratio - 1.0) < 1e-3 else "valeur gelée, écart publié au chapitre 13 (inventaire des hypothèses)"),
    }


OPRISK_COHERENCE = _coherence_oprisk()

# ---------------------------------------------------------------------------
# PARAMÈTRES CALIBRÉS — FRÉQUENCE (PRC 2019-2025)
# ---------------------------------------------------------------------------

FREQUENCY = {
    "source": "PRC 2025 — MLE NegBin",
    "lambda_ref": 341,           # incidents/an (fréquence de référence)
    "r": None,                   # à calibrer sur PRC
    "p": None,
    "dispersion_factor": 9.20,   # Var/Mean observé — MAINTENU constant entre scénarios
    "facteur_recalibration": 1.30,
}

# ---------------------------------------------------------------------------
# PARAMÈTRES DÉPENDANCE — COPULE GUMBEL & FACTEUR COMMUN
# ---------------------------------------------------------------------------

COPULE = {
    "famille": "Gumbel",
    "theta": 1.8,                # = theta_nc (régime non-conforme). τ de Kendall ≈ 0.444
    "theta_nc": 1.8,             # régime non-conforme
    "theta_c": 1.2,              # régime conforme (contrefactuel)
    "justification": "Dépendance de queue supérieure entre les 4 briques",
    # ANCRAGE EMPIRIQUE (notebooks/19_theta_empirical_anchoring.py) : le
    # co-mouvement inter-vecteurs de la PRC (2019-2024, 35 998 incidents)
    # implique θ ∈ [1.10, 1.23] (routes τ de Kendall + λ_U de queue), soit un
    # PLANCHER de la dépendance inter-briques (la contagion intra-incident visée
    # est plus forte qu'un co-mouvement inter-catégories). La FORME queue-
    # supérieure est confirmée (λ_U empirique ≈ 0.25 ≈ 2× celui impliqué par le
    # τ central). Surtout : le Δ_DORA varie <1.4% (OpRisk) / <0.3% (PRC) quand
    # θ_nc balaie [1 (indépendance), 4] → le verdict ne repose PAS sur θ.
    "theta_empirical_band": [1.10, 1.23],
    "theta_delta_dora_sensitivity": "<1.4% sur θ_nc ∈ [1, 4] (invariant)",
    "alternative": "Facteur commun B~Bernoulli(p_sys)",
    # p_sys ancré sur γ de la variable latente (concentration marché cloud),
    # et non plus une valeur arbitraire. Voir compliance/latent.py ANCHORED_PARAMS.
    "p_sys": 0.68,               # proxy = part de marché AWS+Azure+GCP (Synergy/Canalys 2025)
    # Chargement systémique sur l'aggravation (couplage à la copule) —
    # HYPOTHÈSE DE MODÉLISATION non sourcée, bornée conservativement.
    "aggravation_stress_range": [0.0, 0.15],
}

# ---------------------------------------------------------------------------
# RÉSULTATS CENTRAUX — SCR_DORA (synchronisés avec le pipeline)
# ---------------------------------------------------------------------------

# Grille Δ_DORA = SCR(scénario) - SCR(S0), bootstrap deux niveaux.
# Médiane et IC90% en M€. bootstrap_severity=True pour LES DEUX sources
# désormais : OpRisk (91 excès bruts) ET PRC (2258 excès de la sévérité
# dérivée Jacobs, cf. src/severity/prc_analysis.py). Les IC des deux sources
# sont donc data-driven ; celui de PRC reste plus resserré car la conversion
# Jacobs (transformation déterministe X->L) lisse la variance de sévérité.
# Régénérée via notebooks/07_bootstrap_delta_dora.py (n_boot=200, n_sim=20 000)
# APRÈS (a) recalibration de la brique prestataire en surcoût relatif et
# (b) recalibration PRC en propre sur les données brutes (remplacement des
# valeurs externes ξ=1.30/u=0.128/σ=0.257 par ξ=1.033/u=4.176/σ=6.507).
# PRC et OPRISK régénérés et vérifiés dans cet environnement (voir
# outputs/tables/results_delta_dora_bootstrap.csv).
DELTA_DORA_GRID = {
    ("PRC", "S1_partiel"):      {"median": 720.9,  "ic90": [555.7, 895.5],      "bootstrap_sev": True},
    ("PRC", "S2_non_conforme"): {"median": 2014.7, "ic90": [1606.9, 2366.1],    "bootstrap_sev": True},
    ("OPRISK", "S1_partiel"):   {"median": 1514.4, "ic90": [522.5, 7863.2],     "bootstrap_sev": True},
    ("OPRISK", "S2_non_conforme"): {"median": 3879.3, "ic90": [1496.9, 22249.3], "bootstrap_sev": True},
}

# Décomposition par brique (allocation d'Euler, VaR 99.5%, profil médian, theta=0),
# architecture à 3 briques additives (remediation/prestataire/sanction) — l'aggravation
# n'en fait PAS partie : c'est un delta contrefactuel (cf. src/aggregation/lda.py::
# scr_4_briques_report), pas une composante de la somme. Voir
# outputs/tables/results_euler_option_a.csv pour la grille complète par profil et source.
# Prestataire recalibré en surcoût relatif IBM/Ponemon (+8.3%/+11.8%) sur la
# sévérité de remédiation active, au lieu d'une Lognormale à échelle absolue :
# restaure la commensurabilité entre briques et entre sources PRC/OpRisk.
EULER_DECOMPOSITION = {
    "OPRISK": {"remediation": 0.860, "prestataire": 0.140, "sanction": 0.0003},
    "PRC":    {"remediation": 0.854, "prestataire": 0.141, "sanction": 0.005},
    "note": (
        "Remédiation dominante sous les deux sources (85-86%), prestataire "
        "minoritaire mais non négligeable (14%), structure cohérente entre "
        "sources depuis la recalibration du prestataire en surcoût relatif. "
        "La sanction reste marginale sous les deux sources."
    ),
}

SCR_DORA = {
    "cap_eur": 40.0,             # M€ — plafond de sévérité PRC (ξ≥1), ancré capacité réassurance
    "constat_source_domine": (
        "APRÈS recalibration PRC en propre (ξ=1.033 vs OpRisk ξ=0.595), l'écart "
        "entre sources sur le Δ_DORA médian S2 s'est FORTEMENT RESSERRÉ : PRC "
        "2014.7 M€ vs OpRisk 3879.3 M€, soit un facteur ~1.9 (contre ~12 avec "
        "l'ancienne calibration PRC externe). Les deux sources indépendantes "
        "CONVERGENT donc bien plus qu'anticipé une fois PRC calibrée de bout en "
        "bout sur données réelles. Ce qui reste dominant n'est plus l'écart de "
        "NIVEAU entre sources mais l'écart d'INCERTITUDE : l'IC90% OpRisk "
        "(bootstrap 91 excès, facteur ~15 entre bornes) reste bien plus large "
        "que celui de PRC (2258 excès, sévérité lissée par Jacobs)."
    ),
    "note": (
        "Le SCR_DORA est une distribution large, pas un point. Trois sources "
        "d'incertitude se cumulent : calibration (bootstrap), source de sévérité "
        "(PRC vs OpRisk), et scénario de conformité (S0/S1/S2). Les deux sources "
        "sont désormais bootstrappées sur données brutes ; l'IC90% PRC reste plus "
        "resserré que celui d'OpRisk car la conversion Jacobs lisse la variance "
        "de sévérité (transformation déterministe du nombre d'enregistrements)."
    ),
}

# ---------------------------------------------------------------------------
# HACKMAGEDDON — PROPORTIONS S1 2026
# ---------------------------------------------------------------------------

HACKMAGEDDON = {
    "source": "Hackmageddon (Paolo Passeri)",
    "periode": "Janvier–Juin 2026",
    "n_incidents": 1041,
    "n_identifies": 840,
    "taux_identification": 0.807,
    "proportions": {
        "phishing_social_eng": 0.388,
        "exploit_vuln":        0.338,
        "supply_chain_tiers":  0.158,
        "identifiants":        0.063,
        "autres":              0.053,
    },
    "surface_tlpt": 0.496,       # exploit + supply chain (art. 26)
    "surface_tiers": 0.158,      # supply chain seul (art. 28-44)
    # COMPARAISON DE STRUCTURE 2023 CONTRE 2026, ENREGISTREE ICI PLUTOT QUE FLOTTANTE.
    # Ces valeurs vivaient uniquement dans la prose du chapitre données, donc dans aucun
    # script : le harnais ne pouvait ni les confirmer ni les infirmer. Elles ont exactement le
    # même statut que les 1 041 incidents ci-dessus, une CITATION EXTERNE non recalculable, et
    # elles doivent donc figurer au même endroit et sous la même étiquette. Les enregistrer ne
    # rouvre pas le statut de la source, cela le rend vérifiable.
    # L'ARTEFACT QU'ELLES DOCUMENTENT. La catégorie « Ransomware » n'existait pas en 2023, ces
    # incidents étant classés sous « Malware ». Une lecture naïve concluait à une hausse du
    # ransomware de +11 points ; après reclassement par grille de mots-clés commune, il recule
    # de 23 points. C'est l'exemple qui justifie de n'utiliser de cette base que la structure.
    "comparaison_2023_2026": {
        "n_2023": 3019,          # incidents 2023, Q3 manquant
        "trimestre_manquant_2023": "Q3",
        "motivations": {
            # dimension : (part 2023, part 2026) en fraction
            "cybercriminalite": (0.811, 0.739),
            "cyberespionnage": (0.096, 0.206),
            "ransomware_reclasse": (0.358, 0.127),
        },
        "hausse_ransomware_lecture_naive_pts": 11,
        "recul_ransomware_apres_reclassement_pts": 23,
    },
}

# ---------------------------------------------------------------------------
# LUCY 2026 — MARCHÉ FRANÇAIS DE LA CYBERASSURANCE
# ---------------------------------------------------------------------------
# STATUT : CITATION EXTERNE, NON RECALCULABLE, exactement celui de HACKMAGEDDON
# ci-dessus. Le jeu de polices et de sinistres de LUCY n'est pas versionné dans
# data/raw/ et ne le sera pas : il est agrégé et confidentiel par construction.
# Ces valeurs sont enregistrées ici pour être CITABLES, non pour être rejouées.
#
# CE QUE CETTE SOURCE APPORTE AU MÉMOIRE, ET CE QU'ELLE N'APPORTE PAS. Elle
# constate de l'extérieur les deux prémisses du mémoire : qu'aucun module de
# capital standardisé ne couvre le risque de catastrophe cyber sous Solvabilité II,
# et que la queue française est censurée en partie haute, ce qui rend un quantile
# à 99,5 % instable et biaisé vers le bas. Elle ne porte AUCUN niveau de capital du
# mémoire et n'entre dans aucune calibration.
#
# LE PIÈGE D'ÉCHELLE, ET IL EST DU MÊME TYPE QUE CELUI DU 7 AOÛT 2026. La charge de
# LUCY est une charge INDEMNISÉE, c'est-à-dire min(capacité, max(sinistre −
# franchise, 0)) sommée sur un portefeuille de marché. La sévérité du mémoire est
# une perte opérationnelle BRUTE d'entité financière. Les deux ne se comparent pas,
# ni en niveau ni en quantile : rapprocher les 83,2 M EUR indemnisés du marché
# français des 662,78 M EUR de quantile unitaire ferait lire une différence de
# périmètre et de rétention comme une contradiction.
LUCY_2026 = {
    "source": "AMRAE, LUCY 2026 (Lumière sur la Cyberassurance), exercice 2025",
    "analyse": ("Nexialog Consulting, Rapport LUCY 2026, 9 juin 2026, "
                "H. Rapior et K. Kaddouri"),
    "n_polices": 20996,
    "n_sinistres": 1251,
    "n_courtiers": 12,
    "n_assureurs": 1,
    # Ratios sinistres sur primes agrégés, et sur le seul segment des ETI.
    "sp_2024": 0.17,
    "sp_2025": 0.27,
    "sp_eti_2024": 0.13,
    "sp_eti_2025": 0.42,
    # Charge INDEMNISÉE, nette de franchise et plafonnée par la capacité. M EUR.
    "charge_2024_eur": 54.5,
    "charge_2025_eur": 83.2,
    "hausse_charge_2025": 0.53,
    # Reculs de taux de prime, stockés en valeur ABSOLUE et nommés « recul » : un
    # signe négatif dans une sortie se lit mal et le harnais l'a déjà mal lu.
    "recul_taux_prime_grandes": 0.32,
    "recul_taux_prime_eti": 0.23,
    # LA VALEUR QUI COMPTE LE PLUS POUR LE MÉMOIRE : la queue française est vide.
    "n_sinistres_sup_10m_france_2025": 1,
    "seuil_xl_eur": 3.0,
    "seuil_xxl_eur": 10.0,
    # LES DEUX RÉGIMES OPPOSÉS SOUS UNE MÊME TRAJECTOIRE DE RATIO CROISSANTE.
    #
    # UNE IMPRÉCISION DE LA SOURCE, CORRIGÉE ICI ET NON REPRISE. Le rapport nomme
    # « fréquence » le multiplicateur du NOMBRE de sinistres quand il écrit que la
    # charge se décompose. Ce n'est pas la fréquence : la fréquence est un nombre
    # de sinistres par assuré, et ses propres tables la donnent à 1,88 pour 2025
    # quand le nombre de sinistres est à 2,79, l'exposition ayant crû de 1,49.
    # L'identité charge = nombre x sinistre moyen n'est exacte qu'avec le NOMBRE,
    # et c'est donc sous ce nom que les valeurs sont enregistrées. Avec la vraie
    # fréquence il faut trois facteurs : charge = exposition x fréquence x sinistre
    # moyen. La distinction n'est pas byzantine, c'est elle qui sépare un effet de
    # VOLUME d'une dégradation technique à exposition donnée.
    "mult_nombre_2024": 0.73,
    "mult_sinistre_moyen_2024": 1.96,
    "mult_charge_2024": 1.43,
    "mult_nombre_2025": 2.79,
    "mult_sinistre_moyen_2025": 0.55,
    "mult_charge_2025": 1.53,
    "mult_exposition_2025": 1.49,
    "mult_frequence_2025": 1.88,
    # Décomposition 2025 par bloc : (nombre, sinistre moyen, charge). Le bloc
    # intermédiaire est le SEUL où le nombre et le sinistre moyen montent ensemble.
    "blocs_2025": {
        "grandes entreprises": (1.24, 0.82, 1.02),
        "bloc intermediaire": (2.14, 1.65, 3.53),
        "micro-entreprises": (9.52, 0.73, 6.97),
    },
    # -------------------------------------------------------------------
    # TRANSCRIPTION ÉLARGIE, 9 SEPTEMBRE 2026.
    #
    # Le rapport n'entrait dans l'introduction que par trois constats.
    # Kélian a demandé que la lecture de marché en occupe une part
    # substantielle, de façon à ce que le mémoire situe le risque cyber
    # dans le marché qui l'assure avant de le charger en capital. Tout ce
    # qui suit est donc transcrit du rapport co-écrit, sous le MÊME statut
    # de CITATION EXTERNE NON RECALCULABLE : aucune de ces valeurs n'entre
    # dans une calibration, aucune ne porte un niveau de capital.
    #
    # LES SÉRIES SONT STOCKÉES BRUTES ET LES RATIOS SE CALCULENT DANS LE
    # SCRIPT 63. C'est la règle du projet : un rapport écrit à la main
    # pendant la rédaction n'est vérifiable par personne. Le script en
    # tire aussi trois CONTRÔLES D'IDENTITÉ qui valident la transcription
    # elle-même, et c'est leur seul objet :
    #   - sinistres / primes doit redonner la série des S/P, année par année ;
    #   - la somme des quatre classes de taille doit redonner la charge
    #     annuelle indemnisée, année par année ;
    #   - la somme des charges des trois blocs doit redonner la charge 2025.
    # Une valeur recopiée de travers casse l'un des trois.
    # -------------------------------------------------------------------
    # Indicateurs de souscription du marché. Couple (2024, 2025) dans cet
    # ordre, unité en troisième position.
    "marche_2024_2025": {
        "entreprises assurees":         (14124.0, 20996.0, "unite"),
        "primes souscrites":            (316.8, 305.9, "M EUR"),
        "prime unitaire moyenne":       (22427.0, 14567.0, "EUR"),
        "capacite moyenne souscrite":   (2.04, 1.88, "M EUR"),
        "franchise moyenne":            (149.9, 89.0, "k EUR"),
        "taux de prime annuel moyen":   (0.28, 0.26, "%"),
        "sinistres indemnises":         (448.0, 1251.0, "unite"),
        "charge indemnisee":            (54.5, 83.2, "M EUR"),
    },
    # Séries longues 2019-2025, celles que portent les figures du rapport.
    # Elles sont transcrites pour que le mémoire les commente au lieu de
    # lire un point sur un graphique, travers déjà corrigé sur le pic à
    # trois heures du script 08h.
    "annees_serie": (2019, 2020, 2021, 2022, 2023, 2024, 2025),
    "primes_serie_eur": (86.0, 128.0, 183.0, 316.0, 328.0, 317.0, 306.0),
    "sinistres_serie_eur": (74.0, 217.0, 164.0, 71.0, 38.0, 55.0, 83.0),
    "sp_serie": (0.85, 1.69, 0.89, 0.22, 0.12, 0.17, 0.27),
    # Conditions de souscription, mêmes années. La franchise n'est publiée
    # qu'à partir de 2021, d'où les deux None : le panneau correspondant du
    # rapport ne porte que cinq barres quand les deux autres en portent
    # sept, et sa légende commune n'en déclare que six. C'est pour cette
    # raison que le mémoire REPRODUIT ces trois séries en tableau au lieu
    # de reprendre la figure : une légende qui ne compte pas ses séries est
    # un défaut qu'il vaut mieux ne pas importer.
    "capacite_serie_eur": (2.40, 2.68, 1.77, 1.71, 1.65, 2.04, 1.88),
    "franchise_serie_eur": (None, None, 92.0, 218.0, 175.0, 149.9, 89.0),
    "taux_prime_serie": (0.08, 0.10, 0.39, 0.33, 0.29, 0.28, 0.26),
    # Taux de prime annuel moyen par segment, en pourcentage, (2024, 2025).
    #
    # UNE SECONDE IMPRÉCISION DE LA SOURCE, ET ELLE SE VOIT EN RECALCULANT.
    # Le rapport annonce le recul du taux des grandes entreprises à 32 %
    # dans son résumé et sa section 3, puis à 33 % dans sa section 3.1,
    # pour les mêmes niveaux 1,90 % et 1,28 %. Le rapport des deux donne
    # 32,6 %, donc 33 % à l'unité : c'est la seconde valeur qui est juste,
    # et la première arrondit vers le bas. L'écart est immatériel, mais le
    # mémoire cite les NIVEAUX et le recul qu'ils impliquent, jamais un
    # recul transcrit : c'est la seule façon de ne pas propager celui des
    # deux qui est faux. La clé recul_taux_prime_grandes ci-dessus garde la
    # valeur du résumé, pour que la sortie porte les deux et que le
    # désaccord soit visible plutôt que arbitré en silence.
    "taux_prime_segment": {
        "grandes entreprises": (1.90, 1.28),
        "entreprises de taille intermediaire": (1.05, 0.81),
    },
    # Mouvements de conditions propres au bloc intermédiaire. Le taux des
    # petites entreprises monte à CONTRE-COURANT du marché, et le rapport
    # l'attribue à une recomposition du sous-segment vers des profils
    # mieux couverts : ce n'est donc pas un durcissement tarifaire.
    "recul_taux_prime_moyennes": 0.08,
    "hausse_taux_prime_petites": 0.44,
    "recul_franchise_eti": 0.17,
    "recul_franchise_moyennes": 0.41,
    # Conditions du segment mature, exercice 2025.
    "capacite_grandes_2025_eur": 50.0,
    "franchise_grandes_2025_eur": 5.1,
    # Croissance du nombre d'entreprises assurées entre 2024 et 2025.
    "croissance_assures_2025": {
        "grandes entreprises": 0.08,
        "entreprises de taille intermediaire": 0.75,
        "entreprises moyennes": 0.97,
        "petites entreprises": 0.45,
        "micro-entreprises": 0.35,
        "ensemble du marche": 0.49,
    },
    # Les trois blocs : charge indemnisée 2025 en M EUR. La charge des
    # micro-entreprises n'est PAS transcrite, elle est déduite par
    # différence dans le script 63, ce qui en fait le troisième contrôle.
    "blocs_charge_2025_eur": {
        "grandes entreprises": 44.6,
        "bloc intermediaire": 37.3,
    },
    "bloc_intermediaire_charge_2024_eur": 10.5,
    # UNE COQUILLE DE LA SOURCE, TROUVÉE PAR LE CONTRÔLE ET NON REPRISE.
    # La section 7.1 du rapport écrit « 37,3 M€ en 2025 contre 10,5 M€ en
    # 2024 (x2,53) ». Le rapport des deux montants vaut 3,55, et la
    # section 7.6 du MÊME rapport donne bien x3,53 pour ce bloc. Le 2,53
    # de la section 7.1 est donc une coquille sur le chiffre des unités,
    # et c'est le troisième contrôle d'identité de ce bloc qui l'a fait
    # tomber. La valeur retenue est celle de blocs_2025, soit 3,53, et le
    # script 63 imprime la vérification. À signaler comme erratum du
    # rapport publié : la coquille est dans un document co-signé.
    "mult_charge_bloc_intermediaire": 3.53,
    "mult_frequence_blocs_2025": {
        "grandes entreprises": 1.15,
        "bloc intermediaire": 1.30,
        "micro-entreprises": 7.06,
    },
    "sp_2025_segment": {
        "grandes entreprises": 0.22,
        "entreprises de taille intermediaire": 0.42,
        "entreprises moyennes": 0.33,
    },
    # Le sous-segment qui décroche, et celui qui ne décroche pas. Les ETI
    # portent une hausse de fréquence à exposition donnée ; les entreprises
    # moyennes ont une fréquence plate et une charge tirée par la seule
    # exposition. Les deux se ressemblent en charge et diffèrent en nature.
    "eti_2025": {
        "mult_nombre_sinistres": 2.48,
        "mult_charge": 4.22,
        "mult_frequence": 1.42,
        "frequence_2024": 0.087,
        "frequence_2025": 0.124,
    },
    "petites_2025": {"mult_nombre_sinistres": 2.03, "mult_frequence": 1.40},
    "moyennes_2025": {"mult_frequence": 1.01, "mult_exposition": 1.97},
    # Historique du décrochage ETI, 2020 à 2025, et l'indice du taux de
    # prime du même segment en base 100 (2020). Le pic de 2021 dit que le
    # décrochage de 2025 n'est pas un régime inédit : il est CYCLIQUE.
    #
    # PROVENANCE DE CES QUATRE SÉRIES, ET IL FAUT LA DÉCLARER. Le corps du
    # rapport ne donne en clair que quatre de leurs vingt-quatre valeurs
    # (le pic ETI de 261 %, le point bas de 13 %, le 42 % de 2025 et le
    # 22 % des grandes). Les vingt autres sont relevées sur les ÉTIQUETTES
    # DE DONNÉES de la figure du rapport, c'est-à-dire sur des nombres
    # imprimés, et non sur une position de pixel. La distinction n'est pas
    # de confort : lire la hauteur d'une barre est ce que la passation
    # interdit depuis le « pic à trois heures » du script 08h, relever une
    # étiquette imprimée ne l'est pas.
    # Une valeur a d'ailleurs été corrigée à ce titre : le S/P des ETI en
    # 2020 avait d'abord été transcrit à 88 % parce que son étiquette est
    # partiellement recouverte par le marqueur de la courbe d'indice. Le
    # zoom donne 85, et la hauteur de barre le confirme.
    "annees_segment": (2020, 2021, 2022, 2023, 2024, 2025),
    "sp_serie_grandes": (1.90, 0.58, 0.16, 0.09, 0.18, 0.22),
    "sp_serie_eti": (0.85, 2.61, 0.51, 0.21, 0.13, 0.42),
    "sp_serie_moyennes": (0.45, 0.36, 1.00, 0.19, 0.22, 0.33),
    "indice_taux_prime_eti": (100.0, 156.0, 240.0, 257.0, 231.0, 178.0),
    # Répartition du montant indemnisé par taille de sinistre, en M EUR,
    # 2019 à 2025. LA LIGNE XXL EST CELLE QUI COMMANDE LE CHOIX DE DONNÉES
    # DU MÉMOIRE : deux exercices à zéro, un pic à 135, et 19 en 2025 pour
    # un seul sinistre. La queue française est peuplée par accident.
    "taille_sinistre_eur": {
        "XS/S, 0 a 0,3 M EUR": (5.0, 8.0, 10.0, 2.0, 4.0, 7.0, 9.0),
        "M/L, 0,3 a 3 M EUR": (28.0, 22.0, 24.0, 35.0, 22.0, 15.0, 26.0),
        "XL, 3 a 10 M EUR": (40.0, 55.0, 37.0, 20.0, 11.0, 13.0, 29.0),
        "XXL, 10 a 40 M EUR": (0.0, 135.0, 92.0, 15.0, 0.0, 20.0, 19.0),
    },
    # Ce que la queue produit ailleurs, sur la même période. Ces montants
    # sont ce qui interdit de lire la sévérité moyenne contenue de 2025
    # comme une protection structurelle du marché français.
    "comparaisons_etrangeres": {
        "Marks & Spencer, effet sur le resultat operationnel 2025-2026":
            (300.0, "M GBP"),
        "Jaguar Land Rover, garantie d'Etat aux creanciers commerciaux":
            (1.5, "Md GBP"),
        "Rheinmetall, cout declare en 2024 d'une attaque de 2023":
            (10.0, "M EUR"),
        "Association of British Insurers, indemnites cyber versees":
            (200.0, "M GBP"),
    },
    "recul_sinistres_cyber_europe_2024": 0.20,
    # Propension à s'assurer et sinistralité déclarée, baromètre CESIN.
    # Couples (vague precedente, vague 2026) dans cet ordre.
    "cesin_2026": {
        "part des membres assures": (0.72, 0.71),
        "part ayant subi une attaque significative": (0.45, 0.40),
    },
    "cesin_couverture_grandes": 0.78,
    # Ratio de frais et de commissionnement usuel sur la branche cyber.
    # C'est lui qui explique qu'un S/P de 27 % laisse le ratio combiné
    # sous le seuil d'équilibre, et donc que la détente commerciale
    # continue malgré un signal technique qui se dégrade.
    "ratio_frais_bas": 0.30,
    "ratio_frais_haut": 0.35,
    "note": (
        "Citation externe non recalculable, même statut que HACKMAGEDDON. Ne porte "
        "aucun niveau de capital du mémoire et n'entre dans aucune calibration. La "
        "charge est INDEMNISÉE, donc nette de franchise et plafonnée par la "
        "capacité : elle ne se compare pas à une sévérité brute d'entité."
    ),
}

# ---------------------------------------------------------------------------
# DEUX SOURCES DE CONTEXTE 2026, AU MÊME STATUT QUE LUCY_2026 : CITATION
# EXTERNE NON RECALCULABLE.
#
# CE QUE CE STATUT VEUT DIRE, ET IL EST STRICT. Les valeurs ci-dessous sont
# TRANSCRITES depuis les rapports publiés. Le dépôt ne détient ni les réponses
# individuelles de l'enquête, ni les scores élémentaires de la cartographie :
# il ne peut donc PAS les recalculer, seulement vérifier la fidélité de la
# recopie. Aucune n'entre dans une calibration du modèle et aucune ne porte un
# montant de capital. Elles servent à situer le risque, pas à le mesurer.
#
# POURQUOI ELLES SONT ICI ET PAS EN DUR DANS UN CHAPITRE. Parce qu'une valeur
# écrite directement dans le LaTeX n'est vérifiable par personne, et que c'est
# exactement la classe de défaut qui a produit les queues Bâle et le Hill à
# 1,42. Le script 98 les relit, en contrôle les identités et trace les figures.
# ---------------------------------------------------------------------------

CESIN_2026 = {
    "source": ("OpinionWay pour le CESIN, 11e Baromètre de la Cybersécurité "
               "des Entreprises, janvier 2026, portant sur l'exercice 2025"),
    # Base de l'enquête sur les questions de vecteurs et d'impacts.
    "base_attaquees": 159,
    "part_entreprises_attaquees": 0.40,
    # Nombre moyen de vecteurs par entreprise attaquée. LA BAISSE COMPTE
    # AUTANT QUE LE NIVEAU : 4,0 à la vague précédente, 3,3 ici.
    "vecteurs_moyens": 3.3,
    "vecteurs_moyens_vague_precedente": 4.0,
    "vecteurs_moyens_grandes": 3.9,
    # Q5A, vecteurs ayant permis aux cyberattaques de démarrer ou de se
    # déployer. Plusieurs réponses possibles : la somme dépasse 1 par
    # construction, et ce n'est pas une erreur de recopie.
    "vecteurs": {
        "hameçonnage": 0.55,
        "exploitation d'une faille": 0.41,
        "attaque indirecte par un tiers": 0.35,
        "fuite par erreur humaine ou de configuration": 0.27,
        "fraude au président": 0.26,
        "déni de service par botnet": 0.21,
        "noms de domaine illégitimes": 0.17,
        "activation d'un composant malveillant": 0.16,
        "exfiltration par un accès légitime": 0.14,
        "perte ou vol de matériel": 0.13,
        "autre ingénierie sociale": 0.12,
        "système ou application corrompu": 0.12,
        "outils non approuvés": 0.12,
        "contournement du double facteur": 0.11,
        "hypertrucage": 0.05,
        "matériel malveillant introduit": 0.04,
        "extraction de données par IA": 0.03,
        "faille d'un système industriel": 0.03,
        "intrusion physique": 0.01,
    },
    # LE SEUL DÉCOUPAGE PAR TAILLE QUI SERVE LA THÈSE : les vecteurs dont la
    # part MONTE chez les grandes entreprises sont ceux qui passent par un
    # tiers ou par un accès légitime, c'est-à-dire les canaux P4 et P1.
    "vecteurs_grandes_entreprises": {
        "attaque indirecte par un tiers": 0.43,
        "fuite par erreur humaine ou de configuration": 0.36,
        "activation d'un composant malveillant": 0.22,
        "exfiltration par un accès légitime": 0.21,
        "hypertrucage": 0.09,
    },
    # Q7, impact des cyberattaques sur le business.
    "part_avec_impact_business": 0.81,
    "part_sans_impact_business": 0.19,
    "impacts": {
        "perturbation de la production": 0.28,
        "perte d'image": 0.26,
        "compromission de savoir-faire": 0.18,
        "perte de chiffre d'affaires": 0.18,
        "indisponibilité du site": 0.17,
        "arrêt de la production": 0.14,
        "perte financière directe": 0.14,
        "déconnexion par les tiers": 0.12,
        "retard de livraison": 0.08,
        "atteinte aux biens et personnes": 0.08,
        "sanction d'une autorité": 0.03,
        "contrefaçon": 0.01,
    },
    "note": (
        "Citation externe non recalculable. Enquête déclarative : elle mesure "
        "ce que les organisations CONSTATENT, non ce qui survient. La baisse "
        "du nombre moyen de vecteurs et de la part d'entreprises attaquées ne "
        "peut donc pas s'interpréter comme une baisse du risque sans précaution."
    ),
}

CARTO_FA_2026 = {
    "source": ("France Assureurs, Cartographie prospective des risques de la "
               "profession de l'assurance et de la réassurance, édition 2026"),
    # Score moyen du risque de cyberattaques, série rétrospective. Valeurs
    # relevées sur la figure publiée : elles portent l'imprécision d'une
    # lecture graphique et sont arrondies au centième, ce qui suffit à la
    # seule chose qu'on leur demande, montrer une stabilisation au sommet.
    "annees_serie": [2020, 2021, 2022, 2023, 2024, 2025, 2026],
    "score_cyber_serie": [4.24, 4.17, 4.48, 4.03, 4.12, 4.17, 4.10],
    "lecture_graphique": True,
    # Classement 2025 et 2026, scores publiés en clair.
    "rang_2026": [("Cyberattaques", 4.1), ("Environnement économique", 4.0),
                  ("Dérèglement climatique", 3.9), ("Environnement politique", 3.9),
                  ("Événement naturel exceptionnel", 3.3)],
    "rang_2025": [("Cyberattaques", 4.1), ("Dérèglement climatique", 4.1),
                  ("Environnement économique", 3.8), ("Environnement politique", 3.6),
                  ("Événement naturel exceptionnel", 3.3)],
    "annees_consecutives_premier_rang": 8,
    # Déplacements 2026 contre 2025, en points de score. LE COUPLE QUI PORTE
    # LA LECTURE DU MÉMOIRE : le cyber recule en fréquence et monte en
    # sévérité, et le risque qui monte le plus en sévérité est celui de la
    # qualité des données et de la conformité des processus informatiques.
    "deplacements": {
        "Cyberattaques": (-0.45, 0.25),
        "Qualité des données et conformité des processus IT": (-0.20, 0.30),
        "Intelligence artificielle": (0.05, 0.10),
        "Risque de non-conformité et de sanctions": (0.10, -0.15),
        "Environnement politique": (0.30, 0.32),
        "Dérèglement climatique": (-0.10, -0.22),
    },
    "part_cadran_sud_ouest": 0.50,
    "part_cadran_nord_est": 0.25,
    "part_cadran_nord_ouest": 0.13,
    "part_cadran_sud_est": 0.13,
    "note": (
        "Citation externe non recalculable. Les scores de la série "
        "rétrospective sont relevés sur une figure publiée, non sur une table "
        "de valeurs : ils ne supportent pas une lecture au centième et ne "
        "servent qu'à établir une tendance."
    ),
}

# ---------------------------------------------------------------------------
# PREMIER BILAN DES NOTIFICATIONS D'INCIDENTS MAJEURS DORA À L'AMF, EXERCICE
# 2025. Même statut que CESIN_2026 et CARTO_FA_2026 : CITATION EXTERNE NON
# RECALCULABLE. Transcrit le 28 septembre 2026 depuis le rapport lui-même
# (section 1.1 « Les chiffres clés » et section 2.3), et non depuis la
# synthèse Nexialog qui l'a signalé : les comptes y sont identiques, mais le
# rapport porte en plus la décomposition des 87 % entre cyberattaque et
# défaillance accidentelle.
#
# DEUX RÉSERVES À NE PAS PERDRE.
#  - Le périmètre est celui des SOCIÉTÉS DE GESTION, pas des assureurs.
#  - Un incident est compté PAR SGP DÉCLARANTE. Le rapport écrit que la
#    compromission d'un prestataire touche « plusieurs SGP par ruissellement » :
#    les 27 incidents d'origine tierce ne sont donc pas 27 événements distincts.
#    C'est de l'accumulation (pilier P4), vue du côté des victimes.
# La part d'origine tierce mesure l'ORIGINE d'un incident, jamais une
# probabilité de transfert entre domaines de contrôle : elle ne calibre rien.
# Lu et contrôlé par le script 103.
# ---------------------------------------------------------------------------

AMF_DORA_2025 = {
    "source": ("AMF, Rapport sur les incidents majeurs DORA déclarés à l'AMF "
               "en 2025, 24 septembre 2026"),
    "notifications_initiales_sgp": 47,
    "incidents_majeurs_sgp": 31,
    # Origine : prestataire tiers, qu'il s'agisse d'une indisponibilité de
    # ses services ou d'une cyberattaque subie par lui.
    "incidents_origine_tiers": 27,
    "part_origine_tiers_publiee": 0.87,
    # Section 2.3 : les 87 % se répartissent en 68 % de cyberattaques et
    # 19 % de défaillances accidentelles du prestataire.
    "part_tiers_cyber_publiee": 0.68,
    "part_tiers_accidentel_publiee": 0.19,
    # Nature, partition des 31 incidents.
    "incidents_cyber": 22,
    "dont_exfiltration": 15,
    "dont_usurpation": 7,
    "incidents_operationnels": 9,
    "dont_indisponibilite_si": 5,
    "dont_defaillance_sgp": 4,
    "parts_publiees": {"cyber": 0.71, "exfiltration": 0.48, "usurpation": 0.23,
                       "operationnels": 0.29},
    # Hors SGP : PSCA et entreprises de marché. Total AMF = 33.
    "incidents_majeurs_hors_sgp": 2,
    "incidents_majeurs_total_publie": 33,
    # Questionnaire d'auto-évaluation de novembre 2025.
    "part_sgp_dispositif_non_conforme": 0.23,
}

# ---------------------------------------------------------------------------
# MULTIPLICATEURS DORA — sources de calibration (résumé ; détail dans negbin.py)
# ---------------------------------------------------------------------------

MULTIPLICATEURS_SOURCES = {
    "phishing_social_eng": "Ponemon/IBM 2025 — formation réduit le risque jusqu'à 86%",
    "exploit_vuln":        "ENISA Threat Landscape 2025 — conversion 70% vs 27%, ratio ×2.6",
    "supply_chain_tiers":  "IBM/Ponemon + SecurityScorecard — surcoût +11.8%/+8.3%, part 15%→36%",
    "identifiants":        "Microsoft Research arXiv:2305.00945 — MFA réduit 99.22% (tempéré)",
    "multiplicateur_global_S2": 2.49,   # centre ; IC90% λ = [742.7, 947.4]
}

# ---------------------------------------------------------------------------
# VARIABLE LATENTE — paramètres ancrés (Option B)
# ---------------------------------------------------------------------------

LATENT_ANCHORED = {
    "gamma": 0.68,               # proxy concentration cloud (Synergy/Canalys 2025)
    "p_ref_global": 0.50,        # ~50% non pleinement conformes fin 2025 (industry surveys)
    "p_ref_by_pillar": {         # 1 - taux de pleine conformité (Deloitte 2025, ESA dry-run)
        "incident_mgmt": 0.52,
        "ict_risk_mgmt": 0.75,
        "resilience_test": 0.92,
        "third_party": 0.92,
        "roi_data_quality": 0.935,
    },
    "note": "Paramètres illustratifs de la PHASE TRANSITOIRE de DORA (2025), "
            "non représentatifs d'un régime stationnaire. β non estimés (illustratifs).",
}

# ---------------------------------------------------------------------------
# LIMITES ÉPISTÉMIQUES (pour la rédaction)
# ---------------------------------------------------------------------------

LIMITES = {
    "absence_donnees_dora": (
        "DORA est en vigueur depuis janvier 2025. Aucun historique de pertes "
        "conditionnel au niveau de conformité n'existe à ce stade."
    ),
    "biais_prc": (
        "La PRC est une base US de data breaches déclarées. Elle sous-représente "
        "les incidents de disponibilité/continuité et les petites pertes européennes."
    ),
    "biais_oprisk": (
        "OpRisk surreprésente les grandes entités financières et les pertes "
        "judiciarisées (46% US). Les pertes < seuil de notification sont absentes."
    ),
    "biais_hackmageddon": (
        "Base déclarative à fort biais de médiatisation. Le niveau absolu de "
        "fréquence n'est pas calibrant ; seules les proportions relatives le sont. "
        "Taxonomie des types d'attaque instable dans le temps (piège ransomware)."
    ),
    "quantile_995": (
        "Présenter un SCR_DORA ponctuel au 99.5% serait une surinterprétation. "
        "L'incertitude de paramètre seule génère un IC d'un facteur 2.5 sur la VaR."
    ),
    "sanction_marginale": (
        "La brique sanction pèse ~0.3% du capital : le plafond réglementaire "
        "(2-20M€) est négligeable face à la queue lourde du risque opérationnel."
    ),
    "aggravation_stress_non_source": (
        "Le chargement systémique sur l'aggravation (couplage copule) est une "
        "hypothèse de modélisation bornée à 15%, non issue d'une source externe."
    ),
}