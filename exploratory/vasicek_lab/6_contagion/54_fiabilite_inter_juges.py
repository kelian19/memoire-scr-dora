#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
54 : fiabilite inter-juges du codage des post-mortems (kappa de Cohen).

Le script 53 code la direction de W a partir de post-mortems officiels, mais avec UN SEUL
codeur : le risque de biais de confirmation est la limite principale du resultat. Le remede
standard en analyse de contenu est un SECOND CODEUR EN AVEUGLE, puis une mesure d'accord.

DISPOSITIF.
  - Unite de codage : le couple (incident, paire de piliers). 7 incidents x 10 paires non
    ordonnees = 70 decisions, chacune dans 4 modalites :
        "k>j"  la defaillance de k precede et conditionne celle de j
        "j>k"  l'inverse
        "co"   les deux piliers sont touches mais AUCUN ordre n'est etabli par le rapport
        "abs"  la paire n'est pas documentee dans ce rapport
  - Le second codeur recoit le kit `kit_codage_aveugle.pdf` : le protocole, les definitions de
    piliers, et pour chaque incident un resume FACTUEL des defaillances documentees, SANS
    aucune fleche ni attribution causale de ma part (c'est cela, l'aveugle).
  - Il rend un CSV au format `codage_<initiales>.csv` : incident,paire,code

MESURES. Accord brut, kappa de Cohen (accord corrige du hasard), et accord restreint aux
decisions DIRECTIONNELLES (k>j / j>k), qui est ce qui compte pour W. Reperes usuels de kappa :
  < 0,40 faible | 0,40-0,60 modere | 0,60-0,80 substantiel | > 0,80 excellent.

CE SCRIPT NE FABRIQUE AUCUNE DONNEE. Sans second codage sur le disque, il n'invente rien : il
imprime le mode d'emploi et s'arrete. Les chiffres d'accord n'apparaitront que lorsqu'un vrai
second codage sera fourni.

Sortie : diagnostics (+ figure Z18 si un second codage est present).
"""

import os
import sys
import csv
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

WID = 84
PIL = [1, 2, 3, 4, 5]
PAIRS = [f"P{a}-P{b}" for a, b in combinations(PIL, 2)]
CODES = ["k>j", "j>k", "co", "abs"]
CODAGE_DIR = os.path.join(HERE, "notes", "codage")
REF_CODER = "KK"                      # codage de reference (Kelian/moi, script 53)

INCIDENTS = [
    "Microsoft Exchange Online 2023",
    "TSB Bank 2018",
    "Equifax 2017",
    "Knight Capital 2012",
    "Capital One 2019",
    "ION Cleared Derivatives 2023",
    "MOVEit 2023",
    # EXTENSION AU CORPUS DE DIX. Le memoire ne s'appuie plus sur les sept initiaux mais sur
    # le corpus etendu du script 59 (z = +5,12), et les scripts 59 et 64 raisonnent sur dix
    # rapports. Un second codage qui n'en couvrirait que sept mesurerait l'accord sur un
    # corpus que le memoire n'utilise plus.
    "CrowdStrike Falcon 2024",
    "Log4Shell 2021-2022",
    "Raphaels Bank 2015",
]

# L'ORDRE CI-DESSUS EST CELUI DES RECITS DU KIT, ET CE N'EST PAS UN DETAIL. Le kit anonymise
# les incidents ("Fournisseur de messagerie d'entreprise, 2023") pour que le codeur ne
# reconnaisse pas le cas et ne se souvienne pas de la cause racine publiee. Le gabarit qu'il
# remplit doit donc porter les MEMES etiquettes anonymes, sans quoi l'aveuglement est rompu
# par le fichier de reponse lui-meme. Une premiere version du gabarit nommait les entreprises.
ANONYME = {nom: f"Recit {i+1}" for i, nom in enumerate(INCIDENTS)}
REEL = {v: k for k, v in ANONYME.items()}

# --- codage de reference, DERIVE du script 53 (transitions codees -> grille 70 cellules) -----
# Toute paire non listee dans le script 53 est "abs" par defaut ; les paires ou les deux piliers
# apparaissent dans l'incident mais sans ordre etabli seraient "co" (aucune dans le corpus 53).
REF_EDGES = {
    "Microsoft Exchange Online 2023": [(1, 3), (3, 2), (1, 4)],
    "TSB Bank 2018": [(1, 4), (1, 3), (3, 2), (4, 2)],
    "Equifax 2017": [(1, 5), (5, 3), (3, 2), (1, 3)],
    "Knight Capital 2012": [(3, 2), (1, 3)],
    "Capital One 2019": [(1, 4), (1, 3)],
    "ION Cleared Derivatives 2023": [(4, 2)],
    "MOVEit 2023": [(4, 2)],
    # extension : transitions du corpus etendu (script 59), reprises a l'identique
    "CrowdStrike Falcon 2024": [(3, 2)],
    "Log4Shell 2021-2022": [(1, 4), (4, 2)],
    "Raphaels Bank 2015": [(1, 4), (4, 2)],
}


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def pair_key(a, b):
    lo, hi = min(a, b), max(a, b)
    return f"P{lo}-P{hi}"


def ref_grid():
    """Grille de reference : dict (incident, paire) -> code."""
    g = {(inc, p): "abs" for inc in INCIDENTS for p in PAIRS}
    for inc, edges in REF_EDGES.items():
        for (k, j) in edges:
            key = pair_key(k, j)
            lo = int(key.split("-")[0][1:])
            g[(inc, key)] = "k>j" if k == lo else "j>k"
    return g


# --- traduction des reponses EN LANGAGE NATUREL du kit vers les codes internes ---------------
# Le kit remis au second codeur ne contient AUCUN code : on y ecrit le NOM du domaine qui a
# lache en premier, ou une des trois formules libres. Cette table fait la conversion, de sorte
# que le codeur n'ait jamais a manipuler la convention "k>j / j>k".
NOM_PILIER = {1: "gouvernance", 2: "incidents", 3: "tests", 4: "prestataires", 5: "partage"}
SYNONYMES = {
    "gouvernance": 1, "gouv": 1, "gouv.": 1,
    "incidents": 2, "incident": 2, "incid": 2, "incid.": 2,
    "tests": 3, "test": 3,
    "prestataires": 4, "prestataire": 4, "prest": 4, "prest.": 4, "tiers": 4,
    "partage": 5, "partage d'infos": 5, "partage d'info": 5, "infos": 5, "information": 5,
}
LIBRES = {
    "en meme temps": "co", "en même temps": "co", "meme temps": "co", "même temps": "co",
    "non concerne": "abs", "non concerné": "abs", "sans objet": "abs",
    "je ne sais pas": "co", "ne sais pas": "co", "?": "co",   # le doute est traite comme "co"
}


def _normalise(s):
    return " ".join(s.strip().lower().replace("’", "'").split())


def traduire(reponse, paire):
    """Convertit une reponse en langage naturel en code interne, pour la paire donnee.

    `paire` : "P1-P3". Le code "k>j" signifie que le pilier de PLUS PETIT numero precede.
    Le doute ("je ne sais pas") est assimile a "co" : aucun ordre etabli. C'est le choix
    CONSERVATEUR, celui qui ne fabrique pas de direction.
    """
    r = _normalise(reponse)
    if not r:
        return None
    if r in LIBRES:
        return LIBRES[r]
    if r in CODES:                      # tolere aussi les codes bruts (retro-compatibilite)
        return r
    if r in SYNONYMES:
        lo, hi = (int(x[1:]) for x in paire.split("-"))
        p = SYNONYMES[r]
        if p == lo:
            return "k>j"
        if p == hi:
            return "j>k"
        raise ValueError(f"'{reponse}' ne fait pas partie de la paire {paire}")
    raise ValueError(f"reponse non comprise : '{reponse}' (paire {paire})")


def load_coding(path):
    """Lit un CSV incident,paire,code. Accepte les codes internes OU le langage naturel."""
    g = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            inc = row["incident"].strip()
            # le gabarit porte des etiquettes ANONYMES ("Recit 3") ; le codage de reference
            # porte les noms reels. On ramene tout aux noms reels pour pouvoir apparier.
            inc = REEL.get(inc, inc)
            pr = row["paire"].strip()
            cd = traduire(row["code"], pr)
            if cd is None:
                continue                # case laissee vide : on ignore, on n'invente pas
            g[(inc, pr)] = cd
    return g


def cohen_kappa(a, b, labels):
    """Kappa de Cohen sur deux listes de labels appariees."""
    n = len(a)
    idx = {l: i for i, l in enumerate(labels)}
    C = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        C[idx[x], idx[y]] += 1
    po = np.trace(C) / n
    pe = float((C.sum(axis=0) / n) @ (C.sum(axis=1) / n))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


# =====================================================================================
titre("1. Dispositif de fiabilite inter-juges")
# =====================================================================================
ref = ref_grid()
n_dir = sum(1 for v in ref.values() if v in ("k>j", "j>k"))
print(f"  Unites de codage : {len(INCIDENTS)} incidents x {len(PAIRS)} paires = {len(ref)} cellules.")
print(f"  Codage de reference ({REF_CODER}, script 53) : {n_dir} cellules directionnelles, "
      f"{len(ref) - n_dir} 'abs'.")
print(f"  Modalites : " + " | ".join(CODES))
print(f"  Dossier attendu pour les codages : {os.path.relpath(CODAGE_DIR, HERE)}")

os.makedirs(CODAGE_DIR, exist_ok=True)

# ecrit le CSV de reference (trace du codage 53) et un GABARIT VIDE pour le second codeur
ref_path = os.path.join(CODAGE_DIR, f"codage_{REF_CODER}.csv")
with open(ref_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["incident", "paire", "code"])
    for inc in INCIDENTS:
        for p in PAIRS:
            w.writerow([inc, p, ref[(inc, p)]])
tpl_path = os.path.join(CODAGE_DIR, "gabarit_second_codeur.csv")
with open(tpl_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["incident", "paire", "domaines_concernes", "code"])
    for inc in INCIDENTS:
        for p in PAIRS:
            lo, hi = (int(x[1:]) for x in p.split("-"))
            # ETIQUETTE ANONYME, jamais le nom de l'entreprise : voir ANONYME plus haut.
            w.writerow([ANONYME[inc], p, f"{NOM_PILIER[lo]} / {NOM_PILIER[hi]}", ""])
print(f"  Ecrit : codage_{REF_CODER}.csv (reference) et gabarit_second_codeur.csv (a remplir).")
print(f"  Le gabarit porte les etiquettes ANONYMES ({ANONYME[INCIDENTS[0]]} a "
      f"{ANONYME[INCIDENTS[-1]]}), correspondant aux recits")
print("  numerotes du kit. Le codeur ne voit donc aucun nom d'entreprise, ni dans le kit ni")
print("  dans le fichier qu'il remplit : c'est la condition de l'aveuglement.")

# =====================================================================================
titre("2. Second codage : present ou non ?")
# =====================================================================================
others = [f for f in os.listdir(CODAGE_DIR)
          if f.startswith("codage_") and f.endswith(".csv") and f != f"codage_{REF_CODER}.csv"]

if not others:
    print("  AUCUN second codage trouve. Ce script NE FABRIQUE PAS de donnees : il s'arrete ici.")
    print("\n  Mode d'emploi :")
    print("    1. Transmettre `notes/kit_codage_aveugle.pdf` au second codeur (Hugo, ou un")
    print("       praticien hors du projet). Le kit ne contient AUCUNE de mes fleches : seulement")
    print("       les faits documentes par les rapports officiels, et la grille a remplir.")
    print("       Il n'y manipule AUCUN code : il ecrit le NOM du domaine qui a lache en premier")
    print("       (Gouvernance, Incidents, Tests, Prestataires, Partage), ou bien 'en meme temps',")
    print("       'non concerne', 'je ne sais pas'. La conversion est faite par ce script.")
    print("    2. Il remplit `gabarit_second_codeur.csv` et le renomme `codage_<INITIALES>.csv`")
    print(f"       dans {os.path.relpath(CODAGE_DIR, HERE)}.")
    print("    3. Relancer ce script : il calculera l'accord brut, le kappa de Cohen, et")
    print("       l'accord sur les seules decisions directionnelles.")
    print("\n  Interpretation prevue : un kappa >= 0,60 (substantiel) leverait la principale")
    print("  reserve du script 53 (codeur unique) ; un kappa faible signifierait que la")
    print("  direction que je lis dans ces rapports n'est PAS intersubjective, ce qui serait")
    print("  un resultat en soi (et devrait alors etre publie tel quel).")
    sys.exit(0)

# =====================================================================================
titre("3. Accord entre codeurs")
# =====================================================================================
for fn in others:
    coder = fn[len("codage_"):-len(".csv")]
    g2 = load_coding(os.path.join(CODAGE_DIR, fn))
    common = [k for k in ref if k in g2]
    if not common:
        print(f"  {coder} : aucune cellule commune, ignore.")
        continue
    a = [ref[k] for k in common]
    b = [g2[k] for k in common]
    kappa, po = cohen_kappa(a, b, CODES)
    # accord restreint aux cellules ou AU MOINS UN codeur voit une direction
    dir_cells = [k for k in common
                 if ref[k] in ("k>j", "j>k") or g2[k] in ("k>j", "j>k")]
    if dir_cells:
        ad = [ref[k] for k in dir_cells]
        bd = [g2[k] for k in dir_cells]
        acc_dir = float(np.mean([x == y for x, y in zip(ad, bd)]))
        # desaccords de SENS (les deux voient une direction, mais opposee)
        flip = sum(1 for k in dir_cells
                   if ref[k] in ("k>j", "j>k") and g2[k] in ("k>j", "j>k") and ref[k] != g2[k])
    else:
        acc_dir, flip = float("nan"), 0
    lect = ("faible" if kappa < 0.40 else "modere" if kappa < 0.60
            else "substantiel" if kappa < 0.80 else "excellent")
    print(f"  {REF_CODER} vs {coder} : {len(common)} cellules communes")
    print(f"    accord brut          = {100*po:.0f} %")
    print(f"    kappa de Cohen       = {kappa:.2f}  ({lect})")
    print(f"    accord directionnel  = {100*acc_dir:.0f} % sur {len(dir_cells)} cellules")
    print(f"    inversions de sens   = {flip}  (les deux voient une direction, mais opposee)")
    if flip == 0:
        print("    -> aucune inversion : les deux codeurs lisent la MEME orientation.")
