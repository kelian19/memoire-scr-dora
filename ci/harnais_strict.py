#!/usr/bin/env python3
"""Controle strict du harnais des chiffres, pour l'integration continue.

Rejoue verif_chiffres.py sur chaque chapitre du memoire et echoue au moindre ecart :

  - un nombre verifiable qui n'est pas confirme par la sortie versionnee de son script ;
  - un nombre publie hors controle SANS declaration (ni script cite, ni motif ecrit).

C'est le critere de fin de tache du projet, « hors controle non declare = 0 » et aucun non
confirme, rendu executable. verif_tous_chapitres.sh affiche le meme tableau mais sort
toujours avec le code 0 : il informe, celui-ci arrete.

Bibliotheque standard seulement : le harnais ne lit que les sorties versionnees de
sorties_verif/, jamais la donnee brute, qui est sous licence et absente du depot.

    python ci/harnais_strict.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
MEMOIRE = RACINE / "exploratory" / "memoire_cascade"
SORTIES = RACINE / "sorties_verif"

RE_COUV = re.compile(r"COUVERTURE : (\d+) nombres sous controle sur (\d+) publies")
RE_NATURE = re.compile(r"(\d+) nombres hors script PAR NATURE")
RE_VERIF = re.compile(r"(\d+) nombres v\S+ables, (\d+) confirm\S+, (\d+) non confirm")


def analyser(chapitre: Path) -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(
        [sys.executable, str(MEMOIRE / "verif_chiffres.py"), str(SORTIES), str(chapitre)],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    sortie = r.stdout + r.stderr
    if r.returncode != 0:
        raise SystemExit(f"verif_chiffres.py a echoue sur {chapitre.name} :\n{sortie}")
    couv = RE_COUV.search(sortie)
    sous_controle, publies = (int(couv.group(1)), int(couv.group(2))) if couv else (0, 0)
    declares = sum(int(x) for x in RE_NATURE.findall(sortie))
    verif = RE_VERIF.search(sortie)
    verifiables, confirmes, non_confirmes = (
        (int(verif.group(1)), int(verif.group(2)), int(verif.group(3))) if verif else (0, 0, 0)
    )
    return {
        "chapitre": chapitre.stem,
        "verifiables": verifiables,
        "confirmes": confirmes,
        "non_confirmes": non_confirmes,
        "hors": max(publies - sous_controle - declares, 0),
    }


def main() -> int:
    chapitres = sorted((MEMOIRE / "chapitres").glob("*.tex"))
    if not chapitres:
        print("aucun chapitre trouve", file=sys.stderr)
        return 1
    lignes = [analyser(c) for c in chapitres]

    print(f"{'chapitre':36s}{'verif.':>8s}{'confirmes':>11s}{'non conf.':>11s}{'hors':>7s}")
    print("-" * 73)
    for l in lignes:
        marque = "  <--" if l["non_confirmes"] or l["hors"] else ""
        print(f"{l['chapitre']:36s}{l['verifiables']:8d}{l['confirmes']:11d}"
              f"{l['non_confirmes']:11d}{l['hors']:7d}{marque}")
    tot = {k: sum(l[k] for l in lignes) for k in ("verifiables", "confirmes", "non_confirmes", "hors")}
    print("-" * 73)
    print(f"{'TOTAL':36s}{tot['verifiables']:8d}{tot['confirmes']:11d}"
          f"{tot['non_confirmes']:11d}{tot['hors']:7d}")

    if tot["non_confirmes"] or tot["hors"]:
        print(f"\nECHEC : {tot['non_confirmes']} nombre(s) non confirme(s), "
              f"{tot['hors']} nombre(s) hors controle non declare(s).")
        return 1
    print(f"\nOK : {tot['confirmes']} nombres confirmes sur {tot['verifiables']}, "
          "aucun hors controle non declare.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
