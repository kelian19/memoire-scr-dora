#!/usr/bin/env python3
"""Rejoue les scripts de severite de la version Institut a 150 pages sur l'echantillon 2000-2026.

Usage : python outils/ci/lancer_lot_institut.py [numeros...]   (sans argument : les seize scripts)

Pour chaque script, sous OPRISK_ANNEE_MIN=2000 : la sortie est ecrite dans
resultats/sorties_verif_inst/NN.txt, les figures modifiees sont copiees dans
recherche/vasicek_lab/figures_inst/ puis restaurees dans figures/ depuis Git, et une ligne du
MANIFESTE.md (code de retour, erreur, taille, empreinte) est ecrite. Les lignes des scripts non
rejoues sont conservees. Le harnais (harnais_strict.py) refuse un lot que le manifeste n'atteste pas.

Garde-fou : le lanceur s'arrete si figures/ porte deja une modification non committee, car il
restaure ce dossier par Git et detruirait un travail en cours.
"""
import datetime
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
LAB = RACINE / "recherche" / "vasicek_lab"
SORTIE = RACINE / "resultats" / "sorties_verif_inst"
FIGS = LAB / "figures"
FIGS_INST = LAB / "figures_inst"
TOUS = ["63", "47", "72", "14", "16", "46", "48", "51", "71", "89", "90", "91", "102", "104",
        "106", "93"]


def git(*args):
    return subprocess.run(["git", *args], cwd=RACINE, capture_output=True, text=True,
                          encoding="utf-8").stdout


def figures_modifiees():
    return [l[3:].strip().strip('"') for l in git("status", "--porcelain", "--", str(FIGS)).splitlines()
            if l[:2].strip() == "M"]


def enregistrer_manifeste(mf, lignes, commit, numeros, journal_dir):
    """Actualise les exécutions sans effacer les notes de périmètre existantes."""
    date = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    ancien = mf.read_text(encoding="utf-8") if mf.exists() else ""
    entete = ancien.split("| script |", 1)[0] if ancien else (
        "# Manifeste du lot 2000-2026\n\n"
        "Variable : `OPRISK_ANNEE_MIN=2000`. Interpreteur : `python -X utf8 -I`.\n\n"
    )
    entete = re.sub(
        r"Dernier lancement le .*?modifications non committees du depot a cet instant\.",
        f"Dernier lancement le {date}, code du commit `{commit}` plus les\n"
        "modifications non committees du depot a cet instant.",
        entete, count=1, flags=re.S,
    )
    entete = entete.rstrip() + (
        f"\n\nExécution {date} : scripts {', '.join(numeros)}. "
        f"Journaux détaillés : `{journal_dir}`.\n\n"
    )
    table = "| script | fichier | code | Traceback | duree | octets | md5 |\n|---|---|---|---|---|---|---|\n"
    mf.write_text(entete + table + "\n".join(lignes[k] for k in sorted(lignes, key=int)) + "\n",
                  encoding="utf-8")


def main():
    numeros = sys.argv[1:] or TOUS
    if figures_modifiees():
        sys.exit("figures/ porte des modifications non committees : les committer ou les annuler "
                 "avant de lancer le lot.")
    SORTIE.mkdir(parents=True, exist_ok=True)
    FIGS_INST.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPRISK_ANNEE_MIN="2000", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    mf = SORTIE / "MANIFESTE.md"
    lignes = {}
    if mf.exists():
        for l in mf.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\| (\d+) \|", l)
            if m:
                lignes[m.group(1)] = l
    commit = git("rev-parse", "--short", "HEAD").strip()
    maintenant = datetime.datetime.now().astimezone()
    journal_dir = RACINE / "docs/collaboration/local/journaux_reexecution" / maintenant.strftime("%Y-%m-%d_%H%M%S_%f")
    journal_dir.mkdir(parents=True, exist_ok=False)
    numeros_faits = []
    for n in numeros:
        candidats = sorted(LAB.rglob(f"{n}_*.py"))
        if len(candidats) != 1:
            sys.exit(f"Script {n} ambigu ou absent : {len(candidats)} candidats")
        script = candidats[0]
        commande = [sys.executable, "-X", "utf8", "-I", str(script)]
        debut = datetime.datetime.now().astimezone().isoformat()
        t0 = time.time()
        stderr_path = journal_dir / f"{n}.stderr.txt"
        with open(SORTIE / f"{n}.txt", "wb") as out, open(stderr_path, "wb") as err:
            r = subprocess.run(commande, cwd=RACINE, env=env, stdout=out, stderr=err)
        fin = datetime.datetime.now().astimezone().isoformat()
        erreur = "Traceback" in stderr_path.read_text(encoding="utf-8", errors="replace")
        for f in figures_modifiees():
            src = RACINE / f
            shutil.copy2(src, FIGS_INST / src.name)
            subprocess.run(["git", "checkout", "--", f], cwd=RACINE)
        octets = (SORTIE / f"{n}.txt").read_bytes()
        shutil.copy2(SORTIE / f"{n}.txt", journal_dir / f"{n}.sortie.txt")
        versions = {}
        for paquet in ("numpy", "scipy", "scikit-learn", "pandas"):
            try:
                versions[paquet] = importlib.metadata.version(paquet)
            except importlib.metadata.PackageNotFoundError:
                versions[paquet] = None
        trace = dict(commande=commande, cwd=str(RACINE), debut=debut, fin=fin, code=r.returncode,
                     python=sys.version, versions=versions, commit=commit, environnement={
                         k: env[k] for k in ("OPRISK_ANNEE_MIN", "PYTHONIOENCODING", "PYTHONUTF8")},
                     script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                     sortie_sha256=hashlib.sha256(octets).hexdigest(),
                     stderr=str(stderr_path.relative_to(RACINE)))
        (journal_dir / f"{n}.journal.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
        lignes[n] = (f"| {n} | {script.name} | {r.returncode} | {'oui' if erreur else 'non'} | "
                     f"{int(time.time() - t0)} s | {len(octets)} | "
                     f"{hashlib.md5(octets).hexdigest()[:8]} |")
        numeros_faits.append(n)
        enregistrer_manifeste(mf, lignes, commit, numeros_faits, journal_dir.relative_to(RACINE).as_posix())
        print(f"Script {n} : code {r.returncode}, {int(time.time() - t0)} s", flush=True)
        if r.returncode or erreur or not octets:
            sys.exit(f"Échec du script {n} : voir {journal_dir}")
    print("manifeste ecrit :", mf)


if __name__ == "__main__":
    main()
