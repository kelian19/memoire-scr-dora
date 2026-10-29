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

RACINE = (Path(__file__).resolve().parent / '../..').resolve()
MEMOIRE = (Path(__file__).resolve().parent / '../../memoire').resolve()
SORTIES = (Path(__file__).resolve().parent / '../../resultats/sorties_verif').resolve()

SORTIES_INST = (Path(__file__).resolve().parent / '../../resultats/sorties_verif_inst').resolve()
_fusion = {}


def sorties_pour(chapitre: Path) -> Path:
    """Dossier des sorties lues pour ce chapitre.

    La version Institut a 150 pages rejoue ses tests de severite sur l'echantillon 2000-2026
    (OPRISK_ANNEE_MIN=2000) : ses sorties sont dans resultats/sorties_verif_inst/ et remplacent,
    pour ses seuls chapitres (chapitres_inst/), celles de resultats/sorties_verif/. Les autres
    versions gardent les sorties d'origine.
    """
    if chapitre.parent.name != "chapitres_inst" or not SORTIES_INST.is_dir():
        return SORTIES
    surcharges = [f for f in SORTIES_INST.glob("*.txt") if not f.name.startswith("_")]
    if not surcharges:
        return SORTIES
    if "dossier" not in _fusion:
        import shutil
        import tempfile
        dossier = Path(tempfile.mkdtemp(prefix="sorties_inst_"))
        for f in SORTIES.glob("*"):
            if f.is_file():
                shutil.copy2(f, dossier / f.name)
        for f in surcharges:
            shutil.copy2(f, dossier / f.name)
        _fusion["dossier"] = dossier
    return _fusion["dossier"]


def controle_manifeste() -> list:
    """Verifie que le lot de sorties de la version Institut est complet et sain.

    Le manifeste (resultats/sorties_verif_inst/MANIFESTE.md) liste, pour chaque script rejoue, son
    code de retour, la presence d'une erreur, la taille et l'empreinte de sa sortie. Sont refuses :
    un manifeste absent, un code non nul, une erreur, une sortie vide, une empreinte qui ne
    correspond plus au fichier, et un fichier de sortie que le manifeste ne decrit pas."""
    import hashlib
    sorties = [f for f in SORTIES_INST.glob("*.txt") if not f.name.startswith("_")] \
        if SORTIES_INST.is_dir() else []
    if not sorties:
        return []
    mf = SORTIES_INST / "MANIFESTE.md"
    if not mf.is_file():
        return ["MANIFESTE.md absent : le lot de sorties n'est pas atteste"]
    erreurs, vus = [], set()
    for ligne in mf.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (\d+) \| (\S+) \| (\d+) \| (\w+) \| (\d+) s \| (\d+) \| (\w+) \|", ligne)
        if not m:
            continue
        num, _, code, tb, _, taille, empreinte = m.groups()
        vus.add(num)
        f = SORTIES_INST / f"{num}.txt"
        if code != "0":
            erreurs.append(f"script {num} : code de retour {code}")
        if tb != "non":
            erreurs.append(f"script {num} : erreur (Traceback) signalee")
        if not f.is_file() or f.stat().st_size == 0:
            erreurs.append(f"script {num} : sortie absente ou vide")
        else:
            # Le manifeste est calcule sur Windows, ou Git extrait les sorties en CRLF ; sur macOS
            # et Linux elles sont en LF. L'empreinte se compare donc aussi sur la forme CRLF, sans
            # quoi un lot intact est refuse pour une simple difference de fin de ligne.
            brut = f.read_bytes()
            crlf = brut.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            if empreinte not in {hashlib.md5(b).hexdigest()[:8] for b in (brut, crlf)}:
                erreurs.append(f"script {num} : empreinte differente de celle du manifeste")
    for f in sorties:
        if f.stem not in vus:
            erreurs.append(f"{f.name} : sortie non decrite par le manifeste")
    return erreurs


RE_COUV = re.compile(r"COUVERTURE : (\d+) nombres sous controle sur (\d+) publies")
RE_NATURE = re.compile(r"(\d+) nombres hors script PAR NATURE")
RE_VERIF = re.compile(r"(\d+) nombres v\S+ables, (\d+) confirm\S+, (\d+) non confirm")


def analyser(chapitre: Path) -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(
        [sys.executable, str(MEMOIRE / "verif_chiffres.py"), str(sorties_pour(chapitre)), str(chapitre)],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    sortie = r.stdout + r.stderr
    if r.returncode != 0:
        raise SystemExit(f"verif_chiffres.py a echoue sur {chapitre.name} :\n{sortie}")
    couv = RE_COUV.search(sortie)
    verif = RE_VERIF.search(sortie)
    # Un rapport sans ses marqueurs ne vaut pas un bilan nul : il doit faire echouer le controle,
    # sinon une sortie tronquee ou un format change passerait pour un chapitre sans nombre.
    # Seule exception : un chapitre sans aucun nombre n'imprime ni couverture ni bilan, mais
    # « rien à vérifier ».
    rien = "rien à vérifier" in sortie
    if (verif is None and not rien) or (couv is None and not rien):
        raise SystemExit(f"rapport de verif_chiffres.py illisible sur {chapitre.name} :\n{sortie[-2000:]}")
    sous_controle, publies = (int(couv.group(1)), int(couv.group(2))) if couv else (0, 0)
    declares = sum(int(x) for x in RE_NATURE.findall(sortie))
    verifiables, confirmes, non_confirmes = (
        (int(verif.group(1)), int(verif.group(2)), int(verif.group(3))) if verif else (0, 0, 0)
    )
    return {
        "chapitre": ("" if chapitre.parent.name == "chapitres" else chapitre.parent.name + "/")
                    + chapitre.stem,
        "verifiables": verifiables,
        "confirmes": confirmes,
        "non_confirmes": non_confirmes,
        "hors": max(publies - sous_controle - declares, 0),
    }


def main() -> int:
    # Depuis le 1er octobre 2026 (reduction SFE, phase 3), la version ENSAE lit ses chapitres
    # dans chapitres_sfe/, et le contenu qu'elle retire vit dans supplement/ : ils sont
    # verifies comme ceux de la version complete.
    chapitres = (sorted((MEMOIRE / "chapitres").glob("*.tex"))
                 + sorted((MEMOIRE / "chapitres_sfe").glob("*.tex"))
                 + sorted((MEMOIRE / "chapitres_inst").glob("*.tex"))
                 + sorted((MEMOIRE / "supplement").glob("*.tex")))
    if not chapitres:
        print("aucun chapitre trouve", file=sys.stderr)
        return 1
    erreurs_lot = controle_manifeste()
    if erreurs_lot:
        print("ECHEC : lot de sorties Institut non atteste :")
        for e in erreurs_lot:
            print("  -", e)
        return 1
    lignes = [analyser(c) for c in chapitres]

    print(f"{'chapitre':48s}{'verif.':>8s}{'confirmes':>11s}{'non conf.':>11s}{'hors':>7s}")
    print("-" * 85)
    for l in lignes:
        marque = "  <--" if l["non_confirmes"] or l["hors"] else ""
        print(f"{l['chapitre']:48s}{l['verifiables']:8d}{l['confirmes']:11d}"
              f"{l['non_confirmes']:11d}{l['hors']:7d}{marque}")
    tot = {k: sum(l[k] for l in lignes) for k in ("verifiables", "confirmes", "non_confirmes", "hors")}
    print("-" * 85)
    print(f"{'TOTAL':48s}{tot['verifiables']:8d}{tot['confirmes']:11d}"
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
