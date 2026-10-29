"""Exporte la version Institut et ses dependances locales dans un projet Overleaf autonome."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
MEM = ROOT / "memoire"


def clean(text):
    return re.sub(r"(?<!\\)%.*", "", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="Nouveau dossier de preparation")
    parser.add_argument("--nexialog", action="store_true",
                        help="Fond Nexialog, en-tete adapte et figures regroupees pour Overleaf")
    args = parser.parse_args()
    stage = args.destination.resolve()
    stage.mkdir(parents=True, exist_ok=False)
    master = MEM / "overleaf/main.tex" if args.nexialog else MEM / "main_institut_150.tex"
    todo, sources = [master], set()
    while todo:
        src = todo.pop()
        if src in sources:
            continue
        if not src.is_file():
            raise FileNotFoundError(src)
        sources.add(src)
        for name in re.findall(r"\\(?:input|include)\s*\{([^}]+)\}", clean(src.read_text(encoding="utf-8"))):
            if "#" in name:
                raise ValueError(f"Dependance dynamique a verifier : {name}")
            todo.append(MEM / (name if Path(name).suffix else name + ".tex"))

    graphic_dirs = [MEM, ROOT / "recherche/vasicek_lab/figures_inst",
                    ROOT / "recherche/vasicek_lab/figures",
                    ROOT / "recherche/cascade_qualitative/figures", ROOT / "resultats/outputs"]
    assets = set()
    for src in sources:
        text = clean(src.read_text(encoding="utf-8"))
        names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
        names += re.findall(r"\\(?:figover|figcle|pgsigne)\{([^}]+)\}", text)
        for name in names:
            if "#" in name:
                continue
            if args.nexialog and name == "RequiredFiles/fond_latex_2.pdf":
                # Le fond provient du projet Nexialog de l'utilisateur et n'est pas dans le depot.
                continue
            ext = ("",) if Path(name).suffix else (".pdf", ".png", ".jpg", ".jpeg")
            candidates = [base / (name + e) for base in graphic_dirs for e in ext]
            found = next((p.resolve() for p in candidates if p.is_file()), None)
            if found is None:
                if name.startswith("logos/signature_") and re.search(
                    r"\\pgsigne\{" + re.escape(name) + r"\}", text
                ):
                    # Comme \pgsigne : la couverture reste compilable sans signature privee.
                    continue
                raise FileNotFoundError(f"Image manquante : {name}, dans {src.name}")
            assets.add(found)
        for name in re.findall(r"\\bibliography(?:style)?\{([^}]+)\}", text):
            suffix = ".bst" if "plainnat" in name else ".bib"
            for part in name.split(","):
                assets.add(MEM / (part + suffix))

    manifest = []
    for src in sorted(sources | assets):
        if not src.is_file():
            raise FileNotFoundError(src)
        relative = src.relative_to(MEM) if src.is_relative_to(MEM) else src.relative_to(ROOT)
        if args.nexialog and src not in sources and not src.is_relative_to(MEM):
            relative = Path("figures_memoire") / src.name
        if src == master:
            relative = Path("main.tex")
        target = stage / relative
        if target.exists():
            raise FileExistsError(f"Collision de chemins dans l'export : {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if src in sources:
            text = src.read_text(encoding="utf-8")
            # Compilation depuis la racine du ZIP, au lieu du sous-dossier memoire/.
            text = re.sub(r"(\\graphicspath[^\n]*)",
                          lambda m: m.group(0).replace("../", ""), text)
            target.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(src, target)
        manifest.append(dict(source=src.relative_to(ROOT).as_posix(),
                             exported_as=relative.as_posix(),
                             sha256=hashlib.sha256(target.read_bytes()).hexdigest()))

    fond_note = (
        "Version Nexialog : conserver ou ajouter RequiredFiles/fond_latex_2.pdf depuis le projet Nexialog.\n"
        "Le fond n'est pas inclus dans le ZIP ; il est necessaire pour compiler cette variante.\n"
        if args.nexialog else "Version Institut sans fond de page Nexialog.\n"
    )
    (stage / "LIRE_AVANT_IMPORT.txt").write_text(
        "VERSION INSTITUT — 29 octobre 2026\n\n"
        "Importer ce ZIP dans un NOUVEAU projet Overleaf pour conserver le projet actuel.\n"
        "Document principal : main.tex. Compilateur : XeLaTeX.\n"
        "Contenu : version Institut actuelle, dont la methode de downscaling et la comparaison d'entites.\n"
        "PDF local de reference : memoire/main_institut_150.pdf. Le nombre de pages est a verifier apres compilation.\n"
        "Les chemins de figures sont adaptes a la racine de ce projet autonome.\n"
        "Les donnees brutes et les scripts Python ne sont pas necessaires a la compilation.\n"
        + fond_note,
        encoding="utf-8")
    (stage / "MANIFESTE_EXPORT.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    archive = stage.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for src in sorted(stage.rglob("*")):
            if src.is_file():
                z.write(src, src.relative_to(stage).as_posix())
    print(f"{len(sources)} fichiers LaTeX, {len(assets)} dependances, {len(manifest)} fichiers sources")
    print(f"ZIP : {archive} ({archive.stat().st_size / 1024**2:.2f} Mo)")


if __name__ == "__main__":
    main()
