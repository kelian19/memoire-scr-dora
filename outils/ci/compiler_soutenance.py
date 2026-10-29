"""Prépare les polices locales puis compile le support Beamer éditorial."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deck', choices=('ensae', 'institut', 'tous'), default='ensae',
                        help='Support à compiler (ENSAE par défaut).')
    args = parser.parse_args()
    destination = ROOT/'docs/collaboration/local/da_fonts'
    destination.mkdir(parents=True,exist_ok=True)
    source = Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
    for nom in ('calibri.ttf','calibrib.ttf','calibrii.ttf','calibriz.ttf',
                'georgia.ttf','georgiab.ttf','georgiai.ttf','georgiaz.ttf'):
        cible = destination/nom
        if not cible.exists():
            if not (source/nom).exists():
                raise SystemExit(f'Police nécessaire absente : {source/nom}. Installer les polices ou les placer dans {destination}.')
            shutil.copy2(source/nom,cible)
    compilateur = shutil.which('tectonic')
    if not compilateur:
        candidat = ROOT/'outils/tectonic/tectonic.exe'
        if candidat.exists():
            compilateur = str(candidat)
    if not compilateur:
        candidat = Path.home()/'miniconda3/Library/bin/tectonic.exe'
        if candidat.exists():
            compilateur = str(candidat)
    if not compilateur:
        raise SystemExit('Tectonic est nécessaire pour compiler le support.')
    decks = ('ensae', 'institut') if args.deck == 'tous' else (args.deck,)
    for deck in decks:
        print(f'Compilation du support {deck}...', flush=True)
        r=subprocess.run([compilateur,'--keep-logs',f'soutenance_{deck}.tex'],
            cwd=ROOT/'recherche/slides/beamer',capture_output=True,text=True,encoding='utf-8',errors='replace')
        print(r.stdout+r.stderr, flush=True)
        if r.returncode:
            raise SystemExit(r.returncode)


if __name__=='__main__':
    main()
