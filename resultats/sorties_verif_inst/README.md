# Sorties rejouees sur l'echantillon 2000-2026 (version Institut a 150 pages)

Ces fichiers remplacent, pour les seuls chapitres de `memoire/chapitres_inst/`, les sorties de
`resultats/sorties_verif/` du meme numero (`outils/ci/harnais_strict.py`, fonction `sorties_pour`).
Les versions deposees (ENSAE, Institut de 195 pages) gardent les sorties d'origine.

Origine : la calibration gelee (u = 20,03 M EUR, p_u = 0,1509, xi = 0,5954, sigma = 57,97) est celle de
l'echantillon 2000-2026 : 570 pertes, 86 exces, 86/570 = 0,1509. Les comptes 582 et 91 des sorties
d'origine sont ceux de la base courante, qui porte douze pertes d'avant 2000. La frequence gelee,
582/27 = 21,56 par an, reste inchangee et declaree (+2,1 % sur 570/27).

Production : `python outils/ci/lancer_lot_institut.py` (sans argument, les seize scripts ; avec des
numeros, ceux-la seulement). Le lanceur ecrit `MANIFESTE.md` : code de retour, erreur, taille et
empreinte de chaque sortie. Le harnais (`outils/ci/harnais_strict.py`) refuse un lot que le manifeste
n'atteste pas. Les figures rejouees sont dans `recherche/vasicek_lab/figures_inst/`.

La variable `OPRISK_ANNEE_MIN=2000` ne change que la periode du filtre (`filter_finance`) et les
comptes attendus de `config.py` ; le numerateur de la frequence, lu par le moteur, reste 582.
