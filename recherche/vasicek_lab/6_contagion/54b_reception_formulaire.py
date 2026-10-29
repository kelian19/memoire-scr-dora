#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""54b : convertit l'export du formulaire de codage en aveugle vers le format du script 54.

POURQUOI CE SCRIPT EXISTE. Le script 54 lit un CSV en format LONG (une ligne par couple
incident x paire, colonnes incident,paire,code). Le formulaire Google, lui, rend une ligne
LARGE par repondant. Sans ce convertisseur, la reception d'un second codage se termine par une
transcription a la main de 70 cellules, le soir ou le fichier arrive, ce qui est exactement la
maniere de se tromper.

DEUX PIEGES TROUVES EN LISANT LE FORMULAIRE, ET C'EST LA RAISON D'ETRE DU SCRIPT.

  1. LES SEPT GRILLES PORTENT LE MEME INTITULE de question ("Dans cet incident, qu'est-ce qui a
     lache en premier ?"). L'export Google produit donc 70 colonnes aux en-tetes IDENTIQUES,
     et l'appariement colonne -> recit ne peut PAS se faire sur le texte. Il se fait sur la
     POSITION : les grilles sont ajoutees dans l'ordre des recits, donc les colonnes viennent
     par sept blocs consecutifs de dix. Ce script verifie cette structure au lieu de la
     supposer, et refuse de convertir si elle n'est pas exacte.

  2. LE FORMULAIRE PROPOSE DES REPONSES POSITIONNELLES ("le 1er a entraine le 2e") alors que
     le script 54 attend un NOM DE DOMAINE ou une des trois formules libres. Tel quel, le 54
     leverait "ValueError: reponse non comprise" sur la premiere cellule directionnelle.

CE QUE CE SCRIPT NE FAIT PAS, ET C'EST DELIBERE. Il n'interprete RIEN. Il ne connait pas la
convention "k>j / j>k" et ne la reproduit pas : pour une reponse directionnelle il ecrit le NOM
du domaine designe, que le script 54 sait deja traduire. La semantique de l'ordre reste donc
dans UN SEUL fichier, le 54, et ce convertisseur ne peut pas en divergerticement. Les trois
formules libres ("en meme temps", "non concerne", "je ne sais pas") passent VERBATIM, le 54 les
connaissant aussi. Une cellule vide est OMISE et non codee : le 54 restreint l'accord aux
cellules communes, donc une non-reponse ne fabrique aucun code.

GARDE-FOU QUI COMPTE. Le script 54 ramasse N'IMPORTE QUEL fichier codage_*.csv du dossier
notes/codage. Un fichier de test oublie la produirait un kappa sur des donnees inventees. Ce
convertisseur ecrit donc A COTE DU FICHIER D'ENTREE et JAMAIS dans notes/codage : il imprime la
commande de copie, pour que le pas qui rend un codage actif reste un geste manuel et conscient.

Usage :
    python 54b_reception_formulaire.py <export_google.csv> [INITIALES]

    INITIALES : celles du second codeur, par defaut "HR". Elles nomment le fichier de sortie
    codage_<INITIALES>.csv, que le script 54 attend.
"""
import csv
import os
import re
import sys
import unicodedata
from itertools import combinations

# --- structure du formulaire, telle qu'ecrite dans notes/form_codage_aveugle.gs --------------
# L'ORDRE DE CES DEUX LISTES EST CONTRACTUEL : il doit reproduire PAIRES et OPTIONS du .gs.
# Toute divergence est detectee par les controles de `verifier_structure`, pas supposee absente.
TITRE_GRILLE = "Dans cet incident, qu'est-ce qui a lâché en premier ?"

PAIRES_FORM = [
    "1. Gouvernance / Incidents",
    "2. Gouvernance / Tests",
    "3. Gouvernance / Prestataires",
    "4. Gouvernance / Partage d'infos",
    "5. Incidents / Tests",
    "6. Incidents / Prestataires",
    "7. Incidents / Partage d'infos",
    "8. Tests / Prestataires",
    "9. Tests / Partage d'infos",
    "10. Prestataires / Partage d'infos",
]

OPT_PREMIER = "le 1er a entraîné le 2e"
OPT_SECOND = "le 2e a entraîné le 1er"
OPT_LIBRES = ["en même temps", "non concerné", "je ne sais pas"]

N_RECITS_FORM = 7          # le formulaire couvre les sept premiers recits du kit
N_PAIRES = 10

# numero de pilier par domaine, pour le seul controle de la premisse "1er = plus petit numero"
PILIER = {"gouvernance": 1, "incidents": 2, "tests": 3, "prestataires": 4,
          "partage d'infos": 5}

WID = 84


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def norm(s):
    """Comparaison robuste : accents, casse, apostrophes typographiques, espaces multiples."""
    s = s.replace("’", "'")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.strip().lower().split())


def libelle_de_paire(libelle):
    """'3. Gouvernance / Prestataires' -> ('Gouvernance', 'Prestataires')."""
    sans_num = re.sub(r"^\s*\d+\s*\.\s*", "", libelle)
    gauche, droite = (x.strip() for x in sans_num.split("/"))
    return gauche, droite


def paires_internes():
    """P1-P2, P1-P3, ... dans l'ordre des combinaisons, comme PAIRS du script 54."""
    return [f"P{a}-P{b}" for a, b in combinations([1, 2, 3, 4, 5], 2)]


# =====================================================================================
def verifier_structure(entetes):
    """Localise les 70 colonnes de grille et verifie que la structure est bien celle attendue.

    Renvoie la liste des indices de colonnes, dans l'ordre. Leve AssertionError sinon : mieux
    vaut un refus explicite qu'une conversion silencieusement decalee d'une colonne.
    """
    cible = norm(TITRE_GRILLE)
    idx = [i for i, h in enumerate(entetes) if norm(h).startswith(cible)]

    attendu = N_RECITS_FORM * N_PAIRES
    assert len(idx) == attendu, (
        f"{len(idx)} colonnes de grille trouvees, {attendu} attendues "
        f"({N_RECITS_FORM} recits x {N_PAIRES} paires). L'export ne correspond pas au "
        f"formulaire de notes/form_codage_aveugle.gs : ne pas convertir a l'aveugle.")

    # les indices doivent etre CONSECUTIFS par blocs de dix ; un trou signalerait une question
    # inseree entre deux grilles, donc une numerotation de recit fausse.
    for b in range(N_RECITS_FORM):
        bloc = idx[b * N_PAIRES:(b + 1) * N_PAIRES]
        assert bloc == list(range(bloc[0], bloc[0] + N_PAIRES)), (
            f"les dix colonnes du recit {b + 1} ne sont pas consecutives : {bloc}")

    # dans chaque bloc, les libelles de ligne doivent apparaitre dans l'ordre de PAIRES_FORM
    for b in range(N_RECITS_FORM):
        for k in range(N_PAIRES):
            h = entetes[idx[b * N_PAIRES + k]]
            m = re.search(r"\[(.+)\]\s*$", h)
            assert m, f"en-tete sans libelle de ligne entre crochets : {h!r}"
            attendu_lib = norm(PAIRES_FORM[k])
            assert norm(m.group(1)) == attendu_lib, (
                f"recit {b + 1}, colonne {k + 1} : libelle {norm(m.group(1))!r} "
                f"au lieu de {attendu_lib!r}. L'ordre des lignes de la grille a change.")
    return idx


def verifier_premisse_ordre():
    """Le 1er de chaque libelle porte-t-il le plus PETIT numero de pilier ?

    C'est la premisse qui autorise a traduire une reponse positionnelle en nom de domaine, et
    elle se VERIFIE. Si un libelle etait ecrit "Tests / Gouvernance", ecrire le nom du domaine
    designe resterait correct, mais l'appariement avec la paire interne P1-P3 changerait de
    sens : autant refuser.
    """
    internes = paires_internes()
    for k, lib in enumerate(PAIRES_FORM):
        g, d = libelle_de_paire(lib)
        pg, pd = PILIER[norm(g)], PILIER[norm(d)]
        assert pg < pd, f"{lib!r} : le 1er ({g}) n'a pas le plus petit numero"
        assert internes[k] == f"P{pg}-P{pd}", (
            f"la paire {k + 1} du formulaire ({lib}) vaut P{pg}-P{pd}, "
            f"mais la paire {k + 1} du script 54 est {internes[k]}. Les deux ordres divergent.")


# =====================================================================================
def convertir_ligne(ligne, entetes, idx):
    """Une ligne large de reponses -> liste de (incident, paire, code en langage naturel)."""
    internes = paires_internes()
    sorties, vides, comptes = [], 0, {}
    for b in range(N_RECITS_FORM):
        incident = f"Recit {b + 1}"           # etiquette anonyme, celle du kit et du gabarit
        for k in range(N_PAIRES):
            brut = (ligne[idx[b * N_PAIRES + k]] or "").strip()
            if not brut:
                vides += 1
                continue                       # cellule vide : OMISE, jamais codee
            n = norm(brut)
            gauche, droite = libelle_de_paire(PAIRES_FORM[k])
            if n == norm(OPT_PREMIER):
                code = gauche
            elif n == norm(OPT_SECOND):
                code = droite
            elif n in {norm(o) for o in OPT_LIBRES}:
                code = brut                    # verbatim : le script 54 connait ces formules
            else:
                raise ValueError(
                    f"reponse inattendue au recit {b + 1}, paire {PAIRES_FORM[k]} : {brut!r}. "
                    f"Les options du formulaire sont {[OPT_PREMIER, OPT_SECOND] + OPT_LIBRES}.")
            comptes[n] = comptes.get(n, 0) + 1
            sorties.append((incident, internes[k], code))
    return sorties, vides, comptes


# =====================================================================================
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    src = os.path.abspath(sys.argv[1])
    initiales = (sys.argv[2] if len(sys.argv) > 2 else "HR").strip().upper()

    titre("0. Controles de structure, AVANT toute conversion")
    verifier_premisse_ordre()
    print("  premisse 'le 1er porte le plus petit numero de pilier' : verifiee sur les 10 paires")
    print("  ordre des paires du formulaire == ordre des paires du script 54 : verifie")

    with open(src, newline="", encoding="utf-8-sig") as f:
        lignes = list(csv.reader(f))
    assert lignes, "fichier vide"
    entetes, corps = lignes[0], [l for l in lignes[1:] if any(x.strip() for x in l)]
    idx = verifier_structure(entetes)
    print(f"  {len(idx)} colonnes de grille localisees par POSITION (en-tetes identiques)")
    print(f"  {len(corps)} reponse(s) dans l'export")
    assert corps, "aucune reponse dans l'export"

    titre("1. Conversion")
    dossier = os.path.dirname(src)
    ecrits = []
    for n, ligne in enumerate(corps):
        ligne = ligne + [""] * (len(entetes) - len(ligne))     # Google tronque les fins vides
        cells, vides, comptes = convertir_ligne(ligne, entetes, idx)
        suffixe = "" if len(corps) == 1 else f"_{n + 1}"
        dest = os.path.join(dossier, f"codage_{initiales}{suffixe}.csv")
        with open(dest, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["incident", "paire", "code"])
            w.writerows(cells)
        ecrits.append(dest)
        print(f"  reponse {n + 1} : {len(cells)} cellules codees, {vides} laissees vides")
        for k in sorted(comptes):
            print(f"      {k:<28} {comptes[k]:>3}")
        print(f"  ecrit : {dest}")

    titre("2. Ce qu'il reste a faire, ET QUI DOIT RESTER UN GESTE MANUEL")
    print("  Le script 54 ramasse tout fichier codage_*.csv de notes/codage. Un fichier de test")
    print("  oublie la produirait un kappa sur des donnees inventees. La copie est donc a faire")
    print("  a la main, en connaissance de cause :")
    print()
    for d in ecrits:
        print(f'    Copy-Item "{d}" "exploratory\\vasicek_lab\\notes\\codage\\"')
    print()
    print("  puis, depuis la racine du depot :")
    print("    .venv\\Scripts\\python.exe exploratory\\vasicek_lab\\6_contagion"
          "\\54_fiabilite_inter_juges.py")
    print()
    print("  Le 54 imprimera l'accord brut, le kappa de Cohen, l'accord directionnel et le")
    print("  nombre d'INVERSIONS DE SENS. Cette derniere statistique se lit separement du")
    print("  kappa : a zero inversion, les deux codeurs lisent la meme orientation meme si le")
    print("  kappa est mediocre, le desaccord portant alors sur 'direction contre simultaneite'.")


if __name__ == "__main__":
    main()
