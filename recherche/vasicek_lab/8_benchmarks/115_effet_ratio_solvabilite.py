#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
115 : l'ecart entre etats de conformite, lu sur le ratio de solvabilite de quatre assureurs.

LA QUESTION. Un actuaire en poste demande ce que l'ecart DORA fait a SON bilan, et il le lit
en points de ratio de couverture, pas en millions d'euros. Les scripts 65 et 95 donnent tout
ce qu'il faut pour repondre, sans aucun calcul de capital nouveau :
  - 65 : fonds propres eligibles, SCR et couverture publies (SFCR 2024), besoin ORSA au titre
         de DORA par etat de conformite ;
  - 95 : la convention d'agregation. Le risque operationnel entre ADDITIVEMENT dans le SCR
         (art. 103 de la directive : SCR = BSCR + Adj + SCR_op), donc un euro de besoin
         compte pour un euro de SCR.

CE QUE LE SCRIPT CALCULE. Une lecture ORSA, jamais reglementaire :
    ratio publie        = FP / SCR
    ratio ORSA, etat e  = FP / (SCR + besoin_e),   e = conforme, non conforme
    ecart en points     = ratio ORSA conforme - ratio ORSA non conforme.
Aucune valeur n'est saisie : tout est lu dans sorties_verif/65.txt.

TROIS RESERVES, IMPRIMEES AVEC LE RESULTAT.
  1. L'etat de conformite de ces entites n'est pas publie : il est SUPPOSE.
  2. A l'echelle de l'entite, seul le canal de propagation varie entre les etats (65,
     section 6) : l'ecart en points est donc un PLANCHER de l'ecart a quatre canaux.
  3. Le NIVEAU du besoin est une borne superieure d'ordre de grandeur hors du domaine de
     validite de la descente d'echelle (65, verdict 4 : besoin superieur a 10 % du SCR).
     Pour ces entites, seul l'ecart entre etats se lit ; le ratio ORSA lui-meme ne se lit
     pas comme une prevision.

Ne produit aucune figure et ne touche ni config.py, ni un parametre, ni une sortie existante.
Sortie : sorties_verif/115.txt. Lu par l'annexe C du support de soutenance.
"""

import os
import re
import sys

SORTIE_65 = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '../../../resultats/sorties_verif/65.txt'))

# GARDE-FOU, repris du script 95 : les valeurs que le memoire publie pour ces entites.
# Si le 65 derive, ce script s'arrete au lieu de publier.
ATTENDU = {
    "Assureur non-vie A": dict(orsa=112.1, scr=425.0, conforme=77.2),
    "Assureur non-vie B": dict(orsa=135.5, scr=812.0, conforme=91.1),
    "Assureur vie C":     dict(orsa=195.8, scr=1587.0, conforme=133.1),
    "Assureur vie D":     dict(orsa=294.4, scr=14800.0, conforme=208.4),
}
# Le critere du domaine de validite, tel que le verdict du 65 le formule.
SEUIL_VALIDITE = 10.0   # besoin ORSA en % du SCR publie

W_ = 92


def titre(s):
    print("\n" + "=" * W_ + f"\n{s}\n" + "=" * W_)


def nombre_fr(s):
    """« 3 234 » -> 3234.0 ; « 34,8 » -> 34.8."""
    return float(s.replace(" ", "").replace(" ", "").replace(",", "."))


def lire_65():
    if not os.path.exists(SORTIE_65):
        sys.exit(f"ARRET : {SORTIE_65} est introuvable. Relancer le script 65 d'abord.")
    with open(SORTIE_65, encoding="utf-8", errors="replace") as fh:
        txt = fh.read()

    anon = {}
    for m in re.finditer(r"^\s{2}(Assureur (?:non-vie|vie) [A-D])\s+=\s+(.+?)\s*$", txt, re.M):
        anon[m.group(2).strip()] = m.group(1).strip()
    if len(anon) != 4:
        sys.exit(f"ARRET : {len(anon)} etiquettes anonymes lues au lieu de 4.")
    ent = {a: {} for a in anon.values()}

    # Section 0 : SCR et couverture publies.
    bloc = txt.split("CORRESPONDANCE AVEC")[0]
    for ligne in bloc.splitlines():
        m = re.match(r"^(.+?)\s{2,}solo, \S+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)%\s*$", ligne)
        if m and m.group(1).strip() in anon:
            e = ent[anon[m.group(1).strip()]]
            e["couv_publiee"] = float(m.group(5))

    # Provenance : fonds propres eligibles, lus dans la ligne « source » de chaque entite.
    bloc = txt.split("PROVENANCE DE CHAQUE CHAMP")[1].split("DEUX RESERVES")[0]
    courant = None
    for ligne in bloc.splitlines():
        if ligne.strip() in anon:
            courant = anon[ligne.strip()]
            continue
        m = re.search(r"fonds propres eligibles ([\d ,]+?) (M|Md) EUR", ligne)
        if courant and m:
            fp = nombre_fr(m.group(1)) * (1000.0 if m.group(2) == "Md" else 1.0)
            ent[courant]["fp"] = fp

    # Section 4 : besoin ORSA, SCR publie et part du SCR.
    bloc = txt.split("4. Les deux reperes de l'entite")[1].split("STATUT DU DENOMINATEUR")[0]
    for ligne in bloc.splitlines():
        m = re.match(r"^(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)%\s*$", ligne)
        if m and m.group(1).strip() in anon:
            e = ent[anon[m.group(1).strip()]]
            e["orsa"], e["scr"], e["part"] = float(m.group(2)), float(m.group(5)), float(m.group(6))

    # Section 6 : besoin par etat de conformite.
    bloc = txt.split("6. Par etat de conformite")[1].split("VERDICT")[0]
    for ligne in bloc.splitlines():
        m = re.match(r"^(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\+([\d.]+)\s+([\d.]+)\s*$",
                     ligne)
        if m and m.group(1).strip() in anon:
            e = ent[anon[m.group(1).strip()]]
            e["conforme"], e["non_conforme"] = float(m.group(2)), float(m.group(4))
            e["ecart"], e["ecart_pct"] = float(m.group(5)), float(m.group(6))

    requis = {"fp", "couv_publiee", "orsa", "scr", "part", "conforme", "non_conforme", "ecart"}
    manquant = [a for a, e in ent.items() if not requis <= set(e)]
    if manquant:
        sys.exit(f"ARRET : lecture incomplete du 65 pour {manquant}.")
    return ent


def controle(ent):
    for etiquette, att in ATTENDU.items():
        for champ, attendu in att.items():
            lu = ent[etiquette][champ]
            demi = 0.05 if "." in str(attendu) and str(attendu).split(".")[1] != "0" else 0.5
            if abs(lu - attendu) > max(0.006 * abs(attendu), demi):
                sys.exit(f"ARRET : {etiquette}, {champ} lu {lu}, attendu {attendu}. "
                         "Le script 65 a derive : ne pas publier cette sortie.")
    for etiquette, e in ent.items():
        # Le besoin non conforme est le besoin publie, et l'ecart en est la difference.
        if abs(e["non_conforme"] - e["orsa"]) > 0.051:
            sys.exit(f"ARRET : {etiquette}, besoin non conforme {e['non_conforme']} "
                     f"different du besoin publie {e['orsa']}.")
        if abs(e["non_conforme"] - e["conforme"] - e["ecart"]) > 0.11:
            sys.exit(f"ARRET : {etiquette}, l'ecart lu ne vaut pas non conforme - conforme.")
        # Les fonds propres lus doivent redonner la couverture publiee, au point d'arrondi
        # pres (fonds propres et SCR sont eux-memes publies arrondis).
        if abs(100.0 * e["fp"] / e["scr"] - e["couv_publiee"]) > 1.0:
            sys.exit(f"ARRET : {etiquette}, FP / SCR = {100 * e['fp'] / e['scr']:.1f} % "
                     f"contre {e['couv_publiee']:.0f} % publie.")
    print("  controle du 65 : 12 valeurs publiees retrouvees ; besoin non conforme = besoin publie ;")
    print("  ecart = non conforme - conforme ; FP / SCR redonne la couverture publiee a 1 point pres.")


def main():
    titre("115 : l'ecart entre etats de conformite, en points de ratio de solvabilite")
    print("Lecture ORSA, jamais reglementaire. Tout est lu dans sorties_verif/65.txt (SFCR 2024).")
    print("Agregation additive (script 95) : un euro de besoin compte pour un euro de SCR.")
    ent = lire_65()
    controle(ent)

    titre("1. Ratio publie, puis ratio ORSA par etat de conformite")
    print(f"{'entite':<20}{'FP':>9}{'SCR':>8}{'publie':>9}{'conforme':>10}{'non conf.':>11}"
          f"{'ecart':>9}{'besoin/SCR':>12}  domaine")
    print(f"{'':<20}{'M EUR':>9}{'M EUR':>8}{'%':>9}{'%':>10}{'%':>11}{'points':>9}{'%':>12}")
    lignes = {}
    for etiquette in sorted(ent):
        e = ent[etiquette]
        r_pub = 100.0 * e["fp"] / e["scr"]
        r_c = 100.0 * e["fp"] / (e["scr"] + e["conforme"])
        r_nc = 100.0 * e["fp"] / (e["scr"] + e["non_conforme"])
        dans = e["part"] <= SEUIL_VALIDITE
        lignes[etiquette] = (r_pub, r_c, r_nc, r_c - r_nc, dans)
        print(f"{etiquette:<20}{e['fp']:>9.0f}{e['scr']:>8.0f}{r_pub:>9.1f}{r_c:>10.1f}{r_nc:>11.1f}"
              f"{r_c - r_nc:>9.1f}{e['part']:>12.1f}  {'dans' if dans else 'hors (niveau = borne sup.)'}")

    titre("2. Lecture")
    dans = [a for a, l in lignes.items() if l[4]]
    hors = [a for a, l in lignes.items() if not l[4]]
    ecarts = [l[3] for l in lignes.values()]
    print(f"  L'ecart entre etats coute de {min(ecarts):.1f} a {max(ecarts):.1f} points de ratio "
          f"selon l'entite.")
    print(f"  {'entite':<20}{'ecart M EUR':>12}{'en % du SCR':>13}{'points':>9}")
    for etiquette in sorted(lignes):
        e = ent[etiquette]
        print(f"  {etiquette:<20}{e['ecart']:>12.1f}{100 * e['ecart'] / e['scr']:>13.2f}"
              f"{lignes[etiquette][3]:>9.1f}")
    print(f"  Dans le domaine de validite (besoin <= {SEUIL_VALIDITE:.0f} % du SCR) : {', '.join(dans)}.")
    print(f"  Hors du domaine, niveau = borne superieure : {', '.join(hors)}.")
    print("  Pour ces dernieres, les ratios ORSA ne se lisent pas comme une prevision ;")
    print("  seul l'ecart entre etats est interprete, et c'est un plancher (canal de propagation seul).")

    titre("RESERVES")
    print("  1. L'etat de conformite de ces entites n'est pas publie : il est suppose.")
    print("  2. A l'echelle de l'entite seul le canal de propagation varie (65, section 6) :")
    print("     l'ecart en points est un plancher de l'ecart a quatre canaux.")
    print("  3. Le niveau du besoin est une borne superieure hors du domaine de validite.")

    titre("GRANDEURS CITEES (sans separateur de milliers, pour le harnais)")
    for etiquette in sorted(lignes):
        r_pub, r_c, r_nc, d, _ = lignes[etiquette]
        print(f"  {etiquette:<20} publie {r_pub:.1f}  conforme {r_c:.1f}  non conforme {r_nc:.1f}"
              f"  ecart {d:.1f}  FP {ent[etiquette]['fp']:.0f}")


if __name__ == "__main__":
    main()
