#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
95 : ce que l'AGREGATION reglementaire fait du besoin ORSA au titre de DORA.

CE QUI MANQUAIT, ET LE JURY LE DEMANDERA. Le memoire chiffre un besoin de capital et un
ecart entre etats de conformite, puis il s'arrete la. Il ne dit nulle part ce que devient
cet ecart une fois porte au capital total d'une entite. Or la reponse n'est pas neutre :
sous Solvabilite II le risque operationnel entre ADDITIVEMENT, hors de la matrice de
correlation du BSCR (art. 103 : SCR = BSCR + Adj + SCR_op). Un euro d'ecart DORA coute
donc un euro de capital, la ou un euro de risque de marche en coute une fraction.

CE QUE FAIT CE SCRIPT. Il compare deux conventions d'agregation sur les MEMES montants,
ceux que le script 65 calcule sur quatre bilans reels :

  (1) la convention reglementaire, ADDITIVE : Delta SCR = D, taux de reconnaissance 1 ;
  (2) un contrefactuel CORRELE, ou le meme montant serait agrege comme un module de plus,
      correle a rho avec le BSCR pris comme un bloc :
          Delta SCR = sqrt(BSCR^2 + 2 rho BSCR D + D^2) - BSCR,
      soit un taux de reconnaissance tau(rho, d) = (sqrt(1 + 2 rho d + d^2) - 1) / d,
      avec d = D / BSCR.

Le contrefactuel N'EST PAS ce que fait Solvabilite II. Il sert a MESURER ce que la
convention additive implique, et il n'a pas d'autre statut.

CE QUI EST DEMONTRE ICI, et c'est une identite, pas une mesure :
    tau(rho, d) = rho + (d/2)(1 - rho^2) + O(d^2).
Donc pour un montant petit devant le BSCR, le taux de reconnaissance TEND VERS rho
lui-meme : un ecart reconnu en entier par la convention additive ne le serait qu'a
hauteur de rho sous agregation correlee. tau est croissant en d, et tau(1, d) = 1.

CE QUI EST LU, ET CE QUI EST POSE.
  LU : tout vient de sorties_verif/65.txt, donc de chiffres SFCR publies au 31/12/2024.
       Besoin ORSA par entite, ecart entre etats, SCR publie, et pour les deux entites
       non-vie le module de risque operationnel publie a l'etat S.25.01. Aucune valeur
       n'est saisie ici : une derive du 65 arrete ce script au lieu de publier en silence.
  POSE : rho, balaye de 0 a 0,75 et jamais choisi ; et Adj <= 0, qui est une propriete de
       l'ajustement pour capacite d'absorption des pertes, non une hypothese de calibration.

LA BORNE SUR LE BSCR, ET POURQUOI ELLE SUFFIT. Le BSCR n'est pas dans les trois champs
retenus par le 65. On n'en a pas besoin : deux identites le bornent par le bas.
  - Adj <= 0 et SCR = BSCR + Adj + SCR_op donnent    BSCR >= SCR - SCR_op ;
  - le plafond SCR_op <= 0,3 BSCR donne ensuite      BSCR >= SCR / 1,3, toujours.
tau etant croissant en d = D / BSCR, minorer le BSCR MAJORE le taux de reconnaissance du
contrefactuel : les taux imprimes ici sont donc des MAJORANTS, et l'ecart entre les deux
conventions est au moins celui qu'on lit.

CE QUI N'EST PAS FAIT, ET POURQUOI. Aucune figure : les quatre lignes de resultat tiennent
dans une table, et une figure regeneree sur un autre poste ne serait pas identique a
l'octet. Aucune recalibration : ce script ne touche ni config.py, ni un parametre, ni une
sortie existante. Il ne lit pas data/raw et ne depend d'aucun module du laboratoire.

Sortie : sorties_verif/95.txt. Cite au chapitre 12, section de l'agregation reglementaire.
"""

import math
import os
import re
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(os.path.dirname(ICI)))
SORTIE_65 = os.path.join(RACINE, "sorties_verif", "65.txt")

# Le plafond de Formule Standard, art. 204 du reglement delegue 2015/35. C'est la SEULE
# constante reglementaire utilisee ici, et elle sert de borne, pas de calibration.
PLAFOND = 0.30

# Les correlations balayees. Aucune n'est retenue : le resultat est la FORME de tau, et
# l'ordre de grandeur de l'ecart entre les deux conventions.
RHOS = [0.00, 0.25, 0.50, 0.75]

# GARDE-FOU. Les valeurs que le memoire publie au chapitre 12 pour ces quatre entites.
# Si le 65 derive, ce script s'arrete : il ne doit jamais publier un chiffre que le
# document ne porte pas. Tolerance : la demi-unite du dernier chiffre ecrit.
ATTENDU = {
    "Assureur non-vie A": dict(orsa=112.1, scr=425.0),
    "Assureur non-vie B": dict(orsa=135.5, scr=812.0),
    "Assureur vie C":     dict(orsa=195.8, scr=1587.0),
    "Assureur vie D":     dict(orsa=294.4, scr=14800.0),
}

W_ = 92


def titre(s):
    print("\n" + "=" * W_ + f"\n{s}\n" + "=" * W_)


def lire_65():
    """Lit le 65 : correspondance des etiquettes, besoins, SCR publies, module publie,
    et ecart entre etats. Rend un dict par ETIQUETTE ANONYME, jamais par nom de societe :
    les identites restent dans le script 65, qui est le seul endroit qui les porte."""
    if not os.path.exists(SORTIE_65):
        sys.exit(f"ARRET : {SORTIE_65} est introuvable. Relancer le script 65 d'abord.")
    with open(SORTIE_65, encoding="utf-8", errors="replace") as fh:
        txt = fh.read()

    # 1. La table de correspondance : « Assureur non-vie A     = Identite A ».
    anon = {}
    for m in re.finditer(r"^\s{2}(Assureur (?:non-vie|vie) [A-D])\s+=\s+(.+?)\s*$",
                         txt, re.M):
        anon[m.group(2).strip()] = m.group(1).strip()
    if len(anon) != 4:
        sys.exit(f"ARRET : {len(anon)} etiquettes anonymes lues au lieu de 4.")

    ent = {a: dict(nom_interne=n) for n, a in anon.items()}

    # 2. Section 4 : besoin ORSA, forfait, rapport, SCR publie, part du SCR.
    bloc = txt.split("4. Les deux reperes de l'entite")[1].split("STATUT DU DENOMINATEUR")[0]
    for ligne in bloc.splitlines():
        m = re.match(r"^(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)%\s*$",
                     ligne)
        if not m or m.group(1).strip() not in anon:
            continue
        e = ent[anon[m.group(1).strip()]]
        e["orsa"] = float(m.group(2))
        e["scr"] = float(m.group(5))
        e["part"] = float(m.group(6))

    # 3. Le module operationnel PUBLIE (etat S.25.01), disponible sur les deux non-vie.
    bloc = txt.split("LE REPERE FORFAITAIRE CONFRONTE AU MODULE PUBLIE")[1].split("=" * 40)[0]
    for ligne in bloc.splitlines():
        m = re.match(r"^(.+?)\s+([\d.]+)\s+([\d.]+)\s+(-?\d+)%\s*$", ligne)
        if not m or m.group(1).strip() not in anon:
            continue
        ent[anon[m.group(1).strip()]]["op_publie"] = float(m.group(3))

    # 4. Section 6 : l'ecart entre etats de conformite, qui est l'objet de la these.
    bloc = txt.split("6. Par etat de conformite")[1].split("VERDICT")[0]
    for ligne in bloc.splitlines():
        m = re.match(r"^(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\+([\d.]+)\s+([\d.]+)\s*$",
                     ligne)
        if not m or m.group(1).strip() not in anon:
            continue
        e = ent[anon[m.group(1).strip()]]
        e["conforme"] = float(m.group(2))
        e["non_conforme"] = float(m.group(4))
        e["ecart"] = float(m.group(5))

    manquant = [a for a, e in ent.items()
                if not {"orsa", "scr", "ecart"} <= set(e)]
    if manquant:
        sys.exit(f"ARRET : lecture incomplete du 65 pour {manquant}.")
    return ent


def controle(ent):
    """Le 65 n'a pas derive ? Sinon on s'arrete. La tolerance est la demi-unite du dernier
    chiffre ecrit, jamais un plancher absolu : un plancher ecrase ce qui vaut moins que lui."""
    for etiquette, att in ATTENDU.items():
        e = ent[etiquette]
        for champ, attendu in att.items():
            lu = e[champ]
            demi = 0.5 * 10 ** (-len(str(attendu).split(".")[1]) if "." in str(attendu) else 0)
            if abs(lu - attendu) > max(0.006 * abs(attendu), demi):
                sys.exit(f"ARRET : {etiquette}, champ {champ} lu {lu}, attendu {attendu}. "
                         "Le script 65 a derive : ne pas publier cette sortie.")
    print("  controle du 65 : les 8 valeurs publiees au chapitre 12 sont retrouvees.")


def bscr_min(e):
    """Minorant du BSCR a partir des seuls champs publies.
    Adj <= 0 donne BSCR >= SCR - SCR_op ; le plafond donne BSCR >= SCR / 1,3."""
    borne_plafond = e["scr"] / (1.0 + PLAFOND)
    if "op_publie" in e:
        return max(borne_plafond, e["scr"] - e["op_publie"]), "SCR - module publie"
    return borne_plafond, "SCR / 1,3 (plafond seul)"


def tau(rho, d):
    """Taux de reconnaissance sous agregation correlee, BSCR pris comme un bloc."""
    return (math.sqrt(1.0 + 2.0 * rho * d + d * d) - 1.0) / d


def main():
    ent = lire_65()
    ordre = ["Assureur non-vie A", "Assureur non-vie B", "Assureur vie C", "Assureur vie D"]

    titre("95 : l'agregation du besoin ORSA DORA au capital total")
    print("""
Question : le memoire chiffre un ecart de capital entre etats de conformite. Que devient
cet ecart une fois porte au capital TOTAL de l'entite ? Deux conventions sont comparees
sur les memes montants : l'additive, qui est celle de Solvabilite II pour le risque
operationnel, et un contrefactuel correle qui sert de mesure, jamais de proposition.

Tout est lu dans sorties_verif/65.txt, donc de chiffres SFCR publies. Les entites sont
designees par leur classe de taille : les identites restent dans le script 65.""")
    controle(ent)

    titre("1. Ce que le 65 donne, et les bornes du BSCR qui s'en deduisent")
    print("""L'identite SCR = BSCR + Adj + SCR_op avec Adj <= 0 donne BSCR >= SCR - SCR_op, et le
plafond SCR_op <= 0,3 BSCR donne BSCR >= SCR / 1,3. Le minorant retenu est le plus grand
des deux. Minorer le BSCR MAJORE le taux du contrefactuel : les taux qui suivent sont
donc des majorants, et l'ecart entre conventions est au moins celui qu'on lit.
""")
    print(f"{'entite':<22}{'besoin':>9}{'ecart':>9}{'SCR pub.':>10}{'op publie':>11}"
          f"{'BSCR >=':>10}  {'origine de la borne':<26}")
    print(f"{'':<22}{'M EUR':>9}{'M EUR':>9}{'M EUR':>10}{'M EUR':>11}{'M EUR':>10}")
    for a in ordre:
        e = ent[a]
        b, origine = bscr_min(e)
        e["bscr"] = b
        op = f"{e['op_publie']:.1f}" if "op_publie" in e else "non publie"
        print(f"{a:<22}{e['orsa']:>9.1f}{e['ecart']:>9.1f}{e['scr']:>10.0f}{op:>11}"
              f"{b:>10.1f}  {origine:<26}")

    titre("2. La convention reglementaire : additive, donc reconnue en entier")
    print("""Le risque operationnel entre hors de la matrice de correlation du BSCR. Un besoin de
capital de nature operationnelle, porte en ORSA, se transmet donc au capital total sans
aucun benefice de diversification : Delta SCR = D, taux de reconnaissance 1,000.

C'est vrai du NIVEAU comme de l'ECART entre etats. Or c'est l'ecart qui porte la these du
memoire, et le niveau est une borne superieure d'ordre de grandeur (chapitre 12). La suite
lit donc les deux, en donnant le premier rang a l'ecart.
""")
    print(f"{'entite':<22}{'ecart D':>9}{'Delta SCR additif':>20}{'en % du SCR publie':>20}")
    for a in ordre:
        e = ent[a]
        print(f"{a:<22}{e['ecart']:>9.1f}{e['ecart']:>20.1f}{100 * e['ecart'] / e['scr']:>19.2f}%")

    titre("3. Le contrefactuel : le meme montant agrege comme un module de plus")
    print("""tau(rho, d) = (sqrt(1 + 2 rho d + d^2) - 1) / d, avec d = D / BSCR. Le BSCR est pris
comme un bloc unique correle a rho, ce qui MINORE la reconnaissance : agreger module par
module donnerait davantage, la somme des modules majorant le BSCR. Le sens de la
conclusion n'en depend pas, seulement son amplitude.
""")
    # Les taux sont imprimes EN POURCENTAGE, parce que c'est sous cette forme que le
    # chapitre les cite : un taux publie a 29,4 % et imprime ici a 0,294 ne serait pas
    # retrouve par le harnais, qui compare des nombres et non des conventions d'ecriture.
    for objet, champ in (("L'ECART entre etats", "ecart"), ("le NIVEAU du besoin", "orsa")):
        print(f"\n  {objet} :")
        print(f"  {'entite':<22}{'d = D/BSCR':>12}" +
              "".join(f"{('tau rho=' + f'{r:.2f}'):>14}" for r in RHOS) +
              f"{'additif / rho=0,25':>20}")
        for a in ordre:
            e = ent[a]
            d = e[champ] / e["bscr"]
            taux = [tau(r, d) for r in RHOS]
            facteur = 1.0 / taux[1]
            print(f"  {a:<22}{d:>12.4f}" +
                  "".join(f"{100 * t:>13.1f}%" for t in taux) +
                  f"{facteur:>19.2f}x")
        t25 = [tau(0.25, ent[a][champ] / ent[a]["bscr"]) for a in ordre]
        print(f"  soit, a rho = 0,25, de {100 * min(t25):.1f} a {100 * max(t25):.1f} % reconnus contre "
              f"100 % en additif, un facteur {1 / max(t25):.1f} a {1 / min(t25):.1f}.")

    titre("4. La limite analytique, qui est le resultat a retenir")
    print("""Developpement de tau en d = 0 :

    sqrt(1 + 2 rho d + d^2) = 1 + rho d + (d^2 / 2)(1 - rho^2) + O(d^3)
    donc  tau(rho, d) = rho + (d / 2)(1 - rho^2) + O(d^2).

Un montant petit devant le BSCR est donc reconnu a hauteur de rho, et de rho seulement.
La convention additive le reconnait a 1. L'ecart entre les deux conventions n'est pas un
effet de taille : c'est 1 - rho, et il ne se referme qu'a correlation parfaite.

Controle numerique de ce developpement sur l'ecart de l'entite la plus grande :
""")
    e = ent["Assureur vie D"]
    d = e["ecart"] / e["bscr"]
    for r in RHOS[1:]:
        exact = tau(r, d)
        approx = r + 0.5 * d * (1 - r * r)
        print(f"  rho = {r:.2f}  d = {d:.5f}   tau exact = {exact:.5f}   "
              f"rho + (d/2)(1-rho^2) = {approx:.5f}   ecart = {abs(exact - approx):.6f}")

    titre("5. Le plafond de Formule Standard, et la marge qui reste avant saturation")
    print("""SCR_op = min(0,3 x BSCR ; Op). La charge forfaitaire ne peut donc pas depasser 0,3 BSCR,
quelle que soit l'exposition operationnelle reelle. La marge qui reste avant saturation
vaut 0,3 x BSCR - SCR_op publie, et elle se calcule sur les deux entites non-vie, seules a
publier leur module. Lecture de reference Adj = 0, POSEE ET DECLAREE : un Adj strictement
negatif releverait le BSCR, donc le plafond, donc la marge. L'etat S.25.01 de l'entite
permet de lever cette hypothese entite par entite, et c'est ainsi qu'il faut la citer.
""")
    print(f"{'entite':<22}{'BSCR ref.':>11}{'plafond':>10}{'op publie':>11}{'marge':>9}"
          f"{'ecart D':>9}{'besoin D':>10}  verdict")
    for a in ordre:
        e = ent[a]
        if "op_publie" not in e:
            print(f"{a:<22}{'module operationnel non publie : le plafond ne se teste pas':>60}")
            continue
        bscr0 = e["scr"] - e["op_publie"]
        plafond = PLAFOND * bscr0
        marge = plafond - e["op_publie"]
        verdict = "le besoin DEPASSE la marge" if e["orsa"] > marge else "le besoin tient dans la marge"
        print(f"{a:<22}{bscr0:>11.1f}{plafond:>10.1f}{e['op_publie']:>11.1f}{marge:>9.1f}"
              f"{e['ecart']:>9.1f}{e['orsa']:>10.1f}  {verdict}")
    ea = ent["Assureur non-vie A"]
    bscr0 = ea["scr"] - ea["op_publie"]
    marge = PLAFOND * bscr0 - ea["op_publie"]
    print(f"""
Sur l'assureur non-vie A, le besoin de {ea['orsa']:.1f} M EUR vaut {ea['orsa'] / marge:.2f} fois la marge de
{marge:.1f} M EUR qui reste sous le plafond. Meme un forfait qui repondrait a la conformite
saturerait donc avant d'avoir exprime ce besoin. L'ecart entre etats, lui, vaut
{ea['ecart']:.1f} M EUR, soit {ea['ecart'] / marge:.2f} fois cette marge.""")

    titre("VERDICT")
    e_min = min(ordre, key=lambda a: tau(0.25, ent[a]["ecart"] / ent[a]["bscr"]))
    e_max = max(ordre, key=lambda a: tau(0.25, ent[a]["ecart"] / ent[a]["bscr"]))
    t_min = tau(0.25, ent[e_min]["ecart"] / ent[e_min]["bscr"])
    t_max = tau(0.25, ent[e_max]["ecart"] / ent[e_max]["bscr"])
    print(f"""
1. Un euro d'ecart DORA coute un euro de capital total. Le risque operationnel entrant
   hors de la matrice de correlation, la convention reglementaire reconnait l'ecart en
   ENTIER, et cela vaut a toutes les tailles du panel.

2. Le meme montant agrege comme un module de plus ne serait reconnu qu'a hauteur de
   {100 * t_min:.1f} a {100 * t_max:.1f} % a rho = 0,25, soit un facteur {1 / t_max:.1f} a {1 / t_min:.1f} entre les deux
   conventions. Ce n'est pas un effet de taille : pour un montant petit devant le BSCR,
   le taux tend vers rho lui-meme.

3. Consequence pour l'entite, et c'est la lecture decisionnelle : un euro economise sur la
   non-conformite DORA vaut plus, en capital, qu'un euro economise sur un risque qui se
   diversifie. La remediation est donc sous-evaluee par une lecture qui la traiterait
   comme un module parmi d'autres.

4. Consequence sur la Formule Standard, qui s'ajoute a sa cecite deja etablie. Non
   seulement la charge forfaitaire ne repond pas a la conformite, mais elle est PLAFONNEE
   a 0,3 BSCR : sur l'assureur non-vie A, la marge restant sous ce plafond ne couvre meme
   pas le besoin calcule. Un forfait rendu sensible a la conformite ne suffirait donc pas.

5. Ce qui reste hors de portee sans l'etat S.25.01 complet : la valeur exacte du BSCR et
   de l'ajustement Adj, donc le passage des bornes aux valeurs. Les taux imprimes ici sont
   des majorants, ce qui suffit au verdict mais pas a une mesure.""")


if __name__ == "__main__":
    main()
