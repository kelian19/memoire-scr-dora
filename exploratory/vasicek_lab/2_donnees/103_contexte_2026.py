#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
103 : les deux sources de contexte 2026 du chapitre de generalites, transcrites et controlees.

POURQUOI CE SCRIPT EXISTE. Le chapitre de generalites cite deux sources de contexte, la
cartographie prospective 2026 de France Assureurs et le onzieme barometre CESIN. Leurs valeurs
sont transcrites dans config.py sous CARTO_FA_2026 et CESIN_2026, au meme statut que LUCY_2026 :
CITATION EXTERNE NON RECALCULABLE. Sans ce script, elles vivraient en dur dans un fichier LaTeX
et echapperaient au harnais, ce qui est la classe de defaut qui a produit les queues Bale et le
Hill a 1,42.

CE SCRIPT NE PRODUIT AUCUNE FIGURE, ET C'EST VOULU. Les figures de ces deux sources sont
EXTRAITES DES RAPPORTS et vivent dans memoire_cascade/figures_externes/, exception declaree du
projet, au meme titre que les trois figures LUCY autorisees par Hugo Rapior le 17 septembre.
Redessiner ces figures depuis la transcription a ete essaye puis ECARTE : cela aurait cree une
seconde convention d'affichage a cote de celle deja retenue pour LUCY, pour des sources de meme
statut. Une seule regle vaut donc pour les trois : figure extraite, valeurs transcrites et
controlees.

CE QUE LES CONTROLES FONT, ET CE QU'ILS NE FONT PAS. Ils portent sur la COHERENCE INTERNE de la
recopie, jamais sur la collecte des sources : le depot ne detient ni les reponses individuelles
de l'enquete, ni les scores elementaires de la cartographie. Confondre les deux reviendrait a
s'attribuer une verification qu'on n'a pas faite.

QUATRE CONTROLES SUR LES DEUX PREMIERES SOURCES, QUATRE AUTRES (5 a 8) SUR LE BILAN AMF DES
INCIDENTS MAJEURS DORA, troisieme source ajoutee le 28 septembre 2026 (section 4), au meme statut.
  (1) Les parts de vecteurs sont dans [0, 1] et leur somme DEPASSE 1, l'enquete etant a reponses
      multiples : une somme qui vaudrait 1 signalerait une recopie fautive, pas une reussite.
  (2) Chaque vecteur declare plus frequent chez les grandes entreprises l'est effectivement.
  (3) Les parts d'impact sur le business somment a 1 avec la part sans impact.
  (4) Les quatre cadrans de la cartographie somment a 1 a un point d'arrondi pres, la source
      publiant des entiers.

DEUX MISES EN GARDE A NE PAS PERDRE.
  - Le barometre est DECLARATIF : il mesure ce que les organisations constatent, non ce qui
    survient. Une baisse de la part d'entreprises attaquees n'est pas une baisse du risque.
  - La serie retrospective des scores est RELEVEE SUR UNE FIGURE, pas sur une table. Elle ne
    supporte pas une lecture au centieme et ne sert qu'a etablir une tendance. Meme regle que
    pour le S/P des ETI en 2020 du rapport LUCY : une etiquette imprimee se transcrit, une
    hauteur de barre ne se mesure pas.

Sortie : les grandeurs citees par le chapitre, sans figure.
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.config import CESIN_2026 as C                  # noqa: E402
from src.utils.config import CARTO_FA_2026 as F               # noqa: E402
from src.utils.config import AMF_DORA_2025 as A               # noqa: E402

WID = 82


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# =====================================================================================
titre("0. Statut des deux sources")
# =====================================================================================
print(f"  Cartographie : {F['source']}")
print(f"  Barometre    : {C['source']}")
print("  Statut des deux : CITATION EXTERNE NON RECALCULABLE. Les valeurs sont transcrites,")
print("  pas recalculees. Aucune n'entre dans une calibration et aucune ne porte un capital.")
print("  Leurs figures sont EXTRAITES des rapports et vivent dans figures_externes/.")

# =====================================================================================
titre("1. Quatre controles de fidelite de la recopie")
# =====================================================================================
ok = True

parts = list(C["vecteurs"].values())
somme_vect = sum(parts)
c1 = all(0.0 <= v <= 1.0 for v in parts) and somme_vect > 1.0
print(f"  (1) parts de vecteurs dans [0,1] et somme = {somme_vect:.2f} > 1 (reponses multiples)"
      f"   -> {'OK' if c1 else 'ECHEC'}")
ok &= c1

c2 = all(C["vecteurs_grandes_entreprises"][k] > C["vecteurs"][k]
         for k in C["vecteurs_grandes_entreprises"])
print(f"  (2) chaque vecteur 'grandes entreprises' depasse bien sa part globale"
      f"                 -> {'OK' if c2 else 'ECHEC'}")
ok &= c2

somme_impact = C["part_avec_impact_business"] + C["part_sans_impact_business"]
c3 = abs(somme_impact - 1.0) < 1e-9
print(f"  (3) impact + absence d'impact = {somme_impact:.2f}"
      f"                                          -> {'OK' if c3 else 'ECHEC'}")
ok &= c3

somme_cadrans = (F["part_cadran_sud_ouest"] + F["part_cadran_nord_est"]
                 + F["part_cadran_nord_ouest"] + F["part_cadran_sud_est"])
c4 = abs(somme_cadrans - 1.0) <= 0.01 + 1e-9
print(f"  (4) somme des quatre cadrans = {somme_cadrans:.2f}, arrondi de la source tolere"
      f"        -> {'OK' if c4 else 'ECHEC'}")
ok &= c4

print(f"\n  VERDICT : {'les quatre controles passent.' if ok else 'AU MOINS UN CONTROLE ECHOUE.'}")

# =====================================================================================
titre("2. Le barometre : ce que les organisations constatent")
# =====================================================================================
print(f"  Entreprises declarant au moins une attaque : {C['part_entreprises_attaquees']:.0%}"
      f"  (base {C['base_attaquees']} repondants attaques)")
print(f"  Vecteurs par entreprise attaquee          : {C['vecteurs_moyens']}"
      f"  contre {C['vecteurs_moyens_vague_precedente']} a la vague precedente,")
print(f"                                              et {C['vecteurs_moyens_grandes']}"
      f" chez les grandes entreprises")
print(f"  Part avec impact sur le business          : {C['part_avec_impact_business']:.0%}")
print(f"  Part sans impact sur le business          : {C['part_sans_impact_business']:.0%}")

print("\n  Les deux premiers vecteurs :")
for nom in ("hameçonnage", "exploitation d'une faille"):
    print(f"    {nom:<34} {C['vecteurs'][nom]:.0%}")

TIERS = "attaque indirecte par un tiers"
print(f"\n  LE VECTEUR QUI PORTE LA THESE : '{TIERS}'")
print(f"    part globale             : {C['vecteurs'][TIERS]:.0%}  (3e vecteur sur"
      f" {len(C['vecteurs'])})")
print(f"    part grandes entreprises : {C['vecteurs_grandes_entreprises'][TIERS]:.0%}")
print("    Plus l'entite est grande, plus son exposition passe par ses prestataires. C'est le")
print("    canal P4 du memoire, mesure de l'exterieur par une source sans lien avec lui.")

print("\n  Les autres vecteurs dont la part monte avec la taille, et ils relevent tous de P1 :")
for nom, v in sorted(C["vecteurs_grandes_entreprises"].items(), key=lambda kv: -kv[1]):
    if nom == TIERS:
        continue
    print(f"    {nom:<44} {C['vecteurs'][nom]:.0%} -> {v:.0%}")

print("\n  Les quatre premieres consequences :")
for nom, v in sorted(C["impacts"].items(), key=lambda kv: -kv[1])[:4]:
    print(f"    {nom:<34} {v:.0%}")
_SANC = "sanction d'une autorité"
print(f"    {_SANC:<34} {C['impacts'][_SANC]:.0%}   (le canal reglementaire est reel,")
print("                                              mais minoritaire dans le cout total)")

DECO = "déconnexion par les tiers"
print(f"\n  LA PROPAGATION EN SENS INVERSE : '{DECO}' figure a {C['impacts'][DECO]:.0%}"
      " des impacts.")
print("    L'entite n'est pas seulement atteinte PAR ses partenaires, elle est aussi coupee")
print("    PAR eux quand elle est atteinte. La dependance joue dans les deux sens, ce qu'une")
print("    dependance symetrique ne saurait pas distinguer.")

# =====================================================================================
titre("3. La cartographie : le risque sature au sommet et change de regime")
# =====================================================================================
i_max = F["score_cyber_serie"].index(max(F["score_cyber_serie"]))
print(f"  Score du risque de cyberattaques : {F['score_cyber_serie'][i_max]:.2f} en"
      f" {F['annees_serie'][i_max]}, maximum de la serie,")
print(f"                                     {F['score_cyber_serie'][-1]:.2f} en"
      f" {F['annees_serie'][-1]}, et premier rang depuis"
      f" {F['annees_consecutives_premier_rang']} exercices.")

print("\n  Classement 2026 :")
for i, (nom, sc) in enumerate(F["rang_2026"], 1):
    print(f"    {i}. {nom:<34} {sc:.1f}")
print("  Classement 2025, pour memoire :")
for i, (nom, sc) in enumerate(F["rang_2025"], 1):
    print(f"    {i}. {nom:<34} {sc:.1f}")

dx, dy = F["deplacements"]["Cyberattaques"]
print(f"\n  Deplacement 2026 du cyber : frequence {dx:+.2f}, severite {dy:+.2f}")
print("    Le risque ne monte plus en score : il se stabilise au sommet en changeant de")
print("    regime, moins frequent et plus grave.")
# LE SUPERLATIF SE CALCULE, IL NE S'ECRIT PAS. Ce bloc affirmait que la qualite des donnees est
# le risque qui monte le plus en severite. C'EST FAUX : l'environnement politique monte davantage
# (+0,32 contre +0,30). Le constat qui porte l'argument est plus etroit et plus juste : parmi les
# risques qui gagnent en severite TOUT EN RECULANT en frequence, donc dans le meme regime que le
# cyber, c'est la qualite des donnees qui monte le plus.
dep = F["deplacements"]
qd = dep["Qualité des données et conformité des processus IT"]
top_sev = max(dep, key=lambda k: dep[k][1])
regime = {k: v for k, v in dep.items() if v[0] < 0 and v[1] > 0}
top_regime = max(regime, key=lambda k: regime[k][1])
print(f"  Plus forte hausse de severite, tous risques : {top_sev.lower()}, "
      f"{dep[top_sev][1]:+.2f} point")
print(f"    (mais sa frequence monte aussi, {dep[top_sev][0]:+.2f} : ce n'est pas le regime du cyber)")
print(f"  Risques du MEME REGIME que le cyber (severite en hausse, frequence en baisse) : "
      f"{len(regime)}")
for k in sorted(regime, key=lambda k: -regime[k][1]):
    print(f"    {k[:52]:<52s} frequence {regime[k][0]:+.2f}  severite {regime[k][1]:+.2f}")
print(f"  Dans ce regime, la plus forte hausse est : {top_regime.lower()}, {qd[1]:+.2f} point.")
print("    La preoccupation de la profession se deplace de la menace vers sa GOUVERNANCE,")
print("    ce qui est l'objet meme de DORA.")
print(f"  Repartition des cadrans : sud-ouest {F['part_cadran_sud_ouest']:.0%},"
      f" nord-est {F['part_cadran_nord_est']:.0%},")
print(f"                            nord-ouest {F['part_cadran_nord_ouest']:.0%},"
      f" sud-est {F['part_cadran_sud_est']:.0%}")

# =====================================================================================
titre("4. Premier bilan des incidents majeurs DORA declares a l'AMF (exercice 2025)")
# =====================================================================================
# Troisieme source, ajoutee le 28 septembre 2026, meme statut que les deux autres. Les
# controles portent sur la coherence de la recopie : les comptes doivent se partitionner et
# redonner les pourcentages publies A L'UNITE, que la source arrondit.
print(f"  Source : {A['source']}")
print("  Statut : CITATION EXTERNE NON RECALCULABLE. Perimetre : societes de gestion (SGP).")

n = A["incidents_majeurs_sgp"]
ok_a = True
c5 = (A["incidents_cyber"] + A["incidents_operationnels"] == n
      and A["dont_exfiltration"] + A["dont_usurpation"] == A["incidents_cyber"]
      and A["dont_indisponibilite_si"] + A["dont_defaillance_sgp"] == A["incidents_operationnels"])
print(f"  (5) cyber + operationnels = {n}, et chaque sous-total se partitionne"
      f"                -> {'OK' if c5 else 'ECHEC'}")
ok_a &= c5

comptes = {"cyber": A["incidents_cyber"], "exfiltration": A["dont_exfiltration"],
           "usurpation": A["dont_usurpation"], "operationnels": A["incidents_operationnels"]}
c6 = all(round(100 * comptes[k] / n) == round(100 * A["parts_publiees"][k]) for k in comptes)
c6 &= round(100 * A["incidents_origine_tiers"] / n) == round(100 * A["part_origine_tiers_publiee"])
print(f"  (6) chaque part publiee se retrouve a l'unite depuis les comptes"
      f"                  -> {'OK' if c6 else 'ECHEC'}")
ok_a &= c6

somme_tiers = A["part_tiers_cyber_publiee"] + A["part_tiers_accidentel_publiee"]
c7 = abs(somme_tiers - A["part_origine_tiers_publiee"]) < 1e-9
print(f"  (7) tiers cyber + tiers accidentel = {somme_tiers:.2f}, egal a la part d'origine tierce"
      f"  -> {'OK' if c7 else 'ECHEC'}")
ok_a &= c7

c8 = n + A["incidents_majeurs_hors_sgp"] == A["incidents_majeurs_total_publie"]
print(f"  (8) SGP + hors SGP = {n + A['incidents_majeurs_hors_sgp']}, total publie par l'AMF"
      f"                   -> {'OK' if c8 else 'ECHEC'}")
ok_a &= c8
print(f"\n  VERDICT : {'les quatre controles passent.' if ok_a else 'AU MOINS UN CONTROLE ECHOUE.'}")

part_tiers = A["incidents_origine_tiers"] / n
taux_conf = n / A["notifications_initiales_sgp"]
print(f"\n  Notifications initiales : {A['notifications_initiales_sgp']}, confirmees majeures : {n}"
      f"  (taux {taux_conf:.0%})")
print(f"  Origine prestataire tiers : {A['incidents_origine_tiers']} sur {n}, soit {part_tiers:.1%}")
print(f"    dont cyberattaque {A['part_tiers_cyber_publiee']:.0%},"
      f" defaillance accidentelle {A['part_tiers_accidentel_publiee']:.0%}")
print(f"  Nature : {A['incidents_cyber']} cyberattaques (dont {A['dont_exfiltration']} exfiltrations,"
      f" {A['dont_usurpation']} usurpations), {A['incidents_operationnels']} operationnels")
print("  Lecture : l'ecart entre notifications et incidents confirmes est celui d'un seuil de")
print("  materialite ; un incident est compte PAR SGP declarante, et la compromission d'un meme")
print("  prestataire en touche plusieurs. La part tierce mesure une ORIGINE, pas une probabilite")
print("  de transfert entre domaines de controle : elle ne calibre aucun parametre du modele.")

# =====================================================================================
titre("GRANDEURS CITEES (sans separateur de milliers, pour le harnais)")
# =====================================================================================
print(f"  part entreprises attaquees        : {C['part_entreprises_attaquees'] * 100:.0f}")
print(f"  base repondants attaques          : {C['base_attaquees']}")
print(f"  vecteurs moyens                   : {C['vecteurs_moyens']}")
print(f"  vecteurs moyens vague precedente  : {C['vecteurs_moyens_vague_precedente']}")
print(f"  vecteurs moyens grandes           : {C['vecteurs_moyens_grandes']}")
print(f"  hameconnage                       : {C['vecteurs']['hameçonnage'] * 100:.0f}")
print(f"  exploitation d une faille         : "
      f"{C['vecteurs']['exploitation d\'une faille'] * 100:.0f}")
print(f"  attaque indirecte par un tiers    : {C['vecteurs'][TIERS] * 100:.0f}")
print(f"  idem grandes entreprises          : "
      f"{C['vecteurs_grandes_entreprises'][TIERS] * 100:.0f}")
print(f"  fuite par erreur, grandes         : "
      f"{C['vecteurs_grandes_entreprises']['fuite par erreur humaine ou de configuration'] * 100:.0f}")
print(f"  composant malveillant, grandes    : "
      f"{C['vecteurs_grandes_entreprises']['activation d\'un composant malveillant'] * 100:.0f}")
print(f"  acces legitime, grandes           : "
      f"{C['vecteurs_grandes_entreprises']['exfiltration par un accès légitime'] * 100:.0f}")
print(f"  part avec impact business         : {C['part_avec_impact_business'] * 100:.0f}")
print(f"  part sans impact business         : {C['part_sans_impact_business'] * 100:.0f}")
print(f"  perturbation de la production     : "
      f"{C['impacts']['perturbation de la production'] * 100:.0f}")
print(f"  perte d image                     : {C['impacts']['perte d\'image'] * 100:.0f}")
print(f"  compromission de savoir-faire     : "
      f"{C['impacts']['compromission de savoir-faire'] * 100:.0f}")
print(f"  perte de chiffre d affaires       : "
      f"{C['impacts']['perte de chiffre d\'affaires'] * 100:.0f}")
print(f"  sanction d une autorite           : "
      f"{C['impacts']['sanction d\'une autorité'] * 100:.0f}")
print(f"  deconnexion par les tiers         : {C['impacts'][DECO] * 100:.0f}")
print(f"  score cyber maximum               : {F['score_cyber_serie'][i_max]:.2f}")
print(f"  annee du maximum                  : {F['annees_serie'][i_max]}")
print(f"  score cyber 2026                  : {F['score_cyber_serie'][-1]:.2f}")
print(f"  annees consecutives premier rang  : {F['annees_consecutives_premier_rang']}")
print(f"  score environnement economique    : {F['rang_2026'][1][1]:.1f}")
print(f"  score dereglement climatique      : {F['rang_2026'][2][1]:.1f}")
print(f"  deplacement cyber frequence       : {dx:.2f}")
print(f"  deplacement cyber severite        : {dy:.2f}")
print(f"  deplacement qualite donnees sev   : {qd[1]:.2f}")
print(f"  deplacement qualite donnees freq  : {qd[0]:.2f}")
print(f"  deplacement politique severite    : {dep['Environnement politique'][1]:.2f}")
print(f"  deplacement politique frequence   : {dep['Environnement politique'][0]:.2f}")
print(f"  nombre de vecteurs recenses       : {len(C['vecteurs'])}")
print(f"  AMF notifications initiales SGP   : {A['notifications_initiales_sgp']}")
print(f"  AMF incidents majeurs SGP         : {A['incidents_majeurs_sgp']}")
print(f"  AMF incidents origine tiers       : {A['incidents_origine_tiers']}")
print(f"  AMF part origine tiers            : {A['part_origine_tiers_publiee'] * 100:.0f}")
print(f"  AMF part tiers cyber              : {A['part_tiers_cyber_publiee'] * 100:.0f}")
print(f"  AMF part tiers accidentel         : {A['part_tiers_accidentel_publiee'] * 100:.0f}")
print(f"  AMF incidents cyber               : {A['incidents_cyber']}")
print(f"  AMF dont exfiltration             : {A['dont_exfiltration']}")
print(f"  AMF incidents operationnels       : {A['incidents_operationnels']}")
print(f"  AMF part SGP non conformes        : {A['part_sgp_dispositif_non_conforme'] * 100:.0f}")
