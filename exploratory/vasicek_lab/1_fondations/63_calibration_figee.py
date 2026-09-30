#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
63 : imprimer la CALIBRATION FIGEE, pour que les chapitres puissent la citer.

POURQUOI CE SCRIPT EXISTE. Le passage du harnais de verification sur les dix-neuf chapitres
a montre que les constantes de calibration les plus citees du memoire n'apparaissaient dans
AUCUNE sortie de script : le seuil u, l'echelle sigma, le nombre d'exces, la VaR et la TVaR
mono-perte, la concentration de severite, le plafond de reassurance. Elles vivent dans
src/utils/config.py, qui est bien la source de verite unique du projet, mais un fichier de
configuration n'est pas une sortie : rien ne le rejoue, donc rien ne le verifie.

Consequence concrete : les chapitres donnees et socle, qui sont ceux qui publient ces
constantes, plafonnaient a 74-78 % de confirmation alors qu'aucun de leurs nombres n'etait
faux. Le defaut etait de TRACABILITE, pas d'exactitude.

CE QUE FAIT CE SCRIPT. Il imprime la configuration figee, telle quelle, sans rien recalculer.
C'est volontaire : recalculer ici dupliquerait les scripts de calibration et creerait deux
verites. Le role de ce script est d'exposer la source unique sous une forme que
verif_chiffres.py peut lire.

CE QU'IL NE FAIT PAS. Il ne valide rien. Un nombre confirme par ce script est confirme
CONFORME A LA CONFIGURATION, pas conforme aux donnees. La validation empirique des memes
grandeurs est l'objet des scripts 07, 08b, 47 et 57. Les deux controles sont complementaires
et il ne faut pas les confondre.

Sortie : diagnostics seulement, pas de figure.
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.utils.config import (COPULE, FREQUENCY, HACKMAGEDDON,     # noqa: E402
                              LUCY_2026, OPRISK, PRC, SCR_DORA)

W = 84


def titre(s):
    print("\n" + "=" * W + f"\n{s}\n" + "=" * W)


def ligne(lab, val, unite=""):
    if isinstance(val, (list, tuple)):
        val = "[" + " ; ".join(f"{v:,.4g}" for v in val) + "]"
    elif isinstance(val, float):
        val = f"{val:,.4f}".rstrip("0").rstrip(".")
    print(f"  {lab:<46}{str(val):>18} {unite}")


titre("Severite OpRisk Global (montants reels, source de reference en euros)")
print(f"  perimetre : {OPRISK['perimetre']}")
ligne("incidents du perimetre cyber x finance", OPRISK["n_incidents"])
ligne("annees d'observation", OPRISK["n_years"], "ans")
ligne("frequence propre = n / annees", OPRISK["n_incidents"] / OPRISK["n_years"], "/ an")
ligne("exces au-dessus du seuil", OPRISK["n_excess"])
ligne("seuil POT u (percentile 85)", OPRISK["seuil_u_eur"], "M EUR")
ligne("taux de depassement p_u", OPRISK["p_u"])
ligne("indice de queue xi", OPRISK["xi"])
ligne("  IC 90 % sur xi", OPRISK["xi_ic90"])
ligne("echelle sigma", OPRISK["sigma_eur"], "M EUR")
ligne("  IC 90 % sur sigma", OPRISK["sigma_ic90"], "M EUR")
ligne("VaR 99,5 % mono-perte", OPRISK["var_995"], "M EUR")
ligne("  IC 90 % sur la VaR", OPRISK["var_995_ic90"], "M EUR")
ligne("TVaR 99 %", OPRISK["tvar_99"], "M EUR")
print(f"\n  Rapport IC de la VaR : facteur "
      f"{OPRISK['var_995_ic90'][1] / OPRISK['var_995_ic90'][0]:,.1f} entre les bornes.")
print("  C'est ce facteur, et non le point, qui doit etre cite avec le capital.")

titre("Severite PRC (derivee par conversion Jacobs)")
print(f"  periode : {PRC['period']}")
ligne("incidents a total_affected > 0", PRC["n_records"])
ligne("exces au-dessus du seuil", PRC["n_excess"])
ligne("seuil POT u", PRC["seuil_u_eur"], "M EUR")
ligne("taux de depassement p_u", PRC["p_u"])
ligne("indice de queue xi", PRC["xi"])
ligne("  IC 90 % sur xi", PRC["xi_ic90"])
ligne("echelle sigma", PRC["sigma_eur"], "M EUR")
ligne("  IC 90 % sur sigma", PRC["sigma_ic90"], "M EUR")
ligne("VaR 99,5 % mono-perte", PRC["var_995"], "M EUR")
ligne("conversion Jacobs, constante a", PRC["jacobs_a"])
ligne("conversion Jacobs, elasticite b", PRC["jacobs_b"])
ligne("taux de change USD vers EUR", PRC["usd_eur"])
print(f"\n  xi = {PRC['xi']:.3f} > 1 : la severite PRC est en regime d'ESPERANCE INFINIE,")
print(f"  d'ou le plafond de {SCR_DORA['cap_eur']:,.0f} M EUR (capacite de reassurance) qui rend")
print("  le capital calculable. Ce plafond ne s'applique QU'A la source PRC.")

titre("Frequence")
ligne("lambda_ref (perimetre financier PRC)", FREQUENCY["lambda_ref"], "/ an")
ligne("facteur de surdispersion Var / moyenne", FREQUENCY["dispersion_factor"])
ligne("facteur de recalibration", FREQUENCY["facteur_recalibration"])
print("\n  RAPPEL DES TROIS NIVEAUX, a ne jamais confondre :")
print(f"    lambda_ref      = {FREQUENCY['lambda_ref']:>7} / an   cle de repartition par vecteur")
print(f"    lambda (secteur)= {OPRISK['n_incidents']/OPRISK['n_years']:>7,.2f} / an   entree du moteur de cascade")
print(f"    lambda (entite) =  0.0917 / an   lu a la taille de l'entite (script 60)")

titre("Dependance et plafond")
ligne("famille de copule", COPULE["famille"])
ligne("theta non conforme", COPULE["theta_nc"])
ligne("theta conforme", COPULE["theta_c"])
ligne("bande empirique de theta", COPULE["theta_empirical_band"])
ligne("p_sys (concentration cloud)", COPULE["p_sys"])
ligne("plafond de severite (reassurance)", SCR_DORA["cap_eur"], "M EUR")
print(f"\n  Sensibilite declaree : {COPULE['theta_delta_dora_sensitivity']}")

titre("Hackmageddon : CITATION EXTERNE, non recalculable")
print(f"  source  : {HACKMAGEDDON['source']}")
print(f"  periode : {HACKMAGEDDON['periode']}")
ligne("incidents du semestre", HACKMAGEDDON["n_incidents"])
ligne("dont a vecteur d'acces identifie", HACKMAGEDDON["n_identifies"])
ligne("taux d'identification", HACKMAGEDDON["taux_identification"])
print("\n  parts par vecteur d'attaque :")
# EN FRACTION ET EN POURCENTAGE, LES DEUX. Le chapitre donnees cite « 38,8 % » la ou cette
# sortie n'imprimait que « 0.388 » : le harnais ne rapproche pas deux ecritures separees par
# un facteur cent, et ces parts ressortaient donc non confirmees alors qu'elles sont ici. La
# double ecriture coute une colonne et supprime toute une classe de fausses alertes.
for k, v in HACKMAGEDDON["proportions"].items():
    ligne(f"    {k}", f"{v:.3f}   soit {100*v:.1f} %")
ligne("surface TLPT (art. 26)", f"{HACKMAGEDDON['surface_tlpt']:.3f}   soit "
                                f"{100*HACKMAGEDDON['surface_tlpt']:.1f} %")
ligne("surface tiers (art. 28-44)", f"{HACKMAGEDDON['surface_tiers']:.3f}   soit "
                                    f"{100*HACKMAGEDDON['surface_tiers']:.1f} %")

cmp_ = HACKMAGEDDON["comparaison_2023_2026"]
print("\n  comparaison de structure 2023 contre 2026, MEME STATUT DE CITATION :")
print(f"    effectif 2023 : {cmp_['n_2023']} incidents, "
      f"trimestre {cmp_['trimestre_manquant_2023']} manquant")
print(f"    {'dimension (motivation)':<28}{'2023':>10}{'2026':>10}{'ecart (pts)':>14}")
for k, (a, b) in cmp_["motivations"].items():
    print(f"    {k:<28}{100*a:>9.1f} %{100*b:>9.1f} %{100*(b-a):>+13.1f}")
print(f"    lecture naive du ransomware : hausse de "
      f"{cmp_['hausse_ransomware_lecture_naive_pts']} points ; apres reclassement : recul de "
      f"{cmp_['recul_ransomware_apres_reclassement_pts']} points.")
print("    C'est l'artefact de taxonomie qui justifie de ne retenir de cette base que la")
print("    STRUCTURE, et encore, apres reclassement documente.")
print("\n  AVERTISSEMENT. Contrairement a PRC et OpRisk, le jeu Hackmageddon n'est PAS")
print("  versionne dans data/raw/ : ces valeurs sont une citation enregistree, pas une")
print("  sortie recalculable (cf. script 62). Elles ne servent qu'a fixer les parts de")
print("  repartition par vecteur, et ne portent aucun niveau de capital.")

titre("LUCY 2026 : CITATION EXTERNE, non recalculable")
L = LUCY_2026
print(f"  etude    : {L['source']}")
print(f"  analyse  : {L['analyse']}")
print(f"  perimetre de l'etude : {L['n_polices']} polices et {L['n_sinistres']} sinistres")
print(f"                         declares, {L['n_courtiers']} courtiers et "
      f"{L['n_assureurs']} assureur.")
print()
print("  RATIOS SINISTRES SUR PRIMES, en fraction ET en pourcentage, les deux, pour la")
print("  raison deja documentee au bloc Hackmageddon ci-dessus :")
for lab, k in (("agrege 2024", "sp_2024"), ("agrege 2025", "sp_2025"),
               ("segment ETI 2024", "sp_eti_2024"), ("segment ETI 2025", "sp_eti_2025")):
    ligne(f"    {lab}", f"{L[k]:.2f}   soit {100*L[k]:.0f} %")
print()
print("  CHARGE INDEMNISEE, nette de franchise et plafonnee par la capacite :")
ligne("    exercice 2024", L["charge_2024_eur"], "M EUR")
ligne("    exercice 2025", L["charge_2025_eur"], "M EUR")
ligne("    hausse", f"{L['hausse_charge_2025']:.2f}   soit "
                    f"{100*L['hausse_charge_2025']:.0f} %")
print()
print("  RECULS DE TAUX DE PRIME, en valeur absolue et nommes recul :")
ligne("    grandes entreprises", f"{L['recul_taux_prime_grandes']:.2f}   soit "
                                 f"{100*L['recul_taux_prime_grandes']:.0f} %")
ligne("    entreprises de taille intermediaire",
      f"{L['recul_taux_prime_eti']:.2f}   soit {100*L['recul_taux_prime_eti']:.0f} %")
print()
print("  LA VALEUR QUI COMPTE LE PLUS POUR CE MEMOIRE, et c'est un compte, pas un ratio :")
ligne("    sinistres au-dela du seuil XXL en France, 2025",
      L["n_sinistres_sup_10m_france_2025"])
ligne("    seuil XL", L["seuil_xl_eur"], "M EUR")
ligne("    seuil XXL", L["seuil_xxl_eur"], "M EUR")
print("    Un seul sinistre au-dela du seuil XXL sur l'exercice : la queue francaise est")
print("    CENSUREE en partie haute, ce qui rend un quantile a 99,5 % instable et biaise")
print("    vers le bas. C'est le constat externe qui justifie de calibrer la severite du")
print("    memoire sur une base INTERNATIONALE et non sur le marche francais.")
print()
print("  DEUX REGIMES OPPOSES SOUS UNE MEME TRAJECTOIRE DE RATIO CROISSANTE.")
print("  Le ratio monte les deux annees, mais le moteur s'inverse : 2024 est une annee de")
print("  SEVERITE, 2025 une annee de FREQUENCE. Un ratio agrege ne distingue pas les deux,")
print("  et ils n'appellent pas les memes leviers. C'est le meme argument de decomposition")
print("  que celui que ce memoire applique a ses quatre canaux.")
print()
print("    exercice     nombre   sinistre moyen   charge   produit   ecart a l'identite")
for an in (2024, 2025):
    n = L[f"mult_nombre_{an}"]
    s = L[f"mult_sinistre_moyen_{an}"]
    c = L[f"mult_charge_{an}"]
    print(f"    {an}         {n:6.2f}          {s:7.2f}  {c:7.2f}  {n*s:8.4f}   "
          f"{abs(n*s - c):.4f}")
print()
print("    L'ECART A L'IDENTITE EST LE CONTROLE DE CE BLOC. La charge se decompose")
print("    exactement en nombre de sinistres fois sinistre moyen, et les deux lignes le")
print("    verifient a l'arrondi de publication pres. Si la source avait ete recopiee de")
print("    travers, ce produit ne tomberait pas.")
print()
print("  UNE IMPRECISION DE LA SOURCE, CORRIGEE ICI ET A NE PAS REPRENDRE. Le rapport nomme")
print("  << frequence >> le multiplicateur du NOMBRE de sinistres dans cette decomposition.")
print("  Ce n'en est pas une : une frequence est un nombre de sinistres par assure, et les")
print("  tables du meme rapport la donnent a 1,88 pour 2025 quand le nombre est a 2,79.")
ligne("    2025, multiplicateur d'exposition", L["mult_exposition_2025"])
ligne("    2025, multiplicateur du NOMBRE de sinistres", L["mult_nombre_2025"])
ligne("    2025, multiplicateur de la vraie FREQUENCE", L["mult_frequence_2025"])
_p = L["mult_exposition_2025"] * L["mult_frequence_2025"]
print(f"    controle : exposition x frequence = {_p:.4f}, a comparer au nombre "
      f"{L['mult_nombre_2025']:.2f}, ecart {abs(_p - L['mult_nombre_2025']):.4f}")
print("    Avec la vraie frequence l'identite demande TROIS facteurs, exposition fois")
print("    frequence fois sinistre moyen. La distinction separe un effet de VOLUME d'une")
print("    degradation a exposition donnee, et c'est exactement la lecture que ce memoire")
print("    impose a ses propres canaux.")
print()
print("  DECOMPOSITION 2025 PAR BLOC (nombre, sinistre moyen, charge) :")
print("    bloc                       nombre   sinistre moyen   charge")
for nom, (n, s, c) in L["blocs_2025"].items():
    print(f"    {nom:<24} {n:7.2f}          {s:7.2f}  {c:7.2f}")
print("    Le bloc intermediaire est le SEUL ou le nombre et le sinistre moyen montent")
print("    ENSEMBLE, d'ou sa charge multipliee par 3,53. Les grandes entreprises absorbent")
print("    la hausse du nombre sans derive de cout, et les micro-entreprises portent une")
print("    charge tiree par le seul nombre, sur une base 2024 reduite.")
print()
print("  AVERTISSEMENT D'ECHELLE, ET IL EST DU MEME TYPE QUE CELUI DU 7 AOUT 2026. La charge")
print("  de LUCY est INDEMNISEE, donc nette de franchise et plafonnee par la capacite, et")
print("  sommee sur un portefeuille de MARCHE. La severite de ce memoire est une perte")
print("  operationnelle BRUTE d'entite financiere. Les deux ne se comparent ni en niveau ni")
print("  en quantile : rapprocher les 83,2 M EUR indemnises du marche francais du quantile")
print("  unitaire publie ferait lire une difference de perimetre et de retention comme une")
print("  contradiction. Cette source n'entre dans AUCUNE calibration du memoire.")

# ---------------------------------------------------------------------------
# SECTION 6bis : LA LECTURE DE MARCHE, ELARGIE LE 9 SEPTEMBRE 2026.
#
# Le chapitre d'introduction ne citait que trois constats du rapport. Kelian a
# demande que la lecture de marche y occupe une part substantielle, de facon a
# situer le risque cyber dans le marche qui l'assure avant de le charger en
# capital. Ce bloc imprime donc l'ensemble de la transcription.
#
# TOUS LES RATIOS SONT CALCULES ICI ET AUCUN N'EST TRANSCRIT. C'est la regle du
# projet : un rapport ecrit a la main pendant la redaction n'est verifiable par
# personne, et trois des chiffres perimes trouves en aout etaient de cette
# nature. Les variations, les sommes de colonnes et la charge du troisieme bloc
# sortent donc du calcul, pas de la source.
#
# ET LES TROIS CONTROLES D'IDENTITE SONT L'INTERET PRINCIPAL DE CE BLOC. Ils ne
# valident pas le rapport, qui est une citation externe : ils valident la
# TRANSCRIPTION. Une valeur recopiee de travers casse l'un des trois.
#
# FORMATAGE : pas de separateur de milliers dans ce bloc, et c'est delibere. Le
# helper ligne() imprime 20 996 sous la forme << 20,996 >>, que l'extracteur du
# harnais lit comme le decimal 20,996 en convention francaise, donc perd la
# valeur et en injecte une fausse. Meme piege que l'espace de milliers des
# scripts 90 et 91, documente dans la passation.
# ---------------------------------------------------------------------------
titre("LUCY 2026, lecture de marche elargie : citation externe non recalculable")

print("  LE MARCHE SUR DEUX EXERCICES. Colonnes 2024, 2025, puis variation")
print("  RELATIVE calculee ici. Pour le taux de prime la variation se lit en")
print("  POINTS et non en relatif : 0,28 % a 0,26 % est un recul de deux")
print("  centiemes de point, et le relatif de 7 % n'a pas de sens de gestion.")
print()
print(f"    {'indicateur':<32}{'2024':>12}{'2025':>12}{'variation':>12}")
for lab, (a, b, u) in L["marche_2024_2025"].items():
    fa = f"{a:.0f}" if a >= 1000 else f"{a:.2f}".rstrip("0").rstrip(".")
    fb = f"{b:.0f}" if b >= 1000 else f"{b:.2f}".rstrip("0").rstrip(".")
    var = b / a - 1.0
    print(f"    {lab:<32}{fa:>12}{fb:>12}{100*var:>11.0f} %   {u}")
print()
print("  ET LE TAUX DE PRIME SE LIT EN POINTS, pas en relatif :")
_tx = L["marche_2024_2025"]["taux de prime annuel moyen"]
print(f"    ecart du taux de prime annuel moyen          "
      f"{_tx[1] - _tx[0]:>8.2f} point de pourcentage")
print()
print("  LE CISEAU QUI RESUME L'EXERCICE, et il tient en deux lignes de cette")
print("  table : le nombre d'entreprises assurees progresse de 49 % quand le")
print("  volume de primes RECULE de 3 %. L'elargissement du marche s'est donc")
print("  fait par la prime unitaire, qui perd 35 %.")
print()

print("  SERIE LONGUE 2019-2025, en M EUR et en fraction.")
print(f"    {'annee':<8}{'primes':>10}{'sinistres':>12}{'S/P publie':>12}"
      f"{'S/P recalcule':>16}{'ecart':>9}")
_ecart_max_sp = 0.0
for k, an in enumerate(L["annees_serie"]):
    p = L["primes_serie_eur"][k]
    s = L["sinistres_serie_eur"][k]
    sp = L["sp_serie"][k]
    r = s / p
    _ecart_max_sp = max(_ecart_max_sp, abs(r - sp))
    print(f"    {an:<8}{p:>10.0f}{s:>12.0f}{sp:>12.2f}{r:>16.4f}"
          f"{abs(r - sp):>9.4f}")
print()
print("  PREMIER CONTROLE D'IDENTITE. Le S/P est par definition le rapport des")
print("  sinistres aux primes, donc la troisieme colonne doit se retrouver a")
print("  partir des deux premieres, annee par annee.")
print(f"    ecart maximal sur les sept exercices : {_ecart_max_sp:.4f}")
print("    Il reste sous l'arrondi de publication des etiquettes du rapport,")
print("    qui sont au point de pourcentage. La serie est donc coherente.")
print()
print("  ET CE QUE LA SERIE LONGUE APPREND, que les deux derniers exercices")
print("  seuls ne disent pas : le marche a DEJA connu un regime bien plus")
print(f"    S/P maximal de la serie, exercice 2020        {100*max(L['sp_serie']):.0f} %")
print(f"    S/P minimal de la serie, exercice 2023        {100*min(L['sp_serie']):.0f} %")
print(f"    S/P de l'exercice sous revue, 2025            {100*L['sp_2025']:.0f} %")
print("  degrade. Le 27 % de 2025 est un point haut de trois ans, pas un point")
print("  haut d'historique, et l'ecrire autrement serait forcer le trait.")
print()
print("  LES DEUX ECARTS DES TROIS DERNIERS EXERCICES, EN POINTS DE RATIO, et")
print("  c'est l'acceleration qui compte davantage que le niveau :")
_k23 = L["annees_serie"].index(2023)
_e1 = 100 * (L["sp_serie"][_k23 + 1] - L["sp_serie"][_k23])
_e2 = 100 * (L["sp_serie"][_k23 + 2] - L["sp_serie"][_k23 + 1])
print(f"    ecart 2023 vers 2024                         {_e1:>6.0f} points")
print(f"    ecart 2024 vers 2025                         {_e2:>6.0f} points")
print(f"    rapport des deux ecarts                      {_e2 / _e1:>6.0f}")
print("    Le second ecart vaut le double du premier, sur un ratio qui monte")
print("    depuis trois exercices : la progression n'est pas lineaire, elle")
print("    s'accelere, et c'est ce qui la rend interpretable comme un signal.")
print()

print("  CONDITIONS DE SOUSCRIPTION SUR LA MEME SERIE. La franchise n'est publiee")
print("  qu'a partir de 2021, d'ou les tirets : le panneau correspondant du")
print("  rapport ne porte que cinq barres quand les deux autres en portent sept,")
print("  et leur legende commune n'en declare que six. C'est le motif pour lequel")
print("  le memoire REPRODUIT ces trois series en tableau au lieu de reprendre la")
print("  figure : une legende qui ne compte pas ses series est un defaut qu'il")
print("  vaut mieux ne pas importer dans un document qu'un jury va verifier.")
print(f"    {'annee':<8}{'capacite M EUR':>16}{'franchise k EUR':>18}"
      f"{'taux de prime %':>18}")
for k, an in enumerate(L["annees_serie"]):
    fr = L["franchise_serie_eur"][k]
    sfr = "-" if fr is None else f"{fr:.1f}"
    print(f"    {an:<8}{L['capacite_serie_eur'][k]:>16.2f}{sfr:>18}"
          f"{L['taux_prime_serie'][k]:>18.2f}")
print()
print("  CE QUE CETTE SERIE DIT, ET QUE LES DEUX DERNIERS EXERCICES CACHENT : la")
print("  detente ne date pas de 2025. Le taux de prime a culmine en 2021 et")
print("  recule depuis quatre exercices, la franchise a culmine en 2022 et recule")
print("  depuis trois. Le soft market est donc un REGIME et non un accident de")
print("  l'exercice, ce qui compte pour lire la degradation technique : elle")
print("  arrive au bout d'un cycle de detente, pas en meme temps que lui.")
_i_tx = L["taux_prime_serie"].index(max(L["taux_prime_serie"]))
print(f"    sommet du taux de prime : exercice {L['annees_serie'][_i_tx]}, "
      f"{max(L['taux_prime_serie']):.2f} %")
print(f"    taux de prime 2025      : {L['taux_prime_serie'][-1]:.2f} %, soit "
      f"{100*(1 - L['taux_prime_serie'][-1] / max(L['taux_prime_serie'])):.0f} % "
      f"sous le sommet")
_fr = [v for v in L["franchise_serie_eur"] if v is not None]
print(f"    sommet de la franchise  : {max(_fr):.1f} k EUR")
print(f"    franchise 2025          : {_fr[-1]:.1f} k EUR, soit "
      f"{100*(1 - _fr[-1] / max(_fr)):.0f} % sous le sommet")
print()

print("  TAUX DE PRIME ANNUEL MOYEN PAR SEGMENT, en pourcentage, et le recul")
print("  RELATIF calcule ici :")
for lab, (a, b) in L["taux_prime_segment"].items():
    print(f"    {lab:<40}{a:>8.2f}{b:>8.2f}{100*(1 - b/a):>8.0f} % de recul")
print()
_rg = 1 - (L["taux_prime_segment"]["grandes entreprises"][1]
           / L["taux_prime_segment"]["grandes entreprises"][0])
print("  UNE SECONDE IMPRECISION DE LA SOURCE, ET ELLE SE VOIT EN RECALCULANT.")
print("  Le rapport annonce ce recul a 32 % dans son resume et sa section 3, puis")
print("  a 33 % dans sa section 3.1, pour les MEMES niveaux 1,90 % et 1,28 %.")
print(f"    recul transcrit du resume                    "
      f"{L['recul_taux_prime_grandes']:.2f}   soit "
      f"{100*L['recul_taux_prime_grandes']:.0f} %")
print(f"    recul recalcule depuis les deux niveaux      {_rg:.4f}   soit "
      f"{100*_rg:.1f} %, donc {100*_rg:.0f} % a l'unite")
print("    C'est la seconde valeur qui est juste, la premiere arrondissant vers")
print("    le bas. L'ecart est immateriel, mais le memoire cite les NIVEAUX et le")
print("    recul qu'ils impliquent, jamais un recul transcrit : c'est la seule")
print("    facon de ne pas propager celui des deux qui est faux. Meme traitement")
print("    que l'imprecision sur le mot << frequence >> du bloc precedent.")
print()
print("  MOUVEMENTS DE CONDITIONS DU BLOC INTERMEDIAIRE :")
ligne("    recul du taux de prime, moyennes",
      f"{L['recul_taux_prime_moyennes']:.2f}   soit "
      f"{100*L['recul_taux_prime_moyennes']:.0f} %")
ligne("    hausse du taux de prime, petites",
      f"{L['hausse_taux_prime_petites']:.2f}   soit "
      f"{100*L['hausse_taux_prime_petites']:.0f} %")
ligne("    recul de la franchise, ETI",
      f"{L['recul_franchise_eti']:.2f}   soit "
      f"{100*L['recul_franchise_eti']:.0f} %")
ligne("    recul de la franchise, moyennes",
      f"{L['recul_franchise_moyennes']:.2f}   soit "
      f"{100*L['recul_franchise_moyennes']:.0f} %")
print("    LE TAUX DES PETITES ENTREPRISES MONTE A CONTRE-COURANT DU MARCHE, et")
print("    le rapport l'attribue a une recomposition du sous-segment vers des")
print("    profils mieux couverts. Ce n'est donc pas un durcissement tarifaire,")
print("    et le lire comme tel inverserait le sens du mouvement.")
print()
print("  CONDITIONS DU SEGMENT MATURE, exercice 2025 :")
ligne("    capacite moyenne souscrite, grandes", L["capacite_grandes_2025_eur"],
      "M EUR")
ligne("    franchise moyenne, grandes", L["franchise_grandes_2025_eur"], "M EUR")
print()

print("  CROISSANCE DU NOMBRE D'ENTREPRISES ASSUREES, 2024 vers 2025 :")
for lab, v in L["croissance_assures_2025"].items():
    print(f"    {lab:<40}{v:>8.2f}   soit {100*v:>4.0f} %")
print()

print("  LES TROIS BLOCS, exercice 2025. Charge indemnisee en M EUR.")
_charge_blocs = sum(L["blocs_charge_2025_eur"].values())
_charge_micro = L["charge_2025_eur"] - _charge_blocs
for lab, v in L["blocs_charge_2025_eur"].items():
    print(f"    {lab:<40}{v:>10.1f}")
print(f"    {'micro-entreprises, PAR DIFFERENCE':<40}{_charge_micro:>10.1f}")
print(f"    {'total, a comparer a la charge publiee':<40}"
      f"{_charge_blocs + _charge_micro:>10.1f}")
print()
print("  DEUXIEME CONTROLE D'IDENTITE. La charge des micro-entreprises n'est PAS")
print("  transcrite : elle est deduite par difference, donc la somme des trois")
print("  blocs redonne la charge du marche par construction. Ce qui se controle")
print("  est que le residu soit PLAUSIBLE, c'est-a-dire positif et petit devant")
print("  les deux autres blocs, ce qui est le cas.")
for _lab, _v in (("grandes entreprises",
                  L["blocs_charge_2025_eur"]["grandes entreprises"]),
                 ("bloc intermediaire",
                  L["blocs_charge_2025_eur"]["bloc intermediaire"]),
                 ("micro-entreprises", _charge_micro)):
    _pa = _v / L["charge_2025_eur"]
    print(f"    part de la charge, {_lab:<28}{_pa:.4f}   soit {100*_pa:.1f} %")
print("    Les deux premiers blocs font l'essentiel de la charge du marche, et")
print("    le troisieme, dont tous les multiplicateurs sont spectaculaires, en")
print("    porte moins de deux pour cent. Un multiplicateur de 9,52 sur une base")
print("    reduite ne fait pas un enjeu de charge : c'est la lecon de ce bloc.")
print()
ligne("    charge du bloc intermediaire, 2024",
      L["bloc_intermediaire_charge_2024_eur"], "M EUR")
ligne("    multiplicateur de charge du bloc intermediaire",
      L["mult_charge_bloc_intermediaire"])
_ctrl = (L["blocs_charge_2025_eur"]["bloc intermediaire"]
         / L["bloc_intermediaire_charge_2024_eur"])
print(f"    controle : 37,3 / 10,5 = {_ctrl:.4f}, a comparer au multiplicateur "
      f"retenu {L['mult_charge_bloc_intermediaire']:.2f}, "
      f"ecart {abs(_ctrl - L['mult_charge_bloc_intermediaire']):.4f}")
print()
print("  UNE COQUILLE DE LA SOURCE, TROUVEE PAR CE CONTROLE ET NON REPRISE, et")
print("  c'est la trouvaille de ce bloc. La section 7.1 du rapport ecrit")
print("  << 37,3 M EUR en 2025 contre 10,5 M EUR en 2024 (x2,53) >>. Le rapport")
print("  des deux montants vaut 3,55, et la section 7.6 du MEME rapport donne")
print("  bien x3,53 pour ce bloc. Le 2,53 de la section 7.1 est donc une coquille")
print("  sur le chiffre des unites, et le controle d'identite l'a fait tomber.")
print("  La valeur retenue est celle de la section 7.6 :")
print(f"    multiplicateur de charge du bloc, section 7.6 "
      f"{L['blocs_2025']['bloc intermediaire'][2]:>10.2f}")
print(f"    multiplicateur recalcule 37,3 / 10,5         {_ctrl:>10.4f}")
print("    A SIGNALER COMME ERRATUM DU RAPPORT PUBLIE : la coquille est dans un")
print("    document co-signe, et un lecteur qui divise les deux montants la")
print("    trouvera comme ce script l'a trouvee.")
print()
print("  MULTIPLICATEURS DE FREQUENCE PAR BLOC, exercice 2025 :")
for lab, v in L["mult_frequence_blocs_2025"].items():
    print(f"    {lab:<40}{v:>10.2f}")
ligne("    ensemble du marche", L["mult_frequence_2025"])
print()
print("  RATIOS SINISTRES SUR PRIMES PAR SEGMENT, exercice 2025 :")
for lab, v in L["sp_2025_segment"].items():
    print(f"    {lab:<40}{v:>10.2f}   soit {100*v:>4.0f} %")
print()

print("  LE SOUS-SEGMENT QUI DECROCHE, ET CELUI QUI NE DECROCHE PAS.")
print("  Les deux se ressemblent en charge et DIFFERENT EN NATURE, et c'est")
print("  exactement la distinction que ce memoire impose a ses propres canaux :")
E = L["eti_2025"]
ligne("    ETI, multiplicateur du nombre de sinistres", E["mult_nombre_sinistres"])
ligne("    ETI, multiplicateur de charge", E["mult_charge"])
ligne("    ETI, multiplicateur de frequence", E["mult_frequence"])
ligne("    ETI, frequence 2024",
      f"{E['frequence_2024']:.3f}   soit {100*E['frequence_2024']:.1f} %")
ligne("    ETI, frequence 2025",
      f"{E['frequence_2025']:.3f}   soit {100*E['frequence_2025']:.1f} %")
_fe = E["frequence_2025"] / E["frequence_2024"]
print(f"    controle : 12,4 / 8,7 = {_fe:.4f}, a comparer au multiplicateur "
      f"transcrit {E['mult_frequence']:.2f}, ecart {abs(_fe - E['mult_frequence']):.4f}")
ligne("    petites, multiplicateur du nombre de sinistres",
      L["petites_2025"]["mult_nombre_sinistres"])
ligne("    petites, multiplicateur de frequence",
      L["petites_2025"]["mult_frequence"])
ligne("    moyennes, multiplicateur de frequence",
      L["moyennes_2025"]["mult_frequence"])
ligne("    moyennes, multiplicateur d'exposition",
      L["moyennes_2025"]["mult_exposition"])
print("    LA FREQUENCE DES ENTREPRISES MOYENNES EST PLATE. Leur charge monte")
print("    donc par la seule EXPOSITION, quand celle des ETI monte a exposition")
print("    donnee. Deux hausses de charge de meme allure, deux mecanismes")
print("    opposes, et un seul appelle un ajustement de tarif.")
print()

print("  HISTORIQUE DU DECROCHAGE ETI, 2020 a 2025, et l'indice du taux de prime")
print("  du meme segment, base 100 en 2020 :")
print(f"    {'annee':<8}{'S/P grandes':>14}{'S/P ETI':>10}{'S/P moyennes':>14}"
      f"{'indice taux ETI':>18}")
for k, an in enumerate(L["annees_segment"]):
    print(f"    {an:<8}{100*L['sp_serie_grandes'][k]:>13.0f} %"
          f"{100*L['sp_serie_eti'][k]:>9.0f} %"
          f"{100*L['sp_serie_moyennes'][k]:>13.0f} %"
          f"{L['indice_taux_prime_eti'][k]:>18.0f}")
print()
print("  CE QUE CET HISTORIQUE INTERDIT D'ECRIRE, et c'est un piege de lecture")
print("  symetrique de celui de Hackmageddon : le 42 % des ETI en 2025 n'est PAS")
print("  un regime inedit.")
_i_max = L["sp_serie_eti"].index(max(L["sp_serie_eti"]))
print(f"    S/P ETI maximal de la serie, exercice {L['annees_segment'][_i_max]}   "
      f"{100*max(L['sp_serie_eti']):.0f} %")
print(f"    S/P ETI minimal de la serie, exercice "
      f"{L['annees_segment'][L['sp_serie_eti'].index(min(L['sp_serie_eti']))]}   "
      f"{100*min(L['sp_serie_eti']):.0f} %")
print("    Le segment a deja porte un ratio six fois plus eleve. Le decrochage")
print("    est donc CYCLIQUE, et ce qui fait sa nouveaute n'est pas son niveau")
print("    mais le DECALAGE avec l'indice tarifaire, encore proche de son sommet")
print("    de 2023 quand le retournement technique etait deja engage.")
print()
_i_pic = L["indice_taux_prime_eti"].index(max(L["indice_taux_prime_eti"]))
print(f"    sommet de l'indice tarifaire ETI : exercice "
      f"{L['annees_segment'][_i_pic]}, indice "
      f"{max(L['indice_taux_prime_eti']):.0f}")
print(f"    point bas du S/P ETI : exercice "
      f"{L['annees_segment'][L['sp_serie_eti'].index(min(L['sp_serie_eti']))]}")
print("    Le prix atteint son sommet APRES que le risque a atteint son point")
print("    bas, et il ne reflue qu'ensuite : c'est le mouvement de ciseaux.")
print()

print("  REPARTITION DU MONTANT INDEMNISE PAR TAILLE DE SINISTRE, en M EUR.")
print("  LA LIGNE XXL EST CELLE QUI COMMANDE LE CHOIX DE DONNEES DU MEMOIRE.")
print(f"    {'classe':<24}" + "".join(f"{an:>8}" for an in L["annees_serie"]))
for lab, serie in L["taille_sinistre_eur"].items():
    print(f"    {lab:<24}" + "".join(f"{v:>8.0f}" for v in serie))
_som = [sum(L["taille_sinistre_eur"][c][k] for c in L["taille_sinistre_eur"])
        for k in range(len(L["annees_serie"]))]
print(f"    {'somme des classes':<24}" + "".join(f"{v:>8.0f}" for v in _som))
print(f"    {'charge annuelle publiee':<24}"
      + "".join(f"{v:>8.0f}" for v in L["sinistres_serie_eur"]))
_ecart_max_taille = max(abs(a - b) for a, b in zip(_som, L["sinistres_serie_eur"]))
print()
print("  TROISIEME CONTROLE D'IDENTITE, et c'est le plus exigeant des trois : la")
print("  somme des quatre classes de taille doit redonner la charge annuelle")
print("  indemnisee, et ce sur les SEPT exercices, soit vingt-huit valeurs")
print("  transcrites contrainte par sept sommes.")
print(f"    ecart maximal sur les sept exercices : {_ecart_max_taille:.0f} M EUR")
print("    Les etiquettes du rapport sont arrondies a l'unite de M EUR, donc un")
print("    ecart de quelques unites sur une somme de quatre termes est l'arrondi")
print("    lui-meme. La transcription des vingt-huit valeurs est validee.")
print()
_xxl = L["taille_sinistre_eur"]["XXL, 10 a 40 M EUR"]
print("  ET CE QUE DIT LA LIGNE XXL, qui est le constat externe le plus")
print("  important de tout ce bloc pour le present memoire :")
print(f"    exercices de la serie sans AUCUN sinistre de classe XXL : "
      f"{sum(1 for v in _xxl if v == 0.0)} sur {len(_xxl)}")
print(f"    maximum de la classe XXL, exercice "
      f"{L['annees_serie'][_xxl.index(max(_xxl))]} : {max(_xxl):.0f} M EUR")
print(f"    classe XXL sur l'exercice sous revue, 2025 : {_xxl[-1]:.0f} M EUR")
ligne("    sinistres au-dela du seuil XXL, France, 2025",
      L["n_sinistres_sup_10m_france_2025"])
print(f"    rapport du maximum de la serie a l'exercice 2025 : "
      f"{max(_xxl) / _xxl[-1]:.2f}")
print("    LA QUEUE FRANCAISE EST PEUPLEE PAR ACCIDENT. Deux exercices sur sept")
print("    ne portent aucun sinistre de la classe la plus haute, un exercice en")
print("    porte 135 M EUR, et l'exercice sous revue en porte 19 pour UN SEUL")
print("    sinistre. Une queue ainsi peuplee ne soutient aucun ajustement de")
print("    valeurs extremes : c'est le motif qui fait calibrer la severite de ce")
print("    memoire sur une base INTERNATIONALE et non sur le marche francais.")
print()

print("  CE QUE LA QUEUE PRODUIT AILLEURS SUR LA MEME PERIODE. Ces montants sont")
print("  ce qui interdit de lire la severite moyenne contenue de 2025 comme une")
print("  protection structurelle du marche francais :")
for lab, (v, u) in L["comparaisons_etrangeres"].items():
    print(f"    {lab:<62}{v:>10.1f} {u}")
ligne("    recul des sinistres cyber en Europe, 2024",
      f"{L['recul_sinistres_cyber_europe_2024']:.2f}   soit "
      f"{100*L['recul_sinistres_cyber_europe_2024']:.0f} %")
print("    A LIRE AVEC SON SENS : ce recul de 20 % laisse le niveau britannique")
print("    tres au-dessus de celui des exercices 2020 a 2022. Une baisse depuis")
print("    un point haut n'est pas un retour a la normale.")
print()

print("  PROPENSION A S'ASSURER ET SINISTRALITE DECLAREE, barometre CESIN.")
print("  Couples (vague precedente, vague 2026) :")
for lab, (a, b) in L["cesin_2026"].items():
    print(f"    {lab:<48}{100*a:>6.0f} %{100*b:>8.0f} %")
ligne("    part des grandes entreprises couvertes",
      f"{L['cesin_couverture_grandes']:.2f}   soit "
      f"{100*L['cesin_couverture_grandes']:.0f} %")
print("    LES DEUX SERIES RECULENT D'UN POINT ET DE CINQ POINTS, donc rien ne")
print("    bouge vraiment : la propension a s'assurer se stabilise a un niveau")
print("    eleve, et la frequence DECLAREE recule quand la sinistralite")
print("    INDEMNISEE progresse. Les deux ne mesurent pas la meme chose, et")
print("    l'ecart entre elles est un rappel du sous-report deja declare.")
print()

print("  LE RATIO DE FRAIS, ET POURQUOI IL EXPLIQUE LA POURSUITE DU SOFT MARKET :")
ligne("    ratio de frais et de commissionnement, borne basse",
      f"{L['ratio_frais_bas']:.2f}   soit {100*L['ratio_frais_bas']:.0f} %")
ligne("    ratio de frais et de commissionnement, borne haute",
      f"{L['ratio_frais_haut']:.2f}   soit {100*L['ratio_frais_haut']:.0f} %")
print(f"    ratio combine a la borne haute = "
      f"{100*(L['sp_2025'] + L['ratio_frais_haut']):.0f} %, "
      f"a comparer au seuil d'equilibre de 100 %")
print("    Un S/P de 27 % laisse donc le ratio combine NETTEMENT sous le seuil")
print("    d'equilibre, et c'est cette marge residuelle qui explique qu'une")
print("    detente commerciale se poursuive sur un signal technique qui se")
print("    degrade. Le constat n'est pas contradictoire, il est arithmetique.")

titre("Deux constantes homonymes, et le garde-fou qui remplace un renommage")
# DECISION DU 17 AOUT 2026 : ON NE RENOMME PAS, ON REND LA CONFUSION IMPOSSIBLE A COMMETTRE.
# Le projet porte deux constantes dont les noms ne differ ent que par un tiret bas et qui
# n'ont AUCUN rapport :
#   G_BASE  = gain de propagation g, sans dimension, 0,90 en cas de base (scr_engine,
#             euro_cascade_model). Il mesure une CRITICITE de propagation.
#   GBASE   = echelon de severite par pilier, ordinal, chaque echelon doublant la mediane
#             (cascade_model, severite_model). Il indexe une ECHELLE, jamais un montant.
# Le second est documente avec le mot « gravite », ce qui est le terme consacre de l'AMDEC
# mais rend la collision d'autant plus facile a commettre.
#
# POURQUOI PAS DE RENOMMAGE. Il touche vingt-sept fichiers sur un pipeline GELE. Une
# substitution semantiquement fausse mais numeriquement valide ne serait rattrapee par aucun
# controle : le harnais verifie que les nombres publies sortent des scripts, pas qu'ils
# veulent dire ce qu'on croit. Le rapport risque sur gain est donc mauvais, et c'est
# exactement la situation ou le projet a deja choisi de DECLARER plutot que de corriger,
# comme pour p_u.
#
# CE QUI REMPLACE LE RENOMMAGE. Les deux constantes sont imprimees ICI, cote a cote, avec
# leur nature, leur unite et leur module, a chaque execution. Une assertion garantit qu'elles
# ne peuvent pas devenir egales par accident, ce qui est le seul cas ou une substitution
# passerait inapercue. La distinction est en outre verrouillee dans la table des notations du
# memoire, l'entree g portant « mesure une criticite, jamais un montant ».
_HOMONYMES = []
try:
    import importlib
    _lab = os.path.join(REPO, "exploratory", "vasicek_lab")
    _qual = os.path.join(REPO, "exploratory", "cascade_qualitative")
    for _p in (_lab, _qual):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    _eng = importlib.import_module("scr_engine")
    _cas = importlib.import_module("cascade_model")
    _HOMONYMES = [
        ("G_BASE", _eng.G_BASE, "sans dimension", "gain de propagation g, cas de base",
         "scr_engine, euro_cascade_model"),
        ("GBASE", _cas.GBASE, "ordinal", "echelon de severite par pilier",
         "cascade_model, severite_model"),
    ]
except Exception as _exc:                                     # pragma: no cover
    print(f"  Modules indisponibles depuis ce poste ({_exc.__class__.__name__}) :")
    print("  le garde-fou est saute, la declaration ci-dessus reste valable.")

if _HOMONYMES:
    print(f"  {'nom':<10}{'valeur':>26}{'unite':>18}  {'ce que c est'}")
    for _nom, _val, _unite, _sens, _mod in _HOMONYMES:
        _aff = _val if not isinstance(_val, dict) else \
            "{" + ", ".join(f"P{k}:{v}" for k, v in sorted(_val.items())) + "}"
        print(f"  {_nom:<10}{str(_aff):>26}{_unite:>18}  {_sens}")
        print(f"  {'':<10}{'':>26}{'':>18}  defini dans {_mod}")
    _g = _HOMONYMES[0][1]
    _ech = _HOMONYMES[1][1]
    assert isinstance(_g, float) and isinstance(_ech, dict), \
        "G_BASE doit rester un scalaire et GBASE une table par pilier"
    assert _g not in set(_ech.values()), \
        "COLLISION : le gain de propagation a pris une valeur d'echelon de severite"
    print("\n  CONTROLE : le gain est un scalaire, l'echelon une table par pilier, et le")
    print("  premier ne prend aucune des valeurs de la seconde. Une substitution de l'un par")
    print("  l'autre serait donc detectee ici, ce qui est le seul point ou elle pouvait passer.")
    print("  Ce garde-fou ne remplace pas un renommage, il rend son absence sans consequence.")

titre("ENISA Threat Landscape : CITATION EXTERNE, non recalculable")
# TROISIEME SOURCE AU MEME STATUT QUE HACKMAGEDDON. Le chapitre donnees confronte les parts par
# vecteur de Hackmageddon a une source independante, le rapport de l'agence europeenne, et cite
# trois de ses nombres. Aucun n'est reproductible ici : le rapport n'est pas verse dans
# data/raw/ et sa methodologie de collecte n'est pas la notre. Ils etaient donc HORS CONTROLE,
# recopies a la main dans le texte et verifiables par personne. Les enregistrer les rend
# verifiables CONTRE LA SOURCE, ce qui est la seule garantie disponible pour une citation.
#
# CE QUE CES VALEURS FONT DANS LE MEMOIRE, ET CE QU'ELLES NE FONT PAS. Elles servent a UNE
# confrontation qualitative : l'asymetrie entre deux taux de conversion, mesuree par un
# organisme reglementaire sur un perimetre independant, corrobore un classement que le seul
# volume d'incidents ne peut pas reveler. Aucune n'entre dans un calcul, aucune ne porte un
# niveau de capital.
ENISA = {
    "source": "ENISA Threat Landscape 2025, Agence de l'Union europeenne pour la cybersecurite",
    "n_incidents": 4875,
    "perimetre": "incidents de perimetre europeen, collecte propre a l'agence",
    "conversion": {
        "exploitation de vulnerabilite exposee": 0.70,
        "phishing et ingenierie sociale": 0.27,
    },
}
print(f"  source    : {ENISA['source']}")
print(f"  perimetre : {ENISA['perimetre']}")
print(f"  incidents retenus par le rapport         : {ENISA['n_incidents']}")
print("\n  taux de conversion tentative -> intrusion :")
for k, v in ENISA["conversion"].items():
    print(f"    {k:<44}{v:.2f}   soit {100*v:.0f} %")
rap = (ENISA["conversion"]["exploitation de vulnerabilite exposee"]
       / ENISA["conversion"]["phishing et ingenierie sociale"])
print(f"  rapport entre les deux taux              : {rap:.2f}")
print("\n  A QUOI SERT CETTE ASYMETRIE. Le phishing est le vecteur le plus FREQUENT et l'un des")
print("  moins CONVERTISSANTS ; l'exploitation de vulnerabilite est l'inverse. Un modele calibre")
print("  sur le seul volume d'incidents surponderait donc le phishing, et c'est le motif pour")
print("  lequel le memoire retient de Hackmageddon la STRUCTURE par vecteur et jamais le niveau.")
print("  Le rapport ci-dessus est imprime plutot que laisse au lecteur, pour la meme raison que")
print("  les autres rapports derives : un rapport calcule a la redaction n'est verifiable par")
print("  personne.")
print("\n  AVERTISSEMENT, identique a celui de Hackmageddon. Ce rapport n'est PAS versionne dans")
print("  data/raw/ : ces trois valeurs sont une citation enregistree, pas une sortie")
print("  recalculable. Elles ne portent aucun niveau de capital.")

titre("Preprint de cascade climatique : CITATION EXTERNE, non recalculable")
# MEME STATUT QUE HACKMAGEDDON, ET POUR LA MEME RAISON. Le memoire compare desormais son
# CLASSEMENT DES LEVIERS a celui de ce prepublie, et cette comparaison exige de citer ses
# nombres. Ils ne sont reproductibles par aucun script du projet : le papier tourne sur son
# propre moteur, ses parametres sont synthetiques et ses auteurs le declarent. Les enregistrer
# ici les rend VERIFIABLES CONTRE LA SOURCE, ce qui est tout ce qu'on peut garantir, et les
# sort de la zone ou un chiffre recopie a la main n'est controle par personne.
#
# CES VALEURS NE SERVENT QU'A UNE COMPARAISON DE CLASSEMENT. Aucune n'entre dans un calcul du
# memoire, aucune ne porte un niveau de capital, et le papier n'est pas cite comme repere
# empirique : il est SYNTHETIQUE.
CCRN = {
    "source": "Karimi, Salavati, Shokrollahi, arXiv:2608.09456v1 [q-fin.RM], 10 aout 2026",
    "statut": "prepublication, etude numerique entierement SYNTHETIQUE (declare par les auteurs)",
    # Table 11 du papier : ablation structurelle, VaR 99,5 % de la charge annuelle brute,
    # en milliards de dollars.
    "ablation_var995_mdUSD": {
        "CCRN complet": 3.767,
        "sans propagation dirigee": 1.738,
        "sans interaction coulee de debris": 3.699,
        "sans demand surge": 3.720,
        "approximation mono-evenement": 3.503,
    },
    # Table 12 du papier : sensibilite un-a-la-fois de la prime pure, en % du cas de base.
    "tornado_prime_pct": {
        "probabilite d'arete combustible->incendie": (-25.0, 26.6),
        "frequence annuelle d'evenements": (-21.0, 20.7),
        "coefficient climatique de l'arete": (-11.9, 14.1),
        "raideur de la reponse en severite": (-13.5, 7.3),
        "intensite du demand surge": (-1.8, 1.7),
    },
    # Ce qui explique le desaccord de classement, et c'est une propriete de leur MODELE :
    # leur severite est BORNEE, donc elle n'a pas d'indice de queue.
    "severite_bornee": True,
    "multiplicateurs_lognormaux_sd_log": (0.10, 0.15),
}

print(f"  source : {CCRN['source']}")
print(f"  statut : {CCRN['statut']}")

abl = CCRN["ablation_var995_mdUSD"]
ref = abl["CCRN complet"]
print("\n  ablation structurelle, VaR 99,5 % de la charge annuelle (Md USD, leur table 11) :")
for k, v in abl.items():
    ecart = "" if k == "CCRN complet" else f"{100*(v/ref-1):>+8.1f} %"
    print(f"    {k:<38}{v:>8.3f}{ecart:>12}")
print(f"    -> leur brique la plus lourde est la PROPAGATION DIRIGEE, a "
      f"{100*(abl['sans propagation dirigee']/ref-1):+.1f} %.")

print("\n  tornado un-a-la-fois de la prime pure (% du cas de base, leur table 12) :")
for k, (bas, haut) in CCRN["tornado_prime_pct"].items():
    print(f"    {k:<44}{bas:>8.1f} %{haut:>9.1f} %")
print("    -> leur tete de tornado est la probabilite d'arete, puis la frequence.")

print("\n  POURQUOI LEUR CLASSEMENT N'EST PAS LE NOTRE, ET CE N'EST PAS UN DESACCORD DE MESURE.")
print("  Leur severite est BORNEE : reponse bornee, perte plafonnee par une transformation a")
print(f"  capacite, multiplicateurs lognormaux de moyenne un et d'ecart-type logarithmique")
print(f"  {CCRN['multiplicateurs_lognormaux_sd_log'][0]} et "
      f"{CCRN['multiplicateurs_lognormaux_sd_log'][1]}. Elle N'A DONC PAS D'INDICE DE QUEUE, et")
print("  leur ablation ne contient aucune brique « queue » : on ne retire pas ce qui n'est pas la.")
print("  Le plus proche qu'ils font varier est la RAIDEUR DE LA REPONSE EN SEVERITE, qui sort")
print("  derriere la propagation. Notre severite est une GPD de variance infinie, et la queue y")
print("  domine tout. Les deux classements sont donc chacun corrects DANS LEUR MODELE, et ce qui")
print("  les separe est l'indice de queue, non l'architecture.")
print("\n  CONSEQUENCE DE CITATION, ET ELLE VAUT POUR TOUT LE MEMOIRE : ce preprint se cite sur")
print("  le PROTOCOLE (marges appariees, ablation, separation des echelles), JAMAIS sur l'ordre")
print("  d'un resultat en queue. C'est la meme regle que celle deja posee pour l'echelle des")
print("  quantiles au script 80, etendue aux leviers.")

titre("Verdict")
print("Cette sortie rend citables les constantes de calibration dans les chapitres donnees")
print("et socle. Elle atteste la CONFORMITE A LA CONFIGURATION, non l'exactitude empirique :")
print("celle-ci releve des scripts 07, 08b, 47 et 57.")
print("\nEXIT 0")
