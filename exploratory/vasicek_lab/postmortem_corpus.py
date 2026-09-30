# -*- coding: utf-8 -*-
"""Corpus de post-mortems officiels code en transitions dirigees entre piliers DORA.

SOURCE UNIQUE, partagee par les scripts 59 (matrice ordonnee p_jk) et 64 (biais de
narration). Le corpus vivait dans le corps du script 59 ; deux scripts le lisant, il
devait devenir un module, sans quoi la moindre correction d'un codage aurait a etre
recopiee et les deux scripts auraient pu divulguer des chiffres incoherents.

Le script 53, historique, garde son propre litteral : il porte les justifications longues
et les intitules d'origine des sept premiers incidents, et sa sortie est citee telle
quelle dans le memoire. Le rapprochement des deux se fait par SEPT_INITIAUX ci-dessous.

=====================================================================================
PROTOCOLE DE CODAGE (identique au script 53, complete par le denominateur)
-------------------------------------------------------------------------------------
transitions          : (j, k, justification) = le rapport ETABLIT que la defaillance de j a
                       precede ET conditionne celle de k. On ne code pas une co-occurrence
                       sans ordre etabli.
piliers_defaillants  : TOUS les piliers dont le rapport etablit la defaillance, y compris
                       ceux qui n'ont rien entraine. C'est le DENOMINATEUR de p_jk. Il
                       contient necessairement les piliers apparaissant dans les
                       transitions ; il peut en contenir d'autres.
source_type          : "primaire"   = rapport officiel consulte
                       "secondaire" = synthese reglementaire, communique, couverture sectorielle
perspective          : "entite"     = l'organisation etudiee subit et gere l'incident
                       "fournisseur"= l'organisation etudiee EST le tiers defaillant
                       Le melange des deux est un biais a surveiller : vu du fournisseur, une
                       defaillance interne n'est pas un risque P4 ; vue de ses clients, si.
=====================================================================================
"""

CORPUS = [
    # ---------------------------------------------------------------- corpus initial (script 53)
    dict(nom="Microsoft Exchange Online 2023", source="CSRB (DHS/CISA), 20/03/2024",
         source_type="primaire", perspective="fournisseur", secteur="fournisseur TIC critique",
         piliers_defaillants={1, 2, 3, 4},
         transitions=[
             (1, 3, "culture de securite inadequate, priorisation des fonctionnalites -> arret "
                    "de la rotation des cles, pas d'alerte sur l'age des cles"),
             (3, 2, "absence de detection -> compromission signalee par le Departement d'Etat "
                    "et non par Microsoft, puis communication publique inexacte"),
             (1, 4, "lacunes du processus de securite des fusions-acquisitions -> acces via les "
                    "identifiants d'un ingenieur d'une entite acquise"),
         ]),
    dict(nom="TSB Bank 2018", source="Slaughter and May ; FCA/PRA 48,65 M GBP",
         source_type="primaire", perspective="entite", secteur="banque reglementee",
         piliers_defaillants={1, 2, 3, 4},
         transitions=[
             (1, 4, "defaut de gouvernance du programme -> supervision insuffisante des "
                    "prestataires (70+ fournisseurs, sous-traitance SABIS)"),
             (1, 3, "approche 'big bang' -> tests insuffisants, deux centres de donnees non "
                    "testes, 4 424 defauts ouverts au demarrage"),
             (3, 2, "defauts non traites -> incident massif et gestion de crise defaillante"),
             (4, 2, "defaillance des prestataires -> incapacite a diagnostiquer et retablir"),
         ]),
    dict(nom="Equifax 2017", source="GAO-18-559 ; House Oversight, decembre 2018",
         source_type="primaire", perspective="entite", secteur="bureau de credit",
         piliers_defaillants={1, 2, 3, 5},
         transitions=[
             (1, 5, "responsabilites TIC non definies -> l'alerte US-CERT (Apache Struts) "
                    "n'atteint pas les administrateurs (liste de diffusion perimee)"),
             (5, 3, "alerte non routee -> le scan de vulnerabilites ne detecte pas la faille"),
             (3, 2, "certificat de supervision expire -> intrusion non detectee 76 jours"),
             (1, 3, "ecart entre politique TIC et operation -> correctifs defaillants"),
         ]),
    dict(nom="Knight Capital 2012", source="SEC, ordonnance du 16/10/2013",
         source_type="primaire", perspective="entite", secteur="courtier reglemente",
         piliers_defaillants={1, 2, 3},
         transitions=[
             (3, 2, "controles de deploiement insuffisants, code obsolete actif -> absence de "
                    "procedure de reponse, 45 minutes de pertes non maitrisees"),
             (1, 3, "defaut de gestion du risque au niveau de la firme -> deploiement defaillant"),
         ]),
    dict(nom="Capital One 2019", source="OCC 80 M USD ; Federal Reserve cease and desist",
         source_type="primaire", perspective="entite", secteur="banque reglementee",
         piliers_defaillants={1, 3, 4},
         transitions=[
             (1, 4, "pas d'evaluation des risques avant migration cloud -> dependance externe "
                    "mal maitrisee"),
             (1, 3, "manquements de gouvernance non corriges -> pare-feu applicatif mal "
                    "configure, controles non testes"),
         ]),
    dict(nom="ION Cleared Derivatives 2023", source="Communications ION ; regulateurs UE/UK",
         source_type="secondaire", perspective="fournisseur", secteur="fournisseur post-marche",
         piliers_defaillants={2, 4},
         transitions=[
             (4, 2, "ransomware chez un prestataire partage -> gestion d'incident degradee chez "
                    "de nombreux acteurs, retour au manuel, declarations retardees"),
         ]),
    dict(nom="MOVEit 2023", source="Chapitre 5 du memoire ; listes de victimes publiques",
         source_type="secondaire", perspective="fournisseur", secteur="chaine logicielle",
         piliers_defaillants={2, 4},
         transitions=[
             (4, 2, "faille d'un logiciel de transfert tiers exploitee en masse -> gestion et "
                    "notification simultanees chez des centaines d'organisations"),
         ]),

    # ---------------------------------------------------------------- extension (script 59)
    dict(nom="CrowdStrike Falcon 2024", source="CrowdStrike, External Technical Root Cause "
                                               "Analysis, Channel File 291, 06/08/2024",
         source_type="primaire", perspective="fournisseur", secteur="fournisseur TIC critique",
         piliers_defaillants={2, 3},
         transitions=[
             (3, 2, "le Content Validator laisse passer un fichier de contenu dont le nombre de "
                    "champs ne correspond pas au modele (21 attendus, 20 fournis), le defaut "
                    "n'etant pas vu par les couches de test du fait d'un caractere generique -> "
                    "8,5 millions de systemes en erreur, gestion d'incident massive"),
         ]),
    dict(nom="Log4Shell 2021-2022", source="CSRB (DHS/CISA), premier rapport, 11/07/2022",
         source_type="primaire", perspective="entite", secteur="multi-secteurs",
         piliers_defaillants={1, 2, 4},
         transitions=[
             (1, 4, "absence d'inventaire des dependances logicielles et de gouvernance de la "
                    "chaine d'approvisionnement -> incapacite a savoir ou Log4j etait deploye"),
             (4, 2, "dependance tierce non inventoriee -> reponse a incident prolongee, le CSRB "
                    "qualifiant la vulnerabilite d'endemique pour une decennie"),
         ]),
    dict(nom="Raphaels Bank 2015", source="FCA/PRA, amende conjointe 1,89 M GBP",
         source_type="secondaire", perspective="entite", secteur="banque reglementee",
         piliers_defaillants={1, 2, 4},
         transitions=[
             (1, 4, "systemes et controles inadequats pour la supervision et la gouvernance des "
                    "accords d'externalisation, manquement repete"),
             (4, 2, "panne chez le prestataire de traitement de cartes -> interruption du service "
                    "aux clients et remediation tardive"),
         ]),
]

# les sept incidents du corpus initial, dans l'ordre du script 53
SEPT_INITIAUX = [c["nom"] for c in CORPUS[:7]]
