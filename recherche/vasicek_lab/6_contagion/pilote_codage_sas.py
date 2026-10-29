# -*- coding: utf-8 -*-
"""Pilote du 27 octobre 2026 : codage en piliers DORA de 50 descriptions d'incidents SAS OpRisk.

Échantillon : 30 fraudes externes et 20 interruptions d'activité (BDSF), tirées sans remise avec la
graine 20261027 parmi les 570 pertes cyber x finance 2000-2026. Les textes sont sous licence et ne sont
pas versionnés : seuls l'identifiant SAS (Reference ID Code) et les codes le sont.

Codeur : assistant IA (modèle de langage), un seul passage, grille fixée avant lecture. AUCUNE
validation humaine à ce stade : un second codage en aveugle est la condition de tout usage.

Grille (convention du corpus de post-mortems, postmortem_corpus.py) :
  P1 gestion des risques TIC : contrôles d'accès et d'authentification, protection des données,
     sécurité et conception des systèmes, gestion des changements, continuité et secours ;
  P2 gestion des incidents : détection tardive (semaines ou plus), réponse ou notification défaillante ;
  P3 tests : tests insuffisants avant mise en production, vulnérabilité connue non corrigée,
     absence de tests d'intrusion ;
  P4 prestataires tiers : défaillance chez un prestataire, sous-traitant, éditeur ou infrastructure
     de marché qui contribue à l'incident ;
  P5 partage d'information : alerte ou renseignement sur la menace reçu et non exploité, ou non transmis.
Deux niveaux : 'strict' = défaillance établie par le texte ; 'large' = strict plus défaillance que le
texte implique sans l'établir (par exemple une intrusion réussie sans contrôle défaillant nommé).
Une fraude commise contre des clients sans défaillance nommée de l'établissement reçoit l'ensemble vide.
"""

CODES = [
    # id SAS, catégorie, strict, large
    ('22779', 'EF', {4}, {1, 4}),
    ('41776', 'EF', set(), {1}),
    ('40900', 'EF', set(), {1}),
    ('13902', 'EF', set(), set()),
    ('42147', 'EF', {1, 3, 5}, {1, 3, 5}),
    ('41657', 'EF', set(), set()),
    ('36753', 'EF', {1}, {1}),
    ('14465', 'EF', set(), {1}),
    ('19520', 'EF', {4}, {1, 4}),
    ('28958', 'EF', {4}, {1, 4}),
    ('39352', 'EF', set(), {1}),  # intrusion sans contrôle défaillant nommé : niveau large seulement
    ('30061', 'EF', set(), {1}),
    ('21295', 'EF', set(), set()),
    ('27571', 'EF', set(), set()),
    ('35022', 'EF', set(), {1}),
    ('38204', 'EF', set(), set()),
    ('38322', 'EF', set(), {1}),
    ('36185', 'EF', set(), set()),
    ('11549', 'EF', set(), {2}),
    ('38569', 'EF', set(), set()),
    ('36910', 'EF', set(), {1}),
    ('22897', 'EF', set(), set()),
    ('42472', 'EF', {2}, {1, 2}),
    ('24533', 'EF', {2}, {1, 2}),
    ('38300', 'EF', {1}, {1}),
    ('26890', 'EF', set(), set()),
    ('41961', 'EF', {4}, {1, 4}),
    ('23336', 'EF', set(), {1}),
    ('23934', 'EF', {2}, {1, 2}),
    ('20071', 'EF', {1}, {1, 2}),
    ('34574', 'BDSF', {1}, {1}),
    ('41304', 'BDSF', {1, 3}, {1, 3}),
    ('28732', 'BDSF', {1}, {1, 3}),
    ('24603', 'BDSF', {1}, {1, 2}),
    ('12568', 'BDSF', {1}, {1}),
    ('15886', 'BDSF', {1}, {1}),
    ('18968', 'BDSF', {1}, {1}),
    ('29206', 'BDSF', {4}, {4}),
    ('24305', 'BDSF', {1, 2}, {1, 2}),
    ('27884', 'BDSF', {1}, {1, 2}),
    ('6462', 'BDSF', {1}, {1}),
    ('40745', 'BDSF', {1}, {1}),
    ('30525', 'BDSF', {1}, {1, 3}),
    ('10803', 'BDSF', {1}, {1}),
    ('16826', 'BDSF', {1, 3}, {1, 2, 3}),
    ('30103', 'BDSF', {1, 2, 3}, {1, 2, 3}),
    ('24314', 'BDSF', {1, 2}, {1, 2, 3}),
    ('37011', 'BDSF', {1, 4}, {1, 2, 4}),
    ('19530', 'BDSF', {1}, {1, 3}),
    ('38463', 'BDSF', {1}, {1}),
]
