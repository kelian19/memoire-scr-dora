# Corpus de post-mortems : état et candidats à coder

Registre de travail du script `59_corpus_etendu_pij.py`. Objectif annoncé : porter le corpus
de 7 à une trentaine de rapports, ce qui est la condition pour que le postérieur par paire
devienne concluant sur plus de 2 paires sur 6.

## Règle de recevabilité

Un incident n'entre au corpus que s'il satisfait les trois conditions ci-dessous. Les deux
premières sont celles du chapitre 9 ; la troisième est ajoutée par le script 59.

1. **Enquête publique** reconstituant une séquence de défaillances de contrôle, pas une simple
   annonce d'incident.
2. **Attribution possible à des piliers DORA**, c'est-à-dire à des domaines de contrôle et non
   à des vecteurs d'attaque.
3. **Piliers défaillants énumérables**, y compris ceux qui ont cédé sans rien entraîner. C'est
   le dénominateur de `p_jk` : sans lui on ne peut estimer qu'un sens, pas une amplitude.

## Codés (10)

| Incident | Source | Type | Perspective | Secteur |
|---|---|---|---|---|
| Microsoft Exchange Online 2023 | CSRB, 20/03/2024 | primaire | fournisseur | fournisseur TIC critique |
| TSB Bank 2018 | Slaughter and May ; FCA/PRA | primaire | entité | banque réglementée |
| Equifax 2017 | GAO-18-559 ; House Oversight | primaire | entité | bureau de crédit |
| Knight Capital 2012 | SEC, 16/10/2013 | primaire | entité | courtier réglementé |
| Capital One 2019 | OCC ; Federal Reserve | primaire | entité | banque réglementée |
| ION Cleared Derivatives 2023 | communications ION ; régulateurs | secondaire | fournisseur | post-marché |
| MOVEit 2023 | chapitre 5 ; listes publiques | secondaire | fournisseur | chaîne logicielle |
| CrowdStrike Falcon 2024 | RCA externe, 06/08/2024 | primaire | fournisseur | fournisseur TIC critique |
| Log4Shell 2021-2022 | CSRB, 11/07/2022 | primaire | entité | multi-secteurs |
| Raphaels Bank 2015 | FCA/PRA, amende conjointe | secondaire | entité | banque réglementée |

**Les trois derniers sont des propositions de codage à valider.** Ils reposent sur les constats
publics et établis de leurs rapports, mais n'ont pas fait l'objet d'une lecture intégrale
consignée, au même titre que le second codage en aveugle du script 54 reste à obtenir.

## Candidats identifiés, non encore codés

Priorité aux **entités financières réglementées**, parce que le corpus n'en compte que six sur
dix et que la perspective fournisseur introduit un biais de cadrage (vue du fournisseur, sa
propre défaillance n'est pas un risque P4 ; vue de ses clients, si).

### Régulateurs financiers, enquêtes ou sanctions publiées
- **CSRB Lapsus$ (août 2023)** : contournement d'authentification multifacteur, échange de carte
  SIM. À vérifier avant codage : le rapport porte sur un mode opératoire transverse plus que sur
  une entité, ce qui complique l'énumération des piliers défaillants.
- **Bangladesh Bank / SWIFT 2016** : rapports parlementaires et enquêtes SWIFT.
- **Travelex 2020** : administration judiciaire, couverture réglementaire britannique.
- **Optus 2022 et Medibank 2022** : procédures des régulateurs australiens (OAIC, APRA).
- **Sanctions FCA et PRA pour manquements aux systèmes et contrôles TIC** : le registre
  d'enforcement de la FCA est public et exhaustif, c'est le gisement le plus dense en entités
  réglementées.
- **Ordonnances SEC pour manquements de cybersécurité** (au-delà de Knight Capital) : le
  registre d'actions administratives est également public.

### Superviseurs européens
- **Rapport d'incidents TIC majeurs des ESA** : donne l'origination par cause au niveau agrégé.
  Ne fournit pas de séquences par incident, donc ne peut pas entrer au corpus tel quel, mais
  ancre `ROOT` de l'extérieur (déjà utilisé au chapitre 12b).
- **Notifications d'incidents sous NIS2** : horodatage réglementaire à 24 h, 72 h et un mois.
  À surveiller : c'est la source qui pourrait, à terme, lever la limite d'identification.

### Fournisseurs TIC critiques
- **Okta 2023** : compromission du système de gestion des tickets de support.
- **SolarWinds 2020** : action SEC, et rapports publics sur la chaîne de compilation.
- **Kaseya 2021** : chaîne d'approvisionnement des prestataires de services managés.

## Ce que l'extension change, et ce qu'elle ne change pas

Mesuré au 3 août 2026, sur le passage de 7 à 10 rapports :

- le test de direction **se renforce**, z passant de +3,93 à **+5,12**, et il survit aux deux
  restrictions les plus dures : **+4,08** sur les seules sources primaires, **+3,62** sur la
  seule perspective entité ;
- la matrice ordonnée `p_jk` devient estimable, avec ses crédibles ;
- **aucun** lien n'est pour autant établi comme fort : les douze liens dont le crédible exclut
  0,5 l'excluent tous **par le bas**. Le corpus établit où la contagion ne va pas, pas où elle
  va fort ;
- la corrélation avec le classeur d'expert n'est **pas** significative, ni sur la matrice
  complète (rho = +0,08) ni sur la partie antisymétrique (rho = +0,31, p = 0,19).

Autrement dit, l'extension **confirme la frontière à trois niveaux du chapitre 9 sur un objet
plus riche**, elle ne la déplace pas. Il faudra vérifier si cela tient encore à trente rapports :
c'est précisément la question ouverte.
