# Le dépôt Zenodo du matériel complémentaire

## Version 1.5 : publiée le 20 octobre 2026

**Vérifié sur l'API Zenodo le 29 octobre 2026** : la dernière version du DOI de concept est le
[record 23507171](https://zenodo.org/records/23507171), publié le 20 octobre, et ses deux fichiers
ont les md5 des fichiers locaux du tableau « Fichiers locaux à téléverser » ci-dessous
(`98c5b5ba…` et `503968d8…`). Rien n'est à republier tant que ces deux PDF ne changent pas.

**État vérifié sur l'API Zenodo le 20 octobre 2026, avant publication.** Le DOI de concept mène à la
[version 1.4, record 23080143](https://zenodo.org/records/23080143), publiée le 1er octobre.
Les deux documents y sont déjà présents, mais leurs empreintes diffèrent des fichiers locaux
corrigés. Il faut publier une nouvelle version du même dépôt, proposée ici comme **1.5**, avant
de remettre le rapport final. Le DOI de concept imprimé dans le rapport reste inchangé.

| Fichier publié en 1.4 | md5 vérifié sur l'API |
|---|---|
| `materiel_complementaire.pdf` | `b6ba3c692bb9f4799057c6124b5ba105` |
| `supplement_rapport_ENSAE.pdf` | `34ec0fbd8ad8e145601e23450441ba8c` |

**Fichiers locaux à téléverser.** Le tableau suivant doit être actualisé après la dernière
reconstruction. La publication nécessite une session authentifiée sur le compte Zenodo de
l'auteur ; aucune publication n'est effectuée par les scripts de construction.

| Fichier | Pages | Octets | md5 | Construit par |
|---|---|---|---|---|
| `materiel_complementaire.pdf` | 70 | 3 554 650 | `98c5b5ba5185ed45e58a36063912f837` | `build_matcomp.py` |
| `supplement_rapport_ENSAE.pdf` | 127 | 5 222 880 | `503968d83d52c7b2df48f4174fff7811` | `build_supplement.py` |

**Corrections du 20 octobre 2026.** Le matériel complémentaire précise la dispersion entre
graines et les limites des comparaisons avec plafond. Le supplément reprend les corrections
sur l'identification de W, les paramètres de Markov, les états de conformité et l'interpolation
de fréquence, puis les reformulations finales selon les consignes de Caroline. Les fichiers locaux corrigés restent à publier.

### Historique des constructions précédentes

**Reconstruit le 2 octobre 2026** (empreinte ci-dessus, remplace `cbb926a5…`) : le chapitre 5 du
rapport définit les barèmes ROOT, TRANS et GBASE avec une figure (une page de plus avant la
bibliographie) et la bibliographie passe en corps `\small`. Comparaison à la construction
précédente, folios retirés : seules les quatre pages de bibliographie diffèrent, par leur
composition ; aucun texte du supplément, aucun numéro relayé ne change (six relais vérifiés).

- `materiel_complementaire.pdf` a été **reconstruit sur la version Institut déposée** (figée depuis
  le 1er octobre) et comparé au fichier de la 1.2 par `comparer_pdf.py` : **aucune différence de texte
  hors folios**. Les corrections faites après la construction de la 1.2 (figure 11.3, formules du
  chapitre 10, neuf défauts de texte, résumé et notes) n'ont déplacé aucun renvoi des annexes H à J.
  Seule nouveauté : **48 signets**, que l'extraction par pages perdait. Son md5 change donc, son texte
  non.
- `supplement_rapport_ENSAE.pdf` est extrait d'une compilation du rapport ENSAE entier avec S1 à S4
  insérés avant la bibliographie, comme le matériel complémentaire l'est de la version Institut. Il
  porte une page de titre, des signets et la bibliographie. Contrôles du script, tous passés :
  aucun renvoi non résolu ; les six numéros que le rapport imprime vers le supplément (S1.2, S1.4,
  S1.4.1, S4.6, S4.63, S4.68) se retrouvent à l'identique ; aucune chaîne sensible (motif de
  `ci/vitrine/exporter.py` : noms d'entités et de personnes, adresses), aucun chemin de poste. Il
  contient cinq figures de rapports tiers (deux figures LUCY, la cartographie France Assureurs, le
  baromètre CESIN, la carte Avast), couvertes par la réserve de droits de la description.
  **Reconstruit le 1er octobre au soir** (132 → 130 pages, empreinte ci-dessus) : la bibliographie
  du rapport passe en interligne simple et le supplément l'embarque. `comparer_pdf.py` contre la
  construction précédente : huit différences, toutes des titres courants « Bibliographie » déplacés
  par les sauts de page, aucune différence de texte. Six des sept figures remontées dans le corps du
  rapport le même soir restent aussi dans le supplément, sous leurs étiquettes d'origine ; la
  septième, N1, vient des notes de synthèse de la version Institut. **Puis, tard le soir, S5
  (démonstrations du modèle de cascade) est revenu en annexe A du rapport** : le supplément compte
  quatre chapitres et 126 pages (pages et md5 du tableau). `comparer_pdf.py` contre la construction
  à 130 pages : sept différences, toutes attendues, la page de titre (S1 à S4) et le retrait de S5 ;
  aucun numéro du rapport cité par le supplément n'a bougé.

La description proposée ci-dessous couvre les deux documents et l'annexe I sans l'ancien
dispositif de vérification retiré en version 1.2.

### Marche à suivre

1. ouvrir le [DOI de concept](https://doi.org/10.5281/zenodo.22874524), actuellement le record
   `23080143`, et cliquer sur **New version** ;
2. **fichiers** : remplacer les deux anciens PDF par ceux du tableau, depuis
   `memoire/materiel_complementaire/` ;
3. **Version** : `1.5` si la dernière version publiée est toujours la 1.4 ; **date de publication** : le jour de la publication ;
4. **description** : remplacer le texte par celui de la section suivante ;
5. **mots-clés** (facultatif) : le champ ne porte aujourd'hui qu'une entrée, qui recopie toute la
   liste avec son préfixe « Les mots-clés du résumé : ». Le remplacer par dix entrées : risque cyber,
   DORA, SCR, Solvabilité II, théorie des valeurs extrêmes, cascade dirigée, identification
   partielle, bornes de capital, inférence bayésienne, valeur de l'information ;
6. **ne pas toucher** au titre, à la licence (CC BY 4.0), à l'auteur ni au type de ressource ;
7. publier, puis vérifier : `https://doi.org/10.5281/zenodo.22874524` doit aboutir au nouveau
   record ; les deux fichiers servis doivent porter les md5 du tableau ; l'ancienne 1.4 doit
   afficher le bandeau « There is a newer version of the record » ; reporter ici le numéro du
   record et son DOI de version.

**Ne pas créer un nouveau dépôt** : il recevrait un autre DOI de concept, que ni le mémoire ni le
rapport n'impriment.

### Description à coller

> Ce dépôt rassemble deux documents compagnons du mémoire d'actuariat de Kélian Kaddouri (ENSAE
> Paris, filière Actuariat, 2026), « Quantification du SCR lié à la non-conformité au règlement
> DORA ».
>
> **materiel_complementaire.pdf.** Les trois annexes sorties de la version complète du mémoire,
> déposée à l'Institut des actuaires, pour en alléger la lecture. Le document en conserve la
> numérotation : le mémoire porte les annexes A à G, le présent document prend la suite avec les
> annexes H, I et J.
>
> **Annexe H, démonstrations du socle.** Les théorèmes standards de la théorie des valeurs extrêmes
> mobilisés par le chapitre du socle mécaniste (Fisher-Tippett, Balkema-de Haan, Hill, théorème des
> types).
>
> **Annexe I, pièces justificatives.** Les réconciliations entre sources, les corrections
> d'instrument et le registre des limites.
>
> **Annexe J, compléments aux chapitres du corps.** Lecture de marché détaillée, diagnostics de
> queue, choix du seuil, backtests complémentaires et analyses de sensibilité.
>
> **supplement_rapport_ENSAE.pdf.** Le supplément de la version du mémoire déposée à l'ENSAE comme
> rapport de stage de fin d'études, ramenée à moins de cent pages. Ses chapitres S1 à S4 portent les
> numéros auxquels ce rapport renvoie : adaptations par pilier (S1), pistes pour de futurs mémoires
> (S2), questionnaire d'élicitation (S3), compléments retirés du corps du rapport, dont le protocole
> d'élicitation et les relectures de praticiens (S4).
>
> Les démonstrations du modèle de cascade dirigée figurent en annexe A de la version complète du
> mémoire et du rapport. Certaines figures sont reproduites de rapports tiers avec l'autorisation de
> leurs détenteurs et ne sont pas couvertes par la licence de ce dépôt.

Si Zenodo propose un champ de notes de version : « 1.5 : corrections du 20 octobre 2026 ; unités
incident/pilier, portée des diagnostics statistiques, paramètres de Markov et hypothèses de
conformité précisés ; compléments alignés sur le rapport ENSAE final. »

### Reconstruire les deux fichiers

Depuis `memoire/`, après toute modification du mémoire ou du rapport :

```powershell
python materiel_complementaire/build_matcomp.py
& $tec -X compile _tmp_materiel.tex --keep-intermediates
python materiel_complementaire/build_matcomp.py --extract
python materiel_complementaire/comparer_pdf.py <ancien.pdf> materiel_complementaire/materiel_complementaire.pdf
Remove-Item _tmp_materiel.*
python materiel_complementaire/build_supplement.py     # compile, extrait et controle en une fois
```

`build_supplement.py` s'arrête au moindre échec de ses contrôles. Relancer
`reduction/relais_supplement.py` avant lui si un contenu a bougé entre le rapport et le supplément.

## État au 29 septembre 2026 : la version 1.2 est publiée

| | |
|---|---|
| **Lien imprimé dans le mémoire** | le **DOI de concept** `10.5281/zenodo.22874524`, qui résout toujours vers la dernière version, aujourd'hui la 1.2 |
| **Version 1.2, en vigueur** | `https://zenodo.org/records/23040521`, DOI de version `10.5281/zenodo.23040521`, publiée le 29 septembre |
| Version 1.1, remplacée | `https://zenodo.org/records/23033659`, DOI de version `10.5281/zenodo.23033659`, publiée le 29 septembre |
| Version 1.0, remplacée | `https://zenodo.org/records/22874525`, DOI de version `10.5281/zenodo.22874525`, créée le 23 septembre |
| Fichier de la 1.2 | `materiel_complementaire.pdf`, 69 pages, 3 544 358 octets, md5 `9daff1103a1f5c103569aac7f031e4b0` |
| Licence | CC BY 4.0, avec réserve de droits dans la description sur les figures LUCY |

**Lu sur l'API Zenodo le 29 septembre 2026**, et non déclaré : les trois versions portent leur champ
Version renseigné (`1.0`, `1.1`, `1.2`), le concept aboutit à `records/23040521`, et le fichier servi
par la 1.2 a **la même empreinte md5** que le PDF committé. Le champ Version de la 1.1, qui était
vide, est corrigé. **Un écart de date à connaître** : la 1.0 a été créée le 23 septembre à 13 h 49
UTC (champ `created` de l'API), mais sa date de publication *déclarée* affiche le 29 septembre,
comme les deux autres. **L'origine de cet écart n'est pas documentée** : c'est vraisemblablement un
ajustement de la date déclarée, prévu à l'étape 4 de la procédure ci-dessous, mais l'API ne le dit
pas et il n'a pas été confirmé. Le passage du mémoire qui renvoie au dépôt n'imprime que le DOI et
l'adresse, aucune date de version. Les créations réelles sont le 23 septembre (1.0), le 29 à 8 h 08
(1.1) et le 29 à 13 h 49 (1.2).

**Pourquoi une 1.2.** À la demande de Kélian, la section « Le dispositif de vérification des
nombres publiés » a été retirée de l'annexe I : ses comptes étaient périmés et l'annexe G du
mémoire en porte l'essentiel. Le fichier passe de 70 à 69 pages. Le mémoire n'a pas été recompilé
pour autant, son lien étant celui du concept.

## Ce qui a été vérifié après publication de la 1.1, et refait pour la 1.2

Par requêtes réelles, le 29 septembre 2026.

1. `https://doi.org/10.5281/zenodo.22874524` aboutit à `records/23033659` puis, depuis la 1.2, à
   `records/23040521` : **le lien du mémoire a basculé seul deux fois**, sans recompilation ;
2. le fichier servi par Zenodo porte **la même empreinte md5** que le PDF du dépôt : ce qu'un
   lecteur télécharge porte bien les renvois corrigés, §11.6 et suivants ;
3. `records/22874525` affiche le bandeau « There is a newer version of the record », qui mène à la
   1.1 ;
4. titre, description, mots-clés, licence, auteur et type de ressource sont **identiques** entre
   les deux versions, comparés champ par champ par l'API. Seuls diffèrent le fichier et le champ
   Version.

## Pourquoi le mémoire imprime le DOI de concept

Zenodo attribue deux identifiants à un même dépôt. Le **DOI de version** désigne une version
figée ; le **DOI de concept** désigne le dépôt tout entier et renvoie toujours à sa dernière
version.

La 1.1 en donne l'illustration exacte : le DOI de version de la 1.0 ouvre toujours la 1.0, avec un
bandeau que beaucoup de lecteurs ne voient pas, quand le DOI de concept a basculé seul vers la 1.1.
Imprimer le concept rend le mémoire **insensible à toute mise à jour future** du matériel, sans
jamais devoir le recompiler.

## Pourquoi il a fallu une version 1.1

**La version 1.0 renvoyait vers de mauvaises sections du mémoire.** Le commit `a39bcb3` du
28 septembre, « les deux natures de limites », a inséré une section au chapitre 11, ce qui a
décalé d'un cran toutes les sections suivantes. La 1.0, construite le 23, disait « voir §11.5 » là
où le mémoire déposé porte cette section en §11.6, et ainsi de suite sur **quinze pages**.
Comparaison faite titres courants retirés : **aucune autre différence de contenu**, seuls ces
renvois vers les chapitres 10 et 11 avaient bougé. Le mémoire était juste ; c'est le document
compagnon qui pointait à côté.

## Publier une nouvelle version

Les fichiers d'une version publiée sont **immuables** : on ne remplace pas le PDF, on publie une
**nouvelle version du même dépôt**.

1. se connecter à Zenodo et ouvrir la dernière version, aujourd'hui `records/23040521` ;
2. cliquer sur **New version** : Zenodo crée un brouillon prérempli avec les métadonnées de la
   précédente ;
3. retirer l'ancien fichier et téléverser
   `exploratory/memoire_cascade/materiel_complementaire/materiel_complementaire.pdf` ;
4. **renseigner le champ Version**, oublié pour la 1.1 et corrigé depuis, et ajuster la date de
   publication ;
5. **ne pas toucher** au titre, à la description ni à la licence ;
6. publier, puis refaire les quatre vérifications ci-dessus.

**Ne pas créer un nouveau dépôt** : il recevrait un autre DOI de concept, et le lien imprimé dans
le mémoire ne le trouverait jamais.

## Le fichier à téléverser

`materiel_complementaire.pdf`, **69 pages, 0 renvoi non résolu**, md5 `9daff110…`, reconstruit le
29 septembre 2026 au soir depuis le mémoire à 199 pages. C'est le fichier de la **1.2**, publié à
l'identique. Ses renvois concordent avec le mémoire déposé.

**Le reconstruire après toute modification du mémoire**, et pas seulement après une modification des
trois annexes : il est extrait d'une compilation du mémoire entier, donc ses renvois, ses folios et
sa bibliographie portent l'état du document au jour de la construction. C'est l'oubli de cette
règle qui a rendu la 1.0 fausse. **Puis le comparer au fichier publié, titres courants retirés** :
s'il en diffère autrement que par les folios, une nouvelle version est due.

**Il ne porte aucun signet**, l'extraction par pages perdant ceux du mémoire. Le lecteur navigue par
la page de titre, qui annonce le contenu. Ce n'est pas un défaut de fond ; à reprendre seulement si
une version 1.3 se publie pour une autre raison. La 1.2 n'en a pas non plus. **Repris pour la 1.3**,
qui se publie pour le supplément du rapport ENSAE : `build_matcomp.py` pose désormais les signets.

## Ce qui a été vérifié sur le PDF de la 1.1, hors section retirée de la 1.2

| Contrôle | Résultat |
|---|---|
| Renvois non résolus | 0 |
| Annexes H, I et J présentes | oui, pages 2, 5 et 30 |
| Assureurs nommés | **aucun**, l'anonymisation du mémoire tient ici aussi |
| Donnée sous licence redistribuée | **aucune** : seulement le PDF, jamais la base OpRisk |
