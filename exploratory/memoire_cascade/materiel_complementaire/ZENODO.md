# Le dépôt Zenodo du matériel complémentaire

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
une version 1.3 se publie pour une autre raison. La 1.2 n'en a pas non plus.

## Ce qui a été vérifié sur le PDF de la 1.1, hors section retirée de la 1.2

| Contrôle | Résultat |
|---|---|
| Renvois non résolus | 0 |
| Annexes H, I et J présentes | oui, pages 2, 5 et 30 |
| Assureurs nommés | **aucun**, l'anonymisation du mémoire tient ici aussi |
| Donnée sous licence redistribuée | **aucune** : seulement le PDF, jamais la base OpRisk |
