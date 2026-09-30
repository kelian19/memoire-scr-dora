# Matériel complémentaire du mémoire

Trois annexes du mémoire *Quantification du SCR lié à la non-conformité au règlement DORA*
ont été sorties du document déposé le **18 septembre 2026**, sur décision de Kélian, pour
alléger la lecture. Elles vivent ici et sont destinées à un dépôt numérique pérenne
(GitHub ou Zenodo), dont l'adresse est annoncée en tête des annexes du mémoire.

**Aucun de ces trois fichiers n'est appelé par le fichier maître.** `main_ensae.tex` est
désormais le seul (`main.tex` et `main_v2.tex` ont été supprimés le 19 septembre 2026) et il
ne les compile pas. Les remettre dans `chapitres/` et rétablir leurs `\input` suffirait à
revenir en arrière, mais il faudrait aussi défaire les **45** `\matcomp` posés dans le corps
(voir plus bas).

## Le PDF à déposer en ligne

`materiel_complementaire.pdf` — **69 pages, 0 renvoi non résolu**, reconstruit le 29 septembre 2026
au soir et publié le même jour comme **version 1.2** du dépôt Zenodo. La version 1.1, publiée le
matin, portait encore la section « Le dispositif de vérification des nombres publiés », retirée de
l'annexe I depuis. La version 1.0, construite le 23, renvoyait vers des sections des chapitres 10 et
11 décalées d'un cran depuis le 28.

**LE RECONSTRUIRE APRÈS TOUTE MODIFICATION DU MÉMOIRE**, et pas seulement après une modification
des trois annexes : il est extrait d'une compilation du mémoire entier, donc ses renvois et sa
bibliographie portent l'état du document à la date de la construction.

Il est produit par la méthode décrite plus bas : compilation du mémoire **entier** avec les trois
`\input` rétablis, puis extraction des pages. Les renvois sont donc résolus dans les deux sens.
Il s'ouvre sur une page de titre qui le rattache au mémoire et annonce son contenu.

**Le lettrage continue celui du mémoire.** Le document déposé porte les annexes A à G ; celui-ci
prend la suite avec **H** (démonstrations), **I** (pièces justificatives) et **J** (compléments au
corps). Il embarque la bibliographie du mémoire, de sorte que ses citations se résolvent.

Pour le régénérer après une modification : reprendre le script
`build_matcomp.py` (il recopie `main_ensae.tex`, insère la page de titre et les trois `\input`
avant `\bibliographystyle`, dans un master temporaire non versionné), compiler, puis extraire de
la page de titre jusqu'à la fin.

**L'adresse est posée et le dépôt est publié.** La notice qui ouvre les annexes imprime le **DOI de
concept** `10.5281/zenodo.22874524`, cliquable, page physique 156 du PDF ENSAE (folio 155) et 152 du
PDF Institut (folio 151). Il renvoie toujours à la dernière version publiée, la **1.2**
(`records/23040521`) depuis le 29 septembre 2026, vérifiée sur l'API Zenodo. **45** renvois du corps y conduisent par la macro
`\matcomp`, compte relevé sur les deux PDF le 29 septembre. Historique, publication d'une nouvelle
version et vérifications dans `ZENODO.md`, à côté de ce fichier.

## Ce que contient chaque fichier

| Fichier | Ancienne place | Contenu |
|---|---|---|
| `15_demonstrations.tex` | annexe B | démonstrations du socle : Fisher-Tippett, Balkema-de Haan, Hill, théorème des types, VaR contre espérance de VaR |
| `17_pieces_justificatives.tex` | annexe D | vingt sections de pièces justificatives : réconciliations, postures, corrections d'instrument, horloges, plafond et saturation, séquences ordonnées, quantile des configurations, registre des limites. **La table des paramètres en a été retirée le 18 septembre et remise dans le mémoire**, annexe D, `chapitres/17b_table_parametres.tex` |
| `21_complements_corps.tex` | annexe E | dix-huit compléments aux sections du corps : lecture de marché LUCY, cadre réglementaire comparé, Hill, choix du seuil, backtest, sensibilité à la propagation, ancrage d'échelle, détenir contre transférer, corpus étendu, agrégation |

## Ce qu'il faut savoir avant de les compiler à part

Ces fichiers ne sont pas autonomes. Pour en faire un document séparé il faut :

1. **un préambule** : reprendre `preambule_v2.tex` du mémoire, qui porte les macros
   (`\VaR`, `\qsev`, `\figover`, les environnements `cle` et `attention`, les couleurs) ;
2. **les figures** : elles sont lues dans `../../vasicek_lab/figures/` et, pour les trois
   figures LUCY, dans `../figures_externes/` ;
3. **la bibliographie** : `../references.bib` avec le style `../plainnat-fr.bst` ;
4. **accepter les renvois cassés vers le corps** : ces annexes citent des chapitres du
   mémoire qui, eux, ne sont pas ici. Un document autonome imprimera des `??` sur ces
   renvois. Le remède propre est de compiler le mémoire **entier** avec ces trois `\input`
   rétablis, puis d'extraire les pages d'annexe du PDF obtenu : les renvois sont alors
   tous résolus dans les deux sens.

## Les renvois du mémoire vers ces fichiers

Les trois fichiers définissaient **78 étiquettes**. Les renvois que le corps y faisait ont été
remplacés par la macro `\matcomp`, définie dans `preambule_v2.tex` et dans `preambule.tex` :

```latex
\newcommand{\matcomp}{\emph{(voir le Matériel Complémentaire en ligne)}}
```

Elle ne contient aucun `\ref`, donc elle est insensible à la disparition des étiquettes :
le mémoire compile à zéro `??`. **Ne pas la remplacer par un `\ref`** tant que ces annexes
sont déportées.

**Le compte est de 45, et il se calcule.** Trois comptes contradictoires ont circulé ici (51 et
54) et dans l'en-tête de `main_ensae.tex`. Le bon se relève ainsi, et non de mémoire :

```powershell
# occurrences imprimées dans le PDF déposé, espaces normalisés et césure comprise
python -c "import re,pymupdf; d=pymupdf.open('main_ensae.pdf'); print(len(re.findall(r'voir le Mat', re.sub(r'\s+',' ',' '.join(p.get_text() for p in d)))))"
```

Compter sur la source est piégeux : `chapitres/*.tex` en contient 48, dont **3 dans
`02_introduction.tex`, qui est orphelin** depuis que le chapitre 1 l'absorbe, et un découpage
naïf sur `%` ampute le compte d'une unité en coupant sur un `\%` échappé.

## Ce qui est parti avec elles, et qu'il faut savoir

- **deux des trois figures du rapport LUCY** (divergence prix contre risque, répartition
  par taille de sinistre) vivaient dans l'annexe E et ne sont plus dans le mémoire. La
  troisième, la trajectoire primes, sinistres et ratio, reste : elle est appelée par les
  notes de synthèse. Hugo Rapior avait validé la reproduction des trois le 17 septembre ;
  l'autorisation tient, seul l'usage a changé ;
- ~~la table des paramètres~~ : **remise dans le mémoire le 18 septembre**, comme annexe à part
  entière (`chapitres/17b_table_parametres.tex`). Elle a été retirée d'ici, pas recopiée : le
  même tableau à deux endroits divergerait. Le chapitre 11 la cite de nouveau par un vrai
  `\ref` et non par `\matcomp` ;
- ~~la description du dispositif de vérification~~ : **retirée du matériel complémentaire le
  29 septembre 2026**, demande de Kélian. Ses comptes étaient périmés (2 208 nombres, 103
  sorties) et l'annexe G du mémoire en porte déjà l'essentiel : principe, couverture avant le
  taux, bruit publié, limite du dispositif.

Ces pertes sont assumées et signalées ici parce qu'elles portent sur du matériel
qu'un jury peut demander : le lien vers le dépôt doit être en état de marche au dépôt du
30 septembre.

## Le harnais

Ces fichiers ne sont plus balayés par `verif_tous_chapitres.ps1`, qui ne lit que
`chapitres/`. Leurs nombres ne sont donc plus sous contrôle. Au 21 septembre, le mémoire
déposé est à **1 824 nombres publiés, 1 824 confirmés** sur 22 chapitres, hors contrôle non
déclaré à zéro.
Si ces annexes reviennent dans le document, il faut repasser le harnais dessus : elles
citaient leurs scripts section par section et étaient à 100 % avant le déport.
