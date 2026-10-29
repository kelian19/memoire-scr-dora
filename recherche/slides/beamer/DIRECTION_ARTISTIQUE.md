# Édition scientifique

Direction appliquée au support complet le 22 octobre 2026.

**État courant : 22 slides ENSAE, sans annexes**, à la demande de l'auteur.
Le PDF se termine par « Merci ». Les mentions d'annexes ci-dessous décrivent les étapes
antérieures de la refonte ; elles ne constituent plus le contenu du support livré.

Le 23 octobre, le parcours a été resserré de 28 à 22 slides principales, couverture et clôture
comprises. La méthode reste en slide 10. Six développements passent en annexes I–N et les
définitions de la descente sont regroupées en annexe O. Les animations qui répétaient une même
page sont consolidées dans le parcours principal. Le plan, le quantile, le socle, le résultat,
les canaux, la descente, le reporting, le bilan et la conclusion sont allégés.

La présentation reprend les codes d'une publication de recherche : titrage en Georgia,
texte en Calibri, fond blanc, compositions ouvertes, filets fins et grands chiffres.
La navigation Beamer en haut reprend le principe du support de Lucas : sections et pastilles
de progression, avec la slide courante mise en évidence. Son fond reprend le bleu Nexialog
`#223E55` de la charte du projet. Une miniature des huit étapes apparaît uniquement sur
les slides de méthode 11, 13, 14 et 18 : les étapes abordées sont remplies en bleu
et nommées au-dessus. Les résultats, synthèses et annexes disposent de toute la largeur.
Les titres de cartes sur fond coloré sont supprimés.
Les institutions figurent sur la couverture ; la pagination reste discrète sur les pages courantes.

La palette repose sur le graphite `#252C32`, le gris `#626A70` et le rouge `#A52D39`.
Le rouge signale le résultat principal ou une hypothèse à discuter. Les couleurs scientifiques
des figures existantes sont conservées pour respecter leurs légendes.

La slide 10 reprend la référence fournie : état et seuils à gauche, quatre canaux bleus,
trois modèles superposés, pertes et capital réunis dans le bloc rouge, puis passage à l'entité.
Les miniatures reprennent cette disposition avec les huit repères ; la matrice brute est notée $T$.
Le résultat central est redessiné en deux barres proportionnelles ;
le treillis des seize configurations est disponible en annexe H. Couverture et clôture ont
une composition typographique propre. Les références, les réserves et les formules sont conservées.

Le style est défini dans `direction_artistique.tex`, séparément du contenu scientifique.
Depuis la racine du dépôt :

```powershell
.venv/Scripts/python.exe -X utf8 outils/ci/compiler_soutenance.py
```

Ce script prépare les polices installées sur Windows dans un dossier local ignoré par Git,
puis appelle Tectonic. Les fichiers de polices ne sont pas distribués dans le dépôt ; ils sont
incorporés au PDF. Une copie du support avant refonte est conservée localement dans
`docs/collaboration/local/soutenance_avant_da.pdf`.
