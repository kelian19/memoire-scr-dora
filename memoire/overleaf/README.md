# Version Overleaf avec le fond Nexialog

`main.tex` est le fichier principal à placer à la racine du projet Overleaf, à côté de
`chapitres_inst/`, `preambule_v2.tex`, `page_de_garde.tex` et de la bibliographie.
Il conserve le contenu Institut, ajoute le fond Nexialog sur chaque page et retire le titre
du chapitre et le filet supérieur pour éviter leur chevauchement avec le logo. La pagination
reste en haut à droite.

Les figures sont recherchées dans `figures_memoire/`, `figures_pedagogie/`,
`figures_externes/` et `logos/`, avec les chemins de l'export initial en complément.
Le compilateur à sélectionner est **XeLaTeX**.

Depuis la racine du dépôt, créer un export dans un dossier neuf :

```powershell
.venv/Scripts/python.exe -X utf8 outils/ci/exporter_overleaf_institut.py docs/collaboration/local/export_nexialog --nexialog
```

Conserver ou ajouter **`RequiredFiles/fond_latex_2.pdf`** depuis le projet Nexialog.
Ce fond n'est pas disponible dans le dépôt et n'est donc pas inclus dans l'export.

Sans `--nexialog`, l'exporteur utilise `memoire/main_institut_150.tex` et produit la
version Institut sans fond Nexialog. Pour actualiser le contenu de cette variante,
répercuter les changements du document principal Institut tout en conservant les réglages
de fond, d'en-tête et de chemins propres à Overleaf.
