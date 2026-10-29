# Cree des raccourcis Windows vers les fichiers existants, sans les deplacer.
# Depuis la racine : powershell -NoProfile -ExecutionPolicy Bypass -File outils/ci/creer_acces_rapide.ps1
# L'option ne change la politique que pour ce processus PowerShell.
$ErrorActionPreference = 'Stop'
$racineProjet = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$dossierAcces = [IO.Path]::GetFullPath((Join-Path $racineProjet 'outils/acces_rapide'))

# Dossier de destination, nom du raccourci, chemin relatif de sa cible.
$entreesAcces = @(
    @('01_Lire_le_memoire', '01_Memoire_ENSAE', 'memoire/main_ensae.pdf'),
    @('01_Lire_le_memoire', '02_Memoire_Institut', 'memoire/main_institut.pdf'),
    @('01_Lire_le_memoire', '03_Variante_Institut_150', 'memoire/main_institut_150.pdf'),
    @('01_Lire_le_memoire', '04_Soutenance', 'recherche/slides/soutenance_memoire_DORA_finale_19_octobre.pdf'),
    @('01_Lire_le_memoire', '05_Cours_de_preparation', 'recherche/vasicek_lab/notes/cours_memoire_ensae.pdf'),
    @('01_Lire_le_memoire', '06_Complements', 'memoire/materiel_complementaire'),
    @('02_Sources_et_redaction', '01_Chapitres_ENSAE', 'memoire/chapitres_sfe'),
    @('02_Sources_et_redaction', '02_Source_principale_ENSAE', 'memoire/main_ensae.tex'),
    @('02_Sources_et_redaction', '03_Bibliographie_ENSAE', 'memoire/references_ensae.bib'),
    @('02_Sources_et_redaction', '04_Guide_de_redaction', 'docs/redaction'),
    @('02_Sources_et_redaction', '05_Dossier_des_manuscrits', 'memoire'),
    @('03_Calculs_et_donnees', '01_Laboratoire', 'recherche/vasicek_lab'),
    @('03_Calculs_et_donnees', '02_Modules', 'src'),
    @('03_Calculs_et_donnees', '03_Donnees_locales', 'data'),
    @('03_Calculs_et_donnees', '04_Sorties_verifiees', 'resultats/sorties_verif'),
    @('04_Controles_et_consignes', '01_Consignes', 'docs/consignes'),
    @('04_Controles_et_consignes', '02_Controles_du_manuscrit', 'memoire/controles_depot'),
    @('04_Controles_et_consignes', '03_Grilles_de_relecture', 'memoire/controles_caroline'),
    @('04_Controles_et_consignes', '04_Audits', 'docs/audits'),
    @('04_Controles_et_consignes', '05_Outils_de_verification', 'outils/ci'),
    @('05_Archives_et_guides', '01_Anciens_plans', 'docs/archives'),
    @('05_Archives_et_guides', '02_Anciens_audits', 'docs/audits/archives'),
    @('05_Archives_et_guides', '03_Carte_du_projet', 'docs/ARCHITECTURE.md'),
    @('05_Archives_et_guides', '04_Premier_manuscrit', 'archives/premier_memoire')
)

$shellRaccourcis = New-Object -ComObject WScript.Shell
$nombreAcces = 0
foreach ($entreeAcces in $entreesAcces) {
    $cibleAcces = [IO.Path]::GetFullPath((Join-Path $racineProjet $entreeAcces[2]))
    $dossierCategorie = [IO.Path]::GetFullPath((Join-Path $dossierAcces $entreeAcces[0]))
    if (-not $cibleAcces.StartsWith($racineProjet + '\', [StringComparison]::OrdinalIgnoreCase) -or
        -not $dossierCategorie.StartsWith($dossierAcces + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Un chemin de raccourci sort du projet.'
    }
    if (-not (Test-Path -LiteralPath $cibleAcces)) {
        Write-Warning ('Cible absente, raccourci ignore : ' + $entreeAcces[2])
        continue
    }
    New-Item -ItemType Directory -Path $dossierCategorie -Force | Out-Null
    $cheminAcces = Join-Path $dossierCategorie ($entreeAcces[1] + '.lnk')
    $raccourciAcces = $shellRaccourcis.CreateShortcut($cheminAcces)
    $raccourciAcces.TargetPath = $cibleAcces
    $raccourciAcces.WorkingDirectory = $racineProjet
    $raccourciAcces.Description = 'Acces au fichier original du projet de memoire'
    $raccourciAcces.Save()
    $controleAcces = $shellRaccourcis.CreateShortcut($cheminAcces)
    if ($controleAcces.TargetPath -ne $cibleAcces -or
        -not (Test-Path -LiteralPath $controleAcces.TargetPath)) {
        throw ('Raccourci invalide : ' + $cheminAcces)
    }
    $nombreAcces++
}

$aideAcces = @'
# Acces rapide au memoire

Les cinq dossiers regroupent des raccourcis vers les originaux du projet.
Un raccourci ouvre le vrai fichier : aucune copie de PDF a actualiser ici.

01_Lire_le_memoire       : PDF ENSAE, Institut, soutenance et complements.
02_Sources_et_redaction : chapitres, bibliographie, guides et sources.
03_Calculs_et_donnees   : laboratoire, modules, donnees et sorties.
04_Controles_et_consignes : consignes, relectures, audits et verifications.
05_Archives_et_guides   : anciens plans, audits et carte du depot.

Ces raccourcis Windows contiennent des chemins propres a ce poste et restent
hors de Git. Apres un deplacement du projet, les regenerer depuis sa racine :

    powershell -NoProfile -ExecutionPolicy Bypass -File outils/ci/creer_acces_rapide.ps1

Les dossiers techniques conservent leurs chemins pour les imports et LaTeX.
'@
[IO.File]::WriteAllText((Join-Path $dossierAcces 'LIRE_MOI.md'), $aideAcces, [Text.UTF8Encoding]::new($false))
Write-Output ("$nombreAcces raccourcis crees et verifies dans outils/acces_rapide.")
