# Passe verif_chiffres.py sur TOUS les chapitres et resume en une ligne chacun.
# Usage : powershell -ExecutionPolicy Bypass -File verif_tous_chapitres.ps1
#
# Les accents de la sortie Python ne survivent pas toujours au pipeline PowerShell :
# la regex evite donc tout caractere accentue.
$py = Join-Path $PSScriptRoot "..\..\.venv\Scripts\python.exe"
$out = Join-Path $PSScriptRoot "..\..\sorties_verif"
$tot = 0; $ok = 0
"{0,-34}{1,8}{2,10}{3,9}" -f "chapitre", "verif.", "confirmes", "taux"
"-" * 61
foreach ($f in Get-ChildItem (Join-Path $PSScriptRoot "chapitres") -Filter *.tex | Sort-Object Name) {
  $r = & $py (Join-Path $PSScriptRoot "verif_chiffres.py") $out $f.FullName 2>&1
  $m = $r | Select-String -Pattern '(\d+) nombres v.rifiables, (\d+) confirm'
  if ($m) {
    $v = [int]$m.Matches[0].Groups[1].Value
    $c = [int]$m.Matches[0].Groups[2].Value
    $tot += $v; $ok += $c
    $taux = if ($v -gt 0) { "{0:N1} %" -f (100 * $c / $v) } else { "n/a" }
    "{0,-34}{1,8}{2,10}{3,9}" -f $f.BaseName, $v, $c, $taux
  } else {
    "{0,-34}{1,8}{2,10}{3,9}" -f $f.BaseName, "-", "-", "sans script"
  }
}
"-" * 61
if ($tot -gt 0) {
  "{0,-34}{1,8}{2,10}{3,9}" -f "TOTAL", $tot, $ok, ("{0:N1} %" -f (100 * $ok / $tot))
}
