#!/usr/bin/env bash
# Passe verif_chiffres.py sur TOUS les chapitres et resume en une ligne chacun.
# Equivalent macOS / Linux de verif_tous_chapitres.ps1, meme sortie, memes colonnes.
# Usage : bash verif_tous_chapitres.sh
#
# Les accents de la sortie Python ne survivent pas toujours au pipeline :
# la regex evite donc tout caractere accentue.

set -u

ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ICI/../../.venv/bin/python"
OUT="$ICI/../../sorties_verif"

if [ ! -x "$PY" ]; then
  echo "Interpreteur introuvable : $PY" >&2
  echo "Creer l'environnement : python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

tot=0
ok=0
hors=0

# LA COLONNE « hors » EST LA PLUS IMPORTANTE DU TABLEAU. Elle compte les nombres publies
# dans une section qui ne cite aucun script : ni confirmes, ni infirmes, simplement pas
# regardes. Sans elle, on pouvait faire monter le taux en retirant une citation.
printf '%-34s%8s%10s%9s%8s%11s\n' "chapitre" "verif." "confirmes" "taux" "hors" "couverture"
printf '%.0s-' {1..80}; printf '\n'

for f in "$ICI"/chapitres/*.tex; do
  base="$(basename "$f" .tex)"
  r="$("$PY" "$ICI/verif_chiffres.py" "$OUT" "$f" 2>&1)"
  ligne="$(printf '%s\n' "$r" | grep -Eo '[0-9]+ nombres [^,]+, [0-9]+ confirm' | head -1)"
  h="$(printf '%s\n' "$r" | grep -Eo 'COUVERTURE : [0-9]+ nombres sous controle sur [0-9]+' \
       | awk '{print $8-$3}')"
  h="${h:-0}"
  # Les nombres DECLARES hors script par nature ne sont pas hors controle : ils sont hors
  # champ, avec un motif ecrit dans le chapitre. On les retire de la colonne « hors », qui
  # ne doit compter que l'omission.
  d="$(printf '%s\n' "$r" | grep -Eo '[0-9]+ nombres hors script PAR NATURE' \
       | awk '{s+=$1} END{print s+0}')"
  h=$((h - ${d:-0}))
  hors=$((hors + h))
  if [ -n "$ligne" ]; then
    v="$(printf '%s' "$ligne" | awk '{print $1}')"
    c="$(printf '%s' "$ligne" | awk '{print $4}')"
    tot=$((tot + v))
    ok=$((ok + c))
    if [ "$v" -gt 0 ]; then
      taux="$(awk -v c="$c" -v v="$v" 'BEGIN{printf "%.1f %%", 100*c/v}')"
      couv="$(awk -v v="$v" -v h="$h" 'BEGIN{printf "%.1f %%", 100*v/(v+h)}')"
    else
      taux="n/a"; couv="n/a"
    fi
    printf '%-34s%8s%10s%9s%8s%11s\n' "$base" "$v" "$c" "$taux" "$h" "$couv"
  else
    printf '%-34s%8s%10s%9s%8s%11s\n' "$base" "-" "-" "sans script" "$h" "0.0 %"
  fi
done

printf '%.0s-' {1..80}; printf '\n'
if [ "$tot" -gt 0 ]; then
  printf '%-34s%8s%10s%9s%8s%11s\n' "TOTAL" "$tot" "$ok" \
    "$(awk -v c="$ok" -v v="$tot" 'BEGIN{printf "%.1f %%", 100*c/v}')" "$hors" \
    "$(awk -v v="$tot" -v h="$hors" 'BEGIN{printf "%.1f %%", 100*v/(v+h)}')"
fi
