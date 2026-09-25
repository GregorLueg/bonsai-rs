#!/usr/bin/env bash
#
# Score our own stage trees (and any other Newick files given), under our
# scorer, so a search can be compared stage by stage.
#
#   ./score_intermediates.sh <dir> [extra.nwk ...]
#
# Writes <dir>/stages.tsv and prints it.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dir="$(cd "$1" && pwd)"
shift
out="$dir/stages.tsv"
printf 'stage\tloglik\trf\trecovery\tn_nodes\n' > "$out"
for nwk in "$dir"/ours_stages*/*.nwk "$dir"/ours*.nwk "$@"; do
  [[ -f "$nwk" ]] || continue
  s="$(basename "$(dirname "$nwk")")/$(basename "$nwk" .nwk)"
  [[ "$s" == "$(basename "$dir")/"* ]] && s="${s#*/}"
  "$HERE/target/release/harness" score-tree "$dir" "$nwk" \
    | awk -F'\t' -v s="$s" '{printf "%s\t%s\t%s\t%s\t%s\n", s, $3, $5, $7, $9}' >> "$out"
done
column -t -s $'\t' "$out"
