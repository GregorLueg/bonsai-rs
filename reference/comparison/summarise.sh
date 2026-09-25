#!/usr/bin/env bash
#
# Rebuild results.tsv from every metrics.tsv under work/, so that a sweep run in
# pieces still ends up in one table.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$HERE/results.tsv"

printf 'config\tn_leaves\tn_features\timplementation\tdistance_recovery\trobinson_foulds\trf_ours_vs_reference\tloglik\tseconds\n' > "$out"
for m in "$HERE"/work/*/metrics.tsv; do
  [[ -f "$m" ]] || continue
  d="$(dirname "$m")"
  name="$(basename "$d")"
  n="${name#n}"; n="${n%%_*}"
  p="${name##*_p}"
  tail -n +2 "$m" | while IFS= read -r line; do
    printf '%s\t%s\t%s\t%s\n' "$name" "$n" "$p" "$line" >> "$out"
  done
done

# Numeric sort by leaves then features, header kept on top.
{ head -1 "$out"; tail -n +2 "$out" | sort -t$'\t' -k2,2n -k3,3n; } > "$out.tmp"
mv "$out.tmp" "$out"
column -t -s $'\t' "$out"
