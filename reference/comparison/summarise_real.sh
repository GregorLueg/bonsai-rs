#!/usr/bin/env bash
#
# Rebuild results_real.tsv from every work_real/n*/metrics.tsv, and
# steps_real.tsv from every work_real/n*/steps.tsv, so that a ladder run in
# pieces still ends up in two tables. Load averages (one-minute, before the
# run) are carried along because the machine is shared and absolute seconds
# mean nothing without them.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$HERE/results_real.tsv"
steps="$HERE/steps_real.tsv"

load_of() {
  # First field of the "before" line of a load file, or NA.
  [[ -f "$1" ]] && awk -F'\t' '$1=="before"{split($2,a," "); print a[1]}' "$1" || echo NA
}

printf 'n_cells\tn_features\timplementation\tdistance_recovery\trobinson_foulds\trf_ours_vs_reference\tloglik\tseconds\tload_before\tpeak_rss_mb\n' > "$out"
printf 'n_cells\tn_features\tstep\tseconds\tloglik\tgain\tshare\n' > "$steps"
for d in "$HERE"/work_real/n*/; do
  [[ -f "$d/metrics.tsv" ]] || continue
  n="$(basename "$d")"; n="${n#n}"
  p="$(awk -F'\t' '$1=="n_features_ours"{print $2}' "$d/prep.tsv" 2>/dev/null || echo NA)"
  ours_load="$(load_of "$d/ours_load.txt")"
  theirs_load="$(load_of "$d/theirs_load.txt")"
  theirs_rss="$(cat "$d/theirs_peak_rss_mb.txt" 2>/dev/null || echo NA)"
  tail -n +2 "$d/metrics.tsv" | while IFS=$'\t' read -r impl rest; do
    case "$impl" in
      bonsai-rs) l="$ours_load"; r="NA" ;;
      reference) l="$theirs_load"; r="$theirs_rss" ;;
      *) l="NA"; r="NA" ;;
    esac
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$p" "$impl" "$rest" "$l" "$r" >> "$out"
  done
  if [[ -f "$d/steps.tsv" ]]; then
    total="$(awk -F'\t' 'NR>1{s+=$2} END{print s}' "$d/steps.tsv")"
    tail -n +2 "$d/steps.tsv" | while IFS=$'\t' read -r step secs loglik gain _; do
      share="$(echo "scale=3; $secs / $total" | bc)"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$p" "$step" "$secs" "$loglik" "$gain" "$share" >> "$steps"
    done
  fi
done

{ head -1 "$out"; tail -n +2 "$out" | sort -t$'\t' -k1,1n -k3,3; } > "$out.tmp"; mv "$out.tmp" "$out"
{ head -1 "$steps"; tail -n +2 "$steps" | sort -t$'\t' -k1,1n -k3,3; } > "$steps.tmp"; mv "$steps.tmp" "$steps"
column -t -s $'\t' "$out"
echo
column -t -s $'\t' "$steps"
