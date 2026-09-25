#!/usr/bin/env bash
#
# Re-run the scoring step over every configuration already in work/, without
# re-running either implementation. Useful after a change to the metrics.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for d in "$HERE"/work/*/; do
  [[ -f "$d/truth.nwk" ]] || continue
  echo "--- $(basename "$d")"
  "$HERE/target/release/harness" score "$d"
done
"$HERE/summarise.sh"
