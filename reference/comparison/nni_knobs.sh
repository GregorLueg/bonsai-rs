#!/usr/bin/env bash
#
# The NNI attribution experiment: re-run our side over one configuration with
# each suspect knob moved, one at a time, and score the result.
#
#   ./nni_knobs.sh <dir>
#
# Variants, each written as <dir>/ours_<tag>.nwk, steps_<tag>.tsv, moves_<tag>.tsv:
#   default     BonsaiParams::default(), the baseline, for the move counts
#   random1000  NniParams::n_random = 1000, a random-move budget where ours
#               defaults to none
#   mingain0    StarParams::min_gain = 0 on the NNI star, in case the 1e-9
#               floor rejects real moves
#   rounds50    NniParams::max_rounds = 50, to show whether the cap is ever hit
#               (the default is 10 000)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H="$HERE/target/release/harness"
dir="$(cd "$1" && pwd)"

run() {
  local tag="$1"; shift
  echo "--- $tag  (load $(uptime | sed 's/.*load averages*: //'))"
  env OURS_TAG="$tag" "$@" "$H" ours "$dir" | grep -E "nni|spr accepted|in [0-9.]+ s"
  "$H" score-tree "$dir" "$dir/ours_$tag.nwk" | sed "s|$dir/||"
}

run default
run random1000 NNI_RANDOM=1000
run mingain0 NNI_MIN_GAIN=0
run rounds50 NNI_MAX_ROUNDS=50
