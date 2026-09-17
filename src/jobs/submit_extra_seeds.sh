#!/bin/bash
# Submit extra_seeds_transformer.py as 4 balanced single-GPU shards. Run from src/:
#   bash jobs/submit_extra_seeds.sh
# When every shard has finished:
#   ./venv/bin/python extra_seeds_transformer.py --merge
#
# Balanced on measured Softmin cost, depth 50 = 3.5 units per seed and depth 20 = 1 unit
# (1069 s vs 303 s), 112.5 units in all, ~28 per shard.
#
# The five reproduction seeds (depth 50, seeds 0-4) are placed FIRST, with seed 2 -- the outlier
# itself -- at the head of shard a. If those do not reproduce run_transformer.py's numbers
# (333/335/743/363/342) the rest of the run is not comparable with the sweep, so that failure
# should surface in the first ~20 minutes, not after the new seeds.
set -euo pipefail
cd "$(dirname "$0")/.."

seeds() { local depth="$1"; shift; for s in "$@"; do printf '%s:%s ' "$depth" "$s"; done; }

submit() {
    local tag="$1"; shift
    # shellcheck disable=SC2046,SC2086
    sbatch --job-name="extra_seeds_${tag}" jobs/extra_seeds_transformer.sh --tag "$tag" --tasks $@
}

submit a $(seeds 50 2 0 4 8 12 16 20 24)                            # 8 x 3.5      = 28.0
submit b $(seeds 50 1 5 9 13 17 21)  $(seeds 20 $(seq 0 6))         # 21.0 + 7     = 28.0
submit c $(seeds 50 3 6 10 14 18 22) $(seeds 20 $(seq 7 13))        # 21.0 + 7     = 28.0
submit d $(seeds 50 7 11 15 19 23)   $(seeds 20 $(seq 14 24))       # 17.5 + 11    = 28.5
