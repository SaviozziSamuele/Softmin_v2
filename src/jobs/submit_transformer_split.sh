#!/bin/bash
# Submit the transformer sweep as balanced single-GPU shards. Run from src/:
#   bash jobs/submit_transformer_split.sh 3     # or 4, if a fourth A40 is free
# When every shard has finished:
#   ./venv/bin/python run_transformer.py --merge
#
# Cost is roughly linear in depth, so the sweep's 87 depth-units (2+5+10+20+50) are balanced by
# splitting depth 50 across its seeds, 10 units per seed. Splitting by depth alone would leave
# depth 50 as one 50-unit job setting the wall-clock no matter how many GPUs the rest get.
#
#   3 GPUs (makespan ~30 units, ~2.9x over serial)
#     a: 50:0 50:1 50:2          30
#     b: 20 10                   30
#     c: 50:3 50:4 5 2           27
#
#   4 GPUs (makespan ~25 units, ~3.5x over serial)
#     a: 50:0 50:1               20
#     b: 50:2 50:3               20
#     c: 50:4 10 2               22
#     d: 20 5                    25
#
# Depth 50 is listed first in each shard so the most expensive tasks, and any budget or memory
# problem at the deep end, surface early rather than after the cheap ones.
set -euo pipefail
cd "$(dirname "$0")/.."

N_GPUS="${1:?usage: bash jobs/submit_transformer_split.sh 3|4}"

submit() {
    local tag="$1"; shift
    sbatch --job-name="transformer_${tag}" jobs/run_transformer.sh --tag "$tag" --tasks "$@"
}

case "$N_GPUS" in
    3)
        submit a 50:0 50:1 50:2
        submit b 20 10
        submit c 50:3 50:4 5 2
        ;;
    4)
        submit a 50:0 50:1
        submit b 50:2 50:3
        submit c 50:4 10 2
        submit d 20 5
        ;;
    *)
        echo "expected 3 or 4, got: $N_GPUS" >&2
        exit 1
        ;;
esac
