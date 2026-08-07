#!/bin/bash
# One-time (and per-session) environment setup for Softmin_v2 on the Toeplitz cluster.
#
# First time on a fresh clone:
#   bash setup_env.sh
#
# Every session after that, just source the activation part:
#   module load anaconda3
#   eval "$(conda shell.bash hook)"
#   conda activate ./venv

set -e

cd "$(dirname "$0")"

module load anaconda3
eval "$(conda shell.bash hook)"

# /tmp on the login node fills up fast (root fs, ~1GB free) -- redirect pip's
# temp/download dir to scratch before installing anything.
mkdir -p "/scratch/$USER/tmp"
export TMPDIR="/scratch/$USER/tmp"

if [ ! -d "./venv" ]; then
    conda create -p ./venv python=3.11 -y
fi

conda activate ./venv

# requirements.txt no longer carries pywin32 (or any other Windows-only / unrelated-project
# package), so it installs directly on Linux -- no more stripping into a second file.
pip install -r src/requirements.txt

echo "Done. To activate this env in future sessions, run:"
echo '  module load anaconda3 && eval "$(conda shell.bash hook)" && conda activate ./venv'
