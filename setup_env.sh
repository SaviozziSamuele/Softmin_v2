#!/bin/bash
# One-time (and per-session) environment setup for Softmin_v2 on the Toeplitz cluster.
#
# First time on a fresh clone:
#   bash setup_env.sh
#
# Every session after that, just source the activation part:
#   cd src && module load anaconda3 && eval "$(conda shell.bash hook)" && conda activate ./venv

set -e

# The job scripts cd into src/ and then `conda activate ./venv`, so the env must live at
# src/venv, not at the repo root -- venv/ creation and activation both happen from inside src/
# to match that exactly (2026-08-07: an env created at the repo-root path was silently unusable
# by every job script, which look for it at src/venv).
cd "$(dirname "$0")/src"

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
pip install -r requirements.txt

echo "Done. To activate this env in future sessions, run:"
echo '  cd src && module load anaconda3 && eval "$(conda shell.bash hook)" && conda activate ./venv'
