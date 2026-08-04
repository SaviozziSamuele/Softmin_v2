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
mkdir -p /scratch/saviozzi/tmp
export TMPDIR=/scratch/saviozzi/tmp

if [ ! -d "./venv" ]; then
    conda create -p ./venv python=3.11 -y
fi

conda activate ./venv

grep -v -i '^pywin32' src/requirements.txt > src/requirements_linux.txt
pip install -r src/requirements_linux.txt

echo "Done. To activate this env in future sessions, run:"
echo '  module load anaconda3 && eval "$(conda shell.bash hook)" && conda activate ./venv'
