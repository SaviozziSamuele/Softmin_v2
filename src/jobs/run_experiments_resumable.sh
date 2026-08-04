#!/bin/bash
#SBATCH --job-name=run_experiments_resumable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --partition=gpu
# CPU-only: this driver doesn't use the GPUs on this partition, just the AMD nodes' much
# larger core count (256 threads/node vs cl2's 48) -- see the GPU-vs-CPU note in .claude.md.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=samuele.saviozzi@phd.unipi.it
#SBATCH --output=%x_%j.log
# No --time: partition timelimit is "infinite" and this driver is resumable (safe to
# re-submit this same script after a failure/preemption -- run_resumable_pool skips
# already-completed rows in its CSV log). Add --time=<d-hh:mm:ss> here if your allocation
# policy requires an explicit cap.

module purge
module load anaconda3
eval "$(conda shell.bash hook)"

cd "$(dirname "$0")/.."   # src/jobs/ -> src/
conda activate ./venv
python run_experiments_resumable.py
