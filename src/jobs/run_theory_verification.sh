#!/bin/bash
#SBATCH --job-name=run_theory_verification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=cl2
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
python run_theory_verification.py
