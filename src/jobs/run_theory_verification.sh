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
# Verify this is still the current version with `module avail anaconda3` -- pin drifts over time.
module load anaconda3/2021.05-gcc-12.2.0

# TODO: activate the project's conda/venv environment here, e.g.:
# conda activate <your-env-name>

cd "$(dirname "$0")/.."   # src/jobs/ -> src/
python run_theory_verification.py
