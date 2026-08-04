#!/bin/bash
#SBATCH --job-name=run_theory_verification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=cl2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samuele.saviozzi@phd.unipi.it
#SBATCH --output=%x_%j.log
#SBATCH --time=7-00:00:00

module purge
module load anaconda3
eval "$(conda shell.bash hook)"

cd "$(dirname "$0")/.."   # src/jobs/ -> src/
conda activate ./venv
python run_theory_verification.py
