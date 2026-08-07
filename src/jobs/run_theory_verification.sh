#!/bin/bash
#SBATCH --job-name=run_theory_verification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=cl2
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samuele.saviozzi@phd.unipi.it
#SBATCH --output=%x_%j.log
#SBATCH --time=7-00:00:00

module purge
module load anaconda3
eval "$(conda shell.bash hook)"

# Codebase is NumPy-over-tiny-arrays (see .claude.md's GPU-vs-CPU note) with no BLAS-heavy hot
# path, so this is low-risk insurance rather than a fix for an observed problem -- but with
# mp.Pool workers each importing numpy/scipy, pinning threads to 1 removes the question.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

cd /scratch/saviozzi/Softmin_v2/src
conda activate ./venv
python run_theory_verification.py
