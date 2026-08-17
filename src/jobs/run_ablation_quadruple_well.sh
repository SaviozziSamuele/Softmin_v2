#!/bin/bash
#SBATCH --job-name=run_ablation_quadruple_well
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --partition=gpu
# CPU-only: this driver doesn't use the GPUs on this partition, just the AMD nodes' much
# larger core count (256 threads/node vs cl2's 48) -- see the GPU-vs-CPU note in .claude.md.
# Same shape/cost as run_ablation_experiment.py's double-well ablation (2900 tasks x 3
# prefactor-mode variants, MAX_STEPS=10_000_000), so the same sizing applies unchanged.
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samuele.saviozzi@phd.unipi.it
#SBATCH --output=%x_%j.log
#SBATCH --time=7-00:00:00

module purge
module load anaconda3
eval "$(conda shell.bash hook)"

# Codebase is NumPy-over-tiny-arrays (see .claude.md's GPU-vs-CPU note) with no BLAS-heavy hot
# path, so this is low-risk insurance rather than a fix for an observed problem -- but with
# mp.Pool(256) workers each importing numpy/scipy, pinning threads to 1 removes the question.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

cd /scratch/saviozzi/Softmin_v2/src
conda activate ./venv
python run_ablation_quadruple_well.py
