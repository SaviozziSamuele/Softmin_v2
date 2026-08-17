#!/bin/bash
#SBATCH --job-name=run_ablation_ackley
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --partition=gpu
# CPU-only: this driver doesn't use the GPUs on this partition, just the AMD nodes' much
# larger core count -- see the GPU-vs-CPU note in .claude.md.
# 100 (not 256): run_ablation_ackley.py has 100 tasks total (one per seed, d=50 fixed; each
# task's worker_task runs all 3 prefactor-mode variants sequentially), so run_resumable_pool's
# mp.Pool(SLURM_CPUS_PER_TASK) never has more than 100 tasks to hand out -- any cores above that
# would just sit idle for the whole run, same reasoning as run_lennard_jones.sh's 120 cap.
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
# mp.Pool workers each importing numpy/scipy, pinning threads to 1 removes the question.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

cd /scratch/saviozzi/Softmin_v2/src
conda activate ./venv
python run_ablation_ackley.py
