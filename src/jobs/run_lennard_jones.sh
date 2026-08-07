#!/bin/bash
#SBATCH --job-name=run_lennard_jones
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --partition=gpu
# CPU-only: this driver doesn't use the GPUs on this partition, just the AMD nodes' much
# larger core count (256 threads/node vs cl2's 48) -- see the GPU-vs-CPU note in .claude.md.
# N=75 (d=225) is the heaviest condition in this sweep -- sized against the timing smoke test.
# 120 (not 256): N_ATOMS_LIST x SEEDS = 4 x 30 = 120 tasks total, so run_resumable_pool's
# mp.Pool(SLURM_CPUS_PER_TASK) never has more than 120 tasks to hand out -- any cores above that
# would just sit idle for the whole multi-day run.
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
python run_lennard_jones.py
