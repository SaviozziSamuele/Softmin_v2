#!/bin/bash
#SBATCH --job-name=run_particle_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=cl1
# CPU-only: this driver never touches a GPU. It was originally pointed at the gpu partition
# purely for the AMD nodes' core count, but that partition is saturated on CPUs by
# CPU-only jobs (58 pending, 3-day limits) while its 16 A40s sit idle. cl1 is the only
# partition with genuinely free cores and an empty queue, so the sweep starts now instead
# of queuing for weeks. Fewer workers, but wall-clock-to-first-result is far better.
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samuele.saviozzi@phd.unipi.it
#SBATCH --output=%x_%j.log
#SBATCH --time=7-00:00:00

# No `module` / `conda` on the cl1 node (lnx1) -- unlike the gpu and cl2 nodes, where the
# other job scripts' `module load anaconda3` works. The env's interpreter is invoked
# directly instead, which has been verified on lnx1 to import numpy/scipy and the
# project modules without activation.

# Codebase is NumPy-over-tiny-arrays (see .claude.md's GPU-vs-CPU note) with no BLAS-heavy hot
# path, so this is low-risk insurance rather than a fix for an observed problem -- but with
# mp.Pool workers each importing numpy/scipy, pinning threads to 1 removes the question.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

cd /scratch/saviozzi/Softmin_v2/src
./venv/bin/python run_particle_sweep.py
