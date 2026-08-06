#!/bin/bash
#SBATCH --job-name=run_lennard_jones
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --partition=gpu
# CPU-only: this driver doesn't use the GPUs on this partition, just the AMD nodes' much
# larger core count (256 threads/node vs cl2's 48) -- see the GPU-vs-CPU note in .claude.md.
# N=75 (d=225) is the heaviest condition in this sweep -- sized against the timing smoke test.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samuele.saviozzi@phd.unipi.it
#SBATCH --output=%x_%j.log
#SBATCH --time=7-00:00:00

module purge
module load anaconda3
eval "$(conda shell.bash hook)"

cd /scratch/saviozzi/Softmin_v2/src
conda activate ./venv
python run_lennard_jones.py
