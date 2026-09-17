#!/bin/bash
#SBATCH --job-name=extra_seeds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# gpu01 excluded for the glusterfs fault documented in jobs/run_transformer.sh.
#SBATCH --exclude=gpu01
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=samuele.saviozzi@phd.unipi.it
#SBATCH --output=%x_%j.log
# Measured cost is ~2.4 h per shard (Softmin at 2.53 s/step at depth 50, 0.89 at depth 20);
# 12 h leaves room for stalls like the one this run exists to characterise.
#SBATCH --time=12:00:00

# Softmin-only extra seeds at depths 20 and 50 (extra_seeds_transformer.py). Arguments pass
# through; jobs/submit_extra_seeds.sh submits the balanced 4-shard split.

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

cd /scratch/saviozzi/Softmin_v2/src

./venv/bin/python -u -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0))"

./venv/bin/python -u extra_seeds_transformer.py "$@"
