#!/bin/bash
#SBATCH --job-name=probe_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=%x_%j.log
#SBATCH --time=01:00:00
# CANARY ONLY. Measures the real A40 per-step cost at every sweep depth so the ~29 GPU-hour
# projection in the plan can be checked against a walltime limit BEFORE a two-day job is queued.
# Deliberately short: if this overruns an hour, that is itself the answer.

module purge
module load anaconda3
eval "$(conda shell.bash hook)"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

cd /scratch/saviozzi/Softmin_v2/src
conda activate ./venv

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0), flush=True)"
python probe_depth_feasibility.py
