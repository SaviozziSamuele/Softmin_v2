#!/bin/bash
#SBATCH --job-name=run_transformer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# The one driver in this project that genuinely needs a GPU. Every other driver is NumPy over tiny
# arrays and uses --partition=gpu purely for the AMD nodes' core count (see .claude.md's
# GPU-vs-CPU note); this one runs n=20 batched transformer forward+backward passes per step via
# torch.func.vmap, which is real GPU work. Only 8 CPUs are requested because the parallelism is
# inside each task (the vmap'd particle batch), not across tasks -- run_transformer.py runs tasks
# serially in one process on purpose, since CUDA does not survive multiprocessing's fork.
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samuele.saviozzi@phd.unipi.it
#SBATCH --output=%x_%j.log
#SBATCH --time=2-00:00:00

module purge
module load anaconda3
eval "$(conda shell.bash hook)"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

cd /scratch/saviozzi/Softmin_v2/src
conda activate ./venv

# Fail loudly rather than silently spending two days on CPU: this driver is only worth submitting
# to the gpu partition if torch can actually see the A40.
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0))"

# CANARY. The whole budget rests on an ASSUMED GPU speedup over the CPU numbers measured in
# probe_depth_feasibility.py (23,000 CPU-hours at the pre-cut budget; the cut configuration is
# ~1/16 of that, so ~29 GPU-hours only if the GPU is ~50x the probe machine). Measure it before
# committing two days of wall-clock: this prints the real per-step cost at the deepest point, and
# the projection that follows from it.
python probe_depth_feasibility.py

# Recalibrate at the mid-sweep depth before the sweep itself. The shipped LR/beta/sigma were
# selected on a 1-layer model and have no claim to transfer to a 50-layer one; stage D of this
# script re-checks the selection at depth 50 and says outright whether it diverges or stalls.
python calibrate_transformer.py

python run_transformer.py
