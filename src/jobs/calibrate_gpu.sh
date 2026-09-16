#!/bin/bash
#SBATCH --job-name=calibrate_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# gpu01 EXCLUDED (2026-09-16): its glusterfs client wedges on directory lookups under /scratch.
# Measured there: ls of the venv bin directory times out after 15s while stat of a plain file in
# the same tree returns instantly, and torch import blocks in FUSE request_wait_answer forever --
# two jobs (45487, 45506) burned a GPU allocation doing nothing, while the same script ran fine on
# gpu04. Remove this line once the node is fixed; it costs a quarter of the GPU capacity.
#SBATCH --exclude=gpu01
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=samuele.saviozzi@phd.unipi.it
#SBATCH --output=%x_%j.log
#SBATCH --time=08:00:00
# Stage 1 of 2 for the transformer benchmark: measure, then calibrate. Deliberately NOT chained
# into the sweep. run_transformer.py reads LR / NOISE_SIGMA as hardcoded constants, so a chained
# sweep would run on the provisional depth-1 values whatever the calibration below concludes --
# including a stage-D "DIVERGED" verdict at depth 50. The results are reviewed and the constants
# updated before jobs/run_transformer.sh is submitted.

# No `module` or `conda` on the compute nodes (verified the hard way: job 45447 died
# instantly with `module: command not found`). The venv interpreter is invoked
# directly, which needs no activation.

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

cd /scratch/saviozzi/Softmin_v2/src

./venv/bin/python -u -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0), flush=True)"

# Real A40 per-step cost at every sweep depth, and the sweep projection that follows from it.
./venv/bin/python -u probe_depth_feasibility.py

# LR / clip / beta / sigma at depth 10, then the stage-D stability re-check at depth 50.
./venv/bin/python -u calibrate_transformer.py
