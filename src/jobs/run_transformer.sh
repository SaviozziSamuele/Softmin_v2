#!/bin/bash
#SBATCH --job-name=run_transformer
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

# No `module` or `conda` on the compute nodes (verified the hard way: job 45447 died
# instantly with `module: command not found`). The venv interpreter is invoked
# directly, which needs no activation.

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

cd /scratch/saviozzi/Softmin_v2/src

# Fail loudly rather than silently spending two days on CPU: this driver is only worth submitting
# to the gpu partition if torch can actually see the A40.
./venv/bin/python -u -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0))"

# Stage 2 of 2. Submit only after jobs/calibrate_gpu.sh has run and its results have been
# reviewed: LR and NOISE_SIGMA in run_transformer.py are constants, not read from the
# calibration output, so they must be updated by hand first if calibration moved them.
# Arguments pass through, so the same script runs the whole sweep or a single shard, e.g.
#   sbatch jobs/run_transformer.sh --tasks 50:0 50:1 50:2 --tag a
# (jobs/submit_transformer_split.sh submits a balanced set of shards).
./venv/bin/python -u run_transformer.py "$@"
