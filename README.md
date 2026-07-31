# Global Optimization via Softmin Energy Minimization

Code accompanying the paper *Global Optimization via Softmin Energy Minimization*. This repository implements the Softmin Energy interaction $J_\beta(\mathbf{x})$ — an interacting-particle scheme that couples particles through their relative objective values to escape local minima faster than standard stochastic global-optimization methods — together with the baselines it's compared against (Simulated Annealing, overdamped Langevin diffusion, Consensus-Based Optimization, and Particle Swarm Optimization) and the experiment/plotting pipeline used to
produce the paper's figures.

## Repository layout

```
.
├── src/
│   ├── simulations.py          # Simulation base class + all algorithm implementations
│   ├── potentials.py           # Benchmark objective functions (double well, Ackley, ...)
│   ├── viz.py                  # Plotting helpers
│   ├── experiments.py          # Resumable multiprocessing experiment harness
│   ├── run_*.py                # Experiment drivers (see "Running experiments" below)
│   ├── jobs/                   # SLURM sbatch scripts, one per driver
│   ├── ICML_submission_code.ipynb  # Notebook that renders every figure from results/*.pkl
│   ├── requirements.txt        # Pinned Python dependencies
│   ├── results/                # Output of new (post-review) experiment runs -- starts empty
│   └── results_NIPS/           # Archived .pkl/.csv data from the original NeurIPS submission
└── paper/                      # Paper source (LaTeX)
```

## Installation

```bash
pip install -r src/requirements.txt
```

NumPy, SciPy, and sympy versions are pinned deliberately — the published figures depend on exact numerical behavior that can shift subtly across minor versions.

## Reproducing the figures

`src/results/` starts empty in this revision cycle — the `.pkl` data behind the original
submission's figures has been archived to `src/results_NIPS/` (see "Running experiments" below
for why: nearly every series needs a fresh cluster run to reflect the post-review code changes).
Once `src/results/` has been populated by the drivers below:

```bash
cd src
jupyter notebook ICML_submission_code.ipynb
```

Run all cells top to bottom. The notebook contains no simulation logic itself; it only loads
`results/*.pkl` and renders comparison plots. Until then, the cells that depend on `results/`
data will raise `FileNotFoundError` — expected, not a bug.

## Running experiments

Each `src/run_*.py` script is a self-contained, resumable experiment sweep:

| Script | What it sweeps |
|---|---|
| `run_experiments_resumable.py` | Double well / quadruple well / Ackley (radius), all baseline algorithms |
| `run_ackley_dimensions.py` | High-dimensional Ackley, dimensions 10–90 |
| `run_theory_verification.py` | 1D asymmetric double well, exit time from the maximizing regime |
| `run_ablation_experiment.py` | Ascent-descent mechanism ablation (signed / positive-part / absolute prefactor) |
| `run_convergence_diagnostics.py` | Noise / learning-rate / particle-count sweeps checking the Theorem 3.1 exit-time bound |

Each is resumable: if interrupted, re-running the same script picks up where it left off (it skips any `(experiment, parameter, seed)` combination already logged in its CSV). They're designed to run on a cluster with many cores available for `multiprocessing.Pool`, not on a laptop.

## Running on the Toeplitz cluster

These scripts are written for the Dipartimento di Matematica's Toeplitz cluster (`toeplitz.cs.dm.unipi.it`, SLURM-managed).

1. **Clone the repo on the cluster.** This is a private repo, so plain HTTPS won't work
   unauthenticated. From the cluster, either:
   - Generate a dedicated SSH keypair *on Toeplitz* (`ssh-keygen -t ed25519`, no passphrase or a stored one — this key never needs to touch anything but GitHub) and add its public half as a read-only **Deploy Key** under the repo's GitHub Settings → Deploy keys, then
     `git clone git@github.com:SaviozziSamuele/Softmin_v2.git`.
   - Or use a fine-grained Personal Access Token scoped to just this repo:
     `git clone https://<token>@github.com/SaviozziSamuele/Softmin_v2.git` (or let `git` prompt for username + token as the password).

   The repo includes `src/results_NIPS/` (~600 tracked `.pkl`/`.csv` files, archived from the
   original submission), so the initial clone is larger than the code alone.

2. **Submit a job:**
   ```bash
   cd Softmin_v2/src
   sbatch jobs/run_theory_verification.sh
   squeue -u $USER
   ```
   Fill in each script's conda/venv activation `TODO` before first use — see `src/jobs/`.

3. **Branching:** consider keeping a dedicated branch (e.g. `cluster`) that you fast-forward from `master` right before submitting a batch of jobs, rather than pointing the cluster checkout directly at `master`. Long sweeps can run for a long time; if `master` moves while a job is mid-run, you lose the guarantee that a given `results/*.pkl` came from the code you think it did. A dedicated branch (or just noting the commit hash per submission) keeps results reproducible without slowing down day-to-day development on `master`.

## Paper

`paper/main.tex` is the submitted paper; `paper/SoftMin_revised_theory.tex` is an in-progress theory revision — where the two disagree, the revised file is the source of truth for what the code should implement.
