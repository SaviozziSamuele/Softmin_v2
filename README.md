# Global Optimization via Softmin Energy Minimization

Code accompanying the paper *Global Optimization via Softmin Energy Minimization*. The repository
implements the Softmin energy interaction $J_\beta(\mathbf{x})$ — an interacting-particle scheme that
couples particles through their relative objective values in order to escape local minima — together
with the baselines it is compared against (Simulated Annealing, overdamped Langevin dynamics,
Consensus-Based Optimization, Particle Swarm Optimization, and, on the neural-network benchmark, Adam
and SGD), the experiment drivers, and the notebook that renders every figure in the paper.

## Repository layout

```
.
├── src/
│   ├── simulations.py              # Simulation base class + the seven particle dynamics
│   ├── potentials.py               # Analytic benchmarks (wells, Ackley, Lennard-Jones) — NumPy only
│   ├── nn_objective.py             # PyTorch transformer objective, same `setting` contract
│   ├── experiments.py              # Shared resumable multiprocessing harness
│   ├── viz.py                      # Plotting helpers and the shared paper figure style
│   ├── run_*.py                    # Experiment drivers (see "Experiment drivers" below)
│   ├── hyperparameter_diagnostics.py   # Hyperparameter calibration for the analytic benchmarks
│   ├── calibrate_transformer.py        # The same calibration for the transformer benchmark
│   ├── probe_depth_feasibility.py      # Cost/feasibility probe for the transformer depth sweep
│   ├── confirm_particle_anomaly.py     # Re-runs the A=30 entry-time anomaly on fresh seeds
│   ├── extra_seeds_transformer.py      # Extra Softmin seeds characterizing one slow run
│   ├── plot_quadruple_well_landscape.py # Visual check of the quadruple-well sweep bounds
│   ├── jobs/                       # SLURM sbatch scripts, one per driver
│   ├── ICML_submission_code.ipynb  # Notebook rendering every figure from results/
│   ├── requirements.txt            # Dependencies (see "Installation")
│   ├── results/                    # Current results: .pkl per run + master_*.csv logs (tracked)
│   └── results_NIPS/               # Archived data from the earlier submission, kept for reference
└── paper/                          # Paper source (LaTeX)
```

## Installation

```bash
pip install -r src/requirements.txt
```

NumPy, SciPy, sympy, matplotlib and pandas are pinned deliberately: the published figures depend on
exact numerical behavior that can shift across minor versions. PyTorch is deliberately left unpinned
and is needed only for the transformer benchmark; install the CUDA build on a GPU machine, while the
CPU wheel is enough to run everything else. `requirements_linux.txt` is a full freeze of the Linux
environment used on the cluster, kept as a record of what actually ran.

## Reproducing the figures

`src/results/` is populated and tracked, so the figures can be regenerated without re-running any
experiment:

```bash
cd src
jupyter notebook ICML_submission_code.ipynb
```

Run the cells top to bottom. The notebook contains no simulation logic: it loads `results/*.pkl` and
`results/master_*.csv` and writes each figure to `paper/images/`. A cell whose inputs are missing
reports that the corresponding driver has to be run first, rather than failing silently.

## Experiment drivers

Each `src/run_*.py` script is a self-contained sweep that appends to its own `master_*.csv` in
`results/`, and each is resumable: re-running it skips every `(experiment, parameter, seed)`
combination already logged. They assume many cores for `multiprocessing.Pool`, so they belong on a
cluster rather than a laptop; the transformer driver is the one that genuinely needs a GPU, and it
runs its tasks serially in one process because CUDA does not survive `fork`.

| Driver | Sweep | Log |
|---|---|---|
| `run_experiments_resumable.py` | Double well and quadruple well, all baselines | `master_experiment_log_decay.csv` |
| `run_lennard_jones.py` | Lennard-Jones clusters, N = 13, 38, 55, 75 | `master_lennard_jones_log.csv` |
| `run_ackley_dimensions.py` | Ackley in 10–90 dimensions | `master_high_dim_ackley_log_exp_on_dims.csv` |
| `run_particle_sweep.py` | Ensemble size 5–400, Softmin vs. Simulated Annealing on the quadruple well | `master_particle_sweep.csv` |
| `run_transformer.py` | Transformer depth 2–50 (GPU) | `master_transformer.csv` |
| `run_theory_verification.py` | Small-noise exit times against the barrier-flattening bound | `master_theory_verification_init0_0p05.csv` |
| `run_convergence_diagnostics.py` | Noise, learning-rate and particle-count sensitivity of those exit times | `master_convergence_diagnostics.csv` |
| `run_ablation_experiment.py`, `run_ablation_quadruple_well.py` | Ascent-descent ablation (signed / positive-part / absolute pre-factor) on both wells | `master_ablation_log.csv` |
| `run_ablation_ackley.py`, `run_ablation_lj.py` | The same ablation at fixed hard settings (Ackley d=50, Lennard-Jones N=38) | `master_ablation_ackley_log.csv`, `master_ablation_lj_log.csv` |
| `run_ablation_beta2_quadruple_well.py`, `..._hybrid.py` | The same ablation at β=2, where the Gibbs weight no longer masks the sign convention | `master_ablation_beta2_quadruple_well*.csv` |

Supporting scripts: `hyperparameter_diagnostics.py` and `calibrate_transformer.py` implement the
two-stage calibration applied identically to Softmin and to every baseline;
`probe_depth_feasibility.py`, `confirm_particle_anomaly.py` and `extra_seeds_transformer.py` are the
targeted follow-up runs reported in the appendix.

## Running on a cluster

The `src/jobs/*.sh` scripts are SLURM batch scripts, one per driver. They assume a cluster with a
`gpu` partition (used for its core count by the CPU-only drivers, and for an actual GPU by the
transformer driver), and they invoke a virtual-environment interpreter by absolute path rather than
relying on `module` or `conda` being available on the compute nodes.

Before first use, adapt the site-specific lines in each script: the working directory, the
interpreter path, the partition and node constraints, and the notification address. Then:

```bash
cd src
sbatch jobs/run_theory_verification.sh
squeue -u $USER
```

Long sweeps are worth pinning to a fixed revision — a dedicated branch, or a commit hash recorded per
submission — so that a given `results/*.pkl` can always be traced to the code that produced it.

## Paper

`paper/` holds the LaTeX sources for the paper and its appendices, including the figures the notebook
writes into `paper/images/`.
