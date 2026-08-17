import os

import numpy as np

from experiments import run_resumable_pool, reconstruct_pkl_files_from_csv, peak_rss_mb
from potentials import high_dim_ackley_function_setting
from simulations import SoftminEnergyAdagrad

# Fixed at d=50 -- Ackley's own existing diagnostic reference dimension (run_ackley_dimensions.py's
# beta/lr/PSO-hyperparameter tuning was all measured at d=50), so this reuses already-validated
# hyperparameters rather than requiring a fresh tuning pass. Not swept across dimension: the report
# this ablation answers (item 3) asks whether the mechanism generalizes to a harder/different
# landscape at all, not how it scales with dimension -- a single representative point is enough for
# that question, and a full 9-dimension sweep would be 9x the compute for no extra signal here.
ACKLEY_DIM = 50
ACKLEY_TARGET_VALUE = 1.0  # matches run_ackley_dimensions.py -- Ackley's global min is 0 at every d

RESULTS_CSV = 'results/master_ablation_ackley_log.csv'
# Same measured-escape-vs-budget reasoning as run_ackley_dimensions.py -- a d=50 softmin escape
# there was measured at ~92,600 steps, well inside this budget.
MAX_STEPS = 10_000_000

# Same mechanism ablation as run_ablation_experiment.py (see that file for the full rationale),
# applied to the Ackley benchmark to test whether the double-well ablation's finding generalizes to
# a different (non-well-shaped) landscape. 100 tasks total (one per seed, matching
# run_ackley_dimensions.py's own seed count) -- each task's worker_task runs all 3 modes
# sequentially, so this is NOT 300 separate run_resumable_pool tasks.
PREFACTOR_MODES = ['signed', 'positive_part', 'absolute']

HEADER = (
    ['seed'] + PREFACTOR_MODES + [f'{m}_status' for m in PREFACTOR_MODES]
    + [f'{m}_gevals' for m in PREFACTOR_MODES] + [f'{m}_oevals' for m in PREFACTOR_MODES]
    + [f'{m}_secs' for m in PREFACTOR_MODES] + ['peak_rss_mb']
)


def worker_task(args):
    """
    Isolated worker function: runs the three prefactor-mode variants of SoftminEnergyAdagrad on
    the same Ackley-d50 task and returns their entry times.
    """
    (seed,) = args
    np.random.seed(seed * 1000)

    setting = high_dim_ackley_function_setting(d=ACKLEY_DIM)
    f_vec = setting['f_vec']
    f_grad_vec = setting['f_grad_vec']
    bounds = setting['bounds']

    noise_val = 1.0
    lr = 0.001
    beta = 10.0  # matches run_ackley_dimensions.py's ACKLEY_SOFTMIN_BETA

    # Shared initial swarm across the 3 variants (2026), matching run_ackley_dimensions.py's own
    # shared-x0 pattern -- a paired comparison, unlike run_ablation_experiment.py's double-well
    # ablation (each variant there draws its own particle_init from the same sequential RNG
    # stream, an unpaired setup this new driver has no reason to inherit).
    x0 = np.random.uniform(
        [p[0] for p in setting['particle_init']], [p[1] for p in setting['particle_init']],
        (100, ACKLEY_DIM)
    )

    entry_times = []
    statuses = []
    gevals, oevals, secs = [], [], []
    for mode in PREFACTOR_MODES:
        softmin_variant = SoftminEnergyAdagrad(
            n=100, d=ACKLEY_DIM, f=f_vec, df=f_grad_vec, bounds=bounds,
            initial_lr=lr, adaptive_lr=False, noise_intensity=noise_val,
            noise="normal", beta_decay=None, prefactor_mode=mode, beta=beta
        )
        # particle_init=x0 in the constructor would be misinterpreted as a per-dimension [low,
        # high] bounds box (Simulation.__init__ only special-cases a *list*, not an ndarray, that
        # way) -- an ndarray must go through reset(), which shares the same array-vs-box dispatch
        # but treats an ndarray as literal positions (see Simulation.reset).
        softmin_variant.reset(particle_init=x0)
        entry_times.append(
            softmin_variant.first_value_time(target_value=ACKLEY_TARGET_VALUE, max_steps=MAX_STEPS)
        )
        statuses.append(softmin_variant._last_stop_reason)
        gevals.append(softmin_variant.n_gradient_evals)
        oevals.append(softmin_variant.n_objective_evals)
        secs.append(softmin_variant.wall_clock_seconds)

    return (seed, *entry_times, *statuses, *gevals, *oevals, *secs, peak_rss_mb())


def task_key(t):
    return (t[0],)


def row_key(row):
    return (int(row[0]),)


def filename_for(key, algo):
    return f"results/ablation_{algo}_ackley_dim{ACKLEY_DIM}.pkl"


def filename_for_status(key, algo):
    return f"results/ablation_{algo}_ackley_dim{ACKLEY_DIM}_status.pkl"


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Same seed count as the main Ackley sweep (run_ackley_dimensions.py), single fixed dimension.
    all_tasks = [(s,) for s in range(100)]

    run_resumable_pool(
        results_csv=RESULTS_CSV,
        header=HEADER,
        all_tasks=all_tasks,
        task_key_fn=task_key,
        row_key_fn=row_key,
        worker_fn=worker_task,
    )

    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[],     # single fixed dimension -- no sweep key, one file per algo total
        value_start=1,       # skip seed (index 0)
        algo_names=PREFACTOR_MODES,
        filename_fn=filename_for,
    )
    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[],
        value_start=1 + len(PREFACTOR_MODES),
        algo_names=PREFACTOR_MODES,
        filename_fn=filename_for_status,
        value_transform=str,
    )
