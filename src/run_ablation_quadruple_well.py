import os

import numpy as np

from experiments import run_resumable_pool, reconstruct_pkl_files_from_csv, peak_rss_mb
from potentials import unbalanced_quadruple_well_setting
from run_experiments_resumable import _well_target_value
from simulations import SoftminEnergyAdagrad

# Kept as a separate file/CSV from run_ablation_experiment.py's double-well ablation, rather than
# folding this in via an exp_name-keyed task tuple the way run_experiments_resumable.py unifies
# double_well/quadruple_well: that file's master_ablation_log.csv already has 3000 rows on disk
# under a 2-key (parameter, seed) schema from the completed double-well cluster run. Adding an
# exp_name column in place would silently break get_completed_tasks's row_key_fn against every
# existing row (old rows have no experiment column, so row[0] would be read as a delta/seed value
# instead) -- a new file avoids that resumability corruption entirely.
RESULTS_CSV = 'results/master_ablation_quadruple_well_log.csv'
MAX_STEPS = 10_000_000

# Same mechanism ablation as run_ablation_experiment.py (see that file for the full rationale),
# applied to the quadruple-well benchmark to test whether the double-well ablation's finding
# generalizes to a 2D landscape.
PREFACTOR_MODES = ['signed', 'positive_part', 'absolute']

HEADER = (
    ['parameter', 'seed'] + PREFACTOR_MODES + [f'{m}_status' for m in PREFACTOR_MODES]
    + [f'{m}_gevals' for m in PREFACTOR_MODES] + [f'{m}_oevals' for m in PREFACTOR_MODES]
    + [f'{m}_secs' for m in PREFACTOR_MODES] + ['peak_rss_mb']
)


def worker_task(args):
    """
    Isolated worker function: runs the three prefactor-mode variants of SoftminEnergyAdagrad on
    the same quadruple-well task and returns their entry times. Mirrors
    run_ablation_experiment.py's worker_task exactly, substituting the quadruple-well setting/d.
    """
    param, seed = args
    np.random.seed(seed * 1000)

    setting = unbalanced_quadruple_well_setting(param, -1)  # param is A
    d = 2
    particle_init = setting['particle_init']

    f_vec = setting['f_vec']
    f_grad_vec = setting['f_grad_vec']
    bounds = setting['bounds']
    # Value-space target, matching run_experiments_resumable.py's quadruple-well target and
    # run_ablation_experiment.py's double-well ablation -- this ablation shares the exact same
    # benchmark, so it must share the exact same target definition.
    target_value = _well_target_value(setting)

    noise_val = 1.0
    lr = 0.001
    # Same diagnostic-guided value as the quadruple_well series in run_experiments_resumable.py
    # and as run_ablation_experiment.py's double-well ablation -- see that file's matching comment
    # for the full beta sweep (beta=10.0 at or near the optimum, quadruple_well specifically
    # markedly worse above 10, not just flat).
    beta = 10.0

    entry_times = []
    statuses = []
    gevals, oevals, secs = [], [], []
    for mode in PREFACTOR_MODES:
        softmin_variant = SoftminEnergyAdagrad(
            n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
            initial_lr=lr, adaptive_lr=False, noise_intensity=noise_val,
            bounds=bounds, noise="normal", beta_decay=None, prefactor_mode=mode, beta=beta
        )
        entry_times.append(softmin_variant.first_value_time(target_value=target_value, max_steps=MAX_STEPS))
        statuses.append(softmin_variant._last_stop_reason)
        gevals.append(softmin_variant.n_gradient_evals)
        oevals.append(softmin_variant.n_objective_evals)
        secs.append(softmin_variant.wall_clock_seconds)

    return (param, seed, *entry_times, *statuses, *gevals, *oevals, *secs, peak_rss_mb())


def task_key(t):
    return (float(t[0]), t[1])


def row_key(row):
    return (float(row[0]), int(row[1]))


def filename_for(key, algo):
    (param,) = key
    param = float(param)
    if param.is_integer():
        param = int(param)
    return f"results/ablation_{algo}_quadruple_well_{param}.pkl"


def filename_for_status(key, algo):
    (param,) = key
    param = float(param)
    if param.is_integer():
        param = int(param)
    return f"results/ablation_{algo}_quadruple_well_{param}_status.pkl"


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Same A sweep as the main quadruple-well comparison (run_experiments_resumable.py); A=1 is
    # degenerate (see that file's matching comment), so the sweep starts at 2, same as there.
    all_tasks = [(a, s) for a in range(2, 31) for s in range(100)]

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
        key_indices=[0],   # parameter
        value_start=2,      # skip seed (index 1)
        algo_names=PREFACTOR_MODES,
        filename_fn=filename_for,
    )
    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[0],
        value_start=2 + len(PREFACTOR_MODES),
        algo_names=PREFACTOR_MODES,
        filename_fn=filename_for_status,
        value_transform=str,
    )
