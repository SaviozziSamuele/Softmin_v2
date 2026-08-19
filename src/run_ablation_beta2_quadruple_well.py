import os

import numpy as np

from experiments import run_resumable_pool, reconstruct_pkl_files_from_csv, peak_rss_mb
from potentials import unbalanced_quadruple_well_setting
from run_experiments_resumable import _well_target_value
from simulations import SoftminEnergyAdagrad

# Low-beta pilot (2026-08-19), 2D analogue of run_ablation_beta1_double_well.py -- see that
# file's header comment for the full rationale (beta=10.0 concentrates the Gibbs weight enough
# that the three prefactor_mode variants barely differ). Testing whether the mode separation
# found on the 1D double well at beta=1.0 (signed reliably slower than positive_part once
# delta>=6, pooled p=0.0006) generalizes to a landscape where escape isn't just "cross one
# barrier". beta=2.0 here (not 1.0): a quick single-seed timing probe found beta=1.0-scale
# suppression pushes some quadruple-well seeds into tens of seconds per run even at A=25-30, so
# beta=2.0 is a more conservative starting point given this runs locally, not on the cluster.
BETA = 2.0
N_SEEDS = 50

RESULTS_CSV = 'results/master_ablation_beta2_quadruple_well_log.csv'
MAX_STEPS = 10_000_000

PREFACTOR_MODES = ['signed', 'positive_part', 'absolute']

HEADER = (
    ['parameter', 'seed'] + PREFACTOR_MODES + [f'{m}_status' for m in PREFACTOR_MODES]
    + [f'{m}_gevals' for m in PREFACTOR_MODES] + [f'{m}_oevals' for m in PREFACTOR_MODES]
    + [f'{m}_secs' for m in PREFACTOR_MODES] + ['peak_rss_mb']
)


def worker_task(args):
    """Isolated worker function: runs the three prefactor-mode variants of SoftminEnergyAdagrad,
    at BETA=2.0, on the same quadruple-well task and returns their entry times."""
    param, seed = args
    np.random.seed(seed * 1000)

    setting = unbalanced_quadruple_well_setting(param, -1)  # param is A
    d = 2
    particle_init = setting['particle_init']

    f_vec = setting['f_vec']
    f_grad_vec = setting['f_grad_vec']
    bounds = setting['bounds']
    target_value = _well_target_value(setting)

    noise_val = 1.0
    lr = 0.001

    entry_times = []
    statuses = []
    gevals, oevals, secs = [], [], []
    for mode in PREFACTOR_MODES:
        softmin_variant = SoftminEnergyAdagrad(
            n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
            initial_lr=lr, adaptive_lr=False, noise_intensity=noise_val,
            bounds=bounds, noise="normal", beta_decay=None, prefactor_mode=mode, beta=BETA
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
    return f"results/ablation_{algo}_beta2_quadruple_well_{param}.pkl"


def filename_for_status(key, algo):
    (param,) = key
    param = float(param)
    if param.is_integer():
        param = int(param)
    return f"results/ablation_{algo}_beta2_quadruple_well_{param}_status.pkl"


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Extended (2026-08-19) from A=2..15/20 seeds to A=2..20/50 seeds after the first pass turned
    # up a sharp, highly significant crossover around A=8-9 (signed reliably slower below it,
    # reliably faster above it) -- more seeds to pin down the crossover's location/shape, and a
    # wider range to see whether signed's advantage above it keeps growing or saturates.
    all_tasks = [(a, s) for a in range(2, 21) for s in range(N_SEEDS)]

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
