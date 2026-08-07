import os

import numpy as np

from experiments import run_resumable_pool, reconstruct_pkl_files_from_csv, peak_rss_mb
from potentials import asymmetric_double_well
from run_experiments_resumable import _well_target_value
from simulations import SoftminEnergyAdagrad

# Matches run_experiments_resumable.py's DOUBLE_WELL_TILT -- this ablation shares the exact same
# benchmark (see the docstring below), so it must share the exact same potential, not just the
# same delta sweep.
DOUBLE_WELL_TILT = 0.05

# Fixed filename (2026-08-06), matching every other driver's convention -- was previously the
# only remaining driver still keyed by SLURM_JOB_ID, an oversight from the earlier session's
# switch (see run_experiments_resumable.py's HEADER comment for why job-ID-keying never actually
# provided real crash-resumability).
RESULTS_CSV = 'results/master_ablation_log.csv'
# 2e9 -> 1e7 (2026-08-07): same walltime-vs-per-step-cost reasoning as
# run_experiments_resumable.py, which this ablation shares its benchmark with -- see that file's
# MAX_STEPS comment for the measured numbers.
MAX_STEPS = 10_000_000

# Mechanism ablation (per paper/SoftMin_revised_theory.tex's numerical-experiments placeholder):
# replace the signed prefactor a_k with (a_k)_+ (ascending particles freeze) and |a_k| (every
# particle always descends, scaled by magnitude), and compare against the real "signed" method.
# This is a separate, standalone study -- not part of the Softmin-vs-SA-vs-CBO-vs-PSO comparison.
PREFACTOR_MODES = ['signed', 'positive_part', 'absolute']

# Status columns (2026-08-06): trailing block, one per prefactor mode, recording each sim's honest
# self._last_stop_reason -- see run_experiments_resumable.py's HEADER comment for the full
# rationale. No CBO/PSO here (softmin-only ablation), but the same mechanism applies uniformly and
# is worth the same audit trail. Cost columns (2026-08-07): same rationale as
# run_experiments_resumable.py's matching HEADER comment.
HEADER = (
    ['parameter', 'seed'] + PREFACTOR_MODES + [f'{m}_status' for m in PREFACTOR_MODES]
    + [f'{m}_gevals' for m in PREFACTOR_MODES] + [f'{m}_oevals' for m in PREFACTOR_MODES]
    + [f'{m}_secs' for m in PREFACTOR_MODES] + ['peak_rss_mb']
)


def worker_task(args):
    """
    Isolated worker function: runs the three prefactor-mode variants of SoftminEnergyAdagrad on
    the same double-well task and returns their entry times.
    """
    param, seed = args
    np.random.seed(seed * 1000)

    setting = asymmetric_double_well(param, DOUBLE_WELL_TILT)  # param is delta
    d = 1
    particle_init = setting['particle_init']

    f_vec = setting['f_vec']
    f_grad_vec = setting['f_grad_vec']
    bounds = setting['bounds']
    # Value-space target (2026-08-07), matching run_experiments_resumable.py's switch -- this
    # ablation shares the exact same benchmark, so it must share the exact same target definition,
    # not just the same delta sweep (see that file's WELL_TARGET_EPS_FRACTION comment for why).
    target_value = _well_target_value(setting)

    noise_val = 1.0
    lr = 0.001
    # Same diagnostic-guided value as the double_well series in run_experiments_resumable.py --
    # this ablation shares the exact same benchmark. Direct escape-time measurement across
    # beta in {50, 30, 20, 15, 10, 3, 1, 0.3, 0.1, 0.03, 0.01} at delta=15 found beta=10.0 (the
    # class default) at or near the actual optimum -- escape doesn't meaningfully improve above
    # it and fails outright below ~1 (see run_experiments_resumable.py for the full comparison,
    # including the quadruple_well case where going above 10 is markedly worse, not just flat).
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
    return f"results/ablation_{algo}_double_well_{param}.pkl"


def filename_for_status(key, algo):
    (param,) = key
    param = float(param)
    if param.is_integer():
        param = int(param)
    return f"results/ablation_{algo}_double_well_{param}_status.pkl"


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Same double-well delta sweep as the main comparison (run_experiments_resumable.py)
    all_tasks = [(d, s) for d in range(1, 31) for s in range(100)]

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
    # Status pkls (2026-08-07) -- see run_experiments_resumable.py's matching comment.
    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[0],
        value_start=2 + len(PREFACTOR_MODES),
        algo_names=PREFACTOR_MODES,
        filename_fn=filename_for_status,
        value_transform=str,
    )
