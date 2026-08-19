import os

import numpy as np

from experiments import run_resumable_pool, reconstruct_pkl_files_from_csv, peak_rss_mb
from potentials import hybrid_quadruple_well_setting
from run_experiments_resumable import _well_target_value
from simulations import SoftminEnergyAdagrad

# Hybrid-landscape rerun (2026-08-19) of run_ablation_beta2_quadruple_well.py's prefactor_mode
# ablation, on the new hybrid_quadruple_well_setting (separable double-well sum + small bilinear
# coupling; see that function's docstring in potentials.py for the full derivation) instead of
# the old unbalanced_quadruple_well_setting -- the old landscape had bounds too tight relative to
# the actual dynamics' excursion scale, and only 2 (not 3) distinct well depths. TILT_COEF/A**1.5
# scaling (not a fixed tilt) keeps the coupling-to-curvature ratio constant across the A sweep
# instead of decaying -- see potentials.py's hybrid_quadruple_well_setting docstring.
BETA = 2.0
N_SEEDS = 50
TILT_COEF = 0.05 / 2 ** 1.5
KAPPA_FRAC = 0.7

# New, non-colliding namespace: must not mix with the old-landscape A=2..20/50-seed data already
# on disk from run_ablation_beta2_quadruple_well.py (same BETA, different landscape -- silently
# merging them into the same CSV/pkl files would corrupt both).
RESULTS_CSV = 'results/master_ablation_beta2_quadruple_well_hybrid_log.csv'
MAX_STEPS = 10_000_000

PREFACTOR_MODES = ['signed', 'positive_part', 'absolute']

HEADER = (
    ['parameter', 'seed'] + PREFACTOR_MODES + [f'{m}_status' for m in PREFACTOR_MODES]
    + [f'{m}_gevals' for m in PREFACTOR_MODES] + [f'{m}_oevals' for m in PREFACTOR_MODES]
    + [f'{m}_secs' for m in PREFACTOR_MODES] + ['peak_rss_mb']
)


def worker_task(args):
    """Isolated worker function: runs the three prefactor-mode variants of SoftminEnergyAdagrad,
    at BETA=2.0 on the hybrid quadruple well, and returns their entry times."""
    param, seed = args
    np.random.seed(seed * 1000)

    tilt = TILT_COEF * param ** 1.5
    setting = hybrid_quadruple_well_setting(param, tilt, kappa_frac=KAPPA_FRAC)  # param is A
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
    return f"results/ablation_{algo}_beta2_quadruple_well_hybrid_{param}.pkl"


def filename_for_status(key, algo):
    (param,) = key
    param = float(param)
    if param.is_integer():
        param = int(param)
    return f"results/ablation_{algo}_beta2_quadruple_well_hybrid_{param}_status.pkl"


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # SMOKE_TEST env var (default on): quick A={2,5,10,15}/5-seed pass to confirm the plumbing
    # works end-to-end before committing to the full A=2..20/50-seed pilot. Run with
    # SMOKE_TEST=0 for the full pilot.
    if os.environ.get('SMOKE_TEST', '1') == '1':
        all_tasks = [(a, s) for a in [2, 5, 10, 15] for s in range(5)]
    else:
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
