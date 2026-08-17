import os

import numpy as np

from experiments import run_resumable_pool, reconstruct_pkl_files_from_csv, peak_rss_mb
from potentials import lennard_jones_cluster_setting
from run_lennard_jones import _quench, LR, MAX_GRAD_NORM, TARGET_EPS
from simulations import SoftminEnergyAdagrad

# Fixed at N=38 -- the "canonical hard double-funnel case" (icosahedral local funnel vs. the true
# fcc-truncated-octahedron global minimum; see run_lennard_jones.py's driver notes) -- not N=13,
# despite N=13 being far cheaper (LJ cost is O(N^2)/step): the point of this ablation is to test
# whether the mechanism differentiates on a genuinely harder landscape, and N=13 is easy enough
# that it wouldn't exercise that. At N=38, essentially no baseline in the main 8-algorithm sweep
# ever reaches the hit-time target within budget (0% reached-fraction across all 8 methods), so
# quenched energy -- not gated hit-time -- is the metric that actually carries signal here; hit
# time/status is still recorded for completeness and consistency with the other ablations.
LJ_N_ATOMS = 38

RESULTS_CSV = 'results/master_ablation_lj_log.csv'
# Same budget as the main LJ sweep's N=38 row, for direct comparability with cell 17's existing
# N=38 quenched-energy result -- see run_lennard_jones.py's MAX_STEPS comment for the timing
# smoke-test this was derived from.
MAX_STEPS = 100_000

# Same mechanism ablation as run_ablation_experiment.py (see that file for the full rationale),
# applied to the hardest Lennard-Jones cluster size to test whether the double-well ablation's
# finding generalizes to a real-physics, NP-hard landscape. 30 tasks total (one per seed, matching
# run_lennard_jones.py's own seed count) -- each task's worker_task runs all 3 modes sequentially,
# so this is NOT 90 separate run_resumable_pool tasks.
PREFACTOR_MODES = ['signed', 'positive_part', 'absolute']

# _t (hit time) + _e (quenched energy) blocks, matching run_lennard_jones.py's HEADER convention --
# see that file's HEADER comment for why quench evals are folded into gevals/oevals.
HEADER = (
    ['seed']
    + [f'{m}_t' for m in PREFACTOR_MODES] + [f'{m}_e' for m in PREFACTOR_MODES]
    + [f'{m}_status' for m in PREFACTOR_MODES]
    + [f'{m}_gevals' for m in PREFACTOR_MODES] + [f'{m}_oevals' for m in PREFACTOR_MODES]
    + [f'{m}_secs' for m in PREFACTOR_MODES] + ['peak_rss_mb']
)


def worker_task(args):
    """
    Isolated worker function: runs the three prefactor-mode variants of SoftminEnergyAdagrad on
    the same Lennard-Jones N=38 task, quenches each variant's best position, and returns both the
    raw hit time and the quenched energy. Mirrors run_lennard_jones.py's worker_task, restricted
    to the 3 prefactor-mode variants instead of the 8-algorithm comparison.
    """
    (seed,) = args
    np.random.seed(seed * 1000)

    # Clipped setting for the swarm dynamics, unclipped for the quench step -- see
    # run_lennard_jones.py's _quench docstring for why the two must never be mixed.
    setting = lennard_jones_cluster_setting(n_atoms=LJ_N_ATOMS, max_grad_norm=MAX_GRAD_NORM)
    quench_setting = lennard_jones_cluster_setting(n_atoms=LJ_N_ATOMS, max_grad_norm=None)
    f_vec = setting['f_vec']
    f_grad_vec = setting['f_grad_vec']
    bounds = setting['bounds']
    d = setting['d']
    target = setting['e_global'] + TARGET_EPS

    # Shared initial configuration across the 3 variants, matching run_lennard_jones.py's own
    # shared-x0 pattern (a paired comparison; see that file's matching comment).
    x0 = setting['particle_init_fn'](n=100)

    hit_times = []
    quenched_energies = []
    statuses = []
    gevals, oevals, secs = [], [], []
    for mode in PREFACTOR_MODES:
        softmin_variant = SoftminEnergyAdagrad(
            n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds,
            initial_lr=LR, adaptive_lr=False, noise_intensity=1.0,
            noise="normal", beta_decay=None, prefactor_mode=mode,
        )
        softmin_variant.reset(particle_init=x0)
        hit_times.append(softmin_variant.first_value_time(target_value=target, max_steps=MAX_STEPS))
        quench_energy, quench_nfev, quench_njev = _quench(
            quench_setting, softmin_variant.best_global_position
        )
        quenched_energies.append(quench_energy)
        softmin_variant.n_objective_evals += quench_nfev
        softmin_variant.n_gradient_evals += quench_njev
        statuses.append(softmin_variant._last_stop_reason)
        gevals.append(softmin_variant.n_gradient_evals)
        oevals.append(softmin_variant.n_objective_evals)
        secs.append(softmin_variant.wall_clock_seconds)

    return (
        seed, *hit_times, *quenched_energies, *statuses, *gevals, *oevals, *secs, peak_rss_mb()
    )


def task_key(t):
    return (t[0],)


def row_key(row):
    return (int(row[0]),)


def filename_for_t(key, algo):
    return f"results/ablation_{algo}_t_lennard_jones_n{LJ_N_ATOMS}.pkl"


def filename_for_e(key, algo):
    return f"results/ablation_{algo}_e_lennard_jones_n{LJ_N_ATOMS}.pkl"


def filename_for_status(key, algo):
    return f"results/ablation_{algo}_status_lennard_jones_n{LJ_N_ATOMS}.pkl"


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Same seed count as the main LJ sweep (run_lennard_jones.py) -- LJ is O(N^2)/step, so fewer
    # seeds than the wells/Ackley ablations are affordable, matching the main sweep's own choice.
    all_tasks = [(s,) for s in range(30)]

    run_resumable_pool(
        results_csv=RESULTS_CSV,
        header=HEADER,
        all_tasks=all_tasks,
        task_key_fn=task_key,
        row_key_fn=row_key,
        worker_fn=worker_task,
    )

    n_modes = len(PREFACTOR_MODES)
    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[],
        value_start=1,               # hit-time block starts right after seed
        algo_names=PREFACTOR_MODES,
        filename_fn=filename_for_t,
    )
    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[],
        value_start=1 + n_modes,      # quenched-energy block
        algo_names=PREFACTOR_MODES,
        filename_fn=filename_for_e,
    )
    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[],
        value_start=1 + 2 * n_modes,  # status block
        algo_names=PREFACTOR_MODES,
        filename_fn=filename_for_status,
        value_transform=str,
    )
