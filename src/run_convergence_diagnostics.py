import os

import numpy as np

from experiments import run_resumable_pool, peak_rss_mb
from potentials import asymmetric_double_well
from simulations import SoftminEnergyAdagrad

# Fixed filename (2026-08-06), matching every other driver's convention -- see
# run_experiments_resumable.py's HEADER comment for why job-ID-keying never actually provided
# real crash-resumability.
RESULTS_CSV = 'results/master_convergence_diagnostics.csv'

# Freidlin-Wentzell exit-time theory (paper/SoftMin_revised_theory.tex, Theorem 3.1 / Cor.
# exit-time-conditional) is an asymptotic (sigma -> 0), conditional statement:
#   lim_{sigma->0} sigma * log E[tau] = V_beta(...) <= u(min(h, r_c))
# Comparing raw exit times against exp(u(min(h,r_c))/sigma) at one fixed, non-vanishing sigma
# conflates the rate constant with an (h-dependent) sub-exponential prefactor. This driver
# instead sweeps the three knobs that the asymptotic/finite-size assumptions depend on -- noise
# sigma, discretization step (learning rate), and particle count n -- one at a time, holding the
# other two at a shared baseline, at a couple of delta values already in the saturated regime
# (h > r_c) found by run_theory_verification.py. The notebook computes sigma*log(tau) (and, for
# the n-sweep, u(min(h,r_c); n) with the matching n) from the raw per-seed tau values below,
# rather than this driver pre-aggregating them.
TILT = 0.05
BETA = 2.0
DELTAS = [3, 4]

# Still `first_entry_time` (coordinate-space box target), not switched to `first_value_time` --
# same reasoning as run_theory_verification.py's matching comment: this driver validates the same
# state-space Theorem 3.1 exit-time statement, where "escape" means entering a spatial region, not
# crossing a value threshold.

BASE_SIGMA = 0.05
BASE_LR = 0.01
BASE_N = 100

SIGMA_VALUES = [0.2, 0.1, 0.05, 0.025]
LR_VALUES = [0.02, 0.01, 0.005, 0.0025, 0.00125]
N_VALUES = [50, 100, 200, 400]

SEEDS = range(100)

# Fixed *continuous-time* horizon (max_steps = T_MAX / lr), not a fixed step count, so shrinking
# lr doesn't shrink the physical time budget being simulated. Chosen with ~250x headroom above
# the empirical mean exit time observed at the hardest sampled config (delta=4, sigma=0.05).
# Runs that get stuck (swarm collapsed to a point) are aborted early by
# Simulation.first_entry_time's built-in trapped-swarm check, so most tasks finish well short of
# this ceiling regardless of how large it is set.
T_MAX = 20_000.0

# abort_on_stagnation is explicitly OFF here (2026-08-06) -- same reasoning as
# run_theory_verification.py (which shares this driver's beta=2.0 / low-noise regime), sharpened
# further by this driver's own SIGMA_VALUES/LR_VALUES sweeping toward the sigma->0 asymptotic
# limit the underlying theory is actually about: escape here is a genuine rare-event waiting
# process, not continuous improvement, so a short stagnation patience would systematically
# truncate exactly the long, quiet-then-sudden-crossing trajectories this sweep exists to
# characterize. Position-collapse abort (the ceiling this file's own comment above already
# documents relying on) is unaffected and stays on.
ABORT_ON_STAGNATION = False

# Cost columns (2026-08-07): this is where cost-vs-n comes from (the n_particles sweep already
# varies particle count directly) -- see run_experiments_resumable.py's matching HEADER comment.
HEADER = ['sweep', 'delta', 'sigma', 'lr', 'n', 'seed', 'tau', 'status',
          'gevals', 'oevals', 'secs', 'peak_rss_mb']


def worker_task(args):
    """
    Isolated worker function: runs one SoftminEnergyAdagrad instance at the given
    (delta, sigma, lr, n) and returns its continuous-time first entry time into the target well.
    """
    sweep, delta, sigma, lr, n, seed = args
    np.random.seed(seed * 1000)

    setting = asymmetric_double_well(delta, TILT)
    particle_init = [[setting['z_start'], setting['z_start']]]

    max_steps = int(T_MAX / lr)

    sim = SoftminEnergyAdagrad(
        n=n, d=1, f=setting['f_vec'], df=setting['f_grad_vec'], particle_init=particle_init,
        initial_lr=lr, adaptive_lr=False, noise_intensity=sigma, beta=BETA,
        noise="normal", bounds=setting['bounds'], beta_decay=None,
    )
    tau = sim.first_entry_time(
        target_set=setting['set'], max_steps=max_steps, abort_on_stagnation=ABORT_ON_STAGNATION,
    )

    return (sweep, delta, sigma, lr, n, seed, tau, sim._last_stop_reason,
            sim.n_gradient_evals, sim.n_objective_evals, sim.wall_clock_seconds, peak_rss_mb())


def task_key(t):
    sweep, delta, sigma, lr, n, seed = t
    return (sweep, float(delta), float(sigma), float(lr), int(n), int(seed))


def row_key(row):
    sweep, delta, sigma, lr, n, seed = row[:6]  # trailing cost columns not needed for the key
    return (sweep, float(delta), float(sigma), float(lr), int(n), int(seed))


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # (2026-08-07) BASE_SIGMA/BASE_LR/BASE_N are each a member of their own sweep list, so the
    # naive Cartesian product below used to emit three bit-identical tasks per delta/seed --
    # ('sigma', delta, BASE_SIGMA, BASE_LR, BASE_N, seed), ('lr', delta, BASE_SIGMA, BASE_LR,
    # BASE_N, seed), ('n_particles', delta, BASE_SIGMA, BASE_LR, BASE_N, seed) -- three distinct
    # task keys (task_key includes `sweep`) with identical simulation inputs *and* identical
    # seeds, so identical tau: 2 deltas x 100 seeds x 2 redundant copies = 400 wasted full-length
    # runs, and the three sweeps' baseline points were perfectly correlated instead of independent
    # observations. Fixed: the shared point is now computed once, under its own 'baseline' sweep
    # label, and each axis excludes its own value equal to the shared baseline. The notebook cell
    # that plots each axis's tau-vs-parameter curve must pull the point at
    # (sigma=BASE_SIGMA)/(lr=BASE_LR)/(n=BASE_N) from the 'baseline' group, not expect it inside
    # 'sigma'/'lr'/'n_particles' directly.
    all_tasks = []
    for delta in DELTAS:
        all_tasks.extend([
            ('baseline', delta, BASE_SIGMA, BASE_LR, BASE_N, seed)
            for seed in SEEDS
        ])
        all_tasks.extend([
            ('sigma', delta, sigma, BASE_LR, BASE_N, seed)
            for sigma in SIGMA_VALUES if sigma != BASE_SIGMA for seed in SEEDS
        ])
        all_tasks.extend([
            ('lr', delta, BASE_SIGMA, lr, BASE_N, seed)
            for lr in LR_VALUES if lr != BASE_LR for seed in SEEDS
        ])
        all_tasks.extend([
            ('n_particles', delta, BASE_SIGMA, BASE_LR, n, seed)
            for n in N_VALUES if n != BASE_N for seed in SEEDS
        ])

    run_resumable_pool(
        results_csv=RESULTS_CSV,
        header=HEADER,
        all_tasks=all_tasks,
        task_key_fn=task_key,
        row_key_fn=row_key,
        worker_fn=worker_task,
    )

    # No .pkl reconstruction here: unlike the algorithm-comparison drivers, this sweep's natural
    # unit is "one series of tau values per (sweep, delta, param)" across three different
    # parameter axes, which doesn't fit reconstruct_pkl_files_from_csv's single-x-axis shape. The
    # notebook reads RESULTS_CSV directly with pandas and groups by (sweep, delta, param).
