import os

import numpy as np

from experiments import run_resumable_pool
from potentials import asymmetric_double_well
from simulations import SoftminEnergyAdagrad

job_id = os.environ.get("SLURM_JOB_ID", "local")
RESULTS_CSV = f'results/master_convergence_diagnostics_{job_id}.csv'

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

HEADER = ['sweep', 'delta', 'sigma', 'lr', 'n', 'seed', 'tau']


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
    tau = sim.first_entry_time(target_set=setting['set'], max_steps=max_steps)

    return (sweep, delta, sigma, lr, n, seed, tau)


def task_key(t):
    sweep, delta, sigma, lr, n, seed = t
    return (sweep, float(delta), float(sigma), float(lr), int(n), int(seed))


def row_key(row):
    sweep, delta, sigma, lr, n, seed, tau = row
    return (sweep, float(delta), float(sigma), float(lr), int(n), int(seed))


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    all_tasks = []
    for delta in DELTAS:
        all_tasks.extend([
            ('sigma', delta, sigma, BASE_LR, BASE_N, seed)
            for sigma in SIGMA_VALUES for seed in SEEDS
        ])
        all_tasks.extend([
            ('lr', delta, BASE_SIGMA, lr, BASE_N, seed)
            for lr in LR_VALUES for seed in SEEDS
        ])
        all_tasks.extend([
            ('n_particles', delta, BASE_SIGMA, BASE_LR, n, seed)
            for n in N_VALUES for seed in SEEDS
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
