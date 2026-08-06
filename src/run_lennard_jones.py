import os

import numpy as np
from scipy.optimize import minimize

from experiments import run_resumable_pool, reconstruct_pkl_files_from_csv
from potentials import lennard_jones_cluster_setting
from simulations import (
    SoftminEnergyAdagrad,
    SimulatedAnnealing,
    GradientAwareCBO,
    GradientAwarePSO,
    CBO,
    PSO,
)

RESULTS_CSV = 'results/master_lennard_jones_log.csv'

# Set from the timing smoke test (.claude/scratch/smoke_test_lj.py, timing_smoke_test at the
# chosen combo below): at the original placeholder of 1_000_000, N=75 alone would cost roughly
# 280 hours (~11.7 days) per (n_atoms, seed) task -- completely infeasible. 100_000 brings N=75
# to a projected ~28 hours/task (all 30 seeds run in parallel via the resumable pool), keeping
# every N in N_ATOMS_LIST feasible; smaller N finish far faster. Re-check with real cluster
# timing if the projection turns out to be optimistic relative to Toeplitz's hardware.
MAX_STEPS = 100_000

# Learning rate and gradient-norm clip, set by the stiffness grid search
# (.claude/scratch/smoke_test_lj.py): clipping only the LJ term per-atom (not the combined
# LJ+confinement gradient -- see potentials.py's f_grad_vec) at max_grad_norm=1.0 measurably
# improved best_global_value over no clipping / a looser clip=10.0 at every N tested, without
# ever triggering a genuine escape (at_bound stayed False for softmin/softmin_decay/sa across the
# grid). LR unchanged from the original placeholder -- it was already reasonable; the fix was the
# clipping design, not the learning rate.
LR = 0.001
MAX_GRAD_NORM = 1.0

# Tolerance above the literature global minimum counted as "hit" by first_value_time -- loose
# enough to tolerate float/discretization noise, tight enough to require genuinely finding the
# basin (not just approaching it).
TARGET_EPS = 1e-2

# Standard CBO's inverse temperature is annealed *upward* in the literature (opposite direction
# from Softmin's decaying beta) -- see run_ackley_dimensions.py for the same convention.
CBO_BETA_GROWTH = 1.001

N_ATOMS_LIST = [13, 38, 55, 75]
SEEDS = range(30)  # LJ is O(N^2)/step vs. Ackley's O(d); fewer seeds are affordable.

HEADER = [
    'n_atoms', 'seed',
    'softmin_t', 'softmin_decay_t', 'sa_t', 'cbo_t', 'cbo_annealed_t', 'pso_t',
    'standard_cbo_t', 'standard_pso_t',
    'softmin_e', 'softmin_decay_e', 'sa_e', 'cbo_e', 'cbo_annealed_e', 'pso_e',
    'standard_cbo_e', 'standard_pso_e',
]

ALGO_NAMES = ['softmin', 'softmin_decay', 'sa', 'cbo', 'cbo_annealed', 'pso',
              'standard_cbo', 'standard_pso']

FILENAME_BY_ALGO_T = {a: f"results/{a}_t_lennard_jones_n{{n_atoms}}.pkl" for a in ALGO_NAMES}
FILENAME_BY_ALGO_E = {a: f"results/{a}_e_lennard_jones_n{{n_atoms}}.pkl" for a in ALGO_NAMES}


def _quench(quench_setting, position):
    """L-BFGS-B local polish of a candidate configuration, applied identically to every
    algorithm's best position -- standard practice in the LJ literature, and the only way to get
    a scientifically meaningful "which funnel did it find" comparison rather than "who got
    closest mid-dynamics".

    `quench_setting` must come from an UNCLIPPED `lennard_jones_cluster_setting` call
    (max_grad_norm=None), never the clipped setting used for the swarm dynamics: L-BFGS-B's
    quasi-Newton Hessian approximation is built from consecutive (step, gradient-difference)
    pairs assuming they satisfy the secant equation for the true Hessian, which the clipped
    gradient violates wherever clipping is active. Confirmed empirically -- quenching the exact
    same starting configuration with the clipped vs. unclipped gradient converged to visibly
    different (worse, for the clipped case) local minima, not just a slower path to the same one.
    """
    f_vec, f_grad_vec = quench_setting['f_vec'], quench_setting['f_grad_vec']

    def f_scalar(x):
        return f_vec(x[None, :])[0]

    def grad_scalar(x):
        return f_grad_vec(x[None, :])[0]

    res = minimize(f_scalar, position, jac=grad_scalar, method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
    return res.fun


def worker_task(args):
    """Isolated worker function for one (n_atoms, seed) task: builds one shared initial
    configuration, runs all 8 algorithms from it, records each algorithm's first-value hit time
    and its quenched (L-BFGS-B-polished) final energy.
    """
    n_atoms, seed = args

    # 1. Establish deterministic behavior for this worker
    np.random.seed(seed * 1000)

    # 2. Retrieve the LJ setting and a single shared initial configuration. A second, UNCLIPPED
    # setting is built for the quench step only -- the clipped gradient is a dynamics-stability
    # device for the swarm's step() updates and must never reach L-BFGS-B (see _quench).
    setting = lennard_jones_cluster_setting(n_atoms=n_atoms, max_grad_norm=MAX_GRAD_NORM)
    quench_setting = lennard_jones_cluster_setting(n_atoms=n_atoms, max_grad_norm=None)
    f_vec = setting['f_vec']
    f_grad_vec = setting['f_grad_vec']
    bounds = setting['bounds']
    d = setting['d']
    e_global = setting['e_global']
    target = e_global + TARGET_EPS

    # Generated once per (n_atoms, seed) and shared across every algorithm via .reset() below --
    # a deliberate difference from the Ackley driver (where each constructor draws its own init).
    # With only 30 seeds, shared initialization removes a nuisance variance term and makes the
    # comparison paired (documented in the paper appendix).
    x0 = setting['particle_init_fn'](n=100)

    # 3. Instantiate all 8 algorithms, then reset every one onto the shared x0
    softmin = SoftminEnergyAdagrad(
        n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds,
        name=f"SoftMin_lennard_jones_n{n_atoms}",
        initial_lr=LR, adaptive_lr=False, noise_intensity=1.0, noise="normal", beta_decay=None,
    )
    softmin_decay = SoftminEnergyAdagrad(
        n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds,
        name=f"SoftMin_decay_lennard_jones_n{n_atoms}",
        initial_lr=LR, adaptive_lr=False, noise_intensity=1.0, noise="normal", beta_decay=0.999,
    )
    sa = SimulatedAnnealing(
        n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds,
        name=f"SimulatedAnnealing_lennard_jones_n{n_atoms}",
        learning_rate=LR, cooling_rate=0.9999, noise=1.0,
        cooling_schedule="geometric", initial_temperature=10.0,
    )
    cbo = GradientAwareCBO(
        n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds,
        learning_rate=LR, consensus_rate=1.0, noise_intensity=np.sqrt(2.0), beta=10.0,
    )
    cbo_annealed = GradientAwareCBO(
        n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds,
        learning_rate=LR, consensus_rate=1.0, noise_intensity=np.sqrt(2.0), beta=10.0,
        beta_growth=CBO_BETA_GROWTH,
    )
    pso = GradientAwarePSO(
        n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds,
        learning_rate=LR, w=0.729, c1=1.49445, c2=1.49445,
    )
    standard_cbo = CBO(
        n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds,
        learning_rate=LR, consensus_rate=1.0, noise_intensity=1.0, beta=10.0,
    )
    standard_pso = PSO(
        n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds,
        learning_rate=LR, w=0.729, c1=1.49445, c2=1.49445,
    )

    algos = {
        'softmin': softmin, 'softmin_decay': softmin_decay, 'sa': sa,
        'cbo': cbo, 'cbo_annealed': cbo_annealed, 'pso': pso,
        'standard_cbo': standard_cbo, 'standard_pso': standard_pso,
    }
    for sim in algos.values():
        sim.reset(particle_init=x0)

    # 4. Run + quench every algorithm
    hit_times = {}
    quenched_energies = {}
    for name, sim in algos.items():
        hit_times[name] = sim.first_value_time(target_value=target, max_steps=MAX_STEPS)
        quenched_energies[name] = _quench(quench_setting, sim.best_global_position)

    return (n_atoms, seed,
            *(hit_times[a] for a in ALGO_NAMES),
            *(quenched_energies[a] for a in ALGO_NAMES))


def task_key(t):
    return (t[0], t[1])


def row_key(row):
    return (int(row[0]), int(row[1]))


def filename_for_t(key, algo):
    (n_atoms,) = key
    return FILENAME_BY_ALGO_T[algo].format(n_atoms=n_atoms)


def filename_for_e(key, algo):
    (n_atoms,) = key
    return FILENAME_BY_ALGO_E[algo].format(n_atoms=n_atoms)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    all_tasks = [(n, s) for n in N_ATOMS_LIST for s in SEEDS]

    run_resumable_pool(
        results_csv=RESULTS_CSV,
        header=HEADER,
        all_tasks=all_tasks,
        task_key_fn=task_key,
        row_key_fn=row_key,
        worker_fn=worker_task,
    )

    # `zip(algo_names, values)` in reconstruct_pkl_files_from_csv truncates to len(algo_names), so
    # calling it twice against the same CSV -- once pointed at the hit-time columns, once at the
    # quenched-energy columns -- reconstructs both metrics without any change to experiments.py.
    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[0],   # n_atoms
        value_start=2,      # skip seed (index 1); hit times start here
        algo_names=ALGO_NAMES,
        filename_fn=filename_for_t,
    )
    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[0],   # n_atoms
        value_start=2 + len(ALGO_NAMES),  # skip seed + hit times; quenched energies start here
        algo_names=ALGO_NAMES,
        filename_fn=filename_for_e,
    )
