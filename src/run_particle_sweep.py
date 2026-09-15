"""Ensemble-size (particle-count) sweep: Softmin vs. Simulated Annealing on the 2D quadruple well.

Motivation. Every other driver fixes n=100 (Table tab:general_settings), so the paper's reliability
claim is stated at a single ensemble size. This driver varies n directly, at three difficulties
spanning SA's reliability collapse (A=10, where SA reaches 100% of seeds at n=100; A=20, still
reliable; A=30, where it reaches 0%).

What the comparison actually asks. SA here is n INDEPENDENT annealed chains -- the Simulation ABC
ensembles every method identically -- so raising n hands SA more restarts, and its reliability may
well improve with n too. That is the point: the question is whether Softmin's ensemble COUPLING
buys anything beyond embarrassingly-parallel restarts of a single-chain method at matched cost.
Both methods pay the same O(n) work per step, so per-particle evaluation counts are directly
comparable at every n.

Structure follows run_experiments_resumable.py, with `n` promoted to a task dimension. It writes
its OWN CSV rather than widening that driver's schema in place (see the warning at the top of
run_ablation_quadruple_well.py): the two files are not interchangeable and must not be merged.

Run:  cd src && python run_particle_sweep.py     (or: sbatch jobs/run_particle_sweep.sh)
Resumable: re-running only executes tasks whose rows are missing from the CSV.
"""
import os

import numpy as np

from experiments import run_resumable_pool, reconstruct_pkl_files_from_csv, peak_rss_mb
from potentials import unbalanced_quadruple_well_setting
from simulations import SoftminEnergyAdagrad, SimulatedAnnealing
from run_experiments_resumable import MAX_STEPS, _well_target_value

RESULTS_CSV = 'results/master_particle_sweep.csv'

# Ensemble sizes. Spans a factor of 80, centered on the n=100 used everywhere else so the sweep
# contains the published configuration as a directly comparable control point (see the sanity
# check in the plan: at n=100, A=10/20/30 must reproduce master_experiment_log_decay.csv's
# reached-fractions within seed noise, since it is the identical configuration).
N_VALUES = [5, 10, 25, 50, 100, 200, 400]

# Difficulties spanning SA's collapse at n=100: 100% reached at A=10, 100% at A=20, 0% at A=30.
A_VALUES = [10, 20, 30]

N_SEEDS = 100

# Only the two methods the comparison is about. CBO/PSO reach the target in at most 3% of seeds at
# any difficulty on this benchmark (Section 6.1), so sweeping them over n would cost 4x the compute
# to re-measure a flat zero; Langevin is already at 0% by A=10, the easiest point swept here.
ALGO_NAMES = ['softmin', 'sa']

HEADER = [
    'experiment', 'parameter', 'n', 'seed',
    *ALGO_NAMES,
    *(f'{a}_status' for a in ALGO_NAMES),
    *(f'{a}_gevals' for a in ALGO_NAMES),
    *(f'{a}_oevals' for a in ALGO_NAMES),
    *(f'{a}_secs' for a in ALGO_NAMES),
    'peak_rss_mb',
]

# Hyperparameters are held at exactly the values run_experiments_resumable.py uses for this
# benchmark, deliberately NOT re-tuned per n: the sweep measures the effect of ensemble size, so
# re-tuning at each n would confound it with a per-n tuning effect. Both were calibrated at n=100
# (see that driver's comments for the full diagnostic provenance), so n=100 is the in-sample point
# and every other n is out-of-sample for both methods symmetrically.
SOFTMIN_BETA = 10.0
SA_COOLING_RATE = 0.99995
SA_INITIAL_TEMPERATURE = 14.0283


def worker_task(args):
    exp_name, param, n, seed = args
    np.random.seed(seed * 1000)

    if exp_name != 'quadruple_well':
        raise ValueError(f"Unknown experiment: {exp_name}")
    setting = unbalanced_quadruple_well_setting(param, -1)  # param is A
    d = 2

    f_vec = setting['f_vec']
    f_grad_vec = setting['f_grad_vec']
    bounds = setting['bounds']
    particle_init = setting['particle_init']
    target_value = _well_target_value(setting)

    noise_val = 1.0
    lr = 0.001

    # Shared initial swarm, drawn once per task and applied to both algorithms via reset(), exactly
    # as run_experiments_resumable.py does -- removes the starting-position RNG coupling between
    # algorithms. Note the draw is (n, d), so the swarm scales with the swept n.
    x0 = np.random.uniform(
        [p[0] for p in particle_init], [p[1] for p in particle_init], (n, d)
    )

    softmin = SoftminEnergyAdagrad(
        n=n, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        initial_lr=lr, adaptive_lr=False, noise_intensity=noise_val,
        bounds=bounds, noise="normal", beta_decay=None, beta=SOFTMIN_BETA
    )
    sa = SimulatedAnnealing(
        n=n, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        bounds=bounds, learning_rate=lr, cooling_rate=SA_COOLING_RATE,
        noise=noise_val, cooling_schedule="geometric",
        initial_temperature=SA_INITIAL_TEMPERATURE
    )

    algos = {'softmin': softmin, 'sa': sa}
    for sim in algos.values():
        sim.reset(particle_init=x0)

    times, statuses = {}, {}
    for name, sim in algos.items():
        times[name] = sim.first_value_time(target_value=target_value, max_steps=MAX_STEPS)
        statuses[name] = sim._last_stop_reason

    return (exp_name, param, n, seed,
            *(times[a] for a in ALGO_NAMES),
            *(statuses[a] for a in ALGO_NAMES),
            *(algos[a].n_gradient_evals for a in ALGO_NAMES),
            *(algos[a].n_objective_evals for a in ALGO_NAMES),
            *(algos[a].wall_clock_seconds for a in ALGO_NAMES),
            peak_rss_mb())


def task_key(t):
    return (t[0], float(t[1]), int(t[2]), t[3])


def row_key(row):
    return (row[0], float(row[1]), int(row[2]), int(row[3]))


def _stem(key):
    exp, param, n = key
    param = float(param)
    if param.is_integer():
        param = int(param)
    return f"{exp}_{param}_n{int(n)}"


def filename_for(key, algo):
    return f"results/{algo}_{_stem(key)}.pkl"


def filename_for_status(key, algo):
    return f"results/{algo}_{_stem(key)}_status.pkl"


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    all_tasks = [
        ('quadruple_well', a, n, s)
        for a in A_VALUES for n in N_VALUES for s in range(N_SEEDS)
    ]

    run_resumable_pool(
        results_csv=RESULTS_CSV,
        header=HEADER,
        all_tasks=all_tasks,
        task_key_fn=task_key,
        row_key_fn=row_key,
        worker_fn=worker_task,
    )

    # key_indices now carries n as a third key component, and the seed sits at index 3 rather than
    # 2, so every value_start below is shifted by one against run_experiments_resumable.py's.
    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[0, 1, 2],    # experiment, parameter (A), n
        value_start=4,            # skip seed (index 3); values start at softmin
        algo_names=ALGO_NAMES,
        filename_fn=filename_for,
    )

    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[0, 1, 2],
        value_start=4 + len(ALGO_NAMES),
        algo_names=ALGO_NAMES,
        filename_fn=filename_for_status,
        value_transform=str,
    )
