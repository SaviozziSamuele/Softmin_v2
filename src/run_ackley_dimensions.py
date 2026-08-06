import os

import numpy as np

from experiments import run_resumable_pool, reconstruct_pkl_files_from_csv
from potentials import high_dim_ackley_function_setting
from simulations import (
    SoftminEnergyAdagrad,
    SimulatedAnnealing,
    GradientAwareCBO,
    GradientAwarePSO,
    CBO,
    PSO,
)

RESULTS_CSV = 'results/master_high_dim_ackley_log_exp_on_dims.csv'
MAX_STEPS = 10_000_000

# Standard CBO's inverse temperature is annealed *upward* in the literature (opposite direction
# from Softmin's decaying beta) -- see run_experiments_resumable.py for the same convention.
CBO_BETA_GROWTH = 1.001

ALGO_NAMES = ['softmin', 'sa', 'cbo', 'cbo_annealed', 'pso', 'standard_cbo', 'standard_pso']

# Status columns (2026-08-06): trailing block, one per algorithm, recording each sim's honest
# self._last_stop_reason -- see run_experiments_resumable.py's HEADER comment for the full
# rationale (same pattern, applied here for consistency).
HEADER = ['dimension', 'seed', *ALGO_NAMES, *(f'{a}_status' for a in ALGO_NAMES)]

FILENAME_BY_ALGO = {
    'softmin': "results/SoftMin_ackley_function_dim{dim}.pkl",
    'sa': "results/SimulatedAnnealing_ackley_function_dim{dim}.pkl",
    'cbo': "results/CBO_ackley_function_dim{dim}.pkl",
    'cbo_annealed': "results/CBO_annealed_ackley_function_dim{dim}.pkl",
    'pso': "results/PSO_ackley_function_dim{dim}.pkl",
    'standard_cbo': "results/StandardCBO_ackley_function_dim{dim}.pkl",
    'standard_pso': "results/StandardPSO_ackley_function_dim{dim}.pkl",
}


def worker_task(args):
    """
    Isolated worker function for the high-dimensional Ackley evaluation.
    """
    dimension, seed = args

    # 1. Establish deterministic behavior for this worker
    np.random.seed(seed * 1000)

    # 2. Retrieve dimensional setting
    setting = high_dim_ackley_function_setting(d=dimension)
    f_vec = setting['f_vec']
    f_grad_vec = setting['f_grad_vec']
    set_ = setting['set']
    bounds = setting['bounds']

    noise_val = 1.0
    # Unified with every other driver (was 0.01, the only driver in this "main comparison" family
    # that didn't already use 0.001 -- no documented reason for the split was ever found). The
    # existing MAX_STEPS=10_000_000 has ample headroom either way: at the old lr=0.01, a mid-sweep
    # d=50 run reached the target in ~60,000 steps (0.6% of budget); a 10x smaller lr needs
    # roughly 10x more steps for the same physical progress, still only ~6% of budget.
    lr = 0.001

    # 3. Instantiate Softmin with a fixed beta (switched from a beta_decay=0.999 schedule, per
    # user request). The beta*(f_i - J_beta) "ensemble fairness" gap diagnostic alone favored
    # beta=1.0 here, but that diagnostic doesn't reflect what first_entry_time actually rewards
    # (see run_experiments_resumable.py for the full explanation). Directly measuring escape time
    # at d=50 across beta in {50, 30, 10, 3, 1, 0.3, 0.1}: beta=1.0 TIMED OUT completely (100,000
    # step budget), while beta=10.0 (the class default) succeeded in ~92,600 steps, matching
    # beta=30/50's performance almost exactly (beta=3 also worked, marginally slower) -- the same
    # "smaller beta looks fairer but is actually much slower or fails outright" pattern found on
    # every other benchmark tested. Kept explicit rather than left as a silent default.
    ACKLEY_SOFTMIN_BETA = 10.0
    softmin = SoftminEnergyAdagrad(
        n=100, d=dimension, f=f_vec, df=f_grad_vec, bounds=bounds,
        name=f"SoftMin_ackley_function_dim{dimension}",
        initial_lr=lr, adaptive_lr=False, noise_intensity=1.0,
        noise="normal", beta_decay=None, beta=ACKLEY_SOFTMIN_BETA
    )

    # 4. Instantiate Simulated Annealing (cooling_rate=0.9999 diagnostic-confirmed healthy for
    # this benchmark -- t_star/entry_steps ~ 0.5 at d=50, unlike double_well/quadruple_well -- so
    # left unchanged, see run_experiments_resumable.py for where it needed to change)
    sa = SimulatedAnnealing(
        n=100, d=dimension, f=f_vec, df=f_grad_vec, bounds=bounds,
        name=f"SimulatedAnnealing_ackley_function_dim{dimension}",
        learning_rate=lr, cooling_rate=0.9999, noise=1.0,
        cooling_schedule="geometric", initial_temperature=10.0
    )

    cbo = GradientAwareCBO(
        n=100, d=dimension, f=f_vec, df=f_grad_vec,
        bounds=bounds, learning_rate=lr, consensus_rate=1.0,
        noise_intensity=np.sqrt(2 * noise_val), beta=10.0
    )
    cbo_annealed = GradientAwareCBO(
        n=100, d=dimension, f=f_vec, df=f_grad_vec,
        bounds=bounds, learning_rate=lr, consensus_rate=1.0,
        noise_intensity=np.sqrt(2 * noise_val), beta=10.0, beta_growth=CBO_BETA_GROWTH
    )

    pso = GradientAwarePSO(
        n=100, d=dimension, f=f_vec, df=f_grad_vec,
        bounds=bounds, learning_rate=lr, w=0.729, c1=1.49445, c2=1.49445
    )

    standard_cbo = CBO(
        n=100, d=dimension, f=f_vec, df=f_grad_vec,
        bounds=bounds, learning_rate=lr, consensus_rate=1.0, noise_intensity=1.0, beta=10.0
    )
    standard_pso = PSO(
        n=100, d=dimension, f=f_vec, df=f_grad_vec,
        bounds=bounds, learning_rate=lr, w=0.729, c1=1.49445, c2=1.49445
    )

    # 5. Execute vectorized step loops. Each sim's early-stop status is read from
    # self._last_stop_reason right after the call (see run_experiments_resumable.py).
    algos = {
        'softmin': softmin, 'sa': sa, 'cbo': cbo, 'cbo_annealed': cbo_annealed,
        'pso': pso, 'standard_cbo': standard_cbo, 'standard_pso': standard_pso,
    }
    times = {}
    statuses = {}
    for name, sim in algos.items():
        times[name] = sim.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
        statuses[name] = sim._last_stop_reason

    return (dimension, seed,
            *(times[a] for a in ALGO_NAMES),
            *(statuses[a] for a in ALGO_NAMES))


def task_key(t):
    return (t[0], t[1])


def row_key(row):
    return (int(row[0]), int(row[1]))


def filename_for(key, algo):
    (dimension,) = key
    return FILENAME_BY_ALGO[algo].format(dim=dimension)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Generate the flattened workload
    all_tasks = [(d, s) for d in range(10, 100, 10) for s in range(100)]

    run_resumable_pool(
        results_csv=RESULTS_CSV,
        header=HEADER,
        all_tasks=all_tasks,
        task_key_fn=task_key,
        row_key_fn=row_key,
        worker_fn=worker_task,
    )

    # Trailing _status columns are audit-trail data, not plotting data -- see
    # run_experiments_resumable.py's matching comment for why this needs no other change here.
    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[0],   # dimension
        value_start=2,      # skip seed (index 1)
        algo_names=ALGO_NAMES,
        filename_fn=filename_for,
    )
