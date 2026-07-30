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

HEADER = ['dimension', 'seed', 'softmin_decay', 'sa', 'cbo', 'cbo_annealed', 'pso',
          'standard_cbo', 'standard_pso']

FILENAME_BY_ALGO = {
    'softmin_decay': "results/SoftMin_decay_ackley_function_dim{dim}.pkl",
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
    lr = 0.01

    # 3. Instantiate Softmin with Beta Decay
    softmin_decay = SoftminEnergyAdagrad(
        n=100, d=dimension, f=f_vec, df=f_grad_vec, bounds=bounds,
        name=f"SoftMin_decay_ackley_function_dim{dimension}",
        initial_lr=0.01, adaptive_lr=False, noise_intensity=1.0,
        noise="normal", beta_decay=0.999
    )

    # 4. Instantiate Simulated Annealing
    sa = SimulatedAnnealing(
        n=100, d=dimension, f=f_vec, df=f_grad_vec, bounds=bounds,
        name=f"SimulatedAnnealing_ackley_function_dim{dimension}",
        learning_rate=0.01, cooling_rate=0.9999, noise=1.0,
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

    # 5. Execute vectorized step loops
    t_softmin_decay = softmin_decay.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_sa = sa.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_cbo = cbo.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_cbo_annealed = cbo_annealed.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_pso = pso.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_standard_cbo = standard_cbo.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_standard_pso = standard_pso.first_entry_time(target_set=set_, max_steps=MAX_STEPS)

    return (dimension, seed, t_softmin_decay, t_sa, t_cbo, t_cbo_annealed, t_pso,
            t_standard_cbo, t_standard_pso)


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

    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[0],   # dimension
        value_start=2,      # skip seed (index 1)
        algo_names=['softmin_decay', 'sa', 'cbo', 'cbo_annealed', 'pso',
                    'standard_cbo', 'standard_pso'],
        filename_fn=filename_for,
    )
