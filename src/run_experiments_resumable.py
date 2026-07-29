import os

import numpy as np

from experiments import run_resumable_pool, reconstruct_pkl_files_from_csv
from potentials import (
    one_dimensional_double_well,
    unbalanced_quadruple_well_setting,
    ackley_function_setting,
    initialize_on_circle,
)
from simulations import (
    SoftminEnergyAdagrad,
    SimulatedAnnealing,
    GradientAwareCBO,
    GradientAwarePSO,
)

job_id = os.environ.get("SLURM_JOB_ID", "local")
RESULTS_CSV = f'results/master_experiment_log_decay_{job_id}.csv'
MAX_STEPS = 2_000_000_000

# experiment, parameter, seed, softmin, softmin_decay, sa, cbo, pso
HEADER = ['experiment', 'parameter', 'seed', 'softmin', 'softmin_decay', 'sa', 'cbo', 'pso']


def worker_task(args):
    """
    Isolated worker function. Sets up the environment based on the experiment type,
    runs all 5 algorithms, and returns their exit times.
    """
    exp_name, param, seed = args
    np.random.seed(seed * 1000)

    # 1. Environment Setup based on experiment type
    if exp_name == 'double_well':
        setting = one_dimensional_double_well(param)  # param is delta
        d = 1
        particle_init = setting['particle_init']
    elif exp_name == 'quadruple_well':
        setting = unbalanced_quadruple_well_setting(param, -1)  # param is A
        d = 2
        particle_init = setting['particle_init']
    elif exp_name == 'ackley':
        setting = ackley_function_setting()
        d = 2
        # param is radius
        particle_init = initialize_on_circle(100, param, noise=0.1)
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")

    f_vec = setting['f_vec']
    f_grad_vec = setting['f_grad_vec']
    set_ = setting['set']
    bounds = setting['bounds']

    noise_val = 1.0
    lr = 0.001

    # 2. Algorithm Initialization
    softmin = SoftminEnergyAdagrad(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        initial_lr=lr, adaptive_lr=False, noise_intensity=noise_val,
        bounds=bounds, noise="normal", beta_decay=None
    )
    softmin_decay = SoftminEnergyAdagrad(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        initial_lr=lr, adaptive_lr=False, noise_intensity=noise_val,
        bounds=bounds, noise="normal", beta_decay=0.999
    )

    sa = SimulatedAnnealing(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        bounds=bounds, learning_rate=lr, cooling_rate=0.9999,
        noise=noise_val, cooling_schedule="geometric", initial_temperature=10
    )

    cbo = GradientAwareCBO(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        bounds=bounds, learning_rate=lr, consensus_rate=1.0,
        noise_intensity=np.sqrt(2 * noise_val), beta=10.0
    )

    pso = GradientAwarePSO(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        bounds=bounds, learning_rate=lr, w=0.729, c1=1.49445, c2=1.49445
    )

    # Note: For Ackley, the notebook initializes fresh circle particles for each algorithm
    if exp_name == 'ackley':
        softmin.reset(particle_init=initialize_on_circle(100, param, noise=0.1))
        softmin_decay.reset(particle_init=initialize_on_circle(100, param, noise=0.1))
        sa.reset(particle_init=initialize_on_circle(100, param, noise=0.1))
        cbo.reset(particle_init=initialize_on_circle(100, param, noise=0.1))
        pso.reset(particle_init=initialize_on_circle(100, param, noise=0.1))

    # 3. Compute Entry/Exit Times
    t_softmin = softmin.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_softmin_decay = softmin_decay.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_sa = sa.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_cbo = cbo.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_pso = pso.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    return exp_name, param, seed, t_softmin, t_softmin_decay, t_sa, t_cbo, t_pso


def task_key(t):
    return (t[0], float(t[1]), t[2])


def row_key(row):
    return (row[0], float(row[1]), int(row[2]))


def filename_for(key, algo):
    exp, param = key
    param = float(param)
    if param.is_integer():
        param = int(param)
    return f"results/{algo}_{exp}_{param}.pkl"


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Generate full Cartesian product of all desired workloads
    all_tasks = []
    # 1. Double Well (deltas 1 to 30)
    all_tasks.extend([('double_well', d, s) for d in range(1, 31) for s in range(100)])
    # 2. Quadruple Well (As 1 to 30)
    all_tasks.extend([('quadruple_well', a, s) for a in range(1, 31) for s in range(100)])
    # 3. Ackley (radii 1 to 8)
    all_tasks.extend([('ackley', r, s) for r in range(1, 9) for s in range(100)])

    run_resumable_pool(
        results_csv=RESULTS_CSV,
        header=HEADER,
        all_tasks=all_tasks,
        task_key_fn=task_key,
        row_key_fn=row_key,
        worker_fn=worker_task,
    )

    # Automatically convert the flat CSV back into the .pkl format the notebook expects
    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[0, 1],       # experiment, parameter
        value_start=3,            # skip seed (index 2); values start at softmin
        algo_names=['softmin', 'softmin_decay', 'sa', 'cbo', 'pso'],
        filename_fn=filename_for,
    )
