import os

import numpy as np

from experiments import run_resumable_pool, reconstruct_pkl_files_from_csv
from potentials import asymmetric_double_well
from simulations import SoftminEnergyAdagrad

job_id = os.environ.get("SLURM_JOB_ID", "local")
RESULTS_CSV = f'results/master_theory_verification_init0_0p05_{job_id}.csv'
MAX_STEPS = 10_000_000

# Fixed asymmetry for the well: small enough to keep 3 real critical points across the whole
# delta sweep below (see src/potentials.py:asymmetric_double_well), while still making the two
# minima genuinely different depths so this tests escape from a suboptimal basin, not just a
# symmetric barrier crossing.
TILT = 0.05

HEADER = ['experiment', 'parameter', 'noise', 'seed', 'softmin_static']


def worker_task(args):
    """
    Isolated worker function. Sets up the environment based on the experiment type.
    """
    experiment, param, noise_val, seed = args
    np.random.seed(seed * 1000)

    setting = asymmetric_double_well(param, TILT)  # param is delta
    d = 1
    particle_init = [[setting['z_start'], setting['z_start']]]

    f_vec = setting['f_vec']
    f_grad_vec = setting['f_grad_vec']
    set_ = setting['set']
    bounds = setting['bounds']

    lr = 0.01

    # Algorithm Initialization
    softmin_decay = SoftminEnergyAdagrad(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        initial_lr=lr, adaptive_lr=False, noise_intensity=noise_val,
        bounds=bounds, noise="normal", beta_decay=None, beta=2.0
    )

    # Compute Entry/Exit Times
    t_softmin = softmin_decay.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    return experiment, param, noise_val, seed, t_softmin


def task_key(t):
    return (t[0], float(t[1]), float(t[2]), t[3])


def row_key(row):
    return (row[0], float(row[1]), float(row[2]), int(row[3]))


def _clean(value):
    """Format a float parameter for a filename: 12.0 -> '12', 0.05 -> '0p05'."""
    value = float(value)
    if value.is_integer():
        value = int(value)
    return str(value).replace('.', 'p')


def filename_for(key, algo):
    exp, param, noise_val = key
    return f"results/{algo}_{exp}_init0_{_clean(param)}_{_clean(noise_val)}.pkl"


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Generate full Cartesian product of all desired workloads
    all_tasks = []
    # Double Well (deltas 0.25 to 4.0 in steps of 0.125), fixed noise 0.05
    all_tasks.extend([('double_well', d, 0.05, s) for d in np.arange(0.25, 4.25, 0.125) for s in range(100)])

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
        key_indices=[0, 1, 2],   # experiment, parameter, noise
        value_start=4,            # skip seed (index 3)
        algo_names=['softmin_static'],
        filename_fn=filename_for,
    )
