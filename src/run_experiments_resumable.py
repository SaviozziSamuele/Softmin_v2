import os

import numpy as np

from experiments import run_resumable_pool, reconstruct_pkl_files_from_csv
from potentials import (
    one_dimensional_double_well,
    unbalanced_quadruple_well_setting,
)
from simulations import (
    SoftminEnergyAdagrad,
    SimulatedAnnealing,
    GradientAwareCBO,
    GradientAwarePSO,
    CBO,
    PSO,
)

# Fixed filename, matching every other driver's convention -- previously keyed by SLURM_JOB_ID,
# which never actually provided real crash-resumability (no job script sets --requeue) and just
# orphaned a fresh CSV on every resubmission instead of resuming the previous one.
RESULTS_CSV = 'results/master_experiment_log_decay.csv'
MAX_STEPS = 2_000_000_000

# Standard CBO's inverse temperature is annealed *upward* in the literature (opposite direction
# from Softmin's decaying beta) -- see GradientAwareCBO's beta_growth / CBO's beta_growth.
CBO_BETA_GROWTH = 1.001

# experiment, parameter, seed, softmin, sa, cbo, cbo_annealed, pso, standard_cbo, standard_pso
#
# softmin_decay was dropped from the sweep (2026-08-05): a direct escape-time comparison across
# beta in {50, 30, 20, 15, 10, 3, 1, 0.3, 0.1, 0.03, 0.01} and decay/growth schedules at both
# mid-sweep (delta=15/A=15) and hard (delta=30/A=30) settings found a properly-tuned fixed beta
# consistently reaches the target faster than every decaying or increasing schedule tested, with
# the gap widening at harder barriers (up to ~20x on quadruple_well A=30) -- see Remark
# rem:higher-dim and Appendix hyperparameters section. Continuing to run the full decaying-beta
# sweep at MAX_STEPS=2e9 would roughly double softmin's compute cost for a result already settled
# by the single-example comparison; that comparison is reported directly in the paper instead.
HEADER = [
    'experiment', 'parameter', 'seed',
    'softmin', 'sa', 'cbo', 'cbo_annealed', 'pso',
    'standard_cbo', 'standard_pso',
]


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
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")

    f_vec = setting['f_vec']
    f_grad_vec = setting['f_grad_vec']
    set_ = setting['set']
    bounds = setting['bounds']

    noise_val = 1.0
    lr = 0.001

    # Diagnostic-guided (src/hyperparameter_diagnostics.py) -- and a genuine correction along the
    # way: the beta*(f_i - J_beta) "ensemble fairness" gap diagnostic alone favored a much smaller
    # beta (most trailing particles are past the "essentially frozen" threshold at beta=10), but
    # that diagnostic only measures whether EVERY particle gets rescue-drift, not what the
    # first_entry_time metric actually rewards (the FASTEST particle reaching target, matching how
    # a leader's own prefactor -> n as beta grows, per the theory). Directly measuring actual
    # escape time at delta=15/A=15 across beta in {10, 3, 1, 0.3, 0.1, 0.03, 0.01} showed escape
    # getting monotonically SLOWER as beta shrinks below 10, and failing entirely (timeout) below
    # ~1 -- the opposite ranking from the gap-fairness proxy. beta=10.0 (the class default) was the
    # fastest value tested; kept explicit here rather than left as a silent default so this
    # reasoning is auditable.
    SOFTMIN_BETA = 10.0

    # Diagnostic-guided: cooling_rate=0.9999 was found to never escape the quadruple well at all
    # within a 200,000-step probe (cooling reaches 5% of initial temperature by step ~30,000, long
    # before any escape happens) while working fine on the double well -- a real, previously
    # undiagnosed failure mode for this driver's harder quadruple_well settings, not a genuine
    # "SA can't solve this" finding. 0.99995 escapes the same quadruple_well probe in ~1100 steps
    # and doesn't regress the double well (1570 vs. 1616 steps -- no meaningful difference).
    SA_COOLING_RATE = 0.99995

    # 2. Algorithm Initialization
    softmin = SoftminEnergyAdagrad(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        initial_lr=lr, adaptive_lr=False, noise_intensity=noise_val,
        bounds=bounds, noise="normal", beta_decay=None, beta=SOFTMIN_BETA
    )

    sa = SimulatedAnnealing(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        bounds=bounds, learning_rate=lr, cooling_rate=SA_COOLING_RATE,
        noise=noise_val, cooling_schedule="geometric", initial_temperature=10
    )

    cbo = GradientAwareCBO(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        bounds=bounds, learning_rate=lr, consensus_rate=1.0,
        noise_intensity=np.sqrt(2 * noise_val), beta=10.0
    )
    cbo_annealed = GradientAwareCBO(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        bounds=bounds, learning_rate=lr, consensus_rate=1.0,
        noise_intensity=np.sqrt(2 * noise_val), beta=10.0, beta_growth=CBO_BETA_GROWTH
    )

    pso = GradientAwarePSO(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        bounds=bounds, learning_rate=lr, w=0.729, c1=1.49445, c2=1.49445
    )

    standard_cbo = CBO(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        bounds=bounds, learning_rate=lr, consensus_rate=1.0, noise_intensity=1.0, beta=10.0
    )
    standard_pso = PSO(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        bounds=bounds, learning_rate=lr, w=0.729, c1=1.49445, c2=1.49445
    )

    # 3. Compute Entry/Exit Times
    t_softmin = softmin.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_sa = sa.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_cbo = cbo.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_cbo_annealed = cbo_annealed.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_pso = pso.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_standard_cbo = standard_cbo.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    t_standard_pso = standard_pso.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
    return (exp_name, param, seed, t_softmin, t_sa, t_cbo, t_cbo_annealed,
            t_pso, t_standard_cbo, t_standard_pso)


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
        algo_names=['softmin', 'sa', 'cbo', 'cbo_annealed', 'pso',
                     'standard_cbo', 'standard_pso'],
        filename_fn=filename_for,
    )
