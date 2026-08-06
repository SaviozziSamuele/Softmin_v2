import os

import numpy as np

from experiments import run_resumable_pool, reconstruct_pkl_files_from_csv
from potentials import (
    asymmetric_double_well,
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

# Fixed asymmetry tilt (2026-08-06): the main double-well benchmark switched from the symmetric
# `one_dimensional_double_well` (two equally-good minima -- no genuinely suboptimal basin to
# escape from) to `asymmetric_double_well`, matching the value already used by
# run_theory_verification.py. Verified valid (3 real critical points) across the full delta=1..30
# sweep, producing a clean, gently-growing depth gap between the two wells (~0.0707*sqrt(delta)).
DOUBLE_WELL_TILT = 0.05

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
ALGO_NAMES = ['softmin', 'sa', 'cbo', 'cbo_annealed', 'pso', 'standard_cbo', 'standard_pso']

# Status columns (2026-08-06): trailing block, one per algorithm, appended after all the time
# columns above -- records each sim's honest self._last_stop_reason ('target_reached' /
# 'position_collapse' / 'value_stagnation' / 'max_steps_exhausted') so a genuine hit can be told
# apart from an early-stopped run from the CSV alone, instead of the numeric time column being the
# only record (which can't distinguish them -- both a real max_steps run and an aborted one return
# the identical max_steps*lr sentinel). See Simulation._check_early_stop in simulations.py: this
# is what replaced the old, undisclosed position-collapse-only abort that silently mislabeled most
# of CBO/PSO's quadruple-well results as if they'd been run to the real budget.
HEADER = [
    'experiment', 'parameter', 'seed',
    *ALGO_NAMES,
    *(f'{a}_status' for a in ALGO_NAMES),
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
        setting = asymmetric_double_well(param, DOUBLE_WELL_TILT)  # param is delta
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

    # Diagnostic-guided (src/hyperparameter_diagnostics.py). Re-validated 2026-08-06 against the
    # corrected benchmarks (asymmetric double well, curvature-calibrated bounds on both wells,
    # quadruple well A now 2..30): beta=10 still reaches target at both mid-sweep settings
    # (delta=15/A=15), confirming the original 2026-08-05 finding still holds post-correction.
    # Note the quadruple well's working band narrowed under the corrected (tighter) bounds --
    # beta in {50, 30} now times out there too, not just the small-beta end -- so beta=10 sitting
    # centrally in the surviving {20, 15, 10} band is a safer choice than before, not just an
    # unchanged one. See hyperparameter_diagnostics.py's CALIBRATION_BUDGET for the tracked
    # per-algorithm tuning effort (same budget as CBO/SA/PSO below).
    SOFTMIN_BETA = 10.0

    # Diagnostic-guided, re-validated 2026-08-06 against the corrected benchmarks: cooling_rate
    # candidates were swept from timescale-derived rates (t_star = steps to reach 5% of initial
    # temperature, 1000..1,000,000). At the corrected quadruple well, only t_star >= 100,000
    # (rate >= 0.99997) reaches target in the sweep -- but the shipped 0.99995 (t_star ~ 59,900,
    # between the surviving and failing sweep points) was checked directly and reaches in ~1100
    # steps, confirming it still works despite sitting in the sweep's "untested gap". Double well
    # needs rate >= ~0.9999; 0.99995 clears both with margin.
    SA_COOLING_RATE = 0.99995

    # Diagnostic-guided, 2026-08-06: swept the same 11-value beta grid as Softmin plus 2 annealed
    # schedule variants (13 candidates, matching Softmin's budget) at both mid-sweep settings.
    # Genuine negative finding, not a tuning gap: EVERY candidate times out via position_collapse
    # at both double_well (delta=15) and quadruple_well (A=15) -- CBO's consensus dynamics collapse
    # onto the wrong basin regardless of beta or annealing schedule on these benchmarks. beta=10.0
    # kept as the literature-standard value since no alternative empirically helps; see Remark
    # rem:cbo-pso-negative-result in the paper.
    CBO_BETA = 10.0

    # Diagnostic-guided, 2026-08-06: swept 6 named (w, c1, c2) triples (see PSO_CANDIDATES in
    # hyperparameter_diagnostics.py) at both mid-sweep settings. Same negative finding as CBO: every
    # candidate times out at both double_well and quadruple_well regardless of choice -- unlike
    # Ackley (run_ackley_dimensions.py), where two of these same candidates actually escape and the
    # literature-standard default was switched away from. Kept at the literature-standard
    # constriction coefficients here since no alternative empirically helps on these two benchmarks.
    PSO_W, PSO_C1, PSO_C2 = 0.729, 1.49445, 1.49445

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
        noise_intensity=np.sqrt(2 * noise_val), beta=CBO_BETA
    )
    cbo_annealed = GradientAwareCBO(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        bounds=bounds, learning_rate=lr, consensus_rate=1.0,
        noise_intensity=np.sqrt(2 * noise_val), beta=CBO_BETA, beta_growth=CBO_BETA_GROWTH
    )

    pso = GradientAwarePSO(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        bounds=bounds, learning_rate=lr, w=PSO_W, c1=PSO_C1, c2=PSO_C2
    )

    standard_cbo = CBO(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        bounds=bounds, learning_rate=lr, consensus_rate=1.0, noise_intensity=1.0, beta=CBO_BETA
    )
    standard_pso = PSO(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        bounds=bounds, learning_rate=lr, w=PSO_W, c1=PSO_C1, c2=PSO_C2
    )

    # 3. Compute Entry/Exit Times. Each sim's early-stop status is read from
    # self._last_stop_reason right after the call, since first_entry_time's return value alone
    # can't distinguish a real max_steps run from an early-stopped one (see ALGO_NAMES comment).
    algos = {
        'softmin': softmin, 'sa': sa, 'cbo': cbo, 'cbo_annealed': cbo_annealed,
        'pso': pso, 'standard_cbo': standard_cbo, 'standard_pso': standard_pso,
    }
    times = {}
    statuses = {}
    for name, sim in algos.items():
        times[name] = sim.first_entry_time(target_set=set_, max_steps=MAX_STEPS)
        statuses[name] = sim._last_stop_reason

    return (exp_name, param, seed,
            *(times[a] for a in ALGO_NAMES),
            *(statuses[a] for a in ALGO_NAMES))


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
    # 2. Quadruple Well (As 2 to 30 -- A=1 is degenerate: the softest Hessian eigenvalue at the
    # well, 4*A-4 at this benchmark's fixed delta=-1, hits exactly zero there, so it isn't a
    # genuine local minimum in that direction. See potentials.py's ValueError if this is ever
    # called with A=1.)
    all_tasks.extend([('quadruple_well', a, s) for a in range(2, 31) for s in range(100)])

    run_resumable_pool(
        results_csv=RESULTS_CSV,
        header=HEADER,
        all_tasks=all_tasks,
        task_key_fn=task_key,
        row_key_fn=row_key,
        worker_fn=worker_task,
    )

    # Automatically convert the flat CSV back into the .pkl format the notebook expects. The
    # trailing _status columns are audit-trail data for the master CSV, not plotting data --
    # zip(algo_names, values) in reconstruct_pkl_files_from_csv truncates to len(algo_names), so
    # this reads only the time-value block and silently ignores the status block that follows it
    # (same trick run_lennard_jones.py already relies on and documents for its two metric blocks).
    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[0, 1],       # experiment, parameter
        value_start=3,            # skip seed (index 2); values start at softmin
        algo_names=ALGO_NAMES,
        filename_fn=filename_for,
    )
