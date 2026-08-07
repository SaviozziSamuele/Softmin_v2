import os

import numpy as np

from experiments import run_resumable_pool, reconstruct_pkl_files_from_csv, peak_rss_mb
from potentials import high_dim_ackley_function_setting
from simulations import (
    SoftminEnergyAdagrad,
    SimulatedAnnealing,
    GaussianDiffusion,
    GradientAwareCBO,
    GradientAwarePSO,
    CBO,
    PSO,
)

RESULTS_CSV = 'results/master_high_dim_ackley_log_exp_on_dims.csv'
MAX_STEPS = 10_000_000

# Value-space target (2026-08-07), replacing the old fixed +-0.2-per-dimension coordinate box.
# The box volume relative to the domain shrinks as (0.4/20)^d -- 1e-17 at d=10 down to 1e-153 at
# d=90 -- conflating true landscape difficulty with a geometric artifact of a fixed-in-absolute-
# terms box shrinking combinatorially with dimension (see the plan's B9 finding). Ackley's known
# global minimum is exactly 0 regardless of d, so a scalar `first_value_time` target has no
# dimension dependence at all: ACKLEY_TARGET_EPS=1.0 checked directly at d=50 (beta=10, pre-
# recalibration probe) -- reached in ~116,000 steps, the same order of magnitude as the old
# box-target's documented ~92,600 steps, so the two definitions aren't wildly apart in practice.
ACKLEY_TARGET_VALUE = 1.0

# Standard CBO's inverse temperature is annealed *upward* in the literature (opposite direction
# from Softmin's decaying beta) -- see run_experiments_resumable.py for the same convention.
CBO_BETA_GROWTH = 1.001

# 'langevin' added (2026-08-07): plain overdamped Langevin dynamics (GaussianDiffusion) at the
# same noise_intensity=1.0 as Softmin -- see run_experiments_resumable.py's matching comment.
ALGO_NAMES = ['softmin', 'sa', 'langevin', 'cbo', 'cbo_annealed', 'pso', 'standard_cbo', 'standard_pso']

# Status columns (2026-08-06): trailing block, one per algorithm, recording each sim's honest
# self._last_stop_reason -- see run_experiments_resumable.py's HEADER comment for the full
# rationale (same pattern, applied here for consistency).
# Cost columns (2026-08-07): see run_experiments_resumable.py's matching HEADER comment -- this
# sweep is also where cost-vs-dimension comes from, once these columns exist.
HEADER = [
    'dimension', 'seed', *ALGO_NAMES, *(f'{a}_status' for a in ALGO_NAMES),
    *(f'{a}_gevals' for a in ALGO_NAMES),
    *(f'{a}_oevals' for a in ALGO_NAMES),
    *(f'{a}_secs' for a in ALGO_NAMES),
    'peak_rss_mb',
]

FILENAME_BY_ALGO = {
    'softmin': "results/SoftMin_ackley_function_dim{dim}.pkl",
    'sa': "results/SimulatedAnnealing_ackley_function_dim{dim}.pkl",
    'langevin': "results/Langevin_ackley_function_dim{dim}.pkl",
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
    bounds = setting['bounds']

    noise_val = 1.0
    # Unified with every other driver (was 0.01, the only driver in this "main comparison" family
    # that didn't already use 0.001 -- no documented reason for the split was ever found). The
    # existing MAX_STEPS=10_000_000 has ample headroom either way: at the old lr=0.01, a mid-sweep
    # d=50 run reached the target in ~60,000 steps (0.6% of budget); a 10x smaller lr needs
    # roughly 10x more steps for the same physical progress, still only ~6% of budget.
    lr = 0.001

    # Shared initial swarm (2026-08-07) -- see run_experiments_resumable.py's matching comment.
    x0 = np.random.uniform(
        [p[0] for p in setting['particle_init']], [p[1] for p in setting['particle_init']],
        (100, dimension)
    )

    # 3. Instantiate Softmin with a fixed beta (switched from a beta_decay=0.999 schedule, per
    # user request). The beta*(f_i - J_beta) "ensemble fairness" gap diagnostic alone favored
    # beta=1.0 here, but that diagnostic doesn't reflect what first_entry_time actually rewards
    # (see run_experiments_resumable.py for the full explanation). Directly measuring escape time
    # at d=50 across beta in {50, 30, 10, 3, 1, 0.3, 0.1}: beta=1.0 TIMED OUT completely (100,000
    # step budget), while beta=10.0 (the class default) succeeded in ~92,600 steps, matching
    # beta=30/50's performance almost exactly (beta=3 also worked, marginally slower) -- the same
    # "smaller beta looks fairer but is actually much slower or fails outright" pattern found on
    # every other benchmark tested. Kept explicit rather than left as a silent default. (Not
    # re-swept 2026-08-06 -- Ackley's own bounds/setup were only documented, not numerically
    # changed, so this finding is unaffected by that session's benchmark corrections.)
    ACKLEY_SOFTMIN_BETA = 10.0

    # Diagnostic-guided, 2026-08-06: CBO swept across the same 13-candidate (11 beta + 2 annealed
    # schedule) grid as run_experiments_resumable.py. Genuine negative finding: every candidate
    # times out (position_collapse or value_stagnation) at d=50 -- CBO cannot be tuned into working
    # on this benchmark either.
    # ** PARTIALLY UNDER RE-VERIFICATION (2026-08-07) **: whichever candidates timed out via
    # value_stagnation specifically (not position_collapse) were measured under the old, too-short
    # stagnation_patience=50 default -- see simulations.py's matching comment for why that's now
    # suspect. Not yet re-checked for CBO/Ackley specifically (unlike the SA delta>=25 double-well
    # case, which is). PSO's Ackley failure is unaffected -- its comment below records
    # position_collapse only, not value_stagnation.
    # beta: 10.0 -> 0.01 (2026-08-07), matching run_experiments_resumable.py's CBO_BETA switch to
    # the literature-typical 0.01-0.02 range -- see that file's matching comment for the direct
    # empirical confirmation (across beta in {0.01...10.0}) that the collapse here is
    # beta-independent, so this changes nothing about the negative result itself. Spot-checked here
    # too: beta=0.01 and beta=10.0 collapse within one check-interval of each other at d=50
    # (step 12000 vs 13000, both position_collapse).
    ACKLEY_CBO_BETA = 0.01

    # Diagnostic-guided, 2026-08-06: unlike CBO/Softmin, PSO's literature-standard "canonical
    # constriction" default (0.729, 1.49445, 1.49445) was found to FAIL here -- times out via
    # position_collapse at d=50, d=70, AND d=90 in direct testing. Sweeping 6 named (w, c1, c2)
    # triples found two that actually escape: "Eberhart-Shi low-w" (0.4, 2.0, 2.0) and
    # "social-weighted" (0.729, 1.0, 2.0); the former is faster at every dimension tested
    # (d=50: step 311 vs 398; d=70: 656 vs 1835; d=90: 2538 vs 2607) and is adopted here. This
    # reverses the paper's previous claim that PSO fails on Ackley -- see the corresponding
    # \todo{} in main.tex pending the full sweep regeneration this change requires.
    ACKLEY_PSO_W, ACKLEY_PSO_C1, ACKLEY_PSO_C2 = 0.4, 2.0, 2.0
    softmin = SoftminEnergyAdagrad(
        n=100, d=dimension, f=f_vec, df=f_grad_vec, bounds=bounds,
        name=f"SoftMin_ackley_function_dim{dimension}",
        initial_lr=lr, adaptive_lr=False, noise_intensity=1.0,
        noise="normal", beta_decay=None, beta=ACKLEY_SOFTMIN_BETA
    )

    # 4. Instantiate Simulated Annealing (cooling_rate=0.9999 diagnostic-confirmed healthy for
    # this benchmark -- t_star/entry_steps ~ 0.5 at d=50, unlike double_well/quadruple_well -- so
    # left unchanged, see run_experiments_resumable.py for where it needed to change).
    # initial_temperature=10 also left unchanged (2026-08-07) -- NOT the same validated choice as
    # run_experiments_resumable.py's SA_INITIAL_TEMPERATURE=14.0283. The identical
    # initial-acceptance-ratio calibration was attempted here too (see
    # hyperparameter_diagnostics.py's Ackley T0 section) but was inconclusive even at a 100,000-
    # step probe (5x the wells' budget): every T0 candidate tested still failed to reach the
    # target, so nothing validated actually beats this default. Kept as an honest "not yet
    # resolved" rather than picking a number a probe couldn't support.
    sa = SimulatedAnnealing(
        n=100, d=dimension, f=f_vec, df=f_grad_vec, bounds=bounds,
        name=f"SimulatedAnnealing_ackley_function_dim{dimension}",
        learning_rate=lr, cooling_rate=0.9999, noise=1.0,
        cooling_schedule="geometric", initial_temperature=10.0
    )

    langevin = GaussianDiffusion(
        n=100, d=dimension, f=f_vec, df=f_grad_vec, bounds=bounds,
        name=f"Langevin_ackley_function_dim{dimension}",
        learning_rate=lr, noise=1.0,
    )

    cbo = GradientAwareCBO(
        n=100, d=dimension, f=f_vec, df=f_grad_vec,
        bounds=bounds, learning_rate=lr, consensus_rate=1.0,
        noise_intensity=np.sqrt(2 * noise_val), beta=ACKLEY_CBO_BETA
    )
    cbo_annealed = GradientAwareCBO(
        n=100, d=dimension, f=f_vec, df=f_grad_vec,
        bounds=bounds, learning_rate=lr, consensus_rate=1.0,
        noise_intensity=np.sqrt(2 * noise_val), beta=ACKLEY_CBO_BETA, beta_growth=CBO_BETA_GROWTH
    )

    pso = GradientAwarePSO(
        n=100, d=dimension, f=f_vec, df=f_grad_vec,
        bounds=bounds, learning_rate=lr, w=ACKLEY_PSO_W, c1=ACKLEY_PSO_C1, c2=ACKLEY_PSO_C2
    )

    standard_cbo = CBO(
        n=100, d=dimension, f=f_vec, df=f_grad_vec,
        bounds=bounds, learning_rate=lr, consensus_rate=1.0, noise_intensity=1.0, beta=ACKLEY_CBO_BETA
    )
    standard_pso = PSO(
        n=100, d=dimension, f=f_vec, df=f_grad_vec,
        bounds=bounds, learning_rate=lr, w=ACKLEY_PSO_W, c1=ACKLEY_PSO_C1, c2=ACKLEY_PSO_C2
    )

    # 5. Execute vectorized step loops. Each sim's early-stop status is read from
    # self._last_stop_reason right after the call (see run_experiments_resumable.py).
    algos = {
        'softmin': softmin, 'sa': sa, 'langevin': langevin, 'cbo': cbo, 'cbo_annealed': cbo_annealed,
        'pso': pso, 'standard_cbo': standard_cbo, 'standard_pso': standard_pso,
    }
    for sim in algos.values():
        sim.reset(particle_init=x0)

    times = {}
    statuses = {}
    for name, sim in algos.items():
        times[name] = sim.first_value_time(target_value=ACKLEY_TARGET_VALUE, max_steps=MAX_STEPS)
        statuses[name] = sim._last_stop_reason

    return (dimension, seed,
            *(times[a] for a in ALGO_NAMES),
            *(statuses[a] for a in ALGO_NAMES),
            *(algos[a].n_gradient_evals for a in ALGO_NAMES),
            *(algos[a].n_objective_evals for a in ALGO_NAMES),
            *(algos[a].wall_clock_seconds for a in ALGO_NAMES),
            peak_rss_mb())


def task_key(t):
    return (t[0], t[1])


def row_key(row):
    return (int(row[0]), int(row[1]))


def filename_for(key, algo):
    (dimension,) = key
    return FILENAME_BY_ALGO[algo].format(dim=dimension)


def filename_for_status(key, algo):
    (dimension,) = key
    return f"results/{algo}_ackley_function_dim{dimension}_status.pkl"


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

    # Status pkls (2026-08-07) -- see run_experiments_resumable.py's matching comment.
    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[0],
        value_start=2 + len(ALGO_NAMES),
        algo_names=ALGO_NAMES,
        filename_fn=filename_for_status,
        value_transform=str,
    )
