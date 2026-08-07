import os

import numpy as np

from experiments import run_resumable_pool, reconstruct_pkl_files_from_csv, peak_rss_mb
from potentials import (
    asymmetric_double_well,
    unbalanced_quadruple_well_setting,
)
from simulations import (
    SoftminEnergyAdagrad,
    SimulatedAnnealing,
    GaussianDiffusion,
    GradientAwareCBO,
    GradientAwarePSO,
    CBO,
    PSO,
)

# Fixed filename, matching every other driver's convention -- previously keyed by SLURM_JOB_ID,
# which never actually provided real crash-resumability (no job script sets --requeue) and just
# orphaned a fresh CSV on every resubmission instead of resuming the previous one.
RESULTS_CSV = 'results/master_experiment_log_decay.csv'
# 2e9 -> 1e7 (2026-08-07): the double-well objective alone costs ~1.5ms/step (np.vectorize over
# n=100 particles), so 2e9 steps projects to ~36 days for a single algorithm on a single task --
# far past the 7-day SLURM walltime, and a task's CSV row is only written after all algorithms in
# it finish, so a run that hits the wall writes nothing and regenerates identically on resubmit.
# Measured escapes are ~1e3 steps even at the hardest delta/A in the sweep, so 1e7 keeps ~4 orders
# of headroom while capping a genuinely stuck run at a few hours instead of over a month.
MAX_STEPS = 10_000_000

# Standard CBO's inverse temperature is annealed *upward* in the literature (opposite direction
# from Softmin's decaying beta) -- see GradientAwareCBO's beta_growth / CBO's beta_growth.
CBO_BETA_GROWTH = 1.001

# Fixed asymmetry tilt (2026-08-06): the main double-well benchmark switched from the symmetric
# `one_dimensional_double_well` (two equally-good minima -- no genuinely suboptimal basin to
# escape from) to `asymmetric_double_well`, matching the value already used by
# run_theory_verification.py. Verified valid (3 real critical points) across the full delta=1..30
# sweep, producing a clean, gently-growing depth gap between the two wells (~0.0707*sqrt(delta)).
DOUBLE_WELL_TILT = 0.05

# Value-space target (2026-08-07), replacing the old fixed +-0.2 coordinate box for BOTH wells.
# The box width was never checked against either well's own scale: measured, box-width/lengthscale
# drifted 5.3x across the double-well delta sweep and 3.5x across the quadruple-well A sweep
# (getting relatively *more generous*, not held constant, exactly across the sweep meant to test
# difficulty scaling) -- see the plan's B9 finding. `first_value_time` with a target expressed as a
# fraction of each setting's own `depth_gap` (f(start well) - f(target well)) scales automatically
# with delta/A instead, mirroring run_lennard_jones.py's `e_global + TARGET_EPS` pattern. 0.15 was
# checked empirically (asymmetric_double_well delta=30, unbalanced_quadruple_well_setting A=15):
# tight enough that the swarm must genuinely approach the target well (not just its outskirts), but
# clear of the ~10% plateau some runs settle at before fully converging under noise.
WELL_TARGET_EPS_FRACTION = 0.15


def _well_target_value(setting):
    """target_value for `first_value_time` on a well `*_setting` dict: `depth_gap` and `set`
    (now centered exactly on the true target critical point, after the 2026-08-07 quadruple-well
    fix) are enough to build it generically for both the 1D and 2D well benchmarks."""
    target_center = np.array([np.mean(b) for b in setting['set']])
    f_target = float(setting['f_vec'](target_center))
    return f_target + WELL_TARGET_EPS_FRACTION * setting['depth_gap']

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
#
# 'langevin' added (2026-08-07): plain overdamped Langevin dynamics (GaussianDiffusion) at the
# same noise_intensity=1.0 as Softmin -- the one baseline that needs no hyperparameter tuning at
# all, and the exact dynamics Theorem 3.1 is a claim about. Previously implemented but used in no
# driver (only the notebook's cost-comparison cell).
ALGO_NAMES = ['softmin', 'sa', 'langevin', 'cbo', 'cbo_annealed', 'pso', 'standard_cbo', 'standard_pso']

# Status columns (2026-08-06): trailing block, one per algorithm, appended after all the time
# columns above -- records each sim's honest self._last_stop_reason ('target_reached' /
# 'position_collapse' / 'value_stagnation' / 'max_steps_exhausted') so a genuine hit can be told
# apart from an early-stopped run from the CSV alone, instead of the numeric time column being the
# only record (which can't distinguish them -- both a real max_steps run and an aborted one return
# the identical max_steps*lr sentinel). See Simulation._check_early_stop in simulations.py: this
# is what replaced the old, undisclosed position-collapse-only abort that silently mislabeled most
# of CBO/PSO's quadruple-well results as if they'd been run to the real budget.
# Cost columns (2026-08-07, reviewer demand (b)): gradient/objective evaluations (now counted
# per-particle, see Simulation._eval_grad/_eval_obj) and wall-clock seconds per algorithm, plus one
# peak-RSS-MB reading for the whole task (memory is process-wide, not per-algorithm). Appended
# after the status block -- reconstruct_pkl_files_from_csv's zip-truncation already ignores
# everything past value_start+len(ALGO_NAMES), so this needs no change there (same trick the status
# block itself already relies on).
HEADER = [
    'experiment', 'parameter', 'seed',
    *ALGO_NAMES,
    *(f'{a}_status' for a in ALGO_NAMES),
    *(f'{a}_gevals' for a in ALGO_NAMES),
    *(f'{a}_oevals' for a in ALGO_NAMES),
    *(f'{a}_secs' for a in ALGO_NAMES),
    'peak_rss_mb',
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
    bounds = setting['bounds']
    target_value = _well_target_value(setting)

    noise_val = 1.0
    lr = 0.001

    # Shared initial swarm (2026-08-07), drawn once per task and applied to every algorithm via
    # sim.reset(particle_init=x0) below -- removes the largest source of the order-dependent RNG
    # coupling flagged in the plan's B8 finding (each algorithm previously drew its own random
    # start from wherever the shared global stream happened to be after every earlier algorithm's
    # own run, so adding/removing an algorithm or changing MAX_STEPS silently changed every later
    # algorithm's starting positions). Mirrors run_lennard_jones.py's existing x0/reset pattern.
    # Per-step noise draws during first_value_time are still sequential across algorithms (the
    # same as run_lennard_jones.py), so this does not make the comparison fully RNG-independent --
    # only the (typically most consequential) starting-position coupling is removed.
    x0 = np.random.uniform(
        [p[0] for p in particle_init], [p[1] for p in particle_init], (100, d)
    )

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

    # initial_temperature: 10 -> 14.0283 (2026-08-07, revised after a user-flagged review of the
    # first attempt at this fix). The flat T0=10 every driver previously shipped was never
    # calibrated. Two rounds of wrong fixes before landing here, kept in this comment because the
    # reasoning matters for anyone revisiting this:
    #   1. Matching SA's effective diffusion D*T0 to Softmin's noise_intensity throws away SA's
    #      whole design point (a high, annealed starting temperature).
    #   2. Tuning T0 to minimize escape time is degenerate (larger T0 is always faster, no interior
    #      optimum -- eventually indistinguishable from bounded random search).
    #   3. First attempt at an "initial acceptance ratio" (Kirkpatrick et al. 1983) fix picked
    #      T0=2*h with target ratio exp(-h/T0)=0.6-0.8. WRONG: it used the *global* barrier height
    #      h in the acceptance-ratio formula, but Kirkpatrick's ratio is about a *local* trial
    #      move's cost, not the total basin-to-basin separation -- and 2*h makes h/T0=0.5, so the
    #      barrier is thermodynamically almost transparent from step zero (Boltzmann factor 0.61),
    #      which trivializes exactly the obstacle Theorem 3.1 claims Softmin's mechanism reduces.
    #      Confirmed directly: at T0=2*h, SA is statistically indistinguishable from a
    #      gradient-free pure-diffusion control (same seeds, same escape steps to within noise).
    #   4. Re-tried with the literal acceptance-ratio *method* (a short trial run measuring the
    #      empirical fraction of worsening moves, matching Johnson et al. 1989's INITPROB
    #      procedure) applied per-particle to this continuous, always-accept SDE. It doesn't
    #      transfer: the measured worsening-fraction sits at ~0.4-0.5 across FOUR ORDERS OF
    #      MAGNITUDE of T0 (0.001 to 2.0), so it can't discriminate a T0 at all here -- Johnson et
    #      al.'s framework assumes an explicit Metropolis accept/reject step, which this
    #      unconditionally-accepted gradient+noise integrator doesn't have.
    #   5. Landed on: pick T0 so the barrier remains a genuine (not trivialized) thermodynamic
    #      obstacle -- T0 = h(delta=15)/4, giving Boltzmann factor exp(-4)=0.018 -- then validate
    #      directly (escape time + boundary-pinned-fraction), the same "deciding measurement"
    #      convention every other baseline hyperparameter in this file already uses. Reaches
    #      target within a few hundred steps across the full double-well delta=5/15/30 range
    #      (unlike the old T0=10, which failed completely at delta=30) with pinned_frac staying
    #      low (<=0.11), and transfers cleanly to quadruple_well A=15. Its escape is still
    #      diffusion-dominated (confirmed: statistically similar to a pure-diffusion control) --
    #      that is expected, not a red flag: unmodified Langevin/SA has no ascent mechanism, so it
    #      can only ever escape via noise beating its own restoring gradient. That is precisely the
    #      phenomenon Theorem 3.1 claims Softmin's sign-flipping drift improves on, so SA facing a
    #      genuine, noise-driven barrier is the correct, fair baseline -- not a flaw to fix further.
    #
    # Precise transfer boundary (100,000-step probe, fixed T0=14.0283, no per-delta retuning):
    # reaches target through delta=20 (t=0.007 at delta=1 up to t=2.227 at delta=20, monotonically
    # slower), then fails from delta=25 onward (value_stagnation at the 100,000-step probe ceiling
    # for both delta=25 and delta=30).
    #
    # RE-VERIFIED (2026-08-07, prompted by the stagnation_patience 50->500 fix, which found that
    # same old default false-aborting a genuine Softmin escape at quadruple_well A=30 --
    # best_global_value plateaus for up to ~116,000 steps there before resuming, longer than this
    # SA probe's entire old 100,000-step budget, so the delta>=25 claim needed independent
    # confirmation, not just a rerun under the new default). Re-ran delta=25 and delta=30 UNCAPPED
    # (early-stop disabled entirely, manual step loop, no probe ceiling) out to 1,500,000 steps --
    # 15x the original probe and comparable to a meaningful fraction of the real MAX_STEPS=1e7
    # budget. Result: best_global_value is bit-for-bit IDENTICAL from step 10,000 through step
    # 1,500,000 at both deltas (no drift across 3 orders of magnitude in step count) -- unlike
    # quadruple_well A=30's temporary plateau, this is a permanent freeze, consistent with SA's own
    # geometric cooling schedule (rate=0.99995) driving T_t = T0*rate^t to ~0 well before step
    # 200,000 (rate^200000 = exp(200000*ln(0.99995)) = exp(-10.0)), after which the swarm has no
    # noise left to escape with and gradient descent alone leaves it wherever it already sat. This
    # CONFIRMS the delta>=25 exclusion as a genuine, permanent negative result -- not a false-abort
    # artifact -- and is a materially stronger check than the original 100,000-step probe.
    #
    # See hyperparameter_diagnostics.py's sa_initial_temperature_candidates/sa_temperature_report.
    SA_INITIAL_TEMPERATURE = 14.0283

    # Diagnostic-guided, 2026-08-06: swept the same 11-value beta grid as Softmin plus 2 annealed
    # schedule variants (13 candidates, matching Softmin's budget) at both mid-sweep settings.
    # Genuine negative finding, not a tuning gap: EVERY candidate times out via position_collapse
    # at both double_well (delta=15) and quadruple_well (A=15) -- CBO's consensus dynamics collapse
    # onto the wrong basin regardless of beta or annealing schedule on these benchmarks.
    #
    # beta: 10.0 -> 0.01 (2026-08-07, user-prompted literature check). Unlike PSO's Clerc-Kennedy
    # constriction coefficients, CBO papers (e.g. Carrillo, Choi, Totzeck, Tse 2018) don't
    # prescribe a single canonical numeric beta, but 0.01-0.02 is the typical range cited in
    # practice. 10.0 was originally set a priori, pre-dating any diagnostic pass. Before switching,
    # confirmed directly (not assumed) that this doesn't change the result: beta in
    # {0.01, 0.02, 0.1, 1.0, 10.0} all collapse at the IDENTICAL step on both this benchmark and
    # quadruple_well (1000 for GradientAwareCBO, 14000 for standard CBO) -- the collapse here is
    # beta-independent, consistent with vanilla CBO's known structural vulnerability to premature
    # consensus (its diffusion is anisotropic and self-shrinking as particles concentrate, with no
    # reinitialization mechanism, regardless of how sharply beta weights the consensus point) rather
    # than a symptom of beta being poorly chosen. So 0.01 is adopted as the literature-aligned,
    # defensible value with no change to the (still genuine, still structural) negative finding.
    # See Remark rem:cbo-pso-negative-result in the paper.
    CBO_BETA = 0.01

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
        noise=noise_val, cooling_schedule="geometric", initial_temperature=SA_INITIAL_TEMPERATURE
    )

    langevin = GaussianDiffusion(
        n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init,
        bounds=bounds, learning_rate=lr, noise=noise_val,
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

    # 3. Compute hit times against the value-space target (see WELL_TARGET_EPS_FRACTION). Each
    # sim's early-stop status is read from self._last_stop_reason right after the call, since
    # first_value_time's return value alone can't distinguish a real max_steps run from an
    # early-stopped one (see ALGO_NAMES comment).
    algos = {
        'softmin': softmin, 'sa': sa, 'langevin': langevin, 'cbo': cbo, 'cbo_annealed': cbo_annealed,
        'pso': pso, 'standard_cbo': standard_cbo, 'standard_pso': standard_pso,
    }
    for sim in algos.values():
        sim.reset(particle_init=x0)

    times = {}
    statuses = {}
    for name, sim in algos.items():
        times[name] = sim.first_value_time(target_value=target_value, max_steps=MAX_STEPS)
        statuses[name] = sim._last_stop_reason

    return (exp_name, param, seed,
            *(times[a] for a in ALGO_NAMES),
            *(statuses[a] for a in ALGO_NAMES),
            *(algos[a].n_gradient_evals for a in ALGO_NAMES),
            *(algos[a].n_objective_evals for a in ALGO_NAMES),
            *(algos[a].wall_clock_seconds for a in ALGO_NAMES),
            peak_rss_mb())


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


def filename_for_status(key, algo):
    exp, param = key
    param = float(param)
    if param.is_integer():
        param = int(param)
    return f"results/{algo}_{exp}_{param}_status.pkl"


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

    # Status pkls (2026-08-07), parallel to the time pkls above -- lets the notebook mask out
    # timed-out seeds by their honest self._last_stop_reason instead of a numeric mask_below
    # cutoff that silently stops matching once MAX_STEPS changes (see the plan's B3 finding).
    reconstruct_pkl_files_from_csv(
        results_csv=RESULTS_CSV,
        key_indices=[0, 1],
        value_start=3 + len(ALGO_NAMES),  # skip seed + times; status block starts here
        algo_names=ALGO_NAMES,
        filename_fn=filename_for_status,
        value_transform=str,
    )
