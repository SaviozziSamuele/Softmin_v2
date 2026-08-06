"""
Low-effort, reusable diagnostics for sanity-checking each algorithm's hyperparameters against a
benchmark's natural value scale.

Every check below is evaluated at a representative MIDDLE-of-sweep setting for each benchmark
(not the hardest) -- picking hyperparameters against the hardest point in a sweep would just fit
them to the one setting the sweep is supposed to stress-test. The point of tuning at a middle
setting is to let the existing entry-time-vs-difficulty sweeps show, honestly, whether a fixed
hyperparameter choice remains resilient as the problem gets harder.

Run directly (`python src/hyperparameter_diagnostics.py`) to print a report across double_well /
quadruple_well / high-dim Ackley at their sweep midpoints (delta=15, A=15, d=50 -- the midpoints
of the ranges actually swept in run_experiments_resumable.py / run_ackley_dimensions.py).
"""
import numpy as np

from potentials import (
    asymmetric_double_well,
    unbalanced_quadruple_well_setting,
    high_dim_ackley_function_setting,
)

# Matches run_experiments_resumable.py's DOUBLE_WELL_TILT -- these diagnostics must validate
# hyperparameters against the actual benchmark production uses.
DOUBLE_WELL_TILT = 0.05
from simulations import (
    SoftminEnergyAdagrad, SimulatedAnnealing, GradientAwareCBO, GradientAwarePSO, CBO, PSO,
)

# beta=10.0 is the value actually shipped in every production driver (run_experiments_resumable.py,
# run_ackley_dimensions.py, run_ablation_experiment.py, run_lennard_jones.py). It was reached in
# two stages, both preserved below so the audit trail is reproducible from this file alone:
#  1. beta_gap_report/cbo_ess_report (the "fairness" diagnostics) initially pointed to a much
#     smaller beta (~0.03-1.0), since they measure whether EVERY particle keeps getting
#     rescue-drift.
#  2. escape_time_validation -- the metric that actually matters, since first_entry_time only
#     cares about the FASTEST particle -- showed the opposite ranking: escape got monotonically
#     SLOWER as beta shrank below 10 on every benchmark tested (double_well, quadruple_well,
#     high-dim Ackley, and a spot check on Lennard-Jones), and higher values up to 50 never beat
#     10. The same function also settled a second question (should beta decay or grow over time
#     instead of staying fixed?): neither a decaying nor an increasing schedule beat a
#     well-chosen fixed beta at any setting tested, and the gap widened at harder settings (up to
#     ~20x on quadruple_well A=30) -- see Remark rem:higher-dim in the paper.
# Moral: a fairness/concentration diagnostic is a useful cheap first pass, but it is NOT a
# substitute for direct escape-time validation against the actual metric being optimized.


# ---------------------------------------------------------------------------
# Softmin: beta*(f_i - J_beta) gap distribution
# ---------------------------------------------------------------------------

def beta_gap_report(f_vec, f_grad_vec, d, bounds, particle_init, lr, beta_values,
                     n_steps=3000, noise_intensity=1.0, seed=42):
    """For each candidate beta, run SoftminEnergyAdagrad and report the distribution of
    beta*(f_i - J_beta) across the ensemble (only particles behind the leader have a positive
    gap) -- this quantity is exactly what multiplies each trailing particle's own rescue-drift
    inside its Gibbs weight (see SoftminEnergyAdagrad.step()'s velocity_scale), so a large gap
    means that particle's own further progress is exponentially suppressed. Report the median gap
    and the fraction past the "essentially frozen" threshold gap>5, pooled across all steps and
    particles.
    """
    results = {}
    for beta in beta_values:
        np.random.seed(seed)
        sim = SoftminEnergyAdagrad(
            n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds, particle_init=particle_init,
            initial_lr=lr, adaptive_lr=False, noise_intensity=noise_intensity, noise="normal",
            beta_decay=None, beta=beta,
        )
        gaps = []
        for _ in range(n_steps):
            energy = sim.Js[-1]
            gap = beta * (sim.values - energy)
            positive = gap[gap > 0]
            if positive.size:
                gaps.append(positive)
            sim.step()
        gaps = np.concatenate(gaps) if gaps else np.array([0.0])
        results[beta] = {
            'median_gap': float(np.median(gaps)),
            'frac_frozen': float(np.mean(gaps > 5)),
        }
    return results


# ---------------------------------------------------------------------------
# Simulated Annealing: is the cooling schedule matched to this benchmark's timescale?
# ---------------------------------------------------------------------------

def sa_cooling_report(f_vec, f_grad_vec, d, bounds, particle_init, target_set, lr, cooling_rate,
                       initial_temperature=10.0, probe_max_steps=500_000, seed=42):
    """The geometric schedule T(t) = T0 * cooling_rate**t is purely analytic (independent of the
    actual dynamics), so t* (the step at which T first drops to 5% of T0) is computed in closed
    form. What needs an actual run is the comparison point: how many steps does this benchmark's
    mid-sweep setting typically take to resolve (first_entry_time) under the current
    hyperparameters? If escape happens well after t*, cooling has already fully engaged by the
    time it matters (as intended). If escape typically happens well before t*, cooling barely
    engages at all and SA is effectively constant-temperature Langevin for this benchmark -- the
    same failure mode the 2026-08-04 n_steps bug caused everywhere, just via a mismatched
    timescale instead of a frozen self.n_steps.
    """
    t_star = np.log(0.05) / np.log(cooling_rate)

    np.random.seed(seed)
    sim = SimulatedAnnealing(
        n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds, particle_init=particle_init,
        learning_rate=lr, cooling_rate=cooling_rate, noise=1.0,
        cooling_schedule="geometric", initial_temperature=initial_temperature,
    )
    entry_time = sim.first_entry_time(target_set=target_set, max_steps=probe_max_steps)
    entry_steps = entry_time / lr  # first_entry_time returns i * learning_rate

    return {
        't_star_steps': float(t_star),
        'entry_steps': float(entry_steps),
        'timed_out': entry_steps >= probe_max_steps,
        't_star_over_entry': float(t_star / entry_steps) if entry_steps > 0 else float('inf'),
    }


# ---------------------------------------------------------------------------
# CBO: effective sample size of the consensus point's Gibbs weights
# ---------------------------------------------------------------------------

def cbo_ess_report(f_vec, f_grad_vec, d, bounds, particle_init, lr, beta_values, consensus_rate,
                    noise_intensity, n_steps=3000, seed=42):
    """For each candidate beta, run GradientAwareCBO and report the effective sample size
    (ESS = 1 / sum(w_i^2)) of the consensus point's Gibbs weights, as a fraction of swarm size n.
    ESS/n near 0 means the consensus point has collapsed onto effectively one particle (no real
    "consensus", just chasing the current best); ESS/n near 1 means it's close to an unweighted
    average, giving little pull toward the best-found value. The same softmax-concentration
    mechanism as softmin's beta_gap_report, applied to CBO's own use of beta.
    """
    results = {}
    for beta in beta_values:
        np.random.seed(seed)
        sim = GradientAwareCBO(
            n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds, particle_init=particle_init,
            learning_rate=lr, consensus_rate=consensus_rate, noise_intensity=noise_intensity,
            beta=beta,
        )
        ess_fracs = []
        for _ in range(n_steps):
            min_val = np.min(sim.values)
            w = np.exp(-beta * (sim.values - min_val))
            w /= w.sum()
            ess_fracs.append(1.0 / np.sum(w ** 2) / sim.n)
            sim.step()
        results[beta] = {'median_ess_frac': float(np.median(ess_fracs))}
    return results


# ---------------------------------------------------------------------------
# Generic escape-time sweep -- the deciding measurement, shared by every algorithm family below
# ---------------------------------------------------------------------------

def escape_time_sweep(target_set, sim_factories, max_steps=20_000, seed=42):
    """The deciding measurement, used for every algorithm's calibration below: unlike a fairness
    proxy (ensemble-wide, or ESS-based), this directly measures first_entry_time -- the fastest
    particle reaching the target -- which is what the benchmarks actually score.

    `sim_factories` is `{label: zero-arg callable returning a fresh Simulation}`, the same pattern
    already used by `stagnation_check_validation`. Returns
    `{label: (entry_time, timed_out, best_global_value, stop_reason)}` -- `stop_reason` (from
    `Simulation._last_stop_reason`) is what actually distinguishes a real hit from an early-stopped
    or exhausted run; `timed_out` is just `stop_reason != 'target_reached'`.
    """
    results = {}
    for label, factory in sim_factories.items():
        np.random.seed(seed)
        sim = factory()
        t = sim.first_entry_time(target_set=target_set, max_steps=max_steps)
        stop_reason = sim._last_stop_reason
        results[label] = (t, stop_reason != 'target_reached', float(sim.best_global_value), stop_reason)
    return results


# ---------------------------------------------------------------------------
# Softmin: direct escape-time validation -- fixed vs. decaying vs. increasing beta
# ---------------------------------------------------------------------------

def escape_time_validation(f_vec, f_grad_vec, d, bounds, particle_init, target_set, lr,
                            fixed_betas, schedules=(), max_steps=20_000, seed=42):
    """Softmin-specific wrapper around `escape_time_sweep`. `schedules` is a list of
    (label, beta0, beta_decay) triples layered on top of SoftminEnergyAdagrad's own beta_decay
    mechanism (values < 1 anneal down, values > 1 anneal up -- step() does not restrict beta_decay
    to (0, 1); growth rates >= 1.01 were observed to overflow beta to inf well within
    max_steps=20_000, so keep any growth schedule tested here below that).
    """
    def _factory(beta0, beta_decay):
        return lambda: SoftminEnergyAdagrad(
            n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds, particle_init=particle_init,
            initial_lr=lr, adaptive_lr=False, noise_intensity=1.0, noise="normal",
            beta_decay=beta_decay, beta=beta0,
        )
    sim_factories = {f'fixed beta={beta}': _factory(beta, None) for beta in fixed_betas}
    sim_factories.update({label: _factory(beta0, beta_decay) for label, beta0, beta_decay in schedules})
    return escape_time_sweep(target_set, sim_factories, max_steps=max_steps, seed=seed)


# ---------------------------------------------------------------------------
# SA: direct escape-time validation across cooling_rate, derived from target cooling timescales
# ---------------------------------------------------------------------------

def sa_escape_time_validation(f_vec, f_grad_vec, d, bounds, particle_init, target_set, lr,
                               cooling_rates, initial_temperature=10.0, max_steps=20_000, seed=42):
    """SA-specific wrapper around `escape_time_sweep`. `cooling_rates` is a plain list of rates
    (not (label, rate) pairs) -- labeled automatically here so callers can pass the
    timescale-derived list directly (see SA_COOLING_RATES in __main__)."""
    def _factory(rate):
        return lambda: SimulatedAnnealing(
            n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init, bounds=bounds,
            learning_rate=lr, cooling_rate=rate, noise=1.0, cooling_schedule="geometric",
            initial_temperature=initial_temperature,
        )
    sim_factories = {f'cooling_rate={rate}': _factory(rate) for rate in cooling_rates}
    return escape_time_sweep(target_set, sim_factories, max_steps=max_steps, seed=seed)


# ---------------------------------------------------------------------------
# CBO: direct escape-time validation across beta, plus the annealed-schedule question
# ---------------------------------------------------------------------------

def cbo_escape_time_validation(cls, f_vec, f_grad_vec, d, bounds, particle_init, target_set, lr,
                                consensus_rate, noise_intensity, fixed_betas, schedules=(),
                                max_steps=20_000, seed=42):
    """CBO-specific wrapper around `escape_time_sweep`. `cls` is `CBO` or `GradientAwareCBO`.
    `schedules` is a list of (label, beta0, beta_growth) triples -- growth-only, unlike Softmin's
    beta_decay (CBO's beta_growth must be > 1; see GradientAwareCBO/CBO's own validation)."""
    def _factory(beta0, beta_growth):
        return lambda: cls(
            n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds, particle_init=particle_init,
            learning_rate=lr, consensus_rate=consensus_rate, noise_intensity=noise_intensity,
            beta=beta0, beta_growth=beta_growth,
        )
    sim_factories = {f'fixed beta={beta}': _factory(beta, None) for beta in fixed_betas}
    sim_factories.update({label: _factory(beta0, beta_growth) for label, beta0, beta_growth in schedules})
    return escape_time_sweep(target_set, sim_factories, max_steps=max_steps, seed=seed)


# ---------------------------------------------------------------------------
# PSO: direct escape-time validation across a small, named set of (w, c1, c2) triples
# ---------------------------------------------------------------------------

def pso_escape_time_validation(cls, f_vec, f_grad_vec, d, bounds, particle_init, target_set, lr,
                                candidates, max_steps=20_000, seed=42):
    """PSO-specific wrapper around `escape_time_sweep`. `cls` is `PSO` or `GradientAwarePSO`.
    `candidates` is `{label: (w, c1, c2)}` -- PSO has no single temperature-like scalar to sweep,
    so this is a small, named, literature-grounded set instead of a 1D grid (see PSO_CANDIDATES
    in __main__)."""
    sim_factories = {
        label: (lambda w=w, c1=c1, c2=c2: cls(
            n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds, particle_init=particle_init,
            learning_rate=lr, w=w, c1=c1, c2=c2,
        ))
        for label, (w, c1, c2) in candidates.items()
    }
    return escape_time_sweep(target_set, sim_factories, max_steps=max_steps, seed=seed)


# ---------------------------------------------------------------------------
# Early-stop mechanism validation -- is Simulation._check_early_stop itself trustworthy?
# ---------------------------------------------------------------------------

def stagnation_check_validation(f_vec, f_grad_vec, d, bounds, particle_init, target_set,
                                 sim_factories, probe_max_steps=50_000, seed=42,
                                 stagnation_patience=5, stagnation_rtol=1e-4,
                                 stagnation_atol=1e-8, early_stop_check_interval=1000):
    """Validates the early-stop mechanism (2026-08-06, added after CBO/PSO were found to either
    abort near-instantly via position collapse or, worse for PSO, never abort at all and grind the
    real max_steps budget -- see Remark on CBO/PSO exclusion in the paper) against independent
    ground truth, not by re-invoking the same abort logic being validated.

    For each (name, factory) in `sim_factories`: runs once with both mechanisms enabled (the real,
    deployed `first_entry_time` call) and once with a raw, never-aborting manual step loop up to
    the same `probe_max_steps`, recording `best_global_value` at every `early_stop_check_interval`.
    If the enabled run aborted at step S, checks whether the disabled run's `best_global_value`
    ever improved (beyond `stagnation_rtol`/`stagnation_atol`) at any checkpoint after S -- if so,
    the abort likely cut off real progress (a false positive, the failure mode this function exists
    to catch); if not, nothing was lost by stopping early.
    """
    lower = np.array([s[0] for s in target_set])
    upper = np.array([s[1] for s in target_set])
    report = {}

    for name, factory in sim_factories.items():
        np.random.seed(seed)
        sim_on = factory()
        t_on = sim_on.first_entry_time(
            target_set=target_set, max_steps=probe_max_steps,
            stagnation_patience=stagnation_patience, stagnation_rtol=stagnation_rtol,
            stagnation_atol=stagnation_atol, early_stop_check_interval=early_stop_check_interval,
        )
        reason_on, step_on = sim_on._last_stop_reason, sim_on._last_stop_step

        np.random.seed(seed)
        sim_off = factory()
        checkpoints, best_history = [], []
        target_reached_step = None
        for i in range(probe_max_steps):
            sim_off.n_steps = i + 1
            sim_off.step()
            if target_reached_step is None:
                in_bounds = np.all((sim_off.particles > lower) & (sim_off.particles < upper), axis=1)
                if np.any(in_bounds):
                    target_reached_step = i
            if i % early_stop_check_interval == 0:
                checkpoints.append(i)
                best_history.append(float(sim_off.best_global_value))
        checkpoints = np.array(checkpoints)
        best_history = np.array(best_history)

        improved_after_abort = None
        if reason_on in ('position_collapse', 'value_stagnation'):
            before_mask = checkpoints <= step_on
            at_abort = best_history[before_mask][-1] if np.any(before_mask) else best_history[0]
            after = best_history[checkpoints > step_on]
            improved_after_abort = bool(after.size and np.any(
                (after < at_abort) & ~np.isclose(
                    after, at_abort, rtol=stagnation_rtol, atol=stagnation_atol
                )
            ))

        report[name] = {
            'enabled_reason': reason_on, 'enabled_step': step_on, 'enabled_time': t_on,
            'uncapped_target_reached_step': target_reached_step,
            'improved_after_abort_in_uncapped_run': improved_after_abort,
        }
    return report


# ---------------------------------------------------------------------------
# Generic stability probe (lr sanity check; also usable for PSO/w/c1/c2)
# ---------------------------------------------------------------------------

def stability_probe(sim_factory, n_steps=2000):
    """Low-effort divergence check: run n_steps, confirm particles/values stay finite and report
    whether the swarm ended up pinned at the domain boundary (a sign the dynamics is fighting the
    box rather than actually optimizing)."""
    sim = sim_factory()
    for _ in range(n_steps):
        sim.step()
    finite = bool(np.all(np.isfinite(sim.particles)) and np.all(np.isfinite(sim.values)))
    lb = np.array([b[0] for b in sim.bounds])
    ub = np.array([b[1] for b in sim.bounds])
    finite_bounds = bool(np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)))
    at_bound = (finite_bounds and bool(np.any(
        np.isclose(sim.particles, lb, atol=1e-6) | np.isclose(sim.particles, ub, atol=1e-6)
    )))
    return {
        'finite': finite,
        'at_bound': at_bound if finite_bounds else 'n/a (unbounded domain)',
        'best_global_value': float(sim.best_global_value),
    }


LR = 0.001
GAP_CANDIDATES = [10.0, 3.0, 1.0, 0.3, 0.1]
ESCAPE_FIXED_BETAS = [50.0, 30.0, 20.0, 15.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01]
ESCAPE_SCHEDULES = [
    ('decay 10->, r=0.999', 10.0, 0.999),
    ('growth 0.01->, r=1.005', 0.01, 1.005),
]
# Chosen value, shipped in every production driver.
CHOSEN_BETA = 10.0

# SA: candidate cooling_rate list derived the same principled way as the bounds fix (2026-08-06)
# in potentials.py -- not arbitrary decimals, but rates corresponding to clean target cooling
# *timescales* (t_star = steps to reach 5% of initial temperature, exactly what sa_cooling_report
# already computes analytically): cooling_rate = 0.05**(1/t_star). The shipped 0.99995 sits at
# t_star~59,900, between the 30,000- and 100,000-step candidates, so this brackets it on both
# sides rather than only extending in one direction.
SA_T_STARS = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
SA_COOLING_RATES = [0.05 ** (1 / t) for t in SA_T_STARS]

# CBO: beta_growth schedule variants (growth-only, unlike Softmin's beta_decay -- see
# cbo_escape_time_validation). Safe to test the faster variant now that beta_max caps overflow.
CBO_SCHEDULES = [
    ('growth 10->, r=1.001 (shipped)', 10.0, 1.001),
    ('growth 1->, r=1.01', 1.0, 1.01),
]

# PSO: no single temperature-like scalar, so a small, named, literature-grounded set instead of a
# 1D grid. 'aggressive/stress-test' is deliberately non-contractive (w>=1) -- PSO's constructor
# has no stability validation at all (unlike CBO's), so this gives the stability_probe pre-filter
# below something real to catch.
PSO_CANDIDATES = {
    'canonical constriction (default)': (0.729, 1.49445, 1.49445),
    'Kennedy-Eberhart 1995': (1.0, 2.0, 2.0),
    'Eberhart-Shi low-w': (0.4, 2.0, 2.0),
    'Eberhart-Shi high-w': (0.9, 2.0, 2.0),
    'social-weighted': (0.729, 1.0, 2.0),
    'aggressive/stress-test': (1.2, 2.0, 2.0),
}

# Calibration budget, tracked explicitly (2026-08-06) so "same effort across algorithms" is
# checkable, not just asserted -- see paper's app:hyperparameter-methodology.
CALIBRATION_BUDGET = {
    'softmin': (len(ESCAPE_FIXED_BETAS), len(ESCAPE_SCHEDULES)),
    'sa': (len(SA_COOLING_RATES), 0),
    'cbo': (len(ESCAPE_FIXED_BETAS), len(CBO_SCHEDULES)),
    'pso': (len(PSO_CANDIDATES), 0),
}


def _print_escape_results(results):
    for label, (t, timed_out, best_val, reason) in results.items():
        flag = "TIMEOUT" if timed_out else "reached"
        print(f"  {label:>32}: {flag:<8} reason={reason:<18} t={t:.3f} best_global_value={best_val:.4g}")


def report_benchmark(title, setting, d, cbo_noise_intensity=np.sqrt(2.0), cbo_consensus_rate=1.0):
    """Runs the full four-algorithm calibration story (fairness proxy -> deciding measurement)
    against one benchmark setting, printed in one block. Shared across double_well/quadruple_well/
    Ackley so the same report can't silently drift out of sync between benchmarks."""
    f_vec, f_grad_vec = setting['f_vec'], setting['f_grad_vec']
    bounds, particle_init, target_set = setting['bounds'], setting['particle_init'], setting['set']

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("\n-- softmin beta*gap (fairness proxy -- favors SMALL beta; do not tune on this alone) --")
    for beta, stats in beta_gap_report(
        f_vec, f_grad_vec, d, bounds, particle_init, LR, GAP_CANDIDATES,
    ).items():
        print(f"  beta={beta:>6}: median_gap={stats['median_gap']:.3g} "
              f"frac_frozen={stats['frac_frozen']:.3f}")
    print("-- softmin escape time (deciding measurement) --")
    _print_escape_results(escape_time_validation(
        f_vec, f_grad_vec, d, bounds, particle_init, target_set, LR,
        ESCAPE_FIXED_BETAS, ESCAPE_SCHEDULES,
    ))

    print("\n-- SA cooling report at the shipped rate (fairness proxy) --")
    print(f"  {sa_cooling_report(f_vec, f_grad_vec, d, bounds, particle_init, target_set, LR, 0.99995)}")
    print("-- SA escape time across timescale-derived cooling_rate candidates (deciding measurement) --")
    _print_escape_results(sa_escape_time_validation(
        f_vec, f_grad_vec, d, bounds, particle_init, target_set, LR, SA_COOLING_RATES,
    ))

    print("\n-- CBO consensus ESS/n (fairness proxy) --")
    for beta, stats in cbo_ess_report(
        f_vec, f_grad_vec, d, bounds, particle_init, LR, GAP_CANDIDATES,
        consensus_rate=cbo_consensus_rate, noise_intensity=cbo_noise_intensity,
    ).items():
        print(f"  beta={beta:>6}: median_ess_frac={stats['median_ess_frac']:.3f}")
    print("-- CBO escape time across beta + annealed-schedule candidates (deciding measurement) --")
    _print_escape_results(cbo_escape_time_validation(
        GradientAwareCBO, f_vec, f_grad_vec, d, bounds, particle_init, target_set, LR,
        cbo_consensus_rate, cbo_noise_intensity, ESCAPE_FIXED_BETAS, CBO_SCHEDULES,
    ))

    print("\n-- PSO stability_probe pre-filter --")
    filtered_pso = {}
    for label, (w, c1, c2) in PSO_CANDIDATES.items():
        stats = stability_probe(lambda w=w, c1=c1, c2=c2: GradientAwarePSO(
            n=100, d=d, f=f_vec, df=f_grad_vec, bounds=bounds, particle_init=particle_init,
            learning_rate=LR, w=w, c1=c1, c2=c2,
        ))
        verdict = "OK" if stats['finite'] else "EXCLUDED (diverged)"
        print(f"  {label:>32}: {verdict:<20} {stats}")
        if stats['finite']:
            filtered_pso[label] = (w, c1, c2)
    print("-- PSO escape time across surviving candidates (deciding measurement) --")
    _print_escape_results(pso_escape_time_validation(
        GradientAwarePSO, f_vec, f_grad_vec, d, bounds, particle_init, target_set, LR, filtered_pso,
    ))


if __name__ == '__main__':
    dw = asymmetric_double_well(delta=15, tilt=DOUBLE_WELL_TILT)
    report_benchmark("DOUBLE WELL (delta=15, mid-sweep of 1..30)", dw, d=1)

    qw = unbalanced_quadruple_well_setting(A=15, delta=-1)
    report_benchmark("QUADRUPLE WELL (A=15, mid-sweep of 2..30)", qw, d=2)

    ak = high_dim_ackley_function_setting(d=50)
    ak_init = ak.get('particle_init', [[-10.0, 10.0]] * 50)
    ak_full = {**ak, 'particle_init': ak_init}
    report_benchmark("HIGH-DIM ACKLEY (d=50, mid-sweep of 10..90)", ak_full, d=50)

    print("\n" + "=" * 70)
    print("CBO stability boundary spot-check (consensus_rate=1.0, noise_intensity=sqrt(2))")
    print("=" * 70)
    print("Shipped combo exactly saturates the code's own documented '2*lambda > sigma**2' "
          "stability comment (2*1.0 = 2 = (sqrt(2))**2, equality not strict inequality).")
    cbo_stab = stability_probe(lambda: GradientAwareCBO(
        n=100, d=1, f=dw['f_vec'], df=dw['f_grad_vec'], bounds=dw['bounds'],
        particle_init=dw['particle_init'], learning_rate=LR, consensus_rate=1.0,
        noise_intensity=np.sqrt(2.0), beta=CHOSEN_BETA,
    ))
    print(f"  {cbo_stab}")

    print("\n" + "=" * 70)
    print("EARLY-STOP MECHANISM VALIDATION (quadruple well, A=15 mid-sweep)")
    print("=" * 70)
    qw2 = unbalanced_quadruple_well_setting(A=15, delta=-1)

    def _factory(cls, **kwargs):
        return lambda: cls(n=100, d=2, f=qw2['f_vec'], df=qw2['f_grad_vec'], bounds=qw2['bounds'],
                            particle_init=qw2['particle_init'], learning_rate=LR, **kwargs)

    sim_factories = {
        'pso': _factory(PSO, w=0.729, c1=1.49445, c2=1.49445),
        'cbo': _factory(CBO, consensus_rate=1.0, noise_intensity=1.0, beta=CHOSEN_BETA),
        'gradient_aware_pso': _factory(GradientAwarePSO, w=0.729, c1=1.49445, c2=1.49445),
        'gradient_aware_cbo': _factory(GradientAwareCBO, consensus_rate=1.0,
                                        noise_intensity=np.sqrt(2.0), beta=CHOSEN_BETA),
        'softmin': lambda: SoftminEnergyAdagrad(
            n=100, d=2, f=qw2['f_vec'], df=qw2['f_grad_vec'], bounds=qw2['bounds'],
            particle_init=qw2['particle_init'], initial_lr=LR, adaptive_lr=False,
            noise_intensity=1.0, noise="normal", beta_decay=None, beta=CHOSEN_BETA,
        ),
    }
    for name, stats in stagnation_check_validation(
        qw2['f_vec'], qw2['f_grad_vec'], 2, qw2['bounds'], qw2['particle_init'], qw2['set'],
        sim_factories,
    ).items():
        print(f"  {name:>20}: reason={stats['enabled_reason']:<18} step={stats['enabled_step']:<7} "
              f"uncapped_target_reached_step={stats['uncapped_target_reached_step']} "
              f"improved_after_abort={stats['improved_after_abort_in_uncapped_run']}")

    print("\n" + "=" * 70)
    print("CALIBRATION BUDGET (candidates x 3 benchmarks x one escape_time_sweep run each)")
    print("=" * 70)
    print(f"  {'algorithm':<10} {'fixed candidates':>18} {'schedule variants':>18} "
          f"{'per benchmark':>15} {'x3 benchmarks':>15}")
    for algo, (n_fixed, n_schedules) in CALIBRATION_BUDGET.items():
        per_benchmark = n_fixed + n_schedules
        print(f"  {algo:<10} {n_fixed:>18} {n_schedules:>18} {per_benchmark:>15} {per_benchmark*3:>15}")

    print("\n" + "=" * 70)
    print(f"CHOSEN: beta={CHOSEN_BETA} (fixed, no schedule) for all three benchmarks above, plus")
    print("Lennard-Jones (spot-checked separately at N=13/N=38 -- same conclusion, gap widens at")
    print("N=38). SA cooling_rate=0.99995 for all benchmarks (was split 0.9999/0.99995 before the")
    print("quadruple-well bug fix above). lr=0.001 unified across every driver.")
    print("=" * 70)
