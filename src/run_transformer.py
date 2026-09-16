"""Transformer training benchmark: optimizing a small transformer's weights at increasing depth.

The real-world counterpart to the analytic benchmarks. Every other driver optimizes a closed-form
landscape; this one optimizes the training loss of an actual neural network (modular addition, see
nn_objective.py for the task and its literature grounding), with depth as the difficulty axis --
deeper models are harder to optimize and more seed-sensitive, which is exactly the axis the
paper's reliability claim is about.

Baselines. Adam and SGD are included because a neural-network comparison without them is not a
comparison; they are run as n INDEPENDENT chains (see simulations._IndependentOptimizer), so they
receive the same n restarts and the same per-step budget Softmin's ensemble consumes, and the
per-particle evaluation counts are directly comparable. SA and Langevin carry over from the well
benchmarks; CBO and PSO are excluded, see ALGO_NAMES below for why.

Why this driver does NOT use run_resumable_pool. experiments.run_resumable_pool parallelizes across
tasks with multiprocessing.Pool, which forks workers; CUDA does not survive fork, and on Windows
spawned workers would each re-import torch and re-upload the dataset. Here the parallelism is
already inside each task -- the n particles are batched into one vmap'd GPU call -- so tasks are
run SERIALLY in one process, reusing get_completed_tasks for the same crash-resumability and the
same flush-and-fsync-per-row CSV append.

Splitting across GPUs. Tasks are (depth, seed) pairs. Any subset can be selected with --tasks
(`50:3` is one task, a bare `10` is every seed of that depth), and each invocation writes its own
CSV named by --tag, so several single-GPU jobs run the sweep in parallel without sharing a file.
Cost is roughly linear in depth, so depth 50 alone is over half the sweep: balancing has to split
depth 50 by seed, not just split the sweep by depth (jobs/submit_transformer_split.sh). --merge
then combines the shard CSVs into RESULTS_CSV, deduplicating on (depth, seed).

Run:  cd src && python run_transformer.py                     (whole sweep, one process)
      python run_transformer.py --tasks 50:0 50:1 20 --tag a   (one shard)
      python run_transformer.py --merge                        (combine shards)
"""
import argparse
import csv
import glob
import logging
import os
import time

import numpy as np
import torch

from experiments import get_completed_tasks, peak_rss_mb
from nn_objective import transformer_setting
from simulations import (
    SoftminEnergyAdagrad,
    SimulatedAnnealing,
    GaussianDiffusion,
    Adam,
    SGD,
)

logger = logging.getLogger(__name__)

RESULTS_CSV = 'results/master_transformer.csv'
# Per-shard CSVs, one per --tag; --merge combines them into RESULTS_CSV.
SHARD_CSV = 'results/master_transformer_shard_{tag}.csv'

# Depth is the difficulty axis. d_model and n_heads are fixed in nn_objective.transformer_setting,
# so depth is the only thing that varies across the sweep.
#
# Aggressive range, chosen after a feasibility probe (probe_depth_feasibility.py). The earlier
# {1,2,4,6} did not clearly create an optimization-difficulty gradient -- Adam solved depth 2
# FASTER than depth 1 -- so the sweep is pushed an order of magnitude deeper. The probe confirms
# this is sound rather than merely harder: gradient norms stay flat at ~0.8-0.9 from depth 5 to
# depth 50 (no explosion, no vanishing) and Adam moves the loss well below chance at every depth,
# so a 50-layer model here is genuinely trainable and not a guaranteed-failure point. Depth 2 is
# kept as a shallow anchor at ~2% of the sweep's cost.
DEPTHS = [2, 5, 10, 20, 50]

N_SEEDS = 5

# Ensemble size. Far below the n=100 used on the analytic benchmarks: each step costs n full
# forward+backward passes over the training set, so n is the dominant cost term here. 20 -> 10 to
# pay for the deeper sweep (see DEPTHS); 10 still keeps the Gibbs weighting meaningful, and applies
# identically to every method -- including Adam and SGD, which therefore get 10 independent
# restarts rather than one, so the comparison stays cost-matched.
N_PARTICLES = 10

# Step budget. Each step is one full-batch forward+backward per particle. 10,000 -> 2,000 to pay
# for the deeper sweep: at depth 1 Softmin reached the target in ~728 steps and Adam in ~233
# (calibrate_transformer.py), and the feasibility probe had Adam well below chance by step 60 even
# at depth 50, so 2,000 leaves roughly 3x headroom over the largest requirement observed.
#
# CAVEAT worth checking against the results: if a deep model genuinely needs more than 2,000 steps,
# a 'did not reach target' outcome is a budget artifact rather than a real failure, and would
# corrupt exactly the reliability measurement this benchmark exists to make. The checkpoint series
# below is the diagnostic: if best-value is still falling steeply at ckpt1999 for the methods that
# missed the target, the budget -- not the method -- is what failed, and MAX_STEPS must be raised.
MAX_STEPS = 2_000

CHECKPOINTS = [50, 100, 250, 500, 1000, MAX_STEPS - 1]

# Gradient-norm clip, following run_lennard_jones.py's precedent and applied identically to every
# method: Adagrad-style accumulation plus injected noise on a transformer loss diverges without it.
#
# 1.0 -> 10.0 after recalibrating at the mid-sweep depth 10 (calibrate_transformer.py, stage B).
# The first pass, run on a 1-layer model, concluded the clip never binds; that does not hold once
# the model is deeper. At depth 10 the per-particle gradient norm is 0.9311 on average but ranges
# up to 1.0341, so a clip of 1.0 does bite, and it costs a great deal: at eta=0.1 and 400 steps,
# clip=1.0 reaches 1.1648 while clip=10.0 reaches 0.0754. 10.0 and no clip at all give identical
# losses to four decimals, i.e. 10.0 is loose enough never to bind at these depths while still
# bounding a pathological step.
MAX_GRAD_NORM = 10.0

# Learning rate, calibrated for this benchmark family (calibrate_transformer.py, stage B) and
# confirmed at the mid-sweep depth 10. The eta=1e-3 used on the analytic benchmarks does not
# transfer: at depth 10 the gradient norm is ~0.93 over d~5.1e5, so an eta=1e-3 step moves each
# coordinate by ~1.4e-6 against a weight scale of ~1e-1, and the loss does not leave its initial
# value within any feasible budget (measured: 3.9996 after 400 steps, i.e. above the 3.9703 chance
# level, at every clip value tested). eta=0.1 learns, and stage D confirms it is still stable at
# the deepest sweep point: 0.0104 after 400 steps at depth 50, neither diverging nor stalling.
# Adam is exempt from this problem by construction, since it rescales per coordinate -- which is
# exactly why it must not be the only method able to move.
LR = 0.1

# Hyperparameters are carried over unchanged from the analytic benchmarks wherever they transfer,
# so this benchmark stays genuinely out-of-sample for every method (see the paper's appendix on the
# pre-committed calibration protocol). The ONE exception is the noise level, which does not
# transfer to weight space at all and is calibrated per benchmark family below -- the same
# convention already applied to SA's T0 and PSO's coefficients. Adam and SGD use their
# literature-standard defaults, which is the corresponding convention for them: lr=1e-3 is Adam's
# canonical default, and SGD lr=1e-2 with momentum 0.9 is the standard baseline recipe.
# beta stays at 10.0, the value used on every other benchmark. Recalibrated at depth 10
# (calibrate_transformer.py, stage C1): every candidate now clears the 0.10 target, and the top two
# are indistinguishable -- beta=0.1 reaches 0.0157, beta=1 0.0042, beta=10 0.0028 and beta=100
# 0.0024 within 800 steps. The nominal optimum is therefore 100, by a margin of 4e-4 over 10 on a
# target of 0.10. 10 is kept for consistency with the rest of the paper rather than tuned to a
# per-benchmark optimum that the measurement cannot meaningfully separate from it.
SOFTMIN_BETA = 10.0

# Noise level, calibrated for this benchmark family (calibrate_transformer.py, stage C2) rather
# than carried over. sigma=1 is not transferable to weight space: it injects a per-coordinate
# perturbation of standard deviation sqrt(2*sigma*eta) that swamps weights of scale ~1e-1, and
# destroys training outright rather than exploring (measured: 4.15, worse than chance).
#
# The sweep's own optimum is sigma=0, at depth 10 as at depth 1 (loss 0.0024 against 0.0065 at
# 1e-5 -- both far below the 0.10 target, and both an order of magnitude better than 1e-4's
# 0.0485). We deliberately do NOT use 0: it would reduce SimulatedAnnealing and GaussianDiffusion
# to the SAME deterministic gradient descent, collapsing two of the five baselines into duplicates
# of each other and destroying the comparison this benchmark exists to make. 1e-5 is the smallest
# value that keeps every method's dynamics distinct, at a performance difference well inside the
# target margin. Note the cliff just above it: 1e-3 and 1e-2 are at or worse than chance (3.95,
# 4.08), so the usable range is narrow and 1e-5 sits safely inside it.
#
# Applies to Softmin, SA and Langevin alike, so no method is advantaged by the choice.
NOISE_SIGMA = 1e-5

SA_COOLING_RATE = 0.9999
# SA's diffusion is sqrt(2 * noise * eta * T_t), and `noise` above is already NOISE_SIGMA, so its
# starting noise budget is T0-times Softmin's fixed level -- the same ratio it has on the analytic
# benchmarks (T0=10 against sigma=1 there). T0 therefore needs no separate rescaling here.
SA_INITIAL_TEMPERATURE = 10.0
ADAM_LR = 1e-3
# SGD at its standard recipe. Note this is NOT a handicap relative to the calibrated eta=0.1 used
# by the swarm methods: heavy-ball momentum at 0.9 multiplies the asymptotic step by 1/(1-0.9)=10,
# so lr=1e-2 with momentum 0.9 has an effective step size of ~0.1, matching them.
SGD_LR = 1e-2
SGD_MOMENTUM = 0.9

# CBO and PSO (gradient-aware and annealed variants included) are dropped from this benchmark,
# not merely deprioritised: they reach the target in at most 3% of seeds at ANY difficulty on both
# analytic well benchmarks (Section 6.1), so sweeping them over five depths would spend a sixth of
# the budget re-measuring a flat zero. Softmin is compared against the methods that actually work
# on neural-network training (Adam, SGD) plus the two closest stochastic baselines from the well
# benchmarks (SA, Langevin).
ALGO_NAMES = ['softmin', 'adam', 'sgd', 'sa', 'langevin']

# Entry times in this CSV are in STEPS, not in the continuous-time units (steps x eta) used by
# every other driver -- see the note in worker_task for why eta cannot be a shared clock here.
HEADER = [
    'depth', 'seed',
    *ALGO_NAMES,
    *(f'{a}_status' for a in ALGO_NAMES),
    *(f'{a}_best' for a in ALGO_NAMES),
    *(f'{a}_gevals' for a in ALGO_NAMES),
    *(f'{a}_oevals' for a in ALGO_NAMES),
    *(f'{a}_secs' for a in ALGO_NAMES),
    *(f'{a}_ckpt{c}' for a in ALGO_NAMES for c in CHECKPOINTS),
    'peak_rss_mb',
]


def _clipped(f_grad_vec):
    """Per-particle gradient-norm clip, applied identically to every method (the same convention
    run_lennard_jones.py uses for the Lennard-Jones term)."""
    def wrapped(positions):
        g = f_grad_vec(positions)
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        scale = np.minimum(1.0, MAX_GRAD_NORM / np.maximum(norms, 1e-12))
        return g * scale
    return wrapped


def worker_task(args):
    depth, seed = args
    np.random.seed(seed * 1000)
    rng = np.random.RandomState(seed * 1000)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setting = transformer_setting(depth=depth, device=device)
    d = setting['d']
    f_vec = setting['f_vec']
    f_grad_vec = _clipped(setting['f_grad_vec'])
    bounds = setting['bounds']
    particle_init = setting['particle_init']
    target_value = setting['target_loss']

    common = dict(n=N_PARTICLES, d=d, f=f_vec, df=f_grad_vec,
                  particle_init=particle_init, bounds=bounds)

    algos = {
        'softmin': SoftminEnergyAdagrad(
            **common, initial_lr=LR, adaptive_lr=False, noise_intensity=NOISE_SIGMA,
            noise="normal", beta_decay=None, beta=SOFTMIN_BETA),
        'adam': Adam(**common, learning_rate=ADAM_LR),
        'sgd': SGD(**common, learning_rate=SGD_LR, momentum=SGD_MOMENTUM),
        'sa': SimulatedAnnealing(
            **common, learning_rate=LR, cooling_rate=SA_COOLING_RATE, noise=NOISE_SIGMA,
            cooling_schedule="geometric", initial_temperature=SA_INITIAL_TEMPERATURE),
        'langevin': GaussianDiffusion(**common, learning_rate=LR, noise=NOISE_SIGMA),
    }

    # Shared initial swarm: n independent PyTorch inits, drawn once and given to every method, so
    # no method is advantaged by a luckier starting point.
    x0 = setting['particle_init_fn'](N_PARTICLES, rng)
    for sim in algos.values():
        sim.reset(particle_init=x0)

    times, statuses, bests, ckpts = {}, {}, {}, {}
    for name, sim in algos.items():
        # abort_on_collapse / abort_on_stagnation are both disabled deliberately. On a learning
        # objective a collapsed ensemble or a loss plateau routinely means success or a genuine
        # slow phase, not failure -- first_value_time's own docstring flags this case -- and the
        # defaults (True) would silently truncate runs and corrupt the reliability measurement
        # this benchmark exists to make.
        # NOTE the units. first_value_time returns (steps x that sim's own learning_rate), which
        # is a common clock ONLY when every method shares one learning rate -- true on the
        # analytic benchmarks (eta=1e-3 throughout), FALSE here, where the swarm methods use the
        # calibrated eta=0.1 and Adam keeps its own 1e-3. Left raw, Adam's 233 steps would be
        # logged as 0.233 against Softmin's 728 steps logged as 72.8, a 300x apparent gap standing
        # in for a real 3x one. We therefore divide the learning rate back out and record entry
        # time in STEPS, which is comparable across methods because every method performs exactly
        # one objective and one gradient evaluation per particle per step.
        raw = sim.first_value_time(
            target_value=target_value, max_steps=MAX_STEPS, checkpoints=CHECKPOINTS,
            abort_on_collapse=False, abort_on_stagnation=False,
        )
        times[name] = raw / sim.learning_rate
        statuses[name] = sim._last_stop_reason
        bests[name] = float(sim.best_global_value)
        ckpts[name] = sim.value_checkpoints
        logger.info(f"depth={depth} seed={seed} {name}: status={statuses[name]} "
                    f"best={bests[name]:.4f} t={times[name]}")

    return (depth, seed,
            *(times[a] for a in ALGO_NAMES),
            *(statuses[a] for a in ALGO_NAMES),
            *(bests[a] for a in ALGO_NAMES),
            *(algos[a].n_gradient_evals for a in ALGO_NAMES),
            *(algos[a].n_objective_evals for a in ALGO_NAMES),
            *(algos[a].wall_clock_seconds for a in ALGO_NAMES),
            *(ckpts[a].get(c, float('nan')) for a in ALGO_NAMES for c in CHECKPOINTS),
            peak_rss_mb())


def task_key(t):
    return (int(t[0]), int(t[1]))


def row_key(row):
    return (int(row[0]), int(row[1]))


def run_serial(results_csv, header, all_tasks):
    """Serial, crash-resumable task loop. Mirrors run_resumable_pool's contract (skip completed
    tasks, append one fsync'd row per finished task) without the multiprocessing.Pool, which
    cannot be combined with CUDA."""
    os.makedirs(os.path.dirname(results_csv) or '.', exist_ok=True)
    is_new = not os.path.exists(results_csv) or os.path.getsize(results_csv) == 0

    completed = set() if is_new else get_completed_tasks(results_csv, row_key)
    pending = [t for t in all_tasks if task_key(t) not in completed]
    logger.info(f"Total tasks: {len(all_tasks)} | Completed: {len(completed)} "
                f"| Pending: {len(pending)}")

    with open(results_csv, 'a', newline='') as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(header)
            fh.flush()
            os.fsync(fh.fileno())
        for i, task in enumerate(pending, 1):
            t0 = time.perf_counter()
            writer.writerow(worker_task(task))
            fh.flush()
            os.fsync(fh.fileno())
            logger.info(f"[{i}/{len(pending)}] {task} done in {time.perf_counter()-t0:.0f}s")

    logger.info(f"All tasks for {results_csv} finished.")


def parse_tasks(specs):
    """`50:3` -> [(50, 3)]; a bare `10` -> every seed of depth 10. Order is preserved and
    duplicates dropped, so the shard runs its tasks in the order they were listed."""
    tasks = []
    for spec in specs:
        if ':' in spec:
            depth, seed = (int(x) for x in spec.split(':', 1))
            tasks.append((depth, seed))
        else:
            tasks.extend((int(spec), seed) for seed in range(N_SEEDS))
    bad = [t for t in tasks if t[0] not in DEPTHS or not 0 <= t[1] < N_SEEDS]
    if bad:
        raise ValueError(f"not part of the configured sweep (DEPTHS={DEPTHS}, "
                         f"N_SEEDS={N_SEEDS}): {bad}")
    return list(dict.fromkeys(tasks))


def merge_shards(out_csv=RESULTS_CSV, pattern=SHARD_CSV.format(tag='*')):
    """Combine shard CSVs into one, keeping the first row seen per (depth, seed), and refusing to
    merge a shard whose header differs from HEADER (i.e. one run under a different configuration)."""
    rows = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, newline='') as fh:
            reader = csv.reader(fh)
            if next(reader, None) != HEADER:
                raise ValueError(f"{path}: header does not match this driver's HEADER; "
                                 f"refusing to merge a shard from a different configuration")
            for row in reader:
                if row:
                    rows.setdefault(row_key(row), row)
    expected = {(d, s) for d in DEPTHS for s in range(N_SEEDS)}
    missing = sorted(expected - set(rows))
    with open(out_csv, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for key in sorted(rows):
            writer.writerow(rows[key])
    logger.info(f"Merged {len(rows)} rows into {out_csv}; {len(missing)} of {len(expected)} "
                f"tasks missing" + (f": {missing}" if missing else ""))
    return missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='DEPTH:SEED or DEPTH (all seeds); default is the whole sweep')
    parser.add_argument('--tag', default=None,
                        help='write to a shard CSV with this tag instead of RESULTS_CSV')
    parser.add_argument('--merge', action='store_true',
                        help='combine all shard CSVs into RESULTS_CSV and exit')
    args = parser.parse_args(argv)

    os.makedirs('results', exist_ok=True)
    if args.merge:
        merge_shards()
        return

    if args.tasks is None:
        all_tasks = [(depth, seed) for depth in DEPTHS for seed in range(N_SEEDS)]
    else:
        try:
            all_tasks = parse_tasks(args.tasks)
        except ValueError as exc:
            parser.error(str(exc))
    results_csv = SHARD_CSV.format(tag=args.tag) if args.tag else RESULTS_CSV
    run_serial(results_csv, HEADER, all_tasks)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    # One thread per process: the GPU (or the vmap'd batch) is the parallelism here, and letting
    # torch spawn a thread per core on a 256-core node only causes contention.
    torch.set_num_threads(1)
    main()
