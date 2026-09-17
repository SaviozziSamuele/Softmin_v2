"""Extra Softmin seeds on the transformer benchmark, to characterise a single outlier.

The five-seed sweep (run_transformer.py) has one anomalous cell: at depth 50, seed 2, Softmin
reaches the target at step 743 against 333-363 for the other four seeds. Its checkpoint series says
this is a STALL, not a slow start -- at step 250 that seed held the LOWEST loss of the five (2.824
against 3.07-3.46) and then plateaued, clearing the target only at 743. No other method is unusual
at that seed (Adam 86, SGD 316, SA 1179, Langevin 469, all typical), so it is specific to Softmin's
trajectory from that initialisation rather than a bad draw of the shared starting swarm.

One event in five seeds cannot distinguish a rare tail from a fluke, and a mean over five seeds is
not robust to it (it lifts depth 50 from ~343 to 423 and is the only reason Softmin's entry time
looks non-monotone in depth). This script adds seeds cheaply by running ONLY Softmin.

RNG FIDELITY -- why all five algorithms are still constructed. Simulation.__init__ draws its
initial particles from the GLOBAL NumPy stream, so the number of simulations built before a run
determines the noise stream that run sees (the same effect documented for the particle sweep, where
the published driver's 8 constructions made results non-bit-identical to a 2-algorithm rerun). In
run_transformer.worker_task all five are constructed up front and Softmin runs FIRST, so building
all five here and running only Softmin reproduces the original stream exactly, while the four
unused objects cost nothing but their construction. Seeds 0-4 are re-run for precisely this reason:
they must reproduce the sweep's numbers, and if they do not, the comparison with the new seeds is
invalid and the discrepancy must be resolved before anything is concluded.

Run:  cd src && python extra_seeds_transformer.py --tasks 50 --tag a
      python extra_seeds_transformer.py --merge
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
from run_transformer import (
    N_PARTICLES, MAX_STEPS, CHECKPOINTS, LR, SOFTMIN_BETA, NOISE_SIGMA,
    ADAM_LR, SGD_LR, SGD_MOMENTUM, SA_COOLING_RATE, SA_INITIAL_TEMPERATURE,
    _clipped,
)
from simulations import (
    SoftminEnergyAdagrad,
    SimulatedAnnealing,
    GaussianDiffusion,
    Adam,
    SGD,
)

logger = logging.getLogger(__name__)

RESULTS_CSV = 'results/master_transformer_softmin_seeds.csv'
SHARD_CSV = 'results/master_transformer_softmin_seeds_shard_{tag}.csv'

# Depth 50 carries the outlier. Depth 20 is included as a control at 3.5x lower cost: if stalls are
# a property of Softmin on this objective rather than of depth 50 specifically, they should appear
# there too, and 25 seeds at depth 20 cost about a quarter of 25 seeds at depth 50.
DEPTHS = [20, 50]

# 0-4 reproduce the sweep (see the RNG note above); 5-24 are new.
N_SEEDS = 25

HEADER = [
    'depth', 'seed', 'softmin', 'softmin_status', 'softmin_best',
    'softmin_gevals', 'softmin_oevals', 'softmin_secs',
    *(f'softmin_ckpt{c}' for c in CHECKPOINTS),
    'peak_rss_mb',
]


def worker_task(args):
    """A cut-down run_transformer.worker_task: identical construction, Softmin only."""
    depth, seed = args
    np.random.seed(seed * 1000)
    rng = np.random.RandomState(seed * 1000)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setting = transformer_setting(depth=depth, device=device)
    d = setting['d']
    f_vec = setting['f_vec']
    f_grad_vec = _clipped(setting['f_grad_vec'])

    common = dict(n=N_PARTICLES, d=d, f=f_vec, df=f_grad_vec,
                  particle_init=setting['particle_init'], bounds=setting['bounds'])

    # Constructed in run_transformer.worker_task's order and kept, not discarded: see the RNG
    # fidelity note in this module's docstring. Only `softmin` is ever stepped.
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

    x0 = setting['particle_init_fn'](N_PARTICLES, rng)
    for sim in algos.values():
        sim.reset(particle_init=x0)

    sim = algos['softmin']
    raw = sim.first_value_time(
        target_value=setting['target_loss'], max_steps=MAX_STEPS, checkpoints=CHECKPOINTS,
        abort_on_collapse=False, abort_on_stagnation=False,
    )
    # Entry time in STEPS, matching run_transformer.py; see its note on first_value_time's units.
    t = raw / sim.learning_rate
    logger.info(f"depth={depth} seed={seed} softmin: status={sim._last_stop_reason} "
                f"best={sim.best_global_value:.4f} t={t}")

    return (depth, seed, t, sim._last_stop_reason, float(sim.best_global_value),
            sim.n_gradient_evals, sim.n_objective_evals, sim.wall_clock_seconds,
            *(sim.value_checkpoints.get(c, float('nan')) for c in CHECKPOINTS),
            peak_rss_mb())


def task_key(t):
    return (int(t[0]), int(t[1]))


def row_key(row):
    return (int(row[0]), int(row[1]))


def parse_tasks(specs):
    """`50:7` -> [(50, 7)]; a bare `50` -> every seed at depth 50."""
    tasks = []
    for spec in specs:
        if ':' in spec:
            depth, seed = (int(x) for x in spec.split(':', 1))
            tasks.append((depth, seed))
        else:
            tasks.extend((int(spec), seed) for seed in range(N_SEEDS))
    bad = [t for t in tasks if t[0] not in DEPTHS or not 0 <= t[1] < N_SEEDS]
    if bad:
        raise ValueError(f"not part of this run (DEPTHS={DEPTHS}, N_SEEDS={N_SEEDS}): {bad}")
    return list(dict.fromkeys(tasks))


def merge_shards(out_csv=RESULTS_CSV, pattern=SHARD_CSV.format(tag='*')):
    rows = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, newline='') as fh:
            reader = csv.reader(fh)
            if next(reader, None) != HEADER:
                raise ValueError(f"{path}: header does not match HEADER; refusing to merge a "
                                 f"shard from a different configuration")
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


def run_serial(results_csv, all_tasks):
    os.makedirs(os.path.dirname(results_csv) or '.', exist_ok=True)
    is_new = not os.path.exists(results_csv) or os.path.getsize(results_csv) == 0
    completed = set() if is_new else get_completed_tasks(results_csv, row_key)
    pending = [t for t in all_tasks if task_key(t) not in completed]
    logger.info(f"Total tasks: {len(all_tasks)} | Completed: {len(completed)} "
                f"| Pending: {len(pending)}")

    with open(results_csv, 'a', newline='') as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(HEADER)
            fh.flush()
            os.fsync(fh.fileno())
        for i, task in enumerate(pending, 1):
            t0 = time.perf_counter()
            writer.writerow(worker_task(task))
            fh.flush()
            os.fsync(fh.fileno())
            logger.info(f"[{i}/{len(pending)}] {task} done in {time.perf_counter()-t0:.0f}s")
    logger.info(f"All tasks for {results_csv} finished.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tasks', nargs='+', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args(argv)

    os.makedirs('results', exist_ok=True)
    if args.merge:
        merge_shards()
        return

    if args.tasks is None:
        all_tasks = [(d, s) for d in DEPTHS for s in range(N_SEEDS)]
    else:
        try:
            all_tasks = parse_tasks(args.tasks)
        except ValueError as exc:
            parser.error(str(exc))
    run_serial(SHARD_CSV.format(tag=args.tag) if args.tag else RESULTS_CSV, all_tasks)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    torch.set_num_threads(1)
    main()
