"""Confirmation run for the A=30 entry-time anomaly in the ensemble-size sweep.

What is being checked. In results/master_particle_sweep.csv, Softmin's entry time at A=30 is
non-monotone in n in a way seed noise does not obviously explain: the whole n=50 distribution is
about 8x faster than its neighbours (median 1.18, slowest of 100 seeds 5.0), while n=100 is about
8x slower (median 21.3, with 24% of solved seeds above t=50). A=10 is smooth and monotone over the
same range, and reliability is a flat 100% everywhere, so only the cost panel is affected.

It is NOT a configuration artifact: run_particle_sweep.py holds beta, the learning rate, the noise
intensity and SA's cooling schedule fixed across n, and the only thing that varies with n is the
ensemble itself. Either the dip is a real property of the ensemble size, in which case the paper
should say so, or it is an unlucky draw over 100 seeds, in which case the paper must not describe
a trend that does not exist.

How. Re-run the three affected points with a DISJOINT set of seeds -- 100-199 against the
original's 0-99 -- and nothing else changed. worker_task is imported unmodified from
run_particle_sweep, so this is the same computation on fresh randomness (the driver keys its RNG
off seed * 1000, so the streams cannot overlap).

Results go to their own CSV. The original sweep's CSV and the .pkl files reconstructed from it are
the paper's data and are left untouched; mixing 200 seeds into them would silently change every
published figure.

Run:  cd src && python confirm_particle_anomaly.py   (or: sbatch jobs/confirm_particle_anomaly.sh)
      then: python confirm_particle_anomaly.py --report
"""
import os
import sys

from experiments import run_resumable_pool
from run_particle_sweep import HEADER, row_key, task_key, worker_task

RESULTS_CSV = 'results/master_particle_sweep_confirm.csv'

A_VALUE = 30
N_VALUES = [25, 50, 100]
# Disjoint from the original sweep's range(100), so no seed is reused.
SEEDS = range(100, 200)


def report():
    """Compare the original seeds against the fresh ones at each n."""
    import csv
    from collections import defaultdict

    import numpy as np
    from scipy.stats import mannwhitneyu

    def load(path, seeds_wanted):
        out = defaultdict(list)
        if not os.path.exists(path):
            return out
        for r in csv.DictReader(open(path)):
            if int(r['parameter']) != A_VALUE or int(r['seed']) not in seeds_wanted:
                continue
            if r['softmin_status'] == 'target_reached':
                out[int(r['n'])].append(float(r['softmin']))
        return out

    original = load('results/master_particle_sweep.csv', set(range(100)))
    fresh = load(RESULTS_CSV, set(SEEDS))
    if not fresh:
        print(f"{RESULTS_CSV} not found or empty -- run the sweep first.")
        return

    print(f"Softmin entry time at A={A_VALUE}: original seeds 0-99 vs fresh seeds "
          f"{SEEDS.start}-{SEEDS.stop - 1}\n")
    print(f"{'n':>5} {'orig med':>9} {'new med':>9} {'orig IQR':>19} {'new IQR':>19} "
          f"{'ratio':>7} {'MWU p':>9}")
    for n in N_VALUES:
        a, b = np.array(original.get(n, [])), np.array(fresh.get(n, []))
        if not len(a) or not len(b):
            print(f"{n:>5}   missing data (orig {len(a)}, fresh {len(b)})")
            continue
        qa = np.percentile(a, [25, 75])
        qb = np.percentile(b, [25, 75])
        p = mannwhitneyu(a, b, alternative='two-sided').pvalue
        print(f"{n:>5} {np.median(a):>9.2f} {np.median(b):>9.2f} "
              f"[{qa[0]:>7.2f},{qa[1]:>7.2f}] [{qb[0]:>7.2f},{qb[1]:>7.2f}] "
              f"{np.median(b) / np.median(a):>7.2f} {p:>9.2g}")

    print("\nThe dip is REPRODUCED if the fresh medians keep the same shape across n -- n=50 well "
          "below n=25 and n=100 -- regardless of the per-n p-values, which only test whether the "
          "two seed batches agree at a given n.")
    med = {n: np.median(fresh[n]) for n in N_VALUES if len(fresh.get(n, []))}
    if len(med) == len(N_VALUES):
        dipped = med[50] < min(med[25], med[100])
        print(f"\nfresh medians: " + ", ".join(f"n={n}: {med[n]:.2f}" for n in N_VALUES))
        print("VERDICT: dip reproduces -- treat as a real property of n."
              if dipped else
              "VERDICT: dip does NOT reproduce -- the original was a seed artifact.")


if __name__ == '__main__':
    if '--report' in sys.argv:
        report()
    else:
        os.makedirs('results', exist_ok=True)
        all_tasks = [('quadruple_well', A_VALUE, n, s) for n in N_VALUES for s in SEEDS]
        run_resumable_pool(
            results_csv=RESULTS_CSV,
            header=HEADER,
            all_tasks=all_tasks,
            task_key_fn=task_key,
            row_key_fn=row_key,
            worker_fn=worker_task,
        )
        # No reconstruct_pkl_files_from_csv here on purpose: the .pkl files are keyed by
        # (experiment, A, n) and would collide with -- and overwrite -- the originals the paper's
        # figure is generated from. This CSV is a diagnostic, not a source of figures.
