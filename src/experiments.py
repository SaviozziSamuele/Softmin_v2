"""Shared resumable-experiment harness used by the run_*.py drivers.

Each driver (run_experiments_resumable.py, run_ackley_dimensions.py,
run_theory_verification.py) previously reimplemented the same three pieces: a
completed-task lookup against a results CSV, a resumable multiprocessing pool that
flushes/fsyncs each row as it lands, and a CSV -> per-condition .pkl reconstruction pass.
This module factors that pattern out; each driver only supplies its task list, its own
worker function (which algorithms to run and how), and how to key/name things.
"""
import csv
import logging
import multiprocessing as mp
import os
import pickle as pkl
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_completed_tasks(results_csv: str, row_key_fn: Callable[[List[str]], Tuple]) -> set:
    """Reads `results_csv` (if it exists) and returns the set of task keys already logged,
    as produced by `row_key_fn(row)` for each data row (header is skipped)."""
    completed = set()
    if os.path.exists(results_csv):
        with open(results_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row:
                    completed.add(row_key_fn(row))
    return completed


def run_resumable_pool(
    results_csv: str,
    header: Sequence[str],
    all_tasks: Iterable[Tuple],
    task_key_fn: Callable[[Tuple], Tuple],
    row_key_fn: Callable[[List[str]], Tuple],
    worker_fn: Callable[[Tuple], Tuple],
    n_cpu: int = None,
) -> None:
    """Runs `worker_fn` over the tasks in `all_tasks` that aren't already present in
    `results_csv` (per `task_key_fn`/`row_key_fn`), appending each result row to the CSV as
    it completes so the run can be resumed after a crash/preemption.
    """
    os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

    all_tasks = list(all_tasks)
    completed = get_completed_tasks(results_csv, row_key_fn)
    pending_tasks = [t for t in all_tasks if task_key_fn(t) not in completed]

    logger.info(f"Total tasks: {len(all_tasks)} | Completed: {len(completed)} | Pending: {len(pending_tasks)}")

    file_exists = os.path.exists(results_csv)
    with open(results_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)

        if pending_tasks:
            n_cpu = n_cpu or int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
            logger.info(f"Launching pool with {n_cpu} workers...")

            with mp.Pool(n_cpu) as pool:
                for result in pool.imap_unordered(worker_fn, pending_tasks):
                    writer.writerow(result)
                    csvfile.flush()
                    os.fsync(csvfile.fileno())
                    logger.info(f"Completed -> {result}")

    logger.info(f"All tasks for {results_csv} finished.")


def reconstruct_pkl_files_from_csv(
    results_csv: str,
    key_indices: Sequence[int],
    value_start: int,
    algo_names: Sequence[str],
    filename_fn: Callable[[Tuple[str, ...], str], str],
    value_transform: Callable[[str], Any] = float,
) -> None:
    """Rebuilds the per-condition `.pkl` files the notebook reads directly from the flat,
    fault-tolerant CSV log.

    Groups rows by the columns at `key_indices` (raw strings; `filename_fn` is responsible
    for any type cleanup needed to build a filename) and treats the columns from
    `value_start` onward as one series per name in `algo_names`, in order. Any column between
    the last key column and `value_start` (e.g. a per-run seed) is aggregated over and
    dropped, matching how the original per-experiment scripts grouped their results.
    """
    logger.info(f"Reconstructing .pkl files from {results_csv}...")
    data = {}
    with open(results_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            key = tuple(row[i] for i in key_indices)
            values = row[value_start:]
            bucket = data.setdefault(key, {algo: [] for algo in algo_names})
            for algo, v in zip(algo_names, values):
                bucket[algo].append(value_transform(v))

    for key, algos in data.items():
        for algo, values in algos.items():
            path = filename_fn(key, algo)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, 'wb') as f:
                pkl.dump(np.array(values), f)

    logger.info("Pickle reconstruction complete.")
