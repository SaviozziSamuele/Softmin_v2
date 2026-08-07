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

try:
    import resource  # POSIX only (the cluster) -- not available on Windows (local dev)
except ImportError:
    resource = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def peak_rss_mb() -> float:
    """Peak resident-set size of the calling process so far, in MB -- the per-task memory-cost
    signal the reviewer-requested cost analysis needs alongside gradient/objective-evaluation
    counts. `None` on platforms without `resource` (Windows local dev); real drivers run on the
    Linux cluster, where this is always available. `ru_maxrss` is already in KB on Linux (KiB in
    practice), unlike macOS's bytes -- fine here since production only runs on Linux."""
    if resource is None:
        return float('nan')
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def get_completed_tasks(results_csv: str, row_key_fn: Callable[[List[str]], Tuple]) -> set:
    """Reads `results_csv` (if it exists) and returns the set of task keys already logged,
    as produced by `row_key_fn(row)` for each data row (header is skipped).

    A row that fails to parse (e.g. truncated by a crash between `write` and `fsync`) is skipped
    with a warning rather than raising -- one torn line at the end of the file shouldn't make
    every future resume attempt crash on startup.
    """
    completed = set()
    if os.path.exists(results_csv):
        with open(results_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if not row:
                    continue
                try:
                    completed.add(row_key_fn(row))
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping malformed row in {results_csv}: {row!r} ({e})")
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

    # Exclusive lock against a second concurrent run against the same results_csv (e.g. an
    # accidental double `sbatch` of the same driver): both processes would otherwise compute their
    # own pending-task list from the same starting CSV, run every task twice, and interleave rows.
    # Plain `open(..., 'x')`-style exclusive creation is atomic and portable (works the same via
    # os.O_CREAT|O_EXCL on POSIX and Windows), unlike fcntl.flock which is POSIX-only and would
    # break local (Windows) development. A crashed run leaves this file behind; if you're sure no
    # other process is writing this CSV, delete it manually before resubmitting.
    lock_path = results_csv + '.lock'
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"{lock_path} already exists -- either another process is currently writing "
            f"{results_csv}, or a previous run crashed without cleaning up. If you're sure "
            f"nothing else is running against this file, delete the lock and retry."
        )
    try:
        os.write(lock_fd, str(os.getpid()).encode())
        os.close(lock_fd)

        all_tasks = list(all_tasks)
        completed = get_completed_tasks(results_csv, row_key_fn)
        pending_tasks = [t for t in all_tasks if task_key_fn(t) not in completed]

        logger.info(f"Total tasks: {len(all_tasks)} | Completed: {len(completed)} | Pending: {len(pending_tasks)}")

        # Size, not just existence: a 0-byte CSV (e.g. created by makedirs/open but killed before
        # any write) has no header either, and `os.path.exists` alone would skip writing one here,
        # silently losing the first real data row to get_completed_tasks's own header-skip on the
        # next resume.
        file_is_nonempty = os.path.exists(results_csv) and os.path.getsize(results_csv) > 0
        with open(results_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_is_nonempty:
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
    finally:
        os.remove(lock_path)


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
        if next(reader, None) is None:  # Empty file: no header, nothing to reconstruct
            logger.warning(f"{results_csv} is empty -- skipping .pkl reconstruction.")
            return
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
