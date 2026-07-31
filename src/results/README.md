Output directory for the `run_*.py` drivers (post-review-revision experiments).

The `.pkl`/`.csv` data that produced the original NeurIPS submission's figures has been moved to
`../results_NIPS/` for reference — the code no longer reads from or writes to that folder. This
folder starts empty; every `run_*.py` driver creates the files it needs here automatically
(`os.makedirs('results', exist_ok=True)`).

Until the drivers are rerun on the cluster, the notebook cells that read from this folder (the
double-well/quadruple-well/Ackley comparison figures, and the Theorem 3.1 exit-time plot) will
fail with `FileNotFoundError` rather than render — this is expected, not a bug.
