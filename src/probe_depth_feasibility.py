"""Feasibility probe for the aggressive transformer depth sweep.

Probes exactly run_transformer.DEPTHS, so it always reports the configuration that will run.

Three things scale with depth and each could independently make the sweep infeasible or
uninformative, so measure all three rather than assume:

  1. Optimization dimension d (flattened parameter count), which drives the NumPy-side memory in
     simulations.Simulation -- particles, best_positions, gradients, and per-method optimizer
     state are all (n, d) float64 arrays.
  2. Wall-clock per step, which is linear in depth and multiplies through the whole sweep.
  3. Whether the model is TRAINABLE at that depth at all. A 50-layer, d_model=64 pre-LN
     transformer is deep and narrow; if Adam cannot drive the loss below chance there, no
     comparison at that depth means anything, and the point should be dropped rather than
     reported as everyone failing.

Run:  cd src && python probe_depth_feasibility.py
Writes results/depth_feasibility.txt progressively.
"""
import logging
import os
import time

import numpy as np
import torch

logging.disable(logging.INFO)

from nn_objective import transformer_setting
from simulations import Adam
from run_transformer import (_clipped, N_PARTICLES, MAX_STEPS, N_SEEDS,
                             ALGO_NAMES, DEPTHS as SWEEP_DEPTHS)

OUT_PATH = 'results/depth_feasibility.txt'

# Probe exactly the depths the sweep will run, so the projection below is the real budget
# rather than a proxy for it.
DEPTHS = list(SWEEP_DEPTHS)
PROBE_N = 2          # tiny ensemble: this measures scaling, not performance
ADAM_STEPS = 60      # enough to see whether the loss moves off chance at all
ADAM_LR = 1e-3


def main():
    os.makedirs('results', exist_ok=True)
    out = open(OUT_PATH, 'w', buffering=1)

    def emit(line=''):
        out.write(line + '\n')
        print(line, flush=True)

    emit(f"depth feasibility probe | probe n={PROBE_N}, planned n={N_PARTICLES}, "
         f"planned MAX_STEPS={MAX_STEPS}")
    emit()
    emit(f"{'depth':>6} {'d (params)':>12} {'s/step':>9} {'grad norm':>10} "
         f"{'np mem':>9} {'act mem':>9} {'Adam loss':>10}   (memory at n={N_PARTICLES})")

    rows = []
    for depth in DEPTHS:
        setting = transformer_setting(depth=depth)
        d = setting['d']
        rng = np.random.RandomState(0)
        x0 = setting['particle_init_fn'](PROBE_N, rng)
        fg = _clipped(setting['f_grad_vec'])

        # Gradient scale, and time per gradient evaluation.
        t0 = time.perf_counter()
        g = fg(x0)
        t_grad = time.perf_counter() - t0
        gnorm = float(np.linalg.norm(g, axis=1).mean())

        # A Softmin/Adam step costs one f and one grad-f per particle, so ~2 passes; the gradient
        # pass dominates (forward + backward vs forward only), so ~1.5x the grad time is a fair
        # per-step estimate at the same n.
        s_per_step_probe = 1.5 * t_grad
        # Scale to the planned ensemble: cost is linear in n (the particles are vmap'd, but the
        # work is proportional).
        s_per_step = s_per_step_probe * N_PARTICLES / PROBE_N

        # NumPy-side: Simulation keeps ~6 (n, d) float64 arrays per method (particles,
        # best_positions, gradient, plus per-method state such as Adam's m and v).
        np_mem_gb = 6 * N_PARTICLES * d * 8 / 1e9
        # Torch-side activations for backward: ~6 saved tensors per layer, each
        # (n_train, seq, d_model) float32, per particle.
        act_gb = (6 * depth * setting['n_train'] * 3 * 64 * 4 * N_PARTICLES) / 1e9

        # Trainability: can Adam move the loss off chance at all?
        np.random.seed(0)
        ad = Adam(n=PROBE_N, d=d, f=setting['f_vec'], df=fg,
                  particle_init=setting['particle_init'], bounds=setting['bounds'],
                  learning_rate=ADAM_LR)
        ad.reset(particle_init=x0)
        ad.run(ADAM_STEPS)
        adam_loss = float(ad.best_global_value)

        emit(f"{depth:>6} {d:>12,} {s_per_step:>9.2f} {gnorm:>10.4f} "
             f"{np_mem_gb:>8.1f}G {act_gb:>8.1f}G {adam_loss:>10.4f}")
        rows.append((depth, d, s_per_step, adam_loss, setting['chance_loss']))

    chance = rows[0][4]
    emit()
    emit(f"chance loss = {chance:.4f}. An Adam loss at or above it after {ADAM_STEPS} steps means "
         f"the model has not started learning;")
    emit("that is only a 60-step probe, so it is a smoke signal, not proof of untrainability.")
    emit()

    # Sweep cost, measured on THIS machine's CPU, plus a GPU-scaled estimate.
    n_algos = len(ALGO_NAMES)
    emit(f"=== projected sweep cost ({n_algos} algorithms, {N_SEEDS} seeds, "
         f"MAX_STEPS={MAX_STEPS:,}) ===")
    total_cpu = 0.0
    for depth, d, s_per_step, _, _ in rows:
        per_run = s_per_step * MAX_STEPS
        cell = per_run * n_algos * N_SEEDS
        total_cpu += cell
        emit(f"  depth {depth:>2}: {s_per_step:.2f}s/step -> {per_run/3600:8.2f} h per run, "
             f"{cell/3600:9.1f} h for {n_algos} algos x {N_SEEDS} seeds")
    emit(f"  TOTAL (this CPU): {total_cpu/3600:.0f} h = {total_cpu/86400:.1f} days")
    for speedup in (20, 50):
        emit(f"  at {speedup}x GPU speedup: {total_cpu/3600/speedup:.1f} h "
             f"({total_cpu/86400/speedup:.2f} days)")
    emit()
    emit(f"Total depth swept: {sum(DEPTHS)} layer-units across {len(DEPTHS)} points.")
    emit("If the measured GPU per-step cost makes the total above exceed the job's --time limit,")
    emit("cut MAX_STEPS first (it has the most headroom), then N_PARTICLES; cut seeds last, since")
    emit("reliability across seeds is the headline metric this benchmark exists to measure.")
    out.close()


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
