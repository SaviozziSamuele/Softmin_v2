"""Diagnostic-guided hyperparameter calibration for the transformer benchmark.

Same two-stage shape as hyperparameter_diagnostics.py uses for the analytic benchmarks: screen for
a configuration in which the dynamics are actually viable, then validate the free parameter by
direct measurement of the reported metric. Run at the MID-SWEEP depth, matching the convention
every other benchmark family in this paper uses (calibrate at a representative middle setting, not
the easiest or the hardest one). With
DEPTHS = [2, 5, 10, 20, 50] that midpoint is depth 10, not depth 1 as in the first pass -- a value
calibrated on a 1-layer model has no claim to transfer to a 50-layer one.

Stage D then re-checks the selected configuration at the DEEPEST point. That check is the one that
matters most here: eta=0.1 was selected on a shallow model, and a step size that is merely large
at depth 10 can be outright unstable at depth 50 for the non-adaptive methods.

Four stages, in dependency order -- each stage's result fixes the configuration the next one is
measured at:

  A. Scale diagnostic. Measures the raw gradient norm at initialization, which sets whether the
     carried-over clip (MAX_GRAD_NORM=1.0) and learning rate (eta=1e-3) can move the iterate at
     all -- d is ~5.1e5 at depth 10, an order of magnitude above the depth-1 model the first pass
     used.
  B. (eta, clip) grid. The carried-over pair is not viable: at eta=1e-3 the per-coordinate step is
     orders of magnitude below the weight scale, so the loss barely leaves its initial value
     within any feasible budget. Adam is unaffected because it rescales per coordinate, which is
     precisely why it must not be the only method that can move. The clip turns out not to bind at
     all -- the measured gradient norm is below 1.0 -- so the learning rate is the real constraint.
  C. beta sweep at the (eta, clip) chosen in B, plus a sigma re-check, since the best noise level
     can shift once the drift is no longer negligible.
  D. Stability re-check of the whole selected configuration at depth 50, the deepest sweep point.

Writes results progressively to results/transformer_calibration.txt so a long run can be inspected
while in flight.

Run:  cd src && python calibrate_transformer.py     (GPU session -- see runtime note below)
"""
import logging
import os
import time

import numpy as np

logging.disable(logging.INFO)

from nn_objective import transformer_setting
from simulations import SoftminEnergyAdagrad, Adam

OUT_PATH = 'results/transformer_calibration.txt'

# Mid-sweep depth of run_transformer.DEPTHS = [2, 5, 10, 20, 50].
DEPTH = 10
# Deepest point, for the stage-D stability re-check.
DEEP_DEPTH = 50
N_PARTICLES = 4
STEPS_GRID = 400      # stage B: enough to separate "moves" from "does not move"
STEPS_BETA = 800      # stage C: enough to rank candidates
STEPS_DEEP = 400      # stage D: enough to catch divergence or total immobility

# NOTE ON RUNTIME: at depth 10 a gradient evaluation costs roughly ten times what it did at
# depth 1, and stage D at depth 50 fifty times. This script is now a GPU job -- run it via a
# cluster session on --partition=gpu --gres=gpu:1, not on a laptop CPU.

ETA_CANDIDATES = [1e-3, 1e-2, 1e-1]
CLIP_CANDIDATES = [1.0, 10.0, None]
BETA_CANDIDATES = [0.1, 1.0, 10.0, 100.0]
SIGMA_CANDIDATES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]


def clipped(f_grad_vec, max_norm):
    """Per-particle gradient-norm clip; `None` disables it."""
    if max_norm is None:
        return f_grad_vec

    def wrapped(positions):
        g = f_grad_vec(positions)
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        return g * np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))
    return wrapped


def run_softmin(setting, x0, steps, eta, clip, beta, sigma):
    np.random.seed(0)
    sim = SoftminEnergyAdagrad(
        n=N_PARTICLES, d=setting['d'], f=setting['f_vec'],
        df=clipped(setting['f_grad_vec'], clip),
        particle_init=setting['particle_init'], bounds=setting['bounds'],
        initial_lr=eta, adaptive_lr=False, noise_intensity=sigma,
        noise="normal", beta_decay=None, beta=beta,
    )
    sim.reset(particle_init=x0)
    sim.run(steps)
    return float(sim.best_global_value)


def main():
    os.makedirs('results', exist_ok=True)
    out = open(OUT_PATH, 'w', buffering=1)

    def emit(line=''):
        out.write(line + '\n')
        print(line, flush=True)

    setting = transformer_setting(depth=DEPTH)
    rng = np.random.RandomState(0)
    x0 = setting['particle_init_fn'](N_PARTICLES, rng)
    chance = setting['chance_loss']
    emit(f"transformer calibration | depth={DEPTH} d={setting['d']} n={N_PARTICLES} "
         f"n_train={setting['n_train']}")
    emit(f"chance loss = log(p) = {chance:.4f} | target = {setting['target_loss']}")
    emit("A value at or above chance means the method has not learned anything at all.")
    emit()

    # --- Stage A: scale diagnostic --------------------------------------------------------
    raw = setting['f_grad_vec'](x0)
    norms = np.linalg.norm(raw, axis=1)
    emit("=== A. Scale diagnostic at initialization ===")
    emit(f"  raw gradient norm      : mean {norms.mean():.4f}  (min {norms.min():.4f}, "
         f"max {norms.max():.4f})")
    for clip in (1.0, 10.0):
        eff = min(1.0, clip / norms.mean())
        emit(f"  clip={clip:<5} keeps {eff*100:5.1f}% of the gradient; at eta=1e-3 the step norm "
             f"is {1e-3*min(norms.mean(), clip):.2e} over d={setting['d']}")
    emit(f"  per-coordinate step at eta=1e-3, clip=1.0: "
         f"~{1e-3/np.sqrt(setting['d']):.2e} -- compare typical weight scale ~1e-1")
    emit()

    # --- Stage B: (eta, clip) grid --------------------------------------------------------
    emit(f"=== B. (eta, clip) grid, Softmin, beta=10, sigma=0, {STEPS_GRID} steps ===")
    emit(f"  {'eta':>8} {'clip':>6} {'loss':>10}  {'verdict':<28} {'secs':>6}")
    best, best_loss = None, float('inf')
    for eta in ETA_CANDIDATES:
        for clip in CLIP_CANDIDATES:
            t0 = time.perf_counter()
            loss = run_softmin(setting, x0, STEPS_GRID, eta, clip, beta=10.0, sigma=0.0)
            verdict = ('diverged' if not np.isfinite(loss) or loss > 10 * chance
                       else 'no movement' if loss >= chance
                       else 'learning')
            emit(f"  {eta:>8} {str(clip):>6} {loss:>10.4f}  {verdict:<28} "
                 f"{time.perf_counter()-t0:>6.0f}")
            if np.isfinite(loss) and loss < best_loss:
                best, best_loss = (eta, clip), loss
    emit(f"  -> selected (eta, clip) = {best} at loss {best_loss:.4f}")
    emit()

    eta, clip = best

    # --- Stage C: beta and sigma at the selected (eta, clip) ------------------------------
    emit(f"=== C1. beta sweep at eta={eta}, clip={clip}, sigma=0, {STEPS_BETA} steps ===")
    emit(f"  {'beta':>8} {'loss':>10}")
    beta_best, beta_loss = None, float('inf')
    for beta in BETA_CANDIDATES:
        loss = run_softmin(setting, x0, STEPS_BETA, eta, clip, beta, sigma=0.0)
        emit(f"  {beta:>8} {loss:>10.4f}")
        if np.isfinite(loss) and loss < beta_loss:
            beta_best, beta_loss = beta, loss
    emit(f"  -> selected beta = {beta_best} at loss {beta_loss:.4f}")
    emit()

    emit(f"=== C2. sigma re-check at eta={eta}, clip={clip}, beta={beta_best}, "
         f"{STEPS_BETA} steps ===")
    emit(f"  {'sigma':>8} {'noise std/step':>15} {'loss':>10}")
    sigma_best, sigma_loss = None, float('inf')
    for sigma in SIGMA_CANDIDATES:
        loss = run_softmin(setting, x0, STEPS_BETA, eta, clip, beta_best, sigma)
        emit(f"  {sigma:>8} {np.sqrt(2*sigma*eta):>15.6f} {loss:>10.4f}")
        if np.isfinite(loss) and loss < sigma_loss:
            sigma_best, sigma_loss = sigma, loss
    emit(f"  -> selected sigma = {sigma_best} at loss {sigma_loss:.4f}")
    emit()

    # --- Stage D: stability of the selected configuration at the deepest sweep point -------
    emit(f"=== D. Selected config re-checked at depth {DEEP_DEPTH}, {STEPS_DEEP} steps ===")
    deep = transformer_setting(depth=DEEP_DEPTH)
    deep_rng = np.random.RandomState(0)
    deep_x0 = deep['particle_init_fn'](N_PARTICLES, deep_rng)
    emit(f"  depth {DEEP_DEPTH}: d={deep['d']:,}, chance={deep['chance_loss']:.4f}")
    deep_loss = run_softmin(deep, deep_x0, STEPS_DEEP, eta, clip, beta_best, sigma_best)
    if not np.isfinite(deep_loss):
        verdict = "DIVERGED -- eta is too large at this depth, re-run stage B at DEPTH=DEEP_DEPTH"
    elif deep_loss >= deep['chance_loss']:
        verdict = "NO MOVEMENT -- eta may be too small, or 2,000 steps too few at this depth"
    else:
        verdict = "stable and learning"
    emit(f"  Softmin at eta={eta}, beta={beta_best}, sigma={sigma_best}: "
         f"loss {deep_loss:.4f} -> {verdict}")
    emit()

    # --- Adam reference at the same budget ------------------------------------------------
    np.random.seed(0)
    ad = Adam(n=N_PARTICLES, d=setting['d'], f=setting['f_vec'],
              df=clipped(setting['f_grad_vec'], clip),
              particle_init=setting['particle_init'], bounds=setting['bounds'],
              learning_rate=1e-3)
    ad.reset(particle_init=x0)
    ad.run(STEPS_BETA)
    emit(f"=== Reference: Adam (lr=1e-3, clip={clip}) at the same {STEPS_BETA}-step budget ===")
    emit(f"  loss {ad.best_global_value:.4f}")
    emit()
    emit("SUMMARY -- set these in run_transformer.py (and check stage D's verdict before "
         "trusting them at the deep end):")
    emit(f"  LR = {eta}")
    emit(f"  MAX_GRAD_NORM = {clip}")
    emit(f"  SOFTMIN_BETA = {beta_best}")
    emit(f"  NOISE_SIGMA = {sigma_best}")
    out.close()


if __name__ == '__main__':
    main()
