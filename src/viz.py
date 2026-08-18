"""Plotting and animation utilities.

Contains:
  - `plot_simulation`: the landscape/particle visualization used by `Simulation.plot()`,
    deduplicated across the 1D/2D/>2D and static/gif variants.
  - `plot_entry_time_comparison`: the shared "mean entry time vs. parameter" errorbar plot
    used by the double-well, quadruple-well, and Ackley comparison figures in the notebook.
  - `plot_value_comparison`: the value-space analogue (mean +/- IQR of a value-space metric vs.
    parameter, e.g. quenched Lennard-Jones energy gap or re-evaluated RL return).
"""
import os
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ---------------------------------------------------------------------------
# Shared baseline label/color/marker convention (Okabe-Ito colorblind-safe palette), so the
# notebook's double-well/quadruple-well/Ackley/Lennard-Jones comparison cells don't each
# hand-duplicate the same 8-9 label/color/marker triples (2026-08-07).
# ---------------------------------------------------------------------------

BASELINE_STYLE = {
    'softmin':       ('SoftMin', '#0072B2', 'o'),
    'softmin_decay': ('SoftMin Decay', '#F0E442', 'd'),
    'sa':            ('Simulated Annealing', '#E69F00', 's'),
    'langevin':      ('Langevin', '#000000', 'x'),
    'cbo':           ('CBO', '#009E73', '^'),
    'cbo_annealed':  ('CBO Annealed', '#56B4E9', 'v'),
    'pso':           ('PSO', '#D55E00', 'D'),
    'standard_cbo':  ('Standard CBO', '#CC79A7', 'P'),
    'standard_pso':  ('Standard PSO', '#999999', 'X'),
}


def baseline_series(algo_names, filenames, status_filenames=None):
    """Build a `series` list (for `plot_entry_time_comparison`/`plot_value_comparison`) from
    `BASELINE_STYLE`.

    `filenames` (and optional `status_filenames`) is either a single string template containing
    '{p}' (reused for every algo, `.format(p=p)`-compatible with `plot_entry_time_comparison`
    directly -- pass e.g. 'softmin_double_well_{p}.pkl'.format-ready with '{a}' substituted here
    for the algo key), or a `{algo: '{p}'-templated string}` dict for benchmarks whose per-algo
    filenames aren't a uniform pattern (e.g. Ackley's `FILENAME_BY_ALGO`, which capitalizes each
    algo's prefix differently -- import the driver's own dict rather than re-deriving the mapping
    here, so the notebook can't silently drift from what the driver actually writes).
    """
    def _resolve(spec, a):
        if spec is None:
            return None
        return spec[a] if isinstance(spec, dict) else spec.format(a=a, p='{p}')

    series = []
    for a in algo_names:
        label, color, marker = BASELINE_STYLE[a]
        s = {'label': label, 'filename_template': _resolve(filenames, a),
             'color': color, 'marker': marker}
        status_template = _resolve(status_filenames, a)
        if status_template is not None:
            s['status_filename_template'] = status_template
        series.append(s)
    return series


# ---------------------------------------------------------------------------
# Simulation landscape/particle plotting
# ---------------------------------------------------------------------------

def _projected_mesh_eval(f, bounds, proj_axes, full_dim, n=100):
    """Evaluate `f` over a 2D mesh spanning `bounds[proj_axes]`, embedded in `full_dim`
    dimensions (other coordinates held at 0). Returns X, Y, Z (each shape (n, n))."""
    x = np.linspace(bounds[proj_axes[0]][0], bounds[proj_axes[0]][1], n)
    y = np.linspace(bounds[proj_axes[1]][0], bounds[proj_axes[1]][1], n)
    X, Y = np.meshgrid(x, y)
    input_data = np.zeros((X.size, full_dim))
    input_data[:, proj_axes[0]] = X.ravel()
    input_data[:, proj_axes[1]] = Y.ravel()
    Z = f(input_data).reshape(X.shape)
    return X, Y, Z


def _draw_3d_landscape(ax, f, bounds, proj_axes, full_dim, particles, best_position, best_value,
                        cmap, xlabel, ylabel, title):
    X, Y, Z = _projected_mesh_eval(f, bounds, proj_axes, full_dim)
    ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.4)
    ax.scatter(particles[:, proj_axes[0]], particles[:, proj_axes[1]], f(particles),
               label="Particles", color="blue")
    ax.scatter(best_position[proj_axes[0]], best_position[proj_axes[1]], best_value,
               color="red", label="Best Global Position")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("f(x)")
    ax.set_title(title)
    ax.legend()


def _animate_3d_landscape(fig, ax, history, f, proj_axes, best_position, best_value):
    particles_scatter = ax.scatter(history[0][:, proj_axes[0]], history[0][:, proj_axes[1]],
                                    f(history[0]), color="blue", label="Particles")
    best_scatter = ax.scatter(best_position[proj_axes[0]], best_position[proj_axes[1]],
                               best_value, color="red", label="Best Global Position")

    def animate(i):
        particles_scatter._offsets3d = (history[i][:, proj_axes[0]], history[i][:, proj_axes[1]],
                                         f(history[i]))
        best_scatter._offsets3d = ([best_position[proj_axes[0]]], [best_position[proj_axes[1]]],
                                   [best_value])
        return particles_scatter, best_scatter

    return animation.FuncAnimation(fig, animate, frames=len(history), interval=100)


def plot_simulation(sim, path="", make_gif=False, **kwargs):
    """Visualize a Simulation's landscape and particles (static), optionally saving a gif
    of the recorded `sim.history`. `sim` is any `Simulation` subclass instance."""
    figsize = (10, 6)
    cmap = kwargs.get("cmap", "viridis")

    if sim.d > 2:
        proj_axes = kwargs.get("projection_axes", [0, 1])
        assert len(proj_axes) == 2, "projection_axes must be a list of length 2."
        assert all(0 <= p < sim.d for p in proj_axes), "projection_axes must be in the range of the dimension of the space."
        assert proj_axes[0] != proj_axes[1], "projection_axes must be different."

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(sim.bounds[proj_axes[0]][0], sim.bounds[proj_axes[0]][1])
        ax.set_ylim(sim.bounds[proj_axes[1]][0], sim.bounds[proj_axes[1]][1])
        _draw_3d_landscape(ax, sim.f, sim.bounds, proj_axes, sim.d, sim.particles,
                            sim.best_global_position, sim.best_global_value, cmap,
                            f"x{proj_axes[0]}", f"x{proj_axes[1]}",
                            f"3D Particles with Function Landscape - projection axes {proj_axes}")
        plt.show()

    elif sim.d == 2:
        proj_axes = [0, 1]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        _draw_3d_landscape(ax, sim.f, sim.bounds, proj_axes, sim.d, sim.particles,
                            sim.best_global_position, sim.best_global_value, cmap,
                            "x", "y", "3D Particles with Function Landscape")
        ax.set_zlabel("f(x, y)")
        plt.show()

    elif sim.d == 1:
        x = np.linspace(sim.bounds[0][0], sim.bounds[0][1], 1000).reshape(-1, 1)
        plt.figure(figsize=figsize)
        plt.plot(x, sim.f(x))
        plt.scatter(sim.particles, sim.f(sim.particles), color="blue")
        plt.scatter(sim.best_global_position, sim.best_global_value, color="red")
        plt.xlabel("x")
        plt.title("Particles")
        plt.show()

    if make_gif:
        if sim.d > 2:
            proj_axes = kwargs.get("projection_axes", [0, 1])
            assert len(proj_axes) == 2, "projection_axes must be a list of length 2."
            assert all(0 <= p < sim.d for p in proj_axes), "projection_axes must be in the range of the dimension of the space."
            assert proj_axes[0] != proj_axes[1], "projection_axes must be different."

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            X, Y, Z = _projected_mesh_eval(sim.f, sim.bounds, proj_axes, sim.d)
            ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.4)
            ax.set_xlabel(f"x{proj_axes[0]}")
            ax.set_ylabel(f"x{proj_axes[1]}")
            ax.set_zlabel("f(x)")
            ax.set_title(f"3D Particles with Function Landscape - projection axes {proj_axes}")
            ax.legend()

            ani = _animate_3d_landscape(fig, ax, sim.history, sim.f, proj_axes,
                                         sim.best_global_position, sim.best_global_value)
            ani.save(path+f"{sim.name}_{sim.n}_particles_steps{sim.n_steps}_3D.gif", writer="pillow", fps=30)

        elif sim.d == 2:
            proj_axes = [0, 1]
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            X, Y, Z = _projected_mesh_eval(sim.f, sim.bounds, proj_axes, sim.d)
            ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.4)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("f(x, y)")
            ax.set_title("3D Particles with Function Landscape")
            ax.legend()

            ani = _animate_3d_landscape(fig, ax, sim.history, sim.f, proj_axes,
                                         sim.best_global_position, sim.best_global_value)
            ani.save(path+f"{sim.name}_{sim.n}_particles_steps{sim.n_steps}_3D.gif", writer="pillow", fps=30)

        elif sim.d == 1:
            x = np.linspace(sim.bounds[0][0], sim.bounds[0][1], 1000).reshape(-1, 1)
            f_curve = [sim.f(x_i) for x_i in x]
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(x, f_curve)
            line2 = ax.scatter(sim.history[0], sim.f(sim.history[0]))
            line3 = ax.scatter(sim.history[0][np.argmin(sim.f(sim.history[0]))],
                                np.min(sim.f(sim.history[0])), color="red")
            ax.set_xlim(sim.bounds[0][0], sim.bounds[0][1])
            ax.set_ylim(min(f_curve)-0.5, max(f_curve)+0.5)
            plt.xlabel("x")
            plt.title("Particles")

            history_eq_values = getattr(sim, "history_eq_values", None)
            if history_eq_values is None:
                def animate(i):
                    line2.set_offsets(np.c_[sim.history[i], sim.f(sim.history[i])])
                    line3.set_offsets(np.c_[sim.history[i][np.argmin(sim.f(sim.history[i]))],
                                             np.min(sim.f(sim.history[i]))])
                    return line2, line3
            else:
                green_line = ax.axhline(history_eq_values[0], color="green", linestyle="--", label="history_eq_value")
                ax.legend()

                def animate(i):
                    line2.set_offsets(np.c_[sim.history[i], sim.f(sim.history[i])])
                    line3.set_offsets(np.c_[sim.history[i][np.argmin(sim.f(sim.history[i]))],
                                             np.min(sim.f(sim.history[i]))])
                    green_line.set_ydata(history_eq_values[i])
                    return line2, line3, green_line

            ani = animation.FuncAnimation(fig, animate, frames=len(sim.history), interval=100)
            noise = getattr(sim, "noise", None)
            ani.save(path+f"{sim.name}_{sim.n}_particles_steps{sim.n_steps}_noise_{noise}.gif", writer="pillow", fps=30)


# ---------------------------------------------------------------------------
# Entry-time comparison plots (double-well / quadruple-well / Ackley figures)
# ---------------------------------------------------------------------------

def _asymmetric_errorbars(mean, q1, q3):
    mean = np.asarray(mean, dtype=float)
    lower = mean - np.asarray(q1, dtype=float)
    upper = np.asarray(q3, dtype=float) - mean
    return np.vstack([np.maximum(lower, 0), np.maximum(upper, 0)])


# Points at or above this reached-fraction are treated as fully reliable and left unannotated;
# below it (but still >= min_reached_frac, so still plotted) they get a small "NN% reached" label
# so a caller skimming the figure can't mistake "reliable but not perfect" for "100% reliable" --
# the two previously rendered identically once a point cleared min_reached_frac at all.
RELIABLE_ANNOTATION_THRESHOLD = 0.9

_DEFAULT_LEGEND_KWARGS = dict(fontsize=9, ncol=2, framealpha=0.9)


def plot_entry_time_comparison(x, results_dir, series, xlabel, ylabel="Log Mean Entry Time",
                                title="", yscale="log", grid=False, figsize=(11, 6), params=None,
                                min_reached_frac=0.0, legend_kwargs=None):
    """Plot mean (+/- IQR) entry time vs. a swept parameter for several methods.

    Args:
        x: sequence of values for the plot's x-axis (e.g. potential barrier height).
        results_dir: directory containing the `.pkl` files (each an array of entry times).
        series: list of dicts, one per method, with keys:
            label, filename_template (str.format(p=value)), color, marker (optional, default 'o'),
            status_filename_template (optional, str.format(p=value): parallel per-seed status
            strings, e.g. produced by a second reconstruct_pkl_files_from_csv(value_transform=str)
            call against the *_status CSV columns -- see run_experiments_resumable.py. When given,
            each point's mean/IQR is computed only over the seeds whose status is
            'target_reached', so a run that never escaped can't drag the mean toward the
            max_steps*lr timeout sentinel (2026-08-07: replaces the old point-level `mask_below`
            numeric cutoff, which silently stopped matching the sentinel once MAX_STEPS changed --
            see the plan's B3 finding).
            mask_below (optional float, legacy fallback for series with no status pkl available:
            only plot points whose mean is below this cutoff).
        params: sequence used to build each filename (one .pkl per entry, in the same order as
            `x`), when it differs from the x-axis values themselves (e.g. the sweep is indexed
            by delta/A but plotted against a derived barrier height). Defaults to `x`.
        min_reached_frac: if a series has status data, drop any point where the reached-fraction
            (seeds with status=='target_reached' / total seeds) is below this threshold, even if
            the (possibly small-sample) mean of the reached seeds looks fine -- avoids a single
            lucky seed out of 100 producing a misleadingly confident-looking point. Points that
            clear this cutoff but are still below `RELIABLE_ANNOTATION_THRESHOLD` get a small
            reached-fraction label so partial reliability stays visible even once plotted (2026:
            survivorship-bias fix -- see the report that flagged points built from a handful of
            lucky seeds rendering identically to points built from a full successful cohort).
        legend_kwargs: dict merged over `_DEFAULT_LEGEND_KWARGS` (fontsize=9, ncol=2,
            framealpha=0.9) and passed to `ax.legend()` -- the previous bare `ax.legend()` produced
            an oversized single-column legend that crowded an 8-series plot's data.
        xlabel, ylabel, title, yscale, grid, figsize: standard matplotlib figure options.

    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.asarray(x)
    params = x if params is None else list(params)

    for s in series:
        means, q1s, q3s, keep, reached_fracs = [], [], [], [], []
        status_template = s.get('status_filename_template')
        for p in params:
            filename = os.path.join(results_dir, s['filename_template'].format(p=p))
            with open(filename, "rb") as f:
                times = np.asarray(pkl.load(f))

            if status_template is not None:
                status_filename = os.path.join(results_dir, status_template.format(p=p))
                with open(status_filename, "rb") as f:
                    statuses = np.asarray(pkl.load(f))
                reached = statuses == 'target_reached'
                reached_frac = float(np.mean(reached)) if reached.size else 0.0
                point_times = times[reached]
                point_ok = reached_frac >= min_reached_frac and point_times.size > 0
            else:
                point_times = times
                point_ok = True
                reached_frac = 1.0

            means.append(np.mean(point_times) if point_ok else np.nan)
            q1s.append(np.percentile(point_times, 25) if point_ok else np.nan)
            q3s.append(np.percentile(point_times, 75) if point_ok else np.nan)
            keep.append(point_ok)
            reached_fracs.append(reached_frac)

        means = np.asarray(means, dtype=float)
        err = _asymmetric_errorbars(means, q1s, q3s)
        mask = np.asarray(keep, dtype=bool)
        reached_fracs = np.asarray(reached_fracs, dtype=float)

        if status_template is None:
            mask_below = s.get('mask_below')
            if mask_below is not None:
                mask = mask & (means < mask_below)

        ax.errorbar(x[mask], means[mask], yerr=err[:, mask], label=s['label'],
                    marker=s.get('marker', 'o'), color=s.get('color'), capsize=5, linestyle='-')

        if status_template is not None:
            for xi, yi, frac in zip(x[mask], means[mask], reached_fracs[mask]):
                if frac < RELIABLE_ANNOTATION_THRESHOLD:
                    ax.annotate(f"{frac * 100:.0f}%", (xi, yi), textcoords="offset points",
                                xytext=(0, -11), ha="center", fontsize=6.5, color="#666666")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if yscale:
        ax.set_yscale(yscale)
    if grid:
        ax.grid()
    ax.legend(**{**_DEFAULT_LEGEND_KWARGS, **(legend_kwargs or {})})
    return fig, ax


def plot_value_comparison(x, results_dir, series, xlabel, ylabel="Value", title="",
                           yscale=None, grid=False, figsize=(11, 6), params=None,
                           value_transform=None, legend_kwargs=None):
    """Plot mean (+/- IQR) of a value-space metric (e.g. quenched Lennard-Jones energy gap,
    re-evaluated RL return) vs. a swept parameter for several methods.

    Structurally identical to `plot_entry_time_comparison` (same .pkl-per-(algorithm, parameter)
    convention, same asymmetric-IQR errorbars, same `series`/`params` shape) but without that
    function's entry-time-specific defaults (no forced log-scale y-axis, no `mask_below` timeout
    cutoff), and with an optional `value_transform(values, p) -> values` hook applied to each
    loaded array before its mean/IQR are computed -- e.g. subtracting a parameter-dependent
    literature reference (`LJ_GLOBAL_MINIMA[p]`) to turn raw quenched energies into gaps.
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.asarray(x)
    params = x if params is None else list(params)

    for s in series:
        means, q1s, q3s = [], [], []
        for p in params:
            filename = os.path.join(results_dir, s['filename_template'].format(p=p))
            with open(filename, "rb") as f:
                values = pkl.load(f)
            if value_transform is not None:
                values = value_transform(values, p)
            means.append(np.mean(values))
            q1s.append(np.percentile(values, 25))
            q3s.append(np.percentile(values, 75))

        means = np.asarray(means, dtype=float)
        err = _asymmetric_errorbars(means, q1s, q3s)

        ax.errorbar(x, means, yerr=err, label=s['label'],
                    marker=s.get('marker', 'o'), color=s.get('color'), capsize=5, linestyle='-')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if yscale:
        ax.set_yscale(yscale)
    if grid:
        ax.grid()
    ax.legend(**{**_DEFAULT_LEGEND_KWARGS, **(legend_kwargs or {})})
    return fig, ax


# ---------------------------------------------------------------------------
# Ascent-descent prefactor dynamics (A_{t,k})
# ---------------------------------------------------------------------------

def plot_prefactor_dynamics(sim, figsize=(10, 6)):
    """Plot each particle's effective learning rate A_{t,k} over iterations.

    Requires `sim` (a `SoftminEnergyAdagrad` instance) to have been constructed with
    `track_prefactor_history=True` and already run. A_{t,k} > 1 is the "strongly minimizing"
    regime, 0 < A_{t,k} < 1 is plain minimizing (descent), and A_{t,k} < 0 is the maximizing
    (ascent/exploration) regime.
    """
    history = getattr(sim, "prefactor_history", None)
    if not history:
        raise ValueError(
            "sim.prefactor_history is empty -- construct with track_prefactor_history=True "
            "and call .run(...) before plotting."
        )

    history = np.asarray(history)  # shape (n_recorded_steps, n_particles)

    fig, ax = plt.subplots(figsize=figsize)
    for k in range(history.shape[1]):
        ax.plot(history[:, k], alpha=0.5, linewidth=1)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="Descent/ascent threshold (A=0)")
    ax.axhline(1, color="red", linestyle=":", linewidth=1, label="Strongly minimizing threshold (A=1)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$A_{t,k}$")
    ax.set_title("Ascent-Descent Dynamics: per-particle effective learning rate")
    ax.legend()
    return fig, ax


# ---------------------------------------------------------------------------
# Ablation mode distribution (violin/box) -- for ablation settings with no natural sweep axis
# (a single fixed, hard dimension/cluster size), where a per-mode distribution plot is more
# informative than collapsing each mode to a single mean +/- IQR point.
# ---------------------------------------------------------------------------

PREFACTOR_MODE_STYLE = {
    'signed':        ('Signed (default)', '#0072B2'),
    'positive_part': ('Positive part', '#E69F00'),
    'absolute':      ('Absolute value', '#009E73'),
}


def plot_mode_distribution(datasets, ylabel, title="", kind="violin", figsize=(6, 5), modes=None):
    """Plot the per-seed distribution of a scalar metric for each ascent-descent prefactor_mode
    variant (see `SoftminEnergyAdagrad`'s `prefactor_mode` -- 'signed', 'positive_part',
    'absolute').

    Args:
        datasets: {mode_name: array of per-seed values}.
        ylabel, title: standard matplotlib labels.
        kind: 'violin' (default) or 'box'.
        modes: order to plot the modes in (defaults to `PREFACTOR_MODE_STYLE`'s own order,
            filtered to whichever keys are actually present in `datasets`).

    Returns:
        (fig, ax)
    """
    modes = modes or [m for m in PREFACTOR_MODE_STYLE if m in datasets]
    labels = [PREFACTOR_MODE_STYLE[m][0] for m in modes]
    colors = [PREFACTOR_MODE_STYLE[m][1] for m in modes]
    values = [np.asarray(datasets[m], dtype=float) for m in modes]

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(1, len(modes) + 1)

    if kind == "violin":
        parts = ax.violinplot(values, positions=positions, showmeans=True, showmedians=False)
        for body, color in zip(parts['bodies'], colors):
            body.set_facecolor(color)
            body.set_alpha(0.6)
        for key in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            if key in parts:
                parts[key].set_color('black')
    elif kind == "box":
        bp = ax.boxplot(values, positions=positions, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    else:
        raise ValueError("kind must be 'violin' or 'box'")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax
