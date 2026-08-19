"""Combined visual sweep of hybrid_quadruple_well_setting across the local pilot's A range --
one multi-panel figure so bounds-containment can be scanned across every A at once, rather than
spot-checked at 1-2 values (which is what missed the old unbalanced_quadruple_well_setting's
bounds problem at A=15 in the first place). Not a publication figure -- panels are kept small and
compact for a fast visual scan.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from potentials import hybrid_quadruple_well_setting

TILT_COEF = 0.05 / 2 ** 1.5
KAPPA_FRAC = 0.7
A_VALUES = list(range(2, 21))

WELL_STYLE = {
    'start': ('start', 'red', 'o'),
    'neighbor1': ('neighbor', 'white', 's'),
    'neighbor2': ('neighbor', 'white', 's'),
    'target': ('target', '#2ECC71', '*'),
}


def draw_panel(ax, A):
    tilt = TILT_COEF * A ** 1.5
    setting = hybrid_quadruple_well_setting(A, tilt, kappa_frac=KAPPA_FRAC)
    b = setting['bounds']
    wells = setting['wells']

    xs = np.linspace(b[0][0], b[0][1], 150)
    ys = np.linspace(b[1][0], b[1][1], 150)
    X, Y = np.meshgrid(xs, ys)
    Z = setting['f_vec'](np.stack([X.ravel(), Y.ravel()], axis=1)).reshape(X.shape)
    ax.contourf(X, Y, Z, levels=25, cmap='viridis')

    for name, (x, y) in wells.items():
        label, color, marker = WELL_STYLE[name]
        ax.scatter([x], [y], color=color, marker=marker, s=35, edgecolors='black', linewidths=0.5, zorder=5)

    for region, color in [(setting['particle_init'], 'black'), (setting['set'], '#2ECC71')]:
        ax.add_patch(plt.Rectangle(
            (region[0][0], region[1][0]), region[0][1] - region[0][0], region[1][1] - region[1][0],
            fill=False, edgecolor=color, linewidth=0.8))

    ax.set_xlim(b[0])
    ax.set_ylim(b[1])
    ax.set_aspect('equal')
    ax.set_title(f'A={A}', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == '__main__':
    ncols = 5
    nrows = -(-len(A_VALUES) // ncols)  # ceil div
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
    axes = axes.ravel()

    for ax, A in zip(axes, A_VALUES):
        draw_panel(ax, A)
    for ax in axes[len(A_VALUES):]:
        ax.axis('off')

    fig.suptitle(
        'hybrid_quadruple_well_setting bounds-containment sweep\n'
        'red=start, white=neighbors, green star=target; black/green boxes=particle_init/set',
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = 'results/hybrid_quadruple_well_landscape_sweep.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'saved {out}')
