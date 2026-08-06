"""Benchmark potentials/objective functions and particle-initialization helpers.

Each `*_setting` factory returns a dict with:
    f_vec        - vectorized objective f(x) -> shape (n,)
    f_grad_vec   - vectorized gradient df(x) -> shape (n, d)
    particle_init- initial sampling bounds/positions for particles
    set          - target region used by Simulation.first_entry_time
    bounds       - domain clipping bounds
    critical_points (where applicable)
"""
import numpy as np

from typing import Dict, Any

from sympy import symbols, diff, solve, lambdify, Piecewise, exp, sin, cos, sqrt, pi, log
from numpy.polynomial import polynomial as P
from sympy.plotting import plot


def one_dimensional_double_well(delta):
    x = symbols('x')
    f = x**4 - delta * x**2
    df = diff(f, x)
    a, b, c = P.polyroots([0, -2*delta, 0, 4])

    f_num = lambdify(x, f, modules='numpy')
    df_num = lambdify(x, df, modules='numpy')

    def double_well(x):
        return f_num(x)

    def double_well_grad(x):
        return df_num(x)

    double_well_vec = np.vectorize(double_well, signature='(n)->()')
    double_well_grad_vec = np.vectorize(double_well_grad, signature='(n)->(n)')

    return {
        'f_vec': double_well_vec,
        'f_grad_vec': double_well_grad_vec,
        'particle_init': [[a-0.2, a+0.2]],
        'set': [[c-0.2, c+0.2]],
        'bounds': [[-0.5-np.sqrt(delta), np.sqrt(delta)+0.5]],
        'critical_points': [a, b, c],
    }


def asymmetric_double_well(delta, tilt):
    """1D double well with a linear tilt term breaking left/right symmetry, so the two minima
    have genuinely different depths. Needed to test escape from a suboptimal basin (the
    symmetric `one_dimensional_double_well` has two equally-good minima, so there is no
    suboptimal basin to escape from) — see Theorem 3.1 in `paper/SoftMin_revised_theory.tex`,
    which analyzes escape from a starting basin `z_-` into an adjacent basin `z_+` with relative
    height rho_+ = f(z_+) - f(z_-) >= 0.
    """
    x = symbols('x')
    f = x**4 - delta * x**2 + tilt * x
    df = diff(f, x)

    # f'(x) = 4x^3 - 2*delta*x + tilt; the tilt term breaks the closed-form factorization
    # `one_dimensional_double_well` relies on (roots 0, +-sqrt(delta/2)), so this needs the
    # general cubic solver -- still handled numerically by polyroots.
    roots = P.polyroots([tilt, -2 * delta, 0, 4])
    real_roots = np.sort(roots[np.abs(roots.imag) < 1e-8].real)
    if len(real_roots) != 3:
        raise ValueError(
            f"asymmetric_double_well(delta={delta}, tilt={tilt}) does not have 3 real critical "
            f"points (found {len(real_roots)}); reduce |tilt| relative to delta."
        )
    left, saddle, right = real_roots

    f_num = lambdify(x, f, modules='numpy')
    df_num = lambdify(x, df, modules='numpy')

    def well(x):
        return f_num(x)

    def well_grad(x):
        return df_num(x)

    well_vec = np.vectorize(well, signature='(n)->()')
    well_grad_vec = np.vectorize(well_grad, signature='(n)->(n)')

    # The suboptimal (shallower) well is the escape-test starting point z_-; the deeper well is
    # the escape target z_+, regardless of which side of the saddle each one falls on.
    if well(np.array([left])) >= well(np.array([right])):
        z_start, z_target = left, right
    else:
        z_start, z_target = right, left

    return {
        'f_vec': well_vec,
        'f_grad_vec': well_grad_vec,
        'particle_init': [[z_start - 0.2, z_start + 0.2]],
        'set': [[z_target - 0.2, z_target + 0.2]],
        'bounds': [[left - 0.5, right + 0.5]],
        'critical_points': [left, saddle, right],
        'z_start': z_start,
        'saddle': saddle,
        'z_target': z_target,
    }


def unbalanced_quadruple_well_setting(A, delta):
    x, y = symbols('x y')
    f = x**4 + y**4 - A*(x**2 + y**2) - delta*x*y
    f_x = diff(f, x)
    f_y = diff(f, y)

    f_num = lambdify((x, y), f, modules='numpy')
    f_x_num = lambdify((x, y), f_x, modules='numpy')
    f_y_num = lambdify((x, y), f_y, modules='numpy')

    crit_value = 0.5*np.sqrt(2*A+delta)
    critical_points = solve([f_x, f_y], (x, y))

    def quadruple_well(x):
        return f_num(x[0], x[1])

    def quadruple_well_grad(x):
        return np.array([f_x_num(x[0], x[1]), f_y_num(x[0], x[1])])

    quadruple_well_vec = np.vectorize(quadruple_well, signature='(n)->()')
    quadruple_well_grad_vec = np.vectorize(quadruple_well_grad, signature='(n)->(n)')

    return {'f_vec': quadruple_well_vec,
            'f_grad_vec': quadruple_well_grad_vec,
            'particle_init': [[-crit_value-0.2, -crit_value+0.2], [crit_value-0.2, crit_value+0.2]],
            'set': [[crit_value-0.2, crit_value+0.2], [-crit_value-0.2, -crit_value+0.2]],
            'bounds': [[-0.5-np.sqrt(A), np.sqrt(A)+0.5], [-0.5-np.sqrt(A), np.sqrt(A)+0.5]],
            'critical_points': critical_points}


def low_dimensional_quadruple_well_setting(A, delta, dimension):
    setting = unbalanced_quadruple_well_setting(A, delta)
    qw_vec = setting['f_vec']
    qw_grad_vec = setting['f_grad_vec']

    def f(x):
        return qw_vec(x[:2]) + np.sum(0.01*x[2:]**2)

    def f_grad(x):
        return np.concatenate([qw_grad_vec(x[:2]), 0.02*x[2:]])

    f_vec = np.vectorize(f, signature='(n)->()')
    f_grad_vec = np.vectorize(f_grad, signature='(n)->(n)')
    particle_init = setting['particle_init'] + [[-0.2, 0.2]]*(dimension-2)
    set_ = setting['set'] + [[-0.2, 0.2]]*(dimension-2)
    return {'f_vec': f_vec,
            'f_grad_vec': f_grad_vec,
            'particle_init': particle_init,
            'set': set_}


def asymmetric_potential_barrier(beta=1, depth=2, offset=5, plot_=False):
    x = symbols('x')
    a, b, c, d = symbols('a b c d')

    f1 = x**2/(1+exp(-beta*x))
    domain_f1 = x < offset/3
    df1_dx = diff(f1, x)

    f2 = a*x**3+b*x**2+c*x+d
    domain_f2 = (x >= offset/3) & (x < 2/3*offset)
    df2_dx = diff(f2, x)

    f3 = (x-offset)**2-depth
    domain_f3 = x >= 2/3*offset
    df3_dx = diff(f3, x)

    eq1 = f1.subs(x, offset/3)-f2.subs(x, offset/3)
    eq2 = df1_dx.subs(x, offset/3)-df2_dx.subs(x, offset/3)
    eq3 = f2.subs(x, 2/3*offset)-f3.subs(x, 2/3*offset)
    eq4 = df2_dx.subs(x, 2/3*offset)-df3_dx.subs(x, 2/3*offset)
    sol = solve([eq1, eq2, eq3, eq4], (a, b, c, d))

    if sol:
        a_sol, b_sol, c_sol, d_sol = sol[a], sol[b], sol[c], sol[d]
        smooth_f = Piecewise((f1, domain_f1), (f2.subs({a: a_sol, b: b_sol, c: c_sol, d: d_sol}), domain_f2), (f3, domain_f3))
        plot(smooth_f, (x, -4/3*offset, 4/3*offset), title="Asymmetric Potential Barrier", show=plot_)
        smooth_f_num = lambdify(x, smooth_f, modules='numpy')
        smooth_f_vec = np.vectorize(smooth_f_num, signature='(n)->()')
        smooth_f_grad = lambdify(x, diff(smooth_f, x), modules='numpy')
        smooth_f_grad_vec = np.vectorize(smooth_f_grad, signature='(n)->(n)')
        return {'f_vec': smooth_f_vec,
                'f_grad_vec': smooth_f_grad_vec,
                'particle_init': [[-0.2, 0.2]],
                'bounds': [[-4/3*offset, 4/3*offset]],
                'set': [[offset-0.2, offset+0.2]]}
    else:
        print("No solution found for the given parameters.")


def ackley_function_setting(a=20, b=0.2, c=2 * np.pi):
    x, y = symbols('x y')
    f = -a * exp(-b * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(c * x) + cos(c * y))) + exp(1) + 20
    df_dx = diff(f, x)
    df_dy = diff(f, y)

    f_num = lambdify((x, y), f, modules='numpy')
    df_dx_num = lambdify((x, y), df_dx, modules='numpy')
    df_dy_num = lambdify((x, y), df_dy, modules='numpy')

    def ackley(x):
        return f_num(x[0], x[1])

    def ackley_grad(x):
        return np.array([df_dx_num(x[0], x[1]), df_dy_num(x[0], x[1])])

    ackley_vec = np.vectorize(ackley, signature='(n)->()')
    ackley_grad_vec = np.vectorize(ackley_grad, signature='(n)->(n)')

    return {'f_vec': ackley_vec,
            'f_grad_vec': ackley_grad_vec,
            'particle_init': [[-8, 8], [-8, 8]],
            'set': [[-0.2, 0.2], [-0.2, 0.2]],
            'bounds': [[-10, 10], [-10, 10]]}


def high_dim_ackley_function_setting(
    d: int,
    a: float = 20.0,
    b: float = 0.2,
    c: float = 2.0 * np.pi
) -> Dict[str, Any]:
    """
    N-dimensional Ackley function setup using pure NumPy for fast vectorization.

    Returns:
        A dictionary containing the vectorized objective function, vectorized
        analytical gradient, particle initialization bounds, target set, and space bounds.
    """
    def ackley_vec(x: np.ndarray) -> np.ndarray:
        # x is expected to have shape (n_particles, d)
        mean_sq = np.mean(x**2, axis=-1)
        term1 = -a * np.exp(-b * np.sqrt(mean_sq))
        term2 = -np.exp(np.mean(np.cos(c * x), axis=-1))
        return term1 + term2 + a + np.exp(1)

    def ackley_grad_vec(x: np.ndarray) -> np.ndarray:
        # x is expected to have shape (n_particles, d)
        norm_sq = np.mean(x**2, axis=-1, keepdims=True)
        norm = np.sqrt(norm_sq)

        # Prevent division by zero if a particle collapses exactly onto the origin
        norm = np.where(norm == 0.0, 1e-8, norm)

        # Gradient of the first term
        term1_grad = (a * b / d) * (x / norm) * np.exp(-b * norm)

        # Gradient of the second term
        term2_exp = np.exp(np.mean(np.cos(c * x), axis=-1, keepdims=True))
        term2_grad = (c / d) * np.sin(c * x) * term2_exp

        return term1_grad + term2_grad

    return {
        'f_vec': ackley_vec,
        'f_grad_vec': ackley_grad_vec,
        'particle_init': [[-8.0, 8.0]] * d,
        'set': [[-0.2, 0.2]] * d,
        'bounds': [[-10.0, 10.0]] * d
    }


# Cambridge Cluster Database (Wales & Doye, J. Phys. Chem. A 1997) literature global minima for
# the standard 4*[r^-12 - r^-6] Lennard-Jones potential. These are ground truth -- never edit or
# "recompute" them; `lennard_jones_cluster_setting`'s `e_global` and the end-to-end verification
# check in the LJ driver both key off these exact values.
#   N=38  is the canonical "hard" double-funnel case: the true global minimum is an fcc truncated
#          octahedron, but a much larger basin of attraction leads to an easier icosahedral local
#          funnel at E ~= -173.25 -- most local optimizers land there, not at -173.928427.
#   N=75  (Marks decahedron) is also a known hard, multi-funnel case.
LJ_GLOBAL_MINIMA: Dict[int, float] = {
    13: -44.326801,
    38: -173.928427,
    55: -279.248470,
    75: -397.492331,
}


def initialize_lj_cluster(
    n_particles: int,
    n_atoms: int,
    radius: float = None,
    min_dist: float = 0.9,
    max_tries: int = 200,
) -> np.ndarray:
    """
    Samples `n_particles` independent N-atom (N=`n_atoms`) starting configurations for the LJ
    cluster potential, each atom drawn uniformly inside a ball of the given `radius` subject to
    every pairwise distance being >= `min_dist`. Naive uniform-box sampling routinely places two
    atoms nearly coincident; since the LJ potential scales as r^-12, that produces gradients of
    order 1e6+ on the very first optimizer step. Mirrors `initialize_on_hypersphere`'s
    normalize-then-scale idiom for the random direction, but with a random (not fixed) radius so
    points fill the ball rather than just its surface.

    Returns an (n_particles, 3*n_atoms) array, flattened to match the (n, d) shape every
    `Simulation` subclass expects.
    """
    if radius is None:
        radius = (n_atoms ** (1.0 / 3.0)) * 1.1 + 1.0

    configs = np.empty((n_particles, n_atoms, 3))
    for p in range(n_particles):
        pts = np.empty((n_atoms, 3))
        for i in range(n_atoms):
            for _ in range(max_tries):
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                r = radius * np.random.uniform(0.0, 1.0) ** (1.0 / 3.0)
                candidate = direction * r
                if i == 0 or np.all(np.linalg.norm(pts[:i] - candidate, axis=1) >= min_dist):
                    pts[i] = candidate
                    break
            else:
                raise RuntimeError(
                    f"initialize_lj_cluster: could not place atom {i}/{n_atoms} respecting "
                    f"min_dist={min_dist} inside radius={radius} after {max_tries} tries -- "
                    "increase radius or reduce min_dist."
                )
        configs[p] = pts
    return configs.reshape(n_particles, n_atoms * 3)


def lennard_jones_cluster_setting(
    n_atoms: int,
    container_radius: float = None,
    container_strength: float = 10.0,
    r2_floor: float = 0.09,
    max_grad_norm: float = None,
    box_pad: float = 2.0,
) -> Dict[str, Any]:
    """
    N-atom (N=`n_atoms`) Lennard-Jones cluster energy minimization in R^3, d=3*n_atoms.
    Hand-vectorized NumPy, following `high_dim_ackley_function_setting`'s pattern (sympy can't
    express an N-atom pairwise sum, and `np.vectorize` would loop over particles in Python).

    Energy: E(x) = sum_{i<j} 4*[r_ij^-12 - r_ij^-6], plus a confinement penalty (see below).
    Gradient (closed form, no square roots -- 1/r folded into the exponent):
        dE/dx_i = sum_{j != i} (24 r_ij^-8 - 48 r_ij^-14) (x_i - x_j)

    `r2_floor` clamps r^2 away from 0 on the diagonal (self-pairs, where x_i - x_j = 0 exactly):
    without it, r^2=0 raised to a negative power is inf, and inf * 0 is nan in the gradient sum.
    Off-diagonal near-coincident pairs are also clamped as an extra safety net on top of
    `initialize_lj_cluster`'s min-distance rejection.

    Confinement term (prevents "evaporation": atoms drifting to zero-gradient regions under SDE
    noise and never returning): E_conf(X) = container_strength * sum_i max(0, |x_i| - R_c)^2,
    origin-centered on purpose -- it's separable per atom, pins the cluster (removing the
    translational zero mode), and is exactly zero on the reference global minimum once centered
    inside R_c. Don't "fix" this to be centroid-relative -- that adds complexity for no benefit
    here.
    """
    d = 3 * n_atoms
    if container_radius is None:
        # Generous default: comfortably outside the equilibrium cluster radius (~O(N^(1/3)) atoms
        # near touching distance) for every N in LJ_GLOBAL_MINIMA; not performance-critical, just
        # needs to not clip the relaxed cluster.
        container_radius = 1.5 * (n_atoms ** (1.0 / 3.0)) + 1.5
    R_c = container_radius

    eye = np.eye(n_atoms, dtype=bool)

    def f_vec(x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        pos = x.reshape(n, n_atoms, 3)
        diff = pos[:, :, None, :] - pos[:, None, :, :]          # diff[:, i, j] = x_i - x_j
        r2 = np.maximum(np.sum(diff ** 2, axis=-1), r2_floor)   # (n, N, N)
        r6 = r2 ** -3
        pair_energy = 4.0 * (r6 ** 2 - r6)
        pair_energy = np.where(eye[None, :, :], 0.0, pair_energy)
        e_lj = 0.5 * np.sum(pair_energy, axis=(1, 2))           # 0.5: each pair counted twice

        r_atom = np.linalg.norm(pos, axis=-1)                   # (n, N)
        excess = np.maximum(0.0, r_atom - R_c)
        e_conf = container_strength * np.sum(excess ** 2, axis=-1)

        return e_lj + e_conf

    def f_grad_vec(x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        pos = x.reshape(n, n_atoms, 3)
        diff = pos[:, :, None, :] - pos[:, None, :, :]          # diff[:, i, j] = x_i - x_j
        r2 = np.maximum(np.sum(diff ** 2, axis=-1), r2_floor)   # (n, N, N)
        coeff = 24.0 * r2 ** -4 - 48.0 * r2 ** -7
        # No explicit diagonal mask needed here: diff[:, i, i] is exactly the zero vector, so
        # coeff * diff is zero at i==j regardless of the (finite, r2_floor-clamped) coeff value.
        grad_lj = np.sum(coeff[..., None] * diff, axis=2)       # sum over j -> (n, N, 3)

        # Clip only the LJ term, per atom (n, N, 1), BEFORE adding confinement -- not the
        # combined gradient. Confinement's restoring pull (2*container_strength*excess) needs to
        # scale up freely with however far an atom has escaped; capping it at the same ceiling as
        # the near-singular r^-13 LJ term would leave a runaway atom with no way back in, turning
        # a transient excursion into a permanent one.
        if max_grad_norm is not None:
            lj_norms = np.linalg.norm(grad_lj, axis=-1, keepdims=True)  # (n, N, 1)
            scale = np.minimum(1.0, max_grad_norm / np.maximum(lj_norms, 1e-12))
            grad_lj = grad_lj * scale

        r_atom = np.linalg.norm(pos, axis=-1, keepdims=True)    # (n, N, 1)
        r_atom_safe = np.where(r_atom == 0.0, 1e-12, r_atom)
        excess = np.maximum(0.0, r_atom - R_c)
        grad_conf = 2.0 * container_strength * excess * (pos / r_atom_safe)

        return (grad_lj + grad_conf).reshape(n, d)

    bound_extent = R_c + box_pad
    bounds = [[-bound_extent, bound_extent]] * d

    def particle_init_fn(n: int) -> np.ndarray:
        return initialize_lj_cluster(n_particles=n, n_atoms=n_atoms)

    return {
        'f_vec': f_vec,
        'f_grad_vec': f_grad_vec,
        'particle_init': bounds,
        # No coordinate-space target set: the global minimum is degenerate under O(3) rotation x
        # N! atom-permutation symmetry, so no small axis-aligned box captures "this is *a* copy of
        # the global minimum". Value-space (Simulation.first_value_time) is the only well-defined
        # target for this benchmark.
        'set': None,
        'bounds': bounds,
        'particle_init_fn': particle_init_fn,
        'd': d,
        'n_atoms': n_atoms,
        'e_global': LJ_GLOBAL_MINIMA.get(n_atoms),
    }


def initialize_on_circle(n, radius, noise=0.1):
    """
    Initialize n particles on a circle of given radius, with some noise
    """
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.random.normal(radius, noise, n)

    # Convert polar coordinates to Cartesian coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    particles = np.stack((x, y), axis=-1)
    return particles


def initialize_on_hypersphere(n, d, radius, noise=0.1):
    """
    Initialize n particles on a d-dimensional hypersphere of a given radius.
    """
    # Generate random directions using standard normal distribution
    directions = np.random.randn(n, d)
    # Normalize to project them onto the unit hypersphere
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Generate random radii with the specified noise
    radii = np.random.normal(radius, noise, (n, 1))

    # Scale directions by radii
    particles = directions * radii
    return particles
