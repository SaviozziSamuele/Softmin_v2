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
from scipy.optimize import fsolve

# Domain-bounds margin, used by every well-type potential below: extend `bounds` this many
# natural harmonic fluctuation-lengthscales (ell = 1/sqrt(curvature)) beyond the outermost
# critical point, rather than a fixed absolute distance. A fixed margin was checked (2026-08-06)
# and found badly uncalibrated -- it represented anywhere from ~1 to ~5.5 lengthscales depending
# on the difficulty parameter (delta/A), meaning "how much boundary buffer" was inconsistent
# across a sweep for no principled reason. K=5 plays the same role a "5-sigma" truncation plays
# for a locally-Gaussian process: negligible boundary-contamination probability near the wells,
# without wastefully inflating the domain.
BOUNDS_MARGIN_LENGTHSCALES = 5


def _lengthscale_margin(curvature, K=BOUNDS_MARGIN_LENGTHSCALES):
    """K/sqrt(curvature) -- see BOUNDS_MARGIN_LENGTHSCALES. `K` is overridable per-call (default
    unchanged) so a caller can experiment with a more generous margin without touching the shared
    constant every other well-type potential in this file depends on."""
    return K / np.sqrt(curvature)


def asymmetric_double_well(delta, tilt):
    """1D double well with a linear tilt term breaking left/right symmetry, so the two minima
    have genuinely different depths -- the main benchmark family, giving every algorithm a
    genuinely suboptimal starting basin to escape from (a symmetric double well has two
    equally-good minima, so there is nothing suboptimal to escape). See Theorem 3.1 in
    `paper/SoftMin_revised_theory.tex`, which analyzes escape from a starting basin `z_-` into an
    adjacent basin `z_+` with relative height rho_+ = f(z_+) - f(z_-) >= 0.
    """
    x = symbols('x')
    f = x**4 - delta * x**2 + tilt * x
    df = diff(f, x)

    # f'(x) = 4x^3 - 2*delta*x + tilt; without a closed-form factorization (tilt=0 would give
    # roots 0, +-sqrt(delta/2)), this needs the general cubic solver -- handled numerically by
    # polyroots.
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

    # Guaranteed by the branch above, but assert explicitly: this is the exact defect the switch
    # from a symmetric double well to this asymmetric one was meant to fix (see the docstring),
    # so a future edit that reintroduces symmetric parameters should fail loudly here, not produce
    # a benchmark that silently has nothing suboptimal to escape from.
    assert well(np.array([z_target])) < well(np.array([z_start])), (
        f"asymmetric_double_well(delta={delta}, tilt={tilt}): target well is not strictly "
        f"deeper than the start well."
    )
    depth_gap = float(well(np.array([z_start])) - well(np.array([z_target])))

    # f''(x) = 12x^2 - 2*delta (the tilt term is linear, so it drops out of the second
    # derivative) -- bounds margin scales with the true local curvature at each well, not a fixed
    # distance; see BOUNDS_MARGIN_LENGTHSCALES.
    f_pp_left = 12 * left**2 - 2 * delta
    f_pp_right = 12 * right**2 - 2 * delta

    return {
        'f_vec': well_vec,
        'f_grad_vec': well_grad_vec,
        'particle_init': [[z_start - 0.2, z_start + 0.2]],
        'set': [[z_target - 0.2, z_target + 0.2]],
        'bounds': [[left - _lengthscale_margin(f_pp_left), right + _lengthscale_margin(f_pp_right)]],
        'critical_points': [left, saddle, right],
        'z_start': z_start,
        'saddle': saddle,
        'z_target': z_target,
        'depth_gap': depth_gap,
    }


def unbalanced_quadruple_well_setting(A, delta):
    """2D quartic potential f(x,y) = x^4+y^4-A(x^2+y^2)-delta*x*y.

    `f` is symmetric under the x<->y swap (xy=yx), so its non-trivial stationary points split
    into two families, each found in closed form by substituting y=+-x into grad f = 0:
        diagonal      (x=y=+-r_diag),   r_diag = 0.5*sqrt(2A+delta)
        anti-diagonal (x=-y=+-r_anti),  r_anti = 0.5*sqrt(2A-delta)
    Each family's own two points ARE exact mirror images of each other (f(-x,-y)=f(x,y) always)
    -- but the diagonal family and the anti-diagonal family sit at genuinely different depths
    whenever delta != 0: f(r_diag,r_diag) - f(r_anti,-r_anti) = -A*delta (verified numerically).

    A previous version placed BOTH `particle_init` and `set` on the anti-diagonal family (using
    r_diag's formula but anti-diagonal coordinates), making them an exact mirror pair with zero
    depth gap -- the 2D analogue of the defect `asymmetric_double_well`'s tilt term fixes for the
    1D benchmark. Fixed here the same way: `particle_init` starts in the shallower diagonal well,
    `set` targets the deeper anti-diagonal one, so every A has a genuinely suboptimal basin to
    escape from. (The quartic system also has 4 additional real saddle points off both symmetry
    axes, not used here -- `depth_gap`, not a saddle-based barrier, is this benchmark's difficulty
    axis, matching how `asymmetric_double_well`'s notebook figure already uses a coordinate-space
    proxy rather than a rigorous barrier height.)
    """
    x, y = symbols('x y')
    f = x**4 + y**4 - A*(x**2 + y**2) - delta*x*y
    f_x = diff(f, x)
    f_y = diff(f, y)
    f_xx = diff(f_x, x)
    f_xy = diff(f_x, y)

    f_num = lambdify((x, y), f, modules='numpy')
    f_x_num = lambdify((x, y), f_x, modules='numpy')
    f_y_num = lambdify((x, y), f_y, modules='numpy')
    f_xx_num = lambdify((x, y), f_xx, modules='numpy')
    f_xy_num = lambdify((x, y), f_xy, modules='numpy')

    def quadruple_well(x):
        return f_num(x[0], x[1])

    def quadruple_well_grad(x):
        return np.array([f_x_num(x[0], x[1]), f_y_num(x[0], x[1])])

    quadruple_well_vec = np.vectorize(quadruple_well, signature='(n)->()')
    quadruple_well_grad_vec = np.vectorize(quadruple_well_grad, signature='(n)->(n)')

    r_diag_sq = 2 * A + delta
    r_anti_sq = 2 * A - delta
    if r_diag_sq <= 0 or r_anti_sq <= 0:
        raise ValueError(
            f"unbalanced_quadruple_well_setting(A={A}, delta={delta}): both critical-point "
            f"families require 2A+delta>0 and 2A-delta>0 (got {r_diag_sq}, {r_anti_sq})."
        )
    r_diag = 0.5 * np.sqrt(r_diag_sq)
    r_anti = 0.5 * np.sqrt(r_anti_sq)

    # Curvature check at BOTH wells (not just the one bounds/targets used to derive from before):
    # particles now genuinely need to start in one and find the other, so both must be real local
    # minima. At each point the Hessian is [[f_xx, f_xy],[f_xy, f_xx]] by the function's own
    # x<->y symmetry there, so its eigenvalues are f_xx +- |f_xy| and the softest is f_xx-|f_xy|.
    # Degenerate at A=1 for the diagonal well when delta=-1 (softest eigenvalue 4A-4 hits zero
    # there) -- the sweep's own first point.
    eig_soft_diag = f_xx_num(r_diag, r_diag) - abs(f_xy_num(r_diag, r_diag))
    eig_soft_anti = f_xx_num(r_anti, -r_anti) - abs(f_xy_num(r_anti, -r_anti))
    if eig_soft_diag <= 0 or eig_soft_anti <= 0:
        raise ValueError(
            f"unbalanced_quadruple_well_setting(A={A}, delta={delta}) is degenerate: softest "
            f"Hessian eigenvalue is {eig_soft_diag} at the diagonal well / {eig_soft_anti} at "
            f"the anti-diagonal well (need both > 0); increase A."
        )
    margin = _lengthscale_margin(min(eig_soft_diag, eig_soft_anti))
    outer = max(r_diag, r_anti) + margin

    depth_gap = float(f_num(r_diag, r_diag) - f_num(r_anti, -r_anti))
    if depth_gap <= 0:
        raise ValueError(
            f"unbalanced_quadruple_well_setting(A={A}, delta={delta}): diagonal well is not "
            f"shallower than the anti-diagonal well (depth_gap={depth_gap} <= 0) -- 'set' would "
            f"not be a genuinely suboptimal-basin escape target. Check the sign of delta."
        )

    return {
        'f_vec': quadruple_well_vec,
        'f_grad_vec': quadruple_well_grad_vec,
        'particle_init': [[r_diag - 0.2, r_diag + 0.2], [r_diag - 0.2, r_diag + 0.2]],
        'set': [[r_anti - 0.2, r_anti + 0.2], [-r_anti - 0.2, -r_anti + 0.2]],
        'bounds': [[-outer, outer], [-outer, outer]],
        'r_diag': r_diag,
        'r_anti': r_anti,
        'depth_gap': depth_gap,
    }


def _cubic_double_well_roots(A, tilt):
    """3 real critical points of g(t)=t^4-A*t^2+tilt*t, depth-sorted as (z_lo, saddle, z_hi) with
    z_lo the deeper minimum. Deliberately duplicates asymmetric_double_well's inline root-finding
    (P.polyroots + depth sort) rather than refactoring it into a shared helper -- that function
    backs existing production numbers and must not be put at risk by this change. Returns None if
    the cubic doesn't have 3 real roots."""
    roots = P.polyroots([tilt, -2 * A, 0, 4])
    real_roots = np.sort(roots[np.abs(roots.imag) < 1e-8].real)
    if len(real_roots) != 3:
        return None
    left, saddle, right = real_roots
    g_left = left**4 - A * left**2 + tilt * left
    g_right = right**4 - A * right**2 + tilt * right
    return (left, saddle, right) if g_left <= g_right else (right, saddle, left)


def _hybrid_wells(A, tilt, kappa):
    """Locate the 4 critical points of f(x,y) = g(x)+g(y)+kappa*x*y, g(t)=t^4-A*t^2+tilt*t, as
    candidate minima.

    The diagonal pair ('start'/'target') stays exactly closed-form: substituting y=x collapses
    both component gradient equations to the same cubic as `_cubic_double_well_roots`, just with
    `A_eff = A - kappa/2` in place of `A` (verified against an independent fsolve to ~1e-15
    during design). The off-diagonal ('neighbor') pair loses its exact y=-x closed form once both
    `tilt!=0` and `kappa!=0` (the two component equations reduce to `4x^3-(2A+kappa)x=-tilt` and
    `=+tilt`, inconsistent unless tilt=0), so it needs one `fsolve` refinement, seeded at the
    kappa=0 anti-diagonal location. The second neighbor is `(x,y)->(y,x)` of the first, not
    independently solved -- by the potential's exact x<->y swap symmetry this guarantees the two
    neighbor wells are exactly degenerate, regardless of fsolve's numerical precision.

    Returns None if a required cubic lacks 3 real roots or fsolve doesn't converge; otherwise
    `(wells, f_vals, eigs)`, each a dict keyed by 'start'/'neighbor1'/'neighbor2'/'target'.
    """
    base_roots = _cubic_double_well_roots(A, tilt)
    if base_roots is None:
        return None
    z_lo0, _, z_hi0 = base_roots

    eff_roots = _cubic_double_well_roots(A - kappa / 2, tilt)
    if eff_roots is None:
        return None
    z_lo_eff, _, z_hi_eff = eff_roots
    start, target = np.array([z_hi_eff, z_hi_eff]), np.array([z_lo_eff, z_lo_eff])

    def grad(xy):
        x, y = xy
        return np.array([4 * x**3 - 2 * A * x + tilt + kappa * y,
                          4 * y**3 - 2 * A * y + tilt + kappa * x])

    n1, _, ier, _ = fsolve(grad, [z_hi0, z_lo0], full_output=True)
    if ier != 1 or np.linalg.norm(grad(n1)) > 1e-8:
        return None
    n2 = n1[::-1]  # exact swap-mirror, not re-solved -- keeps the two neighbors exactly degenerate

    def f(xy):
        x, y = xy
        return x**4 - A * x**2 + tilt * x + y**4 - A * y**2 + tilt * y + kappa * x * y

    def eigs(xy):
        x, y = xy
        # f_xy = kappa everywhere (constant, from the bilinear coupling term).
        return np.linalg.eigvalsh([[12 * x**2 - 2 * A, kappa], [kappa, 12 * y**2 - 2 * A]])

    wells = {'start': start, 'neighbor1': n1, 'neighbor2': n2, 'target': target}
    f_vals = {name: float(f(w)) for name, w in wells.items()}
    well_eigs = {name: eigs(w) for name, w in wells.items()}
    return wells, f_vals, well_eigs


def _hybrid_ordering_ok(A, tilt, kappa):
    """True iff `kappa` keeps all 4 wells genuine local minima (positive-definite Hessian) with
    the intended strict 3-level ordering f(start) > f(neighbor) > f(target)."""
    result = _hybrid_wells(A, tilt, kappa)
    if result is None:
        return False
    _, f_vals, well_eigs = result
    if any(np.min(e) <= 0 for e in well_eigs.values()):
        return False
    return f_vals['start'] > f_vals['neighbor1'] > f_vals['target']


def _hybrid_kappa_ceiling(A, tilt, kappa_hi=None, tol=1e-6):
    """Largest `kappa` (by bisection) for which `_hybrid_ordering_ok` holds. Assumes a single
    ordering-preserving-to-broken threshold as kappa increases from 0 (true at every A tested
    during design, A=2..30, not proven analytically). `kappa_hi` seeds the search bracket (default
    4*A, generous relative to the ceilings actually observed -- measured ~0.05 at A=2 down to
    ~0.01 at A=30 for tilt=0.05, i.e. 2-3 orders of magnitude below this bracket)."""
    if kappa_hi is None:
        kappa_hi = 4 * A
    if not _hybrid_ordering_ok(A, tilt, 0.0):
        return 0.0
    lo, hi = 0.0, kappa_hi
    if _hybrid_ordering_ok(A, tilt, hi):
        return hi  # search bound itself never broke ordering -- return it as-is
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _hybrid_ordering_ok(A, tilt, mid):
            lo = mid
        else:
            hi = mid
    return lo


def hybrid_quadruple_well_setting(A, tilt, kappa_frac=0.7, kappa=None, bounds_margin_lengthscales=None):
    """2D quartic potential f(x,y) = g(x) + g(y) + kappa*x*y, g(t) = t^4 - A*t^2 + tilt*t -- the
    same cubic `asymmetric_double_well` uses, applied to both axes (a separable base), plus a
    small bilinear coupling term breaking exact separability.

    Unlike `unbalanced_quadruple_well_setting` (whose 4 minima form two exactly mirror-degenerate
    pairs under central symmetry -- only 2 distinct depths), this gives exactly 3 levels: 'start'
    (worst), two 'neighbor' wells (equal to each other, suboptimal), and 'target' diagonally
    opposite start (best). This falls out of the construction directly: x<->y swap symmetry
    (preserved by both g(x)+g(y) and kappa*x*y, for any kappa) guarantees the two neighbor wells
    are exactly degenerate, while `tilt` breaking central symmetry (x,y)->(-x,-y) separates start
    from target. See arXiv:1704.08927 (Zhang & Schuette, "Transition manifolds of complex
    metastable systems"), which builds a 2D multi-well transition-time benchmark the same way
    (separable double-well sum + a small coupling term) for the same reason: a purely separable
    2D double well gives a trivially 1D transition structure.

    `kappa` cannot be chosen independently of `A`/`tilt` -- the 3-level ordering breaks once
    |kappa| exceeds a per-(A,tilt) ceiling (a neighbor well crosses the start or target level).
    That ceiling shrinks in *absolute* terms as A grows if `tilt` is held fixed, because the 1D
    depth gap only grows like sqrt(A) while curvature grows like A -- measured: the
    coupling-to-curvature ratio decays ~57x from A=2 to A=30 at tilt=0.05. Callers sweeping A
    should instead scale `tilt` with `A` (e.g. `tilt = TILT_COEF * A**1.5`, calibrated so
    `tilt(2)` matches whatever fixed value would otherwise be used) -- this keeps that ratio
    exactly constant across the sweep instead of decaying (measured flat at ~0.63% for A=2..30 at
    kappa_frac=1.0). This function does not compute that scaling itself, since A is swept
    externally by the caller.

    Args:
        A: well-depth parameter, plays the same role as `asymmetric_double_well`'s `delta`.
        tilt: linear symmetry-breaking coefficient, same role as `asymmetric_double_well`'s
            `tilt`. Callers sweeping A should scale this with A (see above); this function takes
            whatever value it's given.
        kappa_frac: fraction of the true ordering-preserving `kappa` ceiling (found by bisection
            against the actual ordering/positive-definiteness constraints, not a closed-form
            proxy) to use. Ignored if `kappa` is given explicitly.
        kappa: explicit coupling strength, bypassing `kappa_frac` entirely (e.g. `kappa=0` for a
            purely separable baseline).
        bounds_margin_lengthscales: override for `BOUNDS_MARGIN_LENGTHSCALES`, to experiment with
            a larger domain margin without touching the shared constant every other well-type
            potential in this file depends on.

    Returns a dict matching every other `*_setting` factory in this file (`f_vec`, `f_grad_vec`,
    `particle_init`, `set`, `bounds`, `depth_gap`), plus `wells` (the 4 well coordinates),
    `depth_gap_start_neighbor`, `depth_gap_neighbor_target`, `kappa`, `A_eff`.
    """
    if kappa is None:
        ceiling = _hybrid_kappa_ceiling(A, tilt)
        if ceiling <= 0:
            raise ValueError(
                f"hybrid_quadruple_well_setting(A={A}, tilt={tilt}): even kappa=0 fails to "
                f"produce 4 valid, correctly-ordered wells -- check A/tilt are individually valid "
                f"for the underlying cubic (see asymmetric_double_well) first."
            )
        kappa = kappa_frac * ceiling

    result = _hybrid_wells(A, tilt, kappa)
    if result is None:
        raise ValueError(
            f"hybrid_quadruple_well_setting(A={A}, tilt={tilt}, kappa={kappa}): failed to locate "
            f"all 4 wells (a required cubic had no 3 real roots, or the neighbor-well solve did "
            f"not converge)."
        )
    wells, f_vals, well_eigs = result
    if any(np.min(e) <= 0 for e in well_eigs.values()):
        bad = {name: e for name, e in well_eigs.items() if np.min(e) <= 0}
        raise ValueError(
            f"hybrid_quadruple_well_setting(A={A}, tilt={tilt}, kappa={kappa}) is degenerate: "
            f"non-positive Hessian eigenvalues at {bad}."
        )
    if not (f_vals['start'] > f_vals['neighbor1'] > f_vals['target']):
        raise ValueError(
            f"hybrid_quadruple_well_setting(A={A}, tilt={tilt}, kappa={kappa}): 3-level ordering "
            f"violated ({f_vals}) -- reduce kappa_frac / |kappa|."
        )
    assert abs(f_vals['neighbor1'] - f_vals['neighbor2']) < 1e-6, (
        f"hybrid_quadruple_well_setting(A={A}, tilt={tilt}, kappa={kappa}): neighbor wells are "
        f"not degenerate ({f_vals['neighbor1']} vs {f_vals['neighbor2']}) -- fsolve likely failed "
        f"to converge tightly enough."
    )

    x, y = symbols('x y')
    f_sym = (x**4 - A * x**2 + tilt * x) + (y**4 - A * y**2 + tilt * y) + kappa * x * y
    fx_sym, fy_sym = diff(f_sym, x), diff(f_sym, y)
    f_num = lambdify((x, y), f_sym, modules='numpy')
    fx_num = lambdify((x, y), fx_sym, modules='numpy')
    fy_num = lambdify((x, y), fy_sym, modules='numpy')

    def quadruple_well(pt):
        return f_num(pt[0], pt[1])

    def quadruple_well_grad(pt):
        return np.array([fx_num(pt[0], pt[1]), fy_num(pt[0], pt[1])])

    f_vec = np.vectorize(quadruple_well, signature='(n)->()')
    f_grad_vec = np.vectorize(quadruple_well_grad, signature='(n)->(n)')

    K = BOUNDS_MARGIN_LENGTHSCALES if bounds_margin_lengthscales is None else bounds_margin_lengthscales
    margins = {name: _lengthscale_margin(np.min(well_eigs[name]), K=K) for name in wells}
    x_lo = min(wells[name][0] - margins[name] for name in wells)
    x_hi = max(wells[name][0] + margins[name] for name in wells)
    y_lo = min(wells[name][1] - margins[name] for name in wells)
    y_hi = max(wells[name][1] + margins[name] for name in wells)

    start, target = wells['start'], wells['target']
    return {
        'f_vec': f_vec,
        'f_grad_vec': f_grad_vec,
        'particle_init': [[start[0] - 0.2, start[0] + 0.2], [start[1] - 0.2, start[1] + 0.2]],
        'set': [[target[0] - 0.2, target[0] + 0.2], [target[1] - 0.2, target[1] + 0.2]],
        'bounds': [[x_lo, x_hi], [y_lo, y_hi]],
        'wells': {name: tuple(float(c) for c in w) for name, w in wells.items()},
        'depth_gap': f_vals['start'] - f_vals['target'],
        'depth_gap_start_neighbor': f_vals['start'] - f_vals['neighbor1'],
        'depth_gap_neighbor_target': f_vals['neighbor1'] - f_vals['target'],
        'kappa': kappa,
        'A_eff': A - kappa / 2,
    }


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

    # bounds=[-10,10] is not an arbitrary heuristic: the cosine term's local-minima lattice has
    # period 2*pi/c = 1 (at the default c=2*pi), so [-10,10] spans ~20 periods per dimension --
    # deliberately compact (matches the paper's own framing of Ackley as "a compact domain
    # containing an exponential number of local minima"), with dimension `d` -- not domain size --
    # driving the actual difficulty scaling as this setting is swept to higher d.
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
