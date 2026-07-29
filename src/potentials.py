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
