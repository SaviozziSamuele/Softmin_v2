"""Vectorized particle-swarm simulation classes.

Base class `Simulation` plus seven concrete dynamics:
    SoftminEnergyAdagrad  - this paper's method (Softmin Energy coupling + Adagrad)
    GaussianDiffusion     - overdamped Langevin dynamics (baseline)
    SimulatedAnnealing    - Langevin dynamics with a temperature-cooling schedule (baseline)
    GradientAwareCBO      - Gradient-Aware Consensus-Based Optimization (baseline)
    GradientAwarePSO      - Gradient-Enhanced Particle Swarm Optimization (baseline)
    CBO                   - canonical, gradient-free Consensus-Based Optimization (baseline)
    PSO                   - canonical, gradient-free Particle Swarm Optimization (baseline)
"""
import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Literal, Optional, Union

import numpy as np

import viz

N_CPU = mp.cpu_count()

# Configure production-grade logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class Simulation(ABC):
    """
    Abstract base class for a vectorized particle swarm simulation environment.
    """
    def __init__(
        self,
        n: int,
        d: int,
        f: Callable[[np.ndarray], np.ndarray],
        df: Callable[[np.ndarray], np.ndarray],
        bounds: Optional[List[List[float]]] = None,
        particle_init: Optional[List[List[float]]] = None,
        learning_rate: float = 0.01,
        name: str = ""
    ):
        self.name = name

        if n <= 0 or d <= 0:
            raise ValueError("Number of particles (n) and dimensions (d) must be strictly positive.")

        self.n = n
        self.d = d
        self.learning_rate = learning_rate

        # Safely handle mutable defaults
        if bounds is None:
            bounds = [[-np.inf, np.inf]] * d
        elif len(bounds) == 1:
            bounds = bounds * d

        if len(bounds) != d or not all(len(b) == 2 and b[0] < b[1] for b in bounds):
            raise ValueError("Bounds must be a list of [lower, upper] pairs matching the dimension 'd'.")

        self.f = f
        self.df = df

        # Test vectorization dynamically
        test_vector = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (n, d))
        try:
            f_test, df_test = self.f(test_vector), self.df(test_vector)
            if f_test.shape != (n,) and f_test.shape != (n, 1):
                raise ValueError(f"f(x) must return shape (n,), got {f_test.shape}")
            if df_test.shape != (n, d):
                raise ValueError(f"df(x) must return shape (n, d), got {df_test.shape}")
        except Exception as e:
            raise RuntimeError(f"Function/Gradient vectorization test failed: {e}")

        if particle_init is None:
            particle_init = bounds
        elif len(particle_init) == 1:
            particle_init = particle_init * d

        self.bounds = bounds
        self.particle_init = particle_init

        # Initialize swarm
        self.particles = np.random.uniform(
            [p[0] for p in particle_init],
            [p[1] for p in particle_init],
            (n, d)
        )
        self.values = self.f(self.particles)

        self.best_positions = self.particles.copy()
        self.best_values = self.values.copy()

        best_idx = np.argmin(self.best_values)
        self.best_global_position = self.best_positions[best_idx].copy()
        self.best_global_value = self.best_values[best_idx]

        self.history = [self.particles.copy()]
        self.n_steps = 0

        # Optional attributes some subclasses populate; kept as None here so shared
        # plotting code (viz.plot_simulation) can branch on them safely regardless of
        # which subclass is being plotted.
        self.noise = None
        self.history_eq_values = None

        # Computational-cost instrumentation: evaluation counts (the fair cost currency across
        # gradient-based and gradient-free methods, per the paper's cost-matched-comparison
        # requirement) plus a secondary, implementation-level wall-clock signal. Only step()-time
        # calls are counted, not the one-off vectorization check above or reset()'s initial
        # evaluation, since those are a fixed, identical overhead across all methods.
        self.n_gradient_evals = 0
        self.n_objective_evals = 0
        self.wall_clock_seconds = 0.0

        logger.info(f"Initialized Simulation '{self.name}' with {self.n} particles in {self.d}D space.")

    def _eval_grad(self, positions: np.ndarray) -> np.ndarray:
        """Evaluates df(positions) and counts the call. Subclasses should call this instead of
        `self.df(...)` directly from within `step()`, so gradient-free methods (which never call
        it) naturally report zero gradient evaluations."""
        self.n_gradient_evals += 1
        return self.df(positions)

    def _eval_obj(self, positions: np.ndarray) -> np.ndarray:
        """Evaluates f(positions) and counts the call. Subclasses should call this instead of
        `self.f(...)` directly from within `step()`."""
        self.n_objective_evals += 1
        return self.f(positions)

    @abstractmethod
    def step(self) -> None:
        """Performs a single simulation iteration."""
        pass

    def run(self, n_steps: int) -> None:
        """Executes the simulation loop."""
        logger.info(f"Running simulation for {n_steps} steps...")
        start_time = time.perf_counter()
        for i in range(n_steps):
            self.n_steps = i + 1
            self.step()
            self.history.append(self.particles.copy())
        self.wall_clock_seconds += time.perf_counter() - start_time
        logger.info("Simulation completed.")

    def first_entry_time(self, target_set: List[List[float]], max_steps: int = 10_000) -> float:
        """
        Computes the first entry time into a target region using fully vectorized checks.
        """
        if len(target_set) != self.d or not all(len(s) == 2 and s[0] < s[1] for s in target_set):
            raise ValueError("target_set constraints must match dimensions and be strictly formatted as [[min, max], ...]")

        # Pre-allocate numpy arrays for rapid broadcast comparisons
        lower_bounds = np.array([s[0] for s in target_set])
        upper_bounds = np.array([s[1] for s in target_set])

        start_time = time.perf_counter()
        for i in range(max_steps):
            self.step()
            # Vectorized logical AND across all bounds, then checking if ANY particle satisfies ALL dimension bounds
            in_bounds = np.all((self.particles > lower_bounds) & (self.particles < upper_bounds), axis=1)

            if np.any(in_bounds):
                logger.info(f"Entry set reached at step {i}.")
                self.wall_clock_seconds += time.perf_counter() - start_time
                return i * self.learning_rate
            if i % 1000 == 0:
                # If the max standard deviation across all spatial dimensions is functionally zero
                if np.max(np.std(self.particles, axis=0)) < 1e-6:
                    # The swarm is permanently trapped. Abort the simulation to save CPU time.
                    self.wall_clock_seconds += time.perf_counter() - start_time
                    return max_steps * self.learning_rate

        self.wall_clock_seconds += time.perf_counter() - start_time
        logger.warning(f"Maximum steps ({max_steps}) reached without entering target set.")
        return max_steps * self.learning_rate

    def reset(self, particle_init: Optional[Union[List, np.ndarray]] = None) -> None:
        """Resets the environment."""
        if isinstance(particle_init, list):
            self.particles = np.random.uniform(
                [p[0] for p in particle_init],
                [p[1] for p in particle_init],
                (self.n, self.d)
            )
        elif isinstance(particle_init, np.ndarray):
            if particle_init.shape != (self.n, self.d):
                raise ValueError("Numpy array initialization must match shape (n, d).")
            self.particles = particle_init.copy()
        else:
            self.particles = np.random.uniform(
                [p[0] for p in self.particle_init],
                [p[1] for p in self.particle_init],
                (self.n, self.d)
            )

        self.values = self.f(self.particles)
        self.best_positions = self.particles.copy()
        self.best_values = self.values.copy()
        self.best_global_value = np.min(self.best_values)
        self.history = [self.particles.copy()]
        self.best_global_position = self.best_positions[np.argmin(self.best_values)].copy()
        self.n_steps = 0

    def plot(self, path="", make_gif=False, **kwargs):
        viz.plot_simulation(self, path=path, make_gif=make_gif, **kwargs)


class SoftminEnergyAdagrad(Simulation):
    """
    Vectorized simulation of n particles optimizing a joint Softmin energy J.
    The learning rate is adapted element-wise using the Adagrad algorithm.
    """
    def __init__(
        self,
        n: int,
        d: int,
        f: Callable[[np.ndarray], np.ndarray],
        df: Callable[[np.ndarray], np.ndarray],
        bounds: Optional[List[List[float]]] = None,
        particle_init: Optional[List[List[float]]] = None,
        beta: float = 10.0,
        name: str = "",
        initial_lr: float = 1.0,
        noise_intensity: float = 0.2,
        beta_decay: Optional[float] = None,
        noise: Optional[str] = "normal",
        adaptive_lr: bool = True,
        track_prefactor_history: bool = False,
        prefactor_mode: Literal["signed", "positive_part", "absolute"] = "signed"
    ):
        # Resolve mutable defaults safely
        if bounds is None:
            bounds = [[-5.0, 5.0]] * d

        super().__init__(
            n=n,
            d=d,
            f=f,
            df=df,
            bounds=bounds,
            particle_init=particle_init,
            learning_rate=initial_lr,
            name=name
        )

        if beta <= 0:
            raise ValueError("Beta must be strictly positive.")

        self.beta = float(beta)
        self.beta_decay = beta_decay
        self.noise_intensity = noise_intensity

        if noise not in ["normal", "0-1", None]:
            raise ValueError("Noise must be 'normal', '0-1', or None.")
        self.noise = noise

        self.adaptive_lr = adaptive_lr
        self.alpha = initial_lr
        self.G = np.zeros_like(self.particles)

        if prefactor_mode not in ("signed", "positive_part", "absolute"):
            raise ValueError("prefactor_mode must be 'signed', 'positive_part', or 'absolute'.")
        self.prefactor_mode = prefactor_mode

        # Opt-in per-particle prefactor (A_{t,k}) history, for the ascent-descent diagnostic
        # plot. Left off by default: production sweeps run up to MAX_STEPS = 2e9 steps, and
        # storing a full per-step, per-particle history at that scale would be catastrophic.
        self.track_prefactor_history = track_prefactor_history
        self.prefactor_history = [] if track_prefactor_history else None

        # Initialize energy metrics safely
        initial_energy = self._compute_softmin_energy(self.values)
        self.Js = [initial_energy]
        self.history_eq_values = [1.0 / self.beta + initial_energy]

    def _compute_softmin_energy(self, values: np.ndarray) -> float:
        """
        Computes the Softmin energy J using numerical stabilization to prevent overflow.
        """
        min_val = np.min(values)
        # Shift values by the minimum for numerical stability
        exponentials = np.exp(-self.beta * (values - min_val))
        sum_exp = np.sum(exponentials)

        # Calculate weights and return the weighted sum of original values
        weights = exponentials / sum_exp
        return float(np.sum(weights * values))

    def _get_softmin_weights(self, values: np.ndarray) -> np.ndarray:
        """
        Returns the normalized Boltzmann-Gibbs weights.
        """
        min_val = np.min(values)
        exponentials = np.exp(-self.beta * (values - min_val))
        return exponentials / np.sum(exponentials)

    def lr_schedule(self, gradient: np.ndarray) -> np.ndarray:
        """Adagrad learning rate adaptation."""
        self.G += gradient**2
        return self.learning_rate / (np.sqrt(self.G) + np.finfo(float).eps)

    def generate_noise(self, variance: Union[float, np.ndarray]) -> np.ndarray:
        """Generates stochastic perturbations for the swarm."""
        if self.noise is None:
            return np.zeros_like(self.particles)

        base_noise = np.random.normal(0, variance, self.particles.shape)

        if self.noise == "normal":
            return base_noise

        if self.noise == "0-1":
            noise_mask = np.ones((self.n, 1))
            noise_mask[np.argmin(self.values)] = 0.0
            return noise_mask * base_noise

    def step(self) -> None:
        """Performs a single vectorized simulation update."""
        gradient = self._eval_grad(self.particles)
        energy = self.Js[-1]

        # Compute stabilized weights
        weights = self._get_softmin_weights(self.values)

        # Vectorized velocity computation
        # weights * self.n replaces (n * exponentials / sum(exponentials))
        # velocity_scale = n * A_{t,k} (the paper's per-particle effective learning rate)
        velocity_scale = (1.0 - self.beta * (self.values - energy)) * (self.n * weights)

        if self.track_prefactor_history:
            self.prefactor_history.append(velocity_scale / self.n)

        # Mechanism ablation: which transform of the raw signed prefactor actually drives the
        # particles. "signed" is the paper's mechanism (ascending particles reverse direction);
        # "positive_part" freezes ascending particles instead of letting them climb; "absolute"
        # forces every particle to always descend, scaled by the prefactor's magnitude. Only the
        # applied drift changes -- the Softmin energy bookkeeping above always uses the true
        # signed values, so diagnostics stay meaningful across all three variants.
        if self.prefactor_mode == "signed":
            applied_scale = velocity_scale
        elif self.prefactor_mode == "positive_part":
            applied_scale = np.maximum(velocity_scale, 0.0)
        else:  # "absolute"
            applied_scale = np.abs(velocity_scale)

        velocities = applied_scale[:, np.newaxis] * gradient

        if self.adaptive_lr:
            self.alpha = self.lr_schedule(gradient)

        # Update positions
        noise_variance = np.sqrt(self.alpha) if isinstance(self.alpha, np.ndarray) else np.sqrt(float(self.alpha))
        stochastic_term = np.sqrt(2.0 * self.noise_intensity) * self.generate_noise(variance=noise_variance)

        self.particles = self.particles - (self.alpha * velocities) + stochastic_term

        # Vectorized clipping against bounds
        lower_bounds = [b[0] for b in self.bounds]
        upper_bounds = [b[1] for b in self.bounds]
        self.particles = np.clip(self.particles, lower_bounds, upper_bounds)

        # Single function evaluation per step
        self.values = self._eval_obj(self.particles)

        # Update metrics
        new_energy = self._compute_softmin_energy(self.values)
        self.Js.append(new_energy)

        # Track best configurations
        improved = self.values < self.best_values
        self.best_values[improved] = self.values[improved]
        self.best_positions[improved] = self.particles[improved]

        current_min_idx = np.argmin(self.values)
        if self.values[current_min_idx] < self.best_global_value:
            self.best_global_value = self.values[current_min_idx]
            self.best_global_position = self.particles[current_min_idx].copy()
        if self.beta_decay is not None:
            self.beta *= self.beta_decay

    def run(self, n_steps: int) -> None:
        """Executes the simulation loop."""
        for _ in range(n_steps):
            self.step()
            self.history.append(self.particles.copy())
            self.history_eq_values.append(1.0 / self.beta + self.Js[-1])

        self.n_steps = n_steps

    def reset(self, particle_init: Optional[Union[List, np.ndarray]] = None) -> None:
        """Resets the environment and historical metrics."""
        super().reset(particle_init)

        # Reset Adagrad accumulators and learning rate
        self.G = np.zeros_like(self.particles)
        self.alpha = self.learning_rate

        initial_energy = self._compute_softmin_energy(self.best_values)
        self.Js = [initial_energy]
        self.history_eq_values = [1.0 / self.beta + initial_energy]
        self.prefactor_history = [] if self.track_prefactor_history else None


class GaussianDiffusion(Simulation):
    """
    Vectorized simulation of independent particles diffusing in a potential,
    governed by Overdamped Langevin Dynamics (Euler-Maruyama discretization).
    """
    def __init__(
        self,
        n: int,
        d: int,
        f: Callable[[np.ndarray], np.ndarray],
        df: Callable[[np.ndarray], np.ndarray],
        bounds: Optional[List[List[float]]] = None,
        particle_init: Optional[List[List[float]]] = None,
        noise: float = 0.1,
        learning_rate: float = 0.1,
        name: str = ""
    ):
        # Safely handle default bounds
        if bounds is None:
            bounds = [[-np.inf, np.inf]] * d

        super().__init__(
            n=n,
            d=d,
            f=f,
            df=df,
            bounds=bounds,
            particle_init=particle_init,
            learning_rate=learning_rate,
            name=name
        )

        if noise <= 0:
            raise ValueError("Noise parameter (diffusion coefficient) must be strictly positive.")
        self.noise = noise

        # PRE-ALLOCATION: Cache bounds as numpy arrays to eliminate list comprehensions in the step loop
        self._lower_bounds = np.array([b[0] for b in self.bounds])
        self._upper_bounds = np.array([b[1] for b in self.bounds])

        # PRE-COMPUTATION: Cache the invariant square root multiplier for the stochastic term
        self._noise_scale = np.sqrt(2.0 * self.noise)

    def step(self) -> None:
        """Executes a single vectorized Euler-Maruyama integration step."""
        gradient = self._eval_grad(self.particles)

        # Generate the Wiener process increment
        stochastic_term = self._noise_scale * np.random.normal(
            loc=0.0,
            scale=np.sqrt(self.learning_rate),
            size=self.particles.shape
        )

        # Update positions: X_{t+1} = X_t - eta * grad(X_t) + sqrt(2D) * dW_t
        self.particles = self.particles - (self.learning_rate * gradient) + stochastic_term

        # Fast vectorized clipping using pre-allocated bound arrays
        self.particles = np.clip(self.particles, self._lower_bounds, self._upper_bounds)

        # Evaluate objective function
        self.values = self._eval_obj(self.particles)

        # Update local bests for each particle
        improved = self.values < self.best_values
        self.best_values[improved] = self.values[improved]
        self.best_positions[improved] = self.particles[improved]

        # Update global best efficiently (single array reduction)
        current_min_idx = np.argmin(self.values)
        if self.values[current_min_idx] < self.best_global_value:
            self.best_global_value = self.values[current_min_idx]
            self.best_global_position = self.particles[current_min_idx].copy()


class SimulatedAnnealing(Simulation):
    """
    Vectorized simulation of independent particles driven by Simulated Annealing.
    Utilizes stable Langevin dynamics by scaling the diffusion term by temperature.
    """
    def __init__(
        self,
        n: int,
        d: int,
        f: Callable[[np.ndarray], np.ndarray],
        df: Callable[[np.ndarray], np.ndarray],
        bounds: Optional[List[List[float]]] = None,
        particle_init: Optional[List[List[float]]] = None,
        noise: float = 0.1,
        learning_rate: float = 0.1,
        initial_temperature: float = 1.0,
        cooling_rate: Optional[float] = None,
        cooling_schedule: str = "geometric",
        name: str = ""
    ):
        if bounds is None:
            bounds = [[-np.inf, np.inf]] * d

        super().__init__(
            n=n,
            d=d,
            f=f,
            df=df,
            bounds=bounds,
            particle_init=particle_init,
            learning_rate=learning_rate,
            name=name
        )

        if noise <= 0:
            raise ValueError("Noise parameter (diffusion constant) must be strictly positive.")
        if initial_temperature <= 0:
            raise ValueError("Initial temperature must be strictly positive.")

        implemented_cooling_schedules = {"linear", "geometric", "logarithmic", "power_law", None}
        if cooling_schedule not in implemented_cooling_schedules:
            raise ValueError(f"Cooling schedule must be one of {implemented_cooling_schedules}.")

        default_cooling_rates = {
            "linear": 0.01,
            "geometric": 0.99,
            "logarithmic": 0.1,
            "power_law": 0.5
        }

        if cooling_rate is None:
            cooling_rate = default_cooling_rates.get(cooling_schedule, 0.0)

        # Strict validation of cooling rates
        if cooling_schedule == "linear" and cooling_rate <= 0:
            raise ValueError("With a linear cooling schedule, the cooling rate must be positive.")
        elif cooling_schedule == "geometric" and not (0 < cooling_rate < 1):
            raise ValueError("With a geometric cooling schedule, the cooling rate must be in (0, 1).")
        elif cooling_schedule == "power_law" and cooling_rate <= 0:
            raise ValueError("With a power law cooling schedule, the cooling rate must be positive.")

        self.noise = noise
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.cooling_schedule = cooling_schedule

        # PRE-ALLOCATION: Cache bounds and base noise variance mapping
        self._lower_bounds = np.array([b[0] for b in self.bounds])
        self._upper_bounds = np.array([b[1] for b in self.bounds])

        # Base factor for diffusion std dev calculation: 2 * D * eta
        self._base_variance_factor = 2.0 * self.noise * self.learning_rate

    def temperature(self) -> float:
        """
        Computes the current thermodynamic temperature based on the selected schedule.
        """
        if self.cooling_schedule == "linear":
            return max(self.initial_temperature - self.cooling_rate * self.n_steps, 0.0)
        elif self.cooling_schedule == "geometric":
            return self.initial_temperature * (self.cooling_rate ** self.n_steps)
        elif self.cooling_schedule == "logarithmic":
            return self.initial_temperature / np.log(self.n_steps + np.e)
        elif self.cooling_schedule == "power_law":
            return self.initial_temperature / ((self.n_steps + 1) ** self.cooling_rate)
        else:
            return self.initial_temperature

    def step(self) -> None:
        """Executes a single step of Langevin dynamics with stable annealing."""
        gradient = self._eval_grad(self.particles)
        current_temp = self.temperature()

        # Standard unscaled gradient step: eta * grad(f)
        grad_step = self.learning_rate * gradient

        # Temperature-scaled diffusion: sqrt(2 * D * eta * T) * W_t
        # As T approaches 0, the standard deviation approaches 0 seamlessly.
        current_std_dev = np.sqrt(self._base_variance_factor * current_temp)
        noise_term = np.random.normal(
            loc=0.0,
            scale=current_std_dev,
            size=self.particles.shape
        )

        # Update positions
        self.particles = self.particles - grad_step + noise_term

        # Vectorized bounds clipping
        self.particles = np.clip(self.particles, self._lower_bounds, self._upper_bounds)

        # Single evaluation per step
        self.values = self._eval_obj(self.particles)

        # Local best memory updates
        improved = self.values < self.best_values
        self.best_values[improved] = self.values[improved]
        self.best_positions[improved] = self.particles[improved]

        # Global best update
        current_min_idx = np.argmin(self.values)
        if self.values[current_min_idx] < self.best_global_value:
            self.best_global_value = self.values[current_min_idx]
            self.best_global_position = self.particles[current_min_idx].copy()


class GradientAwareCBO(Simulation):
    """
    Vectorized simulation of Gradient-Aware Consensus-Based Optimization (CBO).
    Blends local gradient descent with global macroscopic consensus and
    variance-collapsing anisotropic diffusion.
    """
    def __init__(
        self,
        n: int,
        d: int,
        f: Callable[[np.ndarray], np.ndarray],
        df: Callable[[np.ndarray], np.ndarray],
        bounds: Optional[List[List[float]]] = None,
        particle_init: Optional[List[List[float]]] = None,
        learning_rate: float = 0.01,         # eta: Gradient step size / dt
        consensus_rate: float = 1.0,         # lambda: Attraction pull to the consensus
        noise_intensity: float = np.sqrt(2), # sigma: Diffusion scaling
        beta: float = 10.0,                  # beta: Temperature inverse for consensus
        beta_growth: Optional[float] = None, # optional annealing: beta *= beta_growth each step
        name: str = "GradientAwareCBO"
    ):
        if bounds is None:
            bounds = [[-np.inf, np.inf]] * d

        super().__init__(
            n=n,
            d=d,
            f=f,
            df=df,
            bounds=bounds,
            particle_init=particle_init,
            learning_rate=learning_rate,
            name=name
        )

        if consensus_rate < 0 or noise_intensity < 0 or beta <= 0:
            raise ValueError("Consensus rate, noise intensity, and beta must be strictly positive.")
        if beta_growth is not None and beta_growth <= 1:
            raise ValueError("beta_growth must be > 1 (CBO's temperature is annealed upward, "
                              "the opposite direction from Softmin's decaying beta).")

        self.consensus_rate = consensus_rate
        self.noise_intensity = noise_intensity
        self.beta = float(beta)
        self.beta_growth = beta_growth

        # PRE-ALLOCATION: Cache bounds and compute static scalars
        self._lower_bounds = np.array([b[0] for b in self.bounds])
        self._upper_bounds = np.array([b[1] for b in self.bounds])

        # dt is functionally equivalent to the learning rate in this discrete approximation
        self._sqrt_dt = np.sqrt(self.learning_rate)

    def _compute_consensus_point(self, positions: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Computes the empirical consensus point c_beta(X) using
        log-sum-exp stabilization to prevent overflow/underflow.
        """
        min_val = np.min(values)

        # Compute stabilized Boltzmann-Gibbs weights
        exponentials = np.exp(-self.beta * (values - min_val))
        weights = exponentials / np.sum(exponentials)

        # Weighted spatial mean (Broadcasting weights over spatial dimensions)
        consensus = np.sum(positions * weights[:, np.newaxis], axis=0)
        return consensus

    def step(self) -> None:
        """Executes a single Euler-Maruyama discretization step for Gradient-Aware CBO."""
        gradient = self._eval_grad(self.particles)

        # 1. Compute global consensus
        consensus_point = self._compute_consensus_point(self.particles, self.values)

        # 2. Compute spatial differences from consensus
        diff_to_consensus = self.particles - consensus_point

        # 3. Calculate components of the SDE
        # -> Gradient Exploitation
        grad_drift = self.learning_rate * gradient

        # -> Consensus Exploration/Drift
        consensus_drift = self.consensus_rate * diff_to_consensus * self.learning_rate

        # -> Variance-Collapsing Diffusion
        # Standard normal increments: dW
        dW = np.random.normal(loc=0.0, scale=1.0, size=self.particles.shape)
        # Element-wise anisotropic scaling (standard CBO diffusion, no ad hoc noise floor)
        diffusion_term = self.noise_intensity * np.abs(diff_to_consensus) * self._sqrt_dt * dW

        # 4. Update Swarm Positions
        self.particles = self.particles - grad_drift - consensus_drift + diffusion_term

        # 5. Fast vectorized clipping using memory views
        self.particles = np.clip(self.particles, self._lower_bounds, self._upper_bounds)

        # 6. Evaluate Objective exactly once
        self.values = self._eval_obj(self.particles)

        # 7. Update memory caches (Local bests)
        improved = self.values < self.best_values
        self.best_values[improved] = self.values[improved]
        self.best_positions[improved] = self.particles[improved]

        # 8. Update global best
        current_min_idx = np.argmin(self.values)
        if self.values[current_min_idx] < self.best_global_value:
            self.best_global_value = self.values[current_min_idx]
            self.best_global_position = self.particles[current_min_idx].copy()

        # 9. Optional temperature annealing (standard CBO practice anneals beta upward, sharpening
        # consensus over time -- the opposite direction from Softmin's decaying beta)
        if self.beta_growth is not None:
            self.beta *= self.beta_growth


class GradientAwarePSO(Simulation):
    """
    Vectorized simulation of Gradient-Enhanced Particle Swarm Optimization (GEPSO).
    Combines the zero-order global exploration of PSO with the first-order
    local exploitation of Gradient Descent.
    """
    def __init__(
        self,
        n: int,
        d: int,
        f: Callable[[np.ndarray], np.ndarray],
        df: Callable[[np.ndarray], np.ndarray],
        bounds: Optional[List[List[float]]] = None,
        particle_init: Optional[List[List[float]]] = None,
        learning_rate: float = 0.01,  # eta: Gradient step size
        w: float = 0.729,             # Inertia weight
        c1: float = 1.49445,          # Cognitive coefficient
        c2: float = 1.49445,          # Social coefficient
        name: str = "GradientAwarePSO"
    ):
        if bounds is None:
            bounds = [[-np.inf, np.inf]] * d

        super().__init__(
            n=n,
            d=d,
            f=f,
            df=df,
            bounds=bounds,
            particle_init=particle_init,
            learning_rate=learning_rate,
            name=name
        )

        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Initialize velocities to zero (or small random vectors)
        self.velocities = np.zeros_like(self.particles)

        # PRE-ALLOCATION: Cache bounds for vectorized clipping
        self._lower_bounds = np.array([b[0] for b in self.bounds])
        self._upper_bounds = np.array([b[1] for b in self.bounds])

    def step(self) -> None:
        """Executes a single step of the Gradient-Enhanced PSO algorithm."""
        # 1. Compute deterministic gradient drift
        gradient = self._eval_grad(self.particles)
        grad_drift = self.learning_rate * gradient

        # 2. Generate stochastic tensors for cognitive and social components
        r1 = np.random.uniform(0, 1, size=self.particles.shape)
        r2 = np.random.uniform(0, 1, size=self.particles.shape)

        # 3. Calculate velocity components
        inertia_term = self.w * self.velocities
        cognitive_term = self.c1 * r1 * (self.best_positions - self.particles)
        social_term = self.c2 * r2 * (self.best_global_position - self.particles)

        # 4. Update Velocities (Standard PSO + Gradient Drift)
        self.velocities = inertia_term + cognitive_term + social_term - grad_drift

        # 5. Update Positions (no ad hoc injected noise floor)
        self.particles = self.particles + self.velocities

        # 6. Fast vectorized clipping using memory views
        # Note: In strict PSO, velocities should also be zeroed out if a particle hits a boundary,
        # but for continuous potentials with gradient awareness, soft clipping is preferred.
        self.particles = np.clip(self.particles, self._lower_bounds, self._upper_bounds)

        # 7. Evaluate Objective strictly once per step
        self.values = self._eval_obj(self.particles)

        # 8. Update Memory Caches (Cognitive/Personal Bests)
        improved = self.values < self.best_values
        self.best_values[improved] = self.values[improved]
        self.best_positions[improved] = self.particles[improved]

        # 9. Update Global Knowledge (Social Best)
        current_min_idx = np.argmin(self.values)
        if self.values[current_min_idx] < self.best_global_value:
            self.best_global_value = self.values[current_min_idx]
            self.best_global_position = self.particles[current_min_idx].copy()

    def reset(self, particle_init: Optional[Union[List, np.ndarray]] = None) -> None:
        """Resets the environment, ensuring velocities are cleared."""
        super().reset(particle_init)
        self.velocities = np.zeros_like(self.particles)


class CBO(Simulation):
    """
    Vectorized simulation of canonical Consensus-Based Optimization (Carrillo et al.).
    Zeroth-order: particles are drawn toward a Gibbs-weighted consensus point with
    variance-collapsing anisotropic diffusion. No gradient information is used.
    """
    def __init__(
        self,
        n: int,
        d: int,
        f: Callable[[np.ndarray], np.ndarray],
        df: Callable[[np.ndarray], np.ndarray],
        bounds: Optional[List[List[float]]] = None,
        particle_init: Optional[List[List[float]]] = None,
        learning_rate: float = 0.01,          # eta: dt
        consensus_rate: float = 1.0,          # lambda: attraction pull to the consensus
        noise_intensity: float = 1.0,         # sigma: diffusion scaling (2*lambda > sigma**2 for stability)
        beta: float = 10.0,                   # beta: temperature inverse for consensus
        beta_growth: Optional[float] = None,  # optional annealing: beta *= beta_growth each step
        name: str = "CBO"
    ):
        if bounds is None:
            bounds = [[-np.inf, np.inf]] * d

        super().__init__(
            n=n,
            d=d,
            f=f,
            df=df,
            bounds=bounds,
            particle_init=particle_init,
            learning_rate=learning_rate,
            name=name
        )

        if consensus_rate < 0 or noise_intensity < 0 or beta <= 0:
            raise ValueError("Consensus rate, noise intensity, and beta must be strictly positive.")
        if beta_growth is not None and beta_growth <= 1:
            raise ValueError("beta_growth must be > 1 (CBO's temperature is annealed upward).")

        self.consensus_rate = consensus_rate
        self.noise_intensity = noise_intensity
        self.beta = float(beta)
        self.beta_growth = beta_growth

        # PRE-ALLOCATION: Cache bounds and compute static scalars
        self._lower_bounds = np.array([b[0] for b in self.bounds])
        self._upper_bounds = np.array([b[1] for b in self.bounds])
        self._sqrt_dt = np.sqrt(self.learning_rate)

    def _compute_consensus_point(self, positions: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Computes the empirical consensus point c_beta(X) using log-sum-exp stabilization."""
        min_val = np.min(values)
        exponentials = np.exp(-self.beta * (values - min_val))
        weights = exponentials / np.sum(exponentials)
        return np.sum(positions * weights[:, np.newaxis], axis=0)

    def step(self) -> None:
        """Executes a single Euler-Maruyama discretization step of standard CBO (no gradient
        term, no ad hoc noise floor)."""
        # 1. Compute global consensus
        consensus_point = self._compute_consensus_point(self.particles, self.values)

        # 2. Compute spatial differences from consensus
        diff_to_consensus = self.particles - consensus_point

        # 3. Consensus drift
        consensus_drift = self.consensus_rate * diff_to_consensus * self.learning_rate

        # 4. Variance-collapsing anisotropic diffusion
        dW = np.random.normal(loc=0.0, scale=1.0, size=self.particles.shape)
        diffusion_term = self.noise_intensity * np.abs(diff_to_consensus) * self._sqrt_dt * dW

        # 5. Update swarm positions
        self.particles = self.particles - consensus_drift + diffusion_term

        # 6. Fast vectorized clipping
        self.particles = np.clip(self.particles, self._lower_bounds, self._upper_bounds)

        # 7. Evaluate objective exactly once (no gradient evaluation at all)
        self.values = self._eval_obj(self.particles)

        # 8. Update memory caches (local bests)
        improved = self.values < self.best_values
        self.best_values[improved] = self.values[improved]
        self.best_positions[improved] = self.particles[improved]

        # 9. Update global best
        current_min_idx = np.argmin(self.values)
        if self.values[current_min_idx] < self.best_global_value:
            self.best_global_value = self.values[current_min_idx]
            self.best_global_position = self.particles[current_min_idx].copy()

        # 10. Optional temperature annealing (standard CBO practice: anneal beta upward)
        if self.beta_growth is not None:
            self.beta *= self.beta_growth


class PSO(Simulation):
    """
    Vectorized simulation of canonical Particle Swarm Optimization (Kennedy & Eberhart,
    Clerc-Kennedy constriction coefficients). Zeroth-order: no gradient information, no
    injected SDE noise.
    """
    def __init__(
        self,
        n: int,
        d: int,
        f: Callable[[np.ndarray], np.ndarray],
        df: Callable[[np.ndarray], np.ndarray],
        bounds: Optional[List[List[float]]] = None,
        particle_init: Optional[List[List[float]]] = None,
        learning_rate: float = 0.01,
        w: float = 0.729,             # Inertia weight
        c1: float = 1.49445,          # Cognitive coefficient
        c2: float = 1.49445,          # Social coefficient
        name: str = "PSO"
    ):
        if bounds is None:
            bounds = [[-np.inf, np.inf]] * d

        super().__init__(
            n=n,
            d=d,
            f=f,
            df=df,
            bounds=bounds,
            particle_init=particle_init,
            learning_rate=learning_rate,
            name=name
        )

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.velocities = np.zeros_like(self.particles)

        self._lower_bounds = np.array([b[0] for b in self.bounds])
        self._upper_bounds = np.array([b[1] for b in self.bounds])

    def step(self) -> None:
        """Executes a single step of canonical PSO (no gradient term, no injected noise)."""
        # 1. Generate stochastic tensors for cognitive and social components
        r1 = np.random.uniform(0, 1, size=self.particles.shape)
        r2 = np.random.uniform(0, 1, size=self.particles.shape)

        # 2. Calculate velocity components
        inertia_term = self.w * self.velocities
        cognitive_term = self.c1 * r1 * (self.best_positions - self.particles)
        social_term = self.c2 * r2 * (self.best_global_position - self.particles)

        # 3. Update velocities (standard PSO, no gradient drift)
        self.velocities = inertia_term + cognitive_term + social_term

        # 4. Update positions (no injected noise)
        self.particles = self.particles + self.velocities

        # 5. Fast vectorized clipping
        self.particles = np.clip(self.particles, self._lower_bounds, self._upper_bounds)

        # 6. Evaluate objective exactly once (no gradient evaluation at all)
        self.values = self._eval_obj(self.particles)

        # 7. Update memory caches (cognitive/personal bests)
        improved = self.values < self.best_values
        self.best_values[improved] = self.values[improved]
        self.best_positions[improved] = self.particles[improved]

        # 8. Update global knowledge (social best)
        current_min_idx = np.argmin(self.values)
        if self.values[current_min_idx] < self.best_global_value:
            self.best_global_value = self.values[current_min_idx]
            self.best_global_position = self.particles[current_min_idx].copy()

    def reset(self, particle_init: Optional[Union[List, np.ndarray]] = None) -> None:
        """Resets the environment, ensuring velocities are cleared."""
        super().reset(particle_init)
        self.velocities = np.zeros_like(self.particles)
