import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle as pkl
import multiprocessing as mp

N_CPU = mp.cpu_count()

from typing import Dict, Union, Optional, Any, List, Callable
from abc import ABC, abstractmethod

from sympy import symbols, diff, solve, lambdify, Matrix, Piecewise, exp, sin, cos, sqrt, pi, log
from numpy.polynomial import polynomial as P
from sympy.plotting import plot

import logging

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
        
        logger.info(f"Initialized Simulation '{self.name}' with {self.n} particles in {self.d}D space.")

    @abstractmethod
    def step(self) -> None:
        """Performs a single simulation iteration."""
        pass
    
    def run(self, n_steps: int) -> None:
        """Executes the simulation loop."""
        logger.info(f"Running simulation for {n_steps} steps...")
        for i in range(n_steps):
            self.n_steps = i + 1
            self.step()
            self.history.append(self.particles.copy())
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

        for i in range(max_steps):
            self.step()
            # Vectorized logical AND across all bounds, then checking if ANY particle satisfies ALL dimension bounds
            in_bounds = np.all((self.particles > lower_bounds) & (self.particles < upper_bounds), axis=1)
            
            if np.any(in_bounds):
                logger.info(f"Entry set reached at step {i}.")
                return i * self.learning_rate
            if i % 1000 == 0:
                # If the max standard deviation across all spatial dimensions is functionally zero
                if np.max(np.std(self.particles, axis=0)) < 1e-6:
                    # The swarm is permanently trapped. Abort the simulation to save CPU time.
                    return max_steps * self.learning_rate
                
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

        figsize = (10,6)
        cmap = kwargs.get("cmap", "viridis")

        if self.d > 2:
            projection_axes = kwargs.get("projection_axes", [0, 1])
            assert len(projection_axes) == 2, "projection_axes must be a list of length 2."
            assert all([0 <= p < self.d for p in projection_axes]), "projection_axes must be in the range of the dimension of the space."
            assert projection_axes[0] != projection_axes[1], "projection_axes must be different."

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(self.bounds[projection_axes[0]][0], self.bounds[projection_axes[0]][1])
            ax.set_ylim(self.bounds[projection_axes[1]][0], self.bounds[projection_axes[1]][1])

            x = np.linspace(self.bounds[projection_axes[0]][0], self.bounds[projection_axes[0]][1], 100)
            y = np.linspace(self.bounds[projection_axes[1]][0], self.bounds[projection_axes[1]][1], 100)
            X, Y = np.meshgrid(x, y)
            input_data = np.zeros((X.size, self.d))
            input_data[:, projection_axes[0]] = X.ravel()
            input_data[:, projection_axes[1]] = Y.ravel()
            Z = self.f(input_data).reshape(X.shape)
            ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.4)

            ax.scatter(self.particles[:, projection_axes[0]], self.particles[:, projection_axes[1]], self.f(self.particles), label="Particles", color="blue")
            ax.scatter(self.best_global_position[projection_axes[0]], self.best_global_position[projection_axes[1]], self.best_global_value, color="red", label="Best Global Position")
            ax.set_xlabel(f"x{projection_axes[0]}")
            ax.set_ylabel(f"x{projection_axes[1]}")
            ax.set_zlabel("f(x)")
            ax.set_title(f"3D Particles with Function Landscape - projection axes {projection_axes}")
            ax.legend()
            plt.show()

        elif self.d == 2:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

            x = np.linspace(self.bounds[0][0], self.bounds[0][1], 100)
            y = np.linspace(self.bounds[1][0], self.bounds[1][1], 100)
            X, Y = np.meshgrid(x, y)
            Z = self.f(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)

            ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.4)

            ax.scatter(self.particles[:, 0], self.particles[:, 1], self.f(self.particles), label="Particles", color="blue")
            ax.scatter(self.best_global_position[0], self.best_global_position[1], self.best_global_value, color="red", label="Best Global Position")
            
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("f(x, y)")
            ax.set_title("3D Particles with Function Landscape")
            ax.legend()
            plt.show()
        
        elif self.d == 1:
            x = np.linspace(self.bounds[0][0], self.bounds[0][1], 1000).reshape(-1,1)
            plt.figure(figsize=figsize)
            plt.plot(x, self.f(x))
            plt.scatter(self.particles, self.f(self.particles), color="blue")
            plt.scatter(self.best_global_position, self.best_global_value, color="red")
            plt.xlabel("x")
            plt.title("Particles")
            plt.show()
        
        if make_gif:
            if self.d > 2:
                projection_axes = kwargs.get("projection_axes", [0, 1])
                assert len(projection_axes) == 2, "projection_axes must be a list of length 2."
                assert all([0 <= p < self.d for p in projection_axes]), "projection_axes must be in the range of the dimension of the space."
                assert projection_axes[0] != projection_axes[1], "projection_axes must be different."

                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection='3d')

                x = np.linspace(self.bounds[projection_axes[0]][0], self.bounds[projection_axes[0]][1], 100)
                y = np.linspace(self.bounds[projection_axes[1]][0], self.bounds[projection_axes[1]][1], 100)
                X, Y = np.meshgrid(x, y)
                input_data = np.zeros((X.size, self.d))
                input_data[:, projection_axes[0]] = X.ravel()
                input_data[:, projection_axes[1]] = Y.ravel()
                Z = self.f(input_data).reshape(X.shape)

                ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.4)

                particles = ax.scatter(self.history[0][:, projection_axes[0]], self.history[0][:, projection_axes[1]], 
                                        self.f(self.history[0]), color="blue", label="Particles")
                best_particle = ax.scatter(self.best_global_position[projection_axes[0]], self.best_global_position[projection_axes[1]], 
                                            self.best_global_value, color="red", label="Best Global Position")

                ax.set_xlabel(f"x{projection_axes[0]}")
                ax.set_ylabel(f"x{projection_axes[1]}")
                ax.set_zlabel("f(x)")
                ax.set_title(f"3D Particles with Function Landscape - projection axes {projection_axes}")
                ax.legend()

                def animate(i):
                    particles._offsets3d = (self.history[i][:, projection_axes[0]], self.history[i][:, projection_axes[1]], 
                                            self.f(self.history[i]))
                    best_particle._offsets3d = ([self.best_global_position[projection_axes[0]]], 
                                                [self.best_global_position[projection_axes[1]]], 
                                                [self.best_global_value])
                    return particles, best_particle

                ani = animation.FuncAnimation(fig, animate, frames=len(self.history), interval=100)
                ani.save(path+f"{self.name}_{self.n}_particles_steps{self.n_steps}_3D.gif", writer="pillow", fps=30)

            elif self.d == 2:
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection='3d')

                x = np.linspace(self.bounds[0][0], self.bounds[0][1], 100)
                y = np.linspace(self.bounds[1][0], self.bounds[1][1], 100)
                X, Y = np.meshgrid(x, y)
                Z = np.array([self.f(x_i) for x_i in np.c_[X.ravel(), Y.ravel()]]).reshape(X.shape)

                ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.4)

                particles = ax.scatter(self.history[0][:, 0], self.history[0][:, 1], 
                                        self.f(self.history[0]), color="blue", label="Particles")
                best_particle = ax.scatter(self.best_global_position[0], self.best_global_position[1], 
                                            self.best_global_value, color="red", label="Best Global Position")

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("f(x, y)")
                ax.set_title("3D Particles with Function Landscape")
                ax.legend()

                def animate(i):
                    particles._offsets3d = (self.history[i][:, 0], self.history[i][:, 1], 
                                            self.f(self.history[i]))
                    best_particle._offsets3d = ([self.best_global_position[0]], 
                                                [self.best_global_position[1]], 
                                                [self.best_global_value])
                    return particles, best_particle

                ani = animation.FuncAnimation(fig, animate, frames=len(self.history), interval=100)
                ani.save(path+f"{self.name}_{self.n}_particles_steps{self.n_steps}_3D.gif", writer="pillow", fps=30)

            elif self.d == 1:
                if self.history_eq_values is None:
                    x = np.linspace(self.bounds[0][0], self.bounds[0][1], 1000).reshape(-1,1)
                    fig, ax = plt.subplots(figsize=figsize)
                    line1 = ax.plot(x, [self.f(x_i) for x_i in x])
                    line2 = ax.scatter(self.history[0], self.f(self.history[0]))
                    line3 = ax.scatter(self.history[0][np.argmin(self.f(self.history[0]))], np.min(self.f(self.history[0])), color="red")
                    ax.set_xlim(self.bounds[0][0], self.bounds[0][1])
                    ax.set_ylim(min([self.f(x_i) for x_i in x])-0.5, max([self.f(x_i) for x_i in x])+0.5)
                    plt.xlabel("x")
                    plt.title("Particles")

                    def animate(i):
                        line2.set_offsets(np.c_[self.history[i], self.f(self.history[i])])
                        line3.set_offsets(np.c_[self.history[i][np.argmin(self.f(self.history[i]))], np.min(self.f(self.history[i]))])
                        return line2, line3
                else:
                    x = np.linspace(self.bounds[0][0], self.bounds[0][1], 1000).reshape(-1,1)
                    fig, ax = plt.subplots(figsize=figsize)
                    line1 = ax.plot(x, [self.f(x_i) for x_i in x])
                    line2 = ax.scatter(self.history[0], self.f(self.history[0]))
                    line3 = ax.scatter(self.history[0][np.argmin(self.f(self.history[0]))], np.min(self.f(self.history[0])), color="red")
                    ax.set_xlim(self.bounds[0][0], self.bounds[0][1])
                    ax.set_ylim(min([self.f(x_i) for x_i in x])-0.5, max([self.f(x_i) for x_i in x])+0.5)
                    plt.xlabel("x")
                    plt.title("Particles")
                    # Add green horizontal line for history_eq_values[0]
                    green_line = ax.axhline(self.history_eq_values[0], color="green", linestyle="--", label="history_eq_value")
                    ax.legend()

                    def animate(i):
                        line2.set_offsets(np.c_[self.history[i], self.f(self.history[i])])
                        line3.set_offsets(np.c_[self.history[i][np.argmin(self.f(self.history[i]))], np.min(self.f(self.history[i]))])
                        # Update green horizontal line
                        green_line.set_ydata(self.history_eq_values[i])
                        return line2, line3, green_line
                    
                ani = animation.FuncAnimation(fig, animate, frames=len(self.history), interval=100)
                ani.save(path+f"{self.name}_{self.n}_particles_steps{self.n_steps}_noise_{self.noise}.gif", writer="pillow", fps=30)


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
        adaptive_lr: bool = True
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
        gradient = self.df(self.particles)
        energy = self.Js[-1]
        
        # Compute stabilized weights
        weights = self._get_softmin_weights(self.values)
        
        # Vectorized velocity computation
        # weights * self.n replaces (n * exponentials / sum(exponentials))
        velocity_scale = (1.0 - self.beta * (self.values - energy)) * (self.n * weights)
        velocities = velocity_scale[:, np.newaxis] * gradient
        
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
        self.values = self.f(self.particles)
        
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
        gradient = self.df(self.particles)
        
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
        self.values = self.f(self.particles)
        
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
        gradient = self.df(self.particles)
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
        self.values = self.f(self.particles)
        
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
        learning_rate: float = 0.01,         # η: Gradient step size / dt
        consensus_rate: float = 1.0,         # λ: Attraction pull to the consensus
        noise_intensity: float = np.sqrt(2), # σ: Diffusion scaling
        beta: float = 10.0,                  # β: Temperature inverse for consensus
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
            
        self.consensus_rate = consensus_rate
        self.noise_intensity = noise_intensity
        self.beta = float(beta)
        
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
        gradient = self.df(self.particles)
        
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
        # Element-wise anisotropic scaling
        # Add a constant noise floor (e.g., 0.5) so diffusion never drops to zero
        noise_floor = 0.5 
        diffusion_term = (self.noise_intensity * np.abs(diff_to_consensus) + noise_floor) * self._sqrt_dt * dW
                
        # 4. Update Swarm Positions
        self.particles = self.particles - grad_drift - consensus_drift + diffusion_term
        
        # 5. Fast vectorized clipping using memory views
        self.particles = np.clip(self.particles, self._lower_bounds, self._upper_bounds)
        
        # 6. Evaluate Objective exactly once
        self.values = self.f(self.particles)
        
        # 7. Update memory caches (Local bests)
        improved = self.values < self.best_values
        self.best_values[improved] = self.values[improved]
        self.best_positions[improved] = self.particles[improved]
        
        # 8. Update global best
        current_min_idx = np.argmin(self.values)
        if self.values[current_min_idx] < self.best_global_value:
            self.best_global_value = self.values[current_min_idx]
            self.best_global_position = self.particles[current_min_idx].copy()

import numpy as np
from typing import Callable, List, Optional

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
        gradient = self.df(self.particles)
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
        
        # 5. Update Positions
        # 5. Update Positions with Injected Langevin Noise
        noise_floor = 0.5
        sde_noise = noise_floor * np.sqrt(self.learning_rate) * np.random.normal(size=self.particles.shape)
        
        self.particles = self.particles + self.velocities + sde_noise
        
        # 6. Fast vectorized clipping using memory views
        # Note: In strict PSO, velocities should also be zeroed out if a particle hits a boundary,
        # but for continuous potentials with gradient awareness, soft clipping is preferred.
        self.particles = np.clip(self.particles, self._lower_bounds, self._upper_bounds)
        
        # 7. Evaluate Objective strictly once per step
        self.values = self.f(self.particles)
        
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

def one_dimensional_double_well(delta):
    x = symbols('x')
    f = x**4 - delta * x**2
    df = diff(f, x)
    a,b,c = P.polyroots([0,-2*delta,0,4])

    f_num = lambdify(x, f, modules='numpy')
    df_num = lambdify(x, df, modules='numpy')

    def double_well(x):
        return f_num(x)
    
    def double_well_grad(x):
        return df_num(x)
    
    double_well_vec = np.vectorize(double_well,signature='(n)->()')
    double_well_grad_vec = np.vectorize(double_well_grad,signature='(n)->(n)')

    return {
        'f_vec': double_well_vec,
        'f_grad_vec': double_well_grad_vec,
        'particle_init': [[a-0.2,a+0.2]],
        'set': [[c-0.2,c+0.2]],
        'bounds': [[-0.5-np.sqrt(delta),np.sqrt(delta)+0.5]],
        'critical_points': [a, b, c],
    }


def unbalanced_quadruple_well_setting(A, delta):
    x, y = symbols('x y')
    f = x**4 + y**4 - A*(x**2 + y**2) -delta*x*y
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
            'particle_init': [[-crit_value-0.2,-crit_value+0.2],[crit_value-0.2,crit_value+0.2]],
            'set': [[crit_value-0.2,crit_value+0.2],[-crit_value-0.2,-crit_value+0.2]],
            'bounds': [[-0.5-np.sqrt(A), np.sqrt(A)+0.5], [-0.5-np.sqrt(A), np.sqrt(A)+0.5]],
            'critical_points': critical_points}

def low_dimensional_quadruple_well_setting(A,delta,dimension):
    dict = unbalanced_quadruple_well_setting(A, delta)
    qw_vec = dict['f_vec']
    qw_grad_vec = dict['f_grad_vec']

    def f(x):
        return qw_vec(x[:2]) + np.sum(0.01*x[2:]**2)
    
    def f_grad(x):
        return np.concatenate([qw_grad_vec(x[:2]), 0.02*x[2:]])
    
    f_vec = np.vectorize(f, signature='(n)->()')
    f_grad_vec = np.vectorize(f_grad, signature='(n)->(n)')
    particle_init = dict['particle_init']+[[-0.2,0.2]]*(dimension-2)
    set_ = dict['set']+[[-0.2,0.2]]*(dimension-2)
    return {'f_vec': f_vec,
            'f_grad_vec': f_grad_vec,
            'particle_init': particle_init,
            'set': set_}

def asymmetric_potential_barrier(beta=1, depth = 2, offset = 5, plot_ = False):

    x = symbols('x')
    a,b,c,d = symbols('a b c d')

    f1 = x**2/(1+exp(-beta*x))
    domain_f1 = x<offset/3
    df1_dx = diff(f1, x)

    f2 = a*x**3+b*x**2+c*x+d
    domain_f2 = (x>=offset/3) & (x<2/3*offset)
    df2_dx = diff(f2, x)

    f3 = (x-offset)**2-depth
    domain_f3 = x>=2/3*offset
    df3_dx = diff(f3, x)

    eq1 = f1.subs(x,offset/3)-f2.subs(x,offset/3)
    eq2 = df1_dx.subs(x,offset/3)-df2_dx.subs(x,offset/3)
    eq3 = f2.subs(x,2/3*offset)-f3.subs(x,2/3*offset)
    eq4 = df2_dx.subs(x,2/3*offset)-df3_dx.subs(x,2/3*offset)
    sol = solve([eq1, eq2, eq3, eq4], (a, b, c, d))
    
    if sol:
        a_sol, b_sol, c_sol, d_sol = sol[a], sol[b], sol[c], sol[d]
        smooth_f = Piecewise((f1, domain_f1), (f2.subs({a: a_sol, b: b_sol, c: c_sol, d: d_sol}), domain_f2), (f3, domain_f3))
        plot(smooth_f, (x, -4/3*offset, 4/3*offset), title = "Asymmetric Potential Barrier", show = plot_)
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

def ackley_function_setting(a = 20, b = 0.2, c = 2 * np.pi):
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
            'particle_init': [[-8,8],[-8,8]],
            'set': [[-0.2, 0.2],[-0.2, 0.2]],
            'bounds': [[-10, 10], [-10, 10]]}


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

if __name__ == "__main__":
    # Example of running simulations on the low dimensional quadruple well potential

    for d in range(5,105, 5):
        setting = low_dimensional_quadruple_well_setting(5, -1, d)
        f_vec = setting['f_vec']
        f_grad_vec = setting['f_grad_vec']
        particle_init = setting['particle_init']
        set_ = setting['set']

        # Initialize the simulations
        softmin_energy = SoftminEnergyAdagrad(n=100, d=d, f=f_vec, df=f_grad_vec, particle_init=particle_init, name=f"SoftMin_quadruple_well_dim{d}", initial_lr=0.01, adaptive_lr=False, noise_intensity=np.sqrt(d), bounds = [[-4,4]], noise="normal", beta_decay=0.9999)

        simulated_annealing = SimulatedAnnealing(n=100, d=d, f=f_vec, df=f_grad_vec, bounds=[[-4,4]], particle_init=particle_init, name=f"SimulatedAnnealing_quadruple_well_dim{d}", learning_rate=0.01, cooling_rate = 0.9999, noise=np.sqrt(d), cooling_schedule="geometric", initial_temperature=10)

        def simulate_entry_times(_):
            softmin_energy.reset(particle_init=particle_init)
            simulated_annealing.reset(particle_init=particle_init)

            entry_time_softmin = softmin_energy.first_entry_time(set=set_)
            entry_time_simulated_annealing = simulated_annealing.first_entry_time(set=set_)

            return entry_time_softmin, entry_time_simulated_annealing

        with mp.Pool(N_CPU) as pool:
            results = pool.map(simulate_entry_times, range(128))

        entry_times_softmin, entry_times_simulated_annealing = zip(*results)
        
        # Save the results
        with open(f"entry_times_softmin_dim{d}.pkl", "wb") as f:
            pkl.dump(np.array(entry_times_softmin), f)
        with open(f"entry_times_simulated_annealing_dim{d}.pkl", "wb") as f:
            pkl.dump(np.array(entry_times_simulated_annealing), f)





