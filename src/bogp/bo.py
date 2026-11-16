"""The Bayesian-optimisation loop (ask / tell interface).

Conventions
-----------
* We **minimise** ``f`` over a box ``bounds`` of shape ``(d, 2)``.
* Inputs are mapped to the unit cube, outputs are standardised inside the GP.
* The surrogate is an ARD Matérn-5/2 GP by default (Snoek et al. 2012), with
  hyperparameters re-estimated by ML-II after every batch of observations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .acquisition import Acquisition, make_acquisition
from .batch import select_batch
from .gp import GaussianProcess
from .kernels import Kernel, make_kernel
from .optimize import maximize_acquisition, sobol, thompson_sample

Array = NDArray[np.float64]


@dataclass
class BOResult:
    """History of an optimisation run (in the user's coordinates)."""

    X: Array
    y: Array
    n_init: int
    times: list[float] = field(default_factory=list)

    @property
    def best_so_far(self) -> Array:
        return np.minimum.accumulate(self.y)

    @property
    def x_best(self) -> Array:
        return self.X[int(np.argmin(self.y))]

    @property
    def y_best(self) -> float:
        return float(self.y.min())


class BayesianOptimizer:
    """GP-based Bayesian optimiser.

    Parameters
    ----------
    bounds:
        Box ``(d, 2)`` of the search space.
    acquisition:
        ``"ei"``, ``"logei"``, ``"pi"``, ``"ucb"`` or ``"ts"`` (Thompson sampling),
        or an :class:`~bogp.acquisition.Acquisition` instance.
    kernel:
        Kernel name (see :func:`bogp.kernels.make_kernel`) or a ``Kernel``.
    n_init:
        Size of the scrambled-Sobol initial design.
    batch_size, batch_strategy:
        ``q`` points per round and how they are chosen (see :mod:`bogp.batch`).
    noise_variance, optimize_noise:
        Likelihood noise (normalised units).  For deterministic benchmarks we
        still learn a small noise term, which regularises the Cholesky.
    seed:
        Seeds the initial design, the inner optimisers and Thompson draws.
    """

    def __init__(
        self,
        bounds: ArrayLike,
        acquisition: str | Acquisition = "ei",
        kernel: str | Kernel = "matern52",
        n_init: int | None = None,
        batch_size: int = 1,
        batch_strategy: str = "kb",
        noise_variance: float = 1e-4,
        noise_bounds: tuple[float, float] = (1e-6, 1e-1),
        optimize_noise: bool = True,
        n_hyper_restarts: int = 2,
        lengthscale_bounds: tuple[float, float] = (1e-2, 1e1),
        variance_bounds: tuple[float, float] = (5e-2, 2e1),
        acq_kwargs: dict | None = None,
        seed: int = 0,
    ) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        self.d = self.bounds.shape[0]
        self.rng = np.random.default_rng(seed)
        self.n_init = n_init if n_init is not None else 2 * self.d + 1
        self.batch_size = batch_size
        self.batch_strategy = batch_strategy
        self.n_hyper_restarts = n_hyper_restarts
        if isinstance(acquisition, str) and acquisition.lower() == "ts":
            self.acq: Acquisition | None = None
            self.acq_name = "TS"
        else:
            self.acq = (
                make_acquisition(acquisition, **(acq_kwargs or {}))
                if isinstance(acquisition, str)
                else acquisition
            )
            self.acq_name = self.acq.name
        if isinstance(kernel, str):
            kernel = make_kernel(kernel, self.d, lengthscale=0.3,
                                 lengthscale_bounds=lengthscale_bounds,
                                 variance_bounds=variance_bounds)
        self.gp = GaussianProcess(kernel, noise_variance=noise_variance,
                                  noise_bounds=noise_bounds, optimize_noise=optimize_noise)
        self._U = np.empty((0, self.d))  # observations in the unit cube
        self._y = np.empty(0)
        self._init_design = sobol(self.n_init, self.d, self.rng)
        self._fitted_n = -1

    # ------------------------------------------------------------ scaling
    def to_unit(self, X: ArrayLike) -> Array:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (np.atleast_2d(X) - lo) / (hi - lo)

    def from_unit(self, U: ArrayLike) -> Array:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + np.atleast_2d(U) * (hi - lo)

    # ------------------------------------------------------------ ask/tell
    def fit_model(self) -> GaussianProcess:
        """(Re)fit the GP on all observations if new data arrived."""
        if self._fitted_n != len(self._y):
            self.gp.fit(self._U, self._y, optimize=True,
                        n_restarts=self.n_hyper_restarts, rng=self.rng)
            self._fitted_n = len(self._y)
        return self.gp

    def ask(self, q: int | None = None) -> Array:
        """Next ``q`` points to evaluate (user coordinates), shape ``(q, d)``."""
        q = q or self.batch_size
        n = len(self._y)
        if n < self.n_init:
            U = self._init_design[n : min(n + q, self.n_init)]
            if len(U) < q:  # pad with random points when the design runs out
                U = np.vstack([U, self.rng.random((q - len(U), self.d))])
            return self.from_unit(U)
        gp = self.fit_model()
        if q == 1 and self.batch_strategy != "fantasy":
            if self.acq is None:
                U = thompson_sample(gp, self.d, self.rng, self._U, self._y, n_samples=1)
            else:
                self.acq.update(t=n, best=float(self._y.min()), dim=self.d, y_scale=gp.y_std)
                x, _ = maximize_acquisition(lambda Z: self.acq.value_and_grad(gp, Z),
                                            self.d, self.rng, self._U, self._y)
                U = x[None]
        else:
            strategy = "ts" if self.acq is None else self.batch_strategy
            U = select_batch(gp, self.acq, q, self.rng, strategy=strategy)  # type: ignore[arg-type]
        return self.from_unit(U)

    def tell(self, X: ArrayLike, y: ArrayLike) -> None:
        """Record observations ``y = f(X)``."""
        self._U = np.vstack([self._U, self.to_unit(X)])
        self._y = np.concatenate([self._y, np.asarray(y, dtype=float).ravel()])

    # ------------------------------------------------------------ driver
    def minimize(
        self,
        f: Callable[[Array], Array],
        n_evals: int,
        callback: Callable[["BayesianOptimizer"], None] | None = None,
    ) -> BOResult:
        """Run until ``n_evals`` function evaluations have been spent.

        ``f`` maps an ``(m, d)`` array to ``(m,)`` values.
        """
        times: list[float] = []
        while len(self._y) < n_evals:
            t0 = time.perf_counter()
            q = min(self.batch_size, n_evals - len(self._y))
            X = self.ask(q)
            times.append(time.perf_counter() - t0)
            self.tell(X, f(X))
            if callback is not None:
                callback(self)
        return BOResult(self.from_unit(self._U), self._y.copy(), self.n_init, times)
