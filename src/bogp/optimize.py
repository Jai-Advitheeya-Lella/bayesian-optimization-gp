"""Inner-loop optimisers: maximise an acquisition over the unit cube, and
Thompson sampling over a candidate set.

All inputs here live in ``[0, 1]^d`` (the BO loop rescales to the user box).
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import qmc

from .gp import GaussianProcess

Array = NDArray[np.float64]

ValueAndGrad = Callable[[Array], tuple[Array, Array]]


def sobol(n: int, d: int, rng: np.random.Generator) -> Array:
    """Scrambled Sobol points in ``[0,1]^d`` (``n`` rounded up to a power of 2)."""
    m = int(np.ceil(np.log2(max(n, 2))))
    s = qmc.Sobol(d, scramble=True, seed=rng)
    return s.random_base2(m)[:n]


def candidate_set(
    d: int,
    rng: np.random.Generator,
    X_obs: Array | None = None,
    y_obs: Array | None = None,
    n_global: int = 1024,
    n_local: int = 512,
    n_top: int = 5,
    local_scale: float = 0.05,
) -> Array:
    """Global Sobol points plus Gaussian perturbations of the best observations.

    The local cloud matters: near the incumbent the acquisition surface is
    sharply peaked and a purely global design would rarely land on the peak.
    """
    C = sobol(n_global, d, rng)
    if X_obs is not None and y_obs is not None and n_local > 0 and len(y_obs) > 0:
        top = X_obs[np.argsort(y_obs)[:n_top]]
        idx = rng.integers(0, len(top), n_local)
        scales = local_scale * rng.choice([0.1, 1.0, 3.0], size=(n_local, 1))
        L = np.clip(top[idx] + scales * rng.standard_normal((n_local, d)), 0.0, 1.0)
        C = np.vstack([C, L])
    return C


def maximize_acquisition(
    value_and_grad: ValueAndGrad,
    d: int,
    rng: np.random.Generator,
    X_obs: Array | None = None,
    y_obs: Array | None = None,
    n_raw: int = 1024,
    n_local: int = 512,
    n_restarts: int = 5,
    maxiter: int = 200,
) -> tuple[Array, float]:
    """Multi-start L-BFGS-B with analytic gradients.

    1. Evaluate the acquisition on a raw candidate set (Sobol + local cloud).
    2. Start L-BFGS-B from the ``n_restarts`` best raw points.
    3. Return the best point found.
    """
    C = candidate_set(d, rng, X_obs, y_obs, n_global=n_raw, n_local=n_local)
    vals, _ = value_and_grad(C)
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    order = np.argsort(-vals)
    best_x, best_v = C[order[0]].copy(), float(vals[order[0]])
    bounds = [(0.0, 1.0)] * d

    def obj(x: Array) -> tuple[float, Array]:
        v, g = value_and_grad(x[None])
        if not np.isfinite(v[0]):
            return 1e10, np.zeros(d)
        return -float(v[0]), -g[0]

    for i in order[:n_restarts]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(obj, C[i], jac=True, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": maxiter})
        if -res.fun > best_v:
            best_x, best_v = np.clip(res.x, 0.0, 1.0), -float(res.fun)
    return best_x, best_v


def thompson_sample(
    gp: GaussianProcess,
    d: int,
    rng: np.random.Generator,
    X_obs: Array,
    y_obs: Array,
    n_samples: int = 1,
    n_global: int = 1024,
    n_local: int = 512,
) -> Array:
    """Draw ``n_samples`` joint posterior functions on a fresh candidate set and
    return the minimiser of each, shape ``(n_samples, d)``.

    Discretised Thompson sampling (as in e.g. Kandasamy et al. 2018): the
    candidate set is re-drawn every call, so over iterations the search covers
    the whole cube while the local cloud resolves the region near the optimum.
    """
    C = candidate_set(d, rng, X_obs, y_obs, n_global=n_global, n_local=n_local)
    F = gp.sample_posterior(C, n_samples=n_samples, rng=rng)
    return C[np.argmin(F, axis=1)]
