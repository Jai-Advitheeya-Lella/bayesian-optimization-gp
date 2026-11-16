"""Reference optimisers used only as baselines: random search,
scikit-optimize's ``gp_minimize`` and Optuna's TPE sampler."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Objective = Callable[[Array], Array]


def random_search(f: Objective, bounds: Array, n_evals: int, seed: int = 0) -> Array:
    """Uniform random search; returns the observed values in order."""
    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = lo + rng.random((n_evals, len(lo))) * (hi - lo)
    return f(X)


def skopt_gp(f: Objective, bounds: Array, n_evals: int, n_init: int, seed: int = 0) -> Array:
    """scikit-optimize GP-EI (Matérn-5/2 ARD, its own defaults otherwise)."""
    from skopt import gp_minimize

    res = gp_minimize(
        lambda x: float(f(np.asarray(x)[None])[0]),
        [tuple(map(float, b)) for b in bounds],
        n_calls=n_evals,
        n_initial_points=n_init,
        initial_point_generator="sobol",
        acq_func="EI",
        random_state=seed,
    )
    return np.asarray(res.func_vals, dtype=float)


def optuna_tpe(f: Objective, bounds: Array, n_evals: int, n_init: int, seed: int = 0) -> Array:
    """Optuna's Tree-structured Parzen Estimator (multivariate variant)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_init, multivariate=True)
    study = optuna.create_study(sampler=sampler, direction="minimize")

    def objective(trial: "optuna.Trial") -> float:
        x = np.array([trial.suggest_float(f"x{i}", float(lo), float(hi))
                      for i, (lo, hi) in enumerate(bounds)])
        return float(f(x[None])[0])

    study.optimize(objective, n_trials=n_evals)
    return np.array([t.value for t in study.trials], dtype=float)
