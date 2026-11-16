"""Batch (parallel) Bayesian optimisation: choose ``q`` points per round.

Greedy sequential strategies (Ginsbourger et al. 2010; Snoek et al. 2012):
the j-th point maximises an acquisition computed from a model that pretends
the first j-1 points have already been evaluated.

* **Kriging believer (KB)** -- pretend ``y_j = mu(x_j)``.
* **Constant liar (CL)**   -- pretend ``y_j = L`` with ``L = min/mean/max(y)``.
* **MC fantasies**         -- integrate the acquisition over the posterior of
  the pending outcomes, ``a(x) = E_{y_P ~ p(y_P | D)}[a(x | D u (X_P, y_P))]``,
  approximated with ``M`` joint posterior samples (Snoek et al. 2012, Sec. 3.3).
* **Batch Thompson sampling** -- ``q`` independent posterior draws.

``mc_qei`` estimates the joint q-EI ``E[max(f* - min_j f(x_j), 0)]`` which
we use to score batches.
"""

from __future__ import annotations

from copy import copy

import numpy as np
from numpy.typing import NDArray

from .acquisition import Acquisition
from .gp import GaussianProcess
from .optimize import maximize_acquisition, thompson_sample

Array = NDArray[np.float64]

STRATEGIES = ("kb", "cl-min", "cl-mean", "cl-max", "fantasy", "ts")


def _fit_acq(acq: Acquisition, gp: GaussianProcess, best: float, d: int) -> Acquisition:
    a = copy(acq)
    a.update(t=len(gp.y), best=best, dim=d, y_scale=gp.y_std)
    return a


def select_batch(
    gp: GaussianProcess,
    acq: Acquisition,
    q: int,
    rng: np.random.Generator,
    strategy: str = "kb",
    n_fantasies: int = 16,
    **opt_kwargs: int,
) -> Array:
    """Return ``q`` points in ``[0,1]^d`` chosen by ``strategy``."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown batch strategy {strategy!r}; choose from {STRATEGIES}")
    assert gp.X is not None and gp.y is not None
    X, y = gp.X, gp.y
    d = X.shape[1]
    if strategy == "ts":
        return thompson_sample(gp, d, rng, X, y, n_samples=q)

    batch: list[Array] = []
    if strategy == "fantasy":
        for _ in range(q):
            if not batch:
                models = [gp]
                bests = [float(y.min())]
            else:
                P = np.array(batch)
                ys = gp.sample_posterior(P, n_samples=n_fantasies, rng=rng)
                models = [gp.condition(P, yk) for yk in ys]
                bests = [min(float(y.min()), float(yk.min())) for yk in ys]
            acqs = [_fit_acq(acq, m, b, d) for m, b in zip(models, bests)]

            def vg(Z: Array, models=models, acqs=acqs) -> tuple[Array, Array]:
                vals, grads = zip(*(a.value_and_grad(m, Z) for m, a in zip(models, acqs)))
                return np.mean(vals, axis=0), np.mean(grads, axis=0)

            x, _ = maximize_acquisition(vg, d, rng, X, y, **opt_kwargs)
            batch.append(x)
        return np.array(batch)

    # Kriging believer / constant liar
    lie = {"cl-min": y.min(), "cl-mean": y.mean(), "cl-max": y.max()}.get(strategy)
    model = gp
    for _ in range(q):
        a = _fit_acq(acq, model, float(model.y.min()), d)
        x, _ = maximize_acquisition(lambda Z, m=model, a=a: a.value_and_grad(m, Z), d, rng,
                                    model.X, model.y, **opt_kwargs)
        batch.append(x)
        y_fake = float(model.predict(x[None])[0][0]) if lie is None else float(lie)
        model = model.condition(x[None], [y_fake])
    return np.array(batch)


def mc_qei(
    gp: GaussianProcess,
    Xq: Array,
    best: float,
    n_samples: int = 4096,
    rng: np.random.Generator | int | None = None,
) -> float:
    """Monte-Carlo estimate of the joint q-point expected improvement."""
    F = gp.sample_posterior(Xq, n_samples=n_samples, rng=rng)
    return float(np.maximum(best - F.min(axis=1), 0.0).mean())
