"""Acquisition functions for *minimisation*, with analytic input gradients.

Every acquisition is written so that **larger is better**.  For a GP posterior
``f(x) ~ N(mu(x), sigma(x)^2)`` and incumbent ``f* = min_i y_i``:

* Expected improvement   ``EI(x) = (f* - mu - xi) Phi(z) + sigma phi(z)``,
  ``z = (f* - mu - xi) / sigma`` (Jones et al. 1998; Snoek et al. 2012).
* Probability of improvement ``PI(x) = Phi(z)`` (Kushner 1964).
* GP-UCB (here the lower confidence bound, negated)
  ``-(mu - sqrt(beta_t) sigma)`` with Srinivas et al. (2010)'s schedule.

Thompson sampling is not a closed-form acquisition; it lives in
:mod:`bogp.optimize` because it needs joint posterior samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfcx, log_ndtr, ndtr

from .gp import GaussianProcess

Array = NDArray[np.float64]

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)
_SQRT_HALF_PI = np.sqrt(np.pi / 2.0)


def _phi(z: Array) -> Array:
    return _INV_SQRT_2PI * np.exp(-0.5 * z * z)


class Acquisition:
    """Base class: ``value_and_grad(gp, X)`` returns ``(a(X), da/dX)``."""

    name: str = "acq"

    best: float = 0.0
    t: int = 1
    dim: int = 1
    y_scale: float = 1.0

    def update(self, t: int, best: float, dim: int, y_scale: float = 1.0) -> None:
        """Hook called once per BO iteration.

        ``t`` is the number of observations, ``best`` the incumbent value and
        ``y_scale`` the standard deviation of the observed targets.
        """
        self.best = best
        self.t = t
        self.dim = dim
        self.y_scale = y_scale

    def from_moments(
        self, mu: Array, sigma: Array
    ) -> tuple[Array, Array, Array]:
        """Return ``a``, ``da/dmu``, ``da/dsigma`` for vectors of moments."""
        raise NotImplementedError

    def __call__(self, gp: GaussianProcess, X: Array) -> Array:
        mu, var = gp.predict(X)
        return self.from_moments(mu, np.sqrt(var))[0]

    def value_and_grad(self, gp: GaussianProcess, X: Array) -> tuple[Array, Array]:
        mu, var, dmu, dvar = gp.predict_with_grad(X)
        sigma = np.sqrt(var)
        a, da_dmu, da_ds = self.from_moments(mu, sigma)
        dsigma = dvar / (2.0 * sigma[:, None])
        return a, da_dmu[:, None] * dmu + da_ds[:, None] * dsigma


@dataclass
class ExpectedImprovement(Acquisition):
    """Closed-form EI with optional exploration margin ``xi``."""

    xi: float = 0.0
    name: str = "EI"

    def from_moments(self, mu: Array, sigma: Array) -> tuple[Array, Array, Array]:
        imp = self.best - mu - self.xi
        z = imp / sigma
        cdf, pdf = ndtr(z), _phi(z)
        ei = imp * cdf + sigma * pdf
        # dEI/dmu = -Phi(z), dEI/dsigma = phi(z)
        return np.maximum(ei, 0.0), -cdf, pdf


@dataclass
class LogExpectedImprovement(Acquisition):
    """``log EI`` -- same maximiser as EI but numerically usable when EI
    underflows far from the incumbent (helps the multi-start optimiser)."""

    xi: float = 0.0
    name: str = "logEI"

    def from_moments(self, mu: Array, sigma: Array) -> tuple[Array, Array, Array]:
        imp = self.best - mu - self.xi
        z = imp / sigma
        # EI = sigma * h(z) with h(z) = z Phi(z) + phi(z) = phi(z) * c(z),
        # c(z) = 1 + z Phi(z)/phi(z) = 1 + z sqrt(pi/2) erfcx(-z/sqrt2).
        # erfcx keeps Phi/phi finite for very negative z; c(z) ~ 1/z^2 there.
        r = _SQRT_HALF_PI * erfcx(-z / np.sqrt(2.0))  # Phi(z) / phi(z)
        c = 1.0 + z * r
        tail = z < -1e3  # cancellation in c: use the asymptotic series
        zt = np.where(tail, z, -1e3)
        c = np.where(tail, 1.0 / zt**2 - 3.0 / zt**4 + 15.0 / zt**6, c)
        c = np.maximum(c, 1e-300)
        a = -0.5 * z * z + np.log(_INV_SQRT_2PI) + np.log(c) + np.log(sigma)
        # dlogEI/dmu = -Phi/(sigma h) = -(r/c)/sigma ; dlogEI/dsigma = phi/(sigma h) = (1/c)/sigma
        return a, -(r / c) / sigma, (1.0 / c) / sigma


@dataclass
class ProbabilityOfImprovement(Acquisition):
    """``PI(x) = P(f(x) < f* - xi)``."""

    xi: float = 0.01
    name: str = "PI"

    def from_moments(self, mu: Array, sigma: Array) -> tuple[Array, Array, Array]:
        # xi is relative to the output scale (std of y) so it is unit-free
        imp = self.best - mu - self.xi * self.y_scale
        z = imp / sigma
        pdf = _phi(z)
        return ndtr(z), -pdf / sigma, -pdf * z / sigma


@dataclass
class UpperConfidenceBound(Acquisition):
    r"""GP-UCB for minimisation: maximise ``-(mu - sqrt(beta_t) sigma)``.

    ``beta_t = beta_scale * 2 log(d t^2 pi^2 / (6 delta))`` follows Theorem 1 of
    Srinivas et al. (2010) with ``|D|`` replaced by the input dimension, a
    common practical choice; the theoretical constants for continuous domains
    are known to over-explore.  A fixed ``beta`` can be passed instead.
    """

    beta: float | None = None
    delta: float = 0.1
    beta_scale: float = 1.0
    name: str = "UCB"

    def beta_t(self) -> float:
        if self.beta is not None:
            return float(self.beta)
        t = max(self.t, 1)
        return float(self.beta_scale * 2.0 * np.log(self.dim * t**2 * np.pi**2 / (6.0 * self.delta)))

    def from_moments(self, mu: Array, sigma: Array) -> tuple[Array, Array, Array]:
        sb = np.sqrt(self.beta_t())
        return -(mu - sb * sigma), -np.ones_like(mu), np.full_like(sigma, sb)


def log_probability_of_improvement(mu: Array, sigma: Array, best: float) -> Array:
    """Numerically stable ``log PI`` (used only for plotting)."""
    return log_ndtr((best - mu) / sigma)


def make_acquisition(name: str, **kwargs: float) -> Acquisition:
    """Factory: ``ei``, ``logei``, ``pi``, ``ucb``."""
    name = name.lower()
    table: dict[str, type[Acquisition]] = {
        "ei": ExpectedImprovement,
        "logei": LogExpectedImprovement,
        "pi": ProbabilityOfImprovement,
        "ucb": UpperConfidenceBound,
    }
    if name not in table:
        raise ValueError(f"unknown acquisition {name!r}")
    return table[name](**kwargs)  # type: ignore[arg-type]
