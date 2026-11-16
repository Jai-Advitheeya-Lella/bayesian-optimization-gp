"""Exact Gaussian-process regression via Cholesky factorisation.

Implements Algorithm 2.1 of Rasmussen & Williams (2006) plus type-II maximum
likelihood for the hyperparameters using the analytic gradient (R&W eq. 5.9)

    d log p(y|X,theta) / d theta_j = 1/2 tr((alpha alpha^T - K_y^{-1}) dK_y/d theta_j),

where ``K_y = K + sigma_n^2 I`` and ``alpha = K_y^{-1} y``.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize

from .kernels import Kernel, _as_2d

Array = NDArray[np.float64]

LOG_2PI = np.log(2.0 * np.pi)


def robust_cholesky(A: Array, max_tries: int = 6) -> tuple[Array, float]:
    """Lower Cholesky factor of ``A``, adding diagonal jitter if needed.

    Returns ``(L, jitter)`` where ``jitter`` is the amount that was added.
    """
    jitter = 0.0
    scale = float(np.mean(np.diag(A))) if A.size else 1.0
    for i in range(max_tries):
        try:
            L = cholesky(A + jitter * np.eye(A.shape[0]), lower=True, check_finite=False)
            if np.all(np.isfinite(L)):
                return L, jitter
        except np.linalg.LinAlgError:
            pass
        jitter = scale * 10.0 ** (-10 + 2 * i)
    raise np.linalg.LinAlgError("matrix is not positive definite even with jitter")


class GaussianProcess:
    """GP regression with a Gaussian likelihood.

    Parameters
    ----------
    kernel:
        Covariance function (its hyperparameters are optimised in ``fit``).
    noise_variance:
        Initial observation-noise variance ``sigma_n^2`` (in normalised units
        when ``normalize_y=True``).
    noise_bounds:
        Box constraint for ``sigma_n^2``.
    optimize_noise:
        Whether ``sigma_n^2`` is learned together with the kernel parameters.
    normalize_y:
        Standardise targets to zero mean / unit variance before fitting
        (same convention as scikit-learn: population std).
    """

    def __init__(
        self,
        kernel: Kernel,
        noise_variance: float = 1e-4,
        noise_bounds: tuple[float, float] = (1e-6, 1e0),
        optimize_noise: bool = True,
        normalize_y: bool = True,
    ) -> None:
        self.kernel = kernel
        self._log_noise = float(np.log(noise_variance))
        self.noise_bounds = noise_bounds
        self.optimize_noise = optimize_noise
        self.normalize_y = normalize_y
        self.X: Array | None = None
        self.y: Array | None = None
        self._y_mean = 0.0
        self._y_std = 1.0

    # ---------------------------------------------------------- parameters
    @property
    def noise_variance(self) -> float:
        return float(np.exp(self._log_noise))

    @property
    def theta(self) -> Array:
        """All learnable log-hyperparameters (kernel first, then noise)."""
        t = self.kernel.theta
        return np.concatenate([t, [self._log_noise]]) if self.optimize_noise else t

    @theta.setter
    def theta(self, value: ArrayLike) -> None:
        value = np.asarray(value, dtype=float)
        p = self.kernel.n_params
        self.kernel.theta = value[:p]
        if self.optimize_noise:
            self._log_noise = float(value[p])

    @property
    def bounds(self) -> Array:
        b = self.kernel.bounds
        if self.optimize_noise:
            b = np.vstack([b, np.log(np.array(self.noise_bounds, dtype=float))[None]])
        return b

    # ------------------------------------------------------------- fitting
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        optimize: bool = True,
        n_restarts: int = 2,
        rng: np.random.Generator | int | None = None,
        renormalize: bool = True,
    ) -> "GaussianProcess":
        """Condition on data, optionally optimising hyperparameters by ML-II.

        ``n_restarts`` extra L-BFGS-B runs start from points drawn uniformly in
        the (log) bounds; the current hyperparameters are always one start.
        """
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        if renormalize:
            if self.normalize_y and y.size > 1:
                self._y_mean = float(y.mean())
                self._y_std = float(y.std()) or 1.0
            elif self.normalize_y:
                self._y_mean, self._y_std = float(y.mean()), 1.0
            else:
                self._y_mean, self._y_std = 0.0, 1.0
        self.X = X
        self.y = y
        self._yn = (y - self._y_mean) / self._y_std
        if optimize:
            self._optimize_hyperparameters(n_restarts, np.random.default_rng(rng))
        self._factorize()
        return self

    def _factorize(self) -> None:
        assert self.X is not None
        K = self.kernel(self.X)
        K[np.diag_indices_from(K)] += self.noise_variance
        self._L, self._jitter = robust_cholesky(K)
        self._alpha = cho_solve((self._L, True), self._yn, check_finite=False)

    def log_marginal_likelihood(
        self, theta: ArrayLike | None = None, eval_gradient: bool = False
    ) -> float | tuple[float, Array]:
        """Log evidence of the (normalised) training targets.

        If ``theta`` is given it is used instead of the current hyperparameters
        (the model itself is left unchanged).
        """
        assert self.X is not None, "call fit() first"
        if theta is None:
            gp = self
        else:
            gp = copy.copy(self)
            gp.kernel = self.kernel.clone()
            gp.theta = theta
        yn = self._yn
        n = yn.shape[0]
        if eval_gradient:
            K, dK = gp.kernel.gradient(self.X)
        else:
            K = gp.kernel(self.X)
        K = K.copy()
        K[np.diag_indices_from(K)] += gp.noise_variance
        try:
            L, _ = robust_cholesky(K)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros(len(gp.theta))) if eval_gradient else -np.inf
        alpha = cho_solve((L, True), yn, check_finite=False)
        lml = -0.5 * yn @ alpha - np.log(np.diag(L)).sum() - 0.5 * n * LOG_2PI
        if not eval_gradient:
            return float(lml)
        Kinv = cho_solve((L, True), np.eye(n), check_finite=False)
        W = np.outer(alpha, alpha) - Kinv
        grad = 0.5 * np.einsum("ij,pij->p", W, dK)
        if gp.optimize_noise:
            grad = np.concatenate([grad, [0.5 * gp.noise_variance * np.trace(W)]])
        return float(lml), grad

    def _optimize_hyperparameters(self, n_restarts: int, rng: np.random.Generator) -> None:
        bounds = self.bounds

        def obj(t: Array) -> tuple[float, Array]:
            val, g = self.log_marginal_likelihood(t, eval_gradient=True)
            if not np.isfinite(val):
                return 1e25, np.zeros_like(t)
            return -val, -g

        starts = [np.clip(self.theta, bounds[:, 0], bounds[:, 1])]
        starts += [rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(n_restarts)]
        best_t, best_f = starts[0], np.inf
        for t0 in starts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(obj, t0, jac=True, method="L-BFGS-B", bounds=bounds)
            if res.fun < best_f:
                best_t, best_f = res.x, res.fun
        self.theta = best_t

    # ---------------------------------------------------------- prediction
    def predict(
        self, Xs: ArrayLike, return_cov: bool = False
    ) -> tuple[Array, Array]:
        """Posterior mean and variance (or full covariance) of the latent ``f``."""
        assert self.X is not None, "call fit() first"
        Xs = _as_2d(Xs)
        Ks = self.kernel(Xs, self.X)
        mu = Ks @ self._alpha
        V = solve_triangular(self._L, Ks.T, lower=True, check_finite=False)
        s2 = self._y_std**2
        if return_cov:
            cov = self.kernel(Xs) - V.T @ V
            return mu * self._y_std + self._y_mean, cov * s2
        var = np.maximum(self.kernel.diag(Xs) - (V * V).sum(0), 1e-15)
        return mu * self._y_std + self._y_mean, var * s2

    def predict_with_grad(self, Xs: ArrayLike) -> tuple[Array, Array, Array, Array]:
        """Posterior mean/variance and their gradients w.r.t. the inputs.

        Returns ``mu (m,)``, ``var (m,)``, ``dmu (m, d)``, ``dvar (m, d)``.
        Uses stationarity: ``d k(x, x) / dx = 0``.
        """
        assert self.X is not None, "call fit() first"
        Xs = _as_2d(Xs)
        Ks = self.kernel(Xs, self.X)  # (m, n)
        dKs = self.kernel.grad_x1(Xs, self.X)  # (m, n, d)
        mu = Ks @ self._alpha
        A = cho_solve((self._L, True), Ks.T, check_finite=False)  # K^{-1} k_*  (n, m)
        var_raw = self.kernel.diag(Xs) - np.einsum("mn,nm->m", Ks, A)
        var = np.maximum(var_raw, 1e-15)
        dmu = np.einsum("mnd,n->md", dKs, self._alpha)
        dvar = -2.0 * np.einsum("mnd,nm->md", dKs, A)
        dvar[var_raw <= 1e-15] = 0.0
        s, s2 = self._y_std, self._y_std**2
        return mu * s + self._y_mean, var * s2, dmu * s, dvar * s2

    def sample_posterior(
        self, Xs: ArrayLike, n_samples: int = 1, rng: np.random.Generator | int | None = None
    ) -> Array:
        """Joint posterior draws of ``f`` at ``Xs``; shape ``(n_samples, m)``."""
        rng = np.random.default_rng(rng)
        mu, cov = self.predict(Xs, return_cov=True)
        L, _ = robust_cholesky(0.5 * (cov + cov.T) + 1e-10 * self._y_std**2 * np.eye(len(mu)))
        z = rng.standard_normal((len(mu), n_samples))
        return (mu[:, None] + L @ z).T

    def condition(self, X_new: ArrayLike, y_new: ArrayLike) -> "GaussianProcess":
        """Return a copy conditioned on extra data with *fixed* hyperparameters
        and fixed output normalisation (used for fantasies / Kriging believer)."""
        assert self.X is not None
        gp = copy.copy(self)
        gp.kernel = self.kernel.clone()
        X = np.vstack([self.X, _as_2d(X_new)])
        y = np.concatenate([self.y, np.asarray(y_new, dtype=float).ravel()])
        return gp.fit(X, y, optimize=False, renormalize=False)

    @property
    def y_std(self) -> float:
        return self._y_std
