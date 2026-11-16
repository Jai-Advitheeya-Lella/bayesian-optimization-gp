"""Covariance functions with analytic hyperparameter and input gradients.

Every kernel exposes its hyperparameters in **log space** through the
``theta`` property, which is the parameterisation used for marginal-likelihood
optimisation (it keeps all parameters positive and makes the optimisation
landscape better conditioned).  ``gradient(X)`` returns the Gram matrix together
with ``dK / dtheta`` stacked along the first axis, and ``grad_x1(X1, X2)``
returns ``d k(x1_i, x2_j) / d x1_i``, which the acquisition optimiser uses.

All kernels here are stationary, so ``k(x, x)`` does not depend on ``x``.  The
GP code relies on that when differentiating the predictive variance.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

__all__ = [
    "Kernel",
    "RBF",
    "Matern32",
    "Matern52",
    "RationalQuadratic",
    "Periodic",
    "SumKernel",
    "ProductKernel",
    "make_kernel",
]


def _as_2d(X: ArrayLike) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return X


class Kernel:
    """Abstract covariance function ``k(x, x')``.

    Subclasses implement ``__call__``, ``diag``, ``gradient`` and ``grad_x1`` and
    manage their own log-parameter vector via ``theta`` / ``bounds``.
    """

    #: Human-readable names of the entries of ``theta``.
    param_names: list[str]

    # ------------------------------------------------------------------ API
    def __call__(self, X1: ArrayLike, X2: ArrayLike | None = None) -> Array:
        raise NotImplementedError

    def diag(self, X: ArrayLike) -> Array:
        """Diagonal of ``k(X, X)`` without forming the full matrix."""
        raise NotImplementedError

    def gradient(self, X: ArrayLike) -> tuple[Array, Array]:
        """Return ``K = k(X, X)`` and ``dK/dtheta`` with shape ``(p, n, n)``."""
        raise NotImplementedError

    def grad_x1(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        """Return ``d k(X1[i], X2[j]) / d X1[i]`` with shape ``(m, n, d)``."""
        raise NotImplementedError

    @property
    def theta(self) -> Array:
        raise NotImplementedError

    @theta.setter
    def theta(self, value: ArrayLike) -> None:
        raise NotImplementedError

    @property
    def bounds(self) -> Array:
        """Log-space box constraints, shape ``(p, 2)``."""
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return int(self.theta.shape[0])

    # ------------------------------------------------------------ algebra
    def __add__(self, other: "Kernel") -> "SumKernel":
        return SumKernel(self, other)

    def __mul__(self, other: "Kernel") -> "ProductKernel":
        return ProductKernel(self, other)

    def clone(self) -> "Kernel":
        return copy.deepcopy(self)

    def clone_with_theta(self, theta: ArrayLike) -> "Kernel":
        k = self.clone()
        k.theta = theta
        return k

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        vals = ", ".join(f"{n}={v:.3g}" for n, v in zip(self.param_names, np.exp(self.theta)))
        return f"{type(self).__name__}({vals})"


# ----------------------------------------------------------------------------
# Stationary ARD kernels that depend on x, x' only through
#     r^2 = sum_d (x_d - x'_d)^2 / l_d^2.
# Each subclass supplies the correlation kappa(r^2) and its derivative
# kappa'(r^2) = d kappa / d r^2.  Then, with Delta = x - x':
#     dK/dlog l_d = v * kappa'(r^2) * (-2 Delta_d^2 / l_d^2)
#     dK/dlog v   = K
#     dK/dx_d     = v * kappa'(r^2) * ( 2 Delta_d   / l_d^2)
# ----------------------------------------------------------------------------
class _StationaryARD(Kernel):
    def __init__(
        self,
        lengthscale: float | Sequence[float] = 1.0,
        variance: float = 1.0,
        lengthscale_bounds: tuple[float, float] = (1e-2, 1e2),
        variance_bounds: tuple[float, float] = (1e-2, 1e2),
    ) -> None:
        ls = np.atleast_1d(np.asarray(lengthscale, dtype=float))
        if np.any(ls <= 0) or variance <= 0:
            raise ValueError("lengthscale and variance must be positive")
        self._log_ls = np.log(ls)
        self._log_var = float(np.log(variance))
        self.lengthscale_bounds = lengthscale_bounds
        self.variance_bounds = variance_bounds

    # --- parameters -------------------------------------------------------
    @property
    def lengthscale(self) -> Array:
        return np.exp(self._log_ls)

    @property
    def variance(self) -> float:
        return float(np.exp(self._log_var))

    @property
    def param_names(self) -> list[str]:  # type: ignore[override]
        if self._log_ls.size == 1:
            return ["lengthscale", "variance"]
        return [f"lengthscale_{i}" for i in range(self._log_ls.size)] + ["variance"]

    @property
    def theta(self) -> Array:
        return np.concatenate([self._log_ls, [self._log_var]])

    @theta.setter
    def theta(self, value: ArrayLike) -> None:
        value = np.asarray(value, dtype=float)
        self._log_ls = value[:-1].copy()
        self._log_var = float(value[-1])

    @property
    def bounds(self) -> Array:
        p = self._log_ls.size
        lb = np.log(np.array(self.lengthscale_bounds))
        vb = np.log(np.array(self.variance_bounds))
        return np.vstack([np.tile(lb, (p, 1)), vb[None, :]])

    # --- correlation function (implemented by subclasses) -------------------
    def _kappa(self, r2: Array) -> Array:
        raise NotImplementedError

    def _dkappa(self, r2: Array) -> Array:
        raise NotImplementedError

    # --- helpers ------------------------------------------------------------
    def _check_dim(self, X: Array) -> None:
        if self._log_ls.size not in (1, X.shape[1]):
            raise ValueError(
                f"kernel has {self._log_ls.size} lengthscales but inputs have {X.shape[1]} dims"
            )

    def _r2(self, X1: Array, X2: Array) -> Array:
        A = X1 / self.lengthscale
        B = X2 / self.lengthscale
        r2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
        return np.maximum(r2, 0.0)

    # --- API ------------------------------------------------------------------
    def __call__(self, X1: ArrayLike, X2: ArrayLike | None = None) -> Array:
        X1 = _as_2d(X1)
        X2 = X1 if X2 is None else _as_2d(X2)
        self._check_dim(X1)
        return self.variance * self._kappa(self._r2(X1, X2))

    def diag(self, X: ArrayLike) -> Array:
        X = _as_2d(X)
        return np.full(X.shape[0], self.variance)

    def gradient(self, X: ArrayLike) -> tuple[Array, Array]:
        X = _as_2d(X)
        self._check_dim(X)
        ls = self.lengthscale
        D = (X[:, None, :] - X[None, :, :]) / ls  # (n, n, d) scaled differences
        r2 = (D * D).sum(-1)
        K = self.variance * self._kappa(r2)
        dk = self.variance * self._dkappa(r2)  # v * kappa'(r^2)
        if ls.size == 1:
            dls = (-2.0 * dk * r2)[None]  # isotropic: sum over dims
        else:
            dls = np.moveaxis(-2.0 * dk[..., None] * D * D, -1, 0)
        return K, np.concatenate([dls, K[None]], axis=0)

    def grad_x1(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        X1, X2 = _as_2d(X1), _as_2d(X2)
        ls = self.lengthscale
        diff = X1[:, None, :] - X2[None, :, :]
        r2 = ((diff / ls) ** 2).sum(-1)
        dk = self.variance * self._dkappa(r2)
        return 2.0 * dk[..., None] * diff / ls**2


class RBF(_StationaryARD):
    r"""Squared-exponential kernel :math:`v \exp(-r^2/2)`."""

    def _kappa(self, r2: Array) -> Array:
        return np.exp(-0.5 * r2)

    def _dkappa(self, r2: Array) -> Array:
        return -0.5 * np.exp(-0.5 * r2)


class Matern32(_StationaryARD):
    r"""Matérn-3/2 kernel :math:`v (1 + \sqrt3 r) e^{-\sqrt3 r}`."""

    def _kappa(self, r2: Array) -> Array:
        s = np.sqrt(3.0 * r2)
        return (1.0 + s) * np.exp(-s)

    def _dkappa(self, r2: Array) -> Array:
        # d/dr kappa = -3 r e^{-sqrt3 r}  =>  d/d(r^2) = -1.5 e^{-sqrt3 r}
        return -1.5 * np.exp(-np.sqrt(3.0 * r2))


class Matern52(_StationaryARD):
    r"""Matérn-5/2 kernel :math:`v (1 + \sqrt5 r + \tfrac53 r^2) e^{-\sqrt5 r}`."""

    def _kappa(self, r2: Array) -> Array:
        s = np.sqrt(5.0 * r2)
        return (1.0 + s + 5.0 / 3.0 * r2) * np.exp(-s)

    def _dkappa(self, r2: Array) -> Array:
        # d/dr kappa = -(5/3) r (1 + sqrt5 r) e^{-sqrt5 r}
        s = np.sqrt(5.0 * r2)
        return -(5.0 / 6.0) * (1.0 + s) * np.exp(-s)


class RationalQuadratic(Kernel):
    r"""Rational-quadratic kernel :math:`v (1 + r^2 / (2\alpha))^{-\alpha}`.

    A scale mixture of RBF kernels; isotropic lengthscale.  Parameters (log):
    ``lengthscale``, ``alpha``, ``variance``.
    """

    def __init__(
        self,
        lengthscale: float = 1.0,
        alpha: float = 1.0,
        variance: float = 1.0,
        lengthscale_bounds: tuple[float, float] = (1e-2, 1e2),
        alpha_bounds: tuple[float, float] = (1e-2, 1e2),
        variance_bounds: tuple[float, float] = (1e-2, 1e2),
    ) -> None:
        self._t = np.log([lengthscale, alpha, variance]).astype(float)
        self._b = np.log(np.array([lengthscale_bounds, alpha_bounds, variance_bounds], dtype=float))
        self.param_names = ["lengthscale", "alpha", "variance"]

    @property
    def theta(self) -> Array:
        return self._t.copy()

    @theta.setter
    def theta(self, value: ArrayLike) -> None:
        self._t = np.asarray(value, dtype=float).copy()

    @property
    def bounds(self) -> Array:
        return self._b.copy()

    def _sq(self, X1: Array, X2: Array) -> Array:
        d2 = (X1 * X1).sum(1)[:, None] + (X2 * X2).sum(1)[None, :] - 2.0 * X1 @ X2.T
        return np.maximum(d2, 0.0)

    def __call__(self, X1: ArrayLike, X2: ArrayLike | None = None) -> Array:
        X1 = _as_2d(X1)
        X2 = X1 if X2 is None else _as_2d(X2)
        ls, a, v = np.exp(self._t)
        return v * (1.0 + self._sq(X1, X2) / (2.0 * a * ls**2)) ** (-a)

    def diag(self, X: ArrayLike) -> Array:
        return np.full(_as_2d(X).shape[0], np.exp(self._t[2]))

    def gradient(self, X: ArrayLike) -> tuple[Array, Array]:
        X = _as_2d(X)
        ls, a, v = np.exp(self._t)
        d2 = self._sq(X, X)
        base = 1.0 + d2 / (2.0 * a * ls**2)
        K = v * base ** (-a)
        # dK/dlog l = K * d2 / (l^2 base)
        dls = K * d2 / (ls**2 * base)
        # dK/dlog a = a * K * ( -log(base) + d2 / (2 a l^2 base) )
        da = a * K * (-np.log(base) + d2 / (2.0 * a * ls**2 * base))
        return K, np.stack([dls, da, K])

    def grad_x1(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        X1, X2 = _as_2d(X1), _as_2d(X2)
        ls, a, v = np.exp(self._t)
        diff = X1[:, None, :] - X2[None, :, :]
        d2 = (diff * diff).sum(-1)
        base = 1.0 + d2 / (2.0 * a * ls**2)
        # dK/dd2 = -a v base^{-a-1} / (2 a l^2) ; dd2/dx = 2 diff
        g = -v * base ** (-a - 1.0) / ls**2
        return g[..., None] * diff


class Periodic(Kernel):
    r"""Periodic kernel (MacKay 1998), applied per input dimension:

    .. math:: k(x, x') = v \exp\Big(-\frac{2}{l^2} \sum_d \sin^2\big(\pi (x_d - x'_d) / p\big)\Big).

    For 1-D inputs this equals scikit-learn's ``ExpSineSquared``.  The
    per-dimension sum (a product of 1-D periodic kernels) is used instead of
    ``sin^2(pi ||x - x'|| / p)`` because the latter is **not** positive
    semi-definite in more than one dimension.
    Parameters (log): ``lengthscale``, ``period``, ``variance``.
    """

    def __init__(
        self,
        lengthscale: float = 1.0,
        period: float = 1.0,
        variance: float = 1.0,
        lengthscale_bounds: tuple[float, float] = (1e-2, 1e2),
        period_bounds: tuple[float, float] = (1e-2, 1e2),
        variance_bounds: tuple[float, float] = (1e-2, 1e2),
    ) -> None:
        self._t = np.log([lengthscale, period, variance]).astype(float)
        self._b = np.log(np.array([lengthscale_bounds, period_bounds, variance_bounds], dtype=float))
        self.param_names = ["lengthscale", "period", "variance"]

    @property
    def theta(self) -> Array:
        return self._t.copy()

    @theta.setter
    def theta(self, value: ArrayLike) -> None:
        self._t = np.asarray(value, dtype=float).copy()

    @property
    def bounds(self) -> Array:
        return self._b.copy()

    def _parts(self, X1: Array, X2: Array) -> tuple[Array, Array, Array]:
        ls, p, v = np.exp(self._t)
        arg = np.pi * (X1[:, None, :] - X2[None, :, :]) / p  # (m, n, d)
        S = (np.sin(arg) ** 2).sum(-1)
        return v * np.exp(-2.0 * S / ls**2), arg, S

    def __call__(self, X1: ArrayLike, X2: ArrayLike | None = None) -> Array:
        X1 = _as_2d(X1)
        X2 = X1 if X2 is None else _as_2d(X2)
        return self._parts(X1, X2)[0]

    def diag(self, X: ArrayLike) -> Array:
        return np.full(_as_2d(X).shape[0], np.exp(self._t[2]))

    def gradient(self, X: ArrayLike) -> tuple[Array, Array]:
        X = _as_2d(X)
        ls = np.exp(self._t[0])
        K, arg, S = self._parts(X, X)
        dls = K * 4.0 * S / ls**2
        # dS/dlog p = -sum_d arg_d sin(2 arg_d)
        dp = K * (2.0 / ls**2) * (arg * np.sin(2.0 * arg)).sum(-1)
        return K, np.stack([dls, dp, K])

    def grad_x1(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        X1, X2 = _as_2d(X1), _as_2d(X2)
        ls, p, _ = np.exp(self._t)
        K, arg, _ = self._parts(X1, X2)
        return -K[..., None] * (2.0 * np.pi / (p * ls**2)) * np.sin(2.0 * arg)


# ----------------------------------------------------------------------------
# Composite kernels
# ----------------------------------------------------------------------------
class _Composite(Kernel):
    def __init__(self, k1: Kernel, k2: Kernel) -> None:
        self.k1, self.k2 = k1, k2

    @property
    def param_names(self) -> list[str]:  # type: ignore[override]
        return [f"k1.{n}" for n in self.k1.param_names] + [f"k2.{n}" for n in self.k2.param_names]

    @property
    def theta(self) -> Array:
        return np.concatenate([self.k1.theta, self.k2.theta])

    @theta.setter
    def theta(self, value: ArrayLike) -> None:
        value = np.asarray(value, dtype=float)
        p1 = self.k1.n_params
        self.k1.theta = value[:p1]
        self.k2.theta = value[p1:]

    @property
    def bounds(self) -> Array:
        return np.vstack([self.k1.bounds, self.k2.bounds])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        op = "+" if isinstance(self, SumKernel) else "*"
        return f"({self.k1!r} {op} {self.k2!r})"


class SumKernel(_Composite):
    """``k1(x, x') + k2(x, x')``."""

    def __call__(self, X1: ArrayLike, X2: ArrayLike | None = None) -> Array:
        return self.k1(X1, X2) + self.k2(X1, X2)

    def diag(self, X: ArrayLike) -> Array:
        return self.k1.diag(X) + self.k2.diag(X)

    def gradient(self, X: ArrayLike) -> tuple[Array, Array]:
        K1, d1 = self.k1.gradient(X)
        K2, d2 = self.k2.gradient(X)
        return K1 + K2, np.concatenate([d1, d2], axis=0)

    def grad_x1(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        return self.k1.grad_x1(X1, X2) + self.k2.grad_x1(X1, X2)


class ProductKernel(_Composite):
    """``k1(x, x') * k2(x, x')``."""

    def __call__(self, X1: ArrayLike, X2: ArrayLike | None = None) -> Array:
        return self.k1(X1, X2) * self.k2(X1, X2)

    def diag(self, X: ArrayLike) -> Array:
        return self.k1.diag(X) * self.k2.diag(X)

    def gradient(self, X: ArrayLike) -> tuple[Array, Array]:
        K1, d1 = self.k1.gradient(X)
        K2, d2 = self.k2.gradient(X)
        return K1 * K2, np.concatenate([d1 * K2[None], K1[None] * d2], axis=0)

    def grad_x1(self, X1: ArrayLike, X2: ArrayLike) -> Array:
        K1, K2 = self.k1(X1, X2), self.k2(X1, X2)
        return self.k1.grad_x1(X1, X2) * K2[..., None] + K1[..., None] * self.k2.grad_x1(X1, X2)


def make_kernel(name: str, dim: int, ard: bool = True, **kwargs: float) -> Kernel:
    """Construct a named kernel for ``dim``-dimensional inputs.

    ``name`` is one of ``rbf``, ``matern32``, ``matern52``, ``rq``, ``periodic``.
    With ``ard=True`` the stationary kernels get one lengthscale per input.
    """
    name = name.lower()
    ls = np.full(dim if ard else 1, kwargs.pop("lengthscale", 0.3))
    table = {"rbf": RBF, "matern32": Matern32, "matern52": Matern52}
    if name in table:
        return table[name](lengthscale=ls, **kwargs)
    if name == "rq":
        return RationalQuadratic(lengthscale=float(ls[0]), **kwargs)
    if name == "periodic":
        return Periodic(lengthscale=float(ls[0]), **kwargs)
    raise ValueError(f"unknown kernel {name!r}")
