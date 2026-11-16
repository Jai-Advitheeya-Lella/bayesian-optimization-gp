"""Kernel values vs scikit-learn and analytic gradients vs finite differences."""

import numpy as np
import pytest
from sklearn.gaussian_process import kernels as sk

from bogp.kernels import RBF, Matern32, Matern52, Periodic, RationalQuadratic

rng = np.random.default_rng(0)
X = rng.uniform(-1, 2, size=(9, 3))
X2 = rng.uniform(-1, 2, size=(5, 3))


def all_kernels():
    return {
        "rbf": RBF([0.7, 1.3, 0.4], 1.7),
        "rbf_iso": RBF(0.8, 0.6),
        "m32": Matern32([0.5, 0.9, 1.4], 1.2),
        "m52": Matern52([1.1, 0.3, 0.8], 0.9),
        "rq": RationalQuadratic(0.9, 1.7, 1.3),
        "per": Periodic(0.8, 1.9, 1.1),
        "sum": RBF([0.7, 1.3, 0.4], 1.7) + Periodic(0.8, 1.9, 1.1),
        "prod": Matern52([1.1, 0.3, 0.8], 0.9) * Periodic(0.6, 2.5, 0.7),
        "nested": (RBF(0.5) + Matern32(1.2, 0.3)) * RationalQuadratic(1.5, 0.5),
    }


SKLEARN_EQUIV = {
    "rbf": sk.ConstantKernel(1.7) * sk.RBF([0.7, 1.3, 0.4]),
    "rbf_iso": sk.ConstantKernel(0.6) * sk.RBF(0.8),
    "m32": sk.ConstantKernel(1.2) * sk.Matern([0.5, 0.9, 1.4], nu=1.5),
    "m52": sk.ConstantKernel(0.9) * sk.Matern([1.1, 0.3, 0.8], nu=2.5),
    "rq": sk.ConstantKernel(1.3) * sk.RationalQuadratic(0.9, 1.7),
    "per": sk.ConstantKernel(1.1) * sk.ExpSineSquared(0.8, 1.9),
    "sum": sk.ConstantKernel(1.7) * sk.RBF([0.7, 1.3, 0.4])
    + sk.ConstantKernel(1.1) * sk.ExpSineSquared(0.8, 1.9),
    "prod": sk.ConstantKernel(0.9) * sk.Matern([1.1, 0.3, 0.8], nu=2.5)
    * sk.ConstantKernel(0.7) * sk.ExpSineSquared(0.6, 2.5),
}


PERIODIC = {"per", "sum", "prod"}  # sklearn's ExpSineSquared uses ||x-x'||: compare in 1-D


@pytest.mark.parametrize("name", list(SKLEARN_EQUIV))
def test_matches_sklearn(name):
    k = all_kernels()[name]
    ref = SKLEARN_EQUIV[name]
    A, B = (X[:, :1], X2[:, :1]) if name in PERIODIC else (X, X2)
    if name in PERIODIC and name != "per":
        # rebuild ARD parts with a single input dimension
        k = {"sum": RBF(0.7, 1.7) + Periodic(0.8, 1.9, 1.1),
             "prod": Matern52(1.1, 0.9) * Periodic(0.6, 2.5, 0.7)}[name]
        ref = {"sum": sk.ConstantKernel(1.7) * sk.RBF(0.7)
               + sk.ConstantKernel(1.1) * sk.ExpSineSquared(0.8, 1.9),
               "prod": sk.ConstantKernel(0.9) * sk.Matern(1.1, nu=2.5)
               * sk.ConstantKernel(0.7) * sk.ExpSineSquared(0.6, 2.5)}[name]
    np.testing.assert_allclose(k(A), ref(A), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(k(A, B), ref(A, B), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(k.diag(A), ref.diag(A), rtol=1e-12)


def test_periodic_is_product_of_1d_periodics():
    k = Periodic(0.8, 1.9, 1.0)
    prod = np.ones((len(X), len(X)))
    for d in range(X.shape[1]):
        prod *= k(X[:, d:d + 1])
    np.testing.assert_allclose(k(X), prod, rtol=1e-12)


@pytest.mark.parametrize("name", list(all_kernels()))
def test_theta_gradient_finite_difference(name):
    k = all_kernels()[name]
    K, dK = k.gradient(X)
    np.testing.assert_allclose(K, k(X), rtol=1e-10, atol=1e-12)
    theta = k.theta
    eps = 1e-6
    for j in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += eps
        tm[j] -= eps
        fd = (k.clone_with_theta(tp)(X) - k.clone_with_theta(tm)(X)) / (2 * eps)
        np.testing.assert_allclose(dK[j], fd, rtol=1e-5, atol=1e-7, err_msg=f"param {j}")


@pytest.mark.parametrize("name", list(all_kernels()))
def test_input_gradient_finite_difference(name):
    k = all_kernels()[name]
    G = k.grad_x1(X2, X)  # (m, n, d)
    eps = 1e-6
    for dim in range(X.shape[1]):
        Xp, Xm = X2.copy(), X2.copy()
        Xp[:, dim] += eps
        Xm[:, dim] -= eps
        fd = (k(Xp, X) - k(Xm, X)) / (2 * eps)
        np.testing.assert_allclose(G[..., dim], fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("name", list(all_kernels()))
def test_gram_is_psd_and_symmetric(name):
    K = all_kernels()[name](X)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-9


def test_input_gradient_zero_at_coincident_points():
    for k in all_kernels().values():
        G = k.grad_x1(X[:3], X[:3])
        for i in range(3):
            np.testing.assert_allclose(G[i, i], 0.0, atol=1e-12)


def test_theta_roundtrip_and_bounds_shape():
    for k in all_kernels().values():
        t = k.theta + 0.1
        k.theta = t
        np.testing.assert_allclose(k.theta, t)
        assert k.bounds.shape == (len(t), 2)
