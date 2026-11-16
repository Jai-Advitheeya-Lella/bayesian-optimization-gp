"""GP posterior / evidence vs scikit-learn, plus gradient and sampling checks."""

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as sk

from bogp.gp import GaussianProcess
from bogp.kernels import RBF, Matern32, Matern52, Periodic

rng = np.random.default_rng(1)
X = rng.uniform(0, 5, size=(25, 2))
y = np.sin(X[:, 0]) * np.cos(0.7 * X[:, 1]) + 0.05 * rng.standard_normal(25)
Xs = rng.uniform(0, 5, size=(40, 2))

X1d = rng.uniform(0, 5, size=(25, 1))
y1d = np.sin(2.0 * X1d[:, 0]) + 0.1 * X1d[:, 0] + 0.05 * rng.standard_normal(25)
Xs1d = rng.uniform(-1, 6, size=(40, 1))

CASES = [
    (RBF([0.9, 1.4], 1.3), sk.ConstantKernel(1.3) * sk.RBF([0.9, 1.4])),
    (Matern32([0.6, 1.1], 0.8), sk.ConstantKernel(0.8) * sk.Matern([0.6, 1.1], nu=1.5)),
    (Matern52([1.2, 0.7], 1.6), sk.ConstantKernel(1.6) * sk.Matern([1.2, 0.7], nu=2.5)),
    (
        RBF(1.0, 0.5) + Periodic(1.2, 3.0, 0.4),
        sk.ConstantKernel(0.5) * sk.RBF(1.0) + sk.ConstantKernel(0.4) * sk.ExpSineSquared(1.2, 3.0),
    ),
    (
        Matern52(1.5, 1.0) * Periodic(0.9, 2.2, 1.0),
        sk.ConstantKernel(1.0) * sk.Matern(1.5, nu=2.5) * sk.ExpSineSquared(0.9, 2.2),
    ),
]


def _data(ours):
    """Periodic cases use 1-D data (sklearn's ExpSineSquared is not PSD in >1-D)."""
    has_per = "period" in " ".join(ours.param_names)
    return (X1d, y1d, Xs1d) if has_per else (X, y, Xs)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("ours,ref", CASES)
def test_posterior_matches_sklearn(ours, ref, normalize):
    X, y, Xs = _data(ours)
    noise = 3e-3
    gp = GaussianProcess(ours.clone(), noise_variance=noise, normalize_y=normalize)
    gp.fit(X, y, optimize=False)
    skgp = GaussianProcessRegressor(ref, alpha=noise, optimizer=None, normalize_y=normalize)
    skgp.fit(X, y)
    mu, var = gp.predict(Xs)
    mu_ref, std_ref = skgp.predict(Xs, return_std=True)
    np.testing.assert_allclose(mu, mu_ref, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(np.sqrt(var), std_ref, rtol=1e-6, atol=1e-8)
    _, cov = gp.predict(Xs, return_cov=True)
    _, cov_ref = skgp.predict(Xs, return_cov=True)
    np.testing.assert_allclose(cov, cov_ref, rtol=1e-6, atol=1e-8)
    # Evidence: sklearn reports it for the (normalised) targets with the same convention.
    np.testing.assert_allclose(gp.log_marginal_likelihood(), skgp.log_marginal_likelihood_value_,
                               rtol=1e-9)


@pytest.mark.parametrize("ours,_", CASES)
def test_lml_gradient_finite_difference(ours, _):
    X, y, _xs = _data(ours)
    gp = GaussianProcess(ours.clone(), noise_variance=1e-2).fit(X, y, optimize=False)
    theta = gp.theta
    lml, grad = gp.log_marginal_likelihood(theta, eval_gradient=True)
    eps = 1e-6
    fd = np.zeros_like(theta)
    for j in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += eps
        tm[j] -= eps
        fd[j] = (gp.log_marginal_likelihood(tp) - gp.log_marginal_likelihood(tm)) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-6)


def test_hyperparameter_optimisation_matches_sklearn_optimum():
    """ML-II from several restarts should reach (at least) sklearn's optimum."""
    gp = GaussianProcess(RBF([1.0, 1.0], 1.0, lengthscale_bounds=(1e-2, 1e2),
                             variance_bounds=(1e-2, 1e2)),
                         noise_variance=1e-2, noise_bounds=(1e-5, 1.0), normalize_y=True)
    gp.fit(X, y, optimize=True, n_restarts=5, rng=0)
    ref = (sk.ConstantKernel(1.0, (1e-2, 1e2)) * sk.RBF([1.0, 1.0], (1e-2, 1e2))
           + sk.WhiteKernel(1e-2, (1e-5, 1.0)))
    skgp = GaussianProcessRegressor(ref, alpha=1e-10, normalize_y=True, n_restarts_optimizer=5,
                                    random_state=0).fit(X, y)
    assert gp.log_marginal_likelihood() >= skgp.log_marginal_likelihood_value_ - 1e-3


@pytest.mark.parametrize("ours,_", CASES)
def test_predictive_gradients(ours, _):
    X, y, Xs = _data(ours)
    gp = GaussianProcess(ours.clone(), noise_variance=1e-3).fit(X, y, optimize=False)
    mu, var, dmu, dvar = gp.predict_with_grad(Xs)
    mu2, var2 = gp.predict(Xs)
    np.testing.assert_allclose(mu, mu2, rtol=1e-10)
    np.testing.assert_allclose(var, var2, rtol=1e-8, atol=1e-12)
    eps = 1e-6
    for dim in range(X.shape[1]):
        Xp, Xm = Xs.copy(), Xs.copy()
        Xp[:, dim] += eps
        Xm[:, dim] -= eps
        (mp, vp), (mm, vm) = gp.predict(Xp), gp.predict(Xm)
        np.testing.assert_allclose(dmu[:, dim], (mp - mm) / (2 * eps), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dvar[:, dim], (vp - vm) / (2 * eps), rtol=1e-4, atol=1e-6)


def test_posterior_samples_have_predicted_moments():
    gp = GaussianProcess(Matern52([1.0, 1.0], 1.0), noise_variance=1e-2).fit(X, y, optimize=False)
    Z = Xs[:6]
    mu, cov = gp.predict(Z, return_cov=True)
    S = gp.sample_posterior(Z, n_samples=40000, rng=0)
    np.testing.assert_allclose(S.mean(0), mu, atol=0.02)
    np.testing.assert_allclose(np.cov(S.T), cov, atol=0.02)


def test_condition_equals_refit_with_fixed_hyperparameters():
    gp = GaussianProcess(RBF([1.0, 1.0], 1.0), noise_variance=1e-3).fit(X[:20], y[:20], optimize=False)
    gp_c = gp.condition(X[20:], y[20:])
    gp_full = GaussianProcess(RBF([1.0, 1.0], 1.0), noise_variance=1e-3)
    gp_full._y_mean, gp_full._y_std = gp._y_mean, gp._y_std
    gp_full.fit(X, y, optimize=False, renormalize=False)
    np.testing.assert_allclose(gp_c.predict(Xs)[0], gp_full.predict(Xs)[0], rtol=1e-8)
    # the original model is untouched
    assert gp.X.shape[0] == 20


def test_interpolates_noise_free_data():
    gp = GaussianProcess(RBF([1.0, 1.0], 1.0), noise_variance=1e-8, optimize_noise=False)
    gp.fit(X, y, optimize=False)
    mu, var = gp.predict(X)
    np.testing.assert_allclose(mu, y, atol=1e-4)
    assert var.max() < 1e-4
