"""Acquisition closed forms vs Monte Carlo and gradients vs finite differences."""

import numpy as np
import pytest

from bogp.acquisition import (
    ExpectedImprovement,
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from bogp.batch import mc_qei
from bogp.gp import GaussianProcess
from bogp.kernels import Matern52

rng = np.random.default_rng(3)
X = rng.random((12, 2))
y = np.sin(6 * X[:, 0]) + X[:, 1] ** 2
gp = GaussianProcess(Matern52([0.3, 0.3], 1.0), noise_variance=1e-4).fit(X, y, optimize=False)
Z = rng.random((30, 2))


def _acqs():
    out = [ExpectedImprovement(), ExpectedImprovement(xi=0.05), LogExpectedImprovement(),
           ProbabilityOfImprovement(), UpperConfidenceBound(), UpperConfidenceBound(beta=4.0)]
    for a in out:
        a.update(t=len(y), best=float(y.min()), dim=2, y_scale=float(y.std()))
    return out


def test_ei_matches_monte_carlo():
    a = _acqs()[0]
    mu, var = gp.predict(Z[:5])
    draws = mu + np.sqrt(var) * np.random.default_rng(0).standard_normal((200000, 5))
    mc = np.maximum(y.min() - draws, 0).mean(0)
    np.testing.assert_allclose(a(gp, Z[:5]), mc, rtol=2e-2, atol=2e-4)


def test_pi_matches_monte_carlo():
    a = ProbabilityOfImprovement(xi=0.0)
    a.update(t=len(y), best=float(y.min()), dim=2)
    mu, var = gp.predict(Z[:5])
    draws = mu + np.sqrt(var) * np.random.default_rng(0).standard_normal((200000, 5))
    np.testing.assert_allclose(a(gp, Z[:5]), (draws < y.min()).mean(0), atol=5e-3)


def test_log_ei_is_log_of_ei():
    ei, logei = _acqs()[0], _acqs()[2]
    v, lv = ei(gp, Z), logei(gp, Z)
    mask = v > 1e-200
    np.testing.assert_allclose(np.log(v[mask]), lv[mask], rtol=1e-6, atol=1e-8)


def test_log_ei_finite_far_from_incumbent():
    a = LogExpectedImprovement()
    a.best = 0.0
    vals, dmu, dsig = a.from_moments(np.array([50.0, 200.0]), np.array([1.0, 1.0]))
    assert np.all(np.isfinite(vals)) and np.all(np.isfinite(dmu)) and np.all(np.isfinite(dsig))
    assert vals[1] < vals[0]


@pytest.mark.parametrize("idx", range(6))
def test_acquisition_gradients(idx):
    a = _acqs()[idx]
    v, g = a.value_and_grad(gp, Z)
    np.testing.assert_allclose(v, a(gp, Z), rtol=1e-8, atol=1e-12)
    eps = 1e-6
    for dim in range(2):
        Zp, Zm = Z.copy(), Z.copy()
        Zp[:, dim] += eps
        Zm[:, dim] -= eps
        fd = (a(gp, Zp) - a(gp, Zm)) / (2 * eps)
        np.testing.assert_allclose(g[:, dim], fd, rtol=1e-4, atol=1e-6)


def test_ei_nonnegative_and_zero_at_data():
    a = _acqs()[0]
    assert np.all(a(gp, Z) >= 0)
    assert np.all(a(gp, X) < 1e-2)


def test_ucb_beta_schedule_grows_logarithmically():
    a = UpperConfidenceBound()
    betas = []
    for t in (1, 10, 100, 1000):
        a.update(t=t, best=0.0, dim=3)
        betas.append(a.beta_t())
    assert np.all(np.diff(betas) > 0)
    # 2 log(t^2) increments: beta(10t) - beta(t) = 4 log 10
    np.testing.assert_allclose(np.diff(betas)[1:], 4 * np.log(10), rtol=1e-12)


def test_qei_single_point_reduces_to_ei():
    a = _acqs()[0]
    x = Z[:1]
    np.testing.assert_allclose(mc_qei(gp, x, float(y.min()), n_samples=200000, rng=0),
                               a(gp, x)[0], rtol=3e-2, atol=1e-4)
