"""End-to-end checks of the BO loop, batch strategies and benchmarks."""

import numpy as np
import pytest

from bogp import BayesianOptimizer, get_benchmark
from bogp.batch import STRATEGIES, mc_qei, select_batch
from bogp.acquisition import ExpectedImprovement


@pytest.mark.parametrize("name", ["branin", "hartmann6", "ackley5", "levy5", "forrester"])
def test_benchmark_minimum_values(name):
    b = get_benchmark(name)
    np.testing.assert_allclose(b(b.x_min[None])[0], b.f_min, atol=1e-5)
    # no random point beats the global minimum
    rng = np.random.default_rng(0)
    U = rng.random((20000, b.dim))
    Xr = b.bounds[:, 0] + U * (b.bounds[:, 1] - b.bounds[:, 0])
    assert b(Xr).min() >= b.f_min - 1e-6


def test_branin_all_three_minimisers():
    b = get_benchmark("branin")
    xs = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
    np.testing.assert_allclose(b(xs), b.f_min, atol=1e-5)


@pytest.mark.parametrize("acq", ["ei", "logei", "pi", "ts"])
def test_bo_solves_branin(acq):
    b = get_benchmark("branin")
    res = BayesianOptimizer(b.bounds, acquisition=acq, seed=0).minimize(b, 35)
    assert res.y_best - b.f_min < 5e-2
    # everything stays in the box
    assert np.all(res.X >= b.bounds[:, 0]) and np.all(res.X <= b.bounds[:, 1])


def test_bo_beats_random_on_hartmann6():
    b = get_benchmark("hartmann6")
    res = BayesianOptimizer(b.bounds, acquisition="ei", seed=1).minimize(b, 40)
    rng = np.random.default_rng(1)
    rand = b(rng.random((40, 6))).min()
    assert res.y_best < rand
    assert res.y_best - b.f_min < 0.3


def test_ask_tell_initial_design_is_space_filling():
    b = get_benchmark("branin")
    opt = BayesianOptimizer(b.bounds, n_init=8, seed=0)
    X = opt.ask(8)
    assert X.shape == (8, 2)
    U = opt.to_unit(X)
    # Sobol: each half of each axis gets exactly half of the points
    assert np.all((U < 0.5).sum(0) == 4)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_batch_strategies_return_distinct_points(strategy):
    b = get_benchmark("branin")
    opt = BayesianOptimizer(b.bounds, seed=0, n_init=10)
    X0 = opt.ask(10)
    opt.tell(X0, b(X0))
    gp = opt.fit_model()
    rng = np.random.default_rng(0)
    U = select_batch(gp, ExpectedImprovement(), 4, rng, strategy=strategy, n_fantasies=8)
    assert U.shape == (4, 2)
    dists = np.linalg.norm(U[:, None] - U[None], axis=-1)[np.triu_indices(4, 1)]
    assert dists.min() > 1e-3
    assert np.all((U >= 0) & (U <= 1))


def test_kriging_believer_batch_has_higher_qei_than_repeating_best_point():
    b = get_benchmark("branin")
    opt = BayesianOptimizer(b.bounds, seed=0, n_init=10)
    X0 = opt.ask(10)
    opt.tell(X0, b(X0))
    gp = opt.fit_model()
    rng = np.random.default_rng(0)
    U = select_batch(gp, ExpectedImprovement(), 4, rng, strategy="kb")
    best = float(gp.y.min())
    q_kb = mc_qei(gp, U, best, rng=0)
    q_rep = mc_qei(gp, np.repeat(U[:1], 4, axis=0), best, rng=0)
    assert q_kb > q_rep


def test_batch_minimize_counts_evaluations():
    b = get_benchmark("branin")
    opt = BayesianOptimizer(b.bounds, seed=0, n_init=8, batch_size=4, batch_strategy="kb")
    res = opt.minimize(b, 20)
    assert len(res.y) == 20
