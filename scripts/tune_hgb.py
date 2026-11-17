"""Real use case: tune 5 hyperparameters of a gradient-boosted tree model.

Model: scikit-learn ``HistGradientBoostingRegressor`` (150 trees) on the
California-housing data (10k training rows; the remaining 10.6k rows are a
held-out test set).  Objective: 3-fold CV RMSE.  Search space (encoded box):

    log10 learning_rate   in [-3, 0]
    log2  max_leaf_nodes  in [2, 6]    (rounded to an integer: 4 .. 64)
    log10 min_samples_leaf in [0, 2.3] (rounded: 1 .. 200)
    log10 l2_regularization in [-4, 1]
    max_features          in [0.1, 1]

    python scripts/tune_hgb.py --seeds 10 --budget 40 --workers 4
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

BOUNDS = np.array([[-3.0, 0.0], [2.0, 6.0], [0.0, 2.3], [-4.0, 1.0], [0.1, 1.0]])
NAMES = ["learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "max_features"]
N_INIT = 8


def decode(x: np.ndarray) -> dict:
    return {
        "learning_rate": float(10 ** x[0]),
        "max_leaf_nodes": int(round(2 ** x[1])),
        "min_samples_leaf": int(round(10 ** x[2])),
        "l2_regularization": float(10 ** x[3]),
        "max_features": float(x[4]),
    }


@lru_cache(maxsize=1)
def data():
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    X, y = fetch_california_housing(data_home="data", return_X_y=True)
    return train_test_split(X, y, train_size=10_000, random_state=0)


def model(params: dict):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(max_iter=150, early_stopping=False, random_state=0, **params)


def cv_rmse(x: np.ndarray) -> float:
    from sklearn.model_selection import KFold, cross_val_score

    Xtr, _, ytr, _ = data()
    s = cross_val_score(model(decode(x)), Xtr, ytr, cv=KFold(3, shuffle=True, random_state=0),
                        scoring="neg_root_mean_squared_error")
    return float(-s.mean())


def test_rmse(params: dict) -> float:
    Xtr, Xte, ytr, yte = data()
    pred = model(params).fit(Xtr, ytr).predict(Xte)
    return float(np.sqrt(np.mean((pred - yte) ** 2)))


def objective(X: np.ndarray) -> np.ndarray:
    return np.array([cv_rmse(x) for x in np.atleast_2d(X)])


def run_one(job: tuple[str, int, int]) -> dict:
    method, seed, budget = job
    warnings.filterwarnings("ignore")
    from bogp import BayesianOptimizer
    from bogp.baselines import optuna_tpe, random_search, skopt_gp

    t0 = time.perf_counter()
    if method == "random":
        rng = np.random.default_rng(seed)
        X = BOUNDS[:, 0] + rng.random((budget, 5)) * (BOUNDS[:, 1] - BOUNDS[:, 0])
        y = objective(X)
    elif method in ("ei", "ts"):
        # CV scores are noisy as a function of the (rounded) hyperparameters,
        # so allow a larger learned noise term than for the synthetic benchmarks.
        opt = BayesianOptimizer(BOUNDS, acquisition=method, n_init=N_INIT, seed=seed,
                                noise_bounds=(1e-6, 0.3))
        res = opt.minimize(objective, budget)
        X, y = res.X, res.y
    else:
        Xs: list[np.ndarray] = []

        def f(Z: np.ndarray) -> np.ndarray:
            Xs.extend(np.atleast_2d(Z))
            return objective(Z)

        fn = {"tpe": optuna_tpe, "skopt": skopt_gp}[method]
        y = fn(f, BOUNDS, budget, N_INIT, seed)
        X = np.array(Xs)
    best = decode(X[int(np.argmin(y))])
    return {"method": method, "seed": seed, "y": y.tolist(), "best_params": best,
            "test_rmse": test_rmse(best), "time": time.perf_counter() - t0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", default=["ei", "ts", "skopt", "tpe", "random"])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--budget", type=int, default=40)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("results/hgb_tuning.json"))
    args = p.parse_args()
    data()  # download once before forking
    x_def = np.array([-1.0, np.log2(31), np.log10(20), -4.0, 1.0])
    jobs = [(m, s, args.budget) for m in args.methods for s in range(args.seeds)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        runs = []
        for i, r in enumerate(pool.imap_unordered(run_one, jobs)):
            runs.append(r)
            print(f"[{i + 1}/{len(jobs)}] {r['method']} seed={r['seed']} best_cv={min(r['y']):.4f} "
                  f"test={r['test_rmse']:.4f} ({r['time']:.0f}s)", flush=True)
    default_cv = cv_rmse(x_def)  # x_def encodes the defaults (l2 = 1e-4 ~ 0)
    out = {"runs": runs, "default_cv_rmse": default_cv, "default_test_rmse": test_rmse(decode(x_def)),
           "bounds": BOUNDS.tolist(), "names": NAMES, "budget": args.budget, "n_init": N_INIT,
           "x_default_encoded": x_def.tolist(), "wall_time_s": time.time() - t0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"saved {args.out}; default CV RMSE {default_cv:.4f}")


if __name__ == "__main__":
    main()
