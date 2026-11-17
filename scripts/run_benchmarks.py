"""Simple-regret benchmark: our BO variants vs random search, skopt and Optuna TPE.

Example
-------
    python scripts/run_benchmarks.py --problems branin hartmann6 ackley5 levy5 \
        --seeds 20 --workers 4 --out results/benchmarks.npz
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from bogp import BayesianOptimizer, get_benchmark
from bogp.baselines import optuna_tpe, random_search, skopt_gp

BUDGET = {"branin": 50, "hartmann6": 100, "ackley5": 100, "levy5": 100}
METHODS = ["random", "tpe", "skopt", "ei", "pi", "ucb", "ucb-0.2", "ts"]


def run_one(job: tuple[str, str, int]) -> tuple[str, str, int, np.ndarray, float]:
    problem, method, seed = job
    warnings.filterwarnings("ignore")
    b = get_benchmark(problem)
    n = BUDGET.get(problem, 100)
    n_init = 2 * b.dim + 1
    t0 = time.perf_counter()
    if method == "random":
        y = random_search(b, b.bounds, n, seed)
    elif method == "tpe":
        y = optuna_tpe(b, b.bounds, n, n_init, seed)
    elif method == "skopt":
        y = skopt_gp(b, b.bounds, n, n_init, seed)
    else:
        acq, kw = method, {}
        if method == "ucb-0.2":
            acq, kw = "ucb", {"beta_scale": 0.2}
        opt = BayesianOptimizer(b.bounds, acquisition=acq, n_init=n_init, acq_kwargs=kw, seed=seed)
        y = opt.minimize(b, n).y
    return problem, method, seed, np.asarray(y, dtype=float), time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problems", nargs="+", default=list(BUDGET))
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("results/benchmarks.npz"))
    args = p.parse_args()

    jobs = [(pb, m, s) for pb in args.problems for m in args.methods for s in range(args.seeds)]
    # put slow jobs first for better load balancing
    jobs.sort(key=lambda j: (j[1] not in ("ts", "skopt"), j))
    out: dict[str, np.ndarray] = {}
    times: dict[str, list[float]] = {}
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (pb, m, s, y, dt) in enumerate(pool.imap_unordered(run_one, jobs)):
            out.setdefault(f"{pb}/{m}", np.zeros((args.seeds, len(y))))[s] = y
            times.setdefault(f"{pb}/{m}", []).append(dt)
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                print(f"[{i + 1}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out, **{f"time/{k}": np.array(v) for k, v in times.items()})
    print(f"saved {args.out} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
