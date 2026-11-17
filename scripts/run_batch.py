"""Batch BO: regret per *round* when q points are evaluated in parallel.

Compares sequential EI (q=1) against q=4 batches chosen by Kriging believer,
constant liar (min), Monte-Carlo fantasies (Snoek et al. 2012) and batch
Thompson sampling, plus random batches.

    python scripts/run_batch.py --problems hartmann6 levy5 --seeds 20 --workers 4
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from bogp import BayesianOptimizer, get_benchmark

N_EVALS = 100
N_INIT = 12
METHODS = {
    # name: (acquisition, q, strategy)
    "seq-EI (q=1)": ("ei", 1, "kb"),
    "KB-EI (q=4)": ("ei", 4, "kb"),
    "CL-min-EI (q=4)": ("ei", 4, "cl-min"),
    "Fantasy-EI (q=4)": ("ei", 4, "fantasy"),
    "Batch-TS (q=4)": ("ts", 4, "ts"),
    "random (q=4)": (None, 4, None),
}


def run_one(job: tuple[str, str, int]) -> tuple[str, str, int, np.ndarray, float]:
    problem, method, seed = job
    warnings.filterwarnings("ignore")
    b = get_benchmark(problem)
    acq, q, strategy = METHODS[method]
    t0 = time.perf_counter()
    if acq is None:
        rng = np.random.default_rng(seed)
        lo, hi = b.bounds[:, 0], b.bounds[:, 1]
        y = b(lo + rng.random((N_EVALS, b.dim)) * (hi - lo))
    else:
        opt = BayesianOptimizer(b.bounds, acquisition=acq, n_init=N_INIT, batch_size=q,
                                batch_strategy=strategy, seed=seed)
        y = opt.minimize(b, N_EVALS).y
    return problem, method, seed, np.asarray(y), time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problems", nargs="+", default=["hartmann6", "levy5"])
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("results/batch.npz"))
    args = p.parse_args()

    jobs = [(pb, m, s) for pb in args.problems for m in args.methods for s in range(args.seeds)]
    jobs.sort(key=lambda j: ("Fantasy" not in j[1], j))
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
    np.savez_compressed(args.out, n_init=N_INIT, **out,
                        **{f"time/{k}": np.array(v) for k, v in times.items()})
    print(f"saved {args.out} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
