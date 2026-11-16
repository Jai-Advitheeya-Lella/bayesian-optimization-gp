"""Standard global-optimisation test functions (vectorised over rows).

Definitions follow the Virtual Library of Simulation Experiments
(Surjanovic & Bingham, https://www.sfu.ca/~ssurjano/optimization.html).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Benchmark:
    name: str
    fn: Callable[[Array], Array]
    bounds: Array  # (d, 2)
    f_min: float
    x_min: Array  # one global minimiser

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    def __call__(self, X: Array) -> Array:
        return self.fn(np.atleast_2d(np.asarray(X, dtype=float)))


def forrester(X: Array) -> Array:
    x = X[:, 0]
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def branin(X: Array) -> Array:
    x1, x2 = X[:, 0], X[:, 1]
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


_H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H6_A = np.array(
    [
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ],
    dtype=float,
)
_H6_P = 1e-4 * np.array(
    [
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ]
)


def hartmann6(X: Array) -> Array:
    inner = (_H6_A[None] * (X[:, None, :] - _H6_P[None]) ** 2).sum(-1)  # (n, 4)
    return -(_H6_ALPHA[None] * np.exp(-inner)).sum(-1)


def ackley(X: Array) -> Array:
    d = X.shape[1]
    a, b, c = 20.0, 0.2, 2 * np.pi
    t1 = -a * np.exp(-b * np.sqrt((X**2).sum(1) / d))
    t2 = -np.exp(np.cos(c * X).sum(1) / d)
    return t1 + t2 + a + np.e


def levy(X: Array) -> Array:
    w = 1.0 + (X - 1.0) / 4.0
    t1 = np.sin(np.pi * w[:, 0]) ** 2
    wi = w[:, :-1]
    t2 = ((wi - 1) ** 2 * (1 + 10 * np.sin(np.pi * wi + 1) ** 2)).sum(1)
    wd = w[:, -1]
    t3 = (wd - 1) ** 2 * (1 + np.sin(2 * np.pi * wd) ** 2)
    return t1 + t2 + t3


def get_benchmark(name: str) -> Benchmark:
    """Look up ``forrester``, ``branin``, ``hartmann6``, ``ackley<d>``, ``levy<d>``."""
    name = name.lower()
    if name == "forrester":
        return Benchmark("Forrester", forrester, np.array([[0.0, 1.0]]), -6.020740055, np.array([0.757249]))
    if name == "branin":
        return Benchmark("Branin", branin, np.array([[-5.0, 10.0], [0.0, 15.0]]),
                         0.397887357729738, np.array([np.pi, 2.275]))
    if name == "hartmann6":
        x = np.array([0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054])
        return Benchmark("Hartmann-6", hartmann6, np.tile([0.0, 1.0], (6, 1)), -3.322368011415515, x)
    for prefix, fn, box, xs in (("ackley", ackley, 32.768, 0.0), ("levy", levy, 10.0, 1.0)):
        if name.startswith(prefix):
            d = int(name[len(prefix):] or 5)
            return Benchmark(f"{prefix.capitalize()}-{d}", fn, np.tile([-box, box], (d, 1)),
                             0.0, np.full(d, xs))
    raise ValueError(f"unknown benchmark {name!r}")
