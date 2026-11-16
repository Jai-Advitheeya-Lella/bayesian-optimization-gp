"""bogp -- Gaussian processes and Bayesian optimisation from scratch in NumPy."""

from .acquisition import (
    ExpectedImprovement,
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    make_acquisition,
)
from .batch import mc_qei, select_batch
from .benchmarks import Benchmark, get_benchmark
from .bo import BayesianOptimizer, BOResult
from .gp import GaussianProcess
from .kernels import RBF, Matern32, Matern52, Periodic, RationalQuadratic, make_kernel

__all__ = [
    "BayesianOptimizer",
    "BOResult",
    "Benchmark",
    "ExpectedImprovement",
    "GaussianProcess",
    "LogExpectedImprovement",
    "Matern32",
    "Matern52",
    "Periodic",
    "ProbabilityOfImprovement",
    "RBF",
    "RationalQuadratic",
    "UpperConfidenceBound",
    "get_benchmark",
    "make_acquisition",
    "make_kernel",
    "mc_qei",
    "select_batch",
]

__version__ = "0.1.0"
