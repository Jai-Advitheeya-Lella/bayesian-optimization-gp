"""GP regression showcase: kernel prior samples and the Mauna Loa CO2 forecast.

The CO2 model follows Rasmussen & Williams (2006, Sec. 5.4.3): a sum of a
long-term smooth trend, a decaying seasonal component, medium-term
irregularities and short-term noise, with all 13 hyperparameters fitted by
maximising the log marginal likelihood (analytic gradients, L-BFGS-B).
We train on 1958-1991 monthly averages and forecast 1992-2001.

    python scripts/gp_regression_demo.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _style as S
from bogp.gp import GaussianProcess
from bogp.kernels import RBF, Matern32, Matern52, Periodic, RationalQuadratic

warnings.filterwarnings("ignore")


def kernel_gallery(out: Path) -> None:
    x = np.linspace(0, 5, 400)[:, None]
    kernels = [
        ("RBF", RBF(0.6)),
        ("Matérn-5/2", Matern52(0.6)),
        ("Matérn-3/2", Matern32(0.6)),
        ("Rational quadratic", RationalQuadratic(0.6, alpha=0.3)),
        ("Periodic", Periodic(1.0, period=1.5)),
        ("RBF × Periodic", RBF(2.0) * Periodic(1.0, period=0.8)),
        ("RBF(long) + RBF(short)", RBF(2.0) + RBF(0.15, variance=0.08)),
        ("Matérn-5/2 × Periodic", Matern52(1.5) * Periodic(0.8, period=1.0)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12, 4.6), sharex=True, sharey=True)
    rng = np.random.default_rng(3)
    for ax, (name, k) in zip(axes.ravel(), kernels):
        K = k(x) + 1e-8 * np.eye(len(x))
        L = np.linalg.cholesky(K)
        draws = (L @ rng.standard_normal((len(x), 3))).T
        for i, d in enumerate(draws):
            ax.plot(x[:, 0], d, color=S.CAT[i], lw=1.5)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_ylim(-3.2, 3.2)
    for ax in axes[1]:
        ax.set_xlabel("x")
    fig.suptitle("Three draws from the GP prior for each kernel (same random seed)",
                 x=0.01, ha="left", fontsize=11, color=S.INK)
    fig.tight_layout()
    fig.savefig(out / "kernel_gallery.png", dpi=120)
    plt.close(fig)
    print("wrote kernel_gallery.png")


def co2_data():
    from sklearn.datasets import fetch_openml

    df = fetch_openml(data_id=41187, as_frame=True, data_home="data").frame
    df["t"] = df["year"] + (df["month"] - 0.5) / 12.0
    m = df.groupby(["year", "month"], as_index=False).agg(t=("t", "first"), co2=("co2", "mean"))
    return m["t"].to_numpy(float)[:, None], m["co2"].to_numpy(float)


def co2_kernel():
    trend = RBF(50.0, variance=5.0, lengthscale_bounds=(5, 1e3), variance_bounds=(1e-2, 1e3))
    season = RBF(100.0, variance=0.05, lengthscale_bounds=(2, 1e3), variance_bounds=(1e-4, 10)) * \
        Periodic(1.0, period=1.0, variance=1.0, lengthscale_bounds=(0.1, 10),
                 period_bounds=(0.5, 2.0), variance_bounds=(0.1, 10))
    medium = RationalQuadratic(1.0, alpha=1.0, variance=0.01, lengthscale_bounds=(0.1, 20),
                               alpha_bounds=(1e-3, 1e3), variance_bounds=(1e-5, 1))
    short = RBF(0.1, variance=1e-3, lengthscale_bounds=(1e-2, 1.0), variance_bounds=(1e-6, 1e-1))
    return trend + season + medium + short


def co2_forecast(out: Path, results: Path) -> None:
    X, y = co2_data()
    split = 1992.0
    tr, te = X[:, 0] < split, X[:, 0] >= split
    t0 = time.perf_counter()
    gp = GaussianProcess(co2_kernel(), noise_variance=1e-4, noise_bounds=(1e-7, 1e-1))
    gp.fit(X[tr], y[tr], optimize=True, n_restarts=3, rng=0)
    fit_time = time.perf_counter() - t0
    mu, var = gp.predict(X[te])
    sd_y = np.sqrt(var + gp.noise_variance * gp.y_std**2)  # predictive for noisy obs
    rmse = float(np.sqrt(np.mean((mu - y[te]) ** 2)))
    cover = float(np.mean(np.abs(mu - y[te]) <= 1.96 * sd_y))

    # scikit-learn with the same kernel structure, as a reference
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process import kernels as sk

    t1 = time.perf_counter()
    skk = (sk.ConstantKernel(5.0, (1e-2, 1e3)) * sk.RBF(50.0, (5, 1e3))
           + sk.ConstantKernel(0.05, (1e-4, 10)) * sk.RBF(100.0, (2, 1e3))
           * sk.ExpSineSquared(1.0, 1.0, (0.1, 10), (0.5, 2.0))
           + sk.ConstantKernel(0.01, (1e-5, 1)) * sk.RationalQuadratic(1.0, 1.0, (0.1, 20), (1e-3, 1e3))
           + sk.ConstantKernel(1e-3, (1e-6, 1e-1)) * sk.RBF(0.1, (1e-2, 1.0))
           + sk.WhiteKernel(1e-4, (1e-7, 1e-1)))
    skgp = GaussianProcessRegressor(skk, normalize_y=True, n_restarts_optimizer=3,
                                    random_state=0).fit(X[tr], y[tr])
    sk_time = time.perf_counter() - t1
    sk_mu = skgp.predict(X[te])
    sk_rmse = float(np.sqrt(np.mean((sk_mu - y[te]) ** 2)))

    res = {
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "ours": {"lml": gp.log_marginal_likelihood(), "test_rmse": rmse, "coverage95": cover,
                 "fit_time_s": fit_time,
                 "hyperparameters": dict(zip(gp.kernel.param_names, np.exp(gp.kernel.theta).tolist())),
                 "noise_variance": gp.noise_variance},
        "sklearn": {"lml": float(skgp.log_marginal_likelihood_value_), "test_rmse": sk_rmse,
                    "fit_time_s": sk_time, "kernel": str(skgp.kernel_)},
    }
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res["ours"].items() if k != "hyperparameters"}, indent=1))
    print("sklearn", res["sklearn"]["lml"], sk_rmse)

    # figure: fit + 10-year forecast + 10 more years of pure extrapolation
    Xg = np.linspace(X[0, 0], 2012, 1200)[:, None]
    mg, vg = gp.predict(Xg)
    sg = np.sqrt(vg + gp.noise_variance * gp.y_std**2)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.1, 1]})
    for a in (ax, ax2):
        a.fill_between(Xg[:, 0], mg - 1.96 * sg, mg + 1.96 * sg, color=S.CAT[0], alpha=0.18, lw=0,
                       label="95% predictive interval")
        a.plot(Xg[:, 0], mg, color=S.CAT[0], lw=1.6, label="GP posterior mean")
        a.scatter(X[tr, 0], y[tr], s=4, color=S.INK, label="training (1958-1991)", zorder=3)
        a.scatter(X[te, 0], y[te], s=6, color=S.CAT[1], label="held-out (1992-2001)", zorder=3)
        a.axvline(split, color=S.AXIS, lw=1)
    ax.set_xlabel("year")
    ax.set_ylabel("CO$_2$ (ppm)")
    ax.set_title(f"Mauna Loa CO$_2$: composite-kernel GP ({len(gp.theta)} hyperparameters, ML-II)",
                 loc="left", fontsize=10)
    ax.legend(loc="upper left")
    ax2.set_xlim(1988, 2003)
    sel = (Xg[:, 0] > 1988) & (Xg[:, 0] < 2003)
    ax2.set_ylim(mg[sel].min() - 4, mg[sel].max() + 4)
    ax2.set_xlabel("year")
    ax2.set_title(f"1992-2001 forecast: RMSE {rmse:.2f} ppm,\n{100 * cover:.0f}% of months inside the 95% band",
                  loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "co2_forecast.png", dpi=120)
    plt.close(fig)
    print("wrote co2_forecast.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("assets"))
    p.add_argument("--results", type=Path, default=Path("results/co2.json"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    S.apply()
    kernel_gallery(args.out)
    co2_forecast(args.out, args.results)


if __name__ == "__main__":
    main()
