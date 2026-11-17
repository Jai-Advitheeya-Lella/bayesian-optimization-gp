"""Turn saved experiment results into figures (assets/) and Markdown tables (results/).

    python scripts/plot_results.py benchmarks
    python scripts/plot_results.py batch
    python scripts/plot_results.py hgb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _style as S
from bogp import get_benchmark

ASSETS = Path("assets")
RESULTS = Path("results")
PROBLEMS = ["branin", "hartmann6", "ackley5", "levy5"]
ORDER = ["ei", "ts", "pi", "ucb-0.2", "ucb", "skopt", "tpe", "random"]


def _band(ax, x, R, color, label, lw=2.0, alpha=0.14):
    med = np.median(R, 0)
    lo, hi = np.percentile(R, 25, 0), np.percentile(R, 75, 0)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0)
    ax.plot(x, med, color=color, lw=lw, label=label)


def _fmt(v: float) -> str:
    return f"{v:.2e}" if v < 1e-2 else f"{v:.3f}"


def benchmarks() -> None:
    d = np.load(RESULTS / "benchmarks.npz")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    rows = []
    for ax, pb in zip(axes.ravel(), PROBLEMS):
        b = get_benchmark(pb)
        for m in ORDER[::-1]:
            key = f"{pb}/{m}"
            if key not in d:
                continue
            Y = d[key]
            R = np.maximum(np.minimum.accumulate(Y, axis=1) - b.f_min, 1e-8)
            x = np.arange(1, Y.shape[1] + 1)
            _band(ax, x, R, S.METHOD_COLORS[m], S.METHOD_LABELS[m],
                  lw=2.4 if m == "ei" else 1.6)
        ax.set_yscale("log")
        ax.axvline(2 * b.dim + 1, color=S.AXIS, lw=1)
        ax.set_title(f"{b.name}  (d = {b.dim})", loc="left")
        ax.set_xlabel("function evaluations")
        ax.set_ylabel("simple regret  $f(x^+) - f^*$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    n_seeds = d[f"{PROBLEMS[0]}/ei"].shape[0]
    fig.suptitle(f"Simple regret, median and interquartile range over {n_seeds} seeds "
                 "(vertical line: end of Sobol initial design)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(ASSETS / "regret_benchmarks.png", dpi=120)
    plt.close(fig)

    # markdown table: median [IQR] final regret, mean wall-clock seconds per run
    hdr = "| Method | " + " | ".join(get_benchmark(p).name for p in PROBLEMS) + " |"
    lines = [hdr, "|" + "---|" * (len(PROBLEMS) + 1)]
    for m in ORDER:
        cells = []
        for pb in PROBLEMS:
            key = f"{pb}/{m}"
            if key not in d:
                cells.append("-")
                continue
            r = d[key].min(1) - get_benchmark(pb).f_min
            q1, q2, q3 = np.percentile(r, [25, 50, 75])
            t = d[f"time/{key}"].mean()
            cells.append(f"{_fmt(q2)} [{_fmt(q1)}, {_fmt(q3)}] ({t:.1f}s)")
        lines.append(f"| {S.METHOD_LABELS[m]} | " + " | ".join(cells) + " |")
    budgets = {pb: d[f"{pb}/ei"].shape[1] for pb in PROBLEMS}
    note = ("Final simple regret, median [IQR] over seeds; budgets "
            + ", ".join(f"{get_benchmark(p).name}={n}" for p, n in budgets.items())
            + "; mean wall-clock seconds per run in parentheses (1 CPU thread).")
    (RESULTS / "benchmarks_table.md").write_text(note + "\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))

    # mean rank of final regret across problems
    ranks = {m: [] for m in ORDER}
    for pb in PROBLEMS:
        med = {m: np.median(d[f"{pb}/{m}"].min(1)) for m in ORDER if f"{pb}/{m}" in d}
        for i, m in enumerate(sorted(med, key=med.get)):
            ranks[m].append(i + 1)
    print("mean rank:", {m: np.mean(v) for m, v in ranks.items() if v})


BATCH_COLORS = {
    "seq-EI (q=1)": S.CAT[0], "KB-EI (q=4)": S.CAT[1], "CL-min-EI (q=4)": S.CAT[2],
    "Fantasy-EI (q=4)": S.CAT[6], "Batch-TS (q=4)": S.CAT[3], "random (q=4)": S.MUTED,
}


def batch() -> None:
    d = np.load(RESULTS / "batch.npz")
    n_init = int(d["n_init"])
    problems = sorted({k.split("/")[0] for k in d.files if "/" in k and not k.startswith("time")},
                      key=lambda p: ["hartmann6", "levy5", "branin", "ackley5"].index(p))
    fig, axes = plt.subplots(2, len(problems), figsize=(5.6 * len(problems), 7.4), squeeze=False)
    lines = ["| Method | " + " | ".join(f"{get_benchmark(p).name}: regret after 10 rounds"
                                         for p in problems) + " | "
             + " | ".join(f"{get_benchmark(p).name}: regret after 100 evals" for p in problems) + " |",
             "|" + "---|" * (2 * len(problems) + 1)]
    table: dict[str, list[str]] = {m: [] for m in BATCH_COLORS}
    table2: dict[str, list[str]] = {m: [] for m in BATCH_COLORS}
    for j, pb in enumerate(problems):
        b = get_benchmark(pb)
        for m, c in BATCH_COLORS.items():
            key = f"{pb}/{m}"
            if key not in d:
                continue
            Y = d[key]
            R = np.maximum(np.minimum.accumulate(Y, axis=1) - b.f_min, 1e-8)
            q = 1 if "q=1" in m else 4
            # regret vs evaluations
            _band(axes[1, j], np.arange(1, Y.shape[1] + 1), R, c, m, lw=1.8)
            # regret vs rounds (round 0 = after the initial design)
            idx = n_init - 1 + q * np.arange(0, (Y.shape[1] - n_init) // 4 + 1)
            idx = idx[idx < Y.shape[1]]
            _band(axes[0, j], np.arange(len(idx)), R[:, idx], c, m, lw=1.8)
            r10 = R[:, n_init - 1 + 10 * q]
            table[m].append(f"{_fmt(np.median(r10))}")
            table2[m].append(f"{_fmt(np.median(R[:, -1]))}")
        for i, xl in enumerate(["parallel rounds after the initial design", "function evaluations"]):
            ax = axes[i, j]
            ax.set_yscale("log")
            ax.set_xlabel(xl)
            ax.set_ylabel("simple regret")
        axes[0, j].set_title(f"{b.name}: regret vs wall-clock rounds", loc="left")
        axes[1, j].set_title(f"{b.name}: regret vs evaluations", loc="left")
    for m in BATCH_COLORS:
        if table[m]:
            lines.append(f"| {m} | " + " | ".join(table[m]) + " | " + " | ".join(table2[m]) + " |")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0))
    n_seeds = d[next(k for k in d.files if k.startswith(problems[0] + "/"))].shape[0]
    fig.suptitle(f"Batch BO with q = 4 parallel evaluations: median and IQR over {n_seeds} seeds",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(ASSETS / "batch_bo.png", dpi=120)
    plt.close(fig)
    (RESULTS / "batch_table.md").write_text("Median simple regret.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


HGB_LABELS = {"ei": "BO-EI (ours)", "ts": "BO-Thompson (ours)", "skopt": "scikit-optimize GP-EI",
              "tpe": "Optuna TPE", "random": "Random search"}


def hgb() -> None:
    res = json.loads((RESULTS / "hgb_tuning.json").read_text())
    runs = res["runs"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    lines = ["| Method | best 3-fold CV RMSE | test RMSE of chosen config | seeds beating the default CV score | median evals to do so |",
             "|---|---|---|---|---|"]
    for m in ["random", "tpe", "skopt", "ts", "ei"]:
        rs = [r for r in runs if r["method"] == m]
        if not rs:
            continue
        Y = np.array([r["y"] for r in rs])
        B = np.minimum.accumulate(Y, axis=1)
        _band(ax, np.arange(1, Y.shape[1] + 1), B, S.METHOD_COLORS[m], HGB_LABELS[m],
              lw=2.4 if m == "ei" else 1.6)
        te = np.array([r["test_rmse"] for r in rs])
        beat = [int(np.argmax(b < res["default_cv_rmse"])) + 1 if (b < res["default_cv_rmse"]).any()
                else np.nan for b in B]
        n_beat = int(np.sum(~np.isnan(beat)))
        med_beat = f"{np.nanmedian(beat):.0f}" if n_beat else "-"
        lines.append(f"| {HGB_LABELS[m]} | {B[:, -1].mean():.4f} ± {B[:, -1].std():.4f} | "
                     f"{te.mean():.4f} ± {te.std():.4f} | {n_beat}/{len(rs)} | {med_beat} |")
    lines.append(f"| scikit-learn defaults (lr=0.1, 31 leaves) | {res['default_cv_rmse']:.4f} | "
                 f"{res['default_test_rmse']:.4f} | - | - |")
    ax.axhline(res["default_cv_rmse"], color=S.INK2, lw=1)
    ax.text(res["budget"], res["default_cv_rmse"], "sklearn defaults ", ha="right", va="bottom",
            color=S.INK2, fontsize=8.5)
    ax.axvline(res["n_init"], color=S.AXIS, lw=1)
    ax.set_xlabel("hyperparameter configurations evaluated")
    ax.set_ylabel("best 3-fold CV RMSE so far")
    lo = min(min(r["y"]) for r in runs)
    ax.set_ylim(lo - 0.005, res["default_cv_rmse"] + 0.06)
    n_seeds = len([r for r in runs if r["method"] == "ei"])
    ax.set_title(f"Tuning 5 HistGradientBoosting hyperparameters (median, IQR; {n_seeds} seeds)",
                 loc="left", fontsize=10)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS / "hgb_tuning.png", dpi=120)
    plt.close(fig)
    (RESULTS / "hgb_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("which", choices=["benchmarks", "batch", "hgb"])
    args = p.parse_args()
    S.apply()
    {"benchmarks": benchmarks, "batch": batch, "hgb": hgb}[args.which]()


if __name__ == "__main__":
    main()
