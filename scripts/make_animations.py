"""Hero animations: BO iterations on the 1-D Forrester function and on 2-D Branin.

    python scripts/make_animations.py --out assets
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import io
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from PIL import Image

import _style as S
from bogp import BayesianOptimizer, get_benchmark

warnings.filterwarnings("ignore")


def _frame(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _save_gif(frames: list[Image.Image], path: Path, ms: int, hold_last: int = 4) -> None:
    frames = frames + [frames[-1]] * hold_last
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=ms, loop=0, optimize=True)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


def animate_1d(out: Path, n_iter: int = 11, seed: int = 4) -> None:
    b = get_benchmark("forrester")
    grid = np.linspace(0, 1, 400)[:, None]
    f_true = b(grid)
    opt = BayesianOptimizer(b.bounds, acquisition="ei", n_init=3, seed=seed, noise_bounds=(1e-6, 1e-3))
    X0 = opt.ask(3)
    opt.tell(X0, b(X0))
    frames = []
    for it in range(n_iter):
        x_next = opt.ask(1)
        gp = opt.gp
        mu, var = gp.predict(grid)
        sd = np.sqrt(var)
        ei = opt.acq(gp, grid)
        X, y = opt.from_unit(opt._U), opt._y

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.6), dpi=100, sharex=True,
                                      gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.08})
        ax.fill_between(grid[:, 0], mu - 2 * sd, mu + 2 * sd, color=S.CAT[0], alpha=0.16, lw=0,
                        label=r"posterior $\mu \pm 2\sigma$")
        ax.plot(grid[:, 0], f_true, color=S.MUTED, lw=1.4, label="objective $f$")
        ax.plot(grid[:, 0], mu, color=S.CAT[0], lw=2, label=r"posterior mean $\mu$")
        ax.scatter(X[:-1, 0], y[:-1], s=34, color=S.INK, zorder=4, edgecolor=S.SURFACE, lw=1.2,
                   label="observations")
        ax.scatter(X[-1:, 0], y[-1:], s=60, color=S.CAT[1], zorder=5, edgecolor=S.SURFACE, lw=1.2,
                   label="latest")
        ax.axvline(x_next[0, 0], color=S.CAT[1], lw=1.0, alpha=0.8)
        ax.set_ylim(-11, 19)
        ax.set_ylabel("f(x)")
        ax.legend(loc="upper left", ncol=2, fontsize=8)
        ax.set_title(f"BO with expected improvement: {len(y)} evaluations, "
                     f"best {y.min():.3f} (global min {b.f_min:.3f})", loc="left", fontsize=10)
        ax2.fill_between(grid[:, 0], 0, ei, color=S.CAT[1], alpha=0.25, lw=0)
        ax2.plot(grid[:, 0], ei, color=S.CAT[1], lw=1.6)
        ax2.axvline(x_next[0, 0], color=S.CAT[1], lw=1.0, alpha=0.8)
        ax2.set_ylabel("EI(x)")
        ax2.set_xlabel("x")
        ax2.set_ylim(bottom=0)
        ax2.text(x_next[0, 0], ax2.get_ylim()[1] * 0.92, "  next", color=S.INK2, fontsize=8.5,
                 va="top")
        fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.11)
        frames.append(_frame(fig))
        if it == n_iter - 1:
            fig.savefig(out / "bo_1d_final.png", dpi=120)
        plt.close(fig)
        opt.tell(x_next, b(x_next))
    _save_gif(frames, out / "bo_1d.gif", ms=900)


def animate_2d(out: Path, n_iter: int = 22, seed: int = 0) -> None:
    b = get_benchmark("branin")
    n = 90
    g1 = np.linspace(*b.bounds[0], n)
    g2 = np.linspace(*b.bounds[1], n)
    G1, G2 = np.meshgrid(g1, g2)
    P = np.column_stack([G1.ravel(), G2.ravel()])
    F = b(P).reshape(n, n)
    minima = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
    opt = BayesianOptimizer(b.bounds, acquisition="ei", n_init=5, seed=seed)
    X0 = opt.ask(5)
    opt.tell(X0, b(X0))
    levels = np.geomspace(0.3, 320, 14)
    norm = LogNorm(0.3, 320)
    cmap = S.blues(reverse=True)  # dark = low (good)
    frames = []
    for it in range(n_iter):
        x_next = opt.ask(1)
        gp = opt.gp
        mu, _ = gp.predict(opt.to_unit(P))
        ei = opt.acq(gp, opt.to_unit(P)).reshape(n, n)
        X, y = opt.from_unit(opt._U), opt._y
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), dpi=90, sharey=True)
        panels = [
            (F, "objective (Branin)"),
            (mu.reshape(n, n), "GP posterior mean"),
            (None, "expected improvement"),
        ]
        for ax, (Z, title) in zip(axes, panels):
            if Z is not None:
                ax.contourf(G1, G2, np.clip(Z, 0.3, 320), levels=levels, norm=norm, cmap=cmap)
                ax.contour(G1, G2, np.clip(Z, 0.3, 320), levels=levels, colors=S.SURFACE,
                           linewidths=0.4, alpha=0.6)
            else:
                Ze = ei / max(ei.max(), 1e-300)
                ax.contourf(G1, G2, Ze, levels=np.linspace(0, 1, 12), cmap=S.blues())
            ax.scatter(minima[:, 0], minima[:, 1], marker="x", s=30, color=S.CAT[2], lw=1.6,
                       zorder=5)
            ax.scatter(X[:-1, 0], X[:-1, 1], s=22, color=S.INK, edgecolor=S.SURFACE, lw=1.0, zorder=6)
            ax.scatter(X[-1, 0], X[-1, 1], s=40, color=S.CAT[1], edgecolor=S.SURFACE, lw=1.0, zorder=7)
            ax.scatter(x_next[0, 0], x_next[0, 1], marker="*", s=150, color=S.CAT[1],
                       edgecolor=S.INK, lw=0.6, zorder=8)
            ax.set_title(title, loc="left", fontsize=10)
            ax.grid(False)
            ax.set_xlabel("$x_1$")
        axes[0].set_ylabel("$x_2$")
        fig.suptitle(f"{len(y)} evaluations  |  best f = {y.min():.4f}  "
                     f"(global min {b.f_min:.4f}; three global minimisers marked x)",
                     x=0.01, ha="left", fontsize=10.5, color=S.INK)
        fig.subplots_adjust(left=0.05, right=0.99, top=0.83, bottom=0.14, wspace=0.06)
        frames.append(_frame(fig))
        if it == n_iter - 1:
            fig.savefig(out / "bo_2d_final.png", dpi=110)
        plt.close(fig)
        opt.tell(x_next, b(x_next))
    _save_gif(frames, out / "bo_2d.gif", ms=700)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("assets"))
    p.add_argument("--which", choices=["1d", "2d", "both"], default="both")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    S.apply()
    if args.which in ("1d", "both"):
        animate_1d(args.out)
    if args.which in ("2d", "both"):
        animate_2d(args.out)


if __name__ == "__main__":
    main()
