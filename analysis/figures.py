#!/usr/bin/env python
"""The three figures of PLAN §7.9, rendered from analysis/metrics.py's tables.

    python analysis/figures.py [--results-dir results] [--out-dir results] [--dataset nat]

    headline.png   every model's `nocot` accuracy against Astra's, as the paired difference
                   with its cluster-bootstrapped CI. The claim is "every CI excludes zero".
    cot_gain.png   `think` - `nocot` per model, with the ordering that is the pre-stated
                   prediction: the Astra post says Astra's gain is the smallest.
    by_device.png  accuracy by device family, **descriptive**. n per family is 13 or fewer
                   outside charade, so the cells carry their n and no comparison is drawn.

Palette and layout follow the dataviz reference: one accent, ink on a warm off-white, no
gridlines competing with the marks, direct labels rather than a legend where there is room.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.metrics import NAT_FAMILY, all_tables, load_runs  # noqa: E402
from crypticbench.config import RESULTS_DIR, SEED  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#2f7d5a"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
       "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE})
    return plt


def _empty(plt, path: Path, title: str, note: str) -> None:
    """A figure that says why it is empty. Better than no file and much better than a chart
    of nothing that reads as a result."""
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.axis("off")
    ax.text(0, 0.7, title, fontsize=11, color=INK)
    ax.text(0, 0.35, note, fontsize=9, color=INK2, wrap=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def headline_figure(tbl: pd.DataFrame, path: Path) -> None:
    plt = _style()
    if tbl.empty or "diff" not in tbl:
        _empty(plt, path, "Headline: nocot accuracy against Astra",
               str(tbl["note"].iloc[0]) if "note" in tbl and len(tbl) else "no data")
        return
    t = tbl.dropna(subset=["diff"]).sort_values("diff")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, max(2.4, 0.5 * len(t) + 1.4)))
    colours = [BLUE if e else GRID for e in t["excludes_zero"]]
    ax.barh(y, t["diff"], height=0.5, color=colours, zorder=2)
    ax.errorbar(t["diff"], y, xerr=[t["diff"] - t["ci_lo"], t["ci_hi"] - t["diff"]],
                fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
    ax.axvline(0, color=INK, linewidth=1, zorder=4)
    for yi, (d, hi, n) in enumerate(zip(t["diff"], t["ci_hi"], t["n_paired"])):
        ax.text(hi + 0.005, yi, f"{d:+.1%}  (n={int(n)})", va="center", ha="left",
                color=INK, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(t["family"])
    ax.set_xlabel("paired difference in nocot accuracy vs "
                  f"{t['reference'].iloc[0]} (negative = {t['reference'].iloc[0]} ahead), "
                  "95 % bootstrap over answers")
    ax.set_title("Headline — no chain of thought, paired within clue", loc="left", color=INK,
                 fontsize=10.5)
    ax.grid(axis="x", color=GRID, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def cot_gain_figure(tbl: pd.DataFrame, path: Path) -> None:
    plt = _style()
    t = tbl.dropna(subset=["diff"]).sort_values("diff") if "diff" in tbl else pd.DataFrame()
    if t.empty:
        _empty(plt, path, "CoT gain — think minus nocot",
               "no family has both a nocot and a think run on the same clues")
        return
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, max(2.4, 0.5 * len(t) + 1.4)))
    ax.barh(y, t["diff"], height=0.5, color=ORANGE, zorder=2)
    ax.errorbar(t["diff"], y, xerr=[t["diff"] - t["ci_lo"], t["ci_hi"] - t["diff"]],
                fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
    ax.axvline(0, color=INK, linewidth=1, zorder=4)
    for yi, (d, hi, a, b) in enumerate(zip(t["diff"], t["ci_hi"], t["acc_nocot"],
                                           t["acc_think"])):
        ax.text(hi + 0.005, yi, f"{d:+.1%}   {a:.0%} → {b:.0%}", va="center", ha="left",
                color=INK, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(t["family"])
    ax.set_xlabel("think − nocot accuracy, paired within clue, 95 % bootstrap over answers")
    ax.set_title("What letting each model reason buys it", loc="left", color=INK, fontsize=10.5)
    ax.grid(axis="x", color=GRID, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def by_device_figure(tbl: pd.DataFrame, path: Path) -> None:
    plt = _style()
    from matplotlib.colors import LinearSegmentedColormap

    if tbl.empty:
        _empty(plt, path, "Accuracy by device family", "no device-tagged runs")
        return
    t = tbl[tbl["task"] == "nocot"] if (tbl["task"] == "nocot").any() else tbl
    systems = sorted(t["system"].unique())
    devices = sorted(t["device"].unique(), key=lambda d: -int(t[t["device"] == d]["n"].max()))
    mat = np.full((len(systems), len(devices)), np.nan)
    for i, s in enumerate(systems):
        for j, d in enumerate(devices):
            cell = t[(t["system"] == s) & (t["device"] == d)]
            if len(cell):
                mat[i, j] = cell["acc"].iloc[0]
    fig, ax = plt.subplots(figsize=(max(6, 0.95 * len(devices) + 3),
                                    max(2.6, 0.45 * len(systems) + 2.0)))
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    cmap.set_bad(GRID)
    ax.pcolormesh(np.arange(len(devices) + 1) - 0.5, np.arange(len(systems) + 1) - 0.5,
                  np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE,
                  linewidth=2)
    for i in range(len(systems)):
        for j in range(len(devices)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if mat[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(devices)))
    ax.set_xticklabels([f"{d}\n(n={int(t[t['device'] == d]['n'].max())})" for d in devices],
                       fontsize=7.5)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems, fontsize=8)
    ax.set_xlim(-0.5, len(devices) - 0.5)
    ax.set_ylim(-0.5, len(systems) - 0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Accuracy by device family — descriptive; n is too small for any within-family "
                 "comparison (PLAN §5.4)", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render(tables: dict[str, pd.DataFrame], figures_dir: Path) -> list[Path]:
    paths = [figures_dir / "headline.png", figures_dir / "cot_gain.png",
             figures_dir / "by_device.png"]
    headline_figure(tables["headline"], paths[0])
    cot_gain_figure(tables["cot_gain"], paths[1])
    by_device_figure(tables["by_device"], paths[2])
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--out-dir")
    ap.add_argument("--dataset", default="nat")
    a = ap.parse_args()
    results = Path(a.results_dir)
    out = Path(a.out_dir) if a.out_dir else results
    df = load_runs(results)
    # `nat` means the nat family, as in metrics.py: `nocot` on the 100 and the `think` arms
    # on `nat50`'s 50 are one study, and cot_gain() pairs them by clue id. Filtering to the
    # single dataset left cot_gain.png empty on the real run (2026-09-11).
    wanted = NAT_FAMILY if a.dataset == "nat" else (a.dataset,)
    split = df[df["dataset"].isin(wanted)]
    if split.empty:
        sys.exit(f"no runs on dataset {a.dataset!r}")
    rng = np.random.default_rng(SEED)
    paths = render(all_tables(split, rng), out / "figures")
    print("wrote " + ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
