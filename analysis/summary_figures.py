#!/usr/bin/env python
"""Draw the README figures from the aggregate tables in results/summary/.

    python analysis/summary_figures.py

Writes figures/*.png. Needs only pandas and matplotlib, and no clue data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary"
OUT = ROOT / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
NEUTRAL = "#a9a8a3"
BLUE = "#2a78d6"     # highlight / reasoning on
ORANGE = "#eb6834"   # broken derivations

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})


def _frame(ax, xmax: float = 1.0) -> None:
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def _title(fig, title: str, subtitle: str) -> None:
    fig.text(0.015, 0.97, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 0.905, subtitle, fontsize=9.5, color=INK2, va="top")


def no_cot(path: Path) -> None:
    df = pd.read_csv(SUMMARY / "no_cot_accuracy.csv").iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.8, bottom=0.11)
    for i, r in df.iterrows():
        lead = r["system"] == "astra-nocot"
        c = BLUE if lead else NEUTRAL
        ax.plot([r.ci_low, r.ci_high], [i, i], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(r.accuracy, i, "o", color=c, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=3)
        ax.text(r.ci_high + 0.015, i, f"{r.accuracy:.1%}", va="center", fontsize=9.5,
                color=INK if lead else INK2, fontweight="bold" if lead else "normal")
    ax.set_yticks(range(len(df)), [m + (" *" if isinstance(n, str) and n else "")
                                   for m, n in zip(df.model, df.note)])
    _frame(ax, 0.85)
    ax.set_xlabel("accuracy, exact match (dot) with 95% CI")
    _title(fig, "Without chain of thought, GPT-6 Astra solves the most clues",
           "100 clues, 4 samples each. Astra's lead over every other model excludes zero "
           "(narrowest: Fable, by 0.009).")
    fig.text(0.015, 0.015, "* run with structured output", fontsize=8.5, color=INK2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def reasoning(path: Path) -> None:
    df = pd.read_csv(SUMMARY / "reasoning_gain.csv").iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 3.9))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.78, bottom=0.2)
    for i, r in df.iterrows():
        ax.plot([r.no_cot, r.reasoning_on], [i, i], color=GRID, linewidth=6,
                solid_capstyle="round", zorder=1)
        ax.plot(r.no_cot, i, "o", color=NEUTRAL, markersize=9, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=3)
        ax.plot(r.reasoning_on, i, "o", color=BLUE, markersize=9, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=3)
        ax.text(r.reasoning_on + 0.02, i, f"{r.reasoning_on:.0%}  (+{r.gain * 100:.0f} pts)",
                va="center", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(df)), [m + (" *" if isinstance(n, str) and n else "")
                                   for m, n in zip(df.model, df.note)])
    _frame(ax, 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("accuracy on the same 50 clues")
    ax.plot([], [], "o", color=NEUTRAL, label="no chain of thought")
    ax.plot([], [], "o", color=BLUE, label="reasoning on")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _title(fig, "With reasoning on, the best models solve 86–94%",
           "Same 50 clues with and without reasoning, 1 sample per clue with reasoning.")
    fig.text(0.015, 0.015, "* high effort; 26% of responses hit the output limit, so a lower bound",
             fontsize=8.5, color=INK2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def derivations(path: Path) -> None:
    df = pd.read_csv(SUMMARY / "derivation_review.csv").iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.subplots_adjust(left=0.3, right=0.97, top=0.76, bottom=0.11)
    for i, r in df.iterrows():
        v = r.broken_share_reviewed
        if v > 0:
            ax.barh(i, v, height=0.6, color=ORANGE)
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9.5,
                color=INK if v > 0 else INK2)
    ax.set_yticks(range(len(df)), [f"{m}   {a:.0%}" for m, a in zip(df.label.str.replace("Claude ", ""), df.accuracy)])
    _frame(ax, 0.5)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel("share of correct answers whose derivation doesn't work, after hand review")
    _title(fig, "The strongest models get right answers for the right reasons",
           "Each correct answer's derivation is machine-checked, and failures were reviewed by "
           "hand.\nModel labels show accuracy with reasoning on.")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    no_cot(OUT / "no_cot_accuracy.png")
    reasoning(OUT / "reasoning_gain.png")
    derivations(OUT / "derivation_quality.png")
    print(f"wrote 3 figures to {OUT}")


if __name__ == "__main__":
    main()
