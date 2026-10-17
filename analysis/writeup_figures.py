#!/usr/bin/env python
"""Four figures for the results write-up, beyond the three of PLAN §7.9.

    python analysis/writeup_figures.py [--results-dir results] [--out-dir results]

    gain_dumbbell.png     what thinking buys: no-CoT and think accuracy per model on the 50
                          shared clues, joined; Astra emphasised. The line lengths are the
                          CoT gains, and the pre-stated prediction was that Astra's is shortest.
    acc_vs_val.png        right for the right reasons: accuracy against validity on the think
                          rows, with the y = x diagonal; the drop below it is the share of right
                          answers whose derivation does not execute. Sonnet 5's rungs as a path.
    solved_by.png         how many of the eight models solve each clue without chain of
                          thought; the clues solved by exactly one model, by which.
    nat_vs_guardian.png   does the margin travel: each model's no-CoT accuracy on the setter's
                          clues and on 200 public Guardian clues, joined. Exploratory, and the
                          figure says so.

Reads the tables metrics.py wrote (results/tables/*.csv) and, for solved_by, the run CSVs
through metrics.load_runs, so it is subject to the same run selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.figures import BLUE, GRID, INK, INK2, ORANGE, SURFACE, _style  # noqa: E402
from analysis.metrics import by_clue, load_runs  # noqa: E402
from crypticbench.config import RESULTS_DIR  # noqa: E402

GREY = "#a9a8a3"
NAMES = {"astra": "GPT-6 Astra", "fable": "Claude Fable 5.1", "sol": "GPT-5.6 Sol",
         "gpt55": "GPT-5.5", "opus47": "Claude Opus 4.7", "opus5": "Claude Opus 5",
         "sonnet5": "Claude Sonnet 5", "haiku45": "Claude Haiku 4.5"}


def _finish(plt, fig, ax, path: Path, title: str, sub: str | None = None) -> None:
    import textwrap
    lines = textwrap.wrap(sub, 95) if sub else []
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=8 + 12 * len(lines))
    if lines:
        ax.text(0, 1.015, "\n".join(lines), transform=ax.transAxes, color=INK2, fontsize=8.5,
                va="bottom", linespacing=1.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def gain_dumbbell(tables: Path, path: Path) -> None:
    plt = _style()
    t = pd.read_csv(tables / "cot_gain.csv").dropna(subset=["diff"])
    t = t.sort_values("acc_nocot").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(t) + 2.0))
    y = np.arange(len(t))
    for i, r in t.iterrows():
        emph = r["family"] == "astra"
        c = BLUE if emph else GREY
        ax.plot([r["acc_nocot"], r["acc_think"]], [i, i], color=c, linewidth=2.2, zorder=2,
                solid_capstyle="round")
        ax.scatter([r["acc_nocot"]], [i], s=70, color=SURFACE, edgecolor=c, linewidth=2.2,
                   zorder=3)
        ax.scatter([r["acc_think"]], [i], s=70, color=c, edgecolor=c, zorder=3)
        lo, hi = r["diff"] - r["ci_lo"], r["ci_hi"] - r["diff"]
        label = f"+{r['diff']:.2f}  [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
        if r["family"] == "sonnet5":
            label += "  (lower bound: 26 % truncated)"
        ax.text(1.005, i, label, va="center", ha="left", color=INK if emph else INK2,
                fontsize=8.5, fontweight="bold" if emph else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([NAMES.get(f, f) for f in t["family"]], fontsize=9)
    for lab, r in zip(ax.get_yticklabels(), t.itertuples()):
        if r.family == "astra":
            lab.set_fontweight("bold"); lab.set_color(INK)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlabel("accuracy on the 50 clues both conditions scored", color=INK2)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # legend as text, not a box
    ax.scatter([0.02], [len(t) - 0.35], s=55, color=SURFACE, edgecolor=INK2, linewidth=2,
               clip_on=False)
    ax.text(0.045, len(t) - 0.35, "no chain of thought", va="center", fontsize=8.5, color=INK2)
    ax.scatter([0.46], [len(t) - 0.35], s=55, color=INK2, edgecolor=INK2, clip_on=False)
    ax.text(0.485, len(t) - 0.35, "thinking on", va="center", fontsize=8.5, color=INK2)
    ax.set_ylim(-0.6, len(t) - 0.0)
    _finish(plt, fig, ax, path, "What letting a model think buys it",
            "The prediction was that Astra's gain (the line's length) would be the shortest. "
            "It is fourth of five. Paired within clue; 95 % bootstrap over answers.")


def acc_vs_val(tables: Path, path: Path) -> None:
    plt = _style()
    ps = pd.read_csv(tables / "per_system.csv")
    rungs = pd.read_csv(tables / "sonnet_rungs.csv")
    t = ps[ps["task"].isin(["think", "think_cot"])].copy()
    t = t.dropna(subset=["val_lower"])
    # After the hand review (ANALYSIS_PLAN Outcome, 2026-09-11): the dot is the reviewed lower
    # bound, the bar runs to the reviewed upper bound, and a small hollow tick marks where the
    # mechanical lower bound was. If the review table is absent the mechanical numbers stand.
    rv_path = tables / "reviewed_validity.csv"
    reviewed = rv_path.exists()
    if reviewed:
        rv = pd.read_csv(rv_path).set_index("system")
        t["val_mech"] = t["val_lower"]
        t["val_lower"] = t["system"].map(rv["val_lower_reviewed"]).fillna(t["val_lower"])
        t["val_upper"] = t["system"].map(rv["val_upper_reviewed"]).fillna(t["val_upper"])
        rungs["val_lower"] = rungs["system"].map(rv["val_lower_reviewed"]).fillna(rungs["val_lower"])
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.2, zorder=1)
    ax.text(0.655, 0.68, "validity = accuracy", color=INK2, fontsize=8, rotation=45,
            ha="center", va="center")
    # Sonnet 5's effort path, off -> low -> high
    order = {"off": 0, "low": 1, "high": 2}
    r = rungs[rungs["on_effort_axis"]].copy()
    r["o"] = r["rung"].map(order)
    r = r.sort_values("o")
    ax.plot(r["acc"], r["val_lower"], color=ORANGE, linewidth=1.6, linestyle=(0, (3, 2)),
            zorder=2)
    for row in t.itertuples():
        fam, sysname = row.family, row.system
        sonnet = fam == "sonnet5"
        c = ORANGE if sonnet else BLUE
        hollow = sysname == "sonnet5-high"
        ax.plot([row.acc, row.acc], [row.val_lower, row.val_upper], color=c, linewidth=2.4,
                alpha=0.35, zorder=2, solid_capstyle="butt")
        ax.scatter([row.acc], [row.val_lower], s=64, color=SURFACE if hollow else c,
                   edgecolor=c, linewidth=2, zorder=4)
        if reviewed and abs(row.val_mech - row.val_lower) > 1e-9:
            ax.scatter([row.acc], [row.val_mech], s=28, marker="_", color=c, linewidth=1.5,
                       zorder=3)
        if sonnet:
            rung = sysname.split("-")[1]
            lab = {"high": "Sonnet 5, effort high\n(26 % truncated)", "low": "Sonnet 5, low",
                   "off": "Sonnet 5, thinking off", "cot": "Sonnet 5, off + prose CoT"}[rung]
        else:
            lab = NAMES.get(fam, fam)
        offsets = {                      # (dx, dy, ha) in data units, hand-placed
            "gpt55-think": (0.0, 0.145, "center"),
            "sol-think": (0.0, -0.045, "center"),
            "astra-think": (0.02, 0.0, "left"),
            "haiku45-think": (-0.02, 0.0, "right"),
            "sonnet5-off": (0.02, -0.02, "left"),
            "sonnet5-cot": (0.02, 0.02, "left"),
            "sonnet5-low": (0.02, -0.005, "left"),
            "sonnet5-high": (0.02, -0.01, "left"),
        }
        dx, dy, ha = offsets.get(sysname, (0.012, -0.012, "left"))
        ax.text(row.acc + dx, row.val_lower + dy, lab, fontsize=8.2,
                color=INK if not sonnet else "#7a2f10", ha=ha, va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
    ax.set_yticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlabel("accuracy, thinking on (50 clues)", color=INK2)
    ax.set_ylabel("validity: the stated derivation executes", color=INK2)
    ax.grid(color=GRID, zorder=0)
    ax.set_aspect("equal")
    ax.text(0.02, 0.96,
            ("dot = lower bound after the setter's hand review (unscoreable claims counted "
             "invalid);\nbar up to the upper bound (counted valid); tick = mechanical lower "
             "bound before review.\nHollow = ceiling artefact." if reviewed else
             "dot = lower bound (unscoreable claims counted invalid);\n"
             "bar up to the upper bound (counted valid). Hollow = ceiling artefact."),
            fontsize=8, color=INK2, va="top", transform=ax.transAxes)
    _finish(plt, fig, ax, path, "Right for the right reasons?",
            ("After review, every right answer from the three strongest models comes with a "
             "derivation that is valid or unscoreable; the gap below the diagonal survives only "
             "on the weak configurations." if reviewed else
             "The gap below the diagonal is right answers whose derivation does not run. "
             "It narrows as models get stronger."))


def solved_by(results: Path, path: Path) -> None:
    plt = _style()
    df = load_runs(results)
    clues = by_clue(df)
    n = clues[(clues["task"] == "nocot") & (clues["dataset"] == "nat")].copy()
    n["solved"] = n["correct"] > 0.5          # majority of the k samples
    wide = n.pivot_table(index="id", columns="family", values="solved", aggfunc="first")
    wide = wide.fillna(False).astype(bool)
    counts = wide.sum(axis=1)
    fams = list(wide.columns)
    hist = counts.value_counts().reindex(range(len(fams) + 1), fill_value=0)
    with_astra = wide[wide["astra"]].sum(axis=1).value_counts() \
        .reindex(range(len(fams) + 1), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(hist))
    ax.bar(x, hist.values, width=0.72, color=GRID, zorder=2)
    ax.bar(x, with_astra.values, width=0.72, color=BLUE, zorder=3)
    for xi, (tot, wa) in enumerate(zip(hist.values, with_astra.values)):
        if tot:
            ax.text(xi, tot + 0.4, str(int(tot)), ha="center", va="bottom", fontsize=8.5,
                    color=INK)
    # who is the lone solver
    lone = wide[counts == 1]
    lone_by = lone.idxmax(axis=1).value_counts()
    note = "solved by exactly one model: " + ", ".join(
        f"{NAMES.get(f, f)} {c}" for f, c in lone_by.items())
    ax.text(0.99, 0.80, note.replace(": ", ":\n"), transform=ax.transAxes, ha="right",
            va="top", fontsize=8.5, color=INK2, linespacing=1.5)
    ax.text(0.99, 0.97, f"Astra solves {int(wide['astra'].sum())} of 100",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=BLUE,
            fontweight="bold")
    ax.set_ylim(0, hist.values.max() + 3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_xlabel("number of the eight models that solve the clue (majority of 4 samples, no chain of thought)",
                  color=INK2)
    ax.set_ylabel("clues", color=INK2)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", length=0)
    # legend in text
    ax.add_patch(plt.Rectangle((0.40, 0.955), 0.016, 0.045, transform=ax.transAxes, color=BLUE,
                               clip_on=False))
    ax.text(0.425, 0.977, "clues Astra solves", transform=ax.transAxes, va="center",
            fontsize=8.5, color=INK2)
    ax.add_patch(plt.Rectangle((0.40, 0.885), 0.016, 0.045, transform=ax.transAxes, color=GRID,
                               clip_on=False))
    ax.text(0.425, 0.907, "clues Astra does not", transform=ax.transAxes, va="center",
            fontsize=8.5, color=INK2)
    _finish(plt, fig, ax, path, "Who solves what",
            "100 of the setter's clues. Is Astra's lead clues nobody else touches, or "
            "marginal wins everywhere?")


def nat_vs_guardian(tables: Path, path: Path) -> None:
    plt = _style()
    ps = pd.read_csv(tables / "per_system.csv")
    nat = ps[(ps["task"] == "nocot") & (ps["dataset"] == "nat")].set_index("family")["acc"]
    g = pd.read_csv(tables / "guardian_line.csv")
    g = g[g["source"] == "this harness"].copy()
    g["family"] = g["system"].str.replace("-nocot", "", regex=False)
    gu = g.set_index("family")["acc"]
    saha = pd.read_csv(tables / "guardian_line.csv")
    saha = saha[saha["source"] == "published"]["acc"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    fams = [f for f in nat.index if f in gu.index]
    # avoid label collisions on the right: sort and nudge
    ypos = None   # set after spread() is defined below
    def spread(pairs, gap):
        """Push labels apart to `gap`, then recentre the moved cluster on its true values."""
        pos, last = {}, None
        for v, f in sorted(pairs):
            yv = v if last is None else max(v, last + gap)
            pos[f] = yv
            last = yv
        shift = np.mean([pos[f] - v for v, f in pairs])
        return {f: y - shift for f, y in pos.items()}

    ypos = spread([(gu[f], f) for f in fams], 0.028)
    for f in fams:
        emph = f in ("astra", "fable")
        c = {"astra": BLUE, "fable": ORANGE}.get(f, GREY)
        ax.plot([0, 1], [nat[f], gu[f]], color=c, linewidth=2.2 if emph else 1.4, zorder=3 if emph else 2)
        ax.scatter([0, 1], [nat[f], gu[f]], s=28, color=c, zorder=4)
        ax.text(1.03, ypos[f], f"{NAMES.get(f, f)}   {nat[f]:.0%} \u2192 {gu[f]:.0%}",
                va="center", fontsize=8.5, color=INK if emph else INK2,
                fontweight="bold" if emph else "normal")
    ax.axhline(saha, xmin=0.62, xmax=1.0, color=INK2, linewidth=1, linestyle=(0, (2, 2)),
               zorder=1)
    ax.text(1.03, saha - 0.028, f"GPT-4-Turbo, Saha et al. 2025  {saha:.1%}", va="center",
            fontsize=8, color=INK2, style="italic")
    ax.set_xlim(-0.1, 2.0)
    ax.set_ylim(0, 0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["the setter's 100 clues\n(k = 4, pre-registered)",
                        "200 public Guardian clues\n(k = 1, exploratory)"], fontsize=9)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(["0", "20 %", "40 %", "60 %"])
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    _finish(plt, fig, ax, path, "Does the margin travel?",
            "Every model drops about 20 points on professionally set clues. Astra's lead over "
            "Fable shrinks from 8.4 to 2.0 points.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    results = Path(a.results_dir)
    out = Path(a.out_dir) if a.out_dir else results
    tables = results / "tables"
    figs = out / "figures"
    gain_dumbbell(tables, figs / "gain_dumbbell.png")
    acc_vs_val(tables, figs / "acc_vs_val.png")
    solved_by(results, figs / "solved_by.png")
    nat_vs_guardian(tables, figs / "nat_vs_guardian.png")
    eci_scatter(tables, figs / "eci_scatter.png")
    by_component_count(results, figs / "by_component_count.png")
    clue_strip(results, figs / "clue_strip.png")
    dose_curve(tables, figs / "dose_curve.png")
    headline_v2(tables, figs / "headline_v2.png")
    headline_eci(tables, figs / "headline_eci.png")
    headline_eci(tables, figs / "headline_eci_guardian.png", dataset="guardian200")
    headline_eci(tables, figs / "headline_eci_think.png", dataset="think")
    pooled_eci(results, tables, figs / "pooled300_eci.png")
    pooled_headline(results, figs / "pooled300_headline.png")
    print("wrote", ", ".join(str(figs / f) for f in
                             ("gain_dumbbell.png", "acc_vs_val.png", "solved_by.png",
                              "nat_vs_guardian.png", "eci_scatter.png",
                              "by_component_count.png", "clue_strip.png", "dose_curve.png", "headline_v2.png", "headline_eci.png", "headline_eci_guardian.png", "headline_eci_think.png",
                              "pooled300_eci.png",
                              "pooled300_headline.png")))



# ---- exploratory figures added 2026-09-11 ---------------------------------------------------

def eci_scatter(tables: Path, path: Path) -> None:
    """No-CoT accuracy against Epoch's ECI. Exploratory: ECI is fitted locally from Epoch's
    public data (data/external/README.md); the accuracy CIs are the study's own."""
    plt = _style()
    from crypticbench.config import DATA_DIR
    eci = pd.read_csv(DATA_DIR / "external" / "eci_scores_2026-09-11.csv").set_index("family")
    ps = pd.read_csv(tables / "per_system.csv")
    nocot = ps[(ps["task"] == "nocot") & (ps["dataset"] == "nat")].set_index("family")
    think = ps[(ps["task"] == "think") & (ps["dataset"] == "nat50") & (ps["role"] == "think")] \
        .set_index("family")
    fig, ax = plt.subplots(figsize=(8, 5.6))
    for fam, r in nocot.iterrows():
        if fam not in eci.index:
            continue
        e = eci.loc[fam]
        emph = fam == "astra"
        c = BLUE if emph else INK2
        ax.plot([e["eci_ci_low"], e["eci_ci_high"]], [r["acc"], r["acc"]], color=c, alpha=0.35,
                linewidth=2, zorder=2)
        ax.plot([e["eci"], e["eci"]], [r["acc_ci_lo"], r["acc_ci_hi"]], color=c, alpha=0.35,
                linewidth=2, zorder=2)
        ax.scatter([e["eci"]], [r["acc"]], s=70, color=c, zorder=4)
        if fam in think.index:
            ax.scatter([e["eci"]], [think.loc[fam, "acc"]], s=55, color=SURFACE, edgecolor=c,
                       linewidth=1.8, zorder=4)
            ax.plot([e["eci"], e["eci"]], [r["acc"], think.loc[fam, "acc"]], color=c,
                    linewidth=1, linestyle=(0, (2, 2)), zorder=1)
        off = {"opus47": (-0.5, -0.03, "right"), "gpt55": (-0.5, -0.035, "right"),
               "sol": (0.5, -0.045, "left"), "opus5": (0.5, -0.045, "left"),
               "fable": (0.5, 0.03, "left"), "astra": (0.5, 0.03, "left"),
               "sonnet5": (0.5, 0.03, "left"), "haiku45": (0.5, 0.03, "left")}
        dx, dy, ha = off.get(fam, (0.5, 0.02, "left"))
        lab = NAMES.get(fam, fam) + (" ¹" if fam == "fable" else "")
        ax.text(e["eci"] + dx, r["acc"] + dy, lab, fontsize=8.5, ha=ha, va="center",
                color=INK if emph else INK2, fontweight="bold" if emph else "normal")
    ax.set_xlabel("Epoch Capabilities Index", color=INK2)
    ax.set_ylabel("accuracy on the setter's clues", color=INK2)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
    ax.grid(color=GRID, zorder=0)
    ax.scatter([], [], s=70, color=INK2, label="no chain of thought, 100 clues")
    ax.scatter([], [], s=55, color=SURFACE, edgecolor=INK2, linewidth=1.8,
               label="thinking on, 50 clues")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.text(0.99, 0.02, "¹ under a structured-output constraint (write-up §7)",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
    _finish(plt, fig, ax, path, "Cryptics against general capability",
            "Exploratory. Without chain of thought the clues track ECI closely; with thinking "
            "on, the top three converge near the ceiling.")


def by_component_count(results: Path, path: Path) -> None:
    """No-CoT accuracy by the number of fodder components in a checker-verified derivation.

    The count is the mode over every *valid* think-arm derivation of the clue, across systems,
    so it is grounded in the clue text by the checker rather than inferred. It is NOT the `hops`
    column PLAN §5.4 asks for — that is the setter's, from the confirmed parses — and this figure
    is exploratory, restricted to the nat50 clues with at least one valid derivation."""
    plt = _style()
    import collections
    import io
    import contextlib
    from crypticbench.derivation import parse
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_runs(results)
    think = df[df["task"].isin(["think", "think_cot"]) & (df["dataset"] == "nat50")
               & (df["verdict"] == "valid")]

    def ncomp(c):
        try:
            return len(parse(str(c)).fodder)
        except Exception:
            return None
    counts = think.assign(n=think["completion"].map(ncomp)).dropna(subset=["n"])
    modal = counts.groupby("id")["n"].agg(lambda s: collections.Counter(s).most_common(1)[0][0])
    nocot = by_clue(df[(df["task"] == "nocot") & (df["dataset"] == "nat")])
    nocot = nocot[nocot["id"].isin(modal.index)].copy()
    nocot["ncomp"] = nocot["id"].map(modal).astype(int).clip(upper=3)
    fams = ["astra", "fable", "sol", "gpt55", "opus47", "opus5", "sonnet5", "haiku45"]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    xs = sorted(nocot["ncomp"].unique())
    n_per = nocot[nocot["family"] == "astra"].groupby("ncomp").size()
    series = {fam: nocot[nocot["family"] == fam].groupby("ncomp")["correct"].mean()
              for fam in fams}
    ends = sorted((g.values[-1], fam) for fam, g in series.items())
    ypos, last = {}, None
    for v, fam in ends:
        yv = v if last is None else max(v, last + 0.04)
        ypos[fam] = yv
        last = yv
    for fam, g in series.items():
        emph = fam == "astra"
        c = BLUE if emph else GREY
        ax.plot(g.index, g.values, color=c, linewidth=2.4 if emph else 1.3, zorder=3 if emph else 2,
                marker="o", markersize=6 if emph else 4)
        ax.text(xs[-1] + 0.08, ypos[fam], NAMES.get(fam, fam), fontsize=8, va="center",
                color=INK if emph else INK2, fontweight="bold" if emph else "normal")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x}{'+' if x == 3 else ''} hop{'s' if x > 1 else ''}"
                        f"\n(n = {int(n_per.get(x, 0))} clues)" for x in xs], fontsize=9)
    ax.set_xlim(xs[0] - 0.3, xs[-1] + 1.1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlabel("hops: fodder components in a checker-verified derivation of the clue\n"
                  "(mode over valid derivations; a proxy, not the setter's count)", color=INK2)
    ax.set_ylabel("no-CoT accuracy", color=INK2)
    ax.grid(axis="y", color=GRID, zorder=0)
    _finish(plt, fig, ax, path, "No-CoT accuracy by number of hops",
            f"{len(modal)} of the 50 think-arm clues have at least one valid derivation and can "
            "be placed; the other 11 cannot.")


def clue_strip(results: Path, path: Path) -> None:
    """Every clue × every model, no-CoT, mean of k. Clues sorted by how many models solve them."""
    plt = _style()
    import io
    import contextlib
    from matplotlib.colors import LinearSegmentedColormap
    from analysis.figures import SEQ
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_runs(results)
    nocot = by_clue(df[(df["task"] == "nocot") & (df["dataset"] == "nat")])
    wide = nocot.pivot_table(index="id", columns="family", values="correct")
    fams = ["astra", "fable", "sol", "gpt55", "opus47", "opus5", "sonnet5", "haiku45"]
    wide = wide[fams]
    order = wide.assign(s=(wide > 0.5).sum(axis=1), m=wide.mean(axis=1)) \
        .sort_values(["s", "m"], ascending=False).index
    mat = wide.loc[order].to_numpy().T
    fig, ax = plt.subplots(figsize=(12, 3.2))
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    # row 0 of `mat` is Astra; draw it at the top, so flip rows and label bottom-up
    ax.pcolormesh(np.arange(mat.shape[1] + 1), np.arange(mat.shape[0] + 1), mat[::-1],
                  cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=0.6)
    ax.set_yticks(np.arange(len(fams)) + 0.5)
    ax.set_yticklabels([NAMES.get(f, f) for f in fams[::-1]], fontsize=8.5)
    ax.set_xticks([])
    ax.set_xlabel("the 100 clues, from the one every model solves to the ones none does; "
                  "cell = share of 4 samples correct", color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    solved = (wide.loc[order] > 0.5).sum(axis=1).to_numpy()
    # mark where the count of solvers drops
    for k in range(8, 0, -1):
        idx = np.where(solved >= k)[0]
        if len(idx):
            ax.axvline(idx[-1] + 1, color=SURFACE, linewidth=2.5)
    _finish(plt, fig, ax, path, "Every clue, every model, without chain of thought",
            "Dark = solved on every sample, light = never. Clues sorted by how many models solve "
            "them; the right-hand quarter is the 25 no model does.")



def dose_curve(tables: Path, path: Path) -> None:
    """Sonnet 5's effort ladder: accuracy and validity against the rung, the pre-registered
    difference-of-slopes test (§1.4), with validity both mechanical and after the hand review."""
    plt = _style()
    r = pd.read_csv(tables / "sonnet_rungs.csv")
    r = r[r["on_effort_axis"]].sort_values("rung_index")
    rv_path = tables / "reviewed_validity.csv"
    rv = pd.read_csv(rv_path).set_index("system") if rv_path.exists() else None
    x = r["rung_index"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, r["acc"], color=INK, marker="o", linewidth=2.2, label="accuracy")
    ax.plot(x, r["val_lower"], color=ORANGE, marker="o", linewidth=1.6, linestyle=(0, (3, 2)),
            label="validity, mechanical (lower bound)")
    slopes = {"accuracy": np.polyfit(x, r["acc"], 1)[0],
              "validity, mechanical": np.polyfit(x, r["val_lower"], 1)[0]}
    if rv is not None:
        vr = r["system"].map(rv["val_lower_reviewed"]).to_numpy()
        ax.plot(x, vr, color=ORANGE, marker="o", linewidth=2.2, label="validity, after review")
        slopes["validity, after review"] = np.polyfit(x, vr, 1)[0]
    for i, (rung, a) in enumerate(zip(r["rung"], r["acc"])):
        ax.text(x[i], a + 0.035, f"{a:.0%}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{rg}\n{int(tok):,} reasoning tokens" for rg, tok in
                        zip(r["rung"], r["mean_reasoning_tokens"])], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlabel("Sonnet 5 reasoning effort (rung index 0, 1, 3; `high` is 26 % truncated)",
                  color=INK2)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    txt = "slope per rung:  " + "   ".join(f"{k} {v:+.3f}" for k, v in slopes.items())
    ax.text(0.99, 0.03, txt, transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    _finish(plt, fig, ax, path, "Does accuracy rise faster than validity with more thinking?",
            "Mechanically yes, by 0.024 per rung. After the hand review the two slopes are "
            "equal to within 0.003. Direction only: the top rung is a ceiling artefact.")



def headline_v2(tables: Path, path: Path) -> None:
    """The headline as two linked panels: each model's no-CoT accuracy with its interval, and
    the paired difference against Astra with the pre-registered criterion (every interval must
    exclude zero) read off directly. The weakest link is annotated."""
    plt = _style()
    ps = pd.read_csv(tables / "per_system.csv")
    hl = pd.read_csv(tables / "headline.csv").set_index("family")
    nocot = ps[(ps["task"] == "nocot") & (ps["dataset"] == "nat")].set_index("family")
    fams = list(nocot.sort_values("acc", ascending=True).index)   # top = highest, bottom = lowest
    y = np.arange(len(fams))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.55 * len(fams) + 2.2), sharey=True,
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    # left: accuracy
    for yi, fam in zip(y, fams):
        r = nocot.loc[fam]
        emph = fam == "astra"
        c = BLUE if emph else INK2
        ax1.plot([r["acc_ci_lo"], r["acc_ci_hi"]], [yi, yi], color=c, linewidth=2.4,
                 alpha=0.9 if emph else 0.55, solid_capstyle="round", zorder=2)
        ax1.scatter([r["acc"]], [yi], s=70 if emph else 52, color=c, zorder=3)
        ax1.text(r["acc_ci_hi"] + 0.015, yi, f"{r['acc']:.1%}", va="center", fontsize=8.5,
                 color=INK if emph else INK2, fontweight="bold" if emph else "normal")
    ax1.set_yticks(y)
    ax1.set_yticklabels([NAMES.get(f, f) + (" ¹" if f == "fable" else "") for f in fams],
                        fontsize=9)
    for lab, f in zip(ax1.get_yticklabels(), fams):
        if f == "astra":
            lab.set_fontweight("bold"); lab.set_color(INK)
    ax1.set_xlim(0, 0.9)
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax1.set_xticklabels(["0", "20 %", "40 %", "60 %", "80 %"])
    ax1.set_xlabel("no-CoT accuracy, 100 clues, mean of 4 samples per clue\n95 % bootstrap over answers",
                   color=INK2)
    ax1.grid(axis="x", color=GRID, zorder=0)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(axis="y", length=0)
    ax1.set_title("How often each model gets the answer", loc="left", fontsize=10, color=INK)
    # right: paired difference vs Astra
    ax2.axvline(0, color=INK, linewidth=1.2, zorder=3)
    for yi, fam in zip(y, fams):
        if fam == "astra":
            ax2.text(0.004, yi, "reference", va="center", fontsize=8.5, color=BLUE,
                     fontweight="bold")
            continue
        r = hl.loc[fam]
        weakest = fam == hl["ci_hi"].idxmax()
        c = ORANGE if weakest else INK2
        ax2.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=c, linewidth=2.4,
                 alpha=0.9 if weakest else 0.55, solid_capstyle="round", zorder=2)
        ax2.scatter([r["diff"]], [yi], s=52, color=c, zorder=3)
        if weakest:
            ax2.annotate(f"the weakest link: clears zero by {abs(r['ci_hi']):.3f}",
                         xy=(r["ci_hi"], yi + 0.12), xytext=(r["ci_hi"] - 0.02, yi - 0.5),
                         fontsize=8.5, color="#7a2f10", ha="right", va="center",
                         arrowprops=dict(arrowstyle="-", color="#7a2f10", linewidth=0.8))
    ax2.set_xlim(-0.6, 0.15)
    ax2.set_xticks([-0.6, -0.4, -0.2, 0])
    ax2.set_xticklabels(["−60", "−40", "−20", "0 pts"])
    ax2.set_xlabel("paired difference vs Astra, same clues\n"
                   "negative = Astra ahead; the claim needs every interval left of zero",
                   color=INK2)
    ax2.grid(axis="x", color=GRID, zorder=0)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("How far behind Astra, clue by clue", loc="left", fontsize=10, color=INK)
    fig.suptitle("Without chain of thought, GPT-6 Astra leads every model tested",
                 x=0.01, ha="left", fontsize=12, color=INK, y=0.995)
    fig.text(0.01, 0.945, "Pre-registered claim: supported. Seven paired comparisons, seven "
             "intervals excluding zero; the narrowest by nine thousandths of a point.   "
             "¹ under a structured-output constraint.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)



VENDOR = {"astra": "openai", "sol": "openai", "gpt55": "openai",
          "fable": "anthropic", "opus47": "anthropic", "opus5": "anthropic",
          "sonnet5": "anthropic", "haiku45": "anthropic"}
VENDOR_COLOUR = {"openai": "#0f7b5f", "anthropic": "#c8501e"}


def headline_eci(tables: Path, path: Path, dataset: str = "nat") -> None:
    """No-CoT accuracy against Epoch's ECI, in the style of the no-CoT index charts: every
    model a labelled point coloured by vendor, a trend line, Astra and Fable starred, and
    Astra's distance above the trend of the other seven annotated. Exploratory placement of a
    pre-registered number; the ECI scores are fitted locally (data/external/README.md)."""
    plt = _style()
    from crypticbench.config import DATA_DIR
    from scipy import stats
    eci = pd.read_csv(DATA_DIR / "external" / "eci_scores_2026-09-11.csv").set_index("family")
    if dataset == "nat":
        ps = pd.read_csv(tables / "per_system.csv")
        nocot = ps[(ps["task"] == "nocot") & (ps["dataset"] == "nat")].set_index("family")
        n_label, k_label = "the setter's 100 clues (mean of 4 samples)", "100 never-curated clues"
    elif dataset == "think":
        ps = pd.read_csv(tables / "per_system.csv")
        nocot = ps[(ps["task"] == "think") & (ps["role"] == "think")
                   & (ps["dataset"] == "nat50")].set_index("family")
        n_label, k_label = "50 of the setter's clues, thinking on", "50 clues, thinking on"
    else:
        g = pd.read_csv(tables / "guardian_line.csv")
        g = g[g["source"] == "this harness"].copy()
        g["family"] = g["system"].str.replace("-nocot", "", regex=False)
        nocot = g.rename(columns={"ci_lo": "acc_ci_lo", "ci_hi": "acc_ci_hi"}).set_index("family")
        n_label, k_label = ("200 public Guardian clues (Saha et al.'s release, 1 sample each)",
                            "200 public Guardian clues, k = 1")
    fams = [f for f in nocot.index if f in eci.index]
    x = np.array([eci.loc[f, "eci"] for f in fams])
    y = np.array([nocot.loc[f, "acc"] for f in fams])
    rho = stats.spearmanr(x, y).correlation
    r = stats.pearsonr(x, y)[0]
    others = [i for i, f in enumerate(fams) if f != "astra"]
    slope, intercept = np.polyfit(x[others], y[others], 1)
    ia = fams.index("astra")
    pred = slope * x[ia] + intercept
    resid = y[ia] - pred
    fig, ax = plt.subplots(figsize=(9.5, 7))
    xs = np.linspace(x.min() - 1.5, x.max() + 1.5, 50)
    ax.plot(xs, slope * xs + intercept, color=INK2, linestyle=(0, (4, 3)), linewidth=1.3,
            zorder=1)
    if dataset == "think":
        xm = 150.5
        ax.text(xm, slope * xm + intercept + 0.05, f"trend of the {len(others)} other models",
                fontsize=8, color=INK2, va="center", ha="center",
                rotation=np.degrees(np.arctan(slope * (x.max() - x.min() + 8) / 1.0 * 7 / 9.5)))
    else:
        y_end = slope * xs[-1] + intercept - (0.0 if dataset == "nat" else 0.035)
        ax.text(xs[-1] + 0.2, y_end, f"trend of the\n{len(others)} other models", fontsize=8,
                color=INK2, va="center")
    for i, f in enumerate(fams):
        c = VENDOR_COLOUR[VENDOR[f]]
        star = f in ("astra", "fable")
        ax.plot([x[i], x[i]], [nocot.loc[f, "acc_ci_lo"], nocot.loc[f, "acc_ci_hi"]], color=c,
                alpha=0.3, linewidth=1.4, zorder=2, solid_capstyle="round")
        hollow = dataset == "think" and f == "sonnet5"
        ax.scatter([x[i]], [y[i]], s=260 if star else 60, marker="*" if star else "o",
                   color=SURFACE if hollow else c, edgecolor=INK if star else c,
                   linewidth=1.6 if hollow else 0.8, zorder=4)
        off = {"astra": (0.7, 0.0, "left", "bold"), "fable": (0.8, -0.03, "left", "bold"),
               "opus47": (-0.4, 0.0, "right", "normal"), "gpt55": (0.4, -0.06, "left", "normal"),
               "sol": (-0.4, 0.035, "right", "normal"), "opus5": (0.4, 0.0, "left", "normal"),
               "sonnet5": (0.4, 0.0, "left", "normal"), "haiku45": (0.4, 0.0, "left", "normal")}
        if dataset != "nat" and f == "fable":
            off["fable"] = (0.8, 0.0, "left", "bold")
        if dataset == "think":
            off["gpt55"] = (0.4, 0.0, "left", "normal")
            off["sol"] = (0.4, 0.0, "left", "normal")
        dx, dy, ha, w = off[f]
        lab = NAMES.get(f, f) + (" ¹" if f == "fable" else "")
        ax.text(x[i] + dx, y[i] + dy, lab, fontsize=9 if star else 8.5, ha=ha, va="center",
                color=INK, fontweight=w)
    # vendor legend as text
    ax.set_xlabel("Epoch Capabilities Index", color=INK2)
    ax.set_ylabel(("accuracy on " if dataset == "think" else "no-CoT accuracy on ") + n_label,
                  color=INK2)
    ax.set_xlim(x.min() - 3, x.max() + 5)
    if dataset == "guardian200":
        saha = pd.read_csv(tables / "guardian_line.csv")
        saha = saha[saha["source"] == "published"]["acc"].iloc[0]
        ax.axhline(saha, color=INK2, linewidth=1, linestyle=(0, (2, 2)), zorder=1)
        ax.text(x.min() - 2.8, saha + 0.012,
                f"GPT-4-Turbo, Saha et al. 2025, as published: {saha:.1%}",
                fontsize=8, color=INK2, style="italic")
    top = {"nat": 0.85, "think": 1.0}.get(dataset, 0.6)
    ax.set_ylim(0, top)
    ticks = [t for t in (0, 0.2, 0.4, 0.6, 0.8, 1.0) if t <= top]
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0"] + [f"{int(t * 100)} %" for t in ticks[1:]])
    ax.grid(color=GRID, zorder=0)
    ax.text(0.01, 0.02, "vertical bars: 95 % bootstrap interval over answers."
            + ("   hollow: 26 % of calls truncated at the token ceiling, lower bound"
               if dataset == "think" else "   ¹ under a structured-output constraint"),
            transform=ax.transAxes, fontsize=7.5, color=INK2)
    what = "Thinking-on" if dataset == "think" else "No-CoT"
    ax.set_title(f"{what} cryptic accuracy against the Epoch Capabilities Index   "
                 f"(ρ = {rho:.2f}, r = {r:.2f})", loc="left", color=INK, fontsize=11, pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)



def _pooled300(results: Path):
    """Per-clue no-CoT scores on the setter's 100 clues and the 200 Guardian clues together.

    EXPLORATORY. ANALYSIS_PLAN §2.5 keeps `nat` and the calibration sets apart in every
    reported number; this pools them for a picture only. Each clue counts once: `nat` clues
    carry a mean of 4 samples, Guardian clues a single sample, so the two thirds of the pool
    that is Guardian is noisier per clue. Clusters for the bootstrap are (dataset, answer)."""
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_runs(results)
    clues = by_clue(df[(df["task"] == "nocot") & df["dataset"].isin(["nat", "guardian200"])])
    clues["cluster"] = clues["dataset"] + ":" + clues["target"].astype(str)
    wide = clues.pivot_table(index=["dataset", "id"], columns="family", values="correct")
    clusters = clues.drop_duplicates(["dataset", "id"]).set_index(["dataset", "id"])["cluster"]
    clusters = clusters.reindex(wide.index)
    return wide, clusters


def pooled_eci(results: Path, tables: Path, path: Path) -> None:
    plt = _style()
    from crypticbench.config import DATA_DIR, SEED
    from scipy import stats
    from analysis.metrics import boot_ci_clustered
    eci = pd.read_csv(DATA_DIR / "external" / "eci_scores_2026-09-11.csv").set_index("family")
    wide, clusters = _pooled300(results)
    rng = np.random.default_rng(SEED)
    fams = [f for f in wide.columns if f in eci.index]
    x = np.array([eci.loc[f, "eci"] for f in fams])
    y = np.array([wide[f].mean() for f in fams])
    ci = {f: boot_ci_clustered(wide[f].to_numpy(float), clusters.to_numpy(), rng) for f in fams}
    rho = stats.spearmanr(x, y).correlation
    r = stats.pearsonr(x, y)[0]
    others = [i for i, f in enumerate(fams) if f != "astra"]
    slope, intercept = np.polyfit(x[others], y[others], 1)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    xs = np.linspace(x.min() - 1.5, x.max() + 1.5, 50)
    ax.plot(xs, slope * xs + intercept, color=INK2, linestyle=(0, (4, 3)), linewidth=1.3, zorder=1)
    ax.text(xs[-1] + 0.2, slope * xs[-1] + intercept - 0.03, "trend of the\nseven other models",
            fontsize=8, color=INK2, va="center")
    off = {"astra": (0.7, 0.0, "left", "bold"), "fable": (0.8, 0.0, "left", "bold"),
           "opus47": (-0.4, 0.0, "right", "normal"), "gpt55": (0.4, -0.05, "left", "normal"),
           "sol": (-0.4, 0.035, "right", "normal"), "opus5": (0.4, 0.0, "left", "normal"),
           "sonnet5": (0.4, 0.0, "left", "normal"), "haiku45": (0.4, 0.0, "left", "normal")}
    for i, f in enumerate(fams):
        c = VENDOR_COLOUR[VENDOR[f]]
        star = f in ("astra", "fable")
        ax.plot([x[i], x[i]], list(ci[f]), color=c, alpha=0.3, linewidth=1.4, zorder=2,
                solid_capstyle="round")
        ax.scatter([x[i]], [y[i]], s=260 if star else 60, marker="*" if star else "o",
                   color=c, edgecolor=INK if star else c, linewidth=0.8, zorder=4)
        dx, dy, ha, w = off[f]
        ax.text(x[i] + dx, y[i] + dy, NAMES.get(f, f) + (" ¹" if f == "fable" else ""),
                fontsize=9 if star else 8.5, ha=ha, va="center", color=INK, fontweight=w)
    ax.set_xlabel("Epoch Capabilities Index", color=INK2)
    ax.set_ylabel("no-CoT accuracy on 100 personal clues with 200 clues from The Guardian",
                  color=INK2)
    ax.set_xlim(x.min() - 3, x.max() + 5)
    ax.set_ylim(0, 0.7)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(["0", "20 %", "40 %", "60 %"])
    ax.grid(color=GRID, zorder=0)
    ax.text(0.01, 0.02, "vertical bars: 95 % bootstrap over (set, answer) clusters.   "
            "¹ under a structured-output constraint", transform=ax.transAxes, fontsize=7.5,
            color=INK2)
    ax.set_title(f"No-CoT cryptic accuracy against the Epoch Capabilities Index, 300 clues   "
                 f"(ρ = {rho:.2f}, r = {r:.2f})", loc="left", color=INK, fontsize=11, pad=16)
    ax.text(0, 1.012, "Eight frontier models, no chain of thought, on 100 personal clues with "
            "200 clues from The Guardian.",
            transform=ax.transAxes, color=INK2, fontsize=7.5, va="bottom")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def pooled_headline(results: Path, path: Path) -> None:
    """The two-panel headline on the pooled 300: accuracy, and the paired difference vs Astra."""
    plt = _style()
    from crypticbench.config import SEED
    from analysis.metrics import boot_ci_clustered, paired_diff
    wide, clusters = _pooled300(results)
    rng = np.random.default_rng(SEED)
    fams = sorted(wide.columns, key=lambda f: wide[f].mean())
    y = np.arange(len(fams))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.55 * len(fams) + 2.2), sharey=True,
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    diffs = {}
    for yi, f in zip(y, fams):
        emph = f == "astra"
        c = BLUE if emph else INK2
        acc = wide[f].mean()
        lo, hi = boot_ci_clustered(wide[f].to_numpy(float), clusters.to_numpy(), rng)
        ax1.plot([lo, hi], [yi, yi], color=c, linewidth=2.4, alpha=0.9 if emph else 0.55,
                 solid_capstyle="round", zorder=2)
        ax1.scatter([acc], [yi], s=70 if emph else 52, color=c, zorder=3)
        ax1.text(hi + 0.012, yi, f"{acc:.1%}", va="center", fontsize=8.5,
                 color=INK if emph else INK2, fontweight="bold" if emph else "normal")
        if not emph:
            diffs[f] = paired_diff(wide[f], wide["astra"], clusters, rng)
    weakest = max(diffs, key=lambda f: diffs[f]["ci_hi"])
    ax2.axvline(0, color=INK, linewidth=1.2, zorder=3)
    for yi, f in zip(y, fams):
        if f == "astra":
            ax2.text(0.004, yi, "reference", va="center", fontsize=8.5, color=BLUE,
                     fontweight="bold")
            continue
        d = diffs[f]
        c = ORANGE if f == weakest else INK2
        ax2.plot([d["ci_lo"], d["ci_hi"]], [yi, yi], color=c, linewidth=2.4,
                 alpha=0.9 if f == weakest else 0.55, solid_capstyle="round", zorder=2)
        ax2.scatter([d["diff"]], [yi], s=52, color=c, zorder=3)
        if f == weakest:
            word = "clears" if d["ci_hi"] < 0 else "does not clear"
            ax2.annotate(f"the weakest link: {word} zero ({d['ci_hi']:+.3f})",
                         xy=(d["ci_hi"], yi + 0.12), xytext=(d["ci_hi"] - 0.02, yi - 0.5),
                         fontsize=8.5, color="#7a2f10", ha="right", va="center",
                         arrowprops=dict(arrowstyle="-", color="#7a2f10", linewidth=0.8))
    n_excl = sum(d["excludes_zero"] and d["diff"] < 0 for d in diffs.values())
    ax1.set_yticks(y)
    ax1.set_yticklabels([NAMES.get(f, f) + (" ¹" if f == "fable" else "") for f in fams],
                        fontsize=9)
    for lab, f in zip(ax1.get_yticklabels(), fams):
        if f == "astra":
            lab.set_fontweight("bold"); lab.set_color(INK)
    ax1.set_xlim(0, 0.7)
    ax1.set_xticks([0, 0.2, 0.4, 0.6])
    ax1.set_xticklabels(["0", "20 %", "40 %", "60 %"])
    ax1.set_xlabel("no-CoT accuracy on 300 clues (100 at k = 4, 200 at k = 1)\n"
                   "95 % bootstrap over (set, answer) clusters", color=INK2)
    ax1.grid(axis="x", color=GRID, zorder=0)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(axis="y", length=0)
    ax1.set_title("How often each model gets the answer", loc="left", fontsize=10, color=INK)
    ax2.set_xlim(-0.6, 0.15)
    ax2.set_xticks([-0.6, -0.4, -0.2, 0])
    ax2.set_xticklabels(["−60", "−40", "−20", "0 pts"])
    ax2.set_xlabel("paired difference vs Astra, same clues\nnegative = Astra ahead", color=INK2)
    ax2.grid(axis="x", color=GRID, zorder=0)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("How far behind Astra, clue by clue", loc="left", fontsize=10, color=INK)
    fig.suptitle("Without chain of thought, on 300 clues", x=0.01, ha="left", fontsize=12,
                 color=INK, y=0.995)
    fig.text(0.01, 0.955, f"EXPLORATORY: the setter's 100 pooled with the 200 public Guardian "
             f"clues, which the plan reports separately.", fontsize=8.5, color=INK2)
    fig.text(0.01, 0.925, f"{n_excl} of 7 paired intervals exclude zero.   "
             f"¹ under a structured-output constraint.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
