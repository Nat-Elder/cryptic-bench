"""The paired bootstrap in analysis/metrics.py (reviewer question 3, answered 2026-09-07).

The reason it exists: marginal intervals at n = 100 are about ±10 points and overlap for
systems that genuinely differ. These tests pin the properties that make the paired headline a
different measurement rather than a restatement of the bars.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.metrics import (  # noqa: E402
    boot_ci,
    boot_ci_clustered,
    cot_gain,
    headline,
    paired_diff,
)


def rng() -> np.random.Generator:
    return np.random.default_rng(20260906)


def frame(scores: dict[str, list[float | None]], *, task: str = "nocot",
          role: str = "nocot", answers: list[str] | None = None) -> pd.DataFrame:
    """Long-form run frame from {system: [per-clue correct]}; None is an errored sample."""
    rows = []
    for system, vals in scores.items():
        for i, v in enumerate(vals):
            rows.append({"system": system, "family": system, "role": role, "task": task,
                         "dataset": "nat", "id": f"cb-{i:04d}",
                         "target": (answers[i] if answers else f"ANS{i:04d}"),
                         "correct": np.nan if v is None else float(v),
                         "device": "charade", "hops": np.nan, "published_date": None})
    return pd.DataFrame(rows)


# ---- the paired difference itself ---------------------------------------------------------

def test_identical_systems_give_a_zero_difference_and_a_zero_width_interval():
    v = pd.Series([1, 0, 1, 1, 0] * 20, dtype=float)
    r = paired_diff(v, v.copy(), pd.Series([f"A{i}" for i in range(len(v))]), rng())
    assert r["diff"] == 0.0
    assert (r["ci_lo"], r["ci_hi"]) == (0.0, 0.0)
    assert not r["excludes_zero"]
    assert r["n_paired"] == len(v)


def test_correlated_pair_gets_a_tighter_interval_than_an_uncorrelated_one():
    """Same marginal accuracies, different agreement — this is what pairing buys, and it is
    invisible to the bars in the figure."""
    n = 200
    g = np.random.default_rng(7)
    a = (g.random(n) < 0.6).astype(float)
    twin = a.copy()
    flip = g.choice(n, size=int(0.05 * n), replace=False)   # 5 % disagreement, same accuracy
    twin[flip] = 1 - twin[flip]
    independent = a.copy()
    g.shuffle(independent)                                   # same accuracy, no per-clue link
    answers = pd.Series([f"A{i}" for i in range(n)])

    close = paired_diff(pd.Series(a), pd.Series(twin), answers, rng())
    far = paired_diff(pd.Series(a), pd.Series(independent), answers, rng())
    close_width = close["ci_hi"] - close["ci_lo"]
    far_width = far["ci_hi"] - far["ci_lo"]
    assert close_width < far_width / 2
    lo, hi = boot_ci(a, rng())
    assert close_width < (hi - lo)      # tighter than either marginal interval


def test_empty_pairing_is_reported_not_raised():
    r = paired_diff(pd.Series([], dtype=float), pd.Series([], dtype=float),
                    pd.Series([], dtype=str), rng())
    assert r["n_paired"] == 0 and not r["excludes_zero"]


# ---- the headline -------------------------------------------------------------------------

def test_headline_orients_every_row_against_the_reference():
    df = pd.concat([frame({"astra": [1] * 10}), frame({"weak": [1] * 3 + [0] * 7})])
    h = headline(df, rng(), reference="astra")
    assert list(h["family"]) == ["weak"]
    r = h.iloc[0]
    assert r["diff"] == pytest.approx(-0.7)      # other minus Astra: negative means Astra ahead
    assert r["excludes_zero"] and r["reference_beats"]


def test_headline_declines_when_the_reference_model_has_no_rows():
    h = headline(frame({"opus5": [1, 0, 1]}), rng(), reference="astra")
    assert len(h) == 1 and "declined" in h.iloc[0]["note"]
    assert "astra" in h.iloc[0]["note"]


def test_headline_uses_nocot_only():
    """A think run must never leak into the headline: the claim is about no-CoT accuracy."""
    df = pd.concat([
        frame({"astra": [0] * 10}),
        frame({"opus5": [0] * 10}),
        frame({"opus5": [1] * 10}, task="think", role="think"),
    ])
    r = headline(df, rng(), reference="astra").iloc[0]
    assert r["acc"] == 0.0 and r["acc_reference"] == 0.0


# ---- the CoT gain -------------------------------------------------------------------------

def test_cot_gain_is_think_minus_nocot_within_family():
    df = pd.concat([
        frame({"opus5": [0, 0, 1, 1] * 10}),
        frame({"opus5": [1, 1, 1, 1] * 10}, task="think", role="think"),
    ])
    r = cot_gain(df, rng()).iloc[0]
    assert r["family"] == "opus5"
    assert r["diff"] == pytest.approx(0.5)
    assert r["acc_nocot"] == pytest.approx(0.5) and r["acc_think"] == 1.0


def test_cot_gain_reports_a_family_with_only_one_arm_rather_than_dropping_it():
    r = cot_gain(frame({"opus5": [1, 0, 1]}), rng()).iloc[0]
    assert r["n_paired"] == 0 and "needs both" in r["note"]


# ---- clustering ---------------------------------------------------------------------------

def test_cluster_bootstrap_widens_when_duplicated_clues_agree():
    """Two clues to the same answer that the model gets both-or-neither carry the information
    of one clue. The clue-level interval assumes two; the by-answer interval should not."""
    base = (rng().random(80) < 0.5).astype(float)
    values = np.concatenate([base, base])
    clusters = np.concatenate([np.arange(80), np.arange(80)])
    clue_lo, clue_hi = boot_ci(values, rng())
    ans_lo, ans_hi = boot_ci_clustered(values, clusters, rng())
    assert (ans_hi - ans_lo) > (clue_hi - clue_lo) * 1.25
    assert abs(((ans_lo + ans_hi) / 2) - values.mean()) < 0.05     # centred, just wider


def test_cluster_bootstrap_falls_back_exactly_when_no_answer_repeats():
    v = (rng().random(50) < 0.4).astype(float)
    assert boot_ci_clustered(v, np.arange(50), rng()) == boot_ci(v, rng())


def test_cluster_bootstrap_handles_an_empty_arm():
    lo, hi = boot_ci_clustered(np.array([]), np.array([]), rng())
    assert np.isnan(lo) and np.isnan(hi)
