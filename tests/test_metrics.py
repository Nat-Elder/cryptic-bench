"""The tables of PLAN §5, exercised on tools/make_fake_runs.py output.

These check shapes, denominators, bounds and refusals — never the numbers, which are random.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from analysis.figures import render  # noqa: E402
from analysis.metrics import (  # noqa: E402
    ACC_FLOOR,
    NAT_FAMILY,
    current_selectors,
    selection_reason,
    TABLES,
    all_tables,
    by_clue,
    by_device,
    by_hops,
    cutoff_split,
    flip_rate,
    load_runs,
    per_system,
    scored,
    sonnet_rungs,
)

# The setter's corpus is not distributed with the repository (data/README.md); this module
# reads it, so on a checkout without it the module skips rather than fails.
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "corpus.jsonl").exists(),
                                reason="the setter's corpus is not on this machine")


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    out = tmp_path_factory.mktemp("fake_results")
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_fake_runs.py"), str(out)],
                   check=True, cwd=ROOT)
    return out


@pytest.fixture(scope="module")
def df(fake):
    return load_runs(fake)


@pytest.fixture(scope="module")
def nat(df):
    """The `nat` family, as `main()` assembles it: `nocot` on the 100 and the `think` arms on
    the 50 of `nat50`, which is a subset of them and not a separate study (PLAN §6)."""
    return df[df["dataset"].isin(NAT_FAMILY)]


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20260906)


def test_every_table_is_produced(nat, rng):
    tables = all_tables(nat, rng)
    assert set(tables) == set(TABLES)
    assert all(isinstance(t, pd.DataFrame) for t in tables.values())


# ---- run selection: what is in the number and what is set aside --------------------------

def test_two_configurations_of_one_system_yield_one_row_with_the_right_k(nat, rng):
    """The defect this replaced: `sonnet5-nocot` was one row over four configurations, k = 4,
    Acc 0.05 — the 8-token probe, its structured re-run, a v1 control and the v2 run averaged
    into a single number that looked like a measurement. The fake results carry the same shape
    (three current runs plus a superseded prompt and a superseded config) and must give k = 3."""
    ps = per_system(nat, rng).set_index("system")
    assert ps.loc["sonnet5-nocot", "k"] == pytest.approx(3.0)
    assert ps.loc["sonnet5-nocot", "n_samples"] == 300
    assert ps.loc["sonnet5-nocot", "n_clues"] == 100


def test_a_superseded_prompt_or_config_is_set_aside_with_its_reason(fake, capsys):
    load_runs(fake)
    out = capsys.readouterr().out
    assert "run selection" in out
    assert "sonnet5-nocot" in out and "set aside 2" in out
    assert "prompt superseded" in out and "config superseded" in out


def test_a_cut_system_and_a_mock_system_are_not_in_any_table(df):
    """`sonnet5-medium` is cut and `mock` is plumbing; both were in the headline
    before run selection existed."""
    assert "sonnet5-medium" not in set(df["system"])
    assert "mock" not in set(df["system"])


def test_all_puts_everything_back_for_archaeology(fake, capsys):
    everything = load_runs(fake, keep_all=True)
    out = capsys.readouterr().out
    assert "run selection OFF" in out and "Not the measurement" in out
    assert {"sonnet5-medium", "mock"} <= set(everything["system"])
    # the superseded sonnet5 runs are back too, which is what makes the pooled k wrong again
    assert len(everything) > len(load_runs(fake))


def test_the_selectors_are_read_off_models_yaml_not_hardcoded():
    sel = current_selectors()
    assert sel["astra-nocot"]["prompt_version"] == "v2"
    assert sel["sonnet5-high"]["prompt_version"] == "v2"   # both think arms, since the gate
    # --sync and the ':batch' id are the same measurement under two ids, so both are accepted
    assert len(sel["astra-nocot"]["fingerprints"]) == 2
    assert sel["sonnet5-medium"]["cut"] and not sel["astra-nocot"]["cut"]
    assert not sel["haiku45-nocot"]["cut"], "un-cut 2026-09-11"


@pytest.mark.parametrize("manifest,expected", [
    ({"system": "nope"}, "not in configs/models.yaml"),
    ({"system": "mock"}, "mock system"),
    ({"system": "sonnet5-medium"}, "cut —"),
    ({"system": "astra-nocot", "prompt_sha256": "x", "config_fingerprint": "y"},
     "prompt superseded"),
])
def test_every_reason_a_run_is_set_aside(manifest, expected):
    reason = selection_reason(manifest, current_selectors())
    assert reason is not None and expected in reason


def test_a_current_run_is_not_set_aside():
    sel = current_selectors()
    good = {"system": "astra-nocot", "prompt_sha256": sel["astra-nocot"]["prompt_sha256"],
            "config_fingerprint": min(sel["astra-nocot"]["fingerprints"])}
    assert selection_reason(good, sel) is None


# ---- nat50: the think arms run on 50 of the 100 ------------------------------------------

def test_the_think_arms_are_on_the_fifty_and_nocot_on_the_hundred(nat, rng):
    ps = per_system(nat, rng).set_index("system")
    assert ps.loc["sonnet5-nocot", "n_clues"] == 100
    assert ps.loc["sonnet5-nocot", "dataset"] == "nat"
    for system in ("sonnet5-high", "sonnet5-off", "sonnet5-cot", "opus5-think"):
        assert ps.loc[system, "n_clues"] == 50, system
        assert ps.loc[system, "dataset"] == "nat50", system


def test_the_cot_gain_pairs_on_the_fifty_and_says_so(nat, rng):
    from analysis.metrics import cot_gain

    t = cot_gain(nat, rng).set_index("family")
    assert t.loc["opus5", "n_paired"] == 50
    assert t.loc["opus5", "paired_on"] == "nat+nat50"
    assert "not on all of nat" in t.loc["opus5", "note"]


def test_a_family_with_no_think_row_declines_rather_than_pairing_against_itself(nat, rng):
    """Astra's `think` arm is cut, so it has a `nocot` row and no CoT gain — which is the
    real shape of the run and must not read as a gain of zero."""
    from analysis.metrics import cot_gain

    t = cot_gain(nat, rng).set_index("family")
    assert t.loc["astra", "n_paired"] == 0
    assert np.isnan(t.loc["astra", "diff"])
    assert "needs both" in t.loc["astra", "note"]


def test_the_headline_is_still_on_all_one_hundred(nat, rng):
    """The headline is `nocot` and `nocot` did not move to the 50."""
    from analysis.metrics import headline

    t = headline(nat, rng)
    assert (t["n_paired"] == 100).all()


# ---- denominators -------------------------------------------------------------------------

def test_k_repeats_collapse_to_one_row_per_clue_as_the_mean_not_a_majority(nat):
    clue = by_clue(nat)
    opus = clue[clue["system"] == "opus5-nocot"]
    assert len(opus) == opus["id"].nunique()
    assert (opus["k"] == 3).all()
    # a mean of k, so intermediate values exist; a binarised score could only be 0 or 1
    assert opus["correct"].isin([0.0, 1.0]).all() is np.False_ or \
        ((opus["correct"] > 0) & (opus["correct"] < 1)).any()
    # on astra k is *per clue*, because a call that reasoned leaves that clue's repeats: k is
    # read off the rows that survived, never assumed from --repeat
    astra = clue[clue["system"] == "astra-nocot"]
    assert (astra["k"] <= 3).all() and (astra["k"] >= 1).all()


def test_accuracy_denominator_is_clues_not_samples(nat, rng):
    ps = per_system(nat, rng).set_index("system")
    astra = ps.loc["astra-nocot"]
    assert astra["n_clues"] == 100
    # 3 x 100 calls, less the ones that reasoned: those are a different condition, not a wrong
    # answer, and they leave the denominator (see the exclusion tests below)
    assert astra["n_samples"] == 300 - astra["n_reasoned_excluded"]
    assert ps.loc["opus5-nocot", "n_samples"] == 300


def test_val_is_undefined_on_the_nocot_arm(nat, rng):
    ps = per_system(nat, rng).set_index("system")
    assert np.isnan(ps.loc["astra-nocot", "val_lower"])
    assert "not defined" in ps.loc["astra-nocot", "notes"]
    assert not np.isnan(ps.loc["opus5-think", "val_lower"])


def test_m2_denominator_is_that_systems_correct_clues(nat, rng):
    ps = per_system(nat, rng).set_index("system")
    think = nat[nat["system"] == "opus5-think"]
    assert ps.loc["opus5-think", "n_correct"] == int((think["correct"] == 1).sum())


# ---- the per-call reasoning exclusion (prompt v2, probe report §7.4) ----------------------

def test_a_nocot_call_that_reasoned_leaves_the_accuracy_and_is_counted(nat, rng):
    """The arm is "asked not to reason, verified per call". A call that came back with
    reasoning tokens is not a no-CoT call, so it is neither scored nor silently kept."""
    ps = per_system(nat, rng).set_index("system")
    n_excluded = ps.loc["astra-nocot", "n_reasoned_excluded"]
    assert n_excluded > 0, "the fake runs must exercise this path"
    raw = nat[(nat["system"] == "astra-nocot") & (nat["reasoning_tokens"] > 0)]
    assert len(raw) == n_excluded
    kept = scored(nat[nat["system"] == "astra-nocot"])
    assert (kept["reasoning_tokens"].fillna(0) == 0).all()


def test_the_excluded_calls_are_not_counted_as_errors(nat, rng):
    """An API failure and a call that reasoned are different facts about the run, and folding
    one into the other would report a condition failure as flakiness."""
    ps = per_system(nat, rng).set_index("system")
    g = nat[nat["system"] == "astra-nocot"]
    assert ps.loc["astra-nocot", "n_errors"] == int(g["correct"].isna().sum())
    assert ps.loc["astra-nocot", "n_reasoned_excluded"] > 0


def test_the_note_says_the_calls_were_excluded_and_why(nat, rng):
    ps = per_system(nat, rng).set_index("system")
    note = ps.loc["astra-nocot", "notes"]
    assert "excluded from Acc" in note and "reasoning_tokens > 0" in note


def test_reasoning_is_still_summarised_over_every_call_including_the_excluded_ones(nat, rng):
    """The exclusion must not hide its own size: the mean is over all calls, so a system whose
    zero is a measurement rather than a setting cannot look identical to one whose is."""
    ps = per_system(nat, rng).set_index("system")
    assert ps.loc["astra-nocot", "mean_reasoning_tokens"] > 0
    assert ps.loc["sonnet5-nocot", "mean_reasoning_tokens"] == 0


def test_an_unreported_count_is_kept_and_reported_not_excluded(nat, rng):
    """A None is the absence of evidence, which the run's own reasoning gate already fails on
    the OpenRouter route. Dropping those rows here would empty the arm instead of flagging it."""
    ps = per_system(nat, rng).set_index("system")
    sonnet = ps.loc["sonnet5-nocot"]
    assert sonnet["n_reasoning_unreported"] == 300
    assert sonnet["n_samples"] == 300
    assert sonnet["n_reasoned_excluded"] == 0
    assert "no reasoning-token count at all" in sonnet["notes"]


def test_the_think_arm_keeps_every_call_however_much_it_reasoned(nat, rng):
    """The exclusion is the `nocot` arm's definition, not a global filter: reasoning is what
    the `think` arm is for."""
    ps = per_system(nat, rng).set_index("system")
    assert ps.loc["opus5-think", "n_reasoned_excluded"] == 0
    assert ps.loc["opus5-think", "mean_reasoning_tokens"] > 0
    think = nat[nat["system"] == "opus5-think"]
    assert len(scored(think)) == int(think["correct"].notna().sum())


def test_the_exclusion_reaches_the_headline_and_the_flip_rate(nat, rng):
    """Every table goes through scored() or by_clue(), so the exclusion is arm-wide."""
    clue = by_clue(nat[nat["system"] == "astra-nocot"])
    assert (clue["k"] <= 3).all() and (clue["k"] < 3).any()


# ---- bounds -------------------------------------------------------------------------------

def test_val_and_m2_bounds_bracket_and_the_upper_credits_the_unscoreable(nat, rng):
    ps = per_system(nat, rng)
    think = ps[ps["task"] != "nocot"]
    assert (think["val_upper"] >= think["val_lower"]).all()
    assert (think["m2_upper"] >= think["m2_lower"]).all()
    # the gap between the bounds is exactly the unscoreable share
    assert np.allclose(think["val_upper"] - think["val_lower"], think["unscoreable_rate"])


def test_every_bound_is_a_proportion(nat, rng):
    ps = per_system(nat, rng)
    for col in ("acc", "val_lower", "val_upper", "m2_lower", "m2_upper"):
        v = ps[col].dropna()
        assert ((v >= 0) & (v <= 1)).all(), col


# ---- the floor policy ----------------------------------------------------------------------

def test_an_arm_below_the_floor_keeps_its_point_estimate_and_loses_its_interval(nat, rng):
    ps = per_system(nat, rng).set_index("system")
    weak = ps.loc["gpt55-nocot"]
    assert weak["acc"] < ACC_FLOOR
    assert weak["below_acc_floor"] and not weak["ci_reliable"]
    assert not np.isnan(weak["acc"])          # the point estimate stands
    assert "floor" in weak["notes"]


def test_a_healthy_arm_is_not_flagged(nat, rng):
    ps = per_system(nat, rng).set_index("system")
    assert ps.loc["astra-nocot", "ci_reliable"]


# ---- refusals ------------------------------------------------------------------------------

def test_the_cutoff_split_declines_while_a_training_cutoff_is_null(nat, rng):
    t = cutoff_split(nat, rng)
    assert len(t) == 1 and "declined" in t.iloc[0]["note"]
    assert "training_cutoff" in t.iloc[0]["note"]
    assert "do not fill from memory" in t.iloc[0]["note"]


def test_the_hop_regression_declines_while_no_clue_carries_a_hop_count(nat, rng):
    t = by_hops(nat, rng)
    assert len(t) == 1 and "declined" in t.iloc[0]["note"]


def test_the_hop_regression_runs_once_hop_counts_exist(nat, rng):
    g = nat.copy()
    # a fabricated hop count, purely to prove the code path exists
    g["hops"] = (g["id"].str.extract(r"(\d+)").astype(int) % 4 + 1).to_numpy()
    t = by_hops(g, rng)
    assert "acc_slope_per_hop" in t and len(t) >= 1
    assert (t["slope_ci_lo"] <= t["acc_slope_per_hop"]).all()


# ---- the rest of §5 ------------------------------------------------------------------------

def test_the_device_table_is_marked_descriptive(nat, rng):
    t = by_device(nat, rng)
    assert len(t) and t["descriptive_only"].all()


def test_the_prose_cot_rung_is_off_the_effort_axis(nat, rng):
    t = sonnet_rungs(nat, rng)
    # `medium` was cut on 2026-09-10 night; three points still carry a slope
    assert set(t["rung"]) == {"off", "low", "high", "cot"}
    assert not t[t["rung"] == "cot"]["on_effort_axis"].iloc[0]
    assert np.isnan(t[t["rung"] == "cot"]["rung_index"].iloc[0])
    assert "difference_of_slopes" in t.attrs


def test_the_flip_rate_is_only_defined_where_there_are_repeats(nat):
    t = flip_rate(nat).set_index("system")
    assert 0 <= t.loc["astra-nocot", "flip_rate"] <= 1
    assert np.isnan(t.loc["opus5-think", "flip_rate"])
    assert "k = 1" in t.loc["opus5-think", "note"]


def test_the_guardian_line_carries_the_published_comparator(df, rng):
    from analysis.metrics import guardian_line

    t = guardian_line(df, rng)
    assert (t["source"] == "published").any()
    assert t[t["source"] == "published"]["acc"].iloc[0] == 0.185


# ---- figures --------------------------------------------------------------------------------

def test_the_three_figures_render(nat, rng, tmp_path):
    paths = render(all_tables(nat, rng), tmp_path)
    assert [p.name for p in paths] == ["headline.png", "cot_gain.png", "by_device.png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_a_declined_table_still_renders_a_figure_that_says_why(rng, tmp_path):
    empty = pd.DataFrame([{"reference": "astra", "note": "no nocot runs"}])
    render({"headline": empty,
            "cot_gain": pd.DataFrame(),
            "by_device": pd.DataFrame()}, tmp_path)
    assert (tmp_path / "headline.png").stat().st_size > 0


def test_the_guardian_line_says_so_when_the_split_was_never_run(rng):
    """Dropped 2026-09-10 (Nat). A table holding only the published 18.5 % must not read as a
    comparison this study was measured against — Saha et al. is prior work, cited."""
    from analysis.metrics import guardian_line

    t = guardian_line(pd.DataFrame(columns=["task", "dataset", "correct", "system", "family",
                                            "role", "target", "device", "hops",
                                            "published_date", "id"]), rng)
    assert (t["source"] == "published").all()
    assert "dropped 2026-09-10" in t.iloc[0]["note"]
    assert "NOT compared item-for-item" in t.iloc[0]["note"]


# ---- a truncated call is not a malformed derivation (gate, 2026-09-10) --------------------

def _think_row(**kw):
    base = {"system": "x-think", "family": "x", "role": "think", "task": "think",
            "dataset": "nat50", "id": "cb-0001", "target": "ANSWER", "correct": 1.0,
            "verdict": "malformed", "stop_reason": "stop", "completion": "DEFINITION: x",
            "truncated": False, "unexecuted": False, "l0": False, "l1": False, "l3": False,
            "output_tokens": 100, "input_tokens": 900, "reasoning_tokens": 500,
            "error": "", "device": "charade", "hops": None,
            "published_date": "", "no_answer_line": False, "len_ok": True,
            "has_extra": False, "structured_output": False, "extra_words": 0,
            "reasoning_excluded": False, "model": "fake/x", "rung": None,
            "n_correct": 0, "sourcing": "", "unverified_ops": 0, "check_reasons": ""}
    base.update(kw)
    return base


def test_a_ceiling_truncation_with_no_block_is_reported_apart_from_malformed(rng):
    """11 of the gate's 18 "malformed" rows were empty completions at `max_tokens`, mean 6,813
    output tokens — the whole budget spent thinking. That is a budget artefact, not a parser
    problem, and mixing them made the 5 % malformed criterion unreadable."""
    rows = [_think_row(id=f"cb-{i:04d}", stop_reason="max_tokens", completion="",
                       truncated=True) for i in range(2)]
    rows += [_think_row(id=f"cb-{i:04d}") for i in range(2, 10)]
    ps = per_system(pd.DataFrame(rows), rng).set_index("system")
    assert ps.loc["x-think", "truncated_no_block_rate"] == pytest.approx(0.2)
    assert ps.loc["x-think", "malformed_rate"] == pytest.approx(0.8)
    assert "no derivation written" in ps.loc["x-think", "notes"]


def test_a_truncated_call_still_counts_against_validity(rng):
    """It leaves the malformed *rate* but not Val's denominator: PLAN §1 says every clue in the
    arm, and failing to deliver a derivation is a failure, not an exemption."""
    rows = [_think_row(id=f"cb-{i:04d}", stop_reason="max_tokens", completion="",
                       truncated=True) for i in range(5)]
    rows += [_think_row(id=f"cb-{i:04d}", verdict="valid", l0=True, l1=True, l3=True)
             for i in range(5, 10)]
    ps = per_system(pd.DataFrame(rows), rng).set_index("system")
    assert ps.loc["x-think", "val_lower"] == pytest.approx(0.5)   # 5 of 10, not 5 of 5
    assert ps.loc["x-think", "n_samples"] == 10


# ---- the hand review of the unexecuted bucket (ANALYSIS_PLAN §2.3) ------------------------

def test_reviewed_validity_rescores_only_class_a():
    from analysis.metrics import reviewed_validity

    rows = []
    for i, (verdict, correct) in enumerate([("valid", 1), ("invalid", 1), ("invalid", 1),
                                            ("invalid", 1), ("invalid", 0), ("unscoreable", 1)]):
        rows.append({"system": "x-think", "family": "x", "role": "think", "task": "think",
                     "dataset": "nat50", "id": f"cb-{i:04d}", "verdict": verdict,
                     "correct": float(correct), "reasoning_excluded": False})
    df = pd.DataFrame(rows)
    review = pd.DataFrame([{"id": "cb-0001", "system": "x-think", "class": "A"},
                           {"id": "cb-0002", "system": "x-think", "class": "B"},
                           {"id": "cb-0003", "system": "x-think", "class": "C"}])
    t = reviewed_validity(df, review).iloc[0]
    assert t["n_class_a"] == 1 and t["n_class_b"] == 1 and t["n_class_c"] == 1
    assert t["val_lower_mechanical"] == pytest.approx(1 / 6)
    assert t["val_lower_reviewed"] == pytest.approx(2 / 6)       # only A moves
    assert t["m2_lower_mechanical"] == pytest.approx(3 / 5)       # 3 invalid of 5 correct
    assert t["m2_lower_reviewed"] == pytest.approx(2 / 5)         # B and C stay


def test_reviewed_validity_declines_without_a_review_file(tmp_path, monkeypatch):
    from analysis import metrics as m

    monkeypatch.setattr(m, "REVIEW_PATH", tmp_path / "none.csv")
    df = pd.DataFrame([{"system": "x-think", "task": "think", "id": "cb-0001", "verdict": "valid",
                        "correct": 1.0, "reasoning_excluded": False}])
    assert "declined" in m.reviewed_validity(df)["note"].iloc[0]


# ---- the blind adjudication of answer strings (PLAN §2.4) ---------------------------------

def test_adjudicated_accuracy_credits_only_accepted_strings():
    from analysis.metrics import adjudicated_accuracy

    rows = [{"system": "x-nocot", "task": "nocot", "dataset": "nat", "id": "cb-0001",
             "correct": 0.0, "extracted": "GRAY", "reasoning_excluded": False},
            {"system": "x-nocot", "task": "nocot", "dataset": "nat", "id": "cb-0001",
             "correct": 1.0, "extracted": "GREY", "reasoning_excluded": False},
            {"system": "x-nocot", "task": "nocot", "dataset": "nat", "id": "cb-0002",
             "correct": 0.0, "extracted": "on hand", "reasoning_excluded": False}]
    df = pd.DataFrame(rows)
    none = pd.DataFrame([{"id": "cb-0001", "string": "GRAY", "accept": "N"}])
    t = adjudicated_accuracy(df, none).iloc[0]
    assert t["acc_exact"] == pytest.approx(0.25) and t["acc_adjudicated"] == pytest.approx(0.25)
    assert t["n_wrong_ruled"] == 1 and t["n_wrong_accepted"] == 0
    some = pd.DataFrame([{"id": "cb-0002", "string": "ONHAND", "accept": "Y"}])
    t = adjudicated_accuracy(df, some).iloc[0]
    assert t["acc_adjudicated"] == pytest.approx(0.75)   # the clue at 0 becomes 1; mean of two clues
