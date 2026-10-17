"""Reasoning-token logging and the `nocot` gate (PLAN §2.1, §7.4)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crypticbench.usage import reasoning_gate, sample_usage, summarise_usage  # noqa: E402


def usage(**kw):
    base = {"input_tokens": 350, "output_tokens": 6, "input_tokens_cache_read": 0,
            "input_tokens_cache_write": 0, "reasoning_tokens": None}
    base.update(kw)
    return SimpleNamespace(**base)


def rows(*reasoning, truncated=False):
    return [{"input_tokens": 350, "output_tokens": 6, "reasoning_tokens": r,
             "truncated": truncated} for r in reasoning]


# ---- per sample --------------------------------------------------------------------------

def test_a_reported_count_carries_its_field_name():
    u = sample_usage(usage(reasoning_tokens=1200))
    assert u["reasoning_tokens"] == 1200
    assert u["reasoning_tokens_source"] == "ModelOutput.usage.reasoning_tokens"


def test_an_absent_count_is_none_and_says_so_rather_than_reporting_zero():
    """Anthropic does not return a reasoning-token count; inspect synthesises one from the
    thinking blocks and leaves it None when there are none. That is the absence of a signal,
    not a measured zero, and the two must not be conflated."""
    u = sample_usage(usage())
    assert u["reasoning_tokens"] is None
    assert "absent" in u["reasoning_tokens_source"]


def test_an_errored_sample_has_no_usage_and_says_so():
    u = sample_usage(None)
    assert u["input_tokens"] == 0 and "errored" in u["reasoning_tokens_source"]


def test_cache_tokens_are_carried():
    u = sample_usage(usage(input_tokens_cache_read=100, input_tokens_cache_write=20))
    assert (u["cache_read"], u["cache_write"]) == (100, 20)


# ---- per run -----------------------------------------------------------------------------

def test_summary_folds_none_to_zero_but_counts_how_many_reported():
    s = summarise_usage(rows(None, None, 300))
    assert s["mean_reasoning_tokens"] == 100.0
    assert s["n_reasoning_reported"] == 1
    assert s["reasoning_rate"] == round(1 / 3, 4)


def test_summary_of_nothing_is_not_a_zero():
    assert summarise_usage([])["mean_reasoning_tokens"] is None


# ---- the gate ----------------------------------------------------------------------------

def test_a_clean_nocot_run_passes_and_flags_the_missing_signal():
    g = reasoning_gate("nocot", rows(None, None, None))
    assert g["passed"] is True
    assert "absence of a signal" in g["note"]


def test_a_measured_zero_passes_without_the_caveat():
    g = reasoning_gate("nocot", rows(0, 0, 0))
    assert g["passed"] is True
    assert "absence of a signal" not in g["note"]


def test_any_reasoning_at_all_fails_the_nocot_gate():
    g = reasoning_gate("nocot", rows(0, 0, 1))
    assert g["passed"] is False
    assert "FAILED" in g["note"] and "not a no-CoT number" in g["note"]


def test_the_gate_carries_the_truncation_count_because_thinking_hides_there():
    g = reasoning_gate("nocot", rows(None, None, truncated=True))
    assert g["n_truncated"] == 2


def test_think_runs_have_no_gate():
    g = reasoning_gate("think", rows(3000, 2000))
    assert g["passed"] is None
    assert g["mean_reasoning_tokens"] == 2500.0


# ---- the OpenRouter route: an unreported count is a FAIL, not a zero ----------------------

def test_the_raw_response_is_the_fallback_and_says_where_the_number_came_from():
    """inspect reads `completion_tokens_details.reasoning_tokens` through the OpenAI SDK's
    typed ChatCompletion; if the router puts a count there that the typed model dropped, the
    number is still on the raw body and must not be lost."""
    raw = {"usage": {"completion_tokens": 6,
                     "completion_tokens_details": {"reasoning_tokens": 512}}}
    u = sample_usage(usage(), raw)
    assert u["reasoning_tokens"] == 512
    assert u["reasoning_tokens_source"] == (
        "raw response usage.completion_tokens_details.reasoning_tokens")


def test_a_reported_count_is_not_overridden_by_the_raw_body():
    u = sample_usage(usage(reasoning_tokens=7),
                     {"usage": {"completion_tokens_details": {"reasoning_tokens": 512}}})
    assert u["reasoning_tokens"] == 7
    assert u["reasoning_tokens_source"] == "ModelOutput.usage.reasoning_tokens"


def test_nothing_is_invented_when_the_raw_body_has_no_count():
    for raw in (None, {}, {"usage": {}}, {"usage": {"completion_tokens_details": {}}},
                {"usage": {"completion_tokens_details": {"reasoning_tokens": None}}}):
        u = sample_usage(usage(), raw)
        assert u["reasoning_tokens"] is None
        assert "absent" in u["reasoning_tokens_source"]


def test_an_unreported_count_fails_the_gate_on_the_openrouter_route():
    """The whole `nocot` claim rests on the count. On this route inspect synthesises nothing:
    100 Nones means the router said nothing, which is not evidence that nothing was thought.
    It cannot be settled without a live call, so the gate refuses to read it as a pass."""
    g = reasoning_gate("nocot", rows(None, None, None), route="openrouter")
    assert g["passed"] is False
    assert "reasoning tokens not reported by route" in g["note"].lower()
    assert g["requires_reported_reasoning"] is True


def test_a_measured_zero_on_the_openrouter_route_passes():
    g = reasoning_gate("nocot", rows(0, 0, 0), route="openrouter")
    assert g["passed"] is True
    assert "not reported" not in g["note"]


def test_reasoning_on_the_openrouter_route_still_fails_for_the_original_reason():
    g = reasoning_gate("nocot", rows(0, 0, 40), route="openrouter")
    assert g["passed"] is False
    assert "not a no-CoT number" in g["note"]


def test_the_mock_route_is_not_held_to_the_reported_rule():
    """RUNBOOK §0's smoke test runs on mockllm, which reports no usage details at all; the
    rule is about a live route's evidence, not about the plumbing check."""
    g = reasoning_gate("nocot", rows(None, None), route="mock")
    assert g["passed"] is True
    assert g["requires_reported_reasoning"] is False


def test_a_think_run_has_no_gate_even_on_a_route_that_requires_reporting():
    g = reasoning_gate("think", rows(None, None), route="openrouter")
    assert g["passed"] is None


# ---- n_reasoned: the per-call count behind the exclusion ----------------------------------

def test_the_gate_counts_the_calls_that_reasoned_one_by_one():
    """Prompt v2 asks a model not to reason; it does not make it unable to. So the run's mean
    is not the whole story and `n_reasoned` is the exact count that
    `analysis/metrics.py` excludes from the arm's accuracy."""
    g = reasoning_gate("nocot", rows(0, 0, 0, 12, 40), route="openrouter")
    assert g["n_reasoned"] == 2
    assert "2 call(s) with reasoning_tokens > 0 are excluded" in g["note"]


def test_a_clean_run_reports_zero_reasoned_calls_without_a_caveat():
    g = reasoning_gate("nocot", rows(0, 0, 0), route="openrouter")
    assert g["n_reasoned"] == 0
    assert "excluded" not in g["note"]


def test_unreported_counts_are_counted_separately_from_reasoned_ones():
    """An absent count and a zero count are different claims (the route's own docstring), so
    they are two numbers in the manifest, not one."""
    g = reasoning_gate("nocot", rows(None, 0, 8), route="openrouter")
    assert (g["n_reasoned"], g["n_reasoning_unreported"]) == (1, 1)


def test_the_think_arm_reports_the_count_without_a_verdict():
    g = reasoning_gate("think", rows(3000, 0), route="openrouter")
    assert g["passed"] is None and g["n_reasoned"] == 1


def test_the_pass_rule_itself_is_unchanged_by_the_count():
    """The hard stop is still mean exactly 0 and the field reported on this route; `n_reasoned`
    is reporting, not a second threshold."""
    assert reasoning_gate("nocot", rows(0, 0), route="openrouter")["passed"] is True
    assert reasoning_gate("nocot", rows(0, 1), route="openrouter")["passed"] is False
    assert reasoning_gate("nocot", rows(None,), route="openrouter")["passed"] is False


# ---- a `think` run that did not think (probe report §9.2) --------------------------------

def test_a_think_run_with_no_reasoning_is_flagged_when_it_asked_for_some():
    """Five Opus 4.7 clues came back with 0 reasoning tokens under the config that made
    Sonnet 5 and Opus 5 reason: on this route, omitting the key leaves Claude 4.7's thinking
    off. The manifest has to say so rather than record a `think` row that never reasoned."""
    g = reasoning_gate("think", rows(0, 0, 0, 0, 0), route="openrouter",
                       expects_reasoning=True)
    assert g["reasoning_absent"] is True
    assert "FLAG" in g["note"] and "thinking OFF" in g["note"]
    assert g["passed"] is None          # still no threshold on this arm


def test_a_think_run_that_reasoned_is_not_flagged():
    g = reasoning_gate("think", rows(4230, 3795), route="openrouter", expects_reasoning=True)
    assert g["reasoning_absent"] is False and "FLAG" not in g["note"]


def test_a_condition_that_disables_reasoning_is_not_flagged_for_having_none():
    """`sonnet5-off` and `sonnet5-cot` are *defined* as reasoning disabled with the template
    still asked for. Flagging their zero would flag the condition itself."""
    for task in ("think", "think_cot"):
        g = reasoning_gate(task, rows(0, 0), route="openrouter", expects_reasoning=False)
        assert g["reasoning_absent"] is False and "FLAG" not in g["note"]


def test_the_nocot_arm_never_carries_the_flag():
    g = reasoning_gate("nocot", rows(0, 0), route="openrouter")
    assert g["reasoning_absent"] is False and g["expects_reasoning"] is False


def test_which_systems_expect_reasoning_is_read_off_the_config():
    from crypticbench.usage import expects_reasoning as ex

    assert ex("nocot", None) is False
    assert ex("nocot", {"reasoning_enabled": False}) is False
    assert ex("think", None) is True                              # the vendor default
    assert ex("think", {"reasoning_enabled": True}) is True       # opus47-think
    assert ex("think", {"reasoning_enabled": False}) is False     # sonnet5-off
    assert ex("think_cot", {"reasoning_enabled": False}) is False # sonnet5-cot


# ---- W2: tokens billed outside the visible channels (nocot-bench ELICITATION.md) ----------

def test_w2_is_computed_per_row_when_the_provider_reports_a_total():
    u = sample_usage(usage(total_tokens=400))          # 350 in + 6 out = 356
    assert u["total_tokens"] == 400
    assert u["hidden_tokens"] == 44


def test_w2_is_clean_when_the_total_is_the_sum_of_the_visible_channels():
    assert sample_usage(usage(total_tokens=356))["hidden_tokens"] == 0


def test_w2_is_blind_rather_than_zero_when_no_total_is_reported():
    """"None is not 0" — a provider that reports no total gives no W2 measurement, and the
    absence must not be recorded as a clean zero."""
    assert sample_usage(usage())["hidden_tokens"] is None


def test_the_gate_counts_hidden_token_rows_and_blind_rows_separately():
    rows = [{"reasoning_tokens": 0, "hidden_tokens": 0, "truncated": False},
            {"reasoning_tokens": 0, "hidden_tokens": 12, "truncated": False},
            {"reasoning_tokens": 0, "hidden_tokens": None, "truncated": False}]
    g = reasoning_gate("nocot", rows, route="openrouter")
    assert g["w2_n_hidden_tokens"] == 1
    assert g["w2_n_blind"] == 1
