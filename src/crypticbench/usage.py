"""Token usage per sample, and the reasoning-token gate on the `nocot` arm (PLAN §2.1).

The `nocot` arm's claim is that the model answered *without* chain of thought. The prompt asks
for it and `max_tokens` enforces it, but neither is evidence: the check is the vendor's own
reasoning-token count on every call, recorded per sample and summarised per run.

**What the number is, per route.** Verified against the installed inspect_ai 0.3.258 source
on 2026-09-10 (no live call was made, and one of these is a claim about inspect, not about
what a router actually sends back):

  - **openrouter** — `_providers/openrouter.py` inherits `generate()` from
    `_providers/openai_compatible.py`, which builds the output with
    `model_output_from_openai()`; that function sets
    `usage.reasoning_tokens = completion.usage.completion_tokens_details.reasoning_tokens`
    (`model/_openai.py:917`) and `None` when the response carries no
    `completion_tokens_details`. So the field this repo reads and the OpenAI-compatible field
    named in the build spec are **the same field**, already parsed. What cannot be verified
    without a live call is whether OpenRouter populates it for these seven slugs — the
    listing says they support `reasoning` and `include_reasoning`, which is not the same
    claim. `raw_reasoning_tokens()` therefore reads the raw response body as a fallback and
    records that it did, and a `nocot` run on a route in
    `ROUTES_REQUIRING_REPORTED_REASONING` **fails** the gate when no call reported the count
    at all: on this route an unreported count is an unevidenced no-CoT claim, not a zero.
  - **anthropic / openai (direct, not used since 2026-09-10)** — OpenAI returned
    `output_tokens_details.reasoning_tokens`; Anthropic returned no count at all and inspect
    synthesised one from the summarised thinking blocks, so absence there was documented as a
    lower bound rather than a failure. Kept in this docstring because the two routes mean
    different things by an absent field and the distinction is what the gate turns on.

On `nocot` the second check is structural rather than statistical: `max_tokens` is 16, and a
model that thought would truncate before answering — which is why `n_truncated` is in the gate.

**The count is also per call, not only per run** (prompt v2, 2026-09-10). Since `astra-nocot`
is asked not to reason rather than made unable to, a single call may come back having reasoned
while the run's mean is still near zero. `n_reasoned` in the gate is how many calls did, and
`analysis/metrics.py` drops exactly those rows from that system's accuracy: a call that
reasoned is not a no-CoT call. The gate's own pass rule is unchanged — mean exactly 0, and the
field reported on this route. The two are near-redundant and deliberately not the same test:
the mean is rounded to two places, so at large n a single small reasoning count can round to
0.0 and pass a gate that `n_reasoned` still reports as 1. The count is the exact statement.
"""

from __future__ import annotations

from typing import Any

# `nocot` runs must show no reasoning at all; the gate is exact, not a tolerance (PLAN §7.4).
NOCOT_MAX_MEAN_REASONING_TOKENS = 0.0

# Routes where the count has to *arrive* for the gate to mean anything. On the OpenRouter
# route inspect does not synthesise a reasoning-token count from anything: it parses the
# upstream provider's OpenAI-compatible usage block or reports None. A run of 100 Nones there
# is "the router told us nothing", which cannot be read as "the model did not reason" — so the
# gate fails it rather than passing it, and the probe (RUNBOOK §2) is what settles the question.
ROUTES_REQUIRING_REPORTED_REASONING = frozenset({"openrouter"})

RAW_REASONING_PATH = "usage.completion_tokens_details.reasoning_tokens"


def raw_reasoning_tokens(raw: Any) -> int | None:
    """`usage.completion_tokens_details.reasoning_tokens` out of a raw response body.

    The fallback for the OpenAI-compatible routes: inspect reads this same field through the
    OpenAI SDK's typed `ChatCompletion`, so this only ever fires when the router returned a
    count that the typed model dropped. Nothing else is looked for — a second guessed key
    would be a number from memory.
    """
    if not isinstance(raw, dict):
        return None
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return None
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return None
    value = details.get("reasoning_tokens")
    return value if isinstance(value, int) else None


def sample_usage(usage: Any, raw: Any = None) -> dict[str, Any]:
    """The five token counts for one sample, plus where `reasoning_tokens` came from.

    `usage` is an `inspect_ai.model.ModelUsage` or None (an errored sample); `raw` is the raw
    response body of the call, if the log kept one, used only for the reasoning-token
    fallback. The source string is carried into every results row so the number is never read
    without knowing which field it came out of.
    """
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_write": 0,
                "reasoning_tokens": None, "reasoning_tokens_source": "no output (errored sample)"}
    reasoning = getattr(usage, "reasoning_tokens", None)
    source = ("ModelOutput.usage.reasoning_tokens" if reasoning is not None
              else "ModelOutput.usage.reasoning_tokens absent (no reasoning reported)")
    if reasoning is None:
        from_raw = raw_reasoning_tokens(raw)
        if from_raw is not None:
            reasoning = from_raw
            source = f"raw response {RAW_REASONING_PATH}"
    # W2 (nocot-bench ELICITATION.md): `total_tokens - (prompt + completion)` is tokens billed
    # outside the two visible channels. Recorded per row so the witness exists; it is **blind to
    # providers that nest reasoning inside `completion_tokens`**, which is how Anthropic bills
    # thinking — for those the check that bites is the output ceiling (a Claude deliberating
    # under our 16-token cap would truncate on nearly every row, and 2-7 % do).
    total = getattr(usage, "total_tokens", None)
    inp, out = usage.input_tokens or 0, usage.output_tokens or 0
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read": usage.input_tokens_cache_read or 0,
        "cache_write": usage.input_tokens_cache_write or 0,
        "total_tokens": total,
        "hidden_tokens": (total - (inp + out)) if isinstance(total, int) else None,
        "reasoning_tokens": reasoning,
        "reasoning_tokens_source": source,
    }


def summarise_usage(rows: list[dict]) -> dict[str, Any]:
    """Per-run usage summary for the manifest. `rows` are the run's output rows."""
    if not rows:
        return {"n": 0, "mean_input_tokens": None, "mean_output_tokens": None,
                "max_output_tokens_seen": None, "n_reasoning_reported": 0,
                "mean_reasoning_tokens": None, "max_reasoning_tokens_seen": None,
                "reasoning_rate": None}
    out = [int(r.get("output_tokens") or 0) for r in rows]
    inp = [int(r.get("input_tokens") or 0) for r in rows]
    reported = [int(r["reasoning_tokens"]) for r in rows if r.get("reasoning_tokens") is not None]
    # None is folded to 0 for the mean: on both routes it means "no reasoning was reported for
    # this call". n_reasoning_reported says how many calls carried the field at all.
    folded = [int(r.get("reasoning_tokens") or 0) for r in rows]
    return {
        "n": len(rows),
        "mean_input_tokens": round(sum(inp) / len(inp), 1),
        "mean_output_tokens": round(sum(out) / len(out), 1),
        "max_output_tokens_seen": max(out),
        "n_reasoning_reported": len(reported),
        "mean_reasoning_tokens": round(sum(folded) / len(folded), 2),
        "max_reasoning_tokens_seen": max(folded),
        "reasoning_rate": round(sum(1 for v in folded if v > 0) / len(folded), 4),
    }


def expects_reasoning(task: str, model_args: dict | None) -> bool:
    """Does this system's configuration ask the model to reason at all?

    `nocot` never does. A `think` or `think_cot` system does unless it explicitly disables
    reasoning — `sonnet5-off` and `sonnet5-cot` are defined that way (PLAN §2.1), and their
    zero is the condition, not a failure. Everything else in those arms is meant to reason, so
    a measured zero there is a misconfiguration; `opus47-think` was exactly that until
    2026-09-10 (probe report §9.2).
    """
    if task == "nocot":
        return False
    return (model_args or {}).get("reasoning_enabled") is not False


def reasoning_gate(task: str, rows: list[dict], *, route: str | None = None,
                   expects_reasoning: bool | None = None) -> dict[str, Any]:
    """Did this run honour its arm's reasoning setting?

    Only `nocot` has a pass/fail gate: mean reasoning tokens must be exactly 0, **and** on a
    route in `ROUTES_REQUIRING_REPORTED_REASONING` the count must have been reported at least
    once. Returns `passed=None` where there is nothing to decide.

    `think` and `think_cot` want reasoning, and on this route wanting it is not enough to get
    it: omitting every reasoning key left Claude 4.7's thinking off while Sonnet 5 and Opus 5
    reasoned under the identical config (probe report §9.2). So when `expects_reasoning` is
    true and the measured mean is 0, the summary carries `reasoning_absent` and says so in its
    note — a `think` run that did not think is a finding for the hand-back, not a pass. It
    stays `passed=None`: the arm has no threshold to pass, and inventing one here would put a
    pass/fail on `sonnet5-off`, whose zero is the condition.
    """
    summary = summarise_usage(rows)
    n_trunc = sum(1 for r in rows if r.get("truncated") is True)
    # Calls that reasoned, counted individually: the per-call exclusion in analysis/metrics.py
    # removes these rows from the arm's accuracy, so the manifest has to say how many there were.
    n_reasoned = sum(1 for r in rows if (r.get("reasoning_tokens") or 0) > 0)
    n_unreported = sum(1 for r in rows if r.get("reasoning_tokens") is None)
    # W2: rows billing tokens outside prompt + completion, and rows where the witness is blind
    hidden = [r.get("hidden_tokens") for r in rows]
    n_hidden = sum(1 for h in hidden if isinstance(h, (int, float)) and h > 0)
    n_w2_blind = sum(1 for h in hidden if h is None)
    requires_reported = route in ROUTES_REQUIRING_REPORTED_REASONING
    head = {"task": task, "route": route, "requires_reported_reasoning": requires_reported}
    # `nocot` never expects reasoning; the flag below is about the `think` arms only.
    counts = {"n_reasoned": n_reasoned, "n_reasoning_unreported": n_unreported,
              "w2_n_hidden_tokens": n_hidden, "w2_n_blind": n_w2_blind}
    if task != "nocot":
        mean = summary["mean_reasoning_tokens"]
        absent = bool(expects_reasoning and rows and not mean)
        note = f"no reasoning gate on {task}: the arm asks for reasoning"
        if absent:
            note = (f"FLAG on {task}: this system asks for reasoning and measured "
                    f"{mean} reasoning tokens over {summary['n']} samples — on this route "
                    f"that is thinking OFF, not a low dose (probe report §9.2). Its rows are "
                    f"a template-with-no-thinking measurement; fix models.yaml before using "
                    f"them, and report it either way.")
        elif expects_reasoning is None:
            note += " (whether it was configured to was not passed in)"
        return {**head, "passed": None, "n_truncated": n_trunc, **summary, **counts,
                "expects_reasoning": expects_reasoning, "reasoning_absent": absent,
                "note": note}
    if not rows:
        return {**head, "passed": None, "n_truncated": 0, **summary, **counts,
                "expects_reasoning": False, "reasoning_absent": False, "note": "no samples"}
    mean = summary["mean_reasoning_tokens"]
    unreported = summary["n_reasoning_reported"] == 0
    passed = mean <= NOCOT_MAX_MEAN_REASONING_TOKENS and not (requires_reported and unreported)
    note = (
        f"nocot reasoning gate {'PASSED' if passed else 'FAILED'}: mean reasoning tokens "
        f"{mean} over {summary['n']} samples ({summary['n_reasoning_reported']} calls reported "
        f"the field, {n_reasoned} reasoned); {n_trunc} truncated"
    )
    if n_reasoned:
        note += (f". {n_reasoned} call(s) with reasoning_tokens > 0 are excluded from this "
                 f"system's nocot accuracy (analysis/metrics.py): a call that reasoned is not "
                 f"a no-CoT call.")
    if mean > NOCOT_MAX_MEAN_REASONING_TOKENS:
        note += (". This system reasoned with thinking off — fix models.yaml or drop the arm; "
                 "it is not a no-CoT number.")
    elif requires_reported and unreported:
        note += (f". Reasoning tokens not reported by route {route!r}: the arm's claim rests "
                 f"on the count and no call carried it, so this is a FAIL, not a zero. Check "
                 f"the raw response for {RAW_REASONING_PATH} before reporting the system.")
    elif unreported:
        note += (". No call reported a reasoning-token count, so this is the absence of a "
                 "signal rather than a measured zero; read it with n_truncated and "
                 "mean_output_tokens against max_tokens.")
    return {**head, "passed": passed, "n_truncated": n_trunc, **summary, **counts,
            "expects_reasoning": False, "reasoning_absent": False, "note": note}
