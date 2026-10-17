"""configs/models.yaml is a claim about what each call asks for.

Every system in it is a row in the headline table, so a wrong field here is a wrong number
there. These tests pin the invariants the runner and the plan depend on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from crypticbench.config import MODELS_YAML  # noqa: E402
from tasks.nocot import NOCOT_MAX_TOKENS  # noqa: E402

SYSTEMS = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
LIVE = {k: v for k, v in SYSTEMS.items() if v["route"] != "mock"}
# PLAN §3. Fable 5.1 is excluded on purpose: its thinking cannot be disabled, so it has no
# `nocot` row and therefore no row in the headline.
PLAN_FAMILIES = {"haiku45", "sonnet5", "opus47", "opus5", "gpt55", "sol", "astra", "fable"}
# **Empty, and that is a finding.** PLAN §3 excluded Fable 5.1 from the `nocot` arm because its
# thinking "cannot be disabled through the API" — the same parameter claim that was made about
# Astra and withdrawn on 2026-09-10 when the arm was redefined as "asked not to reason, verified
# per call". Tested 2026-09-11: Fable returns `reasoning_tokens = 0` with the count reported, on
# 400 of 400 calls. The exclusion was stale, and nocot-bench's 285-model roster carries Fable
# with no-CoT arms too. Every family now has both arms.
THINK_ONLY_FAMILIES: set[str] = set()


@pytest.mark.parametrize("key", sorted(SYSTEMS))
def test_every_system_carries_every_required_field(key):
    cfg = SYSTEMS[key]
    for field in ("route", "model", "model_batch", "batch", "arm", "family", "role",
                  "generate", "training_cutoff", "training_cutoff_source"):
        assert field in cfg, f"{key} is missing {field}"


def test_the_only_live_route_is_openrouter():
    """Nat's ruling of 2026-09-10: all traffic through OpenRouter, no vendor keys.
    A stray `route: anthropic` here would be a second key the .env does not have."""
    assert {cfg["route"] for cfg in LIVE.values()} == {"openrouter"}


@pytest.mark.parametrize("key", sorted(LIVE))
def test_every_live_id_is_an_openrouter_slug(key):
    cfg = LIVE[key]
    assert cfg["model"].startswith("openrouter/"), key
    assert not cfg["model"].endswith(":batch"), key
    assert cfg["model_batch"] == cfg["model"] + ":batch", key


@pytest.mark.parametrize("key", sorted(LIVE))
def test_every_live_system_names_a_batch_id(key):
    """`--sync` is the per-invocation switch between the two ids, not a per-system setting."""
    assert LIVE[key]["batch"] is True, key


@pytest.mark.parametrize("key", sorted(SYSTEMS))
def test_the_generate_block_is_a_valid_generate_config(key):
    from inspect_ai.model import GenerateConfig

    gen = dict(SYSTEMS[key]["generate"])
    gen.pop("structured", None)          # this repo's flag for the 5 % fallback, not inspect's
    GenerateConfig(**gen)                # raises if a field name or type is wrong


@pytest.mark.parametrize("key", sorted(SYSTEMS))
def test_no_system_sets_temperature(key):
    """Claude 4.7+ reject it and GPT-5.5 ignores it with reasoning on, so setting it would
    record a parameter that was never applied (PLAN §3, §8)."""
    assert "temperature" not in SYSTEMS[key]["generate"], key


@pytest.mark.parametrize("key", sorted(SYSTEMS))
def test_every_training_cutoff_is_null_and_says_where_to_get_it(key):
    """None has been taken from a vendor page yet, and none may be filled from memory
    (PLAN §2.6). When Nat fills one, this test is the place to record that it happened."""
    cfg = SYSTEMS[key]
    assert cfg["training_cutoff"] is None, f"{key} has a cutoff — update this test with its source"
    assert cfg["training_cutoff_source"], key


# ---- the arms -------------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(k for k, v in SYSTEMS.items() if v["arm"] == "nocot"))
def test_a_nocot_system_cannot_write_working(key):
    """The ceiling is what stops a plain `nocot` system writing working.

    **Exception: a system under the structured constraint.** There the answer arrives as a JSON
    field and there is no content channel to deliberate in, so the *schema* is the constraint
    and the ceiling only has to fit the envelope. At 16 it did not: `{"answer": "` takes ~11
    tokens, which truncated 94 % of Fable's clues whose gold answer exceeds 8 characters and
    depressed its accuracy — biasing the headline toward our own hypothesis. nocot-bench names
    the same trap and prescribes >= 300 for schema paths.
    """
    cfg = SYSTEMS[key]
    if cfg["generate"].get("structured"):
        assert cfg["generate"]["max_tokens"] >= 300, (
            f"{key} is on the schema path, where a tight ceiling truncates the envelope rather "
            f"than preventing working")
        return
    assert cfg["generate"]["max_tokens"] == NOCOT_MAX_TOKENS, key


@pytest.mark.parametrize("key", sorted(SYSTEMS))
def test_no_system_asks_openrouter_for_an_effort_of_none(key):
    """The openrouter provider passes `reasoning_effort` straight into
    `extra_body.reasoning.effort` (only rewriting `max` to `xhigh`), and OpenRouter accepts
    minimal/low/medium/high/xhigh — **not** `none`. Turning reasoning off on this route is
    `model_args: {reasoning_enabled: false}`; `reasoning_effort: none` would put an unaccepted
    value on the wire. Verified in inspect_ai 0.3.258 openrouter.py, 2026-09-10."""
    assert SYSTEMS[key]["generate"].get("reasoning_effort") != "none", key


# `astra-nocot` is the **one** exception to the two rules below, and the reason is not a
# preference: on this model `reasoning: {enabled: false}` is an HTTP 400 ("Reasoning is
# mandatory for this endpoint and cannot be disabled") and `effort: none`/`minimal` are
# rejected or still reason (probe report §7.1, §7.3). Its row exists by prompt instead —
# `reasoning_effort: low`, the lowest value both routes accept, plus the line `nocot` prompt
# v2 appends for every system — and the zero is *verified per call* rather than set: 7 of 7
# calls returned `reasoning_tokens = 0` (§7.4), each run manifest carries `n_reasoned`, and a
# call that reasoned leaves the accuracy (analysis/metrics.py). Blanket-relaxing either rule
# would let a silent `reasoning_enabled` omission on a Claude system pass as the same thing,
# and that omission would leave reasoning ON with nothing to catch it.
# The systems whose reasoning **cannot** be switched off by any parameter, and whose no-CoT row
# therefore rests on the prompt asking plus the count being checked on every call. Two of them
# now: Astra (probe report §7.1, §7.3, §7.4) and Fable, whose identical exclusion was tested and
# overturned on 2026-09-11. Both run at `low`, the lowest effort their endpoints accept.
ASKED_NOT_TO_REASON = {"astra-nocot", "fable-nocot"}
ASTRA_NOCOT_EXCEPTION = "astra-nocot"


@pytest.mark.parametrize("key", sorted(LIVE))
def test_a_live_nocot_system_turns_reasoning_off_explicitly(key):
    """`nocot` means vendor reasoning off, and on this route that is one switch:
    `reasoning: {enabled: false}`. Omitting it would leave the model's default on and there
    would be no no-CoT arm to report.

    **Exception: `astra-nocot`.** That switch is a 400 on GPT-6 Astra, whose reasoning cannot
    be disabled by any parameter on any route (probe report §7.1, §7.3). It is excluded here
    and pinned instead by `test_astra_nocot_is_asked_not_to_reason_because_no_parameter_can`
    — an explicit hole for one named system, not a relaxed rule for all of them.
    """
    cfg = LIVE[key]
    if cfg["arm"] == "nocot" and key not in ASKED_NOT_TO_REASON:
        assert cfg.get("model_args", {}).get("reasoning_enabled") is False, key
        assert "reasoning_effort" not in cfg["generate"], key


def test_astra_nocot_is_asked_not_to_reason_because_no_parameter_can():
    """The other half of that exception, stated positively.

    `reasoning_enabled: false` returned HTTP 400 on OpenRouter and `effort: none` / `minimal`
    were rejected or still reasoned on OpenAI's own API (probe report §7.1, §7.3), so the
    headline model's no-CoT row rests on prompt v2's instruction under the lowest legal effort,
    with the count checked on every call (§7.4). `low` is that lowest legal value; `none` would
    be a 400 and `minimal` is not off.
    """
    cfg = SYSTEMS["astra-nocot"]
    assert cfg["generate"]["reasoning_effort"] == "low"
    assert "reasoning_enabled" not in cfg.get("model_args", {})
    assert cfg["generate"]["max_tokens"] == NOCOT_MAX_TOKENS
    assert cfg["arm"] == "nocot" and cfg["role"] == "nocot"


def test_only_the_asked_not_to_reason_systems_set_an_effort():
    """Two now, not one. Both are models with no off switch at all, so their zero is a measured
    result rather than a setting; every other `nocot` system disables reasoning by parameter and
    must not carry an effort. A third appearing without a documented 400 would mean the arm had
    acquired a second definition of "no CoT" by drift."""
    with_effort = {k for k, v in SYSTEMS.items()
                   if v["arm"] == "nocot" and "reasoning_effort" in v["generate"]}
    assert with_effort == ASKED_NOT_TO_REASON


@pytest.mark.parametrize("key", sorted(LIVE))
def test_a_think_system_does_not_disable_thinking(key):
    """Except the `off` rung, which is defined as reasoning disabled with the template still
    asked for (PLAN §2.1)."""
    cfg = LIVE[key]
    if cfg["arm"] == "think" and cfg.get("rung") not in ("off",):
        assert cfg.get("model_args", {}).get("reasoning_enabled") is not False, key


@pytest.mark.parametrize("key", sorted(LIVE))
def test_a_think_row_leaves_the_effort_at_the_vendor_default(key):
    """PLAN §2.1 defines `think` as the vendor's own default effort, which on this route is
    the absence of every reasoning key: the provider then sends no `reasoning` block at all.
    The Sonnet 5 ladder is the exception — its rungs pin the effort on purpose."""
    cfg = LIVE[key]
    if cfg["role"] == "think" and cfg["family"] != "sonnet5":
        assert "reasoning_effort" not in cfg["generate"], key


def test_the_sonnet_rungs_are_openrouter_efforts_one_to_one():
    """low / medium / high in models.yaml are `reasoning.effort` low / medium / high; the
    ladder is the dose axis of PLAN §5.5 and a renamed rung would silently move it."""
    for rung in ("low", "medium", "high"):
        cfg = SYSTEMS[f"sonnet5-{rung}"]
        assert cfg["generate"]["reasoning_effort"] == rung
        assert "reasoning_enabled" not in cfg.get("model_args", {})


def test_the_off_rung_and_the_prose_cot_rung_have_reasoning_disabled():
    """`think_cot` puts the reasoning in the output rather than the budget; leaving reasoning
    on would make it a different condition from the one the plan names."""
    for key in ("sonnet5-off", "sonnet5-cot"):
        assert SYSTEMS[key]["model_args"]["reasoning_enabled"] is False, key
        assert "reasoning_effort" not in SYSTEMS[key]["generate"], key
    assert SYSTEMS["sonnet5-cot"]["arm"] == "think_cot"


def test_haiku_is_the_only_system_that_takes_a_thinking_budget():
    """Haiku 4.5 predates adaptive thinking, so a budget is what turns thinking on for it. The
    openrouter provider sends `reasoning_tokens` as `reasoning: {max_tokens: N}` — but only
    when no effort is set, since effort wins and the budget is then warned away (verified in
    inspect_ai 0.3.258 openrouter.py, 2026-09-10). So haiku45-think IS expressible here."""
    with_budget = {k for k, v in SYSTEMS.items() if "reasoning_tokens" in v["generate"]}
    assert with_budget == {"haiku45-think"}
    haiku = SYSTEMS["haiku45-think"]
    assert "reasoning_effort" not in haiku["generate"]
    assert "reasoning_enabled" not in haiku.get("model_args", {})


# ---- the cut guard ---------------------------------------------------------------------------

# `astra-think` was cut on 2026-09-10 evening and un-cut the same night: the cut assumed
# 3,150 output tokens and the gate measured 630. Haiku 4.5 was un-cut 2026-09-11;
# `sonnet5-medium` stays cut.
CUT_SYSTEMS = {"sonnet5-medium"}


def test_the_cut_systems_carry_the_ruling_that_cut_them():
    """Nat, 2026-09-10 evening (PLAN §3, §8). The entries stay for the record, so the string is what run.py refuses on."""
    assert {k for k, v in SYSTEMS.items() if v.get("cut")} == CUT_SYSTEMS
    for key in CUT_SYSTEMS:
        cut = SYSTEMS[key]["cut"]
        assert cut.startswith("2026-09-10 Nat — "), key
    # Haiku 4.5 was cut the same evening and un-cut on 2026-09-11
    assert "rungs trimmed" in SYSTEMS["sonnet5-medium"]["cut"]
    assert not SYSTEMS["haiku45-nocot"].get("cut")


def test_a_cut_entry_is_otherwise_complete():
    """Cut is not the same as broken: each of these must still be a runnable row, because
    `--include-cut` runs it and because the record is only worth keeping if it is accurate."""
    for key in sorted(CUT_SYSTEMS):
        cfg = SYSTEMS[key]
        assert cfg["model"], key
        assert cfg["route"] == "openrouter", key


# ---- coverage against the plan ---------------------------------------------------------------

def test_every_model_in_the_plan_has_both_arms():
    by_family: dict[str, set[str]] = {}
    for cfg in LIVE.values():
        by_family.setdefault(cfg["family"], set()).add(cfg["role"])
    assert set(by_family) == PLAN_FAMILIES | THINK_ONLY_FAMILIES
    for family, roles in by_family.items():
        assert "think" in roles, f"{family} has no think arm — it cannot have a CoT gain"
        if family in THINK_ONLY_FAMILIES:
            assert "nocot" not in roles, (
                f"{family} is think-only by PLAN §3; a nocot row would be a claim its API "
                f"cannot support")
            continue
        assert "nocot" in roles, f"{family} has no nocot arm — it cannot be in the headline"


def test_fable_is_in_the_no_cot_arm_after_all():
    """The reversal, pinned. PLAN §3 said Fable could not be measured without chain of thought;
    it can, on the same terms as Astra. What makes its row *unlike* the others is not the
    reasoning count but the channel: plain, it deliberates in the visible answer at
    `reasoning_tokens = 0` (20 % of rows), so it runs under the 5 % rule's prescribed structured
    constraint — which must be said wherever its number is quoted."""
    families = {cfg["family"] for cfg in LIVE.values() if cfg["arm"] == "nocot"}
    assert "fable" in families
    assert SYSTEMS["fable-nocot"]["generate"]["structured"] is True
    assert not THINK_ONLY_FAMILIES


def test_the_sonnet_ladder_has_every_rung():
    rungs = {v["rung"] for v in SYSTEMS.values() if v.get("rung")}
    assert rungs == {"off", "low", "medium", "high", "cot"}


def test_exactly_one_system_per_family_is_the_think_row():
    counts: dict[str, int] = {}
    for cfg in LIVE.values():
        if cfg["role"] == "think":
            counts[cfg["family"]] = counts.get(cfg["family"], 0) + 1
    assert set(counts) == PLAN_FAMILIES | THINK_ONLY_FAMILIES
    assert set(counts.values()) == {1}


def test_astra_and_sol_are_runnable_now_that_the_route_carries_them():
    """What the route bought (2026-09-10): the headline model has an id, so the probe can
    reach it. Its no-reasoning parameter is still an assumption the probe tests —
    see test_a_live_nocot_system_turns_reasoning_off_explicitly and RUNBOOK §2."""
    for key in ("astra-nocot", "astra-think", "sol-nocot", "sol-think"):
        assert SYSTEMS[key]["model"] is not None, key
        assert SYSTEMS[key]["model_batch"] is not None, key


def test_no_pro_luna_or_terra_variant_is_used():
    """Nat named the seven slugs; the -pro / -luna / -terra siblings are different models and
    would be a different table."""
    for key, cfg in LIVE.items():
        for banned in ("-pro", "-luna", "-terra"):
            assert banned not in cfg["model"], (key, banned)


def test_opus47_is_the_one_think_row_that_has_to_request_thinking():
    """PLAN §2.1's `think` is the vendor's default effort, which on this route is the absence
    of every reasoning key. That leaves Claude 4.7's thinking OFF: five probe clues under it
    returned 0 reasoning tokens and no thinking block, while Sonnet 5 and Opus 5 reasoned
    under the identical config (probe report §9.2). So `opus47-think` requests reasoning
    explicitly and pins no effort and no budget — the dose is still the model's."""
    cfg = SYSTEMS["opus47-think"]
    assert cfg["model_args"]["reasoning_enabled"] is True
    assert "reasoning_effort" not in cfg["generate"]
    assert "reasoning_tokens" not in cfg["generate"]
    requesting = {k for k, v in SYSTEMS.items()
                  if (v.get("model_args") or {}).get("reasoning_enabled") is True}
    assert requesting == {"opus47-think"}, (
        "a second one means the arm has two definitions of `think` in it")


def test_opus47_nocot_still_turns_reasoning_off():
    """The same model, the other arm: 4.7's default is adaptive, so the `nocot` row must keep
    the switch even though its `think` row now needs the opposite one."""
    assert SYSTEMS["opus47-nocot"]["model_args"]["reasoning_enabled"] is False


def test_astra_has_a_think_row_again_because_the_prediction_needs_one():
    """Un-cut 2026-09-10 night. PLAN §5.2's ordering test — the Astra post's claim that Astra's
    CoT gain is the smallest — cannot be run at all without a row for Astra, and the cut's
    premise (3,150 output tokens) was measured at 630."""
    cfg = SYSTEMS["astra-think"]
    assert not cfg.get("cut")
    assert cfg["arm"] == "think" and cfg["role"] == "think"
