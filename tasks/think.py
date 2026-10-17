"""The `think` arms: solve the clue *and* write the derivation, with vendor reasoning on.

Two conditions, one template (PLAN §2.1):

- `think`      — the seven-field block, vendor thinking on at its default effort.
- `think_cot`  — the same block preceded by prose reasoning, thinking off. Sonnet 5's
                 "off + prose CoT" rung, and only that: reasoning in the output rather than in
                 the budget (cot-design-2026-09-09.md §3).

The block writes the derivation **before** the answer, so neither condition is no-CoT. Only
`nocot` is. The scorer keeps answer correctness — solve accuracy falls out of the same run —
and attaches the checker's verdict and level flags, which is what Val and M2 are computed from
downstream (PLAN §1, §2.2).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspect_ai import Task  # noqa: E402
from inspect_ai.scorer import (  # noqa: E402
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate, system_message  # noqa: E402

from crypticbench.checker import check  # noqa: E402
from crypticbench.derivation import parse  # noqa: E402
from crypticbench.normalise import enum_total, norm_answer  # noqa: E402
from crypticbench.prompts import format_examples_think, load_prompt  # noqa: E402
from tasks.common import build_dataset, visible_text  # noqa: E402

CONDITIONS = ("think", "think_cot")


@scorer(metrics=[accuracy(), stderr()])
def derivation_check():
    async def score(state: TaskState, target: Target) -> Score:
        text = visible_text(state.output.completion or "")
        clue = state.metadata.get("clue", "")
        gold = target.text
        d = parse(text)
        c = check(text, clue, gold)
        extracted = norm_answer(d.answer) or None
        enum = state.metadata.get("enumeration", "")
        return Score(
            value=CORRECT if extracted == norm_answer(gold) else INCORRECT,
            answer=extracted,
            metadata={
                "verdict": c.verdict,
                "l0": c.l0, "l1": c.l1, "l2a": c.l2a, "l3": c.l3,
                "unexecuted": c.unexecuted,
                "device_claimed": d.device or None,
                "sourcing": c.sourcing,
                "unverified_ops": c.unverified_ops,
                "reasons": "; ".join(c.reasons)[:500],
                "no_answer_line": extracted is None,
                "len_ok": extracted is not None and enum_total(enum) == len(extracted),
                "stop_reason": state.output.stop_reason,
                # A truncated sample is a budget artefact: it usually has no ANSWER line and so
                # scores INCORRECT, which would read as inability rather than a small ceiling.
                "truncated": state.output.stop_reason == "max_tokens",
            },
        )

    return score


def build_task(rows: list[dict], dev_rows: list[dict], condition: str = "think",
               prompt_version: str = "v1") -> tuple[Task, str]:
    """Returns (task, prompt_sha256). `condition` is one of CONDITIONS."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    system, user_template, digest = load_prompt(condition, prompt_version)
    task = Task(
        dataset=build_dataset(rows, user_template, format_examples_think(dev_rows),
                              f"crypticbench-{condition}"),
        solver=[system_message(system), generate()],
        scorer=derivation_check(),
        name=condition,
    )
    return task, digest
