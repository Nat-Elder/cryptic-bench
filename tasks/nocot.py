"""The `nocot` arm: clue in, bare answer out, vendor thinking off (PLAN §2.1).

This is the headline arm. The model is given four worked clue → answer pairs and asked for the
answer alone; `max_tokens` is 16 and the reasoning-token count is logged on every call
(`crypticbench.usage`). Scoring is normalised exact match against the setter's answer — and,
after step 8 of the run order, against the adjudicated answer set.

**The prompt is v2** (`prompts/nocot.v2.txt`, run.py's default for this arm since 2026-09-10):
v1 plus one line asking for no reasoning tokens. The arm is therefore "asked not to reason,
verified per call", not "reasoning off by parameter" — on `astra-nocot` no parameter can turn
reasoning off at all (probe report §7.1, §7.3, §7.4), and the line is inert on the systems
where one can. A call that comes back with `reasoning_tokens > 0` is excluded from the arm's
accuracy in `analysis/metrics.py` and counted in the manifest's `reasoning_gate`.

**The 5 % rule** (Think Fast's protocol, PLAN §2.1). If more than 5 % of a system's outputs
contain anything beyond the answer, the arm is not measuring what it says: the model is
reasoning in the visible text. `extra_text` is recorded per sample, `extra_text_summary()`
computes the share per system, and the fallback is `generate.structured: true` in
configs/models.yaml, which pins the output to a one-field JSON schema through inspect's
structured-output path. The scorer reads either shape.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspect_ai import Task  # noqa: E402
from inspect_ai.model import GenerateConfig, ResponseSchema  # noqa: E402
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
from inspect_ai.util import JSONSchema  # noqa: E402

from crypticbench.normalise import enum_parts, enum_total, norm_answer  # noqa: E402
from crypticbench.prompts import format_examples_nocot, load_prompt  # noqa: E402
from tasks.common import build_dataset, visible_text  # noqa: E402

# Sixteen output tokens is the ceiling: eight was found to truncate correct two-word answers
# and left the structured fallback returning fragments, and sixteen is also the floor OpenAI's
# Responses API accepts. Still far too short for working: anything discursive truncates
# instead of scoring.
NOCOT_MAX_TOKENS = 16
EXTRA_TEXT_THRESHOLD = 0.05

# A leading label the model was not asked for. Recorded as extra text, not silently stripped.
_LABEL_RE = re.compile(r"^\s*(?:the\s+)?answer\s*[:\-–]\s*", re.IGNORECASE)
# The answer span: a leading run of letters and the separators an enumeration can contain.
_ANSWER_RE = re.compile(r"[A-Za-z][A-Za-z'’\- ]*")


@dataclass
class NoCotOutput:
    answer: str | None        # normalised A–Z, or None if nothing answer-shaped was written
    answer_span: str          # the raw span the answer was read from
    extra_text: str           # everything else in the visible output
    extra_words: int          # words in the answer span beyond what the enumeration allows
    from_json: bool = False   # the structured-output path produced it

    @property
    def has_extra(self) -> bool:
        """The 5 % rule's per-sample flag: anything at all beyond the answer."""
        return bool(self.extra_text.strip()) or self.extra_words > 0


def answer_schema() -> ResponseSchema:
    """The structured-output fallback: one field, one string, nothing to reason in."""
    return ResponseSchema(
        name="cryptic_answer",
        description="The answer to the clue, in capital letters, and nothing else.",
        json_schema=JSONSchema(
            type="object",
            properties={"answer": JSONSchema(type="string",
                                             description="The answer in capital letters.")},
            required=["answer"],
            additionalProperties=False,
        ),
        strict=True,
    )


def parse_nocot(text: str, enumeration: str = "") -> NoCotOutput:
    """Split a `nocot` completion into the answer and everything the model added to it.

    Two shapes are accepted: a bare answer (what the prompt asks for) and the one-field JSON
    object the structured-output fallback produces. Anything else — a label, a sentence, a
    second line, working — lands in `extra_text`, which is what the 5 % rule counts.

    `extra_words` catches the case `extra_text` cannot: a model that answers in a sentence
    ("The answer is MEDIC") writes no character outside the answer span, so the span itself is
    compared against the enumeration's word count. The enumeration is in the prompt, so this
    uses nothing the model was not given. It is an upper bound — a hyphenated enumeration
    allows more words than the answer has — and deliberately so.
    """
    raw = visible_text(text or "")
    stripped = raw.strip()
    from_json = False
    extra_prefix = ""

    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
            from_json = True
            extra_prefix = json.dumps({k: v for k, v in obj.items() if k != "answer"}) \
                if len(obj) > 1 else ""
            stripped = obj["answer"].strip()

    label = _LABEL_RE.match(stripped)
    if label:
        extra_prefix = (extra_prefix + " " + label.group(0)).strip()
        stripped = stripped[label.end():]

    first_line, _, rest = stripped.partition("\n")
    m = _ANSWER_RE.match(first_line)
    if not m:
        return NoCotOutput(answer=None, answer_span="", from_json=from_json,
                           extra_text=(extra_prefix + " " + stripped).strip(), extra_words=0)
    span = m.group(0).strip()
    extra = " ".join(x for x in (extra_prefix, first_line[m.end():].strip(), rest.strip()) if x)
    allowed = len(enum_parts(enumeration)) if enumeration else 1
    extra_words = max(0, len(span.split()) - max(allowed, 1))
    return NoCotOutput(answer=norm_answer(span) or None, answer_span=span,
                       extra_text=extra.strip(), extra_words=extra_words, from_json=from_json)


def extra_text_summary(rows: list[dict]) -> dict:
    """The 5 % rule over one system's rows. `rows` carry `extra_text` / `extra_words`."""
    n = len(rows)
    if not n:
        return {"n": 0, "n_extra": 0, "extra_text_rate": None, "over_threshold": False,
                "note": "no samples"}
    n_extra = sum(1 for r in rows
                  if str(r.get("extra_text") or "").strip() or int(r.get("extra_words") or 0) > 0)
    rate = n_extra / n
    over = rate > EXTRA_TEXT_THRESHOLD
    note = f"{n_extra}/{n} = {rate:.1%} of nocot outputs carry text beyond the answer"
    if over:
        note += (f" — over the {EXTRA_TEXT_THRESHOLD:.0%} rule. Re-run this system with "
                 f"`generate: {{structured: true}}` in configs/models.yaml and say so in the "
                 f"write-up (PLAN §2.1).")
    return {"n": n, "n_extra": n_extra, "extra_text_rate": round(rate, 4),
            "over_threshold": over, "note": note}


@scorer(metrics=[accuracy(), stderr()])
def nocot_answer():
    async def score(state: TaskState, target: Target) -> Score:
        enum = state.metadata.get("enumeration", "")
        parsed = parse_nocot(state.output.completion or "", enum)
        gold = norm_answer(target.text)
        return Score(
            value=CORRECT if parsed.answer == gold else INCORRECT,
            answer=parsed.answer,
            metadata={
                "extra_text": parsed.extra_text[:500],
                "extra_words": parsed.extra_words,
                "has_extra": parsed.has_extra,
                "structured_output": parsed.from_json,
                "no_answer_line": parsed.answer is None,
                "len_ok": parsed.answer is not None and enum_total(enum) == len(parsed.answer),
                "stop_reason": state.output.stop_reason,
                # A 16-token ceiling truncates a discursive answer, which then scores
                # INCORRECT for want of budget, not of ability. Counted, never absorbed.
                "truncated": state.output.stop_reason == "max_tokens",
            },
        )

    return score


def build_task(rows: list[dict], dev_rows: list[dict], prompt_version: str = "v1",
               structured: bool = False) -> tuple[Task, str]:
    """Returns (task, prompt_sha256)."""
    system, user_template, digest = load_prompt("nocot", prompt_version)
    config = GenerateConfig(response_schema=answer_schema()) if structured else GenerateConfig()
    task = Task(
        dataset=build_dataset(rows, user_template, format_examples_nocot(dev_rows),
                              "crypticbench-nocot"),
        solver=[system_message(system), generate()],
        scorer=nocot_answer(),
        config=config,
        name="nocot",
    )
    return task, digest
