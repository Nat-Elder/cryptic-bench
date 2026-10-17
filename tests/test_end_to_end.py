"""One clue through each arm on the mock model, checking what lands in the results row.

This proves the plumbing — that a real answer scores,
that a real derivation reaches the checker with its levels intact, and that the vendor's
reasoning-token count reaches the manifest. No network, no key.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402
from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import ModelOutput, ModelUsage  # noqa: E402

from crypticbench.usage import reasoning_gate, sample_usage  # noqa: E402
from tasks import nocot as nocot_task  # noqa: E402
from tasks import think as think_task  # noqa: E402

# The setter's corpus is not distributed with the repository (data/README.md); this module
# reads it, so on a checkout without it the module skips rather than fails.
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "corpus.jsonl").exists(),
                                reason="the setter's corpus is not on this machine")

# An invented clue: none of the setter's clues are in the repository (data/README.md).
CLUE = {"id": "cb-0001", "clue": "Sing air wildly, lifting", "enumeration": "7",
        "answer": "RAISING", "answer_display": "RAISING", "device": "anagram"}
DERIVATION = "\n".join([
    "DEFINITION: lifting",
    "DEVICE:     anagram",
    "INDICATOR:  wildly",
    "FODDER:     Sing air [literal] -> SINGAIR",
    "OPERATION:  anagram of SINGAIR",
    "RESULT:     RAISING",
    "ANSWER:     RAISING",
])


def reply(content: str, *, reasoning_tokens: int | None = None) -> ModelOutput:
    out = ModelOutput.from_content(model="mockllm/model", content=content)
    out.usage = ModelUsage(input_tokens=350, output_tokens=6, total_tokens=356,
                           reasoning_tokens=reasoning_tokens)
    return out


@pytest.fixture(scope="module", autouse=True)
def log_dir(tmp_path_factory):
    """inspect defaults to ./logs, and a test must not leave eval logs in the working tree —
    they hold completions, which are gitignored for a reason. The directory has to outlive the
    call, because samples are read back from the log file lazily."""
    global LOG_DIR
    LOG_DIR = str(tmp_path_factory.mktemp("eval_logs"))
    return LOG_DIR


LOG_DIR = None


def run(task, outputs):
    logs = inspect_eval(task, model="mockllm/model",
                        model_args={"custom_outputs": outputs},
                        display="none", log_dir=LOG_DIR, log_samples=True)
    assert logs[0].status == "success", logs[0].error
    return logs[0].samples[0]


def test_a_correct_bare_answer_scores_and_carries_no_extra_text():
    task, _ = nocot_task.build_task([CLUE], [])
    s = run(task, [reply("RAISING", reasoning_tokens=0)])
    score = s.scores["nocot_answer"]
    assert score.value == "C"
    assert score.answer == "RAISING"
    assert score.metadata["has_extra"] is False
    assert score.metadata["len_ok"] is True
    assert sample_usage(s.output.usage)["reasoning_tokens"] == 0


def test_a_nocot_answer_with_working_is_still_scored_but_flagged():
    task, _ = nocot_task.build_task([CLUE], [])
    s = run(task, [reply("RAISING (anagram of 'Sing air')")])
    score = s.scores["nocot_answer"]
    assert score.value == "C"          # the answer is right
    assert score.metadata["has_extra"] is True    # and the arm's premise is not met
    assert "anagram" in score.metadata["extra_text"]


def test_a_nocot_run_that_reasoned_fails_the_gate():
    task, _ = nocot_task.build_task([CLUE], [])
    s = run(task, [reply("RAISING", reasoning_tokens=1400)])
    rows = [{"input_tokens": 350, "output_tokens": 6,
             "reasoning_tokens": sample_usage(s.output.usage)["reasoning_tokens"],
             "truncated": False}]
    gate = reasoning_gate("nocot", rows)
    assert gate["passed"] is False and "not a no-CoT number" in gate["note"]


def test_a_valid_derivation_reaches_the_checker_with_its_levels():
    task, _ = think_task.build_task([CLUE], [], "think")
    s = run(task, [reply(DERIVATION, reasoning_tokens=900)])
    score = s.scores["derivation_check"]
    assert score.value == "C"
    md = score.metadata
    assert md["verdict"] == "valid"
    assert (md["l0"], md["l1"], md["l2a"], md["l3"]) == (True, True, True, True)
    assert md["device_claimed"] == "anagram"
    assert md["unexecuted"] is False
    assert sample_usage(s.output.usage)["reasoning_tokens"] == 900


def test_a_right_answer_with_a_derivation_that_does_not_execute_is_the_m2_signature():
    """The whole point of pairing validity with accuracy: this row is correct and invalid."""
    broken = DERIVATION.replace("Sing air [literal] -> SINGAIR", "Sing [literal] -> SING")
    task, _ = think_task.build_task([CLUE], [], "think")
    md = run(task, [reply(broken)]).scores["derivation_check"].metadata
    assert md["verdict"] == "invalid"
    assert md["l1"] and md["l3"] and md["l2a"] is False
    assert md["unexecuted"] is True     # L1 ∧ L3 ∧ ¬L2a: the hand-review bucket (PLAN §2.3)


def test_a_claimed_double_definition_leaves_the_scored_set_rather_than_failing():
    unscoreable = DERIVATION.replace("DEVICE:     anagram", "DEVICE:     double")
    task, _ = think_task.build_task([CLUE], [], "think")
    md = run(task, [reply(unscoreable)]).scores["derivation_check"].metadata
    assert md["verdict"] == "unscoreable"
    assert md["l2a"] is None            # which is why Val is an interval, not a number


def test_the_cot_arm_uses_its_own_prompt_and_the_same_scorer():
    task, digest = think_task.build_task([CLUE], [], "think_cot")
    _, plain_digest = think_task.build_task([CLUE], [], "think")
    assert digest != plain_digest
    md = run(task, [reply("Thinking in prose first...\n\n" + DERIVATION)]) \
        .scores["derivation_check"].metadata
    assert md["verdict"] == "valid"     # prose before the block must not break the parse
