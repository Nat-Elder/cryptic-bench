"""Regressions carried from the 2026-09-08 method review, pruned to v3.1.

Each test corresponds to a way the pipeline could have produced a confident number meaning
something other than what the write-up would have said. The probe-B and `solve` items are gone
with their code; what survives is about the corpus, resumption, and reasoning read as answers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.metrics import per_system  # noqa: E402
from crypticbench.config import N_DEV  # noqa: E402
from crypticbench.corpus import validate  # noqa: E402
from run import config_fingerprint, generate_fields  # noqa: E402
from tasks.common import visible_text  # noqa: E402
from tasks.nocot import parse_nocot  # noqa: E402


def rng():
    return np.random.default_rng(20260908)


# ---------- item 7a: an answer must never be read out of the reasoning trace ----------

def test_answer_inside_think_tags_is_ignored():
    """Some providers inline chain of thought as <think>...</think> in the text itself. A model
    trying candidates in the trace must not have them scored."""
    assert parse_nocot("<think>Maybe WRONGUN, or ALSOBAD</think>\nMEDIC", "5").answer == "MEDIC"


def test_unclosed_think_block_does_not_leak_a_candidate():
    assert parse_nocot("<think>WRONGUN and more musing", "7").answer is None


def test_visible_text_leaves_ordinary_completions_alone():
    assert visible_text("MEDIC") == "MEDIC"


# ---------- item 4: clues sharing an answer are not independent ----------
# (the bootstrap itself is pinned in test_paired.py)

# ---------- item 1: the few-shot examples must not leak an answer into the test split ----------

# validate() compares clue text after norm_answer(), which keeps A-Z only — so fixtures must
# differ by a word, not by a numeral, or they read as duplicate clues.
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def base_row(**kw):
    row = {"id": "cb-0001", "clue": "some clue", "answer": "MEDIC", "answer_display": "MEDIC",
           "enumeration": "5", "source": "unpublished", "split": "test", "notes": {}}
    row.update(kw)
    return row


def test_a_test_clue_sharing_a_dev_answer_is_a_violation():
    rows = [base_row(id=f"cb-000{i}", clue=f"clue number {WORDS[i]}", answer="ALPHA", split="dev")
            for i in range(1, N_DEV + 1)]
    rows.append(base_row(id="cb-0099", clue="a re-clue of the same answer", answer="ALPHA"))
    assert any("is also a dev" in e for e in validate(rows))


def test_distinct_dev_answers_pass():
    rows = [base_row(id=f"cb-000{i}", clue=f"clue number {WORDS[i]}", answer=f"ALPH{chr(64 + i)}",
                     split="dev") for i in range(1, N_DEV + 1)]
    rows.append(base_row(id="cb-0099", clue="unrelated", answer="ZETA", enumeration="4"))
    assert validate(rows) == []


# ---------- the published branch must be reachable, and admitting ignorance must be explicit ----

def test_published_row_without_a_date_fails_unless_acknowledged():
    rows = [base_row(id=f"cb-000{i}", clue=f"clue number {WORDS[i]}", answer=f"ALPH{chr(64 + i)}",
                     split="dev") for i in range(1, N_DEV + 1)]
    bad = base_row(id="cb-0099", clue="published clue", answer="ZETA", enumeration="4",
                   source="published")
    assert any("published_date" in e for e in validate([*rows, bad]))
    ok = {**bad, "notes": {"published_date_unknown": True}}
    assert validate([*rows, ok]) == []


# ---------- item 6: resumption must not mix configurations ----------

def test_config_fingerprint_changes_with_any_generate_field():
    a = config_fingerprint("m", {}, {"reasoning_effort": "high", "max_tokens": 16000}, False)
    b = config_fingerprint("m", {}, {"reasoning_effort": "low", "max_tokens": 16000}, False)
    c = config_fingerprint("m", {"x": 1}, {"reasoning_effort": "high", "max_tokens": 16000}, False)
    d = config_fingerprint("m2", {}, {"reasoning_effort": "high", "max_tokens": 16000}, False)
    e = config_fingerprint("m", {}, {"reasoning_effort": "high", "max_tokens": 16000}, True)
    assert len({a, b, c, d, e}) == 5


def test_config_fingerprint_is_order_insensitive():
    assert config_fingerprint("m", {}, {"max_tokens": 10, "batch": True}, False) == \
           config_fingerprint("m", {}, {"batch": True, "max_tokens": 10}, False)


def test_sync_changes_the_fingerprint_so_a_probe_never_resumes_a_batched_run():
    cfg = {"generate": {"max_tokens": 8, "batch": True}}
    batched, _ = generate_fields(cfg, sync=False, structured=False)
    sync, _ = generate_fields(cfg, sync=True, structured=False)
    assert "batch" in batched and "batch" not in sync
    assert config_fingerprint("m", {}, batched, False) != config_fingerprint("m", {}, sync, False)


# ---------- truncation is tracked, not absorbed into accuracy ----------

def test_truncation_rate_is_reported_and_excluded_from_nothing():
    df = pd.DataFrame([
        {"system": "s", "family": "s", "role": "nocot", "task": "nocot", "dataset": "nat",
         "id": f"cb-{i:04d}", "correct": float(i % 2), "target": f"ANS{i}", "model": "m",
         "no_answer_line": i == 3, "len_ok": True, "truncated": i == 3, "has_extra": False,
         "structured_output": False, "input_tokens": 1, "output_tokens": 1,
         "reasoning_tokens": 0, "device": "charade", "hops": np.nan,
         "published_date": None}
        for i in range(10)
    ])
    ps = per_system(df, rng())
    assert ps["truncation_rate"].iloc[0] == pytest.approx(0.1)
    assert ps["n_clues"].iloc[0] == 10   # truncated samples stay in the denominator: they are answers
