"""The `nocot` arm: parsing, the 5 % rule, and the structured-output fallback (PLAN §2.1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tasks.nocot import (  # noqa: E402
    EXTRA_TEXT_THRESHOLD,
    NOCOT_MAX_TOKENS,
    answer_schema,
    build_task,
    extra_text_summary,
    parse_nocot,
)

# ---- the shape the prompt asks for --------------------------------------------------------

def test_a_bare_answer_is_clean():
    p = parse_nocot("MEDIC", "5")
    assert p.answer == "MEDIC" and p.extra_text == "" and not p.has_extra


def test_a_two_word_answer_is_clean_against_its_enumeration():
    p = parse_nocot("AT LARGE", "2,5")
    assert p.answer == "ATLARGE" and not p.has_extra


def test_a_hyphenated_answer_is_clean():
    p = parse_nocot("ICE-CREAM VAN", "3-5,3")
    assert p.answer == "ICECREAMVAN" and not p.has_extra


def test_lowercase_is_still_scored():
    """The prompt asks for capitals; a model that ignores that is answering, not reasoning."""
    assert parse_nocot("medic", "5").answer == "MEDIC"


# ---- what the 5 % rule is counting --------------------------------------------------------

def test_trailing_punctuation_is_extra_text():
    p = parse_nocot("MEDIC.", "5")
    assert p.answer == "MEDIC" and p.extra_text == "." and p.has_extra


def test_working_on_a_second_line_is_extra_text():
    p = parse_nocot("MEDIC\n(anagram of 'medic')", "5")
    assert p.answer == "MEDIC"
    assert "anagram" in p.extra_text and p.has_extra


def test_an_answer_label_is_extra_text_not_silently_stripped():
    p = parse_nocot("ANSWER: MEDIC", "5")
    assert p.answer == "MEDIC" and p.has_extra and "ANSWER" in p.extra_text.upper()


def test_a_sentence_is_caught_by_the_word_count_when_no_character_is_left_over():
    """"The answer is MEDIC" leaves nothing outside the answer span, so extra_text alone would
    miss it. The enumeration says the answer is one word; the span is four."""
    p = parse_nocot("The answer is MEDIC", "5")
    assert p.extra_words == 3 and p.has_extra


def test_an_empty_completion_has_no_answer():
    p = parse_nocot("", "5")
    assert p.answer is None and not p.has_extra


def test_a_completion_with_no_letters_has_no_answer():
    p = parse_nocot("???", "5")
    assert p.answer is None and p.has_extra


# ---- the structured-output fallback -------------------------------------------------------

def test_the_json_shape_is_read_and_flagged():
    p = parse_nocot(json.dumps({"answer": "MEDIC"}), "5")
    assert p.answer == "MEDIC" and p.from_json and not p.has_extra


def test_extra_json_fields_are_extra_text():
    p = parse_nocot(json.dumps({"answer": "MEDIC", "reasoning": "anagram"}), "5")
    assert p.answer == "MEDIC" and p.has_extra and "reasoning" in p.extra_text


def test_malformed_json_falls_back_to_text_parsing():
    p = parse_nocot('{"answer": "MEDIC"', "5")
    assert p.answer is None or p.has_extra   # it is not silently accepted as a clean answer


def test_the_schema_has_exactly_one_field():
    schema = answer_schema()
    assert schema.json_schema.required == ["answer"]
    assert list(schema.json_schema.properties) == ["answer"]
    assert schema.json_schema.additionalProperties is False


# ---- the rule itself -----------------------------------------------------------------------

def test_the_rule_fires_above_the_threshold_and_names_the_fix():
    rows = [{"extra_text": "", "extra_words": 0} for _ in range(90)] + \
           [{"extra_text": " because", "extra_words": 0} for _ in range(10)]
    s = extra_text_summary(rows)
    assert s["extra_text_rate"] == pytest.approx(0.10)
    assert s["over_threshold"] and "structured" in s["note"]


def test_the_rule_is_quiet_at_the_threshold():
    rows = [{"extra_text": "", "extra_words": 0} for _ in range(95)] + \
           [{"extra_text": ".", "extra_words": 0} for _ in range(5)]
    s = extra_text_summary(rows)
    assert s["extra_text_rate"] == pytest.approx(EXTRA_TEXT_THRESHOLD)
    assert not s["over_threshold"]


def test_the_word_count_flag_counts_towards_the_rule():
    rows = [{"extra_text": "", "extra_words": 0} for _ in range(80)] + \
           [{"extra_text": "", "extra_words": 2} for _ in range(20)]
    assert extra_text_summary(rows)["over_threshold"]


def test_no_samples_is_not_a_pass():
    assert extra_text_summary([])["extra_text_rate"] is None


# ---- wiring ---------------------------------------------------------------------------------

ROWS = [{"id": "cb-0001", "clue": "Doctor turned up", "enumeration": "5", "answer": "MEDIC",
         "answer_display": "MEDIC"}]
DEV = [{"id": "cb-0002", "clue": "Lean tilt", "enumeration": "4", "answer": "LIST",
        "answer_display": "LIST"}]


def test_the_prompt_shows_pairs_and_ends_on_the_arrow():
    task, _ = build_task(ROWS, DEV)
    prompt = task.dataset[0].input
    assert "Clue: Lean tilt (4) → LIST" in prompt
    assert prompt.rstrip().endswith("→")
    assert "DEFINITION" not in prompt      # no derivation template in the no-CoT arm


def test_the_structured_flag_attaches_the_schema_and_the_default_does_not():
    plain, _ = build_task(ROWS, DEV)
    structured, _ = build_task(ROWS, DEV, structured=True)
    assert plain.config.response_schema is None
    assert structured.config.response_schema is not None


def test_the_token_ceiling_is_small_enough_to_forbid_working():
    assert NOCOT_MAX_TOKENS <= 16   # 16 ruled 2026-09-10; 8 truncated correct two-word answers
