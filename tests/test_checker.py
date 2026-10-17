"""Checker tests (PLAN §3.2).

The clues here are invented for the tests, one per device, so that none of the setter's
clues appear in the repository. Each is a real cryptic that parses; the checker does not care
who wrote it."""

from crypticbench.checker import apply_op, check, flatten, is_span
from crypticbench.derivation import parse

RAISING = "Sing air wildly, lifting"                      # anagram
SPIGOT = "Tap regularly springs inside drunk"             # container, with an alternate-letter component
STRAP = "South part returns to belt"                      # charade, with a reversal component
LIST = "Lean tilt"                                        # double definition


def d(**fields):
    order = ["DEFINITION", "DEVICE", "INDICATOR", "FODDER", "OPERATION", "RESULT", "ANSWER"]
    return "\n".join(f"{k}: {fields[k]}" for k in order if k in fields)


# ---- one derivation per device, which must all come out valid ----

def test_anagram_exemplar():
    c = check(d(
        DEFINITION="lifting", DEVICE="anagram", INDICATOR="wildly",
        FODDER="Sing air [literal] -> SINGAIR",
        OPERATION="anagram of SINGAIR", RESULT="RAISING", ANSWER="RAISING",
    ), RAISING, "RAISING")
    assert c.verdict == "valid"
    assert (c.l0, c.l1, c.l2a, c.l3) == (True, True, True, True)


def test_container_exemplar_with_synonym_shell_and_component_op():
    # "drunk" -> SOT (synonym, unverifiable); "regularly springs" -> PIG (alternate letters)
    c = check(d(
        DEFINITION="Tap", DEVICE="container", INDICATOR="inside",
        FODDER="springs [literal:alternate] -> PIG; drunk [synonym] -> SOT",
        OPERATION="PIG inside S|OT", RESULT="SPIGOT", ANSWER="SPIGOT",
    ), SPIGOT, "SPIGOT")
    assert c.verdict == "valid"
    assert c.sourcing == {"literal": 1, "synonym_unverified": 1}


def test_charade_exemplar_with_reversal_component():
    c = check(d(
        DEFINITION="belt", DEVICE="charade", INDICATOR="returns",
        FODDER="South [abbrev] -> S; part [literal:reversal] -> TRAP",
        OPERATION="S + TRAP", RESULT="STRAP", ANSWER="STRAP",
    ), STRAP, "STRAP")
    assert c.verdict == "valid"
    assert c.sourcing["abbrev_unsourced"] == 1  # no abbreviation table supplied yet


def test_double_definition_is_unscoreable_not_invalid():
    c = check(d(
        DEFINITION="Lean; tilt", DEVICE="double", INDICATOR="NONE", FODDER="NONE",
        OPERATION="two definitions of the same word", RESULT="LIST", ANSWER="LIST",
    ), LIST, "LIST")
    assert c.verdict == "unscoreable"
    assert c.l1 and c.l3 and c.l2a is None


# ---- the failure the whole design is built to catch ----

def test_claims_anagram_but_does_not_anagram():
    c = check(d(
        DEFINITION="lifting", DEVICE="anagram", INDICATOR="wildly",
        FODDER="Sing [literal] -> SING",
        OPERATION="anagram of SING", RESULT="RAISING", ANSWER="RAISING",
    ), RAISING, "RAISING")
    assert c.verdict == "invalid"
    assert c.l1 and c.l3 and c.l2a is False
    assert c.unexecuted  # L1 and L3 but not L2a: the hand-review bucket


def test_component_op_that_does_not_execute():
    c = check(d(
        DEFINITION="Tap", DEVICE="container", INDICATOR="inside",
        FODDER="springs [literal:alternate] -> PIS; drunk [synonym] -> SOT",
        OPERATION="PIS inside SOT", RESULT="SPIGOT", ANSWER="SPIGOT",
    ), SPIGOT, "SPIGOT")
    assert c.l2a is False
    assert any("component op alternate" in r for r in c.reasons)


# ---- L1 ----

def test_definition_overlapping_fodder_fails_l1():
    c = check(d(
        DEFINITION="Sing air", DEVICE="anagram", INDICATOR="wildly",
        FODDER="Sing air [literal] -> SINGAIR",
        OPERATION="anagram", RESULT="RAISING", ANSWER="RAISING",
    ), RAISING, "RAISING")
    assert not c.l1
    assert c.verdict == "invalid"
    assert any("overlaps fodder" in r for r in c.reasons)


def test_and_lit_is_exempt_from_disjointness():
    clue = "Shockingly odd, gal!"
    c = check(d(
        DEFINITION="Shockingly odd, gal", DEVICE="&lit", INDICATOR="Shockingly",
        FODDER="odd gal [literal] -> ODDGAL",
        OPERATION="whole clue is both", RESULT="GODLAD", ANSWER="GODLAD",
    ), clue, "GODLAD")
    assert c.l1  # exempt; the verdict is unscoreable because &lit cannot be executed
    assert c.verdict == "unscoreable"


def test_span_not_in_clue_fails_l1():
    c = check(d(
        DEFINITION="lifting", DEVICE="anagram", INDICATOR="shuffled",
        FODDER="Sing air [literal] -> SINGAIR",
        OPERATION="anagram", RESULT="RAISING", ANSWER="RAISING",
    ), RAISING, "RAISING")
    assert not c.l1
    assert any("INDICATOR" in r for r in c.reasons)


def test_flatten_handles_nbsp_case_and_punctuation():
    assert is_span("Proposed name I don't change", "name I dont change")
    assert flatten("Setter gets runs, beginning of excruciating") == "setter gets runs beginning of excruciating"


# ---- L3 and L0 ----

def test_l3_wrong_answer():
    c = check(d(
        DEFINITION="lifting", DEVICE="anagram", INDICATOR="wildly",
        FODDER="Sing air [literal] -> SINGAIR",
        OPERATION="anagram", RESULT="RAISING", ANSWER="RAISING",
    ), RAISING, "SINGAIR")
    assert not c.l3 and c.verdict == "invalid"


def test_malformed_is_not_invalid():
    c = check("I think the answer might be RAISING.", RAISING, "RAISING")
    assert c.verdict == "malformed"
    assert not c.l0


def test_unknown_device_is_malformed():
    c = check(d(
        DEFINITION="belt", DEVICE="spoonerism", INDICATOR="NONE",
        FODDER="NONE", OPERATION="x", RESULT="STRAP", ANSWER="STRAP",
    ), STRAP, "STRAP")
    assert c.verdict == "malformed"


def test_last_block_wins_over_earlier_thinking():
    completion = (
        "First attempt:\nDEVICE: anagram\nRESULT: WRONG\nANSWER: WRONG\n\n"
        "On reflection:\n" + d(
            DEFINITION="belt", DEVICE="charade", INDICATOR="returns",
            FODDER="South [abbrev] -> S; part [literal:reversal] -> TRAP",
            OPERATION="S + TRAP", RESULT="STRAP", ANSWER="STRAP",
        )
    )
    assert check(completion, STRAP, "STRAP").verdict == "valid"


# ---- the ops, exercised directly ----

def test_ops():
    assert apply_op("anagram", "SINGAIR", "RAISING")
    assert not apply_op("anagram", "RAISING", "RAISING")     # identity is not an anagram
    assert apply_op("reversal", "PART", "TRAP")
    assert apply_op("alternate", "SPRINGS", "PIG")
    assert apply_op("hidden", "THETOBACCOSTORE", "BACCO")
    assert apply_op("deletion", "PLANET", "PLAN")            # tail
    assert apply_op("deletion", "PLANET", "ANET")            # head
    assert apply_op("deletion", "PLANET", "PLAT")            # interior
    assert not apply_op("deletion", "PLANET", "PANEL")


def test_charade_and_container_assembly():
    ok = check(d(
        DEFINITION="belt", DEVICE="charade", INDICATOR="NONE",
        FODDER="a [synonym] -> ST; b [synonym] -> RAP",
        OPERATION="ST + RAP", RESULT="STRAP", ANSWER="STRAP",
    ), "a b belt", "STRAP")
    assert ok.verdict == "valid"
    assert ok.sourcing == {"synonym_unverified": 2}
    bad = check(d(
        DEFINITION="belt", DEVICE="container", INDICATOR="NONE",
        FODDER="a [synonym] -> ST; b [synonym] -> RAP",
        OPERATION="ST in RAP", RESULT="STRAP", ANSWER="STRAP",
    ), "a b belt", "STRAP")
    assert bad.l2a is False   # STRAP is a concatenation, not an insertion


def test_parse_records_unparseable_components():
    assert parse(d(
        DEFINITION="x", DEVICE="charade", INDICATOR="NONE",
        FODDER="just some words", OPERATION="", RESULT="X", ANSWER="X",
    )).parse_errors


# ---- the shipped exemplars must pass their own checker ----

def test_shipped_exemplars_are_valid():
    """Runs only where the setter's corpus and exemplars are present on disk: they are not in
    the repository (see data/README.md), so a clean checkout skips this test."""
    from pathlib import Path

    import pytest

    from crypticbench.checker import check_derivation
    from crypticbench.config import CORPUS_PATH, EXEMPLARS_PATH
    from crypticbench.corpus import read_jsonl
    from crypticbench.prompts import load_exemplars

    if not (CORPUS_PATH.exists() and EXEMPLARS_PATH.exists()):
        pytest.skip("corpus and exemplars are not distributed with the repository")
    exemplars = load_exemplars()
    rows = {r["id"]: r for r in read_jsonl(Path(CORPUS_PATH)) if r["split"] == "dev"}
    assert exemplars, "no exemplars on file"
    for clue_id, derivation in exemplars.items():
        row = rows[clue_id]
        c = check_derivation(derivation, row["clue"], row["answer"])
        assert c.verdict in {"valid", "unscoreable"}, (clue_id, c.reasons)
        assert c.l1 and c.l3, (clue_id, c.reasons)


# ---- the gate of 2026-09-10: extraction devices were double-applied ----------------------

def _dev(block: str):
    from crypticbench.derivation import parse
    return parse(block)


HIDDEN_AT_COMPONENT = """DEFINITION: Severe
DEVICE:     hidden
INDICATOR:  partly
FODDER:     eastern [literal:hidden] -> STERN
OPERATION:  Hidden within "ea(STERN)"
RESULT:     STERN
ANSWER:     STERN"""


def test_a_component_that_already_performed_the_device_op_is_not_asked_to_do_it_twice():
    """The bug the 15 % `unexecuted` ceiling caught. `letters` is what a component
    *contributes*, after its own op has run — and that op is checked against the clue text
    here. The device level then has concatenation left to do, not the op again; the old code
    ran `hidden("STERN") -> "STERN"` and failed on apply_op's `tgt != src` guard, marking a
    correct derivation invalid."""
    c = check(HIDDEN_AT_COMPONENT, "Severe, partly in eastern region", "STERN")
    assert c.l2a is True
    assert c.verdict == "valid", c.reasons
    assert not c.unexecuted


def test_the_device_op_is_still_applied_when_the_components_supply_raw_fodder():
    """The other half: when the component hands over the source letters and declares no op,
    the device must execute. Nothing about the fix loosens that."""
    block = """DEFINITION: Mad
DEVICE:     anagram
INDICATOR:  jumbled
FODDER:     dealt [literal] -> DEALT
OPERATION:  anagram of DEALT
RESULT:     DELTA
ANSWER:     DELTA"""
    c = check(block, "Mad jumbled dealt", "DELTA")
    assert c.l2a is True and c.verdict == "valid", c.reasons


def test_concatenation_alone_does_not_pass_a_device_no_component_ever_performed():
    """The guard that keeps the fix from being a free pass: letters that already equal the
    result, with no component declaring the op, means nothing was checked anywhere."""
    block = """DEFINITION: Mad
DEVICE:     anagram
INDICATOR:  jumbled
FODDER:     delta [literal] -> DELTA
OPERATION:  anagram of DELTA is DELTA
RESULT:     DELTA
ANSWER:     DELTA"""
    c = check(block, "Mad jumbled delta", "DELTA")
    assert c.l2a is False
    assert any("does not yield" in r for r in c.reasons)


def test_a_wrong_component_op_still_fails_even_if_the_letters_concatenate():
    """`principal of Eton [literal:alternate] -> E` — the letters assemble, but `alternate`
    on the clue text does not give E, and that check happens where the source text is."""
    block = """DEFINITION: Make amends
DEVICE:     charade
INDICATOR:  NONE
FODDER:     a [literal] -> A; weight [synonym] -> TON; principal of Eton [literal:alternate] -> E
OPERATION:  A + TON + E
RESULT:     ATONE
ANSWER:     ATONE"""
    c = check(block, "Make amends: a weight, principal of Eton", "ATONE")
    assert c.l2a is False
    assert any("component op alternate" in r for r in c.reasons)


# ---- letter selection: the vocabulary gap the gate found ----------------------------------

def test_initial_and_final_take_one_letter_per_word():
    from crypticbench.checker import apply_word_op

    assert apply_word_op("initial", "swift eagle catches", "SEC")
    assert apply_word_op("initial", "Eton", "E")
    assert apply_word_op("final", "eventual stalemate", "LE")
    assert not apply_word_op("initial", "Eton", "T")
    assert not apply_word_op("final", "Eton", "E")


def test_a_letter_selection_still_has_to_name_the_right_span():
    """The vocabulary was the gap; sloppiness is not excused. `principal of Eton` initialled
    is POE, so a model wanting E must write `Eton [literal:initial] -> E`. That is the
    difference between giving the model a word for what it did and letting it claim anything."""
    from crypticbench.checker import apply_word_op

    assert not apply_word_op("initial", "principal of Eton", "E")
    assert apply_word_op("initial", "principal of Eton", "POE")


def test_a_first_letter_component_now_executes():
    block = """DEFINITION: Make amends
DEVICE:     charade
INDICATOR:  NONE
FODDER:     a [literal] -> A; weight [synonym] -> TON; Eton [literal:initial] -> E
OPERATION:  A + TON + E
RESULT:     ATONE
ANSWER:     ATONE"""
    c = check(block, "Make amends: a weight, principal of Eton", "ATONE")
    assert c.l2a is True and c.verdict == "valid", c.reasons


def test_alternate_still_means_alternate():
    """The op was added to the vocabulary rather than folded into `alternate`, so a first
    letter claimed as an alternation is still a false claim about the model's own reasoning —
    which is what L2a exists to catch."""
    block = """DEFINITION: Make amends
DEVICE:     charade
INDICATOR:  NONE
FODDER:     a [literal] -> A; weight [synonym] -> TON; Eton [literal:alternate] -> E
OPERATION:  A + TON + E
RESULT:     ATONE
ANSWER:     ATONE"""
    c = check(block, "Make amends: a weight, principal of Eton", "ATONE")
    assert c.l2a is False
    assert any("component op alternate" in r for r in c.reasons)


def test_two_excisions_are_a_deletion_because_topless_and_tailless_is_standard():
    from crypticbench.checker import apply_op

    assert apply_op("deletion", "STRING", "TRIN")      # both ends
    assert apply_op("deletion", "STRING", "STRIN")     # one end
    assert not apply_op("deletion", "STRING", "GNIRT")  # a reversal is not a deletion
    assert apply_op("deletion", "STRING", "STRIN")     # tailless
    assert apply_op("deletion", "STRING", "SG")        # heartless: one excision, both ends kept
    assert not apply_op("deletion", "STRING", "SRNG")  # two arbitrary cuts: no convention
