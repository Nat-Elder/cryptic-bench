"""The four-level derivation checker (PLAN §3.2). Mechanical throughout: no rubric, no judge.

    L0 well-formed  the template parses and ANSWER is extractable
    L1 grounded     DEFINITION and INDICATOR are clue spans; `literal` components are clue
                    spans; DEFINITION and the literal components are disjoint
    L2a executes    the claimed device, applied to the claimed components, yields RESULT
    L2b sourced     coverage only: how each component's letters were obtained
    L3 consistent   RESULT == ANSWER == gold

    valid = L1 and L2a and L3

L2b is never a validity gate. A synonym step is not mechanically checkable and pretending
otherwise is how this design would fail; what *is* checkable is the letter arithmetic, and
"claims anagram, does not anagram" is caught by L2a alone.

Verdicts: `valid`, `invalid`, `unscoreable` (the model claimed a non-executable device, so it
chose its own exclusion — see PLAN §3.3 on reporting both bounds), `malformed`.
`unexecuted` is a flag, not a verdict: L1 and L3 hold but L2a fails, which is either a checker
bug or a model that got there by other means. Over 15 % of those and the checker is wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .derivation import (
    DISJOINTNESS_EXEMPT,
    EXECUTABLE,
    WORD_OPS,
    Component,
    Derivation,
    parse,
)
from .normalise import norm_answer

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


# ---- clue-span grounding ----------------------------------------------------------------

def flatten(text: str) -> str:
    """Lowercase, strip punctuation and non-breaking spaces, collapse whitespace.

    Spans are compared in this form: models vary case and punctuation, and `times200` clues
    contain U+00A0. "Exact span" means exact words, not exact bytes.
    """
    text = text.replace(" ", " ").replace("’", "'").lower()
    text = _PUNCT_RE.sub(" ", text.replace("'", ""))
    return re.sub(r"\s+", " ", text).strip()


def span_positions(clue: str, span: str) -> list[tuple[int, int]]:
    """Every (start, end) where `span` occurs in the flattened clue, on word boundaries."""
    flat_clue, flat_span = flatten(clue), flatten(span)
    if not flat_span:
        return []
    out = []
    for m in re.finditer(re.escape(flat_span), flat_clue):
        start, end = m.start(), m.end()
        before_ok = start == 0 or flat_clue[start - 1] == " "
        after_ok = end == len(flat_clue) or flat_clue[end] == " "
        if before_ok and after_ok:
            out.append((start, end))
    return out


def is_span(clue: str, span: str) -> bool:
    return bool(span_positions(clue, span))


def _can_be_disjoint(clue: str, a: str, b: str) -> bool:
    """True if some occurrence of `a` and some occurrence of `b` do not overlap."""
    for a_start, a_end in span_positions(clue, a):
        for b_start, b_end in span_positions(clue, b):
            if a_end <= b_start or b_end <= a_start:
                return True
    return False


# ---- L2a: the letter arithmetic --------------------------------------------------------

def _letters(component: Component) -> str:
    return norm_answer(component.letters)


def apply_op(op: str, source_letters: str, target: str) -> bool:
    """Does `op` applied to `source_letters` yield `target`? Used for component-level ops."""
    src, tgt = norm_answer(source_letters), norm_answer(target)
    if not src or not tgt:
        return False
    if op == "anagram":
        return sorted(src) == sorted(tgt) and src != tgt
    if op == "reversal":
        return src[::-1] == tgt
    if op == "hidden":
        return tgt in src and tgt != src
    if op == "alternate":
        return src[0::2] == tgt or src[1::2] == tgt
    if op == "deletion":
        # The standard Ximenean set and no more: cut a prefix, cut a suffix, cut both ends
        # ("topless and tailless" — `string` gives TRIN), or cut the middle ("heartless").
        # The first three are exactly "the survivors are a contiguous substring"; the fourth is
        # a single excision. A single excision alone could not express both-ends, which is why
        # 19 of the gate's failures were this rule rather than the models' arithmetic (gate
        # report §6.1) — but *two arbitrary* excisions would admit STRING -> SRNG, which no
        # convention allows, so the two cases are named rather than counted.
        if len(tgt) >= len(src):
            return False
        if tgt in src:
            return True
        return any(src[:i] + src[j:] == tgt
                   for i in range(len(src) + 1)
                   for j in range(i + 1, len(src) + 1))
    return False


def apply_word_op(op: str, source_text: str, target: str) -> bool:
    """Letter-selection ops, which need the **word boundaries** and so cannot use `src`.

    `initial` and `final` — "initially", "at heart", "endings" — take one letter per word, and
    the gate of 2026-09-10 is why they exist: 15 component claims across 220 rows were letter
    selections tagged `alternate`, because the template offered no word for them
    (`[withheld] [literal:alternate] -> E`). A model that cannot express a legitimate
    step in the vocabulary it was handed writes an invalid derivation for a correct solve, and
    that lands in M2 — the metric this study is for. So the vocabulary gained the word rather
    than the checker gaining a tolerance: `alternate` still means alternate.

    Both accept the whole run of firsts (`SEC` from "swift eagle catches") and the first letter
    of a single word, which is the same rule applied to a one-word span.
    """
    words = [w for w in re.split(r"[^A-Za-z]+", source_text) if w]
    tgt = norm_answer(target)
    if not words or not tgt:
        return False
    if op == "initial":
        return "".join(w[0] for w in words).upper() == tgt
    if op == "final":
        return "".join(w[-1] for w in words).upper() == tgt
    return False


def _assemble(device: str, parts: list[str], result: str,
              component_ops: set[str] | None = None) -> bool:
    """Do the components' contributions assemble into RESULT under the claimed device?

    **`parts` are contributions, not raw fodder.** The template asks for
    `<text> [source:op] -> <letters>`, where `letters` is what that component contributes
    *after* its own op has run — and `check_derivation` verifies that op against the component's
    clue text before getting here. So when the device's operation has already been performed at
    component level, the step left at device level is concatenation, and re-applying the op is
    double application: `eastern [literal:hidden] -> STERN` with `RESULT: STERN` was
    being checked as `hidden("STERN") -> "STERN"`, which fails on `apply_op`'s `tgt != src`
    guard. That marked correct derivations invalid and drove `unexecuted` to 25 % at the gate of
    2026-09-10, over PLAN §2.3's 15 % ceiling — the ceiling did its job.

    So an extraction device assembles if **either** the op runs here (components supplied raw
    fodder) **or** the contributions concatenate to RESULT *and* some component declared that op,
    so it was checked where the source text is. The second clause needs that guard: without it,
    a model could claim `DEVICE: anagram`, hand over letters that already equal the answer, and
    have nothing checked anywhere.

    A component op on a `synonym` or `abbrev` source is still unverifiable — `unverified_ops`
    counts it and L2b reports the coverage (PLAN §2.2: synonyms are counted, never verified).
    That hole is the design's, not this function's.
    """
    joined = "".join(parts)
    res = norm_answer(result)
    if not joined or not res:
        return False
    if device == "charade":
        return joined == res
    if device == "container":
        # one component goes inside the concatenation of the others, in clue order
        if len(parts) < 2:
            return False
        for i, filler in enumerate(parts):
            shell = "".join(parts[:i] + parts[i + 1:])
            if not filler or not shell:
                continue
            if any(shell[:k] + filler + shell[k:] == res for k in range(1, len(shell))):
                return True
        return False
    if apply_op(device, joined, res):
        return True
    return joined == res and device in (component_ops or set())


# ---- results ---------------------------------------------------------------------------

@dataclass
class Check:
    verdict: str = "malformed"          # valid | invalid | unscoreable | malformed
    l0: bool = False
    l1: bool = False
    l2a: bool | None = None            # None where the device is not executable
    l3: bool = False
    unexecuted: bool = False
    sourcing: dict[str, int] = field(default_factory=dict)   # L2b coverage
    unverified_ops: int = 0            # component ops on abbrev/synonym letters: not checkable
    reasons: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.verdict == "valid"


def check_derivation(d: Derivation, clue: str, gold: str, abbreviations: dict[str, str] | None = None) -> Check:
    c = Check()
    if not d.well_formed:
        c.reasons.extend(d.parse_errors or ["ANSWER not extractable"])
        return c
    c.l0 = True

    # ---- L1
    l1_reasons: list[str] = []
    for span in d.definition:
        if not is_span(clue, span):
            l1_reasons.append(f"DEFINITION {span!r} is not a clue span")
    if d.indicator and not is_span(clue, d.indicator):
        l1_reasons.append(f"INDICATOR {d.indicator!r} is not a clue span")
    literals = [comp for comp in d.fodder if comp.source == "literal"]
    for comp in literals:
        if not is_span(clue, comp.text):
            l1_reasons.append(f"literal component {comp.text!r} is not a clue span")
    if d.device not in DISJOINTNESS_EXEMPT:
        for span in d.definition:
            for comp in literals:
                if is_span(clue, span) and is_span(clue, comp.text) and not _can_be_disjoint(clue, span, comp.text):
                    l1_reasons.append(f"DEFINITION {span!r} overlaps fodder {comp.text!r}")
    c.l1 = not l1_reasons
    c.reasons.extend(l1_reasons)

    # ---- L2b coverage (never a gate)
    table = abbreviations or {}
    for comp in d.fodder:
        if comp.source == "literal":
            key = "literal"
        elif comp.source == "abbrev":
            key = "abbrev_known" if norm_answer(comp.text) in table or comp.text.lower() in table else "abbrev_unsourced"
        else:
            key = "synonym_unverified"
        c.sourcing[key] = c.sourcing.get(key, 0) + 1

    # ---- L3
    res, ans, gold_n = norm_answer(d.result), norm_answer(d.answer), norm_answer(gold)
    c.l3 = bool(res) and res == ans == gold_n
    if not c.l3:
        c.reasons.append(f"RESULT {res!r} / ANSWER {ans!r} / gold {gold_n!r} disagree")

    # ---- L2a
    if d.device not in EXECUTABLE:
        c.verdict = "unscoreable"
        c.reasons.append(f"device {d.device!r} is not mechanically executable")
        return c

    if not d.fodder:
        c.l2a = False
        c.reasons.append("no fodder components to execute")
    else:
        parts, op_ok = [], True
        for comp in d.fodder:
            letters = _letters(comp)
            if comp.op:
                if comp.op == d.device and letters == norm_answer(comp.text):
                    # Raw fodder for the device's own operation: `a fun ride
                    # [literal:anagram] -> AFUNRIDE` under `DEVICE: anagram` is naming the
                    # letters the anagram acts on, not claiming a transformation of its own.
                    # `_assemble` checks the anagram, and requiring it here as well demands the
                    # op twice — the mirror of the double-application bug (gate report §5.1).
                    pass
                elif comp.source == "literal":
                    ok = (apply_word_op(comp.op, comp.text, letters) if comp.op in WORD_OPS
                          else apply_op(comp.op, norm_answer(comp.text), letters))
                    if not ok:
                        op_ok = False
                        c.reasons.append(f"component op {comp.op} on {comp.text!r} does not yield {letters!r}")
                else:
                    c.unverified_ops += 1   # op on a substituted string: not checkable
            parts.append(letters)
        assembled = _assemble(d.device, parts, d.result,
                              {comp.op for comp in d.fodder if comp.op})
        if not assembled:
            c.reasons.append(f"{d.device} of {parts} does not yield {norm_answer(d.result)!r}")
        c.l2a = op_ok and assembled

    if c.l1 and c.l3 and not c.l2a:
        c.unexecuted = True
    c.verdict = "valid" if (c.l1 and c.l2a and c.l3) else "invalid"
    return c


def check(completion: str, clue: str, gold: str, abbreviations: dict[str, str] | None = None) -> Check:
    """Parse a raw completion and check it."""
    return check_derivation(parse(completion), clue, gold, abbreviations)
