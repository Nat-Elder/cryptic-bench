"""The v2 derivation template: parse and render (PLAN §3.1).

A derivation is seven labelled fields. `FODDER` is a **tagged component list**, because a
single clue-substring fodder cannot express a charade — STRAP is S ("south") + TRAP ("part"
reversed) — and `times200` marks 66 of its 200 clues `requires_synonym`.

Component syntax, one per semicolon:

    <text> [<source>[:<op>]] -> <LETTERS>

`source` is how the letters were obtained (`literal` = an exact span of the clue, `abbrev` = a
crossword abbreviation, `synonym` = a substitution); `op` is an optional transformation applied
to that component alone, which is what lets one top-level DEVICE cover composite clues.

Parsing is forgiving about case, whitespace and surrounding prose (thinking models write before
the block) and strict about structure: the *last* occurrence of each label wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .normalise import norm_answer

FIELDS = ("DEFINITION", "DEVICE", "INDICATOR", "FODDER", "OPERATION", "RESULT", "ANSWER")

DEVICES = (
    "anagram", "hidden", "reversal", "alternate", "charade",
    "container", "deletion", "homophone", "double", "cryptic-def", "&lit",
)
# L2a is defined only for these; the rest are scored to L1 and reported unscoreable (PLAN §3.3)
EXECUTABLE = ("anagram", "hidden", "reversal", "alternate", "charade", "container", "deletion")
# L1's definition/fodder disjointness test is meaningless for these two
DISJOINTNESS_EXEMPT = ("&lit", "cryptic-def")

SOURCES = ("literal", "abbrev", "synonym")
# Component ops that need word boundaries rather than the concatenated letters: one letter per
# word. Added 2026-09-10 at the gate, where 15 component claims were letter selections tagged
# `alternate` for want of a word for them (gate report §6.1). `checker.apply_word_op` runs them.
WORD_OPS = ("initial", "final")
COMPONENT_OPS = ("anagram", "hidden", "reversal", "alternate", "deletion",
                 "initial", "final")

NONE = "NONE"

_LABEL_RE = {f: re.compile(rf"^\s*{f}\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE) for f in FIELDS}
# `<text> [source]` or `<text> [source:op]`, then `-> <letters>`. The `op` group takes only a
# bare word, so anything else after the colon falls to `gloss` — models write
# `[synonym: "circa"]` and `[synonym: passing 'Go' on the board]`, i.e. a note on *how* the
# substitution was made. That is commentary, not a second claim: the tag is what L2a checks, so
# the gloss is captured and set aside rather than failing the whole component to parse (gate
# report §6.1). A gloss whose text happens to name an op is **not** read as one — a claim has
# to be made in the slot the template provides for it.
_COMPONENT_RE = re.compile(
    r"^\s*(?P<text>.*?)\s*\[\s*(?P<source>\w+)\s*"
    r"(?::\s*(?:(?P<op>[\w-]+)\s*|(?P<gloss>[^\]]*?)\s*))?\]\s*->\s*(?P<letters>.+?)\s*$"
)


@dataclass
class Component:
    text: str
    source: str
    letters: str
    op: str | None = None
    gloss: str | None = None    # free text after the colon: recorded, never checked

    def render(self) -> str:
        tag = self.source if not self.op else f"{self.source}:{self.op}"
        return f"{self.text} [{tag}] -> {self.letters}"


@dataclass
class Derivation:
    definition: list[str] = field(default_factory=list)
    device: str = ""
    indicator: str | None = None
    fodder: list[Component] = field(default_factory=list)
    operation: str = ""
    result: str = ""
    answer: str = ""
    parse_errors: list[str] = field(default_factory=list)

    @property
    def well_formed(self) -> bool:
        """L0: the template parsed and ANSWER is extractable."""
        return not self.parse_errors and bool(norm_answer(self.answer))

    def render(self) -> str:
        fodder = "; ".join(c.render() for c in self.fodder) or NONE
        return "\n".join([
            f"DEFINITION: {'; '.join(self.definition)}",
            f"DEVICE:     {self.device}",
            f"INDICATOR:  {self.indicator or NONE}",
            f"FODDER:     {fodder}",
            f"OPERATION:  {self.operation}",
            f"RESULT:     {self.result}",
            f"ANSWER:     {self.answer}",
        ])


def _last(text: str, label: str) -> str | None:
    matches = _LABEL_RE[label].findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def parse_component(raw: str) -> Component | None:
    m = _COMPONENT_RE.match(raw)
    if not m:
        return None
    op = (m.group("op") or "").lower() or None
    gloss = (m.group("gloss") or "").strip() or None
    return Component(
        text=m.group("text").strip(),
        source=m.group("source").strip().lower(),
        letters=m.group("letters").strip(),
        op=op,
        gloss=gloss,
    )


def parse(text: str) -> Derivation:
    """Parse a completion into a Derivation. Structural problems land in `parse_errors`."""
    d = Derivation()
    values = {f: _last(text, f) for f in FIELDS}

    for f in ("DEVICE", "RESULT", "ANSWER"):
        if values[f] is None:
            d.parse_errors.append(f"missing {f}")
    if values["DEFINITION"] is None:
        d.parse_errors.append("missing DEFINITION")

    d.device = (values["DEVICE"] or "").strip().lower()
    if d.device and d.device not in DEVICES:
        d.parse_errors.append(f"unknown DEVICE {d.device!r}")

    definition = values["DEFINITION"] or ""
    d.definition = [s.strip() for s in definition.split(";") if s.strip()]

    indicator = (values["INDICATOR"] or NONE).strip()
    d.indicator = None if indicator.upper() == NONE else indicator

    fodder = (values["FODDER"] or NONE).strip()
    if fodder.upper() != NONE:
        for raw in fodder.split(";"):
            if not raw.strip():
                continue
            component = parse_component(raw)
            if component is None:
                d.parse_errors.append(f"unparseable component {raw.strip()!r}")
                continue
            if component.source not in SOURCES:
                d.parse_errors.append(f"unknown source {component.source!r}")
            if component.op and component.op not in COMPONENT_OPS:
                d.parse_errors.append(f"unknown component op {component.op!r}")
            d.fodder.append(component)

    d.operation = (values["OPERATION"] or "").strip()
    d.result = (values["RESULT"] or "").strip()
    d.answer = (values["ANSWER"] or "").strip()
    return d
