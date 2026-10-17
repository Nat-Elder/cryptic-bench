"""Answer and enumeration normalisation.

Conventions (design.md §1, §5):
- answers are compared as uppercase A–Z only; spaces, hyphens and apostrophes are dropped;
- enumerations are strings like "5", "3,4", "5-3", "3-5,3" derived from the display form;
- the enumeration in a clue's surface text is the trailing "(...)" group.
"""

from __future__ import annotations

import re

_ENUM_RE = re.compile(r"\s*\(\s*(\d[\d,\-\s]*)\s*\)\s*$")
_DROP_CHARS = " -'’"


def norm_answer(s: str) -> str:
    """Uppercase, drop spaces/hyphens/apostrophes, keep A–Z only."""
    s = s.strip().upper()
    for ch in _DROP_CHARS:
        s = s.replace(ch, "")
    return "".join(ch for ch in s if "A" <= ch <= "Z")


def non_alpha_residue(s: str) -> str:
    """Characters that norm_answer discards *other than* the expected separators.

    Non-empty means the answer needs a ruling (digits, accents, ampersands...).
    """
    s = s.strip().upper()
    for ch in _DROP_CHARS:
        s = s.replace(ch, "")
    return "".join(ch for ch in s if not ("A" <= ch <= "Z"))


def strip_enumeration(clue: str) -> tuple[str, str | None]:
    """Split "Severe, partly in eastern region (5)" -> ("Severe, partly in eastern region", "5")."""
    m = _ENUM_RE.search(clue)
    if not m:
        return clue.strip(), None
    enum = re.sub(r"\s+", "", m.group(1))
    return clue[: m.start()].strip(), enum


def enumeration_from_display(display: str) -> str:
    """"ICE-CREAM VAN" -> "3-5,3"; "ICE CREAM" -> "3,5"; "MEDIC" -> "5"."""
    out = []
    count = 0
    for ch in display.strip():
        if ch in " -":
            if count:
                out.append(str(count))
                out.append("," if ch == " " else "-")
                count = 0
            continue
        if ch in "'’":
            continue
        count += 1
    if count:
        out.append(str(count))
    return "".join(out)


def enum_total(enum: str) -> int:
    """Total letters in an enumeration: "3-5,3" -> 11."""
    return sum(int(n) for n in re.findall(r"\d+", enum))


def enum_parts(enum: str) -> list[tuple[int, str]]:
    """"3-5,3" -> [(3, "-"), (5, ","), (3, "")]: each length and the separator after it."""
    tokens = re.findall(r"\d+|[,\-]", enum)
    parts: list[tuple[int, str]] = []
    i = 0
    while i < len(tokens):
        n = int(tokens[i])
        sep = tokens[i + 1] if i + 1 < len(tokens) and tokens[i + 1] in ",-" else ""
        parts.append((n, sep))
        i += 2 if sep else 1
    return parts


def display_from_grid(letters: str, enum: str) -> str:
    """Re-insert separators: ("ICECREAMVAN", "3-5,3") -> "ICE-CREAM VAN"."""
    letters = norm_answer(letters)
    parts = enum_parts(enum)
    if sum(n for n, _ in parts) != len(letters):
        raise ValueError(f"enumeration {enum} does not fit {letters!r}")
    out = []
    pos = 0
    for n, sep in parts:
        out.append(letters[pos : pos + n])
        pos += n
        if sep:
            out.append(" " if sep == "," else "-")
    return "".join(out)
