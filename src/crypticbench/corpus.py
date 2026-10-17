"""Corpus rows: construction, validation, finalisation."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .config import N_DEV, SEED
from .normalise import (
    display_from_grid,
    enum_total,
    enumeration_from_display,
    non_alpha_residue,
    norm_answer,
    strip_enumeration,
)

# Provenance labels. `published` clues appeared in print (grid and clues online, answers in a
# separate solutions PDF); `unpublished` never appeared anywhere; `other` means provenance is
# not established. The label is not seenness — the cutoff split (PLAN §2.6) uses
# `published_date` against each system's `training_cutoff`, not this field.
SOURCES = {"published", "unpublished", "other"}


@dataclass
class Row:
    clue: str
    answer: str
    answer_display: str
    enumeration: str
    source: str
    published_date: str | None = None
    published_url: str | None = None
    setter: str = "nat"
    split: str = "test"
    id: str | None = None
    notes: dict = field(default_factory=dict)


def make_row(raw_clue: str, raw_answer: str, source: str, *, date_: str | None = None,
             url: str | None = None, trust_answer: bool = False) -> tuple[Row, list[str]]:
    """Build a row from a surface clue (enumeration optional) and an answer in any form.
    Returns the row and a list of warnings that need a human ruling.

    trust_answer: when the clue's enumeration contradicts the answer (grid files: the grid is
    ground truth), fall back to the answer's length and record the clue's enumeration in notes.
    """
    warnings: list[str] = []
    notes: dict = {}
    clue, enum_in_clue = strip_enumeration(raw_clue)
    residue = non_alpha_residue(raw_answer)
    if residue:
        warnings.append(f"answer {raw_answer!r} has non-letter characters {residue!r}")
    answer = norm_answer(raw_answer)
    if any(ch in raw_answer for ch in " -"):
        display = raw_answer.strip().upper()
        enum_from_answer = enumeration_from_display(display)
    else:
        enum_from_answer = None
        display = answer
    if enum_in_clue and enum_from_answer and enum_in_clue != enum_from_answer:
        warnings.append(f"enumeration in clue ({enum_in_clue}) != from answer ({enum_from_answer})")
    enum = enum_in_clue or enum_from_answer or str(len(answer))
    if enum_total(enum) != len(answer):
        if trust_answer:
            notes["enumeration_in_source"] = enum
            warnings.append(f"clue enumeration {enum} does not fit grid answer {answer}; using {len(answer)}")
            enum = enum_from_answer or str(len(answer))
        else:
            warnings.append(f"enumeration {enum} does not fit answer {answer}")
    if enum_in_clue and not enum_from_answer:
        try:
            display = display_from_grid(answer, enum)
        except ValueError:
            pass
    # Which enumerations the setter actually wrote matters: a corpus that mixes "(6,7)" with a
    # derived "(13)" is presenting two different amounts of information to the model, and the
    # difference is invisible once the string is in the row.
    notes["enumeration_source"] = "clue text" if enum_in_clue else (
        "answer display" if enum_from_answer else "derived from answer length")
    return Row(clue=clue, answer=answer, answer_display=display, enumeration=enum,
               source=source, published_date=date_, published_url=url, notes=notes), warnings


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_id_list(path: Path) -> list[str]:
    """Ids from a `#`-commented list file, in file order.

    File order is load-bearing for `nat50`: it is the seed order of the draw, and the gate's
    `--limit 20` takes the first 20 so the full run can reuse them. Sorting here would quietly
    change which 20 clues the gate measures.
    """
    ids: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def validate(rows: list[dict], *, require_dev: bool = True) -> list[str]:
    """Return a list of violations (empty = valid)."""
    errors: list[str] = []
    seen_clues: dict[str, str] = {}
    today = date.today().isoformat()
    for r in rows:
        rid = r.get("id", "?")
        if enum_total(r["enumeration"]) != len(r["answer"]):
            errors.append(f"{rid}: enumeration {r['enumeration']} != len({r['answer']})")
        if not r["answer"].isalpha() or not r["answer"].isupper():
            errors.append(f"{rid}: answer not A–Z: {r['answer']!r}")
        key = norm_answer(r["clue"])
        if key in seen_clues:
            errors.append(f"{rid}: duplicate clue of {seen_clues[key]}")
        seen_clues[key] = rid
        if r["source"] not in SOURCES:
            errors.append(f"{rid}: unknown source {r['source']!r}")
        # The published branch is the forcing function for the contamination arm: a row that
        # names a publication must say when and where, or explicitly admit it does not know.
        # Silence is what produced a corpus of published puzzles tagged `other` with every date
        # null, where this check existed and was unreachable.
        if r["source"] == "published" and not (r.get("published_date") and r.get("published_url")):
            if not (r.get("notes") or {}).get("published_date_unknown"):
                errors.append(f"{rid}: published row lacks published_date/url "
                              f"(set notes.published_date_unknown to acknowledge deliberately)")
        if r.get("published_date") and r["published_date"] > today:
            errors.append(f"{rid}: published_date in the future")
        if r["split"] not in {"dev", "test"}:
            errors.append(f"{rid}: bad split {r['split']!r}")
    n_dev = sum(r["split"] == "dev" for r in rows)
    if require_dev and n_dev != N_DEV:
        errors.append(f"expected {N_DEV} dev rows, found {n_dev}")
    # The dev clues are shown in every prompt. An answer clued twice in the corpus — 26
    # answers are, and one whole puzzle is a re-clue of another — would hand the model the
    # answer to its twin verbatim. Clean as of 2026-09-08, but nothing prevented it.
    dev_answers = {r["answer"] for r in rows if r["split"] == "dev"}
    for r in rows:
        if r["split"] == "test" and r["answer"] in dev_answers:
            errors.append(f"{r.get('id', '?')}: test answer {r['answer']} is also a dev "
                          f"(few-shot) answer; the prompt would leak it")
    return errors


def summary(rows: list[dict]) -> str:
    pub = sum(1 for r in rows if r.get("published_date"))
    lengths = Counter(len(r["answer"]) for r in rows)
    dup_answers = {a: n for a, n in Counter(r["answer"] for r in rows).items() if n > 1}
    lines = [
        f"rows: {len(rows)}  (dev {sum(r['split']=='dev' for r in rows)}, test {sum(r['split']=='test' for r in rows)})",
        f"published: {pub}  unpublished: {len(rows) - pub}",
        "answer lengths: " + ", ".join(f"{k}:{v}" for k, v in sorted(lengths.items())),
        "enumeration source: " + ", ".join(
            f"{k}:{v}" for k, v in sorted(Counter(
                (r.get("notes") or {}).get("enumeration_source", "unrecorded") for r in rows).items())),
        f"duplicate answers (matched-answer groups): {len(dup_answers)}"
        + (": " + ", ".join(f"{a}×{n}" for a, n in sorted(dup_answers.items())) if dup_answers else ""),
    ]
    return "\n".join(lines)


def finalise(raw_rows: list[dict], *, dev_ids_hint: list[str] | None = None,
             existing: list[dict] | None = None, force: bool = False) -> list[dict]:
    """Dedup, shuffle with the fixed seed, assign ids, mark dev rows.

    Ids are keyed to the clue, not to position. Re-running this after new clues arrive keeps
    every id a clue already has and numbers only the new ones from the end, so a cb-NNNN in a
    run CSV written before the new clues still names the same clue afterwards. The dev split
    is carried over for the same reason: it decides which clues are the few-shot examples.
    """
    if existing and not force:
        raise SystemExit("corpus.jsonl exists; pass --force to add clues (existing ids are kept)")
    prior_id = {norm_answer(r["clue"]): r["id"] for r in (existing or []) if r.get("id")}
    prior_dev = {norm_answer(r["clue"]) for r in (existing or []) if r.get("split") == "dev"}
    seen = set()
    rows = []
    for r in raw_rows:
        key = norm_answer(r["clue"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(r))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    next_n = 1 + max((int(i.split("-")[-1]) for i in prior_id.values()), default=0)
    for r in rows:
        key = norm_answer(r["clue"])
        if key in prior_id:
            r["id"] = prior_id[key]
        else:
            r["id"] = f"cb-{next_n:04d}"
            next_n += 1
        r["split"] = "test"
    hinted = [r for r in rows if (r.get("notes") or {}).get("dev_hint")]
    if dev_ids_hint:
        chosen = [r for r in rows if r["id"] in dev_ids_hint]
    elif hinted:
        # the review sheet says which rows are the few-shot exemplars (split column)
        chosen = hinted
    elif prior_dev:
        chosen = [r for r in rows if norm_answer(r["clue"]) in prior_dev]
    else:
        chosen = rows[:N_DEV]
    for r in chosen[:N_DEV]:
        r["split"] = "dev"
    return rows
