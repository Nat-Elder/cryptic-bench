"""Shared between the `nocot` and `think` arms: sample construction and completion cleaning."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402

# Inspect's `output.completion` is `choices[0].message.text`, which concatenates only
# `type == "text"` content blocks, so a thinking model's ContentReasoning never reaches the
# scorer. This regex additionally drops any `<think>...</think>` span, which is how some
# providers inline chain of thought into the text itself: a model that tries candidate answers
# in the trace would otherwise have them read as answers.
_THINK_RE = re.compile(r"<think\b[^>]*>.*?(?:</think>|\Z)", re.IGNORECASE | re.DOTALL)


def visible_text(text: str) -> str:
    """The completion with inlined reasoning spans removed."""
    return _THINK_RE.sub(" ", text or "")


def sample_metadata(row: dict) -> dict:
    return {
        # the clue itself, because the checker grounds every span against it
        "clue": row["clue"],
        "enumeration": row["enumeration"],
        "published": bool(row.get("published_date")),
        # The date, not the flag: the cutoff split is "published before this system's training
        # cutoff", and a clue published in 2026 is post-cutoff for most systems (PLAN §2.6).
        "published_date": row.get("published_date"),
        "source": row.get("source"),
        # the setter's device tag, joined in by run.py from data/devices.jsonl
        "device": row.get("device"),
        "requires_synonym": row.get("requires_synonym"),
    }


def build_dataset(rows: list[dict], user_template: str, examples: str, name: str) -> MemoryDataset:
    samples = [
        Sample(
            id=r["id"],
            input=user_template.format(examples=examples, clue=r["clue"],
                                       enumeration=r["enumeration"]),
            target=r["answer"],
            metadata=sample_metadata(r),
        )
        for r in rows
    ]
    return MemoryDataset(samples=samples, name=name)
