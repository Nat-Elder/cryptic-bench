#!/usr/bin/env python
"""Validate data/corpus.jsonl and print its summary. Exit 1 on any violation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crypticbench.config import CORPUS_PATH  # noqa: E402
from crypticbench.corpus import read_jsonl, summary, validate  # noqa: E402


def main(path: Path = CORPUS_PATH) -> int:
    rows = read_jsonl(path)
    if not rows:
        print(f"{path} is empty or missing")
        return 1
    errors = validate(rows)
    print(summary(rows))
    for e in errors:
        print("VIOLATION", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else CORPUS_PATH))
