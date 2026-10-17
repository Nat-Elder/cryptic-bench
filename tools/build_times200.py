#!/usr/bin/env python
"""Build data/calibration/times200.jsonl from Sadallah et al.'s small explanatory dataset.

Source: github.com/bodasadallah/decrypting-crosswords, data/200_clues.csv (MIT). 200 Times
clues, 40 per type, each with a human wordplay type, the definition and a one-line explanation.
Their five types map onto our eight families as below; the mapping is recorded per row.
"""

from __future__ import annotations

import csv
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crypticbench.config import DATA_DIR, RAW_DIR  # noqa: E402
from crypticbench.corpus import make_row, write_jsonl  # noqa: E402

URL = "https://raw.githubusercontent.com/bodasadallah/decrypting-crosswords/main/data/200_clues.csv"
TYPE_MAP = {"anagram": "anagram", "hidden word": "hidden", "double definition": "semantic",
            "container": "containment", "assemblage": "charade"}


def main() -> None:
    raw = RAW_DIR / "sadallah_200_clues.csv"
    if not raw.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, raw)
    out = []
    n_warn = 0
    with raw.open(encoding="utf-8-sig") as f:
        for i, rec in enumerate(csv.DictReader(f)):
            row, warns = make_row(rec["Clue"], rec["Answer"], "other")
            n_warn += bool(warns)
            d = row.__dict__
            d.update({"id": f"tt-{i:04d}", "source": "sadallah2024", "setter": "times", "split": "test",
                      "device": TYPE_MAP[rec["Type"].strip().lower()], "device_source": rec["Type"].strip().lower(),
                      "definition": rec["Definition"].strip(), "explanation": rec["Explanation"].strip(),
                      "requires_synonym": rec.get("Requires synonym?", "").strip(),
                      "notes": {"blog_url": rec.get("Source", "").strip(), "warnings": warns}})
            out.append(d)
    write_jsonl(DATA_DIR / "calibration" / "times200.jsonl", out)
    print(f"wrote {len(out)} rows ({n_warn} with enumeration warnings, kept) to data/calibration/times200.jsonl")


if __name__ == "__main__":
    main()
