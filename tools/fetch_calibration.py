#!/usr/bin/env python
"""Build data/calibration/guardian200.jsonl from Saha et al.'s post-cutoff Guardian set.

Source: github.com/espressovi/LMCrossword, Data/Datasets/Post_cutoff-Clues/guardian.json
(MIT). 200 clues published after 20 May 2024; answers are lowercase without word breaks, so
the enumeration is the total letter count only (as in their own prompts).
"""

from __future__ import annotations

import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crypticbench.config import GUARDIAN200_PATH as CALIBRATION_PATH, RAW_DIR, SEED  # noqa: E402
from crypticbench.corpus import write_jsonl  # noqa: E402
from crypticbench.normalise import norm_answer, strip_enumeration  # noqa: E402

URL = ("https://raw.githubusercontent.com/espressovi/LMCrossword/master/"
       "Data/Datasets/Post_cutoff-Clues/guardian.json")


def main(n: int = 200) -> None:
    """All 200 by default: Saha et al.'s 18.5 % is measured on these 200 items, so running the
    same 200 makes the comparison item-for-item rather than a random half."""
    raw = RAW_DIR / "lmcrossword_guardian.json"
    if not raw.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, raw)
    data = json.loads(raw.read_text(encoding="utf-8"))
    items = sorted(data.items(), key=lambda kv: int(kv[0]))
    rows = []
    for k, v in items:
        clue, enum_in_clue = strip_enumeration(v["clue"])
        answer = norm_answer(v["answer"])
        rows.append({
            "id": f"gd-{int(k):04d}", "clue": clue, "answer": answer, "answer_display": answer,
            "enumeration": enum_in_clue or str(len(answer)), "source": "saha2025",
            "published_date": None, "published_url": None, "setter": "guardian", "split": "test",
            "notes": {"lmcrossword_index": k},
        })
    rng = random.Random(SEED)
    rng.shuffle(rows)
    write_jsonl(CALIBRATION_PATH, rows[:n])
    print(f"wrote {min(n, len(rows))} of {len(rows)} Guardian clues to {CALIBRATION_PATH}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
