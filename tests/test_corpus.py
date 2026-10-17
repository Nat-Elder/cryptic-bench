"""The corpus file: id stability across a re-finalise, and what the exported corpus contains.

The id-stability tests are carried from the 2026-09-07 review — a cb-NNNN in a run CSV must
still name the same clue after new clues arrive, or every result written before the import
silently changes meaning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crypticbench.config import CORPUS_PATH, DEVICES_PATH, N_DEV  # noqa: E402
from crypticbench.corpus import finalise, read_jsonl, validate  # noqa: E402


def raw(*clues):
    return [{"clue": c, "answer": "X" * 5, "enumeration": "5", "source": "other"} for c in clues]


class TestFinaliseIdStability:
    def test_existing_ids_are_kept_and_new_ones_appended(self):
        first = finalise(raw("alpha one", "beta two", "gamma three"))
        before = {r["clue"]: r["id"] for r in first}
        second = finalise(raw("alpha one", "beta two", "gamma three", "delta four"),
                          existing=first, force=True)
        after = {r["clue"]: r["id"] for r in second}
        for clue, id_ in before.items():
            assert after[clue] == id_, f"{clue} was renumbered {id_} -> {after[clue]}"
        assert after["delta four"] not in before.values()

    def test_dev_split_is_carried_over(self):
        first = finalise(raw(*(f"clue number {i}" for i in range(8))))
        dev_before = {r["clue"] for r in first if r["split"] == "dev"}
        second = finalise(raw(*(f"clue number {i}" for i in range(12))),
                          existing=first, force=True)
        assert {r["clue"] for r in second if r["split"] == "dev"} == dev_before

    def test_refuses_without_force(self):
        first = finalise(raw("alpha one", "beta two"))
        with pytest.raises(SystemExit):
            finalise(raw("alpha one"), existing=first)


@pytest.mark.skipif(not CORPUS_PATH.exists(), reason="the setter's corpus is not on this machine")
class TestTheShippedCorpus:
    """PLAN §2.5: exactly 100 scored clues plus the never-scored dev split, validated."""

    rows = read_jsonl(CORPUS_PATH) if CORPUS_PATH.exists() else []

    def test_it_validates(self):
        assert validate(self.rows) == []

    def test_it_is_a_hundred_test_clues_and_the_dev_split(self):
        assert sum(r["split"] == "test" for r in self.rows) == 100
        assert sum(r["split"] == "dev" for r in self.rows) == N_DEV

    def test_every_clue_has_a_device_tag(self):
        tags = {t["id"] for t in read_jsonl(DEVICES_PATH)}
        assert {r["id"] for r in self.rows} <= tags

    def test_no_tag_is_confirmed_yet(self):
        """Every tag is Claude's suggestion until Nat ticks it (PLAN §7.1). When that happens
        this test is where the change is recorded."""
        assert not any(t.get("confirmed") for t in read_jsonl(DEVICES_PATH))

    def test_no_clue_carries_a_publication_date_yet(self):
        """So the cutoff split declines. Dates per puzzle are a RULING (PLAN §9)."""
        assert not any(r.get("published_date") for r in self.rows)
