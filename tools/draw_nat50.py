#!/usr/bin/env python
"""Draw `nat50` — 50 of the 100 `nat` test clues, once, with the seed recorded.

    python tools/draw_nat50.py [--seed 20260906] [--balance-on hops|device] [--force]

Nat's ruling, 2026-09-10: the `think` arms run on 50 clues rather than 100, to keep the run's
scope manageable at the measured token shapes. `nocot` still runs on all 100. So this is not a second study: the 50 are a
**subset** of the 100, drawn once, written to `data/nat50.txt`, and never re-drawn at run time.

**Balanced, proportionally.** The draw takes each stratum's share of the 100 and keeps that
share of the 50, with largest-remainder rounding so the total is exactly 50. Proportional, not
equalised: PLAN §2.5's corpus is charade-heavy on purpose and the 50 must look like the 100, not
like a balanced dataset nobody has.

**The balance column.** `--balance-on hops` is what PLAN §5.4 wants, and `hops` is what the
graded analysis regresses on. It does not exist yet: no row of `data/devices.jsonl` carries one,
and this repo refuses to infer a hop count by regex from the parse strings (`analysis.by_hops`
declines for the same reason — a regex's count would be this repo's number, not the setter's).
So the default falls back to `device`, loudly, and records which column was used in the file's
own header.

**Ruled 2026-09-10 (Nat): the device-balanced draw is final and `hops` is filled afterwards.**
Filling it does **not** mean re-drawing. The draw's balance column and the analysis's regressor
are two different things: `hops` is what PLAN §5.4 regresses accuracy on, and that regression is
on the `nocot` arm's own 100 clues, which this split does not touch. Re-drawing after any
`nat50` run would change what those runs measured, so it is not a tidy-up — do not do it.

The file is a list of ids in **seed order**, because the gate takes the first 20 of it: a
`--limit 20` gate row is then reused by the full run through `--skip-done`.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crypticbench.config import (  # noqa: E402
    CORPUS_PATH,
    DEVICES_PATH,
    NAT50_PATH,
    NAT50_SIZE,
    SEED,
)
from crypticbench.corpus import read_jsonl  # noqa: E402

BALANCE_COLUMNS = ("hops", "device")


def proportional_targets(counts: dict[object, int], total: int, n: int) -> dict[object, int]:
    """Each stratum's share of `n`, by largest remainder, summing to exactly `n`.

    Largest remainder rather than rounding each independently: independent rounding does not
    sum to n, and truncating the last stratum would make the smallest family bear the whole
    rounding error.
    """
    exact = {k: v * n / total for k, v in counts.items()}
    floors = {k: int(v) for k, v in exact.items()}
    short = n - sum(floors.values())
    # ties broken by stratum key, so the draw is reproducible from the seed alone
    order = sorted(exact, key=lambda k: (-(exact[k] - floors[k]), str(k)))
    for k in order[:short]:
        floors[k] += 1
    return floors


def choose_column(tags: dict[str, dict], requested: str | None) -> str:
    """`hops` if it is on file, else `device`, with a warning. Never a computed hop count."""
    have_hops = sum(1 for t in tags.values() if t.get("hops") is not None)
    if requested == "hops" and not have_hops:
        sys.exit("REFUSED: --balance-on hops, but no row of data/devices.jsonl carries a `hops` "
                 "count. Fill it from the confirmed parses (PLAN §5.4); it will not be inferred "
                 "from the parse strings here.")
    if requested:
        return requested
    if have_hops:
        return "hops"
    print("WARNING: no clue carries a `hops` count, so the draw is balanced on `device` "
          "instead — the tags are Claude's suggestions, unconfirmed (PLAN §7.1). This is a "
          "declared substitution, recorded in the file header. Filling `hops` later means "
          "re-drawing, which invalidates any nat50 run made before it.")
    return "device"


def draw(rows: list[dict], tags: dict[str, dict], *, column: str, seed: int,
         n: int) -> tuple[list[str], dict[object, int], dict[object, int]]:
    """(ids in seed order, per-stratum counts in the 100, per-stratum counts in the 50)."""
    strata: dict[object, list[str]] = {}
    for r in rows:
        value = (tags.get(r["id"], {}) or {}).get(column)
        strata.setdefault(value, []).append(r["id"])
    counts = {k: len(v) for k, v in strata.items()}
    targets = proportional_targets(counts, len(rows), n)
    rng = random.Random(seed)
    chosen: list[str] = []
    for key in sorted(strata, key=str):           # stratum order fixed before any draw
        chosen.extend(rng.sample(sorted(strata[key]), targets[key]))
    rng.shuffle(chosen)                            # seed order: the gate takes the first 20
    assert len(chosen) == n and len(set(chosen)) == n
    return chosen, counts, targets


def render(ids: list[str], *, column: str, seed: int, counts: dict, targets: dict,
           corpus_n: int) -> str:
    head = [
        "# nat50 — 50 of the 100 nat test clues, for the `think` arms (Nat, 2026-09-10).",
        "# `nocot` runs on all 100; think, the Sonnet 5 rungs and think_cot run on these 50.",
        f"# drawn {datetime.now(timezone.utc).date().isoformat()} by tools/draw_nat50.py",
        f"# seed {seed}, proportional on `{column}` by largest remainder, from {corpus_n} clues",
        "# ids are in SEED ORDER: the gate's --limit 20 takes the first 20, which the full run",
        "# then reuses through --skip-done. Do not sort this file.",
        "#",
        f"# per stratum (`{column}`), of the 100 -> of the 50:",
    ]
    for key in sorted(counts, key=str):
        head.append(f"#   {str(key):12} {counts[key]:3} -> {targets[key]:3}")
    if column != "hops":
        head += [
            "#",
            "# NOTE: PLAN §5.4 asks for a draw balanced on `hops`. No clue carried a hop count",
            "# (data/devices.jsonl), and a count inferred from the parse strings would be this",
            "# repo's number, not the setter's. Balanced on `device` instead, declared here.",
            "#",
            "# RULED 2026-09-10 (Nat): this draw is FINAL and `hops` is filled afterwards, for",
            "# the PLAN §5.4 regression — which runs on the nocot arm's own 100 clues, not on",
            "# this split. Filling `hops` is NOT a reason to re-draw: a re-draw changes what",
            "# every nat50 run already measured. Do not regenerate this file.",
        ]
    return "\n".join(head) + "\n" + "\n".join(ids) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--balance-on", choices=BALANCE_COLUMNS, default=None)
    ap.add_argument("--n", type=int, default=NAT50_SIZE)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing draw. A re-draw is a different split: every "
                         "nat50 run made under the old one measures other clues. The draw of "
                         "2026-09-10 is final (Nat) — filling `hops` is not a reason to use "
                         "this flag.")
    a = ap.parse_args()

    if NAT50_PATH.exists() and not a.force:
        sys.exit(f"REFUSED: {NAT50_PATH.name} already exists and a re-draw would silently "
                 f"change what every nat50 run measured. --force if that is intended, and say "
                 f"so in the hand-back.")
    rows = [r for r in read_jsonl(CORPUS_PATH) if r["split"] == "test"]
    tags = {t["id"]: t for t in read_jsonl(DEVICES_PATH)}
    missing = [r["id"] for r in rows if r["id"] not in tags]
    if missing:
        sys.exit(f"REFUSED: {len(missing)} test clues have no tag row: {missing[:5]}")
    column = choose_column(tags, a.balance_on)
    ids, counts, targets = draw(rows, tags, column=column, seed=a.seed, n=a.n)
    NAT50_PATH.write_text(render(ids, column=column, seed=a.seed, counts=counts,
                                 targets=targets, corpus_n=len(rows)), encoding="utf-8")
    print(f"wrote {NAT50_PATH.relative_to(Path(__file__).resolve().parents[1])}: "
          f"{len(ids)} ids, seed {a.seed}, balanced on `{column}`")
    print("per stratum: " + ", ".join(f"{k} {targets[k]}/{counts[k]}"
                                      for k in sorted(counts, key=str)))
    print("first 20 (the gate's): " + " ".join(ids[:20]))


if __name__ == "__main__":
    main()
