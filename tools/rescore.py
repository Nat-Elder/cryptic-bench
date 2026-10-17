#!/usr/bin/env python
"""Re-run the checker over completions already on disk, and diff the verdicts.

    python tools/rescore.py [--results-dir results] [--dataset nat50] [--write]

**Why this exists.** The gate's rule is "fix the checker here or not at all" (PLAN §7.4), and
on 2026-09-10 it caught two checker bugs at once. Changing a checker is worthless unless you
can see what the change does to real derivations — and re-running the models to find out
introduces a second variable (new samples). Every run CSV stores the model's full
`completion`, so the checker can be re-applied offline, to the same 180 rows.

It reports the verdict transition matrix, the rates the gate turns on, and every row whose
verdict moved, so a fix that quietly flips something it should not is visible immediately.

`--write` updates the CSVs in place (and records `rescored_*` fields in the manifest). Without
it, nothing is touched: the default is a dry report. **The eval logs are never rewritten** —
they are the primary record of what the model said.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crypticbench.checker import check_derivation  # noqa: E402
from crypticbench.config import CORPUS_PATH, DATASET_PATHS, RESULTS_DIR  # noqa: E402
from crypticbench.corpus import read_jsonl  # noqa: E402
from crypticbench.derivation import parse  # noqa: E402

THINK_TASKS = ("think", "think_cot")


def clue_index() -> dict[str, dict]:
    rows = {r["id"]: r for r in read_jsonl(CORPUS_PATH)}
    for name, path in DATASET_PATHS.items():
        if name in ("nat", "nat50") or not path.exists() or path.suffix != ".jsonl":
            continue
        for r in read_jsonl(path):
            rows.setdefault(r["id"], r)
    return rows


def rescore_row(row: dict, clue: dict) -> dict:
    """The checker's verdict for one stored completion, in the results-row field names."""
    d = parse(row.get("completion") or "")
    c = check_derivation(d, clue["clue"], clue["answer"])
    return {"verdict": c.verdict, "l0": c.l0, "l1": c.l1, "l2a": c.l2a, "l3": c.l3,
            "unexecuted": c.unexecuted, "check_reasons": "; ".join(c.reasons),
            "device_claimed": d.device or "", "unverified_ops": c.unverified_ops,
            "sourcing": json.dumps(c.sourcing) if c.sourcing else ""}


def rates(rows: list[dict], key: str = "verdict") -> dict:
    """Verdict shares over the rows the gate actually judges.

    Errored calls are dropped first: an API failure leaves the denominator and is counted as
    `n_errors` (PLAN §8, carried ruling), and including them makes every rate wrong in the same
    direction. The 402'd `astra-think` retry of 2026-09-10 put 20 empty rows on disk and moved
    this table's malformed share from 9.5 % to 18.1 % until it was excluded here.
    """
    rows = [r for r in rows if not str(r.get("error") or "").strip()]
    n = len(rows)
    if not n:
        return {}
    v = Counter(r[key] for r in rows)
    unexec = sum(1 for r in rows if str(r.get("unexecuted")).lower() in ("true", "1"))
    # A ceiling truncation with no block written is reported apart from a parse failure, the
    # same split analysis/metrics.py makes — the gate's "malformed < 5 %" is about blocks that
    # would not parse, and the two tools must not disagree about which rows those are.
    no_block = sum(1 for r in rows if r[key] == "malformed" and (
        str(r.get("stop_reason")) == "max_tokens"
        or str(r.get("truncated")).lower() in ("true", "1")
        or not str(r.get("completion") or "").strip()))
    return {"n": n, "valid": v["valid"] / n, "invalid": v["invalid"] / n,
            "unscoreable": v["unscoreable"] / n,
            "malformed": (v["malformed"] - no_block) / n,
            "truncated_no_block": no_block / n,
            "unexecuted": unexec / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--dataset")
    ap.add_argument("--system")
    ap.add_argument("--write", action="store_true",
                    help="update the CSVs in place. Off by default; the eval logs are never "
                         "touched either way.")
    a = ap.parse_args()
    clues = clue_index()
    runs = sorted((Path(a.results_dir) / "runs").glob("*.csv"))
    before: list[dict] = []
    after: list[dict] = []
    moved: list[tuple] = []
    touched = 0

    for csv_path in runs:
        man = json.loads(csv_path.with_suffix(".json").read_text(encoding="utf-8"))
        if man.get("task") not in THINK_TASKS:
            continue
        if a.dataset and man.get("dataset") != a.dataset:
            continue
        if a.system and man.get("system") != a.system:
            continue
        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        changed = 0
        for row in rows:
            clue = clues.get(row["id"])
            if clue is None:
                print(f"  ! {row['id']} not in any dataset; left alone", file=sys.stderr)
                continue
            new = rescore_row(row, clue)
            before.append(dict(row))
            merged = {**row, **{k: str(v) for k, v in new.items()}}
            after.append(merged)
            if row.get("verdict") != new["verdict"]:
                changed += 1
                moved.append((man["system"], row["id"], row.get("verdict"), new["verdict"],
                              new["check_reasons"][:90]))
            row.update({k: v for k, v in new.items()})
        if a.write and changed:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            man["rescored_utc"] = __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat()
            man["rescored_n_verdicts_changed"] = changed
            csv_path.with_suffix(".json").write_text(json.dumps(man, indent=1),
                                                     encoding="utf-8")
        touched += changed

    if not before:
        sys.exit("no think runs matched")
    print(f"{len(before)} rows over {len({r['run_id'] for r in before})} runs "
          f"({rates(before).get('n', 0)} judged, the rest errored); "
          f"{touched} verdicts changed" + ("" if a.write else "  (dry run, use --write)"))
    print("\n            before   after")
    b, af = rates(before), rates(after)
    for key in ("valid", "invalid", "unscoreable", "malformed", "truncated_no_block",
                "unexecuted"):
        flag = ""
        if key == "malformed" and af[key] > 0.05:
            flag = "  <- gate wants < 5%"
        if key == "unexecuted" and af[key] > 0.15:
            flag = "  <- gate wants < 15% (PLAN §2.3)"
        print(f"{key:>12}  {b[key]:6.1%}  {af[key]:6.1%}{flag}")
    print("\ntransitions:")
    for (was, now), n in Counter((r[0][2], r[0][3]) for r in [(m,) for m in moved]).most_common():
        print(f"  {n:3}  {was} -> {now}")
    if moved:
        print("\nrows that moved:")
        for system, rid, was, now, why in moved:
            print(f"  {system:14} {rid} {was:11} -> {now:11} {why}")


if __name__ == "__main__":
    main()
