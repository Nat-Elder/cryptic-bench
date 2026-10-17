#!/usr/bin/env python
"""Export clues from the Ingrid app database to a spreadsheet for the setter to check.

    python tools/export_ingrid_review.py --puzzles "Puzzle 1,Puzzle 2"
    python tools/export_ingrid_review.py --list               # candidates and their scores

Ingrid (a crossword construction app) stores one row per puzzle in `projects`, with the puzzle
itself in `jsonContent`. Inside it, `versions` is a dict of snapshots; each has a `grid` and a
`clues` dict keyed `x,y,direction,length,answer` with the clue text as the value. Cells are
keyed `x,y` with x increasing across. See AIS/crypticbench/ingrid-db-survey-2026-09-07.md.

Two traps this handles:

1. **Stale clue keys.** The key contains the answer, so re-filling a slot leaves the old key
   and its clue text behind. A clue counts only if a slot still starts at (x, y) in the given
   direction *and* the grid spells the key's answer. Some versions carry more stale keys than
   live ones, so "most clues" is not "most complete" — slots actually clued is the score.
2. **Duplicate saves.** The same puzzle exists under several projects. Where two selections
   have an identical clue set they are collapsed; where they share a grid and answers but no
   clue text (a re-clue) both are kept and paired in `reclue_group`, because whether those are
   two items is the setter's call.

Nothing is written back to the database, which is opened read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path


REVIEW_COLUMNS = ["keep", "clue_final", "enumeration_fixed", "published_date", "published_url",
                  "device", "standalone_fair", "notes"]
COLUMNS = (["clue_id", "check", "puzzle", "project_created", "number", "direction", "answer", "length",
            "enumeration", "enumeration_source", "clue"] + REVIEW_COLUMNS
           + ["reclue_group", "unclued_slot", "x", "y", "project_id"])

_ENUM_RE = re.compile(r"\(([\d,\-\s]+)\)\s*$")


def split_enumeration(text: str, answer_len: int) -> tuple[str, str, str, str]:
    """(surface, enumeration, source, check) — Ingrid keeps the enumeration inside the clue.

    The clue text ends "... (6,7)", which is where the *word breaks* live: the answer is stored
    letters-only, so this is the only place a multi-word enumeration can be recovered from.
    The surface is returned without it, matching the corpus convention (docs/data.md schema) and the
    .puz intake, whose 61 clues match these character for character once it is stripped.
    """
    m = _ENUM_RE.search(text.strip())
    if not m:
        return text.strip(), f"({answer_len})", "derived from answer length", "no enumeration in the clue"
    surface = text.strip()[: m.start()].strip()
    parts = [int(x) for x in re.split(r"[,\-]", m.group(1)) if x.strip()]
    enum = f"({m.group(1).strip()})"
    if sum(parts) != answer_len:
        return surface, enum, "clue text", f"enumeration says {sum(parts)} letters, answer has {answer_len}"
    return surface, enum, "clue text", ""


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def slots(grid: dict) -> list[dict]:
    """Every across/down slot of length >= 2 in the current fill, numbered conventionally."""
    cells, w, h = grid["cells"], grid["width"], grid["height"]

    def space(x: int, y: int) -> bool:
        return 0 <= x < w and 0 <= y < h and (cells.get(f"{x},{y}") or {}).get("type") == "space"

    out, num = [], 0
    for y in range(h):
        for x in range(w):
            if not space(x, y):
                continue
            starts = {"across": not space(x - 1, y) and space(x + 1, y),
                      "down": not space(x, y - 1) and space(x, y + 1)}
            if any(starts.values()):
                num += 1
            for direction, is_start in starts.items():
                if not is_start:
                    continue
                word, n = "", 0
                while True:
                    xx, yy = (x + n, y) if direction == "across" else (x, y + n)
                    if not space(xx, yy):
                        break
                    word += (cells.get(f"{xx},{yy}") or {}).get("content") or "."
                    n += 1
                out.append({"num": num, "x": x, "y": y, "dir": direction, "len": n, "answer": word})
    return out


def score_version(version: dict) -> dict:
    """Live clues, unclued slots and stale keys for one version snapshot."""
    sl = slots(version["grid"])
    by_start = {(s["x"], s["y"], s["dir"]): s for s in sl}
    live, stale = [], 0
    for key, text in (version.get("clues") or {}).items():
        x, y, direction, _length, answer = key.split(",")
        slot = by_start.get((int(x), int(y), direction))
        if slot and slot["answer"] == answer and isinstance(text, str) and text.strip():
            live.append({**slot, "clue": " ".join(text.split())})
        else:
            stale += 1
    clued = {(c["x"], c["y"], c["dir"]) for c in live}
    return {"slots": sl, "live": sorted(live, key=lambda c: (c["num"], c["dir"])), "stale": stale,
            "unclued": [s for s in sl if (s["x"], s["y"], s["dir"]) not in clued],
            "empty_cells": sum(1 for c in version["grid"]["cells"].values()
                               if c["type"] == "space" and not c.get("content"))}


def clue_set_hash(live: list[dict]) -> str:
    payload = sorted((c["num"], c["dir"], c["answer"], c["clue"].lower()) for c in live)
    return hashlib.sha1(json.dumps(payload).encode()).hexdigest()[:12]


def candidates(db: Path, wanted: list[str]) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    want = {norm(w) for w in wanted}
    rows = []
    for r in con.execute("select id,name,createdAt,jsonContent from projects "
                         "where jsonContent is not null order by createdAt"):
        if norm(r["name"]) not in want:
            continue
        doc = json.loads(r["jsonContent"])
        for i, (vid, version) in enumerate(sorted(doc["versions"].items(),
                                                  key=lambda kv: kv[1].get("updatedAt", ""))):
            s = score_version(version)
            rows.append({"project_id": r["id"], "project": r["name"], "created": r["createdAt"][:10],
                         "version": i + 1, "version_id": vid, "hash": clue_set_hash(s["live"]), **s})
    return rows


def select(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """One version per puzzle name: most slots clued, then fewest stale keys, then latest."""
    best: dict[str, dict] = {}
    for r in rows:
        key = norm(r["project"])
        rank = (len(r["live"]), -r["stale"], r["created"], r["version"])
        if key not in best or rank > best[key]["_rank"]:
            best[key] = {**r, "_rank": rank}
    chosen = list(best.values())
    # collapse selections that are byte-identical clue sets saved under two projects
    seen: dict[str, dict] = {}
    kept, dropped = [], []
    for r in sorted(chosen, key=lambda r: (r["created"], r["project"])):
        if r["hash"] in seen:
            r["_dropped_for"] = seen[r["hash"]]["project"]
            dropped.append(r)
        else:
            seen[r["hash"]] = r
            kept.append(r)
    return kept, dropped


def rows_for(kept: list[dict]) -> list[dict]:
    # a re-clue group is one answer clued in two selected puzzles with different wording
    by_answer: dict[str, set[str]] = {}
    for p in kept:
        for c in p["live"]:
            by_answer.setdefault(c["answer"], set()).add(norm(p["project"]))
    out = []
    for p in kept:
        slug = norm(p["project"]).replace(" ", "-")
        for c in p["live"] + [{**s, "clue": ""} for s in p["unclued"]]:
            if c["clue"]:
                surface, enum, source, check = split_enumeration(c["clue"], c["len"])
            else:
                surface, enum, source, check = "", f"({c['len']})", "", "unclued slot"
            out.append({
                "clue_id": f"{slug}-{c['num']}{c['dir'][0]}", "check": check, "puzzle": p["project"],
                "project_created": p["created"], "number": c["num"], "direction": c["dir"],
                "answer": c["answer"].upper(), "length": c["len"], "enumeration": enum,
                "enumeration_source": source,
                "clue": surface, "keep": "", "clue_final": "", "enumeration_fixed": enum,
                "published_date": "", "published_url": "", "device": "", "standalone_fair": "",
                "notes": "", "reclue_group": c["answer"].upper() if len(by_answer.get(c["answer"], ())) > 1 else "",
                "unclued_slot": "yes" if not c["clue"] else "", "x": c["x"], "y": c["y"],
                "project_id": p["project_id"],
            })
    return sorted(out, key=lambda r: (r["project_created"], r["puzzle"], r["number"], r["direction"]))


def write_xlsx(path: Path, rows: list[dict], audit: list[dict], kept: list[dict], dropped: list[dict]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "clues"
    ws.append(COLUMNS)
    head = Font(bold=True)
    review = PatternFill("solid", fgColor="FFF6E5")
    for i, col in enumerate(COLUMNS, start=1):
        ws.cell(row=1, column=i).font = head
        if col in REVIEW_COLUMNS:
            ws.cell(row=1, column=i).fill = review
    for r in rows:
        ws.append([r[c] for c in COLUMNS])
    widths = {"clue": 62, "clue_final": 62, "puzzle": 17, "answer": 16, "clue_id": 15,
              "project_created": 14, "notes": 24, "published_url": 22, "project_id": 38,
              "check": 34, "enumeration_source": 20}
    for i, col in enumerate(COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(col, 12)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=ws.cell(row=1, column=cell.column).value in ("clue", "clue_final", "notes"))
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = ws.dimensions

    aw = wb.create_sheet("puzzles")
    cols = ["project", "created", "version", "slots", "clued", "unclued", "stale_keys", "empty_cells", "clue_set_hash", "selected", "why"]
    aw.append(cols)
    for i in range(1, len(cols) + 1):
        aw.cell(row=1, column=i).font = head
    sel = {(r["project_id"], r["version_id"]) for r in kept}
    drop = {(r["project_id"], r["version_id"]): r.get("_dropped_for") for r in dropped}
    for r in audit:
        k = (r["project_id"], r["version_id"])
        why = ("selected: most slots clued" if k in sel else
               f"dropped: identical clue set to {drop[k]}" if k in drop else
               "not selected: fewer slots clued")
        aw.append([r["project"], r["created"], r["version"], len(r["slots"]), len(r["live"]),
                   len(r["unclued"]), r["stale"], r["empty_cells"], r["hash"], "yes" if k in sel else "", why])
    for i, w in enumerate([18, 12, 8, 7, 7, 9, 11, 12, 15, 9, 44], start=1):
        aw.column_dimensions[get_column_letter(i)].width = w

    rw = wb.create_sheet("README")
    for line in README.strip().splitlines():
        rw.append([line])
    rw.column_dimensions["A"].width = 100
    wb.save(path)


README = """
CrypticBench - clue review sheet
Generated by tools/export_ingrid_review.py from the Ingrid app database. Read-only source.

WHAT TO DO
  Work down the 'clues' sheet. The white columns are as stored in Ingrid; the shaded columns
  are yours to fill. Nothing is read back automatically - intake reads this file once you say
  it is done, so edit freely and leave anything you are unsure about blank.

COLUMNS AS STORED (do not need editing)
  clue_id           stable id, e.g. puzzle-3-12a. Used as the corpus id, so keep it.
  puzzle            the Ingrid project the clue came from.
  project_created   when that project was created - a bound on the publication date, not it.
  number, direction conventional numbering computed from the current grid.
  answer, length    from the clue key, verified letter by letter against the grid fill.
  enumeration       parsed off the end of the clue text where Ingrid has it, e.g. (6,7), which
                    is the only place word breaks survive - answers are stored letters-only.
                    Falls back to the total, e.g. (13), where the clue carries no enumeration.
  enumeration_source 'clue text' or 'derived from answer length'.
  check             non-blank means this row needs you: no enumeration in the clue, an
                    enumeration that disagrees with the answer, or an unclued slot. Filter on
                    this column and the rest of the sheet can be skimmed.
  clue              the clue text with the trailing enumeration removed, whitespace collapsed,
                    otherwise verbatim. Matches the .puz exports character for character.

COLUMNS FOR YOU
  keep              blank or 'y' to include, 'n' to exclude. Blank is treated as 'y'.
  clue_final        only if the stored text needs fixing; otherwise leave blank.
  enumeration_fixed pre-filled from the column above. Only needs your attention on rows where
                    'check' is non-blank - elsewhere the clue told us the word breaks.
  published_date    ISO date, e.g. 2026-05-11. Needed for the contamination test: clues with
                    no date all count as unpublished, which inverts that test.
  published_url     if the puzzle is online.
  device            optional; the AI tagger fills this if you leave it blank.
  standalone_fair   'n' if the clue is unfair without crossing letters. Audit flag only.
  notes             anything else.

FLAGS
  reclue_group      this answer is clued in more than one selected puzzle with different
                    wording. Decide whether both are corpus items, or drop one.
  unclued_slot      'yes' means the grid has a slot here with no clue stored. The row exists so
                    the gap is visible; write the clue in clue_final or delete the row.

THE 'puzzles' SHEET
  Every candidate project and version, with why it was or was not selected. A clue counts only
  if its slot still exists in the current grid and the fill still spells its answer, so a
  version can hold more clue records than live clues - those are leftovers from earlier fills.
"""


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(here / "data.sqlite3"))
    ap.add_argument("--puzzles", default="", help="comma-separated Ingrid project names; see --list")
    ap.add_argument("--out", default=str(here / "data" / "raw" / "clues_for_review.xlsx"))
    ap.add_argument("--list", action="store_true", help="print candidates and scores, write nothing")
    a = ap.parse_args()

    wanted = [p for p in a.puzzles.split(",") if p.strip()]
    if not wanted:
        sys.exit('name the puzzles to export: --puzzles "Puzzle 1,Puzzle 2"')
    audit = candidates(Path(a.db), wanted)
    if not audit:
        sys.exit(f"no projects matched {wanted}")
    kept, dropped = select(audit)

    print(f"{'project':18s} {'created':10s} {'v':2s} {'slots':>5s} {'clued':>5s} {'unclued':>7s} {'stale':>5s} {'':3s}")
    sel = {(r['project_id'], r['version_id']) for r in kept}
    for r in audit:
        mark = "<-" if (r["project_id"], r["version_id"]) in sel else ""
        print(f"{r['project'][:17]:18s} {r['created']:10s} v{r['version']:<1d} {len(r['slots']):5d} "
              f"{len(r['live']):5d} {len(r['unclued']):7d} {r['stale']:5d} {mark}")
    for r in dropped:
        print(f"  collapsed: {r['project']} is an identical clue set to {r['_dropped_for']}")
    if a.list:
        return

    rows = rows_for(kept)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(out, rows, audit, kept, dropped)
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    clued = sum(1 for r in rows if not r["unclued_slot"])
    reclued = len({r["reclue_group"] for r in rows if r["reclue_group"]})
    from_text = sum(1 for r in rows if r["enumeration_source"] == "clue text")
    multiword = sum(1 for r in rows if "," in r["enumeration"] or "-" in r["enumeration"])
    flagged = sum(1 for r in rows if r["check"])
    print(f"\n{clued} clues from {len(kept)} puzzles, {len(rows) - clued} unclued slots flagged, "
          f"{reclued} answers clued twice")
    print(f"enumerations: {from_text} parsed from the clue text ({multiword} multi-word), "
          f"{clued - from_text} derived from the answer length")
    print(f"rows needing the setter (filter on 'check'): {flagged}")
    print(f"wrote {out}\n      {csv_path}")


if __name__ == "__main__":
    main()
