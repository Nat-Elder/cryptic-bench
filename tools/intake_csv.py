#!/usr/bin/env python
"""Read the main clue export, and finalise the corpus.

    python tools/intake_csv.py FILE.csv --clue-col Clue --answer-col Answer
        [--date-col Date --url-col URL --source published] --out data/raw/main.jsonl
    python tools/intake_csv.py --finalise data/raw/*.jsonl [--dev cb-0001,...] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crypticbench.config import CORPUS_PATH  # noqa: E402
from crypticbench.corpus import (  # noqa: E402
    finalise,
    make_row,
    read_jsonl,
    summary,
    validate,
    write_jsonl,
)


def read_table(path: Path):
    import pandas as pd

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    sep = "\t" if path.suffix.lower() == ".tsv" else None
    return pd.read_csv(path, sep=sep, engine="python")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--clue-col")
    ap.add_argument("--answer-col")
    ap.add_argument("--date-col")
    ap.add_argument("--url-col")
    ap.add_argument("--source-col")
    ap.add_argument("--enum-col", help="enumeration column, e.g. '(4,2)'; appended to the clue so word breaks survive")
    ap.add_argument("--id-col", help="the review sheet's row id, kept in notes.sheet_id")
    ap.add_argument("--note-cols", help="comma-separated columns copied into notes as-is (e.g. puzzle,device,parse)")
    ap.add_argument("--split-col", help="column holding 'dev' for few-shot rows; everything else is test")
    ap.add_argument("--source", default="unpublished", choices=["published", "unpublished", "other"])
    ap.add_argument("--out")
    ap.add_argument("--finalise", action="store_true")
    ap.add_argument("--dev", help="comma-separated ids to use as few-shot examples")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.finalise:
        raw = [r for f in a.files for r in read_jsonl(Path(f))]
        existing = read_jsonl(CORPUS_PATH)
        rows = finalise(raw, dev_ids_hint=a.dev.split(",") if a.dev else None,
                        existing=existing, force=a.force)
        errors = validate(rows)
        print(summary(rows))
        if errors:
            # Exit before writing (implementation-plan §5.2). Writing first left an invalid
            # corpus on disk that run.py refuses to load and --finalise refuses to overwrite.
            print("\n".join("VIOLATION " + e for e in errors))
            sys.exit(1)
        write_jsonl(CORPUS_PATH, rows)
        print(f"wrote {CORPUS_PATH}")
        return

    if not (a.files and a.clue_col and a.answer_col and a.out):
        ap.error("need FILE --clue-col --answer-col --out (or --finalise)")
    df = read_table(Path(a.files[0]))
    df = df.dropna(subset=[a.clue_col, a.answer_col])
    out: list[dict] = []
    n_warn = 0
    for i, rec in df.iterrows():
        source = str(rec[a.source_col]).strip().lower() if a.source_col else a.source
        date_ = str(rec[a.date_col])[:10] if a.date_col and str(rec[a.date_col]) != "nan" else None
        url = str(rec[a.url_col]) if a.url_col and str(rec[a.url_col]) != "nan" else None
        clue_text = str(rec[a.clue_col]).strip()
        if a.enum_col and str(rec[a.enum_col]) != "nan":
            en = str(rec[a.enum_col]).strip()
            clue_text = f"{clue_text} {en if en.startswith('(') else '(' + en + ')'}"
        row, warns = make_row(clue_text, str(rec[a.answer_col]), source, date_=date_, url=url)
        # Merge, never replace: make_row() puts provenance in notes — `enumeration_source`,
        # which is how the corpus's word-break-versus-total mixture gets reported, and
        # `enumeration_in_source` when a clue's own enumeration contradicted its answer.
        # Assigning a fresh dict here silently discarded both.
        row.notes = {**row.notes, "source_row": int(i)}
        if a.enum_col:
            row.notes["enumeration_source"] = "review sheet (enumeration_fixed)"
        if a.id_col and str(rec[a.id_col]) != "nan":
            row.notes["sheet_id"] = str(rec[a.id_col])
        for col in (a.note_cols.split(",") if a.note_cols else []):
            if col in rec and str(rec[col]) != "nan":
                row.notes[col] = str(rec[col])
        if a.split_col and str(rec[a.split_col]).strip() == "dev":
            row.notes["dev_hint"] = True
        for w in warns:
            n_warn += 1
            print(f"  WARN row {i}: {w}")
        out.append(row.__dict__)
    write_jsonl(Path(a.out), out)
    print(f"wrote {len(out)} rows to {a.out}; {n_warn} warnings need a ruling")


if __name__ == "__main__":
    main()
