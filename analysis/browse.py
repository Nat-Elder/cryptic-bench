#!/usr/bin/env python
"""Every model's answer to every clue, browsable.

    python analysis/browse.py [--results-dir results] [--task nocot] [--dataset nat]

Writes tables/answers_wide.csv (row per clue, column per system) and tables/answers_wide.html
(sortable, filterable, full completion on expand). Every answer stays inspectable: the
adjudication step (PLAN §7.8) grades the distinct answer strings out of this table, and a
`nocot` cell shows both the answer and whatever the model added to it.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.metrics import load_runs  # noqa: E402
from crypticbench.config import CORPUS_PATH, RESULTS_DIR  # noqa: E402
from crypticbench.corpus import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--task", default="nocot", choices=["nocot", "think", "think_cot"])
    ap.add_argument("--dataset", default="nat", choices=["nat", "guardian200", "times200"])
    a = ap.parse_args()
    results = Path(a.results_dir)
    df = load_runs(results)
    df = df[(df["task"] == a.task) & (df["dataset"] == a.dataset)]
    if df.empty:
        raise SystemExit(f"no {a.task} runs on {a.dataset}")
    corpus = {r["id"]: r for r in read_jsonl(CORPUS_PATH)}
    systems = sorted(df["system"].unique())
    ids = sorted(df["id"].unique())

    wide_rows, detail = [], {}
    for cid in ids:
        r = corpus.get(cid, {})
        row = {"id": cid, "clue": r.get("clue", ""), "enumeration": r.get("enumeration", ""),
               "answer": r.get("answer", ""), "device": ""}
        n_right = 0
        for s in systems:
            g = df[(df["id"] == cid) & (df["system"] == s)]
            if len(g):
                x = g.iloc[0]
                row["device"] = x.get("device", "") or row["device"]
                # correct is NaN for an errored sample, which run.py tolerates up to
                # fail_on_error=0.05. int(NaN) raised and took the whole table down.
                scored = pd.notna(x["correct"])
                ok = bool(scored and x["correct"] == 1)
                n_right += ok
                extra = str(x.get("extra_text") or "").strip()
                row[s] = (f"{'✓' if ok else '✗' if scored else '⚠'} {x['extracted'] or '—'}"
                          + (f"  +{extra[:24]}" if extra else ""))
                detail[(cid, s)] = {"completion": x.get("completion", ""), "thinking": x.get("thinking", ""),
                                    "correct": ok, "scored": scored, "extracted": x["extracted"]}
            else:
                row[s] = ""
        row["n_right"] = n_right
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    wide.to_csv(tables / "answers_wide.csv", index=False)

    devices = sorted({r["device"] for r in wide_rows if r["device"]})
    head = "".join(f"<th>{html.escape(s)}</th>" for s in systems)
    body = []
    for r in wide_rows:
        cells = []
        for s in systems:
            d = detail.get((r["id"], s))
            if not d:
                cells.append("<td></td>")
                continue
            cls = "ok" if d["correct"] else ("bad" if d["scored"] else "err")
            comp = html.escape(d["completion"] or "")
            think = html.escape(d["thinking"] or "")
            cells.append(
                f'<td class="{cls}"><details><summary>{html.escape(r[s])}</summary>'
                f'<pre>{comp}</pre>' + (f'<details><summary>thinking</summary><pre>{think}</pre></details>' if think else "") +
                '</details></td>')
        body.append(f'<tr data-device="{html.escape(r["device"])}" data-right="{r["n_right"]}">'
                    f'<td>{r["id"]}</td><td class="clue">{html.escape(r["clue"])} ({r["enumeration"]})</td>'
                    f'<td><b>{r["answer"]}</b></td><td>{html.escape(r["device"])}</td>{"".join(cells)}</tr>')
    n_sys = len(systems)
    page = f"""<!doctype html><meta charset="utf-8"><title>CrypticBench answers</title>
<style>
body{{font:13px/1.35 system-ui,sans-serif;color:#0b0b0b;background:#fcfcfb;margin:16px}}
table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid #e6e5e1;padding:4px 6px;vertical-align:top;text-align:left}}
th{{position:sticky;top:0;background:#fcfcfb;cursor:pointer}} td.ok summary{{color:#008300}} td.bad summary{{color:#e34948}} td.err summary{{color:#52514e}}
td.clue{{min-width:22em}} pre{{white-space:pre-wrap;font-size:11px;max-width:40em;color:#52514e;margin:4px 0}}
.filters{{display:flex;gap:12px;align-items:center;margin-bottom:10px;flex-wrap:wrap}} summary{{cursor:pointer}}
</style>
<h2>CrypticBench — every answer</h2>\n<p style="color:#52514e">A cell reads <b>verdict, extracted answer</b>, then <b>+</b> anything the model wrote beyond it (the 5&nbsp;% rule's evidence). Expand for the full completion.</p>
<div class="filters">
<input id="q" placeholder="search clue / answer" oninput="apply()">
<select id="dev" onchange="apply()"><option value="">all devices</option>{"".join(f'<option>{html.escape(d)}</option>' for d in devices)}</select>
<select id="right" onchange="apply()"><option value="">any outcome</option><option value="0">all models wrong</option><option value="1">exactly one right</option><option value="all">all models right</option></select>
<span id="count"></span>
</div>
<table id="t"><thead><tr><th onclick="sortBy(0)">id</th><th onclick="sortBy(1)">clue</th><th onclick="sortBy(2)">answer</th><th onclick="sortBy(3)">device</th>{head}</tr></thead>
<tbody>{"".join(body)}</tbody></table>
<script>
const NSYS={n_sys};
function apply(){{const q=document.getElementById('q').value.toLowerCase(),d=document.getElementById('dev').value,r=document.getElementById('right').value;let n=0;
for(const tr of document.querySelectorAll('#t tbody tr')){{const txt=tr.children[1].textContent.toLowerCase()+' '+tr.children[2].textContent.toLowerCase();
const nr=+tr.dataset.right;let ok=(!q||txt.includes(q))&&(!d||tr.dataset.device===d);
if(r==='0')ok=ok&&nr===0;else if(r==='1')ok=ok&&nr===1;else if(r==='all')ok=ok&&nr===NSYS;
tr.style.display=ok?'':'none';n+=ok;}}document.getElementById('count').textContent=n+' clues';}}
function sortBy(i){{const tb=document.querySelector('#t tbody');const rows=[...tb.children];const asc=tb.dataset.s!=i;tb.dataset.s=asc?i:'';
rows.sort((a,b)=>a.children[i].textContent.localeCompare(b.children[i].textContent)*(asc?1:-1));rows.forEach(r=>tb.appendChild(r));}}
apply();
</script>"""
    (tables / "answers_wide.html").write_text(page, encoding="utf-8")
    print(f"wrote {tables / 'answers_wide.csv'} and answers_wide.html ({len(ids)} clues × {len(systems)} systems)")


if __name__ == "__main__":
    main()
