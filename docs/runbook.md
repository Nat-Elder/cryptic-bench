# Runbook

The full sequence for running a study with this harness, in order. Each step ends with a
check that decides whether to go on.

All commands run from the repo root with `uv run`, and need a `.env` containing
`OPENROUTER_API_KEY` (see `.env.example`).

## 0. Setup checks

```
uv run pytest
uv run python tools/validate_corpus.py
uv run python run.py --task nocot --systems mock --dataset nat --limit 5 --sync
grep -n "NOT CHECKED" configs/models.yaml
```

**Check:** tests pass; the validator exits 0 and reports your test and dev row counts; the mock
run writes a CSV with a `reasoning_tokens` column. Each line the `grep` prints is a model the
runner will refuse (no model id) or a field the analysis will decline (for example a
blank `training_cutoff`). Fill these from vendor pages, not from memory.

## 1. Build the corpus

From a spreadsheet of clues:

```
uv run python tools/intake_csv.py clues.xlsx \
    --clue-col clue --answer-col answer --enum-col enumeration \
    --id-col id --split-col split --source other --out data/raw/kept.jsonl
uv run python tools/intake_csv.py --finalise data/raw/kept.jsonl --force
uv run python tools/validate_corpus.py
```

If your clues are in the Ingrid app, `tools/export_ingrid_review.py --puzzles "Puzzle 1,Puzzle 2"`
exports them to a review spreadsheet first (`--list` shows what's in the database).

**Check:** zero validation errors, and ids are stable: a `cb-NNNN` in an old run must still
name the same clue (`tests/test_corpus.py` checks this).

To use the public calibration sets, build them with `tools/fetch_calibration.py` (Guardian 200)
and `tools/build_times200.py` (Times 200).

## 2. Probe: five clues per model

```
uv run python run.py --task nocot --systems <nocot models> --dataset nat --limit 5 --sync
uv run python run.py --task think --systems <think models> --dataset nat --limit 5 --sync
```

This is quick and confirms the plumbing before anything else. Re-run it for any model whose
config changes. The runner prints a verdict for each run and stores it in the manifest under
`reasoning_gate` and `extra_text_rule`.

**Check:**

- **The `nocot` reasoning gate passes for every model.** A run where no call reported
  reasoning tokens fails, because the arm depends on that count. If it fails, inspect the raw
  response in the eval log before changing config. Some models (GPT-6 Astra among them) can't
  have reasoning switched off, only asked off by prompt v2; for those, the per-call count is
  the only evidence. Calls that did reason (`n_reasoned`) are excluded from accuracy.
- **No `think` run is flagged `reasoning_absent`.** That flag means the model was asked to
  reason and reported zero reasoning tokens, i.e. thinking was effectively off. Fix the
  model's config (e.g. `reasoning_enabled: true`) and re-probe.
- **Extra text is at most 5% of `nocot` outputs.** If a model writes more than the answer,
  re-run it with `--structured` and record that it needed it.
- **`n_truncated` is 0 on `nocot`.** The limit is 16 tokens, so truncation means the model
  wrote something before its answer.
- **Every `think` response parses** (`l0` true on every row).

## 3. Gate: twenty clues per model

Run every model on the first 20 clues of `data/nat50.txt`. Use the same 20 in both arms:

```
uv run python run.py --task nocot --systems <nocot models> \
    --dataset nat --ids-from data/nat50.txt --limit 20 --sync
uv run python run.py --task think --systems <think models> \
    --dataset nat50 --limit 20 --sync --max-connections 4
uv run python analysis/metrics.py
```

Use `--dataset nat --ids-from` for `nocot`, not `--dataset nat50`: that keeps these rows
inside the `nat` arm instead of creating a second, double-counted group for the same clues.

**Check:** fewer than 5% malformed derivations; `unexecuted_rate` under 15%; reasoning rate and
`n_reasoned` reported for every `nocot` model; no `think` run flagged `reasoning_absent`.

`metrics.py` first prints one line per model saying how many runs it kept and how many it set
aside. Runs whose prompt hash or config no longer match `configs/models.yaml` are excluded.
If a model shows `kept 0`, its config changed after it ran, so re-run it.

This is the last point to fix the checker. Also check that every model's `max_tokens` leaves
room for what the gate measured: a model that truncates often is measuring the ceiling, not
itself.

## 4. Freeze the analysis plan

Write the hypotheses, metrics and decision rules into `docs/analysis_plan.md` and record its hash.
From here on, the analysis is fixed. Deviations are appended to the plan's Outcome section and
never edited into the frozen part. To verify the hash:

```
awk '/FROZEN-BEGIN/{f=1;next} /FROZEN-END/{f=0} f' docs/analysis_plan.md | sha256sum
```

## 5. Full run

```
uv run python run.py --task nocot --systems <nocot models> --dataset nat --repeat 4 --sync --dry-run
uv run python run.py --task nocot --systems <nocot models> --dataset nat --repeat 4 --sync
uv run python run.py --task think --systems <think models> --dataset nat50 \
    --skip-done --sync --max-connections 4
```

Run each command with `--dry-run` first to see what it will do.

- **Always pass `--sync`.** Without it the runner uses OpenRouter's `:batch` model ids, which
  only work through a batch endpoint this harness doesn't implement.
- **Don't pass `--skip-done` on the `nocot` command.** It skips any clue already done under
  the same config regardless of repeat, so clues from the gate would end up with fewer samples
  than the rest. Re-running them is quick.
- **Keep `--max-connections` low on `think`.** OpenRouter refuses too many large reasoning
  calls in flight at once (HTTP 402).
- `--skip-done` resumes by config fingerprint, so a run whose prompt, model id or settings
  changed counts as a new measurement and is never merged with the old one.
- A model with a `cut` field in `configs/models.yaml` is refused unless you pass
  `--include-cut`.

**Check:** `n_errors` small and reported; every
manifest's reasoning gate still passing at full size.

## 6. Adjudicate wrong answers

```
uv run python analysis/browse.py --task nocot --dataset nat
```

Open `results/tables/answers_wide.html` and have the setter grade each distinct wrong answer
string without seeing which model produced it. Record the rulings in the CSV that
`ADJUDICATION_PATH` in `analysis/metrics.py` points to (columns `id,string,samples,accept,ruled,note`;
`accept` is `Y` or `N`). Report exact-match accuracy as primary and adjudicated accuracy
beside it.

## 7. Analyse

```
uv run python analysis/metrics.py --dataset nat
uv run python analysis/figures.py --dataset nat
uv run python analysis/browse.py --task think --dataset nat
```

Tables go to `results/tables/` and figures to `results/figures/`. A `note` column reading
"declined" means a number is missing for a stated reason, not that it is zero.
