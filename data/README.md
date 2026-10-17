# Data

The setter's clues are not in this repository. What is here:

| file | contents |
|---|---|
| `nat50.txt` | ids of the 50 clues used for the reasoning-on arm, with the draw's seed |
| `review/` | the setter's rulings on wrong answers and hand-reviewed derivations (no clue text) |
| `external/` | Epoch Capabilities Index scores for the eight models |

The public calibration sets are downloaded rather than committed. Each script rebuilds the
exact file the runs were hashed against:

| file | build with |
|---|---|
| `calibration/guardian200.jsonl` | `uv run python tools/fetch_calibration.py` (200 Guardian clues, Saha et al. 2025) |
| `calibration/times200.jsonl` | `uv run python tools/build_times200.py` (200 Times clues, Sadallah, Kotova & Kochmar 2024; not used in the study) |

## Supplying a corpus

`run.py`, `tools/validate_corpus.py` and the few-shot prompts read three files from this
directory. To run the harness on your own clues, create them in this shape.

**`corpus.jsonl`** — one clue per line:

```json
{"id": "cb-0001", "clue": "Sing air wildly, lifting", "answer": "RAISING",
 "answer_display": "RAISING", "enumeration": "7", "source": "other",
 "published_date": null, "published_url": null, "setter": "you",
 "split": "test", "notes": {}}
```

`clue` is the surface with the enumeration removed; `answer` is A–Z only; `split` is `test`
or `dev`. Dev rows are the few-shot examples and are never scored. `validate_corpus.py`
refuses a test clue that shares an answer with a dev clue.

**`devices.jsonl`** — one line per clue id, joined in by `run.py`:

```json
{"id": "cb-0001", "device": "anagram", "confirmed": true}
```

**`exemplars.json`** — a derivation for each dev clue, in the seven-field template of
`src/crypticbench/derivation.py`, under a top-level `"derivations"` key. The `think` prompts
render these as their worked examples; a dev clue with no exemplar is skipped.

The clues quoted in `tests/` are invented for the purpose. The test modules that read the
corpus itself (the runner wiring, the analysis tables, the end-to-end plumbing and the
shipped-corpus checks) skip when it is absent; the rest of the suite runs anywhere.
