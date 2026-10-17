# Data: provenance and licences

| file | contents | source | licence | in repo |
|---|---|---|---|---|
| `data/corpus.jsonl` | the setter's 104 clues (100 test, 4 few-shot) | the setter | the setter's, all rights reserved | no |
| `data/devices.jsonl` | device type and one parse per clue | suggested by a model, not yet confirmed by the setter | as the corpus | no |
| `data/exemplars.json` | worked derivations of the 4 few-shot clues | the setter | as the corpus | no |
| `data/nat50.txt` | ids of the 50 clues used for the reasoning-on arm, with the draw's seed | `tools/draw_nat50.py` | MIT | yes |
| `data/review/*.csv` | the setter's rulings on wrong answers and on hand-reviewed derivations | the setter | MIT | yes |
| `data/external/eci_scores_2026-09-11.csv` | Epoch Capabilities Index scores for the eight models | fitted locally with Epoch's public code and data; see `data/external/README.md` | Epoch's terms | yes |
| `data/calibration/guardian200.jsonl` | 200 Guardian clues | [Saha et al. 2025](https://github.com/espressovi/LMCrossword), via `tools/fetch_calibration.py` | MIT (theirs); clues are the Guardian setters' | no, fetched |
| `data/calibration/times200.jsonl` | 200 Times clues with device labels | [Sadallah, Kotova & Kochmar 2024](https://github.com/bodasadallah/decrypting-crosswords), via `tools/build_times200.py` | MIT (theirs); clues are the Times setters' | no, fetched |

## The corpus

100 test clues and 4 few-shot clues that are shown in every prompt and never scored. The
validator rejects a test clue that shares an answer with a few-shot clue. Every manifest in
`results/runs/` records `dataset_sha256`, so each result is tied to the exact file it ran on.

Row fields: `id` (`cb-NNNN`), `clue` (without enumeration), `answer` (A–Z), `answer_display`,
`enumeration`, `source`, `published_date`, `published_url`, `setter`, `split`, `notes`.
[`data/README.md`](../data/README.md) shows how to supply a corpus of your own.

No row has a `published_date`, so the training-cutoff analysis was not run, and no device tag
has been confirmed by the setter, so per-device figures rest on the suggested tags.

## Not committed

- **The setter's clues, tags and exemplars.** They are the test set and are not released.
  The tests don't need them.
- **The calibration sets.** The scripts download them from the original projects and rebuild
  the exact files the manifests hash (guardian200 `34f106a0…`, times200 `9748981f…`).
- **Raw exports and the app database** the corpus was built from (`data/raw/`, `*.sqlite3`).
- **Model answers and logs**, which quote the clues.

The run manifests are committed. They contain the model, config, prompt and corpus hashes,
token usage and the reasoning check, and no clue text or answers.
