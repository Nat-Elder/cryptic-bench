# External data

`eci_scores_2026-09-11.csv` — Epoch Capabilities Index (ECI) scores for the eight systems in
this study. **Not Epoch's published table**: fitted locally on 2026-09-11 with Epoch's own code
(`github.com/epoch-research/eci-public`, `scripts/fit_eci.py`) from their public
`benchmark_data.zip` downloaded that day, with 20 bootstrap draws for the intervals (the
central estimates do not depend on the bootstrap). Scale anchors: Claude 3.5 Sonnet = 130,
GPT-5 = 150. `eci_model_version` is Epoch's row identifier and says which configuration was
scored (e.g. `_max` reasoning, `_none`). Used for one exploratory figure; not in any
pre-registered number.
