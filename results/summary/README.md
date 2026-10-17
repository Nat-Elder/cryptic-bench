# Summary results

Aggregate numbers behind the README and the figures in `figures/`, copied from the Outcome
section of `docs/analysis_plan.md`. They contain no clue text. Rates are proportions (0–1).

| file | contents |
|---|---|
| `no_cot_accuracy.csv` | no-CoT accuracy on 100 clues (k = 4), 95% CI, and paired difference vs GPT-6 Astra |
| `reasoning_gain.csv` | accuracy without and with reasoning on the 50-clue subset, and the gain with 95% CI |
| `derivation_review.csv` | derivation validity before and after hand review, and the share of correct answers whose derivation is broken (lower bounds; double definitions and &lit excluded) |
| `effort_ladder.csv` | Claude Sonnet 5 at three reasoning-effort settings |

`python analysis/summary_figures.py` redraws the figures from these files.
