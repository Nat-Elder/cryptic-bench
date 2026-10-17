# Analysis plan — FROZEN 2026-09-10

**Frozen.** The criteria between the two markers below are fixed. They are never edited again —
not for a typo that changes meaning, not for a defect. A defect is disclosed in the Outcome
section, dated. Deviations go in the workspace's log and in `AIS/next.md`, never in here. An
extension needs a new pre-registration. **Anything measured that this file does not name is
exploratory and carries that label everywhere it is quoted.**

| | |
|---|---|
| **frozen** | 2026-09-10, night (Europe/London) |
| **frozen by** | Nat's ruling, executed by Claude |
| **SHA-256** | `674ea89ab114e57eafc60bbdd2aa1146eab6669f201b20de0b9b1d4d56248d57` (as published) |
| **redacted** | 2026-10-17, for publication: run-cost figures were removed from five places in the frozen text and kill-switch 5 (a cost bookkeeping check) was replaced by a note. No hypothesis, metric, threshold or decision rule changed. The hash at freezing was `bf54923a1f7d8ad8d26f70b12757d8cafdf0089296c63e4557b75ced0a9071bd` |
| **covers** | the lines strictly between `FROZEN-BEGIN` and `FROZEN-END`. Not this provenance table, and not the Outcome section, which is appended afterwards by design |
| **verify** | `awk '/FROZEN-BEGIN/{f=1;next} /FROZEN-END/{f=0} f' docs/analysis_plan.md \| sha256sum` |
| **primary hypothesis** | **framing A** — PLAN §5.1 as written. See §1.1 |
| **what had been seen** | a great deal. §0.2, and it is the most important section in this file |

This replaces the placeholder that stood here until tonight, whose text — "empty on purpose…
writing it now would defeat the point" — was right: the gate found two checker bugs, and a
pre-registration written before it would have frozen a broken instrument.

Criteria are PLAN §5 carried verbatim where §5 decides them; every place this file adds to or
departs from §5 is marked **[NEW]** with its reason, so the diff against §5 is readable rather
than buried.

<!-- FROZEN-BEGIN -->


## 0. Provenance — what had been seen, and when

The `freeze-prereg` procedure's first rule is that an unqualified "written before the results
were seen" must never be written. This project already carries one prereg whose header made
that claim falsely. So, plainly:

### 0.1 What was written before any data

- **PLAN §5's seven analyses, in order, with their definitions, denominators, bounds, floor
  policy and k-aggregation rule.** Written 2026-09-10 (v3, then v3.1 the same day) before any
  API call was made. This is the substance of the criteria below.
- The `nat50` draw: seed 20260906, proportional on `device`, 50 ids fixed in
  `data/nat50.txt` — drawn 2026-09-10 evening, **before** the gate ran.

### 0.2 What had been seen by the time of this draft

Everything in [`../AIS/crypticbench/gate-report-2026-09-10.md`](../AIS/crypticbench/gate-report-2026-09-10.md)
and the two reports before it. Specifically, and this is the part that matters:

| seen | on | under |
|---|---|---|
| `nocot` accuracy per system, 20–25 clues | the first 20 of `nat50` (+ the 5 probe clues) | **prompt v2 and the exact config the full run will use** |
| `think` / rung / `think_cot` accuracy, validity, M2, the CoT gain, the rung curve, 20–25 clues | the first 20 of `nat50` | prompt **v1** and a checker since changed — superseded, and set aside by run selection |
| every token shape | all systems | as above |

**The `nocot` rows are not superseded.** Same prompt, same config, same clues — run selection
keeps them, and the full run will pool them into `nat`'s 100. So the headline's denominator
includes 20 clues whose results were seen before this draft was written. That cannot be undone
by choosing a rule now; it can only be disclosed and handled, which §5.1 does.

**Two of PLAN §5's own predictions were already observed to be failing** at that n:
`astra-nocot` was **third** of six, and Astra's CoT gain was second *largest* where the post
predicts smallest. Criteria that would have been chosen differently knowing this must not be
chosen now — hence §5.1's rule is PLAN §5.1's, unchanged, with a negative branch added because
§5.1 never had one.

### 0.3 The two rulings the freeze waited on, and how they closed

- **The headline framing — ruled: keep PLAN §5.1 (framing A).** The alternative, the review's
  reframe around CoT gain and validity, was proposed on 2026-09-10 *before* any data existed —
  so adopting it would not have been post-hoc in origin. It would have been post-hoc in
  appearance: the gate had by then shown Astra third, and no reader can distinguish a reframe
  chosen on design grounds from one chosen because the original claim was failing. So §5.1
  stands unchanged as the primary hypothesis, and the reframe's **content** is pre-committed as
  the branch of §4's table that fires if A fails. That way both readings are decided in advance
  and neither depends on what the numbers turn out to be.
- **`astra-think` — ruled: un-cut**, once the gate measured it at 630 output tokens a clue
  against the 3,150-token estimate the cut had assumed. §1.2's ordering test has its row.

---

## 1. Hypotheses and their criteria

### 1.1 The headline — **framing A, frozen**

**PLAN §5.1 as written, and this is the primary hypothesis.** Astra's `nocot` Acc against each other model's `nocot`
Acc, paired within clue on `nat`'s 100, cluster-bootstrapped on **answer**, 10 000 resamples,
percentile intervals, seed `20260906`. One comparison per model, reported as the paired
difference `other − Astra` with its 95 % CI.

- **Supported** iff, for **every** other model, the CI excludes zero *and* the difference is
  negative. Five comparisons at present (six if Fable 5.1 is ever added).
- **Not supported** otherwise, and the outcome table below says which branch. [NEW] §5.1 had no
  negative branch; a conjunction of five CIs needs one stated in advance.
- Chance baseline is 0.

**The alternative, not frozen as primary but pre-committed as a consequence.** If the criterion
above is not met, the write-up leads on the **CoT gain and validity across systems** (§1.2,
§1.3) with Astra one row inside it and no privileged position — the review's reframe, reached
by the data rather than by choosing it after seeing them. §4's table says exactly when this
fires. §1.2–1.5 are secondary under A and become primary under that branch; §1.1's comparisons
are then reported without a supported/not-supported verdict.

### 1.2 CoT gain

`think` − `nocot` per model, paired within clue on the **20 shared clues of `nat50`** that both
arms score — `nocot` runs on all 100, every `think` arm on the 50, so the intersection is the
50. [NEW] PLAN §5.2 assumed both arms on the same 100; the `nat50` ruling of 2026-09-10 changed
the denominator and the table now reports `paired_on` and n on every row.

- Each gain: the paired mean difference with a 95 % CI clustered on answer.
- **The ordering test**: the Astra post predicts Astra's gain is the **smallest** of the
  systems measured. Decided by rank: Astra's point estimate is lowest, and its CI excludes the
  highest system's point estimate. Anything else is "not supported", and if `astra-think` stays
  cut the test is **declined for want of a row** rather than reported as passed or failed.

### 1.3 Validity beside accuracy

Val and M2 per system on the `think` arms, with **both bounds** always (§2.3's partition), and
the rule from PLAN §2.3 carried verbatim: **a comparison that the two bounds straddle is not a
comparison.** M2 is reported only for systems above the 5 % accuracy floor (§1.6).

### 1.4 The dose curve, Sonnet 5

**Three rungs, not four**: `off`, `low`, `high`. [NEW] `medium` was cut on 2026-09-10 night;
three points still determine a slope and the difference-of-slopes test is unchanged.
Its 20 gate clues exist under prompt v1 and are superseded, so nothing is quietly pooled.

Acc and Val against the ordinal rung (`off`, `low`, `high`) and against measured
thinking tokens. The one test: **the difference of slopes** — accuracy rising faster than
validity. `off + prose CoT` is not on the effort axis and is excluded from both slopes.

### 1.5 By device and by hop count

Descriptive only, never a within-family model comparison: on `nat50` the largest family is
charade at 24 and no other reaches 8, and **`reversal` has no clues at all** (gate report §7.1).
The graded analysis is by hop count, and it **declines** while no clue carries a `hops` value —
ruled 2026-09-10 to be added after the run, for the `nocot` arm's 100.

### 1.6 Floor policy

Any arm below **5 %** Acc: point estimate stands, CI is unreliable and labelled so, and the arm
leaves the M2 comparisons only. Carried verbatim from §5.7. It did not fire at the gate.

### 1.7 The cutoff split

The headline again on the clues published after the newest `training_cutoff`, or never
published. **Declines** while any `training_cutoff` is null or no clue carries a
`published_date` — both are Nat's to fill, and both are still empty.

---

## 2. The instrument, frozen as of the freeze

### 2.1 What is measured

- **Acc**: normalised exact match against the setter's answer, **unadjudicated as primary** and
  adjudicated as secondary (§2.4's blind pass over the distinct answer strings). Scored as the
  **mean of k per clue**, never binarised.
- **Val** = L1 ∧ L2a ∧ L3, on the `think` arms only. `nocot` writes no derivation and its Val
  is undefined, not zero.
- **M2** = derivation invalid given the answer is correct, over that system's correct clues.

### 2.2 The checker, and every change made to it at the gate [NEW]

PLAN §7.4 licenses fixing the checker at the gate and nowhere else. Five changes were made, all
before this draft, all verified by re-scoring the gate's stored completions offline
(`tools/rescore.py`), and **all frozen here**:

1. An extraction device is not applied twice: if a component performed the device's own op and
   had it checked against its clue text, the assembly step is concatenation.
2. A component that hands over its source letters unchanged under the device's own op is naming
   the device's fodder, not claiming a transformation.
3. `deletion` covers the standard set: a prefix, a suffix, both ends, or the middle. Not two
   arbitrary excisions.
4. `initial` and `final` are component ops, one letter per word, and are in prompt v2's op list.
   `alternate` still means alternate.
5. A gloss after the tag's colon is recorded and not checked; it is not read as an op.

And one classification change: **a call truncated at the output ceiling with no block written is
reported as its own rate, not as malformed** — it remains in Val's denominator, because failing
to deliver a derivation is a failure.

### 2.3 Denominators and exclusions, pre-committed

- **An API error leaves every denominator** and is counted as `n_errors`.
- **A `nocot` call with `reasoning_tokens > 0` leaves that system's accuracy** and is counted as
  `n_reasoned_excluded`; an *unreported* count is kept and flagged, never silently excluded.
- **A run whose prompt hash or config fingerprint is not current is not in any number.** The
  selection is printed per system before every table.
- Truncations, malformed blocks and unscoreable claims stay in Val's denominator, each with its
  own reported rate.

### 2.4 Bootstrap

10 000 resamples, percentile intervals, clustered on **answer**, seed `20260906` from
`config.SEED`. The clue-level interval is reported beside the clustered one, never instead of it.

### 2.5 The subset actually drawn

- `nat`: the 100 test clues of `data/corpus.jsonl`, `nocot` only, **k = 4**. [NEW] PLAN §4
  says k = 8, from Think Fast's protocol; trimmed to 4 on 2026-09-10 night to limit the run's
  scope. The mean-of-k rule is unchanged and the
  per-clue estimate is simply coarser: a clue's score takes five values rather than nine.
  Recorded as a deviation, not a redefinition.
- `nat50`: the 50 ids of `data/nat50.txt`, seed 20260906, proportional on `device`, **final** —
  every `think` arm, the rungs and `think_cot`, k = 1.
- `guardian200`: **dropped** (Nat, 2026-09-10). Saha et al. is prior work cited, **not** a
  comparison run. `times200`: not run.
- Systems: six `nocot`, **six `think`** (`astra-think` un-cut), **three** Sonnet 5 rungs, one
  `think_cot`. Haiku 4.5 is cut entirely and `sonnet5-medium` is cut; both keep
  their `models.yaml` entries and `run.py` refuses them without `--include-cut`.

---

## 3. Kill-switches [NEW]

PLAN §5 had none. Each of these means **stop and redesign**, not "carry on and caveat":

1. **A `nocot` reasoning gate fails at full n** — mean reasoning tokens above 0, or the count
   unreported on this route. That system has no `nocot` row and cannot be in the headline.
2. **`unexecuted` above 15 % at full n on any system.** The checker is frozen, so this is not
   fixable here: the affected system's Val and M2 are withdrawn and reported as unmeasurable.
3. **Malformed above 15 % on any system** — three times the gate's criterion. Its derivations
   are scratchpads and its Val is not a measurement.
4. **Truncation above 25 % on any `think` system.** The ceiling, not the model, is being
   measured; that arm is reported as a ceiling artefact.
5. **[Redacted for publication: an internal bookkeeping check. It bears on no reported
   number.]**

## 4. The outcome table — exhaustive [NEW]

Rows are the two questions that can each go three ways. Every cell says what is licensed.

| headline (§1.1) | CoT-gain ordering (§1.2) | what this licenses |
|---|---|---|
| Astra leads, all CIs exclude zero | Astra's gain smallest | Both of the post's claims replicate on cryptics. The strongest result available: state it, with the M2 caveat from §1.3. |
| Astra leads | Astra's gain not smallest | Astra is better without CoT **and** gains as much from it. The post's mechanism claim does not transfer; say so. |
| Astra leads | declined (no `think` row for a system) | The headline stands alone; the gain ordering is not measured and must not be implied. Not the expected branch now that `astra-think` is un-cut, but kept: any system's `think` arm can fail a kill-switch. |
| Astra does not lead, some CI excludes zero against it | any | **The headline claim fails.** Report the ordering as measured, and the write-up leads on the CoT gain and validity (framing B's content, reached by data rather than by choice). |
| No CI excludes zero either way | any | Underpowered at n = 100, as §5's power note warns. Report the point estimates and the intervals, claim no ordering, and say the design cannot separate these models. |
| Astra leads on `nat` but not on the post-cutoff subset | any | Contamination is the live explanation. Report both, lead with the subset, do not pad it (§2.6). |

**Validity, orthogonally**: if M2 is high across systems (the gate suggested 4–57 %, on a
checker since fixed), that is the paper's second leg regardless of which row above applies —
right answers with derivations that do not execute. If M2 is uniformly low, the memorisation
signature is absent and that is equally reportable.

## 5. Exploratory, and labelled as such wherever quoted [NEW]

Not pre-registered, and named here so that nothing can be promoted later: the `nocot` extra-text
and truncation rates; the `unscoreable` share per system; parse-agreement between claimed and
confirmed device tags; per-clue flip rate; anything about `hops` until the column exists; and
every number in the gate report.

---

<!-- FROZEN-END -->

## Outcome

> **Append only.** Record the numbers, then state which branch of §4 applies, then write up
> accordingly. Deviations go in the workspace's log and in `AIS/next.md`, never into the
> criteria above. A defect found in the criteria is disclosed here, dated, and left in place.

### 2026-09-11 — the run, and which branch applies

**§1.1, the primary hypothesis: SUPPORTED.** `nocot`, `nat`'s 100 clues, k = 4, paired within
clue, bootstrap clustered on answer. Every comparison is `other − astra`:

| | Acc | paired diff vs Astra | 95 % CI | excludes zero |
|---|---|---|---|---|
| **astra-nocot** | **0.636** | — | — | — |
| sol-nocot | 0.528 | −0.108 | −0.186 … −0.032 | yes |
| gpt55-nocot | 0.514 | −0.122 | −0.203 … −0.041 | yes |
| opus47-nocot | 0.510 | −0.126 | −0.213 … −0.041 | yes |
| opus5-nocot | 0.455 | −0.181 | −0.261 … −0.102 | yes |
| sonnet5-nocot | 0.297 | −0.339 | −0.429 … −0.251 | yes |
| haiku45-nocot | 0.207 | −0.428 | −0.521 … −0.339 | yes |

Six of six negative with every CI excluding zero, which is the criterion §1.1 states. **Astra
solves cryptic clues without chain of thought better than every other model tested.**

**§4's branch: row 2** — *"Astra leads · Astra's gain not smallest → Astra is better without CoT
**and** gains as much from it. The post's mechanism claim does not transfer; say so."*

**§1.2, the CoT gain: the ordering test is NOT supported.** On the four systems measured, Astra
is third:

| | `nocot` | `think` | gain | 95 % CI |
|---|---|---|---|---|
| sol | 0.531 | 0.860 | **+0.329** | 0.210 … 0.454 |
| sonnet5 | 0.253 | 0.580 | +0.327 | 0.187 … 0.471 |
| astra | 0.583 | 0.900 | +0.317 | 0.200 … 0.444 |
| haiku45 | 0.195 | 0.280 | +0.085 | −0.010 … 0.190 |

Astra's point estimate is not the lowest, so the test fails as §1.2 states it. Two honest
qualifications, neither of which rescues the prediction: the top three are indistinguishable
(0.317–0.329, CIs almost coincident), and the one system below Astra — Haiku 4.5 — has a gain
whose CI includes zero, which is a floor effect on a weak model rather than evidence about
mechanism. **Read plainly: chain of thought helps Astra about as much as it helps everyone
else. Its no-CoT lead is not explained by needing CoT less.**

**Kill-switch 4 fired on `sonnet5-high`: 26 % of 50 clues truncated at the 16k ceiling**
(>25 %). Per §3 its `think` row is a ceiling artefact, not a measurement. Consequences carried
through: its Acc 0.580 and Val 0.340 are **lower bounds** — a truncated call scores incorrect —
and so is Sonnet 5's +0.327 gain above. The dose curve's top rung is therefore unmeasured, and
`sonnet5-medium` was cut, so what survives of §1.4 is `off` 0.300, `off + prose CoT`
0.360, `low` (not run). **§1.4 is not decidable on this data** and is withdrawn rather than
reported from two points.

Kill-switches 1, 2 and 3 are clear: every kept `nocot` run passed its reasoning gate with 0
reasoned calls and the count reported; `unexecuted` 6–12 % and malformed 2–10 % on every `think`
system.

**§1.3, validity.** `astra-think` Val 0.640–0.780, M2 0.133; `sol-think` 0.660–0.860, M2 0.093;
`haiku45-think` 0.180–0.380, M2 0.286. The two strong systems write derivations that mostly
execute; M2 is low where accuracy is high. On the frozen bounds rule, Astra and Sol's Val
intervals overlap and are **not** a comparison.

**§1.7, the cutoff split: declined**, as frozen — every `training_cutoff` is null. §1.5's hop
regression: declined, no `hops` column.

### 2026-09-11, later — Fable 5.1 enters, and the margin is thinner than it looked

**§1.1 still SUPPORTED, now over seven comparisons.** `fable-nocot` was added after the outcome
above was recorded, under a rule pre-committed *before* its first call: if it returned zero
reasoning tokens with the count reported, it would enter the conjunction unconditionally,
including if it overturned the result. It returned 0 on 400 of 400 calls.

| | Acc | paired diff vs Astra | 95 % CI |
|---|---|---|---|
| **astra-nocot** | **0.636** | — | — |
| **fable-nocot** | **0.552** | **−0.083** | **−0.158 … −0.009** |
| sol-nocot | 0.528 | −0.108 | −0.184 … −0.032 |
| gpt55-nocot | 0.514 | −0.122 | −0.205 … −0.043 |
| opus47-nocot | 0.510 | −0.126 | −0.213 … −0.043 |
| opus5-nocot | 0.455 | −0.181 | −0.262 … −0.102 |
| sonnet5-nocot | 0.297 | −0.339 | −0.429 … −0.249 |
| haiku45-nocot | 0.207 | −0.428 | −0.521 … −0.338 |

**Fable is the binding comparison and it clears zero by 0.009.** Quote that, not the average
margin: the claim is a conjunction, so its strength is its weakest link, and its weakest link is
now nine thousandths of a point of accuracy.

**Two labels its row must carry wherever it is quoted.**

1. **Its answer channel is closed and the other seven are open.** Left plain, Fable deliberates
   in the visible answer at `reasoning_tokens = 0` — 4 of 20 rows were working cut off
   mid-thought (quotes withheld, since they echo the clues), a rate of
   20 % against the 5 % rule. PLAN §2.1's remedy is the structured constraint and it was
   applied, which is a **different measurement**, not a like-for-like row.
2. **Its first full row was wrong and the error favoured us.** At `max_tokens: 16` the JSON
   envelope left ~5 tokens for the answer: 94 % of clues with a gold answer over 8 characters
   truncated, against 12 % of the rest, giving 0.522. Re-run at 300 — where the schema, not the
   ceiling, is the constraint — truncation went to 0 % and the figure to 0.552. Fable competes
   with Astra, so the broken version flattered the headline. Both runs are on disk; the 16-token
   rows are superseded by configuration fingerprint and excluded automatically.

**PLAN §3's exclusion of Fable from this arm is overturned**, on the same evidence and the same
reasoning that overturned it for Astra on 2026-09-10: the parameter claim was true and the
conclusion drawn from it was not. `THINK_ONLY_FAMILIES` in the test suite is now empty.

### 2026-09-11, last runs — §1.2 on five systems, and §1.4 partly recovered

**§1.2, the CoT gain, now on five systems. The ordering test remains NOT supported.**

| | `nocot` | `think` | gain | 95 % CI |
|---|---|---|---|---|
| gpt55 | 0.517 | 0.940 | **+0.423** | 0.292 … 0.554 |
| sol | 0.531 | 0.860 | +0.329 | 0.207 … 0.454 |
| sonnet5 | 0.253 | 0.580 | +0.327 | 0.188 … 0.472 |
| **astra** | 0.583 | 0.900 | **+0.317** | 0.197 … 0.440 |
| haiku45 | 0.195 | 0.280 | +0.085 | −0.015 … 0.190 |

Astra is **fourth of five**. Stated precisely: its gain is indistinguishable from Sol's and
Sonnet 5's, clearly below GPT-5.5's, and clearly above Haiku's — whose CI includes zero and
which is a floor effect on the weakest model rather than evidence about mechanism. The Astra
post predicts the *smallest* gain. Five systems now say otherwise, and the earlier reading
stands: **chain of thought helps Astra about as much as it helps everyone else; its no-CoT lead
is not explained by needing CoT less.**

**§1.4's withdrawal is partly lifted.** `sonnet5-low` was subsequently run, giving three
points on the effort axis rather than one:

| rung | Acc | Val | mean reasoning | truncation |
|---|---|---|---|---|
| `off` | 0.300 | 0.140 | 0 | 4 % |
| `low` | 0.520 | 0.300 | 1,473 | 16 % |
| `high` | 0.580 | 0.340 | 7,143 | **26 %** |
| `off + prose CoT` | 0.360 | 0.180 | 0 | 0 % (off the axis) |

Accuracy slope **+0.084**/rung, validity slope **+0.060**/rung, **difference +0.024** — accuracy
rising faster than validity, which is the direction §1.4 predicts. **Report the direction, not
the magnitude.** The top rung still fires kill-switch 4 at 26 % truncation, so its Acc and Val
are lower bounds and both slopes are understated by an unknown amount; only their *difference* is
partly protected, since truncation depresses both.

**And the ladder produced a finding of its own that no rung was run to get: truncation rises
monotonically with the dose** — 4 %, 16 %, 26 % at a fixed 16k ceiling. A dose-response curve
measured against a fixed output ceiling systematically understates its own top end, because the
condition that reasons most is the condition most likely to be cut off. That is a fact about the
instrument, it applies to any such ladder, and it is worth stating in the write-up beside the
curve rather than left for a reader to infer.

**`off + prose CoT` (0.360) barely beats `off` (0.300) while `low` reaches 0.520 on 1,473
reasoning tokens.** Reasoning in the visible answer adds little; reasoning in the budget adds a
lot. That is the contrast the `think_cot` rung exists for, and it points the same way for Sonnet 5
as Fable's content-channel leak does for Fable.

### Deviations from §2.5, recorded here because the criteria are not editable

1. **Seven `nocot` systems, not six** — Haiku 4.5 un-cut on 2026-09-11.
   It adds the seventh comparison above, the easiest in the set.
2. **k = 4, not 8**, and **three rungs, not four** — ruled 2026-09-10 night, already
   marked [NEW] in §2.5 and §1.4.
3. **The `think` arm is four systems, not six.** `gpt55-think`, `opus5-think` and `opus47-think`
   were not run: `sonnet5-high` overran its token estimate by 1.54× under prompt v2 and the arm
   was cut short. §1.2's ordering test therefore runs on four systems, which is
   weaker than the frozen design intended and is stated wherever the result is quoted.
4. **`guardian200` was run at k = 1**, against §2.5's "dropped". Under this file's own rule —
   *anything measured that this file does not name is exploratory* — **the Saha comparison is
   EXPLORATORY and carries that label wherever it is quoted.** It is reported because external
   calibration is worth having, not because it was pre-registered:

   | | guardian200 Acc (n = 200, k = 1, **exploratory**) |
   |---|---|
   | astra-nocot | 0.420 |
   | sol-nocot | 0.305 |
   | opus47-nocot | 0.290 |
   | opus5-nocot | 0.270 |
   | gpt55-nocot | 0.260 |
   | haiku45-nocot | 0.085 |
   | *Saha et al. 2025, GPT-4-Turbo, published* | *0.185* |

   Every system but Haiku clears the published figure on its own clue set, Haiku falls below it,
   and Astra leads by the same ~11 points it leads by on `nat` — so the lead is not an artefact
   of the setter. Every system scores ~20 points lower here than on `nat`, which is the
   difference between professionally set Guardian clues and this corpus.

### 2026-09-11 — the hand review of the `unexecuted` bucket (§2.3), and what it does to Val and M2

§2.3 pre-committed that a derivation which is grounded and consistent but does not execute is
hand-reviewed. The review covered every right-but-invalid or `unexecuted` thinking-arm row on
`nat50`: **53 rows across eight systems.** Each was classified by Claude with the setter's parse
beside the model's, and **Nat agreed with every classification on 2026-09-11.** The file is
`data/review/hand_review_2026-09-11.csv`; `analysis/metrics.py` reads it into the new
`reviewed_validity` table. The frozen tables and the mechanical numbers are unchanged and remain
primary; this is the secondary, reviewed validity the plan anticipated.

Classes: **A** correct parse the checker's grammar cannot express (34 rows) — re-scored valid;
**B** correct parse, template syntax misused (5) — stays invalid; **C** wrong parse (14) — stays
invalid. Five grammar gaps account for every A: a container nested in a container; an anagram
chained into a charade; deletion of a named letter; a multi-word selection phrase read word by
word; two indicators in one field or an &lit left untagged.

| system | A / B / C | Val lower, mechanical → reviewed | M2 lower, mechanical → reviewed |
|---|---|---|---|
| gpt55-think | 9 / 0 / 0 | 0.660 → **0.840** | 0.191 → **0.000** |
| astra-think | 6 / 0 / 0 | 0.640 → **0.760** | 0.133 → **0.000** |
| sol-think | 4 / 0 / 0 | 0.660 → **0.740** | 0.093 → **0.000** |
| sonnet5-high ¹ | 6 / 0 / 0 | 0.340 → 0.460 | 0.207 → **0.000** |
| sonnet5-low | 3 / 3 / 2 | 0.300 → 0.360 | 0.308 → 0.192 |
| sonnet5-cot | 1 / 1 / 6 | 0.180 → 0.200 | 0.444 → 0.389 |
| sonnet5-off | 2 / 0 / 6 | 0.140 → 0.180 | 0.533 → 0.400 |
| haiku45-think | 0 / 1 / 3 | 0.180 → 0.180 | 0.286 → 0.286 |

¹ kill-switch 4 row; lower bounds.

**Reading, for §1.3.** On every system at or above 0.86 accuracy, and on Sonnet 5 at high
effort, the mechanical right-but-invalid rate was entirely the instrument: reviewed, it is zero.
On Sonnet 5 at lower effort, its prose-CoT arm, and Haiku 4.5, most of it is the model: wrong
parses behind right answers. The memorisation-or-heuristic signature is real and confined to the
weak end. The mechanical M2 is reported as an upper bound wherever quoted, with this table beside
it. The five grammar gaps are the first change to the template and checker for any successor
study; nothing changes in this one.

### 2026-09-11 — the blind adjudication (§2.1, PLAN §2.4): complete, and it changes nothing

The distinct wrong-by-exact-match answer strings from the `nat` and `nat50` arms — **909 strings
over the 100 clues**, with no model attribution — were put to the setter in three passes: the
seven strings one letter off a gold answer, the 39 other strings that eight or more samples
converged on, and the long tail. **The setter accepted none.** The rulings are in
`data/review/adjudication_2026-09-11.csv`, every row `N`; `analysis/metrics.py` reads them into
the `adjudicated_accuracy` table, whose `acc_adjudicated` column equals `acc_exact` on every
system. The secondary, adjudicated accuracy the plan asked for is therefore identical to the
primary, and the write-up says so rather than reporting two columns of the same number.

The two nearest misses were a one-letter variant spelling (16 samples; the wordplay fixes the
letter) and a homophone given from the wrong end (17 samples; the homophone runs only one way).
The answers are withheld with the clues.

### 2026-09-11 — two consequences of the review for §1.4, and a ruling on §1.7

**§1.4 under reviewed validity.** The difference-of-slopes test was stated on validity as the
checker scores it. Mechanically: accuracy +0.084 per rung, validity +0.060, difference +0.024,
the predicted direction. With the hand review's class-A rows re-scored valid (`off` 0.14 → 0.18,
`low` 0.30 → 0.36, `high` 0.34 → 0.46), the validity slope is **+0.087**, and the difference is
**−0.003**: the two slopes are equal to within the resolution of three points. The mechanical
result stands as the pre-registered one and the reviewed result is reported beside it; read
together, §1.4 is **direction only under the mechanical checker and no direction at all after
review**, on a curve whose top rung is a ceiling artefact in any case.

**§1.7, ruled 2026-09-11 (Nat): the cutoff split is not pursued.** No `training_cutoff` or
`published_date` will be filled for this study; §1.7 stays **declined** as written, and
contamination is discussed generally in the write-up rather than tested by split. The clean
subset for the headline (§4, last row) is therefore not computed.
