#!/usr/bin/env python
"""The seven tables of PLAN §5, in that order, from results/runs/*.csv (+ manifests).

    python analysis/metrics.py [--results-dir results] [--out-dir results] [--dataset nat]
                               [--all]

Every function takes the results table and returns a DataFrame; analysis/figures.py renders.
Nothing here decides anything the plan has not already decided — the order below *is* §5, and
the pre-registration freezes it verbatim at run-order step 6.

    1  per_system      Acc with a cluster bootstrap on answer; Val as the three-way partition
                       with both bounds; M2 with the floor policy
    2  headline        paired `nocot` Acc, every model against Astra, within clue
    3  cot_gain        `think` - `nocot` per model, paired within clue, and their ordering
    4  by_device       descriptive only; by_hops is PLAN §5.4's graded analysis
    5  sonnet_rungs    Acc and Val against rung and against measured thinking tokens
    6  cutoff_split    the headline again on the clues after the newest training cutoff
    7  flip_rate       per-clue disagreement across repeats

Bootstrap: 10 000 resamples, seed from config, percentile intervals, **clustered on answer**
(re-clue pairs share an answer, so clues are not independent draws).

**The `nocot` arm excludes calls that reasoned** (prompt v2, 2026-09-10). The arm asks the
model not to reason and verifies the count per call, so a call with `reasoning_tokens > 0` is
not a no-CoT call: it leaves the accuracy and is reported beside it as `n_reasoned_excluded`,
with `n_reasoning_unreported` for the calls that carried no count at all. Neither is folded
into `n_errors`, and `mean_reasoning_tokens` / `reasoning_rate` are still computed over every
call, so the size of what was dropped stays visible.

**Runs are selected, not pooled** (2026-09-10). A results directory accumulates every run ever
made, including ones taken under a prompt or a config that has since changed, and grouping them
on `(system, task, dataset)` alone silently averaged four configurations of `sonnet5-nocot` into
one row. `load_runs()` therefore keeps only the runs whose manifest `prompt_sha256` **and**
`config_fingerprint` match what that system would send today — the same test `--skip-done`
applies — and drops the `mock*` systems and any system carrying a `cut` field. It prints one
line per system saying what it kept, what it set aside and why. `--all` turns the selection off
for archaeology; nothing is moved and nothing is deleted.

Three refusals, all of the same shape as run.py refusing a null model id: the hop regression
declines while no clue carries a hop count, the cutoff split declines while any
`training_cutoff` or `published_date` is null, and the headline declines if Astra has no rows.
A table that says why it is empty is worth more than a number computed from a guess.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from crypticbench.config import MODELS_YAML, RESULTS_DIR, SEED  # noqa: E402
from crypticbench.prompts import load_prompt  # noqa: E402
from run import (  # noqa: E402
    DEFAULT_PROMPT_VERSION,
    config_fingerprint,
    generate_fields,
    resolve_route,
)

N_BOOT = 10_000

# The headline model (PLAN §1). Every other model's nocot Acc is compared against this one.
REFERENCE_FAMILY = "astra"
# PLAN §5.7. Below this accuracy a bootstrap interval on a proportion is not to be believed,
# and the arm leaves the M2 comparisons — it stays in the accuracy table with its point
# estimate and a flag, because "this model cannot do it" is itself the result.
ACC_FLOOR = 0.05
# PLAN §2.3. Above this share of L1 ∧ L3 ∧ ¬L2a the checker is wrong, not the models.
UNEXECUTED_CEILING = 0.15
RUNG_ORDER = {"off": 0, "low": 1, "medium": 2, "high": 3}
# `nat` and `nat50` are one study, not two splits to keep apart: `nocot` runs on all 100 clues
# and every `think` arm on the 50 (PLAN §6, Nat 2026-09-10), so the CoT gain pairs across the
# two by clue id. `guardian200` and `times200` remain separate and are never pooled with either.
NAT_FAMILY = ("nat", "nat50")

# Saha et al. 2025, post-cutoff Guardian set (n=200), 5-shot, exact match, 10-token cap — which
# makes the published number a no-CoT number already, and so the one comparator for `nocot` on
# guardian200. Only GPT-4-Turbo has a post-cutoff Guardian figure on record; the paper's other
# numbers are for different clue sets. Do not add a number here without a source that states it
# for *this* set. Source: AIS/reading/ml/language-models-are-crossword-solvers.md
SAHA_GUARDIAN = {"GPT-4-Turbo (Saha et al. 2025, post-cutoff Guardian)": 0.185}


# ---- loading ----------------------------------------------------------------------------

def load_systems() -> dict:
    import yaml
    try:
        return yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or {}
    except OSError:
        return {}


def current_selectors(systems: dict | None = None) -> dict[str, dict]:
    """What a run of each system would look like today: its prompt hash and its fingerprints.

    The prompt is the arm's default version (`DEFAULT_PROMPT_VERSION`), which is what a bare
    `run.py --task nocot` sends. **Two** fingerprints are accepted per system, the `--sync` one
    and the `:batch` one, because `--sync` changes the model id and nothing about what the
    model is asked — so a gate row and a full-run row of the same configuration belong
    in the same number, even though `--skip-done` deliberately keeps them apart.
    """
    systems = systems if systems is not None else load_systems()
    out: dict[str, dict] = {}
    for key, cfg in (systems or {}).items():
        cfg = cfg or {}
        arm = cfg.get("arm")
        version = DEFAULT_PROMPT_VERSION.get(arm)
        try:
            prompt_sha = load_prompt(arm, version)[2] if version else None
        except OSError:
            prompt_sha = None
        fingerprints = set()
        for sync in (False, True):
            model, _batched = resolve_route(cfg, sync=sync)
            if model is None:
                continue
            gen, structured = generate_fields(cfg, sync=sync, structured=False)
            fingerprints.add(config_fingerprint(model, cfg.get("model_args", {}), gen,
                                                structured))
        out[key] = {"prompt_version": version, "prompt_sha256": prompt_sha,
                    "fingerprints": fingerprints, "cut": cfg.get("cut"),
                    "route": cfg.get("route")}
    return out


def selection_reason(manifest: dict, selectors: dict[str, dict]) -> str | None:
    """Why this run is not part of the current measurement, or None if it is.

    The order matters: a cut system is set aside whatever its config, and a system that is no
    longer in models.yaml cannot be checked against anything.
    """
    key = manifest.get("system")
    sel = selectors.get(key)
    if sel is None:
        return "not in configs/models.yaml"
    # A cut system that was deliberately run (`--include-cut`, recorded in the manifest) is
    # kept: the point of running it was to see the number. The cut still governs the run
    # *shape* — it is why no such row appears without the flag.
    if sel["route"] == "mock" or str(key).startswith("mock"):
        return "mock system (plumbing, not a measurement)"
    if sel["cut"] and not manifest.get("include_cut"):
        return f"cut — {sel['cut']}"
    got_prompt = manifest.get("prompt_sha256")
    if sel["prompt_sha256"] and got_prompt != sel["prompt_sha256"]:
        return (f"prompt superseded (run {str(got_prompt)[:12]}…, current "
                f"{sel['prompt_version']} {str(sel['prompt_sha256'])[:12]}…)")
    got_fp = manifest.get("config_fingerprint")
    if sel["fingerprints"] and got_fp not in sel["fingerprints"]:
        return (f"config superseded (run {got_fp}, current "
                f"{'/'.join(sorted(sel['fingerprints']))})")
    return None


def load_runs(results_dir: Path, *, keep_all: bool = False) -> pd.DataFrame:
    selectors = current_selectors()
    frames = []
    kept: dict[str, int] = {}
    dropped: dict[str, list[str]] = {}
    for csv_path in sorted((results_dir / "runs").glob("*.csv")):
        man_path = csv_path.with_suffix(".json")
        if not man_path.exists():
            sys.exit(f"{csv_path.name} has no manifest; refusing")
        man = json.loads(man_path.read_text(encoding="utf-8"))
        key = man.get("system", "?")
        reason = None if keep_all else selection_reason(man, selectors)
        if reason is not None:
            dropped.setdefault(key, []).append(reason)
            continue
        kept[key] = kept.get(key, 0) + 1
        df = pd.read_csv(csv_path, dtype={"extracted": str, "completion": str, "thinking": str,
                                          "extra_text": str}, keep_default_na=False)
        df["model"] = man.get("model")
        df["batched"] = man.get("batched")
        df["repeat_index"] = man.get("repeat_index", 1)
        df["dataset_sha256"] = man.get("dataset_sha256")
        df["prompt_version"] = man.get("prompt_version")
        frames.append(df)
    report_selection(kept, dropped, keep_all=keep_all)
    if not frames:
        sys.exit("no runs selected. Every run in the directory was set aside — see the lines "
                 "above, or pass --all to read them anyway.")
    df = pd.concat(frames, ignore_index=True)
    # Runs against different versions of a dataset are not comparable. Ids are stable across a
    # re-finalise, so this is a disclosure rather than a refusal — but it must not be silent.
    for dataset, g in df.groupby("dataset"):
        digests = sorted(x for x in g["dataset_sha256"].dropna().unique() if x)
        if len(digests) > 1:
            print(f"WARNING: {dataset} runs span {len(digests)} corpus versions "
                  f"({', '.join(d[:12] for d in digests)}); the tables below mix them.",
                  file=sys.stderr)
    return annotate(df)


def report_selection(kept: dict[str, int], dropped: dict[str, list[str]], *,
                     keep_all: bool) -> None:
    """One line per system: how many runs were kept, how many set aside and why.

    Printed rather than logged, because it is the first thing to read in the output: a table
    computed over the wrong runs looks exactly like a table computed over the right ones.
    """
    if keep_all:
        print("run selection OFF (--all): every run in the directory is included, including "
              "superseded configurations, cut systems and mock runs. Not the measurement.")
        return
    print("run selection (prompt sha + config fingerprint, per configs/models.yaml):")
    for key in sorted(set(kept) | set(dropped)):
        reasons = dropped.get(key, [])
        counts: dict[str, int] = {}
        for r in reasons:
            counts[r] = counts.get(r, 0) + 1
        why = ("; ".join(f"{n} × {r}" for r, n in counts.items()) if counts else "")
        print(f"  {key:16} kept {kept.get(key, 0)}, set aside {len(reasons)}"
              + (f": {why}" if why else ""))


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric coercion, boolean coercion, and the family/role/rung join from models.yaml."""
    df = df.copy()
    for col in ("correct", "input_tokens", "output_tokens", "reasoning_tokens",
                "extra_words", "hops", "repeat_index"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("no_answer_line", "len_ok", "published", "truncated", "unexecuted",
                "has_extra", "l0", "l1", "l3", "structured_output"):
        if col in df:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1"])
    # A `nocot` call that reasoned is not a no-CoT call (prompt v2: the arm is "asked not to
    # reason, verified per call"). NaN is *not* excluded — an unreported count is an absence of
    # evidence, which the run's own reasoning gate already fails on this route; excluding it
    # here would silently empty the arm instead.
    reasoned = df["reasoning_tokens"] > 0 if "reasoning_tokens" in df else False
    df["reasoning_excluded"] = (df["task"] == "nocot") & reasoned
    systems = load_systems()
    for key in ("family", "role", "rung", "training_cutoff"):
        df[key] = df["system"].map(lambda s, k=key: (systems.get(s) or {}).get(k))
    df["family"] = df["family"].fillna(df["system"])
    return df


def _no_block(df: pd.DataFrame) -> pd.Series:
    """Rows whose completion never contained a derivation to check.

    Either the call was cut off at the output ceiling (`stop_reason == max_tokens`) or the
    visible completion is empty — at the 2026-09-10 gate those were the same 11 rows, all with
    the reasoning budget consumed. Kept as a helper rather than inlined because two tables ask
    the question and they must not answer it differently.
    """
    stop = df["stop_reason"].astype(str) if "stop_reason" in df else pd.Series("", index=df.index)
    completion = (df["completion"].astype(str).str.strip() if "completion" in df
                  else pd.Series("x", index=df.index))
    truncated = (df["truncated"].fillna(False) if "truncated" in df
                 else pd.Series(False, index=df.index))
    return (stop == "max_tokens") | truncated.astype(bool) | (completion == "")


def scored(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a score, and — on `nocot` — only the calls that did not reason.

    Two exclusions, for two different reasons. An errored API call is not a wrong answer: it
    leaves the denominator and is counted as `n_errors` (PLAN §8, carried ruling). A `nocot`
    call with `reasoning_tokens > 0` is not a wrong answer either: it is a call from a
    different condition, so it leaves the denominator and is counted as `n_reasoned_excluded`
    (probe report §7.4 (3)). Every table below goes through here or through `by_clue`, so the
    exclusion is arm-wide and not something a single table can forget.
    """
    keep = df["correct"].notna()
    if "reasoning_excluded" in df:
        keep &= ~df["reasoning_excluded"].fillna(False)
    return df[keep]


def by_clue(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse k repeats to one value per (system, clue): the **mean of k**, not a binarised
    majority. A model right 5 times in 8 is 0.625, and pretending it is 1 discards exactly the
    variation k = 8 was chosen to measure (plan-v3.1-review §2)."""
    s = scored(df)
    if s.empty:
        return s
    return (s.groupby(["system", "family", "role", "task", "dataset", "id"], dropna=False)
             .agg(correct=("correct", "mean"), k=("correct", "size"),
                  target=("target", "first"), device=("device", "first"),
                  hops=("hops", "first"), published_date=("published_date", "first"))
             .reset_index())


# ---- bootstrap --------------------------------------------------------------------------

def boot_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_ci_clustered(values: np.ndarray, clusters: np.ndarray, rng: np.random.Generator,
                      n_boot: int = N_BOOT) -> tuple[float, float]:
    """Bootstrap resampling *clusters* of clues rather than clues.

    The clue-level interval assumes clues are independent draws. They are not: answers clued
    twice share their difficulty, and the corpus keeps one surface per re-clued answer only
    after Nat's marks. Resampling by answer treats each answer as the independent unit, which
    widens the interval by however much that dependence is worth. Reported *beside* the
    clue-level interval, never in place of it, so the size of the correction stays visible.
    """
    if len(values) == 0:
        return (np.nan, np.nan)
    groups: dict[object, list[float]] = {}
    for v, c in zip(values, clusters):
        groups.setdefault(c, []).append(float(v))
    keys = list(groups)
    if len(keys) == len(values):  # no clustering present; identical to the clue-level interval
        return boot_ci(values, rng)
    members = [np.asarray(groups[k]) for k in keys]
    sizes = np.array([len(m) for m in members], dtype=float)
    sums = np.array([m.sum() for m in members], dtype=float)
    draw = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    means = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff(a: pd.Series, b: pd.Series, answers: pd.Series,
                rng: np.random.Generator) -> dict:
    """Mean per-clue difference a - b, with a bootstrap clustered on answer.

    Every system sees the identical clues, so the informative quantity is the *per-clue*
    difference: the bootstrap resamples clue ids (through their answer clusters) and reads both
    systems' scores for the ids drawn, which keeps the correlation between them. Resampling
    each system independently would give an interval as wide as the marginal ones and would
    not be a test of a difference at all.
    """
    d = (a - b).to_numpy(float)
    if len(d) == 0:
        return {"n_paired": 0, "diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "excludes_zero": False}
    lo, hi = boot_ci_clustered(d, answers.to_numpy(), rng)
    return {"n_paired": len(d), "diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0)}


# ---- 1. per system ------------------------------------------------------------------------

def per_system(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Acc, Val (three-way, both bounds) and M2 (both bounds) for every system.

    **Val is only defined on the `think` arms.** `nocot` writes no derivation, so its Val
    columns are NaN rather than 0 — PLAN §1's "every clue" denominator cannot mean an arm with
    nothing to check (plan-v3.1-review §2).

    **The partition.** `valid`, `invalid` and `unscoreable` are the three ways a parsed
    derivation can land, and the model chooses which by naming its own device, so the
    unscoreable set is endogenous. Val is therefore an interval: `val_lower` counts only the
    valid, `val_upper` credits every unscoreable claim as well. A comparison that the two
    bounds straddle is not a comparison (PLAN §2.3).
    """
    rows = []
    for system, g in df.groupby("system"):
        s = scored(g)
        clue = by_clue(g)
        v = clue["correct"].to_numpy(float) if len(clue) else np.array([])
        lo, hi = boot_ci(v, rng)
        clo, chi = (boot_ci_clustered(v, clue["target"].to_numpy(), rng) if len(clue)
                    else (np.nan, np.nan))
        acc = float(v.mean()) if len(v) else np.nan
        below_floor = bool(pd.notna(acc) and acc < ACC_FLOOR)
        task = g["task"].iloc[0]
        n = len(s)
        row = {
            "system": system, "family": g["family"].iloc[0], "role": g["role"].iloc[0],
            "task": task, "dataset": g["dataset"].iloc[0], "model": g["model"].iloc[0],
            "n_clues": len(clue), "n_samples": n, "k": (n / len(clue)) if len(clue) else np.nan,
            # n_errors is calls with no score; the reasoned calls are counted separately below
            # rather than folded in, because they are a condition failure, not an API failure.
            "n_errors": int(g["correct"].isna().sum()),
            "n_reasoned_excluded": int(g["reasoning_excluded"].fillna(False).sum())
                                   if "reasoning_excluded" in g else 0,
            "n_reasoning_unreported": int(g["reasoning_tokens"].isna().sum()),
            "acc": acc, "acc_ci_lo": lo, "acc_ci_hi": hi,
            "acc_ci_lo_by_answer": clo, "acc_ci_hi_by_answer": chi,
            "n_answer_clusters": int(clue["target"].nunique()) if len(clue) else 0,
            "below_acc_floor": below_floor,
            "ci_reliable": not below_floor,
            "truncation_rate": float(s["truncated"].mean()) if n else np.nan,
            "no_answer_rate": float(s["no_answer_line"].mean()) if n else np.nan,
            "wrong_length_rate": float((~s["len_ok"]).mean()) if n else np.nan,
            "mean_output_tokens": float(g["output_tokens"].mean()),
            "mean_reasoning_tokens": float(g["reasoning_tokens"].fillna(0).mean()),
        }
        if task == "nocot":
            row["extra_text_rate"] = float(s["has_extra"].mean()) if n else np.nan
            n_excl = row["n_reasoned_excluded"]
            if n_excl:
                notes_head = (f"{n_excl} call(s) reasoned (reasoning_tokens > 0) and are "
                              f"excluded from Acc: asked not to reason is not made unable to "
                              f"(probe report §7.4)")
            else:
                notes_head = None
            row["structured_output_rate"] = float(s["structured_output"].mean()) if n else np.nan
            row.update({k: np.nan for k in ("val_lower", "val_upper", "unscoreable_rate",
                                            "malformed_rate", "unexecuted_rate",
                                            "m2_lower", "m2_upper")})
            notes = [x for x in (notes_head,
                                 "no derivation in this arm; Val and M2 are not defined") if x]
            if row["n_reasoning_unreported"]:
                notes.append(f"{row['n_reasoning_unreported']} call(s) reported no "
                             f"reasoning-token count at all — an absence of evidence, not a "
                             f"measured zero; read the run's reasoning_gate")
        else:
            verdicts = s["verdict"].fillna("malformed")
            n_valid = int((verdicts == "valid").sum())
            n_unsc = int((verdicts == "unscoreable").sum())
            # A truncated call that delivered no block is not a malformed derivation: it is no
            # derivation. The model spent its whole budget thinking (mean 6,813 output tokens
            # at the gate of 2026-09-10) and the ceiling cut it off before the template. It
            # **stays in Val's denominator** — PLAN §1 says every clue in the arm, and failing
            # to answer is a failure — but it is reported as its own rate, because the gate's
            # "malformed < 5 %" criterion is about blocks that would not parse, and mixing the
            # two makes a budget artefact look like a parser problem (build-v2 report §9.1).
            truncated_no_block = (verdicts == "malformed") & _no_block(s)
            n_trunc_no_block = int(truncated_no_block.sum())
            n_mal = int(((verdicts == "malformed") & ~truncated_no_block).sum())
            correct = s[s["correct"] == 1]
            n_c = len(correct)
            cv = correct["verdict"].fillna("malformed") if n_c else pd.Series(dtype=str)
            unexec = float(s["unexecuted"].mean()) if n else np.nan
            notes = []
            if n_trunc_no_block:
                notes.append(f"{n_trunc_no_block} call(s) hit the output ceiling with no "
                             f"derivation written: counted as not-valid, reported separately "
                             f"from malformed, and a `think`-arm failure mode in its own right "
                             f"(PLAN §9.3)")
            if pd.notna(unexec) and unexec > UNEXECUTED_CEILING:
                notes.append(f"checker suspect: unexecuted rate over "
                             f"{UNEXECUTED_CEILING:.0%} (PLAN §2.3)")
            row.update({
                "val_lower": n_valid / n if n else np.nan,
                "val_upper": (n_valid + n_unsc) / n if n else np.nan,
                "unscoreable_rate": n_unsc / n if n else np.nan,
                # parse failures of a delivered block — what the gate's 5 % rule is about
                "malformed_rate": n_mal / n if n else np.nan,
                "truncated_no_block_rate": n_trunc_no_block / n if n else np.nan,
                "unexecuted_rate": unexec,
                # M2: derivation invalid **given the answer is correct**, over that system's
                # correct clues. Bounds for the same reason Val has them.
                "m2_lower": float((cv == "invalid").mean()) if n_c else np.nan,
                "m2_upper": float(cv.isin(["invalid", "unscoreable"]).mean()) if n_c else np.nan,
                "n_correct": n_c,
            })
        # PLAN §5.7's floor policy applies to every arm: the point estimate stands, the
        # interval does not, and on a `think` arm the row leaves the M2 comparisons.
        if below_floor:
            notes.append(f"below the {ACC_FLOOR:.0%} accuracy floor: point estimate only, CI "
                         f"unreliable"
                         + (", excluded from M2 comparisons" if task != "nocot" else "")
                         + " (PLAN §5.7)")
        row["notes"] = "; ".join(notes)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task", "acc"], ascending=[True, False])


# ---- 2. the headline ----------------------------------------------------------------------

def headline(df: pd.DataFrame, rng: np.random.Generator,
             reference: str = REFERENCE_FAMILY) -> pd.DataFrame:
    """PLAN §5.1. Astra's `nocot` Acc against every other model's, paired within clue.

    "Beats every other model" means every row's CI excludes zero and every `diff` is negative
    (the row is other − Astra). One comparison per model; no omnibus test, because the claim is
    a conjunction of pairwise ones.
    """
    clue = by_clue(df[df["task"] == "nocot"])
    if clue.empty:
        return pd.DataFrame([{"reference": reference, "note": "no nocot runs"}])
    wide = clue.pivot_table(index="id", columns="family", values="correct", aggfunc="mean")
    answers = clue.drop_duplicates("id").set_index("id")["target"]
    if reference not in wide.columns:
        return pd.DataFrame([{
            "reference": reference,
            "note": f"no nocot rows for the headline model {reference!r}; the headline is "
                    f"declined. Families present: {', '.join(sorted(wide.columns))}"}])
    rows = []
    for family in sorted(c for c in wide.columns if c != reference):
        both = wide[[family, reference]].dropna()
        r = paired_diff(both[family], both[reference], answers.reindex(both.index), rng)
        rows.append({"family": family, "reference": reference,
                     "acc": float(both[family].mean()),
                     "acc_reference": float(both[reference].mean()),
                     **r,
                     "reference_beats": bool(r["diff"] < 0 and r["excludes_zero"])})
    return pd.DataFrame(rows).sort_values("diff")


# ---- 3. the CoT gain ----------------------------------------------------------------------

def cot_gain(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """PLAN §5.2. `think` − `nocot` per model, paired within clue, then the ordering of gains.

    The `think` row of a family is the system whose `role` is `think` (for Sonnet 5 that is the
    `high` rung, which is why models.yaml carries `role` and `rung` separately).

    **This table is on the 50 clues of `nat50`, not the 100 of `nat`** (Nat, 2026-09-10):
    `nocot` runs on all 100 and every `think` arm on the 50, so the intersection is
    the 50 and the pairing is within clue on those. `n_paired` and `paired_on` say so on every
    row, and the `nocot` accuracy in this table is that system's accuracy **on the 50** — it is
    not the headline number, which is on all 100.
    """
    clue = by_clue(df)
    if clue.empty:
        return pd.DataFrame()
    answers = clue.drop_duplicates("id").set_index("id")["target"]
    rows = []
    for family, g in clue.groupby("family"):
        nocot = g[g["role"] == "nocot"].set_index("id")["correct"]
        think = g[g["role"] == "think"].set_index("id")["correct"]
        shared = nocot.index.intersection(think.index)
        if len(shared) == 0:
            rows.append({"family": family, "acc_nocot": np.nan, "acc_think": np.nan,
                         "n_paired": 0, "diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                         "excludes_zero": False, "paired_on": "",
                         "note": "needs both a nocot and a think run on the same clues "
                                 "(nocot on nat, think on nat50 — the 50 are a subset, so "
                                 "this means one of the two arms has not run)"})
            continue
        r = paired_diff(think.loc[shared], nocot.loc[shared], answers.reindex(shared), rng)
        datasets = sorted(set(g.loc[g["id"].isin(shared), "dataset"]))
        rows.append({"family": family, "acc_nocot": float(nocot.loc[shared].mean()),
                     "acc_think": float(think.loc[shared].mean()), **r,
                     "paired_on": "+".join(datasets),
                     "note": "acc_nocot is on the paired clues, not on all of nat"})
    out = pd.DataFrame(rows).sort_values("diff", ascending=False, na_position="last")
    # the ordering *is* the prediction: the Astra post says Astra's gain is the smallest
    out["gain_rank"] = out["diff"].rank(ascending=False, method="min")
    return out


# ---- 4. device and hops -------------------------------------------------------------------

def by_device(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """PLAN §5.4, descriptive only. The test set is charade 48, anagram 13, container 13,
    hidden 8 and six families of 6 or fewer; a 13-clue proportion carries about ±27 points, so
    this is a figure, never a within-family model comparison."""
    clue = by_clue(df)
    clue = clue[~clue["device"].astype(str).isin(["", "nan", "None", "untagged", "unknown"])]
    rows = []
    for (system, device), g in clue.groupby(["system", "device"]):
        v = g["correct"].to_numpy(float)
        lo, hi = boot_ci(v, rng)
        rows.append({"system": system, "family": g["family"].iloc[0],
                     "task": g["task"].iloc[0], "device": device, "n": len(v),
                     "acc": float(v.mean()), "ci_lo": lo, "ci_hi": hi,
                     "descriptive_only": True})
    return pd.DataFrame(rows)


def by_hops(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """PLAN §5.4's graded analysis: `nocot` Acc and the CoT gain regressed on hop count.

    Hop count is the number of wordplay components plus operations, read off the **confirmed**
    parse. Nothing in data/devices.jsonl carries it yet, so this declines with a reason rather
    than deriving a count from the parse strings — a hop count inferred by regex would be this
    repo's number, not the setter's, and the regression would be measuring the regex.
    """
    clue = by_clue(df)
    have = clue["hops"].notna().sum() if "hops" in clue else 0
    if have == 0:
        return pd.DataFrame([{"note": "declined: no clue carries a hop count. Add `hops` to "
                                      "data/devices.jsonl from the confirmed parses (PLAN §5.4)"}])
    clue = clue[clue["hops"].notna()]
    rows = []
    for family, g in clue.groupby("family"):
        nocot = g[g["role"] == "nocot"]
        if nocot.empty:
            continue
        x = nocot["hops"].to_numpy(float)
        y = nocot["correct"].to_numpy(float)
        slope, intercept = (np.polyfit(x, y, 1) if len(set(x)) > 1 else (np.nan, np.nan))
        # the interval is a bootstrap over answer clusters, like every other interval here
        boots = []
        answers = nocot["target"].to_numpy()
        keys = list(dict.fromkeys(answers))
        index = {k: np.where(answers == k)[0] for k in keys}
        for _ in range(1000):
            draw = np.concatenate([index[keys[i]] for i in
                                   rng.integers(0, len(keys), size=len(keys))])
            if len(set(x[draw])) > 1:
                boots.append(np.polyfit(x[draw], y[draw], 1)[0])
        lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) \
            if boots else (np.nan, np.nan)
        rows.append({"family": family, "n": len(x), "acc_slope_per_hop": float(slope),
                     "intercept": float(intercept), "slope_ci_lo": lo, "slope_ci_hi": hi,
                     "note": "nocot Acc regressed on hop count, bootstrap over answers"})
    return pd.DataFrame(rows)


# ---- 5. the Sonnet 5 rungs ---------------------------------------------------------------

def sonnet_rungs(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """PLAN §5.5. Acc and Val against the ordinal rung and against measured thinking tokens;
    the difference of slopes (accuracy rising faster than validity) is the one test on this row.

    `off + prose CoT` is **not** a point on the effort axis — it is reasoning in the output
    rather than in the budget — so it appears as a row with `rung_index` NaN and is excluded
    from both slopes (cot-design-2026-09-09.md §0).
    """
    g = df[df["rung"].notna() & (df["rung"] != "")]
    if g.empty:
        return pd.DataFrame([{"note": "declined: no run carries a `rung` (see models.yaml)"}])
    rows = []
    for rung, h in g.groupby("rung"):
        s = scored(h)
        clue = by_clue(h)
        n = len(s)
        verdicts = s["verdict"].fillna("malformed")
        rows.append({
            "rung": rung, "rung_index": RUNG_ORDER.get(str(rung), np.nan),
            "system": h["system"].iloc[0], "n_clues": len(clue), "n_samples": n,
            "acc": float(clue["correct"].mean()) if len(clue) else np.nan,
            "val_lower": float((verdicts == "valid").mean()) if n else np.nan,
            "val_upper": float(verdicts.isin(["valid", "unscoreable"]).mean()) if n else np.nan,
            "mean_reasoning_tokens": float(h["reasoning_tokens"].fillna(0).mean()),
            "mean_output_tokens": float(h["output_tokens"].mean()),
            "on_effort_axis": str(rung) in RUNG_ORDER,
        })
    out = pd.DataFrame(rows).sort_values("rung_index", na_position="last")
    axis = out[out["on_effort_axis"]]
    if len(axis) >= 2:
        acc_slope = float(np.polyfit(axis["rung_index"], axis["acc"], 1)[0])
        val_slope = float(np.polyfit(axis["rung_index"], axis["val_lower"], 1)[0])
        out.attrs["acc_slope_per_rung"] = acc_slope
        out.attrs["val_slope_per_rung"] = val_slope
        out.attrs["difference_of_slopes"] = acc_slope - val_slope
    return out


# ---- 6. the cutoff split ------------------------------------------------------------------

def cutoff_split(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """PLAN §5.6 / §2.6. The headline again on the clues no model could have seen.

    "Could have seen" is **published before that system's training cutoff**, not "published at
    all": every clue here was published after most of these models were trained, so splitting
    on publication status would compare two post-cutoff groups and call the gap contamination.
    The clean subset for the cross-model claim is the clues published after the *newest* cutoff,
    or never published.

    Declines, loudly, while any `training_cutoff` is null or no clue carries a
    `published_date`. Both are Nat's to fill (PLAN §7.1) and neither can be guessed.
    """
    clue = by_clue(df[df["task"] == "nocot"])
    if clue.empty:
        return pd.DataFrame([{"note": "declined: no nocot runs"}])
    systems = load_systems()
    families = sorted(clue["family"].unique())
    cutoffs = {}
    for family in families:
        vals = [(systems.get(k) or {}).get("training_cutoff")
                for k, v in systems.items() if (v or {}).get("family") == family]
        cutoffs[family] = next((v for v in vals if v), None)
    missing = [f for f, c in cutoffs.items() if not c]
    dates = pd.to_datetime(clue["published_date"], errors="coerce")
    if missing:
        return pd.DataFrame([{"note": "declined: no training_cutoff for "
                                      + ", ".join(missing)
                                      + ". Take each from the vendor's own page (PLAN §2.6); "
                                        "do not fill from memory."}])
    if dates.notna().sum() == 0:
        return pd.DataFrame([{"note": "declined: no clue carries a published_date, so the "
                                      "post-cutoff subset cannot be formed (PLAN §7.1)"}])
    newest = max(pd.Timestamp(c) for c in cutoffs.values())
    clue = clue.assign(published=dates)
    clean = clue[clue["published"].isna() | (clue["published"] > newest)]
    ids = set(clean["id"])
    sub = clean.pivot_table(index="id", columns="family", values="correct", aggfunc="mean")
    answers = clean.drop_duplicates("id").set_index("id")["target"]
    rows = []
    for family in sorted(c for c in sub.columns if c != REFERENCE_FAMILY):
        if REFERENCE_FAMILY not in sub.columns:
            break
        both = sub[[family, REFERENCE_FAMILY]].dropna()
        r = paired_diff(both[family], both[REFERENCE_FAMILY], answers.reindex(both.index), rng)
        rows.append({"family": family, "reference": REFERENCE_FAMILY,
                     "newest_cutoff": str(newest.date()), "n_clean_clues": len(ids), **r})
    if not rows:
        return pd.DataFrame([{"note": f"declined: no rows for {REFERENCE_FAMILY} on the "
                                      f"post-cutoff subset", "n_clean_clues": len(ids)}])
    out = pd.DataFrame(rows)
    if len(ids) < 20:
        out["small_subset_warning"] = (f"only {len(ids)} clues are after the newest cutoff or "
                                       f"never published — say so, do not pad it (PLAN §2.6)")
    return out


# ---- 7. flip rate -------------------------------------------------------------------------

def flip_rate(df: pd.DataFrame) -> pd.DataFrame:
    """PLAN §5 / §4. Per-clue disagreement across the k repeats: how much noise the paired
    k = 1 comparisons carry. A clue with every repeat agreeing contributes 0."""
    s = scored(df)
    if s.empty:
        return pd.DataFrame()
    rows = []
    for system, g in s.groupby("system"):
        per_clue = g.groupby("id")["correct"].agg(["mean", "size"])
        repeated = per_clue[per_clue["size"] > 1]
        if repeated.empty:
            rows.append({"system": system, "n_repeated_clues": 0, "k": int(per_clue["size"].max()),
                         "flip_rate": np.nan,
                         "note": "k = 1: no repeats, so no flip rate"})
            continue
        # a clue "flips" when its k samples are not unanimous
        flips = ((repeated["mean"] > 0) & (repeated["mean"] < 1)).mean()
        rows.append({"system": system, "n_repeated_clues": len(repeated),
                     "k": int(repeated["size"].max()), "flip_rate": float(flips), "note": ""})
    return pd.DataFrame(rows)


# ---- the calibration line ------------------------------------------------------------------

def guardian_line(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """`nocot` on guardian200 against Saha et al.'s 18.5 %, item for item (PLAN §2.5).

    Dropped on 2026-09-10 at k = 8, **restored at k = 1 on 2026-09-11** when the study pivoted
    to no-CoT: external calibration against a published number is what a hype-adjacent
    accuracy claim most needs, and one repeat is enough for the comparison. So the table
    has harness rows again and the comparison is live — at k = 1, which is worth stating beside
    it, because every `nat` figure is a mean of four.

    The note below still fires if the split is ever absent: a lone published row is not a
    comparison, and Saha et al. would be prior work cited rather than a number this study
    matched.
    """
    clue = by_clue(df[(df["task"] == "nocot") & (df["dataset"] == "guardian200")])
    rows = []
    for system, g in clue.groupby("system"):
        v = g["correct"].to_numpy(float)
        lo, hi = boot_ci(v, rng)
        rows.append({"system": system, "n": len(v), "acc": float(v.mean()),
                     "ci_lo": lo, "ci_hi": hi, "source": "this harness"})
    for name, acc in SAHA_GUARDIAN.items():
        rows.append({"system": name, "n": 200, "acc": acc, "ci_lo": np.nan, "ci_hi": np.nan,
                     "source": "published"})
    out = pd.DataFrame(rows)
    if not (out["source"] == "this harness").any():
        out["note"] = ("guardian200 not run. "
                       "No harness rows, so Saha et al. is cited as prior work and "
                       "NOT compared item-for-item — do not present this row as a benchmark "
                       "this study was measured against.")
    return out


# ---- entry point ---------------------------------------------------------------------------

REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "review" / "hand_review_2026-09-11.csv"


def reviewed_validity(df: pd.DataFrame, review: pd.DataFrame | None = None) -> pd.DataFrame:
    """Val and M2 after the pre-registered hand review of the `unexecuted` bucket (§2.3).

    The review (data/review/) classifies every right-but-invalid or `unexecuted` derivation:
    **A** correct parse the checker cannot express, **B** correct parse with misused syntax,
    **C** wrong parse. Only A is re-scored valid. B and C stay invalid: the checker was right.
    The mechanical columns are repeated beside the reviewed ones so the table reads on its own;
    the mechanical numbers stay primary and nothing in `per_system` moves.
    """
    if review is None:
        if not REVIEW_PATH.exists():
            return pd.DataFrame([{"note": f"declined: no review file at {REVIEW_PATH.name}"}])
        review = pd.read_csv(REVIEW_PATH, dtype=str)
    review = review.assign(**{"class": review["class"].str.strip().str.upper()})
    s = scored(df[df["task"].isin(["think", "think_cot"])])
    if s.empty:
        return pd.DataFrame([{"note": "declined: no think-arm rows"}])
    keyed = s.merge(review[["id", "system", "class"]], on=["id", "system"], how="left")
    rows = []
    for system, g in keyed.groupby("system"):
        verdicts = g["verdict"].fillna("malformed")
        n = len(g)
        is_a = g["class"].eq("A")
        n_valid = int((verdicts == "valid").sum())
        n_a = int(is_a.sum())
        n_unsc = int((verdicts == "unscoreable").sum())
        correct = g[g["correct"] == 1]
        n_c = len(correct)
        cv = correct["verdict"].fillna("malformed")
        m2_mech = float((cv == "invalid").mean()) if n_c else np.nan
        m2_rev = float(((cv == "invalid") & ~correct["class"].eq("A")).mean()) if n_c else np.nan
        rows.append({
            "system": system, "n": n, "n_reviewed": int(g["class"].notna().sum()),
            "n_class_a": n_a, "n_class_b": int(g["class"].eq("B").sum()),
            "n_class_c": int(g["class"].eq("C").sum()),
            "val_lower_mechanical": n_valid / n if n else np.nan,
            "val_lower_reviewed": (n_valid + n_a) / n if n else np.nan,
            "val_upper_mechanical": (n_valid + n_unsc) / n if n else np.nan,
            "val_upper_reviewed": (n_valid + n_a + n_unsc) / n if n else np.nan,
            "n_correct": n_c,
            "m2_lower_mechanical": m2_mech, "m2_lower_reviewed": m2_rev,
            "note": "class A re-scored valid; B and C stay invalid (hand review 2026-09-11)",
        })
    return pd.DataFrame(rows).sort_values("val_lower_reviewed", ascending=False)


ADJUDICATION_PATH = REVIEW_PATH.parent / "adjudication_2026-09-11.csv"


def adjudicated_accuracy(df: pd.DataFrame, rulings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Acc after the blind adjudication of distinct answer strings (PLAN §2.4).

    `rulings` has one row per (id, string) with `accept` Y/N; a string not ruled scores as exact
    match does. A sample is adjudicated-correct if exact match says so or its extracted string
    (A–Z only) is an accepted string for that clue. Mean of k per clue, as everywhere. Exact
    match stays primary; this is the secondary number the plan asked for.
    """
    if rulings is None:
        if not ADJUDICATION_PATH.exists():
            return pd.DataFrame([{"note": f"declined: no rulings at {ADJUDICATION_PATH.name}"}])
        rulings = pd.read_csv(ADJUDICATION_PATH, dtype=str)
    yes = rulings["accept"].str.upper() == "Y"
    accepted = set(zip(rulings.loc[yes, "id"], rulings.loc[yes, "string"]))
    ruled = set(zip(rulings["id"], rulings["string"]))
    s = scored(df[df["dataset"].isin(NAT_FAMILY)]).copy()
    if s.empty:
        return pd.DataFrame([{"note": "declined: no nat rows"}])
    key = s["extracted"].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    s["adj_correct"] = ((s["correct"] == 1)
                        | pd.Series([(i, k) in accepted for i, k in zip(s["id"], key)],
                                    index=s.index)).astype(float)
    s["ruled"] = [(i, k) in ruled for i, k in zip(s["id"], key)]
    rows = []
    for system, g in s.groupby("system"):
        clue = g.groupby("id").agg(exact=("correct", "mean"), adj=("adj_correct", "mean"))
        wrong = g[g["correct"] == 0]
        rows.append({"system": system, "n_clues": len(clue),
                     "acc_exact": float(clue["exact"].mean()),
                     "acc_adjudicated": float(clue["adj"].mean()),
                     "n_wrong_samples": int(len(wrong)),
                     "n_wrong_ruled": int(wrong["ruled"].sum()),
                     "n_wrong_accepted": int((wrong["adj_correct"] == 1).sum()),
                     "n_accepted_strings": len(accepted)})
    return pd.DataFrame(rows).sort_values("acc_adjudicated", ascending=False)


TABLES = ("per_system", "headline", "cot_gain", "by_device", "by_hops", "sonnet_rungs",
          "cutoff_split", "flip_rate", "guardian_line", "reviewed_validity",
          "adjudicated_accuracy")


def all_tables(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    return {
        "per_system": per_system(df, rng),
        "headline": headline(df, rng),
        "cot_gain": cot_gain(df, rng),
        "by_device": by_device(df, rng),
        "by_hops": by_hops(df, rng),
        "sonnet_rungs": sonnet_rungs(df, rng),
        "cutoff_split": cutoff_split(df, rng),
        "flip_rate": flip_rate(df),
        "guardian_line": guardian_line(df, rng),
        "reviewed_validity": reviewed_validity(df),
        "adjudicated_accuracy": adjudicated_accuracy(df),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--out-dir")
    ap.add_argument("--dataset", default="nat",
                    help="which split the headline tables are computed on; `nat` includes "
                         "`nat50`, which is a subset of it, and no other pair is ever pooled "
                         "(PLAN §2.5)")
    ap.add_argument("--all", action="store_true",
                    help="turn run selection off: include superseded prompts and configs, cut "
                         "systems and mock runs. For archaeology, not for a number.")
    a = ap.parse_args()
    results = Path(a.results_dir)
    out = Path(a.out_dir) if a.out_dir else results
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = load_runs(results, keep_all=a.all)
    # `nat` means the nat family: `nocot` on all 100 and the `think` arms on the 50 of `nat50`
    # are one study, and cot_gain() pairs them by clue id.
    wanted = NAT_FAMILY if a.dataset == "nat" else (a.dataset,)
    split = df[df["dataset"].isin(wanted)]
    if split.empty:
        sys.exit(f"no runs on dataset {a.dataset!r}; found "
                 f"{', '.join(sorted(df['dataset'].unique()))}")
    out_tables = all_tables(split, rng)
    out_tables["guardian_line"] = guardian_line(df, rng)   # its own split, by definition
    for name, table in out_tables.items():
        table.to_csv(tables / f"{name}.csv", index=False)
    ps = out_tables["per_system"]
    cols = [c for c in ("system", "task", "n_clues", "k", "n_errors",
                        "n_reasoned_excluded", "n_reasoning_unreported",
                        "truncated_no_block_rate", "acc",
                        "acc_ci_lo_by_answer", "acc_ci_hi_by_answer", "ci_reliable",
                        "val_lower", "val_upper", "m2_lower", "m2_upper", "extra_text_rate",
                        "mean_reasoning_tokens") if c in ps]
    print(ps[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for name in ("headline", "cot_gain", "cutoff_split", "by_hops", "reviewed_validity",
                 "adjudicated_accuracy"):
        t = out_tables[name]
        if len(t):
            print(f"\n{name}:")
            print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {len(out_tables)} tables to {tables}")


if __name__ == "__main__":
    main()
