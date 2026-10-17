#!/usr/bin/env python
"""Fabricate run CSVs + manifests in the current schema, so analysis/ can be exercised without
calling any model.

    python tools/make_fake_runs.py /tmp/fake_results

It emits every shape the tables need: `nocot` with k repeats on `nat` (so the flip rate and
the mean-of-k aggregation are exercised), `think` for the same families **on `nat50`** (so the
CoT gain pairs across the two datasets on the 50, as the real run does), the Sonnet 5 rungs (so
the difference of slopes has an axis), `nocot` on `guardian200` (the Saha line), a family below
the 5 % accuracy floor, and derivation verdicts in all four classes
including `unscoreable` — the endogenous set that makes Val an interval. On `astra-nocot` a
few per cent of calls carry `reasoning_tokens > 0`, because that is the one system that is
*asked* not to reason rather than made unable to (prompt v2): those rows exercise the per-call
exclusion in analysis/metrics.py, which is invisible on data where the count is always zero.

It also emits three runs that `load_runs()` must **set aside**: a cut system, a mock system,
and two superseded configurations of `sonnet5-nocot` (one with the old prompt hash, one with an
old fingerprint). Every manifest therefore carries `prompt_sha256` and `config_fingerprint`,
taken from what each system would send today — a fake run with no fingerprint would be dropped
for the wrong reason and the selection tests would pass vacuously.

The numbers are drawn from a seeded RNG and mean nothing. What they exercise is shape:
denominators, bounds, the floor policy, the paired headline, the cutoff refusal and the
selection.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from analysis.metrics import current_selectors  # noqa: E402
from crypticbench.config import CORPUS_PATH, DEVICES_PATH, NAT50_PATH  # noqa: E402
from crypticbench.corpus import read_id_list, read_jsonl  # noqa: E402

FIELDS = ["run_id", "system", "task", "dataset", "id", "target", "score", "correct",
          "extracted", "extra_text", "extra_words", "has_extra", "structured_output",
          "verdict", "l0", "l1", "l2a", "l3", "unexecuted", "device_claimed", "sourcing",
          "unverified_ops", "check_reasons", "no_answer_line", "len_ok", "stop_reason",
          "truncated", "error", "input_tokens", "cache_read", "output_tokens",
          "reasoning_tokens", "reasoning_tokens_source", "completion", "thinking",
          "device", "hops", "published", "published_date", "source"]

# (system, task, dataset, p(correct), k). Astra is the reference family, so it must be present
# or headline() declines; `gpt55` sits under the 5 % floor on purpose. `nocot` is on `nat` and
# every `think` arm on `nat50`, which is the real shape (Nat, 2026-09-10) and the reason
# cot_gain() has to pair across two datasets. Astra has no `think` row because `astra-think` is
# cut, so its cot_gain row exercises the "needs both arms" branch.
PLAN = [
    ("astra-nocot",   "nocot",     "nat",   0.55, 3),
    ("opus5-nocot",   "nocot",     "nat",   0.35, 3),
    ("opus5-think",   "think",     "nat50", 0.55, 1),
    ("gpt55-nocot",   "nocot",     "nat",   0.03, 3),
    ("gpt55-think",   "think",     "nat50", 0.15, 1),
    ("sonnet5-nocot", "nocot",     "nat",   0.30, 3),
    ("sonnet5-off",   "think",     "nat50", 0.32, 1),
    ("sonnet5-low",   "think",     "nat50", 0.38, 1),
    ("sonnet5-high",  "think",     "nat50", 0.50, 1),
    ("sonnet5-cot",   "think_cot", "nat50", 0.40, 1),
]
# Runs that must be set aside, and the reason each one is: a cut system, a mock system, and a
# `sonnet5-nocot` under the previous prompt and under a previous generate block.
SET_ASIDE = [
    ("sonnet5-medium", "think", "nat50", 0.44, 1, {}),   # the rung that was cut
    ("mock",          "nocot", "nat", 0.20, 1, {}),
    ("sonnet5-nocot", "nocot", "nat", 0.30, 1, {"prompt_sha256": "0" * 64,
                                                "prompt_version": "v1"}),
    ("sonnet5-nocot", "nocot", "nat", 0.30, 1, {"config_fingerprint": "beefbeefbeefbeef"}),
]
VERDICTS = ["valid", "invalid", "unscoreable", "malformed"]


def make_rows(rows, system, task, dataset, p, k, rng, devices, run_id):
    recs = []
    for repeat in range(k):
        for r in rows:
            ok = rng.random() < p
            trunc = rng.random() < 0.03
            wrong = r["answer"][::-1] if rng.random() < 0.6 else r["answer"][:-1] + "XY"
            missing = rng.random() < 0.04
            ext = r["answer"] if ok else (None if missing else wrong)
            tag = devices.get(r["id"], {})
            rec = {f: "" for f in FIELDS}
            rec.update({
                "run_id": run_id, "system": system, "task": task, "dataset": dataset,
                "id": r["id"], "target": r["answer"], "score": "C" if ok else "I",
                "correct": int(ok), "extracted": ext or "",
                "no_answer_line": ext is None,
                "len_ok": ext is not None and len(ext) == len(r["answer"]),
                "stop_reason": "max_tokens" if trunc else "stop", "truncated": trunc,
                "error": "", "cache_read": 0,
                "device": tag.get("device", ""), "hops": tag.get("hops", ""),
                "published": r.get("source") == "published",
                "published_date": r.get("published_date") or "",
                "source": r.get("source", "other"),
            })
            if task == "nocot":
                # ~6 % of outputs carry something beyond the answer, so the 5 % rule fires
                extra = "  (I think)" if rng.random() < 0.06 else ""
                # Astra is asked not to reason, not made unable to, so a few calls reason
                # anyway and the count is reported per call. Every other nocot system has
                # reasoning off by parameter and reports nothing.
                reasoned = system == "astra-nocot" and rng.random() < 0.04
                rec.update({
                    "extra_text": extra, "extra_words": 0, "has_extra": bool(extra),
                    "structured_output": False,
                    "input_tokens": 350, "output_tokens": rng.randint(3, 8),
                    "reasoning_tokens": rng.randint(1, 24) if reasoned else
                                        (0 if system == "astra-nocot" else ""),
                    "reasoning_tokens_source":
                        "ModelOutput.usage.reasoning_tokens" if system == "astra-nocot" else
                        "ModelOutput.usage.reasoning_tokens absent (no reasoning reported)",
                    "completion": (ext or "") + extra, "thinking": "",
                })
            else:
                verdict = (rng.choices(VERDICTS, weights=[6, 3, 2, 1])[0] if ok
                           else rng.choices(VERDICTS, weights=[1, 6, 2, 2])[0])
                rec.update({
                    "verdict": verdict, "l0": verdict != "malformed",
                    "l1": verdict in ("valid", "unscoreable"),
                    "l2a": True if verdict == "valid" else ("" if verdict == "unscoreable"
                                                            else False),
                    "l3": ok, "unexecuted": verdict == "invalid" and ok and rng.random() < 0.3,
                    "device_claimed": tag.get("device", "charade"),
                    "sourcing": json.dumps({"literal": 1}), "unverified_ops": 0,
                    "check_reasons": "" if verdict == "valid" else "fabricated",
                    "input_tokens": 900, "output_tokens": rng.randint(200, 3200),
                    "reasoning_tokens": rng.randint(200, 3000),
                    "reasoning_tokens_source": "ModelOutput.usage.reasoning_tokens",
                    "completion": f"DEFINITION: x\nANSWER: {ext or ''}",
                    "thinking": "summarised thinking",
                })
            recs.append((repeat + 1, rec))
    return recs


def write_run(out: Path, run_id: str, system: str, task: str, dataset: str, recs: list[dict],
              repeat_index: int, override: dict | None = None) -> None:
    (out / "runs").mkdir(parents=True, exist_ok=True)
    with (out / "runs" / f"{run_id}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(recs)
    sel = SELECTORS.get(system, {})
    manifest = {
        "run_id": run_id, "system": system, "task": task, "dataset": dataset,
        "repeat_index": repeat_index, "model": f"fake/{system}", "batched": False,
        "dataset_sha256": "fake", "n": len(recs),
        # what this system would send today, so load_runs() keeps it
        "prompt_version": sel.get("prompt_version"),
        "prompt_sha256": sel.get("prompt_sha256"),
        "config_fingerprint": min(sel.get("fingerprints") or {"fake"}),
    }
    manifest.update(override or {})
    (out / "runs" / f"{run_id}.json").write_text(json.dumps(manifest, indent=1),
                                                 encoding="utf-8")


SELECTORS = current_selectors()


def main(out: Path) -> None:
    corpus = [r for r in read_jsonl(CORPUS_PATH) if r["split"] == "test"]
    nat50_ids = set(read_id_list(NAT50_PATH))
    rows_for = {"nat": corpus, "nat50": [r for r in corpus if r["id"] in nat50_ids]}
    devices = {d["id"]: d for d in read_jsonl(DEVICES_PATH)}
    rng = random.Random(1)
    for i, (system, task, dataset, p, k, *rest) in enumerate(
            [(*x, None) for x in PLAN] + [(*x[:5], x[5]) for x in SET_ASIDE]):
        override = rest[0]
        for repeat, recs in _grouped(make_rows(rows_for[dataset], system, task, dataset, p, k,
                                               rng, devices, "")):
            tail = "_superseded" if override else ""
            run_id = f"{task}_{dataset}_r{repeat}_{system}{tail}{i if override else ''}_fake"
            for rec in recs:
                rec["run_id"] = run_id
            write_run(out, run_id, system, task, dataset, recs, repeat, override)
    # one guardian200 arm, so the Saha line has a row on its own split
    guardian = [{"id": f"gd-{i:04d}", "answer": f"ANSWER{i:03d}", "source": "saha2025"}
                for i in range(60)]
    recs = [rec for _, rec in make_rows(guardian, "astra-nocot", "nocot", "guardian200",
                                        0.30, 1, rng, {}, "nocot_guardian200_astra-nocot_fake")]
    write_run(out, "nocot_guardian200_astra-nocot_fake", "astra-nocot", "nocot", "guardian200",
              recs, 1)
    n = len(list((out / "runs").glob("*.csv")))
    print(f"fake runs in {out}: {n} run files")


def _grouped(pairs):
    by_repeat: dict[int, list[dict]] = {}
    for repeat, rec in pairs:
        by_repeat.setdefault(repeat, []).append(rec)
    return sorted(by_repeat.items())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
