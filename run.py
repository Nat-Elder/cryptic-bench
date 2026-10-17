#!/usr/bin/env python
"""Run one arm on one or more systems.

    python run.py --task nocot --systems mock --dataset nat --dry-run
    python run.py --task nocot --systems haiku45-nocot,sonnet5-nocot --limit 5 --sync
    python run.py --task nocot --systems sonnet5-nocot --dataset nat \
        --ids-from data/nat50.txt --limit 20 --sync
    python run.py --task think --systems sonnet5-high --dataset nat

Three arms (PLAN §2.1): `nocot`, `think`, `think_cot`. Splits (PLAN §2.5): `nat`,
`guardian200`, `times200`, reported separately and never pooled — plus **`nat50`**, which is
not a fourth split but 50 of `nat`'s own 100 clues, drawn once into `data/nat50.txt`.
`nocot` runs on `nat`; every `think` arm runs on `nat50`.

All traffic goes through OpenRouter: every live system is an `openrouter/...` slug. A system
may also name a ':batch' variant of its id, used unless `--sync` is passed (`resolve_route`).

Every run writes results/runs/<run_id>.csv (one row per clue, full completion included),
results/runs/<run_id>.json (manifest), and appends to results/answers.jsonl. It refuses to
start if the system has no model id or if the system's `arm` is not the task asked for.

After a `nocot` run it prints two checks that decide whether the number is a no-CoT number at
all: the reasoning-token gate (PLAN §7.4) and the 5 % extra-text rule (PLAN §2.1).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crypticbench import config as C  # noqa: E402
from crypticbench.corpus import read_id_list, read_jsonl, validate  # noqa: E402
from crypticbench.prompts import load_prompt  # noqa: E402
from crypticbench.usage import (  # noqa: E402
    expects_reasoning,
    reasoning_gate,
    sample_usage,
    summarise_usage,
)
from tasks.nocot import extra_text_summary  # noqa: E402
from tasks.think import CONDITIONS as THINK_CONDITIONS  # noqa: E402

TASKS = ("nocot", *THINK_CONDITIONS)
# Both `think` arms default to **v2** as well since 2026-09-10 night: v1 plus `initial` and
# `final` in the FODDER op list, the vocabulary the gate showed was missing (gate report §6.1).
# Verified before adoption — Sol used both ops unprompted on 4 of 20 clues, `[literal:initial]`
# and `[literal:final]`, at unchanged validity. **The gate's own `think` rows were taken under
# v1**, so run selection now sets them aside: the gate validated the instrument and the full
# run supplies the numbers.
#
# The `nocot` arm's default prompt is **v2**: v1 plus the line asking for no reasoning tokens,
# which is what makes Astra's row exist at all (probe report §7.4) and is inert on the systems
# whose reasoning is off by parameter. One prompt for every system within an arm, so the line
# is not Astra-only. v1 stays on disk and is still selectable with `--prompt-version v1`; the
# hash of whichever file is used goes into the manifest either way.
DEFAULT_PROMPT_VERSION = {"nocot": "v2", "think": "v2", "think_cot": "v2"}
SCORER_NAMES = {"nocot": "nocot_answer", "think": "derivation_check",
                "think_cot": "derivation_check"}

# Printed once per invocation rather than once per call: inspect_ai warns per request when a
# model drops `temperature`, and the reason it is absent is a property of the model family, not
# of any one call. Claude 4.7+ are always adaptive and reject it (PLAN §3, §8).
TEMPERATURE_NOTE = (
    "note: no system sets `temperature`. Claude 4.7+ reject it (always adaptive) and GPT-5.5 "
    "ignores it with reasoning on, so every k > 1 run is default sampling and the frozen "
    "'k = 1 at temperature 0' ruling is void on these families."
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_systems() -> dict:
    return yaml.safe_load(C.MODELS_YAML.read_text(encoding="utf-8"))


def generate_fields(cfg: dict, *, sync: bool, structured: bool) -> tuple[dict, bool]:
    """(GenerateConfig fields, whether the structured-output fallback is on).

    `structured` is not a GenerateConfig field: it is this repo's flag for the 5 % rule's
    fallback, and the schema itself is attached by tasks/nocot.py.

    `generate.batch` is inert on the OpenRouter route (the provider passes `batcher=None`), so
    no system sets it any more; the ':batch' *model id* is handled by `resolve_route`. The pop
    is kept for a direct-vendor entry, should one ever come back.
    """
    gen = dict(cfg.get("generate", {}))
    use_structured = bool(gen.pop("structured", False)) or structured
    if sync:
        # the probe and the gate are sequential decisions and must not wait on a batch
        gen.pop("batch", None)
    return gen, use_structured


def resolve_route(cfg: dict, *, sync: bool) -> tuple[str | None, bool]:
    """(model id, batched) for this invocation, after applying `--sync`.

    A system with `batch: true` names a ':batch' variant of its OpenRouter id in `model_batch`,
    used unless `--sync` is passed. The resolved id is part of the config fingerprint, so the
    two are kept apart by `--skip-done`.
    """
    if bool(cfg.get("batch")) and not sync:
        return cfg.get("model_batch"), True
    return cfg.get("model"), False


def config_fingerprint(model: str, model_args: dict, gen_fields: dict, structured: bool) -> str:
    """Stable hash of everything that changes what a call asks for."""
    payload = json.dumps({"model": model, "model_args": model_args or {},
                          "generate": gen_fields, "structured": structured}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def done_ids(system: str, task: str, dataset: str, prompt_sha: str, fingerprint: str) -> set[str]:
    """Ids already **scored** for this system under this exact configuration.

    Keying on the id alone let a resumed run mix two configurations inside one CSV whenever the
    prompt, models.yaml or a generate field changed between invocations. A run whose prompt
    hash or config fingerprint differs is a different measurement, so its ids do not count.
    """
    ids: set[str] = set()
    if not C.RUNS_DIR.exists():
        return ids
    for m in C.RUNS_DIR.glob("*.json"):
        man = json.loads(m.read_text(encoding="utf-8"))
        if not (man.get("system") == system and man.get("task") == task
                and man.get("dataset") == dataset):
            continue
        if man.get("prompt_sha256") != prompt_sha:
            continue
        if man.get("config_fingerprint") != fingerprint:
            continue
        csv_path = m.with_suffix(".csv")
        if csv_path.exists():
            with csv_path.open(encoding="utf-8") as f:
                # An errored row is not done work. A run that dies partway writes its whole
                # clue list to the CSV with the failures recorded, and counting those as done
                # meant `--skip-done` never retried them: the 402'd astra-think run of
                # 2026-09-10 put 20 errored rows on disk and the retry skipped straight past
                # them to the next 20 clues, dropping those 20 from the arm with no warning.
                ids.update(r["id"] for r in csv.DictReader(f)
                           if not str(r.get("error") or "").strip())
    return ids


def load_rows(dataset: str) -> tuple[list[dict], list[dict]]:
    """(rows to score, dev rows for the few-shot block).

    The dev rows always come from the corpus: the same four worked examples are shown on every
    split, so a `guardian200` number and a `nat100` number differ in their clues and in nothing
    else. Device tags and hop counts are joined in from data/devices.jsonl by id.

    `nat50` is the same corpus rows as `nat`, filtered and **ordered** by `data/nat50.txt` —
    the drawn membership, in seed order, so `--limit 20` is the gate's 20 and the full run
    reuses them. The membership is never re-drawn here; `tools/draw_nat50.py` did it once.
    """
    corpus = read_jsonl(C.CORPUS_PATH)
    errors = validate(corpus)
    if errors:
        sys.exit("corpus invalid:\n" + "\n".join(errors))
    dev = [r for r in corpus if r["split"] == "dev"]
    tags = {t["id"]: t for t in read_jsonl(C.DEVICES_PATH)}

    def tag(rows: list[dict]) -> list[dict]:
        for r in rows:
            t = tags.get(r["id"], {})
            r.setdefault("device", t.get("device"))
            # PLAN §5.4's graded analysis. Absent for now — the confirmed parses are Nat's, and
            # analysis/metrics.py declines the hop regression rather than inventing a count.
            r.setdefault("hops", t.get("hops"))
            r.setdefault("device_confirmed", t.get("confirmed"))
        return rows

    test = [r for r in corpus if r["split"] == "test"]
    if dataset == "nat":
        return tag(test), dev
    if dataset == "nat50":
        ids = read_id_list(C.NAT50_PATH)
        by_id = {r["id"]: r for r in test}
        missing = [i for i in ids if i not in by_id]
        if missing:
            sys.exit(f"{C.NAT50_PATH.name} names {len(missing)} ids that are not `test` rows "
                     f"of the corpus: {missing[:5]}")
        if len(ids) != C.NAT50_SIZE or len(set(ids)) != len(ids):
            sys.exit(f"{C.NAT50_PATH.name} has {len(ids)} ids ({len(set(ids))} distinct); "
                     f"nat50 is exactly {C.NAT50_SIZE}. Re-draw it deliberately or fix the "
                     f"file — do not run a split of unknown size.")
        return tag([by_id[i] for i in ids]), dev
    if dataset in ("guardian200", "times200"):
        path = C.DATASET_PATHS[dataset]
        if not path.exists():
            script = "fetch_calibration.py" if dataset == "guardian200" else "build_times200.py"
            sys.exit(f"{path} is not in the repository (the clues are the newspapers' setters'). "
                     f"Build it with `uv run python tools/{script}`; it downloads from the "
                     f"original project and reproduces the file the runs were hashed against.")
        return tag(read_jsonl(path)), dev
    sys.exit(f"unknown dataset {dataset}")


def reasoning_text(message) -> str:
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return ""
    parts = []
    for c in content:
        if getattr(c, "type", None) == "reasoning":
            parts.append(getattr(c, "reasoning", "") or getattr(c, "summary", "") or "")
    return "\n".join(p for p in parts if p)


def check_system(key: str, cfg: dict, task: str, *, sync: bool = False) -> None:
    """Refuse a system that cannot be addressed or is the wrong arm.

    The id is the *resolved* one, so a system with a ':batch' id missing is refused without
    `--sync` and still allowed with it, rather than silently falling back to the other id.
    """
    model, batched = resolve_route(cfg, sync=sync)
    which = "':batch' " if batched else ""
    if model is None:
        sys.exit(f"[{key}] REFUSED: no {which}model id in configs/models.yaml — "
                 f"fill it from the vendor page")
    arm = cfg.get("arm")
    if arm is not None and arm != task:
        sys.exit(f"[{key}] REFUSED: this system is the {arm!r} arm, not {task!r}. Its "
                 f"generate config (max_tokens, reasoning) belongs to {arm!r}; running it as "
                 f"{task!r} would measure something the tables have no row for.")


def cut_note(cfg: dict) -> str | None:
    """The `cut` string, if this system was cut from the run. Entries stay in models.yaml for
    the record (PLAN §3, §8) and this is what stops one being run by a stale command line."""
    cut = cfg.get("cut")
    return str(cut) if cut else None


def check_cut(key: str, cfg: dict, *, include_cut: bool, dry_run: bool) -> bool:
    """True = skip this system. Refuses a cut system, quoting the string, unless
    `--include-cut`; under `--dry-run` it reports the cut instead of exiting, so the matrix
    still prints and says which rows are not in the run."""
    cut = cut_note(cfg)
    if cut is None or include_cut:
        return False
    if dry_run:
        print(f"[{key}] CUT: {cut} — not run; pass --include-cut to run it anyway")
        return True
    sys.exit(f"[{key}] REFUSED: cut — {cut}. This system is not in the run; its entry stays "
             f"in configs/models.yaml for the record. Pass --include-cut to run it anyway.")


def build(task: str, todo: list[dict], dev: list[dict], prompt_version: str,
          structured: bool):
    import tasks.nocot as nocot_task
    import tasks.think as think_task

    if task == "nocot":
        return nocot_task.build_task(todo, dev, prompt_version, structured=structured)
    return think_task.build_task(todo, dev, task, prompt_version)


def raw_model_response(sample) -> dict | None:
    """The last model call's raw response body, if the log kept one.

    Needed for the reasoning-token count on the OpenRouter route: inspect reads it from the
    typed `usage.completion_tokens_details.reasoning_tokens`, and if the router returns the
    field somewhere the OpenAI SDK does not model, `crypticbench.usage` reads it from here
    instead and records that it did.
    """
    for ev in reversed(list(getattr(sample, "events", None) or [])):
        if getattr(ev, "event", None) != "model":
            continue
        call = getattr(ev, "call", None)
        response = getattr(call, "response", None) if call is not None else None
        if isinstance(response, dict):
            return response
    return None


def row_from_sample(s, *, run_id: str, key: str, task: str, dataset: str) -> dict:
    sc = (s.scores or {}).get(SCORER_NAMES[task])
    err = str(s.error.message) if getattr(s, "error", None) else ""
    has_output = bool(s.output and s.output.choices)
    u = sample_usage(s.output.usage if has_output else None, raw=raw_model_response(s))
    md = (sc.metadata if sc else {}) or {}
    meta = s.metadata or {}
    return {
        "run_id": run_id, "system": key, "task": task, "dataset": dataset, "id": s.id,
        "target": s.target if isinstance(s.target, str) else "|".join(s.target),
        "score": sc.value if sc else None,
        "correct": int(sc.value == "C") if sc else "",
        "extracted": sc.answer if sc else None,
        # --- nocot: the 5 % rule's evidence
        "extra_text": md.get("extra_text"), "extra_words": md.get("extra_words"),
        "has_extra": md.get("has_extra"), "structured_output": md.get("structured_output"),
        # --- think: the checker's output, which Val and M2 are computed from
        "verdict": md.get("verdict"), "l0": md.get("l0"), "l1": md.get("l1"),
        "l2a": md.get("l2a"), "l3": md.get("l3"), "unexecuted": md.get("unexecuted"),
        "device_claimed": md.get("device_claimed"),
        "sourcing": json.dumps(md["sourcing"]) if md.get("sourcing") else "",
        "unverified_ops": md.get("unverified_ops"),
        "check_reasons": md.get("reasons"),
        # --- common
        "no_answer_line": md.get("no_answer_line"), "len_ok": md.get("len_ok"),
        "stop_reason": md.get("stop_reason") or (s.output.stop_reason if has_output else "error"),
        # A truncated sample is a budget artefact: it usually has no answer and so scores
        # INCORRECT, which would read as inability rather than a small ceiling.
        "truncated": md.get("truncated") if md else (
            (s.output.stop_reason == "max_tokens") if has_output else None),
        "error": err[:300],
        "input_tokens": u["input_tokens"], "cache_read": u["cache_read"],
        "output_tokens": u["output_tokens"],
        "total_tokens": u["total_tokens"], "hidden_tokens": u["hidden_tokens"],
        "reasoning_tokens": u["reasoning_tokens"],
        "reasoning_tokens_source": u["reasoning_tokens_source"],
        "completion": (s.output.completion or "") if has_output else "",
        "thinking": reasoning_text(s.output.message) if has_output and s.output.message else "",
        "device": meta.get("device"), "hops": meta.get("hops"),
        "published": meta.get("published"), "published_date": meta.get("published_date"),
        "source": meta.get("source"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--systems", required=True, help="comma-separated keys from configs/models.yaml")
    ap.add_argument("--dataset", choices=["nat", "nat50", "guardian200", "times200"],
                    default="nat",
                    help="`nat` (100 clues, the nocot arm), `nat50` (50 of them, drawn once, "
                         "every think arm), and the calibration splits, which are reported "
                         "separately and never pooled (PLAN §2.5)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="k samples per clue; k=8 on nocot (PLAN §4). Each repeat is its own "
                         "run_id, so resumption never merges them.")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids-from",
                    help="restrict the dataset to the ids in this list file, **in the file's "
                         "order**, before --limit. `--dataset nat --ids-from data/nat50.txt "
                         "--limit 20` is the first 20 clues of the nat50 draw scored as part "
                         "of `nat`: the same clues the think arms gate on, in rows that belong "
                         "to the nat arm's own 100 rather than to a second group.")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sync", action="store_true",
                    help="use the system's plain model id rather than its ':batch' variant. "
                         "The id is in the config fingerprint, so a --sync run is never "
                         "resumed into a non-sync one.")
    ap.add_argument("--max-connections", type=int,
                    help="cap concurrent requests. OpenRouter refuses too many large "
                         "reasoning calls in flight at once (HTTP 402 "
                         "`in_flight_budget_exhausted`), so `think` systems need a small "
                         "number here.")
    ap.add_argument("--structured", action="store_true",
                    help="the 5%% rule's fallback: pin nocot output to a one-field JSON schema")
    ap.add_argument("--prompt-version", default=None,
                    help="default v2 for every arm "
                         "(DEFAULT_PROMPT_VERSION)")
    ap.add_argument("--include-cut", action="store_true",
                    help="run a system carrying a `cut` string in configs/models.yaml")
    a = ap.parse_args()
    prompt_version = a.prompt_version or DEFAULT_PROMPT_VERSION[a.task]

    C.load_env()
    print(TEMPERATURE_NOTE)
    systems = load_systems()
    rows, dev = load_rows(a.dataset)

    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import GenerateConfig

    for repeat_index in range(1, a.repeat + 1):
        for key in a.systems.split(","):
            if key not in systems:
                sys.exit(f"unknown system {key!r}; configs/models.yaml has "
                         f"{', '.join(sorted(systems))}")
            cfg = systems[key]
            if check_cut(key, cfg, include_cut=a.include_cut, dry_run=a.dry_run):
                continue
            check_system(key, cfg, a.task, sync=a.sync)
            model_id, batched = resolve_route(cfg, sync=a.sync)
            gen_fields, structured = generate_fields(cfg, sync=a.sync, structured=a.structured)
            batched = batched or bool(gen_fields.get("batch"))
            prompt_sha = load_prompt(a.task, prompt_version)[2]
            # The resolved id is in the fingerprint, so a ':batch' run and a --sync run are two
            # measurements and `--skip-done` never merges one into the other.
            fingerprint = config_fingerprint(model_id, cfg.get("model_args", {}),
                                             gen_fields, structured)
            todo = rows
            if a.ids_from:
                # Restrict *within* the dataset rather than switching to another one. A `nocot`
                # run on `nat50` would put 20 of `nat`'s own clues in a second per-clue group,
                # which analysis/metrics.py would count twice — once at the gate's k and once
                # at the full run's. Same clues, same population, one group.
                wanted = read_id_list(Path(a.ids_from))
                by_id = {r["id"]: r for r in rows}
                missing = [i for i in wanted if i not in by_id]
                if missing:
                    sys.exit(f"[{key}] REFUSED: --ids-from {a.ids_from} names "
                             f"{len(missing)} ids that are not in {a.dataset!r}: "
                             f"{missing[:5]}")
                todo = [by_id[i] for i in wanted]
            if a.skip_done:
                skip = done_ids(key, a.task, a.dataset, prompt_sha, fingerprint)
                todo = [r for r in rows if r["id"] not in skip]
            if a.limit:
                todo = todo[: a.limit]
            if not todo:
                print(f"[{key}] nothing to do")
                continue
            print(f"[{key}] {a.task}/{a.dataset} r{repeat_index}: {len(todo)} clues, "
                  f"model {model_id}, prompt {prompt_version}, "
                  f"{'batch' if batched else 'sync'}"
                  f"{', structured' if structured else ''}")
            if a.dry_run:
                continue

            task, prompt_hash = build(a.task, todo, dev, prompt_version, structured)
            if a.max_connections:
                gen_fields = {**gen_fields, "max_connections": a.max_connections}
            gen = GenerateConfig(**gen_fields)  # validates the YAML fields
            suffix = f"_r{repeat_index}" if a.repeat > 1 else ""
            run_id = (f"{a.task}_{a.dataset}{suffix}_{key}_"
                      f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
            log_dir = C.LOGS_DIR / run_id

            t0 = time.time()
            logs = inspect_eval(
                task,
                model=model_id,
                model_args=cfg.get("model_args", {}),
                log_dir=str(log_dir),
                fail_on_error=0.05,
                display="plain",
                **gen.model_dump(exclude_none=True),
            )
            log = logs[0]
            if log.status != "success":
                print(f"[{key}] eval status {log.status}: {getattr(log, 'error', '')}")
            out_rows = [row_from_sample(s, run_id=run_id, key=key, task=a.task,
                                        dataset=a.dataset)
                        for s in (log.samples or [])]

            C.RUNS_DIR.mkdir(parents=True, exist_ok=True)
            csv_path = C.RUNS_DIR / f"{run_id}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows
                                   else ["run_id"])
                w.writeheader()
                w.writerows(out_rows)

            gate = reasoning_gate(a.task, out_rows, route=cfg.get("route"),
                                  expects_reasoning=expects_reasoning(
                                      a.task, cfg.get("model_args")))
            extra = extra_text_summary(out_rows) if a.task == "nocot" else None
            manifest = {
                "run_id": run_id, "system": key, "task": a.task, "dataset": a.dataset,
                "repeat_index": repeat_index, "n_repeats": a.repeat,
                "model": model_id, "model_configured": cfg.get("model"),
                "model_batch": cfg.get("model_batch"),
                "route": cfg.get("route"), "arm": cfg.get("arm"),
                # a cut system runs only under --include-cut, and the flag that authorised the
                # run is what authorises the analysis: metrics.py keeps such a run instead of
                # setting it aside for being cut
                "cut": cut_note(cfg), "include_cut": bool(a.include_cut),
                "generate": gen_fields, "model_args": cfg.get("model_args", {}),
                "batched": batched, "structured_output": structured,
                "temperature": gen_fields.get("temperature", "not set"),
                "prompt_version": prompt_version, "prompt_sha256": prompt_hash,
                "config_fingerprint": fingerprint,
                "dataset_sha256": sha256_file(C.DATASET_PATHS[a.dataset]),
                # the restriction is part of what was measured, so it is part of the record
                "ids_from": a.ids_from,
                "ids_from_sha256": sha256_file(Path(a.ids_from)) if a.ids_from else None,
                "models_yaml_sha256": sha256_file(C.MODELS_YAML),
                "inspect_ai_version": __import__("inspect_ai").__version__,
                "started_utc": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
                "duration_s": round(time.time() - t0, 1), "n": len(out_rows),
                "status": log.status,
                "training_cutoff": cfg.get("training_cutoff"),
                "training_cutoff_source": cfg.get("training_cutoff_source"),
                "usage": summarise_usage(out_rows),
                "reasoning_gate": gate,
                "extra_text_rule": extra,
                "n_truncated": sum(1 for r in out_rows if r.get("truncated") is True),
                "n_errors": sum(1 for r in out_rows if r.get("error")),
                "log_dir": str(log_dir.relative_to(C.REPO_ROOT)),
            }
            (C.RUNS_DIR / f"{run_id}.json").write_text(json.dumps(manifest, indent=1),
                                                       encoding="utf-8")
            with C.ANSWERS_PATH.open("a", encoding="utf-8") as f:
                for r in out_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            scored = [r for r in out_rows if r["correct"] != ""]
            acc = sum(r["correct"] == 1 for r in scored) / len(scored) if scored else None
            valid = sum(1 for r in out_rows if r.get("verdict") == "valid")
            print(f"[{key}] done: n={len(out_rows)}"
                  + (f", accuracy={acc:.3f}" if acc is not None else "")
                  + (f", valid={valid / len(out_rows):.3f}"
                     if out_rows and a.task in THINK_CONDITIONS else "")
                  + f", mean_out_tokens={manifest['usage']['mean_output_tokens']}"
                  f"  -> {csv_path.name}")
            print(f"[{key}] {gate['note']}")
            if extra:
                print(f"[{key}] {extra['note']}")


if __name__ == "__main__":
    main()
