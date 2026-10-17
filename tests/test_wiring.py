"""The three arms are wired into the runner, and the runner refuses what it cannot measure."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from crypticbench import config as C  # noqa: E402
from crypticbench.corpus import read_id_list, read_jsonl  # noqa: E402
from crypticbench.prompts import load_prompt  # noqa: E402
from run import (  # noqa: E402
    DEFAULT_PROMPT_VERSION,
    SCORER_NAMES,
    TASKS,
    check_system,
    check_cut,
    config_fingerprint,
    cut_note,
    generate_fields,
    load_rows,
    load_systems,
    resolve_route,
)
from tasks.think import CONDITIONS as THINK_CONDITIONS  # noqa: E402

# The setter's corpus is not distributed with the repository (data/README.md); this module
# reads it, so on a checkout without it the module skips rather than fails.
pytestmark = pytest.mark.skipif(not (C.DATA_DIR / "corpus.jsonl").exists(),
                                reason="the setter's corpus is not on this machine")


def test_the_three_arms_are_the_three_tasks():
    assert set(TASKS) == {"nocot", "think", "think_cot"}
    assert set(SCORER_NAMES) == set(TASKS)
    assert set(THINK_CONDITIONS) == {"think", "think_cot"}


@pytest.mark.parametrize("task", TASKS)
def test_every_arm_has_a_prompt_file(task):
    _, user, digest = load_prompt(task)
    assert "{clue}" in user and "{enumeration}" in user and "{examples}" in user
    assert digest


def test_the_three_prompts_are_three_different_measurements():
    digests = {t: load_prompt(t)[2] for t in TASKS}
    assert len(set(digests.values())) == 3


def test_only_the_cot_prompt_asks_for_prose_and_only_nocot_forbids_working():
    assert "in prose" in load_prompt("think_cot")[0]
    assert "in prose" not in load_prompt("think")[0]
    assert "no working" in load_prompt("nocot")[0]
    assert "DEFINITION" in load_prompt("think")[0]
    assert "DEFINITION" not in load_prompt("nocot")[0]


def test_the_dropped_prompts_are_gone():
    """Derive, probe B and the v1 answer-only templates are out (PLAN §8 'Reversed'). A
    dormant template is a condition someone will run by accident. `nocot.v1.txt` is the
    exception and stays: it is the prompt the 2026-09-10 probe rows were measured under, and
    a superseded prompt whose hash is in a manifest cannot be deleted."""
    names = {p.name for p in (ROOT / "prompts").glob("*")}
    assert names == {"nocot.v1.txt", "nocot.v2.txt", "think.v1.txt", "think.v2.txt",
                     "think_cot.v1.txt", "think_cot.v2.txt"}


# ---- nocot prompt v2 (probe report §7.4) -----------------------------------------------------

V2_LINE = ("Do not use any reasoning tokens. Do not think before answering. "
           "Answer immediately.")


def test_v2_is_v1_plus_the_line_that_makes_astras_row_exist():
    """The recipe from the Astra post: a legal effort setting plus an instruction, with the
    count checked per call. The line is in the **system** prompt, so it is asked once and not
    repeated per clue."""
    v1_system, v1_user, _ = load_prompt("nocot", "v1")
    v2_system, v2_user, _ = load_prompt("nocot", "v2")
    assert V2_LINE in v2_system
    assert V2_LINE not in v1_system
    assert v2_system == v1_system + "\n\n" + V2_LINE
    assert v2_user == v1_user          # the clue side is untouched


def test_v1_is_untouched_and_the_two_hash_differently():
    """v1's hash is recorded in the probe manifests of 2026-09-10; if it moved, those runs
    could no longer be tied to the prompt they were made under."""
    v1_sha = load_prompt("nocot", "v1")[2]
    v2_sha = load_prompt("nocot", "v2")[2]
    assert v1_sha != v2_sha
    assert v1_sha == ("e1472a442a4d37f3b45f54cd48889a74"
                      "ac636dabf59aaef69ca523431e2ff8d3")


def test_v2_still_forbids_working_and_still_has_no_template():
    system, user, _ = load_prompt("nocot", "v2")
    assert "no working" in system and "DEFINITION" not in system
    assert "{clue}" in user and "{enumeration}" in user and "{examples}" in user


def test_the_nocot_arm_defaults_to_v2_and_the_think_arms_do_not():
    """One prompt per arm, and for `nocot` that prompt is now v2 — on every system, or the
    arms are not the same measurement. `--prompt-version v1` still reaches the old file."""
    assert DEFAULT_PROMPT_VERSION["nocot"] == "v2"
    # both think arms moved to v2 on 2026-09-10 night, when `initial` and `final` joined the
    # FODDER op list; the gate's v1 rows are set aside by run selection as a result
    assert DEFAULT_PROMPT_VERSION["think"] == "v2"
    assert DEFAULT_PROMPT_VERSION["think_cot"] == "v2"
    assert set(DEFAULT_PROMPT_VERSION) == set(TASKS)


# ---- the cut guard ---------------------------------------------------------------------------

# `astra-think` was cut on 2026-09-10 evening and un-cut the same night: the cut assumed
# 3,150 output tokens and the gate measured 630. Haiku 4.5 was un-cut 2026-09-11;
# `sonnet5-medium` stays cut.
CUT_SYSTEMS = {"sonnet5-medium"}


def test_the_cut_systems_are_exactly_the_ones_nat_cut():
    """Cut 2026-09-10 (PLAN §3, §8). The entries stay in models.yaml for the
    record, so something other than their absence has to stop them being run."""
    systems = load_systems()
    assert {k for k, v in systems.items() if v.get("cut")} == CUT_SYSTEMS
    for key in CUT_SYSTEMS:
        assert "2026-09-10" in systems[key]["cut"], key
        assert "Nat" in systems[key]["cut"], key


def test_a_cut_system_is_refused_and_the_refusal_quotes_the_ruling():
    systems = load_systems()
    for key in sorted(CUT_SYSTEMS):
        with pytest.raises(SystemExit) as e:
            check_cut(key, systems[key], include_cut=False, dry_run=False)
        assert systems[key]["cut"] in str(e.value), key
        assert "--include-cut" in str(e.value), key


def test_include_cut_runs_it_anyway():
    """The cut is a scope decision, not a defect: the row can be brought back with one flag."""
    systems = load_systems()
    assert check_cut("sonnet5-medium", systems["sonnet5-medium"], include_cut=True,
                     dry_run=False) is False


def test_a_dry_run_reports_a_cut_system_instead_of_exiting(capsys):
    """`--dry-run` is how the run matrix is read; exiting on the first cut row would hide the
    rest of it."""
    systems = load_systems()
    assert check_cut("sonnet5-medium", systems["sonnet5-medium"], include_cut=False,
                     dry_run=True) is True
    out = capsys.readouterr().out
    assert "CUT" in out and "not run" in out and "--include-cut" in out


def test_a_live_system_is_not_cut():
    systems = load_systems()
    for key in ("astra-nocot", "astra-think", "sol-nocot", "sonnet5-nocot", "opus5-nocot",
                "sonnet5-high", "mock"):
        assert cut_note(systems[key]) is None, key
        assert check_cut(key, systems[key], include_cut=False, dry_run=False) is False


# ---- the splits ------------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["nat", "nat50", "guardian200", "times200"])
def test_every_split_has_a_file_and_they_are_not_the_same_file(split):
    """Four addressable datasets, four files. `nat50` is a subset of `nat` rather than a fifth
    corpus, but it still gets its own file — the membership *is* the split, so it is what a
    `nat50` manifest hashes into `dataset_sha256`."""
    assert C.DATASET_PATHS[split].exists(), split
    assert len(set(C.DATASET_PATHS.values())) == 4


def test_the_dev_examples_are_the_same_on_every_split():
    """A guardian200 number and a nat100 number must differ in their clues and nothing else."""
    _, dev_nat = load_rows("nat")
    _, dev_guardian = load_rows("guardian200")
    assert [r["id"] for r in dev_nat] == [r["id"] for r in dev_guardian]
    assert len(dev_nat) == C.N_DEV


def test_dev_clues_are_never_scored():
    rows, dev = load_rows("nat")
    assert len(rows) == 100
    assert not ({r["id"] for r in rows} & {r["id"] for r in dev})


def test_device_tags_are_joined_in_by_id():
    rows, _ = load_rows("nat")
    assert all("device" in r for r in rows)
    assert sum(1 for r in rows if r["device"]) == len(rows)
    # PLAN §5.4's hop count is not on file yet; the column exists and is empty, and
    # analysis.by_hops declines rather than inventing it
    assert all(r["hops"] is None for r in rows)


def test_the_tags_are_flagged_unconfirmed():
    """They are Claude's suggestions until Nat ticks them (PLAN §7.1); nothing may read them
    as the setter's."""
    rows, _ = load_rows("nat")
    assert not any(r["device_confirmed"] for r in rows)


# ---- nat50: 50 of the 100, drawn once (PLAN §6; Nat, 2026-09-10) ---------------------------

def test_nat50_is_fifty_of_the_hundred_test_clues():
    hundred, _ = load_rows("nat")
    fifty, _ = load_rows("nat50")
    assert len(fifty) == C.NAT50_SIZE == 50
    ids = {r["id"] for r in fifty}
    assert len(ids) == 50
    assert ids < {r["id"] for r in hundred}      # a strict subset, not a second corpus


def test_nat50_comes_back_in_seed_order_so_the_gates_twenty_are_fixed():
    """The gate runs `--limit 20` on nat50 and the full run reuses those rows through
    `--skip-done`. That only holds if the order is the file's, not the corpus's or a sort."""
    from crypticbench.corpus import read_id_list

    ids = read_id_list(C.NAT50_PATH)
    rows, _ = load_rows("nat50")
    assert [r["id"] for r in rows] == ids
    assert ids != sorted(ids), "a sorted file would make `seed order` meaningless"


def test_nat50_shares_the_dev_examples_with_every_other_split():
    _, dev = load_rows("nat50")
    assert [r["id"] for r in dev] == [r["id"] for r in load_rows("nat")[1]]


def test_nat50_never_contains_a_dev_clue():
    rows, dev = load_rows("nat50")
    assert not ({r["id"] for r in rows} & {r["id"] for r in dev})


def test_the_nat50_file_records_the_seed_and_the_balance_column():
    """A drawn split whose seed is not on the file is not reproducible, and this one had to
    substitute `device` for the `hops` column PLAN §5.4 asks for — which has to be visible
    without reading the tool."""
    head = C.NAT50_PATH.read_text(encoding="utf-8")
    assert "seed 20260906" in head
    assert "proportional on `device`" in head
    assert "hops" in head            # the substitution is declared in the file itself
    assert "SEED ORDER" in head


def test_the_draw_is_proportional_and_sums_to_exactly_fifty():
    from tools.draw_nat50 import proportional_targets

    counts = {"charade": 48, "anagram": 13, "container": 13, "hidden": 8, "double": 6,
              "deletion": 4, "homophone": 4, "cryptic-def": 2, "alternate": 1, "reversal": 1}
    targets = proportional_targets(counts, 100, 50)
    assert sum(targets.values()) == 50
    assert targets["charade"] == 24 and targets["anagram"] == 7
    for key, n in counts.items():
        assert abs(targets[key] - n / 2) <= 0.5 + 1e-9, key


def test_the_draw_is_reproducible_from_the_seed():
    from tools.draw_nat50 import draw

    rows, _ = load_rows("nat")
    tags = {t["id"]: t for t in read_jsonl(C.DEVICES_PATH)}
    first = draw(rows, tags, column="device", seed=C.SEED, n=50)[0]
    second = draw(rows, tags, column="device", seed=C.SEED, n=50)[0]
    other = draw(rows, tags, column="device", seed=C.SEED + 1, n=50)[0]
    assert first == second == read_id_list(C.NAT50_PATH)
    assert other != first


def test_the_draw_refuses_to_balance_on_a_column_that_does_not_exist():
    """`hops` is what PLAN §5.4 wants and no clue carries one. A hop count inferred from the
    parse strings would be this repo's number, not the setter's, so the tool exits instead."""
    from tools.draw_nat50 import choose_column

    tags = {t["id"]: t for t in read_jsonl(C.DEVICES_PATH)}
    with pytest.raises(SystemExit) as e:
        choose_column(tags, "hops")
    assert "no row of data/devices.jsonl carries a `hops` count" in str(e.value)
    assert choose_column(tags, None) == "device"
    assert choose_column({"x": {"hops": 3}}, None) == "hops"


def test_a_nat50_run_fingerprints_the_membership_not_the_corpus():
    """`dataset_sha256` for a nat50 run is the hash of the id list, which is the thing that
    could change underneath a result. Re-drawing the split therefore shows up as a different
    corpus version in the analysis, which is exactly what it is."""
    import run as runner

    assert runner.sha256_file(C.DATASET_PATHS["nat50"]) != \
        runner.sha256_file(C.DATASET_PATHS["nat"])


# ---- refusals ---------------------------------------------------------------------------------

def cfg(**kw):
    base = {"route": "openrouter",
            "model": "openrouter/anthropic/claude-sonnet-5",
            "model_batch": "openrouter/anthropic/claude-sonnet-5:batch",
            "batch": True, "arm": "nocot"}
    base.update(kw)
    return base


def test_a_null_model_id_is_refused():
    with pytest.raises(SystemExit) as e:
        check_system("astra-nocot", cfg(model=None, model_batch=None), "nocot")
    assert "model id in configs/models.yaml" in str(e.value)


def test_a_deliberately_null_entry_is_refused_on_both_ids():
    """The refusals are the reason a blank is safe to leave in the file (PLAN §3, build spec
    §0.2). A missing id must stop the run rather than guess, on the batch id and on the --sync
    id alike."""
    null_entry = cfg(model=None, model_batch=None)
    for sync in (False, True):
        with pytest.raises(SystemExit) as e:
            check_system("null-nocot", null_entry, "nocot", sync=sync)
        assert "no model id" in str(e.value) or "no ':batch' model id" in str(e.value)


def test_a_missing_batch_id_is_refused_for_the_full_run_but_allowed_under_sync():
    """Silently falling back to the non-batch id would run a different configuration from the
    one the system names."""
    no_batch_id = cfg(model_batch=None)
    with pytest.raises(SystemExit) as e:
        check_system("x", no_batch_id, "nocot")
    assert "':batch' model id" in str(e.value)
    check_system("x", no_batch_id, "nocot", sync=True)


def test_the_wrong_arm_is_refused():
    """A `max_tokens: 16` config handed the seven-field prompt would truncate every sample and
    report it as inability."""
    with pytest.raises(SystemExit) as e:
        check_system("sonnet5-nocot", cfg(arm="nocot"), "think")
    assert "'nocot' arm" in str(e.value)


def test_a_complete_system_on_its_own_arm_passes():
    check_system("sonnet5-nocot", cfg(), "nocot")


@pytest.mark.parametrize("key", sorted(load_systems()))
@pytest.mark.parametrize("sync", [False, True])
def test_every_system_in_the_file_passes_its_own_arm_or_says_why_not(key, sync):
    systems = load_systems()
    c = systems[key]
    model, _ = resolve_route(c, sync=sync)
    if model is None:
        with pytest.raises(SystemExit):
            check_system(key, c, c["arm"], sync=sync)
    else:
        check_system(key, c, c["arm"], sync=sync)


# ---- the ':batch' id and --sync ----------------------------------------------------------

def test_without_sync_a_batched_system_uses_the_batch_id():
    model, batched = resolve_route(cfg(), sync=False)
    assert model.endswith(":batch") and batched is True


def test_sync_switches_to_the_plain_id():
    model, batched = resolve_route(cfg(), sync=True)
    assert model == "openrouter/anthropic/claude-sonnet-5" and batched is False


def test_a_system_that_is_not_batched_ignores_sync():
    for sync in (False, True):
        model, batched = resolve_route(cfg(batch=False), sync=sync)
        assert model == "openrouter/anthropic/claude-sonnet-5" and batched is False


def test_the_batch_and_sync_runs_have_different_fingerprints():
    """--skip-done resumes by fingerprint, so a probe must never be merged into the full run's
    CSV: the two are different ids and different measurements."""
    batch_model, _ = resolve_route(cfg(), sync=False)
    sync_model, _ = resolve_route(cfg(), sync=True)
    assert config_fingerprint(batch_model, {}, {"max_tokens": 8}, False) != \
           config_fingerprint(sync_model, {}, {"max_tokens": 8}, False)


def test_the_mock_systems_survive_the_route_change():
    """RUNBOOK §0's smoke test is a mock run; it must not need a key or a batch id."""
    systems = load_systems()
    for key in ("mock", "mock-think", "mock-think-cot"):
        assert systems[key]["route"] == "mock"
        model, batched = resolve_route(systems[key], sync=False)
        assert model == "mockllm/model" and batched is False
        check_system(key, systems[key], systems[key]["arm"])


# ---- generate fields ---------------------------------------------------------------------------

def test_sync_drops_batch_and_structured_is_lifted_out_of_the_generate_block():
    gen, structured = generate_fields({"generate": {"max_tokens": 8, "batch": True,
                                                    "structured": True}},
                                      sync=True, structured=False)
    assert gen == {"max_tokens": 8}
    assert structured is True


def test_the_command_line_flag_can_turn_structured_on_without_editing_the_file():
    gen, structured = generate_fields({"generate": {"max_tokens": 8}}, sync=False,
                                      structured=True)
    assert structured is True and gen == {"max_tokens": 8}


# ---- --ids-from: the same clues, without changing the population -------------------------

def test_ids_from_restricts_within_the_dataset_and_keeps_the_file_order():
    """The gate's `nocot` rows have to be the *same* 20 clues as the `think` arms', but they
    belong to `nat`: a run on `nat50` would give 20 of nat's own clues a second per-clue group
    and metrics.py would count them twice, once at the gate's k and once at the full run's."""
    from crypticbench.corpus import read_id_list

    wanted = read_id_list(C.NAT50_PATH)
    rows, _ = load_rows("nat")
    by_id = {r["id"]: r for r in rows}
    restricted = [by_id[i] for i in wanted]
    assert [r["id"] for r in restricted[:20]] == wanted[:20]
    # and they are exactly what a nat50 run would score first
    fifty, _ = load_rows("nat50")
    assert [r["id"] for r in fifty[:20]] == [r["id"] for r in restricted[:20]]


def test_the_gate_clue_sets_of_the_two_arms_now_agree():
    """`nocot --dataset nat --ids-from data/nat50.txt --limit 20` and
    `think --dataset nat50 --limit 20` are the same twenty clues (RUNBOOK §3)."""
    from crypticbench.corpus import read_id_list

    nocot_gate = read_id_list(C.NAT50_PATH)[:20]
    think_gate = [r["id"] for r in load_rows("nat50")[0][:20]]
    assert nocot_gate == think_gate
    assert len(set(nocot_gate)) == 20


def test_skip_done_does_not_count_an_errored_row_as_done(tmp_path, monkeypatch):
    """A run that dies partway writes every clue to its CSV with the failures recorded. If
    those count as done, `--skip-done` never retries them and the arm quietly loses coverage —
    which is what the 402'd astra-think run of 2026-09-10 did to 20 nat50 clues."""
    import csv as _csv
    import json as _json

    import run as runner

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(runner.C, "RUNS_DIR", runs)
    (runs / "r1.json").write_text(_json.dumps({
        "system": "x", "task": "think", "dataset": "nat50",
        "prompt_sha256": "sha", "config_fingerprint": "fp"}), encoding="utf-8")
    with (runs / "r1.csv").open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["id", "error"])
        w.writeheader()
        w.writerow({"id": "cb-0001", "error": ""})
        w.writerow({"id": "cb-0002", "error": "Error code: 402 - in_flight_budget_exhausted"})
    done = runner.done_ids("x", "think", "nat50", "sha", "fp")
    assert done == {"cb-0001"}, "an errored clue is not done work"


def test_think_v2_is_v1_plus_the_letter_selection_ops():
    """The vocabulary the gate found missing, and nothing else: v2 differs from v1 only in the
    FODDER op list. A prompt that changed anything else would make the gate's rows
    incomparable with the full run's."""
    v1_system, v1_user, v1_sha = load_prompt("think", "v1")
    v2_system, v2_user, v2_sha = load_prompt("think", "v2")
    assert v1_sha != v2_sha
    assert v1_user == v2_user
    assert "initial" in v2_system and "final" in v2_system
    assert "initial (the first letter of each word" in v2_system
    assert "initial" not in v1_system
    # everything outside the op sentence is untouched
    strip = lambda s: "\n".join(l for l in s.split("\n") if "<op> is" not in l)   # noqa: E731
    assert strip(v1_system) == strip(v2_system)


def test_both_think_arms_got_the_vocabulary():
    """`think_cot` shares the seven-field template, so a tag list that differed between them
    would make the prose-CoT rung a different measurement for a reason nobody chose."""
    for name in ("think", "think_cot"):
        system = load_prompt(name, "v2")[0]
        assert "initial (the first letter of each word" in system, name
