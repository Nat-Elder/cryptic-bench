"""Paths, constants and environment loading.

The .env file is loaded explicitly from the repo root (not from the current working
directory) so scripts behave the same wherever they are invoked from.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
DEVICES_PATH = DATA_DIR / "devices.jsonl"
EXEMPLARS_PATH = DATA_DIR / "exemplars.json"
GUARDIAN200_PATH = DATA_DIR / "calibration" / "guardian200.jsonl"
TIMES200_PATH = DATA_DIR / "calibration" / "times200.jsonl"
# `nat50` is **not** a fourth split of PLAN §2.5: it is 50 of `nat`'s own 100 test clues, drawn
# once with a recorded seed and held in an id list rather
# than in the corpus, so `corpus.jsonl`'s hash — and therefore every existing run's
# `dataset_sha256` — does not move. `nocot` runs on all 100; the `think` arms run on the 50.
# The list is what a `nat50` manifest fingerprints, which is the right thing to fingerprint:
# the membership is the split.
NAT50_PATH = DATA_DIR / "nat50.txt"
NAT50_SIZE = 50
# The splits of PLAN §2.5 plus nat50. `nat` and `nat50` are one study; the calibration sets are
# reported separately and never pooled with either.
DATASET_PATHS = {
    "nat": CORPUS_PATH,
    "nat50": NAT50_PATH,
    "guardian200": GUARDIAN200_PATH,
    "times200": TIMES200_PATH,
}
PROMPTS_DIR = REPO_ROOT / "prompts"
CONFIGS_DIR = REPO_ROOT / "configs"
MODELS_YAML = CONFIGS_DIR / "models.yaml"
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
RUNS_DIR = RESULTS_DIR / "runs"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
ANSWERS_PATH = RESULTS_DIR / "answers.jsonl"
CACHE_DIR = RESULTS_DIR / "cache"

SEED = 20260906
# Few-shot example clues, excluded from scoring. Was 5 (the seeded five); NARRATE left the
# corpus 2026-09-10 and the set is four-shot, a declared deviation (PLAN §8).
N_DEV = 4

_ENV_LOADED = False


def load_env() -> None:
    """Load REPO_ROOT/.env once. Existing environment variables win."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    _ENV_LOADED = True


def require_env(name: str) -> str:
    load_env()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; add it to {REPO_ROOT / '.env'}")
    return value
