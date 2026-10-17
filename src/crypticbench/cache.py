"""Content-hash disk cache for plain SDK calls (device tagging).

Pattern from inoculation-personas/src/personas/judge.py: key = sha256 of the call's
identifying fields; one JSON file per key; atomic write via rename.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from .config import CACHE_DIR


def _key(*fields: Any) -> str:
    blob = json.dumps(list(fields), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def cached(namespace: str, fields: tuple[Any, ...], compute: Callable[[], dict]) -> dict:
    """Return the cached result for `fields`, computing and storing it if absent."""
    d = CACHE_DIR / namespace
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{_key(*fields)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result = compute()
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    Path(tmp).replace(path)
    return result
