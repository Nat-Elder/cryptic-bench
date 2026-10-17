"""Prompt templates: versioned files in prompts/, split into [system] and [user] sections.

`load_prompt("think")` reads prompts/think.v1.txt. There are three templates and they are the
three arms of PLAN §2.1: `nocot` (bare answer), `think` (the seven-field block) and
`think_cot` (prose reasoning, then the block).
"""

from __future__ import annotations

import hashlib
import json

from .config import EXEMPLARS_PATH, PROMPTS_DIR
from .derivation import Component, Derivation


def load_prompt(name: str, version: str = "v1") -> tuple[str, str, str]:
    """Return (system_text, user_template, sha256) for prompts/<name>.<version>.txt."""
    path = PROMPTS_DIR / f"{name}.{version}.txt"
    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if "[system]" not in text or "[user]" not in text:
        raise ValueError(f"{path} must contain [system] and [user] sections")
    system = text.split("[system]", 1)[1].split("[user]", 1)[0].strip()
    user = text.split("[user]", 1)[1].strip()
    return system, user, digest


def format_examples_nocot(dev_rows: list[dict]) -> str:
    """Few-shot block for the `nocot` arm: clue, enumeration, arrow, answer, nothing else.

    The arrow rather than v1's `// ANSWER:` marker, because the arm's whole point is that the
    first generated token is the answer (PLAN §2.1). A label the model has to reproduce is one
    or two forward passes spent on formatting.
    """
    return "\n".join(
        f"Clue: {r['clue']} ({r['enumeration']}) \u2192 {r['answer_display']}" for r in dev_rows
    )


def load_exemplars() -> dict[str, Derivation]:
    """Hand-written derivations of the dev clues, in the seven-field template
    (data/exemplars.json)."""
    raw = json.loads(EXEMPLARS_PATH.read_text(encoding="utf-8"))["derivations"]
    out: dict[str, Derivation] = {}
    for clue_id, r in raw.items():
        out[clue_id] = Derivation(
            definition=list(r["definition"]),
            device=r["device"],
            indicator=r.get("indicator"),
            fodder=[
                Component(text=c["text"], source=c["source"], letters=c["letters"], op=c.get("op"))
                for c in r.get("fodder", [])
            ],
            operation=r["operation"],
            result=r["result"],
            answer=r["answer"],
        )
    return out


def format_examples_think(dev_rows: list[dict]) -> str:
    """Few-shot block for the `think` arms: each dev clue followed by its derivation.

    Dev clues with no exemplar on file are skipped, so the shot count is whatever
    data/exemplars.json covers; PLAN §9 carries the missing fifth (NARRATE).
    """
    exemplars = load_exemplars()
    blocks = []
    for r in dev_rows:
        d = exemplars.get(r["id"])
        if d is None:
            continue
        blocks.append(f"Clue: {r['clue']} ({r['enumeration']})\n{d.render()}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")
