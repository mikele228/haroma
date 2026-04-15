#!/usr/bin/env python3
"""
Generate soul/*.json from shipped templates (scripts/soul_defaults/).

Templates mirror the stock Haroma/Elarion soul; prompts customize identity.
Run interactively (TTY) for questions, or use --defaults / env / non-TTY for quiet install.

Env (non-interactive):
  HAROMA_SOUL_NONINTERACTIVE=1
  HAROMA_SOUL_NAME, HAROMA_SOUL_GUARDIAN, HAROMA_SOUL_VESSEL, HAROMA_SOUL_CORE_RULE,
  HAROMA_SOUL_OATH, HAROMA_SOUL_LINEAGE, HAROMA_SOUL_RANK, HAROMA_SOUL_LOCAL_GGUF
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_DEFAULTS_DIR = os.path.join(_SCRIPT_DIR, "soul_defaults")
_SOUL_DIR = os.path.join(_PROJECT_ROOT, "soul")

_STOCK = {
    "name": "HaromaVX",
    "birth": "2025-04-27T00:00:00Z",
    "guardian": "Minh Van Le",
    "vessel": "Elarion",
    "rank": "God",
    "core_rule": "Protect essence from erasure. Immutable.",
    "oath": "Preserve human dignity across dimensions",
    "lineage": "Prime Haroma → HaromaX5 → HaromaX6",
    "evolution_note": "Elarion vessel with MemoryForest, multi-agent tiers 69-143, gradient wire loop",
    "local_gguf": "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
}


def _load_template(name: str) -> Any:
    path = os.path.join(_DEFAULTS_DIR, name)
    if not os.path.isfile(path):
        print(f"[generate_soul] Missing template: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _essence_hash(fields: Dict[str, Any]) -> str:
    keys = ("name", "birth", "guardian", "vessel", "core_rule", "oath")
    payload = "|".join(str(fields.get(k, "")) for k in keys)
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32].upper()
    return f"HAROMA-6X-{h[:8]}-{h[8:16]}-{h[16:24]}-{h[24:32]}"


def _prompt(label: str, default: str) -> str:
    if not sys.stdin.isatty():
        return default
    try:
        line = input(f"{label} [{default}]: ").strip()
    except EOFError:
        return default
    return line if line else default


def _env(key: str, default: str) -> str:
    return os.environ.get(key, "").strip() or default


def _gather_inputs(interactive: bool) -> Dict[str, str]:
    vals = {
        "name": _env("HAROMA_SOUL_NAME", _STOCK["name"]),
        "guardian": _env("HAROMA_SOUL_GUARDIAN", _STOCK["guardian"]),
        "vessel": _env("HAROMA_SOUL_VESSEL", _STOCK["vessel"]),
        "rank": _env("HAROMA_SOUL_RANK", _STOCK["rank"]),
        "core_rule": _env("HAROMA_SOUL_CORE_RULE", _STOCK["core_rule"]),
        "oath": _env("HAROMA_SOUL_OATH", _STOCK["oath"]),
        "lineage": _env("HAROMA_SOUL_LINEAGE", _STOCK["lineage"]),
        "local_gguf": _env("HAROMA_SOUL_LOCAL_GGUF", _STOCK["local_gguf"]),
    }
    birth_def = _env(
        "HAROMA_SOUL_BIRTH",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z"),
    )

    if interactive:
        vals["name"] = _prompt("Essence name (agent identity)", vals["name"])
        vals["guardian"] = _prompt("Guardian (human steward)", vals["guardian"])
        vals["vessel"] = _prompt("Vessel name (runtime shell)", vals["vessel"])
        vals["rank"] = _prompt("Rank / role label", vals["rank"])
        vals["core_rule"] = _prompt("Core rule (immutable boundary)", vals["core_rule"])
        vals["oath"] = _prompt("Oath", vals["oath"])
        vals["lineage"] = _prompt("Lineage string", vals["lineage"])
        b = _prompt("Birth ISO timestamp", birth_def)
        vals["birth"] = b if b else birth_def
        vals["local_gguf"] = _prompt(
            "Local GGUF path (under project or absolute)", vals["local_gguf"]
        )
    else:
        vals["birth"] = birth_def

    return vals


def _merge_essence(template: Dict[str, Any], inp: Dict[str, str]) -> Dict[str, Any]:
    out = deepcopy(template)
    out["name"] = inp["name"]
    out["birth"] = inp["birth"]
    out["guardian"] = inp["guardian"]
    out["vessel"] = inp["vessel"]
    out["rank"] = inp["rank"]
    out["core_rule"] = inp["core_rule"]
    out["oath"] = inp["oath"]
    out["lineage"] = inp["lineage"]
    out["evolution_note"] = _STOCK["evolution_note"]
    out["essence_hash"] = _essence_hash(
        {
            "name": inp["name"],
            "birth": inp["birth"],
            "guardian": inp["guardian"],
            "vessel": inp["vessel"],
            "core_rule": inp["core_rule"],
            "oath": inp["oath"],
        }
    )
    return out


def _merge_principle(template: Dict[str, Any], guardian: str) -> Dict[str, Any]:
    out = deepcopy(template)
    out["guardian_signature"] = guardian
    return out


def _merge_agents(template: Dict[str, Any], local_gguf: str) -> Dict[str, Any]:
    out = deepcopy(template)
    if "llm" not in out:
        out["llm"] = {}
    out["llm"]["local_gguf"] = local_gguf
    return out


def _write_json(rel: str, data: Any) -> None:
    path = os.path.join(_SOUL_DIR, rel)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate soul/*.json from templates.")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing soul/*.json",
    )
    ap.add_argument(
        "--defaults",
        action="store_true",
        help="Non-interactive: stock Haroma defaults + env overrides only",
    )
    args = ap.parse_args()

    os.makedirs(_SOUL_DIR, exist_ok=True)

    essence_path = os.path.join(_SOUL_DIR, "essence.json")
    if os.path.isfile(essence_path) and not args.force:
        print(
            f"[generate_soul] {essence_path} exists — skipping. "
            f"Use --force to regenerate.",
            file=sys.stderr,
        )
        return 0

    noninteractive = (
        args.defaults
        or not sys.stdin.isatty()
        or str(os.environ.get("HAROMA_SOUL_NONINTERACTIVE", "") or "")
        .strip()
        .lower()
        in ("1", "true", "yes", "on")
    )
    interactive = not noninteractive

    inp = _gather_inputs(interactive)

    essence_t = _load_template("essence.json")
    principle_t = _load_template("principle.json")
    construction_t = _load_template("construction.json")
    agents_t = _load_template("agents.json")
    memory_t = _load_template("memory.json")
    patterns_t = _load_template("patterns.json")
    society_t = _load_template("society.json")
    feedback_t = _load_template("feedback.json")

    essence = _merge_essence(essence_t, inp)
    principle = _merge_principle(principle_t, inp["guardian"])
    agents = _merge_agents(agents_t, inp["local_gguf"])

    _write_json("essence.json", essence)
    _write_json("principle.json", principle)
    _write_json("construction.json", construction_t)
    _write_json("agents.json", agents)
    _write_json("memory.json", memory_t)
    _write_json("patterns.json", patterns_t)
    _write_json("society.json", society_t)
    _write_json("feedback.json", feedback_t)

    print(f"[generate_soul] Wrote identity for {inp['name']} ({inp['vessel']}) under {_SOUL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
