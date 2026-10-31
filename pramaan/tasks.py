from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pramaan.traditions import GRAMMARS
from pramaan.paths import resolve_project_root

_ROOT = resolve_project_root(__file__)
SUITES = {
    "eval-v2": _ROOT / "benchmarks" / "suites" / "eval-v2.jsonl",
    "eval-v2.1": _ROOT / "benchmarks" / "suites" / "eval-v2.1.jsonl",
    "eval-v2.1-common47": _ROOT / "benchmarks" / "suites" / "eval-v2.1-common47.jsonl",
    "train-v1": _ROOT / "benchmarks" / "suites" / "train-v1.jsonl",
    "train-v1.1": _ROOT / "benchmarks" / "suites" / "train-v1.1.jsonl",
    "train-v1.2": _ROOT / "benchmarks" / "suites" / "train-v1.2.jsonl",
}


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    suite_id: str
    state: str
    tradition_id: str
    grammar: str
    level: str
    symmetry: Literal["None", "D2", "D4", "C2", "C4"]
    constraint_profile: dict[str, Any]
    palette: list[str]
    brief: str = Field(min_length=1)

    @field_validator("grammar")
    @classmethod
    def _grammar_is_known(cls, value: str) -> str:
        if value not in GRAMMARS:
            raise ValueError(f"unknown grammar: {value}")
        return value


def load_suite(name: str) -> tuple[list[Task], str]:
    try:
        path = SUITES[name]
    except KeyError as exc:
        raise ValueError(f"unknown suite {name!r}; choose one of {', '.join(SUITES)}") from exc
    payload = path.read_bytes()
    tasks = [Task.model_validate_json(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if not tasks:
        raise ValueError(f"suite {name!r} is empty")
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"suite {name!r} contains duplicate task ids")
    return tasks, hashlib.sha256(canonical_jsonl_bytes(payload)).hexdigest()


def canonical_jsonl_bytes(payload: bytes) -> bytes:
    """Normalize JSONL newlines to CRLF before hashing, independent of checkout OS."""
    return payload.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
