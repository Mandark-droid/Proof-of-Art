"""Build the frozen, outcome-independent 47-task eval-v2.1 subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pramaan.tasks import canonical_jsonl_bytes


COMMON47_SEED = "pramaan-eval-v2.1-common47-v1"
SOURCE_SHA256 = "b04b568d84d2568fd261789819d4e925844dba5afe9c5962275ee571d84e4e3a"
COMMON47_SHA256 = "dd3c9ce96a22deb1af94a1db22dc2a5016d1b1ad1c9c4a12befcd6690a6f7603"


def _rank(label: str, state: str) -> str:
    return hashlib.sha256(f"{COMMON47_SEED}|{label}|{state}".encode("utf-8")).hexdigest()


def build_common47_rows(source_rows: list[dict]) -> list[dict]:
    """Select 47 tasks with a fixed, model-independent balance across all states."""
    if len(source_rows) != 72:
        raise ValueError("common47 requires the frozen 72-row eval-v2.1 suite")
    if any(not isinstance(row, dict) or row.get("suite_id") != "eval-v2.1" for row in source_rows):
        raise ValueError("common47 source rows must all belong to eval-v2.1")
    ids = [row.get("id") for row in source_rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != 72:
        raise ValueError("common47 source task IDs must be unique non-empty strings")
    states = {row.get("state") for row in source_rows}
    if len(states) != 36 or any(not isinstance(state, str) or not state for state in states):
        raise ValueError("common47 source must cover exactly 36 named states")
    by_state = {state: {row["level"]: row for row in source_rows if row["state"] == state}
                for state in states}
    if any(set(levels) != {"easy", "hard"} for levels in by_state.values()):
        raise ValueError("common47 source needs one easy and one hard row per state")

    paired_states = sorted(states, key=lambda state: (_rank("pair", state), state))[:11]
    remaining = states.difference(paired_states)
    single_states = sorted(remaining, key=lambda state: (_rank("single", state), state))
    selected_ids = {
        row["id"]
        for state in paired_states
        for row in by_state[state].values()
    }
    selected_ids.update(by_state[state]["easy"]["id"] for state in single_states[:14])
    selected_ids.update(by_state[state]["hard"]["id"] for state in single_states[14:])
    rows = [row for row in source_rows if row["id"] in selected_ids]
    if len(rows) != 47:
        raise ValueError("common47 selection did not produce exactly 47 tasks")
    return rows


def serialize_common47(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\r\n" for row in rows).encode("utf-8")


def write_common47(source_path: Path, target_path: Path) -> str:
    """Write only from the expected frozen suite and never overwrite an artifact."""
    source_bytes = Path(source_path).read_bytes()
    actual_source_sha = hashlib.sha256(canonical_jsonl_bytes(source_bytes)).hexdigest()
    if actual_source_sha != SOURCE_SHA256:
        raise ValueError(f"eval-v2.1 source digest mismatch: {actual_source_sha}")
    source_rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()]
    content = serialize_common47(build_common47_rows(source_rows))
    digest = hashlib.sha256(canonical_jsonl_bytes(content)).hexdigest()
    if digest != COMMON47_SHA256:
        raise ValueError(f"common47 digest mismatch: {digest}")
    target_path = Path(target_path)
    if target_path.exists():
        raise FileExistsError(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(content)
    return digest


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = root / "benchmarks" / "suites" / "eval-v2.1.jsonl"
    target = root / "benchmarks" / "suites" / "eval-v2.1-common47.jsonl"
    digest = write_common47(source, target)
    print(f"{target}: 47 tasks sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
