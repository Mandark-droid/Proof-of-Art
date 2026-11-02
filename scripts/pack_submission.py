"""Create a local, key-free community submission archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import zipfile

from scripts.push_hub_datasets import EVAL_SUITES, read_jsonl
from pramaan.tasks import canonical_jsonl_bytes

MAX_ARCHIVE_SOURCE_BYTES = 20_000_000
_BLOCKED_KEYS = {"api_key", "provider_api_key", "authorization", "headers", "secret",
                 "secret_key", "access_token", "hf_token", "token", "raw", "raw_response",
                 "completion", "prompt_messages", "messages", "provider_error", "model_calls"}


def _contains_private_data(value) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if (normalized in _BLOCKED_KEYS or normalized.endswith("_api_key")
                    or normalized.endswith("_token") or _contains_private_data(item)):
                return True
    elif isinstance(value, list):
        return any(_contains_private_data(item) for item in value)
    elif isinstance(value, str):
        return bool(re.search(
            r"(?:sk-ant-api\d{2}-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{20,})",
            value,
        ))
    return False

def pack_submission(results_path: Path, output_path: Path) -> dict:
    results_path, output_path = Path(results_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {output_path}")
    payload = results_path.read_bytes()
    if not payload or len(payload) > MAX_ARCHIVE_SOURCE_BYTES:
        raise ValueError("results file is empty or exceeds the 20 MB archive limit")
    rows = read_jsonl(results_path)
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("results must be nonempty JSONL objects")
    if any(_contains_private_data(row) for row in rows):
        raise ValueError("results contain private credentials or raw provider data")
    suite_values = [row.get("suite_id") for row in rows]
    digest_values = [row.get("prompt_suite_sha256") for row in rows]
    model_values = [row.get("model_key") for row in rows]
    prompt_ids = [row.get("prompt_id") for row in rows]
    if (any(not isinstance(value, str) for value in suite_values + digest_values + model_values + prompt_ids)
            or len(set(suite_values)) != 1 or len(set(digest_values)) != 1
            or len(set(model_values)) != 1):
        raise ValueError("submission must contain one model and one frozen suite revision")
    suite_id, suite_sha = suite_values[0], digest_values[0]
    model_keys = set(model_values)
    if suite_id not in EVAL_SUITES or not isinstance(suite_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", suite_sha):
        raise ValueError("submission must identify a supported frozen suite and SHA-256")
    suite_path, _ = EVAL_SUITES[suite_id]
    actual_suite_sha = hashlib.sha256(canonical_jsonl_bytes(suite_path.read_bytes())).hexdigest()
    if suite_sha != actual_suite_sha:
        raise ValueError("submission suite digest does not match the local frozen suite")
    known_ids = {row["id"] for row in read_jsonl(suite_path)}
    if any(not isinstance(prompt_id, str) or prompt_id not in known_ids for prompt_id in prompt_ids):
        raise ValueError("submission contains a prompt outside its frozen suite")
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("submission contains duplicate prompt ids")
    manifest = {
        "schema_version": "pramaan-community-submission-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_id": suite_id,
        "suite_sha256": suite_sha,
        "model_keys": sorted(model_keys),
        "row_count": len(rows),
        "prompt_ids": prompt_ids,
        "results_sha256": hashlib.sha256(payload).hexdigest(),
        "training_eligible": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        archive.writestr("results.jsonl", payload)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="JSONL output from benchmarks/run.py")
    parser.add_argument("archive", type=Path, help="new ZIP path; existing paths are never overwritten")
    args = parser.parse_args()
    try:
        manifest = pack_submission(args.results, args.archive)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: manifest[key] for key in
                      ("suite_id", "suite_sha256", "model_keys", "row_count", "results_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
