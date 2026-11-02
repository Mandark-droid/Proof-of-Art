"""Build a replayable, prompt-complete dataset from a full benchmark result file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run import PALETTE
from pramaan.orchestrate import _call_telemetry, _run_telemetry
from pramaan.prompts import SYSTEM_PROMPT, design_prompt, repair_prompt
from scripts.check_benchmark_results import audit, read_jsonl
from benchmarks.run import load_model_registry


def _profile(row: dict) -> dict:
    for attempt in row.get("attempts", []):
        candidate = attempt.get("candidate")
        if candidate:
            return {**candidate["constraints"], "exclusive_ink_layers": False}
    raise ValueError(f"{row.get('prompt_id')}: cannot reconstruct immutable constraints")


def make_trajectory(row: dict, model_config: dict | None = None) -> dict:
    """Attach exact request/response pairs and verifier outcomes to each model call."""
    profile = _profile(row)
    by_call = {attempt.get("call_index", attempt["index"]): attempt
               for attempt in row.get("attempts", [])}
    calls = []
    previous_attempt = None
    source_calls = row.get("model_calls") or [
        {"call_index": attempt.get("call_index", attempt["index"]),
         "phase": "proposal" if attempt["index"] == 1 else "repair",
         "started_at": None, "completed_at": None,
         "raw_response": attempt.get("raw"), "metadata": attempt.get("metadata", {}), "error": None}
        for attempt in row.get("attempts", [])
    ]
    if row.get("provider_error") and (not source_calls or len(source_calls) < len(row.get("attempts", [])) + 1):
        source_calls.append({"call_index": len(source_calls) + 1,
                             "phase": row["provider_error"].get("phase"),
                             "raw_response": None, "metadata": {},
                             "error": row["provider_error"]})
    for call_index, raw_call in enumerate(source_calls, 1):
        call = json.loads(json.dumps(raw_call))
        phase = call.get("phase")
        if phase == "proposal":
            expected_prompt = design_prompt(row["brief"], PALETTE, profile)
        elif phase == "repair":
            if previous_attempt is None:
                raise ValueError(f"{row.get('prompt_id')}: repair call has no preceding candidate")
            candidate = previous_attempt.get("candidate")
            previous = (json.dumps(candidate, sort_keys=True) if candidate
                        else (previous_attempt.get("raw") or "")[:8000])
            expected_prompt = repair_prompt(row["brief"], profile, previous,
                                            previous_attempt["report"]["checks"])
        else:
            raise ValueError(f"{row.get('prompt_id')}: unsupported call phase {phase!r}")
        expected_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": expected_prompt},
        ]
        # Existing runs predate in-orchestrator capture; reconstruct the exact prompts
        # from their locked prompt, immutable profile, and preceding verifier report.
        if not call.get("messages"):
            call["messages"] = expected_messages
        elif call["messages"] != expected_messages:
            raise ValueError(f"{row.get('prompt_id')}: captured request does not match replayable prompt")
        call["call_index"] = call_index
        if "telemetry" not in call:
            telemetry_metadata = dict(call.get("metadata") or {})
            pricing = (row.get("pricing") or (model_config or {}).get("pricing"))
            created_date = (row.get("created_at") or "")[:10]
            # Backfill an estimate only when the public rate card was observed on
            # the same calendar date as the recorded run; never imply billed cost.
            if pricing and pricing.get("observed_at") == created_date:
                telemetry_metadata["pricing"] = pricing
            call["telemetry"] = _call_telemetry(telemetry_metadata, None)
        call["request"] = {"model": row.get("model_id"),
                           "parameters": row.get("request_params") or {}}
        attempt = by_call.get(call_index)
        if attempt:
            call["candidate_index"] = attempt["index"]
            call["verified_at"] = attempt.get("verified_at")
            call["candidate"] = attempt.get("candidate")
            call["verifier_report"] = attempt.get("report")
            call["accepted"] = bool(attempt.get("report", {}).get("all_hard_constraints_pass"))
            previous_attempt = attempt
        else:
            call["candidate_index"] = None
            call["candidate"] = None
            call["verifier_report"] = None
            call["accepted"] = False
        calls.append(call)

    if len(calls) != len(row.get("attempts", [])) and not row.get("provider_error"):
        raise ValueError(f"{row.get('prompt_id')}: call and candidate counts disagree")
    run_telemetry = _run_telemetry(calls, row.get("attempts", []), row["elapsed_ms"])
    return {
        "trajectory_schema_version": "1.0",
        "trajectory_id": f"{row['model_key']}:{row['prompt_id']}:{row['prompt_suite_sha256'][:12]}",
        "dataset_id": row["benchmark_version"],
        "data_role": "benchmark_evaluation",
        "training_eligible": False,
        "source": "paired_model_benchmark",
        "model_key": row["model_key"], "model_id": row["model_id"],
        "provider": row["provider"], "api_style": row["api_style"],
        "request_params": row.get("request_params"),
        "pricing": row.get("pricing") or (model_config or {}).get("pricing"),
        "prompt_suite_sha256": row["prompt_suite_sha256"],
        "prompt_id": row["prompt_id"], "level": row["level"],
        "brief": row["brief"], "symmetry": row["symmetry"],
        "constraint_profile": profile,
        "validator_version": row["validator_version"],
        "valid_schema_at_1": bool(row.get("attempts") and row["attempts"][0].get("candidate") is not None),
        "validity_at_1": row["validity_at_1"],
        "validity_at_3": row["validity_at_3"],
        "repair_rescued": row["repair_rescued"],
        "attempt_count": len(row.get("attempts", [])),
        "max_attempts": row.get("max_attempts", 3),
        "accepted_mask_sha256": row.get("accepted_mask_sha256"),
        "elapsed_ms": row["elapsed_ms"], "created_at": row["created_at"],
        "telemetry": run_telemetry,
        "provider_error": row.get("provider_error"),
        "calls": calls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Full local benchmark JSONL with raw outputs")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "data" / "beam-hf-trajectories-v1.jsonl")
    parser.add_argument("--prompts", type=Path, default=ROOT / "benchmarks" / "prompts.jsonl")
    parser.add_argument("--models", type=Path, default=ROOT / "benchmarks" / "models.json")
    parser.add_argument("--model", action="append", dest="model_keys")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing dataset: {args.output}")
    try:
        rows = read_jsonl(args.input)
        registry = load_model_registry(args.models)
        report = audit(rows, args.prompts, registry, args.model_keys)
        if not report["complete"]:
            raise ValueError("trajectory export requires complete evaluation-set coverage")
        trajectories = [make_trajectory(row, registry.get(row.get("model_key"))) for row in rows]
    except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for record in trajectories:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(trajectories),
                      "data_role": "benchmark_evaluation", "raw_model_responses_included": True,
                      "candidate_replays": report["candidate_replays"]}, indent=2))


if __name__ == "__main__":
    main()
