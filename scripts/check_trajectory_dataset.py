"""Replay verifier reports and check structural invariants in trajectory JSONL."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pramaan.checks.core import all_pass, verify_pattern
from pramaan.dsl import Pattern


def audit(path: Path) -> dict:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
        if record.get("trajectory_schema_version") != "1.0":
            raise ValueError(f"line {line_number}: unsupported trajectory schema")
        if record.get("data_role") not in {"training", "development", "benchmark_evaluation"}:
            raise ValueError(f"line {line_number}: unknown data role")
        if record.get("data_role") == "benchmark_evaluation" and record.get("training_eligible") is not False:
            raise ValueError(f"line {line_number}: evaluation rows cannot be training eligible")
        run_telemetry = record.get("telemetry")
        if not isinstance(run_telemetry, dict) or not all(
                key in run_telemetry for key in ("elapsed_ms", "latency_ms", "token_usage", "cost")):
            raise ValueError(f"line {line_number}: run telemetry is incomplete")
        calls = record.get("calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError(f"line {line_number}: calls must be a non-empty array")
        expected_index = 1
        previous_candidate = None
        replay_count = 0
        for call in calls:
            if call.get("call_index") != expected_index:
                raise ValueError(f"line {line_number}: call indices must be contiguous")
            expected_index += 1
            if call.get("phase") not in {"proposal", "repair"}:
                raise ValueError(f"line {line_number}: invalid call phase")
            telemetry = call.get("telemetry")
            if not isinstance(telemetry, dict) or not all(
                    key in telemetry for key in ("latency_ms", "token_usage", "cost", "provider_attempts")):
                raise ValueError(f"line {line_number}: call telemetry is incomplete")
            cost = telemetry.get("cost", {})
            rate_card = cost.get("rate_card") if isinstance(cost, dict) else None
            if rate_card and cost.get("estimated_amount") is not None:
                usage = telemetry.get("token_usage", {})
                rates = rate_card.get("per_million_tokens", {})
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                if prompt_tokens is None or completion_tokens is None:
                    raise ValueError(f"line {line_number}: cost estimate lacks token usage")
                details = ((call.get("metadata") or {}).get("usage", {}).get("prompt_tokens_details") or {})
                cached_tokens = min(prompt_tokens, details.get("cached_tokens") or 0)
                input_rate = rates["input"]
                cached_rate = rates.get("cached_input", input_rate)
                expected_cost = ((prompt_tokens - cached_tokens) * input_rate
                                 + cached_tokens * cached_rate
                                 + completion_tokens * rates["output"]) / 1_000_000
                if not math.isclose(cost["estimated_amount"], expected_cost, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(f"line {line_number}: rate-card cost estimate does not replay")
            if not isinstance(call.get("messages"), list) or len(call["messages"]) != 2:
                raise ValueError(f"line {line_number}: each call needs exact system and user messages")
            if [message.get("role") for message in call["messages"]] != ["system", "user"]:
                raise ValueError(f"line {line_number}: message roles must be system then user")
            if call["phase"] == "repair" and previous_candidate is None:
                raise ValueError(f"line {line_number}: repair call lacks a preceding checked candidate")
            raw = call.get("raw_response")
            if call.get("error") is None and not isinstance(raw, str):
                raise ValueError(f"line {line_number}: successful call lacks a raw response")
            candidate = call.get("candidate")
            report = call.get("verifier_report")
            if candidate is not None:
                pattern = Pattern.model_validate_json(json.dumps(candidate))
                checks, _, _ = verify_pattern(pattern)
                saved_checks = (report or {}).get("checks", [])
                # Truncation is a provider finish-state condition layered on top of
                # geometry replay; all geometry checks still need exact equality.
                geometry_checks = [check for check in saved_checks if check.get("id") != "candidate_truncated"]
                if checks != geometry_checks:
                    raise ValueError(f"line {line_number}: candidate {call['candidate_index']} does not replay")
                saved_pass = bool(report and report.get("all_hard_constraints_pass"))
                expected_pass = all_pass(checks) and not any(
                    check.get("id") == "candidate_truncated" for check in saved_checks)
                if saved_pass != expected_pass or call.get("accepted") != expected_pass:
                    raise ValueError(f"line {line_number}: candidate outcome does not replay")
                replay_count += 1
                previous_candidate = candidate
            elif call.get("candidate_index") is not None:
                raise ValueError(f"line {line_number}: missing candidate has a candidate index")
        records.append((record, replay_count))
    ids = [record.get("trajectory_id") for record, _ in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate trajectory_id values")
    return {"records": len(records), "candidate_replays": sum(count for _, count in records),
            "roles": {role: sum(record.get("data_role") == role for record, _ in records)
                      for role in ("training", "development", "benchmark_evaluation")}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.dataset)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "PASS", **result}, indent=2))


if __name__ == "__main__":
    main()
