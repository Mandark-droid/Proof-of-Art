"""Flatten only explicitly training-labeled trajectories into chat SFT JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _conversation(messages: list[dict], response: str) -> list[dict]:
    """Return a complete chat transcript while leaving the source prompt intact."""
    result = [dict(message) for message in messages]
    if result and result[-1].get("role") == "assistant":
        if result[-1].get("content") != response:
            raise ValueError("existing assistant turn conflicts with completion")
        return result
    result.append({"role": "assistant", "content": response})
    return result


def flatten(records: list[dict]) -> tuple[list[dict], int]:
    examples = []
    skipped_evaluation = 0
    for record in records:
        role = record.get("data_role")
        if role != "training" or record.get("training_eligible") is not True:
            if role == "benchmark_evaluation":
                skipped_evaluation += 1
            continue
        for call in record.get("calls", []):
            response = call.get("raw_response")
            if call.get("error") is not None or not isinstance(response, str):
                continue
            report = call.get("verifier_report") or {}
            examples.append({
                "messages": _conversation(call["messages"], response),
                "completion": response,
                "trajectory_id": record["trajectory_id"],
                "call_index": call["call_index"],
                "phase": call["phase"],
                "model_id": record.get("model_id"),
                "provider": record.get("provider"),
                "validator_version": record.get("validator_version"),
                "telemetry": call.get("telemetry"),
                "run_telemetry": record.get("telemetry"),
                "accepted": call.get("accepted", False),
                "failure_check_ids": [check.get("id") for check in report.get("checks", [])
                                      if check.get("status") == "FAIL"],
            })
    return examples, skipped_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        records = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
        examples, skipped = flatten(records)
        if not examples:
            raise ValueError("no explicitly training-eligible calls found; evaluation trajectories are intentionally excluded")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            for example in examples:
                stream.write(json.dumps(example, ensure_ascii=False) + "\n")
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output), "training_examples": len(examples),
                      "evaluation_trajectories_skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
