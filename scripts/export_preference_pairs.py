"""Export preference pairs only from explicitly training-labeled trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _conversation(messages: list[dict], response: str) -> list[dict]:
    """Add an assistant turn without changing the source prompt."""
    result = [dict(message) for message in messages]
    if result and result[-1].get("role") == "assistant":
        if result[-1].get("content") != response:
            raise ValueError("existing assistant turn conflicts with candidate")
        return result
    result.append({"role": "assistant", "content": response})
    return result


def export_sft_examples(records: list[dict]) -> list[dict]:
    examples = []
    for record in records:
        if record.get("data_role") != "training" or record.get("training_eligible") is not True:
            continue
        for call in record.get("calls", []):
            response = call.get("raw_response")
            if call.get("error") is not None or not isinstance(response, str):
                continue
            examples.append({
                "messages": _conversation(call.get("messages", []), response), "completion": response,
                "trajectory_id": record["trajectory_id"], "call_index": call.get("call_index"),
                "phase": call.get("phase"), "reward": call.get("reward", 0.0),
                "reward_version": record.get("reward_version"),
                "model_id": record.get("model_id"), "provider": record.get("provider"),
                "data_role": "training", "training_eligible": True,
            })
    return examples


def export_pairs(records: list[dict]) -> list[dict]:
    pairs = []
    for record in records:
        role = record.get("data_role")
        if role == "benchmark_evaluation":
            continue
        if role != "training" or record.get("training_eligible") is not True:
            continue
        calls = record.get("calls", [])
        rejected = next((call for call in calls if call.get("phase") == "proposal"
                         and call.get("candidate") is not None
                         and call.get("accepted") is False), None)
        chosen = next((call for call in reversed(calls) if call.get("accepted") is True
                       and isinstance(call.get("raw_response"), str)), None)
        proposal = next((call for call in calls if call.get("phase") == "proposal"), None)
        if not (proposal and rejected and chosen and proposal.get("messages")):
            continue
        pairs.append({
            "prompt": proposal["messages"],
            "chosen_messages": _conversation(proposal["messages"], chosen["raw_response"]),
            "rejected_messages": _conversation(proposal["messages"], rejected["raw_response"]),
            "chosen": chosen["raw_response"],
            "rejected": rejected["raw_response"],
            "trajectory_id": record["trajectory_id"],
            "prompt_id": record.get("prompt_id"),
            "chosen_call_index": chosen.get("call_index"),
            "rejected_call_index": rejected.get("call_index"),
            "chosen_failure_check_ids": [],
            "rejected_failure_check_ids": [
                check.get("id") for check in (rejected.get("verifier_report") or {}).get("checks", [])
                if check.get("status") == "FAIL"
            ],
            "model_id": record.get("model_id"),
            "provider": record.get("provider"),
            "data_role": "training", "training_eligible": True,
        })
    return pairs


def export_repair_examples(records: list[dict]) -> list[dict]:
    examples = []
    for record in records:
        if record.get("data_role") != "training" or record.get("training_eligible") is not True:
            continue
        for call in record.get("calls", []):
            response = call.get("raw_response")
            if call.get("phase") != "repair" or call.get("error") is not None or not isinstance(response, str):
                continue
            report = call.get("verifier_report") or {}
            examples.append({
                "messages": _conversation(call.get("messages", []), response), "completion": response,
                "trajectory_id": record["trajectory_id"], "call_index": call.get("call_index"),
                "model_id": record.get("model_id"), "provider": record.get("provider"),
                "failure_check_ids": [check.get("id") for check in report.get("checks", [])
                                      if check.get("status") == "FAIL"],
                "data_role": "training", "training_eligible": True,
            })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair", action="store_true", help="export repair-call SFT rows instead of preference pairs")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        records = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
        pairs = export_repair_examples(records) if args.repair else export_pairs(records)
        if not pairs:
            raise ValueError("no eligible repair rows" if args.repair else
                             "no accepted/rejected pairs from explicitly training-eligible trajectories")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            for pair in pairs:
                stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    key = "repair_examples" if args.repair else "preference_pairs"
    print(json.dumps({"output": str(args.output), key: len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
