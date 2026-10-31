"""Append and merge locally collected, independently bounded benchmark shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pramaan.checks.core import all_pass, verify_pattern
from pramaan.dsl import parse_pattern
from pramaan.reward import reward_for_report
from scripts.export_preference_pairs import export_pairs, export_repair_examples, export_sft_examples
from scripts.push_hub_datasets import make_suite_trajectory, read_jsonl

BUDGET_FIELDS = ("records", "calls", "tokens", "cost_usd", "seconds")
ROLE_MAP = {"training": "training", "evaluation": "benchmark_evaluation"}


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, append: bool) -> None:
    mode = "a" if append else "x"
    with path.open(mode, encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _check_caps(caps: dict[str, Any]) -> None:
    if set(caps) != set(BUDGET_FIELDS):
        raise ValueError("each shard requires records, calls, tokens, cost_usd, and seconds caps")
    for field, value in caps.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"budget cap {field} must be a positive number")


def _usage(rows: list[dict[str, Any]]) -> dict[str, float]:
    calls = tokens = 0
    cost = seconds = 0.0
    for row in rows:
        telemetry = row.get("telemetry") or {}
        row_calls = row.get("calls", row.get("model_calls", []))
        if not row_calls:
            raise ValueError(f"{row.get('prompt_id')}: cannot verify call/token/cost caps without call telemetry")
        calls += int(row.get("provider_attempts") or sum(
            int((call.get("telemetry") or {}).get("provider_attempts") or 1) for call in row_calls
        ))
        row_tokens = row.get("total_tokens")
        if row_tokens is None:
            per_call_tokens = [((call.get("telemetry") or {}).get("token_usage") or {}).get("total_tokens")
                               for call in row_calls]
            if any(value is None for value in per_call_tokens):
                raise ValueError(f"{row.get('prompt_id')}: token usage is missing, cannot certify token cap")
            row_tokens = sum(per_call_tokens)
        tokens += int(row_tokens)
        cost_record = telemetry.get("cost") or row.get("cost") or {}
        value = cost_record.get("amount")
        currency = cost_record.get("currency")
        if value is not None and currency != "USD":
            raise ValueError(f"{row.get('prompt_id')}: reported cost is not USD, cannot certify USD cost cap")
        if value is None:
            value = cost_record.get("estimated_amount")
            currency = cost_record.get("estimated_currency")
        if value is None and isinstance(cost_record.get("estimated_amount"), dict):
            value = cost_record["estimated_amount"].get("amount")
            currency = cost_record["estimated_amount"].get("currency")
        if value is None:
            call_costs = [((call.get("telemetry") or {}).get("cost") or {}) for call in row_calls]
            call_values = []
            for item in call_costs:
                call_value = item.get("amount") if item.get("amount") is not None else item.get("estimated_amount")
                call_currency = item.get("currency") if item.get("amount") is not None else item.get("estimated_currency")
                if call_value is None or call_currency != "USD":
                    raise ValueError(f"{row.get('prompt_id')}: USD cost is missing, cannot certify cost cap")
                call_values.append(call_value)
            value = sum(call_values)
            currency = "USD"
        if currency != "USD":
            raise ValueError(f"{row.get('prompt_id')}: estimated cost is not USD, cannot certify USD cost cap")
        if value is not None:
            cost += float(value)
        elapsed_ms = row.get("elapsed_ms")
        if elapsed_ms is None:
            raise ValueError(f"{row.get('prompt_id')}: elapsed time is missing, cannot certify time cap")
        seconds += float(elapsed_ms) / 1000
    return {"records": len(rows), "calls": calls, "tokens": tokens,
            "cost_usd": round(cost, 12), "seconds": round(seconds, 6)}


def _within_caps(actual: dict[str, float], caps: dict[str, float]) -> None:
    over = [f"{name}={actual[name]} exceeds cap {cap}"
            for name, cap in caps.items() if actual[name] > cap]
    if over:
        raise ValueError("shard budget exceeded: " + "; ".join(over))


def _replay_row(row: dict[str, Any], task: dict[str, Any]) -> None:
    for key in ("suite_id", "state", "tradition_id", "grammar", "level", "symmetry", "brief"):
        if row.get(key) != task.get(key):
            raise ValueError(f"{row.get('prompt_id')}: {key} differs from the frozen suite")
    for attempt in row.get("attempts", []):
        candidate, report = attempt.get("candidate"), attempt.get("report") or {}
        checks = report.get("checks", [])
        if candidate is None:
            if not any(isinstance(check, dict) and check.get("id") == "candidate_parse"
                       and check.get("status") == "FAIL" for check in checks):
                raise ValueError(f"{row.get('prompt_id')}: unparsed candidate lacks a parse failure")
            if reward_for_report(report) != 0:
                raise ValueError(f"{row.get('prompt_id')}: unparsed candidate must have zero reward")
            continue
        pattern = parse_pattern(json.dumps(candidate, ensure_ascii=False))
        replay, _, _ = verify_pattern(pattern)
        saved = [check for check in checks if isinstance(check, dict) and check.get("id") != "candidate_truncated"]
        if replay != saved:
            raise ValueError(f"{row.get('prompt_id')} attempt {attempt.get('index')}: verifier replay mismatch")
        truncated = any(isinstance(check, dict) and check.get("id") == "candidate_truncated" for check in checks)
        if bool(report.get("all_hard_constraints_pass")) != (all_pass(replay) and not truncated):
            raise ValueError(f"{row.get('prompt_id')}: acceptance does not replay")


def append_shard(*, root: Path, shard_id: str, role: str, results_path: Path, suite_path: Path,
                 generator_version: str, seed: str, caps: dict[str, Any]) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", shard_id):
        raise ValueError("shard_id must be 1-80 safe filename characters")
    if role not in ROLE_MAP:
        raise ValueError("role must be training or evaluation")
    _check_caps(caps)
    if not generator_version.strip() or not str(seed).strip():
        raise ValueError("prompt generator version and seed are required provenance")

    suite_sha = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    tasks = read_jsonl(suite_path)
    task_by_id = {task["id"]: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("suite has duplicate task ids")
    result_rows = read_jsonl(results_path)
    result_ids = [row.get("prompt_id") for row in result_rows]
    if not result_rows or len(result_ids) != len(set(result_ids)):
        raise ValueError("shard results must be nonempty and duplicate-free")
    directory = root / "shards" / shard_id
    records_path, manifest_path = directory / "records.jsonl", directory / "manifest.jsonl"
    existing_events = _read_jsonl(manifest_path)
    existing_rows = _read_jsonl(records_path)
    if existing_events:
        initial = existing_events[0]
        identity = {"role": role, "suite_sha256": suite_sha, "prompt_generator_version": generator_version,
                    "prompt_seed": str(seed), "caps": caps}
        if any(initial.get(key) != value for key, value in identity.items()):
            raise ValueError("resume metadata or budget caps differ from the existing shard")
    incoming = []
    for row in result_rows:
        if row.get("prompt_suite_sha256") != suite_sha:
            raise ValueError(f"{row.get('prompt_id')}: suite digest mismatch")
        task = task_by_id.get(row.get("prompt_id"))
        if task is None:
            raise ValueError(f"{row.get('prompt_id')}: task is not in the supplied suite")
        if row.get("suite_id") != task.get("suite_id"):
            raise ValueError(f"{row.get('prompt_id')}: suite identity mismatch")
        if role == "training" and not str(task.get("suite_id", "")).startswith("train-"):
            raise ValueError("evaluation suite rows cannot enter a training shard")
        if role == "evaluation" and not str(task.get("suite_id", "")).startswith("eval-"):
            raise ValueError("training suite rows cannot enter an evaluation shard")
        _replay_row(row, task)
        trajectory = make_suite_trajectory(row, task, ROLE_MAP[role])
        trajectory.update({"shard_id": shard_id, "prompt_generator_version": generator_version,
                           "prompt_seed": str(seed), "reward_version": "pramaan-reward-v1"})
        incoming.append(trajectory)
    by_id = {row["trajectory_id"]: row for row in existing_rows}
    for row in incoming:
        prior = by_id.get(row["trajectory_id"])
        if prior is not None and prior != row:
            raise ValueError(f"{row['trajectory_id']}: duplicate ID has different content")
    new_rows = [row for row in incoming if row["trajectory_id"] not in by_id]
    cumulative = existing_rows + new_rows
    actual = _usage(cumulative)
    _within_caps(actual, caps)
    directory.mkdir(parents=True, exist_ok=True)
    if new_rows:
        _write_jsonl(records_path, new_rows, append=records_path.exists())
    event = {"shard_id": shard_id, "role": ROLE_MAP[role], "suite_id": tasks[0].get("suite_id"),
             "suite_sha256": suite_sha, "prompt_generator_version": generator_version,
             "prompt_seed": str(seed), "caps": caps, "actual": actual,
             "appended_records": len(new_rows), "recorded_at": datetime.now(timezone.utc).isoformat()}
    _write_jsonl(manifest_path, [event], append=manifest_path.exists())
    return event


def _fingerprint(row: dict[str, Any]) -> tuple[str, str, str, str]:
    brief = " ".join(str(row.get("brief", "")).casefold().split())
    return (str(row.get("state", "")).casefold(), str(row.get("tradition_id", "")).casefold(),
            str(row.get("grammar", "")).casefold(), brief)


def merge_training_shards(root: Path, output_dir: Path) -> dict[str, int]:
    shard_root = root / "shards"
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for manifest_path in sorted(shard_root.glob("*/manifest.jsonl")):
        events = _read_jsonl(manifest_path)
        if not events:
            continue
        rows = _read_jsonl(manifest_path.parent / "records.jsonl")
        role = events[0].get("role")
        if role == "training":
            train_rows.extend(rows)
        elif role == "benchmark_evaluation":
            eval_rows.extend(rows)
        else:
            raise ValueError(f"{manifest_path}: unknown shard role")
    if not train_rows:
        raise ValueError("no training shards found")
    ids: set[str] = set()
    prompt_ids: set[str] = set()
    fingerprints: set[tuple[str, str, str, str]] = set()
    for row in train_rows:
        if row.get("data_role") != "training" or row.get("training_eligible") is not True:
            raise ValueError("training merge encountered a non-training record")
        if row.get("trajectory_id") in ids:
            raise ValueError(f"duplicate training trajectory: {row.get('trajectory_id')}")
        ids.add(row["trajectory_id"])
        if row.get("prompt_id") in prompt_ids:
            raise ValueError(f"duplicate training prompt ID: {row.get('prompt_id')}")
        prompt_ids.add(row.get("prompt_id"))
        fingerprint = _fingerprint(row)
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate training prompt fingerprint: {row.get('prompt_id')}")
        fingerprints.add(fingerprint)
    eval_ids = {row.get("prompt_id") for row in eval_rows}
    eval_fingerprints = {_fingerprint(row) for row in eval_rows}
    if prompt_ids.intersection(eval_ids) or fingerprints.intersection(eval_fingerprints):
        raise ValueError("evaluation/training split contamination detected")

    output_dir.mkdir(parents=True, exist_ok=False)
    sft = export_sft_examples(train_rows)
    repairs = export_repair_examples(train_rows)
    preferences = export_pairs(train_rows)
    for name, rows in (("trajectories", train_rows), ("sft", sft), ("repair", repairs),
                       ("preference", preferences)):
        _write_jsonl(output_dir / f"{name}.jsonl", rows, append=False)
    return {"trajectories": len(train_rows), "sft": len(sft), "repair": len(repairs),
            "preference": len(preferences)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    append = commands.add_parser("append", help="append a replay-validated independently capped shard")
    append.add_argument("--root", type=Path, required=True)
    append.add_argument("--shard-id", required=True)
    append.add_argument("--role", choices=ROLE_MAP, required=True)
    append.add_argument("--results", type=Path, required=True)
    append.add_argument("--suite", type=Path, required=True)
    append.add_argument("--prompt-generator-version", required=True)
    append.add_argument("--seed", required=True)
    for field in BUDGET_FIELDS:
        append.add_argument(f"--max-{field.replace('_', '-')}", type=float, required=True)
    merge = commands.add_parser("merge-training", help="merge local training shards after leakage checks")
    merge.add_argument("--root", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "append":
        caps = {field: getattr(args, f"max_{field}") for field in BUDGET_FIELDS}
        event = append_shard(root=args.root, shard_id=args.shard_id, role=args.role,
                             results_path=args.results, suite_path=args.suite,
                             generator_version=args.prompt_generator_version,
                             seed=args.seed, caps=caps)
        print(json.dumps(event, indent=2))
    else:
        print(json.dumps(merge_training_shards(args.root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
