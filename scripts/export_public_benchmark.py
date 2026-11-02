"""Export replay-audited benchmark rows without raw completion text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run import load_model_registry
from pramaan.orchestrate import _call_telemetry, _run_telemetry
from scripts.check_benchmark_results import audit, read_jsonl


def sanitize_rows(rows):
    """Keep source DSL, checks, and usage metadata while dropping duplicate raw text."""
    sanitized = []
    for row in rows:
        clean = json.loads(json.dumps(row))
        calls = clean.get("model_calls") or [
            {"metadata": attempt.get("metadata", {}), "error": None,
             "telemetry": _call_telemetry(attempt.get("metadata", {}), None)}
            for attempt in clean.get("attempts", [])
        ]
        telemetry = _run_telemetry(calls, clean.get("attempts", []), clean.get("elapsed_ms", 0))
        clean["telemetry"] = telemetry
        clean["token_usage"] = telemetry["token_usage"]
        clean["cost"] = telemetry["cost"]
        clean["latency_ms"] = telemetry["latency_ms"]
        clean["provider_attempts"] = telemetry["provider_attempts"]
        clean["provider_retries"] = telemetry["provider_retries"]
        clean["total_tokens"] = telemetry["token_usage"]["total_tokens"]
        for attempt in clean.get("attempts", []):
            attempt.pop("raw", None)
        # Full requests/responses live in the separately labeled trajectory dataset.
        clean.pop("model_calls", None)
        sanitized.append(clean)
    return sanitized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Full local JSONL result file")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "data" / "beam-hf-shared-first-v2.jsonl")
    parser.add_argument("--prompts", type=Path, default=ROOT / "benchmarks" / "prompts.jsonl")
    parser.add_argument("--models", type=Path, default=ROOT / "benchmarks" / "models.json")
    parser.add_argument("--model", action="append", dest="model_keys",
                        help="Expected model key; repeat for multi-model coverage checks")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing public result: {args.output}")
    try:
        rows = read_jsonl(args.input)
        registry = load_model_registry(args.models)
        report = audit(rows, args.prompts, registry, args.model_keys)
        if not report["complete"]:
            raise ValueError("public export requires complete prompt coverage for each included model")
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean = sanitize_rows(rows)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in clean:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), **report, "raw_completion_text_included": False}, indent=2))


if __name__ == "__main__":
    main()
