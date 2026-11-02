"""Materialize replay-verified Beam examples from the sanitized benchmark dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run import load_model_registry
from pramaan.certificate import make_certificate
from pramaan.checks.core import all_pass, verify_pattern
from pramaan.dsl import Pattern
from pramaan.export import render_png
from scripts.check_benchmark_results import audit, read_jsonl


DEFAULT_PROMPT_IDS = ("medium-03", "medium-06", "hard-07")


def prepare(records, prompt_ids):
    by_id = {row.get("prompt_id"): row for row in records}
    prepared = []
    for prompt_id in prompt_ids:
        row = by_id.get(prompt_id)
        if row is None or not row.get("validity_at_3"):
            raise ValueError(f"{prompt_id} is missing or has no verified accepted candidate")
        accepted = next((attempt for attempt in row["attempts"]
                         if attempt.get("report", {}).get("all_hard_constraints_pass")), None)
        if accepted is None:
            raise ValueError(f"{prompt_id} has no passing candidate in its trajectory")
        pattern = Pattern.model_validate_json(json.dumps(accepted["candidate"]))
        checks, masks, grid = verify_pattern(pattern)
        if not all_pass(checks):
            raise ValueError(f"{prompt_id} failed fresh verification")
        certificate = make_certificate(pattern, checks, masks, grid)
        if certificate["mask_sha256"] != row.get("accepted_mask_sha256"):
            raise ValueError(f"{prompt_id} accepted mask digest differs from benchmark result")
        trajectory = {
            "schema_version": 1,
            "source": "sanitized_model_benchmark_record",
            "beam_generated": True,
            "accepted": True,
            "model_key": row["model_key"],
            "model": {"model_id": row["model_id"], "provider": row["provider"]},
            "created_at": row["created_at"],
            "original_prompt": row["brief"],
            "prompt_id": row["prompt_id"],
            "prompt_suite_sha256": row["prompt_suite_sha256"],
            "benchmark_version": row["benchmark_version"],
            "validator_version": row["validator_version"],
            "attempt_count": row["attempt_count"],
            "repair_count": max(0, row["attempt_count"] - 1),
            "request_params": row["request_params"],
            "constraint_profile": pattern.constraints.model_dump(mode="json"),
            "certificate_mask_sha256": certificate["mask_sha256"],
            "certificate_dsl_sha256": certificate["dsl_sha256"],
            "attempts": row["attempts"],
        }
        prepared.append((prompt_id, pattern, certificate, grid, trajectory))
    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks" / "data" / "beam-hf-shared-first-v2.jsonl")
    parser.add_argument("--models", type=Path, default=ROOT / "benchmarks" / "models.json")
    parser.add_argument("--prompts", type=Path, default=ROOT / "benchmarks" / "prompts.jsonl")
    parser.add_argument("--prompt-id", action="append", dest="prompt_ids",
                        help="Accepted benchmark prompt ID to turn into a gallery example; repeatable")
    args = parser.parse_args()
    prompt_ids = args.prompt_ids or list(DEFAULT_PROMPT_IDS)
    try:
        registry = load_model_registry(args.models)
        records = read_jsonl(args.results)
        report = audit(records, args.prompts, registry)
        if not report["complete"]:
            raise ValueError("gallery preparation requires complete, replay-audited results")
        prepared = prepare(records, prompt_ids)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    targets = []
    for prompt_id, *_ in prepared:
        stem = f"beam_{prompt_id.replace('-', '_')}"
        targets.extend([
            ROOT / "examples" / f"{stem}.pattern.json",
            ROOT / "examples" / f"{stem}.certificate.json",
            ROOT / "examples" / f"{stem}.trajectory.json",
            ROOT / "assets" / f"{stem}_preview.png",
        ])
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        parser.error("refusing to overwrite existing gallery artifacts: " + ", ".join(existing))

    for prompt_id, pattern, certificate, grid, trajectory in prepared:
        stem = f"beam_{prompt_id.replace('-', '_')}"
        (ROOT / "examples" / f"{stem}.pattern.json").write_text(
            json.dumps(pattern.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (ROOT / "examples" / f"{stem}.certificate.json").write_text(
            json.dumps(certificate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (ROOT / "examples" / f"{stem}.trajectory.json").write_text(
            json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        render_png(grid, pattern.palette, scale=5, repeats=3).save(ROOT / "assets" / f"{stem}_preview.png")
        print(f"Prepared {prompt_id}: {pattern.title} ({trajectory['attempt_count']} proposal(s))")


if __name__ == "__main__":
    main()
