"""Independently replay model benchmark candidates and audit score provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run import BENCHMARK_VERSION, load_model_registry
from pramaan import VALIDATOR_VERSION
from pramaan.certificate import make_certificate
from pramaan.checks.core import all_pass, verify_pattern
from pramaan.dsl import Pattern, PatternV2, parse_pattern

LOCKED_MAX_OUTPUT_TOKENS = 4096
LOCKED_TEMPERATURE = 0.4
LOCKED_BASE_PARAMS = {"temperature": LOCKED_TEMPERATURE, "max_tokens": LOCKED_MAX_OUTPUT_TOKENS}


def _generation_protocol_profile(model_key, config):
    """Validate one of the recorded, explicitly documented benchmark protocols."""
    provider = config.get("provider")
    request_params = config.get("request_params")
    if not isinstance(request_params, dict) or request_params.get("max_tokens") != LOCKED_MAX_OUTPUT_TOKENS:
        raise ValueError(
            f"model {model_key} must use the shared {LOCKED_MAX_OUTPUT_TOKENS}-token output limit"
        )
    if provider == "Anthropic":
        expected = {"max_tokens": LOCKED_MAX_OUTPUT_TOKENS,
                    "output_config": {"effort": "low"}}
        profile = "anthropic_provider_default_sampling_low_effort"
    elif provider == "CometAPI":
        expected = {**LOCKED_BASE_PARAMS,
                    "extra_body": {"thinking": {"type": "disabled"}}}
        profile = "temperature_0.4_thinking_disabled"
    elif model_key == "beam-hf":
        expected = {**LOCKED_BASE_PARAMS, "reasoning_effort": "minimal"}
        profile = "temperature_0.4_beam_minimal_reasoning"
    else:
        expected = dict(LOCKED_BASE_PARAMS)
        profile = "temperature_0.4"
    if request_params != expected:
        raise ValueError(
            f"model {model_key} uses an unsupported generation protocol; "
            "only the locked temperature-0.4, Anthropic provider-default/low-effort, "
            "and CometAPI thinking-disabled profiles are allowed"
        )
    return profile


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(records, prompts, registry, expected_model_keys=None, frozen_validator_version=None):
    prompt_bytes = prompts.read_bytes()
    suite_digest = hashlib.sha256(prompt_bytes).hexdigest()
    prompt_rows = read_jsonl(prompts)
    prompt_by_id = {prompt.get("id"): prompt for prompt in prompt_rows}
    if len(prompt_by_id) != len(prompt_rows) or None in prompt_by_id:
        raise ValueError("prompt suite has a missing or duplicate prompt id")
    if not records:
        raise ValueError("results file has no records")
    model_keys = set(expected_model_keys or (row.get("model_key") for row in records))
    if not model_keys or any(key not in registry for key in model_keys):
        raise ValueError("results reference an unconfigured or missing model key")
    generation_protocols = {}
    for key in model_keys:
        generation_protocols[key] = _generation_protocol_profile(key, registry[key])

    seen = set()
    initial_messages_by_prompt = {}
    replayed = 0
    scored = 0
    provider_errors = 0
    by_model = {key: set() for key in model_keys}
    for row_number, row in enumerate(records, 1):
        key = row.get("model_key")
        prompt_id = row.get("prompt_id")
        pair = (key, prompt_id)
        if key not in model_keys or key not in registry or prompt_id not in prompt_by_id:
            raise ValueError(f"row {row_number} has an unknown model or prompt")
        if pair in seen:
            raise ValueError(f"row {row_number} duplicates {key}/{prompt_id}")
        seen.add(pair)
        by_model[key].add(prompt_id)
        config = registry[key]
        expected = {
            "benchmark_version": BENCHMARK_VERSION,
            "validator_version": frozen_validator_version or VALIDATOR_VERSION,
            "prompt_suite_sha256": suite_digest,
            "model_id": config["model_id"],
            "provider": config["provider"],
            "api_style": config["api_style"],
            "request_params": config["request_params"],
        }
        if any(row.get(field) != value for field, value in expected.items()):
            raise ValueError(f"row {row_number} provenance does not match current suite, verifier, or model registry")
        if "pricing" in row and row.get("pricing") != config.get("pricing"):
            raise ValueError(f"row {row_number} pricing rate card does not match the model registry")
        prompt = prompt_by_id[prompt_id]
        prompt_fields = (
            "level", "symmetry", "brief", "state", "grammar", "suite_id", "tradition_id"
        )
        if any(row.get(field) != prompt.get(field) for field in prompt_fields if field in prompt):
            raise ValueError(f"row {row_number} prompt data differs from the locked suite")
        if len(model_keys) > 1:
            calls = row.get("model_calls") or row.get("calls")
            if not isinstance(calls, list):
                raise ValueError(f"row {row_number} lacks exact system and user proposal messages")
            for call in calls:
                telemetry = call.get("telemetry") if isinstance(call, dict) else None
                provider_attempts = telemetry.get("provider_attempts") if isinstance(telemetry, dict) else None
                if type(provider_attempts) is not int or provider_attempts != 1:
                    raise ValueError(
                        f"row {row_number} must record exactly one provider attempt per call; "
                        "benchmark transport retries are disabled"
                    )
            proposal = next((call for call in calls
                             if isinstance(call, dict) and call.get("phase") == "proposal"), None)
            messages = (proposal or {}).get("messages")
            if (not isinstance(messages, list) or len(messages) != 2
                    or any(not isinstance(message, dict) for message in messages)
                    or [message.get("role") for message in messages] != ["system", "user"]
                    or any(not isinstance(message.get("content"), str) for message in messages)):
                raise ValueError(f"row {row_number} lacks exact system and user proposal messages")
            previous = initial_messages_by_prompt.setdefault(prompt_id, messages)
            if previous != messages:
                raise ValueError(f"row {row_number} initial proposal messages differ across models")
        if row.get("max_attempts") != 3:
            raise ValueError(f"row {row_number} violates the shared three-proposal attempt limit")
        attempts = row.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != row.get("attempt_count") or len(attempts) > 3:
            raise ValueError(f"row {row_number} has inconsistent attempt count")
        provider_errors += row.get("provider_error") is not None
        first = attempts[0] if attempts else None
        first_pass = bool(first and first.get("report", {}).get("all_hard_constraints_pass"))
        first_schema = bool(first and first.get("candidate") is not None)
        if row.get("valid_schema_at_1") != first_schema or row.get("validity_at_1") != first_pass:
            raise ValueError(f"row {row_number} first-proposal score disagrees with its recorded attempt")
        if attempts:
            scored += 1

        passed_attempts = []
        for attempt in attempts:
            candidate = attempt.get("candidate")
            if candidate is None:
                continue
            pattern = parse_pattern(json.dumps(candidate))
            if prompt.get("grammar") is not None:
                if prompt["grammar"] == "periodic_raster":
                    if not isinstance(pattern, Pattern) or pattern.style != "ajrakh_inspired":
                        raise ValueError(f"row {row_number} attempt {attempt.get('index')} violates the locked periodic-raster contract")
                else:
                    if not isinstance(pattern, PatternV2):
                        raise ValueError(f"row {row_number} attempt {attempt.get('index')} is not a v0.2 pattern")
                    if pattern.grammar != prompt["grammar"] or pattern.tradition_id != prompt.get("tradition_id"):
                        raise ValueError(f"row {row_number} attempt {attempt.get('index')} does not match the locked grammar or tradition")
                if "palette" in prompt and pattern.palette != prompt["palette"]:
                    raise ValueError(f"row {row_number} attempt {attempt.get('index')} does not match the locked palette")
                profile = prompt.get("constraint_profile", {})
                if not isinstance(profile, dict):
                    raise ValueError(f"row {row_number} has an invalid constraint profile")
                for field, expected_value in profile.items():
                    actual_value = (
                        pattern.jaali.frame if field == "frame" and isinstance(pattern, PatternV2)
                        and pattern.jaali is not None
                        else pattern.constraints.symmetry if field == "symmetry"
                        else getattr(pattern.constraints, field, None)
                    )
                    if hasattr(actual_value, "model_dump"):
                        actual_value = actual_value.model_dump(exclude_none=True)
                    if actual_value != expected_value:
                        raise ValueError(f"row {row_number} attempt {attempt.get('index')} violates locked constraint {field}")
            checks, masks, grid = verify_pattern(pattern)
            certificate = make_certificate(pattern, checks, masks, grid)
            report = attempt.get("report", {})
            if report.get("checks") != checks or report.get("all_hard_constraints_pass") != all_pass(checks):
                raise ValueError(f"row {row_number} attempt {attempt.get('index')} does not replay")
            replayed += 1
            if all_pass(checks):
                passed_attempts.append(certificate["mask_sha256"])
        accepted = bool(row.get("validity_at_3"))
        if accepted != bool(passed_attempts):
            raise ValueError(f"row {row_number} repair score disagrees with replayed attempts")
        if accepted and row.get("accepted_mask_sha256") != passed_attempts[0]:
            raise ValueError(f"row {row_number} accepted mask digest does not match replay")
        if row.get("repair_rescued") != (not first_pass and accepted):
            raise ValueError(f"row {row_number} repair-rescued flag is inconsistent")

    coverage = {
        key: {
            "observed": len(by_model[key]),
            "expected": len(prompt_by_id),
            "missing_prompt_ids": sorted(set(prompt_by_id) - by_model[key]),
            "complete": by_model[key] == set(prompt_by_id),
        }
        for key in sorted(model_keys)
    }
    return {
        "result_rows": len(records),
        "scored_rows": scored,
        "candidate_replays": replayed,
        "provider_error_rows": provider_errors,
        "prompt_suite_sha256": suite_digest,
        "coverage_by_model": coverage,
        "generation_protocol_by_model": dict(sorted(generation_protocols.items())),
        "complete": all(value["complete"] for value in coverage.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--prompts", type=Path, default=ROOT / "benchmarks" / "prompts.jsonl")
    parser.add_argument("--models", type=Path, default=ROOT / "benchmarks" / "models.json")
    parser.add_argument("--model", action="append", dest="model_keys",
                        help="Expected model key; repeat for multi-model coverage checks")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--frozen-validator-version",
        choices=["0.2.0"],
        help="Replay the frozen v1 dataset while allowing only its recorded validator version label",
    )
    args = parser.parse_args()
    try:
        registry = load_model_registry(args.models)
        report = audit(
            read_jsonl(args.results),
            args.prompts,
            registry,
            args.model_keys,
            frozen_validator_version=args.frozen_validator_version,
        )
        if args.require_complete and not report["complete"]:
            raise ValueError("results do not cover every locked prompt for every selected model")
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
