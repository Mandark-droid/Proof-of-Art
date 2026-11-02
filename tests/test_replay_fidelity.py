"""Frozen v0.1 check reports and raster digests must survive validator changes."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from benchmarks.run import BENCHMARK_VERSION, load_model_registry
from pramaan.certificate import make_certificate
from pramaan.checks.core import verify_pattern
from pramaan.dsl import Pattern, parse_pattern
from pramaan.render import mask_digest
from scripts.check_benchmark_results import audit


ROOT = Path(__file__).resolve().parents[1]


def _replay(candidate: dict) -> tuple[list[dict], str]:
    pattern = Pattern.model_validate_json(json.dumps(candidate))
    checks, masks, grid = verify_pattern(pattern)
    return checks, mask_digest(masks, grid)


def test_v01_replay_is_byte_identical() -> None:
    example_count = 0
    benchmark_candidate_count = 0
    benchmark_digest_count = 0

    for design_path in sorted((ROOT / "examples").glob("*.pattern.json")):
        certificate_path = design_path.with_name(
            design_path.name.removesuffix(".pattern.json") + ".certificate.json"
        )
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        candidate = json.loads(design_path.read_text(encoding="utf-8"))
        if candidate.get("schema_version") != "0.1":
            continue
        checks, digest = _replay(candidate)
        assert checks == certificate["checks"], design_path.name
        assert digest == certificate["mask_sha256"], design_path.name
        example_count += 1

    results_path = ROOT / "benchmarks" / "data" / "beam-hf-shared-first-v2.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        for attempt in row["attempts"]:
            candidate = attempt.get("candidate")
            if candidate is None:
                continue
            checks, digest = _replay(candidate)
            report = attempt["report"]
            assert checks == report["checks"], (row["prompt_id"], attempt["index"])
            benchmark_candidate_count += 1

            # The frozen v1 JSONL stores a digest only for its accepted candidate.
            # Failing candidates still have their full check report, but no saved digest.
            if row.get("accepted_mask_sha256") and all(
                check["status"] == "PASS" for check in checks
            ):
                assert digest == row["accepted_mask_sha256"], row["prompt_id"]
                benchmark_digest_count += 1

    assert example_count == 6
    assert benchmark_candidate_count == 38
    assert benchmark_digest_count == 22


def test_benchmark_audit_replays_v02_candidates() -> None:
    design_path = ROOT / "examples" / "beam_g7_gj_sidi_saiyyed_jaali.pattern.json"
    candidate = json.loads(design_path.read_text(encoding="utf-8"))
    pattern = parse_pattern(json.dumps(candidate))
    checks, masks, grid = verify_pattern(pattern)
    certificate = make_certificate(pattern, checks, masks, grid)
    prompt = {
        "id": "gj_sidi_saiyyed-easy-0",
        "brief": "A Gujarat Jaali lattice with a framed geometric opening",
        "level": "easy",
        "symmetry": pattern.constraints.symmetry,
        "state": "Gujarat",
        "grammar": "jaali_lattice",
        "suite_id": "eval-v2",
        "tradition_id": pattern.tradition_id,
        "palette": pattern.palette,
        "constraint_profile": {
            "symmetry": pattern.constraints.symmetry,
            "frame": pattern.jaali.frame,
        },
    }
    registry = load_model_registry(ROOT / "benchmarks" / "models.json")
    model = registry["beam-hf"]
    with TemporaryDirectory(prefix=".benchmark-audit-", dir=ROOT) as temp_dir:
        prompts_path = Path(temp_dir) / "eval-v2.jsonl"
        prompts_path.write_text(json.dumps(prompt) + "\n", encoding="utf-8")
        row = {
            "benchmark_version": BENCHMARK_VERSION,
            "validator_version": "0.3.0",
            "model_key": "beam-hf",
            "model_id": model["model_id"],
            "provider": model["provider"],
            "api_style": model["api_style"],
            "request_params": model["request_params"],
            "prompt_suite_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
            "prompt_id": prompt["id"],
            "level": prompt["level"],
            "symmetry": prompt["symmetry"],
            "brief": prompt["brief"],
            "state": prompt["state"],
            "grammar": prompt["grammar"],
            "suite_id": prompt["suite_id"],
            "tradition_id": prompt["tradition_id"],
            "attempt_count": 1,
            "max_attempts": 3,
            "attempts": [{
                "index": 1,
                "candidate": candidate,
                "report": {"checks": checks, "all_hard_constraints_pass": True},
            }],
            "valid_schema_at_1": True,
            "validity_at_1": True,
            "validity_at_3": True,
            "accepted_mask_sha256": certificate["mask_sha256"],
            "repair_rescued": False,
        }
        result = audit([row], prompts_path, registry, ["beam-hf"])

        nonstandard_model = dict(model)
        nonstandard_model["request_params"] = {**model["request_params"], "max_tokens": 2048}
        nonstandard_row = {**row, "request_params": nonstandard_model["request_params"]}
        with pytest.raises(ValueError, match="shared 4096-token output limit"):
            audit([nonstandard_row], prompts_path, {"beam-hf": nonstandard_model}, ["beam-hf"])

        nonstandard_temperature = dict(model)
        nonstandard_temperature["request_params"] = {**model["request_params"], "temperature": 0.2}
        nonstandard_temperature_row = {
            **row, "request_params": nonstandard_temperature["request_params"],
        }
        with pytest.raises(ValueError, match="unsupported generation protocol"):
            audit([nonstandard_temperature_row], prompts_path,
                  {"beam-hf": nonstandard_temperature}, ["beam-hf"])

        shorter_attempt_policy = {**row, "max_attempts": 2}
        with pytest.raises(ValueError, match="shared three-proposal attempt limit"):
            audit([shorter_attempt_policy], prompts_path, registry, ["beam-hf"])

        messages = [
            {"role": "system", "content": "Locked system prompt"},
            {"role": "user", "content": "Locked task prompt"},
        ]
        first_model_row = {**row, "model_calls": [{
            "phase": "proposal", "messages": messages,
            "telemetry": {"provider_attempts": 1},
        }]}
        second_model = {
            **model,
            "key": "second-model",
            "provider": "Second Provider",
            "model_id": "second/provider-model",
            "request_params": {
                key: value for key, value in model["request_params"].items()
                if key != "reasoning_effort"
            },
        }
        second_model_row = {
            **first_model_row,
            "model_key": "second-model",
            "model_id": second_model["model_id"],
            "provider": second_model["provider"],
            "request_params": second_model["request_params"],
        }
        comparison_registry = {"beam-hf": model, "second-model": second_model}
        assert audit([first_model_row, second_model_row], prompts_path, comparison_registry,
                     ["beam-hf", "second-model"])["complete"] is True

        anthropic_model = {
            **model,
            "key": "claude-haiku-5-5",
            "provider": "Anthropic",
            "api_style": "anthropic-messages",
            "model_id": "claude-haiku-test",
            "request_params": {"max_tokens": 4096, "output_config": {"effort": "low"}},
        }
        deepseek_model = {
            **model,
            "key": "deepseek-v4.1-flash",
            "provider": "CometAPI",
            "model_id": "deepseek-v4.1-flash",
            "request_params": {
                "temperature": 0.4, "max_tokens": 4096,
                "extra_body": {"thinking": {"type": "disabled"}},
            },
        }
        anthropic_row = {
            **first_model_row,
            "model_key": anthropic_model["key"],
            "model_id": anthropic_model["model_id"],
            "provider": anthropic_model["provider"],
            "api_style": anthropic_model["api_style"],
            "request_params": anthropic_model["request_params"],
        }
        deepseek_row = {
            **first_model_row,
            "model_key": deepseek_model["key"],
            "model_id": deepseek_model["model_id"],
            "provider": deepseek_model["provider"],
            "api_style": deepseek_model["api_style"],
            "request_params": deepseek_model["request_params"],
        }
        explicit_profiles = {
            "beam-hf": model,
            anthropic_model["key"]: anthropic_model,
            deepseek_model["key"]: deepseek_model,
        }
        protocol_report = audit([first_model_row, anthropic_row, deepseek_row], prompts_path,
                                explicit_profiles, list(explicit_profiles))
        assert protocol_report["complete"] is True
        assert protocol_report["generation_protocol_by_model"] == {
            "beam-hf": "temperature_0.4_beam_minimal_reasoning",
            "claude-haiku-5-5": "anthropic_provider_default_sampling_low_effort",
            "deepseek-v4.1-flash": "temperature_0.4_thinking_disabled",
        }
        bad_deepseek = {
            **deepseek_model,
            "request_params": {
                **deepseek_model["request_params"], "temperature": 0.2,
            },
        }
        with pytest.raises(ValueError, match="unsupported generation protocol"):
            audit([deepseek_row], prompts_path,
                  {bad_deepseek["key"]: bad_deepseek}, [bad_deepseek["key"]])

        parameter_model = {
            **second_model,
            "request_params": {**second_model["request_params"], "top_p": 0.8},
        }
        parameter_row = {**second_model_row, "request_params": parameter_model["request_params"]}
        with pytest.raises(ValueError, match="unsupported generation protocol"):
            audit([first_model_row, parameter_row], prompts_path,
                  {"beam-hf": model, "second-model": parameter_model},
                  ["beam-hf", "second-model"])
        retried_row = json.loads(json.dumps(second_model_row))
        retried_row["model_calls"][0]["telemetry"]["provider_attempts"] = 2
        with pytest.raises(ValueError, match="transport retries are disabled"):
            audit([first_model_row, retried_row], prompts_path, comparison_registry,
                  ["beam-hf", "second-model"])
        second_model_row["model_calls"] = [{"phase": "proposal", "messages": [
            messages[0], {"role": "user", "content": "Changed prompt"},
        ], "telemetry": {"provider_attempts": 1}}]
        with pytest.raises(ValueError, match="initial proposal messages differ across models"):
            audit([first_model_row, second_model_row], prompts_path, comparison_registry,
                  ["beam-hf", "second-model"])

    assert result["candidate_replays"] == 1
    assert result["complete"] is True

    mismatched = json.loads(json.dumps(row))
    mismatched["attempts"][0]["candidate"]["tradition_id"] = "another_tradition"
    with TemporaryDirectory(prefix=".benchmark-audit-mismatch-", dir=ROOT) as temp_dir:
        prompts_path = Path(temp_dir) / "eval-v2.jsonl"
        prompts_path.write_text(json.dumps(prompt) + "\n", encoding="utf-8")
        mismatched["prompt_suite_sha256"] = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="locked grammar or tradition"):
            audit([mismatched], prompts_path, registry, ["beam-hf"])


def test_benchmark_audit_preserves_periodic_raster_v01_contract() -> None:
    candidate = json.loads((ROOT / "examples" / "ajrakh_starfield.pattern.json").read_text(encoding="utf-8"))
    pattern = Pattern.model_validate_json(json.dumps(candidate))
    checks, masks, grid = verify_pattern(pattern)
    certificate = make_certificate(pattern, checks, masks, grid)
    prompt = {
        "id": "g1-legacy-easy-0",
        "brief": "A periodic Ajrakh-inspired tile",
        "level": "easy",
        "symmetry": "D4",
        "state": "Rajasthan",
        "grammar": "periodic_raster",
        "suite_id": "eval-v2",
        "tradition_id": "ajrakh",
        "palette": pattern.palette,
        "constraint_profile": {"symmetry": "D4"},
    }
    registry = load_model_registry(ROOT / "benchmarks" / "models.json")
    model = registry["beam-hf"]
    with TemporaryDirectory(prefix=".benchmark-audit-v01-", dir=ROOT) as temp_dir:
        prompts_path = Path(temp_dir) / "eval-v2.jsonl"
        prompts_path.write_text(json.dumps(prompt) + "\n", encoding="utf-8")
        row = {
            "benchmark_version": BENCHMARK_VERSION,
            "validator_version": "0.3.0",
            "model_key": "beam-hf",
            "model_id": model["model_id"],
            "provider": model["provider"],
            "api_style": model["api_style"],
            "request_params": model["request_params"],
            "prompt_suite_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
            "prompt_id": prompt["id"],
            "level": prompt["level"],
            "symmetry": prompt["symmetry"],
            "brief": prompt["brief"],
            "state": prompt["state"],
            "grammar": prompt["grammar"],
            "suite_id": prompt["suite_id"],
            "tradition_id": prompt["tradition_id"],
            "attempt_count": 1,
            "max_attempts": 3,
            "attempts": [{
                "index": 1,
                "candidate": candidate,
                "report": {"checks": checks, "all_hard_constraints_pass": True},
            }],
            "valid_schema_at_1": True,
            "validity_at_1": True,
            "validity_at_3": True,
            "accepted_mask_sha256": certificate["mask_sha256"],
            "repair_rescued": False,
        }
        result = audit([row], prompts_path, registry, ["beam-hf"])

    assert result["candidate_replays"] == 1
    assert result["complete"] is True
