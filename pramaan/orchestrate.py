from __future__ import annotations

import json
import copy
import re
from datetime import datetime, timezone
from itertools import combinations
import math
from numbers import Real
import time
from typing import TYPE_CHECKING, Protocol

from pydantic import ValidationError

from .beam_client import BeamClient
from .certificate import make_certificate
from .checks.core import all_pass, verify_pattern
from .dsl import Constraints, ConstraintsV2, Pattern, PatternV2, parse_pattern
from .prompts import (
    SYSTEM_PROMPT, _legacy_task_profile, design_prompt, design_prompt_for_task,
    repair_prompt, repair_prompt_for_task, system_prompt_for_task, _task_tile_profile,
)
from .normalize import normalize_model_text

if TYPE_CHECKING:
    from .tasks import Task

MAX_BRIEF_CHARS = 500
MAX_ATTEMPTS = 3
TRAJECTORY_SCHEMA_VERSION = "1.0"
_ERROR_SECRET_RE = re.compile(r"(?i)\b(?:sk-[A-Za-z0-9_-]{10,}|hf_[A-Za-z0-9]{10,})\b|\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_ERROR_LOCAL_PATH_RE = re.compile(r"(?i)(?:\b[A-Z]:\\[^\s\"']+|/(?:home|Users|tmp|workspace|root|mnt)/[^\s\"']+)")


class CompletionClient(Protocol):
    """Minimal model interface required by the deterministic workflow."""

    def complete(self, system: str, prompt: str) -> tuple[str, dict]: ...


def require_integer_control(value, name: str) -> int:
    """Accept integral Gradio slider numbers without silently truncating API input."""
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or not float(value).is_integer()):
        raise ValueError(f"{name} must be a whole number")
    return int(value)


def _parse_candidate(raw: str, immutable_constraints: Constraints, allowed_palette: list[str], exclusive_ink_layers: bool) -> Pattern:
    normalized_raw, _ = normalize_model_text(raw)
    parsed = json.loads(normalized_raw)
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be one JSON object")
    # The UI-owned profile is authoritative; model output cannot weaken it.
    parsed["constraints"] = immutable_constraints.model_dump(mode="json")
    parsed["palette"] = allowed_palette
    parsed["background_index"] = 0
    pattern = Pattern.model_validate_json(json.dumps(parsed))
    if exclusive_ink_layers:
        pattern.constraints.exclusive_layer_pairs = [tuple(pair) for pair in combinations([layer.id for layer in pattern.layers], 2)]
    return pattern


def _task_constraint_profile(task) -> dict:
    profile = dict(getattr(task, "constraint_profile", {}) or {})
    grammar = getattr(task, "grammar", None)
    if grammar == "periodic_raster":
        legacy = _legacy_task_profile(task)
        return Constraints.model_validate({key: legacy[key] for key in Constraints.model_fields}).model_dump(mode="json")
    allowed = set(ConstraintsV2.model_fields)
    values = {key: value for key, value in profile.items() if key in allowed}
    symmetry = getattr(task, "symmetry", values.get("symmetry", "D4"))
    values["symmetry"] = symmetry if symmetry in {"None", "D2", "D4", "C2", "C4"} else "D2"
    return ConstraintsV2.model_validate(values).model_dump(mode="json")


def apply_host_profile(candidate: dict, task: "Task") -> dict:
    """Return a candidate with task-owned identity, palette, background, and constraints applied."""
    if not isinstance(candidate, dict):
        raise ValueError("Model response must be one JSON object")
    result = copy.deepcopy(candidate)
    grammar = getattr(task, "grammar", None)
    if grammar not in {"periodic_raster", "dot_field", "stripe_band", "counted_grid",
                       "rotational_field", "kolam_loop", "jaali_lattice"}:
        raise ValueError(f"unsupported task grammar: {grammar!r}")
    result["palette"] = list(getattr(task, "palette"))
    result["background_index"] = 0
    result["constraints"] = _task_constraint_profile(task)
    task_profile = dict(getattr(task, "constraint_profile", {}) or {})
    result["tile"] = _task_tile_profile(task)
    if task_profile.get("exclusive_ink_layers"):
        layer_ids = [item.get("id") for item in result.get("layers", []) if isinstance(item, dict)]
        result["constraints"]["exclusive_layer_pairs"] = [list(pair) for pair in combinations(layer_ids, 2)]
    if grammar == "periodic_raster":
        # G1 deliberately retains the frozen v0.1 prompt and candidate contract.
        result["schema_version"] = "0.1"
        result["style"] = "ajrakh_inspired"
        result.pop("grammar", None)
        result.pop("tradition_id", None)
        result.pop("kolam", None)
        result.pop("jaali", None)
    else:
        result["schema_version"] = "0.2"
        result["grammar"] = grammar
        result["tradition_id"] = getattr(task, "tradition_id")
        if grammar == "jaali_lattice" and "frame" in task_profile:
            jaali = result.get("jaali")
            if isinstance(jaali, dict):
                jaali["frame"] = task_profile["frame"]
                result["jaali"] = jaali
    return result


def _parse_task_candidate(raw: str, task) -> Pattern | PatternV2:
    try:
        normalized_raw, _ = normalize_model_text(raw)
        candidate = json.loads(normalized_raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid PatternDSL JSON: {exc}") from exc
    parsed = parse_pattern(json.dumps(apply_host_profile(candidate, task)))
    if getattr(task, "grammar") == "periodic_raster" and not isinstance(parsed, Pattern):
        raise ValueError("periodic_raster tasks require the frozen PatternDSL v0.1 candidate shape")
    if getattr(task, "grammar") != "periodic_raster" and not isinstance(parsed, PatternV2):
        raise ValueError("non-G1 tasks require PatternDSL v0.2")
    return parsed


def _provider_error(exc: Exception, phase: str, *, sensitive_values=()) -> dict:
    """Return bounded diagnostics with credentials, prompts, paths, and refusals removed."""
    error_type = getattr(exc, "provider_error_type", type(exc).__name__)
    result = {"phase": phase,
              "type": error_type,
              "status_code": getattr(exc, "status_code", None)}
    category = getattr(exc, "category", None)
    is_refusal = error_type == "AnthropicRefusalError" or type(exc).__name__ == "AnthropicRefusalError"
    if not is_refusal:
        message = str(exc)
        for value in sensitive_values:
            if isinstance(value, str) and len(value) >= 8:
                message = message.replace(value, "[REDACTED_PROMPT]")
        message = _ERROR_SECRET_RE.sub("[REDACTED_KEY]", message)
        message = _ERROR_LOCAL_PATH_RE.sub("[LOCAL_PATH]", message)
        if message:
            result["message"] = message[:300]
    budget_guard = getattr(exc, "budget_guard", None)
    if isinstance(budget_guard, dict):
        result["budget_guard"] = {
            key: budget_guard[key] for key in ("reservation_id", "released", "released_amount_usd", "elapsed_ms")
            if key in budget_guard
        }
    if (is_refusal and isinstance(category, str)
            and len(category) <= 64 and re.fullmatch(r"[A-Za-z0-9_.-]+", category)):
        result["refusal_category"] = category
    return result


def _number(value):
    return value if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)) else None


def _call_telemetry(metadata: dict, request_elapsed_ms: int | None) -> dict:
    usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else {}
    cost = metadata.get("cost")
    if cost is None:
        cost = usage.get("cost")
    amount = currency = None
    cost_source = "not_reported"
    if isinstance(cost, dict):
        amount = _number(cost.get("amount", cost.get("value", cost.get("total"))))
        currency = cost.get("currency", metadata.get("cost_currency", usage.get("cost_currency", usage.get("currency"))))
        if amount is not None:
            cost_source = "provider_reported"
    elif _number(cost) is not None:
        amount = _number(cost)
        currency = metadata.get("cost_currency", usage.get("cost_currency", usage.get("currency")))
        cost_source = "provider_reported"
    if amount is None:
        amount = _number(metadata.get("cost_usd", usage.get("cost_usd")))
        if amount is not None:
            currency, cost_source = "USD", "provider_reported"
    pricing = metadata.get("pricing") if isinstance(metadata.get("pricing"), dict) else None
    estimated_amount = None
    estimate_source = "not_configured"
    if pricing:
        rates = pricing.get("per_million_tokens", {})
        input_rate = _number(rates.get("input"))
        output_rate = _number(rates.get("output"))
        cached_rate = _number(rates.get("cached_input"))
        prompt_tokens = _number(usage.get("prompt_tokens"))
        completion_tokens = _number(usage.get("completion_tokens"))
        cached_tokens = _number((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
        if input_rate is not None and output_rate is not None and prompt_tokens is not None and completion_tokens is not None:
            cached_tokens = min(prompt_tokens, cached_tokens or 0)
            uncached_tokens = max(0, prompt_tokens - cached_tokens)
            effective_cached_rate = cached_rate if cached_rate is not None else input_rate
            estimated_amount = (
                uncached_tokens * input_rate + cached_tokens * effective_cached_rate
                + completion_tokens * output_rate
            ) / 1_000_000
            estimate_source = "published_rate_card_estimate"
        else:
            estimate_source = "missing_rate_or_token_usage"
    return {
        "latency_ms": {
            "request_elapsed": _number(request_elapsed_ms),
            "provider_reported": _number(metadata.get("provider_elapsed_ms")),
        },
        "token_usage": {
            "prompt_tokens": _number(usage.get("prompt_tokens")),
            "cached_prompt_tokens": _number((usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
            "completion_tokens": _number(usage.get("completion_tokens")),
            "total_tokens": _number(usage.get("total_tokens")),
            "source": "provider_usage" if usage else "not_reported",
        },
        "cost": {
            "amount": amount, "currency": currency, "source": cost_source,
            "estimated_amount": estimated_amount,
            "estimated_currency": pricing.get("currency") if pricing else None,
            "estimate_source": estimate_source,
            "rate_card": pricing,
        },
        "provider_attempts": metadata.get("provider_attempts"),
    }


def _run_telemetry(model_calls: list[dict], attempts: list[dict], elapsed_ms: int) -> dict:
    call_telemetry = [call.get("telemetry", {}) for call in model_calls]

    def total_tokens(name):
        values = [item.get("token_usage", {}).get(name) for item in call_telemetry]
        values = [value for value in values if isinstance(value, Real)]
        return int(sum(values)) if values else None

    costs = [item.get("cost", {}) for item in call_telemetry]
    reported_costs = [item for item in costs if item.get("amount") is not None and item.get("currency")]
    currencies = {item["currency"] for item in reported_costs}
    amount_without_currency = any(item.get("amount") is not None and not item.get("currency") for item in costs)
    if reported_costs and len(reported_costs) == len(costs) and len(currencies) == 1 and not amount_without_currency:
        cost = {"amount": sum(item["amount"] for item in reported_costs),
                "currency": next(iter(currencies)), "source": "provider_reported",
                "reported_calls": len(reported_costs)}
    else:
        cost = {"amount": None, "currency": next(iter(currencies)) if len(currencies) == 1 else None,
                "source": "mixed_currencies" if len(currencies) > 1 else
                         "provider_reported_currency_missing" if amount_without_currency else
                         "incomplete_provider_reporting" if reported_costs else "not_reported",
                "reported_calls": len(reported_costs),
                "reported_subtotal": sum(item["amount"] for item in reported_costs) if reported_costs else None}
    estimates = [item for item in costs if item.get("estimated_amount") is not None
                 and item.get("estimated_currency")]
    estimate_currencies = {item["estimated_currency"] for item in estimates}
    if estimates and len(estimates) == len(costs) and len(estimate_currencies) == 1:
        cost["estimated_amount"] = sum(item["estimated_amount"] for item in estimates)
        cost["estimated_currency"] = next(iter(estimate_currencies))
        cost["estimate_source"] = "published_rate_card_estimate"
    else:
        cost["estimated_amount"] = None
        cost["estimated_currency"] = next(iter(estimate_currencies)) if len(estimate_currencies) == 1 else None
        cost["estimate_source"] = ("mixed_rate_cards" if len(estimate_currencies) > 1 else
                                   "incomplete_rate_card_or_usage" if estimates else
                                   "not_configured")
    cost["estimated_calls"] = len(estimates)
    request_latencies = [item.get("latency_ms", {}).get("request_elapsed") for item in call_telemetry]
    provider_latencies = [item.get("latency_ms", {}).get("provider_reported") for item in call_telemetry]
    request_latencies = [value for value in request_latencies if isinstance(value, Real)]
    provider_latencies = [value for value in provider_latencies if isinstance(value, Real)]
    verification = [item.get("verification_elapsed_ms") for item in attempts]
    verification = [value for value in verification if isinstance(value, Real)]
    provider_attempts = [item.get("provider_attempts") for item in call_telemetry]
    provider_attempts = [value for value in provider_attempts if isinstance(value, Real)]
    tokens_complete = all(item.get("token_usage", {}).get("total_tokens") is not None
                          for item in call_telemetry) and bool(call_telemetry)
    provider_attempts_complete = all(item.get("provider_attempts") is not None
                                     for item in call_telemetry) and bool(call_telemetry)
    return {
        "elapsed_ms": elapsed_ms,
        "latency_ms": {
            "request_elapsed_total": int(sum(request_latencies)) if request_latencies else None,
            "provider_reported_total": int(sum(provider_latencies)) if provider_latencies else None,
            "verification_total": int(sum(verification)) if verification else None,
        },
        "token_usage": {"prompt_tokens": total_tokens("prompt_tokens"),
                        "cached_prompt_tokens": total_tokens("cached_prompt_tokens"),
                        "completion_tokens": total_tokens("completion_tokens"),
                        "total_tokens": total_tokens("total_tokens"),
                        "source": "provider_usage" if tokens_complete else "partial_or_not_reported",
                        "calls_with_usage": sum(item.get("token_usage", {}).get("total_tokens") is not None
                                                 for item in call_telemetry),
                        "call_count": len(call_telemetry)},
        "cost": cost,
        "provider_attempts": int(sum(provider_attempts)) if provider_attempts and provider_attempts_complete else None,
        "provider_retries": int(sum(provider_attempts) - len(provider_attempts)) if provider_attempts and provider_attempts_complete else None,
    }


def generate_and_verify(brief: str = "", palette: list[str] | None = None, symmetry: str = "D4", max_colours: int = 4,
                        min_coverage: float = 0.02, max_coverage: float = 0.70,
                        exclusive_ink_layers: bool = False, min_gap: int = 0,
                        max_attempts: int = MAX_ATTEMPTS, client: CompletionClient | None = None,
                        task: "Task | None" = None) -> dict:
    if task is not None:
        brief = getattr(task, "brief")
        palette = list(getattr(task, "palette"))
    elif palette is None:
        raise ValueError("Palette is required")
    if not isinstance(brief, str):
        raise ValueError("Pattern brief must be text")
    brief = brief.strip()
    if not brief:
        raise ValueError("Pattern brief cannot be empty")
    if len(brief) > MAX_BRIEF_CHARS:
        raise ValueError(f"Pattern brief exceeds the {MAX_BRIEF_CHARS}-character limit")
    if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_ATTEMPTS:
        raise ValueError(f"max_attempts must be an integer between 1 and {MAX_ATTEMPTS}")
    if task is None:
        profile = Constraints(symmetry=symmetry, max_colours=max_colours, max_primitives=100,
                              min_gap=min_gap, min_coverage=min_coverage, max_coverage=max_coverage,
                              exclusive_layer_pairs=[])
        profile_json = {**profile.model_dump(mode="json"), "exclusive_ink_layers": exclusive_ink_layers}
        prompt = design_prompt(brief, palette, profile_json)
        system_prompt = SYSTEM_PROMPT
    else:
        profile_json = _task_constraint_profile(task)
        task_profile = getattr(task, "constraint_profile", {}) or {}
        if getattr(task, "grammar") == "jaali_lattice" and "frame" in task_profile:
            profile_json = {**profile_json, "frame": task_profile["frame"]}
        if getattr(task, "grammar") == "periodic_raster":
            profile_json = {**profile_json, "exclusive_ink_layers": False}
        prompt = design_prompt_for_task(task)
        system_prompt = system_prompt_for_task(task)
    attempts = []
    model_calls = []
    run_started = time.perf_counter()
    accepted = None
    provider_error = None
    client = client or BeamClient()

    def call_model(phase: str, request_prompt: str) -> tuple[str | None, dict]:
        started_at = datetime.now(timezone.utc).isoformat()
        call_started = time.perf_counter()
        call = {
            "call_index": len(model_calls) + 1,
            "phase": phase,
            "started_at": started_at,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request_prompt},
            ],
        }
        try:
            response, metadata = client.complete(system_prompt, request_prompt)
            call.update({"raw_response": response, "metadata": metadata, "error": None})
            return response, metadata
        except Exception as exc:
            error = _provider_error(exc, phase, sensitive_values=(system_prompt, request_prompt, brief))
            call.update({"raw_response": None, "metadata": {}, "error": error})
            call["_provider_telemetry"] = {
                "provider_attempts": getattr(exc, "provider_attempts", None),
                "provider_elapsed_ms": getattr(exc, "provider_elapsed_ms", None),
            }
            return None, {}
        finally:
            request_elapsed_ms = round((time.perf_counter() - call_started) * 1000)
            call["completed_at"] = datetime.now(timezone.utc).isoformat()
            telemetry_metadata = dict(call.get("metadata", {}))
            if call.get("error"):
                telemetry_metadata.update(call.pop("_provider_telemetry", {}))
            call["telemetry"] = _call_telemetry(telemetry_metadata, request_elapsed_ms)
            model_calls.append(call)

    raw, meta = call_model("proposal", prompt)
    if raw is None:
        provider_error = model_calls[-1]["error"]
    for index in range(max_attempts if raw is not None else 0):
        verification_started = time.perf_counter()
        parsed = None
        normalized_raw, normalizations = normalize_model_text(raw)
        model_calls[-1]["normalizations"] = list(normalizations)
        try:
            if task is None:
                parsed = _parse_candidate(normalized_raw, profile, palette, exclusive_ink_layers)
            else:
                parsed = _parse_task_candidate(normalized_raw, task)
            checks, masks, grid = verify_pattern(parsed)
            certificate = make_certificate(parsed, checks, masks, grid)
            report = {"checks": checks, "all_hard_constraints_pass": all_pass(checks)}
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            checks, masks, grid, certificate = [], {}, None, None
            report = {"checks": [{"id": "candidate_parse", "status": "FAIL", "assurance": "EXACT_DISCRETE", "message": str(exc)}], "all_hard_constraints_pass": False}
        if meta.get("finish_reason") == "length":
            report["checks"].append({
                "id": "candidate_truncated", "status": "FAIL", "assurance": "NOT_VERIFIED",
                "message": "The model reached the completion-token limit; this response cannot be certified as complete.",
            })
            report["all_hard_constraints_pass"] = False
        verification_elapsed_ms = round((time.perf_counter() - verification_started) * 1000)
        attempts.append({"index": index + 1, "call_index": len(model_calls), "verified_at": datetime.now(timezone.utc).isoformat(), "verification_elapsed_ms": verification_elapsed_ms, "raw": raw, "normalizations": normalizations, "candidate": parsed.model_dump(mode="json") if parsed else None,
                         "report": report, "metadata": meta})
        if parsed and report["all_hard_constraints_pass"]:
            accepted = {"pattern": parsed, "certificate": certificate, "masks": masks, "grid": grid}
            break
        if index + 1 < max_attempts:
            previous = json.dumps(parsed.model_dump(mode="json"), sort_keys=True) if parsed else raw[:8000]
            failures = report["checks"]
            try:
                repair = (repair_prompt(brief, profile_json, previous, failures) if task is None else
                          repair_prompt_for_task(task, previous, failures))
                raw, meta = call_model("repair", repair)
                if raw is None:
                    provider_error = model_calls[-1]["error"]
                    break
            except Exception as exc:
                # Keep this guard for unexpected local errors outside provider invocation.
                provider_error = _provider_error(exc, "repair")
                break

    elapsed_ms = round((time.perf_counter() - run_started) * 1000)
    result = {"trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
              "attempts": attempts, "model_calls": model_calls, "accepted": accepted is not None,
              "model": {"model_id": meta.get("model_id"), "provider": meta.get("provider")} if meta else None,
              "created_at": datetime.now(timezone.utc).isoformat(), "elapsed_ms": elapsed_ms,
              "telemetry": _run_telemetry(model_calls, attempts, elapsed_ms), "constraint_profile": profile_json,
              "provider_error": provider_error}
    if task is not None:
        result.update({
            "task_id": getattr(task, "id"), "state": getattr(task, "state"),
            "tradition_id": getattr(task, "tradition_id"), "grammar": getattr(task, "grammar"),
            "suite_id": getattr(task, "suite_id"), "level": getattr(task, "level"),
        })
    if accepted:
        result.update(accepted)
    return result
