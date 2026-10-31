"""Summarize paired first-proposal and verifier-repair benchmark records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from statistics import median
from pathlib import Path


def _metrics(rows):
    n = len(rows)
    scored = [row for row in rows if row.get("attempt_count", 0) > 0]
    initial_failures = [row for row in scored if not row.get("validity_at_1", False)]
    repair_requested = [
        row for row in initial_failures
        if row.get("attempt_count", 0) > 1
        or (row.get("provider_error") or {}).get("phase") == "repair"
    ]
    rescued = [row for row in repair_requested if row.get("validity_at_3", False)]
    token_rows = [row.get("total_tokens", row.get("telemetry", {}).get("token_usage", {}).get("total_tokens"))
                  for row in rows]
    token_rows = [value for value in token_rows if isinstance(value, (int, float))]
    cached_token_rows = [row.get("telemetry", {}).get("token_usage", {}).get("cached_prompt_tokens")
                         for row in rows]
    cached_token_rows = [value for value in cached_token_rows if isinstance(value, (int, float))]
    request_rows = [row.get("latency_ms", row.get("telemetry", {}).get("latency_ms", {})).get("request_elapsed_total")
                    for row in rows]
    request_rows = [value for value in request_rows if isinstance(value, (int, float))]
    verification_rows = [row.get("verification_elapsed_ms", row.get("telemetry", {}).get("latency_ms", {}).get("verification_total"))
                         for row in rows]
    verification_rows = [value for value in verification_rows if isinstance(value, (int, float))]
    reported_costs = {}
    cost_observations = 0
    estimated_costs = {}
    estimated_cost_observations = 0
    for row in rows:
        cost = row.get("cost") or row.get("telemetry", {}).get("cost") or {}
        amount, currency = cost.get("amount"), cost.get("currency")
        if isinstance(amount, (int, float)) and currency:
            reported_costs[currency] = reported_costs.get(currency, 0) + amount
            cost_observations += 1
        estimated_amount, estimated_currency = cost.get("estimated_amount"), cost.get("estimated_currency")
        if isinstance(estimated_amount, (int, float)) and estimated_currency:
            estimated_costs[estimated_currency] = estimated_costs.get(estimated_currency, 0) + estimated_amount
            estimated_cost_observations += 1
    api_latencies = [
        attempt["metadata"]["provider_elapsed_ms"]
        for row in rows for attempt in row.get("attempts", [])
        if attempt.get("metadata", {}).get("provider_elapsed_ms") is not None
    ]
    api_latencies.extend(
        call["telemetry"]["latency_ms"]["provider_reported"]
        for row in rows for call in row.get("calls", [])
        if call.get("telemetry", {}).get("latency_ms", {}).get("provider_reported") is not None
    )
    candidates = [attempt for row in rows for attempt in row.get("attempts", [])]
    candidates.extend(
        {"report": call.get("verifier_report") or {}}
        for row in rows for call in row.get("calls", []) if call.get("candidate") is not None
    )
    failure_count = sum(
        check.get("status") != "PASS"
        for attempt in candidates for check in attempt.get("report", {}).get("checks", [])
    )
    accepted_hashes = {
        row["accepted_mask_sha256"] for row in rows
        if row.get("validity_at_3") and row.get("accepted_mask_sha256")
    }
    return {
        "n": n,
        "scored_n": len(scored),
        "valid_schema_at_1": (sum(bool(row.get("valid_schema_at_1")) for row in scored)
                              / len(scored) if scored else None),
        "validity_at_1": sum(bool(row.get("validity_at_1")) for row in rows) / n if n else None,
        "validity_at_3": sum(bool(row.get("validity_at_3")) for row in rows) / n if n else None,
        "initial_failures": len(initial_failures),
        "repair_requested": len(repair_requested),
        "repair_rescued": len(rescued),
        "repair_rescue_rate": len(rescued) / len(repair_requested) if repair_requested else None,
        "provider_error_count": sum(row.get("provider_error") is not None for row in rows),
        "mean_elapsed_ms": sum(row["elapsed_ms"] for row in rows) / n if n else None,
        "mean_request_elapsed_ms_per_prompt": sum(request_rows) / len(request_rows) if request_rows else None,
        "mean_verification_elapsed_ms_per_prompt": sum(verification_rows) / len(verification_rows) if verification_rows else None,
        "median_provider_elapsed_ms": median(api_latencies) if api_latencies else None,
        "provider_retries": sum(row.get("provider_retries", row.get("telemetry", {}).get("provider_retries", 0)) or 0
                                 for row in rows),
        "reported_cost_totals_by_currency": reported_costs,
        "reported_cost_observations": cost_observations,
        "estimated_cost_totals_by_currency": estimated_costs,
        "estimated_cost_observations": estimated_cost_observations,
        "mean_failures_per_candidate": failure_count / len(candidates) if candidates else None,
        "mean_tokens_when_available": sum(token_rows) / len(token_rows) if token_rows else None,
        "mean_cached_prompt_tokens_when_available": (sum(cached_token_rows) / len(cached_token_rows)
                                                       if cached_token_rows else None),
        "token_observations": len(token_rows),
        "accepted_designs": sum(bool(row.get("validity_at_3")) for row in rows),
        "unique_accepted_masks": len(accepted_hashes),
    }


def summarize(records, expected_prompt_ids=None, expected_model_keys=None):
    """Summarize independently per model; never pool model quality scores."""
    records = list(records)
    if any("prompt_id" not in record for record in records):
        raise ValueError("every record must include prompt_id")
    model_keys = [record.get("model_key") or record.get("model_id") or "unspecified" for record in records]
    pairs = [(key, record["prompt_id"]) for key, record in zip(model_keys, records)]
    duplicates = sorted(pair for pair, count in Counter(pairs).items() if count > 1)
    if duplicates:
        formatted = ", ".join(f"{model}/{prompt}" for model, prompt in duplicates)
        raise ValueError(f"duplicate model/prompt records: {formatted}")
    expected = set(expected_prompt_ids) if expected_prompt_ids is not None else None
    expected_keys = set(expected_model_keys) if expected_model_keys is not None else None
    grouped = {}
    for model_key, record in zip(model_keys, records):
        grouped.setdefault(model_key, []).append(record)
    keys = sorted(set(grouped) | (expected_keys or set()))
    by_model = {}
    for model_key in keys:
        model_rows = grouped.get(model_key, [])
        observed = {record["prompt_id"] for record in model_rows}
        coverage = {
            "expected_prompt_count": len(expected) if expected is not None else None,
            "observed_prompt_count": len(observed),
            "missing_prompt_ids": sorted(expected - observed) if expected is not None else [],
            "unexpected_prompt_ids": sorted(observed - expected) if expected is not None else [],
            "complete": (observed == expected) if expected is not None else None,
        }
        grouped_metrics = {}
        for field in ("state", "grammar", "level"):
            values = sorted({record.get(field, "unknown") or "unknown" for record in model_rows})
            grouped_metrics[f"by_{field}"] = {
                value: _metrics([record for record in model_rows
                                 if (record.get(field) or "unknown") == value])
                for value in values
            }
        by_model[model_key] = {"study": coverage, "overall": _metrics(model_rows), **grouped_metrics}
    active_keys = [model_key for model_key in keys if grouped.get(model_key)]
    common_ids = (set.intersection(*(
        {record["prompt_id"] for record in grouped[model_key]}
        for model_key in active_keys
    )) if active_keys else set())
    common_rows = {
        model_key: _metrics([record for record in grouped[model_key]
                             if record["prompt_id"] in common_ids])
        for model_key in active_keys
    }
    result = {
        "by_model": by_model,
        "common_completed": {
            "model_keys": active_keys,
            "prompt_count": len(common_ids),
            "prompt_ids": sorted(common_ids),
            "by_model": common_rows,
        },
    }
    # Keep the convenient single-model shape, while avoiding misleading pooled scores.
    if len(by_model) == 1:
        result.update(next(iter(by_model.values())))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--prompts", type=Path, default=Path(__file__).with_name("prompts.jsonl"))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json-output", type=Path, help="Optional path for a machine-readable summary copy")
    parser.add_argument("--model", action="append", dest="model_keys",
                        help="Expected model key; repeat for multi-model coverage accounting")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = [json.loads(line)["id"] for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        result = summarize(rows, expected, args.model_keys)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["model_key", "state", "grammar", "level", *_metrics([]).keys()]
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for model_key, model_result in result["by_model"].items():
                writer.writerow({"model_key": model_key, "level": "all", **model_result["overall"]})
                for dimension in ("state", "grammar", "level"):
                    for value, metrics in model_result[f"by_{dimension}"].items():
                        writer.writerow({"model_key": model_key, dimension: value, **metrics})


if __name__ == "__main__":
    main()
