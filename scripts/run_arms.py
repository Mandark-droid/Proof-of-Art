"""Preflight and launch one isolated benchmark process per model arm.

The default is a no-provider-call preflight. Paid runs require ``--execute``;
each child receives its own result file and spend ledger so concurrent arms
cannot overwrite or share reservations.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import run as runner
from pramaan.tasks import canonical_jsonl_bytes

DEFAULT_MODELS = (
    "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1",
    "deepseek-v4.1-flash",
)


def project_cost_from_usage(usage: dict, per_million_tokens: dict) -> float:
    """Estimate USD from provider-style input/output usage and a rate card."""
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    if any(type(value) is not int or value < 0 for value in (input_tokens, output_tokens)):
        raise ValueError("usage must provide non-negative integer input and output token counts")
    input_rate, output_rate = per_million_tokens.get("input"), per_million_tokens.get("output")
    if any(not isinstance(rate, (int, float)) or isinstance(rate, bool)
           or not math.isfinite(rate) or rate < 0 for rate in (input_rate, output_rate)):
        raise ValueError("pricing must provide finite non-negative input and output rates")
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def effective_model_cap(config: dict, requested_cap: float | None) -> float:
    """Return the requested cap, rejecting any attempt to exceed the frozen cap."""
    budget = config.get("budget") if isinstance(config, dict) else None
    configured = budget.get("cap_usd") if isinstance(budget, dict) else None
    if not isinstance(configured, (int, float)) or isinstance(configured, bool):
        raise ValueError("model has no configured per-model spend cap")
    if requested_cap is None:
        return float(configured)
    if (not isinstance(requested_cap, (int, float)) or isinstance(requested_cap, bool)
            or not math.isfinite(requested_cap) or requested_cap <= 0):
        raise ValueError("requested cap must be a positive finite USD amount")
    if requested_cap > configured:
        raise ValueError(f"requested cap cannot exceed the configured per-model cap of ${configured:.2f}")
    return float(requested_cap)


def build_delta_suite(full_suite: Path, common_suite: Path, destination: Path) -> str:
    """Write the exact ordered rows in eval-v2.1 that are absent from common47."""
    full_lines = [line for line in full_suite.read_text(encoding="utf-8").splitlines() if line.strip()]
    common_lines = [line for line in common_suite.read_text(encoding="utf-8").splitlines() if line.strip()]
    full = [json.loads(line) for line in full_lines]
    common = [json.loads(line) for line in common_lines]
    if len(full) != 72 or len({row.get("id") for row in full}) != 72:
        raise ValueError("full eval-v2.1 must contain 72 unique tasks")
    if len(common) != 47 or len({row.get("id") for row in common}) != 47:
        raise ValueError("common47 must contain 47 unique tasks")
    full_by_id = {row["id"]: row for row in full}
    if any(full_by_id.get(row.get("id")) != row for row in common):
        raise ValueError("common47 contains a task that differs from eval-v2.1")
    common_ids = {row["id"] for row in common}
    delta_lines = [line for line, row in zip(full_lines, full) if row["id"] not in common_ids]
    if len(delta_lines) != 25:
        raise ValueError("the ordered difference from common47 must contain 25 tasks")
    payload = ("\n".join(delta_lines) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return __import__("hashlib").sha256(canonical_jsonl_bytes(payload)).hexdigest()


def _load_plan(models_path: Path, suite_path: Path, selected: list[str],
               ledger_dir: Path, requested_cap: float | None,
               prior_ledger_dir: Path | None = None) -> tuple[dict, dict, str]:
    registry = runner.load_model_registry(models_path, include_paid=True)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("select one or more unique model keys")
    configs = {}
    for key in selected:
        config = registry.get(key)
        if config is None or not config.get("budget_provider"):
            raise ValueError(f"{key} is not a configured budgeted model")
        cap = effective_model_cap(config, requested_cap)
        group = config.get("budget_group", key)
        for previous in configs.values():
            if previous.get("budget_group", previous["key"]) == group and previous["budget"]["cap_usd"] != cap:
                raise ValueError(f"all selected models in {group} must use the same cap")
        configs[key] = {**config, "budget_group": group,
                        "budget": {**config["budget"], "cap_usd": cap}}
    suite_bytes = suite_path.read_bytes()
    prompts = [json.loads(line) for line in suite_bytes.decode("utf-8").splitlines() if line.strip()]
    artifact = runner.validate_paid_eval_suite(prompts)
    if (artifact is None or any(prompt.get("suite_id") != "eval-v2.1" for prompt in prompts)
            or any(not isinstance(prompt.get("id"), str) for prompt in prompts)):
        raise ValueError("paid arms require a canonical eval-v2.1, common47, or delta25 suite")
    digest = __import__("hashlib").sha256(canonical_jsonl_bytes(suite_bytes)).hexdigest()
    ledger_dir.mkdir(parents=True, exist_ok=True)
    group_caps = {}
    for config in configs.values():
        group = config["budget_group"]
        cap = config["budget"]["cap_usd"]
        if group in group_caps and group_caps[group] != cap:
            raise ValueError(f"models sharing budget group {group} must use one cap")
        group_caps[group] = cap
    ledgers = {group: runner.CostLedger(ledger_dir / f"{group}.spend.jsonl", caps={group: cap})
               for group, cap in group_caps.items()}
    prior_ledger_dir = prior_ledger_dir or ROOT / "benchmarks" / "results" / "arm-ledgers"
    all_group_models = {}
    for config in runner.PAID_MODELS:
        group = config.get("budget_group", config["key"])
        all_group_models.setdefault(group, []).append(config["key"])
    for group, ledger in ledgers.items():
        if ledger.path.exists():
            continue
        for key in all_group_models.get(group, []):
            source = prior_ledger_dir / f"{key}.spend.jsonl"
            if source.exists():
                ledger.import_prior_ledger(
                    source, source_provider=key, budget_key=group, model_key=key,
                    scope="aborted-prior-scope",
                )
    preflight = runner.preflight_selected_models(
        configs, {key: len(prompts) for key in configs},
        _CombinedLedger(ledgers),
    )
    preflight.update({
        "suite_path": str(suite_path), "suite_sha256": digest,
        "suite_id": prompts[0].get("suite_id"),
        "protocol_artifact": artifact,
        "task_count": len(prompts),
        "ordered_task_ids": [prompt["id"] for prompt in prompts],
        "arms": {key: {"provider": config["provider"], "model_id": config["model_id"],
                       "request_params": config["request_params"],
                       "cap_usd": config["budget"]["cap_usd"]}
                 for key, config in configs.items()},
    })
    return registry, preflight, digest


class _CombinedLedger:
    """Read-only facade over the shared per-budget-group ledgers."""

    def __init__(self, ledgers: dict):
        self.ledgers = ledgers

    def committed_usd(self, key: str, *, model_key=None) -> float:
        return self.ledgers[key].committed_usd(key, model_key=model_key)

    def committed_by_scope_usd(self, key: str) -> dict:
        return self.ledgers[key].committed_by_scope_usd(key)


def arm_command(*, models_path: Path, suite_path: Path, output_dir: Path,
                ledger_dir: Path, model_key: str, budget_key: str, cap_usd: float,
                resume: bool = False, via_env: bool = False) -> list[str]:
    output_path = output_dir / f"{model_key}.jsonl"
    ledger_path = ledger_dir / f"{budget_key}.spend.jsonl"
    command = [sys.executable, "-m", "benchmarks.run",
               "--suite", str(suite_path), "--models", str(models_path),
               "--model", model_key, "--output", str(output_path),
               "--ledger", str(ledger_path), "--cap-usd", str(cap_usd),
               "--managed-paid-run"]
    if resume:
        command.append("--resume")
    if via_env:
        command.append("--via-env")
    return command


def launch_parallel(commands: list[list[str]], *, cwd: Path = ROOT) -> int:
    """Launch each arm concurrently; return non-zero if any child fails."""
    processes = [subprocess.Popen(command, cwd=cwd) for command in commands]
    statuses = [process.wait() for process in processes]
    return int(any(status != 0 for status in statuses))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "benchmarks" / "models.json")
    parser.add_argument("--suite", type=Path, default=ROOT / "benchmarks" / "suites" / "eval-v2.1-common47.jsonl")
    parser.add_argument("--model", action="append", dest="models_selected",
                        help="Paid model arm; repeat to select multiple arms")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks" / "results" / "eval-v2.1-common47-arms")
    parser.add_argument("--ledger-dir", type=Path, default=ROOT / "benchmarks" / "results" / "eval-v2.1-common47-ledgers")
    parser.add_argument("--prior-ledger-dir", type=Path, default=ROOT / "benchmarks" / "results" / "arm-ledgers",
                        help="Read-only source of prior-scope usage to count against each model cap")
    parser.add_argument("--cap-usd", type=float, help="Lower each selected model's independent cap")
    parser.add_argument("--execute", action="store_true", help="Run the paid arms in parallel after printing preflight")
    parser.add_argument("--resume", action="store_true", help="Resume each arm from its existing result and ledger")
    parser.add_argument("--via-env", action="store_true", help="Replay each proposal through PramaanEnvironment")
    args = parser.parse_args(argv)
    selected = args.models_selected or list(DEFAULT_MODELS)
    try:
        registry, plan, _digest = _load_plan(
            args.models, args.suite, selected, args.ledger_dir, args.cap_usd, args.prior_ledger_dir)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute:
        print("Preflight only; no provider requests were made. Pass --execute on the authorized run date to launch arms.")
        return 0
    runnable = [key for key in selected if
                plan.get("assumptions", {}).get(key, {}).get("conservative_complete_prompts_remaining", 0) > 0]
    skipped = [key for key in selected if key not in runnable]
    if skipped:
        print("Skipping model(s) whose cap cannot reserve one complete task: " + ", ".join(skipped))
    if not runnable:
        parser.error("no selected model can reserve one complete task within its remaining per-model cap")
    missing = sorted({registry[key]["api_key_env"] for key in runnable
                      if not runner.configured_token(registry[key]["api_key_env"])})
    if missing:
        parser.error("missing credentials for selected model(s): " + ", ".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = [arm_command(
        models_path=args.models, suite_path=args.suite, output_dir=args.output_dir,
        ledger_dir=args.ledger_dir, model_key=key,
        budget_key=registry[key].get("budget_group", key),
        cap_usd=effective_model_cap(registry[key], args.cap_usd),
        resume=args.resume, via_env=args.via_env,
    ) for key in runnable]
    return launch_parallel(commands)


if __name__ == "__main__":
    raise SystemExit(main())
