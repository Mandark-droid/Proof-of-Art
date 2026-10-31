"""Publish the prepared evaluation and training datasets to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from huggingface_hub import HfApi, get_token, hf_hub_download
import numpy as np
from PIL import Image
from pramaan.checks.core import all_pass, verify_pattern
from pramaan.dsl import parse_pattern
from pramaan.orchestrate import _run_telemetry
from pramaan.render import render
from pramaan.reward import reward_for_report
from pramaan.tasks import canonical_jsonl_bytes
from pramaan.ui.findings import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
EVAL_REPO = "kshitijthakkar/pramaan-proof-of-art-eval-v2"
EVAL_V21_REPO = "kshitijthakkar/pramaan-proof-of-art-eval-v2.1"
TRAIN_REPO = "kshitijthakkar/pramaan-proof-of-art-train-v1"
EVAL_SUITES = {
    "eval-v2": (ROOT / "benchmarks" / "suites" / "eval-v2.jsonl", EVAL_REPO),
    "eval-v2.1": (ROOT / "benchmarks" / "suites" / "eval-v2.1.jsonl", EVAL_V21_REPO),
}
COMMUNITY_MODEL_PROVIDERS = {
    "claude-haiku-5-5": "Anthropic",
    "claude-sonnet-5-5": "Anthropic",
    "claude-opus-5-5": "Anthropic",
    "claude-fable-5-1": "Anthropic",
    "deepseek-v4.1-flash": "CometAPI",
}
COMMUNITY_API_STYLES = {
    "claude-haiku-5-5": "anthropic-messages",
    "claude-sonnet-5-5": "anthropic-messages",
    "claude-opus-5-5": "anthropic-messages",
    "claude-fable-5-1": "anthropic-messages",
    "deepseek-v4.1-flash": "openai-compatible-chat-completions",
}
MAX_COMMUNITY_UPLOAD_BYTES = 20_000_000
OPENENV_REPO = "kshitijthakkar/pramaan-proof-of-art-openenv"
COMMUNITY_REPO = "kshitijthakkar/pramaan-community-runs"
COMMUNITY_WRITE_TOKEN_ENV = "HF_COMMUNITY_WRITE_TOKEN"
COMMUNITY_PROVIDER_HOSTS = {
    "Anthropic": "api.anthropic.com", "CometAPI": "api.cometapi.com",
    "OpenAI": "api.openai.com", "DeepSeek": "api.deepseek.com",
    "Google Gemini": "generativelanguage.googleapis.com", "Mistral": "api.mistral.ai",
    "Cohere": "api.cohere.ai", "Groq": "api.groq.com", "xAI": "api.x.ai",
    "Together AI": "api.together.xyz", "Fireworks AI": "api.fireworks.ai",
    "Perplexity": "api.perplexity.ai",
}
COMMUNITY_LITELLM_PROVIDERS = {
    "anthropic": "Anthropic", "openai": "OpenAI", "deepseek": "DeepSeek",
    "gemini": "Google Gemini", "mistral": "Mistral", "cohere": "Cohere",
    "groq": "Groq", "xai": "xAI", "together_ai": "Together AI",
    "fireworks_ai": "Fireworks AI", "perplexity": "Perplexity",
}
COMMUNITY_LITELLM_MODEL_RE = re.compile(
    r"(?P<provider>[a-z][a-z0-9_-]{0,31})/(?P<model>[A-Za-z0-9_.:-]+(?:/[A-Za-z0-9_.:-]+)*)"
)


def community_model_identity(model_key):
    """Return trusted provider metadata for legacy keys or strict LiteLLM IDs."""
    if isinstance(model_key, str) and model_key in COMMUNITY_MODEL_PROVIDERS:
        return {
            "provider": COMMUNITY_MODEL_PROVIDERS[model_key], "model_id": model_key,
            "api_style": COMMUNITY_API_STYLES[model_key],
        }
    match = COMMUNITY_LITELLM_MODEL_RE.fullmatch(model_key) if isinstance(model_key, str) else None
    if not match or match.group("provider") not in COMMUNITY_LITELLM_PROVIDERS:
        return None
    model_id = match.group("model")
    if any(part in {".", ".."} for part in model_id.split("/")):
        return None
    return {"provider": COMMUNITY_LITELLM_PROVIDERS[match.group("provider")],
            "model_id": model_id, "api_style": "litellm-chat-completions"}


def _community_safe(value, secret: str | None = None):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [_community_safe(item, secret) for item in value]
    if isinstance(value, dict):
        blocked = {"api_key", "provider_api_key", "authorization", "auth", "headers",
                   "secret", "secret_key", "access_token", "hf_token", "token",
                   "raw", "raw_response", "completion", "prompt_messages", "messages", "provider_error"}
        return {key: _community_safe(item, secret) for key, item in value.items()
                if key.lower().replace("-", "_") not in blocked
                and not key.lower().replace("-", "_").endswith("_api_key")
                and not key.lower().replace("-", "_").endswith("_token")}
    return value


def sanitize_community_eval_rows(rows: list[dict], *, run_id: str,
                                 expected_suite_sha256: str | None = None,
                                 provider_api_key: str | None = None, submitted_by: str | None = None,
                                 suite_path: Path | None = None) -> list[dict]:
    """Whitelist and replay community results before public dataset publication."""
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("run_id must be a safe public identifier")
    if submitted_by is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}", submitted_by):
        raise ValueError("submitted_by must be an authenticated Hugging Face username")
    suite_id = rows[0].get("suite_id") if isinstance(rows, list) and rows and isinstance(rows[0], dict) else None
    if suite_path is None:
        try:
            suite_path = EVAL_SUITES[suite_id][0]
        except KeyError as exc:
            raise ValueError("community run must identify a supported frozen evaluation suite") from exc
    suite_bytes = suite_path.read_bytes()
    suite_sha = hashlib.sha256(canonical_jsonl_bytes(suite_bytes)).hexdigest()
    if expected_suite_sha256 is not None and expected_suite_sha256 != suite_sha:
        raise ValueError("community results must use the digest of their identified frozen suite")
    task_rows = read_jsonl(suite_path)
    tasks = {row["id"]: row for row in task_rows}
    if not isinstance(rows, list) or not rows:
        raise ValueError("community run must contain at least one completed question")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("community run rows must be JSON objects")
    prompt_ids = [row.get("prompt_id") for row in rows]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("community run contains duplicate questions")
    first = rows[0]
    model_key = first.get("model_key")
    identity = community_model_identity(model_key)
    if identity is None:
        raise ValueError("community run model is not on the supported provider list")
    provider, model_id, api_style = identity["provider"], identity["model_id"], identity["api_style"]
    if first.get("model_id") != model_id:
        raise ValueError("community run model id is invalid")
    if api_style == "litellm-chat-completions":
        expected_params = {"temperature": 0.4, "max_tokens": 4096}
    elif api_style == "anthropic-messages":
        expected_params = {"max_tokens": 4096, "output_config": {"effort": "low"}}
    else:
        expected_params = {"temperature": 0.4, "max_tokens": 4096,
                           "extra_body": {"thinking": {"type": "disabled"}}}
    submitted_at = datetime.now(timezone.utc).isoformat()
    safe_rows = []
    from pramaan.checks.core import all_pass, verify_pattern
    from pramaan.dsl import parse_pattern

    for row in rows:
        prompt_id = row.get("prompt_id")
        task = tasks.get(prompt_id)
        if task is None:
            raise ValueError(f"{prompt_id!r} is not a question in the frozen {task_rows[0]['suite_id']} suite")
        if (row.get("model_key") != model_key or row.get("model_id") != model_id
                or row.get("provider") != provider or row.get("prompt_suite_sha256") != suite_sha
                or row.get("api_style") != api_style
                or row.get("request_params") != expected_params):
            raise ValueError("community run mixes providers, settings, or prompt-suite revisions")
        for field in ("suite_id", "state", "tradition_id", "grammar", "level", "brief", "symmetry"):
            if row.get(field) != task.get(field):
                raise ValueError(f"{prompt_id}: {field} differs from frozen {task['suite_id']}")
        attempts = row.get("attempts")
        if not isinstance(attempts, list) or not 1 <= len(attempts) <= 3:
            raise ValueError(f"{prompt_id}: community result must contain 1–3 completed attempts")
        safe_attempts = []
        first_pass = False
        accepted = False
        for expected_index, attempt in enumerate(attempts, 1):
            if not isinstance(attempt, dict) or attempt.get("index") != expected_index:
                raise ValueError(f"{prompt_id}: attempt indexes must be sequential")
            candidate = attempt.get("candidate")
            report = attempt.get("report")
            if (not isinstance(report, dict) or not isinstance(report.get("checks"), list)
                    or not all(isinstance(check, dict) for check in report["checks"])):
                raise ValueError(f"{prompt_id}: attempt report is malformed")
            if candidate is None:
                if not any(check.get("id") == "candidate_parse" and check.get("status") == "FAIL"
                           for check in report["checks"] if isinstance(check, dict)):
                    raise ValueError(f"{prompt_id}: unparsed candidate lacks a parse failure")
                replay_pass = False
            else:
                pattern = parse_pattern(json.dumps(candidate, ensure_ascii=False))
                checks, _, _ = verify_pattern(pattern)
                saved = [check for check in report["checks"] if check.get("id") != "candidate_truncated"]
                if checks != saved:
                    raise ValueError(f"{prompt_id}: verifier checks do not replay")
                truncated = any(check.get("id") == "candidate_truncated" for check in report["checks"])
                replay_pass = all_pass(checks) and not truncated
            if bool(report.get("all_hard_constraints_pass")) != replay_pass:
                raise ValueError(f"{prompt_id}: acceptance does not match verifier replay")
            first_pass = replay_pass if expected_index == 1 else first_pass
            accepted = accepted or replay_pass
            safe_attempts.append(_community_safe({
                "index": expected_index,
                "candidate": candidate,
                "report": report,
                "verified_at": attempt.get("verified_at"),
                "verification_elapsed_ms": attempt.get("verification_elapsed_ms"),
            }, provider_api_key))
        if (bool(row.get("valid_schema_at_1")) != (attempts[0].get("candidate") is not None)
                or bool(row.get("validity_at_1")) != first_pass
                or bool(row.get("validity_at_3")) != accepted):
            raise ValueError(f"{prompt_id}: saved scores do not match replayed attempts")
        usage = row.get("token_usage") if isinstance(row.get("token_usage"), dict) else {}
        cost = row.get("cost") if isinstance(row.get("cost"), dict) else {}
        latency = row.get("latency_ms") if isinstance(row.get("latency_ms"), dict) else {}
        safe_rows.append(_community_safe({
            "community_schema_version": "1.0", "run_id": run_id,
            "data_role": "community_evaluation", "training_eligible": False,
            "source": ("local_community_submission" if row.get("run_mode") == "local_cli"
                       else "public_user_key_evaluation"), "submitted_at": submitted_at,
            "submitted_by": submitted_by,
            "run_mode": row.get("run_mode") if row.get("run_mode") in {"local_cli", "byo_key_ui"}
                        else "byo_key_ui",
            "model_key": model_key, "model_id": model_id,
            "model_label": f"{model_id} (self-reported, unverified)",
            "resolved_model_ids": [value[:128] for value in (row.get("resolved_model_ids") or [])[:5]
                                   if isinstance(value, str)],
            "provider": provider, "provider_host": COMMUNITY_PROVIDER_HOSTS[provider],
            "api_style": row.get("api_style"),
            "request_params": expected_params,
            "prompt_suite_sha256": suite_sha, "suite_id": task["suite_id"],
            "prompt_id": prompt_id, "state": task["state"], "tradition_id": task["tradition_id"],
            "grammar": task["grammar"], "level": task["level"], "brief": task["brief"],
            "symmetry": task["symmetry"], "valid_schema_at_1": bool(row.get("valid_schema_at_1")),
            "validity_at_1": first_pass, "validity_at_3": accepted,
            "repair_rescued": accepted and not first_pass,
            "attempt_count": len(safe_attempts), "max_attempts": 3,
            "accepted_mask_sha256": row.get("accepted_mask_sha256"),
            "elapsed_ms": row.get("elapsed_ms"), "total_tokens": row.get("total_tokens"),
            "token_usage": _community_safe({key: usage.get(key) for key in
                ("prompt_tokens", "cached_prompt_tokens", "completion_tokens", "input_tokens",
                 "output_tokens", "total_tokens", "source", "calls_with_usage", "call_count") if key in usage},
                provider_api_key),
            "cost": _community_safe({key: cost.get(key) for key in
                                      ("amount", "currency", "source", "estimated_amount",
                                       "estimated_currency", "estimate_source", "rate_card")
                                      if key in cost}, provider_api_key),
            "latency_ms": {key: value for key, value in latency.items()
                           if isinstance(value, (int, float)) and not isinstance(value, bool)},
            "provider_attempts": row.get("provider_attempts"),
            "provider_retries": row.get("provider_retries"),
            "validator_version": row.get("validator_version"),
            "output_terms_note": "Governed by the provider terms; evaluation use only",
            "attempts": safe_attempts,
            "run_coverage": {"completed_questions": len(rows), "suite_questions": len(tasks)},
        }, provider_api_key))
    return safe_rows



def community_dataset_card(suite_id: str | None = None) -> str:
    if suite_id is not None and suite_id not in EVAL_SUITES:
        raise ValueError("unsupported community suite")
    return '''---
language:
- en
license: cc-by-4.0
task_categories:
- text-generation
configs:
- config_name: community
  data_files:
  - split: test
    path: community/test/*.jsonl
---

# Pramaan Community Evaluation Runs

Visitor and locally generated evaluations are stored here after deterministic replay against a supported frozen
suite. Each row records its suite identifier and SHA-256; the official evaluation and training datasets are separate.
Model identity is self-reported and unverified. Every published row is forced to `training_eligible: false`;
third-party outputs remain governed by their provider's terms. API keys, raw responses, and prompt message logs
are not published. Space writes require `HF_COMMUNITY_WRITE_TOKEN`, a fine-grained token that can write only this repo.

The benchmark suite, verifier reports, and dataset structure are provided under CC-BY-4.0. This licence does not
change the terms governing third-party model output.
'''

def publish_community_eval_run(rows: list[dict], *, run_id: str,
                               api=None, expected_suite_sha256: str | None = None,
                               provider_api_key: str | None = None, submitted_by: str | None = None) -> dict:
    """Publish one sanitized, uniquely addressed public evaluation run."""
    safe_rows = sanitize_community_eval_rows(rows, run_id=run_id,
                                             expected_suite_sha256=expected_suite_sha256,
                                             provider_api_key=provider_api_key, submitted_by=submitted_by)
    token = os.getenv(COMMUNITY_WRITE_TOKEN_ENV)
    if not token and api is None:
        raise RuntimeError(f"Space {COMMUNITY_WRITE_TOKEN_ENV} secret is required for community publication")
    if not isinstance(submitted_by, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}", submitted_by):
        raise ValueError("public community runs require authenticated Hugging Face submitter attribution")
    hub = api or HfApi(token=token)
    path_in_repo = f"community/test/{run_id}.jsonl"
    submitted_at = safe_rows[0]["submitted_at"]
    stamp = datetime.fromisoformat(submitted_at.replace("Z", "+00:00")).strftime("%Y%m%dT%H%M%S%fZ")
    index_path = f"community/index/{stamp}_{run_id}.json"
    index_payload = json.dumps({"run_id": run_id, "model_key": safe_rows[0]["model_key"],
                                "submitted_at": submitted_at, "row_path": path_in_repo,
                                "completed_questions": len(safe_rows)}, ensure_ascii=False).encode("utf-8")
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in safe_rows).encode("utf-8")
    if len(payload) > MAX_COMMUNITY_UPLOAD_BYTES:
        raise ValueError("community run export exceeds the 20 MB publication limit")
    suite_id = safe_rows[0]["suite_id"]
    repo_id = COMMUNITY_REPO
    try:
        hub.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
        if path_in_repo in hub.list_repo_files(repo_id=repo_id, repo_type="dataset"):
            raise ValueError("community run id already exists; results are append-only")
        hub.upload_file(path_or_fileobj=io.BytesIO(payload), path_in_repo=path_in_repo,
                        repo_id=repo_id, repo_type="dataset",
                        commit_message=f"Add community evaluation {run_id}")
        hub.upload_file(path_or_fileobj=io.BytesIO(index_payload), path_in_repo=index_path,
                        repo_id=repo_id, repo_type="dataset",
                        commit_message=f"Index community evaluation {run_id}")
        hub.upload_file(path_or_fileobj=io.BytesIO(community_dataset_card(suite_id).encode("utf-8")),
                        path_in_repo="README.md", repo_id=repo_id, repo_type="dataset",
                        commit_message="Document community evaluation config")
    except Exception as exc:
        message = str(exc)
        for secret in (token, provider_api_key):
            if secret:
                message = message.replace(secret, "[REDACTED]")
        raise RuntimeError(message[:1000]) from None
    return {"repo_id": repo_id, "path_in_repo": path_in_repo,
            "url": f"https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}",
            "index_path": index_path, "records": len(safe_rows)}



def publish_community_submission_archive(archive_source, *, run_id: str,
                                         api=None, provider_api_key: str | None = None,
                                         expected_suite_sha256: str | None = None,
                                         submitted_by: str | None = None) -> dict:
    """Audit a local ZIP before adding its rows to the separate community dataset."""
    from pramaan.community import audit_submission_archive

    report = audit_submission_archive(archive_source, run_id=run_id,
                                      provider_api_key=provider_api_key)
    if not report.accepted:
        reason = report.reasons[0] if report.reasons else "community submission failed replay audit"
        raise ValueError(reason)
    suite_sha = report.accepted_rows[0]["prompt_suite_sha256"]
    if expected_suite_sha256 is not None and suite_sha != expected_suite_sha256:
        raise ValueError("community submission suite digest differs from the active Space suite")
    return publish_community_eval_run(
        report.accepted_rows, run_id=run_id, api=api,
        expected_suite_sha256=suite_sha, provider_api_key=provider_api_key,
        submitted_by=submitted_by,
    )

def _download_hub_text(path_in_repo: str, *, repo_id: str, api=None) -> str:
    if api is not None and path_in_repo.startswith("community/test/") and hasattr(api, "download_rows"):
        payload = api.download_rows(repo_id=repo_id, repo_type="dataset", path_in_repo=path_in_repo)
        return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
    if api is not None and hasattr(api, "download_file"):
        payload = api.download_file(repo_id=repo_id, repo_type="dataset", path_in_repo=path_in_repo)
        return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=path_in_repo,
                            token=os.getenv("HF_TOKEN") or None)
    return Path(local).read_text(encoding="utf-8")


def load_recent_community_runs(limit: int = 100, *, suite_id: str = "eval-v2.1", api=None) -> list[dict]:
    """Load recent indexed community runs without downloading the append-only corpus."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    try:
        try:
            repo_id = COMMUNITY_REPO
        except KeyError as exc:
            raise ValueError("unsupported community suite") from exc
        hub = api or HfApi(token=os.getenv("HF_COMMUNITY_READ_TOKEN") or os.getenv("HF_COMMUNITY_WRITE_TOKEN") or None)
        paths = hub.list_repo_files(repo_id=repo_id, repo_type="dataset")
        indexes = sorted((path for path in paths
                          if path.startswith("community/index/") and path.endswith(".json")),
                         reverse=True)[:limit]
        runs = []
        for index_path in indexes:
            index = json.loads(_download_hub_text(index_path, repo_id=repo_id, api=hub))
            run_id = index.get("run_id")
            model_key = index.get("model_key")
            row_path = index.get("row_path")
            submitted_at = index.get("submitted_at")
            if (not isinstance(run_id, str) or community_model_identity(model_key) is None
                    or not isinstance(submitted_at, str) or not isinstance(row_path, str)
                    or not row_path.startswith("community/test/") or not row_path.endswith(".jsonl")):
                raise ValueError("invalid community run index")
            text = _download_hub_text(row_path, repo_id=repo_id, api=hub)
            try:
                values = json.loads(text)
            except json.JSONDecodeError:
                values = [json.loads(line) for line in text.splitlines() if line.strip()]
            if isinstance(values, dict):
                values = [values]
            if (not isinstance(values, list) or not values
                    or not all(isinstance(row, dict) for row in values)
                    or any(row.get("run_id") != run_id or row.get("model_key") != model_key for row in values)):
                raise ValueError("community rows do not match their index")
            runs.append({"run_id": run_id, "model_key": model_key, "submitted_at": submitted_at,
                         "rows": _community_safe(values)})
        return runs
    except Exception:
        raise RuntimeError("Unable to load recent community evaluation runs.") from None


def normalize_openenv_transition(row: dict) -> dict:
    """Canonicalize nullable model-id provenance for stable Hub Arrow inference."""
    if not isinstance(row, dict):
        raise ValueError("OpenEnv transition must be an object")
    normalized = dict(row)
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("OpenEnv transition provenance must be an object")
    provenance = dict(provenance)
    resolved_model_ids = provenance.get("resolved_model_ids")
    if resolved_model_ids == []:
        resolved_model_ids = None
    if resolved_model_ids is not None and (
            not isinstance(resolved_model_ids, list)
            or any(not isinstance(value, str) or not value for value in resolved_model_ids)):
        raise ValueError("resolved model IDs must be null or a list of non-empty strings")
    provenance["resolved_model_ids"] = resolved_model_ids
    normalized["provenance"] = provenance
    return normalized


def _openenv_jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                    for row in rows)


def export_openenv_episodes(rows: list[dict], *, suite_path: Path, output_path: Path,
                            role: str) -> dict:
    """Replay benchmark actions through the production environment and export transitions."""
    if role not in {"benchmark_evaluation", "training"}:
        raise ValueError("unsupported OpenEnv dataset role")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output_path}")
    tasks = read_jsonl(suite_path)
    task_by_id = {task["id"]: task for task in tasks}
    suite_sha = hashlib.sha256(canonical_jsonl_bytes(suite_path.read_bytes())).hexdigest()
    if not isinstance(rows, list) or not rows:
        raise ValueError("OpenEnv export needs at least one benchmark result")
    prompt_ids = [row.get("prompt_id") for row in rows]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("OpenEnv export contains duplicate task rows")
    from pramaan_env import PramaanAction
    from pramaan_env.server.environment import PramaanEnvironment

    records = []
    for row in rows:
        prompt_id = row.get("prompt_id")
        task = task_by_id.get(prompt_id)
        if task is None or row.get("prompt_suite_sha256") != suite_sha:
            raise ValueError(f"{prompt_id}: OpenEnv task or suite digest mismatch")
        for key in ("suite_id", "state", "tradition_id", "grammar", "level", "symmetry", "brief"):
            if row.get(key) != task.get(key):
                raise ValueError(f"{prompt_id}: {key} differs from the frozen suite")
        attempts = row.get("attempts")
        if not isinstance(attempts, list) or not 1 <= len(attempts) <= 3:
            raise ValueError(f"{prompt_id}: OpenEnv export needs 1 to 3 sequential attempts")
        resolved_model_ids = row.get("resolved_model_ids")
        if resolved_model_ids == []:
            resolved_model_ids = None
        if resolved_model_ids is not None and (
                not isinstance(resolved_model_ids, list)
                or any(not isinstance(value, str) or not value for value in resolved_model_ids)):
            raise ValueError(f"{prompt_id}: resolved model IDs must be null or a list of non-empty strings")
        episode_id = hashlib.sha256(f"{row.get('model_key')}\0{prompt_id}\0{suite_sha}".encode()).hexdigest()
        env = PramaanEnvironment()
        reset = env.reset(task_id=prompt_id, suite=task["suite_id"], episode_id=episode_id)
        previous = reset.model_dump(mode="json")
        common = {"transition_schema_version": "1.0", "episode_id": episode_id,
                  "data_role": role, "training_eligible": role == "training",
                  "source": "replayed_openenv_benchmark", "prompt_suite_sha256": suite_sha,
                  "prompt_id": prompt_id, "suite_id": task["suite_id"], "state": task["state"],
                  "tradition_id": task["tradition_id"], "grammar": task["grammar"],
                  "level": task["level"], "validator_version": row.get("validator_version"),
                  "provenance": {key: (resolved_model_ids if key == "resolved_model_ids" else row.get(key))
                                 for key in
                                 ("model_key", "model_id", "provider", "api_style", "resolved_model_ids",
                                  "request_params", "pricing")},
                  "episode_usage": {key: row.get(key) for key in
                                    ("token_usage", "cost", "total_tokens", "elapsed_ms")}}
        records.append({**common, "event_type": "reset", "transition_index": 0,
                        "observation": None, "action": None, "next_observation": previous,
                        "reward": None, "reward_version": "pramaan-reward-v1",
                        "checks": [], "terminated": False, "truncated": False, "usage": {}})
        calls = {call.get("call_index"): call for call in row.get("model_calls", [])
                 if isinstance(call, dict)}
        for index, attempt in enumerate(attempts, 1):
            if not isinstance(attempt, dict) or attempt.get("index") != index:
                raise ValueError(f"{prompt_id}: OpenEnv attempt indexes must be sequential")
            source_report = attempt.get("report", {})
            source_truncated = any(check.get("id") == "candidate_truncated"
                                   for check in source_report.get("checks", []) if isinstance(check, dict))
            candidate = attempt.get("candidate")
            action_text = attempt.get("raw")
            pattern_json = (action_text if isinstance(action_text, str) else
                            json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
                            if candidate is not None else "")
            call = calls.get(attempt.get("call_index"), {})
            telemetry = call.get("telemetry", {}) if isinstance(call, dict) else {}
            next_obs = env.step(PramaanAction(pattern_json=pattern_json))
            expected_checks = [check for check in source_report.get("checks", [])
                               if check.get("id") != "candidate_truncated"]
            if next_obs.checks != expected_checks:
                raise ValueError(f"{prompt_id} attempt {index}: OpenEnv verifier checks do not replay")
            expected_reward = reward_for_report(source_report)
            if not source_truncated and next_obs.reward != expected_reward:
                raise ValueError(f"{prompt_id} attempt {index}: OpenEnv reward does not replay")
            current = previous
            next_dump = next_obs.model_dump(mode="json")
            records.append({**common, "event_type": "step", "transition_index": index,
                            "observation": current,
                            "action": {"pattern_json": pattern_json,
                                       "candidate_sha256": hashlib.sha256(pattern_json.encode()).hexdigest()},
                            "next_observation": next_dump, "reward": next_obs.reward,
                            "reward_version": "pramaan-reward-v1", "checks": next_obs.checks,
                            "counterexamples": [check for check in next_obs.checks
                                                if check.get("status") == "FAIL"],
                            "terminated": bool(next_obs.done and next_obs.reward == 1.0),
                            "truncated": bool(next_obs.done and next_obs.reward != 1.0),
                            "source_benchmark_truncated": source_truncated,
                            "source_benchmark_reward": expected_reward,
                            "source_finish_reason": (attempt.get("metadata") or {}).get("finish_reason"),
                            "usage": {"token_usage": telemetry.get("token_usage"),
                                      "total_tokens": ((telemetry.get("token_usage") or {}).get("total_tokens")
                                                       if isinstance(telemetry.get("token_usage"), dict) else None),
                                      "cost": telemetry.get("cost"),
                                      "provider_elapsed_ms": telemetry.get("provider_elapsed_ms")}})
            previous = next_dump
            if next_obs.done and index != len(attempts):
                raise ValueError(f"{prompt_id}: benchmark attempts continue after an OpenEnv terminal state")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"output": str(output_path), "episodes": len(rows), "transitions": len(records),
            "suite_sha256": suite_sha}


def openenv_dataset_card() -> str:
    eval_sha = hashlib.sha256(canonical_jsonl_bytes((ROOT / "benchmarks/suites/eval-v2.jsonl").read_bytes())).hexdigest()
    eval_v21_sha = hashlib.sha256(canonical_jsonl_bytes((ROOT / "benchmarks/suites/eval-v2.1.jsonl").read_bytes())).hexdigest()
    train_sha = hashlib.sha256(canonical_jsonl_bytes((ROOT / "benchmarks/suites/train-v1.jsonl").read_bytes())).hexdigest()
    train_v12_sha = hashlib.sha256(canonical_jsonl_bytes((ROOT / "benchmarks/suites/train-v1.2.jsonl").read_bytes())).hexdigest()
    eval_overall = json.loads((ROOT / "benchmarks/data/eval-v2.1-summary.json").read_text(encoding="utf-8"))[
        "by_model"]["beam-hf"]["overall"]
    return ("---\nlanguage:\n  - en\nlicense: cc-by-4.0\ntask_categories:\n  - reinforcement-learning\n"
            "tags:\n  - openenv\n  - verifiable-rewards\n  - benchmark\n  - generative-art\n  - india\n  - beam\n"
            "pretty_name: Pramaan OpenEnv Episodes and Transitions\nconfigs:\n"
            "  - config_name: transitions-v1.1\n    data_files:\n      - split: test\n        path: transitions-v1.1/test/*.jsonl\n"
            "      - split: train\n        path: transitions-v1.1/train/*.jsonl\n---\n\n"
            "# Pramaan OpenEnv Episodes and Transitions\n\n"
            "## Summary\n\n"
            "Replayable Pramaan evaluation and training episodes stored as schema-normalized OpenEnv JSONL transitions. "
            "The `transitions-v1.1` config has separate `test` (benchmark evaluation) and `train` (training) splits. "
            "Earlier raw shards under `transitions/` remain for provenance and are excluded from this config because "
            "their empty resolved-model ID arrays were inconsistently encoded as null and empty lists.\n\n"
            "## Fields\n\n"
            "Each row records `event_type`, `episode_id`, `transition_index`, `observation`, `action`, "
            "`next_observation`, `checks`, `reward`, `reward_version`, `terminated`, `truncated`, "
            "`prompt_id`, `suite_id`, `state`, `tradition_id`, `grammar`, `level`, `provenance`, `usage`, "
            "`data_role`, and `training_eligible`. Evaluation rows use `data_role=benchmark_evaluation` and "
            "`training_eligible=false`; training rows use `data_role=training` and `training_eligible=true`. "
            "Provider keys and raw provider error bodies are excluded.\n\n"
            "## Splits\n\n"
            "- `test`: replayed benchmark evaluation transitions.\n"
            "- `train`: role-separated training transitions.\n\n"
            "## How to load\n\n"
            "```python\nfrom datasets import load_dataset\n"
            "test_rows = load_dataset(\"kshitijthakkar/pramaan-proof-of-art-openenv\", \"transitions-v1.1\", split=\"test\")\n```\n\n"
            "## Intended use\n\n"
            "Use the evaluation split to inspect or replay the verifier-backed benchmark environment. The training split "
            "is separately labeled for training. Keep their roles separate in analysis and reporting.\n\n"
            "## Limitations\n\n"
            "The records are generated benchmark interactions, not practitioner-authenticated craft documentation. "
            "Rewards measure declared verifier constraints; they do not establish aesthetic quality, cultural authority, "
            "or artisan authenticity. This is a stochastic single-run benchmark record.\n\n"
            "## Version provenance\n\n"
            "Older eval-v2 and train-v1 records are historical and superseded; do not combine versions.\n\n"
            f"Historical eval-v2 prompt SHA-256: `{eval_sha}`. Current eval-v2.1 suite SHA-256: `{eval_v21_sha}`. "
            f"Historical train-v1 prompt SHA-256: `{train_sha}`. Current sanitized train-v1.2 prompt SHA-256: `{train_v12_sha}`; "
            "it has not been evaluated or exported as training data.\n\n"
            f"Current corrected Beam eval-v2.1 baseline: {eval_overall['n']} of 72 tasks; "
            f"schema-valid @1 {eval_overall['valid_schema_at_1']:.1%}, pass@1 {eval_overall['validity_at_1']:.1%}, "
            f"pass@3 {eval_overall['validity_at_3']:.1%}. Estimated cost is USD 0.4382; "
            "provider-reported cost was unavailable.\n")


def publish_openenv_episodes(*, eval_path: Path | None = None, train_path: Path | None = None,
                             api=None) -> dict:
    """Append role-separated OpenEnv transition shards to the public collection dataset."""
    if eval_path is None and train_path is None:
        raise ValueError("provide an evaluation or training transition shard")
    token = os.getenv("HF_TOKEN") or get_token()
    hub = api or HfApi(token=token)
    hub.create_repo(repo_id=OPENENV_REPO, repo_type="dataset", exist_ok=True, private=False)
    existing = set(hub.list_repo_files(repo_id=OPENENV_REPO, repo_type="dataset"))
    uploaded = []
    for role, source in (("benchmark_evaluation", eval_path), ("training", train_path)):
        if source is None:
            continue
        rows = [normalize_openenv_transition(row) for row in read_jsonl(source)]
        for line_number, row in enumerate(rows, 1):
            expected_role = role
            if row.get("data_role") != expected_role or row.get("training_eligible") is not (role == "training"):
                raise ValueError(f"{source}:{line_number}: OpenEnv dataset role metadata is inconsistent")
        split = "test" if role == "benchmark_evaluation" else "train"
        path = f"transitions-v1.1/{split}/{source.name}"
        if path in existing:
            raise ValueError("OpenEnv transition shards are append-only and cannot overwrite an existing file")
        payload = _openenv_jsonl_bytes(rows)
        hub.upload_file(path_or_fileobj=io.BytesIO(payload), path_in_repo=path,
                        repo_id=OPENENV_REPO, repo_type="dataset",
                        commit_message=f"Add OpenEnv {split} transition shard {source.stem}")
        uploaded.append(path)
    card = openenv_dataset_card().encode("utf-8")
    hub.upload_file(path_or_fileobj=io.BytesIO(card), path_in_repo="README.md",
                    repo_id=OPENENV_REPO, repo_type="dataset", commit_message="Document OpenEnv transitions")
    return {"repo_id": OPENENV_REPO, "paths": uploaded,
            "url": f"https://huggingface.co/datasets/{OPENENV_REPO}"}


def read_jsonl(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        rows = payload if isinstance(payload, list) else [payload]
    if not rows:
        raise ValueError(f"{path} is empty")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain JSON objects")
    return rows


def make_suite_trajectory(row: dict, task: dict, role: str) -> dict:
    if role not in {"benchmark_evaluation", "training"}:
        raise ValueError(f"unsupported data role: {role}")
    attempts = row.get("attempts", [])
    by_index = {attempt.get("call_index", attempt.get("index")): attempt for attempt in attempts}
    calls = []
    for source in row.get("model_calls", []):
        call = json.loads(json.dumps(source))
        attempt = by_index.get(call.get("call_index"))
        call["candidate_index"] = attempt.get("index") if attempt and attempt.get("candidate") is not None else None
        call["candidate"] = attempt.get("candidate") if attempt else None
        call["verifier_report"] = attempt.get("report") if attempt else None
        call["verified_at"] = attempt.get("verified_at") if attempt else None
        call["accepted"] = bool(attempt and attempt.get("report", {}).get("all_hard_constraints_pass"))
        call["reward"] = reward_for_report(call["verifier_report"])
        if not isinstance(call.get("messages"), list) or len(call["messages"]) != 2:
            raise ValueError(f"{row.get('prompt_id')}: exact captured request messages are missing")
        calls.append(call)
    if not calls or len(calls) < len(attempts):
        raise ValueError(f"{row.get('prompt_id')}: model calls do not cover all attempts")
    return {
        "trajectory_schema_version": "1.1", "reward_version": "pramaan-reward-v1",
        "trajectory_id": f"{row['model_key']}:{row['prompt_id']}:{row['prompt_suite_sha256'][:12]}",
        "dataset_id": row["suite_id"], "data_role": role,
        "training_eligible": role == "training", "source": "suite_benchmark",
        "model_key": row["model_key"], "model_id": row.get("model_id"),
        "provider": row.get("provider"), "api_style": row.get("api_style"),
        "request_params": row.get("request_params"), "pricing": row.get("pricing"),
        "prompt_suite_sha256": row["prompt_suite_sha256"], "prompt_id": row["prompt_id"],
        "suite_id": row["suite_id"], "state": row["state"], "tradition_id": row["tradition_id"],
        "grammar": row["grammar"], "level": row["level"], "brief": row["brief"],
        "symmetry": row["symmetry"], "constraint_profile": task["constraint_profile"],
        "palette": task["palette"], "validator_version": row["validator_version"],
        "valid_schema_at_1": bool(row.get("valid_schema_at_1")),
        "validity_at_1": bool(row.get("validity_at_1")),
        "validity_at_3": bool(row.get("validity_at_3")),
        "repair_rescued": bool(row.get("repair_rescued")),
        "attempt_count": len(attempts), "max_attempts": row.get("max_attempts", 3),
        "accepted_mask_sha256": row.get("accepted_mask_sha256"),
        "elapsed_ms": row["elapsed_ms"], "created_at": row.get("created_at"),
        "telemetry": _run_telemetry(calls, attempts, row["elapsed_ms"]),
        "provider_error": row.get("provider_error"), "calls": calls,
    }


def compact_suite_row(row: dict, role: str) -> dict:
    compact = json.loads(json.dumps(row))
    compact["data_role"] = role
    compact["training_eligible"] = role == "training"
    compact.pop("model_calls", None)
    for attempt in compact.get("attempts", []):
        attempt.pop("raw", None)
    return compact


def export_previews(trajectories: list[dict], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for record in trajectories:
        if not record.get("validity_at_3"):
            continue
        accepted = next((call for call in reversed(record["calls"]) if call.get("accepted")), None)
        if accepted is None or accepted.get("candidate") is None:
            raise ValueError(f"{record['prompt_id']}: accepted record has no candidate")
        pattern = parse_pattern(json.dumps(accepted["candidate"], ensure_ascii=False))
        _, grid = render(pattern)
        colors = []
        for color in pattern.palette:
            value = color.lstrip("#")
            colors.append(tuple(int(value[i:i + 2], 16) for i in (0, 2, 4)))
        tile = Image.fromarray(np.asarray(colors, dtype=np.uint8)[grid], mode="RGB")
        preview = Image.fromarray(np.tile(np.asarray(tile), (3, 3, 1)), mode="RGB")
        name = "".join(char if char.isalnum() or char in "-_" else "-" for char in record["prompt_id"])
        preview.save(output_dir / f"{name}.png", format="PNG", optimize=True)
        (output_dir / f"{name}.json").write_text(json.dumps({
            "prompt_id": record["prompt_id"], "state": record["state"],
            "tradition_id": record["tradition_id"], "brief": record["brief"],
            "mask_sha256": record.get("accepted_mask_sha256"),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        count += 1
    return count


def export_suite_trajectories(results_path: Path, suite_path: Path, output_path: Path, role: str) -> dict:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output_path}")
    tasks = read_jsonl(suite_path)
    task_by_id = {task["id"]: task for task in tasks}
    suite_sha = hashlib.sha256(canonical_jsonl_bytes(suite_path.read_bytes())).hexdigest()
    rows = read_jsonl(results_path)
    if len(rows) != len(task_by_id) or {row.get("prompt_id") for row in rows} != set(task_by_id):
        raise ValueError("trajectory export requires complete, duplicate-free suite coverage")
    trajectories = []
    replayed = 0
    for row in rows:
        task = task_by_id[row["prompt_id"]]
        if row.get("prompt_suite_sha256") != suite_sha:
            raise ValueError(f"{row['prompt_id']}: result suite digest mismatch")
        for key in ("suite_id", "state", "tradition_id", "grammar", "level", "symmetry", "brief"):
            if row.get(key) != task.get(key):
                raise ValueError(f"{row['prompt_id']}: {key} differs from the frozen suite")
        for attempt in row.get("attempts", []):
            candidate = attempt.get("candidate")
            report = attempt.get("report", {})
            if candidate is None:
                if not any(check.get("id") == "candidate_parse" and check.get("status") == "FAIL"
                           for check in report.get("checks", [])):
                    raise ValueError(f"{row['prompt_id']}: unparsed candidate lacks a parse failure")
                continue
            pattern = parse_pattern(json.dumps(candidate, ensure_ascii=False))
            checks, _, _ = verify_pattern(pattern)
            saved = [check for check in report.get("checks", []) if check.get("id") != "candidate_truncated"]
            if checks != saved:
                raise ValueError(f"{row['prompt_id']} attempt {attempt.get('index')}: verifier replay mismatch")
            truncated = any(check.get("id") == "candidate_truncated" for check in report.get("checks", []))
            if bool(report.get("all_hard_constraints_pass")) != (all_pass(checks) and not truncated):
                raise ValueError(f"{row['prompt_id']}: candidate acceptance does not replay")
            replayed += 1
        trajectories.append(make_suite_trajectory(row, task, role))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in trajectories:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"output": str(output_path), "records": len(trajectories),
            "candidate_replays": replayed, "suite_sha256": suite_sha}


def enforce_data_role(path: Path, repository_kind: str) -> None:
    expected = "benchmark_evaluation" if repository_kind == "eval" else "training"
    for line_number, row in enumerate(read_jsonl(path), 1):
        role = row.get("data_role")
        if role is not None and role != expected:
            raise ValueError(f"{path}:{line_number}: data_role {role!r} conflicts with {repository_kind} repository")
        eligibility = row.get("training_eligible")
        expected_eligibility = repository_kind == "train"
        if role is None and eligibility is not None and eligibility is not expected_eligibility:
            raise ValueError(f"{path}:{line_number}: training_eligible conflicts with {repository_kind} repository")
        if repository_kind == "eval" and role == expected and row.get("training_eligible") is not False:
            raise ValueError(f"{path}:{line_number}: benchmark evaluation rows must set training_eligible=false")
        if repository_kind == "train" and role == expected and row.get("training_eligible") is not True:
            raise ValueError(f"{path}:{line_number}: training rows must explicitly set training_eligible=true")


def validate_eval_v21_rows(path: Path, *, trajectories: bool) -> dict[str, int]:
    """Require the complete ten-model release and the correct row shape."""
    rows = read_jsonl(path)
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, 1):
        model_key, prompt_id = row.get("model_key"), row.get("prompt_id")
        if not isinstance(model_key, str) or not model_key or not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"{path}:{index}: model_key and prompt_id are required")
        identity = (model_key, prompt_id)
        if identity in seen:
            raise ValueError(f"{path}:{index}: duplicate model/task row {identity}")
        seen.add(identity)
        counts[model_key] = counts.get(model_key, 0) + 1
        if (row.get("suite_id") != "eval-v2.1" or row.get("data_role") != "benchmark_evaluation"
                or row.get("training_eligible") is not False):
            raise ValueError(f"{path}:{index}: row is not evaluation-only eval-v2.1 data")
        if trajectories:
            calls = row.get("calls")
            if row.get("trajectory_schema_version") != "1.1" or not isinstance(calls, list) or not calls:
                raise ValueError(f"{path}:{index}: expected a schema 1.1 trajectory with calls")
            if "attempts" in row:
                raise ValueError(f"{path}:{index}: trajectory row unexpectedly contains compact attempts")
            for call in calls:
                if (call.get("phase") not in {"proposal", "repair"}
                        or not isinstance(call.get("messages"), list) or len(call["messages"]) != 2
                        or not isinstance(call.get("metadata"), dict)
                        or not isinstance(call.get("verifier_report"), dict)):
                    raise ValueError(f"{path}:{index}: trajectory call lacks request, model metadata, or verifier evidence")
        elif not isinstance(row.get("attempts"), list) or "calls" in row:
            raise ValueError(f"{path}:{index}: expected compact attempts without trajectory calls")
    if len(rows) != 720 or len(counts) != 10 or any(count != 72 for count in counts.values()):
        raise ValueError(f"{path}: expected 720 rows across ten models (72 each), got {len(rows)} rows: {counts}")
    return dict(sorted(counts.items()))


def validate_eval_v21_pair(compact_path: Path, trajectories_path: Path) -> tuple[dict[str, int], dict[str, int]]:
    if compact_path.resolve() == trajectories_path.resolve():
        raise ValueError("compact and trajectories inputs must be different files")
    if hashlib.sha256(compact_path.read_bytes()).digest() == hashlib.sha256(trajectories_path.read_bytes()).digest():
        raise ValueError("compact and trajectories inputs must contain distinct datasets")
    compact_counts = validate_eval_v21_rows(compact_path, trajectories=False)
    trajectory_counts = validate_eval_v21_rows(trajectories_path, trajectories=True)
    if compact_counts != trajectory_counts:
        raise ValueError("compact and trajectories model/task coverage differs")
    return compact_counts, trajectory_counts


def verify_uploaded_eval_v21(repo_id: str, token: str) -> dict:
    """Force-download resolve/main after upload and re-check both published payloads."""
    compact = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="compact/test.jsonl",
                                   revision="main", force_download=True, token=token))
    trajectories = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="trajectories/test.jsonl",
                                        revision="main", force_download=True, token=token))
    compact_counts, trajectory_counts = validate_eval_v21_pair(compact, trajectories)
    return {
        "compact_rows": sum(compact_counts.values()), "compact_by_model": compact_counts,
        "trajectories_rows": sum(trajectory_counts.values()), "trajectories_by_model": trajectory_counts,
        "compact_sha256": hashlib.sha256(compact.read_bytes()).hexdigest(),
        "trajectories_sha256": hashlib.sha256(trajectories.read_bytes()).hexdigest(),
    }


def dataset_card(repository_kind: str, *, include_community: bool = False,
                 include_references: bool = False, suite_name: str | None = None) -> str:
    if repository_kind == "eval":
        suite_name = suite_name or "eval-v2"
        name = f"Pramaan Proof of Art — Evaluation {suite_name.removeprefix('eval-')}"
        roles = "Benchmark evaluation data. Keep it held out from training, prompt tuning, model selection, and any report that presents the same scores as held-out performance."
        config_names = ["trajectories", "compact", "leaderboard" if suite_name == "eval-v2.1" else "summary"]
        if include_references:
            if suite_name != "eval-v2.1":
                raise ValueError("procedural references are published only with eval-v2.1")
            config_names.append("references")
        repo_id = EVAL_V21_REPO if suite_name == "eval-v2.1" else EVAL_REPO
    else:
        name = "Pramaan Proof of Art — Training v1"
        roles = "Training-eligible records from the separately versioned train-v1 suite. This historical export is superseded; do not reuse it for current training runs."
        config_names = ["trajectories", "sft", "repair", "preference"]
        suite_name = "train-v1"
        repo_id = "kshitijthakkar/pramaan-proof-of-art-train-v1"
    suite_path = ROOT / "benchmarks" / "suites" / f"{suite_name}.jsonl"
    suite_digest = hashlib.sha256(canonical_jsonl_bytes(suite_path.read_bytes())).hexdigest()
    split = "test" if repository_kind == "eval" else "train"
    configs = "\n".join(
        f"  - config_name: {config}\n    data_files:\n      - split: {split}\n        path: {config}/{split}.jsonl"
        for config in config_names
    )
    if repository_kind == "eval" and include_community:
        configs += "\n  - config_name: community\n    data_files:\n      - split: test\n        path: community/test/*.jsonl"
    evaluation_version_note = ""
    if repository_kind == "eval" and suite_name == "eval-v2.1":
        merged = json.loads((ROOT / "benchmarks/data/eval-v2.1-summary.json").read_text(encoding="utf-8"))
        overall = merged["by_model"]["beam-hf"]["overall"]
        model_lines = []
        for key, model in sorted(merged["by_model"].items()):
            metrics = model["overall"]
            label = model.get("label", key)
            model_lines.append(
                f"- {label}: {metrics['n']}/72 tasks; pass@1 {metrics['validity_at_1']:.1%}; "
                f"pass@3 {metrics['validity_at_3']:.1%}."
            )
        fable = merged["by_model"].get("claude-fable-5-1")
        fable_ledger = fable.get("cost_ledger") if fable else None
        common_sha = merged.get("source_suite_digests", {}).get("eval-v2.1-common47", "unavailable")
        delta_sha = merged.get("source_suite_digests", {}).get("eval-v2.1-delta25", "unavailable")
        evaluation_version_note = (
            "This is the current corrected held-out suite. The older eval-v2 is historical and superseded; do not combine its results with eval-v2.1.\n\n"
            f"Current one-run Beam baseline: {overall['n']}/72 tasks, schema-valid @1 {overall['valid_schema_at_1']:.1%}, "
            f"pass@1 {overall['validity_at_1']:.1%}, pass@3 {overall['validity_at_3']:.1%}. Estimated cost USD 0.4382; "
            "provider-reported billed cost was unavailable.\n\n"
            + (
                "## Frozen cross-model report\n\n"
                "Ten model arms were evaluated on the same 72 task records and verifier. The freeze sets a 4,096-token output cap, "
                "up to three proposals per task, and zero provider retries; provider-specific sampling settings are listed per row.\n\n"
                + "\n".join(model_lines) + "\n\n"
                "Claude and DeepSeek results were collected in two sittings: the shared common47 subset, then its exact delta25 complement. "
                "Together these subsets equal the canonical ordered 72-task suite. Rows retain their source `prompt_suite_sha256` and also "
                f"record the canonical suite digest `{suite_digest}`; common47 digest `{common_sha}`, delta25 digest `{delta_sha}`.\n\n"
                + (
                    "Claude Fable 5.1 completed 72/72 under its USD 22 cap. Conservative rate-card exposure was USD "
                    f"{fable_ledger['conservative_exposure_usd']:.6f}; provider-billed cost was unavailable.\n\n"
                    if fable_ledger else ""
                )
                + "Each other paid model arm had an independent USD 10 cap. The full aggregate summary is available as "
                "[`summary/eval-v2.1-summary.json`](https://huggingface.co/datasets/kshitijthakkar/pramaan-proof-of-art-eval-v2.1/blob/main/summary/eval-v2.1-summary.json).\n\n"
                + "Third-party model outputs are evaluation-only, carry `data_role=benchmark_evaluation` and "
                "`training_eligible=false`, and remain governed by their providers' terms. They are not training data.\n\n"
                if fable_ledger and len(merged["by_model"]) == 10 else ""
            )
        )
    elif repository_kind == "eval" and suite_name == "eval-v2":
        evaluation_version_note = "**Historical and superseded by eval-v2.1.** [Open the current dataset](https://huggingface.co/datasets/kshitijthakkar/pramaan-proof-of-art-eval-v2.1).\n\n"
    elif repository_kind == "train" and suite_name == "train-v1":
        current_train_sha = hashlib.sha256(canonical_jsonl_bytes((ROOT / "benchmarks/suites/train-v1.2.jsonl").read_bytes())).hexdigest()
        evaluation_version_note = (
            "**Superseded by the [train-v1.2 prompt suite](https://github.com/Mandark-droid/Proof-of-Art/blob/main/benchmarks/suites/train-v1.2.jsonl).**\n\n"
            f"The corrected train-v1.2 prompts have SHA-256 `{current_train_sha}` but have not yet been evaluated or exported.\n\n"
        )
    return (
        "---\nlanguage:\n  - en\nlicense: cc-by-4.0\ntask_categories:\n  - text-generation\n"
        "tags:\n  - reinforcement-learning\n  - openenv\n  - verifiable-rewards\n  - benchmark\n  - generative-art\n  - india\n  - beam\n"
        f"pretty_name: {name}\nconfigs:\n{configs}\n---\n\n"
        f"# {name}\n\n"
        "## Summary\n\n"
        "Pramaan evaluates geometric patterns from generative models using a deterministic verifier that reports exact declared-constraint outcomes. "
        "Patterns are original designs inspired by South Asian craft vocabularies, not artisan-authentic works or substitutes for practitioners.\n\n"
        + evaluation_version_note
        + "## Fields\n\n"
        "Task and tradition fields include `prompt_id`, `suite_id`, `state`, `tradition_id`, `grammar`, `level`, `brief`, `palette`, `symmetry`, and `constraint_profile`. "
        "Trajectory records include `trajectory_id`, `trajectory_schema_version`, `calls`, `attempt_count`, `max_attempts`, `reward_version` (when present; current runs use `pramaan-reward-v1`), `valid_schema_at_1`, `validity_at_1`, `validity_at_3`, "
        "`data_role`, and `training_eligible`. Nested calls contain proposals and verifier reports; provider credentials and raw provider error bodies are excluded.\n\n"
        "## Splits\n\n"
        + "\n".join(f"- `{config}`: `{split}` split." for config in config_names)
        + ("\n- `community`: `test` split with user-submitted partial runs; these are not included in canonical scores." if repository_kind == "eval" and include_community else "")
        + "\n\n"
        "## How to load\n\n"
        "```python\nfrom datasets import load_dataset\n"
        f"rows = load_dataset(\"{repo_id}\", \"compact\", split=\"{split}\")\n"
        "```\n\n"
        "## Intended use\n\n" + roles + "\n\n"
        + ("## Procedural references\n\n"
           "The `references` config contains verifier-certified procedural examples, with thumbnail PNGs in `references/thumbnails/`. "
           "It includes one or more examples for each of the 36 states and union territories represented by the suite. "
           "Each row has `data_role=reference` and `training_eligible=false`; these are illustrative verifier witnesses, not model results, "
           "not artisan-authentic works, and not benchmark scores. One Puducherry task is unsatisfiable under its frozen constraints and has no accepted reference.\n\n"
           if repository_kind == "eval" and include_references else "")
        + "Evaluation rows use `data_role=benchmark_evaluation` and `training_eligible=false`; training rows use `data_role=training` and `training_eligible=true`. "
        "SFT and repair examples include the full system, user, and assistant turn sequence in `messages`; the assistant text is also retained in `completion` for compatibility. "
        "Preference examples retain the prompt separately and provide full `chosen_messages` and `rejected_messages` conversations. "
        "Repairs remain linked to their task and preceding verifier feedback. Keep the declared suite split and digest attached to any result.\n\n"
        "## Limitations\n\n"
        "The verifier checks declared schema and geometric constraints; it does not measure aesthetic quality, cultural authority, artisan authenticity, or practitioner endorsement. "
        "Model results are stochastic and apply only to the listed suite version and run settings.\n\n"
        f"Prompt suite `{suite_name}` SHA-256: `{suite_digest}`.\n\n"
        "## Attribution and sources\n\n"
        "Beam model: `reflection/Beam-1.0-501B-A23B`.\n\n"
        "Tradition registry source list: see the maintained [citation index](https://github.com/Mandark-droid/Proof-of-Art/blob/main/CITATIONS.md); individual task records retain tradition source metadata.\n\n"
        "See the [trajectory schema and reuse policy](https://github.com/Mandark-droid/Proof-of-Art/blob/main/benchmarks/TRAJECTORY_DATASET.md)."
    )


def _leaderboard_rows(summary_path: Path) -> list[dict]:
    """Flatten the merged summary into small viewer-friendly rows."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    models = summary.get("by_model")
    if not isinstance(models, dict) or len(models) != 10:
        raise ValueError("eval-v2.1 leaderboard export requires all ten model arms")
    rows = []
    for key, document in sorted(models.items()):
        overall = document.get("overall") or {}
        profile = document.get("request_profile") or {}
        params = profile.get("request_params") or {}
        n = overall.get("n")
        if type(n) is not int or n < 0:
            raise ValueError(f"{key}: invalid evaluated task count")
        first = overall.get("passed_at_1")
        if type(first) is not int:
            first = round(float(overall.get("validity_at_1", 0)) * n)
        within = overall.get("accepted_designs")
        if type(within) is not int:
            within = round(float(overall.get("validity_at_3", 0)) * n)
        first_ci = wilson_interval(first, n)
        within_ci = wilson_interval(within, n)
        ledger = document.get("cost_ledger") or {}
        estimated = (overall.get("estimated_cost_totals_by_currency") or {}).get("USD")
        output_limit = params.get("max_tokens", params.get("max_completion_tokens"))
        effort = params.get("reasoning_effort") or (params.get("output_config") or {}).get("effort")
        temperature = params.get("temperature")
        profile_summary = "; ".join(
            value for value in (
                f"output cap {output_limit}" if output_limit else None,
                f"effort {effort}" if effort else None,
                f"temperature {temperature:g}" if isinstance(temperature, (int, float)) else None,
            ) if value
        ) or "provider defaults"
        rows.append({
            "model": key, "label": document.get("label", key), "evaluated": n,
            "first_try": first, "first_try_rate": round(first / n, 6) if n else 0.0,
            "first_try_ci_low": first_ci[0] if first_ci else None,
            "first_try_ci_high": first_ci[1] if first_ci else None,
            "within_three": within, "within_three_rate": round(within / n, 6) if n else 0.0,
            "within_three_ci_low": within_ci[0] if within_ci else None,
            "within_three_ci_high": within_ci[1] if within_ci else None,
            "estimated_cost_usd": estimated, "cap_usd": ledger.get("cap_usd"),
            "stopped_at_cap": bool(document.get("stopped_at_cap") or document.get("cap_reached")),
            "stop_reason": document.get("stop_reason", "complete"),
            "provider_error_count": overall.get("provider_error_count", 0),
            "provider": profile.get("provider", ""), "model_id": profile.get("model_id", ""),
            "api_style": profile.get("api_style", ""), "request_profile_summary": profile_summary,
            "data_role": "benchmark_evaluation", "training_eligible": False,
        })
    return rows


def stage_repository(repository_kind: str, inputs: dict[str, Path], previews: Path | None = None,
                     *, include_community: bool = False, include_references: bool = False,
                     suite_name: str | None = None) -> Path:
    if repository_kind == "eval" and suite_name == "eval-v2.1":
        if "compact" not in inputs or "trajectories" not in inputs:
            raise ValueError("eval-v2.1 publication requires compact and trajectories files")
        validate_eval_v21_pair(inputs["compact"], inputs["trajectories"])
    stage = Path(tempfile.mkdtemp(prefix=f"pramaan-{repository_kind}-"))
    for config, source in inputs.items():
        if repository_kind == "eval" and suite_name == "eval-v2.1" and config == "summary":
            summary_target = stage / "summary" / "eval-v2.1-summary.json"
            summary_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, summary_target)
            leaderboard = stage / "leaderboard" / "test.jsonl"
            leaderboard.parent.mkdir(parents=True, exist_ok=True)
            with leaderboard.open("x", encoding="utf-8", newline="\n") as output:
                for row in _leaderboard_rows(source):
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
            continue
        enforce_data_role(source, repository_kind)
        split = "test" if repository_kind == "eval" else "train"
        target = stage / config / f"{split}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_role = "benchmark_evaluation" if repository_kind == "eval" else "training"
        expected_eligibility = repository_kind == "train"
        with target.open("x", encoding="utf-8") as output:
            for row in read_jsonl(source):
                row.setdefault("data_role", expected_role)
                row.setdefault("training_eligible", expected_eligibility)
                if repository_kind == "train" and config in {"sft", "repair"}:
                    messages = row.get("messages")
                    completion = row.get("completion")
                    if (not isinstance(messages, list) or not messages
                            or not isinstance(completion, str) or not completion):
                        raise ValueError(f"{config} training rows need messages and assistant completion")
                    if messages[-1].get("role") != "assistant":
                        messages = [*messages, {"role": "assistant", "content": completion}]
                        row["messages"] = messages
                    if messages[-1].get("content") != completion:
                        raise ValueError(f"{config} assistant message must match completion")
                if repository_kind == "train" and config == "preference":
                    prompt = row.get("prompt")
                    if not isinstance(prompt, list) or not prompt:
                        raise ValueError("preference rows need a prompt message list")
                    for choice in ("chosen", "rejected"):
                        response = row.get(choice)
                        name = f"{choice}_messages"
                        conversation = row.get(name)
                        expected = [*prompt, {"role": "assistant", "content": response}]
                        if conversation is None:
                            row[name] = expected
                        elif conversation != expected:
                            raise ValueError(f"preference {name} must include its assistant candidate")
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    if repository_kind == "eval" and previews:
        if not previews.is_dir():
            raise ValueError(f"preview directory does not exist: {previews}")
        shutil.copytree(previews, stage / "previews", dirs_exist_ok=True)
    (stage / "README.md").write_text(dataset_card(repository_kind, include_community=include_community,
                                                    include_references=include_references,
                                                    suite_name=suite_name),
                                      encoding="utf-8")
    return stage


def stage_reference_bundle(references_path: Path, suite_path: Path, thumbnails_path: Path,
                           *, include_community: bool = False) -> Path:
    """Stage verified procedural references and require coverage of all 36 suite states."""
    suite = read_jsonl(suite_path)
    references = read_jsonl(references_path)
    suite_digest = hashlib.sha256(canonical_jsonl_bytes(suite_path.read_bytes())).hexdigest()
    suite_by_id = {row.get("id"): row for row in suite if isinstance(row.get("id"), str)}
    expected_states = {row.get("state") for row in suite if isinstance(row.get("state"), str)}
    seen_ids: set[str] = set()
    reference_states: set[str] = set()
    thumbnail_names = {path.stem for path in thumbnails_path.glob("*.png")} if thumbnails_path.is_dir() else set()
    for index, row in enumerate(references, 1):
        task_id = row.get("task_id")
        task = suite_by_id.get(task_id)
        if (not isinstance(task_id, str) or task is None or task_id in seen_ids
                or row.get("suite_id") != "eval-v2.1" or row.get("suite_sha256") != suite_digest
                or row.get("data_role") != "reference" or row.get("training_eligible") is not False
                or row.get("all_hard_constraints_pass") is not True or row.get("state") != task.get("state")):
            raise ValueError(f"reference row {index} is not a unique, verifier-passing eval-v2.1 reference")
        seen_ids.add(task_id)
        reference_states.add(row["state"])
    if not references or len(expected_states) != 36 or reference_states != expected_states:
        raise ValueError("references must cover all 36 states and union territories in eval-v2.1")
    if seen_ids != thumbnail_names:
        raise ValueError("reference thumbnails must match reference task IDs exactly")
    stage = Path(tempfile.mkdtemp(prefix="pramaan-eval-v2.1-references-"))
    target = stage / "references" / "test.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as output:
        for row in references:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    shutil.copytree(thumbnails_path, stage / "references" / "thumbnails")
    (stage / "README.md").write_text(
        dataset_card("eval", suite_name="eval-v2.1", include_references=True,
                     include_community=include_community), encoding="utf-8")
    return stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-trajectories", type=Path)
    parser.add_argument("--eval-compact", type=Path)
    parser.add_argument("--eval-summary", type=Path)
    parser.add_argument("--eval-previews", type=Path)
    parser.add_argument("--train-trajectories", type=Path)
    parser.add_argument("--train-sft", type=Path)
    parser.add_argument("--train-repair", type=Path)
    parser.add_argument("--train-preference", type=Path)
    parser.add_argument("--openenv-eval-transitions", type=Path)
    parser.add_argument("--openenv-train-transitions", type=Path)
    parser.add_argument("--repo", choices=("eval", "eval-v2.1", "train", "both", "openenv"), default="both")
    parser.add_argument("--references-only", action="store_true",
                        help="publish the validated eval-v2.1 references and card without restaging result data")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.references_only:
        token = os.getenv("HF_TOKEN") or get_token()
        if not args.dry_run and not token:
            parser.error("set HF_TOKEN before publishing")
        api = HfApi(token=token) if not args.dry_run else None
        include_community = False
        if api is not None:
            api.create_repo(repo_id=EVAL_V21_REPO, repo_type="dataset", exist_ok=True, private=False)
            existing_files = api.list_repo_files(repo_id=EVAL_V21_REPO, repo_type="dataset")
            include_community = any(path.startswith("community/test/") and path.endswith(".jsonl")
                                    for path in existing_files)
        stage = stage_reference_bundle(ROOT / "benchmarks/data/eval-v2.1-references.jsonl",
                                       ROOT / "benchmarks/suites/eval-v2.1.jsonl",
                                       ROOT / "assets/reference-thumbnails",
                                       include_community=include_community)
        if args.dry_run:
            print(json.dumps({"repo_id": EVAL_V21_REPO, "staged_path": str(stage),
                              "references": 71, "states": 36}, indent=2))
        else:
            api.upload_folder(repo_id=EVAL_V21_REPO, repo_type="dataset", folder_path=str(stage),
                              commit_message="Publish verified references for all 36 states")
            print(f"Published references at https://huggingface.co/datasets/{EVAL_V21_REPO}")
        return
    eval_inputs = (args.eval_trajectories, args.eval_compact, args.eval_summary, args.eval_previews)
    train_inputs = (args.train_trajectories, args.train_sft, args.train_repair, args.train_preference)
    if args.repo in {"eval", "eval-v2.1", "both"} and any(value is None for value in eval_inputs):
        parser.error("eval/both publishing requires all --eval-* paths")
    if args.repo in {"train", "both"} and any(value is None for value in train_inputs):
        parser.error("train/both publishing requires all --train-* paths")
    if args.repo == "openenv" and not (args.openenv_eval_transitions or args.openenv_train_transitions):
        parser.error("openenv publishing requires at least one --openenv-*-transitions path")
    token = os.getenv("HF_TOKEN") or get_token()
    if not args.dry_run and not token:
        parser.error("set HF_TOKEN before publishing")
    jobs = []
    if args.repo in {"eval", "both"}:
        jobs.append(("eval", EVAL_REPO, {"trajectories": args.eval_trajectories,
                     "compact": args.eval_compact, "summary": args.eval_summary}, args.eval_previews, "eval-v2"))
    if args.repo == "eval-v2.1":
        jobs.append(("eval", EVAL_V21_REPO, {"trajectories": args.eval_trajectories,
                     "compact": args.eval_compact, "summary": args.eval_summary}, args.eval_previews, "eval-v2.1"))
    if args.repo in {"train", "both"}:
        jobs.append(("train", TRAIN_REPO, {"trajectories": args.train_trajectories,
                     "sft": args.train_sft, "repair": args.train_repair,
                     "preference": args.train_preference}, None, "train-v1"))
    api = HfApi(token=token) if not args.dry_run else None
    for kind, repo_id, inputs, previews, suite_name in jobs:
        include_community = False
        if api is not None:
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
            if kind == "eval":
                existing_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
                include_community = any(path.startswith("community/test/") and path.endswith(".jsonl")
                                        for path in existing_files)
        stage = stage_repository(kind, inputs, previews, include_community=include_community,
                                 include_references=suite_name == "eval-v2.1",
                                 suite_name=suite_name)
        if args.dry_run:
            print(json.dumps({"repo_id": repo_id, "staged_path": str(stage), "configs": list(inputs)}, indent=2))
            continue
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(stage),
                          commit_message="Add Pramaan dataset configs")
        if suite_name == "eval-v2.1":
            print(json.dumps({"repo_id": repo_id, "revision": "main",
                              **verify_uploaded_eval_v21(repo_id, token)}, sort_keys=True))
        print(f"Published https://huggingface.co/datasets/{repo_id}")
    if args.openenv_eval_transitions or args.openenv_train_transitions:
        if args.dry_run:
            print(json.dumps({"repo_id": OPENENV_REPO, "evaluation": str(args.openenv_eval_transitions),
                              "training": str(args.openenv_train_transitions)}, indent=2))
        else:
            print(json.dumps(publish_openenv_episodes(eval_path=args.openenv_eval_transitions,
                                                      train_path=args.openenv_train_transitions, api=api)))


if __name__ == "__main__":
    main()
