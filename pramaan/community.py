"""Replay-based audit interface for externally produced community evaluation rows."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AuditReport:
    accepted_rows: list[dict] = field(default_factory=list)
    rejected_rows: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return bool(self.accepted_rows) and not self.rejected_rows


def audit_trajectory_rows(rows: list[dict], *, run_id: str,
                          expected_suite_sha256: str | None = None,
                          provider_api_key: str | None = None,
                          suite_path=None) -> AuditReport:
    """Validate a complete community run; never trust client scores or pass flags."""
    from scripts.push_hub_datasets import sanitize_community_eval_rows

    values = rows if isinstance(rows, list) else []
    try:
        safe = sanitize_community_eval_rows(
            values, run_id=run_id, expected_suite_sha256=expected_suite_sha256,
            provider_api_key=provider_api_key, suite_path=suite_path,
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        reason = str(exc)
        if provider_api_key:
            reason = reason.replace(provider_api_key, "[REDACTED]")
        reason = reason[:300] or "community audit rejected the submission"
        return AuditReport(rejected_rows=[{"prompt_id": row.get("prompt_id")
                                          if isinstance(row, dict) else None,
                                          "reason": reason} for row in values] or
                           [{"prompt_id": None, "reason": reason}], reasons=[reason])
    return AuditReport(accepted_rows=safe)


def audit_submission_archive(archive_source, *, run_id: str,
                             provider_api_key: str | None = None) -> AuditReport:
    """Check a pack_submission ZIP manifest and replay its rows before publication."""
    import hashlib
    import io
    import json
    import zipfile
    from pathlib import Path

    from scripts.push_hub_datasets import MAX_COMMUNITY_UPLOAD_BYTES

    try:
        if isinstance(archive_source, (str, Path)):
            payload = Path(archive_source).read_bytes()
        elif isinstance(archive_source, bytes):
            payload = archive_source
        else:
            payload = archive_source.read()
        if not isinstance(payload, bytes) or not payload or len(payload) > MAX_COMMUNITY_UPLOAD_BYTES:
            raise ValueError("submission archive is empty or exceeds the 20 MB limit")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != {"manifest.json", "results.jsonl"}:
                raise ValueError("submission archive must contain exactly manifest.json and results.jsonl")
            if sum(info.file_size for info in archive.infolist()) > MAX_COMMUNITY_UPLOAD_BYTES:
                raise ValueError("submission archive expands beyond the 20 MB limit")
            manifest_bytes = archive.read("manifest.json")
            rows_bytes = archive.read("results.jsonl")
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != "pramaan-community-submission-1":
            raise ValueError("unsupported community submission manifest")
        if hashlib.sha256(rows_bytes).hexdigest() != manifest.get("results_sha256"):
            raise ValueError("submission result digest does not match its manifest")
        rows = [json.loads(line) for line in rows_bytes.decode("utf-8").splitlines() if line.strip()]
        for row in rows:
            if isinstance(row, dict):
                row.setdefault("run_mode", "local_cli")
        if (not rows or manifest.get("row_count") != len(rows)
                or manifest.get("suite_id") != rows[0].get("suite_id")
                or manifest.get("suite_sha256") != rows[0].get("prompt_suite_sha256")
                or manifest.get("prompt_ids") != [row.get("prompt_id") for row in rows]
                or manifest.get("model_keys") != sorted({row.get("model_key") for row in rows})
                or manifest.get("training_eligible") is not False):
            raise ValueError("submission manifest does not match its result rows")
        return audit_trajectory_rows(rows, run_id=run_id,
                                     expected_suite_sha256=manifest.get("suite_sha256"),
                                     provider_api_key=provider_api_key)
    except (OSError, UnicodeError, json.JSONDecodeError, zipfile.BadZipFile,
            KeyError, TypeError, ValueError) as exc:
        reason = str(exc)[:300] or "community submission archive is invalid"
        return AuditReport(rejected_rows=[{"prompt_id": None, "reason": reason}], reasons=[reason])
