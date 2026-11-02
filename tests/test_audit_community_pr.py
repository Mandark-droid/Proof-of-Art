from __future__ import annotations

import json
import zipfile

from pramaan.community import AuditReport
from scripts import audit_community_pr


def _archive(path, *, model_key="lab/ornament-model"):
    manifest = {
        "schema_version": "pramaan-community-submission-1",
        "model_keys": [model_key],
    }
    row = {"model_key": model_key, "model_id": "ornament-model"}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("results.jsonl", json.dumps(row) + "\n")
    return path


def test_local_zip_is_audited_read_only_and_reports_rows_and_self_reported_label(tmp_path, monkeypatch, capsys):
    archive = _archive(tmp_path / "submission.zip")
    calls = []

    def fake_audit(source, **kwargs):
        calls.append((source, kwargs))
        return AuditReport(accepted_rows=[{
            "prompt_id": "task-1", "model_key": "lab/ornament-model",
            "model_label": "ornament-model (self-reported, unverified)",
        }])

    monkeypatch.setattr(audit_community_pr, "audit_submission_archive", fake_audit)
    monkeypatch.setattr(audit_community_pr, "_download_pr_archive", lambda *_: (_ for _ in ()).throw(AssertionError("network used")))

    exit_code = audit_community_pr.run_source(str(archive))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert calls == [(archive, {"run_id": "community-local", "provider_api_key": None})]
    assert "PASS" in output
    assert "rows=1" in output and "accepted=1" in output and "rejected=0" in output
    assert "ornament-model (self-reported, unverified)" in output


def test_pull_request_uses_read_only_downloader_and_reports_failed_row_counts(tmp_path, monkeypatch, capsys):
    archive = _archive(tmp_path / "pr-submission.zip")
    calls = []

    def fake_download(number):
        calls.append(number)
        return archive

    def fake_audit(source, **kwargs):
        assert source == archive
        assert kwargs == {"run_id": "community-pr-27", "provider_api_key": None}
        return AuditReport(
            rejected_rows=[{"prompt_id": "task-1", "reason": "replay mismatch"}],
            reasons=["replay mismatch"],
        )

    monkeypatch.setattr(audit_community_pr, "_download_pr_archive", fake_download)
    monkeypatch.setattr(audit_community_pr, "audit_submission_archive", fake_audit)

    exit_code = audit_community_pr.run_source("27")

    output = capsys.readouterr().out
    assert exit_code == 1
    assert calls == [27]
    assert "FAIL" in output
    assert "rows=1" in output and "accepted=0" in output and "rejected=1" in output
    assert "lab/ornament-model" in output


def test_report_reasons_redact_provider_credentials():
    secret = "sk-ant-api03-" + "x" * 30
    safe = audit_community_pr._safe_reason(f"replay failed with {secret}")
    assert secret not in safe
    assert "[REDACTED]" in safe
