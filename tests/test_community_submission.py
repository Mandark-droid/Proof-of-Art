import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import run as runner
from pramaan.tasks import load_suite


def _failed_row(task, digest):
    provider, params, style = (
        ("Anthropic", {"max_tokens": 4096, "output_config": {"effort": "low"}}, "anthropic-messages")
        if task.id.startswith("jh_") else
        ("CometAPI", {"temperature": 0.4, "max_tokens": 4096,
                      "extra_body": {"thinking": {"type": "disabled"}}},
         "openai-compatible-chat-completions")
    )
    model = "claude-haiku-5-5" if provider == "Anthropic" else "deepseek-v4.1-flash"
    return {
        "model_key": model, "model_id": model, "provider": provider, "api_style": style,
        "request_params": params, "prompt_suite_sha256": digest, "prompt_id": task.id,
        "suite_id": task.suite_id, "state": task.state, "tradition_id": task.tradition_id,
        "grammar": task.grammar, "level": task.level, "brief": task.brief, "symmetry": task.symmetry,
        "valid_schema_at_1": False, "validity_at_1": False, "validity_at_3": False,
        "attempts": [{"index": 1, "candidate": None, "report": {
            "all_hard_constraints_pass": False,
            "checks": [{"id": "candidate_parse", "status": "FAIL"}],
        }}],
    }


def test_community_audit_returns_replay_checked_nontraining_rows():
    from pramaan.community import audit_trajectory_rows

    tasks, digest = load_suite("eval-v2.1")
    task = tasks[0]
    report = audit_trajectory_rows([_failed_row(task, digest)], run_id="audit-1",
                                   expected_suite_sha256=digest)

    assert report.accepted is True
    assert report.rejected_rows == []
    assert report.accepted_rows[0]["training_eligible"] is False
    assert report.accepted_rows[0]["provider_host"] in {"api.anthropic.com", "api.cometapi.com"}


def test_community_audit_rejects_stale_suite_and_tampered_candidate():
    from pramaan.community import audit_trajectory_rows

    tasks, digest = load_suite("eval-v2.1")
    stale = audit_trajectory_rows([_failed_row(tasks[0], digest)], run_id="audit-stale",
                                  expected_suite_sha256="0" * 64)
    assert stale.accepted is False
    assert stale.rejected_rows
    assert "digest" in stale.reasons[0].lower()


def test_pack_submission_writes_rows_and_hash_manifest_without_overwrite(tmp_path):
    from scripts.pack_submission import pack_submission

    tasks, digest = load_suite("eval-v2.1")
    rows_path = tmp_path / "run.jsonl"
    rows_path.write_text(json.dumps(_failed_row(tasks[0], digest)) + "\n", encoding="utf-8")
    archive = tmp_path / "submission.zip"

    manifest = pack_submission(rows_path, archive)

    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {"manifest.json", "results.jsonl"}
        packed = bundle.read("results.jsonl")
        metadata = json.loads(bundle.read("manifest.json"))
    assert hashlib.sha256(packed).hexdigest() == manifest["results_sha256"]
    assert metadata["suite_id"] == "eval-v2.1"
    assert metadata["suite_sha256"] == digest
    assert metadata["row_count"] == 1
    with pytest.raises(FileExistsError):
        pack_submission(rows_path, archive)


def test_pack_submission_accepts_lf_copy_of_frozen_suite(tmp_path, monkeypatch):
    from scripts import pack_submission as packer

    tasks, digest = load_suite("eval-v2.1")
    suite_path = tmp_path / "eval-v2.1.jsonl"
    source = Path("benchmarks/suites/eval-v2.1.jsonl").read_bytes()
    suite_path.write_bytes(source.replace(b"\r\n", b"\n"))
    monkeypatch.setitem(packer.EVAL_SUITES, "eval-v2.1", (suite_path, "fixture/eval-v2.1"))
    rows_path = tmp_path / "run-lf.jsonl"
    rows_path.write_text(json.dumps(_failed_row(tasks[0], digest)) + "\n", encoding="utf-8")

    manifest = packer.pack_submission(rows_path, tmp_path / "run-lf.zip")

    assert manifest["suite_sha256"] == digest


def test_pack_submission_rejects_secret_bearing_rows_and_mixed_suite(tmp_path):
    from scripts.pack_submission import pack_submission

    tasks, digest = load_suite("eval-v2.1")
    rows_path = tmp_path / "private.jsonl"
    row = _failed_row(tasks[0], digest)
    row["provider_api_key"] = "sk-private"
    rows_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="private|credential|secret"):
        pack_submission(rows_path, tmp_path / "private.zip")


def test_submission_archive_audit_checks_manifest_and_replays_before_acceptance(tmp_path):
    from pramaan.community import audit_submission_archive
    from scripts.pack_submission import pack_submission

    tasks, digest = load_suite("eval-v2.1")
    rows_path = tmp_path / "run.jsonl"
    rows_path.write_text(json.dumps(_failed_row(tasks[0], digest)) + "\n", encoding="utf-8")
    archive = tmp_path / "submission.zip"
    pack_submission(rows_path, archive)

    report = audit_submission_archive(archive, run_id="upload-1")

    assert report.accepted is True
    assert report.accepted_rows[0]["training_eligible"] is False
    assert report.accepted_rows[0]["provider_host"] == "api.cometapi.com"


def test_archive_publication_audits_first_and_writes_to_community_repo(tmp_path, monkeypatch):
    from scripts.push_hub_datasets import COMMUNITY_REPO, publish_community_submission_archive
    from scripts.pack_submission import pack_submission

    tasks, digest = load_suite("eval-v2.1")
    rows_path = tmp_path / "run.jsonl"
    rows_path.write_text(json.dumps(_failed_row(tasks[0], digest)) + "\n", encoding="utf-8")
    archive_path = tmp_path / "submission.zip"
    pack_submission(rows_path, archive_path)
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "hf-write-secret")

    class FakeHub:
        def __init__(self): self.uploads = []
        def create_repo(self, **kwargs): pass
        def list_repo_files(self, **kwargs): return []
        def upload_file(self, **kwargs):
            self.uploads.append({**kwargs, "data": kwargs["path_or_fileobj"].read()})

    hub = FakeHub()
    result = publish_community_submission_archive(archive_path, run_id="bundle-1", api=hub, submitted_by="test_user")
    rows_upload = next(item for item in hub.uploads if item["path_in_repo"].endswith(".jsonl"))
    row = json.loads(rows_upload["data"])

    assert result["repo_id"] == COMMUNITY_REPO
    assert rows_upload["repo_id"] == COMMUNITY_REPO
    assert row["training_eligible"] is False
    assert row["run_mode"] == "local_cli"
    assert row["model_label"].endswith("self-reported, unverified)")
    assert "hf-write-secret" not in rows_upload["data"].decode()


def test_archive_publication_rejects_manifest_tampering_before_upload(tmp_path):
    from scripts.push_hub_datasets import publish_community_submission_archive
    from scripts.pack_submission import pack_submission

    tasks, digest = load_suite("eval-v2.1")
    rows_path = tmp_path / "run.jsonl"
    rows_path.write_text(json.dumps(_failed_row(tasks[0], digest)) + "\n", encoding="utf-8")
    archive_path = tmp_path / "submission.zip"
    pack_submission(rows_path, archive_path)
    raw = archive_path.read_bytes()
    changed = raw.replace(b"self-reported", b"different-aa")
    if changed == raw:
        changed = raw[:-4] + b"xxxx"

    with pytest.raises(ValueError):
        publish_community_submission_archive(changed, run_id="bundle-bad", api=object())


def test_space_archive_callback_requires_sign_in_and_attributes_replayed_submission(tmp_path, monkeypatch):
    from pramaan import app
    from pramaan.community import audit_submission_archive
    from scripts.pack_submission import pack_submission

    tasks, digest = load_suite("eval-v2.1")
    rows_path = tmp_path / "run.jsonl"
    rows_path.write_text(json.dumps(_failed_row(tasks[0], digest)) + "\n", encoding="utf-8")
    source_archive = tmp_path / "submission.zip"
    pack_submission(rows_path, source_archive)
    upload_root = tmp_path / "gradio"
    upload_root.mkdir()
    valid_upload = upload_root / "valid.zip"
    valid_upload.write_bytes(source_archive.read_bytes())
    tampered_upload = upload_root / "tampered.zip"
    with zipfile.ZipFile(source_archive) as source, zipfile.ZipFile(tampered_upload, "w") as changed:
        for name in source.namelist():
            payload = source.read(name)
            changed.writestr(name, payload + b" " if name == "results.jsonl" else payload)
    monkeypatch.setattr(app.gr.utils, "get_upload_folder", lambda: str(upload_root))
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "scoped-test-token")
    monkeypatch.setenv("SYSTEM", "spaces")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client-id")

    publish_calls = []
    def publish(archive_path, *, run_id, expected_suite_sha256, submitted_by):
        report = audit_submission_archive(archive_path, run_id=run_id)
        if (not report.accepted
                or report.accepted_rows[0]["prompt_suite_sha256"] != expected_suite_sha256):
            reason = report.reasons[0] if report.reasons else "active suite digest mismatch"
            raise ValueError(reason)
        publish_calls.append({"submitted_by": submitted_by, "records": len(report.accepted_rows)})
        return {"repo_id": "kshitijthakkar/pramaan-community-runs",
                "records": len(report.accepted_rows)}

    monkeypatch.setattr(app, "_community_submission_archive_runtime",
                        lambda: (audit_submission_archive, publish))
    monkeypatch.setattr(app, "_render_custom_app", lambda _state: "rendered")

    valid_event = SimpleNamespace(action="community_archive_submit", active_tab="leaderboard",
                                  community_archive_path=str(valid_upload),
                                  community_archive_consent=True)
    _, valid_state = app._handle_custom_event(
        valid_event, app._initial_custom_state(),
        oauth_profile=SimpleNamespace(username="test_user"),
        oauth_token=SimpleNamespace(token="real-test-oauth-token"),
    )
    valid_status = valid_state["community_archive_status"]
    assert "Published 1 replay-audited record(s)" in valid_status
    assert publish_calls == [{"submitted_by": "test_user", "records": 1}]

    signed_out_upload = upload_root / "signed-out.zip"
    signed_out_upload.write_bytes(source_archive.read_bytes())
    signed_out_event = SimpleNamespace(action="community_archive_submit", active_tab="leaderboard",
                                       community_archive_path=str(signed_out_upload),
                                       community_archive_consent=True)
    _, signed_out_state = app._handle_custom_event(signed_out_event, app._initial_custom_state())
    assert signed_out_state["community_archive_status"] == (
        "Sign in with Hugging Face to publish: use the button at the top of this page"
    )
    assert publish_calls == [{"submitted_by": "test_user", "records": 1}]

    tampered_event = SimpleNamespace(action="community_archive_submit", active_tab="leaderboard",
                                     community_archive_path=str(tampered_upload),
                                     community_archive_consent=True)
    _, tampered_state = app._handle_custom_event(
        tampered_event, app._initial_custom_state(),
        oauth_profile=SimpleNamespace(username="test_user"),
        oauth_token=SimpleNamespace(token="real-test-oauth-token"),
    )
    tampered_status = tampered_state["community_archive_status"]
    assert "Archive not published" in tampered_status
    assert len(publish_calls) == 1


def test_community_audit_rejects_replay_tampered_report(tmp_path):
    from pramaan.community import audit_trajectory_rows
    from pramaan.checks.core import all_pass, verify_pattern
    from pramaan.dsl import parse_pattern

    tasks, digest = load_suite("eval-v2.1")
    task = tasks[0]
    row = _failed_row(task, digest)
    candidate = json.loads((Path("examples/ajrakh_starfield.pattern.json")).read_text(encoding="utf-8"))
    checks, _, _ = verify_pattern(parse_pattern(json.dumps(candidate)))
    row["attempts"] = [{"index": 1, "candidate": candidate,
                        "report": {"checks": checks,
                                   "all_hard_constraints_pass": all_pass(checks)}}]
    row["valid_schema_at_1"] = True
    row["validity_at_1"] = all_pass(checks)
    row["validity_at_3"] = all_pass(checks)
    row["attempts"][0]["report"]["checks"] = [dict(check) for check in checks]
    row["attempts"][0]["report"]["checks"][0]["id"] = "tampered_check"

    report = audit_trajectory_rows([row], run_id="tampered-1", expected_suite_sha256=digest)

    assert report.accepted is False
    assert report.rejected_rows
    assert "replay" in report.reasons[0].lower()


def test_submission_archive_rejects_expansion_over_aggregate_limit():
    import io

    from pramaan.community import audit_submission_archive

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", b"{}")
        archive.writestr("results.jsonl", b"0" * 20_000_000)

    report = audit_submission_archive(buffer.getvalue(), run_id="zip-bomb")

    assert report.accepted is False
    assert "20 MB" in report.reasons[0]
