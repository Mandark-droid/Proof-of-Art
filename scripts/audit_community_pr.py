"""Read-only replay audit for local community ZIPs or dataset pull requests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pramaan.community import AuditReport, audit_submission_archive
from scripts.push_hub_datasets import COMMUNITY_REPO, MAX_COMMUNITY_UPLOAD_BYTES


def _download_pr_archive(number: int) -> Path:
    """Download the sole ZIP under submissions/ from a public dataset PR."""
    from huggingface_hub import HfApi, hf_hub_download

    revision = f"refs/pr/{number}"
    api = HfApi(token=False)
    files = api.list_repo_files(repo_id=COMMUNITY_REPO, repo_type="dataset", revision=revision)
    archives = [name for name in files
                if name.startswith("submissions/") and name.lower().endswith(".zip")
                and len(Path(name).parts) == 2]
    if len(archives) != 1:
        raise ValueError(f"PR {number} must contain exactly one ZIP directly under submissions/; found {len(archives)}")
    path = hf_hub_download(repo_id=COMMUNITY_REPO, repo_type="dataset", filename=archives[0],
                           revision=revision, token=False)
    return Path(path)


def _safe_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    label = re.sub(r"[\x00-\x1f\x7f-\x9f]", "?", value).strip()[:200]
    if not label or re.search(r"(?:sk-ant-api\d{2}-|sk-[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{20,})", label):
        return None
    return label


def _safe_reason(value: str) -> str:
    safe = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", value)
    safe = re.sub(r"sk-ant-api\d{2}-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{20,}",
                  "[REDACTED]", safe)
    return safe[:300]


def _reported_label(archive_path: Path, report: AuditReport) -> str:
    labels = {
        label for row in report.accepted_rows
        if isinstance(row, dict)
        for label in (_safe_label(row.get("model_label")),)
        if label
    }
    if not labels:
        labels = {
            label for row in report.accepted_rows
            if isinstance(row, dict)
            for label in (_safe_label(row.get("model_key")),)
            if label
        }
    if not labels:
        try:
            if archive_path.stat().st_size > MAX_COMMUNITY_UPLOAD_BYTES:
                return "unavailable (archive exceeds size limit)"
            with zipfile.ZipFile(archive_path) as archive:
                if "manifest.json" in archive.namelist():
                    manifest = json.loads(archive.read("manifest.json"))
                    values = manifest.get("model_keys", []) if isinstance(manifest, dict) else []
                    labels.update(label for value in values if (label := _safe_label(value)))
                if not labels and "results.jsonl" in archive.namelist():
                    for line in archive.read("results.jsonl").decode("utf-8").splitlines():
                        row = json.loads(line)
                        if isinstance(row, dict):
                            label = _safe_label(row.get("model_id")) or _safe_label(row.get("model_key"))
                            if label:
                                labels.add(label)
        except (OSError, UnicodeError, json.JSONDecodeError, zipfile.BadZipFile, RuntimeError):
            pass
    return ", ".join(sorted(labels)) if labels else "unavailable"


def _print_report(report: AuditReport, archive_path: Path) -> int:
    accepted = len(report.accepted_rows)
    rejected = len(report.rejected_rows)
    total = accepted + rejected
    status = "PASS" if report.accepted else "FAIL"
    print(f"{status}: rows={total}; accepted={accepted}; rejected={rejected}")
    print(f"Self-reported model label: {_reported_label(archive_path, report)}")
    reasons = report.reasons or [row.get("reason") for row in report.rejected_rows if isinstance(row, dict)]
    for reason in reasons:
        if isinstance(reason, str) and reason.strip():
            print(f"Reason: {_safe_reason(reason)}")
    return 0 if report.accepted else 1


def run_source(source: str) -> int:
    """Audit a local ZIP path or a numeric Hub pull request without publishing."""
    candidate = Path(source)
    try:
        if candidate.is_file():
            archive_path = candidate
            run_id = "community-local"
        elif source.isdecimal() and int(source) > 0:
            number = int(source)
            archive_path = _download_pr_archive(number)
            run_id = f"community-pr-{number}"
        else:
            raise ValueError("source must be an existing ZIP path or a positive Hub pull request number")
        report = audit_submission_archive(archive_path, run_id=run_id, provider_api_key=None)
        return _print_report(report, archive_path)
    except Exception as error:
        print("FAIL: 0 audited rows; 0 accepted; 0 rejected")
        print("Self-reported model label: unavailable")
        print(f"Reason: {_safe_reason(str(error))}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local submission ZIP path or pull request number on the community dataset")
    args = parser.parse_args()
    return run_source(args.source)


if __name__ == "__main__":
    raise SystemExit(main())
