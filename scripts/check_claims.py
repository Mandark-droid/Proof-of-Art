"""Validate festival claims before they are rendered or published."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "pramaan" / "data" / "claims.json"
REGISTRY_PATH = ROOT / "pramaan" / "data" / "traditions.json"
ALLOWED_SURFACES = {"floor", "wall"}
ALLOWED_DIWALI_STATUSES = {"sourced_institutional", "sourced_secondary", "none"}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _has_https_source(claim: dict[str, Any]) -> bool:
    direct = claim.get("url")
    if isinstance(direct, str) and direct.startswith("https://"):
        return True
    sources = claim.get("sources", [])
    return isinstance(sources, list) and any(
        isinstance(source, dict) and isinstance(source.get("url"), str)
        and source["url"].startswith("https://") for source in sources
    )


def audit_claims(claims_path: Path = CLAIMS_PATH, registry_path: Path = REGISTRY_PATH) -> list[str]:
    """Return all broken source references in the ledger and tradition tags."""
    try:
        claims_doc = _load(claims_path)
        registry_doc = _load(registry_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"cannot read claims or registry JSON: {exc}"]

    issues: list[str] = []
    claims = claims_doc.get("claims")
    if not isinstance(claims, list):
        return ["claims.json must contain a claims list"]
    claim_by_id: dict[str, dict[str, Any]] = {}
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            issues.append(f"claims[{index}] must be an object")
            continue
        claim_id = claim.get("id")
        if not isinstance(claim_id, str) or not claim_id.strip():
            issues.append(f"claims[{index}] has no id")
            continue
        if claim_id in claim_by_id:
            issues.append(f"duplicate claim id: {claim_id}")
        claim_by_id[claim_id] = claim
        if claim.get("status") != "sourced":
            issues.append(f"claim {claim_id} must have status sourced")
        if not _has_https_source(claim):
            issues.append(f"claim {claim_id} must have an https source URL")
        if not isinstance(claim.get("text"), str) or not claim["text"].strip():
            issues.append(f"claim {claim_id} must have claim text")

    def check_refs(owner: str, refs: Any) -> None:
        if not isinstance(refs, list):
            issues.append(f"{owner} claim_ids must be a list")
            return
        for claim_id in refs:
            if not isinstance(claim_id, str) or claim_id not in claim_by_id:
                issues.append(f"{owner} references missing claim {claim_id!r}")
                continue
            claim = claim_by_id[claim_id]
            if claim.get("status") != "sourced" or not _has_https_source(claim):
                issues.append(f"{owner} references unsourced claim {claim_id}")

    entries = registry_doc.get("entries")
    if not isinstance(entries, list):
        issues.append("traditions.json must contain an entries list")
        entries = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            issues.append(f"traditions.entries[{index}] must be an object")
            continue
        tradition_id = entry.get("id", f"entry-{index}")
        lens = entry.get("festival_lens")
        if lens is None:
            continue
        if not isinstance(lens, dict):
            issues.append(f"tradition {tradition_id} festival_lens must be an object or null")
            continue
        surface, status, refs = lens.get("surface"), lens.get("diwali_status"), lens.get("claim_ids")
        if surface is not None and surface not in ALLOWED_SURFACES:
            issues.append(f"tradition {tradition_id} has invalid festival surface {surface!r}")
        if status not in ALLOWED_DIWALI_STATUSES:
            issues.append(f"tradition {tradition_id} has invalid Diwali status {status!r}")
        if status == "none":
            if refs not in ([], None):
                issues.append(f"tradition {tradition_id} with status none cannot list claim_ids")
            continue
        if surface not in ALLOWED_SURFACES:
            issues.append(f"tradition {tradition_id} needs a floor or wall surface")
        if not isinstance(refs, list) or not refs:
            issues.append(f"tradition {tradition_id} sourced festival tag needs nonempty claim_ids")
        else:
            check_refs(f"tradition {tradition_id}", refs)

    public_copy = claims_doc.get("public_copy", [])
    if not isinstance(public_copy, list):
        issues.append("claims.json public_copy must be a list")
    else:
        for index, caption in enumerate(public_copy):
            if not isinstance(caption, dict):
                issues.append(f"public_copy[{index}] must be an object")
                continue
            caption_id = caption.get("id", f"caption-{index}")
            if not isinstance(caption.get("text"), str) or not caption["text"].strip():
                issues.append(f"public_copy {caption_id} must have text")
            refs = caption.get("claim_ids")
            if not isinstance(refs, list) or not refs:
                issues.append(f"public_copy {caption_id} needs nonempty claim_ids")
            else:
                check_refs(f"public_copy {caption_id}", refs)
            target = caption.get("target")
            if target is not None:
                target_path = (ROOT / str(target)).resolve()
                if ROOT.resolve() not in target_path.parents or not target_path.is_file():
                    issues.append(f"public_copy {caption_id} has invalid target {target!r}")
                elif str(caption.get("text", "")) not in target_path.read_text(encoding="utf-8"):
                    issues.append(f"public_copy {caption_id} text is missing from {target}")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, default=CLAIMS_PATH)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    args = parser.parse_args()
    issues = audit_claims(args.claims, args.registry)
    if issues:
        for issue in issues:
            print(f"FAIL {issue}")
        return 1
    print(f"PASS claims: sourced references resolve ({args.claims})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
