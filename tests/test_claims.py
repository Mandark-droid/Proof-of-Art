import json

from scripts.check_claims import audit_claims


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_claims_audit_accepts_sourced_registry_tags(tmp_path):
    claims = {
        "claims": [{"id": "c_diwali", "status": "sourced", "url": "https://example.org/source", "text": "Rangoli is made during Diwali."}],
        "public_copy": [{"id": "festival_intro", "text": "A sourced caption.", "claim_ids": ["c_diwali"]}],
    }
    registry = {"entries": [{"id": "tn_kolam", "festival_lens": {"surface": "floor", "diwali_status": "sourced_institutional", "claim_ids": ["c_diwali"]}}]}
    claims_path, registry_path = tmp_path / "claims.json", tmp_path / "traditions.json"
    _write_json(claims_path, claims)
    _write_json(registry_path, registry)

    assert audit_claims(claims_path, registry_path) == []


def test_claims_audit_rejects_missing_or_unverified_claim_references(tmp_path):
    claims = {
        "claims": [{"id": "c_unverified", "status": "unverified", "url": "https://example.org/source", "text": "Unsupported."}],
        "public_copy": [{"id": "diwali_caption", "text": "Unsupported caption.", "claim_ids": ["missing_claim"]}],
    }
    registry = {"entries": [{"id": "tn_kolam", "festival_lens": {"surface": "floor", "diwali_status": "sourced_institutional", "claim_ids": ["c_unverified"]}}]}
    claims_path, registry_path = tmp_path / "claims.json", tmp_path / "traditions.json"
    _write_json(claims_path, claims)
    _write_json(registry_path, registry)

    issues = audit_claims(claims_path, registry_path)

    assert any("status" in issue and "c_unverified" in issue for issue in issues)
    assert any("missing_claim" in issue for issue in issues)


def test_claims_audit_rejects_sourced_tags_without_claims(tmp_path):
    claims_path, registry_path = tmp_path / "claims.json", tmp_path / "traditions.json"
    _write_json(claims_path, {"claims": [], "public_copy": []})
    _write_json(registry_path, {"entries": [{"id": "tn_kolam", "festival_lens": {"surface": "floor", "diwali_status": "sourced_secondary", "claim_ids": []}}]})

    issues = audit_claims(claims_path, registry_path)

    assert any("tn_kolam" in issue and "claim_ids" in issue for issue in issues)


def test_repository_claim_ledger_and_registry_tags_are_consistent():
    from scripts.check_claims import audit_claims

    assert audit_claims() == []
