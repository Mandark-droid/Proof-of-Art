import json
from pathlib import Path

from pramaan.checks.core import verify_pattern
from pramaan.dsl import Pattern
from pramaan.reward import reward_for_report

ROOT = Path(__file__).parents[1]


def _report(path: str, *, attempt: int | None = None):
    data = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if attempt is not None:
        data = data["attempts"][attempt]["candidate"]
    pattern = Pattern.model_validate_json(json.dumps(data))
    checks, _, _ = verify_pattern(pattern)
    return {"checks": checks, "all_hard_constraints_pass": all(c["status"] == "PASS" for c in checks)}


def test_reward_values():
    starfield = _report("examples/ajrakh_starfield.pattern.json")
    d4_counterexample = _report("examples/beam_d4_rosette.trajectory.json", attempt=0)
    assert reward_for_report(starfield) == 1.0
    assert reward_for_report(d4_counterexample) == 0.725
    assert reward_for_report({"parse_error": "not json", "checks": []}) == 0.0
    assert reward_for_report({"checks": [{"id": "candidate_truncated", "status": "FAIL"}]}) == 0.0


def test_reward_for_report_safely_rejects_malformed_checks():
    assert reward_for_report({"checks": [{"id": "schema_valid", "status": "PASS"}, None]}) == 0.0
    assert reward_for_report({"checks": ["not-a-check"]}) == 0.0


def test_reward_for_report_rejects_malformed_top_level_values():
    assert reward_for_report(None) == 0.0
    assert reward_for_report({"checks": []}) == 0.0
    assert reward_for_report({"checks": [{"id": "schema_valid", "status": "FAIL"}]}) == 0.0


def test_reward_rejects_unknown_or_incomplete_verifier_check_schemas():
    valid = _report("examples/ajrakh_starfield.pattern.json")
    unknown = {"checks": [
        {"id": "schema_valid", "status": "PASS"},
        {"id": "made_up", "status": "PASS"},
    ]}
    incomplete = {"checks": [{"id": "candidate_parse", "status": "PASS"}]}
    missing = {"checks": valid["checks"][:-1]}
    duplicated = {"checks": valid["checks"] + [valid["checks"][0]]}

    assert reward_for_report(unknown) == 0.0
    assert reward_for_report(incomplete) == 0.0
    assert reward_for_report(missing) == 0.0
    assert reward_for_report(duplicated) == 0.0


def test_reward_rejects_unknown_statuses_in_known_schema():
    valid = _report("examples/ajrakh_starfield.pattern.json")
    valid["checks"][1]["status"] = "MAYBE"
    assert reward_for_report(valid) == 0.0
