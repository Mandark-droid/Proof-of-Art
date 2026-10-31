from __future__ import annotations

from math import isfinite
from typing import Any


_BASE_CHECK_IDS = frozenset({
    "schema_valid",
    "complexity_bound",
    "palette_limit",
    "periodic_raster",
    "layer_registration",
    "exclusive_ink_collision",
    "minimum_gap",
    "coverage_bound",
})
_SYMMETRY_CHECK_IDS = frozenset({
    "symmetry_none",
    "symmetry_d2",
    "symmetry_d4",
    "symmetry_c2",
    "symmetry_c4",
})
_GRAMMAR_CHECK_SETS = (
    frozenset(),
    frozenset({"dot_clearance"}),
    frozenset({"translation_period"}),
    frozenset({"cell_quantization"}),
    frozenset({"kolam_mirrors_valid", "kolam_loop_count", "kolam_mirror_symmetry"}),
    frozenset({"jaali_single_solid", "jaali_holes_interior", "jaali_open_fraction", "jaali_min_web"}),
)


def _known_check_schema(checks: list[dict[str, Any]]) -> bool:
    ids = [check.get("id") for check in checks]
    if any(not isinstance(check_id, str) for check_id in ids) or len(ids) != len(set(ids)):
        return False
    check_ids = set(ids)
    if not _BASE_CHECK_IDS.issubset(check_ids):
        return False
    symmetry_ids = check_ids & _SYMMETRY_CHECK_IDS
    if len(symmetry_ids) != 1:
        return False
    extras = check_ids - _BASE_CHECK_IDS - symmetry_ids
    return extras in _GRAMMAR_CHECK_SETS


def reward_for_report(report: dict[str, Any]) -> float:
    """Map a complete deterministic verifier report to a bounded scalar reward.

    The verifier's complete known check schema is required. A partial or fabricated
    list cannot earn reward by presenting only passing checks.
    """
    if not isinstance(report, dict) or report.get("parse_error"):
        return 0.0
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks or any(not isinstance(check, dict) for check in checks):
        return 0.0
    if not _known_check_schema(checks):
        return 0.0
    if report.get("candidate_truncated") or any(check.get("id") == "candidate_truncated" for check in checks):
        return 0.0
    if any(check.get("status") not in {"PASS", "FAIL"} for check in checks):
        return 0.0
    schema = next(check for check in checks if check["id"] == "schema_valid")
    if schema["status"] != "PASS":
        return 0.0
    hard = [check for check in checks if check["id"] != "schema_valid"]
    passed = sum(check["status"] == "PASS" for check in hard)
    reward = 1.0 if passed == len(hard) else round(0.2 + 0.6 * passed / len(hard), 12)
    return float(reward) if isfinite(reward) and 0.0 <= reward <= 1.0 else 0.0
