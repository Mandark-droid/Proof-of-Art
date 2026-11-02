"""Regenerate replay certificates for the fixed hand-authored examples."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pramaan.certificate import make_certificate
from pramaan.checks.core import all_pass, verify_pattern
from pramaan.dsl import Pattern


def main() -> int:
    expected_failures = {
        "ajrakh_starfield": set(),
        "ajrakh_d4_counterexample": {"symmetry_d4"},
        "beam_d4_rosette": set(),
        "beam_medium_03": set(),
        "beam_medium_06": set(),
        "beam_hard_07": set(),
    }
    for name, expected in expected_failures.items():
        design_path = ROOT / "examples" / f"{name}.pattern.json"
        certificate_path = ROOT / "examples" / f"{name}.certificate.json"
        pattern = Pattern.model_validate_json(design_path.read_text(encoding="utf-8"))
        checks, masks, grid = verify_pattern(pattern)
        failed = {check["id"] for check in checks if check["status"] != "PASS"}
        if failed != expected or all_pass(checks) != (not expected):
            raise SystemExit(f"{name}: expected failures {sorted(expected)}, got {sorted(failed)}")
        certificate = make_certificate(pattern, checks, masks, grid)
        certificate_path.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
        print(f"Refreshed {certificate_path.relative_to(ROOT)}; failures={sorted(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
