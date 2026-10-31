"""Independent certificate replay CLI: python -m pramaan.verify --help"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .certificate import make_certificate, verify_certificate
from .checks.core import all_pass, verify_pattern
from .dsl import Pattern


def replay(design_path: Path, certificate_path: Path | None = None) -> tuple[int, dict]:
    try:
        pattern = Pattern.model_validate_json(design_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        return 2, {"error": f"Invalid design: {exc}"}
    checks, masks, grid = verify_pattern(pattern)
    fresh = make_certificate(pattern, checks, masks, grid)
    result = {"design": str(design_path), "checks": checks, "all_hard_constraints_pass": all_pass(checks), "certificate": fresh}
    if certificate_path:
        try:
            saved = json.loads(certificate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return 2, {"error": f"Invalid certificate: {exc}", **result}
        matches, message = verify_certificate(saved, fresh)
        result["certificate_replay"] = {"matches": matches, "message": message}
        if not matches:
            return 1, result
    return (0 if all_pass(checks) else 1), result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute a Pramaan design's deterministic checks and certificate.")
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--strict", action="store_true", help="Require a passing design (default behavior).")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)
    code, report = replay(args.design, args.certificate)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for check in report.get("checks", []):
            print(f"{check['status']:4} {check['id']}: {check['message']}")
        if "certificate_replay" in report:
            print(f"Certificate replay: {report['certificate_replay']['message']}")
        if "error" in report:
            print(report["error"], file=sys.stderr)
        print("VERIFIED" if code == 0 else "NOT VERIFIED")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
