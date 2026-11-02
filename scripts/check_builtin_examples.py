"""Replay bundled passing and deliberately failing examples for release checks."""

from __future__ import annotations

import sys
import json
import re
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pramaan.verify import replay
from pramaan.dsl import parse_pattern
from pramaan.checks.core import verify_pattern
from pramaan.certificate import make_certificate, verify_certificate
from pramaan.checks.core import all_pass
from pramaan.export import render_png


def main() -> int:
    fixtures = (
        ("ajrakh_starfield", True, None, "hand_authored"),
        ("ajrakh_d4_counterexample", False, "symmetry_d4", "hand_authored"),
        ("beam_d4_rosette", True, None, "beam_repair"),
        ("beam_medium_03", True, None, "beam_benchmark"),
        ("beam_medium_06", True, None, "beam_benchmark"),
        ("beam_hard_07", True, None, "beam_benchmark"),
        ("beam_g2_gj_bandhani", True, None, "beam_gallery"),
        ("beam_g3_jh_singhbhum_bands", True, None, "beam_gallery"),
        ("beam_g4_as_gamosa_grid", True, None, "beam_gallery"),
        ("beam_g5_br_aripana_rotational", True, None, "beam_gallery"),
        ("beam_g6_tn_kolam_sikku", True, None, "beam_gallery"),
        ("beam_g7_gj_sidi_saiyyed_jaali", True, None, "beam_gallery"),
    )
    for name, expected_pass, expected_failure, provenance_type in fixtures:
        design = ROOT / "examples" / f"{name}.pattern.json"
        certificate = ROOT / "examples" / f"{name}.certificate.json"
        if provenance_type == "beam_gallery":
            pattern = parse_pattern(design.read_text(encoding="utf-8"))
            checks, masks, grid = verify_pattern(pattern)
            fresh_certificate = make_certificate(pattern, checks, masks, grid)
            saved_certificate = json.loads(certificate.read_text(encoding="utf-8"))
            matches, message = verify_certificate(saved_certificate, fresh_certificate)
            report = {
                "checks": checks,
                "all_hard_constraints_pass": all_pass(checks),
                "certificate": fresh_certificate,
                "certificate_replay": {"matches": matches, "message": message},
            }
            code = 0 if all_pass(checks) and matches else 1
        else:
            code, report = replay(design, certificate)
        if report.get("error"):
            raise SystemExit(f"{name}: replay error: {report['error']}")
        replay_match = report.get("certificate_replay", {}).get("matches")
        if replay_match is not True:
            raise SystemExit(f"{name}: saved certificate did not match deterministic replay")

        failed_ids = [check["id"] for check in report["checks"] if check["status"] != "PASS"]
        if expected_pass:
            if code != 0 or not report.get("all_hard_constraints_pass"):
                raise SystemExit(f"{name}: expected a passing design, got exit {code} and failures {failed_ids}")
            pattern = parse_pattern(design.read_text(encoding="utf-8"))
            checks, masks, grid = verify_pattern(pattern)
            expected_preview = render_png(grid, pattern.palette, scale=5, repeats=3)
            preview_path = ROOT / "assets" / f"{name}_preview.png"
            with Image.open(preview_path) as saved_preview:
                if (saved_preview.convert("RGB").size != expected_preview.size
                        or saved_preview.convert("RGB").tobytes() != expected_preview.tobytes()):
                    raise SystemExit(f"{name}: saved preview does not match deterministic rendering")
            if provenance_type in ("beam_repair", "beam_benchmark", "beam_gallery"):
                trace_path = ROOT / "examples" / f"{name}.trajectory.json"
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                attempts = trace.get("attempts", [])
                first_failures = [check["id"] for check in attempts[0]["report"]["checks"]
                                  if check["status"] != "PASS"] if attempts else []
                design_obj = json.loads(design.read_text(encoding="utf-8"))
                if (trace.get("beam_generated") is not True
                        or (trace.get("accepted") is not True if provenance_type in ("beam_repair", "beam_gallery") else False)
                        or not attempts or (provenance_type == "beam_repair" and len(attempts) < 2)
                        or (provenance_type == "beam_gallery" and (
                            trace.get("source") != "independent_gallery_generation"
                            or trace.get("data_role") != "gallery_only_not_dataset"
                            or trace.get("task", {}).get("grammar") != pattern.grammar
                            or trace.get("repair_count") != len(attempts) - 1
                            or len(trace.get("model_calls", [])) == 0
                            or sum(call.get("raw_response") is not None for call in trace.get("model_calls", [])) != len(attempts)
                        ))
                        or trace.get("repair_count") != len(attempts) - 1
                        or (provenance_type == "beam_repair" and first_failures != ["symmetry_d4"])
                        or attempts[-1]["report"].get("all_hard_constraints_pass") is not True
                        or attempts[-1].get("candidate") != design_obj
                        or trace.get("certificate_mask_sha256") != report["certificate"]["mask_sha256"]
                        or trace.get("certificate_dsl_sha256") != report["certificate"]["dsl_sha256"]
                        or re.search(r"hf_[A-Za-z0-9]{20,}", json.dumps(trace))):
                    raise SystemExit(f"{name}: Beam repair provenance does not match the verified design")
                if provenance_type == "beam_repair":
                    detail = "symmetry failure repaired"
                elif provenance_type == "beam_gallery":
                    detail = f"independent {pattern.grammar} gallery sample replayed"
                else:
                    detail = "benchmark candidate replayed"
                print(f"PASS {name}: Beam-generated {detail}; final certificate replay matches; no token found")
            else:
                print(f"PASS {name}: all hard checks pass; certificate replay matches")
        else:
            if code != 1 or report.get("all_hard_constraints_pass") or failed_ids != [expected_failure]:
                raise SystemExit(f"{name}: expected only {expected_failure} to fail, got exit {code} and failures {failed_ids}")
            failed_check = next(check for check in report["checks"] if check["id"] == expected_failure)
            witness = failed_check.get("counterexample") or {}
            if not witness.get("transform") or witness.get("original") == witness.get("transformed"):
                raise SystemExit(f"{name}: symmetry failure lacks distinct witness coordinates")
            print(f"EXPECTED FAILURE {name}: {expected_failure} at {witness['original']} vs {witness['transformed']}; certificate replay matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
