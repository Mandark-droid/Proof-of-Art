from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pramaan.checks.core import verify_pattern
from pramaan.dsl import Pattern, PatternV2
from pramaan.checks.grammars import check_cell_quantization, check_dot_clearance, check_translation_period


def _pattern(grammar: str, motifs: list[dict], symmetry: str = "None", **constraints) -> PatternV2:
    return PatternV2.model_validate_json(json.dumps({
        "schema_version": "0.2",
        "grammar": grammar,
        "tradition_id": "test_tradition",
        "title": "Grammar test",
        "tile": {"width": 64, "height": 64, "wrap_x": True, "wrap_y": True},
        "palette": ["#000000", "#FFFFFF"],
        "background_index": 0,
        "layers": [{"id": "ink", "colour_index": 1, "anchor": [0, 0], "motifs": motifs}],
        "constraints": {"symmetry": symmetry, **constraints},
    }))


def test_dot_clearance_pass_and_toroidal_witness():
    pattern = _pattern("dot_field", [
        {"kind": "dot", "center": [2, 32], "radius": 2},
        {"kind": "dot", "center": [62, 32], "radius": 2},
    ], min_clearance=1)
    check = check_dot_clearance(pattern, wrap=True)
    assert check["status"] == "FAIL"
    assert check["counterexample"]["distance_sq"] == 16
    assert check["counterexample"]["required_distance_sq"] == 25

    separated = _pattern("dot_field", [
        {"kind": "dot", "center": [10, 10], "radius": 1},
        {"kind": "dot", "center": [20, 20], "radius": 1},
    ], min_clearance=1)
    assert check_dot_clearance(separated, wrap=True)["status"] == "PASS"


def test_translation_period_x():
    grid = np.tile(np.tile(np.arange(8, dtype=np.uint8) % 2, 8), (64, 1))
    assert check_translation_period(grid, "x", 8)["status"] == "PASS"
    broken = grid.copy()
    broken[5, 13] = 1 - broken[5, 13]
    check = check_translation_period(broken, "x", 8)
    assert check["status"] == "FAIL"
    assert check["counterexample"]["pixel"]


def test_cell_quantization_4():
    mask = np.kron(np.indices((16, 16))[0] % 2 == 0, np.ones((4, 4), dtype=bool))
    assert check_cell_quantization({"ink": mask}, 4)["status"] == "PASS"
    broken = mask.copy()
    broken[1, 1] = not broken[1, 1]
    check = check_cell_quantization({"ink": broken}, 4)
    assert check["status"] == "FAIL"
    assert check["counterexample"]["block"] == [0, 0]


def test_c4_passes_where_d4_fails():
    centers = [(20, 10), (10, 43), (43, 53), (53, 20)]
    motifs = [{"kind": "dot", "center": list(center), "radius": 1} for center in centers]
    pattern = _pattern("rotational_field", motifs, symmetry="C4")
    checks, _, _ = verify_pattern(pattern)
    assert next(item for item in checks if item["id"] == "symmetry_c4")["status"] == "PASS"

    pattern.constraints.symmetry = "D4"
    checks, _, _ = verify_pattern(pattern)
    assert next(item for item in checks if item["id"] == "symmetry_d4")["status"] == "FAIL"


def test_v01_path_untouched():
    root = Path(__file__).resolve().parents[1]
    pattern_path = root / "examples" / "ajrakh_starfield.pattern.json"
    pattern = Pattern.model_validate_json(pattern_path.read_text(encoding="utf-8"))
    checks, _, _ = verify_pattern(pattern)
    saved = json.loads((root / "examples" / "ajrakh_starfield.certificate.json").read_text(encoding="utf-8"))
    assert checks == saved["checks"]
