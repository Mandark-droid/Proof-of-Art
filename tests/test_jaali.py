from __future__ import annotations

import numpy as np
import json

from pramaan.checks.jaali import (
    check_holes_interior,
    check_min_web,
    check_open_fraction,
    check_single_solid,
    connected_components,
    centered_square_hole_sides,
    min_web_witness,
)
from pramaan.checks.core import verify_pattern
from pramaan.dsl import PatternV2


def test_single_component_pass():
    mask = np.ones((64, 64), dtype=bool)
    assert check_single_solid(mask, wrap=False)["status"] == "PASS"


def test_two_islands_witness():
    mask = np.ones((64, 64), dtype=bool)
    mask[:, 31:33] = False
    check = check_single_solid(mask, wrap=False)
    assert check["status"] == "FAIL"
    assert check["counterexample"]["component_count"] == 2
    assert check["counterexample"]["pixel"] == [0, 0]


def test_toroidal_connectivity_wraps():
    mask = np.ones((64, 64), dtype=bool)
    mask[:, 31:33] = False
    assert connected_components(mask, wrap=False)[0] == 2
    assert connected_components(mask, wrap=True)[0] == 1


def test_hole_touching_frame_fails_bounded_only():
    holes = np.zeros((64, 64), dtype=bool)
    holes[0, 10] = True
    assert check_holes_interior(holes, frame=2, wrap=False)["status"] == "FAIL"
    assert check_holes_interior(holes, frame=2, wrap=True)["status"] == "PASS"


def test_open_fraction_bounds():
    holes = np.zeros((64, 64), dtype=bool)
    holes[:16, :] = True
    assert check_open_fraction(holes, minimum=0.2, maximum=0.3)["status"] == "PASS"
    assert check_open_fraction(holes, minimum=0.4, maximum=0.5)["status"] == "FAIL"


def test_centered_square_hole_sides_respect_fraction_and_frame():
    sides = centered_square_hole_sides(64, 64, minimum=0.12, maximum=0.18, frame=4)

    assert sides
    assert all(4 <= side <= 56 for side in sides)
    assert all(0.12 <= side * side / (64 * 64) <= 0.18 for side in sides)


def test_centered_square_hole_sides_report_no_integer_witness():
    assert centered_square_hole_sides(64, 64, minimum=0.001, maximum=0.001, frame=2) == ()


def test_centered_square_hole_sides_reject_nonfinite_fraction_bounds():
    import pytest

    with pytest.raises(ValueError, match="open-fraction bounds"):
        centered_square_hole_sides(64, 64, minimum=float("nan"), maximum=0.5, frame=2)


def test_min_web_witness_on_1px_bridge():
    solid = np.zeros((64, 64), dtype=bool)
    solid[:, 31] = True
    assert min_web_witness(solid, t=1) == (31, 0)
    check = check_min_web(solid, t=1)
    assert check["status"] == "FAIL"
    assert check["assurance"] == "APPROXIMATE"


def test_jaali_pattern_renders_and_runs_all_structural_checks():
    pattern = PatternV2.model_validate_json(json.dumps({
        "schema_version": "0.2", "grammar": "jaali_lattice", "tradition_id": "test_jaali",
        "title": "Bounded lattice", "tile": {"width": 64, "height": 64, "wrap_x": False, "wrap_y": False},
        "palette": ["#000000", "#FFFFFF"], "background_index": 0,
        "layers": [{"id": "stone", "colour_index": 1, "anchor": [0, 0], "motifs": []}],
        "constraints": {"symmetry": "D4", "min_open_fraction": 0.001, "max_open_fraction": 0.1},
        "jaali": {"frame": 2, "open_colour_index": 0, "holes": [
            {"kind": "rectangle", "center": [32, 32], "size": [4, 4]},
        ]},
    }))
    checks, masks, grid = verify_pattern(pattern)
    structural = {item["id"]: item for item in checks if item["id"].startswith("jaali_")}
    assert set(structural) == {
        "jaali_single_solid", "jaali_holes_interior", "jaali_open_fraction", "jaali_min_web",
    }
    assert all(item["status"] == "PASS" for item in structural.values())
    assert masks["solid"].shape == masks["holes"].shape == grid.shape == (64, 64)
    assert grid[32, 32] == pattern.jaali.open_colour_index
