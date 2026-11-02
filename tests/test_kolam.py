from __future__ import annotations

from collections import Counter
import json
from math import gcd

import pytest
import numpy as np

from pramaan.checks.core import verify_pattern
from pramaan.dsl import PatternV2
from pramaan.kolam import (
    interior_midpoints,
    loop_count,
    mirror_symmetry_orbits,
    trace_loops,
    validate_mirrors,
)


@pytest.mark.parametrize(
    ("rows", "cols", "expected"),
    [(1, 1, 1), (2, 3, 1), (3, 4, 1), (4, 4, 4), (6, 4, 2), (5, 5, 5)],
)
def test_loop_count_is_gcd_without_mirrors(rows, cols, expected):
    assert loop_count(rows, cols, frozenset()) == expected == gcd(rows, cols)


def test_every_non_mirror_midpoint_crossed_exactly_twice():
    rows, cols = 3, 4
    paths = trace_loops(rows, cols, frozenset())
    interior = interior_midpoints(rows, cols)
    crossed = Counter(point for path in paths for point in path[:-1] if point in interior)
    assert set(crossed) == interior
    assert set(crossed.values()) == {2}


def test_mirror_midpoints_never_crossed():
    mirrors = frozenset({(1, 0), (2, 1)})
    paths = trace_loops(3, 3, mirrors)
    for path in paths:
        for index, point in enumerate(path[1:-1], start=1):
            if point in mirrors:
                incoming = (point[0] - path[index - 1][0], point[1] - path[index - 1][1])
                outgoing = (path[index + 1][0] - point[0], path[index + 1][1] - point[1])
                assert outgoing != incoming


def test_single_mirror_changes_loop_count_deterministically():
    assert loop_count(3, 3, frozenset()) == 3
    first = loop_count(3, 3, frozenset({(1, 0)}))
    assert 1 <= first <= len(interior_midpoints(3, 3)) * 2
    assert loop_count(3, 3, frozenset({(1, 0)})) == first


def test_validate_mirrors_reports_invalid_positions():
    invalid = validate_mirrors(3, 3, {(0, 0), (-1, 0), (1, 1), (1, 0)})
    assert set(invalid) == {(0, 0), (-1, 0), (1, 1)}
    assert interior_midpoints(3, 3) == {
        (1, 0), (3, 0), (0, 1), (2, 1), (4, 1),
        (1, 2), (3, 2), (0, 3), (2, 3), (4, 3), (1, 4), (3, 4),
    }


def test_mirror_symmetry_orbits_partition_interior_midpoints():
    interior = interior_midpoints(5, 5)
    orbits = mirror_symmetry_orbits(5, 5, "D2")
    flattened = [point for orbit in orbits for point in orbit]

    assert set(flattened) == interior
    assert len(flattened) == len(set(flattened))
    assert all(len(orbit) in {1, 2, 4} for orbit in orbits)


def test_mirror_symmetry_orbits_reject_incompatible_d4_grid():
    with pytest.raises(ValueError, match="square grid"):
        mirror_symmetry_orbits(3, 5, "D4")


@pytest.mark.parametrize("symmetry", ["None", "C2", "C4", "D2", "D4"])
def test_mirror_symmetry_orbits_cover_each_supported_group(symmetry):
    orbits = mirror_symmetry_orbits(5, 5, symmetry)
    interior = interior_midpoints(5, 5)

    assert {point for orbit in orbits for point in orbit} == interior
    assert all(orbit for orbit in orbits)


def test_mirror_symmetry_orbits_reject_unknown_group():
    with pytest.raises(ValueError, match="unsupported Kolam symmetry"):
        mirror_symmetry_orbits(5, 5, "C6")


def _kolam_pattern(rows=5, cols=5, mirrors=(), symmetry="None", required_loops=None):
    constraints = {"symmetry": symmetry, "min_coverage": 0, "max_coverage": 1}
    if required_loops is not None:
        constraints["required_loops"] = required_loops
    return PatternV2.model_validate_json(json.dumps({
        "schema_version": "0.2", "grammar": "kolam_loop", "tradition_id": "tn_kolam",
        "title": "Kolam test", "tile": {"width": 64, "height": 64, "wrap_x": False, "wrap_y": False},
        "palette": ["#000000", "#FFFFFF", "#C04040"], "background_index": 0,
        "layers": [
            {"id": "curve", "colour_index": 1, "anchor": [0, 0], "motifs": []},
            {"id": "dots", "colour_index": 2, "anchor": [0, 0], "motifs": []},
        ],
        "constraints": constraints,
        "kolam": {"rows": rows, "cols": cols, "spacing": 8, "mirrors": list(mirrors), "stroke": 1, "dot_radius": 1},
    }))


def test_kolam_render_fits_canvas():
    _, _, grid = verify_pattern(_kolam_pattern())
    assert not np.any(grid[:2, :])
    assert not np.any(grid[-2:, :])
    assert not np.any(grid[:, :2])
    assert not np.any(grid[:, -2:])


def test_palette_check_uses_kolam_renderer_masks_not_model_layer_ids():
    pattern = _kolam_pattern()
    pattern.layers[0].id = "marks"

    checks, masks, _ = verify_pattern(pattern)

    palette_check = next(item for item in checks if item["id"] == "palette_limit")
    assert set(masks) == {"curve", "dots"}
    assert palette_check["status"] == "PASS"
    assert palette_check["message"] == "3 colours used; limit 4"


def test_symmetric_mirrors_give_d4_raster():
    mirrors = {(1, 0), (7, 8), (7, 0), (1, 8), (0, 7), (8, 1), (0, 1), (8, 7)}
    checks, _, _ = verify_pattern(_kolam_pattern(mirrors=mirrors, symmetry="D4"))
    assert next(item for item in checks if item["id"] == "symmetry_d4")["status"] == "PASS"
    assert next(item for item in checks if item["id"] == "kolam_mirror_symmetry")["status"] == "PASS"


def test_required_loops_witness():
    checks, _, _ = verify_pattern(_kolam_pattern(rows=4, cols=4, required_loops=1))
    check = next(item for item in checks if item["id"] == "kolam_loop_count")
    assert check["status"] == "FAIL"
    assert check["counterexample"] == {"loops": 4, "required": 1}


def test_invalid_mirror_positions_report_witness():
    checks, _, _ = verify_pattern(_kolam_pattern(rows=3, cols=3, mirrors={(0, 0), (-1, 0), (1, 1)}))
    check = next(item for item in checks if item["id"] == "kolam_mirrors_valid")
    assert check["status"] == "FAIL"
    assert check["counterexample"]["mirror"] in ([0, 0], [-1, 0], [1, 1])
