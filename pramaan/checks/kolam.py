"""Kolam-specific exact checks and concrete counterexamples."""

from __future__ import annotations

from typing import Any

from ..dsl import PatternV2
from ..kolam import interior_midpoints, loop_count, validate_mirrors


def _check(id: str, passed: bool, message: str, witness: Any = None) -> dict:
    result = {"id": id, "status": "PASS" if passed else "FAIL", "assurance": "EXACT_DISCRETE", "message": message}
    if witness is not None:
        result["counterexample"] = witness
    return result


def check_mirrors_valid(pattern: PatternV2) -> dict:
    spec = pattern.kolam
    invalid = validate_mirrors(spec.rows, spec.cols, spec.mirrors)
    witness = None if not invalid else {"mirror": list(invalid[0]), "valid_midpoints": sorted(map(list, interior_midpoints(spec.rows, spec.cols)))}
    return _check(
        "kolam_mirrors_valid", not invalid,
        "All mirrors occupy interior gap midpoints" if not invalid else f"Mirror {invalid[0]} is not an interior gap midpoint",
        witness,
    )


def check_loop_count(pattern: PatternV2) -> dict:
    spec = pattern.kolam
    mirrors = frozenset(tuple(point) for point in spec.mirrors if not validate_mirrors(spec.rows, spec.cols, [point]))
    actual = loop_count(spec.rows, spec.cols, mirrors)
    required = pattern.constraints.required_loops
    passed = required is None or actual == required
    witness = None if passed else {"loops": actual, "required": required}
    message = (
        f"Traced {actual} closed loops; no required count was declared"
        if required is None else
        f"Traced {actual} closed loops; required {required}"
    )
    return _check("kolam_loop_count", passed, message, witness)


def _transforms(rows: int, cols: int, symmetry: str):
    cx, cy = cols - 1, rows - 1

    def r180(x, y):
        return 2 * cx - x, 2 * cy - y

    def vertical(x, y):
        return 2 * cx - x, y

    def horizontal(x, y):
        return x, 2 * cy - y

    def r90(x, y):
        return cx + (y - cy), cy - (x - cx)

    def r270(x, y):
        return cx - (y - cy), cy + (x - cx)

    def diagonal(x, y):
        return cx + (y - cy), cy + (x - cx)

    def antidiagonal(x, y):
        return cx - (y - cy), cy - (x - cx)

    result = [("R180", r180)] if symmetry in {"C2", "C4", "D2", "D4"} else []
    if symmetry in {"D2", "D4"}:
        result.append(("F_vertical", vertical))
    if symmetry == "D4":
        result.extend([("F_horizontal", horizontal), ("R90", r90), ("R270", r270), ("F_diagonal", diagonal), ("F_antidiagonal", antidiagonal)])
    if symmetry == "C4":
        result.extend([("R90", r90), ("R270", r270)])
    return result


def check_mirror_symmetry(pattern: PatternV2) -> dict:
    spec = pattern.kolam
    mirrors = set(tuple(point) for point in spec.mirrors)
    for name, transform in _transforms(spec.rows, spec.cols, pattern.constraints.symmetry):
        for point in sorted(mirrors):
            transformed = transform(*point)
            if transformed not in mirrors:
                return _check(
                    "kolam_mirror_symmetry", False,
                    f"Mirror set is not invariant under {name}",
                    {"transform": name, "mirror": list(point), "missing_mirror": list(transformed)},
                )
    return _check(
        "kolam_mirror_symmetry", True,
        f"Mirror set is invariant under {pattern.constraints.symmetry}",
    )
