from __future__ import annotations

from typing import Any

import numpy as np

from ..dsl import Pattern, PatternV2
from ..render import N, render, render_motif
from .grammars import check_cell_quantization, check_dot_clearance, check_translation_period
from .kolam import check_loop_count, check_mirror_symmetry, check_mirrors_valid
from .jaali import check_holes_interior, check_min_web, check_open_fraction, check_single_solid


def _check(id: str, passed: bool, assurance: str = "EXACT_DISCRETE", message: str = "", witness: Any = None):
    result = {"id": id, "status": "PASS" if passed else "FAIL", "assurance": assurance, "message": message or ("Check passed" if passed else "Check failed")}
    if witness is not None:
        result["counterexample"] = witness
    return result


def _first_difference(a: np.ndarray, b: np.ndarray):
    coords = np.argwhere(a != b)
    if len(coords) == 0:
        return None
    y, x = (int(v) for v in coords[0])
    return x, y


def _dilate_with_witness(mask: np.ndarray, offsets: list[tuple[int, int]]):
    dilated = np.zeros((N, N), dtype=bool)
    source_x = np.full((N, N), -1, dtype=np.int16)
    source_y = np.full((N, N), -1, dtype=np.int16)
    distance_sq = np.zeros((N, N), dtype=np.uint16)
    for dx, dy in offsets:
        shifted = np.roll(mask, shift=(dy, dx), axis=(0, 1))
        newly_reached = shifted & ~dilated
        ys, xs = np.nonzero(newly_reached)
        if len(xs):
            source_x[ys, xs] = (xs - dx) % N
            source_y[ys, xs] = (ys - dy) % N
            distance_sq[ys, xs] = dx * dx + dy * dy
        dilated |= shifted
    return dilated, source_x, source_y, distance_sq


def verify_pattern(pattern: Pattern | PatternV2) -> tuple[list[dict], dict[str, np.ndarray], np.ndarray]:
    masks, grid = render(pattern)
    checks: list[dict] = []

    checks.append(_check("schema_valid", True))
    count = sum(len(layer.motifs) for layer in pattern.layers)
    checks.append(_check("complexity_bound", count <= pattern.constraints.max_primitives,
                         message=f"{count} primitives; limit {pattern.constraints.max_primitives}",
                         witness=None if count <= pattern.constraints.max_primitives else {"actual": count, "limit": pattern.constraints.max_primitives}))

    if isinstance(pattern, PatternV2) and pattern.grammar in {"jaali_lattice", "kolam_loop"}:
        used = {int(value) for value in np.unique(grid)}
    else:
        used = {pattern.background_index}
        used.update(layer.colour_index for layer in pattern.layers if masks[layer.id].any())
    palette_ok = len(used) <= pattern.constraints.max_colours
    checks.append(_check("palette_limit", palette_ok,
                         message=f"{len(used)} colours used; limit {pattern.constraints.max_colours}",
                         witness=None if palette_ok else {"actual": sorted(used), "limit": pattern.constraints.max_colours}))

    symmetry = pattern.constraints.symmetry
    transforms = []
    inverse_coordinates = {
        "R180": lambda x, y: (N - 1 - x, N - 1 - y),
        "F_vertical": lambda x, y: (N - 1 - x, y),
        "R90": lambda x, y: (y, N - 1 - x),
        "R270": lambda x, y: (N - 1 - y, x),
        "F_horizontal": lambda x, y: (x, N - 1 - y),
        "F_diagonal": lambda x, y: (y, x),
        "F_antidiagonal": lambda x, y: (N - 1 - y, N - 1 - x),
    }
    if symmetry in ("D2", "D4", "C2", "C4"):
        transforms.append(("R180", np.rot90(grid, 2)))
    if symmetry in ("D2", "D4"):
        transforms.append(("F_vertical", np.fliplr(grid)))
    if symmetry == "D4":
        transforms.extend([("R90", np.rot90(grid, 3)), ("R270", np.rot90(grid, 1)),
                           ("F_horizontal", np.flipud(grid)), ("F_diagonal", np.transpose(grid)),
                           ("F_antidiagonal", np.fliplr(np.flipud(np.transpose(grid))))])
    if symmetry == "C4":
        transforms.extend([("R90", np.rot90(grid, 3)), ("R270", np.rot90(grid, 1))])
    mismatch = None
    for name, transformed in transforms:
        coord = _first_difference(grid, transformed)
        if coord is not None:
            x, y = coord
            other_x, other_y = inverse_coordinates[name](x, y)
            mismatch = {"transform": name, "original": [x, y],
                        "transformed": [other_x, other_y],
                        "original_colour_id": int(grid[y, x]),
                        "transformed_colour_id": int(grid[other_y, other_x])}
            break
    checks.append(_check("symmetry_" + (symmetry.lower() if symmetry != "None" else "none"), mismatch is None,
                         message="Composited colour grid matches the requested symmetry" if mismatch is None else f"Colour differs under {mismatch['transform']}", witness=mismatch))

    # A canonical tile is repeated by translation; this check confirms that translated copies
    # reproduce the same finite raster cell-for-cell. It does not assert continuous vector seams.
    tiled = np.tile(grid, (3, 3))
    periodic_ok = np.array_equal(tiled[:N, :N], grid) and np.array_equal(tiled[N:2*N, N:2*N], grid)
    checks.append(_check("periodic_raster", periodic_ok,
                         message="Translated copies reproduce the canonical 64x64 raster"))

    anchors = {tuple(layer.anchor) for layer in pattern.layers}
    registration_ok = len(anchors) == 1
    checks.append(_check("layer_registration", registration_ok,
                         message="All layers share one coordinate frame" if registration_ok else "Layers use different anchors",
                         witness=None if registration_ok else {"anchors": sorted([list(a) for a in anchors])}))

    collision = None
    for a, b in pattern.constraints.exclusive_layer_pairs:
        overlap = np.argwhere(masks[a] & masks[b])
        if len(overlap):
            y, x = (int(v) for v in overlap[0])
            collision = {"layers": [a, b], "pixel": [x, y]}
            break
    checks.append(_check("exclusive_ink_collision", collision is None,
                         message="Exclusive ink layers do not overlap" if collision is None else "Exclusive ink layers overlap",
                         witness=collision))

    min_gap = pattern.constraints.min_gap
    gap_witness = None
    if min_gap > 0:
        offsets = sorted(
            ((dx, dy) for dy in range(-min_gap + 1, min_gap) for dx in range(-min_gap + 1, min_gap)
             if dx * dx + dy * dy < min_gap * min_gap),
            key=lambda pair: (pair[0] * pair[0] + pair[1] * pair[1], pair[1], pair[0]),
        )
        entries = []
        for layer in pattern.layers:
            for motif_index, motif in enumerate(layer.motifs):
                mask = render_motif(motif, layer.anchor)
                expanded, source_x, source_y, distance_sq = _dilate_with_witness(mask, offsets)
                entries.append((layer.id, motif_index, mask, expanded, source_x, source_y, distance_sq))
        found = False
        for i, first in enumerate(entries):
            for second in entries[i + 1:]:
                close = first[3] & second[2]
                coords = np.argwhere(close)
                if len(coords):
                    y, x = (int(v) for v in coords[0])
                    sx = int(first[4][y, x])
                    sy = int(first[5][y, x])
                    distance = float(np.sqrt(int(first[6][y, x])))
                    gap_witness = {
                        "motifs": [[first[0], first[1]], [second[0], second[1]]],
                        "pixel": [x, y], "other_pixel": [sx, sy],
                        "distance_pixels": distance, "required_minimum_pixels": min_gap,
                    }
                    found = True
                    break
            if found:
                break
    checks.append(_check(
        "minimum_gap", gap_witness is None,
        message=("Minimum pairwise motif pixel-centre spacing meets the declared toroidal grid threshold" if gap_witness is None
                 else f"Motifs are {gap_witness['distance_pixels']:.3f}px apart; minimum is {min_gap}px"),
        witness=gap_witness,
    ))

    ink_mask = np.zeros((N, N), dtype=bool)
    for mask in masks.values():
        ink_mask |= mask
    ink_pixels = int(np.count_nonzero(ink_mask))
    coverage = ink_pixels / float(N * N)
    coverage_ok = pattern.constraints.min_coverage <= coverage <= pattern.constraints.max_coverage
    if isinstance(pattern, PatternV2) and pattern.grammar == "jaali_lattice":
        checks.append(_check("coverage_bound", True,
                             message="Jaali openness is governed by jaali_open_fraction"))
    else:
        checks.append(_check("coverage_bound", coverage_ok,
                             message=f"Ink coverage {coverage:.4f}; allowed [{pattern.constraints.min_coverage:.4f}, {pattern.constraints.max_coverage:.4f}]",
                             witness=None if coverage_ok else {"coverage": coverage, "allowed": [pattern.constraints.min_coverage, pattern.constraints.max_coverage]}))

    if isinstance(pattern, PatternV2):
        if pattern.grammar == "dot_field":
            checks.append(check_dot_clearance(pattern, (pattern.tile.wrap_x, pattern.tile.wrap_y)))
        elif pattern.grammar == "stripe_band":
            declaration = pattern.constraints.translation_period
            checks.append(
                _check("translation_period", False, "stripe_band requires translation_period")
                if declaration is None
                else check_translation_period(grid, declaration.axis, declaration.period)
            )
        elif pattern.grammar == "counted_grid":
            size = pattern.constraints.cell_size
            checks.append(
                _check("cell_quantization", False, "counted_grid requires cell_size")
                if size is None
                else check_cell_quantization(masks, size)
            )
        elif pattern.grammar == "kolam_loop":
            checks.extend([
                check_mirrors_valid(pattern),
                check_loop_count(pattern),
                check_mirror_symmetry(pattern),
            ])
        elif pattern.grammar == "jaali_lattice":
            wrapped = pattern.tile.wrap_x and pattern.tile.wrap_y
            checks.extend([
                check_single_solid(masks["solid"], wrap=wrapped),
                check_holes_interior(masks["holes"], pattern.jaali.frame, wrap=wrapped),
                check_open_fraction(
                    masks["holes"], pattern.constraints.min_open_fraction,
                    pattern.constraints.max_open_fraction,
                ),
                check_min_web(masks["solid"], pattern.constraints.min_web, wrap=wrapped),
            ])

    return checks, masks, grid


def all_pass(checks: list[dict]) -> bool:
    return all(item["status"] == "PASS" for item in checks)
