"""Exact grid checks for PatternDSL v0.2 grammars G2–G4."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..dsl import PatternV2


def _check(id: str, passed: bool, message: str, witness: Any = None) -> dict:
    result = {
        "id": id,
        "status": "PASS" if passed else "FAIL",
        "assurance": "EXACT_DISCRETE",
        "message": message,
    }
    if witness is not None:
        result["counterexample"] = witness
    return result


def check_dot_clearance(pattern: PatternV2, wrap: bool | tuple[bool, bool]) -> dict:
    """Check center distances between dot motifs, with independent wrapped axes."""
    minimum = pattern.constraints.min_clearance
    if minimum is None:
        return _check("dot_clearance", False, "dot_field requires min_clearance")
    wrap_x, wrap_y = (wrap, wrap) if isinstance(wrap, bool) else wrap
    dots = []
    for layer in pattern.layers:
        for motif_index, motif in enumerate(layer.motifs):
            if motif.kind == "dot":
                dots.append((layer.id, motif_index, motif, layer.anchor))
    for index, first in enumerate(dots):
        first_layer, first_index, first_motif, first_anchor = first
        x1 = int(first_motif.center[0] + first_anchor[0])
        y1 = int(first_motif.center[1] + first_anchor[1])
        for second in dots[index + 1:]:
            second_layer, second_index, second_motif, second_anchor = second
            x2 = int(second_motif.center[0] + second_anchor[0])
            y2 = int(second_motif.center[1] + second_anchor[1])
            dx, dy = x2 - x1, y2 - y1
            if wrap_x:
                dx = min((dx, dx - 64, dx + 64), key=abs)
            if wrap_y:
                dy = min((dy, dy - 64, dy + 64), key=abs)
            distance_sq = dx * dx + dy * dy
            required = int(first_motif.radius + second_motif.radius + minimum)
            if distance_sq < required * required:
                witness = {
                    "motifs": [[first_layer, first_index], [second_layer, second_index]],
                    "centers": [[x1, y1], [x2, y2]],
                    "distance_sq": distance_sq,
                    "required_distance_sq": required * required,
                    "wrapped_delta": [dx, dy],
                }
                return _check(
                    "dot_clearance", False,
                    f"Dot centres are too close: distance² {distance_sq}; required ≥ {required * required}",
                    witness,
                )
    return _check("dot_clearance", True, "Every dot pair meets the declared minimum clearance")


def check_translation_period(grid: np.ndarray, axis: str, period: int) -> dict:
    if axis not in {"x", "y", "xy"} or not isinstance(period, int) or period < 1:
        return _check("translation_period", False, "Invalid translation period declaration")
    axes = {"x": (1,), "y": (0,), "xy": (1, 0)}[axis]
    for array_axis in axes:
        shifted = np.roll(grid, shift=period, axis=array_axis)
        differences = np.argwhere(grid != shifted)
        if len(differences):
            y, x = (int(value) for value in differences[0])
            axis_name = "x" if array_axis == 1 else "y"
            return _check(
                "translation_period", False,
                f"Grid differs after translation by {period} on {axis_name}",
                {
                    "axis": axis_name,
                    "period": period,
                    "pixel": [x, y],
                    "original_colour": int(grid[y, x]),
                    "translated_colour": int(shifted[y, x]),
                },
            )
    return _check("translation_period", True, f"Grid repeats after translation on {axis}")


def check_cell_quantization(masks: dict[str, np.ndarray], cell_size: int) -> dict:
    if cell_size not in {2, 4, 8}:
        return _check("cell_quantization", False, "Cell size must be 2, 4, or 8")
    for layer_id, mask in masks.items():
        if mask.ndim != 2 or mask.shape[0] % cell_size or mask.shape[1] % cell_size:
            return _check(
                "cell_quantization", False,
                "Layer mask dimensions must be divisible by cell_size",
                {"layer": layer_id, "shape": list(mask.shape), "cell_size": cell_size},
            )
        for y in range(0, mask.shape[0], cell_size):
            for x in range(0, mask.shape[1], cell_size):
                block = mask[y:y + cell_size, x:x + cell_size]
                if not np.all(block == block.flat[0]):
                    local_y, local_x = (int(value) for value in np.argwhere(block != block.flat[0])[0])
                    return _check(
                        "cell_quantization", False,
                        f"Layer {layer_id} has a non-uniform {cell_size}x{cell_size} cell",
                        {
                            "layer": layer_id,
                            "block": [x // cell_size, y // cell_size],
                            "pixel": [x + local_x, y + local_y],
                        },
                    )
    return _check("cell_quantization", True, f"Every layer mask is uniform within {cell_size}x{cell_size} cells")
