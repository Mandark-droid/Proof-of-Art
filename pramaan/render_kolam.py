"""Rasterize mirror-curve Kolam into independent curve and dot masks."""

from __future__ import annotations

import numpy as np

from .dsl import KolamSpec
from .kolam import interior_midpoints, trace_loops, validate_mirrors
from .render import N


def render_kolam(spec: KolamSpec, palette_len: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if palette_len < 1:
        raise ValueError("Kolam rendering requires at least one palette colour")
    invalid = set(validate_mirrors(spec.rows, spec.cols, spec.mirrors))
    valid_mirrors = frozenset(tuple(point) for point in spec.mirrors if tuple(point) not in invalid and tuple(point) in interior_midpoints(spec.rows, spec.cols))
    loops = trace_loops(spec.rows, spec.cols, valid_mirrors)

    # Center on the raster's pixel-center axis (31.5); the integer-floor origin
    # shifts an even-spaced dot array by half a pixel and makes D4 impossible.
    ox = (N - 1 - (spec.cols - 1) * spec.spacing) / 2
    oy = (N - 1 - (spec.rows - 1) * spec.spacing) / 2
    yy, xx = np.indices((N, N), dtype=np.float64)

    curve_mask = np.zeros((N, N), dtype=bool)
    for loop in loops:
        points = [
            (ox + x * spec.spacing / 2, oy + y * spec.spacing / 2)
            for x, y in loop
        ]
        for start, end in zip(points, points[1:]):
            x0, y0 = start
            x1, y1 = end
            dx, dy = x1 - x0, y1 - y0
            length_sq = dx * dx + dy * dy
            if length_sq == 0:
                distance_sq = (xx - x0) ** 2 + (yy - y0) ** 2
            else:
                projection = np.clip(((xx - x0) * dx + (yy - y0) * dy) / length_sq, 0, 1)
                nearest_x = x0 + projection * dx
                nearest_y = y0 + projection * dy
                distance_sq = (xx - nearest_x) ** 2 + (yy - nearest_y) ** 2
            curve_mask |= distance_sq <= (spec.stroke / 2) ** 2

    dot_mask = np.zeros((N, N), dtype=bool)
    for row in range(spec.rows):
        for col in range(spec.cols):
            x = ox + col * spec.spacing
            y = oy + row * spec.spacing
            radius = spec.dot_radius + 0.5
            dot_mask |= (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2

    masks = {
        "curve": curve_mask,
        "dots": dot_mask,
    }
    grid = np.zeros((N, N), dtype=np.uint8)
    grid[masks["curve"]] = 1
    if palette_len >= 3:
        grid[masks["dots"]] = 2
    else:
        grid[masks["dots"]] = 1
    return masks, grid
