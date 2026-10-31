"""Connectivity, interior-hole, openness and approximate web checks for G7."""

from __future__ import annotations

from collections import deque
import math
from typing import Any

import numpy as np


def centered_square_hole_sides(
    width: int,
    height: int,
    *,
    minimum: float,
    maximum: float,
    frame: int = 0,
) -> tuple[int, ...]:
    """Return square hole side lengths whose area meets a Jaali open-fraction range.

    A bounded panel keeps the square within ``frame`` pixels of every edge.
    An empty tuple means no integer-sized centered square can satisfy the range.
    """
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in (width, height)):
        raise ValueError("Jaali canvas dimensions must be positive integers")
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise ValueError("Jaali frame must be a non-negative integer")
    fractions = (minimum, maximum)
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
        for value in fractions
    ) or not (0.0 <= minimum <= maximum <= 1.0):
        raise ValueError("Jaali open-fraction bounds must satisfy 0 <= minimum <= maximum <= 1")

    largest = min(width, height) - 2 * frame
    return tuple(
        side for side in range(1, max(0, largest) + 1)
        if minimum <= (side * side) / (width * height) <= maximum
    )


def _check(id: str, passed: bool, message: str, witness: Any = None, assurance="EXACT_DISCRETE") -> dict:
    result = {"id": id, "status": "PASS" if passed else "FAIL", "assurance": assurance, "message": message}
    if witness is not None:
        result["counterexample"] = witness
    return result


def connected_components(mask: np.ndarray, wrap: bool) -> tuple[int, np.ndarray]:
    if mask.ndim != 2:
        raise ValueError("component mask must be a two-dimensional array")
    height, width = mask.shape
    labels = np.full(mask.shape, -1, dtype=np.int32)
    component = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != -1:
                continue
            labels[y, x] = component
            queue = deque([(x, y)])
            while queue:
                cx, cy = queue.popleft()
                for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
                    nx, ny = cx + dx, cy + dy
                    if wrap:
                        nx %= width
                        ny %= height
                    elif not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if mask[ny, nx] and labels[ny, nx] == -1:
                        labels[ny, nx] = component
                        queue.append((nx, ny))
            component += 1
    return component, labels


def check_single_solid(mask: np.ndarray, wrap: bool) -> dict:
    count, labels = connected_components(mask, wrap)
    if count == 1:
        return _check("jaali_single_solid", True, "Solid panel is one 4-connected component")
    if count == 0:
        witness = {"component_count": 0}
    else:
        sizes = np.bincount(labels[labels >= 0])
        smallest = int(np.flatnonzero(sizes == sizes.min())[0])
        y, x = (int(value) for value in np.argwhere(labels == smallest)[0])
        witness = {"component_count": count, "pixel": [x, y], "component_size": int(sizes[smallest])}
    return _check("jaali_single_solid", False, f"Solid panel has {count} separate 4-connected components", witness)


def check_holes_interior(holes: np.ndarray, frame: int, wrap: bool) -> dict:
    if wrap:
        return _check("jaali_holes_interior", True, "Wrapped jaali has no bounded frame")
    coords = np.argwhere(holes)
    height, width = holes.shape
    for y, x in coords:
        y, x = int(y), int(x)
        if x < frame or y < frame or x >= width - frame or y >= height - frame:
            return _check(
                "jaali_holes_interior", False,
                f"Hole pixel [{x}, {y}] enters the {frame}px panel frame",
                {"pixel": [x, y], "frame": frame},
            )
    return _check("jaali_holes_interior", True, f"All hole pixels stay at least {frame}px from the panel edge")


def check_open_fraction(holes: np.ndarray, minimum: float | None, maximum: float | None) -> dict:
    if minimum is None or maximum is None:
        return _check("jaali_open_fraction", False, "jaali_lattice requires minimum and maximum open fractions")
    fraction = float(np.count_nonzero(holes)) / float(holes.size)
    passed = minimum <= fraction <= maximum
    witness = None if passed else {"open_fraction": fraction, "allowed": [minimum, maximum]}
    return _check("jaali_open_fraction", passed, f"Open fraction {fraction:.4f}; allowed [{minimum:.4f}, {maximum:.4f}]", witness)


def _shift(mask: np.ndarray, dx: int, dy: int, wrap: bool) -> np.ndarray:
    if wrap:
        return np.roll(mask, shift=(dy, dx), axis=(0, 1))
    height, width = mask.shape
    shifted = np.zeros_like(mask)
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    if src_x0 < src_x1 and src_y0 < src_y1:
        shifted[src_y0 + dy:src_y1 + dy, src_x0 + dx:src_x1 + dx] = mask[src_y0:src_y1, src_x0:src_x1]
    return shifted


def min_web_witness(mask: np.ndarray, t: int, wrap: bool = False) -> tuple[int, int] | None:
    if not isinstance(t, int) or t < 0:
        raise ValueError("web radius t must be a non-negative integer")
    eroded = np.asarray(mask, dtype=bool).copy()
    for dy in range(-t, t + 1):
        for dx in range(-t, t + 1):
            eroded &= _shift(mask, dx, dy, wrap)
    opened = np.zeros_like(eroded)
    for dy in range(-t, t + 1):
        for dx in range(-t, t + 1):
            opened |= _shift(eroded, dx, dy, wrap)
    removed = np.argwhere(mask & ~opened)
    if not len(removed):
        return None
    y, x = (int(value) for value in removed[0])
    return x, y


def check_min_web(mask: np.ndarray, t: int | None, wrap: bool = False) -> dict:
    if t is None:
        return _check("jaali_min_web", True, "No minimum web thickness was declared", assurance="APPROXIMATE")
    witness = min_web_witness(mask, t, wrap)
    return _check(
        "jaali_min_web", witness is None,
        f"Square opening approximation removes a solid pixel for web radius {t}" if witness else f"All solid pixels survive the square opening for web radius {t}",
        None if witness is None else {"pixel": list(witness), "radius": t},
        assurance="APPROXIMATE",
    )
