"""Doubled-coordinate mirror-curve tracer for Sikku Kolam geometry."""

from __future__ import annotations

from math import gcd
from typing import Iterable

Point = tuple[int, int]
Direction = tuple[int, int]
RayState = tuple[Point, Direction]

_DIRECTIONS: tuple[Direction, ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _validate_dimensions(rows: int, cols: int) -> None:
    if not isinstance(rows, int) or not isinstance(cols, int) or not (1 <= rows <= 9 and 1 <= cols <= 9):
        raise ValueError("Kolam rows and cols must be integers in [1, 9]")


def interior_midpoints(rows: int, cols: int) -> set[Point]:
    _validate_dimensions(rows, cols)
    points = {
        (2 * i + 1, 2 * j)
        for j in range(rows)
        for i in range(cols - 1)
    }
    points.update(
        (2 * i, 2 * j + 1)
        for i in range(cols)
        for j in range(rows - 1)
    )
    return points


def mirror_symmetry_orbits(rows: int, cols: int, symmetry: str) -> list[tuple[Point, ...]]:
    """Partition interior mirror positions into exact symmetry orbits.

    Returned orbits can be combined to enumerate mirror sets that are closed
    under the requested symmetry. C4 and D4 require a square dot grid.
    """
    _validate_dimensions(rows, cols)
    if symmetry not in {"None", "C2", "C4", "D2", "D4"}:
        raise ValueError(f"unsupported Kolam symmetry: {symmetry}")
    if symmetry in {"C4", "D4"} and rows != cols:
        raise ValueError(f"{symmetry} Kolam symmetry requires a square grid")

    cx, cy = cols - 1, rows - 1

    def r180(x: int, y: int) -> Point:
        return 2 * cx - x, 2 * cy - y

    def vertical(x: int, y: int) -> Point:
        return 2 * cx - x, y

    def horizontal(x: int, y: int) -> Point:
        return x, 2 * cy - y

    def r90(x: int, y: int) -> Point:
        return cx + (y - cy), cy - (x - cx)

    def r270(x: int, y: int) -> Point:
        return cx - (y - cy), cy + (x - cx)

    def diagonal(x: int, y: int) -> Point:
        return cx + (y - cy), cy + (x - cx)

    def antidiagonal(x: int, y: int) -> Point:
        return cx - (y - cy), cy - (x - cx)

    transforms = []
    if symmetry in {"C2", "C4", "D2", "D4"}:
        transforms.append(r180)
    if symmetry in {"D2", "D4"}:
        transforms.extend((vertical, horizontal))
    if symmetry == "C4":
        transforms.extend((r90, r270))
    elif symmetry == "D4":
        transforms.extend((r90, r270, diagonal, antidiagonal))

    interior = interior_midpoints(rows, cols)
    unseen = set(interior)
    orbits: list[tuple[Point, ...]] = []
    while unseen:
        seed = min(unseen)
        orbit = {seed}
        frontier = [seed]
        while frontier:
            point = frontier.pop()
            for transform in transforms:
                mapped = transform(*point)
                if mapped not in interior:
                    raise ValueError("requested symmetry does not preserve the Kolam midpoint grid")
                if mapped not in orbit:
                    orbit.add(mapped)
                    frontier.append(mapped)
        unseen.difference_update(orbit)
        orbits.append(tuple(sorted(orbit)))
    return orbits


def _as_point(value) -> Point | None:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(not isinstance(coordinate, int) or isinstance(coordinate, bool) for coordinate in value)
    ):
        return None
    return int(value[0]), int(value[1])


def validate_mirrors(rows: int, cols: int, mirrors: Iterable[Point]) -> list[Point]:
    valid = interior_midpoints(rows, cols)
    invalid = []
    for mirror in mirrors:
        point = _as_point(mirror)
        if point is None or point not in valid:
            invalid.append(tuple(mirror) if isinstance(mirror, (tuple, list)) else mirror)
    return sorted(invalid, key=repr)


def _normalize_mirrors(rows: int, cols: int, mirrors: Iterable[Point]) -> frozenset[Point]:
    positions = frozenset(_as_point(point) for point in mirrors)
    if None in positions:
        raise ValueError("mirror positions must be integer coordinate pairs")
    invalid = validate_mirrors(rows, cols, positions)
    if invalid:
        raise ValueError(f"invalid Kolam mirror positions: {invalid}")
    return positions


def _is_frame(point: Point, rows: int, cols: int) -> bool:
    x, y = point
    return x in {-1, 2 * cols - 1} or y in {-1, 2 * rows - 1}


def _next_state(
    state: RayState,
    rows: int,
    cols: int,
    interior: set[Point],
    mirrors: frozenset[Point],
) -> tuple[RayState, list[Point]]:
    (x, y), (dx, dy) = state
    q = (x + dx, y + dy)
    reflections: list[Point] = []
    reflection_limit = len(interior) * 4
    steps = 0
    while _is_frame(q, rows, cols) or q in mirrors:
        if steps >= reflection_limit:
            raise ValueError("Kolam ray exceeded the bounded reflection-loop guard")
        reflections.append(q)
        qx, qy = q
        if _is_frame(q, rows, cols):
            if qx in {-1, 2 * cols - 1}:
                dx = -dx
            if qy in {-1, 2 * rows - 1}:
                dy = -dy
        else:
            # Vertical-gap mirrors reverse horizontal travel; horizontal-gap
            # mirrors reverse vertical travel, following the construction spec.
            if qx % 2 == 0:
                dx = -dx
            else:
                dy = -dy
        q = (qx + dx, qy + dy)
        steps += 1
    if q not in interior:
        raise ValueError(f"Kolam ray escaped its frame at {q}")
    return (q, (dx, dy)), reflections


def _orbits(rows: int, cols: int, mirrors: frozenset[Point]):
    interior = interior_midpoints(rows, cols)
    states = {
        (point, direction)
        for point in interior - mirrors
        for direction in _DIRECTIONS
    }
    unseen = set(states)
    result = []
    while unseen:
        start = min(unseen)
        current = start
        points = [start[0]]
        local = set()
        while current not in local:
            if current not in unseen:
                raise ValueError("Kolam transitions do not form disjoint state orbits")
            local.add(current)
            unseen.remove(current)
            following, reflections = _next_state(current, rows, cols, interior, mirrors)
            points.extend(reflections)
            points.append(following[0])
            current = following
        if current != start:
            raise ValueError("Kolam ray orbit did not close at its starting state")
        result.append(points)
    return result


def _canonical_loop(points: list[Point]) -> tuple[Point, ...]:
    vertices = points[:-1] if len(points) > 1 and points[0] == points[-1] else points
    if not vertices:
        return ()
    candidates = []
    for sequence in (vertices, list(reversed(vertices))):
        for offset in range(len(sequence)):
            candidates.append(tuple(sequence[offset:] + sequence[:offset]))
    return min(candidates)


def trace_loops(rows: int, cols: int, mirrors: frozenset[Point]) -> list[list[Point]]:
    _validate_dimensions(rows, cols)
    normalized = _normalize_mirrors(rows, cols, mirrors)
    if not interior_midpoints(rows, cols):
        # The 1x1 dot has one degenerate closed loop by gcd(1,1) convention.
        if rows == cols == 1:
            return [[(-1, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]]
        return []
    loops: dict[tuple[Point, ...], list[Point]] = {}
    for points in _orbits(rows, cols, normalized):
        canonical = _canonical_loop(points)
        loops.setdefault(canonical, list(canonical) + ([canonical[0]] if canonical else []))
    return [loops[key] for key in sorted(loops)]


def loop_count(rows: int, cols: int, mirrors: frozenset[Point]) -> int:
    _validate_dimensions(rows, cols)
    normalized = _normalize_mirrors(rows, cols, mirrors)
    if not interior_midpoints(rows, cols):
        return gcd(rows, cols)
    orbit_count = len(_orbits(rows, cols, normalized))
    if orbit_count % 2:
        raise ValueError("Kolam ray permutation produced an unpaired directed orbit")
    return orbit_count // 2
