"""Traced kolam loops as smoothed SVG paths (algorithm from the spec's reference/kolam_svg.py)."""
from __future__ import annotations

import re

from pramaan.kolam import loop_count, trace_loops

from .replay import Run

UNIT = 40.0
_ASKED_GRID = re.compile(r"(\d+)\s*[x×]\s*(\d+)[^.]*?dot grid")


def _point(point: tuple[int, int], unit: float) -> tuple[float, float]:
    x, y = point
    return (x + 1.5) * unit, (y + 1.5) * unit


def _path(loop: list[tuple[int, int]], unit: float) -> str:
    vertices = loop[:-1] if loop and loop[0] == loop[-1] else loop
    points = [_point(p, unit) for p in vertices]
    mid = lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)  # noqa: E731
    start = mid(points[-1], points[0])
    parts = [f"M{start[0]:.1f} {start[1]:.1f}"]
    for i, control in enumerate(points):
        end = mid(control, points[(i + 1) % len(points)])
        parts.append(f"Q{control[0]:.1f} {control[1]:.1f} {end[0]:.1f} {end[1]:.1f}")
    return " ".join(parts) + "Z"


def loop_paths(rows: int, cols: int, mirrors: list[list[int]], unit: float = UNIT) -> list[str]:
    loops = trace_loops(rows, cols, frozenset(tuple(m) for m in mirrors))
    return [_path(loop, unit) for loop in loops]


def asked_grid(brief: str) -> tuple[int, int] | None:
    match = _ASKED_GRID.search(brief)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _try(item) -> dict:
    entry = {"n": item.n, "outcome": item.outcome, "reward": item.reward, "rows": None, "cols": None,
             "mirrors": None, "loops": None, "paths": []}
    spec = item.kolam if item.grid is not None else None
    if isinstance(spec, dict):
        rows, cols, mirrors = spec.get("rows"), spec.get("cols"), spec.get("mirrors") or []
        entry.update(rows=rows, cols=cols, mirrors=len(mirrors))
        try:
            entry["loops"] = loop_count(rows, cols, frozenset(tuple(m) for m in mirrors))
            entry["paths"] = loop_paths(rows, cols, mirrors)
        except ValueError:
            entry["loops"] = None
    return entry


def kolam_data(runs: dict[tuple[str, str], Run], tasks) -> dict:
    """Per kolam brief: the asked grid, then every model's tries as drawings."""
    out = {}
    for task in tasks:
        if task.grammar != "kolam_loop":
            continue
        grid = asked_grid(task.brief)
        out[task.id] = {
            "rows": grid[0] if grid else None, "cols": grid[1] if grid else None,
            "required_loops": task.constraint_profile.get("required_loops"),
            "unit": UNIT,
            "models": {model: [_try(item) for item in run.tries]
                       for (model, task_id), run in sorted(runs.items()) if task_id == task.id},
        }
    return out
