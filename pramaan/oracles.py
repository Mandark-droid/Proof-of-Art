"""Constraint-witness candidates for difficult v0.2 benchmark grammars.

These are procedural verifier fixtures, not culturally grounded model outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any

from .checks.core import all_pass, verify_pattern
from .checks.kolam import _transforms
from .kolam import interior_midpoints, loop_count
from .orchestrate import _parse_task_candidate
from .render import mask_digest
from .tasks import Task, load_suite

_TARGET_GRAMMARS = {"rotational_field", "kolam_loop", "jaali_lattice"}
_ROOT = Path(__file__).resolve().parents[1]


def _base(task: Task, layers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "0.2",
        "grammar": task.grammar,
        "tradition_id": task.tradition_id,
        "title": "Procedural constraint witness",
        "tile": {"width": 64, "height": 64, "wrap_x": False, "wrap_y": False},
        "palette": list(task.palette),
        "background_index": 0,
        "layers": layers,
        "constraints": {"symmetry": task.symmetry},
    }


def _rotational_candidate(task: Task) -> dict[str, Any]:
    # Four equal overlapping disks around the half-pixel centre form a raster
    # invariant under the full square symmetry group, including C4 and D2.
    dots = [
        {"kind": "dot", "center": [x, y], "radius": 6}
        for x, y in ((31, 31), (32, 31), (31, 32), (32, 32))
    ]
    candidate = _base(task, [{"id": "orbit", "colour_index": 1, "anchor": [0, 0], "motifs": dots}])
    return candidate


def _jaali_candidate(task: Task) -> dict[str, Any]:
    minimum = float(task.constraint_profile.get("min_open_fraction", 0.12))
    side = max(32, math.ceil(math.sqrt(minimum * 64 * 64)))
    if side % 2:
        side += 1
    # A single centred square hole is a deterministic symmetric witness; the
    # empty layer asks the renderer for a continuous solid web around it.
    candidate = _base(task, [{"id": "web", "colour_index": 1, "anchor": [0, 0], "motifs": []}])
    candidate["jaali"] = {
        "frame": int(task.constraint_profile.get("frame", 4)),
        "open_colour_index": 2 if len(task.palette) > 2 else 1,
        "holes": [{"kind": "rectangle", "center": [32, 32], "size": [side, side]}],
    }
    return candidate


def _kolam_dimensions(task: Task) -> tuple[int, int]:
    match = re.search(r"\b(\d+)\s*[x×]\s*(\d+)\b", task.brief, flags=re.IGNORECASE)
    return (int(match.group(1)), int(match.group(2))) if match else (3, 3)


def _symmetric_mirror_orbits(rows: int, cols: int, symmetry: str):
    interior = interior_midpoints(rows, cols)
    transforms = [transform for _name, transform in _transforms(rows, cols, symmetry)]
    unseen = set(interior)
    orbits = []
    while unseen:
        seed = min(unseen)
        orbit = {seed}
        frontier = [seed]
        while frontier:
            point = frontier.pop()
            for transform in transforms:
                mapped = transform(*point)
                if mapped in interior and mapped not in orbit:
                    orbit.add(mapped)
                    frontier.append(mapped)
        orbits.append(tuple(sorted(orbit)))
        unseen.difference_update(orbit)
    return orbits


def _kolam_candidate(task: Task) -> dict[str, Any] | None:
    rows, cols = _kolam_dimensions(task)
    required = int(task.constraint_profile.get("required_loops", 1))
    orbits = _symmetric_mirror_orbits(rows, cols, task.symmetry)
    if len(orbits) > 22:
        return None
    # Enumerate symmetry-closed sets in increasing size; all candidates are
    # exact subsets of interior mirror positions, up to the DSL limit of 40.
    for count in range(len(orbits) + 1):
        for selected in combinations(orbits, count):
            mirrors = sorted(point for orbit in selected for point in orbit)
            if len(mirrors) > 40:
                continue
            if loop_count(rows, cols, frozenset(mirrors)) != required:
                continue
            candidate = _base(task, [
                {"id": "curve", "colour_index": 1, "anchor": [0, 0], "motifs": []},
                {"id": "dots", "colour_index": min(2, len(task.palette) - 1), "anchor": [0, 0], "motifs": []},
            ])
            candidate["kolam"] = {
                "rows": rows, "cols": cols, "spacing": 8, "mirrors": mirrors,
                "stroke": 3, "dot_radius": 2,
            }
            candidate["constraints"]["required_loops"] = required
            return candidate
    return None


def find_oracle_candidate(task: Task) -> dict[str, Any] | None:
    """Return a passing host-profiled candidate for a supported task, if found."""
    if task.grammar == "rotational_field":
        candidates = [_rotational_candidate(task)]
    elif task.grammar == "jaali_lattice":
        candidates = [_jaali_candidate(task)]
    elif task.grammar == "kolam_loop":
        candidate = _kolam_candidate(task)
        candidates = [] if candidate is None else [candidate]
    else:
        return None

    for candidate in candidates:
        try:
            pattern = _parse_task_candidate(json.dumps(candidate), task)
            checks, _masks, grid = verify_pattern(pattern)
        except (TypeError, ValueError):
            continue
        if all_pass(checks):
            return pattern.model_dump(mode="json")
    return None


def build_oracle_rows(*, suites: dict[str, tuple[list[Task], str]] | None = None) -> list[dict[str, Any]]:
    """Build replayable procedural witnesses and explicit unsat rows for current suites."""
    if suites is None:
        suites = {name: load_suite(name) for name in ("eval-v2.1", "train-v1.2")}
    rows = []
    for suite_id, (tasks, digest) in suites.items():
        for task in tasks:
            if task.grammar not in _TARGET_GRAMMARS:
                continue
            candidate = find_oracle_candidate(task)
            if candidate is None:
                if task.grammar == "kolam_loop":
                    orbit_count = len(_symmetric_mirror_orbits(*_kolam_dimensions(task), task.symmetry))
                    status = "unsatisfiable" if orbit_count <= 22 else "search_limit"
                    reason = (
                        "Exhaustive search found no symmetry-closed mirror set with the required loop count."
                        if status == "unsatisfiable" else
                        f"Search capped at 22 symmetry orbits; this task has {orbit_count}."
                    )
                else:
                    status = "no_witness_found"
                    reason = "The bounded procedural witness did not pass the verifier."
                rows.append({
                    "suite_id": suite_id, "suite_sha256": digest, "task_id": task.id,
                    "grammar": task.grammar, "source": "procedural_oracle",
                    "oracle_status": status, "reason": reason, "candidate_pattern": None,
                    "checks": [], "mask_sha256": None, "all_hard_constraints_pass": False,
                })
                continue
            pattern = _parse_task_candidate(json.dumps(candidate), task)
            checks, masks, grid = verify_pattern(pattern)
            rows.append({
                "suite_id": suite_id, "suite_sha256": digest, "task_id": task.id,
                "grammar": task.grammar, "source": "procedural_oracle",
                "oracle_status": "verified", "candidate_pattern": candidate,
                "checks": checks, "mask_sha256": mask_digest(masks, grid),
                "all_hard_constraints_pass": all_pass(checks),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", choices=("eval-v2.1", "train-v1.1", "train-v1.2"))
    parser.add_argument("--output", type=Path, default=_ROOT / "benchmarks/data/eval-v2.1-procedural-oracles.jsonl")
    args = parser.parse_args(argv)
    suites = {name: load_suite(name) for name in (args.suite or ["eval-v2.1", "train-v1.2"])}
    rows = build_oracle_rows(suites=suites)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    unsat = [row["task_id"] for row in rows if row["oracle_status"] != "verified"]
    print(f"wrote {len(rows)} procedural oracle rows; {len(rows) - len(unsat)} verified; unsatisfiable={unsat}")
    unresolved = [row["task_id"] for row in rows if row["oracle_status"] not in {"verified", "unsatisfiable"}]
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
