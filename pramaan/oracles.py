"""Constraint-witness candidates for difficult v0.2 benchmark grammars.

These are procedural verifier fixtures, not culturally grounded model outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from math import gcd
from pathlib import Path
from typing import Any

from .checks.core import all_pass, verify_pattern
from .checks.kolam import _transforms
from .kolam import interior_midpoints, loop_count
from .orchestrate import _parse_task_candidate
from .render import mask_digest
from .tasks import Task, load_suite
from .paths import resolve_project_root

_TARGET_GRAMMARS = {"rotational_field", "kolam_loop", "jaali_lattice"}
_ROOT = resolve_project_root(__file__)


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


def _symmetry_expand(candidate: dict[str, Any], symmetry: str) -> dict[str, Any]:
    """Close raster motifs under the host's exact pixel-centred group action."""
    if symmetry not in {"D2", "C4"}:
        return candidate
    names = ["id", "r180", "fx", "fy"] if symmetry == "D2" else ["id", "r90", "r180", "r270"]
    def point(x: int, y: int, name: str) -> tuple[int, int]:
        return {"id": (x, y), "r90": (y, 63-x), "r180": (63-x, 63-y),
                "r270": (63-y, x), "fx": (63-x, y), "fy": (x, 63-y)}[name]
    for layer in candidate.get("layers", []):
        expanded = {}
        for motif in layer.get("motifs", []):
            for name in names:
                item = json.loads(json.dumps(motif))
                if item["kind"] in {"dot", "diamond"}:
                    item["center"] = list(point(*item["center"], name))
                elif item["kind"] == "rectangle":
                    x, y = item["center"]
                    w, h = item["size"]
                    if name == "fx":
                        item["center"] = [64-x if w % 2 == 0 else 63-x, y]
                    elif name == "fy":
                        item["center"] = [x, 64-y if h % 2 == 0 else 63-y]
                    elif name == "r180":
                        item["center"] = [64-x if w % 2 == 0 else 63-x,
                                          64-y if h % 2 == 0 else 63-y]
                    elif name in {"r90", "r270"}:
                        item["center"] = list(point(x, y, name))
                        item["size"] = [h, w]
                else:
                    item["points"] = [list(point(x, y, name)) for x, y in item["points"]]
                expanded[json.dumps(item, sort_keys=True)] = item
        layer["motifs"] = list(expanded.values())
    return candidate


def _rotational_candidate(task: Task) -> dict[str, Any]:
    easy = _easy_rotational_candidate(task)
    if easy is not None:
        return easy
    if task.level == "hard" and task.id.startswith("jh_sohrai"):
        cream = _palette_index(task, "#EFE3C8", 0)
        red = _palette_index(task, "#A33A1E", 1)
        black = _palette_index(task, "#2B2622", 2)
        gold = _palette_index(task, "#D9A441", 3)
        rings = []
        centers = [(31, 31), (32, 31), (31, 32), (32, 32), (32, 14), (32, 6)]
        for label, ink, radius in (("kamal-cream", cream, 6),
                                   ("kamal-red", red, 4),
                                   ("kamal-charcoal", black, 2)):
            rings.append({"id": label, "colour_index": ink, "anchor": [0, 0],
                          "motifs": [{"kind": "dot", "center": [31, 31], "radius": radius},
                                     {"kind": "dot", "center": [32, 31], "radius": radius},
                                     {"kind": "dot", "center": [31, 32], "radius": radius},
                                     {"kind": "dot", "center": [32, 32], "radius": radius},
                                     {"kind": "dot", "center": [32, 14], "radius": radius},
                                     {"kind": "dot", "center": [32, 6], "radius": radius}]})
        edge_dots = [{"kind": "dot", "center": [x, 3], "radius": 1}
                     for x in (3, 9, 15, 21, 27, 36, 42, 48, 54, 60)]
        rings.append({"id": "edge-dots", "colour_index": gold, "anchor": [0, 0],
                      "motifs": edge_dots})
        return _symmetry_expand(_base(task, rings), "C4")
    # Four equal overlapping disks around the half-pixel centre form a raster
    # invariant under the full square symmetry group, including C4 and D2.
    dots = [
        {"kind": "dot", "center": [x, y], "radius": 6}
        for x, y in ((31, 31), (32, 31), (31, 32), (32, 32))
    ]
    candidate = _base(task, [{"id": "orbit", "colour_index": 1, "anchor": [0, 0], "motifs": dots}])
    return candidate


def _easy_rotational_candidate(task: Task) -> dict[str, Any] | None:
    """Brief-shaped C4 rosettes; circles are rasterized as nested disks."""
    if task.id == "jh_sohrai-khovar-wall-painting-easy-9":
        cream = _palette_index(task, "#EFE3C8", 0)
        red = _palette_index(task, "#B3202C", 1)
        white = cream
        black = _palette_index(task, "#1A1A1A", 3)
        centres = [(32, 32), (32, 13), (50, 32), (32, 50), (13, 32)]
        # C4 symmetry permits four outer circles plus the central one; the
        # requested six outer circles cannot form complete C4 orbits.
        layers = [{"id": "outer-white", "colour_index": white, "anchor": [0, 0],
                   "motifs": [{"kind": "dot", "center": list(c), "radius": 8} for c in centres]},
                  {"id": "middle-red", "colour_index": red, "anchor": [0, 0],
                   "motifs": [{"kind": "dot", "center": list(c), "radius": 5} for c in centres]},
                  {"id": "inner-black", "colour_index": black, "anchor": [0, 0],
                   "motifs": [{"kind": "dot", "center": list(c), "radius": 2} for c in centres]}]
        return _base(task, layers)
    if task.id == "kl_pookalam-easy-11":
        white = _palette_index(task, "#F7F3EA", 0)
        orange = _palette_index(task, "#E8541E", 1)
        red = _palette_index(task, "#C8102E", 2)
        yellow = _palette_index(task, "#F2C12E", 3)
        green = _palette_index(task, "#2E7D32", 5)
        rings = []
        for name, colour, radius in (("white", white, 27), ("orange", orange, 21),
                                     ("yellow", yellow, 15)):
            motifs = [{"kind": "diamond", "center": [x, y], "radius": radius}
                      for x, y in ((31, 31), (32, 31), (31, 32), (32, 32))]
            rings.append({"id": f"pookalam-{name}", "colour_index": colour,
                          "anchor": [0, 0], "motifs": motifs})
        centre = [{"kind": "diamond", "center": [x, y], "radius": 6}
                  for x, y in ((31, 31), (32, 31), (31, 32), (32, 32))]
        rings.append({"id": "pookalam-green-centre", "colour_index": green,
                      "anchor": [0, 0], "motifs": centre})
        candidate = _base(task, rings)
        return candidate
    if task.id == "uk_aipan-easy-26":
        white = _palette_index(task, "#F6F2EA", 1)
        # A four-point rosette retains the exact C4 host symmetry. A literal
        # five-point star cannot satisfy C4 without changing the brief.
        star = [{"kind": "diamond", "center": [32, 32], "radius": 22}]
        for x, y in ((32, 11), (53, 32), (32, 53), (11, 32)):
            star.append({"kind": "diamond", "center": [x, y], "radius": 11})
        star.append({"kind": "dot", "center": [32, 32], "radius": 3})
        return _base(task, [{"id": "aipan-c4-rosette", "colour_index": white,
                             "anchor": [0, 0], "motifs": star}])
    return None


def _jaali_candidate(task: Task) -> dict[str, Any]:
    easy = _easy_jaali_candidate(task)
    if easy is not None:
        return easy
    if task.level == "hard" and task.id.startswith("dl_mughal"):
        centers = (16, 47)
        holes = [{"kind": "rectangle", "center": [x, y], "size": [21, 21]}
                 for y in centers for x in centers]
        candidate = _base(task, [{"id": "sandstone-web", "colour_index": 0,
                                  "anchor": [0, 0], "motifs": []}])
        candidate["jaali"] = {"frame": 4, "open_colour_index": 1, "holes": holes}
        return candidate
    if task.level == "hard" and task.id.startswith("jk_khatamband"):
        centers = (10, 24, 40, 54)
        holes = [{"kind": "rectangle", "center": [x, y], "size": [8, 8]}
                 for y in centers for x in centers]
        candidate = _base(task, [{"id": "khatamband-web", "colour_index": 1,
                                  "anchor": [0, 0], "motifs": []}])
        candidate["jaali"] = {"frame": 4, "open_colour_index": 2, "holes": holes}
        return candidate
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


def _easy_jaali_candidate(task: Task) -> dict[str, Any] | None:
    if task.grammar != "jaali_lattice" or "easy-" not in task.id:
        return None
    frame = int(task.constraint_profile.get("frame", 4))
    background = 0
    web_colour = min(1, len(task.palette) - 1)
    open_colour = min(2, len(task.palette) - 1)
    centres = (10, 24, 40, 54)
    size = 8
    if task.id == "dl_mughal-sandstone-and-marble-jaali-screens-of-the-humayun-s-tomb-complex-easy-31":
        # Put the named red sandstone in the rendered tile and use cream for voids.
        web_colour = background = task.palette.index("#A3492B")
        open_colour = task.palette.index("#F2EEE6")
        holes = [{"kind": "rectangle", "center": [x, y], "size": [size, size]}
                 for y in centres for x in centres]
    elif task.id == "up_mughal-pierced-stone-jaali-screens-of-agra-and-fatehpur-sikri-easy-25":
        # Off-axis 7px holes and 6px center-axis holes fit the frozen frame,
        # openness, web, and D2 checks on a half-pixel-centered raster.
        sandstone = task.palette.index("#A0522D")
        background = web_colour = sandstone
        open_colour = task.palette.index("#F2EEE6")
        off_axis = (8, 20, 43, 55)
        holes = [{"kind": "rectangle", "center": [x, y], "size": [7, 7]}
                 for y in off_axis for x in off_axis]
        holes.extend({"kind": "rectangle", "center": [x, 32], "size": [6, 6]}
                     for x in (8, 20, 32, 44, 56))
        holes.extend({"kind": "rectangle", "center": [32, y], "size": [6, 6]}
                     for y in (8, 20, 44, 56))
    else:
        holes = [{"kind": "rectangle", "center": [x, y], "size": [size, size]}
                 for y in centres for x in centres]
    # The DSL has no star-cutout primitive; square openings preserve the
    # certified web and openness constraints. The gap is disclosed below.
    candidate = _base(task, [{"id": "jaali-solid-web", "colour_index": web_colour,
                              "anchor": [0, 0], "motifs": []}])
    candidate["background_index"] = background
    candidate["jaali"] = {"frame": frame, "open_colour_index": open_colour, "holes": holes}
    return candidate


def _counted_grid_candidate(task: Task) -> dict[str, Any]:
    """A centered cell-aligned square is a minimal witness for the cell grid."""
    easy = _easy_counted_grid_candidate(task)
    if easy is not None:
        return easy
    if task.level == "hard":
        return _hard_counted_grid_candidate(task)
    size = int(task.constraint_profile.get("cell_size", 4))
    side = max(size, 4 * size)
    side = min(64, side // size * size)
    return _base(task, [{
        "id": "reference-cells", "colour_index": min(1, len(task.palette) - 1),
        "anchor": [0, 0], "motifs": [{"kind": "rectangle", "center": [32, 32], "size": [side, side]}],
    }])


def _palette_index(task: Task, colour: str, fallback: int = 1) -> int:
    try:
        return task.palette.index(colour.upper())
    except ValueError:
        return min(fallback, len(task.palette) - 1)


def _cell_matrix_candidate(task: Task, matrix: list[list[int]], *, cell: int = 4,
                          background_index: int = 0) -> dict[str, Any]:
    """Convert a cell-grid raster into merged, exactly aligned rectangles."""
    if not matrix or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("reference cell matrix must be rectangular")
    height, width = len(matrix), len(matrix[0])
    by_colour: dict[int, list[dict[str, Any]]] = {}
    for colour in sorted({value for row in matrix for value in row} - {background_index}):
        remaining = {(x, y) for y, row in enumerate(matrix) for x, value in enumerate(row) if value == colour}
        motifs = []
        while remaining:
            x0, y0 = min(remaining, key=lambda point: (point[1], point[0]))
            x1 = x0 + 1
            while x1 < width and (x1, y0) in remaining:
                x1 += 1
            y1 = y0 + 1
            while y1 < height and all((x, y1) in remaining for x in range(x0, x1)):
                y1 += 1
            for y in range(y0, y1):
                for x in range(x0, x1):
                    remaining.remove((x, y))
            motif_width, motif_height = (x1 - x0) * cell, (y1 - y0) * cell
            motifs.append({"kind": "rectangle", "center": [x0 * cell + motif_width // 2,
                                                                y0 * cell + motif_height // 2],
                           "size": [motif_width, motif_height]})
        by_colour[colour] = motifs
    layers = [{"id": f"reference-ink-{colour}", "colour_index": colour,
               "anchor": [0, 0], "motifs": motifs}
              for colour, motifs in sorted(by_colour.items())]
    if not layers:
        raise ValueError("reference cell matrix must contain at least one ink cell")
    if any(len(layer["motifs"]) > 100 for layer in layers):
        raise ValueError("reference cell matrix exceeds the verifier's primitive budget")
    candidate = _base(task, layers)
    candidate["background_index"] = background_index
    return candidate


def _normalize_hard_matrix(matrix: list[list[int]], *, background: int = 0,
                           max_coverage: float = 0.68,
                           periodic_x: bool = False) -> list[list[int]]:
    """Close a hard reference under D2 and stay inside the frozen ink ceiling."""
    height, width = len(matrix), len(matrix[0])
    for y in range(height):
        for x in range(width):
            matrix[y][x] = matrix[min(y, height - 1 - y)][min(x, width - 1 - x)]
    target = int(height * width * max_coverage)
    ink = sum(value != background for row in matrix for value in row)
    if periodic_x:
        row_pairs = sorted(range(height // 2), key=lambda y: abs(y - (height - 1) / 2), reverse=True)
        for y in row_pairs:
            if ink <= target:
                break
            for yy in {y, height - 1 - y}:
                ink -= sum(value != background for value in matrix[yy])
                matrix[yy] = [background] * width
    else:
        cells = sorted(((x, y) for y in range(height // 2) for x in range(width // 2)
                        if matrix[y][x] != background),
                       key=lambda point: abs(point[0] - (width - 1) / 2)
                       + abs(point[1] - (height - 1) / 2), reverse=True)
        for x, y in cells:
            if ink <= target:
                break
            orbit = {(xx, yy) for xx in {x, width - 1 - x}
                     for yy in {y, height - 1 - y}}
            removed = sum(matrix[yy][xx] != background for xx, yy in orbit)
            for xx, yy in orbit:
                matrix[yy][xx] = background
            ink -= removed
    return matrix


def _hard_counted_grid_candidate(task: Task) -> dict[str, Any]:
    """Build a brief-led, D2-closed stitch or woven-cell reference per hard task."""
    n = 16
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    tid = task.id
    red = _palette_index(task, "#C41E3A", 1)
    white = _palette_index(task, "#F4F1EA", 2)
    gold = _palette_index(task, "#E8B42C", 2)
    black = _palette_index(task, "#1A1A1A", 1)
    green = _palette_index(task, "#2E7D4F", 2)
    orange = _palette_index(task, "#E8761B", 3)
    accent = min(1, len(task.palette) - 1)
    accent2 = min(2, len(task.palette) - 1)

    # Work only in one quadrant so every stitch has an exact D2 mirror.
    def set_d2(x: int, y: int, colour: int) -> None:
        for xx in {x, n - 1 - x}:
            for yy in {y, n - 1 - y}:
                matrix[yy][xx] = colour

    if tid.startswith("ap_mangalagiri"):
        for y in range(n):
            for x in range(n):
                if y in {0, 1, 14, 15}:
                    matrix[y][x] = gold
                elif ((x // 2 + y // 2) % 2) == 0:
                    matrix[y][x] = white
        # Inverted-check diamonds and their single-cell separators.
        for y in (0, 15):
            for x in range(1, 15, 2):
                matrix[y][x] = red
    elif tid.startswith("ar_idu"):
        for x in range(n):
            for y in range(n):
                d = min(abs(x - c) + abs(y - 7.5) for c in (3.5, 7.5, 11.5))
                if d <= 4.5:
                    matrix[y][x] = (red, accent2, white)[min(2, int(d // 1.5))]
                if y in {0, 15} and (x // 2) % 2 == 0:
                    matrix[y][x] = accent2 if (x // 2) % 4 == 0 else red
    elif tid.startswith("as_gamosa"):
        for y in range(n):
            for x in range(n):
                if x < 2 or x >= 14:
                    matrix[y][x] = red
                elif y in {0, 1, 14, 15}:
                    matrix[y][x] = white
                elif y in {2, 3, 12, 13} and ((x + y) % 3 == 0):
                    matrix[y][x] = red
        _draw_diamond_cells(matrix, cx=7.5, cy=7.5, radius=4.5, colour=red)
        for x, y in ((5, 7), (10, 7), (5, 8), (10, 8)):
            set_d2(x, y, accent2)
    elif tid.startswith("ga_kunbi"):
        for y in range(n):
            for x in range(n):
                if y < 3 or y >= 13:
                    matrix[y][x] = black if x % 4 in {0, 3} else white
                else:
                    matrix[y][x] = white if ((x // 2 + y // 2) % 2) else red
                    if x % 4 == 0:
                        matrix[y][x] = black
    elif tid.startswith("gj_patan"):
        for y in range(n):
            for x in range(n):
                if x in {0, 15} or y in {0, 15}:
                    matrix[y][x] = white if (x + y) % 2 else black
                else:
                    gx, gy = min(x, 15 - x), min(y, 15 - y)
                    d = abs((gx % 4) - 1.5) + abs((gy % 4) - 1.5)
                    if d <= 2.0:
                        matrix[y][x] = gold if ((gx // 4 + gy // 4) % 2 == 0) else green
                    else:
                        matrix[y][x] = red
    elif tid.startswith("hr_phulkari"):
        for y in range(n):
            for x in range(n):
                d = abs(x - 7.5) + abs(y - 7.5)
                if d <= 7.5:
                    matrix[y][x] = gold if d > 4.5 else orange if d > 1.5 else white
                elif y in {0, 15} and x % 2 == 0:
                    matrix[y][x] = white
                elif x in {0, 15} and y % 2 == 0:
                    matrix[y][x] = white
    elif tid.startswith("mn_moirang"):
        for y in range(n):
            for x in range(n):
                if y in {0, 1, 14, 15}:
                    matrix[y][x] = black if ((x // 2 + y) % 2 == 0) else white
                elif y in {2, 13}:
                    matrix[y][x] = black
                elif abs(y - 7.5) <= 2 and abs(x - 7.5) <= 6:
                    matrix[y][x] = black if ((x + y) % 2 == 0) else white
    elif tid.startswith("ml_garo"):
        for y in range(n):
            for x in range(n):
                if y in {0, 1, 14, 15}:
                    matrix[y][x] = red if x % 4 in {0, 3} else black
                elif y in {2, 13}:
                    matrix[y][x] = white if x % 2 == 0 else black
                else:
                    d = min(abs(x - cx) + abs(y - cy)
                            for cx in (3.5, 7.5, 11.5) for cy in (3.5, 11.5))
                    if d <= 3.5:
                        matrix[y][x] = (red, gold, white)[min(2, int(d // 1.5))]
    elif tid.startswith("od_sambalpuri"):
        for y in range(n):
            for x in range(n):
                if 3 <= x < 13 and 3 <= y < 13:
                    matrix[y][x] = white if ((x // 2 + y // 2) % 2) else black
                elif x in {0, 1, 14, 15} or y in {0, 1, 14, 15}:
                    matrix[y][x] = white if (x + y) % 2 == 0 else red
    elif tid.startswith("pb_phulkari"):
        # Four quadrant motifs: nested lozenges, chevrons, stitch dashes, and dots.
        for y in range(n):
            for x in range(n):
                gx, gy = min(x, 15 - x), min(y, 15 - y)
                patch = (0 if gx < 4 and gy < 4 else
                         1 if gx >= 4 and gy < 4 else
                         2 if gx < 4 else 3)
                lx, ly = gx % 4, gy % 4
                if patch == 0:
                    ink = abs(lx - 1.5) + abs(ly - 1.5) <= 2 and (lx, ly) not in {(1, 1), (2, 2)}
                elif patch == 1:
                    ink = (lx, ly) in {(0, 0), (1, 1), (2, 2), (3, 3), (0, 3), (1, 2), (2, 1), (3, 0)}
                elif patch == 2:
                    ink = (ly in {1, 2} and lx in {0, 3}) or (lx in {1, 2} and ly in {0, 3})
                else:
                    ink = (lx, ly) in {(0, 0), (3, 0), (1, 1), (2, 1), (1, 2), (2, 2), (0, 3), (3, 3)}
                if ink:
                    matrix[y][x] = gold if patch in {0, 3} else white
                if gx % 4 == 0 or gy % 4 == 0:
                    matrix[y][x] = gold
    elif tid.startswith("tn_toda"):
        for y in range(n):
            for x in range(n):
                band = min(y, n - 1 - y)
                if band in {2, 3}:
                    matrix[y][x] = red if abs(x - 7.5) <= (band + 1) else 0
                elif band in {0, 1, 4, 5}:
                    matrix[y][x] = black if (x + y) % 2 == 0 else 0
    elif tid.startswith("tg_pochampally"):
        for y in range(n):
            for x in range(n):
                cx, cy = (x // 4) * 4 + 1.5, (y // 4) * 4 + 1.5
                d = abs(x - cx) + abs(y - cy)
                if d <= 3:
                    matrix[y][x] = red if d >= 2 else black
                elif x % 4 in {0, 3} or y % 4 in {0, 3}:
                    matrix[y][x] = black
    elif tid.startswith("wb_nakshi"):
        for y in range(n):
            for x in range(n):
                dx, dy = abs(x - 7.5), abs(y - 7.5)
                d = dx + dy
                if (abs(dx - dy) <= 1.0 and max(dx, dy) <= 6.5) or d in {6.5, 7.5}:
                    matrix[y][x] = gold
                elif (x in {0, 15} or y in {0, 15}) and (x + y) % 2 == 0:
                    matrix[y][x] = red
                elif d <= 4.5 and (x + y) % 2 == 0:
                    matrix[y][x] = red if d > 1.5 else black
    elif tid.startswith("an_nicobari"):
        for y in range(n):
            for x in range(n):
                xx, yy = min(x, 15 - x), min(y, 15 - y)
                matrix[y][x] = accent if (xx - yy) % 4 in {0, 1} else 0
                if x in {0, 15} or y in {0, 15}:
                    matrix[y][x] = accent2
    elif tid.startswith("ch_phulkari"):
        for y in range(n):
            for x in range(n):
                d = abs(x - 7.5) + abs(y - 7.5)
                if d <= 7.5:
                    matrix[y][x] = gold if d > 5.5 else white if d > 3.5 else orange if d > 1.5 else 0
                elif d <= 8.5:
                    matrix[y][x] = gold
    elif tid.startswith("ld_lakshadweep"):
        for y in range(n):
            for x in range(n):
                matrix[y][x] = accent if ((x // 2 + y // 2) % 2) else 0
                if y % 4 == 0:
                    matrix[y][x] = red if x % 2 == 0 else green
                d = abs(x - 7.5) + abs(y - 7.5)
                if 3.5 <= d <= 5.5:
                    matrix[y][x] = accent2
    elif tid.startswith("up_lucknow"):
        blue_ink = _palette_index(task, "#9DB7D9", 2)
        white_ink = _palette_index(task, "#FFFFFF", 1)
        for y in range(n):
            for x in range(n):
                d = abs(x - 7.5) + abs(y - 7.5)
                if 4.5 <= d <= 6.5:
                    matrix[y][x] = blue_ink
                elif d < 4.5 and ((x - y) % 4 == 0 or (x + y) % 4 == 3):
                    matrix[y][x] = white_ink
                if (x, y) in {(2, 2), (13, 2), (2, 13), (13, 13)}:
                    matrix[y][x] = white_ink
    # Keep 4px stitch cells when the DSL primitive cap permits; otherwise compress
    # each 2x2 patch into an 8px cell while preserving the frozen alignment.
    _normalize_hard_matrix(matrix, background=0)
    native = _cell_matrix_candidate(task, matrix, cell=4)
    if sum(len(layer.get("motifs", [])) for layer in native["layers"]) <= 100:
        return native
    reduced = []
    for y in range(0, n, 2):
        row = []
        for x in range(0, n, 2):
            block = [matrix[yy][xx] for yy in (y, y + 1) for xx in (x, x + 1)]
            non_background = [value for value in block if value != 0]
            if not non_background:
                row.append(0)
            else:
                row.append(max(set(non_background), key=non_background.count))
        reduced.append(row)
    _normalize_hard_matrix(reduced, background=0)
    return _cell_matrix_candidate(task, reduced, cell=8)


def _draw_diamond_cells(matrix: list[list[int]], *, cx: float, cy: float, radius: float,
                        colour: int, outline: bool = False, width: float = 1.0) -> None:
    for y in range(len(matrix)):
        for x in range(len(matrix[0])):
            distance = abs(x - cx) + abs(y - cy)
            if (distance <= radius and (not outline or distance >= radius - width)):
                matrix[y][x] = colour


def _easy_counted_grid_candidate(task: Task) -> dict[str, Any] | None:
    """Brief-shaped easy references for the counted-grid subset."""
    task_id = task.id
    # Counted-grid maps use 4px cells; paired axes make the host D2 symmetry
    # visible in every design. Colours come from the task-owned palette.
    if task_id == "ap_mangalagiri-sarees-and-fabrics-easy-0":
        m = [[0] * 16 for _ in range(16)]
        white, gold = _palette_index(task, "#F4F1EA", 3), _palette_index(task, "#C9A227", 2)
        for y in range(1, 15):
            for x in range(1, 15):
                if (((x-1) // 2 + (y-1) // 2) % 2 == 0):
                    m[y][x] = white
        for x in range(16):
            m[0][x] = m[15][x] = gold
        return _cell_matrix_candidate(task, m)
    if task_id == "as_gamosa-red-on-white-border-and-motif-band-easy-2":
        m = [[0] * 16 for _ in range(16)]
        red = _palette_index(task, "#C41E3A")
        for k in range(16):
            m[0][k] = m[15][k] = m[k][0] = m[k][15] = red
        _draw_diamond_cells(m, cx=7.5, cy=7.5, radius=5.5, colour=red)
        return _cell_matrix_candidate(task, m)
    if task_id == "ga_kunbi-check-sari-easy-5":
        m = [[0] * 16 for _ in range(16)]
        white, black = _palette_index(task, "#F3EEE3"), _palette_index(task, "#1E1E1E", 2)
        for y in range(16):
            for x in range(16):
                if ((x-1) // 2 + (y-1) // 2) % 2:
                    m[y][x] = white
        for x in range(16):
            m[0][x] = m[15][x] = black
        return _cell_matrix_candidate(task, m)
    if task_id == "hr_phulkari-and-bagh-counted-thread-embroidery-easy-7":
        m = [[0] * 16 for _ in range(16)]
        yellow, white = _palette_index(task, "#F2B705", 2), _palette_index(task, "#FFF8E7", 3)
        _draw_diamond_cells(m, cx=7.5, cy=7.5, radius=6.5, colour=yellow)
        for y in (7, 8):
            for x in (7, 8):
                m[y][x] = white
        return _cell_matrix_candidate(task, m)
    if task_id == "ka_kasuti-embroidery-easy-10":
        m = [[0] * 16 for _ in range(16)]
        red = _palette_index(task, "#B11226")
        # Two mirrored double-running stepped diamonds joined by a stitch line.
        for cy in (4.5, 10.5):
            for cx in (4.5, 10.5):
                for y in range(16):
                    for x in range(16):
                        d = abs(x - cx) + abs(y - cy)
                        if d in (3.0, 4.0):
                            m[y][x] = red
        for x in range(5, 11):
            m[7][x] = m[8][x] = red
        return _cell_matrix_candidate(task, m)
    if task_id == "mn_moirang-phee-moirang-pheejin-stepped-triangle-border-easy-14":
        white, black = _palette_index(task, "#F6F2EA",1), _palette_index(task, "#1A1A1A",0)
        m = [[black] * 16 for _ in range(16)]
        for y in range(9,15):
            for x in range(4,12):
                height = 1 + min(x-4, 11-x)
                if y >= 15-height:
                    m[y][x] = white
                    m[15-y][x] = white
        return _cell_matrix_candidate(task, m)
    if task_id == "ml_garo-dakmanda-easy-15":
        m = [[0] * 16 for _ in range(16)]
        red, white = _palette_index(task, "#B3202C"), _palette_index(task, "#F6F2EA", 3)
        # Four nested 8x8-cell diamonds, alternating inks, plus paired edge bands.
        for x in range(16):
            m[0][x] = m[15][x] = red
        for radius, colour in ((7.0, red), (5.0, white), (3.0, red), (1.0, white)):
            _draw_diamond_cells(m, cx=7.5, cy=7.5, radius=radius, colour=colour, outline=True, width=.5)
        return _cell_matrix_candidate(task, m)
    if task_id == "od_sambalpuri-bandha-easy-18":
        m = [[0] * 16 for _ in range(16)]
        black, white = _palette_index(task, "#141414"), _palette_index(task, "#F4EFE4", 2)
        for y in range(1, 15):
            for x in range(1, 15):
                if (((x-1) // 2 + (y-1) // 2) % 2):
                    m[y][x] = black
        for i in range(16):
            m[0][i] = m[15][i] = m[i][0] = m[i][15] = white
        return _cell_matrix_candidate(task, m)
    if task_id == "pb_phulkari-and-bagh-counted-thread-embroidery-easy-19":
        m = [[0] * 16 for _ in range(16)]
        yellow, white = _palette_index(task, "#F2B705"), _palette_index(task, "#FFF8E7", 2)
        for cx, cy, radius, colour in ((3.5, 3.5, 3.5, yellow), (11.5, 3.5, 3.5, yellow),
                                       (3.5, 11.5, 3.5, yellow), (11.5, 11.5, 3.5, yellow),
                                       (7.5, 7.5, 5.5, white)):
            _draw_diamond_cells(m, cx=cx, cy=cy, radius=radius, colour=colour)
        return _cell_matrix_candidate(task, m)
    if task_id == "tg_pochampally-ikat-easy-23":
        red, cream = _palette_index(task, "#B22222"), _palette_index(task, "#F3EFE6", 2)
        m = [[red] * 16 for _ in range(16)]
        # The host fixes red as the background; cream foreground marks make
        # both the square and stepped diamond visible with a red gap between.
        for i in range(2, 14):
            m[2][i] = m[13][i] = m[i][2] = m[i][13] = cream
        _draw_diamond_cells(m, cx=7.5, cy=7.5, radius=3.5, colour=cream)
        return _cell_matrix_candidate(task, m)
    if task_id == "wb_nakshi-kantha-easy-27":
        m = [[0] * 16 for _ in range(16)]
        red = _palette_index(task, "#A8231B")
        # Central eight-point star: an axis-and-diagonal union with D4 symmetry.
        for y in range(16):
            for x in range(16):
                dx, dy = abs(x - 7.5), abs(y - 7.5)
                if (dx <= 1.5 and dy <= 5.5) or (dy <= 1.5 and dx <= 5.5) or abs(dx - dy) <= 1 and dx + dy <= 7:
                    m[y][x] = red
        for cy in (2, 13):
            for cx in (2, 13):
                for y in range(16):
                    for x in range(16):
                        dx, dy = abs(x - cx), abs(y - cy)
                        if (dx <= 1 and dy <= 1) or (dx == 0 and dy == 2) or (dy == 0 and dx == 2):
                            m[y][x] = red
        return _cell_matrix_candidate(task, m)
    if task_id == "an_nicobari-plaited-pandanus-mat-easy-28":
        m = [[0] * 16 for _ in range(16)]
        dark = _palette_index(task, "#8A6A3E")
        for y in range(1, 15):
            for x in range(1, 15):
                if (((x-1) // 2 + (y-1) // 2) % 2):
                    m[y][x] = dark
        return _cell_matrix_candidate(task, m)
    if task_id == "ch_phulkari-easy-29":
        m = [[0] * 16 for _ in range(16)]
        yellow, white = _palette_index(task, "#F2B705"), _palette_index(task, "#FFF8E7", 2)
        _draw_diamond_cells(m, cx=7.5, cy=7.5, radius=5.5, colour=yellow)
        for y in (1, 14):
            for x in (1, 14):
                m[y][x] = white
        return _cell_matrix_candidate(task, m)
    if task_id == "ld_lakshadweep-coir-and-screw-pine-mat-weave-easy-34":
        m = [[0] * 16 for _ in range(16)]
        cream = _palette_index(task, "#E9DCBE")
        for y in range(1, 15):
            for x in range(1, 15):
                if (((x-1) // 2 + (y-1) // 2) % 2):
                    m[y][x] = cream
        return _cell_matrix_candidate(task, m)
    return None


def _stripe_band_candidate(task: Task) -> dict[str, Any]:
    """Construct a narrow periodic stripe pair and mirror the pair for host symmetry."""
    easy = _easy_stripe_band_candidate(task)
    if easy is not None:
        return easy
    if task.level == "hard":
        return _hard_stripe_band_candidate(task)
    period_spec = task.constraint_profile.get("translation_period", {})
    axis = period_spec.get("axis", "x") if isinstance(period_spec, dict) else "x"
    period = period_spec.get("period", 8) if isinstance(period_spec, dict) else 8
    if axis not in {"x", "y", "xy"} or type(period) is not int or not 1 <= period <= 64:
        return _base(task, [{"id": "reference-stripes", "colour_index": min(1, len(task.palette) - 1),
                            "anchor": [0, 0], "motifs": []}])
    # A translation by period is equivalent to gcd(period, 64) on this torus.
    repeat = gcd(period, 64)
    width = 1
    residues = sorted(set(range(width)) | set(range(max(0, repeat - width), repeat)))
    vertical = axis in {"x", "xy"}
    horizontal = axis in {"y", "xy"}
    if task.symmetry in {"D4", "C4"}:
        vertical = horizontal = True
    motifs = []
    if vertical:
        motifs.extend({"kind": "rectangle", "center": [x, 32], "size": [1, 64]}
                      for x in range(64) if x % repeat in residues)
    if horizontal:
        motifs.extend({"kind": "rectangle", "center": [32, y], "size": [64, 1]}
                      for y in range(64) if y % repeat in residues)
    return _base(task, [{"id": "reference-bands", "colour_index": min(1, len(task.palette) - 1),
                         "anchor": [0, 0], "motifs": motifs}])


def _hard_stripe_band_candidate(task: Task) -> dict[str, Any]:
    """Create a D2-symmetric stripe sample with the suite's enforced x-period."""
    tid = task.id
    background = _palette_index(task, "#D4A017", 3) if tid.startswith("ka_ilkal") else 0
    m = [[background for _ in range(64)] for _ in range(64)]
    colors = [i for i in range(len(task.palette))]
    def colour(hex_value: str, fallback: int) -> int:
        return _palette_index(task, hex_value, fallback)
    def xband(x: int, sequence: list[int]) -> int:
        # A mirror-symmetric eight-pixel motif repeats across the warp axis.
        phase = x % 8
        phase = min(phase, 7 - phase)
        return sequence[phase % len(sequence)]
    def rowband(y: int, spans: list[tuple[int, int, int]], default: int = 0) -> int:
        yy = min(y, 63 - y)
        for lo, hi, ink in spans:
            if lo <= yy < hi:
                return ink
        return default

    cream = colour("#F3EFE6", min(1, len(colors) - 1))
    red = colour("#B3202C", min(2, len(colors) - 1))
    gold = colour("#C9A227", min(1, len(colors) - 1))
    green = colour("#2E7D32", min(2, len(colors) - 1))
    black = colour("#1A1A1A", 0)
    blue = colour("#1F5F8B", min(1, len(colors) - 1))
    orange = colour("#E07A1F", min(2, len(colors) - 1))

    if tid.startswith("hp_kullu"):
        bands = [(0, 3, gold), (3, 9, green), (9, 12, cream), (12, 24, red),
                 (24, 27, cream), (27, 33, green), (33, 36, gold)]
        for y in range(64):
            base = rowband(y, bands, 0)
            for x in range(64):
                m[y][x] = xband(x, [base, base, base, (red if y % 2 == 0 else base)])
    elif tid.startswith("ka_ilkal"):
        red_ka = colour("#B11226", 0)
        cream_ka = colour("#F3EFE6", 5)
        green_ka = colour("#1D7A6B", 1)
        indigo_ka = colour("#2B3A67", 2)
        sequence = [red_ka, cream_ka, indigo_ka, red_ka,
                    red_ka, indigo_ka, cream_ka, red_ka]
        for y in range(64):
            for x in range(64):
                phase = x % 8
                m[y][x] = sequence[min(phase, 7 - phase)]
    elif tid.startswith("kl_balaramapuram"):
        spans = [(0, 4, gold), (4, 6, cream), (6, 12, red), (12, 14, cream), (14, 18, gold)]
        for y in range(64):
            for x in range(64):
                m[y][x] = rowband(y, spans, 0)
                if y in {0, 63}:
                    m[y][x] = green
    elif tid.startswith("mp_maheshwari"):
        spans = [(0, 22, blue), (22, 24, gold), (24, 27, red),
                 (27, 29, gold), (29, 32, red)]
        for y in range(64):
            base = rowband(y, spans, blue)
            for x in range(64):
                ink = base
                phase = min(x % 8, 7 - (x % 8))
                if 24 <= min(y, 63-y) < 28 and phase in {0, 1, 6, 7}:
                    ink = gold if phase in {0, 1} else blue
                m[y][x] = ink
    elif tid.startswith("mz_mizo"):
        sequence = [0] * 9 + [2, 2, 0, 0, 4, 4, 5, 5, 5, 5, 5, 5, 5, 4, 2, 0, 5, 5, 5, 5, 5, 5, 5]
        for y in range(64):
            yy = min(y, 63 - y)
            base = sequence[min(yy % len(sequence), len(sequence)-1)]
            for x in range(64):
                m[y][x] = base
        # Two mirrored transverse bands follow the brief; x remains invariant,
        # satisfying the frozen eight-pixel translation profile.
        for y in range(64):
            if 20 <= min(y, 63-y) <= 23:
                for x in range(64):
                    m[y][x] = black
    elif tid.startswith("nl_chakhesang"):
        sequence = [5, 5, 2, 2, 2, 5, 5, 5]
        for y in range(64):
            for x in range(64):
                m[y][x] = xband(x, sequence)
        for y in (30, 31, 32, 33):
            for x in range(64):
                if x % 4 in {1, 2}:
                    m[y][x] = cream
        for x in range(64):
            if x % 8 in {0, 7}:
                m[0][x] = m[63][x] = cream
    elif tid.startswith("rj_leheriya"):
        # Five requested colours exceed the four-colour host ceiling; retain
        # yellow, green, magenta and blue in a mirror-closed eight-pixel repeat.
        sequence = [0, 2, 3, 4, 4, 3, 2, 0]
        for y in range(64):
            for x in range(64):
                # The requested diagonal bands reduce to repeated narrow stripes
                # under the frozen x-period and D2 constraints.
                m[y][x] = sequence[x % 8]
    elif tid.startswith("sk_lepcha"):
        sequence = [0, 0, 2, 3, 4, 1, 1, 1]
        for y in range(64):
            for x in range(64):
                m[y][x] = xband(x, sequence)
        # Short stepped chevrons sit in each repeated group.
        for y in (27, 28, 35, 36):
            for x in range(64):
                if x % 8 in {1, 2, 5, 6}:
                    m[y][x] = gold
    elif tid.startswith("tr_risa"):
        sequence = [2, 1, 2, 0, 0, 0, 0, 2]
        for y in range(64):
            for x in range(64):
                m[y][x] = xband(x, sequence)
        for y in (30, 31, 32, 33):
            for x in range(64):
                if x % 8 in {1, 2, 5, 6}:
                    m[y][x] = cream
    else:
        raise ValueError(f"No hard stripe reference defined for {tid}")
    _normalize_hard_matrix(m, background=background, periodic_x=True)
    return _cell_matrix_candidate(task, m, cell=1, background_index=background)


def _easy_stripe_band_candidate(task: Task) -> dict[str, Any] | None:
    """Brief-shaped stripe panels with explicit negative space and x-period 8."""
    tid = task.id
    layers: list[dict[str, Any]] = []
    def stripe_layer(label: str, colour: int, boxes: list[tuple[int,int,int,int]]) -> None:
        motifs = [_rect_motif(*box) for box in boxes]
        if motifs:
            layers.append({"id": label, "colour_index": colour, "anchor": [0, 0], "motifs": motifs})
    if tid == "ar_apatani-textiles-easy-1":
        white = _palette_index(task, "#F3EEE3", 3); red = _palette_index(task, "#B03A2E", 1)
        yellow = _palette_index(task, "#E3A12A", 2); indigo = _palette_index(task, "#27405E", 0)
        sequence = (white, yellow, red, indigo, indigo, red, yellow, white)
        for colour in set(sequence) - {white}:
            stripe_layer(f"apatani-{colour}", colour, [(x,16,x+1,48) for x,v in enumerate(sequence) if v == colour for x in range(x,64,8)])
        layers.insert(0, {"id":"apatani-white-ground-panel","colour_index":white,"anchor":[0,0],
                          "motifs":[_rect_motif(0,12,64,52)]})
        c = _base(task, layers); c["background_index"] = white; return c
    if tid == "mz_mizo-puanchei-easy-16":
        red = _palette_index(task, "#B3202C"); green = _palette_index(task, "#1E6B3A",1); black = _palette_index(task, "#1A1A1A",4)
        stripe_layer("mizo-black-ends", black, [(0,0,64,6),(0,58,64,64)])
        stripe_layer("mizo-green-lines", green, [(0,18,64,20),(0,44,64,46)])
        stripe_layer("mizo-red-pair", red, [(0,21,64,25),(0,39,64,43)])
        layers.insert(0,{"id":"mizo-white-ground-panel","colour_index":_palette_index(task,"#F6F2EA",5),
                         "anchor":[0,0],"motifs":[_rect_motif(0,16,64,48)]})
        c = _base(task, layers); c["background_index"] = _palette_index(task,"#F6F2EA",5); return c
    if tid == "nl_chakhesang-shawls-easy-17":
        black = _palette_index(task, "#1A1A1A"); white = _palette_index(task, "#F3EEE3",2); red = _palette_index(task,"#B3202C",1)
        stripe_layer("naga-black-bands", black, [(0,y,64,y+4) for y in (16,22,38,44)])
        stripe_layer("naga-white-check-row", white, [(x,31,x+1,33) for x in range(64) if x%8 in {1,2,5,6}])
        layers.insert(0,{"id":"naga-red-ground-panel","colour_index":red,"anchor":[0,0],
                         "motifs":[_rect_motif(0,12,64,52)]})
        c = _base(task, layers); c["background_index"] = red; return c
    if tid == "sk_lepcha-thara-weave-stripes-easy-21":
        white = _palette_index(task,"#F2EFE6",0); red = _palette_index(task,"#B81D24",2); yellow = _palette_index(task,"#E0B52A",3); green = _palette_index(task,"#2F7A3E",4)
        sequence = (white,red,yellow,green,green,yellow,red,white)
        for colour in set(sequence)-{white}:
            stripe_layer(f"lepcha-{colour}",colour,[(x,16,x+1,48) for i,v in enumerate(sequence) if v==colour for x in range(i,64,8)])
        c=_base(task,layers); c["background_index"]=white; return c
    if tid == "tr_risa-easy-24":
        black=_palette_index(task,"#1A1A1A"); red=_palette_index(task,"#B3202C",1); white=_palette_index(task,"#F6F2EA",2)
        sequence=(white,red,white,red,red,white,red,white)
        for colour in set(sequence)-{black}:
            stripe_layer(f"risa-{colour}",colour,[(x,24,x+1,40) for i,v in enumerate(sequence) if v==colour for x in range(i,64,8)])
        c=_base(task,layers); c["background_index"]=black; return c
    return None


def _periodic_raster_candidate(task: Task) -> dict[str, Any]:
    """A centered 32px square is symmetric and stays inside legacy coverage limits."""
    easy = _easy_periodic_raster_candidate(task)
    if easy is not None:
        return easy
    if task.level == "hard":
        return _hard_periodic_raster_candidate(task)
    return _base(task, [{
        "id": "reference-raster", "colour_index": min(1, len(task.palette) - 1),
        "anchor": [0, 0], "motifs": [{"kind": "rectangle", "center": [32, 32], "size": [32, 32]}],
    }])


def _hard_periodic_raster_candidate(task: Task) -> dict[str, Any]:
    """Compose a task-specific border or central field within the raster grammar."""
    tid = task.id
    if tid.startswith("br_madhubani"):
        paper = _palette_index(task, "#F6EFDD", 5)
        black = _palette_index(task, "#1A1714", 0)
        yellow = _palette_index(task, "#D9A300", 1)
        red = _palette_index(task, "#B5261E", 2)
        border = [_rect_motif(3, 3, 61, 4), _rect_motif(3, 60, 61, 61),
                  _rect_motif(3, 3, 4, 61), _rect_motif(60, 3, 61, 61),
                  _rect_motif(8, 8, 56, 9), _rect_motif(8, 55, 56, 56),
                  _rect_motif(8, 8, 9, 56), _rect_motif(55, 8, 56, 56)]
        teeth = [_polygon([(x, 4), (x + 2, 8), (x + 4, 4)]) for x in (8, 16, 24)]
        circles = [{"kind": "dot", "center": [x, y], "radius": 4}
                   for x, y in ((16, 16), (32, 16))]
        hatching = [_line([(20, 20), (32, 32)], 1), _line([(20, 32), (32, 20)], 1)]
        candidate = _base(task, [
            {"id": "rice-paper-fish-eyes", "colour_index": yellow, "anchor": [0, 0],
             "motifs": border + teeth + circles},
            {"id": "madhubani-hatch", "colour_index": black, "anchor": [0, 0], "motifs": hatching},
            {"id": "vermilion-accents", "colour_index": red, "anchor": [0, 0],
             "motifs": [{"kind": "diamond", "center": [31, 31], "radius": 5}]},
        ])
        candidate["background_index"] = paper
        return _symmetry_expand(candidate, "D2")
    if tid.startswith("cg_bastar"):
        ink = _palette_index(task, "#3B2A1E", 1)
        chevron = [_line([(4, 8), (16, 20), (28, 8)], 1),
                   _line([(8, 12), (20, 24), (32, 12)], 1),
                   _line([(12, 4), (24, 16), (36, 4)], 1)]
        sun = [{"kind": "dot", "center": [x, y], "radius": 5}
               for x, y in ((31, 31), (32, 31), (31, 32), (32, 32))]
        lozenge = [{"kind": "diamond", "center": [31, 31], "radius": 16},
                   {"kind": "diamond", "center": [32, 32], "radius": 16}]
        return _symmetry_expand(_base(task, [{"id": "bastar-woodcut", "colour_index": ink,
                                              "anchor": [0, 0], "motifs": chevron + lozenge + sun}]), "D2")
    if tid.startswith("mh_warli"):
        ochre = _palette_index(task, "#9A4A2A", 1)
        white = _palette_index(task, "#F5F1E6", 0)
        dots = [{"kind": "dot", "center": [x, y], "radius": 1}
                for x in range(9, 55, 3) for y in (8, 55)]
        dots += [{"kind": "dot", "center": [x, y], "radius": 1}
                 for y in range(9, 55, 3) for x in (8, 55)]
        zigzags = [_polygon([(x, 12), (x + 4, 16), (x + 8, 12)])
                   for x in (12, 20, 28, 36, 44)]
        candidate = _base(task, [{"id": "warli-dots", "colour_index": white,
                                  "anchor": [0, 0], "motifs": dots + zigzags}])
        candidate["background_index"] = ochre
        return _symmetry_expand(candidate, "D2")
    if tid.startswith("dd_warli"):
        brown = _palette_index(task, "#8B5A3C", 1)
        white = _palette_index(task, "#F5F1E6", 0)
        yellow = _palette_index(task, "#E3B23C", 3)
        edge = (12, 20, 28, 35, 43, 51)
        square_dots = [{"kind": "dot", "center": [x, y], "radius": 1}
                       for x in edge for y in (12, 51)]
        square_dots += [{"kind": "dot", "center": [x, y], "radius": 1}
                        for y in edge[1:-1] for x in (12, 51)]
        triangles = [_polygon([(28, 14), (34, 14), (31, 8)]),
                     _polygon([(49, 28), (55, 31), (49, 34)])]
        center_dots = [{"kind": "dot", "center": [x, y], "radius": 2}
                       for x in (31, 32) for y in (31, 32)]
        candidate = _base(task, [{"id": "warli-white", "colour_index": white,
                                  "anchor": [0, 0], "motifs": square_dots + triangles},
                                 {"id": "warli-yellow-center", "colour_index": yellow,
                                  "anchor": [0, 0], "motifs": center_dots}])
        candidate["background_index"] = brown
        return _symmetry_expand(candidate, "D2")
    raise ValueError(f"No hard periodic-raster reference defined for {tid}")


def _rect_motif(left: int, top: int, right: int, bottom: int) -> dict[str, Any]:
    """Inclusive integer raster rectangle from half-open bounds."""
    return {"kind": "rectangle", "center": [(left + right) // 2, (top + bottom) // 2],
            "size": [right - left, bottom - top]}


def _line(points: list[tuple[int, int]], stroke: int = 1) -> dict[str, Any]:
    return {"kind": "line", "points": [list(point) for point in points], "stroke": stroke}


def _polygon(points: list[tuple[int, int]]) -> dict[str, Any]:
    return {"kind": "polygon", "points": [list(point) for point in points]}


def _easy_periodic_raster_candidate(task: Task) -> dict[str, Any] | None:
    """Compact raster constructions that follow each easy prompt's named motif."""
    task_id = task.id
    if task_id == "br_madhubani-easy-3":
        yellow = _palette_index(task, "#D9A300")
        black = _palette_index(task, "#1A1714")
        border = []
        for lo, hi, width in ((6, 58, 2), (12, 52, 1)):
            border.extend([_rect_motif(lo, lo, hi, lo + width), _rect_motif(lo, hi - width, hi, hi),
                           _rect_motif(lo, lo, lo + width, hi), _rect_motif(hi - width, lo, hi, hi)])
        diamond = [{"kind": "diamond", "center": [32, 32], "radius": 16}]
        hatch = []
        for offset in range(-10, 11, 5):
            hatch.append(_line([(22 + max(0, offset), 32 - min(10, abs(offset))),
                                (42 - max(0, -offset), 32 + min(10, abs(offset)))], 1))
            hatch.append(_line([(22 + max(0, offset), 32 + min(10, abs(offset))),
                                (42 - max(0, -offset), 32 - min(10, abs(offset)))], 1))
        return _base(task, [
            {"id": "reference-yellow", "colour_index": yellow, "anchor": [0, 0], "motifs": border + diamond},
            {"id": "reference-black-hatch", "colour_index": black, "anchor": [0, 0], "motifs": hatch},
        ])
    if task_id == "cg_bastar-wooden-craft-easy-4":
        ink = _palette_index(task, "#3B2A1E")
        motifs = [{"kind": "diamond", "center": [32, 32], "radius": 13}]
        zig = [(0, 9), (8, 16), (16, 9), (24, 16), (32, 9), (40, 16), (48, 9), (56, 16), (63, 9)]
        motifs.extend([_line(zig, 1), _line([(x, 63 - y) for x, y in zig], 1)])
        side = [(9, 0), (16, 8), (9, 16), (16, 24), (9, 32), (16, 40), (9, 48), (16, 56), (9, 63)]
        motifs.extend([_line(side, 1), _line([(63 - x, y) for x, y in side], 1)])
        return _base(task, [{"id": "reference-woodcut", "colour_index": ink, "anchor": [0, 0], "motifs": motifs}])
    if task_id == "hp_kinnauri-shawl-geometric-border-easy-8":
        white = _palette_index(task, "#F2EFE6")
        yellow = _palette_index(task, "#E6B422", 2)
        hexagon = _polygon([(32, 12), (44, 19), (44, 45), (32, 52), (20, 45), (20, 19)])
        return _base(task, [
            {"id": "reference-hexagon", "colour_index": white, "anchor": [0, 0], "motifs": [hexagon]},
            {"id": "reference-core", "colour_index": yellow, "anchor": [0, 0],
             "motifs": [{"kind": "dot", "center": [32, 32], "radius": 3}]},
        ])
    if task_id == "mp_bagh-hand-block-print-easy-12":
        black, red = _palette_index(task, "#1B1B1B"), _palette_index(task, "#A3211F", 2)
        lattice = [_rect_motif(x, 0, x + 2, 64) for x in (0, 15, 16, 31, 32, 47, 48, 63)]
        lattice += [_rect_motif(0, y, 64, y + 2) for y in (0, 15, 16, 31, 32, 47, 48, 63)]
        diamonds = [{"kind": "diamond", "center": [x, y], "radius": 3}
                    for y in (8, 24, 40, 56) for x in (8, 24, 40, 56)]
        return _base(task, [
            {"id": "reference-lattice", "colour_index": black, "anchor": [0, 0], "motifs": lattice},
            {"id": "reference-butis", "colour_index": red, "anchor": [0, 0], "motifs": diamonds},
        ])
    if task_id == "mh_warli-painting-easy-13":
        ochre, white = _palette_index(task, "#6B4A2B", 2), _palette_index(task, "#F5F1E6", 0)
        dots = [{"kind": "dot", "center": [x, y], "radius": 2}
                for x, y in ((20, 20), (32, 20), (44, 20), (20, 32), (44, 32),
                             (20, 44), (32, 44), (44, 44))]
        triangles = [
            _polygon([(14, 14), (20, 17), (17, 20)]), _polygon([(50, 14), (44, 17), (47, 20)]),
            _polygon([(14, 50), (20, 47), (17, 44)]), _polygon([(50, 50), (44, 47), (47, 44)]),
        ]
        candidate = _base(task, [{"id":"warli-ochre-panel","colour_index":ochre,"anchor":[0,0],
                                  "motifs":[_rect_motif(6,6,58,58)]},
                                 {"id": "reference-warli-mark", "colour_index": white,
                                  "anchor": [0, 0], "motifs": dots + triangles}])
        return candidate
    if task_id == "rj_bagru-hand-block-print-easy-20":
        cream = _palette_index(task, "#F1E3C3", 4)
        black, red, indigo = (_palette_index(task, "#1C1C1C", 2), _palette_index(task, "#9B2226", 1), 0)
        outer = [{"kind": "diamond", "center": [32, y], "radius": 7} for y in (21, 43)]
        outer += [{"kind": "diamond", "center": [x, 32], "radius": 7} for x in (21, 43)]
        inner = [{"kind": "diamond", "center": [32, y], "radius": 4} for y in (21, 43)]
        inner += [{"kind": "diamond", "center": [x, 32], "radius": 4} for x in (21, 43)]
        corners = [{"kind": "dot", "center": [x, y], "radius": 2}
                   for x, y in ((6, 6), (57, 6), (6, 57), (57, 57))]
        candidate = _base(task, [
            {"id": "reference-cream-field", "colour_index": cream, "anchor": [0, 0],
             "motifs": [_rect_motif(6, 6, 58, 58)]},
            {"id": "reference-outlined-petals", "colour_index": black, "anchor": [0, 0], "motifs": outer},
            {"id": "reference-buti", "colour_index": red, "anchor": [0, 0], "motifs": inner},
            {"id": "reference-corner-dots", "colour_index": indigo, "anchor": [0, 0], "motifs": corners},
        ])
        candidate["background_index"] = cream
        return candidate
    if task_id == "dd_warli-painting-easy-30":
        brown, white = _palette_index(task, "#8B5A3C", 1), _palette_index(task, "#F5F1E6", 0)
        dots = [{"kind": "dot", "center": [x, y], "radius": 2}
                for x, y in ((20, 20), (32, 20), (44, 20), (20, 32), (44, 32),
                             (20, 44), (32, 44), (44, 44))]
        triangles = [
            _polygon([(32, 10), (28, 16), (36, 16)]), _polygon([(32, 54), (28, 48), (36, 48)]),
            _polygon([(10, 32), (16, 28), (16, 36)]), _polygon([(54, 32), (48, 28), (48, 36)]),
            _polygon([(16, 16), (22, 18), (18, 22)]), _polygon([(48, 16), (42, 18), (46, 22)]),
            _polygon([(16, 48), (22, 46), (18, 42)]), _polygon([(48, 48), (42, 46), (46, 42)]),
        ]
        candidate = _base(task, [{"id":"warli-brown-panel","colour_index":brown,"anchor":[0,0],
                                  "motifs":[_rect_motif(6,6,58,58)]},
                                 {"id": "reference-warli-marks", "colour_index": white,
                                  "anchor": [0, 0], "motifs": dots + triangles}])
        return candidate
    return None


def _dot_field_candidate(task: Task) -> dict[str, Any]:
    """Build a brief-specific dot witness, then let the verifier enforce its grid rules."""
    if task.level == "hard" and task.id.startswith("uk_rangwali"):
        maroon = _palette_index(task, "#7A1F2B", 2)
        lattice = [{"kind": "dot", "center": [x, y], "radius": 3}
                   for x in (6, 16, 47, 57) for y in (6, 16, 47, 57)]
        ring_centers = [(39, 19), (24, 19), (39, 44), (24, 44),
                        (44, 25), (19, 25), (44, 38), (19, 38),
                        (45, 29), (18, 29), (45, 34), (18, 34)]
        ring = [{"kind": "dot", "center": [x, y], "radius": 1} for x, y in ring_centers]
        return _base(task, [{"id": "pichhoda-lattice", "colour_index": maroon,
                             "anchor": [0, 0], "motifs": lattice + ring}])
    if task.level == "hard" and task.id.startswith("la_thigma"):
        white = _palette_index(task, "#F4EBD0", 1)
        gold = _palette_index(task, "#D9A300", 2)
        xs = (6, 16, 26, 37, 47, 57)
        dots = [{"kind": "dot", "center": [x, y], "radius": 2}
                for x in xs for y in (24, 39)]
        crosses = [{"kind": "dot", "center": [cx, cy], "radius": 1}
                   for x, y in ((16, 24), (47, 24), (16, 39), (47, 39))
                   for cx, cy in ((x - 5, y), (x + 5, y), (x, y - 5), (x, y + 5))]
        return _base(task, [{"id": "thigma-dots", "colour_index": white,
                             "anchor": [0, 0], "motifs": dots},
                            {"id": "thigma-crosses", "colour_index": gold,
                             "anchor": [0, 0], "motifs": crosses}])
    if task.id == "gj_bandhani-easy-6":
        # A 2,4,4,2 diamond gives twelve centre dots; the four inset corner
        # marks keep the motif visible after tile wrapping. All sixteen marks
        # use the brief's white ink and form a D2-symmetric set.
        centres = [
            (28, 22), (35, 22),
            (22, 28), (28, 28), (35, 28), (41, 28),
            (22, 35), (28, 35), (35, 35), (41, 35),
            (28, 41), (35, 41),
            (6, 6), (57, 6), (6, 57), (57, 57),
        ]
        white = next((i for i, colour in enumerate(task.palette) if colour.upper() == "#F6EFDD"),
                     min(1, len(task.palette) - 1))
        dots = [{"kind": "dot", "center": list(point), "radius": 2} for point in centres]
        return _base(task, [{"id": "reference-dots", "colour_index": white,
                             "anchor": [0, 0], "motifs": dots}])

    if task.id == "la_thigma-resist-dyed-circles-easy-33":
        centres = (18, 28, 35, 45)
        white = _palette_index(task, "#F4EBD0", 1)
        dots = [{"kind": "dot", "center": [x, y], "radius": 2}
                for y in centres for x in centres]
        return _base(task, [{"id": "reference-white-dot-lattice", "colour_index": white,
                             "anchor": [0, 0], "motifs": dots}])

    # A compact symmetric witness for the remaining unconstrained dot fields.
    dots = [{"kind": "dot", "center": [x, y], "radius": 3}
            for x, y in ((26, 26), (37, 26), (26, 37), (37, 37))]
    return _base(task, [{"id": "reference-dots", "colour_index": min(1, len(task.palette) - 1),
                         "anchor": [0, 0], "motifs": dots}])


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
                {"id": "curve", "colour_index": 1,
                 "anchor": [0, 0], "motifs": []},
                {"id": "dots", "colour_index": min(2, len(task.palette) - 1),
                 "anchor": [0, 0], "motifs": []},
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
    elif task.grammar == "counted_grid":
        candidates = [_counted_grid_candidate(task)]
    elif task.grammar == "stripe_band":
        candidates = [_stripe_band_candidate(task)]
    elif task.grammar == "periodic_raster":
        candidates = [_periodic_raster_candidate(task)]
    elif task.grammar == "dot_field":
        candidates = [_dot_field_candidate(task)]
    else:
        return None

    for candidate in candidates:
        if task.level == "easy" and task.grammar not in {"jaali_lattice", "kolam_loop"}:
            expanded = _symmetry_expand(json.loads(json.dumps(candidate)), task.symmetry)
            if sum(len(layer.get("motifs", [])) for layer in expanded.get("layers", [])) <= 100:
                candidate = expanded
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
