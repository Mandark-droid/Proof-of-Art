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
    # Four-by-four even square openings give exact D2 symmetry, 25% openness,
    # and a connected web at the declared minimum thickness.
    frame = int(task.constraint_profile.get("frame", 4))
    size = 8
    centres = (10, 24, 40, 54)
    holes = [{"kind": "rectangle", "center": [x, y], "size": [size, size]}
             for y in centres for x in centres]
    # The DSL has no star-cutout primitive; square openings preserve the
    # certified web and openness constraints. The gap is disclosed below.
    candidate = _base(task, [{"id": "jaali-solid-web", "colour_index": min(1, len(task.palette)-1),
                              "anchor": [0, 0], "motifs": []}])
    candidate["jaali"] = {"frame": frame, "open_colour_index": min(2, len(task.palette)-1),
                          "holes": holes}
    return candidate


def _counted_grid_candidate(task: Task) -> dict[str, Any]:
    """A centered cell-aligned square is a minimal witness for the cell grid."""
    easy = _easy_counted_grid_candidate(task)
    if easy is not None:
        return easy
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
        m = [[0] * 16 for _ in range(16)]
        cream = _palette_index(task, "#F3EFE6", 2)
        # Square frame and a stepped D4 diamond nested inside it.
        for i in range(3, 13):
            m[3][i] = m[12][i] = m[i][3] = m[i][12] = cream
        _draw_diamond_cells(m, cx=7.5, cy=7.5, radius=5.5, colour=cream)
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
    return _base(task, [{
        "id": "reference-raster", "colour_index": min(1, len(task.palette) - 1),
        "anchor": [0, 0], "motifs": [{"kind": "rectangle", "center": [32, 32], "size": [32, 32]}],
    }])


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
