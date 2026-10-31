from .dsl import ConstraintsV2


def _symmetry_guidance(symmetry: str) -> str:
    if symmetry == "D4":
        return (
            "Exact D4 is required on pixel coordinates 0..63, whose center is (31.5,31.5). "
            "Generate every off-axis dot/diamond as a complete orbit using (x,y), "
            "(63-y,x), (63-x,63-y), (y,63-x), (63-x,y), (x,63-y), (y,x), "
            "and (63-y,63-x); include each distinct result once with identical size and colour. "
            "Do not treat coordinate 32 as the tile center. Transform every vertex for lines/polygons and swap rectangle width/height on quarter turns. "
        )
    if symmetry == "D2":
        return (
            "Exact D2 is required on pixel coordinates 0..63. Generate every off-axis dot/diamond with its orbit "
            "under (x,y), (63-x,63-y), (63-x,y), and (x,63-y); include each distinct result once with identical size and colour. "
        )
    return "No symmetry is required unless another user constraint specifies it. "


def _v2_symmetry_guidance(symmetry: str) -> str:
    """Give v0.2 tasks explicit pixel-center orbits without changing frozen v0.1 prompts."""
    transforms = {
        "D2": "(x,y), (63-x,y), (x,63-y), and (63-x,63-y)",
        "D4": "(x,y), (63-y,x), (63-x,63-y), (y,63-x), (63-x,y), (x,63-y), (y,x), and (63-y,63-x)",
        "C2": "(x,y) and (63-x,63-y)",
        "C4": "(x,y), (63-y,x), (63-x,63-y), and (y,63-x)",
    }
    if symmetry not in transforms:
        return "No symmetry is required unless another host constraint specifies it. "
    return (
        f"Exact {symmetry} is required on pixel coordinates 0..63, centered at (31.5,31.5). "
        f"For every off-axis mark, include the complete orbit under {transforms[symmetry]}; "
        "include each distinct result once with identical shape, size, and colour. Transform every polygon/line vertex; "
        "swap rectangle width and height on quarter turns. Do not treat coordinate 32 as the tile center. "
    )


SYSTEM_PROMPT = """You are PRAMAAN's computational geometric ornament designer.
Generate original art inspired by the selected craft tradition, without claiming it is an authentic artisan product.
Return a single PatternDSL v0.1 JSON object only. Use only the listed primitive types, allowed coordinates and palette IDs.
Preserve all requirements from the separate fixed constraint profile. Do not return SVG, HTML, Python, markdown, URLs or explanations.
A separate deterministic validator will decide whether constraints pass.

The required object fields are schema_version, style, title, tile, palette, background_index, layers, and constraints. Add no other fields.
Use schema_version "0.1", style "ajrakh_inspired", and a wrapped 64x64 tile.
The tile object must contain exactly width: 64, height: 64, wrap_x: true, and wrap_y: true. Do not use a "wrap" shorthand or add tile fields.
For D4 on this grid, the pixel center is (31.5,31.5), not (32,32); use the full coordinate orbits specified in the task prompt.
Each layer has id, colour_index, anchor [0,0], and motifs.
Primitive forms: dot/diamond = {"kind":"dot","center":[x,y],"radius":r}; rectangle = {"kind":"rectangle","center":[x,y],"size":[w,h]};
line = {"kind":"line","points":[[x1,y1],[x2,y2]],"stroke":n}; polygon = {"kind":"polygon","points":[[x1,y1],[x2,y2],[x3,y3]]} (3 to 12 vertices).
Use integer coordinates in [-64,127], at most 4 layers and 100 total motifs. Coordinates and dimensions are integers.
The palette contains unique six-digit hex colours. Constraints are host-controlled: include only the PatternDSL constraint fields and do not copy app-only controls such as exclusive_ink_layers into the DSL. All dimensions, ids, and fields must be valid JSON."""


SYSTEM_PROMPT_V2 = """You are PRAMAAN's computational geometric ornament designer.
Generate original art informed by the selected craft tradition without claiming it is an authentic artisan product.
Return exactly one PatternDSL v0.2 JSON object. Follow the grammar and schema in the user message. Do not return SVG, HTML, Python, markdown, URLs, or explanations.
Use the exact 64x64 tile object, valid integer coordinates, 1 to 4 layers with string ids, and at most 100 total primitives. Use only the provided palette (1 to 6 colours).
The host owns the grammar, tradition, palette, background, and all constraints. Preserve them exactly as supplied; never weaken or remove a constraint.
A deterministic verifier decides whether the pattern passes. Put grammar-specific controls only inside constraints. Include every required grammar block and no block for another grammar. Do not invent top-level fields or grammar blocks."""


def design_prompt(brief: str, palette: list[str], constraints: dict) -> str:
    return (
        "Create one original Ajrakh-inspired repeating geometric tile. Make a visually balanced, detailed but uncluttered motif.\n"
        f"ART BRIEF: {brief}\nALLOWED_PALETTE: {palette}\n"
        f"IMMUTABLE_CONSTRAINTS: {constraints}\n"
        "Use a tile object with exactly {\"width\":64,\"height\":64,\"wrap_x\":true,\"wrap_y\":true}; do not add a wrap field or any other properties. "
        f"{_symmetry_guidance(constraints.get('symmetry', 'D4'))}"
        "Use style ajrakh_inspired, at most 4 layers and 100 total primitives. "
        "Keep distinct motif masks at least IMMUTABLE_CONSTRAINTS.min_gap pixels apart when that value is greater than zero. "
        "Do not copy app-only controls such as exclusive_ink_layers into the PatternDSL constraints object. "
        "Coordinates are integers. Return the complete PatternDSL v0.1 object as JSON with no extra fields."
    )


def repair_prompt(brief: str, profile: dict, previous_json: str, failures: list[dict]) -> str:
    import json

    return (
        "Revise the previous PatternDSL candidate while preserving its visual intent. These checks came from a deterministic validator. "
        "Fix the concrete counterexamples. Return a complete schema-valid PatternDSL JSON object only, with no extra fields. "
        "The tile object must contain exactly width, height, wrap_x, and wrap_y; omit app-only controls such as exclusive_ink_layers from the DSL. "
        f"{_symmetry_guidance(profile.get('symmetry', 'D4'))}\n"
        f"ORIGINAL_BRIEF: {brief}\nIMMUTABLE_CONSTRAINTS: {json.dumps(profile, sort_keys=True)}\n"
        f"PREVIOUS_DESIGN: {previous_json}\nVERIFIER_FAILURES: {json.dumps(failures, sort_keys=True)}"
    )


_V1_PROFILE_DEFAULTS = {
    "symmetry": "D4",
    "max_colours": 4,
    "max_primitives": 100,
    "min_gap": 0,
    "min_coverage": 0.02,
    "max_coverage": 0.70,
    "exclusive_layer_pairs": [],
}


def _task_value(task, name, default=None):
    if isinstance(task, dict):
        return task.get(name, default)
    return getattr(task, name, default)


def _legacy_task_profile(task) -> dict:
    source = _task_value(task, "constraint_profile", {}) or {}
    profile = {**_V1_PROFILE_DEFAULTS, **{key: source[key] for key in _V1_PROFILE_DEFAULTS if key in source}}
    symmetry = _task_value(task, "symmetry", profile["symmetry"])
    profile["symmetry"] = symmetry if symmetry in {"None", "D2", "D4"} else "D2"
    profile["exclusive_ink_layers"] = bool(source.get("exclusive_ink_layers", False))
    return profile


def _v2_task_profile(task) -> dict:
    source = _task_value(task, "constraint_profile", {}) or {}
    values = {key: value for key, value in source.items() if key in ConstraintsV2.model_fields}
    symmetry = _task_value(task, "symmetry", values.get("symmetry", "D4"))
    values["symmetry"] = symmetry if symmetry in {"None", "D2", "D4", "C2", "C4"} else "D2"
    profile = {
        **ConstraintsV2.model_validate(values).model_dump(mode="json"),
        "exclusive_ink_layers": bool(source.get("exclusive_ink_layers", False)),
    }
    if _task_value(task, "grammar") == "jaali_lattice" and "frame" in source:
        profile["frame"] = source["frame"]
    return profile


def _task_tile_profile(task) -> dict:
    """Return the host-owned 64x64 wrap contract for a grammar task."""
    grammar = _task_value(task, "grammar")
    source = _task_value(task, "constraint_profile", {}) or {}
    bounded = grammar in {"rotational_field", "kolam_loop"} or (
        grammar == "jaali_lattice" and int(source.get("frame", 0)) > 0
    )
    return {"width": 64, "height": 64, "wrap_x": not bounded, "wrap_y": not bounded}


_GRAMMAR_GUIDANCE = {
    "dot_field": (
        'Use only dot motifs. Put the host minimum clearance in constraints.min_clearance; '
        "do not use diamonds, rectangles, lines, or polygons. Do not add a top-level spacing field. "
        "Expand every motif into literal JSON arrays: no comprehensions, variables, ranges, or comments. "
        "Do not include a top-level dot_field key; this grammar has no dot_field block. "
        "Set kolam and jaali to null. For D2 or D4, use complete reflected symmetry orbits "
        "with the same radius and colour for every dot."
    ),
    "stripe_band": (
        'Use repeated horizontal or vertical bands and set constraints.translation_period to the exact '
        '{"axis": "x" or "y" or "xy", "period": integer} supplied by the host. '
        "Do not add a top-level stripe_band or bands field; there is no such grammar block. "
        "Keep the palette to the exact host list, at most 6 colours, and use no more than 4 layers."
    ),
    "counted_grid": (
        "Quantize every mark to the host cell grid. Set constraints.cell_size to the host value. Do not add "
        "a top-level grid or counted_grid field; cell_size belongs only in constraints. Use full "
        "host-sized cells only, with each filled cell uniform within every layer. Do not use dots, diamonds, "
        "or diagonal motifs; use axis-aligned rectangles whose edges align exactly to cell boundaries. "
        "For each rectangle, both center minus half its size and center plus half its size must be multiples "
        "of cell_size on both axes; its width and height must be whole cell multiples."
    ),
    "rotational_field": (
        "Use a bounded 64x64 field (wrap_x and wrap_y are false) and satisfy the exact host symmetry "
        "(C2 or C4 when specified). Do not add a top-level rotational_field field; rotation is declared only "
        "through constraints.symmetry and the host TILE_SCHEMA."
    ),
    "kolam_loop": (
        'Provide a "kolam" block with rows, cols, spacing, mirrors, stroke, and dot_radius. rows and cols must be 1..9; '
        "spacing must be exactly 6, 8, 10, or 12; stroke must be 1, 2, or 3; dot_radius must be 0, 1, or 2. "
        "Mirrors are placed "
        "at doubled-coordinate edge midpoints between neighboring dots; coordinates must be interior, unique, "
        "and for each [x,y] exactly one coordinate is odd and the other is even; never choose both odd "
        "or both even. Coordinates must be within 0..2*cols-2 for x and 0..2*rows-2 for y. "
        'and the reflected curves must close into the host required_loops count. Set tile to exactly '
        '{"width":64,"height":64,"wrap_x":false,"wrap_y":false}; use a bounded canvas.'
    ),
    "jaali_lattice": (
        "Provide a \"jaali\" block with frame, open_colour_index, and holes. Every hole motif must include its required kind field named \"kind\", "
        "for example {\"kind\":\"rectangle\",\"center\":[x,y],\"size\":[w,h]} or "
        "{\"kind\":\"polygon\",\"points\":[[x1,y1],[x2,y2],[x3,y3]]}; holes are Motif objects, not bare "
        "coordinate/size objects. Include at least one layer object; layers cannot be []. For a solid panel "
        "carved only by holes, use a layer such as {\"id\":\"solid\",\"colour_index\":1,\"anchor\":[0,0],\"motifs\":[]} "
        "(an empty motifs array fills the panel before the declared holes are cut). Holes must remain inside the frame, "
        "form complete orbits under the host symmetry, leave one connected solid web, and satisfy the host "
        "min_web/open-fraction constraints."
    ),
}

_V2_SHAPE_GUIDANCE = (
    'Layers use the exact keys "id", "colour_index", "anchor", "motifs". '
    'A dot motif is {"kind": "dot", "center": [x, y], "radius": r}; '
    'diamond uses the same fields, rectangle uses {"kind": "rectangle", "center": [x, y], "size": [w, h]}, '
    'line uses {"kind": "line", "points": [[x1, y1], [x2, y2]], "stroke": n}, '
    'and polygon uses {"kind": "polygon", "points": [[x1, y1], [x2, y2], [x3, y3]]}. '
    "Never use name, dots, type, cx, or centre as field names. The root object has only schema_version, grammar, "
    "tradition_id, title, tile, palette, background_index, layers, constraints, kolam, and jaali as applicable. "
    "tile must be an object with width, height, wrap_x, wrap_y; palette has 1..6 colours; layers has 1..4 objects; "
    "each layer id is a string and motifs is an array. Never use integer layer ids, a tile array, or extra fields."
)


def system_prompt_for_task(task) -> str:
    """Return the frozen v0.1 system prompt for G1, otherwise the v0.2 prompt."""
    return SYSTEM_PROMPT if _task_value(task, "grammar") == "periodic_raster" else SYSTEM_PROMPT_V2


def design_prompt_for_task(task) -> str:
    """Build a task-bound prompt; preserve the original G1 prompt byte for byte."""
    grammar = _task_value(task, "grammar")
    brief = _task_value(task, "brief", "")
    palette = _task_value(task, "palette", [])
    if grammar == "periodic_raster":
        return design_prompt(brief, palette, _legacy_task_profile(task))
    if grammar not in _GRAMMAR_GUIDANCE:
        raise ValueError(f"unsupported task grammar: {grammar!r}")
    import json

    profile = _v2_task_profile(task)
    tile = _task_tile_profile(task)
    return (
        f"Create one original pattern for tradition { _task_value(task, 'tradition_id', '') }.\n"
        f"GRAMMAR: {grammar}\nART_BRIEF: {brief}\nALLOWED_PALETTE: {json.dumps(palette)}\n"
        f"HOST_CONSTRAINTS: {json.dumps(profile, sort_keys=True)}\n"
        f"TILE_SCHEMA: {json.dumps(tile)}; tile must be an object, never an array.\n"
        f"GRAMMAR_RULES: {_GRAMMAR_GUIDANCE[grammar]}\n"
        f"SYMMETRY_RULES: {_v2_symmetry_guidance(profile['symmetry'])}\n"
        f"LAYER_AND_MOTIF_SCHEMA: {_V2_SHAPE_GUIDANCE}\n"
        "Return a complete PatternDSL v0.2 JSON object with schema_version, grammar, tradition_id, title, tile, "
        "palette, background_index, layers, and constraints. Use the grammar block required by the grammar; "
        "do not include a block for another grammar. The host overwrites constraints, palette, background_index, "
        "and the Jaali frame when specified. "
        "No additional keys or prose."
    )


def repair_prompt_for_task(task, previous_json: str, failures: list[dict]) -> str:
    """Ask for a full repaired candidate while retaining task-specific geometry rules."""
    if _task_value(task, "grammar") == "periodic_raster":
        return repair_prompt(
            _task_value(task, "brief", ""), _legacy_task_profile(task), previous_json, failures,
        )
    import json

    kolam_repair_guidance = (
        " KOLAM_REPAIR: When verifier feedback includes valid_midpoints, replace the invalid mirror list "
        "using only those listed coordinates; do not copy an invalid mirror. Recompute the mirror set so the "
        "traced loop count equals required_loops and preserve the requested symmetry."
        if _task_value(task, "grammar") == "kolam_loop" else ""
    )
    tile = _task_tile_profile(task)

    return (
        f"Repair the PatternDSL v0.2 candidate for grammar {_task_value(task, 'grammar')}. "
        f"Preserve this task brief: {_task_value(task, 'brief', '')}.\n"
        f"GRAMMAR_RULES: {_GRAMMAR_GUIDANCE.get(_task_value(task, 'grammar'), 'Follow the required grammar block exactly.')}\n"
        f"{kolam_repair_guidance}\n"
        f"LAYER_AND_MOTIF_SCHEMA: {_V2_SHAPE_GUIDANCE}\n"
        f"HOST_CONSTRAINTS: {json.dumps(_v2_task_profile(task), sort_keys=True)}\n"
        f"TILE_SCHEMA: {json.dumps(tile)}; tile must be an object, never an array.\n"
        f"SYMMETRY_RULES: {_v2_symmetry_guidance(_v2_task_profile(task)['symmetry'])}\n"
        f"PREVIOUS_DESIGN: {previous_json}\nVERIFIER_FAILURES: {json.dumps(failures, sort_keys=True)}\n"
        "Return one complete schema-valid PatternDSL v0.2 JSON object only. The host owns grammar, tradition, "
        "palette, background_index, constraints, and the Jaali frame when specified; keep them as given. Do not add prose or fields."
    )
