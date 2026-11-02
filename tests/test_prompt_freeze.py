import hashlib
from types import SimpleNamespace

from pramaan.prompts import design_prompt, design_prompt_for_task, repair_prompt_for_task


def test_g1_prompts_unchanged():
    profile = {
        "symmetry": "D4",
        "max_colours": 4,
        "max_primitives": 100,
        "min_gap": 0,
        "min_coverage": 0.02,
        "max_coverage": 0.7,
        "exclusive_layer_pairs": [],
        "exclusive_ink_layers": False,
    }
    expected = "d8fbc469589270694c4d6e94df74932a9e2a81d9fd0e874188ff69c363e78c41"
    original = design_prompt("balanced geometry", ["#000000", "#FFFFFF"], profile)
    task = SimpleNamespace(
        grammar="periodic_raster", brief="balanced geometry", palette=["#000000", "#FFFFFF"],
        symmetry="D4", constraint_profile=profile,
    )
    assert hashlib.sha256(original.encode()).hexdigest() == expected
    assert design_prompt_for_task(task) == original


def test_kolam_prompt_explains_doubled_coordinates_and_mirrors():
    task = SimpleNamespace(
        grammar="kolam_loop", brief="a balanced loop", palette=["#000000", "#FFFFFF"],
        symmetry="D4", tradition_id="tn_sikku_kolam", constraint_profile={"required_loops": 1},
    )
    prompt = design_prompt_for_task(task)
    assert "doubled-coordinate" in prompt.lower()
    assert "mirror" in prompt.lower()
    assert "required_loops" in prompt
    assert "exactly one coordinate is odd" in prompt
    assert "never choose both odd or both even" in prompt


def test_kolam_repair_prompt_replaces_invalid_mirrors_from_witness():
    task = SimpleNamespace(
        grammar="kolam_loop", brief="a balanced loop", palette=["#000000", "#FFFFFF"],
        symmetry="D4", tradition_id="tn_sikku_kolam", constraint_profile={"required_loops": 1},
    )
    prompt = repair_prompt_for_task(
        task, '{"kolam":{"mirrors":[[1,3]]}}',
        [{"id": "kolam_mirrors_valid", "status": "FAIL",
          "counterexample": {"mirror": [1, 3], "valid_midpoints": [[1, 0], [0, 1]]}}],
    )
    assert "replace the invalid mirror list" in prompt
    assert "valid_midpoints" in prompt


def test_jaali_prompt_explains_holes_and_frame():
    task = SimpleNamespace(
        grammar="jaali_lattice", brief="a framed lattice", palette=["#000000", "#FFFFFF"],
        symmetry="D2", tradition_id="gj_jaali", constraint_profile={"frame": 2, "min_web": 1},
    )
    prompt = design_prompt_for_task(task)
    assert "holes" in prompt.lower()
    assert "frame" in prompt.lower()
    assert '"jaali"' in prompt
    assert '"kind": "rectangle", "center": [x, y], "size": [w, h]' in prompt
    assert "every hole motif must include its required kind field" in prompt.lower()
    assert "include at least one layer object" in prompt.lower()


def test_jaali_repair_prompt_repeats_required_hole_and_layer_shapes():
    task = SimpleNamespace(
        grammar="jaali_lattice", brief="a framed lattice", palette=["#000000", "#FFFFFF"],
        symmetry="D2", tradition_id="gj_jaali", constraint_profile={"frame": 2, "min_web": 1},
    )
    prompt = repair_prompt_for_task(
        task, '{"layers":[],"jaali":{"holes":[{"center":[12,12],"size":[8,8]}]}}',
        [{"id": "candidate_parse", "status": "FAIL", "message": "jaali.holes.0.kind Field required"}],
    )
    assert '"kind": "rectangle", "center": [x, y], "size": [w, h]' in prompt
    assert "every hole motif must include its required kind field" in prompt.lower()
    assert "include at least one layer object" in prompt.lower()


def test_v2_design_and_repair_prompts_name_exact_layer_and_motif_fields():
    task = SimpleNamespace(
        grammar="dot_field", brief="separated dots", palette=["#000000", "#FFFFFF"],
        symmetry="D2", tradition_id="gj_bandhani", constraint_profile={"min_clearance": 3},
    )
    prompts = [
        design_prompt_for_task(task),
        repair_prompt_for_task(task, '{"layers": []}', []),
    ]

    for prompt in prompts:
        assert '"id", "colour_index", "anchor", "motifs"' in prompt
        assert '"kind": "dot", "center": [x, y], "radius": r' in prompt
        assert "Never use name, dots, type, cx, or centre" in prompt


def test_dot_field_design_and_repair_forbid_unsupported_top_level_spacing():
    task = SimpleNamespace(
        grammar="dot_field", brief="a symmetric dot pattern", palette=["#000000", "#FFFFFF"],
        symmetry="D2", tradition_id="gj_bandhani", constraint_profile={"min_clearance": 3},
    )
    prompts = [
        design_prompt_for_task(task),
        repair_prompt_for_task(task, '{"spacing": 6}', []),
    ]
    for prompt in prompts:
        assert "Do not add a top-level spacing field" in prompt


def test_dot_field_prompts_require_expanded_json_arrays_not_code():
    task = SimpleNamespace(
        grammar="dot_field", brief="a symmetric dot pattern", palette=["#000000", "#FFFFFF"],
        symmetry="D2", tradition_id="gj_bandhani", constraint_profile={"min_clearance": 3},
    )
    prompts = [
        design_prompt_for_task(task),
        repair_prompt_for_task(task, '{"layers": []}', []),
    ]
    for prompt in prompts:
        assert "literal JSON arrays" in prompt
        assert "no comprehensions, variables, ranges, or comments" in prompt


def test_dot_field_prompts_forbid_unrecognized_grammar_block():
    task = SimpleNamespace(
        grammar="dot_field", brief="a symmetric dot pattern", palette=["#000000", "#FFFFFF"],
        symmetry="D2", tradition_id="gj_bandhani", constraint_profile={"min_clearance": 3},
    )
    prompts = [
        design_prompt_for_task(task),
        repair_prompt_for_task(task, '{"dot_field": {}}', []),
    ]
    for prompt in prompts:
        assert "Do not include a top-level dot_field key" in prompt
        assert "set kolam and jaali to null" in prompt.lower()


def test_dot_field_prompts_build_complete_symmetry_orbits():
    task = SimpleNamespace(
        grammar="dot_field", brief="a symmetric dot pattern", palette=["#000000", "#FFFFFF"],
        symmetry="D2", tradition_id="gj_bandhani", constraint_profile={"min_clearance": 3},
    )
    prompts = [
        design_prompt_for_task(task),
        repair_prompt_for_task(task, '{"layers": []}', []),
    ]
    for prompt in prompts:
        assert "complete reflected symmetry orbits" in prompt
        assert "same radius and colour" in prompt


def test_counted_grid_prompts_define_full_cells_and_object_tile_schema():
    task = SimpleNamespace(
        grammar="counted_grid", brief="red and white counted motifs", palette=["#FFFFFF", "#AA0000"],
        symmetry="D2", tradition_id="as_gamosa", constraint_profile={"cell_size": 4},
    )
    prompts = [
        design_prompt_for_task(task),
        repair_prompt_for_task(task, '{"tile": [64, 64]}', []),
    ]
    for prompt in prompts:
        assert '"wrap_x": true, "wrap_y": true' in prompt
        assert "tile must be an object, never an array" in prompt
        assert "full host-sized cells only" in prompt
        assert "Do not use dots, diamonds, or diagonal motifs" in prompt
        assert "for each rectangle" in prompt.lower()
        assert "center plus half its size must be multiples of cell_size" in prompt
        assert "Do not add a top-level grid or counted_grid field" in prompt


def test_v2_grammar_prompts_forbid_model_invented_top_level_grammar_blocks():
    cases = [
        ("stripe_band", {}, "stripe_band or bands"),
        ("rotational_field", {}, "rotational_field"),
    ]
    for grammar, profile, forbidden in cases:
        task = SimpleNamespace(
            grammar=grammar, brief="symmetric geometric pattern", palette=["#000000", "#FFFFFF"],
            symmetry="D2", tradition_id="test_pattern", constraint_profile=profile,
        )
        prompts = [
            design_prompt_for_task(task),
            repair_prompt_for_task(task, '{"rotational_field":{}}', []),
        ]
        for prompt in prompts:
            assert f"Do not add a top-level {forbidden} field" in prompt


def test_v2_prompts_include_complete_host_symmetry_orbits():
    cases = [
        ("dot_field", "D2", "(63-x,y)"),
        ("stripe_band", "D4", "(63-y,x)"),
        ("rotational_field", "C4", "(63-y,x)"),
    ]
    for grammar, symmetry, transform in cases:
        task = SimpleNamespace(
            grammar=grammar, brief="balanced pattern", palette=["#000000", "#FFFFFF"],
            symmetry=symmetry, tradition_id="test_pattern", constraint_profile={},
        )
        prompt = design_prompt_for_task(task)
        assert transform in prompt
        assert "include each distinct result once" in prompt


def test_kolam_prompt_lists_exact_schema_bounds_from_dsl():
    task = SimpleNamespace(
        grammar="kolam_loop", brief="a balanced loop", palette=["#000000", "#FFFFFF"],
        symmetry="D4", tradition_id="tn_sikku_kolam", constraint_profile={"required_loops": 1},
    )
    prompt = design_prompt_for_task(task)
    assert "spacing must be exactly 6, 8, 10, or 12" in prompt
    assert "dot_radius must be 0, 1, or 2" in prompt
    assert "stroke must be 1, 2, or 3" in prompt
