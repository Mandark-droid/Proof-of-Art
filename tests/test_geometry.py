import numpy as np
from PIL import Image, ImageDraw

from tests.helpers import pattern_obj
from pramaan.checks.core import verify_pattern
from pramaan.dsl import Motif
from pramaan.render import N, _draw_motif, render_layer, render_motif


def test_edge_crossing_shape_wraps_to_opposite_edge():
    p = pattern_obj()
    p.layers[0].motifs = [p.layers[0].motifs[0].model_copy(update={"center": (63, 32)})]
    mask = render_layer(p.layers[0])
    assert mask[32, 0]
    assert mask[32, 63]


def test_corner_crossing_shape_wraps_on_both_axes():
    p = pattern_obj()
    p.layers[0].motifs = [p.layers[0].motifs[0].model_copy(update={"center": (0, 0)})]
    mask = render_layer(p.layers[0])
    assert mask[0, 0]
    assert mask[0, 63]
    assert mask[63, 0]
    assert mask[63, 63]


def test_large_boundary_motif_wraps_copies_from_beyond_adjacent_tile():
    p = pattern_obj()
    for center_x in (127, -64):
        p.layers[0].motifs = [p.layers[0].motifs[0].model_copy(update={"center": (center_x, 32), "radius": 64})]
        mask = render_layer(p.layers[0])
        # At y=0 this diamond covers 65 source x pixels; its periodic image must
        # cover every canonical x pixel even when it crosses multiple tile copies.
        assert mask[0].all()


def test_thick_line_wraps_when_stroke_crosses_multiple_tile_copies():
    motif = Motif.model_validate({"kind": "line", "points": [(127, 0), (127, 63)], "stroke": 16})
    mask = render_motif(motif)
    assert mask[:, :8].all()
    assert mask[:, 56:].all()


def test_periodic_translation_selection_matches_broad_brute_force_for_every_primitive():
    motifs = [
        Motif.model_validate({"kind": "dot", "center": (60, 32), "radius": 64}),
        Motif.model_validate({"kind": "diamond", "center": (-64, 32), "radius": 64}),
        Motif.model_validate({"kind": "rectangle", "center": (60, 32), "size": (128, 128)}),
        Motif.model_validate({"kind": "line", "points": [(60, -64), (60, 127)], "stroke": 16}),
        Motif.model_validate({"kind": "polygon", "points": [(-64, -64), (60, -64), (60, 127), (-64, 127)]}),
    ]
    anchor = (67, 0)
    for motif in motifs:
        image = Image.new("1", (N, N), 0)
        draw = ImageDraw.Draw(image)
        # The allowed coordinate and size bounds fit inside this deliberately wider
        # translation range, independent of the optimized bounds calculation.
        for dx in range(-4 * N, 5 * N, N):
            for dy in range(-4 * N, 5 * N, N):
                _draw_motif(draw, motif, dx + anchor[0], dy + anchor[1])
        expected = np.asarray(image, dtype=bool)
        assert np.array_equal(render_motif(motif, anchor), expected), motif.kind


def test_palette_limit_fails_when_too_many_colours_used():
    p = pattern_obj()
    p.constraints.max_colours = 1
    checks, _, _ = verify_pattern(p)
    assert next(c for c in checks if c["id"] == "palette_limit")["status"] == "FAIL"


def test_palette_limit_counts_hidden_print_layers():
    p = pattern_obj()
    p.constraints.max_colours = 2
    hidden = p.layers[0].model_copy(update={"id": "hidden_red", "colour_index": 2})
    p.layers.append(hidden)
    checks, _, _ = verify_pattern(p)
    palette = next(c for c in checks if c["id"] == "palette_limit")
    assert palette["status"] == "FAIL"
    assert palette["counterexample"]["actual"] == [0, 1, 2]


def test_exclusive_layer_overlap_returns_pixel():
    p = pattern_obj()
    duplicate = p.layers[0].model_copy(update={"id": "second", "colour_index": 2})
    p.layers.append(duplicate)
    p.constraints.exclusive_layer_pairs = [("ink", "second")]
    checks, _, _ = verify_pattern(p)
    collision = next(c for c in checks if c["id"] == "exclusive_ink_collision")
    assert collision["status"] == "FAIL"
    assert collision["counterexample"]["pixel"]


def test_registration_mismatch_fails():
    p = pattern_obj()
    duplicate = p.layers[0].model_copy(update={"id": "second", "anchor": (1, 0)})
    p.layers.append(duplicate)
    checks, _, _ = verify_pattern(p)
    assert next(c for c in checks if c["id"] == "layer_registration")["status"] == "FAIL"


def test_coverage_bound_fails():
    p = pattern_obj()
    p.constraints.min_coverage = 0.9
    checks, _, _ = verify_pattern(p)
    assert next(c for c in checks if c["id"] == "coverage_bound")["status"] == "FAIL"


def test_minimum_gap_is_disabled_by_default():
    p = pattern_obj()
    p.constraints.symmetry = "None"
    p.layers[0].motifs = [p.layers[0].motifs[0].model_copy(update={"center": (20, 20)}),
                          p.layers[0].motifs[0].model_copy(update={"center": (24, 20)})]
    checks, _, _ = verify_pattern(p)
    gap = next(c for c in checks if c["id"] == "minimum_gap")
    assert p.constraints.min_gap == 0
    assert gap["status"] == "PASS"


def test_minimum_gap_fails_with_concrete_pixel_pair():
    p = pattern_obj()
    p.constraints.symmetry = "None"
    p.constraints.min_gap = 5
    p.layers[0].motifs = [p.layers[0].motifs[0].model_copy(update={"center": (20, 20)}),
                          p.layers[0].motifs[0].model_copy(update={"center": (26, 20)})]
    checks, _, _ = verify_pattern(p)
    gap = next(c for c in checks if c["id"] == "minimum_gap")
    witness = gap["counterexample"]
    assert gap["status"] == "FAIL"
    assert witness["distance_pixels"] < witness["required_minimum_pixels"]
    assert len(witness["pixel"]) == len(witness["other_pixel"]) == 2


def test_minimum_gap_checks_wrapped_tile_seam():
    p = pattern_obj()
    p.constraints.symmetry = "None"
    p.constraints.min_gap = 2
    p.layers[0].motifs = [p.layers[0].motifs[0].model_copy(update={"center": (0, 20)}),
                          p.layers[0].motifs[0].model_copy(update={"center": (63, 20)})]
    checks, _, _ = verify_pattern(p)
    gap = next(c for c in checks if c["id"] == "minimum_gap")
    assert gap["status"] == "FAIL"
    assert gap["counterexample"]["distance_pixels"] < 2


def test_minimum_gap_accepts_separated_raster_masks():
    p = pattern_obj()
    p.constraints.symmetry = "None"
    p.constraints.min_gap = 5
    p.layers[0].motifs = [p.layers[0].motifs[0].model_copy(update={"center": (10, 20)}),
                          p.layers[0].motifs[0].model_copy(update={"center": (22, 20)})]
    checks, _, _ = verify_pattern(p)
    assert next(c for c in checks if c["id"] == "minimum_gap")["status"] == "PASS"
