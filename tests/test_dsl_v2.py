from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pramaan.dsl import Pattern, PatternV2, parse_pattern
from tests.helpers import pattern_obj


def _v2(**overrides) -> dict:
    data = {
        "schema_version": "0.2",
        "grammar": "periodic_raster",
        "tradition_id": "gj_ajrakh",
        "title": "Test repeat",
        "tile": {"width": 64, "height": 64, "wrap_x": True, "wrap_y": True},
        "palette": ["#000000", "#FFFFFF"],
        "background_index": 0,
        "layers": [{
            "id": "ink", "colour_index": 1, "anchor": [0, 0],
            "motifs": [{"kind": "dot", "center": [16, 16], "radius": 2}],
        }],
        "constraints": {"symmetry": "D4"},
    }
    data.update(overrides)
    return data


def _json(data: dict) -> str:
    return json.dumps(data)


def test_v02_requires_grammar():
    data = _v2()
    del data["grammar"]
    with pytest.raises(ValidationError):
        PatternV2.model_validate_json(_json(data))


def test_kolam_block_only_for_kolam_grammar():
    kolam = {
        "rows": 3, "cols": 3, "spacing": 8, "mirrors": [],
        "stroke": 1, "dot_radius": 1,
    }
    with pytest.raises(ValidationError):
        PatternV2.model_validate_json(_json(_v2(kolam=kolam)))
    with pytest.raises(ValidationError):
        PatternV2.model_validate_json(_json(_v2(grammar="kolam_loop")))

    valid = _v2(
        grammar="kolam_loop", tile={"width": 64, "height": 64, "wrap_x": False, "wrap_y": False},
        kolam=kolam,
    )
    assert PatternV2.model_validate_json(_json(valid)).kolam.rows == 3


def test_wrap_false_only_for_bounded_grammars():
    bounded_tile = {"width": 64, "height": 64, "wrap_x": False, "wrap_y": False}
    with pytest.raises(ValidationError):
        PatternV2.model_validate_json(_json(_v2(tile=bounded_tile)))
    assert PatternV2.model_validate_json(_json(_v2(grammar="rotational_field", tile=bounded_tile)))
    jaali = {"frame": 4, "open_colour_index": 1, "holes": []}
    assert PatternV2.model_validate_json(
        _json(_v2(grammar="jaali_lattice", tile=bounded_tile, jaali=jaali))
    )
    with pytest.raises(ValidationError, match="both axes"):
        PatternV2.model_validate_json(_json(_v2(
            grammar="jaali_lattice",
            tile={"width": 64, "height": 64, "wrap_x": True, "wrap_y": False},
            jaali={"frame": 0, "open_colour_index": 1, "holes": []},
        )))


def test_dot_field_rejects_non_dot_motifs():
    data = _v2(grammar="dot_field", constraints={"min_clearance": 2})
    data["layers"][0]["motifs"][0] = {
        "kind": "rectangle", "center": [16, 16], "size": [4, 4]
    }
    with pytest.raises(ValidationError):
        PatternV2.model_validate_json(_json(data))


def test_parse_pattern_dispatches_on_schema_version():
    assert isinstance(parse_pattern(pattern_obj().model_dump_json()), Pattern)
    assert isinstance(parse_pattern(_json(_v2())), PatternV2)
    with pytest.raises(ValueError, match="unsupported schema_version"):
        parse_pattern('{"schema_version":"0.3"}')


def test_tradition_id_accepts_sourced_registry_ids_up_to_80_characters():
    accepted = _v2(tradition_id="x" * 80)
    assert PatternV2.model_validate_json(_json(accepted)).tradition_id == "x" * 80

    too_long = _v2(tradition_id="x" * 81)
    with pytest.raises(ValidationError):
        PatternV2.model_validate_json(_json(too_long))


def test_kolam_bounds():
    tile = {"width": 64, "height": 64, "wrap_x": False, "wrap_y": False}
    base = _v2(
        grammar="kolam_loop", tile=tile,
        kolam={"rows": 3, "cols": 3, "spacing": 8, "mirrors": [], "stroke": 1, "dot_radius": 1},
    )
    too_many_rows = json.loads(_json(base))
    too_many_rows["kolam"]["rows"] = 10
    with pytest.raises(ValidationError):
        PatternV2.model_validate_json(_json(too_many_rows))

    too_wide = json.loads(_json(base))
    too_wide["kolam"]["cols"] = 9
    too_wide["kolam"]["spacing"] = 8
    with pytest.raises(ValidationError):
        PatternV2.model_validate_json(_json(too_wide))
