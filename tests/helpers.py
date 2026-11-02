import json

from pramaan.dsl import Pattern


def pattern_obj(**overrides):
    data = {
        "schema_version": "0.1", "style": "ajrakh_inspired", "title": "Test",
        "tile": {"width": 64, "height": 64, "wrap_x": True, "wrap_y": True},
        "palette": ["#000000", "#FFFFFF", "#FF0000"], "background_index": 0,
        "layers": [{"id": "ink", "colour_index": 1, "anchor": [0, 0], "motifs": [
            {"kind": "diamond", "center": [16, 16], "radius": 3},
            {"kind": "diamond", "center": [47, 16], "radius": 3},
            {"kind": "diamond", "center": [16, 47], "radius": 3},
            {"kind": "diamond", "center": [47, 47], "radius": 3}
        ]}],
        "constraints": {"symmetry": "D4", "max_colours": 3, "max_primitives": 100,
                        "min_coverage": 0, "max_coverage": 1, "exclusive_layer_pairs": []}
    }
    data.update(overrides)
    return Pattern.model_validate_json(json.dumps(data))
