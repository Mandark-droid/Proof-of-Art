import json
import unittest
from pydantic import ValidationError

from pramaan.dsl import Pattern
from tests.helpers import pattern_obj


def test_valid_pattern_parses():
    assert pattern_obj().tile.width == 64


def test_extra_executable_field_is_rejected():
    data = json.loads(pattern_obj().model_dump_json())
    data["script"] = "__import__('os').system('bad')"
    with unittest.TestCase().assertRaises(ValidationError):
        Pattern.model_validate_json(json.dumps(data))


def test_unsupported_tile_size_is_rejected():
    data = json.loads(pattern_obj().model_dump_json())
    data["tile"]["width"] = 128
    with unittest.TestCase().assertRaises(ValidationError):
        Pattern.model_validate_json(json.dumps(data))


def test_out_of_palette_layer_is_rejected():
    data = json.loads(pattern_obj().model_dump_json())
    data["layers"][0]["colour_index"] = 5
    with unittest.TestCase().assertRaises(ValidationError):
        Pattern.model_validate_json(json.dumps(data))


def test_shape_rejects_irrelevant_parameters():
    data = json.loads(pattern_obj().model_dump_json())
    data["layers"][0]["motifs"][0]["stroke"] = 2
    with unittest.TestCase().assertRaises(ValidationError):
        Pattern.model_validate_json(json.dumps(data))


def test_duplicate_palette_colours_are_rejected_case_insensitively():
    data = json.loads(pattern_obj().model_dump_json())
    data["palette"] = ["#000000", "#ffffff", "#FFFFFF"]
    with unittest.TestCase().assertRaises(ValidationError):
        Pattern.model_validate_json(json.dumps(data))


def test_minimum_gap_accepts_only_bounded_integer_pixels():
    data = json.loads(pattern_obj().model_dump_json())
    data["constraints"]["min_gap"] = 8
    assert Pattern.model_validate_json(json.dumps(data)).constraints.min_gap == 8
    for invalid in (-1, 9, 2.5):
        bad = json.loads(json.dumps(data))
        bad["constraints"]["min_gap"] = invalid
        with unittest.TestCase().assertRaises(ValidationError):
            Pattern.model_validate_json(json.dumps(bad))
