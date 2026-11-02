import json
from pathlib import Path

import pytest

from pramaan.normalize import normalize_model_text
from pramaan.orchestrate import _parse_candidate, generate_and_verify
from tests.helpers import pattern_obj


def test_fenced_json_parses():
    raw = '```json\n{"ok": true}\n```'
    normalized, actions = normalize_model_text(raw)
    assert json.loads(normalized) == {"ok": True}
    assert actions == ["strip_code_fence"]


def test_candidate_parser_applies_the_shared_normalizer():
    pattern = pattern_obj()
    fenced = f"```json\n{pattern.model_dump_json()}\n```"
    parsed = _parse_candidate(fenced, pattern.constraints, pattern.palette, False)
    assert parsed.title == pattern.title


def test_unlabelled_fence_parses():
    normalized, actions = normalize_model_text('```\n{"ok": true}\n```')
    assert json.loads(normalized) == {"ok": True}
    assert actions == ["strip_code_fence"]


def test_think_block_then_json_parses():
    normalized, actions = normalize_model_text('<think>reasoning</think>\n{"ok": true}')
    assert json.loads(normalized) == {"ok": True}
    assert actions == ["strip_think_block"]


def test_prose_around_json_still_fails():
    normalized, actions = normalize_model_text('Here is the result: {"ok": true}')
    assert actions == []
    with pytest.raises(json.JSONDecodeError):
        json.loads(normalized)


def test_two_json_objects_fail():
    normalized, actions = normalize_model_text('{"a": 1}\n{"b": 2}')
    assert actions == []
    with pytest.raises(json.JSONDecodeError):
        json.loads(normalized)


def test_unfenced_json_unchanged_and_no_normalizations_recorded():
    raw = '{"ok": true}'
    assert normalize_model_text(raw) == (raw, [])


def test_v1_beam_rows_unchanged():
    path = Path(__file__).parents[1] / "benchmarks" / "data" / "beam-hf-trajectories-v1.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 24
    for row in rows:
        for call in row["calls"]:
            raw = call["raw_response"]
            normalized, actions = normalize_model_text(raw)
            assert actions == []
            assert json.loads(normalized) == json.loads(raw)


def test_attempt_and_call_record_normalization_actions():
    class FencedClient:
        def complete(self, system, prompt):
            return f"```json\n{pattern_obj().model_dump_json()}\n```", {"model_id": "fixture", "provider": "test"}

    result = generate_and_verify("balanced geometry", ["#000000", "#FFFFFF"],
                                 client=FencedClient(), max_attempts=1)
    assert result["accepted"] is True
    assert result["attempts"][0]["normalizations"] == ["strip_code_fence"]
    assert result["model_calls"][0]["normalizations"] == ["strip_code_fence"]
