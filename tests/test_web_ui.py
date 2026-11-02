from __future__ import annotations

import base64
import json
from io import BytesIO

from PIL import Image

from pramaan_env import PramaanAction
from pramaan_env.server.environment import PramaanEnvironment
from pramaan_env.server.web_ui import (
    PASSING_TASK_ID,
    UNREADABLE_EXAMPLE,
    _javascript,
    _load_saved_kinnauri_examples,
    _markup,
    _render_pattern,
    _render_preview_request,
    _witness_markers,
)


def test_web_playground_replays_saved_kinnauri_failure_and_pass():
    task_id, failing, passing = _load_saved_kinnauri_examples()
    assert task_id == PASSING_TASK_ID == "hp_kinnauri-shawl-geometric-border-easy-8"

    env = PramaanEnvironment()
    env.reset(task_id=task_id, suite="eval-v2.1")
    failed = env.step(PramaanAction(pattern_json=failing))
    assert failed.reward == 0.725
    assert failed.attempt == 1 and failed.done is False
    symmetry = next(check for check in failed.checks if check["id"] == "symmetry_d2")
    assert symmetry["status"] == "FAIL"
    assert symmetry["counterexample"]["transform"] == "F_vertical"
    assert symmetry["counterexample"]["original"] == [16, 21]
    assert symmetry["counterexample"]["transformed"] == [47, 21]

    env.reset(task_id=task_id, suite="eval-v2.1")
    passed = env.step(PramaanAction(pattern_json=passing))
    assert passed.reward == 1.0
    assert passed.attempt == 1 and passed.done is True
    assert len(passed.checks) == 9


def test_witness_preview_marks_both_replayed_pixels():
    task_id, failing, _ = _load_saved_kinnauri_examples()
    env = PramaanEnvironment()
    env.reset(task_id=task_id, suite="eval-v2.1")
    result = env.step(PramaanAction(pattern_json=failing))
    markers = _witness_markers(result.checks)

    assert markers == [{"x": 16, "y": 21}, {"x": 47, "y": 21}]
    image_data = _render_pattern(failing, markers)
    assert image_data.startswith("data:image/png;base64,")
    image = Image.open(BytesIO(base64.b64decode(image_data.split(",", 1)[1])))
    for marker in markers:
        x, y = marker["x"] * 6 + 3, marker["y"] * 6 + 3
        assert image.getpixel((x + 8, y)) == (236, 72, 153)
    request_image = _render_preview_request(json.dumps({
        "pattern_json": failing,
        "witness": {"original": [16, 21], "transformed": [47, 21]},
    }))
    assert request_image.startswith("data:image/png;base64,")


def test_web_panel_discloses_reward_bands_attempt_done_and_unreadable_action():
    markup = _markup()
    script = _javascript("{}")

    assert "0.2 to under 0.8" in markup
    assert 'id="poa-reward-meter"' in markup
    assert 'id="poa-attempt"' in markup and 'id="poa-done"' in markup
    assert "Load an unreadable example" in markup
    assert "Reward 0" in script
    assert "attempt" in script and "done" in script
    assert "0 < reward < 1" not in markup + script


def test_unreadable_web_example_returns_zero_reward():
    env = PramaanEnvironment()
    env.reset(task_id=PASSING_TASK_ID, suite="eval-v2.1")
    result = env.step(PramaanAction(pattern_json=UNREADABLE_EXAMPLE))
    assert result.reward == 0.0
