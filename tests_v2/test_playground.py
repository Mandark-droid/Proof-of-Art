"""The playground at /web/ (spec 8.9 and 10, ``Playground.dc.html``)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
KINNAURI = "hp_kinnauri-shawl-geometric-border-easy-8"
NO_PASS = "dl_mughal-sandstone-and-marble-jaali-screens-of-the-humayun-s-tomb-complex-hard-31"
UNREADABLE = "{ this is intentionally not valid PatternDSL JSON"


def _replay(task_id: str) -> dict:
    return json.loads((ROOT / "pramaan_v2" / "site_data" / "replays" / f"{task_id}.json").read_text(encoding="utf-8"))


def _beam_try(n: int) -> dict:
    beam = next(m for m in _replay(KINNAURI)["models"] if m["key"] == "beam-hf")
    return next(t for t in beam["tries"] if t["n"] == n)


def _without_nulls(value):
    if isinstance(value, dict):
        return {k: _without_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_nulls(v) for v in value]
    return value


@pytest.fixture(scope="module")
def client():
    from pramaan_v2.server import app

    return TestClient(app)


def _episode(client, task_id: str, *patterns: str) -> tuple[dict, list[dict]]:
    reset = client.post("/web/reset", json={"task_id": task_id})
    assert reset.status_code == 200
    steps = []
    for pattern in patterns:
        response = client.post("/web/step", json={"action": {"pattern_json": pattern}})
        assert response.status_code == 200
        steps.append({"pattern_json": pattern, "result": response.json()})
    return reset.json(), steps


# ---------------------------------------------------------------- builder and mount

def test_builder_returns_blocks():
    import gradio as gr

    from pramaan_v2.apps.playground import PLAYGROUND_TITLE, build_playground

    blocks = build_playground(None, [], None, False, PLAYGROUND_TITLE, "")
    assert isinstance(blocks, gr.Blocks)


def test_builder_is_v2_not_v1():
    from pramaan_v2.apps import playground

    assert playground.build_playground.__module__ == "pramaan_v2.apps.playground"


def test_web_page_serves(client):
    response = client.get("/web/")
    assert response.status_code == 200


def test_web_config_carries_the_v2_page(client):
    config = client.get("/web/config").json()
    text = json.dumps(config)
    assert "Play one episode yourself." in text
    assert "/static/v2/v2.css" in text


# ---------------------------------------------------------------- contract, unchanged

def test_reset_takes_a_non_default_task_id(client):
    reset, _ = _episode(client, "tn_kolam-easy-22")
    assert reset["observation"]["task_id"] == "tn_kolam-easy-22"
    assert reset["done"] is False


def test_reset_takes_suite_and_seed(client):
    response = client.post("/web/reset", json={"suite": "train-v1", "seed": 5})
    assert response.status_code == 200
    assert response.json()["observation"]["metadata"]["suite_id"] == "train-v1"


def test_saved_beam_failure_scores_0725_with_failing_symmetry(client):
    from pramaan_v2.apps.playground_render import examples

    failure = examples(KINNAURI, "eval-v2.1")[0]
    _, steps = _episode(client, KINNAURI, failure["pattern"])
    result = steps[0]["result"]
    assert result["reward"] == 0.725
    assert result["done"] is False
    failed = [c["id"] for c in result["observation"]["checks"] if c["status"] == "FAIL"]
    assert failed == ["symmetry_d2"]


def test_unreadable_scores_zero_and_advances_the_attempt(client):
    _, steps = _episode(client, KINNAURI, UNREADABLE, UNREADABLE)
    assert [s["result"]["reward"] for s in steps] == [0.0, 0.0]
    assert [s["result"]["observation"]["attempt"] for s in steps] == [1, 2]
    assert steps[0]["result"]["observation"]["parse_error"].startswith("invalid PatternDSL JSON")


# ---------------------------------------------------------------- briefs and examples

def test_brief_list_covers_every_brief_of_each_suite():
    from pramaan.tasks import load_suite

    from pramaan_v2.apps.playground_render import SUITES, brief_options

    assert [s for s, _ in SUITES] == ["eval-v2.1", "train-v1"]
    for suite in ("eval-v2.1", "train-v1"):
        tasks, _ = load_suite(suite)
        options = brief_options(suite)
        assert sorted(task_id for task_id, _ in options) == sorted(t.id for t in tasks)
        assert len({label for _, label in options}) == len(options)
    labels = dict(brief_options("eval-v2.1"))
    assert labels[KINNAURI] == "Himachal Pradesh · Kinnauri shawl border · easy"


def test_suite_labels_count_from_the_suites():
    from pramaan_v2.apps.playground_render import suite_label

    assert suite_label("eval-v2.1") == "eval-v2.1 · 72 held-out briefs"
    assert suite_label("train-v1") == "train-v1 · 144 training briefs"


def test_example_chips_come_from_the_replay_file():
    from pramaan_v2.apps.playground_render import examples

    chips = examples(KINNAURI, "eval-v2.1")
    assert [c["label"] for c in chips] == ["Saved failure · Beam, try 1", "Saved pass · Beam, try 3",
                                           "Unreadable input"]
    assert json.loads(chips[0]["pattern"]) == _without_nulls(_beam_try(1)["proposal"])
    assert json.loads(chips[1]["pattern"]) == _without_nulls(_beam_try(3)["proposal"])
    assert chips[2]["pattern"] == UNREADABLE


def test_example_chips_fall_back_to_the_certified_reference():
    from pramaan_v2.apps.playground_render import examples

    chips = examples(NO_PASS, "eval-v2.1")
    assert chips[1]["label"] == "Certified reference"
    reference = next(json.loads(line) for line in
                     (ROOT / "benchmarks" / "data" / "eval-v2.1-references.jsonl").read_text(encoding="utf-8").splitlines()
                     if line.strip() and json.loads(line)["task_id"] == NO_PASS)
    assert json.loads(chips[1]["pattern"]) == _without_nulls(reference["candidate_pattern"])


def test_every_eval_brief_has_three_chips():
    from pramaan_v2.apps.playground_render import brief_options, examples

    for task_id, _ in brief_options("eval-v2.1"):
        chips = examples(task_id, "eval-v2.1")
        assert chips[-1]["label"] == "Unreadable input"
        assert 2 <= len(chips) <= 3, task_id


def test_saved_patterns_are_reflowed_but_equal():
    from pramaan_v2.apps.playground_render import format_pattern

    proposal = _without_nulls(_beam_try(1)["proposal"])
    text = format_pattern(proposal)
    assert json.loads(text) == proposal
    assert text.splitlines()[1].strip().startswith('"schema_version"')
    assert max(len(line) for line in text.splitlines()) <= 72


# ---------------------------------------------------------------- render: the three shapes

def test_idle_view_before_any_step(client):
    from pramaan_v2.apps.playground_render import render_view

    reset, _ = _episode(client, KINNAURI)
    view = render_view({"suite": "eval-v2.1", "reset": reset, "steps": []})
    assert "Nothing sent yet." in view["observation"]
    assert view["title"] == "Waiting for a step"
    assert view["status"] == "step 0 of 3 · done: false · best reward 0"
    assert view["done"] is False
    assert "A brick-red tile" in view["reset_panel"] and KINNAURI in view["reset_panel"]
    assert "D2 · left-right and top-bottom flips" in view["reset_panel"]
    assert view["slots"].count("Not sent") == 3


def test_rule_broke_view(client):
    from pramaan_v2.apps.playground_render import examples, render_view

    failure = examples(KINNAURI, "eval-v2.1")[0]["pattern"]
    reset, steps = _episode(client, KINNAURI, failure)
    view = render_view({"suite": "eval-v2.1", "reset": reset, "steps": steps})
    obs = view["observation"]
    assert ">0.725<" in obs
    assert "7 of 8 rules hold: 0.2 + 0.6 × 7/8" in obs
    assert "attempt 1 of 3 · done: false" in obs
    assert "One rule broke: the tile is not symmetric." in obs
    assert "symmetry_d2 · F_vertical<br>(16, 21) ≠ (47, 21)" in obs
    assert "24 of 4,096 pixels disagree with a mirror image." in obs
    assert obs.count("border: 2px solid #4FD1C5") == 2
    assert obs.count('class="v2-px"') == 2  # the clean tile and the evidence mask
    assert obs.count(">PASS<") == 8 and obs.count(">FAIL<") == 1
    assert "<details" in obs and "Raw observation JSON" in obs
    assert view["status"] == "step 1 of 3 · done: false · best reward 0.725"
    assert view["done"] is False
    assert "0.725 · symmetry broke" in view["slots"] and "Two steps left." in view["slots"]


def test_evidence_mask_matches_the_saved_mask(client):
    from pramaan_v2.apps.playground_render import evidence

    failure = _without_nulls(_beam_try(1)["proposal"])
    tile = evidence(KINNAURI, "eval-v2.1", json.dumps(failure))
    assert tile["mismatch_px"] == _beam_try(1)["mismatch_px"] == 24
    assert tile["mask_png"] == _beam_try(1)["mask_png"]


def test_every_rule_holds_view(client):
    from pramaan_v2.apps.playground_render import examples, render_view

    passing = examples(KINNAURI, "eval-v2.1")[1]["pattern"]
    reset, steps = _episode(client, KINNAURI, passing)
    view = render_view({"suite": "eval-v2.1", "reset": reset, "steps": steps})
    obs = view["observation"]
    assert "Every rule holds. The episode is over." in obs
    assert "outline: 1px solid #F4B63F" in obs
    assert "2a851a1377b4909d921d89000ff37d2b7446dc4d77bc87fceff200aa4f6cb141" in obs
    assert "All nine checks pass" in obs
    assert "#4FD1C5" not in obs
    assert view["done"] is True
    assert view["run_disabled"] is True
    assert view["hint"] == "The episode is over. Reset to play again."
    assert view["slots"].count("Not needed") == 2
    assert "1.0 · accepted" in view["slots"]


def test_unreadable_view(client):
    from pramaan_v2.apps.playground_render import render_view

    reset, steps = _episode(client, KINNAURI, UNREADABLE)
    view = render_view({"suite": "eval-v2.1", "reset": reset, "steps": steps})
    obs = view["observation"]
    assert "The checker could not read this." in obs
    assert "Expecting property name enclosed in double quotes" in obs
    assert "Two remain." in obs
    assert "<img" not in obs
    assert "the pattern could not be read" in obs
    assert view["done"] is False and view["run_disabled"] is False
    assert "0 · unreadable" in view["slots"]


def test_run_button_disabled_after_three_failing_steps(client):
    from pramaan_v2.apps.playground_render import examples, render_run_button, render_view

    failure = examples(KINNAURI, "eval-v2.1")[0]["pattern"]
    reset, steps = _episode(client, KINNAURI, failure, UNREADABLE, failure)
    view = render_view({"suite": "eval-v2.1", "reset": reset, "steps": steps})
    assert view["done"] is True and view["run_disabled"] is True
    assert "Not needed" not in view["slots"]
    assert view["status"] == "step 3 of 3 · done: true · best reward 0.725"
    assert " disabled" in render_run_button(True) and " disabled" not in render_run_button(False)


def test_saved_replay_link_and_sentence():
    from pramaan_v2.apps.playground_render import render_saved_line

    line = render_saved_line(KINNAURI, "eval-v2.1")
    assert "Beam's saved run on this brief went 0.725, 0.725, 1.0." in line
    assert f'href="/replays/{KINNAURI}"' in line


# ---------------------------------------------------------------- page

def test_page_markup():
    from pramaan_v2.apps.playground_render import render_page

    html = render_page()
    assert html.count("<h1") == 1
    assert "Play one episode yourself." in html
    assert "BUILD WITH IT</a> / PLAYGROUND · OPENENV · /web" in html
    assert 'aria-current="page"' in html  # Build with it in the shared header
    assert re.search(r'<label for="pg-pattern"[^>]*>', html) and '<textarea id="pg-pattern"' in html
    assert 'maxlength="20000"' in html
    assert 'role="status"' in html and 'aria-live="polite"' in html
    assert "Every step counts toward the three, readable or not." in html
    for href in ("/build#contract", 'href="/build"',
                 "https://huggingface.co/datasets/kshitijthakkar/pramaan-proof-of-art-openenv"):
        assert href in html
    for fact in (">3<", ">20,000<", ">0<"):
        assert fact in html
    assert "Random brief from a seed" in html
    assert html.count("<option") == 72 + 1 + 2


def test_snippets_name_the_routes_and_client():
    from pramaan_v2.apps.playground_render import client_snippet, http_snippet

    http = http_snippet(KINNAURI, "https://example.hf.space")
    assert '{base}/web/reset' in http and '{base}/web/step' in http and KINNAURI in http
    typed = client_snippet(KINNAURI, "https://example.hf.space")
    assert "PramaanEnv(base_url=url).sync()" in typed and "pattern_json =" in typed


def test_script_reads_the_deep_link():
    from pramaan_v2.apps.playground import PLAYGROUND_JS

    assert "task_id" in PLAYGROUND_JS and "URLSearchParams" in PLAYGROUND_JS


def test_view_does_not_trust_client_loop_bounds():
    from pramaan_v2.apps.playground_render import render_view

    state = {"reset": {"observation": {"task_id": "hp_kinnauri-shawl-geometric-border-easy-8",
                                       "max_attempts": 10**9}},
             "steps": [{"pattern_json": "x", "result": {"reward": 0.0, "done": False}}] * 500}
    out = render_view(state)
    assert out["slots"].count("STEP ") <= 3


def test_evidence_refuses_oversized_or_non_text_patterns():
    from pramaan_v2.apps.playground_render import MAX_ACTION_CHARS, evidence

    task = "hp_kinnauri-shawl-geometric-border-easy-8"
    assert evidence(task, "eval-v2.1", "{" + " " * MAX_ACTION_CHARS + "}") is None
    assert evidence(task, "eval-v2.1", {"not": "text"}) is None


def test_view_refuses_an_oversized_state_payload():
    from pramaan_v2.apps.playground import MAX_STATE_CHARS, view

    out = view("[" + " " * (MAX_STATE_CHARS + 1) + "]")
    assert out["task_id"] == "hp_kinnauri-shawl-geometric-border-easy-8"


@pytest.mark.parametrize("state", [
    {"reset": "x"}, {"reset": {"observation": "x"}}, {"reset": {"observation": {"metadata": "x", "task_id": 5}}},
    {"steps": "x"}, {"steps": [{"result": "x"}]}, {"steps": [{"result": {"observation": "x"}}]},
    {"steps": [{"result": {"observation": {"checks": "x"}, "reward": "x"}}]},
    {"steps": [{"result": {"observation": {"attempt": "<b>1</b>", "max_attempts": "<i>3</i>"}, "reward": 0.5}}]},
    {"suite": 5, "seed": "<s>"},
])
def test_view_survives_a_malformed_state(state):
    from pramaan_v2.apps.playground import view

    out = view(json.dumps(state))
    assert isinstance(out, dict) and out["task_id"]
    text = json.dumps(out)
    assert "<b>" not in text and "<i>" not in text and "<s>" not in text
