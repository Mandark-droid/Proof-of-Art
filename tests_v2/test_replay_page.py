"""The one-run page: one model on one brief, try by try (spec 8.3)."""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

KINNAURI = "hp_kinnauri-shawl-geometric-border-easy-8"
BEAM = f"/replays/{KINNAURI}/beam-hf"
SHA = "2a851a1377b4909d921d89000ff37d2b7446dc4d77bc87fceff200aa4f6cb141"


@pytest.fixture(scope="module")
def client():
    from pramaan_v2.server import app

    return TestClient(app)


@pytest.fixture(scope="module")
def data():
    from pramaan_v2.site.data import site_data

    return site_data()


@pytest.fixture(scope="module")
def beam1(client):
    response = client.get(f"{BEAM}?try=1")
    assert response.status_code == 200
    return response.text


def _main(html: str) -> str:
    return re.search(r"<main>.*</main>", html, re.S).group(0)


def test_every_run_renders(data):
    from pramaan_v2.site.render.replay import render_run

    runs = data.wall["runs"]
    assert len(runs) == len(data.models) * 72
    for run in runs:
        html = render_run(data, run["task"], run["model"], None, "http://localhost")
        assert html is not None, (run["task"], run["model"])
        assert html.count("<h1") == 1


def test_routes_answer_200(client, data):
    for run in data.wall["runs"][::37]:
        assert client.get(f'/replays/{run["task"]}/{run["model"]}').status_code == 200


def test_featured_run_try_one(beam1):
    assert "Beam, three tries, one border." in beam1
    assert "Passed on try 3 of 3." in beam1
    assert "0.725" in beam1
    assert "7 of 8 rules hold: 0.2 + 0.6 × 7/8" in beam1
    assert "(16, 21)" in beam1 and "(47, 21)" in beam1
    assert "24 of 4,096 pixels disagree with a mirror image" in beam1
    assert "Try 1 · first proposal" in beam1
    assert "WHAT THE MODEL WAS TOLD" in beam1
    assert '&quot;symmetry_d2&quot;' in beam1 or '"symmetry_d2"' in beam1
    assert "EXACT_DISCRETE" in beam1
    assert "PASSED ON TRY 3" in beam1
    assert "8,087 tokens" in beam1 and "8.3 s of model time" in beam1 and "3 ms of checking" in beam1
    assert "est. $0.0063" in beam1 and "validator 0.3.0" in beam1
    assert "Symmetric" in beam1 and "Repeat tiles match" in beam1 and "Ink coverage is within range" in beam1
    assert "0.2705 of the tile; allowed 0.02 to 0.70" in beam1


def test_both_views_are_in_the_page_and_the_checker_view_is_default(beam1):
    assert "What the eye sees" in beam1 and "What the checker sees" in beam1
    assert re.search(r'aria-pressed="true"[^>]*>What the checker sees', beam1)
    assert 'data-rv-overlay' in beam1
    assert "border-left: 1px dashed" in beam1  # the mirror axis


def test_try_picker_links_work_without_javascript(beam1):
    for n in (1, 2, 3):
        assert f'href="{BEAM}?try={n}"' in beam1
    assert "0.725 · broke symmetry" in beam1 and "1.0 · passed" in beam1


def test_featured_run_try_three_shows_the_proof(client):
    html = client.get(f"{BEAM}?try=3").text
    assert "PROOF YOU CAN RECOMPUTE" in html
    assert SHA in html
    assert "None. Anyone can recompute it; nobody has to trust it." in html
    assert "all 8 rules hold" in html
    assert "Try 3 · after two counterexamples" in html
    assert "WHAT THE MODEL WAS TOLD" not in html


@pytest.mark.parametrize("raw", ["0", "9", "abc", ""])
def test_bad_try_numbers_fall_back_to_try_one(client, beam1, raw):
    response = client.get(f"{BEAM}?try={raw}")
    assert response.status_code == 200
    assert _main(response.text) == _main(beam1)


def test_no_try_is_try_one(client, beam1):
    assert _main(client.get(BEAM).text) == _main(beam1)


def test_unknown_model_or_brief_is_404(client):
    assert client.get(f"/replays/{KINNAURI}/no-such-model").status_code == 404
    assert client.get("/replays/no-such-brief/beam-hf").status_code == 404


def _find(data, predicate):
    for task_id in data.task_ids:
        for model in data.replay(task_id)["models"]:
            if predicate(model):
                return task_id, model
    raise AssertionError("no such run")


def test_never_readable_run_has_no_tile(client, data):
    task_id, model = _find(data, lambda m: m["status"] == "-")
    html = client.get(f'/replays/{task_id}/{model["key"]}').text
    main = _main(html)
    viewer = main.split('data-rv-detail', 1)[0]
    assert "<img" not in viewer.split("Same brief", 1)[0]
    assert "Never produced a readable pattern." in html
    assert "NEVER READABLE" in html


def test_unreadable_try_shows_its_parse_message(client, data):
    task_id, model = _find(data, lambda m: m["status"] in "23" and m["tries"][0]["outcome"] == "unreadable")
    html = client.get(f'/replays/{task_id}/{model["key"]}?try=1').text
    first_line = model["tries"][0]["checks"][0]["message"].splitlines()[0]
    assert first_line in html.replace("&#x27;", "'").replace("&quot;", '"')
    assert "could not be read" in html


def test_failure_without_pixels_shows_the_measured_value(client):
    html = client.get("/replays/py_kolam-as-practised-in-puducherry-easy-35/gpt-6-astra").text
    assert "Traced 2 closed loops; required 1" in html


def test_head_carries_title_description_and_link_image(beam1):
    assert "<title>Beam on Kinnauri shawl border: passed on try 3 · Pramaan</title>" in beam1
    assert re.search(r'<meta name="description" content="Passed on try 3 of 3\. [^"]*brick-red', beam1)
    assert f'<meta property="og:image" content="http://testserver/og/replay/{KINNAURI}/beam-hf.png">' in beam1 \
        or re.search(rf'<meta property="og:image" content="https?://[^"]+/og/replay/{KINNAURI}/beam-hf\.png">', beam1)


def test_actions_and_strip(beam1, data):
    assert "Copy link" in beam1 and f'/og/replay/{KINNAURI}/beam-hf.png' in beam1
    assert "Open the dataset row" in beam1 and "huggingface.co/datasets/" in beam1
    from pramaan_v2.constants import number_word

    assert f"Same brief, {number_word(len(data.models))} models" in beam1
    for model in data.models:
        assert f'href="/replays/{KINNAURI}/{model["key"]}"' in beam1
    assert "try 3 · this run" in beam1
    assert f"/playground?task_id={KINNAURI}" in beam1
    assert f"/replays/{KINNAURI}/beam-hf</span>" in beam1


def test_check_labels_use_the_design_wording():
    from pramaan_v2.site.labels import check_label

    assert check_label("symmetry_d2") == "Symmetric"
    assert check_label("periodic_raster") == "Repeat tiles match"
    assert check_label("layer_registration") == "Layers line up"
    assert check_label("exclusive_ink_collision") == "Ink layers do not overlap"
    assert check_label("minimum_gap") == "Motifs keep their distance"
    assert check_label("coverage_bound") == "Ink coverage is within range"
    assert check_label("candidate_truncated") == "Cut off at the token limit"
    assert check_label("candidate_parse") == "Unreadable proposal"
    assert check_label("schema_valid") == "Readable pattern"
    assert check_label("kolam_loop_count") == "Kolam stroke count"
