"""The Replays page: one brief, every model's tries side by side (spec 8.2)."""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

KINNAURI = "hp_kinnauri-shawl-geometric-border-easy-8"
PUDUCHERRY = "py_kolam-as-practised-in-puducherry-easy-35"


@pytest.fixture(scope="module")
def client():
    from pramaan_v2.server import app

    return TestClient(app)


@pytest.fixture(scope="module")
def data():
    from pramaan_v2.site.data import site_data

    return site_data()


@pytest.fixture(scope="module")
def kinnauri(client):
    response = client.get(f"/replays/{KINNAURI}")
    assert response.status_code == 200
    return response.text


def test_every_brief_renders(client, data):
    assert len(data.task_ids) == 72
    for task_id in data.task_ids:
        response = client.get(f"/replays/{task_id}")
        assert response.status_code == 200, task_id
        assert response.text.count("<h1") == 1, task_id


def test_unknown_brief_is_404(client):
    response = client.get("/replays/no-such-brief")
    assert response.status_code == 404
    assert "That page does not exist." in response.text


def test_default_route_is_the_kinnauri_brief(client, kinnauri):
    from pramaan_v2.site.render.replays import DEFAULT_TASK

    assert DEFAULT_TASK == KINNAURI
    body = lambda html: re.search(r"<main>.*</main>", html, re.S).group(0)  # noqa: E731
    assert body(client.get("/replays").text) == body(kinnauri)


def test_heading_eyebrow_and_counts_come_from_data(kinnauri, data):
    from pramaan_v2.constants import number_word

    n = len(data.models)
    assert f"One brief. {number_word(n).capitalize()} answers." in kinnauri
    assert "REPLAYS · BRIEF 17 OF 72" in kinnauri
    assert "HIMACHAL PRADESH · KINNAURI SHAWL BORDER · REPEAT TILES · D2 SYMMETRY" in kinnauri
    summary = data.replay(KINNAURI)["summary"]
    assert f'{summary["first_try"]}</strong> models passed on the first try' in kinnauri
    assert f'{summary["repaired"]}</strong> repaired it' in kinnauri
    assert f'{summary["never"]}</strong> never passed' in kinnauri
    assert 'href="/traditions/HP"' in kinnauri
    assert "Left: the certified reference." in kinnauri


def test_lunas_repeated_pair_is_the_observation(kinnauri):
    assert "GPT-6 Luna was handed the same pixel pair twice and returned it a third time." in kinnauri
    assert kinnauri.count("(18, 20) ≠ (45, 43)") >= 3


def test_every_model_name_links_to_its_run(kinnauri, data):
    rows = re.findall(r'<th scope="row"[^>]*>(.*?)</th>', kinnauri, re.S)
    assert len(rows) == len(data.models)
    for model, row in zip(data.models, rows):
        assert f'href="/replays/{KINNAURI}/{model["key"]}"' in row
        assert model["label"] in row


def test_cells_follow_the_board(kinnauri):
    assert "PASSED" in kinnauri and "not needed" in kinnauri
    assert "24 PX OFF" in kinnauri and "(16, 21) ≠ (47, 21)" in kinnauri
    assert "UNREADABLE" in kinnauri
    assert 'class="v2-px"' in kinnauri


def test_controls_navigate_without_javascript(kinnauri):
    assert '<label for="brief-state"' in kinnauri
    assert re.search(r'<option value="hp_kinnauri-shawl-geometric-border-easy-8"[^>]*selected', kinnauri)
    assert 'href="/replays/hp_kullu-shawl-end-bands-hard-8"' in kinnauri
    assert "Back to the wall" in kinnauri
    # Without JavaScript the state choice is a list of links.
    assert "<noscript>" in kinnauri and 'href="/replays/ap_mangalagiri-sarees-and-fabrics-easy-0"' in kinnauri


def test_previous_and_next_brief_in_wall_order(kinnauri):
    assert "← Haryana · hard" in kinnauri
    assert "Himachal Pradesh · hard →" in kinnauri


def test_impossible_brief_has_no_reference_and_says_so(client):
    html = client.get(f"/replays/{PUDUCHERRY}").text
    assert "Left: the certified reference." not in html
    assert "cannot be met as written" in html
    assert 'aria-label="Certified reference' not in html


def test_first_and_last_brief_have_one_neighbour(client, data):
    first = client.get(f"/replays/{data.task_ids[0]}").text
    last = client.get(f"/replays/{data.task_ids[-1]}").text
    assert "→</a>" in first and "← " not in re.search(r"<footer.*</footer>", first, re.S).group(0)
    assert "← " in last
