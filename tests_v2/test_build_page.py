"""Build with it: /build (spec 8.8)."""
from __future__ import annotations

import ast
import html
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    from pramaan_v2.server import app

    return TestClient(app)


@pytest.fixture(scope="module")
def data():
    from pramaan_v2.site.data import site_data

    return site_data()


@pytest.fixture(scope="module")
def page(client):
    response = client.get("/build")
    assert response.status_code == 200
    return response.text


def _text(markup: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", markup)))


def _section(page: str, anchor: str) -> str:
    return page.split(f'id="{anchor}"', 1)[1].split("</section>", 1)[0]


def test_one_h1_and_the_header(page):
    assert page.count("<h1") == 1
    assert "The checker is the benchmark. And the reward." in page
    assert "BUILD WITH IT · BENCHMARK · OPENENV ENVIRONMENT · DATASETS" in page
    assert re.search(r'<a href="/build" aria-current="page"', page)


def test_lede_counts_come_from_meta(page, data):
    rendered = data.meta["totals"]["rendered"]
    assert f"replays all {rendered:,} saved proposals" in _text(page)


def _rows(section: str) -> list[str]:
    return [html.unescape(name) for name in re.findall(r'<th scope="row"[^>]*>([^<]+)</th>', section)]


def test_contract_tables_are_the_pydantic_fields(page):
    from pramaan_env.models import PramaanAction, PramaanObservation

    contract = _section(page, "contract")
    action = contract.split('data-model="PramaanAction"', 1)[1].split("</table>", 1)[0]
    observation = contract.split('data-model="PramaanObservation"', 1)[1].split("</table>", 1)[0]
    assert sorted(_rows(action)) == sorted(PramaanAction.model_fields)
    assert sorted(_rows(observation)) == sorted(PramaanObservation.model_fields)
    assert len(_rows(observation)) == len(PramaanObservation.model_fields)


def test_contract_types_are_generated(page):
    contract = _section(page, "contract")
    assert re.search(r">pattern_json</th><td[^>]*>str</td>", contract)
    assert re.search(r">certificate</th><td[^>]*>dict \| None</td>", contract)
    assert re.search(r">attempt</th><td[^>]*>int</td>", contract)


def test_playground_button_and_dataset_links(page, data):
    assert re.search(r'<a href="/playground"[^>]*>Open the interactive environment</a>', page)
    for repo in (data.meta["dataset"], "kshitijthakkar/pramaan-proof-of-art-train-v1",
                 "kshitijthakkar/pramaan-proof-of-art-openenv"):
        assert f'href="https://huggingface.co/datasets/{repo}"' in page
    assert 'href="/submit"' in page


def test_dataset_counts(page, data):
    refs = sum(1 for line in (ROOT / "benchmarks/data/eval-v2.1-references.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip())
    text = _text(page)
    assert f"{data.meta['totals']['runs']} model and brief records with full trajectories, plus {refs} certified references." in text
    counts = {name: sum(1 for line in (ROOT / f"benchmarks/data/train-v1-{name}.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()) for name in ("sft", "repair", "preference", "trajectories")}
    assert (f"{counts['sft']} supervised rows, {counts['repair']} repairs, {counts['preference']} preference pairs, "
            f"{counts['trajectories']} trajectories.") in text


def test_episode_is_drawn_from_kolam_data(page, data):
    tries = data.kolam["tn_kolam-easy-22"]["models"]["beam-hf"]
    assert [t["mirrors"] for t in tries] == [12, 8, 4]
    assert [round(t["reward"], 3) for t in tries] == [0.691, 0.745, 1.0]
    episode = _section(page, "episode")
    text = _text(episode)
    assert "step 1 · 0 loops" in text and "reward 0.69" in text
    assert "step 2 · 2 loops" in text and "reward 0.75" in text
    assert "step 3 · 1 loop · done" in text and "reward 1.0" in text
    for t in tries:
        for d in t["paths"]:
            assert d in episode
    assert "Beam on the Tamil Nadu kolam brief, one closed loop required." in text


def test_open_problem_numbers_come_from_data(page, data):
    models = data.leaderboard["models"]
    n = len(models)
    jaali_briefs = next(g["briefs"] for g in data.meta["grammars"] if g["id"] == "jaali_lattice")
    jaali = [m["by_grammar"]["jaali_lattice"] for m in models]
    kolam_none = sum(1 for m in models if m["by_grammar"]["kolam_loop"] == 0)
    from pramaan_v2.constants import number_word

    text = _text(_section(page, "open-problems"))
    assert f"{sum(jaali)} of {n * jaali_briefs} results pass" in text
    assert f"the best model manages {max(jaali)} of {jaali_briefs}" in text
    assert f"{number_word(kolam_none).capitalize()} of {number_word(n)} models pass none of the three briefs." in text
    totals = data.meta["totals"]
    assert f"{totals['low_ink']} of {totals['passed']} accepted designs are nearly blank." in text
    assert "Two models passed an impossible brief that way." in text
    assert (sum(jaali), n * jaali_briefs, max(jaali), kolam_none) == (12, 55, 3, 6)
    assert len(re.findall(r"<li", _section(page, "open-problems"))) == 6


def test_grammar_table_counts_from_meta(page, data):
    table = _section(page, "grammars")
    rows = re.findall(r'<th scope="row"[^>]*>([^<]+)</th><td[^>]*>(\d+)</td>', table)
    assert rows == [(g["label"], str(g["briefs"])) for g in data.meta["grammars"]]
    assert "Jaali minimum web thickness uses an approximation" in table


def test_nobody_has_trained_empty_state(page):
    assert "Nobody has trained on this yet." in page
    assert "[YOUR TRAINING CURVE]" in page


def test_footer_versions_from_meta(page, data):
    assert f"Validator {data.meta['validator']} · reward {data.meta['reward']}" in page


def _pre_blocks(page: str) -> list[str]:
    return [html.unescape(re.sub(r"<[^>]+>", "", block)) for block in re.findall(r"<pre[^>]*>(.*?)</pre>", page, re.S)]


def test_commands_use_real_flags_and_files(page):
    score = _pre_blocks(page)[0]
    run_source = (ROOT / "benchmarks/run.py").read_text(encoding="utf-8")
    run_part = score.split("benchmarks/run.py")[1].split("\n\n")[0]
    for flag in re.findall(r"(--[a-z-]+)", run_part):
        assert f'"{flag}"' in run_source, flag
    assert (ROOT / "benchmarks/suites/eval-v2.1.jsonl").exists() and (ROOT / "benchmarks/models.json").exists()
    assert (ROOT / "scripts/pack_submission.py").exists()


def test_python_snippets_parse_and_import_real_names(page, data):
    blocks = _pre_blocks(page)
    reward, client_snippet = blocks[1], blocks[2]
    ast.parse(reward)
    ast.parse(client_snippet)
    from pramaan_env.reward_fn import pramaan_reward  # noqa: F401
    from pramaan_env import PramaanAction, PramaanEnv  # noqa: F401

    task = re.search(r'task_id=\["([^"]+)"\]', reward).group(1)
    train_ids = {line.split('"id": "', 1)[1].split('"', 1)[0]
                 for line in (ROOT / "benchmarks/suites/train-v1.jsonl").read_text(encoding="utf-8").splitlines()
                 if '"id": "' in line}
    assert task in train_ids
    # the reward snippet runs as written
    namespace: dict = {}
    exec(compile(reward, "reward_snippet", "exec"), namespace)  # noqa: S102
    assert namespace["rewards"] == [0.0]
    assert 'url = "http://testserver"' in client_snippet or "url = " in client_snippet
    assert "hp_kinnauri-shawl-geometric-border-easy-8" in client_snippet
