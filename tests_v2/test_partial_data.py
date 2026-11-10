"""The site built from incomplete data: a model missing runs, a model with no summary document.

Each fixture rebuilds the site data from the published rows into a temporary directory and
points the routes at it, so every page is rendered from data the builder really writes.
"""
from __future__ import annotations

import html as htmllib
import json
import re
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

KEEP = ("gpt-6-astra", "claude-opus-5-5", "beam-hf", "gpt-6.1-sol")
SHORT = "gpt-6.1-sol"            # loses ten runs
UNSUMMARISED = "claude-opus-5-5"  # has runs but no summary document
FEATURED = "hp_kinnauri-shawl-geometric-border-easy-8"
SPECIAL = {FEATURED, "tn_kolam-easy-22", "py_kolam-as-practised-in-puducherry-easy-35",
           "mn_moirang-phee-moirang-pheejin-stepped-triangle-border-easy-14"}


def _dropped() -> list[str]:
    """Ten hard briefs near the start of the wall row, so most of the row sits after the gap."""
    from pramaan.tasks import load_suite
    from pramaan_v2.constants import STATE_ORDER

    tasks, _ = load_suite("eval-v2.1")
    hard = sorted((t for t in tasks if t.level == "hard" and t.id not in SPECIAL),
                  key=lambda t: STATE_ORDER.index(t.state))
    return [t.id for t in hard[1:11]]


DROPPED = _dropped()


def _build(tmp_path_factory, published, name, rows, summary):
    from pramaan_v2.build.site_data import build
    from pramaan_v2.build.sources import Published
    from pramaan_v2.site.data import SiteData

    out = tmp_path_factory.mktemp(name)
    build(out, Published(rows=rows, summary=summary, dataset=published.dataset, revision=published.revision))
    return SiteData(root=out)


def _summary(published, keys):
    summary = dict(published.summary)
    summary["by_model"] = {k: v for k, v in published.summary["by_model"].items() if k in keys}
    return summary


@pytest.fixture(scope="session")
def partial(tmp_path_factory, published):
    rows = [r for r in published.rows if r["model_key"] in KEEP
            and not (r["model_key"] == SHORT and r["prompt_id"] in DROPPED)]
    return _build(tmp_path_factory, published, "partial", rows, _summary(published, KEEP))


@pytest.fixture(scope="session")
def unsummarised(tmp_path_factory, published):
    rows = [r for r in published.rows if r["model_key"] in KEEP]
    keys = set(KEEP) - {UNSUMMARISED}
    return _build(tmp_path_factory, published, "unsummarised", rows, _summary(published, keys))


@contextmanager
def serving(data, monkeypatch):
    """A client whose pages and link images all read ``data``; the community snapshot is empty."""
    from pramaan_v2.server import app
    from pramaan_v2.site import community, routes
    from pramaan_v2.site.render import og

    monkeypatch.setattr(routes, "site_data", lambda: data)
    monkeypatch.setattr(og, "site_data", lambda: data)
    monkeypatch.setattr(community, "community_snapshot",
                        lambda: {"status": community.EMPTY, "rows": [], "provenance": "community"})
    og.og_home.cache_clear()
    og._og_replay.cache_clear()
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        og.og_home.cache_clear()
        og._og_replay.cache_clear()


def _urls(data, focus: str) -> list[str]:
    urls = ["/", "/leaderboard", "/leaderboard?sort=cost", "/leaderboard?sort=first",
            "/leaderboard?sort=evaluated", "/leaderboard?prov=canonical", "/leaderboard?q=gpt",
            "/leaderboard?compare=" + ",".join(KEEP[:3]), "/leaderboard?compare=" + focus,
            "/replays", "/traditions", "/traditions/TN", "/build", "/og/home.png"]
    urls += [f"/replays/{task}" for task in data.task_ids]
    seen = set()
    for run in data.wall["runs"]:
        if run["model"] == focus or run["model"] not in seen:
            seen.add(run["model"])
            urls.append(f"/replays/{run['task']}/{run['model']}")
    urls += [f"/og/replay/{r['task']}/{r['model']}.png" for r in data.wall["runs"] if r["model"] == focus][:4]
    return urls


# ---------------------------------------------------------------- the loader


def test_site_data_reads_from_its_own_root(partial):
    assert partial.meta["totals"]["models"] == len(KEEP)
    assert partial.meta["totals"]["runs"] == len(KEEP) * 72 - len(DROPPED)
    assert len(partial.leaderboard["models"]) == len(KEEP)
    assert len(partial.wall["runs"]) == partial.meta["totals"]["runs"]
    replay = partial.replay(DROPPED[0])
    assert {m["key"] for m in replay["models"]} == set(KEEP) - {SHORT}


# ---------------------------------------------------------------- A: a model with fewer runs than briefs


def test_every_route_renders_when_a_model_is_missing_runs(partial, monkeypatch):
    assert next(m for m in partial.models if m["key"] == SHORT)["rank"] is None
    with serving(partial, monkeypatch) as client:
        failed = {url: client.get(url).status_code for url in _urls(partial, SHORT)}
        failed = {url: code for url, code in failed.items() if code != 200}
        assert failed == {}
        assert client.get(f"/replays/{DROPPED[0]}/{SHORT}").status_code == 404


def test_leaderboard_marks_the_unranked_model_and_its_missing_runs(partial, monkeypatch):
    with serving(partial, monkeypatch) as client:
        page = client.get("/leaderboard").text
    row = re.search(r'<div class="lb-rank-row" data-model="%s".*?</div>' % re.escape(SHORT), page, re.S).group(0)
    assert ">–</span>" in row
    short = next(m for m in partial.leaderboard["models"] if m["key"] == SHORT)
    assert f'{short["within3"]} of {short["n"]}' in row and short["n"] == 72 - len(DROPPED)
    matrix = re.search(r'<div class="lb-mrow" data-model="%s".*?</div></div>' % re.escape(SHORT), page, re.S).group(0)
    assert matrix.count('class="lb-c lb-gap"') == len(DROPPED)
    assert all(f"/replays/{task}/{SHORT}" not in matrix for task in DROPPED)


# ---------------------------------------------------------------- B: a model with no summary document


def test_every_route_renders_when_a_model_has_no_summary(unsummarised, monkeypatch):
    m = next(m for m in unsummarised.leaderboard["models"] if m["key"] == UNSUMMARISED)
    assert m["spend_usd"] is None and m["tokens_per_brief"] is None and m["provider"] is None
    with serving(unsummarised, monkeypatch) as client:
        failed = {url: client.get(url).status_code for url in _urls(unsummarised, UNSUMMARISED)}
        assert {url: code for url, code in failed.items() if code != 200} == {}


def test_models_without_cost_stay_off_the_scatter_and_sort_last(unsummarised, monkeypatch):
    with serving(unsummarised, monkeypatch) as client:
        page = client.get("/leaderboard?sort=cost").text
        compare = client.get("/leaderboard?compare=" + UNSUMMARISED).text
    cost = re.search(r'id="cost"(.*?)<!-- /cost -->', page, re.S).group(1)
    assert f'data-model="{UNSUMMARISED}"' not in cost
    results = re.search(r'id="results"(.*?)<!-- /results -->', page, re.S).group(1)
    order = re.findall(r'<tr class="lb-res-row" data-model="([^"]+)"', results)
    assert order[-1] == UNSUMMARISED
    row = re.search(r'<tr class="lb-res-row" data-model="%s".*?</tr>' % re.escape(UNSUMMARISED), results, re.S).group(0)
    assert row.count(">–</td>") == 3
    values = re.search(r"<table data-lb-values.*?</table>", compare, re.S).group(0)
    assert values.count(">–</td>") == 3
    facts = json.loads(re.search(r'<script type="application/json" id="lb-facts">(.*?)</script>', page, re.S).group(1))
    assert facts[UNSUMMARISED]["provider"] == "" and facts[UNSUMMARISED]["spend"] is None


# ---------------------------------------------------------------- C: the Home wall with a gap


def test_home_wall_anchors_carry_their_own_cell(partial, monkeypatch):
    from pramaan_v2.site.render.replays import run_url

    with serving(partial, monkeypatch) as client:
        page = client.get("/").text
    wall = partial.wall
    grid = re.search(r'<div class="hw-grid" data-home-wall([^>]*)>', page).group(1)
    assert f'data-cols="{wall["cols"]}"' in grid and f'data-rows="{wall["rows"]}"' in grid
    anchors = re.findall(r'<a class="hw-t[^"]*" href="([^"]+)" data-col="(\d+)" data-row="(\d+)" '
                         r'style="--c:(\d+);--r:(\d+);grid-area:(\d+) / (\d+)"', page)
    assert len(anchors) == len(wall["runs"]) == partial.meta["totals"]["runs"]
    expected = {(htmllib.escape(run_url(r["task"], r["model"])), r["col"], r["row"]) for r in wall["runs"]}
    got = set()
    for href, col, row, c, r, grow, gcol in anchors:
        assert (col, row) == (c, r) and (int(grow), int(gcol)) == (int(row) + 1, int(col) + 1)
        got.add((href, int(col), int(row)))
    assert got == expected


def test_home_script_reads_each_tiles_own_cell():
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "pramaan_v2" / "static" / "js" / "20-home.js").read_text(encoding="utf-8")
    assert "data-col" in js and "data-row" in js and 'getAttribute("data-rows")' in js
    assert "Math.floor(i / COLS)" not in js and "tiles.length / COLS" not in js


# ---------------------------------------------------------------- I (11): JSON inside <script>


def test_script_json_escapes_every_angle_bracket(partial, tmp_path, monkeypatch):
    import shutil

    from pramaan_v2.site.data import SiteData
    from pramaan_v2.site.render.leaderboard import script_json

    root = tmp_path / "site"
    shutil.copytree(partial.root, root)
    data = SiteData(root=root)
    evil = "Evil <!--<script>alert(1)</script>"
    key = data.meta["models"][0]["key"]
    data.meta["models"][0]["label"] = evil
    next(m for m in data.leaderboard["models"] if m["key"] == key)["label"] = evil
    with serving(data, monkeypatch) as client:
        for url in ("/", "/leaderboard"):
            page = client.get(url).text
            blobs = re.findall(r'<script type="application/json"[^>]*>(.*?)</script>', page, re.S)
            assert blobs, url
            for blob in blobs:
                assert "<" not in blob, (url, blob[:200])
            assert any(evil in json.dumps(json.loads(blob), ensure_ascii=False) for blob in blobs), url
    assert "<" not in script_json("x", {"a": "<!--<script"}).removeprefix('<script type="application/json" id="x">')[:-9]
