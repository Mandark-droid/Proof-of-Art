"""Community runs: aggregated per model from v1's replay-validated snapshot, served without waiting on the Hub."""
from __future__ import annotations

import json
import re
import threading
import time

import pytest
from fastapi.testclient import TestClient

MODELS = ("community-alpha", "community-beta")
BRIEFS_EACH = {"community-alpha": 5, "community-beta": 3}


@pytest.fixture()
def community(monkeypatch):
    from pramaan_v2.site import community as module

    module._reset()
    yield module
    module.wait_idle(10)
    module._reset()


@pytest.fixture(scope="module")
def snapshot(published):
    """A snapshot shaped exactly like v1's: per-brief trajectory rows plus v1's own summary."""
    from pramaan.app import _summarize_community_rows

    rows = []
    for key, source in zip(MODELS, ("gpt-6-astra", "claude-opus-5-5")):
        picked = [r for r in published.rows if r["model_key"] == source][:BRIEFS_EACH[key]]
        rows += [dict(r, model_key=key, run_id=f"run-{key}") for r in picked]
    prompt_ids = sorted({r["prompt_id"] for r in published.rows})
    summary = _summarize_community_rows(rows, prompt_ids, ["beam-hf", *MODELS])
    return {"status": "Community results loaded.", "rows": rows, "beam_rows": [],
            "selected": {k: {"run_id": f"run-{k}", "submitted_at": ""} for k in MODELS}, "summary": summary}


def _expected(snapshot, key):
    rows = [r for r in snapshot["rows"] if r["model_key"] == key]
    return len(rows), sum(bool(r["validity_at_3"]) for r in rows), sum(bool(r["validity_at_1"]) for r in rows)


def _until(predicate, timeout=8.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    return predicate()


# ---------------------------------------------------------------- D: the per-model aggregate


def test_aggregate_is_one_row_per_model_with_its_own_n(community, snapshot):
    public = community.aggregate(snapshot)
    assert public["provenance"] == "community"
    assert [r["key"] for r in public["rows"]] == list(MODELS)
    for row in public["rows"]:
        n, within, first = _expected(snapshot, row["key"])
        assert (row["n"], row["within3"], row["first_try"]) == (n, within, first)
        assert row["provenance"] == "community" and row["label"]
        assert set(row) == {"key", "label", "provider", "n", "within3", "first_try", "provenance"}
    assert "attempts" not in json.dumps(public) and "candidate" not in json.dumps(public)


def test_aggregate_prefers_the_summary_counts(community, snapshot):
    edited = json.loads(json.dumps(snapshot))
    edited["summary"]["by_model"]["community-beta"]["overall"].update(n=4, validity_at_3=0.75, validity_at_1=0.25)
    beta = next(r for r in community.aggregate(edited)["rows"] if r["key"] == "community-beta")
    assert (beta["n"], beta["within3"], beta["first_try"]) == (4, 3, 1)


def test_aggregate_without_a_summary_counts_the_rows(community, snapshot):
    bare = dict(snapshot, summary={})
    for row in community.aggregate(bare)["rows"]:
        assert (row["n"], row["within3"], row["first_try"]) == _expected(snapshot, row["key"])


@pytest.mark.parametrize("bad", [None, "x", {"rows": "x"}, {"rows": []}, {"rows": [1, {"model_key": 3}]}])
def test_aggregate_of_nothing_is_the_empty_state(community, bad):
    assert community.aggregate(bad) == {"status": community.EMPTY, "rows": [], "provenance": "community"}


def test_populated_leaderboard_and_api(community, snapshot, monkeypatch):
    from pramaan_v2.server import app

    monkeypatch.setattr(community, "community_snapshot", lambda: community.aggregate(snapshot))
    client = TestClient(app)
    page = client.get("/leaderboard").text
    block = re.search(r'id="community"(.*?)<!-- /community -->', page, re.S).group(1)
    assert "2 community runs have been merged." in block
    assert "data-lb-community-table" in block and 'data-lb-community-table class="lb-scroll" style="overflow-x: auto; margin-top: 20px" hidden' not in block
    for key in MODELS:
        n, within, first = _expected(snapshot, key)
        row = re.search(r"<tr[^>]*>(?:(?!</tr>).)*?%s(?:(?!</tr>).)*</tr>" % re.escape(
            community.aggregate(snapshot)["rows"][MODELS.index(key)]["label"]), block, re.S).group(0)
        assert "COMMUNITY" in row and f"{within} of {n}" in row and f">{first}</td>" in row
    api = client.get("/api/community.json").json()
    assert api == community.aggregate(snapshot)
    assert all(set(r) == {"key", "label", "provider", "n", "within3", "first_try", "provenance"} for r in api["rows"])


# ---------------------------------------------------------------- E: never waiting on the Hub


def test_a_slow_loader_never_blocks_and_its_rows_arrive_later(community, snapshot, monkeypatch):
    import pramaan.app as v1

    def slow():
        time.sleep(3)
        return snapshot

    monkeypatch.setattr(v1, "_community_comparison_snapshot", slow)
    started = time.perf_counter()
    first = community.community_snapshot()
    assert time.perf_counter() - started < 0.2
    assert first["rows"] == [] and first["status"] == community.EMPTY
    started = time.perf_counter()
    community.community_snapshot()
    assert time.perf_counter() - started < 0.2
    later = _until(lambda: community.community_snapshot()["rows"])
    assert [r["key"] for r in later] == list(MODELS)


def test_a_failed_refresh_keeps_the_last_good_snapshot(community, snapshot, monkeypatch):
    import pramaan.app as v1

    monkeypatch.setattr(v1, "_community_comparison_snapshot", lambda: snapshot)
    community.community_snapshot()
    assert _until(lambda: community.community_snapshot()["rows"])

    def broken():
        time.sleep(0.5)
        raise RuntimeError("hub unavailable")

    monkeypatch.setattr(v1, "_community_comparison_snapshot", broken)
    monkeypatch.setattr(community, "REFRESH_AFTER", 0)
    started = time.perf_counter()
    kept = community.community_snapshot()
    assert time.perf_counter() - started < 0.2
    assert community.wait_idle(5)
    assert [r["key"] for r in kept["rows"]] == list(MODELS)
    assert [r["key"] for r in community.community_snapshot()["rows"]] == list(MODELS)


def test_a_failing_first_load_serves_the_empty_state(community, monkeypatch):
    import pramaan.app as v1

    def broken():
        raise RuntimeError("hub unavailable")

    monkeypatch.setattr(v1, "_community_comparison_snapshot", broken)
    assert community.community_snapshot()["rows"] == []
    assert community.wait_idle(5)
    snap = community.community_snapshot()
    assert snap == {"status": community.EMPTY, "rows": [], "provenance": "community"}


def test_refresh_clears_v1_cache_and_runs_one_at_a_time(community, snapshot, monkeypatch):
    import pramaan.app as v1

    calls, clears, gate = [], [], threading.Event()

    def loader():
        calls.append(1)
        gate.wait(5)
        return snapshot

    loader.cache_clear = lambda: clears.append(1)
    monkeypatch.setattr(v1, "_community_comparison_snapshot", loader)
    monkeypatch.setattr(community, "REFRESH_AFTER", 0)
    for _ in range(5):
        community.community_snapshot()
    gate.set()
    assert community.wait_idle(5)
    assert len(calls) == 1 and len(clears) == 1


def test_refresh_waits_its_interval(community, snapshot, monkeypatch):
    import pramaan.app as v1

    calls = []
    monkeypatch.setattr(v1, "_community_comparison_snapshot", lambda: calls.append(1) or snapshot)
    community.community_snapshot()
    assert community.wait_idle(5)
    for _ in range(3):
        community.community_snapshot()
    assert community.wait_idle(5) and len(calls) == 1
    assert community.REFRESH_AFTER == 120


def test_the_server_starts_a_refresh_at_startup():
    from pramaan_v2.server import app
    from pramaan_v2.site import community

    handlers = app.router.on_startup
    assert community.start_refresh in handlers
