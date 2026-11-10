import gzip

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from pramaan_v2.server import app
    from pramaan_v2.site import pagecache

    pagecache.clear()
    return TestClient(app)


def test_a_page_is_rendered_once_then_served_from_memory(client, monkeypatch):
    from pramaan_v2.site.render import home

    calls = []
    real = home.render_home
    monkeypatch.setattr(home, "render_home", lambda *a, **k: calls.append(1) or real(*a, **k))
    first = client.get("/")
    second = client.get("/")
    assert first.status_code == second.status_code == 200
    assert first.text == second.text
    assert len(calls) == 1


def test_cached_pages_go_out_gzip_encoded_once(client):
    response = client.get("/replays", headers={"Accept-Encoding": "gzip"})
    assert response.headers["content-encoding"] == "gzip"
    assert "One brief." in response.text
    raw = client.get("/replays", headers={"Accept-Encoding": "identity"})
    assert raw.headers.get("content-encoding") in (None, "identity")
    assert raw.text == response.text


def test_missing_pages_are_not_cached(client):
    from pramaan_v2.site import pagecache

    assert client.get("/replays/no-such-brief").status_code == 404
    assert not any("no-such-brief" in key for key in pagecache.keys())


def test_leaderboard_cache_follows_the_community_snapshot(client, monkeypatch):
    from pramaan_v2.site import community

    rows = {"status": "x", "rows": [], "provenance": "community"}
    monkeypatch.setattr(community, "community_snapshot", lambda: rows)
    before = client.get("/leaderboard").text
    rows = {"status": "x", "rows": [{"key": "community-model", "label": "Community Model Z", "provider": "p",
                                     "n": 12, "within3": 9, "first_try": 4, "provenance": "community"}],
            "provenance": "community"}
    monkeypatch.setattr(community, "community_snapshot", lambda: rows)
    after = client.get("/leaderboard").text
    assert "Community Model Z" in after and "Community Model Z" not in before


def test_warm_up_renders_every_read_only_page():
    from pramaan_v2.site import pagecache, routes
    from pramaan_v2.site.data import site_data

    pagecache.clear()
    count = routes.warm_pages("http://localhost")
    data = site_data()
    expected = 1 + 1 + 1 + 1 + len(data.task_ids) + len(data.wall["runs"]) + 1 + len(data.meta["states"])
    assert count >= expected
    assert len(pagecache.keys()) >= expected


def test_the_server_warms_pages_at_startup():
    from pramaan_v2.server import app
    from pramaan_v2.site import routes

    handlers = getattr(app.router, "on_startup", [])
    assert routes.start_warming in handlers


def test_warm_up_base_comes_from_configuration(monkeypatch):
    from pramaan_v2.site import routes

    monkeypatch.setenv("SPACE_HOST", "kshitijthakkar-pramaan-proof-of-art-v2.hf.space")
    assert routes.configured_base() == "https://kshitijthakkar-pramaan-proof-of-art-v2.hf.space"


def test_arbitrary_query_strings_never_create_cache_entries(client):
    from pramaan_v2.site import pagecache

    client.get("/traditions/HP")
    client.get("/leaderboard")
    client.get("/replays/hp_kinnauri-shawl-geometric-border-easy-8/beam-hf")
    before = len(pagecache.keys())
    for i in range(30):
        assert client.get(f"/traditions/HP?grammar=junk{i}").status_code == 200
        assert client.get(f"/leaderboard?x={i}&q=a{i}").status_code == 200
        assert client.get(f"/replays/hp_kinnauri-shawl-geometric-border-easy-8/beam-hf?try={10 + i}").status_code == 200
    assert len(pagecache.keys()) == before


def test_warm_up_yields_between_pages(monkeypatch):
    from pramaan_v2.site import pagecache, routes

    pauses = []
    monkeypatch.setattr(routes.time, "sleep", lambda seconds: pauses.append(seconds))
    pagecache.clear()
    count = routes.warm_pages("http://localhost")
    assert len(pauses) >= count - 1 and all(0 < p <= 0.05 for p in pauses)
