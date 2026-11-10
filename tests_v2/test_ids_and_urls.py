"""Model keys and brief ids in URLs: quoted when linked, and restricted to a safe alphabet at build time."""
from __future__ import annotations

import pytest


def test_run_url_quotes_both_segments():
    from pramaan_v2.site.render.replays import run_url

    assert run_url("tn_kolam-easy-22", "beam-hf") == "/replays/tn_kolam-easy-22/beam-hf"
    assert run_url("a b", 'x/y?"<') == "/replays/a%20b/x%2Fy%3F%22%3C"


def test_matrix_links_go_through_run_url():
    from pramaan_v2.site.render import charts

    out = charts.matrix([{"key": "m/x", "label": "M"}], [{"grammar": "counted_grid", "tasks": ["t 1"]}],
                        [{"model": "m/x", "task": "t 1", "status": "1", "low_ink": False, "state": "TN",
                          "level": "easy"}], {"TN": "Tamil Nadu"})
    assert 'href="/replays/t%201/m%2Fx"' in out


def test_home_links_go_through_run_url():
    import inspect

    from pramaan_v2.site.render import home

    source = inspect.getsource(home)
    assert 'href="/replays/{' not in source
    assert '+ (f"/{accepted' not in source and 'href += f"/{key}"' not in source


def _row(published, **change):
    row = dict(published.rows[0])
    row.update(change)
    return row


@pytest.mark.parametrize("field,value", [("model_key", "bad key"), ("model_key", "../x"), ("model_key", "-x"),
                                         ("prompt_id", "tn kolam"), ("prompt_id", "a/b")])
def test_builder_rejects_unsafe_ids(tmp_path, published, field, value):
    from pramaan_v2.build.site_data import build
    from pramaan_v2.build.sources import Published

    bad = Published(rows=[_row(published, **{field: value})] + published.rows[1:], summary=published.summary,
                    dataset=published.dataset, revision=published.revision)
    with pytest.raises(ValueError, match=r"\[A-Za-z0-9\]"):
        build(tmp_path, bad)
    assert not (tmp_path / "meta.json").exists()


def test_source_reader_rejects_unsafe_ids(tmp_path, published):
    import json

    from pramaan_v2.build.sources import _read_rows

    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(_row(published, model_key="a b")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model_key"):
        _read_rows(path)


@pytest.mark.parametrize("raw", ["%C2%B2", "%E2%91%A0", "①", "9" * 5000, "-1", "1e3"])
def test_try_query_falls_back_to_the_first_try(raw):
    import re

    from fastapi.testclient import TestClient

    from pramaan_v2.server import app

    response = TestClient(app, raise_server_exceptions=False).get(
        f"/replays/hp_kinnauri-shawl-geometric-border-easy-8/beam-hf?try={raw}")
    assert response.status_code == 200
    copy = re.search(r'<a href="([^"]+)" data-rv-copy', response.text).group(1)
    assert copy.endswith("?try=1")


def test_run_route_has_no_unused_try_parameter():
    import inspect

    from pramaan_v2.site import routes

    assert "try_" not in inspect.signature(routes.run_page).parameters
