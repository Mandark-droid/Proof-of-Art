import re

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from pramaan_v2.server import app

    return TestClient(app)


def test_static_css_is_served_with_immutable_cache(client):
    response = client.get("/static/v2/v2.css")
    assert response.status_code == 200
    assert "immutable" in response.headers["cache-control"]
    assert "--turmeric: #F4B63F" in response.text
    assert "font-display: swap" in response.text


def test_fonts_are_self_hosted(client):
    js = client.get("/static/v2/v2.js")
    assert js.status_code == 200 and "immutable" in js.headers["cache-control"]
    assert client.get("/static/v2/fonts/anek-latin-latin-wdth-normal.woff2").status_code == 200


def test_unknown_page_is_the_v2_404(client):
    response = client.get("/no-such-page")
    assert response.status_code == 404
    assert "That page does not exist." in response.text
    assert 'href="/"' in response.text and 'href="/replays"' in response.text
    assert response.text.count("<h1") == 1


def test_healthz_reports_the_data_revision(client):
    body = client.get("/healthz").json()
    assert body["ok"] is True and len(body["revision"]) == 40


def test_header_has_four_primary_links_and_submit():
    from pramaan_v2.site.shell import header

    html = header("replays")
    nav = re.search(r'<nav aria-label="Primary".*?</nav>', html, re.S).group(0)
    assert re.findall(r">([^<>]+)</a>", nav) == ["Replays", "Leaderboard", "Try it", "Build with it"]
    assert nav.count('aria-current="page"') == 1
    assert re.search(r'aria-current="page"[^>]*>Replays<', nav)
    assert "Submit a run</a>" in html
    assert "प्रमाण" in html and 'lang="hi"' in html


def test_page_carries_link_preview_metadata():
    from pramaan_v2.site.shell import page

    html = page(title="T", description="D", path="/replays", body="<h1>x</h1>", footer="",
                current="replays", base_url="https://example.hf.space", og_image="/og/home.png")
    assert '<meta property="og:title" content="T">' in html
    assert '<meta property="og:image" content="https://example.hf.space/og/home.png">' in html
    assert '<meta name="twitter:card" content="summary_large_image">' in html
    assert '<link rel="canonical" href="https://example.hf.space/replays">' in html
    assert html.startswith("<!doctype html>")


def test_shell_escapes_model_labels():
    from pramaan_v2.site.shell import esc, page

    label = 'A<b>&"c"'
    html = page(title=label, description=label, path="/", body=f"<h1>{esc(label)}</h1>", footer="",
                current=None, base_url="https://example.hf.space")
    assert "<b>" not in html
    assert "A&lt;b&gt;&amp;&quot;c&quot;" in html


def test_public_base_url_comes_from_configuration_not_headers(monkeypatch):
    from starlette.requests import Request
    from pramaan_v2.site.routes import base_url

    def request(headers):
        raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
        return Request({"type": "http", "scheme": "http", "server": ("localhost", 8000), "path": "/",
                        "headers": raw, "query_string": b""})

    hostile = {"host": "evil.example", "x-forwarded-host": "evil.example", "x-forwarded-proto": "javascript"}
    monkeypatch.setenv("SPACE_HOST", "kshitijthakkar-pramaan-proof-of-art-v2.hf.space")
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    assert base_url(request(hostile)) == "https://kshitijthakkar-pramaan-proof-of-art-v2.hf.space"
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://pramaan.example/")
    assert base_url(request(hostile)) == "https://pramaan.example"
    monkeypatch.delenv("PUBLIC_BASE_URL")
    monkeypatch.delenv("SPACE_HOST")
    loopback = ".".join(["127", "0", "0", "1"]) + ":8765"
    assert base_url(request({"host": loopback})) == "http://" + loopback
    assert base_url(request({"host": "localhost:8765"})) == "http://localhost:8765"
    assert base_url(request({"host": 'evil"><script>'})) == "http://localhost"
    assert base_url(request({"host": "evil.example", "x-forwarded-proto": "javascript"})) == "http://localhost"


def test_wall_sprite_is_served_with_immutable_cache(client):
    response = client.get("/site/wall.png")
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    assert "immutable" in response.headers["cache-control"]


V1_SPACE = "https://kshitijthakkar-pramaan-proof-of-art.hf.space"
V2_SPACE = "https://kshitijthakkar-pramaan-proof-of-art-v2.hf.space"


def _home_snippet(base: str = "http://testserver") -> str:
    import html
    import re

    from pramaan_v2.site.data import site_data
    from pramaan_v2.site.render.home import render_home

    page = render_home(site_data(), base)
    return html.unescape(re.search(r"<pre[^>]*>(from pramaan_env import.*?)</pre>", page, re.S).group(1))


def test_code_snippets_point_at_the_v2_space_by_default(monkeypatch):
    from pramaan_v2.apps.playground_render import client_snippet, public_base

    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("SPACE_HOST", raising=False)
    assert public_base() == V2_SPACE
    assert f'url = "{V2_SPACE}"' in _home_snippet() and V1_SPACE + '"' not in _home_snippet()
    assert f'url = "{V2_SPACE}"' in client_snippet("tn_kolam-easy-22", public_base())


def test_code_snippets_follow_the_configured_base(monkeypatch):
    from pramaan_v2.apps.playground_render import public_base

    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.setenv("SPACE_HOST", "someone-pramaan.hf.space")
    assert public_base() == "https://someone-pramaan.hf.space"
    assert 'url = "https://someone-pramaan.hf.space"' in _home_snippet()
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://pramaan.example/")
    assert public_base() == "https://pramaan.example"
    assert 'url = "https://pramaan.example"' in _home_snippet()
