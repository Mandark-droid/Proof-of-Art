"""Link preview images (spec 11)."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

KINNAURI = "hp_kinnauri-shawl-geometric-border-easy-8"
PUDUCHERRY = "py_kolam-as-practised-in-puducherry-easy-35"


@pytest.fixture(scope="module")
def client():
    from pramaan_v2.server import app

    return TestClient(app)


def _png(response):
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    image = Image.open(io.BytesIO(response.content))
    assert image.format == "PNG"
    return image


@pytest.mark.parametrize("url", ["/og/home.png", f"/og/replay/{KINNAURI}/beam-hf.png",
                                 f"/og/replay/{PUDUCHERRY}/claude-opus-5-5.png"])
def test_link_images_are_1200_by_630(client, url):
    image = _png(client.get(url))
    assert image.size == (1200, 630)
    assert image.convert("RGB").getpixel((2, 2)) == (0x15, 0x11, 0x0F)


def test_unknown_run_is_404(client):
    assert client.get(f"/og/replay/{KINNAURI}/no-such-model.png").status_code == 404
    assert client.get("/og/replay/no-such-brief/beam-hf.png").status_code == 404


def test_images_are_cached(client):
    from pramaan_v2.site.render.og import og_home, og_replay

    assert og_home() is og_home()
    assert og_replay(KINNAURI, "beam-hf") is og_replay(KINNAURI, "beam-hf")
    first = client.get(f"/og/replay/{KINNAURI}/beam-hf.png").content
    assert client.get(f"/og/replay/{KINNAURI}/beam-hf.png").content == first


def test_never_readable_run_still_draws(client):
    from pramaan_v2.site.data import site_data

    data = site_data()
    run = next(r for r in data.wall["runs"] if r["status"] == "-")
    image = _png(client.get(f'/og/replay/{run["task"]}/{run["model"]}.png'))
    assert image.size == (1200, 630)


def test_featured_image_shows_the_evidence(client):
    image = _png(client.get(f"/og/replay/{KINNAURI}/beam-hf.png")).convert("RGB")
    colours = {image.getpixel((x, y)) for x in range(59, 571, 4) for y in range(59, 571, 4)}
    assert (0x4F, 0xD1, 0xC5) in colours  # evidence pixels in the overlay


def _replay_cache():
    from pramaan_v2.site.render import og

    return getattr(og, "_og_replay", og.og_replay)


def test_garbage_urls_neither_fill_nor_poison_the_cache(client):
    before = _replay_cache().cache_info()
    for i in range(20):
        assert client.get(f"/og/replay/junk-{i}/beam-hf.png").status_code == 404
        assert client.get(f"/og/replay/{KINNAURI}/junk-{i}.png").status_code == 404
    after = _replay_cache().cache_info()
    assert (after.currsize, after.misses) == (before.currsize, before.misses)


def test_replay_image_cache_holds_every_run():
    from pramaan_v2.site.data import site_data

    assert _replay_cache().cache_info().maxsize >= site_data().meta["totals"]["runs"]
