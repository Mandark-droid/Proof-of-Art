"""Home page (`/`): structure and every number checked against the site data."""
from __future__ import annotations

import html as htmllib
import re

import pytest
from fastapi.testclient import TestClient

FEATURED = "hp_kinnauri-shawl-geometric-border-easy-8"
IMPOSSIBLE = "py_kolam-as-practised-in-puducherry-easy-35"
KOLAM = "tn_kolam-easy-22"


@pytest.fixture(scope="module")
def data():
    from pramaan_v2.site.data import site_data

    return site_data()


@pytest.fixture(scope="module")
def page():
    from pramaan_v2.server import app

    response = TestClient(app).get("/")
    assert response.status_code == 200
    return response.text


def _section(page: str, marker: str) -> str:
    match = re.search(r'<section[^>]*data-home="' + marker + r'"[^>]*>.*?</section>', page, re.S)
    assert match, f"section {marker} missing"
    return match.group(0)


def _visible_text(fragment: str) -> str:
    fragment = re.sub(r"<(script|style)\b.*?</\1>", " ", fragment, flags=re.S)
    return htmllib.unescape(re.sub(r"<[^>]+>", " ", fragment))


def _wall_anchors(page: str) -> list[tuple[str, str]]:
    wall = _section(page, "wall")
    return re.findall(r'<a class="hw-t[^"]*" href="([^"]+)"[^>]*aria-label="([^"]+)"', wall)


def _n(value: int) -> str:
    return f"{value:,}"


def test_one_h1_with_the_board_title(page):
    assert page.count("<h1") == 1
    assert re.search(r"<h1[^>]*>AI drew it\.<br>Math checked it\.</h1>", page)


def test_hero_eyebrow_and_featured_replay(page, data):
    totals = data.meta["totals"]
    eyebrow = (f"{totals['models']} MODELS · {totals['briefs']} BRIEFS · "
               f"{len(data.meta['states'])} STATES AND UNION TERRITORIES")
    assert eyebrow in page
    assert f'href="/replays/{FEATURED}/beam-hf"' in page


def test_wall_has_one_link_per_run(page, data):
    anchors = _wall_anchors(page)
    assert len(anchors) == data.meta["totals"]["runs"] == len(data.wall["runs"])
    runs = {f"/replays/{run['task']}/{run['model']}" for run in data.wall["runs"]}
    assert {href for href, _ in anchors} == runs


def test_wall_accessible_name_on_a_known_run(page, data):
    run = next(r for r in data.wall["runs"] if r["task"] == FEATURED and r["model"] == "beam-hf")
    label = next(m["label"] for m in data.models if m["key"] == "beam-hf")
    verdict = {"1": "passed on the first try", "2": "passed on try 2", "3": "passed on try 3",
               "x": "broke a rule on every try", "-": "never readable"}[run["status"]]
    expected = f"{label}, Himachal Pradesh, easy: {verdict}"
    assert (f"/replays/{FEATURED}/beam-hf", expected) in _wall_anchors(page)


def test_wall_row_labels_follow_the_model_order(page, data):
    wall = _section(page, "wall")
    labels = re.findall(r'<div class="hw-row"[^>]*><span>([^<]+)</span><span[^>]*>(\d+)</span>', wall)
    board = {m["key"]: m for m in data.leaderboard["models"]}
    assert labels == [(htmllib.escape(m["label"]), str(board[m["key"]]["within3"])) for m in data.models]


def test_default_inspector_is_beam_himachal_easy(page, data):
    wall = _section(page, "wall")
    inspector = wall[wall.index('class="hw-insp"'):]
    inspector = inspector[:inspector.index("</a>") + 4]
    assert "<strong" in inspector and ">Beam<" in inspector
    assert "Himachal Pradesh" in inspector and "Kinnauri shawl border" in inspector
    assert "PASSED ON TRY 3" in inspector
    assert f'href="/replays/{FEATURED}/beam-hf"' in inspector and "Open replay" in inspector


def test_legend_counts_come_from_meta(page, data):
    totals = data.meta["totals"]
    legend = _visible_text(_section(page, "legend"))
    assert f"{_n(totals['passed'])} passed every check" in legend
    assert f"{_n(totals['broke'])} drew a pattern that broke a rule" in legend
    assert f"{_n(totals['never'])} never produced a readable pattern" in legend


def test_four_numbers_come_from_meta(page, data):
    totals = data.meta["totals"]
    numbers = re.findall(r'<div class="hn-v"[^>]*>([^<]+)</div>', _section(page, "numbers"))
    assert numbers == [_n(totals["runs"]), _n(totals["proposals"]), _n(totals["pixel_evidence"]), "0"]
    text = _visible_text(_section(page, "numbers"))
    assert f"all {_n(totals['rendered'])} rendered proposals" in text


def test_kolam_chips_and_all_three_drawings_are_in_the_page(page, data):
    kolam = _section(page, "kolam")
    chips = re.findall(r'<button[^>]*aria-pressed="(true|false)"[^>]*>([^<]+)</button>', kolam)
    assert [label for _, label in chips] == ["Beam", "Claude Opus 5.5", "GPT-6 Astra"]
    assert [pressed for pressed, _ in chips] == ["true", "false", "false"]
    groups = re.findall(r'<div class="hk-tries"[^>]*data-model="([^"]+)"([^>]*)>', kolam)
    assert [key for key, _ in groups] == ["beam-hf", "claude-opus-5-5", "gpt-6-astra"]
    assert "hidden" not in groups[0][1] and all("hidden" in rest for _, rest in groups[1:])
    beam = data.kolam[KOLAM]["models"]["beam-hf"]
    for item in beam:
        for path in item["paths"]:
            assert path in kolam
    assert "gcd(rows, columns)" in kolam


def test_spot_the_break_is_revealed_without_js(page, data):
    spot = _section(page, "spot")
    replay = data.replay(FEATURED)
    first = next(m for m in replay["models"] if m["key"] == "beam-hf")["tries"][0]
    answer = re.search(r'<div class="hs-answer"([^>]*)>', spot).group(1)
    assert "hidden" not in answer
    assert f"{first['mismatch_px']} pixels say no." in spot
    x, y = first["pair"]["original"]
    tx, ty = first["pair"]["transformed"]
    assert f"({x}, {y}) ≠ ({tx}, {ty}) under a left-right flip" in htmllib.unescape(spot)
    assert "Looks symmetric" in spot and "Something is off" in spot and "Cover it again" in spot


def test_impossible_kolam_card_names_every_accepted_model(page, data):
    card = _visible_text(re.search(r'<article[^>]*data-card="kolam".*?</article>', page, re.S).group(0))
    accepted = [m["label"] for m in data.models
                if any(t["outcome"] == "passed" for t in data.kolam[IMPOSSIBLE]["models"].get(m["key"], []))]
    assert accepted
    for label in accepted:
        assert label in card
    failed = len(data.models) - len(accepted)
    from pramaan_v2.constants import number_word

    assert f"{number_word(failed).capitalize()} models failed." in card


def test_blank_tile_card_counts(page, data):
    card = _visible_text(re.search(r'<article[^>]*data-card="blank".*?</article>', page, re.S).group(0))
    totals = data.meta["totals"]
    assert f"{_n(totals['low_ink'])} of {_n(totals['passed'])} accepted designs" in card


def test_anchor_card_shows_the_top_three(page, data):
    card = re.search(r'<article[^>]*data-card="anchor".*?</article>', page, re.S).group(0)
    top = sorted(data.leaderboard["models"], key=lambda m: -m["anchor_only"])[:3]
    briefs = data.meta["totals"]["briefs"]
    for model in top:
        assert f"{model['anchor_only']}/{briefs}" in card and htmllib.escape(model["label"]) in card


def test_strip_has_one_row_per_model_in_order(page, data):
    strip = _section(page, "strip")
    rows = re.findall(r'<div class="hl-row"[^>]*><span[^>]*>(\d\d)</span><span[^>]*>([^<]+)</span>', strip)
    assert [label for _, label in rows] == [htmllib.escape(m["label"]) for m in data.models]
    assert [n for n, _ in rows] == [f"{i:02d}" for i in range(1, len(data.models) + 1)]
    assert 'href="/leaderboard"' in strip


def test_open_problems_numbers(page, data):
    build = _visible_text(_section(page, "build"))
    models = data.leaderboard["models"]
    jaali = sum(m["by_grammar"]["jaali_lattice"] for m in models)
    briefs = data.leaderboard["grammar_briefs"]["jaali_lattice"]
    assert f"{jaali} of {briefs * len(models)} model and brief pairs pass" in build
    for href in ('href="/build"', 'href="/build#contract"', 'href="/try"', 'href="/playground"'):
        assert href in _section(page, "build")


def test_footer_links(page, data):
    footer = re.search(r"<footer.*?</footer>", page, re.S).group(0)
    assert f"The {data.traditions['totals']['traditions']} traditions and their sources" in footer
    for href in ("/traditions", "https://huggingface.co/datasets/kshitijthakkar/pramaan-proof-of-art-eval-v2.1",
                 "https://huggingface.co/datasets/kshitijthakkar/pramaan-proof-of-art-train-v1",
                 "https://huggingface.co/datasets/kshitijthakkar/pramaan-proof-of-art-openenv",
                 "https://github.com/Mandark-droid/Proof-of-Art",
                 "https://huggingface.co/collections/kshitijthakkar/pramaan-proof-of-art"):
        assert f'href="{href}"' in footer
    assert f"Validator {data.meta['validator']} · suite {data.meta['suite_id']}" in footer


def test_no_v1_vocabulary_in_visible_text(page):
    # "counterexample" is allowed: the build teaser keeps the board copy (spec 3.1 allows it on builder text).
    text = _visible_text(page.split("<main>", 1)[1])
    labels = " ".join(re.findall(r'(?:aria-label|alt)="([^"]*)"', page))
    for blob in (text, labels):
        for word in ("Studio", "Arena", "Pass@1", "Pass@3"):
            assert not re.search(r"\b" + re.escape(word) + r"(?!\w)", blob), word
        for word in ("arm", "arms", "candidate", "witness", "quilt", "mosaic", "attempt", "attempts",
                     "task"):
            assert not re.search(r"\b" + word + r"\b", blob, re.I), word


def test_every_tile_image_has_a_text_alternative(page):
    for img in re.findall(r"<img\b[^>]*>", page):
        assert re.search(r'alt="[^"]+"', img) or 'aria-hidden="true"' in img or 'alt=""' in img, img
    for div in re.findall(r'<div[^>]*class="[^"]*hw-tile[^"]*"[^>]*>', page):
        assert 'role="img"' in div and "aria-label=" in div


def test_wall_sprite_is_cache_busted(page, data):
    assert f"/site/wall.png?v={data.meta['revision'][:10]}" in page


def test_home_script_and_style_fragments_are_bundled():
    from pramaan_v2.site.shell import bundle_css, bundle_js

    assert b"[data-home-wall]" in bundle_js() or b"data-home" in bundle_js()
    assert b".hw-t" in bundle_css()
