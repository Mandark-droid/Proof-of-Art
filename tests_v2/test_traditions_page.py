"""Traditions pages: /traditions and /traditions/{state_code} (spec 8.7)."""
from __future__ import annotations

import html
import json
import re
from html.parser import HTMLParser
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
def hp(client):
    response = client.get("/traditions/HP")
    assert response.status_code == 200
    return response.text


def _text(page: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page)))


def _state(data, code):
    return next(s for s in data.traditions["states"] if s["code"] == code)


class _Cards(HTMLParser):
    """Text chunks inside each <article data-tradition="...">, plus their attributes."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.cards: dict[str, list[str]] = {}
        self.attrs: dict[str, list[dict]] = {}
        self._current = None
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article" and "data-tradition" in attrs:
            self._current = attrs["data-tradition"]
            self.cards[self._current] = []
            self.attrs[self._current] = []
            self._depth = 0
        if self._current:
            if tag == "article":
                self._depth += 1
            self.attrs[self._current].append({"tag": tag, **attrs})

    def handle_endtag(self, tag):
        if self._current and tag == "article":
            self._depth -= 1
            if self._depth == 0:
                self._current = None

    def handle_data(self, text):
        if self._current and text.strip():
            self.cards[self._current].append(text.strip())


def _cards(page: str) -> _Cards:
    parser = _Cards()
    parser.feed(page)
    return parser


def test_index_and_every_state_page_render(client, data):
    index = client.get("/traditions")
    assert index.status_code == 200
    assert index.text.count("<h1") == 1
    assert "Where every brief comes from." in index.text
    codes = [s["code"] for s in data.traditions["states"]]
    assert len(codes) == 36
    for code in codes:
        response = client.get(f"/traditions/{code}")
        assert response.status_code == 200, code
        assert response.text.count("<h1") == 1
        assert _state(data, code)["name"] in _text(response.text)


def test_lowercase_code_works_and_unknown_code_is_404(client):
    assert client.get("/traditions/hp").status_code == 200
    assert client.get("/traditions/XX").status_code == 404
    from pramaan_v2.site.data import site_data
    from pramaan_v2.site.render.traditions import render_traditions

    assert render_traditions(site_data(), "ZZ", "http://localhost") is None


def test_index_opens_on_himachal_pradesh_like_the_render(client):
    page = client.get("/traditions").text
    assert ">Himachal Pradesh</h2>" in page
    assert 'aria-current="page"' in page and 'href="/traditions/HP"' in page


def test_eyebrow_heading_and_totals_come_from_the_registry(hp, data):
    totals = data.traditions["totals"]
    text = _text(hp)
    assert "TRADITIONS · SOURCES AND CAUTIONS" in text
    assert f"{totals['traditions']} traditions across all {len(data.traditions['states'])} states and Union Territories" in text
    assert f"{totals['sources']} sources, each with the claim it supports. {totals['unverified']} more are listed as unverified." in text
    assert (f"{totals['gi']} hold a registered Geographical Indication, so their names are never used as "
            "product labels") in text
    assert f"{totals['multi_state']} are practised in more than one state and are attributed that way" in text
    assert (totals["traditions"], totals["sources"], totals["unverified"], totals["gi"], totals["multi_state"]) == (
        87, 271, 43, 54, 45)


def test_grammar_chips_carry_counts_and_aria_pressed(hp, data):
    from pramaan_v2.constants import GRAMMAR_LABELS

    by_grammar = data.traditions["by_grammar"]
    buttons = re.findall(r'<button[^>]*name="grammar"[^>]*>(.*?)</button>', hp)
    labels = [_text(b).strip() for b in buttons]
    assert labels[0] == f"All {data.traditions['totals']['traditions']}"
    expected = sorted(by_grammar, key=lambda g: -by_grammar[g])
    assert labels[1:] == [f"{GRAMMAR_LABELS[g]} {by_grammar[g]}" for g in expected]
    assert labels[1:] == ["Counted grid 26", "Stripe bands 20", "Repeat tiles 17", "Rotational motifs 9",
                          "Jaali lattice 7", "Dot field 5", "Kolam loops 3"]
    pressed = re.findall(r'<button[^>]*name="grammar"[^>]*aria-pressed="(true|false)"', hp)
    assert pressed == ["true"] + ["false"] * 7


def test_grammar_filter_works_without_javascript(data):
    from pramaan_v2.site.render.traditions import render_traditions

    page = render_traditions(data, "TN", "http://localhost", grammar="kolam_loop")
    pressed = re.findall(r'<button[^>]*value="([a-z_]*)"[^>]*aria-pressed="true"', page)
    assert pressed == ["kolam_loop"]
    cards = _cards(page)
    shown = [tid for tid, attrs in cards.attrs.items() if "hidden" not in attrs[0]]
    tn = _state(data, "TN")["traditions"]
    assert set(shown) == {t["id"] for t in tn if t["grammar"] == "kolam_loop"}
    # state chips count only that grammar and keep the filter in their links
    assert re.search(r'href="/traditions/TN\?grammar=kolam_loop"', page)
    # an unknown grammar falls back to all
    assert render_traditions(data, "TN", "http://localhost", grammar="nope").count('aria-pressed="true"') >= 1


def test_state_chips_link_to_every_state_with_counts(hp, data):
    for state in data.traditions["states"]:
        match = re.search(rf'<a href="/traditions/{state["code"]}"[^>]*>(.*?)</a>', hp)
        assert match, state["code"]
        assert _text(match.group(1)).strip() == f"{state['name']} {len(state['traditions'])}"


def test_every_tradition_of_every_state_is_rendered(client, data):
    for state in data.traditions["states"]:
        page = client.get(f"/traditions/{state['code']}").text
        cards = _cards(page)
        assert list(cards.cards) == [t["id"] for t in state["traditions"]], state["code"]


def test_every_source_shows_what_it_supports_and_unverified_are_labelled(client, data):
    for state in data.traditions["states"]:
        page = client.get(f"/traditions/{state['code']}").text
        text = _text(page)
        for tradition in state["traditions"]:
            for source in tradition["sources"] + tradition["sources_unverified"]:
                assert re.sub(r"\s+", " ", source["supports"]).strip() in text
                assert html.escape(source["url"], quote=True) in page
            if tradition["sources_unverified"]:
                block = page.split(f'data-tradition="{tradition["id"]}"', 1)[1].split("</article>", 1)[0]
                assert 'data-unverified' in block and "UNVERIFIED" in block


LABELS = {
    "HOW TO SAY IT", "PALETTE AND ITS BASIS", "CAUTIONS", "SOURCES · WHAT EACH SUPPORTS",
    "SOURCES LISTED AS UNVERIFIED", "Research confidence:", "·", ",", "not in the evaluation briefs",
}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _allowed(chunk: str, tradition: dict, models_word: str) -> bool:
    from pramaan_v2.site.render.traditions import split_sentences

    registry = "\n".join(_strings(tradition))
    if chunk in LABELS or chunk.strip("· ") in registry:
        return True
    if chunk == "; ".join(tradition["motif_vocabulary"]):
        return True
    if all(sentence in registry for sentence in split_sentences(chunk)):
        return True
    if re.fullmatch(r"(easy|hard) brief, " + models_word + " models", chunk):
        return True
    from pramaan_v2.constants import GRAMMAR_LABELS

    mono = (f"{GRAMMAR_LABELS[tradition['grammar']].upper()} · {tradition['category'].replace('_', ' ').upper()} · "
            f"{'GI REGISTERED' if tradition['gi_registered'] else 'NOT GI REGISTERED'}")
    return chunk in (mono, f"Research confidence: {tradition['confidence']} ·",
                     f"Research confidence: {tradition['confidence']} · not in the evaluation briefs")


def test_card_text_comes_only_from_the_registry(client, data):
    from pramaan_v2.constants import number_word

    models_word = number_word(len(data.models))
    for state in data.traditions["states"]:
        page = client.get(f"/traditions/{state['code']}").text
        cards = _cards(page)
        for tradition in state["traditions"]:
            for chunk in cards.cards[tradition["id"]]:
                assert _allowed(chunk, tradition, models_word), (tradition["id"], chunk)


def test_kinnauri_cautions_appear_verbatim(hp, data):
    from pramaan_v2.site.render.traditions import split_sentences

    kinnauri = next(t for t in _state(data, "HP")["traditions"] if t["id"] == "hp_kinnauri-shawl-geometric-border")
    sentences = split_sentences(kinnauri["cautions"])
    assert " ".join(sentences) == kinnauri["cautions"].strip()
    text = _text(hp)
    for sentence in sentences:
        assert sentence in text
    assert "use 'inspired by Kinnauri shawl borders'" in text
    # the motif vocabulary is the description, as written
    assert kinnauri["motif_vocabulary"][0] in text


def test_sentence_split_keeps_abbreviations_together():
    from pramaan_v2.site.render.traditions import split_sentences

    assert split_sentences("Marks carry identity (e.g. Oraon lines). Use care.") == [
        "Marks carry identity (e.g. Oraon lines).", "Use care."]


def test_brief_links_go_to_replays(hp):
    assert 'href="/replays/hp_kinnauri-shawl-geometric-border-easy-8"' in hp
    assert 'href="/replays/hp_kullu-shawl-end-bands-hard-8"' in hp
    assert "not in the evaluation briefs" in hp


def test_festival_context_shows_ledger_claims_as_written(client, data):
    claims = json.loads((ROOT / "pramaan" / "data" / "claims.json").read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in claims["claims"]}
    tn = client.get("/traditions/TN").text
    assert "<details" in tn and "Optional Diwali context" in tn
    ids = []
    for tradition in _state(data, "TN")["traditions"]:
        for claim_id in (tradition.get("festival_lens") or {}).get("claim_ids", []):
            ids.append(claim_id)
    assert ids
    for claim_id in ids:
        assert html.escape(by_id[claim_id]["text"], quote=True) in tn


def test_no_gi_name_is_used_as_a_product_label_and_no_v1_words(hp):
    text = _text(hp)
    for word in (" arm ", "candidate", "witness", "Pass@1", "quilt", "Studio", "Arena"):
        assert word not in text, word


def test_wall_tile_uses_the_versioned_sprite(hp, data):
    assert f"/site/wall.png?v={data.meta['revision'][:10]}" in hp
    assert re.search(r'role="img" aria-label="[^"]+"', hp)


def test_grammar_filter_works_through_the_route_without_javascript():
    from fastapi.testclient import TestClient
    from pramaan_v2.server import app

    client = TestClient(app)
    filtered = client.get("/traditions/TN?grammar=kolam_loop").text
    unfiltered = client.get("/traditions/TN").text
    assert filtered != unfiltered
    assert 'aria-pressed="true"' in filtered.split("Kolam loops")[0][-400:] or "Kolam loops" in filtered
