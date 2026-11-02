import re

from pramaan.traditions import ALL_STATES, GRAMMARS, load_registry, traditions_for_state


def test_every_state_has_a_tradition():
    registry = load_registry()
    assert len(ALL_STATES) == 36
    assert all(traditions_for_state(state) for state in ALL_STATES)
    assert len(registry) > 0


def test_every_tradition_has_grammar_palette_and_source():
    for tradition in load_registry().values():
        assert tradition.grammar in GRAMMARS
        assert len(tradition.palette_presets["traditional"]) >= 3
        assert any(source.url.startswith("https://") for source in tradition.sources)


def test_state_selector_only_lists_traditions_owned_by_that_state():
    choices = traditions_for_state("Gujarat")
    assert choices
    assert all(item.primary_state == "Gujarat" for item in choices)
    assert all(item.id != "up_banarasi-butidar-buti-and-jaal-brocade-layouts" for item in choices)


def test_registry_ids_unique_and_slug_like():
    ids = list(load_registry())
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-z]{2}_[a-z0-9]+(?:-[a-z0-9]+)*", item) for item in ids)


def test_research_exclusions_are_not_in_registry():
    names = [entry.name for entry in load_registry().values()]
    assert not any("Nek Chand" in name for name in names)
    assert not any(name == "Rangoli" for name in names)
    assert not any("generic coastal" in name.lower() for name in names)

def test_citation_index_renders_only_verified_registry_sources(tmp_path):
    import json

    from scripts.render_citations import render_source_index

    registry = {
        "entries": [{
            "id": "tn_test",
            "name": "Test tradition",
            "states": ["Tamil Nadu"],
            "sources": [
                {"title": "Verified source", "url": "https://example.org/verified", "publisher": "Example", "verified": True},
                {"title": "Curated verified source", "url": "https://example.org/curated", "publisher": "Example"},
                {"title": "Unverified source", "url": "https://example.org/unverified", "publisher": "Example", "verified": False},
            ],
        }],
    }
    path = tmp_path / "traditions.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    rendered = render_source_index(path)
    assert "https://example.org/verified" in rendered
    assert "https://example.org/curated" in rendered
    assert "https://example.org/unverified" not in rendered
    assert "Tamil Nadu" in rendered


def test_festival_lens_registry_metadata_is_typed_and_scoped():
    registry = load_registry()
    tn_kolam = registry["tn_kolam"].festival_lens
    puducherry_kolam = registry["py_kolam-as-practised-in-puducherry"].festival_lens
    aipan = registry["uk_aipan"].festival_lens
    pookalam = registry["kl_pookalam"].festival_lens
    sohrai = registry["jh_sohrai-khovar-wall-painting"].festival_lens

    assert tn_kolam.surface == "floor"
    assert tn_kolam.diwali_status == "sourced_institutional"
    assert "c_kolam_grid" in tn_kolam.claim_ids
    assert puducherry_kolam.diwali_status == "sourced_institutional"
    assert aipan.diwali_status == "sourced_institutional"
    assert pookalam.diwali_status == "none"
    assert pookalam.claim_ids == []
    assert sohrai.surface == "wall"
    assert sohrai.diwali_status == "sourced_secondary"
    assert sohrai.claim_ids == ["c_sohrai_diwali"]
