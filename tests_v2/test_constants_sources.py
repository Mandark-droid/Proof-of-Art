import json
from pathlib import Path

from pramaan.traditions import ALL_STATES


def test_state_codes_cover_every_state_once():
    from pramaan_v2.constants import STATE_CODES, STATE_ORDER

    assert STATE_ORDER == tuple(ALL_STATES)
    assert set(STATE_CODES) == set(ALL_STATES)
    assert len(set(STATE_CODES.values())) == 36
    assert STATE_CODES["Himachal Pradesh"] == "HP"
    assert STATE_CODES["Dadra and Nagar Haveli and Daman and Diu"] == "DD"
    assert STATE_CODES["Puducherry"] == "PY"


def test_grammar_order_and_labels():
    from pramaan_v2.constants import GRAMMAR_LABELS, GRAMMAR_ORDER

    assert GRAMMAR_ORDER == ("counted_grid", "stripe_band", "periodic_raster", "jaali_lattice",
                             "dot_field", "rotational_field", "kolam_loop")
    assert GRAMMAR_LABELS["periodic_raster"] == "Repeat tiles"
    assert GRAMMAR_LABELS["rotational_field"] == "Rotational motifs"
    assert set(GRAMMAR_LABELS) == set(GRAMMAR_ORDER)


def test_number_word():
    from pramaan_v2.constants import number_word

    assert number_word(1) == "one"
    assert number_word(11) == "eleven"
    assert number_word(20) == "twenty"
    assert number_word(21) == "21"


def test_load_published_reads_the_pinned_revision(published):
    pinned = json.loads((Path(__file__).resolve().parents[1] / "pramaan_v2/build/sources.json")
                        .read_text(encoding="utf-8"))
    assert published.revision == pinned["revision"]
    assert published.dataset == pinned["dataset"]
    assert len(published.rows) == 792
    assert len({row["model_key"] for row in published.rows}) == 11
    assert len({row["prompt_id"] for row in published.rows}) == 72
    assert "claude-fable-5-2" in published.summary["by_model"]


def test_load_published_rejects_rows_from_another_suite(tmp_path, published):
    import pytest
    from pramaan_v2.build.sources import load_published

    compact = tmp_path / "compact.jsonl"
    row = dict(published.rows[0], suite_id="eval-v2")
    compact.write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(published.summary), encoding="utf-8")
    with pytest.raises(ValueError, match="eval-v2.1"):
        load_published(compact=compact, summary=summary)
