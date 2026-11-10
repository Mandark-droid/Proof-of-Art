"""Chart builders for the leaderboard: geometry, encodings and the rules in spec 6.5 and 9."""
from __future__ import annotations

import math
import re

import pytest

from pramaan_v2.site.data import site_data
from pramaan_v2.site.render import charts


@pytest.fixture(scope="module")
def rows():
    return site_data().leaderboard["models"]


def test_wilson_label_rounds_both_bounds():
    assert charts.wilson_label([0.8475202051632649, 0.9699746184342299]) == "85–97%"
    assert charts.wilson_label([0.047896319973655774, 0.1873504579512686]) == "5–19%"


def test_cost_axes_match_the_board():
    # Board: $0.10 at x=56, $1 at 282.1, $10 at 508.1; 72 briefs at y=16, 0 at y=336.
    assert charts.cost_x(0.1) == pytest.approx(56, abs=0.1)
    assert charts.cost_x(1) == pytest.approx(282.1, abs=0.1)
    assert charts.cost_x(7.06) == pytest.approx(474.0, abs=0.2)
    assert charts.cost_y(72, 72) == 16 and charts.cost_y(0, 72) == 336
    assert charts.cost_y(67, 72) == pytest.approx(38.2, abs=0.1)


def test_frontier_is_models_beating_every_cheaper_model(rows):
    keys = [m["key"] for m in charts.frontier(rows)]
    assert keys == [m["key"] for m in sorted(rows, key=lambda m: m["spend_usd"]) if m["on_frontier"]]
    best = -1
    for m in sorted(rows, key=lambda m: m["spend_usd"]):
        assert (m["within3"] > best) == (m["key"] in keys)
        best = max(best, m["within3"])


def test_cost_labels_do_not_overlap_each_other_or_points(rows):
    placed = charts.place_cost_labels(rows, 72)
    boxes = [p["box"] for p in placed]
    for i, a in enumerate(boxes):
        assert a[0] >= 0 and a[2] <= 640 and a[1] >= 0 and a[3] <= 348
        for b in boxes[i + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], (a, b)
        for m in rows:
            cx, cy = charts.cost_x(m["spend_usd"]), charts.cost_y(m["within3"], 72)
            near_x = min(max(cx, a[0]), a[2])
            near_y = min(max(cy, a[1]), a[3])
            assert math.hypot(cx - near_x, cy - near_y) >= 6, (m["key"], a)


@pytest.mark.parametrize("count,n", [(c, n) for n in (3, 4, 5, 11, 14, 31) for c in range(n + 1)])
def test_heat_cells_never_use_the_forbidden_alpha_band(count, n):
    style = charts.heat_style(count, n)
    if count == 0:
        assert "#1D1815" in style
        return
    alpha = float(re.search(r"rgba\(244, 182, 63, ([0-9.]+)\)", style).group(1))
    assert not (0.47 < alpha < 0.66)
    assert ("color: #15110F" in style) == (alpha >= 0.66)


def test_heat_alpha_matches_board_samples():
    # Values from the board: 31/31 -> 0.9, 1/5 -> 0.18, 2/3 -> 0.66, 7/14 -> 0.45, 26/31 -> 0.75.
    for count, n, alpha in ((31, 31, "0.9"), (1, 5, "0.18"), (2, 3, "0.66"), (7, 14, "0.45"), (26, 31, "0.75")):
        assert f"rgba(244, 182, 63, {alpha})" in charts.heat_style(count, n)


def test_radar_caption_rules():
    grammars = [("counted_grid", 31), ("stripe_band", 14), ("periodic_raster", 11), ("jaali_lattice", 5),
                ("dot_field", 4), ("rotational_field", 4), ("kolam_loop", 3)]
    full = dict(counted_grid=31, stripe_band=14, periodic_raster=11, jaali_lattice=1, dot_field=4,
                rotational_field=4, kolam_loop=2)
    assert charts.radar_caption(full, grammars) == "Lowest: jaali, 1 of 5."
    two = dict(full, jaali_lattice=0, kolam_loop=0)
    assert charts.radar_caption(two, grammars) == "No passes in jaali or kolam."
    three = dict(two, dot_field=0)
    assert charts.radar_caption(three, grammars) == "No passes in 3 of 7 grammars."


def test_small_radar_geometry_matches_reference():
    grammars = [("counted_grid", 31), ("stripe_band", 14), ("periodic_raster", 11), ("jaali_lattice", 5),
                ("dot_field", 4), ("rotational_field", 4), ("kolam_loop", 3)]
    model = {"label": "GPT-6 Astra", "by_grammar": dict(counted_grid=31, stripe_band=14, periodic_raster=11,
             jaali_lattice=1, dot_field=4, rotational_field=4, kolam_loop=2)}
    svg = charts.small_radar(model, grammars)
    # Points copied from the board's GPT-6 Astra card.
    assert 'points="140.0,46.0 185.3,67.8 196.5,116.9 145.0,114.5 114.8,156.3 83.5,116.9 109.8,79.9"' in svg
    assert "<title>Jaali lattice: 1 of 5</title>" in svg


def test_compare_radar_geometry_matches_reference(rows):
    astra = next(m for m in rows if m["key"] == "gpt-6-astra")
    svg = charts.compare_radar([astra])
    assert 'points="330.0,93.0 461.1,164.3 414.2,288.6 330.0,360.0 274.3,272.2 294.2,219.3"' in svg


def test_compare_values_clamp_to_the_rim():
    model = {"key": "x", "label": "X", "within3": 72, "readable": 72, "repaired": [0, 0], "first": {"passed": 72},
             "tokens_per_brief": 500, "latency_s": 1, "spend_usd": 0.01,
             "radar": {"passed": 1, "readable": 1, "repaired": 1.4, "tokens": 1.6, "speed": -0.2, "spend": 2}}
    assert charts.radar_values(model) == [1, 1, 1, 1, 0, 1]


def test_first_try_bar_segments_sum_to_one_hundred(rows):
    for m in rows:
        widths = [float(w) for w in re.findall(r"width: ([0-9.]+)%", charts.first_try_bar(m))]
        assert len(widths) == 3 and sum(widths) == pytest.approx(100, abs=0.011)


def test_request_settings_wording():
    assert charts.request_settings({"max_tokens": 4096, "output_config": {"effort": "low"}}) == "output effort low"
    assert charts.request_settings({"max_tokens": 4096, "reasoning_effort": "minimal", "temperature": 0.4}) == \
        "temperature 0.4 · reasoning_effort minimal"
    assert charts.request_settings({"extra_body": {"thinking": {"type": "disabled"}}, "max_tokens": 4096,
                                    "temperature": 0.4}) == "temperature 0.4 · thinking disabled"


def test_cost_grid_ticks_follow_the_brief_count():
    svg = charts.cost_scatter([], 40)
    ticks = re.findall(r'<text x="48" y="([0-9.]+)" text-anchor="end" font-size="11" fill="#948980">(\d+)</text>', svg)
    assert [t for _, t in ticks] == ["40", "30", "20", "10", "0"]
    assert float(ticks[0][0]) == charts.cost_y(40, 40) + 4 == 20
    assert charts.cost_y(0, 40) == 336


def test_shares_use_each_models_own_brief_count():
    model = {"key": "x", "label": "X", "n": 50, "within3": 25, "readable": 40, "repaired": [1, 2],
             "first": {"passed": 10, "broke": 20, "unreadable": 20}, "wilson95": [0.36, 0.64],
             "tokens_per_brief": None, "latency_s": None, "spend_usd": None, "radar": {}}
    row = charts.ranking_row(model, None, hidden=False)
    assert "25 of 50" in row and "width: 50.00%" in row and ">–</span>" in row
    assert charts.radar_display(model) == ["25 of 50", "80.0%", "1 of 2", "–", "–", "–"]
    assert charts.first_try_split(model) == [20.0, 40.0, 40.0]
    assert "0 of 40" in charts.spoke_table(40) and "40 of 40" in charts.spoke_table(40)


def test_leaderboard_script_has_no_literal_brief_count():
    from pathlib import Path

    js = (Path(charts.__file__).resolve().parents[2] / "static" / "js" / "40-leaderboard.js").read_text(encoding="utf-8")
    assert not re.search(r"\b72\b", js)
    assert "data-briefs" in js or "dataset.briefs" in js
