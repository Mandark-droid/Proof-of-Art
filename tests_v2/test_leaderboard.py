"""The /leaderboard page: every count, name and mark comes from the site data."""
from __future__ import annotations

import html
import re

import pytest
from fastapi.testclient import TestClient

from pramaan_v2.constants import number_word
from pramaan_v2.site.data import site_data


@pytest.fixture(scope="module")
def client():
    from pramaan_v2.server import app

    return TestClient(app)


@pytest.fixture(scope="module")
def data():
    return site_data()


@pytest.fixture(scope="module")
def page(client):
    response = client.get("/leaderboard")
    assert response.status_code == 200
    return response.text


def section(text: str, anchor: str) -> str:
    """The markup between an element with this id and the next leaderboard section marker."""
    match = re.search(r'id="%s"(.*?)<!-- /%s -->' % (re.escape(anchor), re.escape(anchor)), text, re.S)
    assert match, f"section {anchor} not found"
    return match.group(1)


def visible_rows(fragment: str, cls: str) -> list[str]:
    rows = re.findall(r'<(?:div|tr) class="%s"([^>]*)>' % cls, fragment)
    return [re.search(r'data-model="([^"]+)"', attrs).group(1) for attrs in rows if " hidden" not in attrs]


def by_key(data):
    return {m["key"]: m for m in data.leaderboard["models"]}


def test_one_h1_with_the_number_word(page, data):
    assert page.count("<h1") == 1
    n = len(data.models)
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S).group(1)
    assert h1 == f"{number_word(n).capitalize()} models. One checker. Three tries."
    assert "LEADERBOARD · SUITE EVAL-V2.1 · VALIDATOR 0.3.0" in page


def test_section_nav_matches_the_board(page):
    nav = re.search(r'<nav aria-label="Leaderboard sections".*?</nav>', page, re.S).group(0)
    hrefs = re.findall(r'href="([^"]+)"', nav)
    assert hrefs == ["#ranking", "#results", "#matrix", "#failing", "#cost", "#grammar", "#profiles",
                     "/replays", "#method", "#community", "#gaps"]
    for anchor in ("ranking", "results", "matrix", "failing", "cost", "grammar", "profiles", "method",
                   "community", "gaps"):
        assert f'id="{anchor}"' in page


def test_ranking_lists_every_model_in_leaderboard_order(page, data):
    ranking = section(page, "ranking")
    assert visible_rows(ranking, "lb-rank-row") == [m["key"] for m in data.models]
    for m in data.leaderboard["models"]:
        assert html.escape(m["label"]) in ranking


def test_ranking_labels_carry_count_and_wilson_interval(page, data):
    ranking = section(page, "ranking")
    astra = by_key(data)["gpt-6-astra"]
    lo, hi = astra["wilson95"]
    assert f"{astra['within3']} of 72" in ranking
    assert f"{int(lo * 100 + 0.5)}–{int(hi * 100 + 0.5)}%" in ranking
    assert "67 of 72" in ranking and "85–97%" in ranking
    assert "left: 84.8%" in ranking  # whisker starts at the lower bound


def test_ranking_caveat_names_every_model_overlapping_the_leader(page, data):
    rows = data.leaderboard["models"]
    leader_lo = rows[0]["wilson95"][0]
    overlapping = [m["label"] for m in rows if m["wilson95"][1] >= leader_lo]
    caveat = re.search(r'<p class="lb-caveat"[^>]*>(.*?)</p>', page, re.S).group(1)
    assert f"top {number_word(len(overlapping))} intervals overlap" in caveat
    for label in overlapping:
        assert html.escape(label) in caveat
    for m in rows[len(overlapping):]:
        assert html.escape(m["label"]) not in caveat


def test_results_table_has_every_model_and_three_way_first_try(page, data):
    results = section(page, "results")
    assert visible_rows(results, "lb-res-row") == [m["key"] for m in data.models]
    for m in data.leaderboard["models"]:
        f = m["first"]
        assert f"{f['passed']} passed · {f['broke']} broke a rule · {f['unreadable']} unreadable" in results
        assert f"{m['passed_of_readable'][0]} of {m['passed_of_readable'][1]}" in results
        assert f"{m['repaired'][0]} of {m['repaired'][1]}" in results
        assert f"${m['spend_usd']:.2f}" in results


def test_filter_by_query_keeps_only_matching_rows(client, data):
    text = client.get("/leaderboard", params={"q": "gpt"}).text
    gpt = [m["key"] for m in data.models if "gpt" in m["label"].lower() or "gpt" in m["provider"].lower()]
    assert len(gpt) == 4
    assert visible_rows(section(text, "ranking"), "lb-rank-row") == gpt
    assert visible_rows(section(text, "results"), "lb-res-row") == gpt
    assert f"4 of {len(data.models)} model rows shown" in text
    assert 'value="gpt"' in text
    # Filtering never changes a number.
    assert "67 of 72" in section(text, "ranking")


def test_sort_by_cost_reorders_rows_without_changing_ranks(client, data):
    text = client.get("/leaderboard", params={"sort": "cost"}).text
    rows = sorted(data.leaderboard["models"], key=lambda m: m["spend_usd"])
    assert visible_rows(section(text, "ranking"), "lb-rank-row") == [m["key"] for m in rows]
    assert re.search(r'<option value="cost" selected', text)
    luna = re.search(r'data-model="gpt-6-luna"[^>]*><span[^>]*>(\d\d)</span>', section(text, "ranking"))
    rank = next(m["rank"] for m in data.models if m["key"] == "gpt-6-luna")
    assert luna.group(1) == "%02d" % rank


def test_provenance_community_hides_canonical_rows(client, data):
    text = client.get("/leaderboard", params={"prov": "community"}).text
    assert visible_rows(section(text, "ranking"), "lb-rank-row") == []
    assert f"0 of {len(data.models)} model rows shown" in text


def test_default_count_reads_all_rows(page, data):
    n = len(data.models)
    assert f"{n} of {n} model rows shown" in page


def test_matrix_links_every_run(page, data):
    matrix = section(page, "matrix")
    cells = re.findall(r'<a class="lb-c ([^"]*)" href="([^"]+)" title="([^"]+)"', matrix)
    assert len(cells) == len(data.models) * 72
    runs = {(r["task"], r["model"]): r for r in data.wall["runs"]}
    for classes, href, title in cells:
        _, _, task, model = href.split("/")
        run = runs[(task, model)]
        status = {"x": "sx", "-": "sn"}.get(run["status"], "s" + run["status"])
        assert status in classes.split()
        assert ("ink" in classes.split()) == bool(run["low_ink"])
    notches = sum("ink" in c.split() for c, _, _ in cells)
    assert notches == data.meta["totals"]["low_ink"]
    assert f"({data.meta['totals']['low_ink']} cells)" in matrix


def test_matrix_columns_follow_grammar_groups(page, data):
    matrix = section(page, "matrix")
    first_row = re.search(r'data-model="gpt-6-astra"(.*?)</div></div></div>', matrix, re.S).group(1)
    tasks = re.findall(r'href="/replays/([^/]+)/gpt-6-astra"', first_row)
    expected = [t for g in data.leaderboard["matrix"]["groups"] for t in g["tasks"]]
    assert tasks == expected


def test_matrix_cell_title_reads_like_the_spec(page):
    matrix = section(page, "matrix")
    assert re.search(r'title="GPT-6 Astra · [A-Z][A-Za-z ]+ (easy|hard) · passed on try 1 · low ink"', matrix)
    assert "broke a rule" in matrix and "never readable" in matrix


def test_failing_checks_counts_come_from_the_data(page, data):
    failing = section(page, "failing")
    rows = sorted(data.leaderboard["failing_first_try"], key=lambda r: -r["count"])
    bars = re.findall(r'<div class="lb-fail-row"[^>]*data-count="(\d+)"', failing)
    top, rest = rows[:11], rows[11:]
    assert [int(b) for b in bars] == [r["count"] for r in top] + [sum(r["count"] for r in rest)]
    assert f"{number_word(len(rest)).capitalize()} other checks" in failing
    runs = data.meta["totals"]["runs"]
    parse = next(r["count"] for r in rows if r["check"] == "candidate_parse")
    assert f"An unreadable proposal is the largest single failure: {parse} of {runs} first tries." in failing
    assert f"Counted over the {runs} first proposals." in failing


def test_cost_frontier_passes_exactly_through_frontier_models(page, data):
    from pramaan_v2.site.render.charts import cost_x, cost_y

    cost = section(page, "cost")
    points = re.search(r'<polyline class="lb-frontier" points="([^"]+)"', cost).group(1)
    frontier = sorted((m for m in data.leaderboard["models"] if m["on_frontier"]), key=lambda m: m["spend_usd"])
    expected = " ".join("%.1f,%.1f" % (cost_x(m["spend_usd"]), cost_y(m["within3"], data.meta["totals"]["briefs"])) for m in frontier)
    assert points == expected
    for m in data.leaderboard["models"]:
        assert f'data-model="{m["key"]}"' in cost
    for tick in ("$0.10", "$0.30", "$1", "$3", "$10", "$30"):
        assert f">{tick}</text>" in cost


def test_cost_callout_is_generated(page, data):
    rows = data.leaderboard["models"]
    frontier = sorted((m for m in rows if m["on_frontier"]), key=lambda m: -m["within3"])
    top, second = frontier[0], frontier[1]
    cost = section(page, "cost")
    assert (f"{html.escape(second['label'])} passes {second['within3']} of 72 for an estimated "
            f"${second['spend_usd']:.2f}.") in cost


def test_heatmap_cells_equal_by_grammar(page, data):
    grammar = section(page, "grammar")
    order = [g["id"] for g in data.meta["grammars"]]
    for m in data.leaderboard["models"]:
        row = re.search(r'<tr data-model="%s">(.*?)</tr>' % re.escape(m["key"]), grammar, re.S).group(1)
        counts = [int(c) for c in re.findall(r"<td[^>]*>(\d+)</td>", row)]
        assert counts == [m["by_grammar"][g] for g in order]


def test_grammar_callouts_from_data(page, data):
    grammar = section(page, "grammar")
    rows = data.leaderboard["models"]
    jaali = sum(m["by_grammar"]["jaali_lattice"] for m in rows)
    briefs = data.leaderboard["grammar_briefs"]["jaali_lattice"]
    assert f"{jaali} of {len(rows) * briefs}" in grammar
    no_kolam = sum(1 for m in rows if m["by_grammar"]["kolam_loop"] == 0)
    assert f"{no_kolam} of {len(rows)}" in grammar
    assert "2 × 3 for a 2 × 2" in grammar and "5 × 4 for a 5 × 5" in grammar


def radar_series(text: str) -> dict[str, list[float]]:
    found = re.findall(r'<polygon class="lb-series" data-model="([^"]+)" data-values="([^"]+)"', text)
    return {k: [float(v) for v in vals.split(",")] for k, vals in found}


SPEC_16_6 = {
    "gpt-6-astra": [0.931, 0.958, 0.615, 0.759, 0.407, 0.262],
    "claude-opus-5-5": [0.903, 0.194, 0.894, 0.261, 0.202, 0.220],
    "beam-hf": [0.653, 0.819, 0.545, 0.339, 0.759, 0.743],
}


def test_default_comparison_radar_values(page, data):
    series = radar_series(section(page, "profiles"))
    assert list(series) == ["gpt-6-astra", "claude-opus-5-5", "beam-hf"]
    spokes = ("passed", "readable", "repaired", "tokens", "speed", "spend")
    for key, values in series.items():
        assert values == SPEC_16_6[key]
        assert values == [round(by_key(data)[key]["radar"][s], 3) for s in spokes]


def test_compare_query_selects_models(client):
    text = client.get("/leaderboard", params={"compare": "gpt-6.1-sol,beam-hf"}).text
    profiles = section(text, "profiles")
    assert list(radar_series(profiles)) == ["gpt-6.1-sol", "beam-hf"]
    pressed = re.findall(r'<button[^>]*data-key="([^"]+)"[^>]*aria-pressed="true"', profiles)
    assert pressed == ["gpt-6.1-sol", "beam-hf"]


def test_compare_ignores_unknown_keys_and_keeps_three(client):
    text = client.get("/leaderboard", params={"compare": "nope,gpt-6-astra,gpt-6.1-sol,beam-hf,gpt-6-luna"}).text
    assert list(radar_series(section(text, "profiles"))) == ["gpt-6.1-sol", "beam-hf", "gpt-6-luna"]


def test_small_radars_one_per_model_with_captions(page, data):
    profiles = section(page, "profiles")
    cards = re.findall(r'<article class="lb-radar-card"', profiles)
    assert len(cards) == len(data.models)
    assert "Lowest: jaali, 1 of 5." in profiles
    assert "No passes in 5 of 7 grammars." in profiles  # GPT-6 Luna
    assert "Jaali lattice: 1 of 5" in profiles


def test_methodology_lists_request_settings(page, data):
    method = section(page, "method")
    assert visible_rows(method, "lb-method-row") == [m["key"] for m in data.models]
    assert "reasoning_effort low" in method
    assert "output effort low" in method
    assert "temperature 0.4 · reasoning_effort minimal" in method
    assert "temperature 0.4 · thinking disabled" in method
    assert "4,096-token output ceiling" in method
    assert f"{data.meta['totals']['proposals']:,}" in method


def test_community_empty_state(page):
    community = section(page, "community")
    assert "No community runs have been merged yet." in community
    assert "Refresh community runs" in community
    assert "/api/community.json" in page or "data-lb-refresh" in community


def test_known_gaps_facts_from_data(page, data):
    gaps = section(page, "gaps")
    assert "KNOWN GAPS · PUBLISHED, NOT PATCHED" in gaps
    assert "Read this before quoting a number." in gaps
    assert gaps.count("<h3") == 4
    totals = data.meta["totals"]
    assert f"{totals['low_ink']} of {totals['passed']} accepted designs" in gaps
    assert "Claude Opus 5.5" in gaps and "Claude Fable 5.2" in gaps  # both accepted on the impossible kolam
    assert re.search(r"Opus:? 52", gaps) and re.search(r"Haiku:? 25", gaps)


def test_every_chart_mark_has_a_tooltip(page):
    for anchor in ("ranking", "failing", "cost"):
        assert "title=" in section(page, anchor) or "<title>" in section(page, anchor)
    profiles = section(page, "profiles")
    assert "<title>GPT-6 Astra · passed within three: 67 of 72</title>" in profiles


def test_no_v1_words_in_visitor_copy(page):
    visible = re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>", "", page, flags=re.S))
    for word in (" arm ", "candidate", "witness", "Pass@1", "quilt", "Studio", "Arena"):
        assert word not in visible
