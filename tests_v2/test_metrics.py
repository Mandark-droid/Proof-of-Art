import pytest

# Spec 16.1, recomputed by the spec author from the ten-model data.
SPEC_16_1 = [
    # key, within3, (passed, broke, unreadable), readable %, passed-of-readable, repaired, wilson %, tokens k, latency s, spend
    ("gpt-6-astra", 67, (59, 10, 3), 95.8, (59, 69), (8, 13), (84.8, 97.0), 3.3, 20.7, 7.06),
    ("gpt-6.1-sol", 66, (57, 11, 4), 94.4, (57, 68), (9, 15), (83.0, 96.1), 3.5, 20.2, 1.51),
    ("claude-opus-5-5", 65, (6, 8, 58), 19.4, (6, 14), (59, 66), (81.3, 95.2), 9.3, 36.6, 8.97),
    ("claude-fable-5-1", 63, (30, 12, 30), 58.3, (30, 42), (33, 42), (77.9, 93.3), 8.8, 58.7, 20.63),
    ("gpt-6.1-terra", 57, (38, 26, 8), 88.9, (38, 64), (19, 34), (68.4, 86.9), 5.4, 25.5, 0.96),
    ("claude-sonnet-5-5", 50, (6, 15, 51), 29.2, (6, 21), (44, 66), (58.0, 78.9), 9.8, 26.8, 4.49),
    ("beam-hf", 47, (17, 42, 13), 81.9, (17, 59), (30, 55), (53.8, 75.2), 7.9, 7.8, 0.44),
    ("claude-haiku-5-5", 39, (16, 27, 29), 59.7, (16, 43), (23, 56), (42.7, 65.2), 12.9, 36.9, 0.34),
    ("gpt-6-luna", 9, (2, 16, 54), 25.0, (2, 18), (7, 70), (6.7, 22.1), 10.7, 21.9, 0.16),
    ("deepseek-v4.1-flash", 7, (3, 39, 30), 58.3, (3, 42), (4, 69), (4.8, 18.7), 11.5, 21.8, 0.45),
]
SPEC_16_2 = {  # counted, stripe, repeat, jaali, dot, rotational, kolam
    "gpt-6-astra": (31, 14, 11, 1, 4, 4, 2), "gpt-6.1-sol": (31, 14, 11, 1, 4, 3, 2),
    "claude-opus-5-5": (31, 13, 11, 2, 3, 3, 2), "claude-fable-5-1": (30, 13, 10, 3, 4, 3, 0),
    "gpt-6.1-terra": (30, 11, 10, 0, 3, 3, 0), "claude-sonnet-5-5": (26, 8, 10, 1, 3, 2, 0),
    "beam-hf": (24, 7, 11, 0, 2, 2, 1), "claude-haiku-5-5": (25, 1, 8, 1, 3, 1, 0),
    "gpt-6-luna": (7, 0, 2, 0, 0, 0, 0), "deepseek-v4.1-flash": (5, 1, 1, 0, 0, 0, 0),
}
SPEC_16_4 = {"candidate_parse": 280, "symmetry_d2": 107, "candidate_truncated": 53, "cell_quantization": 36,
             "coverage_bound": 29, "jaali_single_solid": 27, "dot_clearance": 19, "kolam_loop_count": 17,
             "translation_period": 16, "jaali_open_fraction": 13, "palette_limit": 9, "symmetry_c4": 5,
             "jaali_min_web": 5, "kolam_mirror_symmetry": 4, "layer_registration": 3,
             "jaali_holes_interior": 3, "kolam_mirrors_valid": 2}
SPEC_LOW_INK = {"gpt-6-astra": 8, "beam-hf": 8, "gpt-6.1-sol": 7, "gpt-6.1-terra": 6, "claude-opus-5-5": 4,
                "gpt-6-luna": 4, "claude-fable-5-1": 3, "claude-haiku-5-5": 2, "claude-sonnet-5-5": 1,
                "deepseek-v4.1-flash": 1}
SPEC_ANCHOR_ONLY = {"claude-opus-5-5": 52, "claude-sonnet-5-5": 42, "claude-fable-5-1": 17, "beam-hf": 5,
                    "gpt-6-luna": 4, "claude-haiku-5-5": 1}
SPEC_TRUNCATED = {"claude-haiku-5-5": 25, "claude-fable-5-1": 12, "deepseek-v4.1-flash": 6,
                  "claude-opus-5-5": 4, "claude-sonnet-5-5": 4, "gpt-6-luna": 2}
SPEC_16_6 = {
    "gpt-6-astra": (0.931, 0.958, 0.615, 0.759, 0.407, 0.262),
    "claude-opus-5-5": (0.903, 0.194, 0.894, 0.261, 0.202, 0.220),
    "beam-hf": (0.653, 0.819, 0.545, 0.339, 0.759, 0.743),
}
RADAR_KEYS = ("passed", "readable", "repaired", "tokens", "speed", "spend")


@pytest.fixture(scope="module")
def ten_metrics(ten_model_rows, ten_model_summary):
    from pramaan.tasks import load_suite
    from pramaan_v2.build.metrics import model_metrics
    from pramaan_v2.build.replay import replay_all

    runs, _ = replay_all(ten_model_rows)
    tasks, _ = load_suite("eval-v2.1")
    return runs, model_metrics(runs, ten_model_summary, tasks)


def test_order_and_spec_table_16_1(ten_metrics):
    _, metrics = ten_metrics
    assert [m.key for m in metrics] == [row[0] for row in SPEC_16_1]
    for m, (key, within, first, readable, por, repaired, wilson, tokens, latency, spend) in zip(metrics, SPEC_16_1):
        assert m.within3 == within, key
        assert (m.first["passed"], m.first["broke"], m.first["unreadable"]) == first, key
        assert round(100 * m.readable / 72, 1) == readable, key
        assert m.passed_of_readable == por, key
        assert m.repaired == repaired, key
        assert tuple(round(100 * value, 1) for value in m.wilson95) == wilson, key
        assert round(m.tokens_per_brief / 1000, 1) == tokens, key
        assert round(m.latency_s, 1) == latency, key
        assert round(m.spend_usd, 2) == spend, key
        assert m.complete and m.n == 72


def test_grammar_table_16_2(ten_metrics):
    from pramaan_v2.constants import GRAMMAR_ORDER

    _, metrics = ten_metrics
    for m in metrics:
        assert tuple(m.by_grammar[g] for g in GRAMMAR_ORDER) == SPEC_16_2[m.key], m.key
    assert sum(m.by_grammar["jaali_lattice"] for m in metrics) == 9
    assert sum(m.by_grammar["kolam_loop"] == 0 for m in metrics) == 6


def test_failing_first_try_16_4(ten_metrics):
    from pramaan_v2.build.metrics import failing_first_try

    runs, _ = ten_metrics
    rows = failing_first_try(runs)
    assert {row["check"]: row["count"] for row in rows} == SPEC_16_4
    assert [row["check"] for row in rows][:3] == ["candidate_parse", "symmetry_d2", "candidate_truncated"]
    kinds = {row["check"]: row["kind"] for row in rows}
    assert kinds["candidate_parse"] == kinds["candidate_truncated"] == "format"
    assert kinds["symmetry_d2"] == "geometry"


def test_low_ink_anchor_only_and_truncation_per_model(ten_metrics):
    _, metrics = ten_metrics
    by_key = {m.key: m for m in metrics}
    assert {k: m.low_ink for k, m in by_key.items() if m.low_ink} == SPEC_LOW_INK
    assert {k: m.anchor_only for k, m in by_key.items() if m.anchor_only} == SPEC_ANCHOR_ONLY
    assert sum(m.anchor_only for m in metrics) == 121
    assert {k: m.truncated for k, m in by_key.items() if m.truncated} == SPEC_TRUNCATED
    assert sum(m.truncated for m in metrics) == 53


def test_frontier_and_radar_16_6(ten_metrics):
    _, metrics = ten_metrics
    assert [m.key for m in sorted(metrics, key=lambda m: m.spend_usd) if m.on_frontier] == [
        "gpt-6-luna", "claude-haiku-5-5", "beam-hf", "gpt-6.1-terra", "gpt-6.1-sol", "gpt-6-astra"]
    by_key = {m.key: m for m in metrics}
    for key, values in SPEC_16_6.items():
        assert tuple(round(by_key[key].radar[k], 3) for k in RADAR_KEYS) == values, key
    assert all(0.0 <= v <= 1.0 for m in metrics for v in m.radar.values())


def test_partial_model_is_unranked_and_wall_has_gaps(ten_model_rows, ten_model_summary):
    from pramaan.tasks import load_suite
    from pramaan_v2.build.metrics import model_metrics
    from pramaan_v2.build.replay import replay_all

    rows = [r for r in ten_model_rows if r["model_key"] != "gpt-6-astra"]
    rows += [r for r in ten_model_rows if r["model_key"] == "gpt-6-astra"][:62]
    runs, _ = replay_all(rows)
    tasks, _ = load_suite("eval-v2.1")
    metrics = model_metrics(runs, ten_model_summary, tasks)
    astra = next(m for m in metrics if m.key == "gpt-6-astra")
    assert astra.complete is False and astra.n == 62
    assert metrics[-1].key == "gpt-6-astra"
    assert all(m.complete for m in metrics[:-1])


def test_a_run_without_attempts_counts_as_never_readable(published):
    from pramaan.tasks import load_suite
    from pramaan_v2.build.metrics import failing_first_try, model_metrics
    from pramaan_v2.build.replay import replay_all

    rows = [r for r in published.rows if r["model_key"] == "beam-hf"]
    empty = rows[0]["prompt_id"]
    rows[0] = dict(rows[0], attempts=[])
    runs, _ = replay_all(rows)
    tasks, _ = load_suite("eval-v2.1")
    assert runs[("beam-hf", empty)].status == "-"
    (m,) = model_metrics(runs, published.summary, tasks)
    assert m.n == 72 and sum(m.first.values()) == 71 and m.readable <= 71
    assert sum(r["count"] for r in failing_first_try(runs)) >= 0


def test_builder_refuses_two_briefs_for_one_state_and_level(tmp_path, published, monkeypatch):
    from pramaan.tasks import load_suite
    from pramaan_v2.build import site_data

    tasks, digest = load_suite("eval-v2.1")
    a, b = tasks[0], next(t for t in tasks if t.state != tasks[0].state)
    clash = [t.model_copy(update={"state": a.state, "level": a.level}) if t.id == b.id else t for t in tasks]
    monkeypatch.setattr(site_data, "load_suite", lambda suite: (clash, digest))
    with pytest.raises(ValueError, match="same state and level"):
        site_data.build(tmp_path, published)
