from pathlib import Path
import re

from pramaan.ui.leaderboard import render_model_report
from pramaan.ui.leaderboard import render_canonical_leaderboard, render_community_leaderboard


def test_community_leaderboard_renders_common_scores_and_escapes_model_ids():
    snapshot = {
        "status": "Community results loaded.",
        "selected": {"provider/model<&": {"run_id": "run-1"}},
        "summary": {
            "common_completed": {
                "model_keys": ["beam-hf", "provider/model<&"],
                "prompt_count": 2,
                "by_model": {
                    "beam-hf": {"validity_at_1": 0.5, "validity_at_3": 1.0},
                    "provider/model<&": {"validity_at_1": 0.5, "validity_at_3": 0.5},
                },
            },
            "by_model": {
                "beam-hf": {"study": {"observed_prompt_count": 2}, "overall": {}},
                "provider/model<&": {"study": {"observed_prompt_count": 2}, "overall": {}},
            },
        },
    }

    rendered = render_community_leaderboard(snapshot)

    assert "Compared on 2 identical completed questions" in rendered
    assert "Rank 1" in rendered
    assert "Community · self-reported model ID" in rendered
    assert "provider/model&lt;&amp;" in rendered
    assert "provider/model<&" not in rendered


def test_canonical_leaderboard_uses_injected_gallery_and_suite_metadata():
    index = {
        "suite_id": "eval-test",
        "suite_sha256": "a" * 64,
        "common_prompt_ids": ["task-1"],
        "arms": [{"key": "beam-hf", "label": "Beam", "provider": "Hugging Face",
                  "completed_prompt_ids": ["task-1"]}],
        "baselines": [],
    }
    rendered = render_canonical_leaderboard(
        index,
        root=Path("/fixture"),
        suite_id="eval-test",
        suite_sha256="a" * 64,
        load_gallery_index=lambda: None,
        gallery_json=lambda _path: {
            "tasks": {"task-1": {"status": "completed", "passed_at": 1,
                                  "grammar": "rotational_field"}},
        },
        wilson_interval=lambda _successes, _total: (0.2, 1.0),
        comparison_charts=lambda _records: '<svg aria-label="comparison"></svg>',
    )

    assert "Canonical model results · eval-test" in rendered
    assert "Shared questions: 1" in rendered
    assert "Beam" in rendered
    assert "G5" in rendered
    assert "comparison" in rendered
    assert 'data-testid="leaderboard-hero"' in rendered
    assert 'data-testid="leaderboard-hero-beam-hf"' in rendered
    assert 'aria-label="Beam: pass at three 1 of 1, 100.0%; 95 percent Wilson interval' in rendered
    assert "Baseline · no other complete arm" in rendered
    assert "Text alternative:" in rendered


def test_leaderboard_hero_keeps_partial_arm_visible_without_ranking_it():
    index = {
        "suite_id": "eval-test", "suite_sha256": "b" * 64,
        "common_prompt_ids": ["task-1", "task-2"],
        "arms": [
            {"key": "beam-hf", "label": "Beam", "completed_prompt_ids": ["task-1", "task-2"]},
            {"key": "claude-haiku-5-5", "label": "Claude Haiku", "completed_prompt_ids": ["task-1"]},
        ],
        "baselines": [],
    }
    docs = {
        "beam-hf": {"tasks": {key: {"status": "completed", "passed_at": 1} for key in ("task-1", "task-2")}},
        "claude-haiku-5-5": {"tasks": {"task-1": {"status": "completed", "passed_at": 1}}},
    }
    rendered = render_canonical_leaderboard(
        index, root=Path("/fixture"), suite_id="eval-test", suite_sha256="b" * 64,
        load_gallery_index=lambda: None,
        gallery_json=lambda path: docs[path.stem],
        wilson_interval=lambda successes, total: (0.1, 0.9),
        comparison_charts=lambda _records: "",
    )
    assert 'data-testid="leaderboard-hero-claude-haiku-5-5"' in rendered
    assert 'data-coverage="1/2"' in rendered
    assert "Partial coverage 1/2; not ranked" in rendered


def test_leaderboard_hero_marks_equal_complete_scores_as_ties():
    index = {
        "suite_id": "eval-test", "suite_sha256": "c" * 64,
        "common_prompt_ids": ["task-1"],
        "arms": [
            {"key": "beam-hf", "label": "Beam", "completed_prompt_ids": ["task-1"]},
            {"key": "claude-haiku-5-5", "label": "Claude Haiku", "completed_prompt_ids": ["task-1"]},
        ],
        "baselines": [],
    }
    docs = {key: {"tasks": {"task-1": {"status": "completed", "passed_at": 1}}}
            for key in ("beam-hf", "claude-haiku-5-5")}
    rendered = render_canonical_leaderboard(
        index, root=Path("/fixture"), suite_id="eval-test", suite_sha256="c" * 64,
        load_gallery_index=lambda: None,
        gallery_json=lambda path: docs[path.stem],
        wilson_interval=lambda successes, total: (0.2, 0.9),
        comparison_charts=lambda _records: "",
    )
    assert rendered.count("Tie · rank 1") == 4  # chart and detailed table for each arm


def _model_report_fixture():
    names = [
        ("beam-hf", "Beam"), ("claude-haiku-5-5", "Haiku"),
        ("claude-sonnet-5-5", "Sonnet"), ("claude-opus-5-5", "Opus"),
        ("claude-fable-5-1", "Fable"), ("deepseek-v4.1-flash", "DeepSeek"),
        ("gpt-6.1-sol", "Sol"), ("gpt-6.1-terra", "Terra"),
        ("gpt-6-luna", "Luna"), ("gpt-6-astra", "Astra"),
    ]
    rows = {}
    for index, (key, _label) in enumerate(names):
        n = 34 if key == "claude-fable-5-1" else 72
        passed1, passed3 = 8 + index, 20 + index
        rows[key] = {
            "study": {"expected_prompt_count": 72, "observed_prompt_count": n + 2,
                      "provider_error_count": 2, "cap_reached": key == "claude-fable-5-1"},
            "overall": {"n": n, "validity_at_1": passed1 / n,
                        "validity_at_3": passed3 / n, "estimated_cost_totals_by_currency": {"USD": 1.25}},
            "label": _label,
        }
    return {"expected_prompt_count": 72, "by_model": rows}


def test_model_report_never_ranks_partial_fable_above_complete_models():
    rendered = render_model_report(_model_report_fixture())
    complete_row = rendered.index('data-testid="model-report-row-gpt-6-astra"')
    fable_start = rendered.index('data-testid="model-report-row-claude-fable-5-1"')
    fable_end = rendered.index("</tr>", fable_start)
    fable_row = rendered[fable_start:fable_end]

    assert complete_row < fable_start
    assert "Not ranked" in fable_row
    assert "34 of 72 evaluated" in fable_row
    assert "USD 10 cap" in fable_row


def test_model_report_shows_ten_unique_models_evaluated_denominators_and_provider_errors():
    rendered = render_model_report(_model_report_fixture())
    assert rendered.count('data-testid="model-report-row-') == 10
    assert 'data-model-key="claude-fable-5-1"' in rendered
    assert "34 of 72 evaluated" in rendered
    assert "USD 10 cap" in rendered
    assert "Provider errors are excluded from evaluated counts" in rendered
    assert 'data-testid="model-report-chart"' in rendered
    assert "95% Wilson" in rendered
    assert "First try" in rendered
    assert rendered.count('data-testid="model-chart-') == 10
    assert "First try" in rendered
    assert "Text alternative:" in rendered


def test_model_report_chart_stacks_first_try_outcomes_from_saved_attempts():
    counts_by_model = {
        "gpt-6-astra": (59, 10, 3), "gpt-6.1-sol": (57, 11, 4),
        "claude-opus-5-5": (6, 8, 58), "claude-fable-5-1": (30, 12, 30),
        "gpt-6.1-terra": (38, 26, 8), "claude-sonnet-5-5": (6, 15, 51),
        "beam-hf": (17, 42, 13), "claude-haiku-5-5": (16, 27, 29),
        "deepseek-v4.1-flash": (3, 39, 30), "gpt-6-luna": (2, 16, 54),
    }
    within_three = {
        "gpt-6-astra": 67, "gpt-6.1-sol": 66, "claude-opus-5-5": 65,
        "claude-fable-5-1": 63, "gpt-6.1-terra": 57, "claude-sonnet-5-5": 50,
        "beam-hf": 47, "claude-haiku-5-5": 39, "deepseek-v4.1-flash": 7,
        "gpt-6-luna": 9,
    }
    summary = _model_report_fixture()
    documents = {}
    for model, (passed, broke_rule, unreadable) in counts_by_model.items():
        overall = summary["by_model"][model]["overall"]
        overall.update(n=72, passed_at_1=passed, validity_at_1=passed / 72,
                       accepted_designs=within_three[model],
                       validity_at_3=within_three[model] / 72)
        outcomes = ([{"passed": True, "failing_checks": []}] * passed
                    + [{"passed": False, "failing_checks": ["symmetry_d2"]}] * broke_rule
                    + [{"passed": False, "failing_checks": ["candidate_parse"]}] * unreadable)
        documents[model] = {"tasks": {str(index): {"attempts": [outcome]}
                                       for index, outcome in enumerate(outcomes)}}

    rendered = render_model_report(summary, gallery_documents=documents)

    assert "First try: 17 passed · 42 broke a rule · 13 unreadable" in rendered
    assert "Within three: 47/72" in rendered
    assert 'aria-label="First try: 17 passed, 42 broke a rule, 13 unreadable"' in rendered
    for model, expected_counts in counts_by_model.items():
        match = re.search(rf'<li[^>]*data-testid="model-chart-{re.escape(model)}"([^>]*)>', rendered)
        assert match
        attributes = match.group(1)
        actual = tuple(int(re.search(rf'data-first-{field}="(\d+)"', attributes).group(1))
                       for field in ("pass", "rule-fail", "unreadable"))
        assert actual == expected_counts
        assert sum(actual) == 72


def test_model_report_adds_wilson_intervals_and_near_invisible_column():
    from pramaan.ui.findings import wilson_interval

    summary = _model_report_fixture()
    rendered = render_model_report(summary)
    beam_start = rendered.index('data-testid="model-report-row-beam-hf"')
    beam_row = rendered[beam_start:rendered.index("</tr>", beam_start)]
    first_interval = wilson_interval(8, 72)
    within_interval = wilson_interval(20, 72)

    assert '<th scope="col">Near-invisible</th>' in rendered
    assert (f'95% Wilson {first_interval[0]:.1%}–{first_interval[1]:.1%}'
            in beam_row)
    assert (f'95% Wilson {within_interval[0]:.1%}–{within_interval[1]:.1%}'
            in beam_row)
    assert "medal" not in rendered.lower() and "podium" not in rendered.lower()


def test_model_report_uses_model_specific_cap_after_fable_completion():
    summary = _model_report_fixture()
    fable = summary["by_model"]["claude-fable-5-1"]
    fable["overall"]["n"] = 72
    fable["overall"]["validity_at_3"] = 0.875
    fable["cost_ledger"] = {"cap_usd": 22.0}
    rendered = render_model_report(summary)
    start = rendered.index('data-testid="model-report-row-claude-fable-5-1"')
    row = rendered[start:rendered.index("</tr>", start)]

    assert "72 of 72 evaluated" in row
    assert "USD 22 cap" in row
    assert "USD 10 cap" not in row


def test_model_report_ties_share_rank_and_duplicate_keys_are_impossible():
    summary = _model_report_fixture()
    summary["by_model"]["claude-haiku-5-5"]["overall"]["n"] = summary["by_model"]["beam-hf"]["overall"]["n"]
    summary["by_model"]["claude-haiku-5-5"]["overall"]["validity_at_3"] = summary["by_model"]["beam-hf"]["overall"]["validity_at_3"]
    summary["by_model"]["claude-haiku-5-5"]["overall"]["validity_at_1"] = summary["by_model"]["beam-hf"]["overall"]["validity_at_1"]
    rendered = render_model_report(summary)
    assert "Tie · rank" in rendered
    assert rendered.count('data-testid="model-report-row-beam-hf"') == 1
    assert rendered.count('data-testid="model-report-row-claude-haiku-5-5"') == 1


def test_model_report_never_treats_provider_errors_as_failed_tasks():
    summary = _model_report_fixture()
    beam = summary["by_model"]["beam-hf"]
    beam["study"]["observed_prompt_count"] = 72
    beam["study"]["provider_error_count"] = 10
    beam["overall"]["n"] = 62
    beam["overall"]["validity_at_1"] = 31 / 62
    beam["overall"]["validity_at_3"] = 40 / 62
    rendered = render_model_report(summary)
    assert 'data-model-key="beam-hf"' in rendered
    assert "62 of 72 evaluated" in rendered
    assert "40 / 62" in rendered
    assert "72 of 72 failed" not in rendered
