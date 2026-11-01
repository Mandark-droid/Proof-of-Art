from pramaan.ui.findings import render_findings_strip, wilson_interval
import pytest


def test_findings_strip_uses_live_summary_counts_and_discloses_methods():
    summary = {
        "by_model": {
            "beam-hf": {"study": {"expected_prompt_count": 72, "observed_prompt_count": 72},
                        "overall": {"n": 72, "validity_at_1": 17 / 72,
                                    "repair_requested": 55, "repair_rescued": 30,
                                    "accepted_designs": 47}},
        },
    }
    rendered = render_findings_strip(summary)
    assert 'data-testid="finding-first-try"' in rendered
    assert "17/72 · 23.6%" in rendered
    assert 'data-testid="finding-repair-gain"' in rendered
    assert "30/55 · 54.5%" in rendered
    assert "47/72 · 65.3%" in rendered
    assert "95% Wilson" in rendered
    assert "Scores measure geometry, not cultural authenticity" in rendered


def test_findings_strip_contains_three_live_metrics_without_a_stale_roster():
    summary = {
        "by_model": {
            "beam-hf": {"study": {"expected_prompt_count": 72, "observed_prompt_count": 72},
                        "overall": {"n": 72, "validity_at_1": 17 / 72,
                                    "repair_requested": 55, "repair_rescued": 30,
                                    "accepted_designs": 47}},
        },
    }
    rendered = render_findings_strip(summary)
    assert rendered.count("<article data-testid=\"finding-") == 3
    assert "17/72 · 23.6%" in rendered
    assert "30/55 · 54.5%" in rendered
    assert "47/72 · 65.3%" in rendered
    assert "finding-arm-" not in rendered


def test_wilson_interval_handles_bounds_and_invalid_counts():
    assert wilson_interval(0, 10)[0] == 0
    assert wilson_interval(10, 10)[1] == pytest.approx(1)
    assert wilson_interval(2, 0) is None
    assert wilson_interval(11, 10) is None
