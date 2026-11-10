from pramaan.ui.findings import render_findings_strip, wilson_interval
import pytest


def test_findings_strip_uses_saved_repairs_grammar_latency_and_visibility():
    from pramaan.app import (_benchmark_summary, _home_suite_tasks, _load_gallery_arm_documents,
                             _load_gallery_index)

    summary = _benchmark_summary()
    documents = _load_gallery_arm_documents(_load_gallery_index())
    backgrounds = {task["id"]: task["palette"][0] for task in _home_suite_tasks()}
    rendered = render_findings_strip(summary, gallery_documents=documents,
                                     task_backgrounds=backgrounds)

    assert 'data-testid="finding-repair-origins"' in rendered
    assert "25 of Beam’s 30 came from a readable design that broke a rule" in rendered
    assert "7 of Opus’s 59 did" in rendered
    assert "Jaali had 12 accepted results out of 55 model-task results" in rendered
    assert "Beam averaged 7.8 s per request" in rendered
    assert "next fastest was 20.2 s" in rendered
    assert "51 of 536 accepted designs show little or no visible ink" in rendered
    assert "Claude Opus 5.5 61, Claude Fable 5.1 60, Claude Fable 5.2 59, GPT-6 Astra 59" in rendered
    assert "Published ranks stay as run" in rendered


def test_findings_strip_shows_unavailable_when_saved_evidence_is_missing():
    rendered = render_findings_strip({"by_model": {"beam-hf": {"overall": {}}}})

    assert rendered.count('<article data-testid="finding-') == 4
    assert "Repair starting points are not available" in rendered
    assert "Jaali results are not available" in rendered
    assert "Request latency is not available" in rendered
    assert "Near-invisible totals are not available" in rendered


def test_wilson_interval_handles_bounds_and_invalid_counts():
    assert wilson_interval(0, 10)[0] == 0
    assert wilson_interval(10, 10)[1] == pytest.approx(1)
    assert wilson_interval(2, 0) is None
    assert wilson_interval(11, 10) is None
