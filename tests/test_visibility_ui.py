import base64
from io import BytesIO
import re

from PIL import Image

from pramaan.ui.visibility import accepted_pattern_is_low_visibility


def _tile(visible_pixels: int) -> str:
    pixels = [(245, 241, 230)] * (64 * 64)
    for index in range(visible_pixels):
        pixels[index] = (20, 40, 60)
    image = Image.new("RGB", (64, 64))
    image.putdata(pixels)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_visibility_threshold_is_shared_from_runtime_package_not_runtime_scripts_dependency():
    from pramaan.visibility import MIN_VISIBLE_SHARE as package_threshold
    from scripts.audit_visible_designs import MIN_VISIBLE_SHARE as audit_threshold
    from pathlib import Path

    runtime_source = (Path(__file__).resolve().parents[1] / "pramaan" / "ui" / "visibility.py").read_text(encoding="utf-8")
    assert package_threshold == audit_threshold == 0.02
    assert "from scripts.audit_visible_designs" not in runtime_source


def test_visibility_threshold_matches_saved_audit_strict_two_percent_rule():
    under_threshold = {"passed": True, "tile_png_b64": _tile(81)}
    at_threshold = {"passed": True, "tile_png_b64": _tile(82)}
    assert accepted_pattern_is_low_visibility(under_threshold, "#F5F1E6")
    assert not accepted_pattern_is_low_visibility(at_threshold, "#F5F1E6")
    assert not accepted_pattern_is_low_visibility(
        {"passed": False, "tile_png_b64": _tile(0)}, "#F5F1E6",
    )


def test_home_state_sheet_flags_accepted_barely_visible_pattern():
    from pramaan.ui.home import render_reference_wall

    task = {"state": "Andhra Pradesh", "id": "ap-easy", "level": "easy",
            "brief": "A brief", "palette": ["#F5F1E6", "#14283C"]}
    accepted = {"index": 1, "passed": True, "tile_png_b64": _tile(81)}
    rendered = render_reference_wall(
        [], [], {"beam-hf": {"tasks": {"ap-easy": {"passed_at": 1, "attempts": [accepted]}}}},
        [task], [{"key": "beam-hf", "label": "Beam"}],
    )
    assert "Passes the checks, but almost nothing is visible" in rendered


def test_arena_flags_accepted_barely_visible_pattern():
    from pramaan.ui.compare import render_arena

    key = "beam-hf"
    rendered = render_arena(
        {"suite_id": "eval-test", "suite_sha256": "a" * 64,
         "arms": [{"key": key, "label": "Beam"}]},
        {key: {"tasks": {"prompt": {
            "brief": "Same brief", "passed_at": 1,
            "attempts": [{"index": 1, "passed": True, "tile_png_b64": _tile(81)}],
        }}}},
        suite_id="eval-test", suite_sha256="a" * 64, prompt_id="prompt",
        replay_prompt_id="other", task_backgrounds={"prompt": "#F5F1E6"},
    )
    assert "Passes the checks, but almost nothing is visible" in rendered


def test_visibility_limitation_is_in_models_and_readme():
    from pathlib import Path
    from pramaan.app import (_benchmark_summary, _home_suite_tasks, _load_gallery_arm_documents,
                             _load_gallery_index, render_model_report)

    message = "The checker measures drawn shapes, not visibility: 44 of 470 accepted designs show little or no visible ink."
    report = render_model_report(
        _benchmark_summary(), gallery_documents=_load_gallery_arm_documents(_load_gallery_index()),
        task_backgrounds={task["id"]: task["palette"][0] for task in _home_suite_tasks()},
    )
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert message in report and message in readme
    assert report.index('data-testid="model-report-table"') < report.index(message)


def test_model_report_near_invisible_counts_sum_to_saved_44_of_470():
    from pramaan.app import (_benchmark_summary, _home_suite_tasks, _load_gallery_arm_documents,
                             _load_gallery_index)
    from pramaan.ui.leaderboard import render_model_report

    index = _load_gallery_index()
    documents = _load_gallery_arm_documents(index)
    backgrounds = {task["id"]: task["palette"][0] for task in _home_suite_tasks()}
    rendered = render_model_report(_benchmark_summary(), gallery_documents=documents,
                                   task_backgrounds=backgrounds)
    row_counts = [tuple(map(int, match)) for match in re.findall(
        r'data-near-invisible="(\d+)" data-accepted-designs="(\d+)"', rendered,
    )]

    assert len(documents) == 10
    assert len(row_counts) == 10
    assert sum(low for low, _accepted in row_counts) == 44
    assert sum(accepted for _low, accepted in row_counts) == 470
    assert "44 of 470 accepted designs show little or no visible ink" in rendered
