from pramaan.ui.home import render_reference_wall
from pramaan.app import _home_reference_rows
import base64
from io import BytesIO
from PIL import Image


def _tiny_png_b64():
    buffer = BytesIO()
    Image.new("RGBA", (1, 1), (20, 30, 40, 255)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_bundled_home_references_load_all_71_pre_rendered_thumbnails():
    references = _home_reference_rows()

    assert len(references) == 71
    assert len({row["task_id"] for row in references}) == 71
    assert all(row["source"] == "procedural_reference" and row["data_role"] == "reference"
               and row["training_eligible"] is False
               and row["thumbnail_uri"].startswith("data:image/png;base64,")
               for row in references)


def test_wall_uses_map_grid_two_marks_model_chips_and_state_sheets():
    png = _tiny_png_b64()
    references = [
        {"state": "Andhra Pradesh", "task_id": "ap-easy", "level": "easy",
         "grammar": "counted_grid", "brief": "Easy brief.",
         "checks": [{"id": "schema_valid", "status": "PASS"},
                    {"id": "dot_clearance", "status": "PASS"}],
         "thumbnail_uri": "data:image/png;base64," + png},
        {"state": "Andhra Pradesh", "task_id": "ap-hard", "level": "hard",
         "grammar": "counted_grid", "brief": "Hard brief.", "checks": [],
         "thumbnail_uri": "data:image/png;base64," + png},
    ]
    suite_tasks = [
        {"state": "Andhra Pradesh", "id": "ap-easy", "level": "easy", "brief": "Easy brief."},
        {"state": "Andhra Pradesh", "id": "ap-hard", "level": "hard", "brief": "Hard brief."},
    ]
    states = [{"state": "Andhra Pradesh", "grid": [5, 6], "tradition": "Mangalagiri"}]
    arm_documents = {
        "beam-hf": {"tasks": {
            "ap-easy": {"state": "Andhra Pradesh", "level": "easy", "passed_at": 1,
                         "attempts": [{"index": 1, "passed": True, "tile_png_b64": png}]},
            "ap-hard": {"state": "Andhra Pradesh", "level": "hard", "passed_at": 2,
                         "attempts": [{"index": 1, "passed": False, "failing_checks": ["symmetry_d2"],
                                       "witness": [[12, 20, "pixel"]]},
                                      {"index": 2, "passed": True, "tile_png_b64": png}]},
        }},
        "claude-haiku-5-5": {"tasks": {
            "ap-easy": {"state": "Andhra Pradesh", "level": "easy", "attempts": []},
            "ap-hard": {"state": "Andhra Pradesh", "level": "hard", "attempts": []},
        }},
    }
    arms = [{"key": "beam-hf", "label": "Beam"},
            {"key": "claude-haiku-5-5", "label": "Haiku"}]

    rendered = render_reference_wall(references, states, arm_documents, suite_tasks, arms,
                                     selected_model="beam-hf", display_mode="reference")

    assert 'data-home-wall-tile' in rendered
    assert 'role="grid"' in rendered and 'grid-column:6;grid-row:7' in rendered
    assert rendered.count('data-wall-mark=') == 72
    tile = rendered.split('data-state="Andhra Pradesh"', 1)[1].split('</article>', 1)[0]
    assert tile.count('data-wall-mark=') == 2
    assert 'data-wall-model-choice="beam-hf"' in rendered
    assert 'data-wall-model-choice="claude-haiku-5-5"' in rendered
    assert '>Beam<small>1 first · 1 repaired · 0 failed · 0 not run' in rendered
    assert '>Haiku<small>0 first · 0 repaired · 0 failed · 2 not run' in rendered
    assert 'aria-label="Andhra Pradesh; easy task right first try; hard task right after repair"' in rendered
    assert 'Easy brief.' in rendered and 'Hard brief.' in rendered
    assert "Pattern structure" in rendered and "Dot spacing" in rendered
    assert "schema_valid" not in rendered and "dot_clearance" not in rendered
    assert "Symmetry (D2)" in rendered and "symmetry_d2" not in rendered
    assert "Passed means the geometric checks passed; it does not mean the design matches the brief." in rendered
    assert rendered.index("Beam attempt · Right first try") < rendered.index("Passed means the geometric checks passed")
    assert 'data-wall-state-sheet' in rendered
    assert 'All model results (2)' in rendered
    assert 'Other model results' not in rendered
    assert 'No reference: this brief has no valid solution' not in rendered


def test_unsatisfiable_easy_task_has_no_reference_but_keeps_state_tile():
    rendered = render_reference_wall(
        [], [], {},
        [{"state": "Puducherry", "id": "py_kolam-as-practised-in-puducherry-easy-35", "level": "easy",
          "brief": "Proved unsatisfiable."}],
        [], selected_model="beam-hf", display_mode="reference",
    )

    assert 'data-state="Puducherry"' in rendered
    assert "No reference: this brief has no valid solution" in rendered
    assert "frozen evaluation" in rendered


def test_attempt_mode_shows_selected_model_design_or_an_explicit_missing_state():
    png = _tiny_png_b64()
    refs = [{"state": "Andhra Pradesh", "task_id": "ap-easy", "level": "easy",
             "thumbnail_uri": "data:image/png;base64," + png}]
    arms = [{"key": "beam-hf", "label": "Beam"}]
    docs = {"beam-hf": {"tasks": {"ap-easy": {"state": "Andhra Pradesh", "level": "easy",
                                                   "passed_at": 2, "attempts": [
                                                       {"index": 2, "passed": True,
                                                        "tile_png_b64": png}]}}}}

    rendered = render_reference_wall(refs, [], docs,
                                     [{"state": "Andhra Pradesh", "id": "ap-easy", "level": "easy"}],
                                     arms, selected_model="beam-hf", display_mode="attempt")

    assert 'data-wall-display="attempt"' in rendered
    assert 'alt="Beam model attempt for Andhra Pradesh easy brief"' in rendered
    assert 'aria-label="Andhra Pradesh; easy task right after repair; hard task not run"' in rendered
    assert 'data:image/png;base64,' + png in rendered


def test_parse_failure_explains_missing_wall_preview_in_plain_language():
    task = {"state": "Andhra Pradesh", "level": "easy", "attempts": [
        {"index": 1, "passed": False, "failing_checks": ["candidate_parse"]},
    ]}
    rendered = render_reference_wall(
        [], [], {"beam-hf": {"tasks": {"ap-easy": task}}},
        [{"state": "Andhra Pradesh", "id": "ap-easy", "level": "easy", "brief": "Unreadable proposal."}],
        [{"key": "beam-hf", "label": "Beam"}], selected_model="beam-hf", display_mode="attempt",
    )
    assert "Could not be read as a pattern." in rendered
    assert "Readable pattern" in rendered
    assert "candidate_parse" not in rendered


def test_wall_uses_leaderboard_model_names_and_all_model_results_label():
    key = "claude-fable-5-1"
    rendered = render_reference_wall(
        [], [], {key: {"tasks": {}}}, [], [{"key": key, "label": key}],
    )
    assert '>Claude Fable 5.1<' in rendered
    assert '<strong>Claude Fable 5.1</strong>' in rendered
    assert ">All model results (1)<" in rendered
    assert "Other model results" not in rendered


def test_wall_uses_short_visible_names_without_losing_full_state_accessible_names():
    rendered = render_reference_wall([], [], {}, [], [])

    for full, short in (
        ("Jammu and Kashmir", "J&amp;K"),
        ("Dadra and Nagar Haveli and Daman and Diu", "DNH &amp; DD"),
        ("Arunachal Pradesh", "Arunachal"),
        ("Andaman and Nicobar Islands", "Andaman"),
    ):
        assert f'data-state="{full}"' in rendered
        assert f'title="{full}"' in rendered
        assert f'class="pramaan-wall-state-name" title="{full}">{short}</span>' in rendered


def test_phone_wall_switches_to_three_columns_with_readable_art_and_compact_model_row():
    from pramaan.app import PRAMAAN_CSS
    from pramaan.ui.quilt import STATE_GRID

    rendered = render_reference_wall([], [], {}, [], [])
    phone_rules = PRAMAAN_CSS.split("@media (max-width: 600px)", 1)[1]

    assert rendered.count('class="pramaan-wall-state"') == len(STATE_GRID) == 36
    assert "grid-template-columns: repeat(3,minmax(0,1fr))" in phone_rules
    assert "grid-column: auto !important" in phone_rules
    assert "grid-row: auto !important" in phone_rules
    assert ".pramaan-wall-models { flex-wrap: nowrap; overflow-x: auto;" in phone_rules
    assert ".pramaan-wall-state-name { font-size: 12px; }" in phone_rules
    assert ".pramaan-wall-mark { font-size: 9px;" in phone_rules
