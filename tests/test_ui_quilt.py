from pramaan.ui.quilt import STATE_GRID, render_data_quilt, state_grid_cell, state_grid_style


def test_state_quilt_layout_covers_36_distinct_states_and_cells():
    assert len(STATE_GRID) == 36
    assert len(set(STATE_GRID.values())) == 36
    assert all(0 <= col < 10 and 0 <= row < 9 for col, row in STATE_GRID.values())


def test_state_quilt_uses_documented_geographic_anchor_points():
    assert state_grid_cell("Jammu and Kashmir") == (3, 0)
    assert state_grid_cell("Gujarat") == (1, 4)
    assert state_grid_cell("Kerala") == (3, 8)
    assert state_grid_cell("Andaman and Nicobar Islands") == (8, 8)


def test_state_grid_style_uses_one_based_css_grid_positions():
    assert state_grid_style("Gujarat") == "grid-column:2;grid-row:5"
    assert state_grid_style("missing") == ""
    assert state_grid_cell("missing") is None


def test_benchmark_quilt_renders_all_states_in_geographic_cells():
    from pramaan import app

    assert set(STATE_GRID) == set(app.ALL_STATES)
    html = app._benchmark_state_mosaic_html([], {})
    assert 'class="pramaan-state-mosaic-scroll"' in html
    assert 'role="grid" aria-label="36 state and Union Territory pattern tiles"' in html
    for state, (column, row) in STATE_GRID.items():
        assert f'data-state="{state}"' in html
        assert f'data-grid-col="{column}" data-grid-row="{row}"' in html
        assert f'grid-column:{column + 1};grid-row:{row + 1}' in html


def _quilt_fixture():
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/nKgAAAAASUVORK5CYII="
    state = {"state": "Gujarat", "state_type": "state", "grid": [1, 4],
             "tradition_id": "gj_bandhani", "tradition": "Bandhani dots",
             "caution": "Inspired by local practice; not a substitute.",
             "sources": [{"title": "Source", "url": "https://example.org/craft"}]}
    task = {"state": "Gujarat", "level": "easy", "brief": "A <script>dot pattern</script>.",
            "passed_at": 1, "attempts": [{"index": 1, "passed": True, "failing_checks": [],
                                          "witness": [], "tile_png_b64": png}]}
    index = {"states": [state], "arms": [{"key": "beam-hf"}, {"key": "model-b"}]}
    docs = {"beam-hf": {"tasks": {"gj_easy": task}},
            "model-b": {"tasks": {"gj_easy": {**task, "passed_at": 2}}}}
    return index, docs


def test_data_quilt_model_view_has_accessible_tile_cultural_context_and_attempts():
    index, docs = _quilt_fixture()
    rendered = render_data_quilt(index, docs, model_key="beam-hf", level="easy")
    assert 'aria-label="36 state and Union Territory model quilt"' in rendered
    assert 'aria-label="Gujarat, beam-hf, passed on attempt 1"' in rendered
    assert "A &lt;script&gt;dot pattern&lt;/script&gt;." in rendered
    assert "Cultural care:" in rendered and "Bandhani dots" in rendered
    assert "Attempt 1: passed" in rendered
    assert 'loading="lazy" decoding="async"' in rendered


def test_data_quilt_compare_and_hardness_views_use_same_state_records():
    index, docs = _quilt_fixture()
    compare = render_data_quilt(index, docs, model_key="beam-hf", compare_model="model-b", mode="compare")
    hardness = render_data_quilt(index, docs, model_key="beam-hf", mode="hardness")
    assert "beam-hf" in compare and "model-b" in compare
    assert "passed at 1" in compare and "passed at 2" in compare
    assert "1/2" in hardness


def test_data_quilt_comparison_uses_exact_shared_task_and_hardness_excludes_not_run():
    index, docs = _quilt_fixture()
    docs["model-b"]["tasks"] = {
        "different_prompt": docs["model-b"]["tasks"]["gj_easy"],
        "not_run": {"state": "Gujarat", "level": "easy", "status": "not_run", "attempts": []},
    }
    compare = render_data_quilt(index, docs, model_key="beam-hf", compare_model="model-b", mode="compare")
    hardness = render_data_quilt(index, docs, model_key="beam-hf", mode="hardness")
    assert "model-b</b><small>no matching shared task</small>" in compare
    assert "1/1" in hardness


def test_data_quilt_rejects_non_png_or_invalid_base64_previews():
    index, docs = _quilt_fixture()
    task = docs["beam-hf"]["tasks"]["gj_easy"]
    task["attempts"][0]["tile_png_b64"] = "bm90IGFuIGltYWdl"
    rendered = render_data_quilt(index, docs, model_key="beam-hf")
    assert "data:image/png;base64,bm90IGFuIGltYWdl" not in rendered


def test_data_quilt_fits_phone_width_and_exposes_stable_replay_selectors():
    index, docs = _quilt_fixture()
    rendered = render_data_quilt(index, docs, model_key="beam-hf")
    assert 'data-testid="quilt-scroll-region"' in rendered
    assert 'data-testid="quilt-grid"' in rendered
    assert 'data-testid="quilt-state-gujarat"' in rendered
    assert 'aria-haspopup="dialog"' in rendered
    assert 'data-testid="quilt-state-sheet"' in rendered
    assert 'data-testid="quilt-state-sheet-close"' in rendered
    assert 'Watch the featured proof replay' in rendered
    # Inline min-width overrides the legacy 340px phone minimum without
    # changing the 10-column geography, so a 265px content viewport can fit.
    assert 'style="min-width:0;width:100%;max-width:960px"' in rendered
