from pathlib import Path

from pramaan.ui.compare import ARENA_PICKER_TASK_IDS, render_arena, render_arena_picker, render_same_brief_wall


def test_same_brief_wall_pairs_only_matching_prompt_ids_and_briefs():
    index = {
        "suite_id": "eval-test",
        "suite_sha256": "a" * 64,
        "arms": [
            {"key": "beam-hf", "label": "Beam"},
            {"key": "model-b", "label": "Model B"},
        ],
    }
    docs = {
        "beam-hf": {"tasks": {
            "task-1": {"state": "Gujarat", "brief": "Shared brief", "status": "completed",
                       "passed_at": 1, "attempts": [{"passed": True}]},
            "task-2": {"state": "Gujarat", "brief": "Brief A", "attempts": []},
            "task-3": {"state": "Gujarat", "brief": "Brief C", "attempts": []},
        }},
        "model-b": {"tasks": {
            "task-1": {"state": "Gujarat", "brief": "Shared brief", "status": "completed",
                       "passed_at": 1, "attempts": [{"passed": True}]},
            "task-2": {"state": "Gujarat", "brief": "Brief B", "attempts": []},
            "task-3": {"state": "Gujarat", "attempts": []},
        }},
    }

    rendered = render_same_brief_wall(
        index, "Gujarat", root=Path("/fixture"), suite_id="eval-test",
        suite_sha256="a" * 64, gallery_json=lambda path: docs[path.stem],
    )

    assert 'aria-label="Same-brief model comparison"' in rendered
    assert "task-1" in rendered and "Shared brief" in rendered
    assert "Beam" in rendered and "Model B" in rendered
    assert "task-2" in rendered
    assert "task-3" in rendered
    assert "Brief A" not in rendered and "Brief B" not in rendered
    assert rendered.count("pramaan-model-tile-card") == 2
    assert "not compared" in rendered


def test_same_brief_wall_rejects_stale_suite_and_bad_tile_payload():
    stale = {"suite_id": "eval-old", "suite_sha256": "b" * 64, "arms": []}
    unavailable = render_same_brief_wall(
        stale, "Gujarat", root=Path("/fixture"), suite_id="eval-test",
        suite_sha256="a" * 64, gallery_json=lambda _path: {},
    )
    assert "not available" in unavailable

    index = {"suite_id": "eval-test", "suite_sha256": "a" * 64,
             "arms": [{"key": "beam-hf", "label": "<Beam>"}]}
    doc = {"tasks": {"task<&": {
        "state": "Gujarat", "brief": "<script>bad</script>", "status": "completed",
        "passed_at": 1, "attempts": [{"passed": True, "tile_png_b64": "PHNjcmlwdD4="}],
    }}}
    rendered = render_same_brief_wall(
        index, "Gujarat", root=Path("/fixture"), suite_id="eval-test",
        suite_sha256="a" * 64, gallery_json=lambda _path: doc,
    )
    assert "<script>bad</script>" not in rendered
    assert "&lt;script&gt;bad&lt;/script&gt;" in rendered
    assert "&lt;Beam&gt;" in rendered
    assert "task&lt;&amp;" in rendered
    assert "data:image/png;base64" not in rendered


def test_arena_shows_pass_fail_per_attempt_and_featured_replay_link():
    index = {
        "suite_id": "eval-test", "suite_sha256": "a" * 64,
        "arms": [{"key": "beam-hf", "label": "Beam"},
                 {"key": "claude-haiku-5-5", "label": "Claude Haiku 5.5"}],
    }
    docs = {
        "beam-hf": {"tasks": {"shared-task": {
            "brief": "A shared geometric brief", "status": "completed", "passed_at": 2,
            "attempts": [
                {"index": 1, "passed": False, "failing_checks": ["symmetry_d2"], "tile_png_b64": None},
                {"index": 2, "passed": True, "failing_checks": [], "tile_png_b64": None},
            ],
        }}},
        "claude-haiku-5-5": {"tasks": {}},
    }
    rendered = render_arena(
        index, docs, suite_id="eval-test", suite_sha256="a" * 64,
        prompt_id="shared-task", replay_prompt_id="shared-task",
    )
    assert "A shared geometric brief" in rendered
    assert 'data-testid="arena-attempt-beam-hf-1" data-attempt-state="fail"' in rendered
    assert 'data-testid="arena-attempt-beam-hf-2" data-attempt-state="pass"' in rendered
    assert "Symmetry (D2)" in rendered and "symmetry_d2" not in rendered
    assert "× Fail" in rendered and "✓ Pass" in rendered
    assert 'data-testid="arena-arm-claude-haiku-5-5"' in rendered
    assert 'data-arena-model-card="claude-haiku-5-5" data-arm-state="not-run" hidden' in rendered
    assert 'data-arena-model-choice="beam-hf" checked' in rendered
    assert 'data-testid="arena-replay-beam-hf"' in rendered and 'href="#proof-replay"' in rendered
    assert 'data-testid="arena-replay-claude-haiku-5-5"' not in rendered


def test_arena_uses_leaderboard_display_name_for_raw_model_key():
    key = "claude-fable-5-1"
    rendered = render_arena(
        {"suite_id": "eval-test", "suite_sha256": "a" * 64,
         "arms": [{"key": key, "label": key}]},
        {key: {"tasks": {"prompt": {"brief": "Same brief", "attempts": [], "passed_at": None}}}},
        suite_id="eval-test", suite_sha256="a" * 64, prompt_id="prompt", replay_prompt_id="other",
    )
    assert ">Claude Fable 5.1</h4>" in rendered
    assert "> Claude Fable 5.1</label>" in rendered


def test_arena_does_not_compare_different_briefs_or_stale_data():
    index = {"suite_id": "eval-test", "suite_sha256": "a" * 64,
             "arms": [{"key": "beam-hf"}, {"key": "model-b"}]}
    docs = {
        "beam-hf": {"tasks": {"shared-task": {"brief": "Brief A", "attempts": []}}},
        "model-b": {"tasks": {"shared-task": {"brief": "Brief B", "attempts": []}}},
    }
    rendered = render_arena(index, docs, suite_id="eval-test", suite_sha256="a" * 64,
                            prompt_id="shared-task", replay_prompt_id="shared-task")
    assert "not compared" in rendered
    stale = render_arena(index, docs, suite_id="eval-other", suite_sha256="a" * 64,
                         prompt_id="shared-task", replay_prompt_id="shared-task")
    assert "unavailable" in stale


def test_arena_brief_picker_pre_renders_six_saved_briefs_for_client_side_switching():
    suite_tasks = [
        {"id": task_id, "state": f"State {index}", "level": "easy", "brief": f"Saved brief {index}"}
        for index, task_id in enumerate(ARENA_PICKER_TASK_IDS)
    ]
    index = {"suite_id": "eval-test", "suite_sha256": "a" * 64,
             "arms": [{"key": "beam-hf", "label": "Beam"},
                      {"key": "claude-haiku-5-5", "label": "Claude Haiku 5.5"}]}
    tasks = {
        task_id: {"brief": f"Saved brief {i}", "passed_at": 1,
                  "attempts": [{"index": 1, "passed": True, "failing_checks": []}]}
        for i, task_id in enumerate(ARENA_PICKER_TASK_IDS)
    }
    docs = {"beam-hf": {"tasks": tasks}, "claude-haiku-5-5": {"tasks": tasks}}

    rendered = render_arena_picker(
        index, docs, suite_tasks, suite_id="eval-test", suite_sha256="a" * 64,
        replay_prompt_id=ARENA_PICKER_TASK_IDS[0], selected_prompt_id=ARENA_PICKER_TASK_IDS[0],
        selected_model_keys=["beam-hf"],
    )

    assert 'data-testid="arena-brief-picker"' in rendered
    assert rendered.count('data-arena-brief-panel=') == 6
    assert rendered.count('data-arena-model-choice="beam-hf"') == 1
    assert 'data-testid="arena-brief-option-hp_kinnauri-shawl-geometric-border-easy-8" selected' in rendered
    for task_id in ARENA_PICKER_TASK_IDS:
        assert f'data-testid="arena-brief-panel-{task_id}"' in rendered
        assert f'data-testid="arena-brief-option-{task_id}"' in rendered
    assert rendered.count(' data-arena-brief-panel="') == 6
    assert 'data-action="arena_task"' not in rendered
