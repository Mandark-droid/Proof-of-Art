def test_build_tab_uses_readme_sync_client_form_and_prioritizes_web_link():
    from pramaan.app import _render_custom_app

    rendered = _render_custom_app({"active_tab": "rl"})
    build = rendered.split('data-tab-panel="rl"', 1)[1]

    assert 'PramaanEnv(base_url="https://kshitijthakkar-pramaan-proof-of-art.hf.space").sync()' in build
    assert 'href="/web/"' in build
    assert 'class="pramaan-ui-button primary" href="/web/"' in build
    assert build.index('data-testid="openenv-contract"') < build.index("How the benchmark is measured")
    assert "Reward function tested." in build
    assert "No model has been trained on the Pramaan environment yet." in build
    assert "train-v1" in build and "held-out evaluation suite" in build
    assert 'data-testid="known-visibility-gap"' in build
    assert "mn_moirang-phee-moirang-pheejin-stepped-triangle-border-easy-14" in build
    assert "Accepted design · reward 1.0" in build


def test_models_visibility_note_links_to_saved_manipur_example():
    from pramaan.app import (_benchmark_summary, _home_suite_tasks, _load_gallery_arm_documents,
                             _load_gallery_index)
    from pramaan.ui.leaderboard import render_model_report

    backgrounds = {task["id"]: task["palette"][0] for task in _home_suite_tasks()}
    rendered = render_model_report(
        _benchmark_summary(), gallery_documents=_load_gallery_arm_documents(_load_gallery_index()),
        task_backgrounds=backgrounds,
    )

    assert 'href="#/rl">See the Manipur example in Build with it.</a>' in rendered
