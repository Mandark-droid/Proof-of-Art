from pramaan import app


def test_try_it_starts_with_g_column_example_result_and_collapsed_advanced_rules():
    state = app._initial_custom_state()
    result = state["result"]

    assert state["form"]["state"] == "Gujarat"
    assert state["form"]["tradition"] == "gj_bandhani"
    assert "bandhani-inspired" in state["form"]["brief"].lower()
    assert "indigo ground" in state["form"]["brief"].lower()
    assert result.get("image") is not None
    assert result.get("source")
    assert result.get("certificate")
    rendered = app._render_custom_app(state)
    studio = rendered.split('data-tab-panel="studio"', 1)[1].split('data-tab-panel="rl"', 1)[0]

    assert 'class="pramaan-ui-preview"' in studio
    assert 'data-local-panel="checks" aria-pressed="true"' in studio
    assert 'data-panel="checks"><ul class="pramaan-ui-certificate-checks"' in studio
    assert 'data-panel="source" hidden' in studio
    assert '<details class="pramaan-ui-advanced-rules">' in studio
    assert '<summary>Advanced rules</summary>' in studio
    assert "Load example" in studio and "Generate with Beam" in studio and "Recheck my edit" in studio
    assert studio.index('data-action="generate"') < studio.index('data-field="palette"')


def test_build_with_it_leads_with_environment_and_dataset_resources_and_collapses_sources():
    rendered = app._render_custom_app(app._initial_custom_state())
    build = rendered.split('data-tab-panel="rl"', 1)[1]

    assert build.index("Use the verifier as an environment") < build.index("Evaluation dataset")
    assert build.index("Training dataset") < build.index("Tradition references")
    assert "https://huggingface.co/collections/kshitijthakkar/pramaan-proof-of-art" in build
    assert "https://github.com/Mandark-droid/Proof-of-Art" in build
    assert "Add and run a model" in build and "--model" in build
    assert '<details class="pramaan-ui-sources">' in build
    assert '<summary>Sources</summary>' in build
    assert '<details class="pramaan-ui-diwali">' in build


def test_add_and_run_a_model_is_a_full_width_row_below_the_four_resource_cards():
    rendered = app._render_custom_app(app._initial_custom_state())
    build = rendered.split('data-tab-panel="rl"', 1)[1]
    resources = build.split('<section aria-label="Public datasets and source code">', 1)[1].split('</section>', 1)[0]
    grid, runner = resources.split('</div><article class="pramaan-ui-card pramaan-builder-runner">', 1)

    assert grid.count('<article class="pramaan-ui-card">') == 4
    assert "Add and run a model" in runner
    assert "--model" in runner
    assert "Open the source repository" in runner
    assert runner.endswith("</article>")


def test_studio_checks_are_plain_language_first_with_certificate_json_collapsed():
    state = app._initial_custom_state()
    rendered = app._render_custom_app(state)
    studio = rendered.split('data-tab-panel="studio"', 1)[1].split('data-tab-panel="rl"', 1)[0]
    checks = studio.split('data-panel="checks">', 1)[1].split('data-panel="attempts"', 1)[0]

    assert "Readable pattern" in checks
    assert "Symmetric" in checks
    assert "At most 3 colours" in checks
    assert "Dots keep their distance" in checks
    assert '<details class="pramaan-ui-certificate-details"><summary>Technical details</summary>' in checks
    assert "mask_sha256" in checks
    assert checks.index("Readable pattern") < checks.index('<summary>Technical details</summary>')
    assert "mask_sha256" not in checks.split("<details", 1)[0]


def test_home_has_one_visible_replay_artwork_plain_language_checks_and_readable_wall_labels():
    rendered = app._render_custom_app(app._initial_custom_state())
    home = rendered.split('data-tab-panel="benchmark"', 1)[1].split('data-tab-panel="leaderboard"', 1)[0]

    assert home.count('data-testid="proof-replay"') == 1
    assert ".pramaan-proof-step .pramaan-proof-tile { display: none; }" in app.PRAMAAN_CSS
    assert "Symmetric" in home
    assert "At most 4 colours" in home
    assert "Dots keep their distance" in home
    assert '<summary>Technical check details</summary>' in home
    assert "36 states, one tile each" in home
    assert "Tap a tile" in home
    assert "See how 10 models compare" in home
    for short_name in ("J&amp;K", "DNH &amp; DD", "Arunachal", "Andaman"):
        assert short_name in home
