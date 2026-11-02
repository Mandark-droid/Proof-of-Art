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
    assert 'data-panel="checks"><section class="pramaan-proof-card is-proven"' in studio
    assert 'data-panel="source" hidden' in studio
    assert '<details class="pramaan-ui-advanced-rules">' in studio
    assert '<summary>Advanced rules</summary>' in studio
    assert "Load example" in studio and "Generate with Beam" in studio and "Recheck my edit" in studio
    assert studio.index('data-action="generate"') < studio.index('data-field="palette"')


def test_build_with_it_leads_with_environment_and_dataset_resources_and_collapses_sources():
    rendered = app._render_custom_app(app._initial_custom_state())
    build = rendered.split('data-tab-panel="rl"', 1)[1]

    assert build.index("Use the verifier as an environment") < build.index("Evaluation dataset")
    assert 'class="pramaan-ui-button primary" href="/web/"' in build
    assert 'with PramaanEnv(base_url="https://kshitijthakkar-pramaan-proof-of-art.hf.space").sync() as env:' in build
    assert 'task_id="hp_kinnauri-shawl-geometric-border-easy-8"' in build
    assert build.index("Training dataset") < build.index("Tradition references")
    assert "https://huggingface.co/collections/kshitijthakkar/pramaan-proof-of-art" in build
    assert "https://github.com/Mandark-droid/Proof-of-Art" in build
    assert "Add and run a model" in build and "--model" in build
    assert "benchmarks/run.py --suite benchmarks/suites/eval-v2.1.jsonl --model your-model-key --limit 12 --output benchmarks/results/my-run.jsonl" in build
    assert "-m scripts.pack_submission benchmarks/results/my-run.jsonl benchmarks/results/my-run.zip" in build
    assert "scripts/run_arms.py" not in build
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
    assert "Proof of Art" in checks
    assert "Copy certificate JSON" in checks
    assert checks.index("Readable pattern") < checks.index("Copy certificate JSON")
    assert "mask_sha256" not in checks and "dsl_sha256" not in checks


def test_passing_proof_card_matches_certificate_and_keeps_exclusions_visible():
    import base64
    import json
    import re
    from pramaan.certificate import make_certificate
    from pramaan.ui.proof_card import render_proof_card

    state = app._initial_custom_state()
    pattern, checks, masks, grid, expected = app._verify_json(state["result"]["source"])
    certificate = make_certificate(pattern, checks, masks, grid)
    rendered = render_proof_card(
        certificate, image_uri=app._image_data_uri(state["result"]["image"]),
        state=state["form"]["state"], tradition=app._tradition_by_id(state["form"]["tradition"]).name,
        reward=1.0, download_html='<a download="run.zip">Download run bundle</a>',
    )

    assert expected == certificate
    assert "Proof of Art" in rendered
    assert f'{len(checks)} of {len(checks)} checks passed' in rendered
    assert f'Design SHA-256</dt><dd><code>{certificate["dsl_sha256"][:12]}…</code>' in rendered
    assert f'Pixels SHA-256</dt><dd><code>{certificate["mask_sha256"][:12]}…</code>' in rendered
    assert certificate["dsl_sha256"] not in rendered
    assert certificate["mask_sha256"] not in rendered
    assert "Copy certificate JSON" in rendered
    match = re.search(r'data-certificate-b64="([A-Za-z0-9+/=]+)"', rendered)
    assert match and json.loads(base64.b64decode(match.group(1))) == certificate
    for exclusion in certificate["exclusions"]:
        assert exclusion in rendered
    assert "Anyone can recompute this. It is not a digital signature." in rendered
    assert "Download run bundle" in rendered


def test_failing_proof_card_shows_witness_but_no_design_or_pixel_hash():
    from pramaan.ui.proof_card import render_proof_card

    certificate = {
        "all_hard_constraints_pass": False,
        "checks": [{"id": "symmetry_d2", "status": "FAIL", "message": "Colour mismatch",
                    "counterexample": {"original": [1, 2], "transformed": [3, 2],
                                       "transform": "F_vertical"}}],
        "exclusions": ["Not a proof of cultural authenticity"],
        "dsl_sha256": "d" * 64, "mask_sha256": "m" * 64,
    }
    rendered = render_proof_card(certificate, state="Himachal Pradesh", tradition="Kinnauri")

    assert "Not proven" in rendered
    assert "Symmetry (D2)" in rendered
    assert "Pixel (1, 2) and its mirror (3, 2) differ" in rendered
    assert "dsl_sha256" not in rendered and "mask_sha256" not in rendered
    assert "dddddddddddd" not in rendered and "mmmmmmmmmmmm" not in rendered
    assert certificate["exclusions"][0] in rendered


def test_try_it_pairs_preview_and_proof_card_on_desktop_and_stacks_them_on_phone():
    css = app.PRAMAAN_CSS
    rendered = app._render_custom_app(app._initial_custom_state())
    studio = rendered.split('data-tab-panel="studio"', 1)[1].split('data-tab-panel="rl"', 1)[0]
    checks = studio.split('data-panel="checks">', 1)[1].split('data-panel="attempts"', 1)[0]

    assert "max-width: 420px" in css
    assert "@media (min-width: 1100px)" in css
    assert '.pramaan-ui-result > [data-panel="checks"] { grid-column: 2; grid-row: 4;' in css
    assert '.pramaan-ui-result > .pramaan-ui-preview, .pramaan-ui-result > .pramaan-ui-placeholder { grid-column: 1; grid-row: 4;' in css
    assert "@media (max-width: 1099px)" in css
    assert ".pramaan-ui-result { display: block; }" in css
    assert '<section class="pramaan-proof-card is-proven"' in checks


def test_home_has_one_visible_replay_artwork_plain_language_checks_and_readable_wall_labels():
    rendered = app._render_custom_app(app._initial_custom_state())
    home = rendered.split('data-tab-panel="benchmark"', 1)[1].split('data-tab-panel="leaderboard"', 1)[0]

    assert home.count('data-testid="proof-replay"') == 1
    assert ".pramaan-proof-step .pramaan-proof-tile { display: none; }" in app.PRAMAAN_CSS
    assert "Symmetric" in home
    assert "At most 4 colours" in home
    assert "Motifs keep their distance" in home
    assert '<summary>Technical check details</summary>' in home
    assert "36 states, one tile each" in home
    assert "Tap a tile" in home
    assert "See how 10 models compare" not in home
    for short_name in ("J&amp;K", "DNH &amp; DD", "Arunachal", "Andaman"):
        assert short_name in home
