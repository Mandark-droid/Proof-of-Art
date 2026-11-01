import base64
import re

from pramaan.app import HEADLINE, _initial_custom_state, _proof_replay_html, _render_custom_app


def test_first_screen_uses_shared_headline_and_replay_is_included():
    rendered = _render_custom_app(_initial_custom_state())

    assert HEADLINE == "AI drew it. Math checked it."
    assert HEADLINE in rendered
    assert "Proof of art that you can verify." in rendered
    assert 'data-testid="proof-replay"' in rendered
    assert 'data-testid="proof-replay-play"' in rendered
    assert 'data-testid="proof-replay-step"' in rendered
    assert 'data-testid="proof-verify-yourself"' in rendered
    assert "Try it yourself" in rendered


def test_saved_replay_renders_reverified_attempts_witness_and_certificate():
    rendered = _proof_replay_html()

    assert "Gujarat" in rendered
    assert "Bandhani" in rendered
    assert "Dot Field" in rendered
    assert "gj_bandhani" not in rendered
    assert "dot_field" not in rendered
    assert "dark red field with a diamond of 12 small white dots" in rendered
    assert "Attempt 1 · did not pass" in rendered
    assert "What needs attention · attempt 1" in rendered
    assert "dot_clearance" in rendered
    assert "Dots keep their distance" in rendered
    assert '<summary>Technical check details</summary>' in rendered
    assert "is-fail" in rendered and "is-pass" in rendered
    assert "Repair · all checks pass" in rendered
    assert "Mask SHA-256" in rendered
    match = re.search(r'data-pattern-b64="([A-Za-z0-9+/=]+)"', rendered)
    assert match
    assert b'"grammar":"dot_field"' in base64.b64decode(match.group(1))


def test_try_it_uses_plain_language_pattern_type_label():
    rendered = _render_custom_app(_initial_custom_state())

    assert "Pattern type: Dot field" in rendered
    assert "Grammar: <code>dot_field</code>" not in rendered


def test_home_replay_starts_on_failed_attempt_and_removes_repeated_tile_caption():
    from pramaan import app

    rendered = app._proof_replay_html()
    first_step = re.search(
        r'<div class="pramaan-proof-step" data-proof-step="1"([^>]*)>(.*?)</div>', rendered, re.S,
    )
    result_step = re.search(
        r'<div class="pramaan-proof-step" data-proof-step="\d+" data-proof-result([^>]*)>', rendered,
    )
    first_image = re.search(r'<img class="pramaan-proof-tile" src="([^"]+)"', first_step.group(2) if first_step else "")
    active_image = re.search(r'<img class="pramaan-proof-tile" data-proof-active-tile src="([^"]+)"', rendered)

    assert first_step and "hidden" not in first_step.group(1)
    assert result_step and "hidden" in result_step.group(1)
    assert first_image and active_image and first_image.group(1) == active_image.group(1)
    assert "Repair · all checks pass" in rendered
    assert "data-proof-figure-caption" not in rendered


def test_initial_failed_attempt_names_failed_check_and_marks_its_witness_on_tile():
    from pramaan import app

    rendered = app._proof_replay_html()
    first_step = re.search(
        r'<div class="pramaan-proof-step" data-proof-step="1"[^>]*>(.*?)</div>', rendered, re.S,
    )
    assert first_step
    assert "Dots keep their distance: failed" in first_step.group(1)
    # The witness is two motif centres, not a pixel coordinate. Both are circled
    # on the initial image so the opening replay frame contains visible evidence.
    image = re.search(
        r'<img class="pramaan-proof-tile" src="data:image/png;base64,([^"]+)" data-proof-witness-mark',
        first_step.group(1),
    )
    assert image
    from io import BytesIO
    from PIL import Image
    marked_tile = Image.open(BytesIO(base64.b64decode(image.group(1)))).convert("RGB")
    assert sum(1 for pixel in marked_tile.get_flattened_data() if pixel == (209, 47, 138)) > 0


def test_every_proof_replay_step_has_visible_explanatory_content():
    rendered = _proof_replay_html()
    steps = re.findall(
        r'<div class="pramaan-proof-step" data-proof-step="(\d+)"([^>]*)>(.*?)</div>',
        rendered, re.S,
    )
    assert len(steps) == 6
    for _index, _attrs, body in steps:
        text = re.sub(r'<[^>]+>', ' ', body)
        assert re.search(r'\w', text) or '<img ' in body
    assert "Dots keep their distance: failed" in steps[1][2]


def test_mobile_proof_replay_uses_the_large_tile_once_and_wall_details_are_collapsed():
    from pramaan import app

    css = app.PRAMAAN_CSS
    rendered = app._render_custom_app(_initial_custom_state())
    wall = rendered.split('class="pramaan-home-wall"', 1)[1].split('</section>', 1)[0]

    assert ".pramaan-proof-step .pramaan-proof-tile { display: none; }" in css
    assert "<summary>How to read this</summary>" in wall
    assert "Tap a tile for its verified reference and model results" in wall
    assert "Marks show easy and hard outcomes" in wall
    assert "not visual similarity or cultural authenticity" in wall
    assert "36 states, one tile each" in wall
    assert ".pramaan-wall-mark { font-size: 12px;" in css
    assert ".pramaan-wall-legend li { align-items: center; display: inline-flex; gap: .35rem; font-size: 13px; }" in css
    assert ".pramaan-wall-state-name { bottom: auto;" in css
    assert "unsatisfiable" in wall
    intro = re.search(r'<p>Tap a tile for its verified reference.*?</p>', wall)
    assert intro and "easy and hard outcomes" in intro.group(0)


def test_home_metrics_name_beam_as_the_evaluated_model():
    from pramaan.app import _home_metrics_html

    rendered = _home_metrics_html()

    assert "Beam designs correct on the first try" in rendered
    assert "Beam designs correct after checker feedback" in rendered
