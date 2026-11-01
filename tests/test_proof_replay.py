import base64
import re

from pramaan.app import HEADLINE, _initial_custom_state, _proof_replay_html, _render_custom_app


def test_first_screen_uses_shared_headline_and_replay_is_included():
    rendered = _render_custom_app(_initial_custom_state())

    assert HEADLINE == "Art you can prove."
    assert HEADLINE in rendered
    assert 'data-testid="proof-replay"' in rendered
    assert 'data-testid="proof-replay-play"' in rendered
    assert 'data-testid="proof-replay-step"' in rendered
    assert 'data-testid="proof-verify-yourself"' in rendered
    assert "Try it yourself" in rendered


def test_saved_replay_renders_reverified_attempts_witness_and_certificate():
    rendered = _proof_replay_html()

    assert "Gujarat" in rendered
    assert "gj_bandhani" in rendered
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
