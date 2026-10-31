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


def test_mobile_proof_replay_uses_the_large_tile_once_and_wall_details_are_collapsed():
    from pramaan import app

    css = app.PRAMAAN_CSS
    rendered = app._render_custom_app(_initial_custom_state())
    wall = rendered.split('class="pramaan-home-wall"', 1)[1].split('</section>', 1)[0]

    assert ".pramaan-proof-step .pramaan-proof-tile { display: none; }" in css
    assert "<summary>How to read this</summary>" in wall
    assert "Tap a tile to inspect its verified reference" in wall
    assert "two marks show easy and hard task outcomes" in wall
    assert "do not score visual similarity" in wall
    assert "36 states, one tile each" in wall
    assert ".pramaan-wall-mark { font-size: 12px;" in css
    assert ".pramaan-wall-legend li { align-items: center; display: inline-flex; gap: .35rem; font-size: 13px; }" in css
    assert ".pramaan-wall-state-name { font-size: 12px; }" in css
    assert "unsatisfiable" in wall
    intro = re.search(r'<p>Tap a tile to inspect.*?</p>', wall)
    assert intro and "easy and hard task outcomes" in intro.group(0)


def test_home_metrics_name_beam_as_the_evaluated_model():
    from pramaan.app import _home_metrics_html

    rendered = _home_metrics_html()

    assert "Beam designs correct on the first try" in rendered
    assert "Beam designs correct after checker feedback" in rendered
