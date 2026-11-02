import base64
import json
import re

from pramaan.app import (
    HEADLINE,
    PRAMAAN_CSS,
    SUBTITLE,
    _initial_custom_state,
    _proof_replay_html,
    _render_custom_app,
)


def test_first_screen_uses_shared_headline_and_replay_is_included():
    rendered = _render_custom_app(_initial_custom_state())

    assert HEADLINE == "AI drew it. Math checked it."
    assert SUBTITLE == "Proof of art that you can verify."
    assert HEADLINE in rendered
    home_intro = re.search(r'<div class="pramaan-home-intro">(.*?)</div>', rendered, re.S)
    assert home_intro
    assert home_intro.group(1).startswith(f"<h1>{HEADLINE}</h1><p>{SUBTITLE} ")
    assert rendered.count(f"<h1>{HEADLINE}</h1>") == 1
    assert "Art you can prove." not in rendered
    assert 'data-testid="proof-replay"' in rendered
    assert 'data-testid="proof-replay-play"' in rendered
    assert 'data-testid="proof-replay-step"' in rendered
    assert 'data-testid="proof-verify-yourself"' in rendered
    assert rendered.count('data-testid="proof-replay"') == 1
    assert "Try it yourself" in rendered
    assert ".pramaan-proof-replay h1" in PRAMAAN_CSS
    assert "text-wrap: balance" in PRAMAAN_CSS
    home_heading = PRAMAAN_CSS.split(".pramaan-home-intro h1", 1)[1].split("}", 1)[0]
    assert "font-size: clamp(2rem,4vw,3.4rem)" in home_heading
    assert "text-wrap: balance" in home_heading


def test_saved_replay_renders_reverified_attempts_witness_and_certificate():
    rendered = _proof_replay_html()

    assert "Himachal Pradesh" in rendered
    assert "Art you can prove." not in rendered
    assert "pramaan-proof-subtitle" not in rendered
    assert "Kinnauri" in rendered
    assert "Kinnauri shawl geometric border" in rendered
    assert "hp_kinnauri" not in rendered
    assert "A brick-red tile with a single stepped white hexagon centred on it" in rendered
    assert "Attempt 1 · did not pass" in rendered
    assert "What needs attention · attempt 1" in rendered
    assert "symmetry_d2" in rendered
    assert "Symmetric" in rendered
    assert '<summary>Technical check details</summary>' in rendered
    assert "is-fail" in rendered and "is-pass" in rendered
    assert "Attempt 3 · all checks pass" in rendered
    assert "Pixels SHA-256" in rendered
    assert 'data-proof-comparison-failed' in rendered
    assert 'data-proof-comparison-passing' in rendered
    assert 'data-witness-transform="F_vertical"' in rendered
    assert 'data-proof-card' in rendered
    assert "Proof of Art" in rendered
    assert "9 of 9 checks passed" in rendered
    assert "Anyone can recompute this. It is not a digital signature." in rendered
    assert "cryptographic" not in rendered and "immutable" not in rendered
    match = re.search(r'data-pattern-b64="([A-Za-z0-9+/=]+)"', rendered)
    assert match
    assert b'"style":"ajrakh_inspired"' in base64.b64decode(match.group(1))


def test_home_witness_summary_and_marks_come_from_replayed_saved_candidates():
    from pramaan.app import EVAL_TRAJECTORY_PATH, PROOF_REPLAY_TRAJECTORY_ID, parse_pattern, verify_pattern

    record = next(
        row for line in EVAL_TRAJECTORY_PATH.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line)).get("trajectory_id") == PROOF_REPLAY_TRAJECTORY_ID
    )
    replayed = []
    for call in record["calls"]:
        pattern = parse_pattern(json.dumps(call["candidate"], ensure_ascii=False))
        checks, _, _ = verify_pattern(pattern)
        replayed.append((call, checks))
    failed_call, failed_checks = next(
        (call, checks) for call, checks in replayed
        if any(check.get("status") == "FAIL" for check in checks)
    )
    passing_call, _ = next(
        (call, checks) for call, checks in reversed(replayed)
        if all(check.get("status") == "PASS" for check in checks)
    )
    failure = next(check for check in failed_checks if check.get("status") == "FAIL")
    witness = failure["counterexample"]
    original, transformed = witness["original"], witness["transformed"]
    relation = "its mirror" if witness["transform"] == "F_vertical" else "its transformed pixel"
    expected = (
        f"Pixel ({original[0]}, {original[1]}) and {relation} "
        f"({transformed[0]}, {transformed[1]}) differ · reward "
        f"{failed_call['reward']} → {passing_call['reward']}"
    )
    rendered = _proof_replay_html()

    assert expected in rendered
    assert f'data-witness-transform="{witness["transform"]}"' in rendered
    assert 'data-proof-comparison data-witness-transform=' in rendered


def test_readme_and_standalone_replay_use_the_shared_branding_copy():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    replay = (root / "assets" / "proof-replay.html").read_text(encoding="utf-8")

    assert f"**{HEADLINE}**" in readme
    assert f"*{SUBTITLE}*" in readme
    assert f'<h1>{HEADLINE}</h1>' in replay
    assert "Art you can prove." not in replay


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
    assert "Attempt 3 · all checks pass" in rendered
    assert "data-proof-figure-caption" not in rendered


def test_initial_failed_attempt_names_failed_check_and_marks_its_witness_on_tile():
    from pramaan import app

    rendered = app._proof_replay_html()
    first_step = re.search(
        r'<div class="pramaan-proof-step" data-proof-step="1"[^>]*>(.*?)</div>', rendered, re.S,
    )
    assert first_step
    assert "Symmetric: failed" in first_step.group(1)
    # Distinct replay-derived colors mark the original, mirror, and mirror axis.
    image = re.search(
        r'<img class="pramaan-proof-tile"[^>]*src="data:image/png;base64,([^"]+)"',
        first_step.group(1),
    )
    assert image
    from io import BytesIO
    from PIL import Image
    marked_tile = Image.open(BytesIO(base64.b64decode(image.group(1)))).convert("RGB")
    pixels = set(marked_tile.get_flattened_data())
    assert (255, 45, 149) in pixels
    assert (0, 229, 255) in pixels
    magenta_pixels = sum(pixel == (255, 45, 149) for pixel in marked_tile.get_flattened_data())
    assert magenta_pixels >= 180
    assert all(marked_tile.getpixel((marked_tile.width // 2, y)) == (255, 230, 0)
               for y in range(marked_tile.height))


def test_every_proof_replay_step_has_visible_explanatory_content():
    rendered = _proof_replay_html()
    steps = re.findall(
        r'<div class="pramaan-proof-step" data-proof-step="(\d+)"([^>]*)>(.*?)</div>',
        rendered, re.S,
    )
    assert len(steps) == 7
    for _index, _attrs, body in steps:
        text = re.sub(r'<[^>]+>', ' ', body)
        assert re.search(r'\w', text) or '<img ' in body
    assert "Symmetric: failed" in steps[1][2]


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

    assert "Beam designs passing every check on the first try" in rendered
    assert "Beam designs passing every check after checker feedback" in rendered
