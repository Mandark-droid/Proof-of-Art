from types import SimpleNamespace
import hashlib
import json
import re
import uuid
import zipfile
from pathlib import Path
import pytest

from PIL import Image

from pramaan import app
from pramaan.checks.core import verify_pattern
from pramaan.dsl import parse_pattern
from pramaan.certificate import make_certificate
from pramaan.export import trajectory_bundle
from pramaan.tasks import canonical_jsonl_bytes


def test_local_app_config_check_uses_lazy_lock_factory_on_windows(monkeypatch):
    import asyncio
    import gradio.queueing
    import gradio.utils
    import scripts.check_app_configuration as check_app

    monkeypatch.setattr(check_app.sys, "platform", "win32")
    monkeypatch.setattr(gradio.queueing, "safe_get_lock", gradio.queueing.safe_get_lock)
    monkeypatch.setattr(gradio.utils, "safe_get_lock", gradio.utils.safe_get_lock)
    monkeypatch.setattr(gradio.utils, "safe_get_stop_event", gradio.utils.safe_get_stop_event)
    check_app.install_windows_gradio_lock_shim()

    assert gradio.queueing.safe_get_lock is asyncio.Lock
    assert gradio.utils.safe_get_lock is asyncio.Lock
    assert gradio.utils.safe_get_stop_event is asyncio.Event


def test_execution_trace_bundle_preserves_failed_attempts_without_credentials(monkeypatch, tmp_path):
    import pramaan.export as export

    monkeypatch.setattr(export, "_EXPORT_DIR", tmp_path)
    record = {
        "trajectory_id": "studio-run-123",
        "data_role": "unassigned",
        "training_eligible": False,
        "model": "beam-hf",
        "original_prompt": "Indigo geometric repeat",
        "attempts": [{"index": 1, "candidate": None,
                       "report": {"checks": [{"id": "candidate_parse", "status": "FAIL"}]}}],
        "model_calls": [{"call_index": 1, "raw_response": "not json",
                         "telemetry": {"token_usage": {"total_tokens": 37}}}],
        "api_key": "must-not-be-exported",
    }

    path = trajectory_bundle(record)

    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {"trajectory.json", "README.txt"}
        exported = json.loads(archive.read("trajectory.json"))
    assert exported["trajectory_id"] == "studio-run-123"
    assert exported["attempts"] == record["attempts"]
    assert exported["model_calls"] == record["model_calls"]
    assert exported["training_eligible"] is False
    assert "api_key" not in exported


def test_failed_studio_generation_returns_a_downloadable_trace(monkeypatch, tmp_path):
    import pramaan.export as export

    _stub_state_tradition_ui(monkeypatch)
    monkeypatch.setattr(export, "_EXPORT_DIR", tmp_path)
    monkeypatch.setattr(app, "generate_and_verify", lambda **_kwargs: {
        "trajectory_schema_version": "1.1",
        "model": "beam-hf",
        "created_at": "2026-10-30T00:00:00+00:00",
        "constraint_profile": {"symmetry": "D4"},
        "elapsed_ms": 8,
        "telemetry": {},
        "provider_error": {"phase": "proposal", "type": "Timeout"},
        "model_calls": [],
        "attempts": [],
        "provider_attempts": 0,
        "accepted": False,
    })

    values = app.generate(
        "A tiny dot composition", "Gujarat", "gj_bandhani", "default", "D4", 1,
        0.02, 0.7, False, 0, 1, 4, "x", 8, 0, 0, 0,
    )

    assert values[5] == "Not verified · Beam request failed (Timeout)"
    with zipfile.ZipFile(values[6]) as archive:
        trace = json.loads(archive.read("trajectory.json"))
    assert trace["original_prompt"] == "A tiny dot composition"
    assert trace["data_role"] == "unassigned"
    assert trace["training_eligible"] is False


def _stub_state_tradition_ui(monkeypatch):
    traditions = {
        "gj_bandhani": SimpleNamespace(
            id="gj_bandhani", name="Bandhani", states=("Gujarat",),
            grammar="dot_field", palette_presets={"default": ["#172F4A"]},
            default_profile={"symmetry": "D4"}, sources=[], sources_unverified=[],
            confidence="high", cautions="Synthetic fixture.",
        ),
        "gj_zardozi": SimpleNamespace(
            id="gj_zardozi", name="Zardozi", states=("Gujarat",),
            grammar="counted_grid", palette_presets={"default": ["#172F4A"]},
            default_profile={"symmetry": "D2"}, sources=[], sources_unverified=[],
            confidence="high", cautions="Synthetic fixture.",
        ),
        "tn_sikku_kolam": SimpleNamespace(
            id="tn_sikku_kolam", name="Sikku Kolam", states=("Tamil Nadu",),
            grammar="kolam_loop", palette_presets={"default": ["#172F4A"]},
            default_profile={"symmetry": "D2"}, sources=[], sources_unverified=[],
            confidence="high", cautions="Synthetic fixture.",
        ),
    }
    examples = {
        ("Gujarat", "gj_bandhani"): "Bandhani Dot Field",
        ("Gujarat", "gj_zardozi"): "Zardozi Grid",
        ("Tamil Nadu", "tn_sikku_kolam"): "Sikku Kolam Loop",
    }
    monkeypatch.setattr(app, "_tradition_by_id", lambda key: traditions.get(key))
    monkeypatch.setattr(app, "_tradition_choices", lambda state: [
        item for item in traditions.values() if state in item.states
    ])
    monkeypatch.setattr(app, "_example_choices", lambda state, tradition_id: (
        [(f"{examples[(state, tradition_id)]} · {state} / {traditions[tradition_id].name}",
          examples[(state, tradition_id)])]
        if (state, tradition_id) in examples else []
    ))
    return traditions


def _stub_custom_render_sources(monkeypatch):
    snapshot = {"status": "Community results unavailable in this test.", "rows": [],
                "beam_rows": [], "selected": {}, "summary": {}}
    monkeypatch.setattr(app, "_community_model_choices", lambda: [])
    monkeypatch.setattr(app, "_community_comparison_snapshot", lambda: snapshot)
    monkeypatch.setattr(app, "_benchmark_state_view", lambda _state: ([], []))
    monkeypatch.setattr(app, "_festival_lens_states", lambda: set())
    monkeypatch.setattr(app, "_load_gallery_index", lambda: None)
    monkeypatch.setattr(app, "_benchmark_summary_html", lambda: "")


def test_periodic_raster_profile_uses_an_available_symmetry(monkeypatch):
    tradition = SimpleNamespace(
        id="test_periodic",
        name="Test periodic tradition",
        grammar="periodic_raster",
        palette_presets={"test": ["#172F4A"]},
        default_profile={"symmetry": "C4"},
    )
    monkeypatch.setattr(app, "_tradition_by_id", lambda _tradition_id: tradition)

    result = app._tradition_defaults("test_periodic")

    assert result[2]["value"] in result[2]["choices"]
    assert result[2]["value"] == "D2"


def test_launch_demo_applies_gradio6_theme_and_css(monkeypatch):
    captured = {}

    class Demo:
        def launch(self, **kwargs):
            captured.update(kwargs)
            return "launched"

    monkeypatch.setattr(app, "build_demo", lambda: Demo())

    assert app.launch_demo() == "launched"
    assert captured["theme"] is not None
    assert captured["css"] == app.PRAMAAN_CSS


def test_brand_header_uses_embedded_logo_without_static_request():
    header = app._brand_header_html()

    assert '<img class="pramaan-logo" src="data:image/png;base64,' in header
    assert "gradio_api/file=" not in header + app.PRAMAAN_CSS


def test_brand_logo_is_embedded_as_a_small_retina_png():
    import base64
    from io import BytesIO
    from PIL import Image

    uri = app._brand_logo_data_uri()
    assert uri.startswith("data:image/png;base64,")
    payload = base64.b64decode(uri.partition(",")[2], validate=True)
    with Image.open(BytesIO(payload)) as logo:
        assert logo.size == (216, 216)
    assert len(payload) < 40_000


def test_brand_logo_lfs_pointer_falls_back_to_text_wordmark(monkeypatch, tmp_path, caplog):
    pointer = tmp_path / "pramaan-mark.png"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "0" * 64 + "\n"
        "size 582760\n",
        encoding="ascii",
    )
    monkeypatch.setattr(app, "_brand_logo_path", lambda: pointer)
    app._brand_logo_data_uri.cache_clear()
    try:
        assert app._brand_logo_data_uri() == ""
        header = app._brand_header_html()
    finally:
        app._brand_logo_data_uri.cache_clear()

    assert '<h1 class="pramaan-brand-name">Pramaan <em>Proof of Art</em></h1>' in header
    assert '<img class="pramaan-logo"' not in header
    assert "text-only header" in caplog.text


def test_proof_seal_is_outlined_and_included_in_downloadable_art_bundles(tmp_path):
    import xml.etree.ElementTree as ET

    seal_path = app.ROOT / "assets" / "pramaan-seal.svg"
    root = ET.parse(seal_path).getroot()
    tags = [node.tag.rsplit("}", 1)[-1] for node in root.iter()]
    assert "path" in tags and "circle" in tags
    assert "text" not in tags
    assert "प्रमाण" in "".join(node.text or "" for node in root.iter())
    assert app._brand_favicon_path().is_file()
    with Image.open(app._brand_favicon_path()) as icon:
        assert icon.size == (64, 64)
        favicon = icon.convert("RGBA")
    with Image.open(app._brand_logo_path()) as source:
        source = source.convert("RGBA")
        source.thumbnail((64, 64), Image.Resampling.LANCZOS)
    expected = Image.new("RGBA", (64, 64), (255, 253, 249, 255))
    expected.alpha_composite(source, ((64 - source.width) // 2, (64 - source.height) // 2))
    assert favicon.tobytes() == expected.tobytes()

    bundle = tmp_path / "artifact.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("certificate.json", "{}")
    app._add_proof_seal_to_bundle(str(bundle))
    with zipfile.ZipFile(bundle) as archive:
        assert "proof-seal.svg" in archive.namelist()
        assert archive.read("proof-seal.svg") == seal_path.read_bytes()


def test_direct_gradio_launch_uses_the_pramaan_favicon(monkeypatch):
    captured = {}

    class Demo:
        def launch(self, **kwargs):
            captured.update(kwargs)
            return "launched"

    monkeypatch.setattr(app, "build_demo", lambda: Demo())

    assert app.launch_demo() == "launched"
    assert captured["favicon_path"] == str(app._brand_favicon_path())


def test_brand_header_keeps_logo_legible_at_mobile_width():
    header = app._brand_header_html()

    assert '<header class="pramaan-brandbar">' in header
    assert '<h1 class="pramaan-brand-name">Pramaan <em>Proof of Art</em></h1>' in header
    assert 'class="pramaan-brand-seal"' not in header
    assert 'href="#pramaan-proof-seal"' not in header
    assert ".pramaan-brandbar {" in app.PRAMAAN_CSS
    assert "grid-template-columns: 42px minmax(0,1fr)" in app.PRAMAAN_CSS
    assert "grid-template-rows: 42px auto" in app.PRAMAAN_CSS
    assert "grid-column: 2; grid-row: 2; justify-content: flex-start" in app.PRAMAAN_CSS
    assert "grid-template-columns: 38px minmax(0,1fr)" in app.PRAMAAN_CSS
    assert "object-fit: contain" in app.PRAMAAN_CSS
    assert "huggingface.co/spaces/kshitijthakkar/pramaan-proof-of-art/resolve/main/assets/pramaan-mark.png" not in header + app.PRAMAAN_CSS


def test_brand_header_exposes_project_links_without_emoji():
    header = app._brand_header_html()

    assert '<nav class="pramaan-brand-links" aria-label="Project links">' in header
    for label, href in (
        ("Dataset", "https://huggingface.co/datasets/kshitijthakkar/pramaan-proof-of-art-eval-v2.1"),
        ("OpenEnv", "https://kshitijthakkar-pramaan-proof-of-art.hf.space/web/"),
        ("Collection", "https://huggingface.co/collections/kshitijthakkar/pramaan-proof-of-art"),
        ("Code", "https://github.com/Mandark-droid/Proof-of-Art"),
    ):
        assert f'href="{href}"' in header
        assert f">{label}</a>" in header
    assert 'target="_blank" rel="noopener noreferrer"' in header
    assert "@media (max-width: 600px)" in app.PRAMAAN_CSS
    assert ".pramaan-brand-links a { font-size: .64rem; }" in app.PRAMAAN_CSS
    assert not any(ord(character) > 0xFFFF for character in header)


def test_offline_configuration_checker_accepts_the_docker_space_contract(monkeypatch, tmp_path):
    from scripts import check_app_configuration

    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "pramaan_env" / "server").mkdir(parents=True)
    (tmp_path / "README.md").write_text(
        "---\nsdk: docker\napp_port: 8000\nbase_path: /studio\nhf_oauth: true\n"
        "models:\n  - reflection/Beam-1.0-501B-A23B\n---\n",
        encoding="utf-8",
    )
    (tmp_path / "Dockerfile").write_text(
        "RUN uv sync --locked --no-dev\nEXPOSE 8000\n"
        'CMD ["uv", "run", "--no-sync", "--project", ".", "server"]\n',
        encoding="utf-8",
    )
    (tmp_path / "pramaan_env" / "server" / "app.py").write_text("app = None\n", encoding="utf-8")
    (tmp_path / "benchmarks" / "models.json").write_text(json.dumps({
        "schema_version": 1,
        "models": [{"model_id": "reflection/Beam-1.0-501B-A23B:baseten"}],
    }), encoding="utf-8")
    monkeypatch.setattr(check_app_configuration, "ROOT", tmp_path)

    check_app_configuration.check_space_metadata()


def test_offline_configuration_checker_rejects_a_docker_mount_mismatch(monkeypatch, tmp_path):
    import pytest

    from scripts import check_app_configuration

    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "README.md").write_text(
        "---\nsdk: docker\napp_port: 8000\nbase_path: /wrong\nhf_oauth: true\n"
        "models:\n  - reflection/Beam-1.0-501B-A23B\n---\n",
        encoding="utf-8",
    )
    (tmp_path / "Dockerfile").write_text(
        "RUN uv sync --locked --no-dev\nEXPOSE 8000\n"
        'CMD ["uv", "run", "--no-sync", "--project", ".", "server"]\n',
        encoding="utf-8",
    )
    (tmp_path / "pramaan_env" / "server").mkdir(parents=True)
    (tmp_path / "pramaan_env" / "server" / "app.py").write_text("app = None\n", encoding="utf-8")
    (tmp_path / "benchmarks" / "models.json").write_text(json.dumps({
        "schema_version": 1,
        "models": [{"model_id": "reflection/Beam-1.0-501B-A23B:baseten"}],
    }), encoding="utf-8")
    monkeypatch.setattr(check_app_configuration, "ROOT", tmp_path)

    with pytest.raises(SystemExit, match="base_path"):
        check_app_configuration.check_space_metadata()


def test_benchmark_summary_html_shows_per_model_metrics_and_cost_source(monkeypatch):
    monkeypatch.setattr(app, "_benchmark_summary", lambda: {
        "by_model": {
            "beam-hf": {
                "overall": {
                    "n": 72,
                    "validity_at_1": 11 / 72,
                    "validity_at_3": 38 / 72,
                    "reported_cost_totals_by_currency": {},
                    "estimated_cost_totals_by_currency": {"USD": 0.4257},
                    "mean_elapsed_ms": 8580,
                }
            }
        }
    })

    rendered = app._benchmark_summary_html()

    assert 'class="pramaan-benchmark-card"' in rendered
    assert "beam-hf" in rendered
    assert "72 prompts" in rendered
    assert "15.3%" in rendered
    assert "52.8%" in rendered
    assert "USD 0.43" in rendered
    assert "8.6 s" in rendered
    assert "<dt>Reported cost</dt><dd>not available</dd>" in rendered


def test_benchmark_summary_html_reports_missing_summary_safely(monkeypatch):
    monkeypatch.setattr(app, "_benchmark_summary", lambda: {
        "status": "<summary>not ready</summary>",
    })

    rendered = app._benchmark_summary_html()

    assert 'role="status"' in rendered
    assert "&lt;summary&gt;not ready&lt;/summary&gt;" in rendered
    assert "<summary>not ready</summary>" not in rendered


def test_benchmark_tab_wires_the_custom_html_protocol_and_summary(monkeypatch):
    # The sandbox blocks asyncio's Windows self-pipe socketpair; this Gradio
    # lock is only needed by a running queue, which this construction test skips.
    monkeypatch.setattr(app.gr.utils, "safe_get_lock", lambda: None)
    monkeypatch.setattr(app.gr.utils, "safe_get_stop_event", lambda: object())
    from gradio import routes
    monkeypatch.setattr(routes, "attach_oauth", lambda _app: None)
    monkeypatch.setattr(app.gr.Blocks, "queue", lambda self, **_kwargs: self)
    monkeypatch.setattr(app, "_community_model_choices", lambda: [])
    monkeypatch.setattr(app, "_community_comparison_snapshot", lambda: {
        "status": "Community results unavailable in this test.", "rows": [], "beam_rows": [],
        "selected": {}, "summary": {},
    })
    monkeypatch.setattr(app, "_festival_lens_states", lambda: {"Kerala"})
    _stub_state_tradition_ui(monkeypatch)
    monkeypatch.setattr(app, "_load_gallery_index", lambda: {
        "suite_id": app.EVAL_SUITE_ID, "suite_sha256": app.EVAL_SUITE_SHA256,
        "validator_version": app.VALIDATOR_VERSION, "states": [], "common_prompt_ids": [],
        "arms": [{"key": "claude-sonnet-5-5", "label": "Claude Sonnet"}], "baselines": [],
    })
    monkeypatch.setattr(app, "_home_reference_rows", lambda: [])

    demo = app.build_demo()
    assert any(component.get("props", {}).get("value") == "Sign in to publish"
               for component in demo.config["components"])
    html_panels = [component["props"].get("value", "")
                   for component in demo.config["components"] if component["type"] == "html"]

    assert len([panel for panel in html_panels if 'role="tab" data-tab-button=' in panel]) == 1
    ui_html = next(panel for panel in html_panels if 'data-tab-button="studio"' in panel)
    assert 'data-tab-button="benchmark"' in ui_html and "><span>Home</span></a>" in ui_html
    assert 'data-tab-button="leaderboard"' in ui_html and "><span>Models</span></a>" in ui_html
    assert 'data-tab-button="studio"' in ui_html and "><span>Try it</span></a>" in ui_html
    assert 'data-tab-button="rl"' in ui_html and "><span>Build with it</span></a>" in ui_html
    assert "AI designs patterns from 36 Indian craft traditions. An independent checker verifies every rule the design claims, or shows the exact pixel that breaks it." in ui_html
    assert 'data-testid="proof-replay"' in ui_html
    proof_hero = ui_html.split('data-testid="proof-replay"', 1)[1].split('</article>', 1)[0]
    assert proof_hero.count('class="pramaan-ui-button primary"') == 1
    assert 'data-testid="proof-verify-yourself"' in proof_hero
    assert 'data-testid="proof-replay-play"' in proof_hero and 'class="pramaan-ui-button secondary"' in proof_hero
    home_actions = re.search(r'<nav class="pramaan-home-actions"[^>]*>(.*?)</nav>', ui_html, re.S)
    assert home_actions
    assert "Watch a proof" not in home_actions.group(1)
    assert "Try it" not in home_actions.group(1)
    assert "Models" in home_actions.group(1)
    assert "Use it as an RL environment" in home_actions.group(1)
    home_panel = ui_html.split('data-tab-panel="benchmark">', 1)[1].split('<div class="pramaan-ui-tabpanel"', 1)[0]
    assert 'See how 10 models compare' not in home_panel
    assert "Model leaderboard" not in home_panel and "USD 10" not in home_panel
    assert 'id="state-wall"' in home_panel and 'data-home-wall-tile' in home_panel
    assert home_panel.index('class="pramaan-home-intro"') < home_panel.index('data-testid="proof-replay"') < home_panel.index('class="pramaan-home-metrics"') < home_panel.index('id="state-wall"')
    models_panel = ui_html.split('data-tab-panel="leaderboard"', 1)[1].split('<div class="pramaan-ui-tabpanel"', 1)[0]
    assert "Model leaderboard" in models_panel and 'data-testid="arena"' in models_panel
    assert "Design gallery" not in models_panel
    assert "Arena" in models_panel
    assert "How the benchmark is measured" in ui_html
    assert 'role="tablist"' in ui_html
    assert 'role="tab" data-tab-button="benchmark" aria-selected="true"' in ui_html
    assert 'href="#/overview"' in ui_html
    assert 'href="#/leaderboard"' in ui_html
    assert 'href="#/build"' in ui_html
    assert 'href="#/studio"' in ui_html
    assert 'href="/web/"' in ui_html
    assert 'class="pramaan-ui-disclaimer"' not in ui_html
    assert 'id="pramaan-proof-seal"' not in ui_html
    assert 'class="pramaan-seal-stamp"' not in ui_html
    assert 'data-field="gallery_model"' not in ui_html
    assert "claude-sonnet-5-5" in ui_html
    assert "Same-brief model wall" not in ui_html
    assert "Design gallery" not in ui_html
    assert "Canonical model results" in ui_html
    assert 'data-field="festival_lens" disabled' not in ui_html
    assert ui_html.index('data-field="state"') < ui_html.index('data-field="tradition"')
    assert ui_html.index('data-field="tradition"') < ui_html.index('data-field="example"')
    assert "Showing examples for Gujarat · Bandhani. Change either selector to update the examples." in ui_html
    custom_html = next(block for block in demo.blocks.values() if isinstance(block, app.gr.HTML))
    assert "ArrowRight" in custom_html.js_on_load and "ArrowDown" in custom_html.js_on_load
    assert "hashchange" in custom_html.js_on_load and "popstate" in custom_html.js_on_load
    assert "history.pushState" in custom_html.js_on_load
    assert "#/states?state=" in custom_html.js_on_load
    assert "data-dismiss-disclaimer" in custom_html.js_on_load
    assert "key === 'state' ? 'state' : key === 'tradition' ? 'tradition'" in custom_html.js_on_load
    assert "const uploaded = await upload(file)" in custom_html.js_on_load
    assert "values.community_archive_path = uploaded.path" in custom_html.js_on_load
    assert "Sign in with Hugging Face to publish: use the button at the top of this page" in custom_html.js_on_load
    assert "Archive submission timed out after 90 seconds" in custom_html.js_on_load
    assert "}, 90000)" in custom_html.js_on_load


def test_home_proof_first_paint_shows_failed_attempt_then_replays_the_repair():
    import re

    rendered = app._proof_replay_html()
    active_image = re.search(r'<img[^>]+data-proof-active-tile src="([^"]+)"', rendered)
    first_step = re.search(r'<div class="pramaan-proof-step" data-proof-step="1"([^>]*)>(.*?)</div>', rendered)
    passing_step = re.search(r'<div class="pramaan-proof-step" data-proof-step="\d+" data-proof-result([^>]*)>(.*?)</div>', rendered)

    assert active_image is not None and first_step is not None and passing_step is not None
    assert "hidden" not in first_step.group(1)
    assert "hidden" in passing_step.group(1)
    assert active_image.group(1) == re.search(r'<img[^>]+src="([^"]+)"', first_step.group(2)).group(1)
    assert 'data-proof-step="1"' in rendered
    assert "Himachal Pradesh · Kinnauri shawl geometric border · Periodic Raster" in rendered
    assert "hp_kinnauri" not in rendered
    assert "Attempt 3 · all checks pass" in rendered


def test_home_wall_has_model_and_artwork_controls_without_implying_cultural_scores(monkeypatch):
    monkeypatch.setattr(app, "_home_reference_rows", lambda: [])

    rendered = app._render_custom_app(app._initial_custom_state())

    home = rendered.split('data-tab-panel="benchmark"', 1)[1].split('<div class="pramaan-ui-tabpanel"', 1)[0]
    assert 'data-wall-model-choice=' in home
    assert 'data-wall-mode-choice="reference"' in home
    assert 'cultural authenticity' in home
    assert '<details class="pramaan-home-reference-note"><summary>How to read this</summary>' in home


def test_benchmark_summary_rejects_superseded_suite_digest(monkeypatch, tmp_path):
    summary_path = tmp_path / app.EVAL_SUMMARY_PATH
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps({"suite_id": "eval-v2", "suite_sha256": "f" * 64,
                                        "by_model": {"beam-hf": {}}}), encoding="utf-8")
    monkeypatch.setattr(app, "ROOT", tmp_path)

    result = app._benchmark_summary()

    assert result["status"] == "The bundled benchmark summary belongs to a superseded suite."
    assert "by_model" not in result


def test_benchmark_gallery_loads_only_replayed_accepted_published_trajectories(monkeypatch, tmp_path):
    source = (app.ROOT / "examples" / "ajrakh_starfield.pattern.json").read_text(encoding="utf-8")
    candidate = json.loads(source)
    pattern = parse_pattern(json.dumps(candidate))
    checks, masks, grid = verify_pattern(pattern)
    certificate = make_certificate(pattern, checks, masks, grid)
    record = {
        "prompt_id": "gj_ajrakh-easy-0",
        "state": "Gujarat",
        "grammar": "periodic_raster",
        "level": "easy",
        "prompt_suite_sha256": "a" * 64,
        "validity_at_3": True,
        "accepted_mask_sha256": certificate["mask_sha256"],
        "calls": [{
            "call_index": 1,
            "phase": "proposal",
            "messages": [{"role": "system", "content": "system"},
                         {"role": "user", "content": "Exact locked benchmark prompt"}],
            "candidate": candidate,
            "verifier_report": {"checks": checks, "all_hard_constraints_pass": True},
            "accepted": True,
        }],
    }
    trajectory_path = tmp_path / "benchmarks" / "data" / "eval-v2.1-trajectories.jsonl"
    trajectory_path.parent.mkdir(parents=True)
    trajectory_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(app, "ROOT", tmp_path)
    monkeypatch.setattr(app, "EVAL_SUITE_SHA256", "a" * 64)

    gallery, details = app._benchmark_state_view("Gujarat")

    assert len(gallery) == 1
    assert details[0]["exact_prompt"] == "Exact locked benchmark prompt"
    assert details[0]["accepted_mask_sha256"] == certificate["mask_sha256"]

    trajectory_path.write_text(json.dumps({**record, "prompt_suite_sha256": "f" * 64}) + "\n",
                               encoding="utf-8")
    stale_gallery, stale_details = app._benchmark_state_view("Gujarat")
    assert stale_gallery == []
    assert not isinstance(stale_details, list)

    trajectory_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    all_gallery, all_details = app._benchmark_state_view(None)
    mosaic = app._benchmark_state_mosaic_html(all_gallery, all_details)
    assert mosaic.count('data-state="') == 36
    assert 'data-state="Gujarat"' in mosaic
    assert "gj_ajrakh-easy-0" in mosaic
    assert 'loading="lazy" decoding="async"' in mosaic


def test_community_run_requires_explicit_consent_before_loading_provider_runtime(monkeypatch):
    monkeypatch.setattr(app, "_community_runtime", lambda: (_ for _ in ()).throw(AssertionError("runtime loaded")))

    status = app._run_community_evaluation("claude-haiku-5-5", "visitor-secret", False)

    assert "Confirm that provider charges may apply" in status
    assert "visitor-secret" not in status


def test_community_run_rejects_non_provider_model_before_loading_runtime(monkeypatch):
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "community-write-secret")
    monkeypatch.setattr(app, "_community_runtime", lambda: (_ for _ in ()).throw(
        AssertionError("invalid model reached provider runtime")))

    status = app._run_community_evaluation("https://untrusted.example/v1", "visitor-secret", True)

    assert "valid provider/model identifier" in status
    assert "No provider request was made" in status
    assert "visitor-secret" not in status


def test_community_run_accepts_nested_provider_model_and_guides_unpriced_models_to_zip(monkeypatch):
    model_id = "huggingface/together/meta-llama/Llama-3.3-70B-Instruct"
    seen = []

    def no_call_preflight(received):
        seen.append(received)
        raise ValueError("model is not in LiteLLM's priced catalog; run it locally and upload its replay-audited ZIP instead")

    runner = SimpleNamespace(
        community_litellm_model_config=no_call_preflight,
        run_community_evaluation=lambda *_args, **_kwargs: pytest.fail("unpriced model was dispatched"),
    )
    monkeypatch.setattr(app, "_community_runtime", lambda: (runner, SimpleNamespace()))

    status = app._run_community_evaluation(model_id, "visitor-secret", True)

    assert seen == [model_id]
    assert "Run it locally, then upload the replay-audited ZIP" in status
    assert "No provider request was made" in status


def test_nested_model_identifier_rejects_empty_and_traversal_segments_before_runtime(monkeypatch):
    monkeypatch.setattr(app, "_community_runtime", lambda: pytest.fail("invalid ID reached preflight"))

    for model_id in ("huggingface/together//model", "huggingface/together/../model"):
        status = app._run_community_evaluation(model_id, "visitor-secret", True)
        assert "valid provider/model identifier" in status
        assert "No provider request was made" in status


def test_community_model_field_documents_nested_huggingface_route():
    rendered = app._render_custom_app(app._initial_custom_state())

    assert "LiteLLM model ID (provider/model or nested route)" in rendered
    assert "huggingface/together/meta-llama/Llama-3.3-70B-Instruct" in rendered


def test_community_run_rejects_wrong_token_limit_or_cap_before_dispatch(monkeypatch):
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "community-write-secret")
    seen = []
    runner = SimpleNamespace(
        community_litellm_model_config=lambda _name: {
            "request_params": {"max_tokens": 8192}, "budget": {"cap_usd": 10.0}},
        run_community_evaluation=lambda *_args: seen.append("dispatch"),
    )
    publisher = SimpleNamespace()
    monkeypatch.setattr(app, "_community_runtime", lambda: (runner, publisher))

    status = app._run_community_evaluation("anthropic/claude-haiku-5-5", "visitor-secret", True)

    assert "fixed 4,096-token protocol and USD 10 spend cap" in status
    assert "No provider request was made" in status
    assert seen == []


def test_community_preflight_error_confirms_no_provider_request(monkeypatch):
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "community-write-secret")
    runner = SimpleNamespace(
        community_litellm_model_config=lambda _name: (_ for _ in ()).throw(ValueError("private catalog detail")),
        run_community_evaluation=lambda *_args: pytest.fail("preflight rejection must not dispatch"),
    )
    monkeypatch.setattr(app, "_community_runtime", lambda: (runner, SimpleNamespace()))

    status = app._run_community_evaluation("anthropic/unknown-model", "visitor-secret", True)

    assert "No provider request was made" in status
    assert "private catalog detail" not in status
    assert "visitor-secret" not in status


def test_community_runtime_error_discloses_possible_provider_charges(monkeypatch):
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "community-write-secret")
    def run(*_args, task_limit, full_run_cost_acknowledged):
        raise RuntimeError("private provider error")
    runner = SimpleNamespace(
        community_litellm_model_config=lambda _name: {
            "request_params": {"max_tokens": 4096}, "budget": {"cap_usd": 10.0}},
        run_community_evaluation=run,
    )
    monkeypatch.setattr(app, "_community_runtime", lambda: (runner, SimpleNamespace()))

    status = app._run_community_evaluation("anthropic/claude-haiku-5-5", "visitor-secret", True)

    assert "charges may have occurred" in status
    assert "private provider error" not in status
    assert "visitor-secret" not in status


def test_loading_example_syncs_state_and_tradition_controls_to_example(monkeypatch):
    tradition = SimpleNamespace(
        id="tn_sikku_kolam", name="Sikku Kolam", states=("Tamil Nadu",), default_profile={"symmetry": "D2"},
        palette_presets={"default": ["#172F4A"]},
    )
    monkeypatch.setattr(app, "_tradition_by_id", lambda tradition_id: tradition
                        if tradition_id == tradition.id else None)
    rendered_state = {}
    monkeypatch.setattr(app, "_render_custom_app", lambda state: rendered_state.update(state) or "rendered")
    previous = app._initial_custom_state()
    previous["form"].update(state="Gujarat", tradition="gj_ajrakh")
    example_name = next(name for name in app.EXAMPLES if name.startswith("Sikku Kolam Loop"))
    event = SimpleNamespace(action="example", example=example_name)

    rendered, state = app._handle_custom_event(event, previous)

    assert state["form"]["state"] == "Tamil Nadu"
    assert state["form"]["tradition"] == "tn_sikku_kolam"
    assert state["form"]["example"] == example_name
    assert rendered == "rendered"
    assert rendered_state["form"]["state"] == "Tamil Nadu"
    assert rendered_state["form"]["tradition"] == "tn_sikku_kolam"


def test_state_change_refreshes_tradition_and_worked_example_in_same_ui(monkeypatch):
    _stub_state_tradition_ui(monkeypatch)
    _stub_custom_render_sources(monkeypatch)
    monkeypatch.setattr(app, "_example_result_for_ui", lambda name: {
        "image": "data:image/png;base64,preview", "gallery": [],
        "source": '{"tradition_id":"tn_sikku_kolam"}', "certificate": "{}",
        "history": '{"original_prompt":"A checked Kolam"}', "status": "Verified", "bundle": None,
    })
    previous = app._initial_custom_state()
    assert previous["form"]["tradition"] == "gj_bandhani"
    event = SimpleNamespace(action="state", state="Tamil Nadu", active_tab="studio")

    rendered, state = app._handle_custom_event(event, previous)

    assert state["form"]["state"] == "Tamil Nadu"
    assert state["form"]["tradition"] == "tn_sikku_kolam"
    assert 'data-field="state"' in rendered
    assert 'value="Tamil Nadu" selected' in rendered
    assert 'data-field="tradition"' in rendered
    assert 'value="tn_sikku_kolam" selected' in rendered
    assert 'data-field="example"' in rendered
    assert 'value="Sikku Kolam Loop" selected' in rendered
    assert "Showing examples for Tamil Nadu · Sikku Kolam. Change either selector to update the examples." in rendered
    assert "Bandhani Dot Field" not in rendered
    assert state["form"]["brief"] == "A checked Kolam"
    assert state["result"]["source"] == state["form"]["source"]
    assert state["result"]["image"] == "data:image/png;base64,preview"


def test_state_changes_keep_all_36_state_forms_and_evidence_consistent(monkeypatch):
    monkeypatch.setattr(app, "_render_custom_app", lambda _state: "rendered")
    starting = app._initial_custom_state()

    for state_name in app.ALL_STATES:
        event = SimpleNamespace(action="state", state=state_name, active_tab="studio")
        _rendered, state = app._handle_custom_event(event, starting)
        form = state["form"]
        traditions = app._tradition_choices(state_name)
        assert form["state"] == state_name
        choices = app._example_choices(state_name, form["tradition"])

        if choices:
            expected_example = choices[0][1]
            result = state["result"]
            descriptor = app._saved_example_descriptor(expected_example)
            assert form["example"] == expected_example
            source = json.loads(result["source"])
            if descriptor:
                task = descriptor["task"]
                assert form["state"] == task["state"]
                assert form["tradition"] == task["tradition_id"]
                assert form["brief"] == task["brief"]
                assert "verified" in result["status"].lower() or "checks pass" in result["status"].lower()
            else:
                assert form["tradition"] == (traditions[0].id if traditions else "")
                assert app.EXAMPLE_STATES[Path(app.EXAMPLES[expected_example]).name.removesuffix(".pattern.json")] == state_name
                assert source["tradition_id"] == form["tradition"]
            assert result["image"].startswith("data:image/png;base64,")
            assert form["source"] == result["source"]
            assert form["brief"] == app._example_brief(result)
        else:
            assert form["tradition"] == (traditions[0].id if traditions else "")
            assert form["example"] == form["brief"] == form["source"] == ""
            assert "image" not in state["result"]
            assert state["result"]["status"] == (
                "No saved example for this state; edit the brief and use Generate with Beam"
            )


def test_tradition_change_refreshes_matching_example_in_same_ui(monkeypatch):
    _stub_state_tradition_ui(monkeypatch)
    _stub_custom_render_sources(monkeypatch)
    previous = app._initial_custom_state()
    event = SimpleNamespace(action="tradition", state="Gujarat",
                            tradition="gj_zardozi", active_tab="studio")

    rendered, state = app._handle_custom_event(event, previous)

    assert state["form"]["state"] == "Gujarat"
    assert state["form"]["tradition"] == "gj_zardozi"
    assert state["form"]["example"] == "Zardozi Grid"
    assert 'value="gj_zardozi" selected' in rendered
    assert 'value="Zardozi Grid" selected' in rendered
    assert "Showing examples for Gujarat · Zardozi. Change either selector to update the examples." in rendered
    assert "Bandhani Dot Field" not in rendered


def test_worked_example_choices_are_scoped_and_labeled_by_state_and_tradition(monkeypatch):
    tradition = SimpleNamespace(id="gj_bandhani", name="Bandhani", states=("Gujarat",))
    monkeypatch.setattr(app, "_tradition_by_id", lambda tradition_id: tradition
                        if tradition_id == tradition.id else None)

    monkeypatch.setattr(app, "_home_suite_tasks", lambda: [])
    monkeypatch.setattr(app, "_saved_benchmark_examples_by_task", lambda: {})
    choices = app._example_choices("Gujarat", "gj_bandhani")

    assert len(choices) == 1
    label, value = choices[0]
    assert "Bandhani" in label
    assert "Bandhani Dot Field" in label
    assert value == "Bandhani Dot Field · Beam-generated, verified"
    assert app._example_choices("Tamil Nadu", "gj_bandhani") == []


def test_every_state_and_tradition_has_a_replay_verified_loadable_sample():
    app._home_suite_tasks.cache_clear()
    app._saved_benchmark_examples_by_task.cache_clear()
    app._saved_example_result.cache_clear()
    checked_pairs = 0
    for state_name in app.ALL_STATES:
        traditions = app._tradition_choices(state_name)
        assert traditions, f"No registered traditions for {state_name}"
        for tradition in traditions:
            choices = app._example_choices(state_name, tradition.id)
            assert choices, f"No worked sample for {state_name} / {tradition.id}"
            assert 'data-action="example"' in app._load_example_button_html(choices)
            assert " disabled" not in app._load_example_button_html(choices)
            _label, value = choices[0]
            descriptor = app._saved_example_descriptor(value)
            assert descriptor is not None, f"First choice is not benchmark-backed: {value}"
            task = descriptor["task"]
            form = app._form_for_example(value, {"state": state_name, "tradition": tradition.id})
            assert form["state"] == task["state"]
            assert form["tradition"] == task["tradition_id"]
            assert form["brief"].strip()
            result = app._example_result_for_ui(value)
            assert result["image"].startswith("data:image/png;base64,")
            assert result["source"]
            certificate = json.loads(result["certificate"])
            assert certificate["all_hard_constraints_pass"] is True
            assert form["brief"] == app._example_brief(result) == task["brief"]
            if descriptor["kind"] == "beam":
                assert descriptor["kind"] == "beam"
                assert "Beam attempt, verified" in result["status"]
            else:
                assert "Reference design" in result["status"]
                assert "may differ from the brief" in result["status"]
            checked_pairs += 1
    assert checked_pairs >= 36


def test_home_replay_comparison_copy_distinguishes_appearance_from_check_result():
    rendered = app._proof_replay_html()

    assert "What the eye sees" in rendered and "attempt 1 looks symmetric" in rendered
    assert "What the checker found" in rendered and "one pixel and its mirror differ" in rendered


def test_home_proof_hero_gives_comparison_art_the_larger_column_and_tiles():
    rendered = app._proof_replay_html()

    assert 'data-proof-comparison-failed' in rendered and 'data-proof-comparison-passing' in rendered
    assert ".pramaan-proof-replay {" in app.PRAMAAN_CSS
    assert "grid-template-columns: minmax(0,.85fr) minmax(210px,1.15fr)" in app.PRAMAAN_CSS
    assert '.pramaan-proof-comparison { align-items: start; display: grid; gap: .55rem; grid-template-areas: "failed passed" "summary summary";' in app.PRAMAAN_CSS
    assert '.pramaan-proof-comparison .pramaan-proof-tile { max-height: 320px; width: min(100%,320px); }' in app.PRAMAAN_CSS
    assert ".pramaan-proof-comparison .pramaan-proof-tile { max-height: 150px; width: min(100%,150px); }" in app.PRAMAAN_CSS


def test_home_proof_witness_summary_appears_once_between_comparison_tiles():
    rendered = app._proof_replay_html()

    assert rendered.count('data-testid="proof-witness-summary"') == 1
    failed = rendered.index('data-proof-comparison-failed')
    witness = rendered.index('data-testid="proof-witness-summary"')
    passed = rendered.index('data-proof-comparison-passing')
    assert failed < witness < passed


def test_home_proof_stage_is_hidden_until_replay_starts():
    css = app.PRAMAAN_CSS

    assert ".pramaan-proof-stage { display: none; }" in css
    assert '.pramaan-proof-replay[data-proof-started="true"] .pramaan-proof-stage {' in css
    assert "display: block; min-height: 120px;" in css


def test_home_proof_progress_is_hidden_until_replay_starts():
    rendered = app._proof_replay_html()

    assert 'data-proof-progress role="status"' in rendered
    assert ".pramaan-proof-progress { color: var(--pramaan-muted); display: none;" in app.PRAMAAN_CSS
    assert '.pramaan-proof-replay[data-proof-started="true"] .pramaan-proof-progress { display: inline; }' in app.PRAMAAN_CSS


def test_manipur_reward_gap_copy_is_honest_about_training_status(monkeypatch, tmp_path):
    from pramaan.ui import known_gap

    monkeypatch.setattr(known_gap, "accepted_pattern_is_low_visibility", lambda *_args: True)
    thumbnail = tmp_path / "reference.png"
    thumbnail.write_bytes(b"saved preview")
    documents = {"model": {"tasks": {known_gap._TASK_ID: {
        "attempts": [{"passed": True, "tile_png_b64": "AA=="}], "reward": 1.0,
    }}}}

    rendered = known_gap.render_known_gap_panel(documents, {known_gap._TASK_ID: "#000000"}, thumbnail)

    assert "Reward gap found before training" in rendered
    assert "None of these models was trained on this reward" in rendered
    assert "not learned gaming" in rendered
    assert "the hole a trained policy could find" in rendered


def test_models_dashboard_has_sticky_section_navigation_in_requested_order():
    from pramaan.ui.leaderboard import render_models_section_nav

    anchors = ["leaderboard-composition", "leaderboard-results-matrix",
               "leaderboard-failing-checks", "leaderboard-cost", "leaderboard-full-results",
               "model-arena", "leaderboard-methodology", "leaderboard-community-results"]
    rendered = render_models_section_nav(" ".join(f'id="{anchor}"' for anchor in anchors))
    labels = ["Composition", "Results matrix", "Failing checks", "Cost",
              "Full results", "Arena", "Methodology", "Community results"]

    assert 'data-testid="models-section-menu"' in rendered
    assert 'aria-label="Models dashboard sections"' in rendered
    assert [rendered.index(label) for label in labels] == sorted(rendered.index(label) for label in labels)
    assert 'href="#leaderboard-results-matrix">Results matrix</a>' in rendered
    assert 'href="#leaderboard-full-results">Full results</a>' in rendered
    assert 'Ranking' not in rendered
    assert 'href="#leaderboard-community-results">Community results</a>' in rendered
    assert ".pramaan-leaderboard-nav.is-sticky" in app.PRAMAAN_CSS


def test_models_dashboard_navigation_omits_unrendered_sections():
    from pramaan.ui.leaderboard import render_models_section_nav

    rendered = render_models_section_nav('<section id="leaderboard-composition"></section>')

    assert 'href="#leaderboard-composition">Composition</a>' in rendered
    assert "Results matrix" not in rendered
    assert "Cost" not in rendered
    assert "Arena" not in rendered


def test_full_results_table_preserves_saved_wilson_intervals_without_redundant_ranking_chart():
    import re
    from pramaan.ui.findings import wilson_interval
    from pramaan.ui.leaderboard import render_model_report

    summary = app._benchmark_summary()
    rendered = render_model_report(summary)
    models = summary["by_model"]
    assert 'data-testid="model-report-table"' in rendered
    for key, document in models.items():
        overall = document["overall"]
        n = overall["n"]
        first = overall.get("passed_at_1", round(overall.get("validity_at_1", 0) * n))
        within = overall.get("accepted_designs", round(overall.get("validity_at_3", 0) * n))
        row_match = re.search(
            rf'<tr[^>]*data-testid="model-report-row-{re.escape(key)}"[^>]*>(.*?)</tr>', rendered, re.S,
        )
        assert row_match, key
        first_interval = wilson_interval(first, n)
        within_interval = wilson_interval(within, n)
        assert f'95% Wilson {first_interval[0]:.1%}–{first_interval[1]:.1%}' in row_match.group(1)
        assert f'95% Wilson {within_interval[0]:.1%}–{within_interval[1]:.1%}' in row_match.group(1)


def test_composition_chart_legend_and_stacked_counts_use_saved_attempts():
    import re
    from pramaan.ui.leaderboard import render_model_report

    summary = app._benchmark_summary()
    gallery = app._load_gallery_arm_documents(app._load_gallery_index())
    rendered = render_model_report(summary, gallery_documents=gallery)

    for label in ("Passed", "Broke a rule", "Unreadable"):
        assert f">{label}</li>" in rendered
    for model_key, document in gallery.items():
        passed = broke_rule = unreadable = observed = 0
        tasks = document.get("tasks", {})
        for task in tasks.values():
            attempts = task.get("attempts", []) if isinstance(task, dict) else []
            first = attempts[0] if attempts and isinstance(attempts[0], dict) else None
            if first is None:
                continue
            failing = first.get("failing_checks")
            if first.get("passed") is True:
                passed += 1
            elif first.get("passed") is False and isinstance(failing, list) and failing:
                if "candidate_parse" in failing:
                    unreadable += 1
                else:
                    broke_rule += 1
            else:
                continue
            observed += 1
        if observed == summary["by_model"][model_key]["overall"]["n"]:
            row = re.search(rf'<li[^>]*data-testid="model-chart-{re.escape(model_key)}"([^>]*)>', rendered)
            assert row
            assert f'data-first-pass="{passed}"' in row.group(1)
            assert f'data-first-rule-fail="{broke_rule}"' in row.group(1)
            assert f'data-first-unreadable="{unreadable}"' in row.group(1)
    assert ".pramaan-model-bar-stacked { border:" in app.PRAMAAN_CSS
    assert ".pramaan-model-composition-legend" in app.PRAMAAN_CSS


def test_cost_frontier_uses_saved_estimates_and_within_three_results():
    import re
    from pramaan.ui.leaderboard import render_model_report

    summary = app._benchmark_summary()
    rendered = render_model_report(summary)
    models = summary["by_model"]
    observed = []
    for key, document in models.items():
        overall = document["overall"]
        expected_cost = overall["estimated_cost_totals_by_currency"]["USD"]
        expected_count = overall["accepted_designs"]
        match = re.search(rf'<g class="point[^>]*data-testid="model-cost-point-{re.escape(key)}"([^>]*)>', rendered)
        assert match, key
        attrs = match.group(1)
        assert f'data-estimated-cost="{expected_cost:.8f}"' in attrs
        assert f'data-within-three="{expected_count}"' in attrs
        assert 'data-denominator="72"' in attrs
        observed.append((key, expected_cost, expected_count))
    frontier = {key for key, cost, count in observed if not any(
        other_key != key and other_cost <= cost and other_count >= count
        and (other_cost < cost or other_count > count)
        for other_key, other_cost, other_count in observed
    )}
    for key, _cost, _count in observed:
        assert f'data-testid="model-cost-point-{key}"' in rendered
        point = re.search(rf'<g class="point[^>]*data-testid="model-cost-point-{re.escape(key)}"([^>]*)>', rendered).group(1)
        assert f'data-frontier="{str(key in frontier).lower()}"' in point
    assert "log scale" in rendered
    assert "Costs are estimates, request settings differ by provider" in rendered
    assert "Pareto frontier" in rendered
    assert "$0.15639" not in rendered
    tick_values = re.findall(r'data-cost-tick="([^"]+)"', rendered)
    assert tick_values and set(tick_values) <= {"0.2", "0.5", "1", "2", "5", "10", "20"}
    label_boxes = []
    for key in models:
        point = re.search(rf'data-testid="model-cost-point-{re.escape(key)}"[^>]*>(.*?)</g>', rendered)
        assert point and 'class="label-leader"' in point.group(1)
        box = re.search(r'data-label-box="([^"]+)"', point.group(1))
        assert box, key
        bounds = tuple(float(value) for value in box.group(1).split(","))
        assert len(bounds) == 4
        for other in label_boxes:
            left, top, right, bottom = bounds
            other_left, other_top, other_right, other_bottom = other
            assert not (left < other_right and right > other_left and top < other_bottom and bottom > other_top), (key, bounds, other)
        label_boxes.append(bounds)


def test_results_matrix_covers_saved_model_task_outcomes_and_arena_links(monkeypatch):
    import re
    from html import escape as html_escape
    from pramaan.ui.leaderboard import render_model_report

    summary = app._benchmark_summary()
    index = app._load_gallery_index()
    gallery = app._load_gallery_arm_documents(index)
    tasks = app._home_suite_tasks()
    backgrounds = {task["id"]: task["palette"][0] for task in tasks}
    rendered = render_model_report(summary, gallery_documents=gallery,
                                   task_backgrounds=backgrounds, suite_tasks=tasks)
    cell_count = rendered.count('data-testid="matrix-cell-')
    assert cell_count == len(summary["by_model"]) * len(tasks)
    for model_key, document in gallery.items():
        for task in tasks:
            task_id = task["id"]
            saved = document.get("tasks", {}).get(task_id)
            attempts = saved.get("attempts", []) if isinstance(saved, dict) else []
            first_pass = bool(attempts and attempts[0].get("passed") is True)
            passing = next((attempt for attempt in attempts if attempt.get("passed") is True), None)
            expected_state = ("first" if first_pass else "repair" if passing else "fail" if attempts else "not-run")
            cell = re.search(
                rf'<a class="pramaan-matrix-cell[^>]*data-testid="matrix-cell-{re.escape(model_key)}-{re.escape(task_id)}"[^>]*>',
                rendered,
            )
            assert cell, (model_key, task_id)
            assert f'data-cell-state="{expected_state}"' in cell.group(0)
            assert f'href="#/leaderboard?task={task_id}"' in cell.group(0)
            assert html_escape(task["state"], quote=True) in cell.group(0)
            cell_html = rendered[cell.start():rendered.find("</a>", cell.end())]
            passing = next((attempt for attempt in attempts if attempt.get("passed") is True), None)
            if passing:
                attempt_number = passing.get("index")
                if type(attempt_number) is not int or not 1 <= attempt_number <= 3:
                    attempt_number = next(i for i, attempt in enumerate(attempts, 1) if attempt is passing)
                assert f'<span aria-hidden="true">{attempt_number}</span>' in cell_html
            else:
                expected_mark = "×" if expected_state == "fail" else "—"
                assert f'<span aria-hidden="true">{expected_mark}</span>' in cell_html
    assert 'aria-label="Scrollable model by task results matrix"' in rendered
    assert 'Passed on attempt 1' in rendered and 'Passed on attempt 2 or 3' in rendered
    assert 'class="matrix-model matrix-corner"' in rendered
    assert 'is-near-invisible' in rendered


def test_matrix_task_link_selects_the_matching_same_brief_arena(monkeypatch):
    from html import escape as html_escape
    from pramaan.ui.compare import render_arena

    monkeypatch.setattr(app, "_render_custom_app", lambda state: "rendered")
    previous = app._initial_custom_state()
    tasks = app._home_suite_tasks()
    task = tasks[0]
    event = SimpleNamespace(action="arena_task", task_id=task["id"], active_tab="leaderboard")

    rendered, state = app._handle_custom_event(event, previous)

    assert rendered == "rendered"
    assert state["active_tab"] == "leaderboard"
    assert state["arena_prompt_id"] == task["id"]
    gallery_index = app._load_gallery_index()
    gallery = app._load_gallery_arm_documents(gallery_index)
    backgrounds = {row["id"]: row["palette"][0] for row in tasks}
    arena = render_arena(
        gallery_index, gallery, suite_id=app.EVAL_SUITE_ID,
        suite_sha256=app.EVAL_SUITE_SHA256, prompt_id=state["arena_prompt_id"],
        replay_prompt_id=app.PROOF_REPLAY_PROMPT_ID, task_backgrounds=backgrounds,
    )
    assert html_escape(task["brief"]) in arena
    assert 'data-testid="arena"' in arena


def test_first_try_failing_checks_chart_matches_saved_attempts_and_separates_unreadable():
    from collections import Counter
    from pramaan.ui.leaderboard import render_model_report

    summary = app._benchmark_summary()
    gallery = app._load_gallery_arm_documents(app._load_gallery_index())
    rendered = render_model_report(summary, gallery_documents=gallery)
    expected = Counter()
    failed_first_attempts = 0
    for document in gallery.values():
        for task in document.get("tasks", {}).values():
            attempts = task.get("attempts", []) if isinstance(task, dict) else []
            first = attempts[0] if attempts and isinstance(attempts[0], dict) else None
            if not first or first.get("passed") is not False:
                continue
            failures = first.get("failing_checks")
            if not isinstance(failures, list):
                continue
            failed_first_attempts += 1
            if "candidate_parse" in failures:
                expected["__unreadable__"] += 1
            else:
                expected.update(set(check_id for check_id in failures if isinstance(check_id, str)))

    assert 'id="leaderboard-failing-checks"' in rendered
    assert "unreadable outputs are counted separately" in rendered
    for check_id, count in expected.items():
        assert f'data-testid="first-try-failure-{check_id}" data-check-id="{check_id}" data-count="{count}"' in rendered
    assert f"summarize {failed_first_attempts} failed first attempts" in rendered
    assert '<span class="pramaan-failure-label">Unreadable output</span>' in rendered


def test_full_results_table_uses_saved_mean_request_time_values():
    from pramaan.ui.leaderboard import render_model_report

    summary = app._benchmark_summary()
    rendered = render_model_report(summary)
    assert 'id="leaderboard-latency"' not in rendered
    assert 'data-testid="model-latency-chart"' not in rendered
    assert "<th scope=\"col\">Mean request time</th>" in rendered
    for key, model in summary["by_model"].items():
        overall = model["overall"]
        row = next(line for line in rendered.splitlines() if f'data-testid="model-report-row-{key}"' in line)
        latency = overall.get("mean_request_elapsed_ms_per_prompt")
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            assert f'data-latency-ms="{latency:.8f}"' in row
            assert f'<td>{latency / 1000:.1f}s per request</td>' in row
        else:
            assert 'data-latency-ms=""' in row
            assert '<td>not reported</td>' in row


def test_matrix_and_scatter_details_are_available_on_touch_and_keyboard_activation():
    from pathlib import Path
    from pramaan.ui.leaderboard import render_model_report

    tasks = app._home_suite_tasks()
    backgrounds = {task["id"]: task["palette"][0] for task in tasks if task.get("palette")}
    gallery = app._load_gallery_arm_documents(app._load_gallery_index())
    rendered = render_model_report(
        app._benchmark_summary(), gallery_documents=gallery, suite_tasks=tasks,
        task_backgrounds=backgrounds,
    )

    assert 'data-testid="chart-detail-dialog"' in rendered
    assert 'data-testid="matrix-cell-beam-hf-' in rendered
    assert 'data-touch-detail="matrix"' in rendered
    assert 'data-detail-result="' in rendered and 'data-detail-brief="' in rendered
    assert 'data-touch-detail="cost"' in rendered and 'data-estimated-cost="' in rendered
    assert 'data-touch-detail="latency"' not in rendered
    assert 'data-chart-detail-arena hidden' in rendered
    assert 'data-chart-detail-close' in rendered

    source = Path(app.__file__).read_text(encoding="utf-8")
    assert "event.target.closest('[data-touch-detail]')" in source
    assert "dialog.showModal()" in source
    assert "detailTarget.click()" in source
    assert "detailClose.closest('[data-testid=\"chart-detail-dialog\"]')?.close()" in source


def test_all_frozen_suite_check_ids_have_specific_plain_language_labels():
    from pramaan.ui.check_labels import plain_check_label

    check_ids = {
        "candidate_parse", "candidate_truncated", "schema_valid", "complexity_bound", "palette_limit",
        "periodic_raster", "layer_registration", "exclusive_ink_collision", "minimum_gap",
        "coverage_bound", "dot_clearance", "translation_period", "cell_quantization",
        "kolam_mirrors_valid", "kolam_loop_count", "kolam_mirror_symmetry",
        "jaali_single_solid", "jaali_holes_interior", "jaali_open_fraction", "jaali_min_web",
        "symmetry_none", "symmetry_d2", "symmetry_d4", "symmetry_c2", "symmetry_c4",
    }
    assert all(plain_check_label({"id": check_id}) != "Check the pattern" for check_id in check_ids)


def test_community_panel_routes_public_contributions_through_dataset_pull_requests(monkeypatch):
    _stub_custom_render_sources(monkeypatch)
    monkeypatch.setattr(app, "_community_model_choices", lambda: [])

    rendered = app._render_custom_app(app._initial_custom_state())

    assert "Space publishing is off" in rendered
    assert "Contribute by pull request" in rendered
    assert "create_pr=True" in rendered
    assert "https://huggingface.co/datasets/kshitijthakkar/pramaan-community-runs" in rendered
    assert 'id="pramaan-community-archive"' not in rendered
    assert 'data-action="community_archive_submit"' not in rendered
    assert 'data-action="community_publish"' not in rendered
    assert "data-action=\"community_run\"" in rendered


def test_community_callback_never_persists_visitor_key(monkeypatch):
    monkeypatch.setattr(app, "_community_runtime", lambda: None)
    event = SimpleNamespace(
        action="community_run", active_tab="benchmark", community_model="anthropic/claude-haiku-5-5",
        community_api_key="visitor-secret", community_charge_consent=True,
    )

    rendered, state = app._handle_custom_event(event, app._initial_custom_state())

    assert "visitor-secret" not in rendered
    assert "visitor-secret" not in json.dumps(state)
    assert "not enabled in this Space build" in state["community_status"]


def test_consent_community_run_uses_fixed_suite_and_uploads_only_after_completion(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "community-write-secret")
    monkeypatch.setenv("SYSTEM", "spaces")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client-id")
    suite_sha = "a" * 64
    run_id = "community-run-1"
    config = {"request_params": {"max_tokens": 4096, "output_config": {"effort": "low"}},
              "budget": {"cap_usd": 10.0},
              "pricing": {"currency": "USD", "per_million_tokens": {"input": 1.0, "output": 2.0},
                          "source": "test rate card", "observed_at": "2026-10-30"}}
    rows = [{"model_key": "anthropic/claude-haiku-5-5", "model_id": "anthropic/claude-haiku-5-5",
             "run_id": run_id, "prompt_id": f"question-{index}", "attempts": [{"index": 1}],
             "validity_at_1": True, "validity_at_3": True,
                "total_tokens": 12, "elapsed_ms": 100, "verification_elapsed_ms": 2,
             "latency_ms": {"request_elapsed_total": 90},
             "cost": {"amount": 0.001, "currency": "USD"}}
            for index in range(72)]
    seen = {"config_model": None, "run": None, "rows": None, "upload_key": None}

    def model_config(model_name):
        seen["config_model"] = model_name
        return config

    def run(model_name, api_key, *, task_limit, full_run_cost_acknowledged):
        seen["run"] = (model_name, api_key, task_limit, full_run_cost_acknowledged)
        return rows, suite_sha

    runner = SimpleNamespace(community_litellm_model_config=model_config,
                             estimate_community_evaluation_cost=lambda _name, **_kwargs: {
                                 "reference_suite": app.EVAL_SUITE_ID, "reference_tasks": 72,
                                 "task_count": 72, "source": "official_beam_eval_v2.1_measured_usage",
                                 "per_run_cap_usd": 10.0,
                                 "estimated_cost_exceeds_per_run_cap": False,
                                 "estimated_usd": 0.0288, "mean_input_tokens_per_task": 200.0,
                                 "mean_output_tokens_per_task": 100.0},
                             run_community_evaluation=run)

    def publish(published_rows, *, run_id, expected_suite_sha256, provider_api_key, submitted_by):
        seen.update(rows=published_rows, upload_key=provider_api_key, submitted_by=submitted_by)
        assert expected_suite_sha256 == suite_sha
        assert run_id == "community-run-1"
        return {"url": "https://huggingface.co/datasets/example/community"}

    publisher = SimpleNamespace(
        COMMUNITY_REPO="example/community",
        EVAL_REPO="example/community",
        publish_community_eval_run=publish,
    )
    monkeypatch.setattr(app, "_community_runtime", lambda: (runner, publisher))
    estimate = app._estimate_community_full_run_cost(
        "anthropic/claude-haiku-5-5", runner=runner, config=config)

    result = app._run_community_evaluation(
        "anthropic/claude-haiku-5-5", "visitor-secret", True,
        publish=True, submitted_by="reviewer", task_limit=72,
        full_run_cost_acknowledged=True, reviewed_estimate=estimate,
    )

    assert "72/72 requested questions" in result
    assert "Tokens 864" in result
    assert "request latency 90 ms mean/prompt" in result
    assert "verifier latency 2.0 ms mean/prompt" in result
    assert "visitor-secret" not in result
    assert seen["config_model"] == "anthropic/claude-haiku-5-5"
    assert seen["run"] == ("anthropic/claude-haiku-5-5", "visitor-secret", 72, True)
    assert seen["upload_key"] == "visitor-secret"
    assert seen["submitted_by"] == "reviewer"
    assert len(seen["rows"]) == 72


def test_community_run_does_not_use_read_only_hf_token_for_publication(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "read-only-token")
    monkeypatch.setenv("SYSTEM", "spaces")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.delenv("HF_COMMUNITY_WRITE_TOKEN", raising=False)
    monkeypatch.setattr(app, "_community_runtime", lambda: (_ for _ in ()).throw(
        AssertionError("runner was loaded without a community write token")))

    status = app._run_community_evaluation(
        "claude-haiku-5-5", "visitor-provider-key", True,
        publish=True, submitted_by="reviewer",
    )

    assert "Public community uploads are not configured" in status
    assert "read-only-token" not in status
    assert "visitor-provider-key" not in status


def test_community_controls_keep_runs_private_and_route_sharing_to_pull_requests(monkeypatch):
    monkeypatch.setattr(app, "_community_model_choices", lambda: [])

    rendered = app._render_custom_app(app._initial_custom_state())

    assert 'type="password"' in rendered
    assert 'data-field="community_charge_consent"' in rendered
    assert "Your key passes through this Space to the provider" in rendered
    assert "disposable key" in rendered
    assert "Results stay private in this session and are not added to a dataset" in rendered
    assert "Contribute by pull request" in rendered
    assert 'data-action="community_run" disabled' in rendered
    assert 'data-action="community_publish"' not in rendered
    assert 'data-action="community_archive_submit"' not in rendered
    assert 'data-field="community_charge_consent" checked' not in rendered


def test_community_model_control_accepts_provider_model_text_and_explains_cost_guard(monkeypatch):
    monkeypatch.setattr(app, "_community_model_choices", lambda: [("claude-haiku-5-5", "Claude Haiku")])

    rendered = app._render_custom_app(app._initial_custom_state())

    assert '<label for="pramaan-community-model">LiteLLM model ID (provider/model or nested route)</label>' in rendered
    assert '<input id="pramaan-community-model" data-field="community_model" type="text"' in rendered
    assert 'data-field="community_model" type="text"' in rendered
    assert 'data-field="community_model" type="password"' not in rendered
    assert "Only priced models are accepted" in rendered
    assert "hard USD 10 run cap" in rendered
    assert '<select id="pramaan-community-model"' not in rendered


def test_private_community_run_works_without_hf_write_token_and_never_publishes(monkeypatch):
    monkeypatch.delenv("HF_COMMUNITY_WRITE_TOKEN", raising=False)
    config = {"request_params": {"max_tokens": 4096}, "budget": {"cap_usd": 10.0}}
    rows = [{"run_id": "local-run", "validity_at_1": True, "validity_at_3": True,
             "total_tokens": 32, "elapsed_ms": 400, "verification_elapsed_ms": 2,
             "latency_ms": {"request_elapsed_total": 350},
             "cost": {"estimated_amount": 0.01, "estimated_currency": "USD"}}]
    calls = []
    runner = SimpleNamespace(
        community_litellm_model_config=lambda _model: config,
        run_community_evaluation=lambda model, key, *, task_limit, full_run_cost_acknowledged:
            calls.append((model, key, {"task_limit": task_limit,
                                       "full_run_cost_acknowledged": full_run_cost_acknowledged}))
            or (rows, "a" * 64),
    )
    publisher = SimpleNamespace(
        publish_community_eval_run=lambda *_args, **_kwargs: pytest.fail("private result must not publish"),
    )
    monkeypatch.setattr(app, "_community_runtime", lambda: (runner, publisher))

    status = app._run_community_evaluation("anthropic/claude-haiku-5-5", "visitor-secret", True)

    assert calls == [("anthropic/claude-haiku-5-5", "visitor-secret", {
        "task_limit": 12, "full_run_cost_acknowledged": False,
    })]
    assert "not added to a dataset" in status
    assert "1/12 requested questions" in status
    assert "visitor-secret" not in status


def test_public_community_run_requires_hf_sign_in_before_provider_call(monkeypatch):
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "community-write-secret")
    monkeypatch.setattr(app, "_community_runtime", lambda: pytest.fail("unauthenticated request reached runner"))
    event = SimpleNamespace(
        action="community_publish", active_tab="benchmark", community_model="anthropic/claude-haiku-5-5",
        community_api_key="visitor-secret", community_charge_consent=True,
        community_publish_consent=True,
    )
    rendered_states = []
    monkeypatch.setattr(app, "_render_custom_app", lambda state: rendered_states.append(state) or "rendered")

    rendered, state = app._handle_custom_event(event, app._initial_custom_state(), oauth_profile=None)

    assert rendered == "rendered"
    assert "Public publishing is disabled" in state["community_status"]
    assert "No provider request was made" in state["community_status"]
    assert "visitor-secret" not in json.dumps(state)


def test_mocked_hf_owner_profile_cannot_publish_community_run_or_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("SYSTEM", "spaces")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "must-not-be-used")
    monkeypatch.setattr(app, "_community_runtime", lambda: pytest.fail("mock profile reached the runner"))
    monkeypatch.setattr(app, "_community_submission_archive_runtime",
                        lambda: pytest.fail("mock profile reached the archive publisher"))
    monkeypatch.setattr(app, "_render_custom_app", lambda _state: "rendered")
    upload_root = tmp_path / "gradio"
    upload_root.mkdir()
    archive_path = upload_root / "submission.zip"
    archive_path.write_bytes(b"mock must never publish")
    monkeypatch.setattr(app.gr.utils, "get_upload_folder", lambda: str(upload_root))
    owner_profile = SimpleNamespace(username="space-owner")
    mock_token = SimpleNamespace(token="mock-oauth-token-for-local-dev")

    run_event = SimpleNamespace(
        action="community_publish", active_tab="benchmark", community_model="anthropic/claude-haiku-5-5",
        community_api_key="visitor-secret", community_charge_consent=True,
        community_publish_consent=True,
    )
    _, run_state = app._handle_custom_event(
        run_event, app._initial_custom_state(), oauth_profile=owner_profile, oauth_token=mock_token,
    )
    assert "Public publishing is disabled" in run_state["community_status"]
    assert "No provider request was made" in run_state["community_status"]

    archive_event = SimpleNamespace(
        action="community_archive_submit", active_tab="leaderboard",
        community_archive_path=str(archive_path), community_archive_consent=True,
    )
    _, archive_state = app._handle_custom_event(
        archive_event, app._initial_custom_state(), oauth_profile=owner_profile, oauth_token=mock_token,
    )
    assert "Public publishing is disabled" in archive_state["community_archive_status"]
    assert not archive_path.exists()


def test_publication_controls_fail_closed_without_real_hf_oauth(monkeypatch):
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "must-not-enable-publishing")
    monkeypatch.delenv("SYSTEM", raising=False)
    monkeypatch.delenv("OAUTH_CLIENT_ID", raising=False)
    monkeypatch.setattr(app, "_community_model_choices", lambda: [("ready", "available")])
    monkeypatch.setattr(app, "_community_submission_archive_runtime", lambda: (lambda *_a, **_k: None, lambda *_a, **_k: {}))

    assert app._community_publish_ready() is False
    assert app._community_submission_archive_ready() is False
    rendered = app._render_custom_app(app._initial_custom_state())

    assert 'data-action="community_publish"' not in rendered
    assert 'data-action="community_archive_submit"' not in rendered
    assert "Space publishing is off" in rendered


def test_authenticated_public_submission_passes_submitter_to_publisher(monkeypatch):
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "community-write-secret")
    monkeypatch.setenv("SYSTEM", "spaces")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client-id")
    config = {"request_params": {"max_tokens": 4096}, "budget": {"cap_usd": 10.0}}
    rows = [{"run_id": "public-run", "prompt_id": "q1", "validity_at_1": True,
             "validity_at_3": True, "total_tokens": 2, "elapsed_ms": 10,
             "verification_elapsed_ms": 1, "latency_ms": {"request_elapsed_total": 8},
             "cost": {"estimated_amount": 0.01, "estimated_currency": "USD"}}]
    submitted = {}
    runner = SimpleNamespace(
        community_litellm_model_config=lambda _model: config,
        run_community_evaluation=lambda _model, _key, *, task_limit, full_run_cost_acknowledged: (rows, "b" * 64),
    )
    def publish(*_args, **kwargs):
        submitted.update(kwargs)
        return {"url": "https://huggingface.co/datasets/example/community"}
    publisher = SimpleNamespace(COMMUNITY_REPO="example/community", publish_community_eval_run=publish)
    monkeypatch.setattr(app, "_community_runtime", lambda: (runner, publisher))
    event = SimpleNamespace(
        action="community_publish", active_tab="benchmark", community_model="anthropic/claude-haiku-5-5",
        community_api_key="visitor-secret", community_charge_consent=True,
        community_publish_consent=True,
    )
    profile = SimpleNamespace(username="craft-reviewer")
    monkeypatch.setattr(app, "_render_custom_app", lambda state: "rendered")

    _rendered, state = app._handle_custom_event(
        event, app._initial_custom_state(), oauth_profile=profile,
        oauth_token=SimpleNamespace(token="real-test-oauth-token"),
    )

    assert submitted["submitted_by"] == "craft-reviewer"
    assert "Published Community run public-run" in state["community_status"]
    assert "visitor-secret" not in json.dumps(state)


def test_public_community_run_fails_closed_before_call_without_submitter_support(monkeypatch):
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "community-write-secret")
    monkeypatch.setenv("SYSTEM", "spaces")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client-id")
    runner = SimpleNamespace(
        community_litellm_model_config=lambda _model: {
            "request_params": {"max_tokens": 4096}, "budget": {"cap_usd": 10.0}},
        run_community_evaluation=lambda *_args: pytest.fail("must not call provider without attributed publisher"),
    )
    def publish(_rows, *, run_id, expected_suite_sha256, provider_api_key):
        pytest.fail("must not publish without submitter attribution")
    monkeypatch.setattr(app, "_community_runtime", lambda: (runner, SimpleNamespace(
        publish_community_eval_run=publish,
    )))

    status = app._run_community_evaluation(
        "anthropic/claude-haiku-5-5", "visitor-secret", True,
        publish=True, submitted_by="craft-reviewer",
    )

    assert "authenticated submitter attribution support" in status
    assert "No provider request was made" in status


def test_initial_custom_render_does_not_fetch_public_community_runs(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "_community_comparison_snapshot", lambda: calls.append("fetch") or {
        "status": "loaded", "rows": [], "beam_rows": [], "selected": {}, "summary": {},
    })

    rendered = app._render_custom_app(app._initial_custom_state())

    assert calls == []
    assert "Select Refresh public community runs to load the latest public results." in rendered


def test_space_archive_form_is_replaced_by_community_dataset_pull_request_instructions(monkeypatch):
    monkeypatch.delenv("HF_COMMUNITY_WRITE_TOKEN", raising=False)
    monkeypatch.setenv("SYSTEM", "spaces")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setattr(app, "_community_submission_archive_runtime", lambda: (object(), object()))
    _stub_custom_render_sources(monkeypatch)

    rendered = app._render_custom_app(app._initial_custom_state())

    assert 'data-field="community_archive_file"' not in rendered
    assert 'data-field="community_archive_consent"' not in rendered
    assert 'data-action="community_archive_submit"' not in rendered
    assert "Contribute by pull request" in rendered
    assert "Space publishing is off" in rendered
    assert "create_pr=True" in rendered


def test_pull_request_contribution_instructions_do_not_enable_space_publishing(monkeypatch):
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "scoped-write-token")
    monkeypatch.setenv("SYSTEM", "spaces")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setattr(app, "_community_submission_archive_runtime",
                        lambda: (lambda *_a, **_k: None, lambda *_a, **_k: {}))
    _stub_custom_render_sources(monkeypatch)

    rendered = app._render_custom_app(app._initial_custom_state())

    assert "Contribute by pull request" in rendered
    assert 'data-action="community_publish"' not in rendered
    assert 'data-action="community_archive_submit"' not in rendered


def test_archive_submit_refuses_without_consent_and_does_not_persist_upload_path(monkeypatch):
    monkeypatch.setenv("SYSTEM", "spaces")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setattr(app, "_community_submission_archive_runtime",
                        lambda: (_ for _ in ()).throw(AssertionError("archive runtime loaded")))
    rendered_state = {}
    monkeypatch.setattr(app, "_render_custom_app", lambda state: rendered_state.update(state) or "rendered")
    event = SimpleNamespace(
        action="community_archive_submit", active_tab="leaderboard",
        community_archive_path="C:/temp/private-submission.zip", community_archive_consent=False,
    )

    rendered, state = app._handle_custom_event(
        event, app._initial_custom_state(), oauth_profile=SimpleNamespace(username="test-user"),
        oauth_token=SimpleNamespace(token="real-test-oauth-token"),
    )

    assert "Check the publication consent box" in state["community_archive_status"]
    assert "C:/temp/private-submission.zip" not in json.dumps(state)
    assert "C:/temp/private-submission.zip" not in rendered
    assert rendered == "rendered"
    assert rendered_state["community_archive_status"] == state["community_archive_status"]


def test_archive_publish_validates_managed_upload_and_uses_only_scoped_publisher(monkeypatch, tmp_path):
    monkeypatch.setenv("SYSTEM", "spaces")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client-id")
    upload_root = tmp_path / "gradio"
    upload_root.mkdir()
    archive_path = upload_root / "submission.zip"
    archive_path.write_bytes(b"zip fixture")
    monkeypatch.setattr(app.gr.utils, "get_upload_folder", lambda: str(upload_root))
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "scoped-write-token")
    captured = {}

    def publish(source, **kwargs):
        captured["source"] = source
        captured["kwargs"] = kwargs
        return {"repo_id": "kshitijthakkar/pramaan-community-runs", "records": 3}

    monkeypatch.setattr(app, "_community_submission_archive_runtime",
                        lambda: (lambda *_a, **_k: None, publish))

    status = app._publish_community_archive(str(archive_path), submitted_by="test_user")

    assert "Published 3 replay-audited record(s)" in status
    assert captured["source"] == archive_path.resolve()
    assert captured["kwargs"]["expected_suite_sha256"] == app.EVAL_SUITE_SHA256
    assert captured["kwargs"]["submitted_by"] == "test_user"
    assert "api" not in captured["kwargs"] and "provider_api_key" not in captured["kwargs"]
    assert "scoped-write-token" not in status


def test_archive_publish_rejects_upload_outside_gradio_temp_and_cleans_managed_upload(monkeypatch, tmp_path):
    monkeypatch.setenv("SYSTEM", "spaces")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client-id")
    upload_root = tmp_path / "gradio"
    upload_root.mkdir()
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"private")
    managed = upload_root / "submission.zip"
    managed.write_bytes(b"private")
    monkeypatch.setattr(app.gr.utils, "get_upload_folder", lambda: str(upload_root))
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "scoped-write-token")
    publisher_calls = []
    monkeypatch.setattr(app, "_community_submission_archive_runtime",
                        lambda: (lambda *_a, **_k: None,
                                 lambda *_a, **_k: publisher_calls.append(True)))
    monkeypatch.setattr(app, "_render_custom_app", lambda _state: "rendered")

    event = SimpleNamespace(action="community_archive_submit", active_tab="leaderboard",
                            community_archive_path=str(outside), community_archive_consent=True)
    _, state = app._handle_custom_event(
        event, app._initial_custom_state(), oauth_profile=SimpleNamespace(username="test_user"),
        oauth_token=SimpleNamespace(token="real-test-oauth-token"),
    )

    assert "Choose a ZIP uploaded to this Space" in state["community_archive_status"]
    assert outside.exists()
    assert not publisher_calls

    event.community_archive_path = str(managed)
    event.community_archive_consent = False
    _, state = app._handle_custom_event(
        event, app._initial_custom_state(), oauth_profile=SimpleNamespace(username="test_user"),
        oauth_token=SimpleNamespace(token="real-test-oauth-token"),
    )
    assert "Check the publication consent box" in state["community_archive_status"]
    assert not managed.exists()
    assert not publisher_calls


def test_space_archive_publish_status_is_not_exposed_in_public_contribution_ui(monkeypatch, tmp_path):
    monkeypatch.setenv("SYSTEM", "spaces")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client-id")
    upload_root = tmp_path / "gradio"
    upload_root.mkdir()
    archive_path = upload_root / "submission.zip"
    archive_path.write_bytes(b"zip fixture")
    monkeypatch.setattr(app.gr.utils, "get_upload_folder", lambda: str(upload_root))
    monkeypatch.setenv("HF_COMMUNITY_WRITE_TOKEN", "scoped-write-token")
    monkeypatch.setattr(app, "_community_submission_archive_runtime",
                        lambda: (lambda *_a, **_k: None, lambda *_a, **_k: (_ for _ in ()).throw(
                            ValueError("<script>alert(1)</script> invalid digest"))))

    status = app._publish_community_archive(str(archive_path), submitted_by="test_user")
    monkeypatch.setattr(app, "_community_model_choices", lambda: [])
    monkeypatch.setattr(app, "_community_comparison_snapshot", lambda: {"rows": [], "beam_rows": [],
                        "selected": {}, "summary": {}, "status": "unavailable"})
    monkeypatch.setattr(app, "_benchmark_state_view", lambda _state: ([], []))
    monkeypatch.setattr(app, "_festival_lens_states", lambda: set())
    monkeypatch.setattr(app, "_load_gallery_index", lambda: None)
    monkeypatch.setattr(app, "_benchmark_summary_html", lambda: "")
    rendered = app._render_custom_app({**app._initial_custom_state(), "community_archive_status": status})
    assert "Archive not published" in status and "invalid digest" in status
    assert "invalid digest" not in rendered
    assert "<script>" not in rendered and "&lt;script&gt;" not in rendered


def test_community_model_eligibility_requires_fixed_token_limit_and_run_cap():
    assert app._community_model_config_is_eligible({
        "request_params": {"max_tokens": 4096, "output_config": {"effort": "low"}},
        "budget": {"cap_usd": 10.0},
    })
    assert not app._community_model_config_is_eligible({
        "request_params": {"max_tokens": 8192}, "budget": {"cap_usd": 10.0},
    })
    assert not app._community_model_config_is_eligible({
        "request_params": {"max_tokens": 4096}, "budget": {"cap_usd": 20.0},
    })
    assert not app._community_model_config_is_eligible(None)


def test_community_full_run_estimate_uses_runner_official_usage_estimator(monkeypatch):
    config = {
        "key": "provider/model", "request_params": {"max_tokens": 4096},
        "budget": {"cap_usd": 10.0},
        "pricing": {"currency": "USD", "per_million_tokens": {"input": 1.0, "output": 2.0},
                    "source": "test rate card", "observed_at": "2026-10-30"},
    }
    calls = []
    def estimate(_name, *, task_count):
        calls.append(task_count)
        return {"reference_suite": app.EVAL_SUITE_ID, "reference_tasks": 72,
                "task_count": task_count, "source": "official_beam_eval_v2.1_measured_usage",
                "per_run_cap_usd": 10.0, "estimated_cost_exceeds_per_run_cap": False,
                "estimated_usd": 0.0288, "mean_input_tokens_per_task": 200.0,
                "mean_output_tokens_per_task": 100.0}
    runner = SimpleNamespace(community_litellm_model_config=lambda _name: config,
                             estimate_community_evaluation_cost=estimate)
    monkeypatch.setattr(app, "_community_runtime", lambda: (runner, object()))
    estimate = app._estimate_community_full_run_cost("provider/model")
    assert estimate["estimated_usd"] == pytest.approx(0.0288)
    assert estimate["task_count"] == 72
    assert estimate["cap_usd"] == 10.0
    assert estimate["estimated_cost_exceeds_per_run_cap"] is False
    assert estimate["rate_card_source"] == "test rate card"
    assert calls == [72]


def test_community_estimate_fails_closed_when_official_usage_estimate_is_malformed(monkeypatch):
    runner = SimpleNamespace(community_litellm_model_config=lambda _name: {
        "key": "provider/model", "request_params": {"max_tokens": 4096},
        "budget": {"cap_usd": 10.0},
        "pricing": {"currency": "USD", "per_million_tokens": {"input": 1.0, "output": 2.0}},
    }, estimate_community_evaluation_cost=lambda *_args, **_kwargs: {})
    monkeypatch.setattr(app, "_community_runtime", lambda: (runner, object()))
    monkeypatch.setattr(runner, "estimate_community_evaluation_cost", lambda *_args, **_kwargs: {
        "reference_suite": "old-suite", "reference_tasks": 72, "task_count": 72,
        "source": "official_beam_eval_v2.1_measured_usage", "estimated_usd": 0.01,
        "per_run_cap_usd": 10.0, "estimated_cost_exceeds_per_run_cap": False,
        "mean_input_tokens_per_task": 1, "mean_output_tokens_per_task": 1,
    })
    assert app._estimate_community_full_run_cost("provider/model") is None


def test_full_community_run_requires_a_current_estimate_and_separate_ack(monkeypatch):
    calls = []
    config = {
        "key": "provider/model", "request_params": {"max_tokens": 4096},
        "budget": {"cap_usd": 10.0},
        "pricing": {"currency": "USD", "per_million_tokens": {"input": 1.0, "output": 2.0},
                    "source": "test rate card", "observed_at": "2026-10-30"},
    }
    def run(_model, _key, *, task_limit, full_run_cost_acknowledged):
        calls.append((task_limit, full_run_cost_acknowledged))
        return ([{"run_id": "run-1", "validity_at_1": True, "validity_at_3": True}], "suite-sha")
    runner = SimpleNamespace(community_litellm_model_config=lambda _name: config,
                             estimate_community_evaluation_cost=lambda _name, **_kwargs: {
                                 "reference_suite": app.EVAL_SUITE_ID, "reference_tasks": 72,
                                 "task_count": 72, "source": "official_beam_eval_v2.1_measured_usage",
                                 "per_run_cap_usd": 10.0,
                                 "estimated_cost_exceeds_per_run_cap": False,
                                 "estimated_usd": 0.0288, "mean_input_tokens_per_task": 200.0,
                                 "mean_output_tokens_per_task": 100.0},
                             run_community_evaluation=run)
    monkeypatch.setattr(app, "_community_runtime", lambda: (runner, object()))

    missing_ack = app._run_community_evaluation(
        "provider/model", "secret", True, task_limit=72)
    assert "confirm it before starting" in missing_ack
    assert calls == []

    estimate = app._estimate_community_full_run_cost("provider/model", runner=runner, config=config)
    stale_estimate = {**estimate, "estimated_usd": estimate["estimated_usd"] + 1}
    stale = app._run_community_evaluation(
        "provider/model", "secret", True, task_limit=72,
        full_run_cost_acknowledged=True, reviewed_estimate=stale_estimate)
    assert "Estimate this model's full run first" in stale
    assert calls == []

    complete = app._run_community_evaluation(
        "provider/model", "secret", True, task_limit=72,
        full_run_cost_acknowledged=True, reviewed_estimate=estimate)
    assert "72 requested questions" in complete
    assert calls == [(72, True)]


def test_community_runner_receives_explicit_quick_run_limit(monkeypatch):
    calls = []
    config = {
        "key": "provider/model", "request_params": {"max_tokens": 4096},
        "budget": {"cap_usd": 10.0},
        "pricing": {"currency": "USD", "per_million_tokens": {"input": 1.0, "output": 2.0}},
    }
    def run(_model, _key, *, task_limit, full_run_cost_acknowledged):
        calls.append((task_limit, full_run_cost_acknowledged))
        return ([{"run_id": "run-2", "validity_at_1": False, "validity_at_3": False}], "suite-sha")
    runner = SimpleNamespace(community_litellm_model_config=lambda _name: config,
                             run_community_evaluation=run)
    monkeypatch.setattr(app, "_community_runtime", lambda: (runner, object()))
    result = app._run_community_evaluation("provider/model", "secret", True)
    assert "12 requested questions" in result
    assert calls == [(12, False)]


def test_community_daily_identity_uses_hf_account_only_with_real_oauth(monkeypatch):
    monkeypatch.setenv("SYSTEM", "spaces")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client-id")
    anonymous = app._community_daily_identity(None, SimpleNamespace(session_hash="browser-session"))
    real_token = SimpleNamespace(token="real-test-oauth-token")
    first = app._community_daily_identity(SimpleNamespace(username="Craft-Reviewer"), None, real_token)
    same = app._community_daily_identity(SimpleNamespace(username="craft-reviewer"), None, real_token)
    mocked = app._community_daily_identity(
        SimpleNamespace(username="Space-Owner"), SimpleNamespace(session_hash="browser-session"),
        SimpleNamespace(token="mock-oauth-token-for-local-dev"),
    )
    assert anonymous != "browser-session"
    assert first == same
    assert first != anonymous
    assert mocked == anonymous


def test_community_daily_cap_blocks_second_run_before_provider_dispatch(monkeypatch):
    identity = hashlib.sha256(f"daily-test:{uuid.uuid4()}".encode()).hexdigest()
    calls = []
    config = {"request_params": {"max_tokens": 4096}, "budget": {"cap_usd": 10.0},
              "pricing": {"currency": "USD", "per_million_tokens": {"input": 1.0, "output": 2.0}}}
    def run(_model, _key, *, task_limit, full_run_cost_acknowledged):
        calls.append(task_limit)
        return ([{"run_id": "daily-run", "validity_at_1": False, "validity_at_3": False}], "suite-sha")
    runner = SimpleNamespace(community_litellm_model_config=lambda _name: config,
                             run_community_evaluation=run)
    monkeypatch.setattr(app, "_community_runtime", lambda: (runner, object()))

    first = app._run_community_evaluation(
        "provider/model", "secret", True, daily_identity=identity, enforce_daily_limit=True)
    second = app._run_community_evaluation(
        "provider/model", "secret", True, daily_identity=identity, enforce_daily_limit=True)

    assert "12 requested questions" in first
    assert "one USD 10-capped evaluation for today" in second
    assert calls == [12]


def test_estimate_action_warns_when_full_suite_projection_exceeds_run_cap(monkeypatch):
    config = {"request_params": {"max_tokens": 4096}, "budget": {"cap_usd": 10.0},
              "pricing": {"source": "test rates", "observed_at": "2026-10-30"}}
    runner = SimpleNamespace(
        community_litellm_model_config=lambda _name: config,
        estimate_community_evaluation_cost=lambda _name, *, task_count: {
            "reference_suite": app.EVAL_SUITE_ID, "reference_tasks": 72,
            "task_count": task_count, "source": "official_beam_eval_v2.1_measured_usage",
            "per_run_cap_usd": 10.0, "estimated_cost_exceeds_per_run_cap": True,
            "estimated_usd": 16.76917, "mean_input_tokens_per_task": 2500.0,
            "mean_output_tokens_per_task": 4100.0,
        },
        run_community_evaluation=lambda *_args, **_kwargs: pytest.fail("estimate action must not dispatch"),
    )
    monkeypatch.setattr(app, "_community_runtime", lambda: (runner, object()))
    monkeypatch.setattr(app, "_render_custom_app", lambda _state: "rendered")
    event = SimpleNamespace(action="community_estimate", active_tab="benchmark",
                            community_model="anthropic/claude-fable-5-1")

    _rendered, state = app._handle_custom_event(event, app._initial_custom_state())

    assert "USD 16.7692" in state["community_status"]
    assert "exceeds the USD 10 run cap" in state["community_status"]
    assert "partial coverage is likely" in state["community_status"]
    assert "No provider request was made" in state["community_status"]


def test_community_ui_defaults_to_quick_run_and_discloses_estimate_gate():
    rendered = app._render_custom_app(app._initial_custom_state())
    assert 'data-field="community_task_limit"' in rendered
    assert '<option value="12" selected>Quick run · 12 questions</option>' in rendered
    assert 'data-action="community_estimate"' in rendered
    assert 'data-field="community_full_run_cost_consent"' in rendered
    assert 'Estimate full-suite cost (no provider call)' in rendered
    assert "one capped run per account or browser session per UTC day" in rendered


def test_traditions_panel_adds_scoped_diwali_context_and_sources():
    rendered = app._render_custom_app(app._initial_custom_state())

    assert "Diwali: light, welcome, and pattern" in rendered
    assert "In some communities" in rendered
    assert "not a claim that every listed tradition is Diwali-specific" in rendered
    assert "https://www.incredibleindia.gov.in/en/festivals-and-events/diwali" in rendered
    assert "https://ignca.gov.in/walls-and-floors/" in rendered
    assert "36-entry survey spans Indian states and Union Territories" in rendered
    assert "State / Union Territory" in rendered
    assert len(app.ALL_STATES) == 36


def test_diwali_mosaic_has_one_explicit_tile_per_state_or_territory(monkeypatch):
    monkeypatch.setattr(app, "_benchmark_state_view", lambda _state: (
        [(Image.new("RGB", (2, 2), "#a84436"), "Gujarat · q-1 · periodic_raster · easy")],
        [{"state": "Gujarat", "prompt_id": "q-1", "grammar": "periodic_raster", "level": "easy"}],
    ))

    rendered = app._benchmark_state_mosaic_html()

    assert rendered.count('data-state="') == 36
    assert 'data-state="Gujarat"' in rendered
    assert "q-1" in rendered
    assert rendered.count("No replay-verified passing example in this build.") == 35


def test_community_comparison_selects_one_latest_run_per_model_and_uses_common_ids(monkeypatch, tmp_path):
    suite_path = tmp_path / "benchmarks" / "suites" / "eval-v2.1.jsonl"
    suite_path.parent.mkdir(parents=True)
    prompts = [{"id": f"q-{index}"} for index in range(72)]
    suite_bytes = ("".join(json.dumps(prompt) + "\n" for prompt in prompts)).encode("utf-8")
    suite_path.write_bytes(suite_bytes)
    digest = hashlib.sha256(canonical_jsonl_bytes(suite_bytes)).hexdigest()
    monkeypatch.setattr(app, "EVAL_SUITE_SHA256", digest)
    beam_path = tmp_path / app.EVAL_RESULT_PATH
    beam_path.parent.mkdir(parents=True)
    beam_rows = [{"prompt_id": prompt["id"], "model_key": "beam-hf",
                  "prompt_suite_sha256": digest, "validity_at_1": False,
                  "validity_at_3": True, "attempt_count": 2} for prompt in prompts]
    beam_path.write_text("".join(json.dumps(row) + "\n" for row in beam_rows), encoding="utf-8")
    monkeypatch.setattr(app, "ROOT", tmp_path)

    def rows_for(model_key, run_id, pass1, pass3, suite_digest=digest):
        return [{"prompt_id": prompt["id"], "model_key": model_key, "run_id": run_id,
                 "prompt_suite_sha256": suite_digest, "validity_at_1": pass1,
                 "validity_at_3": pass3, "attempt_count": 1} for prompt in prompts]

    recent = [
        {"run_id": "haiku-new", "model_key": "claude-haiku-5-5", "submitted_at": "2026-10-30T12:00:00Z",
         "rows": rows_for("claude-haiku-5-5", "haiku-new", True, True)},
        {"run_id": "deepseek-new", "model_key": "deepseek-v4.1-flash", "submitted_at": "2026-10-30T11:00:00Z",
         "rows": rows_for("deepseek-v4.1-flash", "deepseek-new", False, False)},
        {"run_id": "haiku-old", "model_key": "claude-haiku-5-5", "submitted_at": "2026-10-29T11:00:00Z",
         "rows": rows_for("claude-haiku-5-5", "haiku-old", False, False)},
        {"run_id": "opus-old-suite", "model_key": "claude-opus-5-5", "submitted_at": "2026-10-30T13:00:00Z",
         "rows": rows_for("claude-opus-5-5", "opus-old-suite", True, True, "f" * 64)},
    ]
    replayed = []
    publisher = SimpleNamespace(
        COMMUNITY_MODEL_PROVIDERS={"claude-haiku-5-5": "Anthropic", "claude-opus-5-5": "Anthropic",
                                   "deepseek-v4.1-flash": "CometAPI"},
        load_recent_community_runs=lambda limit: recent[:limit],
        sanitize_community_eval_rows=lambda values, **kwargs: (
            (_ for _ in ()).throw(ValueError("superseded suite"))
            if any(row.get("prompt_suite_sha256") != kwargs["expected_suite_sha256"] for row in values)
            else replayed.append(kwargs["run_id"]) or values
        ),
    )
    monkeypatch.setattr(app, "_community_runtime", lambda: (SimpleNamespace(), publisher))
    captured = {}

    def summarize(records, prompt_ids, model_keys):
        captured.update(records=records, prompt_ids=prompt_ids, model_keys=model_keys)
        models = {key: {"study": {"observed_prompt_count": 72}, "overall": {
            "mean_tokens_when_available": 800, "mean_elapsed_ms": 2000,
            "mean_request_elapsed_ms_per_prompt": 1500,
            "mean_verification_elapsed_ms_per_prompt": 2,
            "reported_cost_totals_by_currency": {"USD": 0.25},
            "estimated_cost_totals_by_currency": {"USD": 0.30},
        }} for key in model_keys}
        metrics = {"beam-hf": {"validity_at_1": 0.3, "validity_at_3": 0.6},
                   "claude-haiku-5-5": {"validity_at_1": 0.7, "validity_at_3": 0.8},
                   "deepseek-v4.1-flash": {"validity_at_1": 0.2, "validity_at_3": 0.4}}
        return {"by_model": models, "common_completed": {
            "model_keys": model_keys, "prompt_count": 60, "by_model": metrics}}

    monkeypatch.setattr(app, "_summarize_community_rows", summarize)
    app._community_comparison_snapshot.cache_clear()

    snapshot = app._community_comparison_snapshot()
    rendered = app._community_leaderboard_html(snapshot)

    assert snapshot["selected"]["claude-haiku-5-5"]["run_id"] == "haiku-new"
    assert replayed == ["haiku-new", "deepseek-new"]
    assert "claude-opus-5-5" not in snapshot["selected"]
    assert len(captured["records"]) == 216
    assert captured["model_keys"] == ["beam-hf", "claude-haiku-5-5", "deepseek-v4.1-flash"]
    assert "Compared on 60 identical completed questions" in rendered
    assert "Tokens 800 mean" in rendered
    assert "request 1500 ms mean/prompt" in rendered
    assert "reported USD 0.2500" in rendered
    assert "Rank 1" in rendered and "Rank 3" in rendered
    assert '<table class="pramaan-leaderboard"' in rendered
    assert "Model / provenance" in rendered
    assert "Pass@3 · shared tasks" in rendered
    assert "Community · self-reported model ID" in rendered
    assert "Rate-card estimate" in rendered
    app._community_comparison_snapshot.cache_clear()


def test_community_gallery_replays_candidate_before_rendering(monkeypatch):
    candidate = json.loads((app.ROOT / "examples" / "ajrakh_starfield.pattern.json").read_text(encoding="utf-8"))
    pattern = parse_pattern(json.dumps(candidate))
    checks, masks, grid = verify_pattern(pattern)
    certificate = make_certificate(pattern, checks, masks, grid)
    row = {
        "model_key": "claude-haiku-5-5", "prompt_id": "q-1", "state": "Gujarat",
        "tradition_id": "gj_ajrakh", "accepted_mask_sha256": certificate["mask_sha256"],
        "attempts": [{"candidate": candidate,
                      "report": {"checks": checks, "all_hard_constraints_pass": True}}],
    }

    other_model_row = {**row, "model_key": "deepseek-v4.1-flash", "prompt_id": "q-2"}
    snapshot = {"rows": [row, other_model_row]}
    rendered = app._community_state_gallery(snapshot, "Gujarat")
    haiku_only = app._community_state_gallery(snapshot, "Gujarat", "claude-haiku-5-5")
    deepseek_only = app._community_state_gallery(snapshot, "Gujarat", "deepseek-v4.1-flash")
    tampered = {**row, "accepted_mask_sha256": "0" * 64}
    rejected = app._community_state_gallery({"rows": [tampered]}, "Gujarat")

    assert "Replayed accepted community design" in rendered
    assert "claude-haiku-5-5" in rendered and "q-1" in rendered
    assert "q-1" in haiku_only and "q-2" not in haiku_only
    assert "q-2" in deepseek_only and "q-1" not in deepseek_only
    assert "Replayed accepted community design" not in rejected


def test_gallery_model_filter_is_persisted_by_the_custom_ui_event(monkeypatch):
    rendered_state = {}
    monkeypatch.setattr(app, "_render_custom_app", lambda state: rendered_state.update(state) or "rendered")
    event = SimpleNamespace(action="gallery_model", gallery_model="claude-haiku-5-5", active_tab="leaderboard")

    rendered, state = app._handle_custom_event(event, app._initial_custom_state())

    assert rendered == "rendered"
    assert state["gallery_model"] == "claude-haiku-5-5"
    assert rendered_state["gallery_model"] == "claude-haiku-5-5"
    assert state["active_tab"] == "leaderboard"
    assert rendered_state["active_tab"] == "leaderboard"


def test_gallery_state_tile_selection_updates_existing_state_filter(monkeypatch):
    rendered_state = {}
    monkeypatch.setattr(app, "_render_custom_app", lambda state: rendered_state.update(state) or "rendered")
    event = SimpleNamespace(action="gallery_state_select", gallery_state="Tamil Nadu", active_tab="benchmark")

    rendered, state = app._handle_custom_event(event, app._initial_custom_state())

    assert rendered == "rendered"
    assert state["benchmark_state"] == "Tamil Nadu"
    assert rendered_state["benchmark_state"] == "Tamil Nadu"
    assert state["active_tab"] == "benchmark"


def test_gallery_contains_selectable_36_state_overview_and_preserves_data_empty_state(monkeypatch):
    monkeypatch.setattr(app, "_load_gallery_index", lambda: None)
    monkeypatch.setattr(app, "_benchmark_state_view", lambda _state: ([], {
        "status": "Replay-audited eval-v2.1 gallery data unavailable in this build."
    }))
    monkeypatch.setattr(app, "_community_comparison_snapshot", lambda: {
        "status": "Community runs unavailable.", "rows": [], "selected": {}, "summary": {},
    })
    monkeypatch.setattr(app, "_home_reference_rows", lambda: [
        {"state": state, "task_id": f"{index}", "level": "easy", "grammar": "counted_grid",
         "checks": [], "mask_sha256": "a" * 64, "thumbnail_uri": "data:image/png;base64,aGVsbG8="}
        for index, state in enumerate(app.ALL_STATES)
    ])

    rendered = app._render_custom_app(app._initial_custom_state())

    home_html = rendered.split('data-tab-panel="benchmark"', 1)[1].split('<div class="pramaan-ui-tabpanel"', 1)[0]
    assert home_html.count('data-home-wall-tile') == 36
    assert home_html.count('data-wall-mark=') == 72
    assert home_html.count('aria-hidden="true">E○</span>') == 36
    assert home_html.count('aria-hidden="true">H○</span>') == 36
    assert 'class="pramaan-wall-name-row"><span class="pramaan-wall-state-name" title=' in home_html
    assert 'class="pramaan-wall-marks"><span class="pramaan-wall-mark" data-wall-mark="easy"' in home_html
    assert '<details class="pramaan-wall-model-picker"><summary>Compare models</summary>' in home_html
    assert '<details class="pramaan-wall-model-picker" open' not in home_html
    assert 'role="grid" aria-label="36 state and Union Territory pattern map"' in home_html
    assert 'data-wall-mode-choice="reference"' in home_html
    assert 'data-wall-mode-choice="attempt"' in home_html
    assert 'data-wall-state-sheet' in home_html
    assert "Design gallery" not in rendered
    assert "No replay-verified passing example in this build." not in rendered


def test_home_wall_controls_render_without_legacy_gallery_controls(monkeypatch):
    index = {
        "suite_id": app.EVAL_SUITE_ID,
        "suite_sha256": app.EVAL_SUITE_SHA256,
        "validator_version": app.VALIDATOR_VERSION,
        "states": [{"state": state, "grid": list(app.state_grid_cell(state)), "tradition": "Example tradition"}
                   for state in app.ALL_STATES],
        "arms": [{"key": "beam-hf", "label": "Beam"}],
        "common_prompt_ids": ["gj_easy"],
    }
    arm = {"tasks": {"gj_easy": {"state": "Gujarat", "level": "easy", "passed_at": 1,
                                      "attempts": []}}}
    monkeypatch.setattr(app, "_load_gallery_index", lambda: index)
    monkeypatch.setattr(app, "_gallery_json", lambda _path: arm)
    rendered = app._render_custom_app(app._initial_custom_state())
    assert 'data-field="gallery_model"' not in rendered
    assert 'data-wall-model-choice="beam-hf"' in rendered
    assert '<details class="pramaan-wall-model-picker"><summary>Compare models</summary>' in rendered
    assert '<details class="pramaan-wall-model-picker" open' not in rendered
    assert 'data-wall-mode-choice="attempt"' in rendered
    assert 'data-tab-button="leaderboard"' in rendered
    assert 'data-tab-button="benchmark"' in rendered
    assert 'aria-label="Choose a model for the Wall"' in rendered
    assert 'title="Gujarat · Example tradition"' in rendered


def test_wall_model_and_art_mode_preferences_persist_in_custom_event(monkeypatch):
    monkeypatch.setattr(app, "_render_custom_app", lambda _state: "rendered")
    state = app._initial_custom_state()

    _rendered, state = app._handle_custom_event(
        SimpleNamespace(action="wall_model", wall_model="claude-haiku-5-5", active_tab="benchmark"), state)
    _rendered, state = app._handle_custom_event(
        SimpleNamespace(action="wall_mode", wall_mode="attempt", active_tab="benchmark"), state)

    assert state["wall_model"] == "claude-haiku-5-5"
    assert state["wall_mode"] == "attempt"
    assert state["active_tab"] == "benchmark"


def test_quilt_controls_persist_in_custom_ui_event(monkeypatch):
    monkeypatch.setattr(app, "_render_custom_app", lambda _state: "rendered")
    event = SimpleNamespace(action="quilt_mode", quilt_mode="compare", active_tab="benchmark")
    _rendered, state = app._handle_custom_event(event, app._initial_custom_state())
    assert state["quilt_mode"] == "compare"


def test_leaderboard_tab_survives_refresh_event(monkeypatch):
    rendered_state = {}
    monkeypatch.setattr(app, "_render_custom_app", lambda state: rendered_state.update(state) or "rendered")
    event = SimpleNamespace(action="community_refresh", active_tab="leaderboard")

    rendered, state = app._handle_custom_event(event, app._initial_custom_state())

    assert rendered == "rendered"
    assert state["active_tab"] == "leaderboard"
    assert rendered_state["active_tab"] == "leaderboard"
    assert state["community_loaded"] is True
    assert rendered_state["community_loaded"] is True


def test_festival_lens_includes_only_traditions_with_sourced_claims(monkeypatch):
    records = {
        "Kerala": [SimpleNamespace(primary_state="Kerala", festival_lens={
            "surface": "floor", "diwali_status": "sourced_institutional", "claim_ids": ["museum-1"]})],
        "Tamil Nadu": [SimpleNamespace(primary_state="Tamil Nadu", festival_lens={
            "surface": "floor", "diwali_status": "sourced_secondary", "claim_ids": ["source-2"]})],
        "Assam": [SimpleNamespace(primary_state="Assam", festival_lens={
            "surface": None, "diwali_status": "none", "claim_ids": []})],
        "Delhi": [SimpleNamespace(primary_state="Delhi", festival_lens={
            "surface": "wall", "diwali_status": "sourced_institutional", "claim_ids": []})],
        "West Bengal": [SimpleNamespace(primary_state="West Bengal", festival_lens={
            "surface": "floor", "diwali_status": "sourced_institutional", "claim_ids": [""]})],
        "Mizoram": [SimpleNamespace(primary_state="Mizoram", festival_lens={
            "surface": None, "diwali_status": "sourced_institutional", "claim_ids": ["source-3"]})],
        "Karnataka": [SimpleNamespace(primary_state="Tamil Nadu", festival_lens={
            "surface": "floor", "diwali_status": "sourced_secondary", "claim_ids": ["source-2"]})],
    }
    monkeypatch.setattr(app, "ALL_STATES", tuple(records))
    monkeypatch.setattr(app, "_tradition_choices", lambda state: records[state])

    assert app._festival_lens_states() == {"Kerala", "Tamil Nadu"}


def test_state_mosaic_marks_only_source_supported_festival_tiles():
    image = Image.new("RGB", (2, 2), "white")

    mosaic = app._benchmark_state_mosaic_html(
        [(image, "sample")], [{"state": "Kerala", "prompt_id": "k-1", "grammar": "kolam_loop"}],
        festival_lens_states={"Kerala"}, festival_lens_active=True,
    )

    assert 'class="pramaan-state-mosaic is-festival-lens"' in mosaic
    assert 'data-state="Kerala" data-festival-linked="true"' in mosaic
    assert 'data-state="Tamil Nadu" data-festival-linked="false"' in mosaic


def test_state_mosaic_labels_missing_artifacts_instead_of_implying_zero_passes():
    mosaic = app._benchmark_state_mosaic_html(
        [], {"status": "Accepted benchmark previews will appear when audited eval-v2.1 trajectories are included in this Space build."}
    )

    assert "trajectories are included in this Space build" in mosaic
    assert "evaluation data unavailable" in mosaic
    assert "no replay-verified passing example in this build" not in mosaic


def test_home_wall_layout_is_bounded_and_has_outcome_legend():
    rendered = app._render_custom_app(app._initial_custom_state())

    assert "grid-template-columns: repeat(10, minmax(0, 1fr))" in app.PRAMAAN_CSS
    assert "width: 100%" in app.PRAMAAN_CSS
    assert "width: max-content" not in app.PRAMAAN_CSS
    assert "min-width: 640px" in app.PRAMAAN_CSS
    assert "@media (max-width: 600px)" in app.PRAMAAN_CSS
    assert ".pramaan-state-mosaic-scroll { box-sizing: border-box; max-width: 100%; min-width: 0; overflow-x: auto; overscroll-behavior-inline: contain; width: 100%; }" in app.PRAMAAN_CSS
    assert ".pramaan-ui, .pramaan-state-mosaic-scroll { max-width: 100%; min-width: 0; width: 100%; }" in app.PRAMAAN_CSS
    assert 'class="pramaan-wall-legend"' in rendered
    for label in ("Passed first try", "Passed after repair", "Did not pass", "Not run"):
        assert label in rendered


def test_model_quilt_marks_each_evidence_state_with_a_distinct_shape():
    index = {
        "states": [],
        "arms": [{"key": "beam-hf", "label": "Beam"},
                 {"key": "claude-sonnet-5-5", "label": "Claude Sonnet"}],
    }
    documents = {
        "beam-hf": {"tasks": {
            "gj_easy": {"state": "Gujarat", "level": "easy", "passed_at": 1, "attempts": []},
            "as_easy": {"state": "Assam", "level": "easy", "attempts": [{"index": 1, "passed": False}]},
            "kl_easy": {"state": "Kerala", "level": "easy", "source": "community", "attempts": []},
        }},
        "claude-sonnet-5-5": {"tasks": {
            "py_easy": {"state": "Puducherry", "level": "easy", "passed_at": 2, "attempts": []},
            "tn_easy": {"state": "Tamil Nadu", "level": "easy", "passed_at": 3, "attempts": []},
        }},
    }

    rendered = app.render_data_quilt(index, documents, model_key="beam-hf", level="easy",
                                     mode="single", compare_model="claude-sonnet-5-5")

    assert 'data-tile-state="verified-first"' in rendered
    assert 'data-tile-state="failed-all"' in rendered
    assert 'data-tile-state="accepted-other-model"' in rendered
    assert 'data-tile-state="not-evaluated"' in rendered
    assert 'data-tile-state="community-run"' in rendered
    assert 'class="pramaan-seal-stamp"' not in rendered


def test_festival_lens_toggle_persists_and_keeps_benchmark_active(monkeypatch):
    rendered_state = {}
    monkeypatch.setattr(app, "_render_custom_app", lambda state: rendered_state.update(state) or "rendered")
    event = SimpleNamespace(action="festival_lens", festival_lens=True, active_tab="benchmark")

    rendered, state = app._handle_custom_event(event, app._initial_custom_state())

    assert rendered == "rendered"
    assert state["festival_lens"] is True
    assert state["active_tab"] == "benchmark"
    assert rendered_state["festival_lens"] is True


def test_community_leaderboard_does_not_claim_missing_beam_baseline():
    snapshot = {"status": "Community results loaded.", "selected": {
        "claude-haiku-5-5": {"run_id": "run-1"}}, "summary": {
        "by_model": {
            "beam-hf": {"study": {"observed_prompt_count": 0}, "overall": {}},
            "claude-haiku-5-5": {"study": {"observed_prompt_count": 2}, "overall": {}},
        },
        "common_completed": {"model_keys": ["claude-haiku-5-5"], "prompt_count": 2,
                              "by_model": {"claude-haiku-5-5": {}}},
    }}

    rendered = app._community_leaderboard_html(snapshot)

    assert "No canonical Beam rows in this build" in rendered
    assert "Rank 1" not in rendered


def test_leaderboard_and_gallery_show_explicit_empty_states_without_artifacts(monkeypatch):
    monkeypatch.setattr(app, "_load_gallery_index", lambda: None)
    empty_snapshot = {"status": "No consented community runs are available yet.", "rows": [],
                      "beam_rows": [], "selected": {}, "summary": {}}

    leaderboard = app._canonical_leaderboard_html()
    gallery_unavailable = app._canonical_state_gallery_html(None, "Gujarat")
    gallery_empty = app._canonical_state_gallery_html({
        "suite_id": app.EVAL_SUITE_ID, "suite_sha256": app.EVAL_SUITE_SHA256, "arms": [],
    }, "Gujarat")
    community_leaderboard = app._community_leaderboard_html(empty_snapshot)
    community_gallery = app._community_state_gallery(empty_snapshot, "Gujarat")

    assert "replay-audited gallery index for this suite is not available" in leaderboard
    assert 'role="status"' in leaderboard
    assert "replay-audited model gallery is not available" in gallery_unavailable
    assert "No model task records are available for Gujarat" in gallery_empty
    assert "No consented community runs are available yet." in community_leaderboard
    assert "No replay-verified community designs are available for this state yet." in community_gallery


def test_missing_model_gallery_data_does_not_restore_the_removed_gallery(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "ROOT", tmp_path)
    monkeypatch.setattr(app, "_load_gallery_index", lambda: None)

    rendered = app._render_custom_app(app._initial_custom_state())

    assert 'aria-label="36 state and Union Territory pattern map"' in rendered
    assert 'data-testid="arena"' in rendered
    assert "Design gallery" not in rendered
    assert "No replay-verified passing example in this build." not in rendered


def test_canonical_leaderboard_ranks_only_the_shared_prompt_intersection(monkeypatch, tmp_path):
    gallery_dir = tmp_path / "assets" / "gallery"
    gallery_dir.mkdir(parents=True)
    arms = [
        {"key": "beam-hf", "label": "Beam", "provider": "Hugging Face", "run_kind": "official",
         "n_tasks": 3, "completed_prompt_ids": ["x", "y", "z"], "mean_reward": 0.8,
         "tokens_per_task": 70, "est_cost_per_verified": 0.01, "median_latency_ms": 120,
         "normalizations": {"strip_code_fence": 2, "strip_think_block": 1}, "normalization_calls": 8,
         "by_grammar": {"G1": {"n": 2, "passed_at_3": 2}}},
        {"key": "model-b", "label": "Model B", "provider": "Provider B", "run_kind": "official",
         "n_tasks": 2, "completed_prompt_ids": ["x", "y"], "mean_reward": 0.6,
         "normalizations": {"strip_code_fence": 0, "strip_think_block": 3}, "normalization_calls": 10,
         "tokens_per_task": 80, "est_cost_per_verified": 0.02, "median_latency_ms": 140,
         "by_grammar": {"G1": {"n": 2, "passed_at_3": 1}}},
        {"key": "model-c", "label": "Model C", "provider": "Provider C", "run_kind": "official",
         "n_tasks": 2, "completed_prompt_ids": ["x", "y"], "mean_reward": 0.8,
         "tokens_per_task": 90, "est_cost_per_verified": 0.03, "median_latency_ms": 160},
        {"key": "model-partial", "label": "Partial model", "provider": "Provider D", "run_kind": "community",
         "n_tasks": 1, "completed_prompt_ids": ["x"], "mean_reward": 0.5},
    ]
    index = {
        "suite_id": app.EVAL_SUITE_ID, "suite_sha256": app.EVAL_SUITE_SHA256,
        "validator_version": app.VALIDATOR_VERSION, "states": [{} for _ in app.ALL_STATES],
        "common_prompt_ids": ["x", "y"],
        "arms": arms, "baselines": [{"key": "empty-tile", "n_tasks": 3, "passed": 0, "pass_at_1": 0.0}],
    }
    (gallery_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (gallery_dir / "beam-hf.json").write_text(json.dumps({"key": "beam-hf", "tasks": {
        "x": {"status": "completed", "passed_at": 1, "grammar": "periodic_raster"},
        "y": {"status": "completed", "passed_at": 3, "grammar": "periodic_raster"},
        "z": {"status": "completed", "passed_at": None, "grammar": "jaali_lattice"},
    }}), encoding="utf-8")
    (gallery_dir / "model-b.json").write_text(json.dumps({"key": "model-b", "tasks": {
        "x": {"status": "completed", "passed_at": 3, "grammar": "periodic_raster"},
        "y": {"status": "completed", "passed_at": None, "grammar": "periodic_raster"},
    }}), encoding="utf-8")
    (gallery_dir / "model-c.json").write_text(json.dumps({"key": "model-c", "tasks": {
        "x": {"status": "completed", "passed_at": 1, "grammar": "periodic_raster"},
        "y": {"status": "completed", "passed_at": 3, "grammar": "periodic_raster"},
    }}), encoding="utf-8")
    (gallery_dir / "model-partial.json").write_text(json.dumps({"key": "model-partial", "tasks": {
        "x": {"status": "completed", "passed_at": 1, "grammar": "periodic_raster"},
    }}), encoding="utf-8")
    monkeypatch.setattr(app, "ROOT", tmp_path)

    rendered = app._canonical_leaderboard_html()

    assert rendered.index("Beam") < rendered.index("Model B")
    assert "Shared questions: 2" in rendered
    assert "50.0%" in rendered and "100.0%" in rendered
    assert "95% Wilson" in rendered
    assert "Tie · rank 1" in rendered
    assert "Not ranked · incomplete shared coverage" in rendered
    assert "empty-tile" in rendered and "0 / 3" in rendered
    assert "G1" in rendered and "G7" in rendered
    assert "Output cleanup" in rendered
    assert "Fence 25.0% (2/8); think 12.5% (1/8)" in rendered
    assert "Fence 0.0% (0/10); think 30.0% (3/10)" in rendered
    assert 'aria-label="Repair arc, pass at one to pass at three"' in rendered
    assert 'aria-label="Cost per verified result versus pass at three"' in rendered


def test_leaderboard_exposes_anchored_sections_and_filterable_sortable_results(monkeypatch, tmp_path):
    gallery_dir = tmp_path / "assets" / "gallery"
    gallery_dir.mkdir(parents=True)
    (gallery_dir / "index.json").write_text(json.dumps({
        "suite_id": app.EVAL_SUITE_ID, "suite_sha256": app.EVAL_SUITE_SHA256,
        "validator_version": app.VALIDATOR_VERSION, "states": [{} for _ in app.ALL_STATES],
        "common_prompt_ids": [app.PROOF_REPLAY_PROMPT_ID], "arms": [
            {"key": "beam-hf", "label": "Beam", "provider": "Hugging Face",
             "completed_prompt_ids": [app.PROOF_REPLAY_PROMPT_ID]},
        ], "baselines": [],
    }), encoding="utf-8")
    (gallery_dir / "beam-hf.json").write_text(json.dumps({"tasks": {
        app.PROOF_REPLAY_PROMPT_ID: {
            "status": "completed", "passed_at": 1, "grammar": "periodic_raster",
            "brief": "A shared test brief for the compact Arena.", "attempts": [],
        },
    }}), encoding="utf-8")
    assets_dir = tmp_path / "assets"
    (assets_dir / "pramaan-mark.png").write_bytes((app.ROOT / "assets" / "pramaan-mark.png").read_bytes())
    monkeypatch.setattr(app, "ROOT", tmp_path)
    monkeypatch.setattr(app, "_benchmark_summary", lambda: {
        "expected_prompt_count": 72, "by_model": {"beam-hf": {
            "label": "Beam", "study": {"expected_prompt_count": 72, "observed_prompt_count": 1},
            "overall": {"n": 1, "validity_at_1": 1.0, "validity_at_3": 1.0},
        }},
    })
    monkeypatch.setattr(app, "_community_comparison_snapshot", lambda: {
        "status": "No consented community runs are available yet.",
        "rows": [], "selected": {}, "summary": {},
    })

    rendered = app._render_custom_app(app._initial_custom_state())

    assert 'href="#model-arena"' in rendered
    assert 'href="#leaderboard-community-results"' in rendered
    assert 'href="#leaderboard-methodology"' in rendered
    assert 'data-leaderboard-search' in rendered
    assert 'data-leaderboard-provenance' in rendered
    assert 'data-leaderboard-sort' in rendered
    assert 'data-leaderboard-row' in rendered
    assert 'data-score-pass3="1.00000000"' in rendered
    models_panel = rendered.split('data-tab-panel="leaderboard"', 1)[1].split('<div class="pramaan-ui-tabpanel"', 1)[0]
    methodology = models_panel.split('<details id="leaderboard-methodology"', 1)[1].split('</details>', 1)[0]
    assert 'open' not in methodology[:80]
    ordered_sections = [
        'data-testid="model-report-chart"', 'data-testid="model-results-matrix"',
        'data-testid="first-try-failures-chart"', 'data-testid="model-cost-chart"',
        'id="leaderboard-full-results"', 'id="model-arena"',
        'id="leaderboard-methodology"', 'id="leaderboard-community-results"',
    ]
    assert [models_panel.index(section) for section in ordered_sections] == sorted(
        models_panel.index(section) for section in ordered_sections
    )
    assert 'data-testid="arena"' in models_panel
    assert 'data-testid="model-results-matrix"' in models_panel
    assert 'data-testid="model-report-table"' in models_panel
    assert '<section id="leaderboard-full-results"' in models_panel
    assert 'data-testid="model-latency-chart"' not in models_panel
    assert models_panel.index('data-testid="model-cost-chart"') < models_panel.index('id="leaderboard-full-results"')
    assert 'data-testid="model-protocol-table"' not in models_panel
    for removed in (
        "Design gallery", "No replay-verified passing example in this build",
        "Selected state details", "Beam examples", "Replayed task evidence",
        "Same-brief model wall", "Community designs", "Sign in to publish",
    ):
        assert removed not in models_panel


def test_models_panel_uses_final_eleven_model_summary_without_partial_rank_text():
    rendered_report = app.render_model_report(app._benchmark_summary())
    row_keys = __import__("re").findall(
        r'data-testid="model-report-row-([^\"]+)"', rendered_report,
    )

    assert len(row_keys) == 11
    assert len(set(row_keys)) == 11
    assert "Not ranked" not in rendered_report
    assert "Claude Fable 5.1" in rendered_report
    assert "72 of 72 evaluated" in rendered_report
    assert rendered_report.count("<table") == 1
    assert "pramaan-model-wilson" not in rendered_report
    assert "Not ranked" not in rendered_report


def test_readme_and_models_page_do_not_describe_retired_charts():
    from pathlib import Path

    readme = (Path(app.ROOT) / "README.md").read_text(encoding="utf-8").casefold()
    rendered = app._render_custom_app(app._initial_custom_state())
    models_panel = rendered.split('data-tab-panel="leaderboard"', 1)[1].split(
        '<div class="pramaan-ui-tabpanel"', 1
    )[0].casefold()
    for phrase in ("ranking chart", "stacked models chart", "passes-by-attempt",
                   "passes by attempt", "latency chart", "latency scatter"):
        assert phrase not in readme
        assert phrase not in models_panel


def test_readme_and_space_copy_have_no_stale_benchmark_or_replay_claims():
    from html.parser import HTMLParser

    class VisibleCopy(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in {"script", "style"}:
                self._skip += 1
            if not self._skip:
                for name, value in attrs:
                    if name in {"aria-label", "alt", "title", "placeholder"} and value:
                        self.parts.append(value)

        def handle_endtag(self, tag):
            if tag in {"script", "style"} and self._skip:
                self._skip -= 1

        def handle_data(self, data):
            if not self._skip and data.strip():
                self.parts.append(data.strip())

    def visible_text(markup):
        parser = VisibleCopy()
        parser.feed(markup)
        return " ".join(parser.parts)

    readme = (Path(app.ROOT) / "README.md").read_text(encoding="utf-8")
    rendered_space = visible_text(app._render_custom_app(app._initial_custom_state()))
    combined = f"{readme}\n{rendered_space}"
    lower = combined.casefold()

    assert not re.search(r"(?:benchmark|eval-v2\.1|suite).{0,100}ajrakh|ajrakh.{0,100}(?:benchmark|eval-v2\.1|suite)",
                         combined, re.I | re.S)
    assert not re.search(r"\b47\s+tasks?\b|\b47[- ]task(?:s|\b)", combined, re.I)
    assert "right first try" not in lower
    assert not re.search(r"\b(?:correct|correctly|accuracy)\b.{0,50}\b(?:pass(?:ed|es|ing)?|checks?|score|rate)\b|"
                         r"\b(?:pass(?:ed|es|ing)?|checks?|score|rate)\b.{0,50}\b(?:correct|correctly|accuracy)\b",
                         combined, re.I | re.S)
    for phrase in ("ranking chart", "stacked models chart", "passes-by-attempt",
                   "passes by attempt", "latency chart"):
        assert phrase not in lower

    fable_row = next(line for line in readme.splitlines() if line.startswith("| Claude Fable 5.1 |"))
    assert "72/72" in fable_row and "$22" in fable_row and "$10" not in fable_row
    replay = visible_text(app._proof_replay_html()).casefold()
    assert "himachal pradesh" in replay and "kinnauri" in replay
    assert "gujarat" not in replay and "bandhani" not in replay
    readme_replay = readme.split("### A replay with a visible counterexample", 1)[1].split(
        "### Beam designs that the checker passed", 1
    )[0].casefold()
    assert "himachal pradesh" in readme_replay and "kinnauri" in readme_replay
    assert "gujarat" not in readme_replay and "bandhani" not in readme_replay

    header = app._brand_header_html().casefold()
    header_text = visible_text(header).casefold()
    assert "seal" not in header_text
    assert "pramaan-brand-seal" not in header and "#pramaan-proof-seal" not in header


def test_readme_examples_exist_and_every_local_image_resolves():
    import re
    from pathlib import Path
    from urllib.parse import unquote, urlsplit

    root = Path(app.ROOT)
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "## Examples" in readme
    assert "## Replay certificates and trajectories" in readme

    image_paths = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    image_paths.extend(re.findall(r'<img\b[^>]*\bsrc=["\']([^"\']+)', readme, re.I))
    assert image_paths
    local_paths = [urlsplit(path).path for path in image_paths
                   if not urlsplit(path).scheme and not path.startswith("#")]
    assert local_paths
    missing = [path for path in local_paths if not (root / unquote(path)).is_file()]
    assert not missing, f"README local image paths do not resolve: {missing}"


def test_model_methodology_lists_current_models_and_provider_request_differences():
    arms = []
    for index in range(10):
        params = {"max_tokens": 4096, "output_config": {"effort": "low"}}
        provider = "Anthropic"
        if index == 0:
            provider, params = "Hugging Face", {"temperature": 0.4, "max_tokens": 4096,
                                                  "reasoning_effort": "minimal"}
        elif index == 4:
            provider, params = "CometAPI", {"temperature": 0.4, "max_tokens": 4096,
                                              "extra_body": {"thinking": {"type": "disabled"}}}
        elif index >= 5:
            provider, params = "OpenAI", {"max_completion_tokens": 4096,
                                           "reasoning_effort": "none" if index % 2 else "low"}
        arms.append({"key": f"model-{index}", "label": f"Model {index}",
                     "provider": provider, "request_params": params})
    rendered = app._model_methodology_html({
        "common_prompt_ids": [f"task-{index}" for index in range(47)], "arms": arms,
    }, {"expected_prompt_count": 72})

    assert '<details id="leaderboard-methodology"' in rendered
    assert "10 models · 72 suite tasks" in rendered
    assert rendered.count('data-model-protocol-row=') == 10
    assert "temperature 0.4" in rendered and "effort low" in rendered
    assert 'Model 4' in rendered and 'thinking disabled' in rendered
    assert 'Model 5' in rendered and 'effort none' in rendered
    assert 'reasoning cap' in rendered
    assert "<table" not in rendered
    single = app._model_methodology_html({
        "common_prompt_ids": ["task-only"], "arms": [arms[0]],
    }, {"expected_prompt_count": 1})
    assert "Methodology · 1 model · 1 suite task" in single


def test_build_with_it_protocol_note_and_methodology_read_saved_run_caps():
    summary = app._benchmark_summary()
    note = app._benchmark_protocol_note(summary)
    methodology = app._model_methodology_html(app._load_gallery_index(), summary)

    assert note == (
        "Every model ran all 72 tasks with the same verifier, a 4,096-token cap and up to 3 proposals. "
        "Each paid model had a USD 10 cap, except Claude Fable 5.1 and Claude Fable 5.2 at USD 22. "
        "Request settings differ by provider and are listed in Methodology."
    )
    assert methodology.count("spend cap USD 22") == 2
    assert methodology.count("spend cap USD 10") == 8
    assert "47 tasks" not in note


def test_arena_defaults_to_beam_and_three_highest_scoring_evaluated_models():
    summary = {"by_model": {
        "beam-hf": {"overall": {"n": 72, "validity_at_1": .2, "validity_at_3": .4}},
        "model-a": {"overall": {"n": 40, "validity_at_1": .7, "validity_at_3": .8}},
        "model-b": {"overall": {"n": 40, "validity_at_1": .6, "validity_at_3": .9}},
        "model-c": {"overall": {"n": 40, "validity_at_1": .5, "validity_at_3": .7}},
        "model-d": {"overall": {"n": 0, "validity_at_1": 0, "validity_at_3": 0}},
    }}
    assert app._arena_default_model_keys(summary) == ["beam-hf", "model-b", "model-a", "model-c"]


def test_gallery_index_loader_rejects_a_superseded_suite(monkeypatch, tmp_path):
    gallery_dir = tmp_path / "assets" / "gallery"
    gallery_dir.mkdir(parents=True)
    stale_index = {"suite_id": app.EVAL_SUITE_ID, "suite_sha256": "f" * 64,
                   "states": [], "arms": [], "common_prompt_ids": []}
    (gallery_dir / "index.json").write_text(json.dumps(stale_index), encoding="utf-8")
    monkeypatch.setattr(app, "ROOT", tmp_path)

    assert app._load_gallery_index() is None


def test_canonical_gallery_groups_same_brief_verified_and_failed_tiles(monkeypatch, tmp_path):
    import base64
    from io import BytesIO

    gallery_dir = tmp_path / "assets" / "gallery"
    gallery_dir.mkdir(parents=True)
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "teal").save(buffer, format="PNG")
    tile = base64.b64encode(buffer.getvalue()).decode("ascii")
    index = {"suite_id": app.EVAL_SUITE_ID, "suite_sha256": app.EVAL_SUITE_SHA256,
             "common_prompt_ids": ["gj-bandhani-easy"], "arms": [
                 {"key": "beam-hf", "label": "Beam", "completed_prompt_ids": ["gj-bandhani-easy"]},
                 {"key": "model-b", "label": "Model B", "completed_prompt_ids": ["gj-bandhani-easy"]},
             ]}
    beam = {"key": "beam-hf", "tasks": {"gj-bandhani-easy": {
        "state": "Gujarat", "brief": "A shared Bandhani diamond and dots.", "status": "completed",
        "passed_at": 2, "attempts": [{"passed": False, "failing_checks": ["symmetry_d4"], "tile_png_b64": tile},
                                      {"passed": True, "failing_checks": [], "tile_png_b64": tile}],
        "cert_sha256": "a" * 64, "mask_sha256": "b" * 64,
    }}}
    model_b = {"key": "model-b", "tasks": {"gj-bandhani-easy": {
        "state": "Gujarat", "brief": "A shared Bandhani diamond and dots.", "status": "completed",
        "passed_at": None, "attempts": [{"passed": False, "failing_checks": ["symmetry_d4"], "tile_png_b64": tile}],
    }}}
    (gallery_dir / "beam-hf.json").write_text(json.dumps(beam), encoding="utf-8")
    (gallery_dir / "model-b.json").write_text(json.dumps(model_b), encoding="utf-8")
    monkeypatch.setattr(app, "ROOT", tmp_path)

    rendered = app._canonical_state_gallery_html(index, "Gujarat")
    beam_only = app._canonical_state_gallery_html(index, "Gujarat", "beam-hf")

    assert "Same-brief model comparison" in rendered
    assert "A shared Bandhani diamond and dots." in rendered
    assert "Verified · passed on attempt 2" in rendered
    assert "Not verified · no passing attempt" in rendered
    assert "Symmetry (D4)" in rendered
    assert "symmetry_d4" not in rendered
    assert "certificate SHA-256" in rendered and "mask SHA-256" in rendered
    assert "Model B" not in beam_only and "Beam" in beam_only


def test_arena_explains_parse_failure_without_showing_raw_check_ids():
    from pramaan.ui.compare import render_arena

    rendered = render_arena(
        {"suite_id": "suite", "suite_sha256": "a" * 64,
         "arms": [{"key": "beam-hf", "label": "Beam"}]},
        {"beam-hf": {"tasks": {"prompt-1": {
            "brief": "A shared geometry brief.", "passed_at": None,
            "attempts": [{"index": 1, "passed": False, "failing_checks": ["candidate_parse"]}],
        }}}},
        suite_id="suite", suite_sha256="a" * 64, prompt_id="prompt-1",
        replay_prompt_id="other-prompt",
    )
    assert "Could not be read as a pattern." in rendered
    assert "Readable pattern" in rendered
    assert "candidate_parse" not in rendered


def test_selected_wall_model_status_is_full_size_and_uses_accent_text():
    css = app.PRAMAAN_CSS
    assert ".pramaan-wall-model-chip small { color: var(--pramaan-muted); font-size: 12px; opacity: 1; }" in css
    assert '.gradio-container .pramaan-ui .pramaan-wall-model-chip[aria-checked="true"] small' in css
    assert "color: var(--pramaan-on-accent) !important; font-size: 12px !important; opacity: 1 !important" in css
