"""The studio at /studio: Try it (spec 8.6) and Submit a run (spec 8.5)."""
from __future__ import annotations

import copy
import html
import json
import re
from datetime import datetime, timedelta, timezone

import gradio as gr
import pytest
from fastapi.testclient import TestClient

V1_FORM_FIELDS = {
    "brief", "state", "tradition", "example", "palette", "symmetry", "max_colours", "min_coverage",
    "max_coverage", "exclusive_ink_layers", "min_gap", "required_loops", "cell_size",
    "translation_axis", "translation_period", "min_clearance", "frame", "min_web", "source",
}
V1_COMMUNITY_FIELDS = {
    "community_model", "community_task_limit", "community_api_key", "community_charge_consent",
    "community_full_run_cost_consent", "community_archive_file", "community_archive_consent",
}
V1_ACTIONS = {"generate", "recheck", "example", "community_estimate", "community_run",
              "community_archive_submit"}
SECRET = "sk-test-NEVER-SHOWN-8c1f0b"


def _text(markup: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", markup)))


def _view(markup: str, name: str) -> str:
    match = re.search(rf'<div data-studio-view="{name}"[^>]*>(.*?)<!-- /{name} -->', markup, re.S)
    assert match, name
    return match.group(1)


@pytest.fixture(scope="module")
def v1():
    from pramaan import app as v1_app

    return v1_app


@pytest.fixture(scope="module")
def initial(v1):
    return v1._initial_custom_state()


@pytest.fixture(scope="module")
def render():
    from pramaan_v2.apps.studio_render import render_studio

    return render_studio


@pytest.fixture(scope="module")
def try_page(render, initial):
    return render(copy.deepcopy(initial))


def _submit_state(initial, **extra):
    state = copy.deepcopy(initial)
    state["active_tab"] = "leaderboard"
    state.update(extra)
    return state


# ------------------------------------------------------------------ Try it


def test_try_it_is_the_default_view(try_page):
    assert re.search(r'<div data-studio-view="try"(?![^>]*hidden)', try_page)
    assert re.search(r'<div data-studio-view="submit"[^>]*hidden', try_page)
    view = _view(try_page, "try")
    assert view.count("<h1") == 1
    text = _text(view)
    assert "TRY IT · THE STUDIO" in text
    assert "Pick a tradition. Watch it get checked." in text
    assert 'aria-current="page"' in view and ">Try it</a>" in view


def test_try_it_has_the_three_numbered_steps_in_order(try_page):
    text = _text(_view(try_page, "try"))
    positions = [text.index(s) for s in ("1 · CHOOSE", "2 · MAKE", "3 · EDIT AND RECHECK")]
    assert positions == sorted(positions)
    for label in ("State or Union Territory", "Tradition", "Worked example for this tradition",
                  "Pattern brief", "Advanced rules", "Pattern source"):
        assert label in text, label


def test_choose_step_selects_the_initial_example(try_page, initial):
    view = _view(try_page, "try")
    form = initial["form"]
    assert re.search(rf'<option value="{re.escape(form["state"])}" selected>', view)
    assert re.search(rf'<option value="{re.escape(form["tradition"])}" selected>', view)
    assert 'Bandhani Dot Field · Beam-generated, verified</option>' in view
    assert 'href="/traditions/GJ"' in view
    assert "Pattern type: dot field." in _text(view)


def test_every_control_has_a_real_label(try_page):
    view = _view(try_page, "try")
    ids = re.findall(r'<(?:select|textarea|input)[^>]*\bid="([^"]+)"', view)
    assert ids
    for control_id in ids:
        assert f'for="{control_id}"' in view, control_id
    assert re.search(r'<label for="st-src"[^>]*>Pattern source</label>', view)
    assert re.search(r'<textarea id="st-src"[^>]*data-field="source"', view)


def test_event_field_names_are_the_ones_v1_reads(render, initial):
    markup = render(copy.deepcopy(initial))
    fields = set(re.findall(r'data-field="([a-z_]+)"', markup))
    assert fields <= V1_FORM_FIELDS | V1_COMMUNITY_FIELDS, fields - V1_FORM_FIELDS - V1_COMMUNITY_FIELDS
    assert {"brief", "state", "tradition", "example", "source"} <= fields
    actions = set(re.findall(r'data-action="([a-z_]+)"', markup))
    assert actions <= V1_ACTIONS, actions - V1_ACTIONS
    assert {"generate", "recheck", "example", "community_estimate", "community_run"} <= actions


def test_make_step_buttons_and_allowance_note(try_page):
    from pramaan.budget import MODEL_CALL_BUDGET

    view = _view(try_page, "try")
    assert re.search(r'data-action="generate"[^>]*>Generate with Beam</button>', view)
    assert re.search(r'data-action="example"[^>]*>Load example</button>', view)
    assert f"{MODEL_CALL_BUDGET.daily_limit} model calls per UTC day" in _text(view)
    assert "one at a time" in _text(view)
    assert re.search(r"<details[^>]*>\s*<summary[^>]*>Advanced rules</summary>", view)
    assert re.search(r'data-action="recheck"[^>]*>Recheck my edit</button>', view)


def test_progress_regions_sit_next_to_the_buttons(try_page):
    view = _view(try_page, "try")
    make = re.search(r'data-action="generate".*?</div>', view, re.S).group(0)
    assert 'role="status"' in make
    recheck = re.search(r'data-action="recheck".*?</div>', view, re.S).group(0)
    assert 'role="status"' in recheck


def test_result_panel_shows_the_verified_example(try_page, initial):
    view = _view(try_page, "try")
    text = _text(view)
    assert re.search(r'aria-pressed="false"[^>]*>One tile</button>', view)
    assert re.search(r'aria-pressed="true"[^>]*>As a 3 × 3 repeat</button>', view)
    assert "GUJARAT · BANDHANI · DOT FIELD" in text
    assert re.search(r'<img[^>]+src="data:image/png;base64,[^"]+"[^>]+alt="[^"]+"', view)
    for tab in ("Checks and proof", "Attempts", "Proposals"):
        assert re.search(rf'role="tab"[^>]*>{tab}</button>', view), tab
    assert "10 of 10" in text and "checks passed" in text
    assert "Verified at 64 × 64 · Beam proposal accepted after 1 try" in text
    assert view.count('aria-label="Passed"') == 10
    assert 'aria-label="Failed"' not in view
    for label in ("Readable pattern", "Within the design size limit", "At most 3 colours", "Symmetric",
                  "Repeat tiles match", "Layers line up", "Ink layers do not overlap",
                  "Motifs keep their distance", "Ink coverage is within range", "Dots keep their distance"):
        assert label in text, label
    for detail in ("64 primitives of 100", "3 used", "left-right and top-bottom",
                   "0.3281; allowed up to 0.70", "every pair clears 3 px"):
        assert detail in text, detail


def test_proof_card_carries_hashes_and_actions(try_page, initial):
    certificate = json.loads(initial["result"]["certificate"])
    view = _view(try_page, "try")
    text = _text(view)
    assert "PROOF YOU CAN RECOMPUTE" in text
    assert certificate["dsl_sha256"] in text and certificate["mask_sha256"] in text
    assert "0.3.0 · exhaustive on this 64 × 64 raster" in text
    assert "Cultural authenticity, physical print quality, continuous vector seams" in text
    copy_button = re.search(r'<button[^>]*data-copy-certificate[^>]*data-certificate-b64="([^"]+)"', view)
    assert copy_button
    import base64

    assert json.loads(base64.b64decode(copy_button.group(1)))["dsl_sha256"] == certificate["dsl_sha256"]
    assert "Download run bundle" in text
    assert "It is not a digital signature" in text and "Runs stay in your session" in text
    assert re.search(r'href="/traditions/GJ"[^>]*>Where this tradition(’|&#x27;|\')s brief comes from', view)


def test_bundle_download_is_a_data_link_when_a_bundle_exists(render, initial, tmp_path):
    bundle = tmp_path / "pramaan-test.zip"
    bundle.write_bytes(b"PK\x05\x06" + b"\0" * 18)
    state = copy.deepcopy(initial)
    state["result"] = {**state["result"], "bundle": str(bundle)}
    view = _view(render(state), "try")
    assert re.search(r'<a[^>]+download="pramaan-run-bundle.zip"[^>]+href="data:application/zip;base64,', view)


def test_bundle_without_a_file_rechecks_then_downloads(try_page):
    view = _view(try_page, "try")
    assert re.search(r'<button[^>]*data-action="recheck"[^>]*data-bundle-request[^>]*>Download run bundle</button>',
                     view)


def test_failed_recheck_names_the_broken_rule(render, initial, v1):
    source = json.loads(initial["form"]["source"])
    layer = source["layers"][0]
    layer["motifs"][0]["center"] = [layer["motifs"][0]["center"][0] + 1, layer["motifs"][0]["center"][1]]
    event = gr.EventData(None, {"action": "recheck", "source": json.dumps(source), "active_tab": "studio"})
    from pramaan_v2.apps.studio import on_event

    markup, state = on_event(event, copy.deepcopy(initial), None, None, None)
    view = _view(markup, "try")
    assert 'aria-label="Failed"' in view
    assert "checks passed" in _text(view)
    assert "PROOF YOU CAN RECOMPUTE" not in _text(view)


# ------------------------------------------------------------------ Submit a run


def test_submit_view_is_chosen_by_the_v1_tab(render, initial):
    markup = render(_submit_state(initial))
    assert re.search(r'<div data-studio-view="submit"(?![^>]*hidden)', markup)
    assert re.search(r'<div data-studio-view="try"[^>]*hidden', markup)
    view = _view(markup, "submit")
    assert view.count("<h1") == 1
    text = _text(view)
    assert "LEADERBOARD / COMMUNITY RUNS / SUBMIT" in text
    assert "Put your model on the board." in text
    assert "Sign in to publish" in text


def test_submit_view_renders_path_a(render, initial):
    view = _view(render(_submit_state(initial)), "submit")
    text = _text(view)
    assert "A · RUN IT HERE, PRIVATELY" in text
    assert "Run one model on the frozen briefs with your own key." in text
    assert re.search(r'<label for="cm-model"[^>]*>LiteLLM model ID', view)
    assert re.search(r'<select id="cm-size"[^>]*data-field="community_task_limit"', view)
    assert '<option value="12" selected>Quick run · 12 briefs</option>' in view
    assert '<option value="72">Full suite · 72 briefs</option>' in view
    assert "Estimate full-suite cost, no provider call" in text
    key = re.search(r'<input id="cm-key"[^>]*>', view).group(0)
    assert 'type="password"' in key and 'autocomplete="off"' in key and "value=" not in key
    assert view.count('data-field="community_charge_consent"') == 1
    assert view.count('data-field="community_full_run_cost_consent"') == 1
    assert re.search(r'data-action="community_run"[^>]*>Run privately</button>', view)
    assert "No community run has been started in this session." in text


def test_submit_view_renders_path_b_with_real_commands(render, initial):
    view = _view(render(_submit_state(initial)), "submit")
    text = _text(view)
    assert "B · UPLOAD A PACKED RUN" in text
    assert ("uv run --project . python benchmarks/run.py \\ --suite benchmarks/suites/eval-v2.1.jsonl \\ "
            "--model your-model-key \\ --output benchmarks/results/my-run.jsonl") in text
    assert ("uv run --project . python -m scripts.pack_submission \\ benchmarks/results/my-run.jsonl "
            "benchmarks/results/my-run.zip") in text
    file_input = re.search(r'<input id="cm-zip"[^>]*>', view).group(0)
    assert 'type="file"' in file_input and 'accept=".zip"' in file_input
    assert 'data-field="community_archive_file"' in file_input
    assert "One ZIP, 20 MB at most, holding exactly manifest.json and results.jsonl." in text
    assert 'data-field="community_archive_consent"' in view
    assert "https://huggingface.co/datasets/kshitijthakkar/pramaan-community-runs" in view
    assert "create_pr=True" in text


def test_submit_commands_match_the_scripts():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    run = (root / "benchmarks" / "run.py").read_text(encoding="utf-8")
    for flag in ("--suite", "--model", "--output"):
        assert f'"{flag}"' in run
    pack = (root / "scripts" / "pack_submission.py").read_text(encoding="utf-8")
    assert 'add_argument("results"' in pack and 'add_argument("archive"' in pack
    assert (root / "benchmarks" / "suites" / "eval-v2.1.jsonl").is_file()


def test_submit_view_explainers(render, initial):
    text = _text(_view(render(_submit_state(initial)), "submit"))
    assert "Your scores are never trusted. Your proposals are replayed." in text
    steps = [text.index(s) for s in ("1 · MANIFEST", "2 · SUITE", "3 · REPLAY", "4 · SCRUB", "5 · PUBLISH")]
    assert steps == sorted(steps)
    for item in ("Verified:", "Not verified:", "Kept apart:", "Licence:"):
        assert item in text
    assert "After a private run" in text and "[YOUR RUN]" in text
    assert "Submission schema pramaan-community-submission-1" in text


def test_submit_lede_counts_the_canonical_models_from_data(render, initial):
    from pramaan_v2.constants import number_word
    from pramaan_v2.site.data import site_data

    count = len(site_data().models)
    text = _text(_view(render(_submit_state(initial)), "submit"))
    assert f"apart from the canonical {number_word(count)}." in text
    assert "canonical ten." not in text or count == 10


# ------------------------------------------------------------------ spec 8.5 states


@pytest.fixture()
def publishing_on(monkeypatch):
    from pramaan_v2.apps import studio_render

    monkeypatch.setattr(studio_render, "archive_publishing_ready", lambda: True)


@pytest.fixture()
def publishing_off(monkeypatch):
    from pramaan_v2.apps import studio_render

    monkeypatch.setattr(studio_render, "archive_publishing_ready", lambda: False)


def test_state_signed_out(render, initial, publishing_on):
    view = _view(render(_submit_state(initial)), "submit")
    assert re.search(r'<a[^>]+href="/login/huggingface\?_target_url=[^"]+"[^>]*>Sign in with Hugging Face to publish</a>',
                     view)
    assert "Sign in with Hugging Face, choose a ZIP and give consent to begin." in _text(view)
    assert 'data-action="community_run"' in view  # path A works signed out


def test_state_signed_in(render, initial, publishing_on):
    view = _view(render(_submit_state(initial, v2_hf_user="ada")), "submit")
    assert re.search(r'data-action="community_archive_submit"[^>]*>Validate and publish</button>', view)
    assert "Sign in with Hugging Face to publish</a>" not in view
    assert "Sign out (ada)" in view


def test_state_publishing_switched_off(render, initial, publishing_off):
    view = _view(render(_submit_state(initial, v2_hf_user="ada")), "submit")
    assert ("Publishing from the Space is switched off right now. Use the pull request route below."
            in _text(view))
    assert re.search(r'data-action="community_archive_submit"[^>]*disabled', view)


def test_state_estimate_shown(render, initial):
    estimate = {"model_key": "openai/gpt-x", "estimated_usd": 3.1415, "cap_usd": 10.0,
                "estimated_cost_exceeds_per_run_cap": False, "rate_card_source": "LiteLLM price list",
                "rate_card_observed_at": "2026-11-01", "mean_input_tokens_per_task": 900,
                "mean_output_tokens_per_task": 2100, "task_count": 72}
    state = _submit_state(initial, community_estimate=estimate,
                          community_status="Full-suite estimate for openai/gpt-x: USD 3.1415 ...")
    text = _text(_view(render(state), "submit"))
    assert "openai/gpt-x" in text
    assert "LiteLLM price list" in text and "2026-11-01" in text
    assert "USD 3.1415" in text and "72 briefs" in text
    assert "USD 10" in text
    assert "No provider request was made." in text


def test_state_daily_limit_reached(render, initial):
    status = ("This account or browser session has already used its one USD 10-capped evaluation for today "
              "(UTC). No provider request was made.")
    text = _text(_view(render(_submit_state(initial, community_status=status)), "submit"))
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
    assert f"00:00 UTC on {tomorrow}" in text


def test_state_run_in_progress(render, initial):
    state = _submit_state(initial, community_progress={"done": 5, "total": 12, "passed": 3, "spend_usd": 0.42})
    text = _text(_view(render(state), "submit"))
    assert "Brief 5 of 12" in text and "3 passed so far" in text and "USD 0.42 spent so far" in text


def test_state_archive_rejected(render, initial, publishing_on):
    status = ("Archive not published: submission archive must contain exactly manifest.json and results.jsonl; "
              "submission result digest does not match its manifest")
    view = _view(render(_submit_state(initial, v2_hf_user="ada", community_archive_status=status)), "submit")
    assert "The archive was rejected. Nothing was published." in _text(view)
    reasons = re.search(r'data-community-archive-status.*?<ul[^>]*>(.*?)</ul>', view, re.S).group(1)
    assert reasons.count("<li") == 2
    assert "results.jsonl" in reasons and "digest does not match" in reasons


def test_state_published(render, initial, publishing_on):
    status = ("Published 72 replay-audited record(s) to the public community dataset "
              "(kshitijthakkar/pramaan-community-runs).")
    view = _view(render(_submit_state(initial, v2_hf_user="ada", community_archive_status=status)), "submit")
    text = _text(view)
    assert "Published 72 replay-audited rows" in text
    assert 'href="https://huggingface.co/datasets/kshitijthakkar/pramaan-community-runs"' in view
    assert 'href="/leaderboard#community"' in view


def test_private_run_summary_fills_the_after_card(render, initial):
    status = ("Community run run-abc: 12/12 requested questions; pass@1 4/12; pass@3 7/12. Tokens 41000; wall time "
              "300.0 s; provider-reported cost USD 0.8120; rate-card estimate USD 0.8000; request latency 900 ms "
              "mean/prompt; verifier latency 40.0 ms mean/prompt. This run is not ranked unless compared on the "
              "same completed question IDs. Results stay in this run and are not added to a dataset.")
    view = _view(render(_submit_state(initial, community_status=status)), "submit")
    text = _text(view)
    assert "[YOUR RUN]" not in text
    assert "7 of 12" in text and "4 of 12" in text and "USD 0.8120" in text
    assert "pass@1" not in text.lower() and "pass@3" not in text.lower()


# ------------------------------------------------------------------ handler and secrets


def test_on_event_recheck_of_a_passing_example(initial):
    from pramaan_v2.apps.studio import on_event

    event = gr.EventData(None, {"action": "recheck", "source": initial["form"]["source"], "active_tab": "studio"})
    markup, state = on_event(event, copy.deepcopy(initial), None, None, None)
    view = _view(markup, "try")
    assert "10 of 10" in _text(view)
    assert view.count('aria-label="Passed"') == 10
    assert 'aria-label="Failed"' not in view
    assert state["result"]["bundle"]
    assert re.search(r'<a[^>]+download="pramaan-run-bundle.zip"', view)
    assert "Verified at 64 × 64" in _text(view)


def test_on_event_state_change_uses_v1_handler(initial):
    from pramaan_v2.apps.studio import on_event

    event = gr.EventData(None, {"action": "state", "state": "Tamil Nadu", "active_tab": "studio"})
    markup, state = on_event(event, copy.deepcopy(initial), None, None, None)
    assert state["form"]["state"] == "Tamil Nadu"
    assert re.search(r'<option value="Tamil Nadu" selected>', _view(markup, "try"))
    assert 'href="/traditions/TN"' in markup


def test_api_key_never_appears_in_html_or_state(initial):
    from pramaan_v2.apps.studio import on_event

    for action in ("community_estimate", "community_run"):
        event = gr.EventData(None, {
            "action": action, "active_tab": "leaderboard", "community_model": "not a model id",
            "community_task_limit": "12", "community_api_key": SECRET, "community_charge_consent": True,
            "community_full_run_cost_consent": False,
        })
        markup, state = on_event(event, copy.deepcopy(initial), None, None, None)
        assert SECRET not in markup
        assert SECRET not in repr(state)
        assert re.search(r'<div data-studio-view="submit"(?![^>]*hidden)', markup)


def test_signed_in_profile_reaches_the_render(initial, publishing_on):
    from pramaan_v2.apps.studio import on_event

    profile = gr.OAuthProfile({"preferred_username": "ada", "name": "Ada", "picture": "",
                               "profile": "https://huggingface.co/ada"})
    event = gr.EventData(None, {"action": "noop", "active_tab": "leaderboard"})
    markup, state = on_event(event, copy.deepcopy(initial), profile, None, None)
    assert state["v2_hf_user"] == "ada"
    assert "Validate and publish" in _view(markup, "submit")


def test_rendered_copy_uses_v2_vocabulary(render, initial):
    for state in (copy.deepcopy(initial), _submit_state(initial)):
        # The raw run record (a collapsed <pre> of JSON) keeps its field names; visitor copy does not.
        text = _text(re.sub(r"<pre.*?</pre>", " ", render(state), flags=re.S))
        for word in ("Pass@1", "pass@3", "candidate", "Arena", "quilt", "witness"):
            assert word.lower() not in text.lower(), word


def test_no_banned_phrases_in_studio_files():
    from pathlib import Path

    from scripts.check_public_tree import BANNED_PHRASES

    root = Path(__file__).resolve().parents[1]
    for rel in ("pramaan_v2/apps/studio.py", "pramaan_v2/apps/studio_render.py",
                "pramaan_v2/static/css/70-studio.css"):
        text = (root / rel).read_text(encoding="utf-8")
        for phrase in BANNED_PHRASES:
            assert not re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.I), (rel, phrase)


# ------------------------------------------------------------------ mount


def test_studio_route_serves_the_app():
    from pramaan_v2.server import app

    client = TestClient(app)
    response = client.get("/studio/")
    assert response.status_code == 200
    config = client.get("/studio/config")
    assert config.status_code == 200
    body = config.text
    assert "PUT YOUR MODEL ON THE BOARD" in body.upper() or "Put your model on the board." in body
    assert "Pick a tradition." in body


def test_studio_build_wires_the_handler():
    from pramaan_v2.apps.studio import build_studio

    demo = build_studio()
    names = [getattr(dep.get("fn") if isinstance(dep, dict) else getattr(dep, "fn", None), "__name__", "")
             for dep in demo.fns.values()] if hasattr(demo, "fns") else []
    assert "on_event" in names
    assert any(type(block).__name__ == "LoginButton" for block in demo.blocks.values())


def test_api_key_field_is_cleared_after_every_action_that_sends_it():
    from pramaan_v2.apps.studio import STUDIO_JS

    sent_with = re.search(r"community_api_key' && !\[([^\]]*)\]\.includes\(action\)\) return;", STUDIO_JS).group(1)
    actions = re.findall(r"'(\w+)'", sent_with)
    assert set(actions) == {"community_run", "community_publish"}
    handler = STUDIO_JS[STUDIO_JS.index("element.addEventListener('click'"):]
    for action in actions:
        branch = re.search(r"if \(action === '%s'\) \{(.*?)\n  \}" % action, handler, re.S)
        assert branch, action
        assert "clearKey();" in branch.group(1) and "return;" in branch.group(1), action
    assert re.search(r"function clearKey\(\) \{[^}]*community_api_key[^}]*value = ''", STUDIO_JS)
