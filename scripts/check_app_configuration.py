"""Verify offline app behavior and load the real Beam repair example without inference."""

from __future__ import annotations

import os
import sys
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_space_metadata() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\r?\n(.*?)\r?\n---(?:\r?\n|$)", readme, re.DOTALL)
    if not match:
        raise SystemExit("Space metadata YAML front matter is missing from README.md")
    metadata = yaml.safe_load(match.group(1)) or {}
    sdk = metadata.get("sdk")
    if metadata.get("hf_oauth") is not True:
        raise SystemExit("Space metadata must enable Hugging Face OAuth for consented community submissions")
    if sdk == "gradio":
        running_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        if metadata.get("python_version") != running_python:
            raise SystemExit(f"Space Python version must match the local release-check runtime ({running_python})")

        app_file = metadata.get("app_file")
        if not isinstance(app_file, str) or not app_file:
            raise SystemExit("Space metadata app_file is missing")
        resolved_app = (ROOT / app_file).resolve()
        try:
            resolved_app.relative_to(ROOT)
        except ValueError as exc:
            raise SystemExit("Space app_file must stay inside the repository") from exc
        if not resolved_app.is_file():
            raise SystemExit(f"Space app_file does not exist: {app_file}")

        dev_requirements = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
        pinned_gradio = re.search(r"(?m)^gradio==([^\s;]+)\s*$", dev_requirements)
        if not pinned_gradio or metadata.get("sdk_version") != pinned_gradio.group(1):
            raise SystemExit("Space sdk_version must match the locally tested pinned Gradio version")
    elif sdk == "docker":
        if type(metadata.get("app_port")) is not int or metadata["app_port"] != 8000:
            raise SystemExit("Docker Space app_port must match the container port 8000")
        if metadata.get("base_path") != "/studio":
            raise SystemExit("Docker Space base_path must match the mounted Studio path /studio")
        dockerfile = ROOT / "Dockerfile"
        if not dockerfile.is_file():
            raise SystemExit("Docker Space metadata requires a Dockerfile")
        docker_text = dockerfile.read_text(encoding="utf-8")
        if not re.search(r"(?m)^EXPOSE\s+8000\s*$", docker_text):
            raise SystemExit("Dockerfile must expose the Space app_port 8000")
        if not re.search(r"(?m)^RUN\s+uv\s+sync\s+--locked\s+--no-dev\s*$", docker_text):
            raise SystemExit("Dockerfile must install the pinned production lock with uv sync --locked")
        if not (ROOT / "pramaan_env" / "server" / "app.py").is_file():
            raise SystemExit("Docker Space OpenEnv server entrypoint is missing")
        if 'CMD ["uv", "run", "--no-sync", "--project", ".", "server"]' not in docker_text:
            raise SystemExit("Dockerfile must start the declared server entrypoint")
    else:
        raise SystemExit("Space metadata sdk must select either gradio or docker")

    registry = json.loads((ROOT / "benchmarks" / "models.json").read_text(encoding="utf-8"))
    registry_model_ids = {model["model_id"].split(":", 1)[0] for model in registry["models"]}
    if not registry_model_ids.issubset(set(metadata.get("models", []))):
        raise SystemExit("Space metadata models must include the configured Beam model")


def install_windows_gradio_lock_shim() -> None:
    """Avoid Gradio's blocking socketpair while building the app in Windows sandboxes."""
    if sys.platform != "win32":
        return
    import asyncio
    import gradio.queueing
    import gradio.utils

    # asyncio.Lock binds to a running loop only when first used; Gradio's default
    # helper eagerly creates a Proactor loop and opens a socketpair at import time.
    gradio.queueing.safe_get_lock = asyncio.Lock
    gradio.utils.safe_get_lock = asyncio.Lock
    gradio.utils.safe_get_stop_event = asyncio.Event


def main() -> int:
    # An explicit empty value prevents a developer's ignored .env from enabling live calls.
    os.environ["HF_TOKEN"] = ""
    # Let Gradio register the real Space OAuth routes instead of calling `whoami()`
    # to manufacture a local mock profile during this offline configuration check.
    os.environ.setdefault("SYSTEM", "spaces")
    os.environ.setdefault("SPACE_ID", "pramaan-offline-release-check")
    os.environ.setdefault("OAUTH_CLIENT_ID", "offline-release-check")
    os.environ.setdefault("OAUTH_CLIENT_SECRET", "offline-release-check")
    os.environ.setdefault("OAUTH_SCOPES", "openid profile")
    os.environ.setdefault("OPENID_PROVIDER_URL", "https://huggingface.co")
    check_space_metadata()
    install_windows_gradio_lock_shim()

    import pramaan.app as app_module

    EXAMPLES, build_demo, load_example = app_module.EXAMPLES, app_module.build_demo, app_module.load_example
    # Do not contact the Hub while checking an offline build; use the truthful
    # empty-community state so this gate remains credential-free and deterministic.
    app_module._community_comparison_snapshot = lambda: {
        "status": "Community results unavailable in this offline check.",
        "rows": [], "beam_rows": [], "selected": {}, "summary": {},
    }

    demo = build_demo()
    login_button = next((component for component in demo.config["components"]
                         if component.get("props", {}).get("value") == "Sign in to publish"), None)
    if login_button is None:
        raise SystemExit("Hugging Face sign-in control is missing from the community submission interface")
    buttons = [component for component in demo.config["components"] if component["type"] == "button"]
    generation = next((button for button in buttons
                       if button.get("props", {}).get("value") == "Generate with Beam"), None)
    if generation is None or generation.get("props", {}).get("interactive") is not False:
        raise SystemExit("offline mode must render Generate with Beam as disabled")
    offline_status = any(
        component["type"] == "markdown"
        and "Offline example mode" in component.get("props", {}).get("value", "")
        for component in demo.config["components"]
    )
    if not offline_status:
        raise SystemExit("offline mode status is missing")
    custom_html = next((component for component in demo.config["components"]
                        if component["type"] == "html"
                        and component.get("props", {}).get("elem_id") == "pramaan-custom-app"), None)
    if custom_html is None:
        raise SystemExit("custom HTML Space interface is missing")
    html = custom_html.get("props", {}).get("value", "")
    navigation = {
        "benchmark": "#/overview",
        "leaderboard": "#/leaderboard",
        "studio": "#/studio",
        "rl": "#/build",
    }
    for tab, route in navigation.items():
        if f'data-tab-button="{tab}"' not in html or f'href="{route}"' not in html:
            raise SystemExit(f"custom HTML interface is missing the {tab} screen or {route} route")
    beam_example = next((name for name in EXAMPLES if "Beam-generated" in name), None)
    if beam_example is None:
        raise SystemExit("Beam-generated example is missing from the selector")
    image, gallery, source, certificate, history, status, bundle = load_example(beam_example)
    if (image is None or len(gallery) != 2 or "Proposal 1 · FAIL" not in gallery[0][1]
            or "Proposal 2 · PASS" not in gallery[1][1]
            or '"beam_generated": true' not in history
            or "after 2 proposals" not in status or bundle is None):
        raise SystemExit("Beam example did not load its real failed/repaired trajectory and verified bundle")
    benchmark_examples = [name for name in EXAMPLES if "Beam-generated" in name and "D4 rosette" not in name]
    for name in benchmark_examples:
        image, gallery, source, certificate_text, history, status, bundle = load_example(name)
        certificate = json.loads(certificate_text)
        if (image is None or not gallery or '"beam_generated": true' not in history
                or certificate.get("all_hard_constraints_pass") is not True
                or "Beam " not in status or "accepted after" not in status or bundle is None):
            raise SystemExit(f"benchmark gallery example failed to load: {name}")
    print("PASS offline app: build and generation disabled; Beam repair example, gallery, and bundle load")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
