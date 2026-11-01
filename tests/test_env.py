import json
from pathlib import Path

from pramaan.dsl import Constraints
from pramaan.orchestrate import generate_and_verify
from pramaan.prompts import SYSTEM_PROMPT, design_prompt
from pramaan.tasks import Task, load_suite
from pramaan_env import PramaanAction
from pramaan_env.server.environment import PramaanEnvironment

ROOT = Path(__file__).parents[1]


def test_environment_defaults_to_corrected_eval_suite():
    env = PramaanEnvironment()
    obs = env.reset(seed=0)
    assert obs.metadata["suite_id"] == "eval-v2.1"
    assert env.state.prompt_suite_sha256 == load_suite("eval-v2.1")[1]


def _v1_g1_task():
    tasks, digest = load_suite("eval-v2.1")
    source = next(task for task in tasks if task.grammar == "periodic_raster")
    return source, digest


def _candidate(path="examples/ajrakh_starfield.pattern.json"):
    return (ROOT / path).read_text(encoding="utf-8")


def test_reset_by_task_id_and_seed_is_deterministic():
    tasks, _ = load_suite("eval-v2.1")
    first = PramaanEnvironment()
    second = PramaanEnvironment()
    first_obs = first.reset(task_id=tasks[0].id)
    assert first_obs.task_id == tasks[0].id
    assert first.state.task_id == tasks[0].id
    assert first.state.prompt_suite_sha256 == load_suite("eval-v2.1")[1]
    assert first.state.step_count == 0
    assert first.reset(seed=23).task_id == second.reset(seed=23).task_id


def test_passing_action_ends_episode_with_certificate():
    task, _ = _v1_g1_task()
    env = PramaanEnvironment()
    env.reset(task_id=task.id)
    obs = env.step(PramaanAction(pattern_json=_candidate()))
    assert obs.reward == 1.0
    assert obs.done is True
    assert obs.certificate is not None
    assert obs.certificate["schema_version"] == "0.1"
    assert obs.all_hard_constraints_pass is True
    assert env.state.accepted is True
    assert env.state.step_count == 1


def test_three_failures_end_episode_done_true():
    task, _ = _v1_g1_task()
    env = PramaanEnvironment()
    env.reset(task_id=task.id)
    for attempt in range(1, 4):
        obs = env.step(PramaanAction(pattern_json="not json"))
        assert obs.attempt == attempt
        assert obs.reward == 0.0
    assert obs.done is True


def test_garbage_action_is_zero_reward_not_crash():
    env = PramaanEnvironment()
    tasks, _ = load_suite("eval-v2.1")
    g6_task = next(task for task in tasks if task.grammar == "kolam_loop")
    env.reset(task_id=g6_task.id, suite="eval-v2.1")
    obs = env.step(PramaanAction(pattern_json="x" * 20_000))
    assert obs.reward == 0.0
    assert obs.done is False
    assert obs.attempt == 1
    assert obs.parse_error


def test_v01_action_is_rejected_for_v2_task():
    tasks, _ = load_suite("eval-v2.1")
    task = next(task for task in tasks if task.grammar == "kolam_loop")
    env = PramaanEnvironment()
    env.reset(task_id=task.id)
    obs = env.step(PramaanAction(pattern_json=_candidate()))
    assert obs.reward == 0.0
    assert obs.parse_error
    assert obs.attempt == 1
    assert obs.done is False


def test_env_and_orchestrator_agree():
    task, _ = _v1_g1_task()
    raw = _candidate()

    class FixedClient:
        def complete(self, system, prompt):
            return raw, {"finish_reason": "stop"}

    direct = generate_and_verify(task.brief, task.palette, symmetry=task.symmetry, max_attempts=1, client=FixedClient())
    env = PramaanEnvironment()
    env.reset(task_id=task.id)
    observation = env.step(PramaanAction(pattern_json=raw))
    assert observation.checks == direct["attempts"][0]["report"]["checks"]
    assert observation.certificate["mask_sha256"] == direct["certificate"]["mask_sha256"]


def test_prompt_messages_match_benchmark_prompts():
    task, _ = _v1_g1_task()
    env = PramaanEnvironment()
    obs = env.reset(task_id=task.id)
    profile = Constraints(symmetry=task.symmetry)
    prompt = design_prompt(task.brief, task.palette, {**profile.model_dump(mode="json"), "exclusive_ink_layers": False})
    assert obs.prompt_messages == [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def test_validation_replay_fixture_rewards():
    fixture = json.loads((ROOT / "validation" / "replay.json").read_text(encoding="utf-8"))
    for case in fixture["episodes"]:
        env = PramaanEnvironment()
        env.reset(**case["reset"])
        obs = env.step(PramaanAction(**case["action"]))
        assert {"reward": obs.reward, "done": obs.done} == case["expected"]
def test_openenv_routes_survive_gradio_mount():
    source = (ROOT / "pramaan_env" / "server" / "app.py").read_text(encoding="utf-8")
    web_ui = (ROOT / "pramaan_env" / "server" / "web_ui.py").read_text(encoding="utf-8")
    assert "gradio_builder=build_pramaan_web" in source
    assert 'path="/studio"' in source
    assert 'custom_tab_name="Pramaan RL"' in source
    assert "server.reset_episode" in web_ui and "server.run_step" in web_ui


def test_web_branding_uses_studio_tokens_and_public_identity():
    from pramaan_env.server.web_ui import WEB_TITLE, _javascript, _markup

    markup = _markup()
    script = _javascript("{}")

    assert WEB_TITLE == "Pramaan: AI drew it. Math checked it. · RL environment"
    assert "AI drew it. Math checked it." in markup
    assert "Proof of art that you can verify." in markup
    assert 'src="/favicon.ico"' in markup
    assert "https://huggingface.co/collections/kshitijthakkar/pramaan-proof-of-art" in markup
    assert "--pramaan-paper:#f7f3ec" in markup
    assert "--pramaan-paper:#151b22" in markup
    assert "prefers-color-scheme:dark" in markup
    assert "server.render_pattern" in script
    assert "Raw observation JSON" in markup
    assert "**" not in markup
    assert ".poa-head a,.poa-links a{color:var(--pramaan-ink)!important;background:var(--pramaan-card)" in markup
    assert ".poa-code pre,.poa-code code" in markup
    assert "color:var(--pramaan-on-accent)!important" in markup
    assert "body,.gradio-container{width:100%;min-width:0;max-width:100vw!important" in markup
    audit = (ROOT / "scripts" / "ui_contrast_audit.py").read_text(encoding="utf-8")
    assert '"web": "/web/"' in audit
    assert 'page.wait_for_selector("#pramaan-rl"' in audit


def test_space_mount_redirects_are_path_only_and_preserve_queries():
    source = (ROOT / "pramaan_env" / "server" / "app.py").read_text(encoding="utf-8")
    assert '"/": "/studio/"' in source
    assert '"/studio": "/studio/"' in source
    assert '"/web": "/web/"' in source
    assert "RedirectResponse(target, status_code=307)" in source
    assert 'query.decode("latin-1")' in source


def test_uvicorn_trusts_space_proxy_headers():
    source = (ROOT / "pramaan_env" / "server" / "app.py").read_text(encoding="utf-8")
    assert "proxy_headers=True" in source
    assert 'forwarded_allow_ips="*"' in source


def test_studio_mount_receives_custom_theme_and_css():
    source = (ROOT / "pramaan_env" / "server" / "app.py").read_text(encoding="utf-8")
    assert "theme=create_brand_theme()" in source
    assert "css=PRAMAAN_CSS" in source
    assert 'head=\'<link rel="icon" href="/favicon.ico" type="image/png">\'' in source


def test_space_favicon_route_serves_original_mark():
    source = (ROOT / "pramaan_env" / "server" / "app.py").read_text(encoding="utf-8")
    assert '@app.get("/favicon.ico"' in source
    assert 'media_type="image/png"' in source
    assert '"pramaan-mark.png"' in source


def test_space_frontmatter_targets_docker_mount():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    frontmatter = text.split("---", 2)[1]

    assert "sdk: docker" in frontmatter
    assert "app_port: 8000" in frontmatter
    assert any(line.strip() in {"base_path: /studio", "base_path: /studio/"}
               for line in frontmatter.splitlines())
    description = next(line.split(":", 1)[1].strip() for line in frontmatter.splitlines()
                       if line.startswith("short_description:"))
    assert len(description) <= 60


def test_env_works_without_hf_token(monkeypatch):
    from pathlib import Path

    from pramaan.tasks import load_suite
    from pramaan_env import PramaanAction
    from pramaan_env.server.environment import PramaanEnvironment

    monkeypatch.delenv("HF_TOKEN", raising=False)
    tasks, _ = load_suite("eval-v2.1")
    task = next(task for task in tasks if task.grammar == "periodic_raster")
    env = PramaanEnvironment()
    env.reset(task_id=task.id)
    observation = env.step(PramaanAction(pattern_json=Path("examples/ajrakh_starfield.pattern.json").read_text(encoding="utf-8")))
    assert observation.reward == 1.0
    assert observation.done is True


def test_non_g1_tasks_use_task_aware_design_and_repair_prompts():
    from pramaan.prompts import design_prompt_for_task, repair_prompt_for_task, system_prompt_for_task

    tasks, _ = load_suite("eval-v2.1")
    task = next(item for item in tasks if item.grammar == "dot_field")
    env = PramaanEnvironment()
    observation = env.reset(task_id=task.id)
    assert observation.prompt_messages == [
        {"role": "system", "content": system_prompt_for_task(task)},
        {"role": "user", "content": design_prompt_for_task(task)},
    ]

    raw = "not json"
    failed = env.step(PramaanAction(pattern_json=raw))
    assert failed.parse_error
    assert env._prompt_messages == [
        {"role": "system", "content": system_prompt_for_task(task)},
        {"role": "user", "content": repair_prompt_for_task(task, raw, failed.checks)},
    ]


def test_v02_environment_action_uses_host_profile_and_parser():
    tasks, _ = load_suite("eval-v2.1")
    task = next(item for item in tasks if item.grammar == "dot_field")
    env = PramaanEnvironment()
    env.reset(task_id=task.id)
    candidate = {
        "schema_version": "0.2",
        "grammar": "periodic_raster",
        "tradition_id": "attacker-selected",
        "title": "One centered dot",
        "tile": {"width": 64, "height": 64, "wrap_x": True, "wrap_y": True},
        "palette": ["#000000"],
        "background_index": 8,
        "layers": [{
            "id": "dots", "colour_index": 1, "anchor": [0, 0],
            "motifs": [{"kind": "dot", "center": [31, 31], "radius": 1}],
        }],
        "constraints": {"symmetry": "None", "min_clearance": 0},
    }
    observation = env.step(PramaanAction(pattern_json=json.dumps(candidate)))
    assert observation.parse_error is None
    assert any(check["id"] == "dot_clearance" for check in observation.checks)
    assert observation.reward > 0.0


def test_environment_import_does_not_build_gradio_demo():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pramaan_env.server.environment; "
            "assert 'pramaan.app' not in sys.modules; "
            "assert 'pramaan_env.server.app' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90 if sys.platform == "win32" else 30,
    )
    assert result.returncode == 0, result.stderr


def test_docker_image_copies_community_runner_and_publisher_helpers():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY benchmarks ./benchmarks" in dockerfile
    assert "COPY scripts/push_hub_datasets.py ./scripts/push_hub_datasets.py" in dockerfile

def test_docker_runtime_adds_project_root_for_dynamic_platform_helpers():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "PYTHONPATH=/app" in dockerfile
    assert (
        'RUN uv run --no-sync --project . python -c "from pramaan import app; '
        "assert app._community_runtime() is not None; "
        'assert app._community_model_choices()"'
    ) in dockerfile


def test_pramaan_app_imports_from_only_the_files_copied_by_dockerfile(tmp_path):
    import os
    import shlex
    import shutil
    import subprocess
    import sys

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    image_root = tmp_path / "docker-context"
    image_root.mkdir()
    for line in dockerfile.splitlines():
        if not line.lstrip().startswith("COPY "):
            continue
        tokens = shlex.split(line)
        if any(token.startswith("--from=") for token in tokens[1:-1]):
            continue
        sources, destination = tokens[1:-1], tokens[-1]
        for source in sources:
            source_path = ROOT / source
            assert source_path.exists(), f"Dockerfile COPY source is missing: {source}"
            if source_path.is_dir():
                target = image_root / destination
                shutil.copytree(source_path, target, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                target = image_root / (Path(source).name if destination in {".", "./"} or len(sources) > 1 else destination)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(image_root)
    env["PRAMAAN_PROJECT_ROOT"] = str(image_root)
    for name in ("HF_TOKEN", "HF_COMMUNITY_WRITE_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-c", "from pramaan import app; assert app is not None"],
        cwd=image_root, env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr
