import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _deploy_module():
    spec = importlib.util.spec_from_file_location("v2_deploy", ROOT / "v2" / "tools" / "deploy_space.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    deploy = _deploy_module()
    dest = tmp_path_factory.mktemp("tree")
    paths = deploy.build_tree(dest)
    return dest, {p.as_posix() for p in paths}


def test_tree_has_the_space_entry_points(staged):
    dest, paths = staged
    for required in ("README.md", "Dockerfile", "LICENSE", "pyproject.toml", "uv.lock",
                     "pramaan_v2/server.py", "pramaan_v2/site_data/meta.json", "pramaan_v2/site_data/wall.png",
                     "pramaan_v2/static/css/00-base.css", "pramaan/dsl.py", "benchmarks/suites/eval-v2.1.jsonl",
                     "benchmarks/data/train-v1-trajectories.jsonl"):
        assert required in paths, required
    assert (dest / "README.md").read_text(encoding="utf-8").startswith("---\n")


def test_tree_leaves_out_everything_else(staged):
    _, paths = staged
    banned = ("tests/", "tests_v2/", "local/", ".pramaan-exports/", "assets/", "v2/", "dev/",
              ".superpowers/", "docs/")
    assert [p for p in paths if p.startswith("benchmarks/results/")] == ["benchmarks/results/eval-v2.1-beam-20261030.jsonl"]
    assert not [p for p in paths if p.startswith(banned)]
    assert [p for p in paths if p.startswith("scripts/")] == ["scripts/push_hub_datasets.py"]
    assert not [p for p in paths if "__pycache__" in p or p.endswith(".pyc")]


def test_space_card_front_matter():
    import yaml

    text = (ROOT / "v2" / "README.md").read_text(encoding="utf-8")
    front = yaml.safe_load(text.split("---")[1])
    assert front["sdk"] == "docker" and front["app_port"] == 8000
    assert "base_path" not in front
    assert len(front["short_description"]) <= 60


def test_dockerfile_runs_the_v2_server():
    text = (ROOT / "v2" / "Dockerfile").read_text(encoding="utf-8")
    assert "pramaan_v2.server" in text and "uv sync --locked --no-dev" in text


def test_staged_tree_passes_the_public_scan(staged):
    import sys

    sys.path.insert(0, str(ROOT))
    from scripts.check_public_tree import audit_tree

    dest, _ = staged
    assert audit_tree(dest) == []


def test_dockerfile_copies_everything_the_tree_stages(staged):
    import re

    _, paths = staged
    text = (ROOT / "v2" / "Dockerfile").read_text(encoding="utf-8")
    copied = set()
    for line in text.splitlines():
        if line.startswith("COPY ") and "--from" not in line:
            copied.update(part.rstrip("/") for part in line.split()[1:-1])
    for path in paths - {"Dockerfile"}:
        top = path.split("/")[0] if "/" in path else path
        assert path in copied or top in copied or any(path.startswith(c + "/") for c in copied), path
    assert "_community_runtime() is not None" in text


def test_tree_ships_only_committed_files(tmp_path):
    deploy = _deploy_module()
    probe = ROOT / "pramaan_v2" / "_untracked_deploy_probe.txt"
    probe.write_text("local only", encoding="utf-8")
    try:
        paths = {p.as_posix() for p in deploy.build_tree(tmp_path / "t")}
    finally:
        probe.unlink()
    assert "pramaan_v2/_untracked_deploy_probe.txt" not in paths


def test_redeploy_never_changes_visibility_unless_asked():
    deploy = _deploy_module()
    calls = []

    class FakeApi:
        def create_repo(self, *args, **kwargs):
            calls.append(("create", kwargs.get("private")))

        def update_repo_settings(self, *args, **kwargs):
            calls.append(("settings", kwargs.get("private")))

    deploy.ensure_space(FakeApi(), visibility=None)
    assert ("settings", True) not in calls and ("settings", False) not in calls
    deploy.ensure_space(FakeApi(), visibility="public")
    assert ("settings", False) in calls
