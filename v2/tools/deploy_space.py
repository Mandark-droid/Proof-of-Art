"""Stage, check and deploy the v2 Space (kshitijthakkar/pramaan-proof-of-art-v2).

    python v2/tools/deploy_space.py --output DIR          # stage and scan only
    python v2/tools/deploy_space.py --deploy [--visibility public|private]   # stage, scan, upload, smoke
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPACE_ID = "kshitijthakkar/pramaan-proof-of-art-v2"
SPACE_URL = "https://kshitijthakkar-pramaan-proof-of-art-v2.hf.space"
RENAMES = {"v2/README.md": "README.md", "v2/Dockerfile": "Dockerfile"}
FILES = ("LICENSE", "pyproject.toml", "uv.lock", "benchmarks/data/eval-v2.1-references.jsonl",
         # Read-only reuse of v1 modules the studio and community runs import.
         "benchmarks/run.py", "benchmarks/models.json", "scripts/push_hub_datasets.py",
         "benchmarks/data/eval-v2.1-trajectories.jsonl", "benchmarks/data/eval-v2.1-summary.json",
         "benchmarks/results/eval-v2.1-beam-20261030.jsonl", "benchmarks/data/train-v1-trajectories.jsonl")
DIRS = ("pramaan", "pramaan_env", "pramaan_v2", "benchmarks/suites", "examples")
SKIP = ("__pycache__", ".pyc", ".pytest_cache")
SMOKE = ("/", "/replays", "/replays/hp_kinnauri-shawl-geometric-border-easy-8/beam-hf", "/og/home.png", "/healthz")


def _copy(source: Path, target: Path, out: list[Path], dest: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    out.append(target.relative_to(dest))


def _tracked(paths: tuple[str, ...]) -> list[str]:
    """Files committed at HEAD under ``paths``: untracked or ignored local files never ship."""
    out = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", "HEAD", "--", *paths],
                         check=True, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line and not any(part in line for part in SKIP)]


def uncommitted(paths: tuple[str, ...]) -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", *paths],
                         check=True, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def build_tree(dest: Path) -> list[Path]:
    """Copy exactly the committed files the Space needs into ``dest``; return their relative paths."""
    dest = Path(dest)
    paths: list[Path] = []
    for source, name in RENAMES.items():
        _copy(ROOT / source, dest / name, paths, dest)
    for relative in _tracked(FILES + DIRS):
        _copy(ROOT / relative, dest / relative, paths, dest)
    return sorted(paths)


def scan(dest: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.check_public_tree import audit_tree

    issues = audit_tree(dest)
    if issues:
        raise SystemExit("public scan failed:\n  " + "\n  ".join(issues))


def _get(url: str, token: str | None) -> int:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def ensure_space(api, *, visibility: str | None) -> None:
    """Create the Space if needed (private by default); change visibility only when asked."""
    api.create_repo(SPACE_ID, repo_type="space", space_sdk="docker", private=visibility != "public",
                    exist_ok=True)
    if visibility is not None:
        api.update_repo_settings(SPACE_ID, repo_type="space", private=visibility == "private")


def deploy(dest: Path, *, visibility: str | None) -> None:
    from huggingface_hub import HfApi, get_token

    token = get_token()
    api = HfApi(token=token)
    ensure_space(api, visibility=visibility)
    commit = api.upload_folder(repo_id=SPACE_ID, repo_type="space", folder_path=str(dest),
                               commit_message="Publish the Pramaan v2 site", delete_patterns=["*"])
    print(f"Space commit: {commit.oid}", flush=True)
    for _ in range(90):
        stage = api.get_space_runtime(SPACE_ID).stage
        print(f"Space runtime: {stage}", flush=True)
        if stage == "RUNNING":
            break
        if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"}:
            raise SystemExit(f"Space failed: {stage}")
        time.sleep(20)
    else:
        raise SystemExit("Space did not reach RUNNING")
    failures = [(path, status) for path in SMOKE if (status := _get(SPACE_URL + path, token)) != 200]
    if failures:
        raise SystemExit(f"smoke failed: {failures}")
    print("PASS smoke: " + ", ".join(SMOKE), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--visibility", choices=("public", "private"),
                        help="change the Space's visibility; omitted, a redeploy leaves it as it is")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="pramaan-v2-") as temp:
        dest = args.output or Path(temp) / "tree"
        paths = build_tree(dest)
        scan(dest)
        print(f"Staged and scanned {len(paths)} files in {dest}", flush=True)
        if args.deploy:
            dirty = uncommitted(tuple(RENAMES) + FILES + DIRS)
            if dirty:
                raise SystemExit("commit these first; the Space ships committed files only:\n  "
                                 + "\n  ".join(dirty))
            deploy(dest, visibility=args.visibility)
    return 0


if __name__ == "__main__":
    sys.exit(main())
