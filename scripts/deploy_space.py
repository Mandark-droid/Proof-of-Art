"""Build an allowlisted Space tree, run release gates, deploy one commit, and smoke it."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import json

from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[1]
SPACE_ID = "kshitijthakkar/pramaan-proof-of-art"
SPACE_URL = "https://kshitijthakkar-pramaan-proof-of-art.hf.space"
EMBED_URL = "https://huggingface.co/spaces/kshitijthakkar/pramaan-proof-of-art"
ROOT_FILES = (
    "CITATIONS.md", "Dockerfile", "LICENSE", "README.md", "app.py",
    "client.py", "models.py", "openenv.yaml", "pyproject.toml", "uv.lock",
)
ALLOWED_FILES = (
    "benchmarks/README.md", "benchmarks/TRAJECTORY_DATASET.md", "benchmarks/collect_shards.py",
    "benchmarks/models.json", "benchmarks/prompts.jsonl", "benchmarks/run.py", "benchmarks/summarize.py",
    "benchmarks/results/eval-v2.1-beam-20261030.jsonl", "docs/benchmark-protocol.md",
    "scripts/check_public_tree.py", "scripts/deploy_space.py",
    "scripts/push_hub_datasets.py", "scripts/update_hub_cards.py",
)
ALLOWED_DIRS = (
    "assets", "benchmarks/data", "benchmarks/suites", "examples", "pramaan",
    "pramaan_env", "server",
)
HUB_EMAIL = "kshitijthakkar" + "@users.noreply.huggingface.co"
HUB_AUTHOR = "kshitijthakkar"
RAW_DRAFT_PATHS = ("docs/" + "social_posts.md", "docs/" + "video_script.md", "docs/" + "demo-and-submission.md")


def allowlisted_paths(source: Path, gitattributes: Path | None = None) -> list[Path]:
    paths: set[Path] = set()
    for item in ROOT_FILES + ALLOWED_FILES:
        path = source / item
        if not path.is_file():
            raise FileNotFoundError(f"required allowlisted file is missing: {item}")
        paths.add(Path(item))
    for item in ALLOWED_DIRS:
        directory = source / item
        if not directory.is_dir():
            raise FileNotFoundError(f"required allowlisted directory is missing: {item}")
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"symlinks are not allowed in deploy input: {path.relative_to(source)}")
            if path.is_file():
                if "__pycache__" in path.parts or path.suffix.lower() in {".pyc", ".pyo"}:
                    continue
                paths.add(path.relative_to(source))
    if (source / ".gitattributes").is_file() or (gitattributes and gitattributes.is_file()):
        paths.add(Path(".gitattributes"))
    return sorted(paths, key=lambda path: path.as_posix().casefold())


def build_tree(source: Path, destination: Path, gitattributes: Path | None = None) -> list[Path]:
    source, destination = source.resolve(), destination.resolve()
    gitattributes = gitattributes.resolve() if gitattributes else None
    paths = allowlisted_paths(source, gitattributes)
    destination.mkdir(parents=True, exist_ok=False)
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        input_path = source / relative
        if relative.as_posix() == ".gitattributes" and not input_path.is_file() and gitattributes:
            input_path = gitattributes
        shutil.copy2(input_path, target)
    return paths


def run_scanner(root: Path, *, history: bool = False) -> None:
    command = [sys.executable, "scripts/check_public_tree.py", "--root", str(root)]
    if history:
        command.append("--history")
    subprocess.run(command, cwd=ROOT, check=True)


def validate_images(root: Path, paths: list[Path]) -> None:
    from PIL import Image

    image_suffixes = {".png", ".gif", ".jpg", ".jpeg", ".webp"}
    for relative in paths:
        if relative.suffix.lower() not in image_suffixes:
            continue
        path = root / relative
        with path.open("rb") as image_file:
            head = image_file.read(128)
        if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"allowlisted image is an unresolved Git LFS pointer: {relative}")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            raise RuntimeError(f"allowlisted image is unreadable: {relative}: {error}") from error
    print("PASS allowlisted raster images are readable", flush=True)


def run_release_gates(source: Path) -> None:
    subprocess.run([sys.executable, "scripts/run_release_checks.py"], cwd=source, check=True)


def wait_until_running(api: HfApi, timeout_seconds: int = 900) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        runtime = api.get_space_runtime(repo_id=SPACE_ID)
        stage = getattr(runtime, "stage", None)
        stage_value = getattr(stage, "value", stage)
        print(f"Space runtime: {stage_value}", flush=True)
        if str(stage_value).upper() == "RUNNING":
            return
        if str(stage_value).upper() in {"RUNTIME_ERROR", "BUILD_ERROR"}:
            error_message = getattr(runtime, "error_message", None)
            detail = f"; error: {error_message}" if error_message else ""
            raise RuntimeError(f"Space did not reach RUNNING: {stage_value}{detail}")
        time.sleep(15)
    raise TimeoutError("Space did not reach RUNNING within 15 minutes")


def assert_space_head_identity(expected_commit: str) -> None:
    with tempfile.TemporaryDirectory(prefix="pramaan-space-preflight-") as temp_dir:
        clone = Path(temp_dir) / "space"
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run([
            "git", "clone", "--quiet", "--depth=1", "--single-branch", "--branch", "main",
            "https://huggingface.co/spaces/" + SPACE_ID, str(clone),
        ], check=True, env=env)
        head = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if head != expected_commit:
            raise RuntimeError(f"Space main advanced before the release check: expected {expected_commit}, found {head}")
        identities = subprocess.run(
            ["git", "-C", str(clone), "log", "-1", "--format=%an%x09%ae%x09%cn%x09%ce"],
            check=True, capture_output=True, text=True,
        ).stdout.strip().split("\t")
        if (len(identities) != 4 or identities[0].lower() != HUB_AUTHOR.lower()
                or identities[1].lower() != HUB_EMAIL.lower() or identities[2].lower() != HUB_AUTHOR.lower()
                or identities[3].lower() != HUB_EMAIL.lower()):
            raise RuntimeError("new Space commit author and committer are not the Hub no-reply identity")
    print("PASS Space release commit identity and head check", flush=True)


def check_public_cleanup(api: HfApi) -> None:
    issues: list[str] = []
    with urlopen("https://huggingface.co/api/spaces/" + SPACE_ID + "/refs", timeout=30) as response:
        refs = json.loads(response.read().decode("utf-8"))
    branch_names = sorted(item.get("name") for item in refs.get("branches", []))
    if branch_names != ["main"] or refs.get("tags") or refs.get("converts"):
        raise RuntimeError(f"Space refs are not clean: branches={branch_names}, tags={refs.get('tags')}, converts={refs.get('converts')}")
    commits = api.list_repo_commits(repo_id=SPACE_ID, repo_type="space", revision="main")
    if not commits:
        raise RuntimeError("Space main has no commits")
    for index, commit in enumerate(commits, start=1):
        commit_authors = commit.authors
        if commit_authors != [HUB_AUTHOR]:
            issues.append(f"Space commit {index} has unexpected author metadata: {commit_authors}")
        trailers = [line for line in (commit.message or "").splitlines()
                    if line.lower().startswith("co-authored-by:")]
        if trailers:
            issues.append(f"Space commit {index} has co-author trailers: {trailers}")
    with tempfile.TemporaryDirectory(prefix="pramaan-space-audit-") as temp_dir:
        clone = Path(temp_dir) / "space"
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run([
            "git", "clone", "--quiet", "--depth=100",
            "https://huggingface.co/spaces/" + SPACE_ID, str(clone),
        ], check=True, env=env)
        refs_output = subprocess.run(
            ["git", "-C", str(clone), "ls-remote", "--heads", "origin"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        if len(refs_output) != 1 or not refs_output[0].endswith("refs/heads/main"):
            raise RuntimeError(f"Space Git refs are not main-only: {refs_output}")
        identities = subprocess.run(
            ["git", "-C", str(clone), "log", "-1", "--format=%an%x09%ae%x09%cn%x09%ce"],
            check=True, capture_output=True, text=True,
        ).stdout.strip().split("\t")
        if (len(identities) != 4 or identities[1].lower() != HUB_EMAIL.lower()
                or "@users.noreply.huggingface.co" not in identities[3].lower()):
            raise RuntimeError("latest Space commit author is not the Hub no-reply identity")
        count = subprocess.run(
            ["git", "-C", str(clone), "rev-list", "--all", "--count"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if count != str(len(commits)):
            raise RuntimeError(f"Space API lists {len(commits)} commits but fresh clone contains {count}")
    for path in RAW_DRAFT_PATHS:
        url = f"https://huggingface.co/spaces/{SPACE_ID}/raw/main/{path}"
        try:
            with urlopen(Request(url, method="GET"), timeout=30) as response:
                status = response.status
        except HTTPError as error:
            status = error.code
        if status != 404:
            raise RuntimeError(f"removed raw file still responds HTTP {status}: {path}")
    print(f"PASS Space structure: only main ref, {len(commits)} no-reply commit(s), draft URLs return 404", flush=True)
    if issues:
        raise RuntimeError("Space history cleanup is incomplete: " + "; ".join(issues))


def deploy(source: Path, paths: list[Path], stage: Path) -> None:
    token = get_token()
    if not token:
        raise RuntimeError("Log in to Hugging Face before deploying")
    api = HfApi(token=token)
    parent_commit = api.repo_info(repo_id=SPACE_ID, repo_type="space", revision="main").sha
    wanted = {path.as_posix() for path in paths}
    with tempfile.TemporaryDirectory(prefix="pramaan-space-release-git-") as temp_dir:
        clone = Path(temp_dir) / "space"
        askpass = Path(temp_dir) / "askpass.cmd"
        askpass.write_text("@echo off\r\necho %HF_GIT_PASSWORD%\r\n", encoding="ascii")
        env = os.environ.copy()
        env.update({
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS_REQUIRE": "force",
            "GIT_ASKPASS": str(askpass),
            "HF_GIT_PASSWORD": token,
            "GIT_AUTHOR_NAME": HUB_AUTHOR,
            "GIT_AUTHOR_EMAIL": HUB_EMAIL,
            "GIT_COMMITTER_NAME": HUB_AUTHOR,
            "GIT_COMMITTER_EMAIL": HUB_EMAIL,
        })
        remote_url = "https://" + HUB_AUTHOR + "@huggingface.co/spaces/" + SPACE_ID
        subprocess.run([
            "git", "clone", "--quiet", "--depth=1", "--single-branch", "--branch", "main",
            remote_url, str(clone),
        ], check=True, env=env)
        head = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if head != parent_commit:
            raise RuntimeError(f"Space main changed during allowlist preparation: expected {parent_commit}, found {head}")

        tracked = set(subprocess.run(
            ["git", "-C", str(clone), "ls-files", "-z"],
            check=True, capture_output=True,
        ).stdout.decode("utf-8").split("\0")) - {""}
        removed = sorted(tracked - wanted)
        for offset in range(0, len(removed), 80):
            subprocess.run(["git", "-C", str(clone), "rm", "-f", "--", *removed[offset:offset + 80]], check=True, env=env)

        for relative in paths:
            target = clone / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage / relative, target)
        add_paths = sorted(wanted)
        for offset in range(0, len(add_paths), 80):
            subprocess.run(["git", "-C", str(clone), "add", "--", *add_paths[offset:offset + 80]], check=True, env=env)
        subprocess.run([
            "git", "-C", str(clone), "commit", "--allow-empty",
            "-m", "Publish allowlisted Pramaan release",
        ], check=True, env=env)
        identity = subprocess.run(
            ["git", "-C", str(clone), "log", "-1", "--format=%an%x09%ae%x09%cn%x09%ce"],
            check=True, capture_output=True, text=True,
        ).stdout.strip().split("\t")
        if identity != [HUB_AUTHOR, HUB_EMAIL, HUB_AUTHOR, HUB_EMAIL]:
            raise RuntimeError(f"new Space commit did not use the required Hub identity: {identity}")
        subprocess.run(["git", "-C", str(clone), "-c", "credential.helper=", "push", "origin", "main"],
                       check=True, env=env)
        commit = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    actual = api.repo_info(repo_id=SPACE_ID, repo_type="space", revision="main").sha
    if actual != commit:
        raise RuntimeError(f"Space main moved after deploy: expected {commit}, found {actual}")
    print(f"Space release commit: {commit}", flush=True)
    wait_until_running(api)
    subprocess.run([
        sys.executable, str(source / "scripts" / "smoke_embed.py"),
        "--base-url", SPACE_URL, "--embed-url", EMBED_URL,
    ], cwd=source, check=True)
    assert_space_head_identity(commit)
    check_public_cleanup(api)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--deploy", action="store_true", help="run release checks, deploy one commit, smoke, and verify Space refs")
    parser.add_argument("--output", type=Path, help="copy the allowlisted tree here for inspection")
    parser.add_argument("--gitattributes", type=Path, help="optional LFS rules preserved from the current Space")
    args = parser.parse_args()
    source = args.source.resolve()
    if args.deploy:
        branch = subprocess.run(["git", "-C", str(source), "branch", "--show-current"],
                                check=True, capture_output=True, text=True).stdout.strip()
        if branch != "main":
            raise RuntimeError("Space deploy source must be the integrated local main branch")
        run_release_gates(source)
    if args.output:
        output = args.output.resolve()
        paths = build_tree(source, output, args.gitattributes)
        run_scanner(output)
        print(f"Allowlisted tree ready: {output} ({len(paths)} files)", flush=True)
        return 0
    with tempfile.TemporaryDirectory(prefix="pramaan-space-release-") as temp:
        stage = Path(temp) / "tree"
        paths = build_tree(source, stage, args.gitattributes)
        run_scanner(stage)
        validate_images(stage, paths)
        print(f"Allowlisted tree scanned: {len(paths)} files", flush=True)
        if args.deploy:
            deploy(source, paths, stage)
        else:
            print("No deploy requested; no remote was contacted.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
