"""Run the offline release gates locally without contacting model providers."""

from __future__ import annotations

import os
import subprocess
import sys
import json
import tempfile
from pathlib import Path

from pramaan.tasks import canonical_jsonl_bytes

ROOT = Path(__file__).resolve().parents[1]
CURRENT_SUITE = Path("benchmarks/suites/eval-v2.1.jsonl")
CURRENT_RESULT = Path("benchmarks/results/eval-v2.1-beam-20261030.jsonl")
CURRENT_TRAJECTORY = Path("benchmarks/data/eval-v2.1-trajectories.jsonl")
LEGACY_TRAJECTORY = Path("benchmarks/data/beam-hf-trajectories-v1.jsonl")


def build_checks(root: Path = ROOT) -> list[list[str]]:
    """Choose current eval-v2.1 release gates, preserving legacy replay gates."""
    checks = [
        [sys.executable, "-m", "pytest", "-q", "tests"],
        [sys.executable, "scripts/check_builtin_examples.py"],
        [sys.executable, "scripts/check_app_configuration.py"],
    ]
    current_paths = [root / CURRENT_SUITE, root / CURRENT_RESULT, root / CURRENT_TRAJECTORY]
    present = [path.is_file() for path in current_paths]
    if any(present) and not all(present):
        missing = [path.as_posix() for path, exists in zip(current_paths, present) if not exists]
        raise ValueError("current eval-v2.1 release artifacts are incomplete: " + ", ".join(missing))

    if all(present):
        checks.append([
            sys.executable, "scripts/check_benchmark_results.py", CURRENT_RESULT.as_posix(),
            "--prompts", CURRENT_SUITE.as_posix(), "--model", "beam-hf", "--require-complete",
        ])
    checks.extend([
        [sys.executable, "scripts/check_trajectory_dataset.py", LEGACY_TRAJECTORY.as_posix()],
        [sys.executable, "scripts/check_public_tree.py"],
    ])
    return checks


def check_current_trajectory_export(root: Path = ROOT) -> dict:
    """Regenerate the current trajectory export via B's replay gate and compare bytes."""
    sys.path.insert(0, str(root))
    from scripts.push_hub_datasets import export_suite_trajectories

    results_path = root / CURRENT_RESULT
    suite_path = root / CURRENT_SUITE
    committed_path = root / CURRENT_TRAJECTORY
    with tempfile.TemporaryDirectory(prefix="pramaan-release-export-") as temp_dir:
        regenerated_path = Path(temp_dir) / "eval-v2.1-trajectories.jsonl"
        report = export_suite_trajectories(
            results_path, suite_path, regenerated_path, role="benchmark_evaluation",
        )
        if canonical_jsonl_bytes(regenerated_path.read_bytes()) != canonical_jsonl_bytes(committed_path.read_bytes()):
            raise ValueError("committed eval-v2.1 trajectory export differs from the replayed export")
    return report


def main() -> int:
    if sys.version_info[:2] != (3, 12):
        print("Release checks require Python 3.12 to match the Hugging Face Space runtime.", flush=True)
        return 2
    environment = os.environ.copy()
    environment["HF_TOKEN"] = ""
    try:
        checks = build_checks(ROOT)
    except ValueError as exc:
        print(f"FAIL: {exc}", flush=True)
        return 1
    current_release = (ROOT / CURRENT_SUITE).is_file()
    if not current_release:
        print("NOTE: eval-v2.1 artifacts are not integrated; this is a legacy development check, not final release evidence.", flush=True)
    for command in checks:
        print(f"$ {' '.join(command[1:])}", flush=True)
        result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if result.returncode:
            print(f"FAIL: release check exited with status {result.returncode}", flush=True)
            return result.returncode
    if current_release:
        print("$ replay and compare the eval-v2.1 trajectory export", flush=True)
        try:
            print(json.dumps(check_current_trajectory_export(ROOT), indent=2), flush=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"FAIL: eval-v2.1 trajectory export audit: {exc}", flush=True)
            return 1
    print("PASS: all local release checks completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
