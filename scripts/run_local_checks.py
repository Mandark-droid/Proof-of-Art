"""Run simple pytest-style test functions using only Python's standard library."""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODULES = [
    "tests.test_schema", "tests.test_symmetry", "tests.test_geometry", "tests.test_budget", "tests.test_beam_client",
    "tests.test_certificates", "tests.test_orchestrator", "tests.test_benchmarks", "tests.test_benchmark_audit",
    "tests.test_public_tree",
]


def main():
    passed = failed = 0
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            try:
                fn()
            except Exception as exc:
                failed += 1
                print(f"FAIL {module_name}.{name}: {type(exc).__name__}: {exc}")
            else:
                passed += 1
                print(f"PASS {module_name}.{name}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
