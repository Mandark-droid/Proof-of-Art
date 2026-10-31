"""Resolve project data paths when Pramaan runs from source or an installed wheel."""

from __future__ import annotations

import os
from pathlib import Path


_PROJECT_MARKER = Path("benchmarks") / "suites" / "eval-v2.1.jsonl"


def resolve_project_root(module_file: str | Path, *, cwd: str | Path | None = None) -> Path:
    """Find the checked-out/Space project root before falling back to package paths.

    In the Space, application modules may be imported from ``.venv/site-packages``
    while the allowlisted data tree remains under ``/app``. Prefer the explicit
    runtime override, then the working directory and its ancestors, and only then
    the installed module's ancestors.
    """
    candidates: list[Path] = []
    configured = os.environ.get("PRAMAAN_PROJECT_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    working_root = Path(cwd if cwd is not None else Path.cwd()).expanduser()
    candidates.extend((working_root, *working_root.parents))
    module_path = Path(module_file).expanduser()
    try:
        module_path = module_path.resolve()
    except OSError:
        module_path = module_path.absolute()
    candidates.extend(module_path.parents)

    visited: set[Path] = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            candidate = candidate.absolute()
        if candidate in visited:
            continue
        visited.add(candidate)
        if (candidate / _PROJECT_MARKER).is_file():
            return candidate
    return module_path.parents[1] if len(module_path.parents) > 1 else module_path.parent
