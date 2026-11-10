"""Shared fixtures for the v2 Space tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ADDED_AFTER_SPEC = "claude-fable-5-2"


@pytest.fixture(scope="session")
def published():
    from pramaan_v2.build.sources import load_published

    return load_published()


@pytest.fixture(scope="session")
def ten_model_rows(published):
    """The ten models the v2 spec's own numbers were computed from."""
    return [row for row in published.rows if row["model_key"] != ADDED_AFTER_SPEC]


@pytest.fixture(scope="session")
def ten_model_summary(published):
    summary = dict(published.summary)
    summary["by_model"] = {key: value for key, value in published.summary["by_model"].items()
                           if key != ADDED_AFTER_SPEC}
    return summary
