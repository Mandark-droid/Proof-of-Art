"""Pin the current published data. When the data revision changes this fails on purpose:
review the new numbers, then regenerate tests_v2/snapshot_eval_v2_1.json."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = json.loads((Path(__file__).with_name("snapshot_eval_v2_1.json")).read_text(encoding="utf-8"))
SITE = ROOT / "pramaan_v2" / "site_data"


def _site(name):
    return json.loads((SITE / name).read_text(encoding="utf-8"))


def test_snapshot_revision_is_current():
    revision = _site("meta.json")["revision"]
    assert revision == SNAPSHOT["revision"], (
        "data revision changed: review the new numbers, then regenerate tests_v2/snapshot_eval_v2_1.json")


def test_totals_match_snapshot():
    assert _site("meta.json")["totals"] == SNAPSHOT["totals"]


@pytest.mark.parametrize("expected", SNAPSHOT["models"], ids=[m["key"] for m in SNAPSHOT["models"]])
def test_model_numbers_match_snapshot(expected):
    models = _site("leaderboard.json")["models"]
    m = next(item for item in models if item["key"] == expected["key"])
    assert [item["key"] for item in models].index(expected["key"]) == SNAPSHOT["models"].index(expected)
    for field in ("within3", "first", "readable", "passed_of_readable", "repaired", "by_grammar", "low_ink",
                  "anchor_only", "truncated", "on_frontier", "complete"):
        assert m[field] == expected[field], field
    assert [round(x, 4) for x in m["wilson95"]] == expected["wilson95"]
    assert round(m["tokens_per_brief"] / 1000, 1) == expected["tokens_k"]
    assert round(m["latency_s"], 1) == expected["latency_s"]
    assert round(m["spend_usd"], 2) == expected["spend_usd"]
    assert {k: round(v, 3) for k, v in m["radar"].items()} == expected["radar"]


def test_failing_first_try_matches_snapshot():
    rows = _site("leaderboard.json")["failing_first_try"]
    assert {r["check"]: r["count"] for r in rows} == SNAPSHOT["failing_first_try"]
