"""Load the canonical published results the v2 site is built from.

The site is built from the public results dataset at a pinned revision, so a
new model reaches the site by publishing it, bumping ``sources.json`` and
rebuilding.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SOURCES = Path(__file__).with_name("sources.json")
SUITE_ID = "eval-v2.1"
COMPACT_FILE = "compact/test.jsonl"
SUMMARY_FILE = "summary/eval-v2.1-summary.json"
# Model keys and brief ids become URL path segments and file names on the site.
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class Published:
    rows: list[dict]
    summary: dict
    dataset: str
    revision: str


def pinned_source() -> dict[str, str]:
    return json.loads(SOURCES.read_text(encoding="utf-8"))


def check_ids(rows: list[dict], where: str = "rows") -> None:
    """Fail unless every model key and brief id matches ``SAFE_ID``."""
    for number, row in enumerate(rows, 1):
        for key in ("model_key", "prompt_id"):
            value = row.get(key)
            if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
                raise ValueError(f"{where}:{number}: {key} {value!r} must match [A-Za-z0-9][A-Za-z0-9._-]*, "
                                 "because it becomes a URL path segment and a file name")


def _read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, row in enumerate(rows, 1):
        if not all(isinstance(row.get(key), str) and row[key] for key in ("model_key", "prompt_id")):
            raise ValueError(f"{path}:{number}: model_key and prompt_id are required")
        if not isinstance(row.get("attempts"), list):
            raise ValueError(f"{path}:{number}: attempts are required")
        if row.get("suite_id") != SUITE_ID:
            raise ValueError(f"{path}:{number}: row is not from suite {SUITE_ID}")
    check_ids(rows, str(path))
    return rows


def load_published(compact: Path | None = None, summary: Path | None = None) -> Published:
    """Read rows and summary from local files, or from the pinned dataset revision."""
    source = pinned_source()
    if compact is None or summary is None:
        from huggingface_hub import hf_hub_download

        kwargs = {"repo_id": source["dataset"], "repo_type": "dataset", "revision": source["revision"]}
        compact = compact or Path(hf_hub_download(filename=COMPACT_FILE, **kwargs))
        summary = summary or Path(hf_hub_download(filename=SUMMARY_FILE, **kwargs))
    rows = _read_rows(Path(compact))
    document = json.loads(Path(summary).read_text(encoding="utf-8"))
    if not isinstance(document.get("by_model"), dict):
        raise ValueError(f"{summary}: summary has no by_model mapping")
    return Published(rows=rows, summary=document, dataset=source["dataset"], revision=source["revision"])
