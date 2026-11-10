"""Community runs for the leaderboard, from v1's replay-validated loader (read-only reuse).

Rows are marked Community and never mix into canonical ranks. v1's snapshot holds one
trajectory row per brief plus v1's own summary; the site publishes only one aggregate row per
model (``aggregate``), never the trajectories.

A page request never waits on the Hub: ``community_snapshot`` answers at once from the last good
snapshot (or the empty state), and starts a background refresh when that snapshot is older than
``REFRESH_AFTER`` seconds. At most one refresh runs at a time; a failed refresh keeps the last
good snapshot. The server also starts one refresh at startup (``start_refresh``).
"""
from __future__ import annotations

import threading
import time

EMPTY = "No community runs have been merged yet."
REFRESH_AFTER = 120  # seconds
LABEL_MAX = 120

_lock = threading.Lock()
_state: dict = {"snapshot": None, "attempted": None, "thread": None}


def _empty() -> dict:
    return {"status": EMPTY, "rows": [], "provenance": "community"}


# ---------------------------------------------------------------- the per-model aggregate

def _text(*values) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()[:LABEL_MAX]
    return ""


def _count(overall: dict, field: str, n: int, rows: list[dict]) -> int:
    share = overall.get(field)
    if isinstance(share, (int, float)) and not isinstance(share, bool) and "n" in overall:
        return int(round(share * n))
    return sum(bool(row.get(field)) for row in rows)


def aggregate(snapshot) -> dict:
    """One row per community model: key, label, provider, n, within3, first_try, provenance.

    Counts come from the snapshot's ``summary["by_model"][key]["overall"]`` (v1's summarizer) when
    present, else from the model's own trajectory rows. ``n`` is that run's own brief count.
    """
    rows = snapshot.get("rows") if isinstance(snapshot, dict) else None
    if not isinstance(rows, list):
        return _empty()
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("model_key"), str) and row["model_key"]:
            groups.setdefault(row["model_key"], []).append(row)
    if not groups:
        return _empty()
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    documents = summary.get("by_model") if isinstance(summary.get("by_model"), dict) else {}
    out = []
    for key, items in groups.items():
        document = documents.get(key) if isinstance(documents.get(key), dict) else {}
        overall = document.get("overall") if isinstance(document.get("overall"), dict) else {}
        n = overall.get("n")
        n = n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else len(items)
        first = items[0]
        out.append({
            "key": key[:LABEL_MAX],
            "label": _text(first.get("model_label"), first.get("label"), document.get("label"), first.get("model_id"), key),
            "provider": _text(first.get("provider")),
            "n": n, "within3": _count(overall, "validity_at_3", n, items),
            "first_try": _count(overall, "validity_at_1", n, items), "provenance": "community",
        })
    count = len(out)
    status = f"{count} community {'run has' if count == 1 else 'runs have'} been merged."
    return {"status": status, "rows": out, "provenance": "community"}


# ---------------------------------------------------------------- non-blocking refresh

def _load() -> dict:
    """v1's snapshot, bypassing its ``lru_cache`` so a refresh really lists the Hub again."""
    import pramaan.app as v1

    loader = v1._community_comparison_snapshot
    clear = getattr(loader, "cache_clear", None)
    if callable(clear):
        clear()
    return loader()


def _refresh() -> None:
    try:
        snapshot = _load()
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("rows"), list):
            raise ValueError("community snapshot is malformed")
        public = aggregate(snapshot)
        with _lock:
            previous = _state["snapshot"]
            # v1 reports a failed listing as an empty snapshot; never let that erase merged rows.
            if public["rows"] or not (previous and previous["rows"]):
                _state["snapshot"] = public
    except Exception:  # noqa: BLE001 - a failed refresh keeps the last good snapshot
        pass
    finally:
        with _lock:
            _state["thread"] = None


def start_refresh() -> bool:
    """Start a background refresh unless one is running or the last attempt is recent. Never blocks."""
    with _lock:
        thread = _state["thread"]
        if thread is not None and thread.is_alive():
            return False
        attempted = _state["attempted"]
        if attempted is not None and time.monotonic() - attempted < REFRESH_AFTER:
            return False
        _state["attempted"] = time.monotonic()
        thread = threading.Thread(target=_refresh, name="community-refresh", daemon=True)
        _state["thread"] = thread
    thread.start()
    return True


def community_snapshot() -> dict:
    """The last good aggregate, or the empty state; returns at once and may start a refresh."""
    start_refresh()
    with _lock:
        snapshot = _state["snapshot"]
    if snapshot is None:
        return _empty()
    return {**snapshot, "rows": [dict(row) for row in snapshot["rows"]]}


def wait_idle(timeout: float) -> bool:
    """Wait for a running refresh to finish (tests and tools); True when none is running."""
    with _lock:
        thread = _state["thread"]
    if thread is not None:
        thread.join(timeout)
        return not thread.is_alive()
    return True


def _reset() -> None:
    """Forget every snapshot and attempt (tests)."""
    with _lock:
        _state.update(snapshot=None, attempted=None, thread=None)
