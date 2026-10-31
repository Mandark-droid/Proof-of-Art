"""Accessible geographic layout helpers for the 36-state Pramaan quilt."""

from __future__ import annotations

import html
import re
import base64

from .check_labels import check_labels


_ABBREVIATED_GRID = {
    "JK": (3, 0), "LA": (4, 0),
    "PB": (3, 1), "HP": (4, 1),
    "CH": (3, 2), "HR": (4, 2), "UK": (5, 2), "SK": (7, 2), "AR": (9, 2),
    "RJ": (2, 3), "DL": (4, 3), "UP": (5, 3), "BR": (6, 3), "AS": (8, 3), "NL": (9, 3),
    "GJ": (1, 4), "MP": (3, 4), "CG": (4, 4), "JH": (6, 4), "WB": (7, 4), "ML": (8, 4), "MN": (9, 4),
    "DD": (1, 5), "MH": (3, 5), "OD": (5, 5), "TR": (8, 5), "MZ": (9, 5),
    "GA": (2, 6), "TS": (4, 6), "AP": (5, 6),
    "KA": (3, 7),
    "LD": (1, 8), "KL": (3, 8), "TN": (4, 8), "PY": (5, 8), "AN": (8, 8),
}

_ABBREVIATION_TO_STATE = {
    "JK": "Jammu and Kashmir", "LA": "Ladakh", "PB": "Punjab", "HP": "Himachal Pradesh",
    "CH": "Chandigarh", "HR": "Haryana", "UK": "Uttarakhand", "SK": "Sikkim",
    "AR": "Arunachal Pradesh", "RJ": "Rajasthan", "DL": "Delhi", "UP": "Uttar Pradesh",
    "BR": "Bihar", "AS": "Assam", "NL": "Nagaland", "GJ": "Gujarat",
    "MP": "Madhya Pradesh", "CG": "Chhattisgarh", "JH": "Jharkhand", "WB": "West Bengal",
    "ML": "Meghalaya", "MN": "Manipur", "DD": "Dadra and Nagar Haveli and Daman and Diu",
    "MH": "Maharashtra", "OD": "Odisha", "TR": "Tripura", "MZ": "Mizoram", "GA": "Goa",
    "TS": "Telangana", "AP": "Andhra Pradesh", "KA": "Karnataka", "LD": "Lakshadweep",
    "KL": "Kerala", "TN": "Tamil Nadu", "PY": "Puducherry", "AN": "Andaman and Nicobar Islands",
}

STATE_GRID: dict[str, tuple[int, int]] = {
    _ABBREVIATION_TO_STATE[code]: cell for code, cell in _ABBREVIATED_GRID.items()
}


def state_grid_cell(state: str) -> tuple[int, int] | None:
    """Return a state's zero-based (column, row) cell in the 10 by 9 quilt."""
    return STATE_GRID.get(state)


def state_grid_style(state: str) -> str:
    """Return safe one-based CSS grid placement, or empty for unknown states."""
    cell = state_grid_cell(state)
    if cell is None:
        return ""
    column, row = cell
    return f"grid-column:{column + 1};grid-row:{row + 1}"


def _tile_uri(attempts: list[dict]) -> str | None:
    accepted = next((attempt for attempt in reversed(attempts)
                     if isinstance(attempt, dict) and attempt.get("passed") is True
                     and isinstance(attempt.get("tile_png_b64"), str)), None)
    if accepted is None:
        return None
    encoded = accepted["tile_png_b64"]
    if len(encoded) > 20_000 or not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", encoded):
        return None
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error):
        return None
    if not decoded.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    return "data:image/png;base64," + encoded


def _task_for_state(tasks: dict, state: str, level: str, task_id: str | None = None) -> tuple[str, dict] | None:
    if task_id is not None:
        task = tasks.get(task_id)
        return (task_id, task) if isinstance(task, dict) and task.get("state") == state and task.get("level") == level else None
    matches = [(task_id, task) for task_id, task in tasks.items()
               if isinstance(task_id, str) and isinstance(task, dict)
               and task.get("state") == state and task.get("level") == level]
    if not matches:
        return None
    return sorted(matches, key=lambda item: item[0])[0]


def _attempt_history(task: dict) -> str:
    attempts = task.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return "<p class=\"pramaan-muted\">No proposal-level evidence is available.</p>"
    rows = []
    for attempt in attempts[:3]:
        if not isinstance(attempt, dict):
            continue
        index = attempt.get("index")
        if type(index) is not int or not 1 <= index <= 3:
            continue
        outcome = "passed" if attempt.get("passed") is True else "did not pass"
        checks_text = check_labels(attempt.get("failing_checks"), limit=12)
        witness = attempt.get("witness")
        witness_text = html.escape(str(witness[:4])) if isinstance(witness, list) and witness else "none recorded"
        uri = _tile_uri([attempt])
        image = f'<img class="pramaan-quilt-attempt" src="{uri}" loading="lazy" decoding="async" alt="Attempt {index} preview, {outcome}">' if uri else ""
        rows.append(f'<li>{image}<strong>Attempt {index}: {outcome}</strong>'
                    f'<small>Failed checks: {html.escape(checks_text or "none")}; witness: {witness_text}</small></li>')
    return "<ol class=\"pramaan-quilt-attempts\">" + "".join(rows) + "</ol>"


def render_data_quilt(index: dict, arm_documents: dict[str, dict], *, model_key: str,
                      level: str = "easy", mode: str = "single", compare_model: str | None = None) -> str:
    """Render a model-aware state quilt from replay-audited gallery artifacts."""
    states = index.get("states") if isinstance(index.get("states"), list) else []
    arms = index.get("arms") if isinstance(index.get("arms"), list) else []
    keys = [arm.get("key") for arm in arms if isinstance(arm, dict)
            and isinstance(arm.get("key"), str) and arm["key"] in arm_documents]
    if not keys:
        return '<p class="pramaan-muted" role="status">Replay-audited model tiles are not available in this build.</p>'
    selected_key = model_key if model_key in keys else keys[0]
    secondary_key = compare_model if compare_model in keys and compare_model != selected_key else next(
        (key for key in keys if key != selected_key), None)
    safe_level = level if level in {"easy", "hard"} else "easy"
    safe_mode = mode if mode in {"single", "compare", "hardness"} else "single"
    state_rows = {row.get("state"): row for row in states if isinstance(row, dict)
                  and isinstance(row.get("state"), str)}
    if not state_rows:
        state_rows = {name: {"state": name, "grid": list(cell)} for name, cell in STATE_GRID.items()}
    model_tasks = {
        key: (arm_documents[key].get("tasks") if isinstance(arm_documents[key].get("tasks"), dict) else {})
        for key in keys
    }

    tiles = []
    for state_name, fallback_cell in STATE_GRID.items():
        state_row = state_rows.get(state_name, {})
        cell = state_row.get("grid", list(fallback_cell))
        if (not isinstance(cell, (list, tuple)) or len(cell) != 2
                or any(type(value) is not int for value in cell)
                or not (0 <= cell[0] < 10 and 0 <= cell[1] < 9)):
            cell = list(fallback_cell)
        column, row = cell
        placement = f"grid-column:{column + 1};grid-row:{row + 1}"
        safe_state = html.escape(state_name, quote=True)
        main_pair = _task_for_state(model_tasks[selected_key], state_name, safe_level)
        shared_task_id = main_pair[0] if main_pair else None
        state_tasks = {
            key: (_task_for_state(model_tasks[key], state_name, safe_level, shared_task_id)
                  if shared_task_id is not None else None)
            for key in keys
        }
        state_tasks[selected_key] = main_pair
        secondary_pair = state_tasks.get(secondary_key) if secondary_key else None
        main_task = main_pair[1] if main_pair else None
        secondary_task = secondary_pair[1] if secondary_pair else None
        attempted_pairs = [pair for pair in state_tasks.values() if pair
                           and isinstance(pair[1].get("attempts"), list) and pair[1].get("attempts")]
        passed_first = sum(type(pair[1].get("passed_at")) is int and pair[1].get("passed_at") == 1
                           for pair in attempted_pairs)

        if safe_mode == "hardness":
            content = f'<span class="pramaan-quilt-hardness-count">{passed_first}/{len(attempted_pairs)}</span><small>models passed on first proposal</small>'
            aria = f"{safe_state}: {passed_first} of {len(attempted_pairs)} models with attempts passed on the first proposal"
        elif safe_mode == "compare":
            def compare_tile(key, task):
                if task is None:
                    return f'<div class="pramaan-quilt-compare-cell"><b>{html.escape(key)}</b><small>no matching shared task</small></div>'
                passed_at = task.get("passed_at")
                outcome = f"passed at {passed_at}" if type(passed_at) is int else "did not pass"
                uri = _tile_uri(task.get("attempts", [])) if isinstance(task.get("attempts"), list) else None
                image = f'<img src="{uri}" loading="lazy" decoding="async" alt="{html.escape(key)} {html.escape(outcome)}">' if uri else ""
                return f'<div class="pramaan-quilt-compare-cell"><b>{html.escape(key)}</b>{image}<small>{outcome}</small></div>'
            content = '<div class="pramaan-quilt-compare">' + compare_tile(selected_key, main_task)
            if secondary_key:
                content += compare_tile(secondary_key, secondary_task)
            else:
                content += '<div class="pramaan-quilt-compare-cell"><small>A second complete model is not available.</small></div>'
            content += '</div>'
            aria = f"{safe_state}: model comparison"
        else:
            task = main_task
            passed_at = task.get("passed_at") if task else None
            outcome = f"passed on attempt {passed_at}" if type(passed_at) is int else "no accepted result"
            attempts = task.get("attempts", []) if task and isinstance(task.get("attempts"), list) else []
            uri = _tile_uri(attempts)
            image = f'<img src="{uri}" loading="lazy" decoding="async" alt="Replay-verified {safe_level} design for {safe_state}, {outcome}">' if uri else ""
            content = image + f'<small>{html.escape(selected_key)} · {outcome}</small>'
            aria = f"{safe_state}, {selected_key}, {outcome}"

        if main_task and main_task.get("source") == "community":
            evidence_state = "community-run"
        elif main_task and type(main_task.get("passed_at")) is int:
            evidence_state = "verified-first" if main_task.get("passed_at") == 1 else "verified-repaired"
        elif main_task and isinstance(main_task.get("attempts"), list) and main_task["attempts"]:
            evidence_state = "failed-all"
        elif secondary_task and type(secondary_task.get("passed_at")) is int:
            evidence_state = "accepted-other-model"
        else:
            evidence_state = "not-evaluated"
        brief_task = main_task or secondary_task
        brief = brief_task.get("brief", "") if brief_task else ""
        caution = state_row.get("caution", "")
        sources = state_row.get("sources", []) if isinstance(state_row.get("sources"), list) else []
        source_links = []
        for source in sources[:8]:
            if not isinstance(source, dict):
                continue
            title, url = source.get("title"), source.get("url")
            if isinstance(title, str) and isinstance(url, str) and url.startswith("https://"):
                source_links.append(f'<li><a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(title)}</a></li>')
        details = (f'<details class="pramaan-quilt-details"><summary>{html.escape(str(state_row.get("tradition", state_row.get("tradition_id", "Tradition details"))))}</summary>'
                   + (f'<p><strong>Brief:</strong> {html.escape(brief)}</p>' if brief else '<p>No task brief is available for this level.</p>')
                   + (f'<p><strong>Cultural care:</strong> {html.escape(caution)}</p>' if isinstance(caution, str) and caution else "")
                   + ("<ul>" + "".join(source_links) + "</ul>" if source_links else "")
                   + (_attempt_history(brief_task) if brief_task else "")
                   + '<p><a href="#proof-replay">Watch the featured proof replay</a></p></details>')
        tab_index = "0" if not tiles else "-1"
        tiles.append(
            f'<article class="pramaan-state-tile pramaan-data-quilt-tile" role="gridcell" tabindex="{tab_index}" '
            f'data-state-tile data-state="{safe_state}" data-tile-state="{evidence_state}" '
            'data-quilt-state-tile aria-haspopup="dialog" '
            f'data-testid="quilt-state-{html.escape(state_name.lower().replace(" ", "-"), quote=True)}" '
            f'data-grid-col="{column}" data-grid-row="{row}" '
            f'style="{placement}" aria-label="{html.escape(aria, quote=True)}">'
            f'<strong>{safe_state}</strong>{content}{details}</article>'
        )
    return ('<div class="pramaan-state-mosaic-scroll" data-testid="quilt-scroll-region" tabindex="0" aria-label="Geographic quilt">'
            '<div class="pramaan-state-mosaic pramaan-data-quilt" data-testid="quilt-grid" role="grid" aria-label="36 state and Union Territory model quilt" style="min-width:0;width:100%;max-width:960px">'
            + "".join(tiles) + '</div></div>'
            + '<ul class="pramaan-quilt-legend" aria-label="Quilt evidence legend">'
            + '<li data-tile-state="verified-first"><span aria-hidden="true"></span>Verified on first proposal</li>'
            + '<li data-tile-state="verified-repaired"><span aria-hidden="true"></span>Verified after repair</li>'
            + '<li data-tile-state="failed-all"><span aria-hidden="true"></span>Failed all proposals</li>'
            + '<li data-tile-state="accepted-other-model"><span aria-hidden="true"></span>Accepted by another model</li>'
            + '<li data-tile-state="not-evaluated"><span aria-hidden="true"></span>Not evaluated</li>'
            + '<li data-tile-state="community-run"><span aria-hidden="true"></span>Community run</li></ul>'
            + '<dialog class="pramaan-quilt-dialog" data-testid="quilt-state-sheet" aria-labelledby="quilt-sheet-title">'
            + '<div class="pramaan-quilt-dialog-head"><h3 id="quilt-sheet-title" data-quilt-sheet-title>State evidence</h3>'
            + '<button type="button" class="pramaan-ui-button secondary" data-quilt-close aria-label="Close state evidence">Close</button></div>'
            + '<div data-quilt-sheet-content></div></dialog>')
