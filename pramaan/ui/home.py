"""Home screen and source-backed model/reference wall rendering."""

from __future__ import annotations

import base64
import html
import re

from .quilt import STATE_GRID
from .check_labels import check_label, check_labels
from .leaderboard import model_display_name


_SAFE_B64 = re.compile(r"[A-Za-z0-9+/]+={0,2}\Z")
_STATUS = {
    "first": ("●", "Right first try"),
    "repair": ("◐", "Right after repair"),
    "fail": ("×", "Not yet right"),
    "none": ("○", "Not run"),
}


def _result(task: dict | None) -> tuple[str, str]:
    if not isinstance(task, dict):
        return "none", "Not run"
    passed_at = task.get("passed_at")
    if type(passed_at) is int and passed_at == 1:
        return "first", "Right first try"
    if type(passed_at) is int and passed_at > 1:
        return "repair", "Right after repair"
    attempts = task.get("attempts")
    if isinstance(attempts, list) and attempts:
        return "fail", "Not yet right"
    return "none", "Not run"


def _image_uri(value: object) -> str:
    if not isinstance(value, str):
        return ""
    if value.startswith("data:image/png;base64,"):
        encoded = value.partition(",")[2]
    else:
        encoded = value
    if len(encoded) > 20_000 or not _SAFE_B64.fullmatch(encoded):
        return ""
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error):
        return ""
    if not decoded.startswith(b"\x89PNG\r\n\x1a\n"):
        return ""
    return "data:image/png;base64," + encoded


def _attempt_uri(task: dict | None) -> str:
    attempts = task.get("attempts") if isinstance(task, dict) else None
    if not isinstance(attempts, list):
        return ""
    passing = [row for row in attempts if isinstance(row, dict) and row.get("passed") is True]
    candidates = passing[-1:] or attempts[-1:]
    return _image_uri(candidates[0].get("tile_png_b64")) if candidates and isinstance(candidates[0], dict) else ""


def _img(uri: str, alt: str, *, cls: str = "") -> str:
    if not uri:
        return f'<span class="pramaan-wall-placeholder" role="img" aria-label="{html.escape(alt, quote=True)}">No image</span>'
    return (f'<img class="{html.escape(cls, quote=True)}" src="{uri}" alt="{html.escape(alt, quote=True)}" '
            'loading="lazy" decoding="async">')


def _task_attempts(model_task: dict | None) -> str:
    attempts = model_task.get("attempts") if isinstance(model_task, dict) else None
    if not isinstance(attempts, list) or not attempts:
        return "<p>No proposal-level evidence was recorded.</p>"
    items = []
    for attempt in attempts[:3]:
        if not isinstance(attempt, dict):
            continue
        outcome = "passed" if attempt.get("passed") is True else "did not pass"
        checks = check_labels(attempt.get("failing_checks", []))
        checks = html.escape(checks)
        witness = attempt.get("witness", [])
        points = []
        if isinstance(witness, list):
            for point in witness:
                if isinstance(point, (list, tuple)) and len(point) >= 2 and all(type(v) is int for v in point[:2]):
                    x, y = point[:2]
                    if 0 <= x < 64 and 0 <= y < 64:
                        points.append(f"({x}, {y})")
        evidence = "; ".join(part for part in [checks and f"failed checks: {checks}", points and "witness pixels: " + ", ".join(points[:8])] if part)
        items.append(f'<li><strong>Attempt {html.escape(str(attempt.get("index", "?")))}: {outcome}</strong>'
                     + (f'<small>{evidence}</small>' if evidence else "") + "</li>")
    return "<ol>" + "".join(items) + "</ol>"


def _state_model_list(model_keys: list[tuple[str, str]], model_tasks: dict[str, dict],
                      easy_id: str, hard_id: str) -> str:
    rows = []
    for key, label in model_keys:
        easy = model_tasks.get(key, {}).get(easy_id) if easy_id else None
        hard = model_tasks.get(key, {}).get(hard_id) if hard_id else None
        outcomes = []
        for level, task in (("Easy", easy), ("Hard", hard)):
            status, result = _result(task)
            image = _attempt_uri(task)
            preview = (_img(image, f"{label} {level.lower()} attempt", cls="pramaan-wall-model-thumb") if image
                       else '<span class="pramaan-wall-model-thumb pramaan-wall-placeholder" aria-hidden="true"></span>')
            outcomes.append(f'<span class="pramaan-wall-model-outcome"><small>{level}</small>{preview}'
                            + f'<b data-outcome="{status}">{_STATUS[status][0]} {html.escape(result)}</b></span>')
        rows.append(f'<li class="pramaan-wall-model-row" data-model-row="{html.escape(key, quote=True)}">'
                    + f'<strong>{html.escape(label)}</strong><div class="pramaan-wall-model-outcomes">'
                    + "".join(outcomes) + '</div></li>')
    return ('<details class="pramaan-wall-model-list"><summary>All model results ('
            + str(len(model_keys)) + ')</summary><ul>' + "".join(rows) + '</ul></details>')


def _task_sheet_section(state: str, level: str, spec: dict | None, reference: dict | None,
                        selected_label: str, selected_task: dict | None,
                        ) -> str:
    title = "Easy task" if level == "easy" else "Hard task"
    brief = spec.get("brief", "") if isinstance(spec, dict) else ""
    if reference:
        reference_html = _img(reference.get("thumbnail_uri", ""), f"Verified procedural reference for {state}, {level}")
        check_rows = reference.get("checks", []) if isinstance(reference.get("checks"), list) else []
        check_text = " · ".join(html.escape(check_label(row.get("id", "check"))) for row in check_rows
                               if isinstance(row, dict) and row.get("status") == "PASS")
        reference_note = f'<small>Procedural reference · all saved checks pass{": " + check_text if check_text else ""}</small>'
    elif isinstance(spec, dict) and spec.get("id") == "py_kolam-as-practised-in-puducherry-easy-35":
        reference_html = '<span class="pramaan-wall-placeholder" role="note">No reference: this brief has no valid solution</span>'
        reference_note = "<small>The frozen brief remains in the evaluation; its unsatisfiability is documented.</small>"
    else:
        reference_html = '<span class="pramaan-wall-placeholder" role="note">No verified reference is published for this task.</span>'
        reference_note = ""
    selected_uri = _attempt_uri(selected_task)
    selected_status, selected_result = _result(selected_task)
    attempts = selected_task.get("attempts") if isinstance(selected_task, dict) else []
    attempts = attempts if isinstance(attempts, list) else []
    unreadable = any(isinstance(attempt, dict)
                     and "candidate_parse" in (attempt.get("failing_checks") or [])
                     for attempt in attempts)
    selected_html = ('<p class="pramaan-wall-unreadable" role="status">Could not be read as a pattern.</p>'
                     if unreadable and not selected_uri else
                     _img(selected_uri, f"{selected_label} model attempt for {state} {level} brief"))
    verification_note = ('<small class="pramaan-wall-verification-note">Passed means the geometric checks passed; it does not mean the design matches the brief.</small>'
                         if selected_status in {"first", "repair"} else "")
    compare = ('<div class="pramaan-wall-pair"><figure><figcaption>Reference</figcaption>'
               + reference_html + reference_note + '</figure><figure>'
               + f'<figcaption>{html.escape(selected_label)} attempt · {html.escape(selected_result)}</figcaption>'
               + verification_note + selected_html + _task_attempts(selected_task) + '</figure></div>')
    return (f'<section class="pramaan-wall-task" data-wall-task-level="{level}">'
            + f'<p class="pramaan-ui-eyebrow">{title}</p><h4>{html.escape(str(spec.get("tradition_id", "")) if isinstance(spec, dict) else "Task brief")}</h4>'
            + f'<p class="pramaan-wall-brief">{html.escape(str(brief))}</p>'
            + compare + '</section>')


def render_reference_wall(references: list[dict], state_rows: list[dict], arm_documents: dict[str, dict],
                         suite_tasks: list[dict], arms: list[dict], *, selected_model: str = "beam-hf",
                         display_mode: str = "reference") -> str:
    """Render the 10 by 9 state map, paired reference/model marks and evidence sheets."""
    reference_by_task = {row["task_id"]: row for row in references
                         if isinstance(row, dict) and isinstance(row.get("task_id"), str)}
    task_by_state: dict[str, dict[str, dict]] = {}
    for task in suite_tasks:
        if isinstance(task, dict) and task.get("state") in STATE_GRID and task.get("level") in {"easy", "hard"}:
            task_by_state.setdefault(task["state"], {})[task["level"]] = task
    state_metadata = {row.get("state"): row for row in state_rows if isinstance(row, dict)
                      and isinstance(row.get("state"), str)}
    model_keys = [(row.get("key"), model_display_name(row.get("key"), row.get("label"))) for row in arms
                  if isinstance(row, dict) and isinstance(row.get("key"), str)
                  and row["key"] in arm_documents]
    if not model_keys:
        model_keys = [(key, model_display_name(key)) for key, document in arm_documents.items()
                      if isinstance(key, str) and isinstance(document.get("tasks"), dict)]
    model_tasks = {key: (arm_documents[key].get("tasks") if isinstance(arm_documents[key].get("tasks"), dict) else {})
                   for key, _label in model_keys}
    if selected_model not in model_tasks and model_keys:
        selected_model = model_keys[0][0]
    labels = dict(model_keys)
    chosen_label = labels.get(selected_model, selected_model)
    display_mode = display_mode if display_mode in {"reference", "attempt"} else "reference"
    tally = {key: {name: 0 for name in ("first", "repair", "fail", "none")} for key, _ in model_keys}
    for key, tasks in model_tasks.items():
        for task in tasks.values():
            status, _label = _result(task)
            tally[key][status] += 1
    chips = []
    for key, label in model_keys:
        counts = tally[key]
        pressed = key == selected_model
        chips.append(
            f'<button class="pramaan-wall-model-chip" type="button" role="radio" '
            f'data-wall-model-choice="{html.escape(key, quote=True)}" aria-checked="{str(pressed).lower()}" '
            f'tabindex="{0 if pressed else -1}">{html.escape(label)}'
            f'<small>{counts["first"]} first · {counts["repair"]} repaired · '
            f'{counts["fail"]} failed · {counts["none"]} not run</small></button>')
    state_tiles = []
    for state in STATE_GRID:
        easy = task_by_state.get(state, {}).get("easy")
        hard = task_by_state.get(state, {}).get("hard")
        easy_id = easy.get("id") if easy else ""
        hard_id = hard.get("id") if hard else ""
        easy_ref = reference_by_task.get(easy_id)
        hard_ref = reference_by_task.get(hard_id)
        # If the easy brief is unsatisfiable, the hard reference fills the geographic tile;
        # the exact easy exception remains visible in the state sheet and note.
        tile_ref = easy_ref or hard_ref
        selected_tasks = model_tasks.get(selected_model, {})
        easy_model = selected_tasks.get(easy_id) if easy_id else None
        hard_model = selected_tasks.get(hard_id) if hard_id else None
        easy_status, easy_label = _result(easy_model)
        hard_status, hard_label = _result(hard_model)
        image_uri = _attempt_uri(easy_model) if display_mode == "attempt" else (
            tile_ref.get("thumbnail_uri", "") if isinstance(tile_ref, dict) else "")
        shown_level = "easy" if easy_ref else "hard" if hard_ref else "none"
        if image_uri:
            art = _img(image_uri, f"{chosen_label} attempt for {state} easy task" if display_mode == "attempt"
                       else f"Procedural reference artwork for {state} ({shown_level} task)", cls="pramaan-wall-art")
        else:
            art = '<span class="pramaan-wall-empty-art" aria-hidden="true"></span>'
        marks = ''.join(
            f'<span class="pramaan-wall-mark" data-wall-mark="{level}" data-status="{status}" '
            f'aria-label="{level.title()} task: {html.escape(label, quote=True)}">'
            f'<span aria-hidden="true">{_STATUS[status][0]}</span></span>'
            for level, status, label in (("easy", easy_status, easy_label), ("hard", hard_status, hard_label)))
        coordinate = state_metadata.get(state, {}).get("grid", list(STATE_GRID[state]))
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2 or any(type(v) is not int for v in coordinate):
            coordinate = list(STATE_GRID[state])
        column, row = coordinate
        if not (0 <= column < 10 and 0 <= row < 9):
            column, row = STATE_GRID[state]
        aria = f"{state}; easy task {easy_label.lower()}; hard task {hard_label.lower()}"
        metadata = state_metadata.get(state, {})
        tradition = metadata.get("tradition", metadata.get("tradition_id", "State tradition"))
        details = (
            _task_sheet_section(state, "easy", easy, easy_ref, chosen_label, easy_model)
            + _task_sheet_section(state, "hard", hard, hard_ref, chosen_label, hard_model)
            + _state_model_list(model_keys, model_tasks, str(easy_id or ""), str(hard_id or ""))
            + f'<p class="pramaan-wall-care"><strong>{html.escape(str(tradition))}</strong>'
            + (f'<br>{html.escape(metadata["caution"])}' if isinstance(metadata.get("caution"), str) and metadata["caution"] else "")
            + '</p>')
        state_tiles.append(
            f'<article class="pramaan-wall-state" data-state="{html.escape(state, quote=True)}" '
            f'style="grid-column:{column + 1};grid-row:{row + 1}">'
            f'<button type="button" class="pramaan-wall-tile" data-home-wall-tile data-state="{html.escape(state, quote=True)}" '
            f'aria-label="{html.escape(aria, quote=True)}" aria-haspopup="dialog">'
            + art + f'<span class="pramaan-wall-state-name">{html.escape(state)}</span>'
            + f'<span class="pramaan-wall-marks">{marks}</span></button>'
            + f'<template data-wall-state-sheet data-state="{html.escape(state, quote=True)}">{details}</template></article>')
    state_model_counts = sum(tally[selected_model].values()) if selected_model in tally else 0
    selected_count_line = (f'<span data-wall-summary>{html.escape(chosen_label)} · {state_model_counts} task records</span>'
                           if selected_model else '<span data-wall-summary>Model results unavailable</span>')
    mode_buttons = ''.join(
        f'<button type="button" class="pramaan-wall-mode" data-wall-mode-choice="{mode}" '
        f'aria-pressed="{str(display_mode == mode).lower()}">{label}</button>'
        for mode, label in (("reference", "Reference"), ("attempt", "Model’s attempt")))
    return (
        '<section class="pramaan-home-wall" id="state-wall" aria-labelledby="state-wall-heading" '
        f'data-wall-display="{display_mode}" data-wall-selected-model="{html.escape(selected_model, quote=True)}">'
        f'<div class="pramaan-home-wall-heading"><div><p class="pramaan-ui-eyebrow">{len(STATE_GRID)} states and territories</p>'
        '<h2 id="state-wall-heading">The Wall</h2>'
        '<p>Each tile shows a verified reference for one state or Union Territory. Its two marks show the selected model’s easy and hard task outcomes.</p>'
        '</div>' + selected_count_line + '</div>'
        '<details class="pramaan-home-reference-note"><summary>How to read this</summary>'
        '<p>References are independently generated constraint witnesses, not model submissions or leaderboard scores. '
        'The marks show verifier-versus-brief outcomes; they do not score visual similarity or cultural authenticity.</p>'
        '<p>The easy Puducherry Kolam brief is unsatisfiable under its stated rules and remains in the frozen evaluation as a documented exception. '
        'No reference is invented for it.</p></details>'
        '<div class="pramaan-wall-controls">'
        '<div class="pramaan-wall-models" role="radiogroup" aria-label="Choose a model for the Wall">' + "".join(chips) + '</div>'
        '<div class="pramaan-wall-mode-toggle" role="group" aria-label="Choose artwork shown in the tiles">' + mode_buttons + '</div>'
        '<input type="hidden" data-field="wall_model" value="' + html.escape(selected_model, quote=True) + '">'
        '<input type="hidden" data-field="wall_mode" value="' + display_mode + '"></div>'
        '<ul class="pramaan-wall-legend" aria-label="Outcome marks">'
        '<li><span data-status="first">●</span>Right first try</li><li><span data-status="repair">◐</span>Right after repair</li>'
        '<li><span data-status="fail">×</span>Not yet right</li><li><span data-status="none">○</span>Not run</li></ul>'
        f'<div class="pramaan-home-wall-grid" role="grid" aria-label="{len(STATE_GRID)} state and Union Territory pattern map">'
        + "".join(state_tiles) + '</div>'
        '<dialog class="pramaan-wall-dialog" data-wall-dialog aria-labelledby="pramaan-wall-dialog-title">'
        '<div class="pramaan-wall-dialog-head"><h3 id="pramaan-wall-dialog-title" data-wall-dialog-title>State evidence</h3>'
        '<button type="button" class="pramaan-ui-button secondary" data-wall-dialog-close>Close</button></div>'
        '<div data-wall-dialog-content></div></dialog></section>')
