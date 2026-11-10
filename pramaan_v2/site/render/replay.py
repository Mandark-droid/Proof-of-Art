"""One run: one model on one brief, try by try (spec 8.3, ``Replay.dc.html``)."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from ...constants import number_word
from ..data import SiteData
from ..labels import check_label, check_row_label, short_rule
from ..shell import FONT_MONO, esc, footer, page
from .replays import brief_label, failing_checks, is_truncated, models_in_order, pair_text, run_url, tile_img, xy

FEATURED = Path(__file__).resolve().parents[1] / "featured.json"
MONO = f"font-family: {FONT_MONO}"
TILE = 64
PIXELS = TILE * TILE

TICK = ('<svg width="18" height="18" viewBox="0 0 18 18" role="img" aria-label="Passed"><path d="M4 9.5l3.2 3.2L14 5.8" '
        'stroke="#F4B63F" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"></path></svg>')
CROSS = ('<svg width="18" height="18" viewBox="0 0 18 18" role="img" aria-label="Failed"><path d="M5 5l8 8M13 5l-8 8" '
         'stroke="#E8613C" stroke-width="2" stroke-linecap="round" fill="none"></path></svg>')
PILL = ("display: inline-flex; align-items: center; min-height: 44px; box-sizing: border-box; padding: 0 18px; "
        "border-radius: 22px; border: 1px solid #4A3F38; color: #F4EFE6; text-decoration: none; font-size: 15px; "
        "font-weight: 600")
H3 = (f'style="margin: 0 0 8px; {MONO}; font-size: 12px; font-weight: 500; letter-spacing: 0.12em; '
      'color: #948980"')
PRE = (f"margin: 0; padding: 16px 18px; overflow-x: auto; background: #1D1815; border: 1px solid #2E2621; "
       f"border-radius: 6px; {MONO}; font-size: 13px; line-height: 1.55; color: #F4EFE6")
TRANSFORM_WORDS = {"F_vertical": "a mirror image", "F_horizontal": "a mirror image", "R180": "the tile turned 180°",
                   "R90": "the tile turned 90°", "R270": "the tile turned 270°"}
# Checks whose message carries a measured value worth showing on a passing row; the rest show their id.
MEASURED = {"complexity_bound", "kolam_loop_count", "kolam_mirror_symmetry", "jaali_open_fraction", "jaali_min_web",
            "jaali_holes_interior", "jaali_single_solid"}
LOWER_FIRST = {"a", "an", "the", "two", "three", "four", "five", "six", "one"}


@lru_cache(maxsize=1)
def featured() -> dict[str, str]:
    return json.loads(FEATURED.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- words

def reward_text(value: float | None) -> str:
    if value is None:
        return "0.0"
    text = f"{value:.3f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def reward_reason(attempt: dict) -> str:
    if attempt["outcome"] == "unreadable":
        return "cut off at the token limit" if is_truncated(attempt) else "the proposal could not be read"
    held, total = attempt.get("rules_held") or (0, 0)
    if held == total:
        return f"all {total} rules hold"
    reason = f"{held} of {total} rules hold"
    if total and abs(0.2 + 0.6 * held / total - attempt["reward"]) < 1e-6:
        reason += f": 0.2 + 0.6 × {held}/{total}"
    return reason


def verdict(model: dict) -> tuple[str, str, str]:
    """(mono tag, sentence, lower-case phrase for the title)."""
    status, tries = model["status"], model["tries"]
    if status == "1":
        return "PASSED ON THE FIRST TRY", "Passed on the first try.", "passed on the first try"
    if status in ("2", "3"):
        return f"PASSED ON TRY {status}", f"Passed on try {status} of 3.", f"passed on try {status}"
    if status == "-":
        return "NEVER READABLE", "Never produced a readable pattern.", "never produced a readable pattern"
    broke = sum(1 for a in tries if a["outcome"] == "broke")
    unreadable = sum(1 for a in tries if a["outcome"] == "unreadable")
    if not unreadable:
        return ("BROKE A RULE ON EVERY TRY", f"Broke a rule on all {len(tries)} tries.",
                "broke a rule on every try")
    return ("NEVER PASSED",
            f"Never passed: broke a rule on {number_word(broke)} {'try' if broke == 1 else 'tries'}, "
            f"{number_word(unreadable)} unreadable.", "never passed")


def try_title(model: dict, n: int) -> str:
    if n == 1:
        return "Try 1 · first proposal"
    before = [a for a in model["tries"] if a["n"] < n]
    c = sum(1 for a in before if a["outcome"] == "broke")
    u = sum(1 for a in before if a["outcome"] == "unreadable")
    parts = []
    if c:
        parts.append(f"{number_word(c)} counterexample{'s' if c != 1 else ''}")
    if u:
        parts.append(f"{number_word(u)} unreadable proposal{'s' if u != 1 else ''}")
    return f"Try {n} · after {' and '.join(parts)}"


def short_result(attempt: dict) -> tuple[str, str]:
    """('0.725 · broke symmetry', colour)."""
    reward = reward_text(attempt["reward"])
    if attempt["outcome"] == "passed":
        return f"{reward} · passed", "#F4B63F"
    if attempt["outcome"] == "unreadable":
        return f"{reward} · {'cut off' if is_truncated(attempt) else 'unreadable'}", "#B8ADA1"
    fails = failing_checks(attempt)
    what = short_rule(fails[0]["id"]) if len(fails) == 1 else f"{number_word(len(fails))} rules"
    return f"{reward} · broke {what}", "#E8613C"


def brief_sentence(replay: dict) -> str:
    text = replay["brief"].strip()
    first = text.split(" ", 1)[0]
    if first.lower() in LOWER_FIRST:
        text = text[0].lower() + text[1:]
    symmetry = replay["symmetry"]
    rule = {"D2": "Must hold under left-right and top-bottom flips.", "C4": "Must hold under quarter turns."}.get(
        symmetry, f"Must hold under {symmetry} symmetry.")
    return f"{text} {rule}"


# ---------------------------------------------------------------- checks

def _secondary(check: dict) -> str:
    cid, message = check.get("id", ""), str(check.get("message", ""))
    if cid.startswith("symmetry_"):
        return "colour grid matches the requested symmetry"
    if cid == "coverage_bound":
        match = re.match(r"Ink coverage ([0-9.]+); allowed \[([0-9.]+), ([0-9.]+)\]", message)
        if match:
            value, low, high = (float(v) for v in match.groups())
            return f"{value:.4f} of the tile; allowed {low:.2f} to {high:.2f}"
        return cid
    if cid == "palette_limit":
        return re.sub(r";\s*limit\s+\d+\s*$", "", message)
    return message if cid in MEASURED and "\n" not in message else cid


def _evidence(check: dict) -> str:
    message = str(check.get("message", "")).split("\n", 1)[0]
    example = check.get("counterexample") or {}
    if example.get("original") is not None and example.get("transformed") is not None:
        return f'{message}: {xy(example["original"])} ≠ {xy(example["transformed"])}'
    pixel = example.get("pixel")
    if pixel is not None and str(pixel[0]) not in message:
        return f"{message}: {xy(pixel)}"
    return message


def _check_rows(attempt: dict) -> str:
    checks = attempt.get("checks") or []
    rows = []
    for i, check in enumerate(checks):
        last = "; border-bottom: 1px solid #2E2621" if i == len(checks) - 1 else ""
        label = esc(check_row_label(check))
        if check.get("status") == "PASS":
            rows.append(
                '<li style="display: grid; grid-template-columns: 22px minmax(0, 1fr); gap: 10px; padding: 9px 0; '
                f'border-top: 1px solid #2E2621{last}">{TICK}<span>{label} '
                f'<span style="color: #948980">· {esc(_secondary(check))}</span></span></li>')
        else:
            rows.append(
                '<li style="display: grid; grid-template-columns: 22px minmax(0, 1fr); gap: 10px; '
                'padding: 11px 10px 11px 0; border-top: 1px solid #E8613C; border-bottom: 1px solid #E8613C; '
                f'background: #1D1815">{CROSS}<span><strong>{label}</strong> <span style="color: #948980">· '
                f'{esc(check.get("id", ""))}</span><br><span style="{MONO}; font-size: 13px; color: #4FD1C5; '
                f'overflow-wrap: anywhere">{esc(_evidence(check))}</span></span></li>')
    count = number_word(len(checks)).upper()
    noun = "CHECK" if len(checks) == 1 else "CHECKS"
    return (f'<div><h3 {H3}>{count} {noun}</h3><ul style="margin: 0; padding: 0; list-style: none; display: flex; '
            f'flex-direction: column; font-size: 15px; line-height: 1.35">{"".join(rows)}</ul></div>')


def _value(value) -> str:
    if isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return xy(value)
    if isinstance(value, list) and value and all(isinstance(v, list) for v in value):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def told_text(attempt: dict) -> str:
    blocks = []
    for check in failing_checks(attempt):
        lines = [f'"id": {_value(check.get("id"))}', f'"status": {_value(check.get("status"))}',
                 f'"assurance": {_value(check.get("assurance"))}', f'"message": {_value(check.get("message"))}']
        example = check.get("counterexample")
        if example:
            lines.append('"counterexample":')
            if example.get("original") is not None and example.get("transformed") is not None:
                left = [f'"original": {xy(example["original"])}', f'"transformed": {xy(example["transformed"])}']
                width = max(len(v) for v in left) + 4
                colours = [example.get("original_colour_id"), example.get("transformed_colour_id")]
                for text, colour in zip(left, colours):
                    lines.append("    " + (text.ljust(width) + f"colour {colour}" if colour is not None else text))
            else:
                lines += [f'    "{key}": {_value(value)}' for key, value in example.items()]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _told_or_proof(model: dict, attempt: dict) -> str:
    if attempt["outcome"] == "passed":
        exact = all(c.get("assurance") == "EXACT_DISCRETE" for c in attempt.get("checks") or [])
        approximate = [check_label(c["id"]) for c in attempt.get("checks") or []
                       if c.get("assurance") not in (None, "EXACT_DISCRETE")]
        assurance = "Exhaustive on this 64 × 64 raster" if exact else (
            "Exhaustive on this 64 × 64 raster, except " + ", ".join(approximate).lower() + ", which is approximate")
        dt = 'style="color: #948980"'
        return (
            f'<div><h3 {H3}>PROOF YOU CAN RECOMPUTE</h3>'
            '<dl class="rv-proof" style="margin: 0; padding: 16px 18px; display: grid; grid-template-columns: 130px '
            'minmax(0, 1fr); gap: 8px 16px; background: #1D1815; border: 1px solid #F4B63F; border-radius: 6px; '
            'font-size: 14px">'
            f'<dt {dt}>Pixels SHA-256</dt><dd style="margin: 0; {MONO}; overflow-wrap: anywhere">'
            f'{esc(attempt.get("pixels_sha256", ""))}</dd>'
            f'<dt {dt}>Assurance</dt><dd style="margin: 0">{esc(assurance)}</dd>'
            f'<dt {dt}>Not covered</dt><dd style="margin: 0">Cultural authenticity, print quality, continuous vector '
            'seams</dd>'
            f'<dt {dt}>Signature</dt><dd style="margin: 0">None. Anyone can recompute it; nobody has to trust it.</dd>'
            "</dl></div>")
    last = attempt["n"] >= max(a["n"] for a in model["tries"])
    heading = "WHAT THE CHECKER RETURNED" if last else "WHAT THE MODEL WAS TOLD"
    return (f'<div><h3 {H3}>{heading}</h3><pre class="rv-told" style="{PRE}; white-space: pre-wrap; '
            f'overflow-wrap: anywhere">{esc(told_text(attempt))}</pre></div>')


# ---------------------------------------------------------------- viewer

def _pct(v: int) -> str:
    return f"{(v + 0.5) / TILE * 100:.2f}%"


def _ring(x: int, y: int) -> str:
    return (f'<div style="position: absolute; left: {_pct(x)}; top: {_pct(y)}; width: 30px; height: 30px; '
            'margin: -15px 0 0 -15px; box-sizing: border-box; border: 2px solid #4FD1C5; border-radius: 50%"></div>')


def _overlay(replay: dict, attempt: dict) -> str:
    if attempt["outcome"] != "broke":
        return ""
    pair = attempt.get("pair") or {}
    parts: list[str] = []
    points = [p for p in (pair.get("original"), pair.get("transformed"), pair.get("pixel")) if p is not None]
    if attempt.get("mask_png") or points:
        parts.append('<div style="position: absolute; inset: 0; background: rgba(21, 17, 15, 0.7)"></div>')
    if attempt.get("mask_png"):
        parts.append(tile_img(attempt["mask_png"], "", "position: absolute; inset: 0; width: 100%; height: 100%",
                              hidden=True))
    parts += [_ring(p[0], p[1]) for p in points]
    transform = pair.get("transform")
    if transform and replay["symmetry"].startswith("D"):
        if transform != "F_horizontal":
            parts.append('<div style="position: absolute; left: 50%; top: 0; bottom: 0; width: 0; '
                         'border-left: 1px dashed rgba(244, 239, 230, 0.55)"></div>')
        if transform != "F_vertical":
            parts.append('<div style="position: absolute; top: 50%; left: 0; right: 0; height: 0; '
                         'border-top: 1px dashed rgba(244, 239, 230, 0.55)"></div>')
    if attempt.get("mismatch_px") is not None:
        caption = (f'{attempt["mismatch_px"]:,} of {PIXELS:,} pixels disagree with '
                   f'{TRANSFORM_WORDS.get(transform, "the required symmetry")}')
    else:
        caption = (attempt.get("measured") or "").split("\n", 1)[0]
        if pair.get("pixel") is not None and str(pair["pixel"][0]) not in caption:
            caption += f" at {xy(pair['pixel'])}"
    parts.append(f'<div style="position: absolute; left: 12px; bottom: 12px; max-width: calc(100% - 24px); '
                 f'box-sizing: border-box; padding: 6px 10px; background: #15110F; {MONO}; font-size: 13px; '
                 f'color: #4FD1C5">{esc(caption)}</div>')
    return f'<div class="rv-ov" data-rv-overlay style="position: absolute; inset: 0">{"".join(parts)}</div>'


def _chip(on: bool) -> str:
    colours = ("border: 1px solid #F4EFE6; background: #F4EFE6; color: #15110F" if on
               else "border: 1px solid #4A3F38; background: transparent; color: #F4EFE6")
    return (f"min-height: 44px; padding: 0 16px; border-radius: 22px; {colours}; font-family: inherit; "
            "font-size: 15px; font-weight: 600; cursor: pointer")


def _viewer(replay: dict, model: dict, attempt: dict, task_id: str) -> str:
    stage_box = "position: relative; width: 100%; max-width: 576px; aspect-ratio: 1 / 1; box-sizing: border-box"
    if attempt["outcome"] == "unreadable":
        message = "\n".join(str(failing_checks(attempt)[0].get("message", "")).splitlines()[:3]) \
            if failing_checks(attempt) else ""
        what = "CUT OFF AT THE TOKEN LIMIT" if is_truncated(attempt) else "UNREADABLE"
        top = ""
        stage = (
            f'<div class="rv-stage rv-none" style="{stage_box}; display: flex; flex-direction: column; '
            'justify-content: center; gap: 14px; padding: 32px; border: 1px dashed #6B625B">'
            f'<p style="margin: 0; {MONO}; font-size: 12px; letter-spacing: 0.1em; color: #948980">{what} · NO TILE</p>'
            '<p style="margin: 0; font-size: 18px; color: #B8ADA1; text-wrap: pretty">This proposal could not be read, '
            "so there is nothing to draw. The checker's parse message:</p>"
            f'<pre style="margin: 0; {MONO}; font-size: 13px; line-height: 1.5; color: #F4EFE6; white-space: pre-wrap; '
            f'overflow-wrap: anywhere">{esc(message)}</pre></div>')
    else:
        overlay = _overlay(replay, attempt)
        alt = f'The tile {model["label"]} drew on try {attempt["n"]}'
        top = (
            '<div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px">'
            '<div role="group" aria-label="View" class="rv-toggle" style="display: flex; gap: 8px">'
            f'<button type="button" data-rv-view="eye" aria-pressed="false" style="{_chip(False)}">What the eye sees</button>'
            f'<button type="button" data-rv-view="checker" aria-pressed="true" style="{_chip(True)}">'
            "What the checker sees</button></div>"
            f'<span style="{MONO}; font-size: 12px; color: #948980">64 × 64 PIXELS · SHOWN AT 9×</span></div>')
        stage = (
            f'<div class="rv-stage" style="{stage_box}; outline: 1px solid #4A3F38">'
            + tile_img(attempt["tile_png"], alt, "width: 100%; height: 100%") + overlay + "</div>")
        if overlay:
            stage += (
                '<figure class="rv-eye-copy" style="display: none; margin: 0; max-width: 576px">'
                + tile_img(attempt["tile_png"], alt + ", without the checker's marks", "width: 100%; height: auto; "
                           "outline: 1px solid #4A3F38")
                + f'<figcaption style="margin-top: 8px; {MONO}; font-size: 12px; color: #948980">WHAT THE EYE SEES'
                "</figcaption></figure>")
    return (
        '<div class="rv-viewer" style="flex: 1 1 520px; min-width: 0; display: flex; flex-direction: column; gap: 16px">'
        f"{top}{stage}{_picker(model, attempt, task_id)}</div>")


def _picker(model: dict, current: dict, task_id: str) -> str:
    items = []
    for attempt in model["tries"]:
        n = attempt["n"]
        on = n == current["n"]
        if attempt.get("tile_png"):
            thumb = tile_img(attempt["tile_png"], "", "flex: 0 0 56px; width: 56px; height: 56px", hidden=True)
        else:
            thumb = ('<span aria-hidden="true" style="flex: 0 0 56px; width: 56px; height: 56px; box-sizing: border-box; '
                     'border: 1px dashed #6B625B"></span>')
        text, colour = short_result(attempt)
        current_attr = ' aria-current="true"' if on else ""
        items.append(
            f'<a href="{run_url(task_id, model["key"])}?try={n}"{current_attr} style="display: flex; align-items: center; '
            f'gap: 12px; min-height: 76px; box-sizing: border-box; padding: 8px; border: 1px solid '
            f'{"#F4EFE6" if on else "#2E2621"}; border-radius: 6px; background: #1D1815; color: #F4EFE6; '
            f'text-decoration: none; text-align: left">{thumb}'
            f'<span style="display: flex; flex-direction: column; font-size: 14px; line-height: 1.3"><strong>Try {n}</strong>'
            f'<span style="{MONO}; font-size: 12px; color: {colour}">{esc(text)}</span></span></a>')
    return ('<nav aria-label="Choose a try" class="rv-picker" style="display: grid; grid-template-columns: '
            'repeat(auto-fill, minmax(min(170px, 100%), 1fr)); gap: 12px; max-width: 576px">' + "".join(items) + "</nav>")


def _meter(reward: float) -> str:
    pct = f"{max(0.0, min(1.0, reward or 0.0)) * 100:g}%"
    return (
        '<div aria-hidden="true" style="position: relative; margin-top: 16px; height: 12px; display: flex; gap: 2px">'
        '<span style="flex: 0 0 20%; background: #2E2621"></span>'
        '<span style="flex: 0 0 calc(60% - 4px); background: #4A3F38"></span>'
        '<span style="flex: 1 1 0; background: #F4B63F"></span>'
        f'<span style="position: absolute; top: -5px; left: {pct}; width: 2px; height: 22px; margin-left: -1px; '
        'background: #F4EFE6"></span></div>'
        f'<div style="margin-top: 8px; display: flex; justify-content: space-between; gap: 8px; {MONO}; font-size: 11px; '
        'color: #948980"><span>0 UNREADABLE</span><span>0.2 TO 0.8 PARTIAL</span><span>1.0 ALL PASS</span></div>')


def _detail(model: dict, attempt: dict) -> str:
    tokens = f'{attempt["tokens"]:,} tokens' if attempt.get("tokens") is not None else "tokens not recorded"
    latency = f' · {attempt["latency_ms"] / 1000:.2f} s' if attempt.get("latency_ms") is not None else ""
    passed = attempt["outcome"] == "passed"
    return (
        '<div class="rv-detail" data-rv-detail style="flex: 1 1 480px; min-width: 0; display: flex; '
        'flex-direction: column; gap: 24px"><div>'
        '<div style="display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px 16px">'
        f'<h2 style="margin: 0; font-size: 30px; line-height: 1.1; font-weight: 700; font-stretch: 112%">'
        f'{esc(try_title(model, attempt["n"]))}</h2>'
        f'<span style="{MONO}; font-size: 13px; color: #948980">{tokens}{latency}</span></div>'
        '<div style="margin-top: 16px; display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 14px">'
        f'<span style="font-size: 54px; line-height: 1; font-weight: 700; font-stretch: 112%; '
        f'color: {"#F4B63F" if passed else "#F4EFE6"}">{reward_text(attempt["reward"])}</span>'
        f'<span style="font-size: 15px; color: #B8ADA1">reward · {esc(reward_reason(attempt))}</span></div>'
        f'{_meter(attempt["reward"])}</div>'
        f"{_check_rows(attempt)}{_told_or_proof(model, attempt)}</div>")


# ---------------------------------------------------------------- the rest of the page

def _strip(data: SiteData, replay: dict, models: list[dict], current_key: str, task_id: str) -> str:
    items = []
    for model in models:
        here = model["key"] == current_key
        status = model["status"]
        passed = next((a for a in model["tries"] if a["outcome"] == "passed"), None)
        shown = passed or next((a for a in reversed(model["tries"]) if a.get("tile_png")), None)
        ring = "; outline: 2px solid #F4B63F; outline-offset: 3px" if here else ""
        if shown is None:
            picture = (f'<span aria-hidden="true" style="width: 100%; aspect-ratio: 1 / 1; box-sizing: border-box; '
                       f'border: 1px dashed #6B625B{ring}"></span>')
        else:
            dim = "" if passed else "; opacity: 0.45"
            picture = tile_img(shown["tile_png"], "", f"width: 100%; height: auto; aspect-ratio: 1 / 1{dim}{ring}",
                               hidden=True)
        if passed:
            note, colour = f'try {passed["n"]}', "#F4B63F"
        elif status == "-":
            note, colour = "never readable", "#B8ADA1"
        else:
            note, colour = "never passed", "#E8613C"
        if here:
            note += " · this run"
        name = f"<strong>{esc(model['label'])}</strong>" if here else esc(model["label"])
        current = ' aria-current="page"' if here else ""
        items.append(
            f'<a href="{run_url(task_id, model["key"])}"{current} style="display: flex; flex-direction: column; gap: 8px; '
            f'text-decoration: none; color: #F4EFE6; font-size: 13px; line-height: 1.25">{picture}'
            f'<span>{name}<br><span style="color: {colour}">{note}</span></span></a>')
    return (
        '<section style="padding: 88px 0 0">'
        '<div style="display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; '
        'gap: 8px 24px; margin-bottom: 20px">'
        '<h2 style="margin: 0; font-size: 30px; line-height: 1.1; font-weight: 700; font-stretch: 112%">'
        f'Same brief, {number_word(len(models))} models</h2>'
        f'<a href="/replays/{esc(task_id)}" style="display: inline-flex; align-items: center; min-height: 44px; '
        'font-weight: 600">Compare every try side by side →</a></div>'
        '<div class="rv-strip" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(96px, 100%), '
        f'1fr)); gap: 14px">{"".join(items)}</div></section>')


def _reward_comment(model: dict) -> str:
    groups: list[tuple[str, list[int]]] = []
    for attempt in model["tries"]:
        value = reward_text(attempt["reward"])
        if groups and groups[-1][0] == value:
            groups[-1][1].append(attempt["n"])
        else:
            groups.append((value, [attempt["n"]]))
    parts = []
    for value, ns in groups:
        if len(ns) == 1:
            parts.append(f"{value} for try {ns[0]}")
        else:
            parts.append(f"{value} for tries {', '.join(map(str, ns[:-1]))} and {ns[-1]}")
    return ", ".join(parts)


def _recompute(model: dict, task_id: str, suite: str) -> str:
    code = (
        "import json\n"
        "from pramaan_env import PramaanAction\n"
        "from pramaan_env.server.environment import PramaanEnvironment\n\n"
        "env = PramaanEnvironment()\n"
        f'env.reset(task_id="{task_id}", suite="{suite}")\n'
        "result = env.step(PramaanAction(pattern_json=json.dumps(candidate)))\n"
        f"print(result.reward)   # {_reward_comment(model)}")
    return (
        '<section class="rv-recompute" style="padding: 88px 0 0; display: flex; flex-wrap: wrap; gap: 24px 56px; '
        'align-items: flex-start"><div style="flex: 1 1 320px; min-width: 0">'
        '<h2 style="margin: 0 0 12px; font-size: 30px; line-height: 1.1; font-weight: 700; font-stretch: 112%">'
        "Recompute this run</h2>"
        '<p style="margin: 0; color: #B8ADA1; text-wrap: pretty">Every saved proposal is a small JSON pattern. Feed it '
        "to the environment and you get the same checks, the same counterexample and the same reward, on any machine, "
        "without a model key.</p>"
        f'<a href="/playground?task_id={esc(task_id)}" style="display: inline-flex; align-items: center; '
        'min-height: 44px; margin-top: 8px; font-weight: 600">Open this brief in the playground →</a></div>'
        f'<pre style="flex: 2 1 560px; min-width: 0; margin: 0; padding: 18px 20px; overflow-x: auto; '
        f'background: #1D1815; border: 1px solid #2E2621; border-radius: 6px; {MONO}; font-size: 13px; '
        f'line-height: 1.6; color: #F4EFE6">{esc(code)}</pre></section>')


def render_run(data: SiteData, task_id: str, model_key: str, try_n: int | None, base_url: str) -> str | None:
    replay = data.replay(task_id)
    if replay is None:
        return None
    models = models_in_order(data, replay)
    model = next((m for m in models if m["key"] == model_key), None)
    if model is None or not model["tries"]:
        return None
    numbers = [a["n"] for a in model["tries"]]
    n = try_n if try_n in numbers else numbers[0]
    attempt = next(a for a in model["tries"] if a["n"] == n)
    path = run_url(task_id, model_key)
    og_path = "/og/replay/" + path.removeprefix("/replays/") + ".png"
    base = base_url.rstrip("/")
    tag, sentence, phrase = verdict(model)
    passed = model["status"] in ("1", "2", "3")
    headline = featured().get(f"{task_id}/{model_key}") or f'{model["label"]} · {replay["tradition"]}'
    dataset = data.meta.get("dataset", "")
    dataset_url = f"https://huggingface.co/datasets/{dataset}/viewer/compact/test?q={task_id}"
    crumb = " / ".join([replay["state"], replay["tradition"]]).upper() + f' · {replay["level"].upper()}'
    tag_colour = "#F4B63F" if passed else ("#E8613C" if model["status"] == "x" else "#B8ADA1")
    tag_border = "#F4B63F" if passed else ("#E8613C" if model["status"] == "x" else "#6B625B")
    fact = 'style="padding: 7px 12px; border: 1px solid #2E2621; color: #B8ADA1"'
    cost = model.get("est_cost_usd")
    facts = [
        f'<span style="padding: 7px 12px; border: 1px solid {tag_border}; color: {tag_colour}">{tag}</span>',
        f'<span {fact}>{model["total_tokens"]:,} tokens</span>' if model.get("total_tokens") is not None else "",
        f'<span {fact}>{model["model_time_ms"] / 1000:.1f} s of model time</span>'
        if model.get("model_time_ms") is not None else "",
        f'<span {fact}>{model["check_time_ms"]:,} ms of checking</span>' if model.get("check_time_ms") is not None else "",
        f'<span {fact}>est. ${cost:.4f}</span>' if cost is not None else "",
        f'<span {fact}>validator {esc(data.meta.get("validator", ""))}</span>',
    ]
    head = (
        '<section style="display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; '
        'gap: 20px 48px; padding: 32px 0 28px"><div style="flex: 1 1 560px; min-width: 0">'
        f'<p style="margin: 0 0 12px; {MONO}; font-size: 12px; letter-spacing: 0.1em; color: #948980">'
        f'<a href="/replays/{esc(task_id)}" style="color: #B8ADA1">REPLAYS</a> / {esc(crumb)}</p>'
        '<h1 style="margin: 0 0 12px; font-size: 50px; line-height: 1; font-weight: 700; font-stretch: 116%; '
        f'text-wrap: balance">{esc(headline)}</h1>'
        f'<p style="margin: 0 0 4px; font-size: 18px; font-weight: 600; color: #F4EFE6">{esc(sentence)}</p>'
        '<p style="margin: 0; max-width: 680px; font-size: 18px; color: #B8ADA1; text-wrap: pretty">'
        f'<span style="color: #948980">The brief:</span> {esc(brief_sentence(replay))}</p></div>'
        '<div class="rv-actions" style="display: flex; flex-wrap: wrap; gap: 10px">'
        f'<a href="{esc(base + path)}?try={n}" data-rv-copy style="{PILL}">Copy link</a>'
        f'<a href="{esc(og_path)}" style="{PILL}">Share card</a>'
        f'<a href="{esc(dataset_url)}" style="{PILL}">Open the dataset row</a>'
        '<span class="v2-sr" role="status" data-rv-copied></span></div></section>'
        f'<section class="rv-facts" style="display: flex; flex-wrap: wrap; gap: 8px; padding-bottom: 28px; {MONO}; '
        f'font-size: 13px">{"".join(facts)}</section>')
    main = (
        '<section style="display: flex; flex-wrap: wrap; gap: 32px 48px; align-items: flex-start">'
        f"{_viewer(replay, model, attempt, task_id)}{_detail(model, attempt)}</section>")
    body = (f'<div data-rv-run>{head}{main}{_strip(data, replay, models, model_key, task_id)}'
            f'{_recompute(model, task_id, data.meta.get("suite_id", "eval-v2.1"))}</div>')
    i = next(k for k, m in enumerate(models) if m["key"] == model_key)
    next_model = models[(i + 1) % len(models)]
    ids = data.task_ids
    next_task = ids[(ids.index(task_id) + 1) % len(ids)]
    links = [("← All replays", f"/replays/{task_id}"),
             (f'{next_model["label"]} on this brief →', run_url(task_id, next_model["key"])),
             (f'{model["label"]} on {brief_label(data, next_task)} →', run_url(next_task, model_key)),
             ("Leaderboard", "/leaderboard"), ("Build with it", "/build")]
    title = f'{model["label"]} on {replay["tradition"]}: {phrase} · Pramaan'
    description = f'{sentence} The brief: {brief_sentence(replay)}'
    noscript = ("<noscript><style>.rv-toggle{display:none!important}.rv-eye-copy{display:block!important}"
                "</style></noscript>")
    return page(title=title, description=description, path=path, body=body, footer=footer(links, path),
                current="replays", base_url=base_url, og_image=og_path,
                head_extra=noscript)
