"""Replays: one brief, every model's tries side by side (spec 8.2, ``Replays.dc.html``)."""
from __future__ import annotations

import base64
import io
from functools import lru_cache
from urllib.parse import quote

from ...constants import GRAMMAR_LABELS, number_word
from ..data import SiteData
from ..labels import short_rule
from ..shell import FONT_MONO, esc, page

DEFAULT_TASK = "hp_kinnauri-shawl-geometric-border-easy-8"

MONO = f"font-family: {FONT_MONO}"
CELL_TEXT = f"{MONO}; font-size: 12px"


# ---------------------------------------------------------------- shared helpers (also used by replay.py)

def png_src(b64: str) -> str:
    return f"data:image/png;base64,{b64}"


def tile_img(b64: str, alt: str, style: str, *, hidden: bool = False) -> str:
    alt_attr = 'alt="" aria-hidden="true"' if hidden else f'alt="{esc(alt)}"'
    return f'<img class="v2-px" src="{png_src(b64)}" {alt_attr} style="display: block; {style}">'


def xy(point) -> str:
    return f"({point[0]}, {point[1]})"


def pair_text(pair: dict | None) -> str:
    """'(16, 21) ≠ (47, 21)' for a symmetry pair, '(14, 0)' for a single reported pixel."""
    if not pair:
        return ""
    if pair.get("original") is not None and pair.get("transformed") is not None:
        return f'{xy(pair["original"])} ≠ {xy(pair["transformed"])}'
    if pair.get("pixel") is not None:
        return xy(pair["pixel"])
    return ""


def run_url(task_id: str, model_key: str) -> str:
    """The run page path, each segment percent-quoted (escape it again when it goes into an attribute)."""
    return f"/replays/{quote(task_id, safe='')}/{quote(model_key, safe='')}"


def models_in_order(data: SiteData, replay: dict) -> list[dict]:
    order = data.model_index
    return sorted(replay["models"], key=lambda m: order.get(m["key"], len(order)))


def failing_checks(attempt: dict) -> list[dict]:
    return [c for c in attempt.get("checks") or [] if c.get("status") != "PASS"]


def is_truncated(attempt: dict) -> bool:
    return any(c.get("id") == "candidate_truncated" for c in failing_checks(attempt))


def state_index(data: SiteData, code: str) -> int:
    return next(i for i, s in enumerate(data.meta["states"]) if s["code"] == code)


def brief_label(data: SiteData, task_id: str) -> str:
    """'Himachal Pradesh · hard' for a brief in wall order."""
    for state in data.meta["states"]:
        for level in ("easy", "hard"):
            if state[level] == task_id:
                return f'{state["name"]} · {level}'
    return task_id


@lru_cache(maxsize=128)
def dominant_colour(b64: str) -> str:
    """The most common colour of a tile, for the legend swatch."""
    from PIL import Image

    image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA").convert("RGB")
    _, (r, g, b) = max(image.getcolors(64 * 64))
    return f"#{r:02X}{g:02X}{b:02X}"


# ---------------------------------------------------------------- the page

def _counts(replay: dict, models: list[dict]) -> str:
    summary = replay["summary"]
    first, repaired, never = summary["first_try"], summary["repaired"], summary["never"]
    observation, observed_model = _observation(models)
    items = [
        f'<li><strong style="color: #F4EFE6">{first}</strong> {"model" if first == 1 else "models"} passed on the first try</li>',
        f"<li><strong style=\"color: #F4EFE6\">{repaired}</strong> repaired it after the checker's feedback</li>",
    ]
    never_text = f'<strong style="color: #F4EFE6">{never}</strong> never passed'
    if observation and observed_model["status"] in ("x", "-"):
        items.append(f"<li>{never_text}. {esc(observation)}</li>")
    else:
        items.append(f"<li>{never_text}</li>")
        if observation:
            items.append(f"<li>{esc(observation)}</li>")
    return ('<ul style="flex: 1 1 280px; min-width: 0; margin: 0; padding: 0; list-style: none; display: flex; '
            'flex-direction: column; gap: 6px; font-size: 15px; color: #B8ADA1">' + "".join(items) + "</ul>")


def _observation(models: list[dict]) -> tuple[str | None, dict | None]:
    """One generated observation: a model returning the same reported pair on consecutive tries."""
    best, best_model = 1, None
    for model in models:
        streak, previous = 1, None
        for attempt in model["tries"]:
            pair = attempt.get("pair") or {}
            key = (tuple(pair["original"]), tuple(pair["transformed"])) \
                if pair.get("original") is not None and pair.get("transformed") is not None else None
            streak = streak + 1 if key is not None and key == previous else 1
            previous = key
            if streak > best:
                best, best_model = streak, model
    if best_model is None:
        return None, None
    name = best_model["label"]
    if best >= 3:
        return f"{name} was handed the same pixel pair twice and returned it a third time.", best_model
    return f"{name} was handed a pixel pair and returned the same pair on its next try.", best_model


def _cell(task_id: str, model: dict, n: int, passed_before: bool) -> str:
    label = f"TRY {n}"
    if passed_before:
        return (f'<td class="rv-cell rv-skip" data-try="{label}" style="padding: 10px 12px; color: #948980">'
                "not needed</td>")
    attempt = next((a for a in model["tries"] if a["n"] == n), None)
    if attempt is None:
        return f'<td class="rv-cell rv-skip" data-try="{label}" style="padding: 10px 12px; color: #948980">none</td>'
    who = f'{model["label"]}, try {n}'
    inner: list[str] = []
    tile = "flex: 0 0 80px; width: 80px; height: 80px"
    if attempt["outcome"] == "unreadable":
        inner.append(f'<span class="rv-empty" role="img" aria-label="{esc(who)}: unreadable, no tile" '
                     f'style="{tile}; box-sizing: border-box; border: 1px dashed #4A3F38"></span>')
        sub = (f'<br><span style="color: #948980">CUT OFF</span>' if is_truncated(attempt) else "")
        inner.append(f'<span class="rv-txt" style="{CELL_TEXT}; line-height: 1.4; color: #B8ADA1">UNREADABLE{sub}</span>')
    elif attempt["outcome"] == "passed":
        inner.append(tile_img(attempt["tile_png"], f"{who}: passed", tile))
        inner.append(f'<span class="rv-txt" style="{CELL_TEXT}; color: #F4B63F">PASSED</span>')
    else:
        inner.append(tile_img(attempt["tile_png"], f"{who}: broke a rule", tile))
        if attempt.get("mask_png"):
            inner.append(tile_img(attempt["mask_png"], "", f"{tile}; background-color: #1D1815", hidden=True))
        if attempt.get("mismatch_px") is not None:
            head = f'{attempt["mismatch_px"]} PX OFF'
        else:
            fails = failing_checks(attempt)
            head = short_rule(fails[0]["id"]).upper() if fails else "BROKE A RULE"
        detail = pair_text(attempt.get("pair")) or (attempt.get("measured") or "")
        inner.append(f'<span class="rv-txt" style="{CELL_TEXT}; line-height: 1.4; color: #E8613C">{esc(head)}'
                     f'<br><span style="color: #948980">{esc(detail)}</span></span>')
    return (f'<td class="rv-cell" data-try="{label}" style="padding: 10px 12px">'
            f'<div class="rv-in" style="display: flex; align-items: center; gap: 10px">{"".join(inner)}</div></td>')


def _row(task_id: str, model: dict, last: bool) -> str:
    border = "border-top: 1px solid #2E2621" + ("; border-bottom: 1px solid #2E2621" if last else "")
    cells, passed = [], False
    for n in (1, 2, 3):
        cells.append(_cell(task_id, model, n, passed))
        attempt = next((a for a in model["tries"] if a["n"] == n), None)
        passed = passed or bool(attempt and attempt["outcome"] == "passed")
    return (f'<tr style="{border}"><th scope="row" style="padding: 10px 12px 10px 0; text-align: left; font-weight: 600">'
            f'<a href="{run_url(task_id, model["key"])}">{esc(model["label"])}</a></th>{"".join(cells)}</tr>')


def _controls(data: SiteData, replay: dict) -> str:
    level, code = replay["level"], replay["state_code"]
    options, links = [], []
    for state in data.meta["states"]:
        target = state[level]
        selected = ' selected=""' if state["code"] == code else ""
        options.append(f'<option value="{esc(target)}"{selected}>{esc(state["name"])}</option>')
        links.append(f'<li><a href="/replays/{esc(target)}" style="display: inline-flex; align-items: center; '
                     f'min-height: 44px">{esc(state["name"])}</a></li>')
    state = data.meta["states"][state_index(data, code)]
    chips = []
    for lv, text in (("easy", "Easy brief"), ("hard", "Hard brief")):
        on = lv == level
        colours = ("border: 1px solid #F4EFE6; background: #F4EFE6; color: #15110F" if on
                   else "border: 1px solid #4A3F38; background: transparent; color: #F4EFE6")
        current = ' aria-current="page"' if on else ""
        chips.append(f'<a href="/replays/{esc(state[lv])}"{current} style="display: inline-flex; align-items: center; '
                     f'min-height: 44px; box-sizing: border-box; padding: 0 18px; border-radius: 22px; {colours}; '
                     f'font-size: 15px; font-weight: 600; text-decoration: none">{text}</a>')
    return (
        '<div class="rv-controls" style="display: flex; flex-wrap: wrap; align-items: flex-end; gap: 12px">'
        '<div style="display: flex; flex-direction: column; gap: 4px">'
        '<label for="brief-state" style="font-size: 13px; color: #948980">State or Union Territory</label>'
        '<select id="brief-state" name="task" data-rv-state style="min-height: 44px; padding: 0 12px; '
        'border: 1px solid #4A3F38; border-radius: 6px; background: #1D1815; color: #F4EFE6; font-family: inherit; '
        f'font-size: 15px">{"".join(options)}</select></div>'
        f'<div role="group" aria-label="Brief difficulty" style="display: flex; gap: 8px">{"".join(chips)}</div>'
        '<a href="/" style="display: inline-flex; align-items: center; min-height: 44px; padding: 0 6px; '
        'font-weight: 600">Back to the wall</a>'
        '<noscript><details style="flex-basis: 100%"><summary style="display: flex; align-items: center; '
        'min-height: 44px; cursor: pointer; color: #B8ADA1">Choose a state or Union Territory</summary>'
        '<ul style="margin: 0; padding: 0; list-style: none; columns: 14em; font-size: 15px">'
        + "".join(links) + "</ul></details></noscript></div>"
    )


def _brief_card(data: SiteData, replay: dict, models: list[dict]) -> str:
    grammar = GRAMMAR_LABELS.get(replay["grammar"], replay["grammar"])
    mono_line = " · ".join([replay["state"], replay["tradition"], grammar, f'{replay["symmetry"]} symmetry']).upper()
    sources = (f'<a href="/traditions/{esc(replay["state_code"])}">Sources for this tradition</a>')
    reference = (replay.get("reference") or {}).get("tile_png")
    if reference:
        picture = tile_img(reference, "Certified reference tile for this brief",
                           "flex: 0 0 112px; width: 112px; height: 112px; outline: 1px solid #4A3F38")
        note = ("Left: the certified reference. It shows the brief can be met; it is not the only valid answer. "
                + sources)
    else:
        picture = (f'<div role="img" aria-label="No certified reference" style="flex: 0 0 112px; width: 112px; '
                   f'height: 112px; box-sizing: border-box; display: flex; align-items: center; justify-content: center; '
                   f'border: 1px dashed #6B625B; {MONO}; font-size: 11px; letter-spacing: 0.08em; color: #948980; '
                   f'text-align: center">NO<br>REFERENCE</div>')
        note = ("There is no certified reference for this brief: it cannot be met as written. A pass here means the "
                "declared rules hold, not that the brief was met. " + sources)
    return (
        '<section class="rv-brief" style="display: flex; flex-wrap: wrap; gap: 20px 32px; align-items: center; '
        'padding: 20px 24px; background: #1D1815; border: 1px solid #2E2621; border-radius: 6px">'
        f'{picture}'
        '<div style="flex: 1 1 420px; min-width: 0">'
        f'<p style="margin: 0 0 6px; {MONO}; font-size: 12px; letter-spacing: 0.08em; color: #948980">{esc(mono_line)}</p>'
        f'<p style="margin: 0; font-size: 20px; line-height: 1.35; text-wrap: pretty">{esc(replay["brief"])}</p>'
        f'<p style="margin: 8px 0 0; font-size: 14px; color: #948980">{note}</p></div>'
        f'{_counts(replay, models)}</section>'
    )


def _legend(replay: dict, models: list[dict]) -> str:
    swatch = None
    reference = (replay.get("reference") or {}).get("tile_png")
    if reference:
        swatch = dominant_colour(reference)
    else:
        for model in models:
            tile = next((a.get("tile_png") for a in model["tries"] if a.get("tile_png")), None)
            if tile:
                swatch = dominant_colour(tile)
                break
    disagree = ("Every pixel that disagrees with a mirror image" if replay["symmetry"].startswith("D")
                else "Every pixel that disagrees with the tile turned")
    item = 'style="display: inline-flex; align-items: center; gap: 8px"'
    return (
        '<div style="display: flex; flex-wrap: wrap; gap: 8px 24px; margin-bottom: 12px; font-size: 14px; color: #B8ADA1">'
        f'<span {item}><span style="width: 14px; height: 14px; background: {swatch or "#9A2B1F"}"></span>What the model drew</span>'
        f'<span {item}><span style="width: 14px; height: 14px; background: #4FD1C5"></span>{disagree}</span>'
        '<span style="color: #948980">The pair in brackets is the first counterexample the checker returned.</span>'
        "</div>"
    )


def _footer(data: SiteData, task_id: str, path: str) -> str:
    ids = data.task_ids
    i = ids.index(task_id)
    link = ('<a href="/replays/{}" style="display: inline-flex; align-items: center; min-height: 44px; '
            'padding-right: 16px">{}</a>')
    parts = []
    if i > 0:
        parts.append(link.format(esc(ids[i - 1]), "← " + esc(brief_label(data, ids[i - 1]))))
    if i < len(ids) - 1:
        parts.append(link.format(esc(ids[i + 1]), esc(brief_label(data, ids[i + 1])) + " →"))
    return (
        '<footer style="display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: center; margin-top: 40px; '
        'padding: 20px 0 56px; font-size: 15px">' + "".join(parts)
        + f'<span style="margin-left: auto; {MONO}; font-size: 12px; color: #948980; overflow-wrap: anywhere">'
        f"/replays/{esc(task_id)}</span></footer>"
    )


def render_replays(data: SiteData, task_id: str, base_url: str, path: str) -> str | None:
    replay = data.replay(task_id)
    if replay is None:
        return None
    models = models_in_order(data, replay)
    n_models = len(data.models)
    total = len(data.task_ids)
    head = (
        '<section style="display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; '
        'gap: 20px 48px; padding: 32px 0 24px">'
        '<div style="flex: 1 1 480px; min-width: 0">'
        f'<p style="margin: 0 0 12px; {MONO}; font-size: 12px; letter-spacing: 0.1em; color: #948980">'
        f'REPLAYS · BRIEF {replay["number"]} OF {total}</p>'
        '<h1 style="margin: 0; font-size: 50px; line-height: 1; font-weight: 700; font-stretch: 116%; '
        f'text-wrap: balance">One brief. {number_word(n_models).capitalize()} answers.</h1></div>'
        f"{_controls(data, replay)}</section>"
    )
    rows = "".join(_row(task_id, m, i == len(models) - 1) for i, m in enumerate(models))
    th = 'scope="col" style="padding: 8px 12px; font-weight: 500"'
    table = (
        '<section style="padding: 28px 0 0">' + _legend(replay, models)
        + '<div class="rv-scroll" style="overflow-x: auto">'
        '<table class="rv-table" style="width: 100%; min-width: 980px; border-collapse: collapse; font-size: 15px">'
        f'<thead><tr style="{MONO}; font-size: 11px; letter-spacing: 0.1em; color: #948980; text-align: left">'
        '<th scope="col" style="padding: 8px 12px 8px 0; font-weight: 500; width: 170px">MODEL</th>'
        f"<th {th}>TRY 1</th><th {th}>TRY 2</th><th {th}>TRY 3</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div></section>"
    )
    body = f'<div data-rv-brief>{head}{_brief_card(data, replay, models)}{table}</div>'
    summary = replay["summary"]
    level = replay["level"]
    title = f'One brief, {number_word(n_models)} answers: {replay["tradition"]}, {level} · Pramaan'
    description = (f'{replay["brief"]} {summary["first_try"]} of {n_models} models passed on the first try, '
                   f'{summary["repaired"]} repaired it, {summary["never"]} never passed.')
    return page(title=title, description=description, path=path, body=body,
                footer=_footer(data, task_id, path), current="replays", base_url=base_url)
