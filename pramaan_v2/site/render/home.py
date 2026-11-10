"""Home `/`: the wall, one check happening, the three findings, the strip and the build hand-off.

Markup and inline styles are ported from the design board `Main.dc.html` (desktop) with the
phone layout from `Phone.dc.html` in `static/css/20-home.css`. Every number comes from the data.
"""
from __future__ import annotations

from functools import lru_cache

from .. import copy as c
from ..data import SiteData
from ..shell import FONT_MONO, esc, footer, json_for_script, page, public_base
from .replays import run_url

FEATURED = "hp_kinnauri-shawl-geometric-border-easy-8"
FEATURED_MODEL = "beam-hf"
KOLAM_TASK = "tn_kolam-easy-22"
KOLAM_MODELS = ("beam-hf", "claude-opus-5-5", "gpt-6-astra")
IMPOSSIBLE_TASK = "py_kolam-as-practised-in-puducherry-easy-35"
BLANK_TASK = "mn_moirang-phee-moirang-pheejin-stepped-triangle-border-easy-14"
COLLECTION = "https://huggingface.co/collections/kshitijthakkar/pramaan-proof-of-art"
LOOP_COLOURS = ("#F4EFE6", "#F4B63F", "#4FD1C5", "#F0704A", "#B79CF2")
COLS = 36

MONO = f"font-family: {FONT_MONO}"
EYEBROW = f"margin: 0 0 12px; {MONO}; font-size: 12px; letter-spacing: 0.12em; color: #948980"
STATEMENT = "margin: 0; font-size: 46px; line-height: 1.02; font-weight: 700; font-stretch: 114%; text-wrap: balance"
PRIMARY = ("display: inline-flex; align-items: center; min-height: 48px; padding: 0 24px; border-radius: 24px; "
           "background: #F4B63F; color: #15110F; font-weight: 700; text-decoration: none")
TEXT_LINK = "display: inline-flex; align-items: center; min-height: 44px; font-weight: 600"


def _png(b64: str | None) -> str:
    return f"data:image/png;base64,{b64}" if b64 else ""


def _pct(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") + "%"


def _sprite_pos(col: int, row: int, rows: int, cols: int = COLS) -> str:
    return f"{_pct(col / max(cols - 1, 1) * 100)} {_pct(row / max(rows - 1, 1) * 100)}"


# Hero --------------------------------------------------------------------------------------------

def _hero(data: SiteData) -> str:
    totals = data.meta["totals"]
    return (
        '<section data-home="hero" class="hh" style="display: flex; flex-wrap: wrap; align-items: flex-end; '
        'gap: 20px 64px; padding: 36px 0 26px">'
        '<div style="flex: 1 1 520px; min-width: 0">'
        f'<p class="hh-eyebrow" style="margin: 0 0 14px; {MONO}; font-size: 12px; letter-spacing: 0.12em; '
        f'color: #948980">{esc(c.hero_eyebrow(totals, len(data.meta["states"])))}</p>'
        '<h1 style="margin: 0; font-size: 66px; line-height: 0.98; font-weight: 700; font-stretch: 118%; '
        'letter-spacing: -0.01em; text-wrap: balance">AI drew it.<br>Math checked it.</h1>'
        f'<p class="hh-phone" style="margin: 10px 0 0; font-size: 15px; color: #B8ADA1">{esc(c.phone_lede(totals))}</p>'
        "</div>"
        '<div class="hh-side" style="flex: 1 1 420px; min-width: 0; display: flex; flex-direction: column; gap: 18px">'
        '<p style="margin: 0; font-size: 18px; color: #B8ADA1; text-wrap: pretty">Every tile on this wall is something '
        "a model actually drew for a brief from an Indian craft tradition. Bright tiles passed every geometric check. "
        "Dim tiles broke a rule. Empty cells never produced a readable pattern.</p>"
        '<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 12px 20px">'
        f'<a href="{esc(run_url(FEATURED, FEATURED_MODEL))}" style="{PRIMARY}">Open a replay</a>'
        '<span style="font-size: 15px; color: #948980">or pick any tile below</span>'
        "</div></div></section>"
    )


# The wall ----------------------------------------------------------------------------------------

def _wall(data: SiteData) -> str:
    meta, wall = data.meta, data.wall
    rows, cols = wall["rows"], wall["cols"]
    totals = meta["totals"]
    board = {m["key"]: m for m in data.leaderboard["models"]}
    labels = {m["key"]: m["label"] for m in data.models}
    states = {s["code"]: s["name"] for s in meta["states"]}
    codes = [s["code"] for s in sorted(meta["states"], key=lambda s: s["index"])]
    sprite = f"/site/wall.png?v={meta['revision'][:10]}"
    runs = sorted(wall["runs"], key=lambda r: (r["row"], r["col"]))

    row_labels = "".join(
        '<div class="hw-row" style="display: flex; align-items: center; justify-content: space-between; gap: 8px; '
        f'border-top: 1px solid #2E2621"><span>{esc(m["label"])}</span><span style="{MONO}; color: #F4B63F">'
        f'{board[m["key"]]["within3"]}</span></div>' for m in data.models)
    head = (f'<div style="{MONO}; font-size: 10px; letter-spacing: 0.1em; color: #948980; display: flex; '
            f'align-items: center; justify-content: space-between"><span>MODEL</span><span>OF {totals["briefs"]}</span></div>')

    tiles = []
    for run in runs:
        kind = {"x": "hw-b", "-": "hw-n"}.get(run["status"], "hw-p")
        name = c.run_name(labels[run["model"]], states[run["state"]], run["level"], run["status"])
        # Each anchor names its own cell and is placed there, so a missing run leaves a gap, not a shift.
        col_, row_ = run["col"], run["row"]
        tiles.append(f'<a class="hw-t {kind}" href="{esc(run_url(run["task"], run["model"]))}" '
                     f'data-col="{col_}" data-row="{row_}" style="--c:{col_};--r:{row_};grid-area:{row_ + 1} / {col_ + 1}" '
                     f'aria-label="{esc(name)}"></a>')

    default = next((r for r in runs if r["task"] == FEATURED and r["model"] == FEATURED_MODEL), runs[0])
    col, row = default["col"], default["row"]
    flip, up = int(col >= 20), int(row >= rows / 2)
    status = default["status"]
    tile_style = ("" if status == "-" else
                  f"background-position: {_sprite_pos(col, row, rows, cols)}")
    tile_label = ("No readable pattern" if status == "-" else
                  f"Tile drawn by {labels[default['model']]} for {states[default['state']]}, {default['level']} brief")
    inspector = (
        '<div class="hw-insp" aria-live="polite" style="position: absolute; z-index: 3; '
        'left: calc((var(--sc) + 1 - var(--flip)) * 100% / 36); '
        'top: calc(24px + (var(--sr) + var(--up)) * (100% - 24px) / var(--rows)); '
        'transform: translate(calc(var(--flip) * (-100% - 10px) + (1 - var(--flip)) * 10px), calc(var(--up) * -100%)); '
        'width: 316px; box-sizing: border-box; padding: 14px; background: #262019; border: 1px solid #4A3F38; '
        'border-radius: 6px; box-shadow: 0 14px 40px rgba(0, 0, 0, 0.55)">'
        f'<div class="hw-tile{" hw-empty" if status == "-" else ""}" role="img" aria-label="{esc(tile_label)}" '
        f'style="{tile_style}"></div>'
        '<div class="hw-info" style="min-width: 0; display: flex; flex-direction: column; gap: 3px; font-size: 14px; line-height: 1.3">'
        f'<strong class="hi-model" style="font-size: 16px">{esc(labels[default["model"]])}</strong>'
        '<span class="hi-where" style="display: flex; flex-direction: column; gap: 3px; color: #B8ADA1">'
        f'<span class="hi-place">{esc(states[default["state"]])}</span>'
        f'<span class="hi-trad"><span class="hi-tn">{esc(default["tradition"])}</span>'
        f'<span class="hi-lv"> · {esc(default["level"])}</span></span></span>'
        f'<span class="hi-verdict" style="margin-top: 4px; {MONO}; font-size: 12px; '
        f'color: {c.VERDICT_TONES[status]}">{c.verdict(status)}</span>'
        "</div>"
        f'<a class="hi-open" href="{esc(run_url(default["task"], default["model"]))}" '
        'style="margin-top: auto; display: inline-flex; align-items: center; min-height: 44px; font-weight: 600">'
        '<span>Open replay</span><span class="hi-arrow" aria-hidden="true">&#160;→</span></a>'
        "</div>"
    )

    lookup = {
        "models": [m["label"] for m in data.models],
        "states": [[s["name"], s["code"]] for s in sorted(meta["states"], key=lambda s: s["index"])],
        "trad": {f'{r["col"]}{r["level"][0]}': r["tradition"] for r in runs},
        "verdicts": c.VERDICTS, "tones": c.VERDICT_TONES,
    }
    blob = json_for_script(lookup, ensure_ascii=False)
    best = data.leaderboard["models"][0] if data.leaderboard["models"] else None
    hint = c.phone_top(best["label"], best["within3"], totals["briefs"]) if best else ""

    return (
        f'<section data-home="wall" class="hw" aria-label="{esc(c.wall_label(totals))}" '
        f'style="--rows:{rows};--sc:{col};--sr:{row};--flip:{flip};--up:{up};--hw-sprite:url(\'{sprite}\'); '
        'display: flex; flex-wrap: wrap; align-items: stretch; gap: 12px">'
        f'<div class="hw-hint" aria-hidden="true" style="{MONO}"><span>{esc(hint)}</span>'
        f'<span>SWIPE FOR ALL {len(codes)} STATES</span></div>'
        f'<div class="hw-labels" style="flex: 0 0 150px; display: grid; grid-template-rows: 24px '
        f'repeat({len(data.models)}, minmax(0, 1fr)); font-size: 14px; line-height: 1.15">{head}{row_labels}</div>'
        '<div class="hw-main" style="flex: 999 1 560px; min-width: 0; position: relative">'
        f'<div class="hw-codes" aria-hidden="true" style="display: grid; grid-template-columns: repeat(36, minmax(0, 1fr)); '
        f'height: 24px; align-items: center; text-align: center; {MONO}; font-size: 10px; color: #948980">'
        + "".join(f"<span>{code}</span>" for code in codes) + "</div>"
        f'<div class="hw-scroll"><div class="hw-grid" data-home-wall data-cols="{cols}" data-rows="{rows}">'
        + "".join(tiles)
        + '<div class="hw-sel" aria-hidden="true"></div></div></div>'
        + inspector
        + "</div>"
        f'<script type="application/json" id="hw-data">{blob}</script>'
        "</section>"
    )


def _legend(data: SiteData) -> str:
    swatch = {
        "passed": "width: 14px; height: 14px; background: #F4EFE6",
        "broke": "width: 14px; height: 14px; background: rgba(244, 239, 230, 0.28)",
        "never": "width: 14px; height: 14px; box-sizing: border-box; border: 1px solid #6B625B",
    }
    items = "".join(
        f'<span style="display: inline-flex; align-items: center; gap: 8px"><span style="{swatch[kind]}"></span>'
        f"{esc(text)}</span>" for kind, text in c.legend(data.meta["totals"]))
    return (
        '<section data-home="legend" class="hw-legend" style="display: flex; flex-wrap: wrap; gap: 10px 28px; '
        f'padding: 16px 0 0 162px; font-size: 14px; color: #B8ADA1">{items}'
        f'<span style="color: #948980">{esc(c.WALL_ROWS_NOTE)}</span></section>'
    )


def _numbers(data: SiteData) -> str:
    cells = "".join(
        '<div style="background: #15110F; padding: 22px 24px">'
        f'<div class="hn-v" style="font-size: 44px; font-weight: 700; font-stretch: 112%; line-height: 1'
        f'{"; color: #4FD1C5" if cyan else ""}">{esc(value)}</div>'
        f'<div style="margin-top: 8px; font-size: 15px; color: #B8ADA1">{esc(caption)}</div></div>'
        for value, caption, cyan in c.four_numbers(data.meta["totals"]))
    return (
        '<section data-home="numbers" style="display: grid; grid-template-columns: repeat(auto-fit, '
        'minmax(min(240px, 100%), 1fr)); gap: 1px; margin-top: 36px; background: #2E2621; border: 1px solid #2E2621">'
        f"{cells}</section>"
    )


# Kolam -------------------------------------------------------------------------------------------

def _dots(rows: int, cols: int, unit: float) -> str:
    return "".join(f"M{(2 * col + 1.5) * unit:g} {(2 * row + 1.5) * unit:g}h.01"
                   for row in range(rows) for col in range(cols))


def kolam_svg(rows: int, cols: int, paths: list[str], label: str, *, unit: float = 40.0,
              stroke: float = 6.8, dot: float = 13) -> str:
    """Traced loops (one colour per loop when there is more than one) over the dot grid."""
    width, height = (2 * cols + 1) * unit, (2 * rows + 1) * unit
    loops = "".join(
        f'<path class="hk-loop" pathLength="1" style="--i:{i}" d="{d}" '
        f'stroke="{LOOP_COLOURS[i % len(LOOP_COLOURS)] if len(paths) > 1 else LOOP_COLOURS[0]}" '
        f'stroke-width="{stroke:g}" stroke-linejoin="round" fill="none"></path>' for i, d in enumerate(paths))
    return (f'<svg class="v2-draw" viewBox="0 0 {width:g} {height:g}" role="img" aria-label="{esc(label)}" '
            'style="display: block; width: 100%; height: auto">'
            f'{loops}<path d="{_dots(rows, cols, unit)}" stroke="#F4EFE6" stroke-width="{dot:g}" '
            'stroke-linecap="round" fill="none"></path></svg>')


def _kolam_label(item: dict) -> str:
    dots = c.Word(item["rows"] * item["cols"])
    loops = item["loops"]
    if not loops:
        return f"{dots} dots and no line"
    if loops == 1:
        return f"{dots} dots and one unbroken line"
    return f"{dots} dots and {c.word(loops)} separate loops in {c.word(loops)} colours"


def _anchor_error(checks: list[dict]) -> bool:
    return any(ch["id"] == "candidate_parse" and ch["status"] == "FAIL" and ".anchor" in ch.get("message", "")
               for ch in checks)


def _parse_line(message: str) -> str:
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if len(lines) >= 3 and "validation error" in lines[0]:
        return f"{lines[1]}: {lines[2].split(' [')[0]}"
    return lines[0] if lines else "Unreadable proposal"


def _try_code(item: dict, checks: list[dict]) -> str:
    if item["outcome"] == "unreadable":
        bad = next((ch for ch in checks if ch["status"] == "FAIL"), None)
        return _parse_line(bad["message"]) if bad else "Unreadable proposal"
    loop = next((ch for ch in checks if ch["id"] == "kolam_loop_count"), None)
    if loop:
        return loop["message"]
    bad = next((ch for ch in checks if ch["status"] == "FAIL"), None)
    return bad["message"] if bad else "Every check passed"


FIGURE = ("margin: 0; padding: 16px; background: #1D1815; border: 1px solid {border}; border-radius: 6px; "
          "display: flex; flex-direction: column; gap: 12px")
DASHED = ("padding: 16px; border: 1px dashed #4A3F38; border-radius: 6px; display: flex; align-items: center; "
          "color: #948980; font-size: 15px")


def _kolam_tries(data: SiteData, key: str, shown: bool) -> str:
    tries = data.kolam[KOLAM_TASK]["models"].get(key, [])
    replay = data.replay(KOLAM_TASK) or {"models": []}
    checks = {t["n"]: t["checks"] for m in replay["models"] if m["key"] == key for t in m["tries"]}
    passed = any(t["outcome"] == "passed" for t in tries)
    figures = []
    for i, item in enumerate(tries):
        last = i == len(tries) - 1
        badge, tone = c.kolam_badge(item, last=last and not passed)
        try_checks = checks.get(item["n"], [])
        caption = c.kolam_caption(item, tries[i - 1] if i else None, anchor_error=_anchor_error(try_checks))
        code = (c.reward_trail([t["reward"] for t in tries]) if last and len(tries) > 1
                else _try_code(item, try_checks))
        if item["rows"] and item["cols"]:
            art = kolam_svg(item["rows"], item["cols"], item["paths"], _kolam_label(item))
        else:
            art = ('<div style="aspect-ratio: 1 / 1; display: flex; align-items: center; justify-content: center; '
                   'border: 1px dashed #4A3F38; color: #948980; font-size: 14px; text-align: center; padding: 12px">'
                   "No pattern to draw</div>")
        figures.append(
            f'<figure style="{FIGURE.format(border="#F4B63F" if item["outcome"] == "passed" else "#2E2621")}">'
            f'<div style="display: flex; justify-content: space-between; {MONO}; font-size: 12px; letter-spacing: 0.08em">'
            f'<span style="color: #948980">TRY {item["n"]}</span><span style="color: {tone}">{esc(badge)}</span></div>'
            f"{art}"
            f'<figcaption style="font-size: 14px; line-height: 1.4; color: #B8ADA1">{esc(caption)}</figcaption>'
            f'<code style="{MONO}; font-size: 11px; color: #948980">{esc(code)}</code></figure>')
    if len(tries) < 3 and passed:
        figures.append(f'<div style="{DASHED}">No {"second" if len(tries) == 1 else "third"} try needed.</div>')
        if len(tries) == 1:
            all_models = data.kolam[KOLAM_TASK]["models"]
            labels = {m["key"]: m["label"] for m in data.models}
            others = [labels[m["key"]] for m in data.models if m["key"] not in KOLAM_MODELS
                      and any(t["n"] == 1 and t["outcome"] == "passed" for t in all_models.get(m["key"], []))]
            never = sum(1 for m in data.models
                        if not any(t["outcome"] == "passed" for t in all_models.get(m["key"], [])))
            figures.append(f'<div style="{DASHED}">{esc(c.kolam_first_try_note(others, never, len(data.models)))}</div>')
    return (f'<div class="hk-tries" data-model="{key}"{"" if shown else " hidden"} style="display: grid; '
            'grid-template-columns: repeat(auto-fit, minmax(min(190px, 100%), 1fr)); gap: 16px">'
            + "".join(figures) + "</div>")


def _kolam(data: SiteData) -> str:
    labels = {m["key"]: m["label"] for m in data.models}
    keys = [k for k in KOLAM_MODELS if k in labels and k in data.kolam[KOLAM_TASK]["models"]]
    state = next(s for s in data.meta["states"] if s["easy"] == KOLAM_TASK or s["hard"] == KOLAM_TASK)
    chips = "".join(
        f'<button type="button" class="hk-chip" data-model="{k}" aria-pressed="{"true" if i == 0 else "false"}" '
        'style="min-height: 44px; padding: 0 18px; border-radius: 22px; '
        f'border: 1px solid {"#F4EFE6" if i == 0 else "#4A3F38"}; background: {"#F4EFE6" if i == 0 else "transparent"}; '
        f'color: {"#15110F" if i == 0 else "#F4EFE6"}; font-family: inherit; font-size: 15px; font-weight: 600; '
        f'cursor: pointer">{esc(labels[k])}</button>' for i, k in enumerate(keys))
    return (
        '<section data-home="kolam" class="hk" style="padding: 104px 0 0">'
        f'<p style="{EYEBROW}">WATCH ONE GET CHECKED · {esc(state["name"].upper())} · SIKKU KOLAM</p>'
        '<div style="display: flex; flex-wrap: wrap; gap: 32px 56px; align-items: flex-start">'
        '<div style="flex: 1 1 360px; min-width: 0; display: flex; flex-direction: column; gap: 18px">'
        f'<h2 class="v2-statement" style="{STATEMENT}">Draw it in one unbroken line.</h2>'
        '<p style="margin: 0; color: #B8ADA1; text-wrap: pretty">The line runs diagonally between nine dots and turns '
        "only at the edge, or at a mirror the model chooses to place. The brief asks for a single closed loop. The "
        "checker traces the line and counts.</p>"
        f'<div role="group" aria-label="Choose a model" style="display: flex; flex-wrap: wrap; gap: 8px">{chips}</div>'
        f'<p style="margin: 0; font-size: 15px; color: #948980; text-wrap: pretty">{esc(c.NO_MIRRORS_NOTE)}</p>'
        "</div>"
        '<div style="flex: 2 1 620px; min-width: 0">'
        + "".join(_kolam_tries(data, k, i == 0) for i, k in enumerate(keys))
        + "</div></div></section>"
    )


# Spot the break ----------------------------------------------------------------------------------

def _spot(data: SiteData) -> str:
    replay = data.replay(FEATURED)
    model = next(m for m in replay["models"] if m["key"] == FEATURED_MODEL)
    first = model["tries"][0]
    pair = first["pair"]
    rings = "".join(
        f'<div style="position: absolute; left: {_pct((x + 0.5) / 64 * 100)}; top: {_pct((y + 0.5) / 64 * 100)}; '
        'width: 26px; height: 26px; margin: -13px 0 0 -13px; box-sizing: border-box; border: 2px solid #4FD1C5; '
        'border-radius: 50%"></div>' for x, y in (pair["original"], pair["transformed"]))
    href = esc(run_url(FEATURED, FEATURED_MODEL))
    alt = f"{model['label']}'s first try at the brief: {replay['brief']}"
    return (
        '<section data-home="spot" class="hs" style="padding: 112px 0 0">'
        f'<p style="{EYEBROW}">TRY IT · {esc(replay["state"].upper())} · {esc(replay["tradition"].upper())}</p>'
        '<div style="display: flex; flex-wrap: wrap; gap: 32px 64px; align-items: center">'
        '<div style="flex: 0 1 440px; min-width: 0">'
        '<div style="position: relative; width: 100%; aspect-ratio: 1 / 1; outline: 1px solid #4A3F38">'
        f'<img class="v2-px" src="{_png(first["tile_png"])}" alt="{esc(alt)}" width="64" height="64" '
        'style="display: block; width: 100%; height: 100%">'
        '<div class="hs-overlay" aria-hidden="true" style="position: absolute; inset: 0">'
        '<div style="position: absolute; inset: 0; background: rgba(21, 17, 15, 0.66)"></div>'
        f'<img class="v2-px" src="{_png(first["mask_png"])}" alt="" width="64" height="64" '
        'style="position: absolute; inset: 0; width: 100%; height: 100%">'
        f"{rings}</div></div></div>"
        '<div style="flex: 1 1 420px; min-width: 0">'
        '<div class="hs-q" hidden style="display: flex; flex-direction: column; gap: 18px">'
        f'<h2 class="v2-statement" tabindex="-1" style="{STATEMENT}">Is this tile mirror-symmetric?</h2>'
        f'<p style="margin: 0; color: #B8ADA1; text-wrap: pretty">{esc(model["label"])} drew it on its first try. The brief '
        "asks for a stepped white hexagon on brick red with a yellow dot at its core, the same when flipped left to "
        "right and top to bottom.</p>"
        '<div style="display: flex; flex-wrap: wrap; gap: 12px">'
        '<button type="button" class="hs-reveal v2-primary" style="min-height: 48px; padding: 0 24px; border-radius: 24px; '
        'border: 0; background: #F4B63F; color: #15110F; font-family: inherit; font-size: 17px; font-weight: 700; '
        'cursor: pointer">Looks symmetric</button>'
        '<button type="button" class="hs-reveal" style="min-height: 48px; padding: 0 24px; border-radius: 24px; '
        'border: 1px solid #F4EFE6; background: transparent; color: #F4EFE6; font-family: inherit; font-size: 17px; '
        'font-weight: 700; cursor: pointer">Something is off</button>'
        "</div></div>"
        '<div class="hs-answer" style="display: flex; flex-direction: column; gap: 18px">'
        f'<h2 class="v2-statement" tabindex="-1" style="{STATEMENT}">Your eye says yes. {first["mismatch_px"]} '
        "pixels say no.</h2>"
        f'<p style="margin: 0; color: #B8ADA1; text-wrap: pretty">Of the tile\'s {c.n(64 * 64)} pixels, '
        f'{first["mismatch_px"]} disagree with their mirror image, all along the slanted edges of the hexagon. '
        "The checker hands back the first pair it finds.</p>"
        f'<p style="margin: 0; padding: 14px 16px; background: #1D1815; border: 1px solid #2E2621; {MONO}; '
        f'font-size: 15px; color: #4FD1C5">{esc(c.pair_text(pair))}</p>'
        f'<p style="margin: 0; color: #B8ADA1; text-wrap: pretty">{esc(c.spot_after(model["label"], model["status"]))}</p>'
        '<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 12px 20px">'
        f'<a href="{href}" style="{PRIMARY}">Open this replay</a>'
        '<button type="button" class="hs-cover" style="min-height: 44px; padding: 0 6px; border: 0; background: '
        'transparent; color: #B8ADA1; font-family: inherit; font-size: 15px; text-decoration: underline; '
        'cursor: pointer">Cover it again</button>'
        "</div></div></div></div></section>"
    )


# Three findings ----------------------------------------------------------------------------------

CARD = ("display: flex; flex-direction: column; gap: 14px; padding: 24px; background: #1D1815; "
        "border: 1px solid #2E2621; border-radius: 6px")
CARD_H3 = "margin: 6px 0 0; font-size: 24px; line-height: 1.15; font-weight: 700"
CARD_P = "margin: 0; font-size: 16px; color: #B8ADA1; text-wrap: pretty"
CARD_TAG = f"margin: auto 0 0; {MONO}; font-size: 12px; letter-spacing: 0.06em; color: #948980"
FIG_CAP = f"margin-top: 8px; {MONO}; font-size: 11px; line-height: 1.5"


def _card_title(text: str, href: str) -> str:
    return f'<h3 style="{CARD_H3}"><a class="hf-link" href="{href}">{esc(text)}</a></h3>'


def _impossible_card(data: SiteData) -> str:
    brief = data.kolam[IMPOSSIBLE_TASK]
    asked = (brief["rows"], brief["cols"])
    labels = {m["key"]: m["label"] for m in data.models}
    order = [m["key"] for m in data.models if m["key"] in brief["models"]]
    accepted = [(k, next(t for t in brief["models"][k] if t["outcome"] == "passed")) for k in order
                if any(t["outcome"] == "passed" for t in brief["models"][k])]
    missed = next((t for k in order for t in brief["models"][k]
                   if (t["rows"], t["cols"]) == asked and t["loops"] and t["loops"] != brief["required_loops"]), None)
    figures, fr = [], []
    if missed:
        fr.append(2 * missed["cols"] + 1)
        figures.append(
            '<figure style="margin: 0">'
            + kolam_svg(missed["rows"], missed["cols"], missed["paths"],
                        f"{c.Word(missed['loops'])} loops on a {c.word(asked[0])} by {c.word(asked[1])} dot grid",
                        stroke=6, dot=12)
            + f'<figcaption style="{FIG_CAP}; color: #E8613C">{asked[0]} × {asked[1]} DOTS<br>'
              f'{c.word(missed["loops"]).upper()} LOOPS</figcaption></figure>')
    if accepted:
        shown = accepted[0][1]
        fr.append(2 * shown["cols"] + 1)
        figures.append(
            '<figure style="margin: 0">'
            + kolam_svg(shown["rows"], shown["cols"], shown["paths"],
                        f"One loop on a {c.word(shown['rows'])} by {c.word(shown['cols'])} dot grid", stroke=6, dot=12)
            + f'<figcaption style="{FIG_CAP}; color: #F4B63F">{shown["rows"]} × {shown["cols"]} DOTS<br>'
              "ONE LOOP · ACCEPTED</figcaption></figure>")
    columns = " ".join(f"minmax(0, {f / fr[0]:.2f}fr)".replace(".00fr", "fr") for f in fr) or "1fr"
    text = c.impossible_kolam(asked, [(labels[k], t["rows"], t["cols"]) for k, t in accepted],
                              len(data.models) - len(accepted))
    href = esc(run_url(IMPOSSIBLE_TASK, accepted[0][0]) if accepted else f"/replays/{IMPOSSIBLE_TASK}")
    return (
        f'<article data-card="kolam" style="{CARD}">'
        f'<div style="display: grid; grid-template-columns: {columns}; gap: 16px; align-items: end">{"".join(figures)}</div>'
        f'{_card_title("The impossible kolam", href)}<p style="{CARD_P}">{esc(text)}</p>'
        f'<p style="{CARD_TAG}">CHECKER GAP · BRIEF FIDELITY</p></article>'
    )


def _blank_card(data: SiteData) -> str:
    replay = data.replay(BLANK_TASK)
    totals = data.meta["totals"]
    blank_runs = [r for r in data.wall["runs"] if r["task"] == BLANK_TASK and r["status"] in "123" and r["low_ink"]]
    order = {m["key"]: i for i, m in enumerate(data.models)}
    blank_runs.sort(key=lambda r: order.get(r["model"], 99))
    figures = []
    if replay and replay.get("reference"):
        figures.append(
            '<figure style="margin: 0">'
            f'<img class="v2-px" src="{_png(replay["reference"]["tile_png"])}" width="64" height="64" '
            'alt="Reference tile: white stepped triangles on black" '
            'style="display: block; width: 100%; height: auto; aspect-ratio: 1 / 1; outline: 1px solid #4A3F38">'
            f'<figcaption style="{FIG_CAP}; color: #948980">REFERENCE</figcaption></figure>')
    href = f"/replays/{BLANK_TASK}"
    if blank_runs and replay:
        key = blank_runs[0]["model"]
        model = next(m for m in replay["models"] if m["key"] == key)
        accepted = next(t for t in model["tries"] if t["outcome"] == "passed")
        href = run_url(BLANK_TASK, key)
        figures.append(
            '<figure style="margin: 0">'
            f'<img class="v2-px" src="{_png(accepted["tile_png"])}" width="64" height="64" '
            f'alt="Accepted tile from {esc(model["label"])}: completely black" '
            'style="display: block; width: 100%; height: auto; aspect-ratio: 1 / 1; outline: 1px solid #4A3F38">'
            f'<figcaption style="{FIG_CAP}; color: #F4B63F">ACCEPTED · REWARD {c.reward_text(accepted["reward"])}'
            "</figcaption></figure>")
    text = c.blank_tile(len(blank_runs), len(data.models), totals["low_ink"], totals["passed"])
    return (
        f'<article data-card="blank" style="{CARD}">'
        '<div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; align-items: end">'
        f'{"".join(figures)}</div>'
        f'{_card_title("A blank tile scores 1.0", esc(href))}<p style="{CARD_P}">{esc(text)}</p>'
        f'<p style="{CARD_TAG}">REWARD GAP · VISIBILITY</p></article>'
    )


def _anchor_card(data: SiteData) -> str:
    briefs = data.meta["totals"]["briefs"]
    top = sorted(data.leaderboard["models"], key=lambda m: -m["anchor_only"])[:3]
    bars = "".join(
        '<div style="display: grid; grid-template-columns: 132px minmax(0, 1fr) 54px; gap: 10px; align-items: center">'
        f'<span>{esc(m["label"])}</span><span style="height: 16px; background: #2E2621"><span style="display: block; '
        f'width: {_pct(m["anchor_only"] / briefs * 100)}; height: 100%; background: #B8ADA1"></span></span>'
        f'<span style="{MONO}; text-align: right">{m["anchor_only"]}/{briefs}</span></div>' for m in top)
    href = esc(run_url(KOLAM_TASK, "claude-opus-5-5"))
    return (
        f'<article data-card="anchor" style="{CARD}">'
        f'<div style="display: flex; flex-direction: column; gap: 10px; padding: 8px 0 4px; font-size: 14px">{bars}'
        f'<span style="{MONO}; font-size: 11px; color: #948980">FIRST TRIES REJECTED FOR THIS FIELD ALONE</span></div>'
        f'{_card_title("One untyped field decided first tries", href)}'
        f'<p style="{CARD_P}">The prompt names layers[].anchor and never says what it holds. Claude models sent a '
        "string. The schema wants [x, y]. For those models the first-try score mostly measures that field, not "
        "geometry, so this site leads with the score within three tries.</p>"
        f'<p style="{CARD_TAG}">HARNESS GAP · PROMPT SCHEMA</p></article>'
    )


def _findings(data: SiteData) -> str:
    return (
        '<section data-home="findings" style="padding: 112px 0 0">'
        f'<p style="{EYEBROW}">FOUND IN THE REPLAYS</p>'
        f'<h2 class="v2-statement" style="{STATEMENT.replace("margin: 0", "margin: 0 0 12px")}">'
        "Three things the scoreboard hid.</h2>"
        '<p style="margin: 0 0 32px; max-width: 720px; color: #B8ADA1; text-wrap: pretty">Each one is a gap in our own '
        "benchmark. We publish them with the runs that expose them, and leave the frozen scores as they are.</p>"
        '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(360px, 100%), 1fr)); gap: 20px">'
        f"{_impossible_card(data)}{_blank_card(data)}{_anchor_card(data)}</div></section>"
    )


# Leaderboard strip -------------------------------------------------------------------------------

def _strip(data: SiteData) -> str:
    briefs = data.meta["totals"]["briefs"]
    board = {m["key"]: m for m in data.leaderboard["models"]}
    rows = []
    for i, model in enumerate(data.models, start=1):
        m = board[model["key"]]
        first, total = m["first"]["passed"], m["within3"]
        later = max(total - first, 0)
        edge = "" if first else "; border-left: 2px solid #F4B63F"
        bottom = "; border-bottom: 1px solid #2E2621" if i == len(data.models) else ""
        rows.append(
            '<div class="hl-row" style="display: grid; grid-template-columns: 28px minmax(110px, 190px) minmax(0, 1fr) 62px; '
            f'gap: 14px; align-items: center; min-height: 46px; border-top: 1px solid #2E2621{bottom}">'
            f'<span class="hl-n" style="{MONO}; font-size: 13px; color: #948980">{i:02d}</span>'
            f'<span class="hl-name" style="font-weight: 600">{esc(model["label"])}</span>'
            f'<span class="hl-bar" role="img" aria-label="{first} on the first try, {later} more on try 2 or 3" '
            'style="display: flex; height: 18px; background: #211B18">'
            f'<span style="width: {_pct(first / briefs * 100)}; background: #F4B63F"></span>'
            f'<span style="width: {_pct(later / briefs * 100)}; box-sizing: border-box; border: 2px solid #F4B63F; '
            f'border-left: 0{edge}"></span></span>'
            f'<span class="hl-total" style="{MONO}; text-align: right">{total}</span></div>')
    return (
        '<section data-home="strip" style="padding: 112px 0 0">'
        '<div style="display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; '
        'gap: 12px 32px; margin-bottom: 28px"><div>'
        f'<p style="{EYEBROW}">LEADERBOARD · {esc(data.meta["suite_id"].upper())} · ONE RUN PER MODEL</p>'
        f'<h2 class="v2-statement" style="{STATEMENT}">{esc(c.strip_heading(briefs))}</h2></div>'
        f'<a href="/leaderboard" style="{TEXT_LINK}">Full leaderboard, cost and failure types →</a></div>'
        '<div style="display: flex; flex-wrap: wrap; gap: 8px 24px; margin-bottom: 14px; font-size: 14px; color: #B8ADA1">'
        '<span style="display: inline-flex; align-items: center; gap: 8px"><span style="width: 22px; height: 12px; '
        'background: #F4B63F"></span>Passed on the first try</span>'
        '<span style="display: inline-flex; align-items: center; gap: 8px"><span style="width: 22px; height: 12px; '
        "box-sizing: border-box; border: 2px solid #F4B63F\"></span>Passed on try 2 or 3, after the checker's feedback"
        "</span></div>"
        f'<div style="display: flex; flex-direction: column">{"".join(rows)}</div>'
        f'<p style="margin: 14px 0 0; font-size: 14px; color: #948980">{esc(c.strip_caption(briefs))}</p></section>'
    )


# Build teaser and open problems ------------------------------------------------------------------

SNIPPET = """from pramaan_env import PramaanEnv, PramaanAction

url = "{base}"
with PramaanEnv(base_url=url).sync() as env:
    obs = env.reset(task_id="tn_kolam-easy-22").observation
    result = env.step(PramaanAction(pattern_json=pattern_json))
    print(result.reward, result.done)"""

PROBLEM_LINKS = ("/build", "/leaderboard#grammar", f"/replays/{KOLAM_TASK}", f"/replays/{IMPOSSIBLE_TASK}",
                 run_url(KOLAM_TASK, "claude-opus-5-5"))


def _build(data: SiteData, base: str) -> str:
    models = data.leaderboard["models"]
    briefs = data.leaderboard["grammar_briefs"]
    problems = c.open_problems(
        jaali_passed=sum(m["by_grammar"].get("jaali_lattice", 0) for m in models),
        jaali_pairs=briefs["jaali_lattice"] * len(models),
        jaali_best=max((m["by_grammar"].get("jaali_lattice", 0) for m in models), default=0),
        jaali_briefs=briefs["jaali_lattice"],
        kolam_none=sum(1 for m in models if not m["by_grammar"].get("kolam_loop", 0)),
        models=len(models), kolam_briefs=briefs["kolam_loop"])
    items = "".join(
        '<li style="display: grid; grid-template-columns: 34px minmax(0, 1fr); gap: 8px; padding: 14px 0; '
        f'border-top: 1px solid #2E2621{"; border-bottom: 1px solid #2E2621" if i == len(problems) else ""}">'
        f'<span style="{MONO}; font-size: 13px; color: #F4B63F">{i:02d}</span>'
        f'<span><a class="hp-link" href="{PROBLEM_LINKS[i - 1]}"><strong>{esc(title)}</strong></a> '
        f'<span style="color: #B8ADA1">{esc(text)}</span></span></li>'
        for i, (title, text) in enumerate(problems, start=1))
    return (
        '<section data-home="build" style="padding: 112px 0 0">'
        f'<p style="{EYEBROW}">BUILD WITH IT · BENCHMARK AND OPENENV ENVIRONMENT</p>'
        '<div style="display: flex; flex-wrap: wrap; gap: 32px 56px; align-items: flex-start">'
        '<div style="flex: 1 1 420px; min-width: 0; display: flex; flex-direction: column; gap: 18px">'
        f'<h2 class="v2-statement" style="{STATEMENT}">The checker is the reward.</h2>'
        '<p style="margin: 0; color: #B8ADA1; text-wrap: pretty">The same deterministic verifier that scored this wall '
        "is an OpenEnv environment. Send a pattern, get back every check, the counterexample and a reward between 0 "
        "and 1. No model credentials, no judge model.</p>"
        '<pre style="margin: 0; padding: 18px 20px; overflow-x: auto; background: #1D1815; border: 1px solid #2E2621; '
        f'border-radius: 6px; {MONO}; font-size: 13px; line-height: 1.6; color: #F4EFE6">{esc(SNIPPET.replace("{base}", base))}</pre>'
        '<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 12px 20px">'
        f'<a href="/build" style="{PRIMARY}">Run your model</a>'
        f'<a href="/build#contract" style="{TEXT_LINK}">Environment contract →</a>'
        f'<a href="/try" style="{TEXT_LINK}">Try it in the studio →</a>'
        f'<a href="/playground" style="{TEXT_LINK}">Play an episode →</a>'
        "</div></div>"
        '<div style="flex: 1 1 420px; min-width: 0">'
        '<h3 style="margin: 0 0 6px; font-size: 24px; font-weight: 700">Open problems</h3>'
        '<p style="margin: 0 0 14px; font-size: 15px; color: #948980">Unclaimed as of this release. Each links to the '
        "runs behind it.</p>"
        f'<ol style="margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column">{items}</ol>'
        "</div></div></section>"
    )


def _limits(data: SiteData) -> str:
    count = data.traditions["totals"]["traditions"]
    return (
        '<section data-home="limits" style="margin-top: 112px; padding: 32px 0; border-top: 1px solid #2E2621; '
        'display: flex; flex-wrap: wrap; gap: 20px 56px">'
        '<h2 style="flex: 1 1 260px; margin: 0; font-size: 28px; line-height: 1.1; font-weight: 700; font-stretch: 112%">'
        "What a pass means, and what it does not.</h2>"
        '<p style="flex: 2 1 480px; margin: 0; color: #B8ADA1; text-wrap: pretty">A pass means every declared rule holds '
        "on a 64 × 64 raster: symmetry, repeats, spacing, loop counts, lattice connectivity. It says nothing about "
        "beauty and never certifies cultural authenticity. These designs are mathematical reinterpretations inspired "
        "by craft traditions, not a substitute for the people who keep them alive. "
        f'<a href="/traditions">The {count} traditions and their sources</a></p></section>'
    )


def footer_links(data: SiteData) -> list[tuple[str, str]]:
    return [
        (f"The {data.traditions['totals']['traditions']} traditions and their sources", "/traditions"),
        ("Results dataset", f"https://huggingface.co/datasets/{data.meta['dataset']}"),
        ("Training data", "https://huggingface.co/datasets/kshitijthakkar/pramaan-proof-of-art-train-v1"),
        ("OpenEnv data", "https://huggingface.co/datasets/kshitijthakkar/pramaan-proof-of-art-openenv"),
        ("Code", "https://github.com/Mandark-droid/Proof-of-Art"),
        ("Everything in one place", COLLECTION),
    ]


@lru_cache(maxsize=4)
def _body(data: SiteData, base: str) -> str:
    return (f'<div class="hm">{_hero(data)}{_wall(data)}{_legend(data)}{_numbers(data)}{_kolam(data)}'
            f"{_spot(data)}{_findings(data)}{_strip(data)}{_build(data, base)}{_limits(data)}</div>")


def render_home(data: SiteData, base_url: str) -> str:
    totals = data.meta["totals"]
    description = (f"{c.Word(totals['models'])} AI models drew {totals['briefs']} briefs from Indian craft traditions. "
                   "A deterministic checker verified every tile. Open any run and see the exact pixel that broke.")
    return page(title="Pramaan · AI drew it. Math checked it.", description=description, path="/",
                body=_body(data, public_base()), footer=footer(footer_links(data), c.footer_right(data.meta), mono=False, margin_top=0),
                current=None, base_url=base_url, og_image="/og/home.png")
