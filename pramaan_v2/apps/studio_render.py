"""Pure render functions for the studio: v1 studio state in, v2 markup out.

Try it is ported from ``TryIt.dc.html`` (spec 8.6) and Submit a run from ``Community.dc.html`` (spec 8.5).
The state is v1's (``pramaan.app._initial_custom_state`` / ``_handle_custom_event``); a few ``v2_`` keys
written by ``pramaan_v2.apps.studio.on_event`` carry what v1 does not keep (sign-in, last action, the
community form without the key). Event field names and actions are exactly the ones v1 reads.
"""
from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from pramaan import app as v1
from pramaan.budget import MODEL_CALL_BUDGET
from pramaan.orchestrate import MAX_BRIEF_CHARS

from ..constants import GRAMMAR_LABELS, STATE_CODES, number_word
from ..site.labels import check_label, check_row_label
from ..site.shell import FONT_MONO, FONT_SANS, esc, footer, header

SHORT_NAMES = Path(__file__).resolve().parents[1] / "short_names.json"
COMMUNITY_REPO = "kshitijthakkar/pramaan-community-runs"
COMMUNITY_URL = f"https://huggingface.co/datasets/{COMMUNITY_REPO}"
# The community protocol v1 enforces (``_community_model_config_is_eligible``, ``_run_community_evaluation``).
QUICK_BRIEFS = 12
TOKEN_CEILING = 4096
CAP_USD = 10
TRIES = 3
SUBMIT_TARGET = "/studio/#/submit"
LOGIN_HREF = "/login/huggingface?_target_url=" + quote(SUBMIT_TARGET, safe="")
LOGOUT_HREF = "/logout?_target_url=" + quote(SUBMIT_TARGET, safe="")

MONO = f"font-family: {FONT_MONO}"
STEP = f"margin: 0; {MONO}; font-size: 12px; letter-spacing: 0.1em; color: #F4B63F"
EYEBROW = f"margin: 0 0 12px; {MONO}; font-size: 12px; letter-spacing: 0.12em; color: #948980"
FIELD = "display: flex; flex-direction: column; gap: 4px"
LABEL = "font-size: 14px; color: #B8ADA1"
SMALL_LABEL = "font-size: 13px; color: #B8ADA1"
HINT = "font-size: 13px; color: #948980"
CONTROL = ("min-height: 44px; padding: 0 12px; border: 1px solid #4A3F38; border-radius: 6px; background: #1D1815; "
           "color: #F4EFE6; font-family: inherit; font-size: 16px")
SMALL_CONTROL = ("min-height: 44px; padding: 0 10px; box-sizing: border-box; border: 1px solid #4A3F38; "
                 "border-radius: 6px; background: #1D1815; color: #F4EFE6; font-family: inherit; font-size: 15px")
GROUND_CONTROL = ("min-height: 44px; padding: 0 12px; box-sizing: border-box; border: 1px solid #4A3F38; "
                  f"border-radius: 6px; background: #15110F; color: #F4EFE6; {MONO}; font-size: 14px")
PRIMARY = ("min-height: 48px; padding: 0 22px; border-radius: 24px; border: 0; background: #F4B63F; color: #15110F; "
           "font-family: inherit; font-size: 17px; font-weight: 700; cursor: pointer")
PRIMARY_WIDE = PRIMARY.replace("padding: 0 22px", "padding: 0 24px")
SECONDARY = ("min-height: 48px; padding: 0 22px; border-radius: 24px; border: 1px solid #F4EFE6; "
             "background: transparent; color: #F4EFE6; font-family: inherit; font-size: 17px; font-weight: 600; "
             "cursor: pointer")
PILL = ("min-height: 44px; padding: 0 16px; border-radius: 22px; border: 1px solid #4A3F38; background: transparent; "
        "color: #F4EFE6; font-family: inherit; font-size: 15px; font-weight: 600; cursor: pointer")
PILL_LINK = PILL.replace("cursor: pointer", "display: inline-flex; align-items: center; box-sizing: border-box; "
                                             "text-decoration: none")
STATUS = "font-size: 14px; color: #948980"
CHECKBOX = "flex: none; width: 20px; height: 20px; margin: 2px 0 0; accent-color: #F4B63F"
CHECK_ROW = "display: flex; align-items: flex-start; gap: 10px; min-height: 44px; font-size: 15px"
PRE = (f"margin: 6px 0 0; padding: 10px 12px; overflow-x: auto; background: #15110F; border: 1px solid #2E2621; "
       f"border-radius: 6px; {MONO}; font-size: 12px; line-height: 1.55")
TICK = ('<svg width="18" height="18" viewBox="0 0 18 18" role="img" aria-label="Passed"><path d="M4 9.5l3.2 3.2L14 5.8" '
        'stroke="#F4B63F" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"></path></svg>')
CROSS = ('<svg width="18" height="18" viewBox="0 0 18 18" role="img" aria-label="Failed"><path d="M5 5l8 8M13 5l-8 8" '
         'stroke="#E8613C" stroke-width="2" stroke-linecap="round" fill="none"></path></svg>')
ARROW = ('<svg width="24" height="24" viewBox="0 0 28 28" aria-hidden="true" style="flex: 0 0 24px; align-self: center">'
         '<path d="M5 14h17M16 8l6 6-6 6" stroke="#948980" stroke-width="2" stroke-linecap="round" '
         'stroke-linejoin="round" fill="none"></path></svg>')
SYMMETRY_WORDS = {"d2": "left-right and top-bottom", "d4": "every flip and quarter turn", "c2": "a half turn",
                  "c4": "quarter turns", "none": "none requested"}
TRANSFORM_WORDS = {"F_vertical": "a left-right flip", "F_horizontal": "a top-bottom flip", "R90": "a quarter turn",
                   "R180": "a half turn", "R270": "a three-quarter turn", "F_diagonal": "a diagonal flip",
                   "F_antidiagonal": "an anti-diagonal flip"}
EXCLUSION_WORDS = {
    "Not a proof of cultural authenticity": "Cultural authenticity",
    "Not a guarantee of physical manufacturing quality": "physical print quality",
    "Raster checks do not prove continuous vector seam geometry": "continuous vector seams",
}
MEASURED = {"kolam_loop_count", "kolam_mirror_symmetry", "jaali_open_fraction", "jaali_min_web",
            "jaali_holes_interior", "jaali_single_solid"}


# ------------------------------------------------------------------ deployment flags (patched in tests)

def archive_publishing_ready() -> bool:
    return bool(v1._community_submission_archive_ready())


def runner_ready() -> bool:
    return bool(v1._community_model_choices())


def beam_ready() -> bool:
    return bool(os.getenv("HF_TOKEN"))


@lru_cache(maxsize=1)
def _short_names() -> dict[str, str]:
    return json.loads(SHORT_NAMES.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def canonical_model_count() -> int:
    from ..site.data import site_data

    return len(site_data().models)


@lru_cache(maxsize=1)
def suite_briefs() -> int:
    from ..site.data import site_data

    return int(site_data().meta["totals"]["briefs"])


# ------------------------------------------------------------------ words

_WORDING = (
    (re.compile(r"64×64 discrete resolution"), "64 × 64"),
    (re.compile(r"\bafter 1 proposal\b"), "after 1 try"),
    (re.compile(r"\bafter (\d+) proposals\b"), r"after \1 tries"),
    (re.compile(r"\bpass@1 (\d+)/(\d+)"), r"\1 of \2 passed on the first try"),
    (re.compile(r"\bpass@3 (\d+)/(\d+)"), r"\1 of \2 passed within three"),
    (re.compile(r"\bBeam attempt\b"), "Beam proposal"),
    (re.compile(r"\battempts\b"), "tries"),
    (re.compile(r"\battempt\b"), "try"),
    (re.compile(r"\bcandidates\b"), "proposals"),
    (re.compile(r"\bcandidate\b"), "proposal"),
    (re.compile(r"\bquestions\b"), "briefs"),
    (re.compile(r"\bquestion\b"), "brief"),
    (re.compile(r"\bPatternDSL could not be verified\. Check the JSON and retry\."),
     "The pattern could not be checked. Fix the JSON and recheck."),
    (re.compile(r"\bPatternDSL\b"), "pattern"),
    (re.compile(r"\bPASS\b"), "passed"),
    (re.compile(r"\bFAIL\b"), "broke a rule"),
    (re.compile(r"\btask tradition\b"), "brief's tradition"),
    (re.compile(r"\bper task\b"), "per brief"),
)


def plain(text: object) -> str:
    """v1 status text in v2 vocabulary (spec 3.1)."""
    value = str(text or "")
    for pattern, replacement in _WORDING:
        value = pattern.sub(replacement, value)
    return value


def _json(text: object) -> object:
    try:
        return json.loads(str(text))
    except (TypeError, ValueError):
        return None


def _xy(point) -> str:
    return f"({point[0]}, {point[1]})"


# ------------------------------------------------------------------ small builders

def _select(control_id: str, field: str, value, options, *, style: str = CONTROL, attrs: str = "") -> str:
    items = []
    for option_value, option_label in options:
        selected = " selected" if str(option_value) == str(value) else ""
        items.append(f'<option value="{esc(option_value)}"{selected}>{esc(option_label)}</option>')
    return (f'<select id="{control_id}" data-field="{field}"{attrs} style="{style}">'
            + "".join(items) + "</select>")


def _number(control_id: str, field: str, value, lo, hi, step) -> str:
    return (f'<input id="{control_id}" data-field="{field}" type="number" min="{lo}" max="{hi}" step="{step}" '
            f'value="{esc(value)}" placeholder="{esc(value)}" style="{SMALL_CONTROL}">')


def _small_field(control_id: str, label: str, control: str) -> str:
    return (f'<div style="{FIELD}"><label for="{control_id}" style="{SMALL_LABEL}">{esc(label)}</label>'
            f'{control}</div>')


def _image_uri(image) -> str:
    try:
        return v1._image_data_uri(image) if image is not None else ""
    except Exception:
        return ""


# ------------------------------------------------------------------ Try it: choose / make / edit

def _example_label(label: str, value: str) -> str:
    if value in v1.EXAMPLES:
        return value
    label = plain(label)
    return re.sub(r" · (easy|hard) · ", r" · \1 brief · ", label)


def _choose(form: dict, item) -> str:
    state_name = str(form.get("state", ""))
    tradition_id = str(form.get("tradition", ""))
    states = [(name, name) for name in v1.ALL_STATES]
    traditions = [(row.id, row.name) for row in v1._tradition_choices(state_name)]
    examples = [(value, _example_label(label, value)) for label, value in v1._example_choices(state_name, tradition_id)]
    example = form.get("example", "")
    if example not in {value for value, _ in examples}:
        example = examples[0][0] if examples else ""
    grammar = GRAMMAR_LABELS.get(item.grammar, item.grammar).lower() if item else "choose a tradition"
    code = STATE_CODES.get(state_name)
    sources = (f' <a href="/traditions/{code}">Sources and cautions for this tradition</a>' if code else "")
    if examples:
        example_select = _select("st-ex", "example", example, examples)
    else:
        example_select = _select("st-ex", "example", "", [("", "No saved example for this tradition")],
                                 attrs=" disabled")
    return (
        '<div class="st-step st-choose" style="display: flex; flex-direction: column; gap: 14px">'
        f'<p style="{STEP}">1 · CHOOSE</p>'
        f'<div style="{FIELD}"><label for="st-state" style="{LABEL}">State or Union Territory</label>'
        + _select("st-state", "state", state_name, states) + "</div>"
        f'<div style="{FIELD}"><label for="st-trad" style="{LABEL}">Tradition</label>'
        + _select("st-trad", "tradition", tradition_id, traditions or [("", "No sourced tradition yet")])
        + f'<span style="{HINT}">The tradition sets the grammar, the palette and the starting rules. '
          f'Pattern type: {esc(grammar)}.{sources}</span></div>'
        f'<div style="{FIELD}"><label for="st-ex" style="{LABEL}">Worked example for this tradition</label>'
        + example_select + "</div></div>"
    )


def _advanced(form: dict, item) -> str:
    palettes = item.palette_presets if item else {}
    palette_options = [(name, " · ".join(colours)) for name, colours in palettes.items()] or [("", "No preset")]
    grammar = item.grammar if item else ""
    symmetry = (["None", "D2", "D4"] if grammar == "periodic_raster" else ["None", "C2", "D2", "C4", "D4"])
    symmetry_value = form.get("symmetry", "D2")
    if symmetry_value not in symmetry:
        symmetry_value = "D2"
    fields = [
        _small_field("st-pal", "Palette preset",
                     _select("st-pal", "palette", form.get("palette", ""), palette_options, style=SMALL_CONTROL)),
        _small_field("st-sym", "Symmetry",
                     _select("st-sym", "symmetry", symmetry_value, [(v, v) for v in symmetry], style=SMALL_CONTROL)),
        _small_field("st-col", "Maximum colours", _number("st-col", "max_colours", form.get("max_colours", 4), 1, 6, 1)),
        _small_field("st-min", "Minimum coverage",
                     _number("st-min", "min_coverage", form.get("min_coverage", 0.02), 0, 0.5, 0.01)),
        _small_field("st-max", "Maximum coverage",
                     _number("st-max", "max_coverage", form.get("max_coverage", 0.7), 0.1, 1, 0.05)),
    ]
    if grammar == "dot_field":
        fields.append(_small_field("st-clr", "Minimum dot clearance",
                                   _number("st-clr", "min_clearance", form.get("min_clearance", 0), 0, 64, 1)))
    elif grammar == "kolam_loop":
        fields.append(_small_field("st-loops", "Required kolam loops",
                                   _number("st-loops", "required_loops", form.get("required_loops", 1), 1, 81, 1)))
    elif grammar == "counted_grid":
        fields.append(_small_field("st-cell", "Counted-grid cell size",
                                   _select("st-cell", "cell_size", form.get("cell_size", 4),
                                           [(v, v) for v in (2, 4, 8)], style=SMALL_CONTROL)))
    elif grammar == "stripe_band":
        fields.append(_small_field("st-axis", "Repeat axis",
                                   _select("st-axis", "translation_axis", form.get("translation_axis", "x"),
                                           [(v, v) for v in ("x", "y", "xy")], style=SMALL_CONTROL)))
        fields.append(_small_field("st-period", "Repeat period",
                                   _number("st-period", "translation_period", form.get("translation_period", 8),
                                           1, 64, 1)))
    elif grammar == "jaali_lattice":
        fields.append(_small_field("st-frame", "Jaali frame",
                                   _number("st-frame", "frame", form.get("frame", 0), 0, 12, 1)))
        fields.append(_small_field("st-web", "Minimum jaali web",
                                   _number("st-web", "min_web", form.get("min_web", 0), 0, 32, 1)))
    fields.append(_small_field("st-gap", "Minimum motif spacing",
                               _number("st-gap", "min_gap", form.get("min_gap", 0), 0, 8, 1)))
    checked = " checked" if form.get("exclusive_ink_layers") else ""
    return (
        '<details style="border: 1px solid #2E2621; border-radius: 6px">'
        '<summary style="min-height: 44px; display: flex; align-items: center; padding: 0 14px; font-weight: 600; '
        'cursor: pointer">Advanced rules</summary>'
        '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(170px, 100%), 1fr)); gap: 12px; '
        'padding: 4px 14px 16px">' + "".join(fields)
        + f'<label for="st-excl" style="{CHECK_ROW}; grid-column: 1 / -1; align-items: center; font-size: 14px">'
          f'<input id="st-excl" type="checkbox" data-field="exclusive_ink_layers"{checked} style="{CHECKBOX}; '
          'margin: 0">Ink layers must not overlap</label>'
        + "</div></details>"
    )


def _make(form: dict, item, state: dict) -> str:
    beam = beam_ready()
    last = state.get("v2_last_action")
    status = ""
    if last in {"generate", "example", "state", "tradition"}:
        status = plain((state.get("result") or {}).get("status", ""))
    if not beam and not status:
        status = "Beam is not configured in this build, so Generate is off. The examples and recheck still work."
    has_examples = bool(v1._example_choices(form.get("state"), form.get("tradition")))
    generate_attrs = "" if beam else ' disabled title="Beam is not configured in this build"'
    example_attrs = "" if has_examples else " disabled"
    return (
        '<div class="st-step st-make" style="display: flex; flex-direction: column; gap: 14px">'
        f'<p style="{STEP}">2 · MAKE</p>'
        f'<div style="{FIELD}"><label for="st-brief" style="{LABEL}">Pattern brief</label>'
        f'<textarea id="st-brief" data-field="brief" rows="3" maxlength="{MAX_BRIEF_CHARS}" '
        'placeholder="Describe the pattern you want, in a sentence or two." style="padding: 12px; box-sizing: border-box; '
        'border: 1px solid #4A3F38; border-radius: 6px; background: #1D1815; color: #F4EFE6; font-family: inherit; '
        f'font-size: 16px; line-height: 1.4; resize: vertical">{esc(form.get("brief", ""))}</textarea></div>'
        '<div class="st-actions" style="display: flex; flex-wrap: wrap; align-items: center; gap: 10px">'
        f'<button type="button" class="v2-primary" data-action="generate"{generate_attrs} style="{PRIMARY}">'
        'Generate with Beam</button>'
        f'<button type="button" data-action="example"{example_attrs} style="{SECONDARY}">Load example</button>'
        f'<span role="status" data-status="make" style="flex: 1 1 100%; {STATUS}">{esc(status)}</span></div>'
        f'<p style="margin: 0; {HINT}; text-wrap: pretty">A generation uses the Space owner\'s inference allowance: '
        f'one at a time, {MODEL_CALL_BUDGET.daily_limit} model calls per UTC day across the Space. Your prompt goes to '
        'Hugging Face Inference Providers only when you press Generate.</p>'
        + _advanced(form, item) + "</div>"
    )


def _edit(form: dict, item, state: dict) -> str:
    status = plain((state.get("result") or {}).get("status", "")) if state.get("v2_last_action") == "recheck" else ""
    hint = ("Move one dot and recheck to see the checker name the pair that now sits too close."
            if item and item.grammar == "dot_field" else
            "Change any value and recheck. If a rule breaks, the checker names the pixels that break it.")
    return (
        '<div class="st-step st-edit" style="display: flex; flex-direction: column; gap: 14px">'
        f'<p style="{STEP}">3 · EDIT AND RECHECK</p>'
        f'<div style="{FIELD}"><label for="st-src" style="{LABEL}">Pattern source</label>'
        '<textarea id="st-src" data-field="source" rows="11" spellcheck="false" style="padding: 14px; '
        'box-sizing: border-box; border: 1px solid #4A3F38; border-radius: 6px; background: #1D1815; color: #F4EFE6; '
        f'{MONO}; font-size: 13px; line-height: 1.55; resize: vertical">{esc(form.get("source", ""))}</textarea>'
        f'<span style="{HINT}">{esc(hint)}</span></div>'
        '<div class="st-actions" style="display: flex; flex-wrap: wrap; align-items: center; gap: 10px">'
        f'<button type="button" data-action="recheck" style="{SECONDARY}">Recheck my edit</button>'
        f'<span role="status" data-status="recheck" style="flex: 1 1 100%; {STATUS}">{esc(status)}</span></div></div>'
    )


# ------------------------------------------------------------------ Try it: result panel

def _detail(check: dict, constraints: dict) -> str:
    cid, message = str(check.get("id", "")), str(check.get("message", ""))
    if cid.startswith("symmetry_"):
        return SYMMETRY_WORDS.get(cid.removeprefix("symmetry_"), "")
    if cid == "complexity_bound":
        match = re.match(r"(\d+) primitives; limit (\d+)", message)
        return f"{match.group(1)} primitives of {match.group(2)}" if match else ""
    if cid == "palette_limit":
        match = re.match(r"(\d+) colours? used", message)
        return f"{match.group(1)} used" if match else ""
    if cid == "coverage_bound":
        match = re.match(r"Ink coverage ([0-9.]+); allowed \[([0-9.]+), ([0-9.]+)\]", message)
        if not match:
            return ""
        value, low, high = (float(v) for v in match.groups())
        return (f"{value:.4f}; allowed up to {high:.2f}" if low == 0
                else f"{value:.4f}; allowed {low:.2f} to {high:.2f}")
    if cid == "dot_clearance" and check.get("status") == "PASS":
        clearance = constraints.get("min_clearance") if isinstance(constraints, dict) else None
        return f"every pair clears {clearance} px" if isinstance(clearance, int) else ""
    if cid in MEASURED and "\n" not in message:
        return message
    return ""


def _evidence(check: dict) -> str:
    message = str(check.get("message", "")).split("\n", 1)[0]
    example = check.get("counterexample") or {}
    original, transformed = example.get("original"), example.get("transformed")
    if isinstance(original, list) and isinstance(transformed, list):
        words = TRANSFORM_WORDS.get(str(example.get("transform")), "the transform")
        return f"Pixel {_xy(original)} and {_xy(transformed)} differ under {words}"
    pixel, other = example.get("pixel"), example.get("other_pixel")
    if isinstance(pixel, list) and isinstance(other, list):
        return f"{message}: pixels {_xy(pixel)} and {_xy(other)}"
    if isinstance(pixel, list):
        return f"{message}: pixel {_xy(pixel)}"
    return message


def _check_list(checks: list[dict], constraints: dict) -> str:
    rows = []
    for check in checks:
        label = esc(check_row_label(check) if check.get("status") == "PASS" else check_label(check.get("id", "")))
        if check.get("status") == "PASS":
            detail = _detail(check, constraints)
            extra = f' <span style="color: #948980">· {esc(detail)}</span>' if detail else ""
            rows.append('<li style="display: grid; grid-template-columns: 22px minmax(0, 1fr); gap: 10px; padding: 9px 0; '
                        f'border-top: 1px solid #2E2621">{TICK}<span>{label}{extra}</span></li>')
        else:
            rows.append(
                '<li style="display: grid; grid-template-columns: 22px minmax(0, 1fr); gap: 10px; padding: 9px 10px 9px 0; '
                'border-top: 1px solid #E8613C; border-bottom: 1px solid #E8613C; background: #1D1815">'
                f'{CROSS}<span><strong>{label}</strong> <span style="color: #948980">· {esc(check.get("id", ""))}</span><br>'
                f'<span style="{MONO}; font-size: 13px; color: #4FD1C5; overflow-wrap: anywhere">'
                f'{esc(_evidence(check))}</span></span></li>')
    return ('<ul class="st-checks" style="margin: 0; padding: 0; list-style: none; display: grid; '
            'grid-template-columns: repeat(auto-fit, minmax(min(280px, 100%), 1fr)); gap: 0 28px; font-size: 15px; '
            'line-height: 1.35">' + "".join(rows) + "</ul>")


def _bundle_control(result: dict, verified: bool) -> str:
    bundle = result.get("bundle")
    if isinstance(bundle, str) and bundle and Path(bundle).is_file():
        try:
            encoded = base64.b64encode(Path(bundle).read_bytes()).decode("ascii")
        except OSError:
            encoded = ""
        if encoded:
            return (f'<a data-bundle-download download="pramaan-run-bundle.zip" '
                    f'href="data:application/zip;base64,{encoded}" style="{PILL_LINK}">Download run bundle</a>')
    if verified:
        return (f'<button type="button" data-action="recheck" data-bundle-request style="{PILL}">'
                'Download run bundle</button>')
    return ""


def _proof(result: dict, certificate: dict, code: str | None) -> str:
    passed = certificate.get("all_hard_constraints_pass") is True
    traditions = (f'<a href="/traditions/{code}" style="display: inline-flex; align-items: center; min-height: 44px; '
                  'padding: 0 16px; font-size: 15px; font-weight: 600">Where this tradition’s brief comes from →</a>'
                  if code else "")
    if not passed:
        bundle = _bundle_control(result, False)
        return (
            '<div class="st-proof" style="padding: 20px 22px; background: #1D1815; border: 1px solid #E8613C; '
            'border-radius: 6px">'
            f'<h2 style="margin: 0 0 12px; {MONO}; font-size: 12px; font-weight: 500; letter-spacing: 0.12em; '
            'color: #E8613C">NO PROOF YET</h2>'
            '<p style="margin: 0; font-size: 15px; color: #B8ADA1">A rule is broken, so there is nothing to recompute. '
            'The failed check above names the pixels that break it. Fix the pattern and recheck it.</p>'
            f'<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px">{bundle}{traditions}</div></div>'
        )
    grid = certificate.get("grid") or [64, 64]
    size = f"{grid[0]} × {grid[1]}" if isinstance(grid, list) and len(grid) == 2 else "64 × 64"
    exclusions = [EXCLUSION_WORDS.get(item, item) for item in certificate.get("exclusions", []) if isinstance(item, str)]
    covered = ", ".join(exclusions) if exclusions else "Not recorded"
    encoded = base64.b64encode(json.dumps(certificate, ensure_ascii=False, sort_keys=True, indent=2)
                               .encode("utf-8")).decode("ascii")
    rows = []
    for field, label in (("dsl_sha256", "Design SHA-256"), ("mask_sha256", "Pixels SHA-256")):
        digest = certificate.get(field)
        if isinstance(digest, str):
            rows.append(f'<dt style="color: #948980">{label}</dt><dd style="margin: 0; {MONO}; '
                        f'overflow-wrap: anywhere">{esc(digest)}</dd>')
    rows.append(f'<dt style="color: #948980">Validator</dt><dd style="margin: 0">'
                f'{esc(certificate.get("validator_version", "not recorded"))} · exhaustive on this {size} raster</dd>')
    rows.append(f'<dt style="color: #948980">Does not cover</dt><dd style="margin: 0">{esc(covered)}</dd>')
    return (
        '<div class="st-proof" style="padding: 20px 22px; background: #1D1815; border: 1px solid #F4B63F; '
        'border-radius: 6px">'
        f'<h2 style="margin: 0 0 12px; {MONO}; font-size: 12px; font-weight: 500; letter-spacing: 0.12em; '
        'color: #F4B63F">PROOF YOU CAN RECOMPUTE</h2>'
        '<dl style="margin: 0; display: grid; grid-template-columns: 130px minmax(0, 1fr); gap: 8px 16px; '
        f'font-size: 14px">{"".join(rows)}</dl>'
        '<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-top: 16px">'
        f'<button type="button" data-copy-certificate data-certificate-b64="{encoded}" style="{PILL}">'
        'Copy certificate JSON</button>'
        + _bundle_control(result, True) + traditions
        + f'<span role="status" data-status="copy" style="{STATUS}"></span></div>'
        '<p style="margin: 14px 0 0; font-size: 13px; color: #948980">Anyone can recompute this. It is not a digital '
        'signature. Runs stay in your session and downloads expire after one hour.</p></div>'
    )


def _attempts(history_text: str) -> str:
    history = _json(history_text)
    if not isinstance(history, dict):
        return ('<p style="margin: 0; font-size: 15px; color: #B8ADA1">Tries appear here after you generate with '
                'Beam.</p>')
    rows = []
    for attempt in history.get("attempts") or []:
        if not isinstance(attempt, dict):
            continue
        report = attempt.get("report") or {}
        number = attempt.get("index", len(rows) + 1)
        if attempt.get("candidate") is None:
            outcome, colour = "unreadable", "#B8ADA1"
            detail = ""
        elif report.get("all_hard_constraints_pass"):
            outcome, colour, detail = "passed", "#F4B63F", ""
        else:
            outcome, colour = "broke a rule", "#E8613C"
            failed = [check_label(c.get("id", "")) for c in report.get("checks", [])
                      if isinstance(c, dict) and c.get("status") == "FAIL"]
            detail = ", ".join(failed)
        extra = f' <span style="color: #948980">· {esc(detail)}</span>' if detail else ""
        rows.append(f'<li style="padding: 10px 0; border-top: 1px solid #2E2621"><strong>Try {esc(number)}</strong> '
                    f'<span style="color: {colour}">{outcome}</span>{extra}</li>')
    if rows:
        summary = ('<ol style="margin: 0; padding: 0; list-style: none; font-size: 15px; border-bottom: 1px solid '
                   f'#2E2621">{"".join(rows)}</ol>')
    else:
        summary = ('<p style="margin: 0; font-size: 15px; color: #B8ADA1">This pattern was loaded from a saved run, '
                   'so there are no live tries. Its record is below.</p>')
    record = json.dumps(history, indent=2, ensure_ascii=False)
    return (summary + '<details style="margin-top: 14px; border: 1px solid #2E2621; border-radius: 6px">'
            '<summary style="min-height: 44px; display: flex; align-items: center; padding: 0 14px; font-weight: 600; '
            'cursor: pointer">Full run record (JSON)</summary>'
            f'<pre style="margin: 0; padding: 14px; max-height: 420px; overflow: auto; {MONO}; font-size: 12px; '
            f'line-height: 1.55; border-top: 1px solid #2E2621">{esc(record)}</pre></details>')


def _proposals(gallery) -> str:
    figures = []
    for entry in gallery or []:
        try:
            image, label = entry
        except (TypeError, ValueError):
            continue
        uri = _image_uri(image)
        if not uri:
            continue
        caption = plain(label)
        figures.append(f'<figure style="margin: 0; display: flex; flex-direction: column; gap: 6px">'
                       f'<img class="v2-px" src="{uri}" alt="{esc(caption)}" decoding="async" '
                       'style="display: block; width: 100%; height: auto; outline: 1px solid #4A3F38">'
                       f'<figcaption style="font-size: 14px; color: #B8ADA1">{esc(caption)}</figcaption></figure>')
    if not figures:
        return ('<p style="margin: 0; font-size: 15px; color: #B8ADA1">Proposal previews appear here after you '
                'generate or recheck.</p>')
    return ('<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(min(180px, 100%), 1fr)); '
            f'gap: 16px">{"".join(figures)}</div>')


def _tab(key: str, label: str, selected: bool) -> str:
    border = "#F4B63F" if selected else "transparent"
    colour = "#F4EFE6" if selected else "#B8ADA1"
    return (f'<button type="button" role="tab" id="st-tab-{key}" aria-controls="st-panel-{key}" '
            f'aria-selected="{"true" if selected else "false"}" tabindex="{"0" if selected else "-1"}" '
            f'data-tab="{key}" style="min-height: 44px; padding: 0 14px; border: 0; border-bottom: 2px solid {border}; '
            f'background: transparent; color: {colour}; font-family: inherit; font-size: 15px; font-weight: 600; '
            f'cursor: pointer">{label}</button>')


def _result(form: dict, item, state: dict) -> str:
    result = state.get("result") or {}
    certificate = _json(result.get("certificate"))
    certificate = certificate if isinstance(certificate, dict) else None
    source = _json(result.get("source"))
    constraints = source.get("constraints") if isinstance(source, dict) else {}
    title = source.get("title") if isinstance(source, dict) and isinstance(source.get("title"), str) else "The pattern"
    state_name = str(form.get("state", ""))
    code = STATE_CODES.get(state_name)
    tradition = _short_names().get(str(form.get("tradition", "")), item.name if item else "")
    grammar = GRAMMAR_LABELS.get(item.grammar, "") if item else ""
    eyebrow = " · ".join(part.upper() for part in (state_name, tradition, grammar) if part)
    uri = _image_uri(result.get("image"))
    if uri:
        figure = (f'<div class="st-preview" data-preview="repeat"><img class="v2-px st-preview-img" src="{uri}" '
                  f'alt="{esc(title)}: the checked tile repeated three by three." style="display: block; width: 100%; '
                  'height: auto; image-rendering: pixelated; outline: 1px solid #4A3F38"></div>')
    else:
        figure = ('<div class="st-empty" style="display: flex; align-items: center; justify-content: center; '
                  'min-height: 240px; max-width: 640px; padding: 24px; box-sizing: border-box; border: 1px dashed '
                  '#6B625B; border-radius: 6px; text-align: center; font-size: 15px; color: #B8ADA1">'
                  'Load an example or generate a pattern to see it here, repeated three by three.</div>')
    status = plain(result.get("status", ""))
    if certificate is not None:
        checks = [c for c in certificate.get("checks", []) if isinstance(c, dict)]
        passed = sum(1 for c in checks if c.get("status") == "PASS")
        colour = "#F4B63F" if checks and passed == len(checks) else "#E8613C"
        summary = (
            '<div style="display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 20px">'
            f'<span style="font-size: 44px; line-height: 1; font-weight: 700; font-stretch: 112%; color: {colour}">'
            f'{passed} of {len(checks)}</span><span style="font-size: 18px">checks passed</span>'
            f'<span style="font-size: 14px; color: #948980">{esc(status)}</span></div>')
        checks_panel = _check_list(checks, constraints or {}) + _proof(result, certificate, code)
    else:
        summary = (f'<p style="margin: 0; font-size: 18px; color: #B8ADA1">{esc(status or "Choose a worked example to see its checks.")}</p>')
        bundle = _bundle_control(result, False)
        checks_panel = ('<p style="margin: 0; font-size: 15px; color: #B8ADA1">The checks and the proof appear here '
                        'once a pattern has been checked.</p>'
                        + (f'<div style="margin-top: 16px">{bundle}</div>' if bundle else ""))
    return (
        '<div class="st-result" style="flex: 1.35 1 560px; min-width: 0; display: flex; flex-direction: column; gap: 20px">'
        '<div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px">'
        '<div role="group" aria-label="Preview" style="display: flex; gap: 8px">'
        '<button type="button" data-preview-choice="tile" aria-pressed="false" style="min-height: 44px; padding: 0 16px; '
        'border-radius: 22px; border: 1px solid #4A3F38; background: transparent; color: #F4EFE6; font-family: inherit; '
        'font-size: 15px; font-weight: 600; cursor: pointer">One tile</button>'
        '<button type="button" data-preview-choice="repeat" aria-pressed="true" style="min-height: 44px; padding: 0 16px; '
        'border-radius: 22px; border: 1px solid #F4EFE6; background: #F4EFE6; color: #15110F; font-family: inherit; '
        'font-size: 15px; font-weight: 600; cursor: pointer">As a 3 × 3 repeat</button></div>'
        f'<span style="{MONO}; font-size: 12px; color: #948980">{esc(eyebrow)}</span></div>'
        + figure + summary
        + '<div role="tablist" aria-label="Verifier evidence" style="display: flex; flex-wrap: wrap; gap: 4px; '
          'border-bottom: 1px solid #2E2621">'
        + _tab("checks", "Checks and proof", True) + _tab("attempts", "Attempts", False)
        + _tab("proposals", "Proposals", False) + "</div>"
        + f'<div role="tabpanel" id="st-panel-checks" aria-labelledby="st-tab-checks" data-tabpanel="checks" '
          f'style="display: flex; flex-direction: column; gap: 20px">{checks_panel}</div>'
        + f'<div role="tabpanel" id="st-panel-attempts" aria-labelledby="st-tab-attempts" data-tabpanel="attempts" '
          f'hidden>{_attempts(result.get("history", ""))}</div>'
        + f'<div role="tabpanel" id="st-panel-proposals" aria-labelledby="st-tab-proposals" data-tabpanel="proposals" '
          f'hidden>{_proposals(result.get("gallery"))}</div>'
        + "</div>"
    )


def render_try(state: dict) -> str:
    form = dict(state.get("form") or {})
    item = v1._tradition_by_id(form.get("tradition"))
    models = canonical_model_count()
    body = (
        '<section style="display: flex; flex-wrap: wrap; align-items: flex-end; gap: 20px 64px; padding: 36px 0 36px">'
        '<div style="flex: 1 1 520px; min-width: 0">'
        f'<p style="{EYEBROW}">TRY IT · THE STUDIO</p>'
        '<h1 style="margin: 0; font-size: 58px; line-height: 0.98; font-weight: 700; font-stretch: 118%">'
        'Pick a tradition.<br>Watch it get checked.</h1></div>'
        '<p style="flex: 0 1 480px; min-width: 0; margin: 0; font-size: 18px; color: #B8ADA1; text-wrap: pretty">'
        'Load a verified example, ask Beam for a new pattern, or edit the pattern yourself and recheck it. No sign-in. '
        'The built-in examples work without a single model call.</p></section>'
        '<section class="st-layout" style="display: flex; flex-wrap: wrap; gap: 32px 48px; align-items: flex-start">'
        '<div class="st-steps" style="flex: 1 1 420px; min-width: 0; display: flex; flex-direction: column; gap: 28px">'
        + _choose(form, item) + _make(form, item, state) + _edit(form, item, state) + "</div>"
        + _result(form, item, state) + "</section>"
    )
    links = [(f"See how {number_word(models)} models did on the same traditions", "/replays"),
             ("Use the checker in your own code", "/build")]
    return (header("try") + f"<main>{body}</main>"
            + footer(links, "Inspired by craft traditions. Not artisan-authentic work.", mono=False))


# ------------------------------------------------------------------ Submit a run

def _tomorrow_utc() -> str:
    return (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()


_RUN_SUMMARY = re.compile(r"Community run (?P<run>\S+): (?P<count>\d+)/(?P<limit>\d+) requested questions; "
                          r"pass@1 (?P<p1>\d+)/(?P<n1>\d+); pass@3 (?P<p3>\d+)/(?P<n3>\d+)\.")


def _run_summary(status: str) -> dict | None:
    match = _RUN_SUMMARY.search(status or "")
    if not match:
        return None
    found = match.groupdict()
    for key, pattern in (("reported", r"provider-reported cost ([^;]+);"), ("estimate", r"rate-card estimate ([^;]+);")):
        hit = re.search(pattern, status)
        found[key] = hit.group(1).strip() if hit else "not reported"
    return found


def _path_a_status(state: dict) -> str:
    progress = state.get("community_progress")
    if isinstance(progress, dict):
        done, total = progress.get("done", 0), progress.get("total", QUICK_BRIEFS)
        spend = progress.get("spend_usd", 0) or 0
        return (f"Run in progress: Brief {done} of {total} · {progress.get('passed', 0)} passed so far · "
                f"USD {float(spend):.2f} spent so far.")
    status = str(state.get("community_status", "") or "")
    if not status:
        return ("No community run has been started in this session." if runner_ready()
                else "The community runner is not enabled in this Space build yet.")
    if status.startswith("Full-suite estimate for") and isinstance(state.get("community_estimate"), dict):
        return "Estimate ready. No provider request was made."
    if "already used its one" in status:
        return (f"This account or browser session has already used today's capped run. The next run is allowed from "
                f"00:00 UTC on {_tomorrow_utc()}. No provider request was made.")
    summary = _run_summary(status)
    if summary:
        return (f"Run finished: {summary['p3']} of {summary['n3']} briefs passed within three. Results stay in this "
                "session and are not added to any dataset.")
    return plain(status)


def _estimate(state: dict) -> str:
    estimate = state.get("community_estimate")
    if not isinstance(estimate, dict):
        return ""
    observed = estimate.get("rate_card_observed_at")
    source = esc(estimate.get("rate_card_source", "provider rate card")) + (f", observed {esc(observed)}" if observed else "")
    briefs = estimate.get("task_count", suite_briefs())
    over = estimate.get("estimated_cost_exceeds_per_run_cap")
    cap_note = (" The estimate is above the cap, so the run is likely to stop before every brief is done."
                if over else " The estimate is within the cap; actual provider billing can differ.")
    row = 'style="color: #948980"'
    return (
        '<div class="cm-estimate" style="padding: 14px 16px; background: #15110F; border: 1px solid #2E2621; '
        'border-radius: 6px; font-size: 14px">'
        '<dl style="margin: 0; display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 6px 14px">'
        f'<dt {row}>Model</dt><dd style="margin: 0; {MONO}; overflow-wrap: anywhere">{esc(estimate.get("model_key", ""))}</dd>'
        f'<dt {row}>Price source</dt><dd style="margin: 0">{source}</dd>'
        f'<dt {row}>Estimated cost</dt><dd style="margin: 0">USD {float(estimate.get("estimated_usd", 0)):.4f} for '
        f'{esc(briefs)} briefs</dd>'
        f'<dt {row}>Hard cap</dt><dd style="margin: 0">USD {CAP_USD} per run</dd></dl>'
        f'<p style="margin: 10px 0 0; color: #948980">{cap_note.strip()} No provider request was made.</p></div>'
    )


def _path_a(state: dict) -> str:
    saved = state.get("v2_community") or {}
    ready = runner_ready()
    off = "" if ready else ' disabled title="Community evaluation is not enabled in this Space build"'
    briefs = suite_briefs()
    limit = str(saved.get("task_limit", QUICK_BRIEFS))
    charge = " checked" if saved.get("charge") else ""
    full = " checked" if saved.get("full") else ""
    return (
        '<article style="display: flex; flex-direction: column; gap: 16px; min-width: 0; padding: 28px; '
        'background: #1D1815; border: 1px solid #2E2621; border-radius: 6px">'
        f'<p style="{STEP}">A · RUN IT HERE, PRIVATELY</p>'
        '<h2 style="margin: 0; font-size: 28px; line-height: 1.1; font-weight: 700">Run one model on the frozen briefs '
        'with your own key.</h2>'
        f'<p style="margin: 0; font-size: 15px; color: #B8ADA1; text-wrap: pretty">Run {QUICK_BRIEFS} briefs, or '
        f'estimate the full {briefs} before you commit. Same {TOKEN_CEILING:,}-token ceiling and {number_word(TRIES)} '
        'tries per brief as the canonical runs. Only models with a published price are accepted, and a hard USD '
        f'{CAP_USD} cap allows one capped run per account per UTC day.</p>'
        f'<div style="{FIELD}"><label for="cm-model" style="{LABEL}">LiteLLM model ID (provider/model or nested route)'
        '</label><input id="cm-model" data-field="community_model" type="text" autocapitalize="none" spellcheck="false" '
        f'value="{esc(saved.get("model", ""))}" placeholder="e.g. huggingface/together/meta-llama/Llama-3.3-70B-Instruct" '
        f'style="{GROUND_CONTROL}"{" disabled" if not ready else ""}></div>'
        '<div style="display: flex; flex-wrap: wrap; align-items: flex-end; gap: 12px">'
        f'<div style="{FIELD}; flex: 1 1 220px; min-width: 0"><label for="cm-size" style="{LABEL}">Run size</label>'
        + _select("cm-size", "community_task_limit", limit,
                  [(str(QUICK_BRIEFS), f"Quick run · {QUICK_BRIEFS} briefs"), (str(briefs), f"Full suite · {briefs} briefs")],
                  style=GROUND_CONTROL.replace(f"{MONO}; font-size: 14px", "font-family: inherit; font-size: 15px"))
        + "</div>"
        f'<button type="button" data-action="community_estimate"{off} style="{PILL}">Estimate full-suite cost, '
        'no provider call</button></div>'
        + _estimate(state)
        + f'<div style="{FIELD}"><label for="cm-key" style="{LABEL}">Your provider API key</label>'
          '<input id="cm-key" data-field="community_api_key" type="password" autocomplete="off" autocapitalize="none" '
          f'spellcheck="false" style="{GROUND_CONTROL}">'
          f'<span style="{HINT}">The key passes through this Space to the provider and is not saved in your profile or '
          'in the run. Use a disposable key with a provider-side limit.</span></div>'
        f'<label for="cm-charge" style="{CHECK_ROW}"><input id="cm-charge" type="checkbox" '
        f'data-field="community_charge_consent"{charge} style="{CHECKBOX}"><span>This run may incur provider charges; '
        'I authorize use of my key.</span></label>'
        f'<label for="cm-full" style="{CHECK_ROW}"><input id="cm-full" type="checkbox" '
        f'data-field="community_full_run_cost_consent"{full} style="{CHECKBOX}"><span>For a full suite, I reviewed the '
        f'no-call estimate and authorize charges up to the Space\'s USD {CAP_USD} hard cap.</span></label>'
        '<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 12px 20px">'
        f'<button type="button" class="v2-primary" data-action="community_run"{off} style="{PRIMARY_WIDE}">'
        'Run privately</button>'
        f'<span role="status" data-community-status style="flex: 1 1 260px; {STATUS}">{esc(_path_a_status(state))}</span>'
        '</div>'
        '<p style="margin: 0; padding-top: 14px; border-top: 1px solid #2E2621; font-size: 14px; color: #948980; '
        'text-wrap: pretty">Results stay private in this session and are not added to any dataset. When the run ends '
        'you can see how it did here and publish a packed run with path B.</p></article>'
    )


def _archive_status(state: dict, signed_in: bool, publishing: bool) -> str:
    status = str(state.get("community_archive_status", "") or "")
    if status.startswith("Archive not published:"):
        reasons = [part.strip() for part in status.removeprefix("Archive not published:").split(";") if part.strip()]
        items = "".join(f"<li>{esc(reason[:1].upper() + reason[1:])}</li>" for reason in reasons)
        return ('<p style="margin: 0; color: #E8613C">The archive was rejected. Nothing was published.</p>'
                f'<ul style="margin: 6px 0 0; padding-left: 20px">{items}</ul>')
    match = re.match(r"Published (\d+) replay-audited record\(s\)", status)
    if match:
        return (f'<p style="margin: 0; color: #F4EFE6">Published {match.group(1)} replay-audited rows to '
                f'<a href="{COMMUNITY_URL}">{COMMUNITY_REPO}</a>. They appear on the '
                '<a href="/leaderboard#community">leaderboard under Community</a> once the snapshot refreshes.</p>')
    if status:
        return f'<p style="margin: 0">{esc(plain(status))}</p>'
    if not publishing:
        return ('<p style="margin: 0">Publishing from the Space is switched off right now. Use the pull request route '
                'below.</p>')
    if not signed_in:
        return '<p style="margin: 0">Sign in with Hugging Face, choose a ZIP and give consent to begin.</p>'
    return '<p style="margin: 0">Choose a ZIP and give consent to begin.</p>'


def _path_b(state: dict, user: str | None) -> str:
    publishing = archive_publishing_ready()
    saved = state.get("v2_community") or {}
    consent = " checked" if saved.get("archive_consent") else ""
    if publishing and not user:
        action = (f'<a href="{LOGIN_HREF}" class="v2-primary" data-signin-link style="{PRIMARY_WIDE}; display: inline-flex; '
                  'align-items: center; box-sizing: border-box; text-decoration: none">Sign in with Hugging Face to '
                  'publish</a>')
    else:
        disabled = "" if publishing else " disabled aria-disabled=\"true\""
        action = (f'<button type="button" class="v2-primary" data-action="community_archive_submit"{disabled} '
                  f'style="{PRIMARY_WIDE}">Validate and publish</button>')
    code = (f'<pre style="{PRE}">uv run --project . python benchmarks/run.py \\\n'
            '  --suite benchmarks/suites/eval-v2.1.jsonl \\\n  --model your-model-key \\\n'
            '  --output benchmarks/results/my-run.jsonl</pre>')
    pack = (f'<pre style="{PRE}">uv run --project . python -m scripts.pack_submission \\\n'
            '  benchmarks/results/my-run.jsonl benchmarks/results/my-run.zip</pre>')
    num = f'style="{MONO}; font-size: 13px; color: #F4B63F"'
    li = 'style="display: grid; grid-template-columns: 28px minmax(0, 1fr); gap: 8px"'
    return (
        '<article style="display: flex; flex-direction: column; gap: 16px; min-width: 0; padding: 28px; '
        'background: #1D1815; border: 1px solid #2E2621; border-radius: 6px">'
        f'<p style="{STEP}">B · UPLOAD A PACKED RUN</p>'
        '<h2 style="margin: 0; font-size: 28px; line-height: 1.1; font-weight: 700">Ran it on your own machine? Upload '
        'the archive.</h2>'
        '<ol style="margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 12px; '
        'font-size: 15px">'
        f'<li {li}><span {num}>01</span><span style="min-width: 0">Run the frozen suite.{code}</span></li>'
        f'<li {li}><span {num}>02</span><span style="min-width: 0">Pack it. The packer writes a SHA-256 manifest.{pack}'
        '</span></li>'
        f'<li {li}><span {num}>03</span><span>Upload the ZIP below.</span></li></ol>'
        '<div style="display: flex; flex-direction: column; gap: 6px; padding: 20px; border: 1px dashed #6B625B; '
        'border-radius: 6px">'
        '<label for="cm-zip" style="font-size: 15px; font-weight: 600">Submission archive</label>'
        '<input id="cm-zip" data-field="community_archive_file" type="file" accept=".zip" style="min-height: 44px; '
        'font-family: inherit; font-size: 15px; color: #B8ADA1">'
        f'<span style="{HINT}">One ZIP, 20 MB at most, holding exactly manifest.json and results.jsonl.</span></div>'
        f'<label for="cm-consent" style="{CHECK_ROW}"><input id="cm-consent" type="checkbox" '
        f'data-field="community_archive_consent"{consent} style="{CHECKBOX}"><span>I consent to these rows being '
        'published in the public community dataset under CC BY 4.0. They contain no API keys or personal information.'
        '</span></label>'
        '<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 12px 20px">' + action
        + f'<div role="status" data-community-archive-status style="flex: 1 1 260px; {STATUS}">'
          f'{_archive_status(state, bool(user), publishing)}</div></div>'
        '<p style="margin: 0; padding-top: 14px; border-top: 1px solid #2E2621; font-size: 14px; color: #948980; '
        f'text-wrap: pretty">Rather use a pull request? Upload the same ZIP to <a href="{COMMUNITY_URL}">{COMMUNITY_REPO}'
        '</a> under submissions/ with create_pr=True. It goes through the same audit before it is merged.</p></article>'
    )


AUDIT_STEPS = (
    ("1 · MANIFEST", "Schema version, row count, brief IDs and the SHA-256 of results.jsonl must all match.", False),
    ("2 · SUITE", "The run must be on the frozen suite, checked by its digest. Other suites are refused.", False),
    ("3 · REPLAY", "Every saved proposal is re-verified. Pass flags and rewards in the file are recomputed, not read. "
                   "No model is called.", True),
    ("4 · SCRUB", "Anything that looks like a credential is removed. An archive that carries a key is rejected.", False),
    ("5 · PUBLISH", "Accepted rows are appended to the community dataset and appear on the leaderboard marked "
                    "Community.", False),
)
ROW_MEANS = (
    ("Verified:", "the submitted designs and their outcomes, by replay against the frozen suite and verifier."),
    ("Not verified:", "which model or provider produced them. Names are self-reported."),
    ("Kept apart:", "community rows never enter the canonical leaderboard or the training data."),
    ("Licence:", "Creative Commons Attribution 4.0 International."),
)


def _audit() -> str:
    cards = []
    for i, (title, text, accent) in enumerate(AUDIT_STEPS):
        if i:
            cards.append(ARROW)
        border = "#F4B63F" if accent else "#2E2621"
        cards.append(f'<div style="flex: 1 1 150px; min-width: 0; box-sizing: border-box; padding: 18px; '
                     f'border: 1px solid {border}; border-radius: 6px"><p style="margin: 0 0 6px; {MONO}; font-size: 13px; '
                     f'color: #F4B63F">{title}</p><p style="margin: 0; font-size: 15px; color: #B8ADA1; '
                     f'text-wrap: pretty">{esc(text)}</p></div>')
    return (
        '<section class="cm-audit" style="padding: 104px 0 0">'
        f'<p style="{EYEBROW}">WHAT HAPPENS TO AN ARCHIVE</p>'
        '<h2 class="v2-statement" style="margin: 0 0 28px; font-size: 46px; line-height: 1.02; font-weight: 700; '
        'font-stretch: 114%; text-wrap: balance">Your scores are never trusted. Your proposals are replayed.</h2>'
        f'<div class="cm-steps" style="display: flex; flex-wrap: wrap; align-items: stretch; gap: 10px">{"".join(cards)}'
        '</div></section>'
    )


def _after_run(state: dict) -> str:
    summary = _run_summary(str(state.get("community_status", "") or ""))
    box = ('display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; '
           'min-height: 196px; padding: 24px; box-sizing: border-box; border: 1px dashed #6B625B; border-radius: 6px; '
           'text-align: center')
    if not summary:
        inner = (f'<p style="margin: 0; {MONO}; font-size: 14px; color: #F4B63F">[YOUR RUN]</p>'
                 '<p style="margin: 0; max-width: 380px; font-size: 15px; color: #B8ADA1; text-wrap: pretty">Briefs '
                 'passed, first-try split, cost so far and a link into the replays for each brief. Shown only to you '
                 'until you publish.</p>')
        return f'<div style="{box}">{inner}</div>'
    stat = 'style="margin: 0; font-size: 15px; color: #B8ADA1"'
    return (
        '<div style="display: flex; flex-direction: column; gap: 10px; min-height: 196px; padding: 24px; '
        'box-sizing: border-box; background: #1D1815; border: 1px solid #2E2621; border-radius: 6px">'
        f'<p style="margin: 0; {MONO}; font-size: 12px; letter-spacing: 0.12em; color: #F4B63F">YOUR RUN · '
        f'{esc(summary["run"])}</p>'
        f'<p style="margin: 0"><span style="font-size: 44px; line-height: 1; font-weight: 700; font-stretch: 112%; '
        f'color: #F4B63F">{summary["p3"]} of {summary["n3"]}</span> <span style="font-size: 18px">briefs passed within '
        'three</span></p>'
        f'<p {stat}>{summary["p1"]} of {summary["n1"]} passed on the first try · {summary["count"]} of '
        f'{summary["limit"]} requested briefs completed</p>'
        f'<p {stat}>Cost so far: provider-reported {esc(summary["reported"])}; rate-card estimate '
        f'{esc(summary["estimate"])}.</p>'
        '<p style="margin: 0; font-size: 14px; color: #948980">Shown only to you. To put it on the board, run it '
        'locally, pack it and publish it with <a href="#cm-zip">path B</a>.</p></div>'
    )


def render_submit(state: dict) -> str:
    user = state.get("v2_hf_user") if isinstance(state.get("v2_hf_user"), str) else None
    models = canonical_model_count()
    head = header(None, submit_label=f"Sign out ({user})" if user else "Sign in to publish")
    head = head.replace('<a href="/submit" class="v2-submit"',
                        f'<a href="{LOGOUT_HREF if user else LOGIN_HREF}" class="v2-submit" data-signin-link', 1)
    head = head.replace('"v2-nav"', '"v2-nav-submit"')  # both views share the page, so ids must differ
    means = "".join(
        f'<li style="padding: 12px 0; border-top: 1px solid #2E2621'
        f'{"; border-bottom: 1px solid #2E2621" if i == len(ROW_MEANS) - 1 else ""}"><strong>{term}</strong> '
        f'<span style="color: #B8ADA1">{esc(text)}</span></li>' for i, (term, text) in enumerate(ROW_MEANS))
    body = (
        '<section style="display: flex; flex-wrap: wrap; align-items: flex-end; gap: 20px 64px; padding: 36px 0 36px">'
        '<div style="flex: 1 1 520px; min-width: 0">'
        f'<p style="{EYEBROW}"><a href="/leaderboard" style="color: #B8ADA1">LEADERBOARD</a> / COMMUNITY RUNS / SUBMIT</p>'
        '<h1 style="margin: 0; font-size: 58px; line-height: 0.98; font-weight: 700; font-stretch: 118%; '
        'text-wrap: balance">Put your model on the board.</h1></div>'
        '<p style="flex: 1 1 420px; min-width: 0; margin: 0; font-size: 18px; color: #B8ADA1; text-wrap: pretty">Two ways '
        'to get a row. Either way the checker replays every proposal you send before anything is published, and '
        f'community results stay apart from the canonical {number_word(models)}.</p></section>'
        '<section style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(460px, 100%), 1fr)); gap: 20px; '
        'align-items: start">' + _path_a(state) + _path_b(state, user) + "</section>"
        + _audit()
        + '<section style="padding: 88px 0 0; display: flex; flex-wrap: wrap; gap: 32px 56px; align-items: flex-start">'
          '<div style="flex: 1 1 420px; min-width: 0">'
          '<h2 style="margin: 0 0 14px; font-size: 30px; line-height: 1.1; font-weight: 700; font-stretch: 112%">'
          'What a community row means</h2>'
          f'<ul style="margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; font-size: 16px">'
          f'{means}</ul></div>'
          '<div style="flex: 1 1 420px; min-width: 0">'
          '<h2 style="margin: 0 0 14px; font-size: 30px; line-height: 1.1; font-weight: 700; font-stretch: 112%">'
          'After a private run</h2>' + _after_run(state) + "</div></section>"
    )
    links = [("← Leaderboard", "/leaderboard"), ("Community dataset", COMMUNITY_URL), ("Build with it", "/build")]
    return head + f"<main>{body}</main>" + footer(links, "Submission schema pramaan-community-submission-1", mono=False)


# ------------------------------------------------------------------ page

def render_studio(state: dict) -> str:
    """Both views; the one v1's ``active_tab`` names is shown, the client script follows the URL hash."""
    state = state or {}
    submit = state.get("active_tab") == "leaderboard"
    signed_in = "true" if isinstance(state.get("v2_hf_user"), str) else "false"
    return (
        f'<div class="v2-root v2-studio" data-studio data-view="{"submit" if submit else "try"}" '
        f'data-signed-in="{signed_in}" style="background: #15110F; color: #F4EFE6; font-family: {FONT_SANS}; '
        'font-size: 17px; line-height: 1.5">'
        '<div class="v2-wrap" style="max-width: 1296px; margin: 0 auto; padding: 0 24px; box-sizing: border-box">'
        f'<div data-studio-view="try"{" hidden" if submit else ""}>' + render_try(state) + "<!-- /try --></div>"
        f'<div data-studio-view="submit"{"" if submit else " hidden"}>' + render_submit(state) + "<!-- /submit --></div>"
        "</div></div>"
    )
