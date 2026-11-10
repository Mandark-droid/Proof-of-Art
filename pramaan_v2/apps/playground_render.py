"""Markup for the playground at /web/ (spec 8.9 and 10, ``Playground.dc.html``).

Pure functions: the Gradio builder in ``playground.py`` calls the OpenEnv routes from the browser
and asks :func:`render_view` to draw what came back. Nothing here talks to the environment.
"""
from __future__ import annotations

import base64
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from pramaan.checks.core import all_pass, verify_pattern
from pramaan.dsl import parse_pattern
from pramaan.orchestrate import apply_host_profile
from pramaan.tasks import Task, load_suite

from ..build.replay import symmetry_mismatch
from ..build.site_data import mask_png, tile_png
from ..constants import GRAMMAR_LABELS, STATE_ORDER, number_word
from ..site.data import site_data
from ..site.labels import check_row_label, short_rule
from ..site.render.replay import featured, reward_reason, reward_text
from ..site.render.replays import DEFAULT_TASK, png_src, xy
from ..site.shell import FONT_MONO, FONT_SANS, esc, footer, header, public_base

ROOT = Path(__file__).resolve().parents[2]
SHORT_NAMES = Path(__file__).resolve().parents[1] / "short_names.json"
REFERENCES = ROOT / "benchmarks" / "data" / "eval-v2.1-references.jsonl"
TRAIN_RUNS = ROOT / "benchmarks" / "data" / "train-v1-trajectories.jsonl"
DATASET = "https://huggingface.co/datasets/kshitijthakkar/pramaan-proof-of-art-openenv"
GITHUB = "https://github.com/Mandark-droid/Proof-of-Art"
COLLECTION = "https://huggingface.co/collections/kshitijthakkar/pramaan-proof-of-art"

SUITES = (("eval-v2.1", "held-out"), ("train-v1", "training"))
DEFAULT_SUITE = "eval-v2.1"
UNREADABLE_EXAMPLE = "{ this is intentionally not valid PatternDSL JSON"
FORMAT_CHECKS = frozenset({"candidate_parse", "candidate_truncated"})
MAX_ACTION_CHARS = 20_000
TILE = 64
PIXELS = TILE * TILE
SEED_VALUE = "seed"

MONO = f"font-family: {FONT_MONO}"
EYEBROW = f"{MONO}; font-size: 12px; letter-spacing: 0.12em; color: #948980"
H2 = "margin: 0; font-size: 30px; line-height: 1.1; font-weight: 700; font-stretch: 112%"
FIELD = ("min-height: 44px; padding: 0 12px; border: 1px solid #4A3F38; border-radius: 6px; background: #15110F; "
         "color: #F4EFE6; font-family: inherit; font-size: 16px")
CHIP = ("min-height: 44px; padding: 0 16px; border-radius: 22px; border: 1px solid #4A3F38; background: transparent; "
        "color: #F4EFE6; font-family: inherit; font-size: 15px; font-weight: 600; cursor: pointer")
ROW = "display: flex; gap: 8px; padding: 8px 0; border-top: 1px solid"
SYMMETRY_WORDS = {"D2": "left-right and top-bottom flips", "C4": "quarter turns", "D4": "quarter turns and flips",
                  "C2": "half turns", "R180": "half turns"}
TRANSFORM_WORDS = {"F_vertical": "a mirror image", "F_horizontal": "a mirror image", "R180": "the tile turned 180°",
                   "R90": "the tile turned 90°", "R270": "the tile turned 270°"}
# "One rule broke: <phrase>." for the rules a visitor meets most.
BROKE_PHRASES = {
    "palette_limit": "too many colours", "complexity_bound": "the design is over the size limit",
    "periodic_raster": "the repeat tiles do not match", "layer_registration": "the layers do not line up",
    "exclusive_ink_collision": "two ink layers overlap", "minimum_gap": "motifs sit too close together",
    "coverage_bound": "the ink coverage is out of range", "dot_clearance": "dots sit too close together",
}
KEY_ORDER = ("schema_version", "style", "title", "tile", "palette", "background_index", "layers", "constraints",
             "id", "colour_index", "anchor", "motifs", "kind", "width", "height", "wrap_x", "wrap_y")


# ---------------------------------------------------------------- words and numbers

def number(value: float | None) -> str:
    """Rewards as the board writes them: 0, 0.725, 1.0."""
    if not value:
        return "0"
    return reward_text(float(value))


# ---------------------------------------------------------------- suites and briefs

@lru_cache(maxsize=4)
def _tasks(suite: str) -> dict[str, Task]:
    tasks, _digest = load_suite(suite)
    return {task.id: task for task in tasks}


@lru_cache(maxsize=1)
def _short_names() -> dict[str, str]:
    return json.loads(SHORT_NAMES.read_text(encoding="utf-8"))


def _tradition_name(tradition_id: str) -> str:
    name = _short_names().get(tradition_id)
    if name:
        return name
    from pramaan.traditions import load_registry

    entry = load_registry().get(tradition_id)
    return entry.name if entry is not None else tradition_id


def _level(task: Task) -> str:
    match = re.search(r"-train-(\d+)$", task.id)
    return f"train {match.group(1)}" if task.level == "train" and match else task.level


def suite_label(suite: str) -> str:
    word = dict(SUITES)[suite]
    return f"{suite} · {len(_tasks(suite))} {word} briefs"


@lru_cache(maxsize=4)
def brief_options(suite: str) -> list[tuple[str, str]]:
    """Every brief of a suite in wall order, as (task id, "{State} · {tradition} · {level}")."""
    order = {name: i for i, name in enumerate(STATE_ORDER)}
    levels = {"easy": 0, "hard": 1}

    def key(task: Task):
        match = re.search(r"-(\d+)$", task.id)
        return (order.get(task.state, len(order)), task.state, levels.get(task.level, 2), task.tradition_id,
                int(match.group(1)) if match else 0, task.id)

    tasks = sorted(_tasks(suite).values(), key=key)
    return [(task.id, f"{task.state} · {_tradition_name(task.tradition_id)} · {_level(task)}") for task in tasks]


def suite_of(task_id: str) -> str | None:
    for suite, _ in SUITES:
        if task_id in _tasks(suite):
            return suite
    return None


# ---------------------------------------------------------------- saved examples

def _without_nulls(value):
    if isinstance(value, dict):
        return {k: _without_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_nulls(v) for v in value]
    return value


def _ordered(value):
    if isinstance(value, dict):
        rank = {k: i for i, k in enumerate(KEY_ORDER)}
        keys = sorted(value, key=lambda k: (rank.get(k, len(rank)), k))
        return {k: _ordered(value[k]) for k in keys}
    if isinstance(value, list):
        return [_ordered(v) for v in value]
    return value


def _compact(value) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        return "{ " + ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {_compact(v)}" for k, v in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_compact(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False)


NL = "\n"


def _hug(value: dict, column: int, width: int) -> str:
    """A nested object the way the board writes it: ``{ "a": 1, "b": 2,`` filled to ``width``, continuation
    lines under the first key, and a member that needs several lines on lines of its own."""
    inner = column + 2
    out, pos, multi = "{ ", inner, False
    for i, (key, item) in enumerate(value.items()):
        prefix = f"{json.dumps(key, ensure_ascii=False)}: "
        single = prefix + _compact(item)
        if i == 0 and pos + len(single) + 2 <= width:
            out, pos = out + single, pos + len(single)
            continue
        if i > 0 and not multi and pos + 2 + len(single) + 2 <= width:
            out, pos = out + ", " + single, pos + 2 + len(single)
            continue
        if i > 0:
            out += "," + NL + " " * inner
        if inner + len(single) + 2 <= width:
            out, pos, multi = out + single, inner + len(single), False
            continue
        text = prefix + _flow(item, inner + len(prefix), inner, width)
        out += text
        multi = NL in text
        pos = len(text.rsplit(NL, 1)[-1]) + (0 if multi else inner)
    return out + (NL + " " * column + "}" if multi else " }")


def _flow(value, column: int, indent: int, width: int) -> str:
    """``value`` starting at ``column`` on a line indented by ``indent``."""
    compact = _compact(value)
    if column + len(compact) <= width or not isinstance(value, (dict, list)) or not value:
        return compact
    inner = indent + 2
    if isinstance(value, dict):
        if column > 0:
            return _hug(value, column, width)
        lines = []
        for key, item in value.items():
            prefix = f"{json.dumps(key, ensure_ascii=False)}: "
            lines.append(" " * inner + prefix + _flow(item, inner + len(prefix), inner, width))
        return "{" + NL + ("," + NL).join(lines) + NL + " " * indent + "}"
    parts = [_compact(item) for item in value]
    if not any(isinstance(item, dict) for item in value) and all(column + 1 + len(p) + 2 <= width for p in parts):
        # Fill each line and align continuation lines under the first item.
        lines, line = [], ""
        for part in parts:
            candidate = f"{line}, {part}" if line else part
            if line and column + 1 + len(candidate) + 2 > width:
                lines.append(line + ",")
                line = part
            else:
                line = candidate
        lines.append(line)
        return "[" + (NL + " " * (column + 1)).join(lines) + "]"
    lines = [" " * inner + _flow(item, inner, inner, width) for item in value]
    return "[" + NL + ("," + NL).join(lines) + NL + " " * indent + "]"


def format_pattern(pattern: dict, width: int = 72) -> str:
    """Readable PatternDSL JSON: short objects on one line, long lists wrapped to ``width``."""
    return _flow(_ordered(_without_nulls(pattern)), 0, 0, width)


@lru_cache(maxsize=1)
def _references() -> dict[str, dict]:
    rows = {}
    for line in REFERENCES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("all_hard_constraints_pass") and isinstance(row.get("candidate_pattern"), dict):
                rows[row["task_id"]] = row["candidate_pattern"]
    return rows


@lru_cache(maxsize=1)
def _train_runs() -> dict[str, dict]:
    """Saved training runs by brief: {task_id: {"key", "tries": [{"n", "outcome", "reward", "proposal"}]}}."""
    from pramaan.reward import reward_for_report

    runs: dict[str, dict] = {}
    if not TRAIN_RUNS.is_file():
        return runs
    for line in TRAIN_RUNS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tries = []
        for call in row.get("calls") or []:
            report = call.get("verifier_report") or {}
            candidate = call.get("candidate")
            failed = {c.get("id") for c in report.get("checks") or [] if c.get("status") == "FAIL"}
            if not isinstance(candidate, dict) or failed & FORMAT_CHECKS:
                outcome = "unreadable"
            else:
                outcome = "passed" if report.get("all_hard_constraints_pass") else "broke"
            tries.append({"n": call.get("candidate_index") or len(tries) + 1, "outcome": outcome,
                          "checks": report.get("checks") or [],
                          "reward": reward_for_report(report) if report else 0.0,
                          "proposal": candidate if isinstance(candidate, dict) else None})
        runs.setdefault(row["prompt_id"], {"key": row["model_key"], "tries": tries})
    return runs


def _model_label(key: str) -> str:
    for model in site_data().meta["models"]:
        if model["key"] == key:
            return model["label"]
    return key


def _saved_runs(task_id: str, suite: str) -> list[dict]:
    """Saved runs for a brief, the featured run first, then leaderboard order."""
    if suite == "train-v1":
        run = _train_runs().get(task_id)
        return [{"key": run["key"], "label": _model_label(run["key"]), "tries": run["tries"]}] if run else []
    replay = site_data().replay(task_id)
    if replay is None:
        return []
    runs = [{"key": m["key"], "label": m["label"], "tries": m["tries"]} for m in replay["models"]]
    shown = {key.split("/", 1)[1] for key in featured() if key.split("/", 1)[0] == task_id}
    return sorted(runs, key=lambda run: run["key"] not in shown)


def _first(runs: list[dict], outcome: str) -> tuple[dict, dict] | None:
    return next(((run, t) for run in runs for t in run["tries"] if t["outcome"] == outcome and t.get("proposal")),
                None)


def _story(task_id: str, suite: str) -> tuple[tuple[dict, dict] | None, tuple[dict, dict] | None, dict | None]:
    """((run, failing try), (run, accepted try), the run the episode strip quotes).

    The failure is the first saved try that broke a rule (the featured run first, then leaderboard order).
    The pass prefers the same model's accepted try, so the two chips tell one model's repair.
    """
    runs = _saved_runs(task_id, suite)
    broke = _first(runs, "broke")
    passed = _first([broke[0]], "passed") if broke else None
    passed = passed or _first(runs, "passed")
    quoted = broke[0] if broke else passed[0] if passed else runs[0] if runs else None
    return broke, passed, quoted


def examples(task_id: str, suite: str = DEFAULT_SUITE) -> list[dict]:
    """The example chips for a brief: a saved failure, a saved pass (or the certified reference) and a fixed
    unreadable string, each labelled with where it came from."""
    broke, passed, _quoted = _story(task_id, suite)
    chips = []
    if broke is not None:
        run, failing = broke
        chips.append({"kind": "fail", "label": f"Saved failure · {run['label']}, try {failing['n']}",
                      "pattern": format_pattern(failing["proposal"])})
    if passed is not None:
        run, accepted = passed
        chips.append({"kind": "pass", "label": f"Saved pass · {run['label']}, try {accepted['n']}",
                      "pattern": format_pattern(accepted["proposal"])})
    elif suite == DEFAULT_SUITE and task_id in _references():
        chips.append({"kind": "pass", "label": "Certified reference", "pattern": format_pattern(_references()[task_id])})
    chips.append({"kind": "bad", "label": "Unreadable input", "pattern": UNREADABLE_EXAMPLE})
    return chips


# ---------------------------------------------------------------- the tile a step draws

@lru_cache(maxsize=64)
def _evidence(task_id: str, suite: str, pattern_json: str) -> dict | None:
    task = _tasks(suite).get(task_id)
    if task is None:
        return None
    try:
        candidate = apply_host_profile(json.loads(pattern_json), task)
        pattern = parse_pattern(json.dumps(candidate))
        checks, _masks, grid = verify_pattern(pattern)
    except Exception:  # noqa: BLE001 - an unreadable pattern simply has no tile
        return None
    grid = np.asarray(grid)
    colours = np.array([tuple(bytes.fromhex(value.lstrip("#"))) for value in pattern.palette], dtype=np.uint8)
    rgba = np.concatenate([colours[grid], np.full(grid.shape + (1,), 255, dtype=np.uint8)], axis=2)
    mismatch = None
    if not all_pass(checks) and any(c["status"] == "FAIL" and c["id"].startswith("symmetry_") for c in checks):
        mismatch = symmetry_mismatch(grid, pattern.constraints.symmetry)
    b64 = lambda data: base64.b64encode(data).decode("ascii")  # noqa: E731
    return {"tile_png": b64(tile_png(rgba)),
            "mask_png": b64(mask_png(mismatch)) if mismatch is not None else None,
            "mismatch_px": int(mismatch.sum()) if mismatch is not None else None}


def evidence(task_id: str, suite: str, pattern_json: str) -> dict | None:
    """The clean tile and, for a symmetry failure, the evidence mask, as base64 PNGs."""
    if not isinstance(pattern_json, str) or len(pattern_json) > MAX_ACTION_CHARS:
        return None
    return _evidence(task_id, suite, pattern_json)


# ---------------------------------------------------------------- one step

def _outcome(result: dict) -> str:
    obs = result.get("observation") or {}
    checks = obs.get("checks") or []
    if obs.get("parse_error") or any(c.get("id") in FORMAT_CHECKS and c.get("status") == "FAIL" for c in checks):
        return "unreadable"
    if obs.get("all_hard_constraints_pass") or result.get("reward") == 1.0:
        return "passed"
    return "broke"


def _rules_held(checks: list[dict]) -> tuple[int, int]:
    hard = [c for c in checks if c.get("id") != "schema_valid"]
    return sum(c.get("status") == "PASS" for c in hard), len(hard)


def _failing(checks: list[dict]) -> list[dict]:
    return [c for c in checks if c.get("status") != "PASS"]


def _reason(result: dict) -> str:
    outcome = _outcome(result)
    if outcome == "unreadable":
        return "the pattern could not be read"
    checks = (result.get("observation") or {}).get("checks") or []
    return reward_reason({"outcome": outcome, "rules_held": _rules_held(checks), "reward": result.get("reward") or 0})


def _meter(reward: float) -> str:
    pct = f"{max(0.0, min(1.0, reward or 0.0)) * 100:g}%"
    return (
        '<div aria-hidden="true" style="position: relative; margin-top: 16px; height: 12px; display: flex; gap: 2px">'
        '<span style="flex: 0 0 20%; background: #2E2621"></span>'
        '<span style="flex: 0 0 calc(60% - 4px); background: #4A3F38"></span>'
        '<span style="flex: 1 1 0; background: #F4B63F"></span>'
        f'<span style="position: absolute; top: -5px; left: {pct}; width: 2px; height: 22px; margin-left: -1px; '
        'background: #F4EFE6"></span></div>'
        f'<div class="pg-legend" style="margin-top: 8px; display: flex; justify-content: space-between; gap: 8px; {MONO}; '
        'font-size: 11px; color: #948980"><span>0 UNREADABLE</span><span>0.2 TO 0.8 PARTIAL</span>'
        "<span>1.0 ALL PASS</span></div>")


def _pct(v: int) -> str:
    return f"{(v + 0.5) / TILE * 100:.2f}%"


def _ring(point) -> str:
    return (f'<div style="position: absolute; left: {_pct(point[0])}; top: {_pct(point[1])}; width: 24px; height: 24px; '
            'margin: -12px 0 0 -12px; box-sizing: border-box; border: 2px solid #4FD1C5; border-radius: 50%"></div>')


def _img(b64: str, style: str) -> str:
    return f'<img class="v2-px" src="{png_src(b64)}" alt="" aria-hidden="true" style="display: block; {style}">'


def _pair(check: dict) -> dict:
    example = check.get("counterexample") or check.get("witness") or {}
    return example if isinstance(example, dict) else {}


def _verdict(failing: list[dict]) -> str:
    if not failing:
        return "The checker did not report which rule broke."
    if len(failing) == 1:
        cid = failing[0].get("id", "")
        phrase = "the tile is not symmetric" if cid.startswith("symmetry_") else BROKE_PHRASES.get(
            cid, f"{short_rule(cid)} does not hold")
        return f"One rule broke: {phrase}."
    names = [short_rule(c.get("id", "")) for c in failing]
    return f"{number_word(len(failing)).capitalize()} rules broke: {', '.join(names[:-1])} and {names[-1]}."


def _evidence_line(check: dict) -> str:
    example = _pair(check)
    head = esc(check.get("id", ""))
    if example.get("transform"):
        head += f" · {esc(example['transform'])}"
    if example.get("original") is not None and example.get("transformed") is not None:
        detail = f"{xy(example['original'])} ≠ {xy(example['transformed'])}"
    elif example.get("pixel") is not None:
        detail = xy(example["pixel"])
    else:
        detail = str(check.get("message", "")).split("\n", 1)[0]
    return f"{head}<br>{esc(detail)}"


def _check_rows(checks: list[dict]) -> str:
    rows = []
    for check in checks:
        label = esc(check_row_label(check))
        if check.get("status") == "PASS":
            rows.append(f'<li style="{ROW} #2E2621"><span style="{MONO}; color: #F4B63F">PASS</span>'
                        f"<span>{label}</span></li>")
        else:
            rows.append(f'<li style="{ROW} #E8613C"><span style="{MONO}; color: #E8613C">FAIL</span>'
                        f"<strong>{label}</strong></li>")
    return ('<ul class="pg-checks" style="margin: 0; padding: 0; list-style: none; display: grid; grid-template-columns: '
            'repeat(auto-fit, minmax(min(240px, 100%), 1fr)); gap: 0 24px; font-size: 14px; line-height: 1.35">'
            + "".join(rows) + "</ul>")


def _raw(result: dict) -> str:
    text = json.dumps(result, indent=2, ensure_ascii=False)
    return ('<details style="border: 1px solid #2E2621; border-radius: 6px">'
            '<summary style="min-height: 44px; display: flex; align-items: center; padding: 0 16px; font-weight: 600; '
            'cursor: pointer">Raw observation JSON</summary>'
            f'<pre style="margin: 0; padding: 4px 16px 16px; max-height: 480px; overflow: auto; {MONO}; font-size: 12px; '
            f'line-height: 1.55; color: #F4EFE6">{esc(text)}</pre></details>')


def _tile_box(tile: dict, label: str, outline: str, overlay: str = "") -> str:
    return (f'<div class="pg-tile" role="img" aria-label="{esc(label)}" style="position: relative; flex: 0 1 256px; '
            f'min-width: 200px; aspect-ratio: 1 / 1; outline: 1px solid {outline}">'
            + _img(tile["tile_png"], "width: 100%; height: 100%") + overlay + "</div>")


def _broke(result: dict, tile: dict | None) -> str:
    checks = (result.get("observation") or {}).get("checks") or []
    failing = _failing(checks)
    example = next((_pair(c) for c in failing if _pair(c)), {})
    points = [p for p in (example.get("original"), example.get("transformed"), example.get("pixel"))
              if isinstance(p, (list, tuple)) and len(p) == 2]
    picture = ""
    if tile is not None:
        overlay = ""
        if tile.get("mask_png"):
            overlay += _img(tile["mask_png"], "position: absolute; inset: 0; width: 100%; height: 100%")
        overlay += "".join(_ring(p) for p in points)
        picture = _tile_box(tile, "The tile this pattern draws, with the mismatched pixels marked" if overlay
                            else "The tile this pattern draws", "#4A3F38", overlay)
    transform = example.get("transform")
    if tile is not None and tile.get("mismatch_px") is not None:
        caption = (f"{tile['mismatch_px']:,} of {PIXELS:,} pixels disagree with "
                   f"{TRANSFORM_WORDS.get(transform, 'the required symmetry')}. ")
    else:
        caption = ""
    if len(points) == 2:
        caption += "The pair ringed on the tile is what the model is shown before its next try."
    elif points:
        caption += "The pixel ringed on the tile is what the model is shown before its next try."
    else:
        caption += "This message is what the model is shown before its next try."
    lines = "<br>".join(_evidence_line(c) for c in failing)
    return (
        '<div style="display: flex; flex-wrap: wrap; gap: 20px 24px; align-items: flex-start">'
        f'{picture}<div style="flex: 1 1 240px; min-width: 0">'
        '<p style="margin: 0 0 8px; font-size: 20px; line-height: 1.25; font-weight: 600; text-wrap: pretty">'
        f"{esc(_verdict(failing))}</p>"
        f'<p style="margin: 0 0 10px; {MONO}; font-size: 13px; line-height: 1.5; color: #4FD1C5; overflow-wrap: anywhere">'
        f"{lines}</p>"
        f'<p style="margin: 0; font-size: 15px; color: #B8ADA1; text-wrap: pretty">{esc(caption)}</p></div></div>'
        + _check_rows(checks) + _raw(result))


def _passed(result: dict, tile: dict | None) -> str:
    obs = result.get("observation") or {}
    checks = obs.get("checks") or []
    digest = (obs.get("certificate") or {}).get("mask_sha256", "")
    picture = _tile_box(tile, "The tile this pattern draws", "#F4B63F") if tile is not None else ""
    return (
        '<div style="display: flex; flex-wrap: wrap; gap: 20px 24px; align-items: flex-start">'
        f'{picture}<div style="flex: 1 1 240px; min-width: 0">'
        '<p style="margin: 0 0 8px; font-size: 20px; line-height: 1.25; font-weight: 600; text-wrap: pretty">'
        "Every rule holds. The episode is over.</p>"
        '<p style="margin: 0 0 10px; font-size: 15px; color: #B8ADA1; text-wrap: pretty">'
        f"All {number_word(len(checks))} checks pass on the full 64 × 64 raster. The observation carries a "
        "certificate anyone can recompute.</p>"
        f'<p style="margin: 0; {MONO}; font-size: 12px; line-height: 1.5; color: #B8ADA1; overflow-wrap: anywhere">'
        f"Pixels SHA-256<br>{esc(digest)}</p></div></div>"
        + _check_rows(checks) + _raw(result))


def _unreadable(result: dict) -> str:
    obs = result.get("observation") or {}
    left = max(0, (obs.get("max_attempts") or 3) - (obs.get("attempt") or 0))
    if left:
        remain = f"The step still counts. {number_word(left).capitalize()} remain{'s' if left == 1 else ''}."
    else:
        remain = "The step still counts, and it was the last one."
    failing = _failing(obs.get("checks") or []) or [{"id": "candidate_parse", "status": "FAIL",
                                                      "message": obs.get("parse_error") or ""}]
    check = failing[0]
    text = "\n".join(f"{json.dumps(k)}: {json.dumps(check.get(k), ensure_ascii=False)}"
                     for k in ("id", "status", "message"))
    return (
        '<div style="padding: 20px 24px; border: 1px solid #4A3F38; border-radius: 6px">'
        '<p style="margin: 0 0 8px; font-size: 20px; line-height: 1.25; font-weight: 600; text-wrap: pretty">'
        "The checker could not read this.</p>"
        '<p style="margin: 0 0 12px; font-size: 15px; color: #B8ADA1; text-wrap: pretty">Nothing was drawn, so there is '
        f"no tile and no pixel to point at. {remain}</p>"
        f'<pre style="margin: 0; padding: 12px 14px; overflow-x: auto; background: #1D1815; border-radius: 6px; {MONO}; '
        'font-size: 12px; line-height: 1.55; color: #F4EFE6; white-space: pre-wrap; overflow-wrap: anywhere">'
        f"{esc(text)}</pre></div>" + _raw(result))


def render_idle(checks: int | None = None) -> str:
    what = f"the {number_word(checks)} checks" if checks else "the checks"
    return ('<div style="padding: 28px 24px; border: 1px dashed #4A3F38; border-radius: 6px">'
            '<p style="margin: 0 0 8px; font-size: 20px; font-weight: 600">Nothing sent yet.</p>'
            '<p style="margin: 0; font-size: 15px; color: #B8ADA1; text-wrap: pretty">Reset returned the brief, the '
            f"palette and the rules. Send a pattern to get a reward, {what} and, if one fails, the pixel pair that "
            "breaks it.</p></div>")


def checks_count(task_id: str, suite: str) -> int | None:
    """How many checks a step on this brief returns, from a saved try."""
    for run in _saved_runs(task_id, suite):
        for item in run["tries"]:
            if item["outcome"] != "unreadable" and item.get("checks"):
                return len(item["checks"])
    return None


def _int(value, default: int) -> int:
    """An int from browser-sent state, or the default (never a string spliced into markup)."""
    return value if type(value) is int else default


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def render_observation(steps: list[dict], suite: str, task_id: str = "") -> str:
    """The observation column below its heading: the empty state, or the last step's result."""
    if not steps:
        return render_idle(checks_count(task_id, suite) if task_id else None)
    step = steps[-1]
    result = step.get("result") or {}
    obs = result.get("observation") or {}
    reward = result.get("reward") or 0.0
    outcome = _outcome(result)
    tone = "#F4B63F" if outcome == "passed" else "#F4EFE6"
    done = "true" if result.get("done") else "false"
    head = (
        '<div><div style="display: flex; align-items: baseline; flex-wrap: wrap; gap: 6px 14px">'
        f'<span class="pg-reward" style="font-size: 54px; line-height: 1; font-weight: 700; font-stretch: 112%; '
        f'color: {tone}">{number(reward)}</span>'
        f'<span style="font-size: 15px; color: #B8ADA1">reward · {esc(_reason(result))}</span></div>'
        + _meter(reward) +
        f'<p style="margin: 12px 0 0; {MONO}; font-size: 13px; color: #B8ADA1">attempt {_int(obs.get("attempt"), 0)} of '
        f'{_int(obs.get("max_attempts"), 3)} · done: {done}</p></div>')
    if outcome == "unreadable":
        return head + _unreadable(result)
    task_id = obs.get("task_id") or ""
    tile = evidence(task_id, (obs.get("metadata") or {}).get("suite_id") or suite, step.get("pattern_json") or "")
    return head + (_passed(result, tile) if outcome == "passed" else _broke(result, tile))


# ---------------------------------------------------------------- the episode strip

def _slot_title(result: dict) -> str:
    outcome = _outcome(result)
    reward = number(result.get("reward"))
    if outcome == "passed":
        return f"{reward} · accepted"
    if outcome == "unreadable":
        return f"{reward} · unreadable"
    failing = _failing((result.get("observation") or {}).get("checks") or [])
    what = short_rule(failing[0]["id"]) if len(failing) == 1 else f"{number_word(len(failing))} rules"
    return f"{reward} · {what} broke"


def _steps_left(left: int) -> str:
    return f"{number_word(left).capitalize()} step{'' if left == 1 else 's'} left."


def render_slots(steps: list[dict], max_attempts: int = 3) -> str:
    done = bool(steps and (steps[-1].get("result") or {}).get("done"))
    items = []
    for i in range(max_attempts):
        label = f'<p style="margin: 0 0 6px; {MONO}; font-size: 12px; letter-spacing: 0.1em; color: #948980">STEP {i + 1}</p>'
        if i < len(steps):
            result = steps[i].get("result") or {}
            border = {"passed": "#F4B63F", "broke": "#E8613C"}.get(_outcome(result), "#4A3F38")
            last = i == len(steps) - 1
            sub = "Episode over. Reset to play again." if last and done else _steps_left(max_attempts - i - 1)
            items.append(
                f'<li style="min-width: 0; padding: 16px 18px; background: #1D1815; border: 1px solid {border}; '
                f'border-radius: 6px">{label}<p style="margin: 0; font-size: 22px; font-weight: 700">'
                f'{esc(_slot_title(result))}</p><p style="margin: 4px 0 0; font-size: 14px; color: #B8ADA1">{sub}</p></li>')
        else:
            if done:
                title, sub = "Not needed", f"The episode ended at step {len(steps)}."
            else:
                title = "Not sent"
                sub = _steps_left(max_attempts) if not steps and i == 0 else "Waiting."
            items.append(
                '<li style="min-width: 0; padding: 16px 18px; border: 1px dashed #4A3F38; border-radius: 6px">'
                f'{label}<p style="margin: 0; font-size: 22px; font-weight: 700; color: #948980">{title}</p>'
                f'<p style="margin: 4px 0 0; font-size: 14px; color: #948980">{sub}</p></li>')
    return "".join(items)


def render_saved_line(task_id: str, suite: str = DEFAULT_SUITE) -> str:
    """One sentence on a saved run of the same brief, with a link to its replay."""
    run = _story(task_id, suite)[2]
    if run is None:
        return "No saved run of this brief yet."
    rewards = ", ".join(number(t["reward"]) for t in run["tries"])
    if suite == "train-v1":
        return f"{esc(run['label'])}'s saved training run on this brief went {rewards}. Training briefs have no replay page."
    n = len(run["tries"])
    link = f"Watch all {number_word(n)} tries as a replay." if n > 1 else "Watch it as a replay."
    return (f"{esc(run['label'])}'s saved run on this brief went {rewards}. "
            f'<a href="/replays/{esc(task_id)}">{link}</a>')


# ---------------------------------------------------------------- reset panel and status

def render_reset_panel(reset: dict, seed: int | None = None) -> str:
    obs = reset.get("observation") or {}
    symmetry = (obs.get("constraint_profile") or {}).get("symmetry") or ""
    words = SYMMETRY_WORDS.get(symmetry)
    eyebrow = f"WHAT RESET RETURNED · {obs.get('task_id', '')}" + (f" · seed {seed}" if seed is not None else "")
    facts = (("Grammar", GRAMMAR_LABELS.get(obs.get("grammar"), obs.get("grammar") or "")),
             ("Symmetry", f"{symmetry} · {words}" if words else symmetry or "none"),
             ("Tile", "64 × 64, wraps on both axes"), ("Steps", f"{obs.get('max_attempts', 3)} at most"))
    dl = "".join(f'<div style="display: flex; gap: 8px"><dt style="color: #948980">{k}</dt>'
                 f'<dd style="margin: 0">{esc(v)}</dd></div>' for k, v in facts)
    return (f'<p style="margin: 0 0 8px; {EYEBROW}; overflow-wrap: anywhere">{esc(eyebrow)}</p>'
            f'<p style="margin: 0 0 14px; font-size: 22px; line-height: 1.3; text-wrap: pretty">{esc(obs.get("brief", ""))}</p>'
            f'<dl style="margin: 0; display: flex; flex-wrap: wrap; gap: 8px 28px; font-size: 15px">{dl}</dl>')


def status_line(steps: list[dict]) -> str:
    if not steps:
        return "step 0 of 3 · done: false · best reward 0"
    result = steps[-1].get("result") or {}
    obs = result.get("observation") or {}
    best = max((s.get("result") or {}).get("reward") or 0.0 for s in steps)
    done = "true" if result.get("done") else "false"
    return (f"step {_int(obs.get('attempt'), len(steps))} of {_int(obs.get('max_attempts'), 3)} · done: {done} · "
            f"best reward {number(best)}")


RUN_STYLE = ("min-height: 48px; padding: 0 28px; border: 0; border-radius: 24px; background: #F4B63F; color: #15110F; "
             "font-family: inherit; font-size: 17px; font-weight: 700; cursor: pointer")
HINT_OPEN = "Edit the JSON freely. Every step counts toward the three, readable or not."
HINT_OVER = "The episode is over. Reset to play again."


def render_run_button(disabled: bool) -> str:
    return (f'<button type="button" id="pg-run" class="v2-primary"{" disabled" if disabled else ""} '
            f'style="{RUN_STYLE}">Run one step</button>')


def render_chips(chips: list[dict], pressed: int | None = 0) -> str:
    return "".join(
        f'<button type="button" data-pg-example="{i}" aria-pressed="{"true" if i == pressed else "false"}" '
        f'style="{CHIP}">{esc(chip["label"])}</button>' for i, chip in enumerate(chips))


# ---------------------------------------------------------------- code

def http_snippet(task_id: str, base: str) -> str:
    return ("import requests\n\n"
            f'base = "{base}"\n'
            f'task = "{task_id}"\n'
            'requests.post(f"{base}/web/reset",\n'
            '              json={"task_id": task}).raise_for_status()\n'
            'result = requests.post(f"{base}/web/step", json={\n'
            '    "action": {"pattern_json": open("design.pattern.json").read()}\n'
            "}).json()\n"
            'print("reward:", result["reward"], "done:", result["done"])')


def client_snippet(task_id: str, base: str) -> str:
    return ("from pramaan_env import PramaanEnv, PramaanAction\n\n"
            f'url = "{base}"\n'
            f'task = "{task_id}"\n'
            'pattern_json = open("design.pattern.json").read()\n'
            "with PramaanEnv(base_url=url).sync() as env:\n"
            "    obs = env.reset(task_id=task).observation\n"
            "    result = env.step(PramaanAction(pattern_json=pattern_json))\n"
            "    print(result.reward, result.done)")


# ---------------------------------------------------------------- the whole view

def render_view(state: dict) -> dict:
    """Everything the page redraws after a reset or a step.

    ``state``: ``{"suite", "reset": /web/reset response, "steps": [{"pattern_json", "result"}], "seed"}``.
    The state comes from the browser, so a part of the wrong type falls back to the empty view.
    """
    try:
        return _render_view(state if isinstance(state, dict) else {})
    except (TypeError, ValueError, AttributeError, KeyError):
        return _render_view({})


def _clean_step(step) -> dict | None:
    if not isinstance(step, dict):
        return None
    result = _dict(step.get("result"))
    obs = _dict(result.get("observation"))
    if not isinstance(obs.get("checks", []), list):
        obs = {k: v for k, v in obs.items() if k != "checks"}
    if not isinstance(result.get("reward", 0.0), (int, float)) or isinstance(result.get("reward"), bool):
        result = {k: v for k, v in result.items() if k != "reward"}
    pattern = step.get("pattern_json")
    return {"pattern_json": pattern if isinstance(pattern, str) else "", "result": {**result, "observation": obs}}


def _render_view(state: dict) -> dict:
    reset = _dict(state.get("reset"))
    obs = _dict(reset.get("observation"))
    reset = {**reset, "observation": obs}
    suite = _dict(obs.get("metadata")).get("suite_id") or state.get("suite") or DEFAULT_SUITE
    suite = suite if suite in dict(SUITES) else DEFAULT_SUITE
    task_id = obs.get("task_id") if isinstance(obs.get("task_id"), str) and obs.get("task_id") else DEFAULT_TASK
    # The state comes from the browser: never let it set a loop bound or an amount of work.
    from pramaan_env.server.environment import PramaanEnvironment

    limit = PramaanEnvironment.MAX_ATTEMPTS
    raw_steps = state.get("steps") if isinstance(state.get("steps"), list) else []
    steps = [s for s in (_clean_step(item) for item in raw_steps[:limit]) if s is not None]
    requested = obs.get("max_attempts")
    max_attempts = min(max(requested, 1), limit) if type(requested) is int else limit
    done = bool(steps and (steps[-1].get("result") or {}).get("done"))
    chips = examples(task_id, suite)
    return {
        "task_id": task_id, "suite": suite,
        "reset_panel": render_reset_panel(reset, _int(state.get("seed"), None)),
        "title": "What came back" if steps else "Waiting for a step",
        "observation": render_observation(steps, suite, task_id),
        "status": status_line(steps),
        "done": done, "run_disabled": done, "hint": HINT_OVER if done else HINT_OPEN,
        "slots": render_slots(steps, max_attempts),
        "saved": render_saved_line(task_id, suite),
        "examples": chips, "chips": render_chips(chips),
        "http": http_snippet(task_id, public_base()), "client": client_snippet(task_id, public_base()),
    }


# ---------------------------------------------------------------- page

def _options(items: list[tuple[str, str]], selected: str) -> str:
    return "".join(f'<option value="{esc(value)}"{" selected" if value == selected else ""}>{esc(label)}</option>'
                   for value, label in items)


def _code(title: str, ident: str, text: str) -> str:
    return (f'<div style="min-width: 0"><h3 style="margin: 0 0 8px; {MONO}; font-size: 12px; font-weight: 500; '
            f'letter-spacing: 0.12em; color: #948980">{title}</h3>'
            f'<pre id="{ident}" style="margin: 0; padding: 16px 18px; overflow-x: auto; background: #1D1815; '
            f'border: 1px solid #2E2621; border-radius: 6px; {MONO}; font-size: 13px; line-height: 1.6; '
            f'color: #F4EFE6"><code style="font-family: inherit">{esc(text)}</code></pre></div>')


REWARD_BANDS = (
    ("0", "#B8ADA1", "Unreadable. Not valid PatternDSL JSON."),
    ("0.2 to 0.8", "#B8ADA1", "Readable, but a rule broke. 0.2 plus 0.6 times the share of rules that hold."),
    ("1.0", "#F4B63F", "Every rule holds. A certificate comes back and the episode ends."),
)
LINK = "display: inline-flex; align-items: center; min-height: 44px"


def render_page(task_id: str = DEFAULT_TASK, suite: str = DEFAULT_SUITE) -> str:
    """The page as first drawn: the default brief reset, nothing sent. The script then resets the real episode."""
    from pramaan_env.server.environment import PramaanEnvironment

    max_attempts = PramaanEnvironment.MAX_ATTEMPTS
    task = _tasks(suite)[task_id]
    reset = {"observation": {"task_id": task.id, "brief": task.brief, "grammar": task.grammar,
                             "constraint_profile": {"symmetry": task.symmetry}, "max_attempts": max_attempts,
                             "metadata": {"suite_id": suite}}}
    view = render_view({"suite": suite, "reset": reset, "steps": []})
    chips = view["examples"]
    concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "4"))
    meta = site_data().meta

    hero = (
        '<section style="display: flex; flex-wrap: wrap; align-items: flex-end; gap: 20px 64px; padding: 36px 0 32px">'
        '<div style="flex: 1 1 560px; min-width: 0">'
        f'<p style="margin: 0 0 12px; {EYEBROW}"><a href="/build" style="color: #948980">BUILD WITH IT</a> / '
        "PLAYGROUND · OPENENV · /web</p>"
        '<h1 style="margin: 0; font-size: 58px; line-height: 0.98; font-weight: 700; font-stretch: 118%; '
        'text-wrap: balance">Play one episode yourself.</h1></div>'
        '<p style="flex: 1 1 420px; min-width: 0; margin: 0; font-size: 18px; color: #B8ADA1; text-wrap: pretty">This is '
        "the environment a training loop talks to. Send a pattern. It returns a reward from 0 to 1 and the exact pixel "
        f"that broke a rule. {number_word(max_attempts).capitalize()} steps at most, then the episode ends.</p></section>")

    bar = (
        '<section aria-label="Episode" class="pg-bar" style="display: flex; flex-wrap: wrap; align-items: flex-end; '
        'gap: 16px 20px; padding: 20px 24px; background: #1D1815; border: 1px solid #2E2621; border-radius: 6px">'
        '<div style="display: flex; flex-direction: column; gap: 4px; flex: 0 1 240px; min-width: 0">'
        '<label for="pg-suite" style="font-size: 13px; color: #B8ADA1">Suite</label>'
        f'<select id="pg-suite" style="{FIELD}">{_options([(s, suite_label(s)) for s, _ in SUITES], suite)}</select></div>'
        '<div style="display: flex; flex-direction: column; gap: 4px; flex: 1 1 380px; min-width: 0">'
        '<label for="pg-brief" style="font-size: 13px; color: #B8ADA1">Brief</label>'
        f'<select id="pg-brief" style="{FIELD}">'
        f'{_options(brief_options(suite) + [(SEED_VALUE, "Random brief from a seed")], task_id)}</select></div>'
        '<button type="button" id="pg-reset" style="min-height: 44px; padding: 0 20px; border: 1px solid #F4EFE6; '
        'border-radius: 22px; background: transparent; color: #F4EFE6; font-family: inherit; font-size: 16px; '
        'font-weight: 600; cursor: pointer">Reset episode</button>'
        f'<p role="status" id="pg-status" style="margin: 0 0 0 auto; padding: 10px 0; {MONO}; font-size: 13px; '
        f'color: #B8ADA1">{esc(view["status"])}</p></section>')

    bands = "".join(f'<dt style="{MONO}; color: {colour}">{esc(v)}</dt><dd style="margin: 0">{esc(text)}</dd>'
                    for v, colour, text in REWARD_BANDS)
    returned = (
        '<section style="display: flex; flex-wrap: wrap; gap: 20px; padding: 20px 0 0">'
        '<div id="pg-reset-panel" style="flex: 3 1 520px; min-width: 0; padding: 20px 24px; border: 1px solid #2E2621; '
        f'border-radius: 6px">{view["reset_panel"]}</div>'
        '<div style="flex: 2 1 380px; min-width: 0; padding: 20px 24px; border: 1px solid #2E2621; border-radius: 6px">'
        f'<p style="margin: 0 0 12px; {EYEBROW}">HOW THE REWARD IS SET</p>'
        '<dl style="margin: 0; display: grid; grid-template-columns: 96px minmax(0, 1fr); gap: 8px 14px; font-size: 15px; '
        f'line-height: 1.35">{bands}</dl></div></section>')

    action = (
        '<div class="pg-action" style="flex: 1 1 520px; min-width: 0; display: flex; flex-direction: column; gap: 14px">'
        f'<div><p style="margin: 0 0 6px; {MONO}; font-size: 12px; letter-spacing: 0.12em; color: #F4B63F">ACTION</p>'
        f'<h2 style="{H2}">Send a pattern</h2></div>'
        '<div role="group" aria-label="Saved examples" id="pg-chips" style="display: flex; flex-wrap: wrap; gap: 8px">'
        f'{view["chips"]}</div>'
        f'<label for="pg-pattern" style="margin: 0; {MONO}; font-size: 13px; color: #B8ADA1">pattern_json · up to '
        f"{MAX_ACTION_CHARS:,} characters</label>"
        f'<textarea id="pg-pattern" spellcheck="false" autocomplete="off" maxlength="{MAX_ACTION_CHARS}" '
        'style="display: block; width: 100%; margin: 0; min-height: 560px; padding: 16px; box-sizing: border-box; '
        'resize: vertical; overflow: auto; border: 1px solid #4A3F38; border-radius: 6px; background: #1D1815; '
        f'color: #F4EFE6; {MONO}; font-size: 13px; line-height: 1.55; white-space: pre; tab-size: 2">'
        f'{esc(chips[0]["pattern"])}</textarea>'
        '<div class="pg-runrow" style="display: flex; flex-wrap: wrap; align-items: center; gap: 12px 20px">'
        f'{render_run_button(False)}<span id="pg-hint" style="font-size: 14px; color: #948980">{HINT_OPEN}</span></div>'
        "</div>")

    observation = (
        '<div class="pg-observe" aria-live="polite" style="flex: 1 1 520px; min-width: 0; display: flex; '
        'flex-direction: column; gap: 18px">'
        f'<div><p style="margin: 0 0 6px; {MONO}; font-size: 12px; letter-spacing: 0.12em; color: #F4B63F">OBSERVATION</p>'
        f'<h2 id="pg-obs-title" style="{H2}">{view["title"]}</h2></div>'
        f'<div id="pg-obs" style="display: flex; flex-direction: column; gap: 18px">{view["observation"]}</div></div>')

    work = ('<section class="pg-work" style="display: flex; flex-wrap: wrap; gap: 40px 48px; align-items: flex-start; '
            f'padding: 48px 0 0">{action}{observation}</section>')

    episode = (
        '<section style="padding: 72px 0 0">'
        f'<h2 style="{H2.replace("margin: 0", "margin: 0 0 16px")}">This episode</h2>'
        '<ol id="pg-slots" style="margin: 0; padding: 0; list-style: none; display: grid; grid-template-columns: '
        f'repeat(auto-fit, minmax(min(280px, 100%), 1fr)); gap: 12px">{view["slots"]}</ol>'
        '<p id="pg-saved" style="margin: 14px 0 0; max-width: 900px; font-size: 15px; color: #B8ADA1; '
        f'text-wrap: pretty">{view["saved"]}</p></section>')

    facts = (
        (str(max_attempts), "steps per episode. It ends early on reward 1.0."),
        (f"{MAX_ACTION_CHARS:,}", "characters per action."),
        (str(concurrent), "episodes at once on the hosted Space."),
        ("0", "model calls. The checker is plain arithmetic on a CPU."),
    )
    fact_items = "".join(
        f'<div><dt style="font-size: 30px; line-height: 1; font-weight: 700; font-stretch: 112%">{esc(n)}</dt>'
        f'<dd style="margin: 6px 0 0; color: #B8ADA1">{esc(text)}</dd></div>' for n, text in facts)
    code = (
        '<section style="padding: 72px 0 0">'
        f'<h2 style="{H2.replace("margin: 0", "margin: 0 0 8px")}">The same episode from code</h2>'
        '<p style="margin: 0 0 20px; max-width: 900px; font-size: 15px; color: #B8ADA1; text-wrap: pretty">This page '
        "calls two routes. Your code can call the same two, or use the typed client.</p>"
        '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(480px, 100%), 1fr)); gap: 20px">'
        + _code("PLAIN HTTP", "pg-http", view["http"]) + _code("TYPED CLIENT", "pg-client", view["client"]) +
        "</div>"
        '<dl style="margin: 24px 0 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(min(220px, 100%), '
        f'1fr)); gap: 16px 32px; font-size: 15px">{fact_items}</dl>'
        '<p style="margin: 28px 0 0; display: flex; flex-wrap: wrap; gap: 8px 24px; font-size: 16px">'
        f'<a href="/build#contract" style="{LINK}">Action and observation fields</a>'
        f'<a href="/build" style="{LINK}">Use it as a training reward</a>'
        f'<a href="{DATASET}" style="{LINK}">Saved reset and step records</a></p></section>')

    foot = footer([("Code and benchmark protocol", GITHUB), ("Everything in one place", COLLECTION),
                   ("Replays", "/replays")],
                  f"Validator {meta['validator']} · reward {meta['reward']}", mono=False, margin_top=88, border=True)
    return (
        f'<div class="v2-root pg-root" data-pg style="background: #15110F; color: #F4EFE6; font-family: {FONT_SANS}; '
        'font-size: 17px; line-height: 1.5">'
        '<div class="v2-wrap" style="max-width: 1296px; margin: 0 auto; padding: 0 24px; box-sizing: border-box">'
        f'{header("build")}<div class="pg-body">{hero}{bar}{returned}{work}{episode}{code}</div>{foot}</div></div>')
