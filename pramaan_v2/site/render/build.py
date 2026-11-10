"""Build with it: /build (spec 8.8, Build.dc.html).

Counts come from the site data; the environment contract is generated from the pydantic models;
every command and snippet runs as written in the repository.
"""
from __future__ import annotations

import json
import types
import typing
from functools import lru_cache
from pathlib import Path

from ...constants import number_word
from ..data import SiteData
from ..shell import FONT_MONO, esc, page

ROOT = Path(__file__).resolve().parents[3]
REPO = "https://github.com/Mandark-droid/Proof-of-Art"
COLLECTION = "https://huggingface.co/collections/kshitijthakkar/pramaan-proof-of-art"
HF = "https://huggingface.co/datasets/"
TRAIN_REPO = "kshitijthakkar/pramaan-proof-of-art-train-v1"
OPENENV_REPO = "kshitijthakkar/pramaan-proof-of-art-openenv"
EPISODE_TASK, EPISODE_MODEL = "tn_kolam-easy-22", "beam-hf"
CONTRACT_TASK = "hp_kinnauri-shawl-geometric-border-easy-8"
# train-v1 export sizes as published (benchmarks/data/train-v1-*.jsonl; the files do not ship with the Space).
TRAIN_V1 = {"sft": 339, "repair": 195, "preference": 50, "trajectories": 144}
LOOP_COLOURS = ("#F4EFE6", "#F4B63F", "#4FD1C5", "#F0704A", "#B79CF2")

MONO = f"font-family: {FONT_MONO}"
EYEBROW = f"margin: 0 0 12px; {MONO}; font-size: 12px; letter-spacing: 0.12em; color: #948980"
CARD = ("display: flex; flex-direction: column; gap: 14px; min-width: 0; padding: 24px; background: #1D1815; "
        "border: 1px solid #2E2621; border-radius: 6px")
CARD_EYEBROW = f"margin: 0; {MONO}; font-size: 12px; letter-spacing: 0.1em; color: #F4B63F"
CARD_H2 = "margin: 0; font-size: 26px; line-height: 1.12; font-weight: 700"
PRE = (f"margin: 0; padding: 14px 16px; overflow-x: auto; background: #15110F; border: 1px solid #2E2621; "
       f"border-radius: 6px; {MONO}; font-size: 12px; line-height: 1.6; color: #F4EFE6")
STATEMENT = ("margin: 0 0 28px; font-size: 46px; line-height: 1.02; font-weight: 700; font-stretch: 114%; "
             "text-wrap: balance")

# Board copy for each contract field; any field without one falls back to its pydantic description.
FIELD_TEXT = {
    "pattern_json": "The whole pattern as one JSON string. This is the only thing the agent controls.",
    "action.metadata": "Optional notes carried with the action. Never read by the checker.",
    "brief": "The art brief for this task.",
    "palette": "The colours the host allows.",
    "constraint_profile": "Symmetry, periods, loop counts and the other host rules.",
    "prompt_messages": "Messages ready to send to a chat model, including repair feedback after a miss.",
    "attempt": "Which try this is.",
    "max_attempts": "How many tries the episode allows: {max_attempts}.",
    "checks": "Every check with its status, message and counterexample.",
    "all_hard_constraints_pass": "True only when every rule holds.",
    "reward": "0, between 0.2 and 0.8, or 1.0.",
    "done": "The episode is over: passed, or out of tries.",
    "certificate": "The replayable proof, on a full pass.",
    "parse_error": "Why an unreadable action was rejected.",
    "task_id": "Which brief this is.",
    "state": "The state or Union Territory the brief comes from.",
    "tradition_id": "The registry tradition behind the brief.",
    "grammar": "The craft grammar the brief is checked under.",
    "observation.metadata": "Carries the suite_id of the brief.",
}
ACTION_ORDER = ("pattern_json", "metadata")
OBSERVATION_ORDER = ("brief", "palette", "constraint_profile", "prompt_messages", "attempt", "max_attempts", "checks",
                     "all_hard_constraints_pass", "reward", "done", "certificate", "parse_error", "task_id", "state",
                     "tradition_id", "grammar", "metadata")

GRAMMAR_RULES = {
    "counted_grid": ("Every mark fills whole cells of the host's thread grid.", "cell_quantization"),
    "stripe_band": ("The band sequence repeats exactly at the stated period.", "translation_period"),
    "periodic_raster": ("The tile meets itself at every edge and holds its mirror symmetry.",
                        "periodic_raster, symmetry_d2"),
    "jaali_lattice": ("The pierced screen stays one connected piece, with holes inside the frame and open area in "
                      "range.", "jaali_single_solid and three more"),
    "dot_field": ("No two dots sit closer than the minimum distance, across tile edges too.", "dot_clearance"),
    "rotational_field": ("The tile is unchanged by the required turns about its centre.", "symmetry_c4"),
    "kolam_loop": ("The traced line closes into exactly the required number of loops.",
                   "kolam_loop_count and two more"),
}


# Data ------------------------------------------------------------------------------------------

def _count_lines(path: Path) -> int | None:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return None


@lru_cache(maxsize=1)
def _references() -> int | None:
    return _count_lines(ROOT / "benchmarks" / "data" / "eval-v2.1-references.jsonl")


@lru_cache(maxsize=1)
def _train_task() -> str:
    ids = []
    try:
        for line in (ROOT / "benchmarks" / "suites" / "train-v1.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.append(json.loads(line)["id"])
    except (OSError, ValueError, KeyError):
        return "tn_kolam-train-1"
    return next((i for i in ids if i.startswith("tn_kolam-train")), ids[0] if ids else "tn_kolam-train-1")


def _type_name(annotation) -> str:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        return " | ".join(_type_name(arg) for arg in typing.get_args(annotation))
    if annotation is type(None):
        return "None"
    if origin is not None:
        return getattr(origin, "__name__", str(origin))
    return getattr(annotation, "__name__", str(annotation))


def contract_rows(model: type, order: tuple[str, ...], scope: str) -> list[tuple[str, str, str]]:
    """One row per pydantic field: name, type, plain description."""
    from pramaan_env.server.environment import PramaanEnvironment

    fields = model.model_fields
    names = [n for n in order if n in fields] + [n for n in fields if n not in order]
    rows = []
    for name in names:
        field = fields[name]
        text = FIELD_TEXT.get(f"{scope}.{name}") or FIELD_TEXT.get(name) or (field.description or "").rstrip(".") + "."
        rows.append((name, _type_name(field.annotation),
                     text.format(max_attempts=number_word(PramaanEnvironment.MAX_ATTEMPTS))))
    return rows


def _models_passing(tries: list[dict]) -> bool:
    return any(t["outcome"] == "passed" for t in tries)


def _impossible_task(data: SiteData) -> str | None:
    for task in data.kolam:
        replay = data.replay(task)
        if replay is not None and replay.get("reference") is None:
            return task
    return None


def _blank_task(data: SiteData) -> str | None:
    counts: dict[str, int] = {}
    for run in data.wall["runs"]:
        if run.get("low_ink"):
            counts[run["task"]] = counts.get(run["task"], 0) + 1
    return max(counts, key=lambda t: (counts[t], t)) if counts else None


def _round2(value: float) -> str:
    if value >= 1:
        return "1.0"
    return f"{int(value * 100 + 0.5 + 1e-9) / 100:.2f}"


# Markup ----------------------------------------------------------------------------------------

def _dots(rows: int, cols: int, unit: float) -> str:
    return "".join(f"M{(2 * c + 1.5) * unit:g} {(2 * r + 1.5) * unit:g}h.01" for r in range(rows) for c in range(cols))


def _kolam_svg(rows: int, cols: int, unit: float, paths: list[str], label: str) -> str:
    width, height = (2 * cols + 1) * unit, (2 * rows + 1) * unit
    loops = "".join(
        f'<path d="{esc(d)}" stroke="{LOOP_COLOURS[i % len(LOOP_COLOURS)]}" stroke-width="{unit * 0.17:g}" '
        'stroke-linejoin="round" fill="none"></path>' for i, d in enumerate(paths))
    return (f'<svg viewBox="0 0 {width:g} {height:g}" role="img" aria-label="{esc(label)}" '
            f'style="display: block; width: 100%; height: auto">{loops}'
            f'<path d="{_dots(rows, cols, unit)}" stroke="#F4EFE6" stroke-width="{unit * 0.325:g}" '
            'stroke-linecap="round" fill="none"></path></svg>')


def _pre(code: str, label: str) -> str:
    return (f'<div class="bd-code"><pre style="{PRE}" aria-label="{esc(label)}"><code style="font-family: inherit">{esc(code)}</code></pre>'
            '</div>')


def _cards(data: SiteData) -> str:
    totals = data.meta["totals"]
    score_code = (
        "uv run --project . python benchmarks/run.py \\\n"
        f"  --suite benchmarks/suites/{data.meta['suite_id']}.jsonl \\\n"
        "  --models benchmarks/models.json \\\n"
        "  --model my-lab/new-model \\\n"
        "  --output benchmarks/results/my-run.jsonl\n\n"
        "uv run --project . python -m \\\n"
        "  scripts.pack_submission \\\n"
        "  benchmarks/results/my-run.jsonl \\\n"
        "  benchmarks/results/my-run.zip")
    reward_code = (
        "from pramaan_env.reward_fn import (\n"
        "    pramaan_reward,\n"
        ")\n\n"
        "rewards = pramaan_reward(\n"
        '    ["{}"],  # completions\n'
        f'    task_id=["{_train_task()}"],\n'
        ")\n"
        "print(rewards)  # [0.0]: unreadable\n\n"
        "# In TRL:\n"
        "# GRPOTrainer(\n"
        "#     reward_funcs=[pramaan_reward], ...)")
    refs = _references()
    refs_text = f", plus {refs} certified references" if refs else ""
    datasets = (
        ("Evaluation results", HF + data.meta["dataset"],
         f"{totals['runs']} model and brief records with full trajectories{refs_text}."),
        ("Training data", HF + TRAIN_REPO,
         f"{TRAIN_V1['sft']} supervised rows, {TRAIN_V1['repair']} repairs, {TRAIN_V1['preference']} preference "
         f"pairs, {TRAIN_V1['trajectories']} trajectories."),
        ("OpenEnv transitions", HF + OPENENV_REPO,
         "Reset and step records with actions, verifier evidence and rewards, in separate training and evaluation "
         "splits."),
    )
    items = "".join(
        f'<li style="padding: 12px 0; border-top: 1px solid #2E2621{"; border-bottom: 1px solid #2E2621" if i == 2 else ""}">'
        f'<a href="{esc(href)}" style="font-weight: 600">{esc(name)}</a><br>'
        f'<span style="color: #B8ADA1">{esc(text)}</span></li>' for i, (name, href, text) in enumerate(datasets))
    return (
        '<section class="bd-cards" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(360px, 100%), 1fr)); '
        'gap: 20px">'
        f'<article id="score" style="{CARD}">'
        f'<p style="{CARD_EYEBROW}">01 · SCORE A MODEL</p>'
        f'<h2 style="{CARD_H2}">Run the frozen {totals["briefs"]} briefs with your own key.</h2>'
        '<p style="margin: 0; font-size: 15px; color: #B8ADA1; text-wrap: pretty">Add one entry to '
        "benchmarks/models.json, run the suite, pack the result. The packer writes a SHA-256 manifest and refuses "
        "mismatched suites or anything that carries a credential.</p>"
        + _pre(score_code, "Commands to score a model") +
        '<p style="margin: auto 0 0; font-size: 14px; color: #948980"><a href="/submit">Upload the archive</a>, or run '
        "a model privately on the Space with your own key. Every row is replayed before it is published.</p></article>"
        f'<article id="train" style="{CARD}">'
        f'<p style="{CARD_EYEBROW}">02 · TRAIN A POLICY</p>'
        f'<h2 style="{CARD_H2}">Use the verifier as a reward function.</h2>'
        '<p style="margin: 0; font-size: 15px; color: #B8ADA1; text-wrap: pretty">The adapter resets a separate '
        "train-v1 episode for each completion and returns one reward per completion. It is tested on saved passing "
        "designs, failing designs and unreadable input.</p>"
        + _pre(reward_code, "The reward function") +
        f'<p style="margin: auto 0 0; font-size: 14px; color: #948980">Train on train-v1 only. The {totals["briefs"]} '
        "evaluation briefs are held out.</p></article>"
        f'<article id="data" style="{CARD}">'
        f'<p style="{CARD_EYEBROW}">03 · READ THE DATA</p>'
        f'<h2 style="{CARD_H2}">Every row behind this site is on the Hub.</h2>'
        '<ul style="margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; font-size: 15px">'
        f"{items}</ul>"
        '<p style="margin: auto 0 0; font-size: 14px; color: #948980">Third-party model outputs are evaluation data '
        "only and are not training-eligible.</p></article></section>")


def _episode(data: SiteData) -> str:
    arrow = ('<svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true" class="bd-arrow" '
             'style="flex: 0 0 28px; align-self: center"><path d="M5 14h17M16 8l6 6-6 6" stroke="#948980" '
             'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"></path></svg>')
    step = ('<div style="flex: 1 1 260px; min-width: 0; padding: 20px; border: 1px solid #2E2621; border-radius: 6px">'
            f'<p style="margin: 0 0 8px; {MONO}; font-size: 14px; color: #F4B63F">{{code}}</p>'
            '<p style="margin: 0; font-size: 15px; color: #B8ADA1; text-wrap: pretty">{text}</p></div>')
    steps = arrow.join((
        step.format(code="env.reset(task_id)", text="You receive the brief, the palette, the constraint profile and "
                    "the prompt messages. The host owns palette and constraints; the agent cannot weaken them."),
        step.format(code="env.step(pattern_json)", text="The action is one JSON pattern. It is parsed as data and "
                    "drawn on a 64 × 64 raster. Model-written code, HTML or SVG is never executed or rendered."),
        step.format(code="result.observation, result.reward", text="Every check with its status, the first "
                    "counterexample for each failure, a reward and a done flag. On a full pass you also get a "
                    "certificate you can replay."),
    ))

    brief = data.kolam.get(EPISODE_TASK) or {}
    tries = (brief.get("models") or {}).get(EPISODE_MODEL) or []
    label = next((m["label"] for m in data.models if m["key"] == EPISODE_MODEL), EPISODE_MODEL)
    state = next((s["name"] for s in data.meta["states"] if EPISODE_TASK in (s["easy"], s["hard"])), "")
    required = brief.get("required_loops", 1)
    unit = brief.get("unit", 40.0)
    figures = []
    for t in tries:
        rows, cols = t.get("rows") or brief.get("rows", 3), t.get("cols") or brief.get("cols", 3)
        loops = t.get("loops") or 0
        passed = t["outcome"] == "passed"
        dots = number_word(rows * cols)
        if passed:
            alt = f"One unbroken line around all {dots} dots"
        elif loops == 0:
            alt = f"{dots.capitalize()} dots and no line"
        else:
            alt = f"{number_word(loops).capitalize()} separate loop{'s' if loops != 1 else ''}"
        alt += f", from {t.get('mirrors')} mirrors"
        caption = f"step {t['n']} · {loops} loop{'' if loops == 1 else 's'}" + (" · done" if passed else "")
        reward_colour = "#F4B63F" if passed else "#F4EFE6"
        figures.append(
            f'<figure style="margin: 0; padding: 14px; background: #1D1815; border: 1px solid '
            f'{"#F4B63F" if passed else "#2E2621"}; border-radius: 6px">'
            + _kolam_svg(rows, cols, unit, t.get("paths") or [], alt) +
            f'<figcaption style="margin-top: 10px; {MONO}; font-size: 12px; line-height: 1.5; color: #B8ADA1">'
            f'{esc(caption)}<br><span style="color: {reward_colour}">reward {_round2(t["reward"])}</span>'
            "</figcaption></figure>")
    loops_word = number_word(required)
    intro = (f"A saved episode: {esc(label)} on the {esc(state)} kolam brief, {loops_word} closed "
             f"loop{'' if required == 1 else 's'} required.")
    bands = (
        '<div style="flex: 2 1 380px; min-width: 0">'
        '<h3 style="margin: 0 0 14px; font-size: 22px; font-weight: 700">Reward bands</h3>'
        '<div aria-hidden="true" style="display: flex; gap: 2px; height: 14px"><span style="flex: 0 0 20%; '
        'background: #2E2621"></span><span style="flex: 0 0 calc(60% - 4px); background: #4A3F38"></span>'
        '<span style="flex: 1 1 0; background: #F4B63F"></span></div>'
        '<dl style="margin: 16px 0 0; display: grid; grid-template-columns: 96px minmax(0, 1fr); gap: 10px 16px; '
        'font-size: 15px">'
        f'<dt style="{MONO}; color: #F4EFE6">0</dt><dd style="margin: 0; color: #B8ADA1">Unreadable, cut off or '
        "rejected input, including a report with missing checks.</dd>"
        f'<dt style="{MONO}; color: #F4EFE6">0.2 to 0.8</dt><dd style="margin: 0; color: #B8ADA1">A readable pattern '
        "that holds some rules: 0.2 plus 0.6 times the share of rules that pass.</dd>"
        f'<dt style="{MONO}; color: #F4B63F">1.0</dt><dd style="margin: 0; color: #B8ADA1">Every rule holds. The '
        "episode ends and returns a certificate.</dd></dl>"
        '<p style="margin: 16px 0 0; font-size: 14px; color: #948980; text-wrap: pretty">Reward measures declared '
        "geometry. It does not test visible ink, beauty or cultural authenticity.</p></div>")
    return (
        '<section id="episode" style="padding: 104px 0 0">'
        f'<p style="{EYEBROW}">THE ENVIRONMENT · ONE EPISODE</p>'
        f'<h2 class="v2-statement" style="{STATEMENT}">Three tries. Each one comes back with evidence.</h2>'
        f'<div class="bd-steps" style="display: flex; flex-wrap: wrap; align-items: stretch; gap: 12px">{steps}</div>'
        '<div style="display: flex; flex-wrap: wrap; gap: 28px 56px; align-items: flex-start; margin-top: 36px">'
        '<div style="flex: 3 1 560px; min-width: 0">'
        f'<p style="margin: 0 0 12px; font-size: 15px; color: #948980">{intro}</p>'
        '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(150px, 100%), 1fr)); gap: 12px">'
        + "".join(figures) + f'</div></div>{bands}</div></section>')


def _table(rows: list[tuple[str, str, str]], model: str, *, name_w: int, type_w: int, pad: int,
           min_width: str = "") -> str:
    body = []
    for i, (name, kind, text) in enumerate(rows):
        border = "border-top: 1px solid #2E2621" + ("; border-bottom: 1px solid #2E2621" if i == len(rows) - 1 else "")
        w1 = f"width: {name_w}px; " if i == 0 else ""
        w2 = f"width: {type_w}px; " if i == 0 else ""
        body.append(
            f'<tr style="{border}"><th scope="row" style="padding: {pad}px 16px {pad}px 0; {w1}{MONO}; font-size: 13px; '
            f'font-weight: 500; color: #F4B63F; white-space: nowrap">{esc(name)}</th>'
            f'<td style="padding: {pad}px 16px; {w2}{MONO}; font-size: 13px; color: #948980">{esc(kind)}</td>'
            f'<td style="padding: {pad}px 0 {pad}px 16px; color: #B8ADA1">{esc(text)}</td></tr>')
    return (f'<div style="overflow-x: auto"><table data-model="{model}" aria-label="{model} fields" '
            f'style="width: 100%; {min_width}border-collapse: collapse; font-size: 15px; text-align: left">'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _contract(base_url: str) -> str:
    from pramaan_env.models import PramaanAction, PramaanObservation

    action = contract_rows(PramaanAction, ACTION_ORDER, "action")
    observation = contract_rows(PramaanObservation, OBSERVATION_ORDER, "observation")
    client = (
        "from pramaan_env import PramaanEnv, PramaanAction\n\n"
        f'url = "{base_url.rstrip("/")}"\n'
        f'task = "{CONTRACT_TASK}"\n'
        "with PramaanEnv(base_url=url).sync() as env:\n"
        "    obs = env.reset(task_id=task).observation\n"
        '    pattern_json = "{}"  # your proposal for obs.brief\n'
        "    result = env.step(PramaanAction(pattern_json=pattern_json))\n"
        "    print(result.reward, result.done)")
    client_pre = (f'<div class="bd-code"><pre style="{PRE.replace("background: #15110F", "background: #1D1815").replace("padding: 14px 16px", "padding: 16px 18px").replace("font-size: 12px", "font-size: 13px")}" '
                  f'aria-label="Typed client"><code style="font-family: inherit">{esc(client)}</code></pre>'
                  '</div>')
    return (
        '<section id="contract" style="padding: 104px 0 0">'
        f'<p style="{EYEBROW}">ENVIRONMENT CONTRACT · OPENENV</p>'
        f'<h2 class="v2-statement" style="{STATEMENT}">What an agent sends, and what comes back.</h2>'
        '<div style="display: flex; flex-wrap: wrap; gap: 32px 56px; align-items: flex-start">'
        '<div style="flex: 1 1 440px; min-width: 0; display: flex; flex-direction: column; gap: 18px">'
        '<h3 style="margin: 0; font-size: 22px; font-weight: 700">Action · PramaanAction</h3>'
        + _table(action, "PramaanAction", name_w=150, type_w=60, pad=11) +
        '<h3 style="margin: 10px 0 0; font-size: 22px; font-weight: 700">Use the hosted environment</h3>'
        + client_pre +
        '<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 12px 20px">'
        '<a href="/playground" style="display: inline-flex; align-items: center; min-height: 48px; padding: 0 24px; '
        'border-radius: 24px; background: #F4B63F; color: #15110F; font-weight: 700; text-decoration: none">'
        "Open the interactive environment</a>"
        '<a href="/try" style="display: inline-flex; align-items: center; min-height: 44px; font-weight: 600">'
        "Or try it in the studio →</a></div>"
        '<p style="margin: 0; font-size: 14px; color: #948980; text-wrap: pretty">The playground steps through an '
        "episode by hand and ships with Beam's saved Kinnauri symmetry failure and its counterexample.</p></div>"
        '<div style="flex: 1.2 1 520px; min-width: 0">'
        '<h3 style="margin: 0 0 18px; font-size: 22px; font-weight: 700">Observation · PramaanObservation</h3>'
        + _table(observation, "PramaanObservation", name_w=210, type_w=110, pad=10, min_width="min-width: 520px; ")
        + "</div></div></section>")


def _problems(data: SiteData) -> str:
    models = data.leaderboard["models"]
    n = len(models)
    briefs = {g["id"]: g["briefs"] for g in data.meta["grammars"]}
    jaali = [m["by_grammar"].get("jaali_lattice", 0) for m in models]
    kolam_none = sum(1 for m in models if m["by_grammar"].get("kolam_loop", 0) == 0)
    totals = data.meta["totals"]
    impossible = _impossible_task(data)
    passed_impossible = sum(1 for tries in (data.kolam.get(impossible, {}).get("models") or {}).values()
                            if _models_passing(tries)) if impossible else 0
    blank = _blank_task(data)
    who = (f"{number_word(passed_impossible).capitalize()} model{'' if passed_impossible == 1 else 's'} passed an "
           "impossible brief that way.")
    items = (
        ("Train the first policy.", "#train",
         "The reward adapter is tested. The trainer wiring has not been run end to end."),
        ("Jaali lattices.", "/leaderboard#grammar",
         f"{sum(jaali)} of {n * briefs.get('jaali_lattice', 0)} results pass, and the best model manages "
         f"{max(jaali) if jaali else 0} of {briefs.get('jaali_lattice', 0)}. One connected screen with holes inside the "
         "frame is where every model struggles."),
        ("Kolam loops.", f"/replays/{EPISODE_TASK}",
         f"{number_word(kolam_none).capitalize()} of {number_word(n)} models pass none of the "
         f"{number_word(briefs.get('kolam_loop', 0))} briefs."),
        ("A visible-ink check.", f"/replays/{blank}" if blank else "/leaderboard#matrix",
         f"{totals['low_ink']} of {totals['passed']} accepted designs are nearly blank. Add the rule without breaking "
         "replay of the frozen runs."),
        ("A brief-fidelity check.", f"/replays/{impossible}" if impossible else "/leaderboard",
         f"The checker verifies loop counts, not the dot grid the brief names. {who}"),
        ("Shaped partial credit.", f"/replays/{EPISODE_TASK}",
         "Two loops and five loops both score 0.75. Reward counts rules passed, not how near a miss was."),
    )
    rows = []
    for i, (title, href, text) in enumerate(items):
        border = "border-top: 1px solid #2E2621" + ("; border-bottom: 1px solid #2E2621" if i == len(items) - 1 else "")
        rows.append(
            f'<li style="display: grid; grid-template-columns: 34px minmax(0, 1fr); gap: 8px; padding: 14px 0; {border}">'
            f'<span style="{MONO}; font-size: 13px; color: #F4B63F">{i + 1:02d}</span>'
            f'<span><a href="{esc(href)}" class="bd-problem" style="color: #F4EFE6; font-weight: 700">{esc(title)}</a> '
            f'<span style="color: #B8ADA1">{esc(text)}</span></span></li>')
    return (
        '<section style="padding: 104px 0 0; display: flex; flex-wrap: wrap; gap: 32px 56px; align-items: stretch">'
        '<div id="first-run" style="flex: 1 1 460px; min-width: 0; display: flex; flex-direction: column">'
        f'<p style="{EYEBROW}">FIRST TRAINING RUN</p>'
        '<h2 style="margin: 0 0 20px; font-size: 38px; line-height: 1.04; font-weight: 700; font-stretch: 114%; '
        'text-wrap: balance">Nobody has trained on this yet.</h2>'
        '<div style="flex: 1 1 auto; min-height: 280px; display: flex; flex-direction: column; align-items: center; '
        'justify-content: center; gap: 10px; padding: 24px; border: 1px dashed #6B625B; border-radius: 6px; '
        'text-align: center">'
        f'<p style="margin: 0; {MONO}; font-size: 14px; color: #F4B63F">[YOUR TRAINING CURVE]</p>'
        '<p style="margin: 0; max-width: 380px; font-size: 15px; color: #B8ADA1; text-wrap: pretty">Held-out briefs '
        "passed, before and after training on train-v1, with the replays to prove it. The first run that moves this "
        "number is shown here with its author.</p></div></div>"
        '<div id="open-problems" style="flex: 1 1 460px; min-width: 0">'
        f'<p style="{EYEBROW}">OPEN PROBLEMS</p>'
        '<ol style="margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column">'
        + "".join(rows) + "</ol></div></section>")


def _grammars(data: SiteData) -> str:
    rows = []
    grammars = data.meta["grammars"]
    for i, g in enumerate(grammars):
        rule, check = GRAMMAR_RULES.get(g["id"], ("", g["id"]))
        border = "border-top: 1px solid #2E2621" + ("; border-bottom: 1px solid #2E2621" if i == len(grammars) - 1 else "")
        rows.append(
            f'<tr style="{border}"><th scope="row" style="padding: 13px 16px 13px 0; font-weight: 600">{esc(g["label"])}</th>'
            f'<td style="padding: 13px 16px; {MONO}">{g["briefs"]}</td>'
            f'<td style="padding: 13px 16px; color: #B8ADA1">{esc(rule)}</td>'
            f'<td style="padding: 13px 0 13px 16px; {MONO}; font-size: 13px; color: #948980">{esc(check)}</td></tr>')
    head = (f'<tr style="{MONO}; font-size: 11px; letter-spacing: 0.08em; color: #948980">'
            '<th scope="col" style="padding: 8px 16px 8px 0; font-weight: 500; width: 200px">GRAMMAR</th>'
            '<th scope="col" style="padding: 8px 16px; font-weight: 500; width: 80px">BRIEFS</th>'
            '<th scope="col" style="padding: 8px 16px; font-weight: 500">THE RULE THAT DEFINES IT</th>'
            '<th scope="col" style="padding: 8px 0 8px 16px; font-weight: 500; width: 230px">CHECK</th></tr>')
    return (
        '<section id="grammars" style="padding: 104px 0 0">'
        f'<p style="{EYEBROW}">{number_word(len(grammars)).upper()} GRAMMARS · WHAT EACH ONE CHECKS</p>'
        f'<h2 class="v2-statement" style="{STATEMENT.replace("0 0 28px", "0 0 24px")}">One rule per craft grammar, in a sentence.</h2>'
        '<div style="overflow-x: auto"><table style="width: 100%; min-width: 760px; border-collapse: collapse; '
        f'font-size: 16px; text-align: left"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'
        '<p style="margin: 14px 0 0; max-width: 900px; font-size: 14px; color: #948980; text-wrap: pretty">Every '
        "pattern also passes shared checks for palette size, primitive count, ink coverage, layer alignment and tile "
        "continuity. Checks are exhaustive on the 64 × 64 raster. Jaali minimum web thickness uses an approximation "
        "and is labelled as such. Certificates are unsigned: anyone can recompute them.</p></section>")


def render_build(data: SiteData, base_url: str) -> str:
    totals = data.meta["totals"]
    hero = (
        '<section style="display: flex; flex-wrap: wrap; align-items: flex-end; gap: 20px 64px; padding: 36px 0 36px">'
        '<div style="flex: 1 1 560px; min-width: 0">'
        f'<p style="{EYEBROW}">BUILD WITH IT · BENCHMARK · OPENENV ENVIRONMENT · DATASETS</p>'
        '<h1 style="margin: 0; font-size: 58px; line-height: 0.98; font-weight: 700; font-stretch: 118%; '
        'text-wrap: balance">The checker is the benchmark. And the reward.</h1></div>'
        '<p style="flex: 1 1 420px; min-width: 0; margin: 0; font-size: 18px; color: #B8ADA1; text-wrap: pretty">'
        "One deterministic verifier does both jobs. It runs on a CPU, needs no model key and no judge model, and "
        f"replays all {totals['rendered']:,} saved proposals in about two seconds with identical results.</p></section>")
    body = (f'<div data-build>{hero}{_cards(data)}{_episode(data)}{_contract(base_url)}{_problems(data)}'
            f"{_grammars(data)}</div>")
    footer = (
        '<footer style="display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: center; margin-top: 88px; '
        'padding: 20px 0 56px; border-top: 1px solid #2E2621; font-size: 15px">'
        f'<a href="{REPO}" style="display: inline-flex; align-items: center; min-height: 44px; padding-right: 16px">'
        "Code and benchmark protocol</a>"
        f'<a href="{COLLECTION}" style="display: inline-flex; align-items: center; min-height: 44px; '
        'padding-right: 16px">Everything in one place</a>'
        '<a href="/replays" style="display: inline-flex; align-items: center; min-height: 44px; padding-right: 16px">'
        "Replays</a>"
        f'<span style="margin-left: auto; color: #948980">Validator {esc(data.meta["validator"])} · reward '
        f'{esc(data.meta["reward"])}</span></footer>')
    return page(title="Build with Pramaan · Pramaan",
                description="One deterministic checker is both the benchmark and the reward: run the frozen briefs, "
                            "train against the OpenEnv environment, read every row on the Hub.",
                path="/build", body=body, footer=footer, current="build", base_url=base_url)
