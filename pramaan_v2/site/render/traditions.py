"""Traditions: /traditions and /traditions/{state_code} (spec 8.7, Traditions.dc.html).

Every word inside a tradition card is the registry's own text (spec 8.7 outranks the board's sample
copy): the description is ``motif_vocabulary``; "How to say it" is the cautions sentences that give
attribution or naming guidance; "Cautions" is the rest of ``cautions``; the palette, sources and
unverified sources are listed as recorded.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from ...constants import GRAMMAR_LABELS, GRAMMAR_ORDER, number_word
from ..data import SiteData
from ..shell import FONT_MONO, esc, page

CLAIMS = Path(__file__).resolve().parents[3] / "pramaan" / "data" / "claims.json"
REPO = "https://github.com/Mandark-droid/Proof-of-Art"
DEFAULT_STATE = "HP"

EYEBROW = f"margin: 0 0 12px; font-family: {FONT_MONO}; font-size: 12px; letter-spacing: 0.12em; color: #948980"
LABEL = f"font-family: {FONT_MONO}; font-size: 11px; letter-spacing: 0.08em"
CHIP_ON = ("border: 1px solid #F4EFE6; background: #F4EFE6; color: #15110F")
CHIP_OFF = ("border: 1px solid #4A3F38; background: transparent; color: #F4EFE6")
CHIP = ("min-height: 44px; padding: 0 16px; border-radius: 22px; {state}; font-family: inherit; font-size: 15px; "
        "font-weight: 600; cursor: pointer")
STATE_ON = "border: 1px solid #F4B63F; background: #F4B63F; color: #15110F; font-weight: 700"
STATE_OFF = "border: 1px solid #2E2621; background: transparent; color: #F4EFE6"
STATE_CHIP = ("display: inline-flex; align-items: center; gap: 4px; min-height: 44px; padding: 0 12px; "
              "border-radius: 6px; box-sizing: border-box; {state}; font-size: 14px; text-decoration: none")

_SENTENCE = re.compile(r"(?<!e\.g\.)(?<!i\.e\.)(?<=[.!?])\s+(?=[A-Z'\"(])")
_SAY = re.compile(r"inspired by|attribut|call it|describ|\bsay\b", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE.split((text or "").strip()) if part]


def say_and_cautions(cautions: str) -> tuple[list[str], list[str]]:
    """Sentences giving attribution or naming guidance, and the remaining cautions, both verbatim."""
    sentences = split_sentences(cautions)
    return [s for s in sentences if _SAY.search(s)], [s for s in sentences if not _SAY.search(s)]


@lru_cache(maxsize=1)
def _claims() -> dict:
    doc = json.loads(CLAIMS.read_text(encoding="utf-8"))
    return {"claims": {c["id"]: c for c in doc.get("claims", [])},
            "public": {c["id"]: c for c in doc.get("public_copy", [])}}


def _grammar_order(by_grammar: dict[str, int]) -> list[str]:
    return sorted(by_grammar, key=lambda g: (-by_grammar[g], GRAMMAR_ORDER.index(g) if g in GRAMMAR_ORDER else 99))


def _task_tiles(data: SiteData) -> dict[str, dict]:
    """Per brief: the wall cell of the first model (leaderboard order) whose accepted tile has visible ink."""
    best: dict[str, dict] = {}
    order = data.model_index
    for run in data.wall["runs"]:
        if run["status"] not in ("1", "2", "3") or run.get("low_ink"):
            continue
        current = best.get(run["task"])
        if current is None or order.get(run["model"], 99) < order.get(current["model"], 99):
            best[run["task"]] = run
    return best


def _tile(data: SiteData, run: dict, name: str) -> str:
    wall = data.wall
    cols, rows = wall["cols"], wall["rows"]
    x = run["col"] / (cols - 1) * 100 if cols > 1 else 0
    y = run["row"] / (rows - 1) * 100 if rows > 1 else 0
    label = next((m["label"] for m in data.models if m["key"] == run["model"]), run["model"])
    sprite = f"/site/wall.png?v={data.meta['revision'][:10]}"
    return (
        f'<a href="/replays/{esc(run["task"])}/{esc(run["model"])}" class="tr-tile" style="flex: 0 0 96px; '
        'display: block; width: 96px; height: 96px">'
        f'<span role="img" aria-label="A passing design for the {esc(name)} brief, by {esc(label)}" class="v2-px" '
        f'style="display: block; width: 96px; height: 96px; background-image: url({sprite}); '
        f'background-size: {cols * 100}% {rows * 100}%; background-repeat: no-repeat; '
        f'background-position: {x:.3f}% {y:.3f}%; image-rendering: pixelated; outline: 1px solid #4A3F38"></span></a>'
    )


def _dark(hex_colour: str) -> bool:
    try:
        r, g, b = (int(hex_colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return True
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 90


def _sources(items: list[dict], *, unverified: bool) -> str:
    rows = []
    for source in items:
        rows.append(
            '<li style="padding: 10px 0; border-top: 1px solid #2E2621">'
            f'<a href="{esc(source["url"])}">{esc(source["title"])}</a> '
            f'<span style="color: #948980">· {esc(source["publisher"])}</span><br>'
            f'<span style="color: #B8ADA1">{esc(source["supports"])}</span></li>')
    attr = ' data-unverified' if unverified else ""
    return (f'<ul{attr} style="margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; '
            f'font-size: 14px; line-height: 1.4">{"".join(rows)}</ul>')


def _card(data: SiteData, tradition: dict, tiles: dict[str, dict], shown: bool, models_word: str) -> str:
    name = tradition["name"]
    grammar = tradition["grammar"]
    mono = " · ".join((GRAMMAR_LABELS.get(grammar, grammar).upper(),
                       str(tradition.get("category") or "").replace("_", " ").upper(),
                       "GI REGISTERED" if tradition.get("gi_registered") else "NOT GI REGISTERED"))
    tasks = tradition.get("tasks") or {}
    links = [f'<a href="/replays/{esc(tasks[level])}">{level} brief, {models_word} models</a>'
             for level in ("easy", "hard") if tasks.get(level)]
    briefs = " · ".join(links) if links else "not in the evaluation briefs"
    head = (
        '<div style="min-width: 0">'
        f'<p style="margin: 0 0 6px; font-family: {FONT_MONO}; font-size: 11px; letter-spacing: 0.06em; '
        f'color: #948980">{esc(mono)}</p>'
        f'<h3 style="margin: 0; font-size: 24px; line-height: 1.12; font-weight: 700">{esc(name)}</h3>'
        f'<p style="margin: 6px 0 0; font-size: 14px; color: #B8ADA1">Research confidence: '
        f'{esc(tradition.get("confidence"))} · {briefs}</p></div>')
    run = next((tiles[tasks[level]] for level in ("easy", "hard") if tasks.get(level) in tiles), None)
    if run is not None:
        head = f'<div style="display: flex; gap: 16px; align-items: flex-start">{_tile(data, run, name)}{head}</div>'

    parts = [head]
    vocabulary = "; ".join(tradition.get("motif_vocabulary") or [])
    if vocabulary:
        parts.append(f'<p style="margin: 0; font-size: 15px; color: #B8ADA1; text-wrap: pretty">{esc(vocabulary)}</p>')
    say, cautions = say_and_cautions(tradition.get("cautions", ""))
    if say:
        parts.append(f'<div><p style="margin: 0 0 6px; {LABEL}; color: #F4B63F">HOW TO SAY IT</p>'
                     f'<p style="margin: 0; font-size: 15px; text-wrap: pretty">{esc(" ".join(say))}</p></div>')
    palette = tradition.get("palette_notes") or []
    if palette:
        swatches = "".join(
            f'<span title="{esc(p["name"])}" style="width: 36px; height: 36px; background: {esc(p["hex"])}'
            f'{"; outline: 1px solid #4A3F38" if _dark(p["hex"]) else ""}"></span>' for p in palette)
        basis = "".join(
            f'<li><span style="color: #B8ADA1">{esc(p["name"])}</span> · {esc(p["basis"])}</li>' for p in palette)
        parts.append(
            f'<div><p style="margin: 0 0 8px; {LABEL}; color: #948980">PALETTE AND ITS BASIS</p>'
            f'<div style="display: flex; flex-wrap: wrap; gap: 6px" aria-hidden="true">{swatches}</div>'
            '<ul style="margin: 8px 0 0; padding: 0; list-style: none; display: flex; flex-direction: column; '
            f'gap: 4px; font-size: 13px; line-height: 1.45; color: #948980; text-wrap: pretty">{basis}</ul></div>')
    if cautions:
        parts.append(f'<div><p style="margin: 0 0 6px; {LABEL}; color: #948980">CAUTIONS</p>'
                     f'<p style="margin: 0; font-size: 15px; color: #B8ADA1; text-wrap: pretty">'
                     f'{esc(" ".join(cautions))}</p></div>')
    if tradition.get("sources"):
        parts.append(f'<div><p style="margin: 0 0 6px; {LABEL}; color: #948980">SOURCES · WHAT EACH SUPPORTS</p>'
                     f'{_sources(tradition["sources"], unverified=False)}</div>')
    if tradition.get("sources_unverified"):
        parts.append(f'<div><p style="margin: 0 0 6px; {LABEL}; color: #948980">SOURCES LISTED AS UNVERIFIED</p>'
                     f'{_sources(tradition["sources_unverified"], unverified=True)}</div>')
    return (
        f'<article data-tradition="{esc(tradition["id"])}" data-grammar="{esc(grammar)}"{"" if shown else " hidden"} '
        'style="display: flex; flex-direction: column; gap: 16px; min-width: 0; padding: 24px; background: #1D1815; '
        'border: 1px solid #2E2621; border-radius: 6px">' + "".join(parts) + "</article>")


def _festival(state: dict) -> str:
    ledger = _claims()
    ids: list[str] = []
    for tradition in state["traditions"]:
        for claim_id in (tradition.get("festival_lens") or {}).get("claim_ids") or []:
            if claim_id in ledger["claims"] and claim_id not in ids:
                ids.append(claim_id)
    note = ("This survey spans every state and Union Territory; it does not claim that each listed tradition is "
            "tied to the festival.")
    if ids:
        items = "".join(
            f'<li style="padding: 8px 0; border-top: 1px solid #2E2621">{esc(ledger["claims"][cid]["text"])} '
            + " ".join(f'<a href="{esc(s["url"])}" style="white-space: nowrap">{esc(s["publisher"])}</a>'
                       for s in ledger["claims"][cid].get("sources", [])[:2])
            + "</li>" for cid in ids)
        inner = (f'<ul style="margin: 0; padding: 0 16px; list-style: none; font-size: 15px; color: #B8ADA1">{items}</ul>'
                 f'<p style="margin: 0; padding: 8px 16px 16px; font-size: 14px; color: #948980; text-wrap: pretty">'
                 f'{note}</p>')
    else:
        general = ledger["public"].get("festival_context", {}).get("text", "")
        inner = (f'<p style="margin: 0; padding: 0 16px 16px; font-size: 15px; color: #B8ADA1; text-wrap: pretty">'
                 f'{esc(general)} {note}</p>')
    return ('<details class="tr-details" style="border: 1px solid #2E2621; border-radius: 6px">'
            '<summary style="min-height: 44px; display: flex; align-items: center; padding: 0 16px; font-weight: 600; '
            f'cursor: pointer">Optional Diwali context</summary>{inner}</details>')


def render_traditions(data: SiteData, state_code: str | None, base_url: str, grammar: str | None = None) -> str | None:
    registry = data.traditions
    states = registry["states"]
    code = (state_code or DEFAULT_STATE).upper()
    state = next((s for s in states if s["code"] == code), None)
    if state is None:
        return None
    by_grammar = registry["by_grammar"]
    if grammar not in by_grammar:
        grammar = None
    totals = registry["totals"]
    order = _grammar_order(by_grammar)
    query = f"?grammar={grammar}" if grammar else ""
    path = f"/traditions/{code}" if state_code else "/traditions"

    hero = (
        '<section style="display: flex; flex-wrap: wrap; align-items: flex-end; gap: 20px 64px; padding: 36px 0 32px">'
        '<div style="flex: 1 1 560px; min-width: 0">'
        f'<p style="{EYEBROW}">TRADITIONS · SOURCES AND CAUTIONS</p>'
        '<h1 style="margin: 0; font-size: 58px; line-height: 0.98; font-weight: 700; font-stretch: 118%; '
        'text-wrap: balance">Where every brief comes from.</h1></div>'
        '<p style="flex: 1 1 420px; min-width: 0; margin: 0; font-size: 18px; color: #B8ADA1; text-wrap: pretty">'
        "Each brief starts from a documented craft tradition. This is the registry behind them: what the tradition "
        "is, where its palette comes from, how to describe a generated design without overclaiming, and the source "
        "for each statement.</p></section>")

    stat = ('<div style="background: #15110F; padding: 20px 22px"><div style="font-size: 40px; font-weight: 700; '
            'font-stretch: 112%; line-height: 1">{n}</div><div style="margin-top: 8px; font-size: 15px; '
            'color: #B8ADA1">{text}</div></div>')
    stats = (
        '<section class="tr-stats" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(200px, 100%), 1fr)); '
        'gap: 1px; background: #2E2621; border: 1px solid #2E2621">'
        + stat.format(n=totals["traditions"], text=f"traditions across all {len(states)} states and Union Territories")
        + stat.format(n=totals["sources"], text=f"sources, each with the claim it supports. {totals['unverified']} "
                      "more are listed as unverified.")
        + stat.format(n=totals["gi"], text="hold a registered Geographical Indication, so their names are never used "
                      "as product labels")
        + stat.format(n=totals["multi_state"], text="are practised in more than one state and are attributed that way")
        + "</section>")

    chips = [f'<button type="submit" name="grammar" value="" aria-pressed="{"false" if grammar else "true"}" '
             f'style="{CHIP.format(state=CHIP_OFF if grammar else CHIP_ON)}">'
             f'All {totals["traditions"]}</button>']
    for g in order:
        on = g == grammar
        chips.append(f'<button type="submit" name="grammar" value="{esc(g)}" aria-pressed="{"true" if on else "false"}" '
                     f'style="{CHIP.format(state=CHIP_ON if on else CHIP_OFF)}">'
                     f'{esc(GRAMMAR_LABELS.get(g, g))} {by_grammar[g]}</button>')

    state_chips = []
    for s in states:
        counts = {g: sum(1 for t in s["traditions"] if t["grammar"] == g) for g in by_grammar}
        count = counts[grammar] if grammar else len(s["traditions"])
        here = s["code"] == code
        number = (f'<span data-count>{count}</span>' if here
                  else f'<span data-count style="color: #948980">{count}</span>')
        state_chips.append(
            f'<a href="/traditions/{s["code"]}{query}"{" aria-current=" + chr(34) + "page" + chr(34) if here else ""} '
            f'data-state="{s["code"]}" data-all="{len(s["traditions"])}" data-counts="{esc(json.dumps(counts))}" '
            f'style="{STATE_CHIP.format(state=STATE_ON if here else STATE_OFF)}">{esc(s["name"])} {number}</a>')

    filters = (
        '<section style="padding: 40px 0 0; display: flex; flex-direction: column; gap: 18px">'
        f'<form method="get" action="{esc(path)}" data-tr-filter>'
        '<p style="margin: 0 0 8px; font-size: 14px; color: #948980">Pattern type</p>'
        '<div role="group" aria-label="Filter by pattern type" style="display: flex; flex-wrap: wrap; gap: 8px">'
        + "".join(chips) + "</div></form>"
        '<div><p style="margin: 0 0 8px; font-size: 14px; color: #948980">State or Union Territory, with the number '
        "of traditions documented there</p>"
        '<div role="group" aria-label="Choose a state or Union Territory" class="tr-states" '
        'style="display: flex; flex-wrap: wrap; gap: 6px; font-size: 14px">' + "".join(state_chips) + "</div></div>"
        "</section>")

    tiles = _task_tiles(data)
    models_word = number_word(len(data.models))
    traditions = state["traditions"]
    used = sum(1 for t in traditions if t.get("tasks"))
    cards = "".join(_card(data, t, tiles, grammar is None or t["grammar"] == grammar, models_word) for t in traditions)
    visible = sum(1 for t in traditions if grammar is None or t["grammar"] == grammar)
    empty_label = GRAMMAR_LABELS.get(grammar, "") if grammar else ""
    empty = (f'<p data-tr-empty{"" if visible == 0 else " hidden"} style="margin: 0; padding: 24px; border: 1px dashed '
             f'#6B625B; border-radius: 6px; color: #B8ADA1; text-align: center">No <span data-tr-empty-label>'
             f'{esc(empty_label.lower())}</span> tradition is documented for {esc(state["name"])}. '
             f'<a href="/traditions/{code}">Show all {len(traditions)}</a></p>')
    detail = (
        f'<section style="padding: 56px 0 0" aria-labelledby="tr-state">'
        '<div style="display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; '
        'gap: 8px 32px; margin-bottom: 20px">'
        f'<h2 id="tr-state" class="v2-statement" style="margin: 0; font-size: 46px; line-height: 1.02; font-weight: 700; '
        f'font-stretch: 114%">{esc(state["name"])}</h2>'
        f'<span style="font-size: 15px; color: #B8ADA1">{len(traditions)} tradition{"" if len(traditions) == 1 else "s"} '
        f'· {used} used by the evaluation briefs · <a href="/try">open in the studio</a></span></div>'
        + empty +
        '<div class="tr-cards" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(min(380px, 100%), 1fr)); '
        f'gap: 20px; align-items: start">{cards}</div></section>')

    about = (
        '<section style="padding: 56px 0 0; display: flex; flex-wrap: wrap; gap: 24px 56px; align-items: flex-start">'
        '<div style="flex: 1 1 420px; min-width: 0">'
        '<h2 style="margin: 0 0 12px; font-size: 30px; line-height: 1.1; font-weight: 700; font-stretch: 112%">'
        "What this registry is, and is not</h2>"
        '<p style="margin: 0; color: #B8ADA1; text-wrap: pretty">These are computational interpretations inspired by '
        "documented geometric ideas. The checker tests geometry, never cultural authenticity. Protected names are not "
        "used as labels, ritual and devotional imagery is kept out of the briefs, and shared traditions are "
        "attributed to every region that practises them.</p></div>"
        '<div style="flex: 1 1 420px; min-width: 0; display: flex; flex-direction: column; gap: 12px">'
        + _festival(state)
        + f'<a href="{REPO}/blob/main/CITATIONS.md" style="display: inline-flex; align-items: center; min-height: 44px; '
        'align-self: flex-start; font-weight: 600">Every citation in one file →</a>'
        f'<a href="{REPO}/issues" style="display: inline-flex; align-items: center; min-height: 44px; '
        'align-self: flex-start; font-weight: 600">Correct a source or a caution →</a></div></section>')

    footer = (
        '<footer style="display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: center; margin-top: 72px; '
        'padding: 20px 0 56px; border-top: 1px solid #2E2621; font-size: 15px">'
        '<a href="/try" style="display: inline-flex; align-items: center; min-height: 44px; padding-right: 16px">'
        "Try a tradition in the studio</a>"
        '<a href="/replays" style="display: inline-flex; align-items: center; min-height: 44px; padding-right: 16px">'
        "See how the models did</a>"
        '<span style="margin-left: auto; color: #948980">Inspired by these traditions. Not a substitute for the people '
        "who keep them alive.</span></footer>")

    body = (f'<div data-traditions data-grammar="{esc(grammar or "")}">'
            + hero + stats + filters + detail + about + "</div>")
    title = "Traditions and sources · Pramaan" if not state_code else f"{state['name']} · Traditions · Pramaan"
    description = (f"{totals['traditions']} documented craft traditions across {len(states)} states and Union "
                   f"Territories, with {totals['sources']} sources and what each one supports.")
    return page(title=title, description=description, path=path + query, body=body, footer=footer,
                current=None, base_url=base_url)
